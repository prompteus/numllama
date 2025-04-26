import contextlib
import warnings
from typing import Any, List, Optional, Tuple, Union

import hydra
import tokenizers
import tokenizers.pre_tokenizers
import torch
import transformers
import transformers.modeling_outputs
import transformers.models
from torch import Tensor

import numllama.addition
import numllama.nn


class NumLlamaConfig(transformers.LlamaConfig):
    model_type = "NumLlama"

    def __init__(
        self,
        numeric_input_emb_config: dict[str, Any] | None = None,
        numeric_encoder_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.numeric_input_emb = numeric_input_emb_config
        self.numeric_encoder = numeric_encoder_config


class EmbeddingGlue(numllama.nn.DualEmbedding):
    def __init__(self, embed_dim: int, model_dim: int, embedding: numllama.nn.DualEmbedding):
        super().__init__()
        self.to_model_dim = torch.nn.Linear(embed_dim, model_dim)
        self.to_embed_dim = torch.nn.Linear(model_dim, embed_dim)
        self.embedding = embedding
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = model_dim
        self.reset_params()

    @torch.no_grad()
    def reset_params(self):
        torch.nn.init.zeros_(self.to_model_dim.bias)
        torch.nn.init.zeros_(self.to_embed_dim.bias)
        torch.nn.init.xavier_uniform_(self.to_model_dim.weight)
        torch.nn.init.xavier_uniform_(self.to_embed_dim.weight)

    def _encode(self, x: Tensor) -> Tensor:
        return self.to_model_dim(self.embedding._encode(x))

    def _decode(self, x: Tensor) -> Tensor:
        return self.embedding._decode(self.to_embed_dim(x))


class NumLlamaForCausalLM(transformers.LlamaForCausalLM):
    config_class = NumLlamaConfig

    def apply_numeric_patch(self):
        if not isinstance(self.config, NumLlamaConfig):
            raise ValueError(f"Expected self.config to be {NumLlamaConfig.__name__}, got {type(self.config).__name__}")
        if self.config.numeric_input_emb is None:
            raise ValueError("Numeric input embedding config is required.")
        num_emb_config = numllama.addition.NumEmbeddingConfig(**self.config.numeric_input_emb)
        num_encoder_config = self.config.numeric_encoder
        supported_nums = torch.arange(num_emb_config.min_value, num_emb_config.max_value + 1)
        numeric_emb: numllama.nn.DualEmbedding = numllama.nn.LatentEmbedding(
            input_embedding=numllama.nn.TokenEmbedding.from_pretrained(
                numllama.nn.sinusoidal_encode(x=supported_nums, **num_emb_config.model_dump()),
                freeze=True,
            ),
            encoder=hydra.utils.instantiate(num_encoder_config),
            embedding_dim=num_emb_config.embedding_dim,
        )
        self.build_num_latents = numeric_emb.build_latents
        if numeric_emb.embedding_dim != self.model.embed_tokens.embedding_dim:
            wrapper = EmbeddingGlue(
                embed_dim=numeric_emb.embedding_dim,
                model_dim=self.model.embed_tokens.embedding_dim,
                embedding=numeric_emb,
            )
            numeric_emb = wrapper
        if not self.config.tie_word_embeddings:
            raise NotImplementedError("Independent embeddings and lm-head are not implemented.")
        string_emb = numllama.nn.TokenEmbedding.from_pretrained(
            self.model.embed_tokens.weight,
            freeze=False,
        )
        del self.model.embed_tokens  # type: ignore
        del self.lm_head  # type: ignore
        self.embedding = numllama.nn.MultiEmbedding({"str": string_emb, "num": numeric_emb})
        self.model.embed_tokens = numllama.nn.Lambda(self.embedding.encode)
        self.lm_head = numllama.nn.Lambda(self.embedding.decode)
        self.config.vocab_size = self.embedding.num_embeddings

    def get_numeric_emb(self) -> numllama.nn.LatentEmbedding:
        emb = self.embedding.embeddings["num"]
        if isinstance(emb, EmbeddingGlue):
            emb = emb.embedding
        assert isinstance(emb, numllama.nn.LatentEmbedding)
        return emb

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[transformers.cache_utils.Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: transformers.processing_utils.Unpack[transformers.models.llama.modeling_llama.KwargsForCausalLM],
    ) -> Union[Tuple, transformers.modeling_outputs.CausalLMOutputWithPast]:
        if self.training:
            ctx = self.get_numeric_emb().use_latents
        else:
            # in eval mode, latents are already computed and cached
            ctx = contextlib.nullcontext
        with ctx():
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

    def prepare_inputs_for_generation(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        is_first_generation_call = "past_key_values" in kwargs and not kwargs["past_key_values"]
        if is_first_generation_call and "labels" in kwargs:
            # this needs to modify input_ids in-place, otherwise the further calls will aggregate
            # the generated outputs behind the labels
            input_ids = args[0] if args else kwargs["input_ids"]  # must be filled
            attention_mask = args[2] if len(args) > 2 else kwargs["attention_mask"]

            # shorten the generation inputs for the ids of labels
            assert len(input_ids) == 1 and len(attention_mask) == 1, "Following code assumes eval batch size of 1."
            input_ids_eos_pos = torch.argmax((input_ids[0] == self.config.eos_token_id).int())
            # zero out the input_ids on the position of labels
            input_ids.resize_(1, input_ids_eos_pos+1)
            attention_mask.resize_(1, input_ids_eos_pos+1)

        return super().prepare_inputs_for_generation(*args, **kwargs)

def patch_llama_digit_splitting(tokenizer: transformers.PreTrainedTokenizerFast):
    warnings.warn(
        "This function assumes the specific behavior of llama 3 1b tokenizer. "
        "Make sure to verify it works as expected on your particular tokenizer."
    )
    tokenizer._tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [
            tokenizers.pre_tokenizers.Split(
                # Pattern is same except removal of splitting numbers to triplets of digits
                pattern=tokenizers.Regex(
                    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                ),
                behavior="isolated",
                invert=False,
            ),
            tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )


def add_num_tokens_to_tokenizer(
    minimum: int,
    maximum: int,
    tokenizer: transformers.PreTrainedTokenizerFast,
    model: NumLlamaForCausalLM | None = None,
) -> None:
    warnings.warn(
        "This function assumes the specific behavior of llama 3 1b tokenizer. "
        "Make sure to verify it works as expected on your particular tokenizer."
    )
    supported_numbers = range(minimum, maximum + 1)
    num_tokens_str = [" " + str(num) for num in supported_numbers]
    orig_vocab_size = len(tokenizer.get_vocab())
    tokenizer.add_tokens(num_tokens_str)  # type: ignore
    # verify that the new tokens are added:
    # - at the end of the vocab
    # - in a contiguous block
    # - in a correct order
    if model is not None:
        num_string_embeddings = model.embedding.embeddings["str"].num_embeddings
        assert num_string_embeddings == orig_vocab_size
    expected_ids = list(range(orig_vocab_size, orig_vocab_size + len(supported_numbers)))
    actual_ids = tokenizer.convert_tokens_to_ids(num_tokens_str)
    if actual_ids != expected_ids:
        raise ValueError(
            "Failed to add number tokens to tokenizer. "
            "Number tokens were not placed correctly "
            "(and your tokenizer is now broken)."
        )


transformers.AutoConfig.register("NumLlama", NumLlamaConfig)
transformers.AutoModelForCausalLM.register(NumLlamaConfig, NumLlamaForCausalLM)
