import warnings
from typing import Any

import hydra
import tokenizers
import tokenizers.pre_tokenizers
import torch
import transformers
import transformers.modeling_outputs
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

    def get_numeric_emb(self) -> numllama.nn.LatentEmbedding:
        emb = self.embedding.embeddings["num"]
        if isinstance(emb, EmbeddingGlue):
            emb = emb.embedding
        assert isinstance(emb, numllama.nn.LatentEmbedding)
        return emb

    def forward(self, *args, **kwargs) -> transformers.modeling_outputs.CausalLMOutputWithPast:
        if self.training:
            with self.get_numeric_emb().use_latents():
                return super().forward(*args, **kwargs)
        # in eval mode, latents are already cached
        return super().forward(*args, **kwargs)


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
