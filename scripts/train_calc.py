import logging
import sys
import traceback
from typing import Optional

import datasets
import numpy as np
import transformers
import typer
import wandb
import re

import numllama.metrics
import numllama as gadgets
from scripts import utils
from scripts.permutation_utils import StepPermuter

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
)

logger = logging.getLogger()

sys.modules["svgai"] = numllama
sys.modules["svgai.train"] = numllama.addition


@app.command()
def main(
    use_instructions_train: bool = False,
    use_instructions_val: bool = False,
    permute_train_numbers: bool = False,
    permute_val_numbers: bool = False,
    only_addition_in_train: bool = False,
    only_addition_in_val: bool = False,
    model_name: str = "meta-llama/Llama-3.2-1B",
    num_embeddings_model: Optional[str] = "None",
    freeze_input_embeddings: bool = False,
    limit_train_set_per_ds: int = -1,
    limit_val_set_per_ds: int = 25,  # TODO
    wandb_entity: str = "transformersclub",
    wandb_project: str = "numllama",
    wandb_group: Optional[str] = "",
    wandb_dir: str = ".wandb",
    checkpoint_dir: str = "checkpoints",
    train_ds: str = "MU-NLPC/Calc-X",
    train_ds_split_name: str = "train",
    input_col: str = "question",
    train_label_col: str = "chain",
    valid_label_col: str = "chain",
    valid_ds: str = "MU-NLPC/Calc-X",
    valid_ds_subset: Optional[str] = None,
    max_output_length: int = 1024,
    batch_size: int = 4,
    effective_batch_size: int = 32,
    eval_batch_size: int = 1,
    optim="adamw_torch",
    save_total_limit: int = 5,
    eval_steps: int = 2000,
    save_steps: int = 20000,
    learning_rate: float = 4e-5,
    early_stopping_patience: Optional[int] = 100,
    early_stopping_threshold: float = 0.01,
) -> None:
    cli_params = locals()

    grad_accum = effective_batch_size // batch_size

    # # ORIGINAL CALC-X code: T5 + gadgets
    #
    # model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cpu")
    # model_class = gadgets.model.gadget_assisted_model(model.__class__)
    # del model
    # gc.collect()
    # model = model_class.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    # assert isinstance(model, gadgets.model.GadgetAssist)
    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if num_embeddings_model.lower() != "none":
        addition_model = numllama.addition.AdditionLightning.load_from_checkpoint(num_embeddings_model)
        numeric_input_emb_config = addition_model.model.embedding_config.model_dump()
        numeric_encoder_config = addition_model.model.num_encoder_config

        original_config = transformers.LlamaConfig.from_pretrained(model_name)
        config = numllama.numllama.NumLlamaConfig(
                numeric_input_emb_config=numeric_input_emb_config,
                numeric_encoder_config=numeric_encoder_config,
                **original_config.to_dict(),
        )
        model = numllama.numllama.NumLlamaForCausalLM.from_pretrained(model_name, config=config)

        # create the new numeric embedding layer inside llama
        model.apply_numeric_patch()

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        # change how llama tokenizes numbers
        numllama.numllama.patch_llama_digit_splitting(tokenizer)

        numllama.numllama.add_num_tokens_to_tokenizer(
                numeric_input_emb_config["min_value"],
                numeric_input_emb_config["max_value"],
                tokenizer,
                model
        )
        num_state_dict = addition_model.model.embedding.state_dict()
        model.get_numeric_emb().load_state_dict(num_state_dict)

        # resolve linear glue if possible
        glue_dims_match = (
            model.embedding.embs["num"].to_model_dim.in_features == addition_model.model.to_model_dim.in_features
            and model.embedding.embs["num"].to_model_dim.out_features == addition_model.model.to_model_dim.out_features
            and model.embedding.embs["num"].to_embed_dim.in_features == addition_model.model.from_model_dim.in_features
            and model.embedding.embs["num"].to_embed_dim.out_features == addition_model.model.from_model_dim.out_features
        )
        if glue_dims_match:
            model.embedding.embs["num"].to_model_dim.load_state_dict(addition_model.model.to_model_dim.state_dict())
            model.embedding.embs["num"].to_embed_dim.load_state_dict(addition_model.model.from_model_dim.state_dict())
            logger.info("The glue linear layers successfully reloaded.")
        else:
            logger.warning("Model and embedding glue dims do not match. The glue linear layers are newly trained!")

    else:
        model = numllama.numllama.BaselineLlamaForCausalLM.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if not tokenizer:  # tokenizer apparently does not support use_fast=False -> skip it
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.tokenizer = tokenizer
    model.generation_config.max_new_tokens = max_output_length

    if freeze_input_embeddings:
        if num_embeddings_model.lower() != "none":
            print("Freezing num embeddings in training")
            model.build_num_latents()
            for p in model.get_numeric_emb().parameters():
                p.requires_grad = False
            if glue_dims_match:
                # pre-trained glue -> freeze
                print("Also freezing pre-trained glue to num embeddings")
                for p in model.embedding.embs["num"].to_model_dim.parameters():
                    p.requires_grad = False
                for p in model.embedding.embs["num"].to_embed_dim.parameters():
                    p.requires_grad = False
        else:
            print("Freezing input embeddings in training")
            for p in model.model.embed_tokens.parameters():
                p.requires_grad = False
    else:
        print("Not freezing any input embeddings")

    val_dataset_tags = []
    if only_addition_in_val:
        val_dataset_tags.append("addition")
    if permute_val_numbers:
        val_dataset_tags.append("permuted")

    tags = ["+".join(val_dataset_tags)+"_val"] if val_dataset_tags else []

    wandb.init(entity=wandb_entity,
               project=wandb_project,
               tags=tags,
               group=wandb_group,
               dir=wandb_dir)
    wandb.config.update({"cli_params": cli_params})

    print("Running with arguments: ", " ".join(sys.argv[1:]))

    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model)
    ds_train = datasets.load_dataset(train_ds, split=train_ds_split_name)
    ds_valid = datasets.load_dataset(valid_ds, split="validation")


    def is_only_addition(example, chain_col: str) -> bool:
        import bs4
        chain = example[chain_col]
        doc = bs4.BeautifulSoup(chain, features="html.parser")
        gadget_tags: list[bs4.Tag] = doc.find_all(gadgets.markup.GADGET_TAG)
        only_addition = True
        for tag in gadget_tags:
            gadget_input = tag.get_text()
            operators = re.findall("[+\-*/^%=<>]", gadget_input)
            only_addition = only_addition and all(op == "+" or op == "-" for op in operators)

        return only_addition

    if only_addition_in_train:
        ds_train = ds_train.filter(lambda example: is_only_addition(example, train_label_col))
        print('Subsetting train dataset to %s samples containing only "+"/"-"' % len(ds_train))

    if only_addition_in_val:
        ds_valid = ds_valid.filter(lambda example: is_only_addition(example, train_label_col))
        print('Subsetting val dataset to %s samples containing only "+"/"-"' % len(ds_valid))

    instructions_ds = datasets.load_dataset("MU-NLPC/Calc-X_style-instructions")

    random_generator = np.random.default_rng(0)

    def add_instruction(example):
        source_ds = example["source_ds"]
        template: str = random_generator.choice(instructions_ds[source_ds]["template"],
                                                p=instructions_ds[source_ds]["weight"])
        return {input_col: template.format(example[input_col])}

    if limit_train_set_per_ds is not None and limit_train_set_per_ds > 0:
        df_train = ds_train.to_pandas()
        df_train = df_train.groupby("source_ds").sample(limit_train_set_per_ds, random_state=0)
        ds_train = datasets.Dataset.from_pandas(df_train)

    if valid_ds_subset is not None:
        ds_valid = ds_valid.filter(lambda x: x["source_ds"] == valid_ds_subset)
    if limit_val_set_per_ds is not None and limit_val_set_per_ds > 0:
        df_valid = ds_valid.to_pandas()
        df_valid = df_valid.groupby("source_ds").sample(limit_val_set_per_ds, random_state=0)
        ds_valid = datasets.Dataset.from_pandas(df_valid)

    if use_instructions_train:
        ds_train = ds_train.map(add_instruction)
    if use_instructions_val:
        ds_valid = ds_valid.map(add_instruction)

    def _preproc_nums(input_str: str, space_nums: bool = True) -> str:
        # transformations needed for a correct functioning of the num-augmented tokenizer
        flat_numbers_input_str = input_str.replace("_", "")
        if space_nums:
            flat_numbers_input_str = re.sub(r'\s*(\d+(?:\.\d+)?)\s*', r' \1 ', flat_numbers_input_str)
        return flat_numbers_input_str

    def preprocess(example, label_col, permute: bool = True, permuter: Optional[StepPermuter] = None):
        questions = example[input_col]
        chains = example[label_col]
        questions = [_preproc_nums(text) for text in questions]
        chains = [_preproc_nums(text) for text in chains]

        if permute:
            for i in range(len(example[input_col])):
                questions[i], chains[i] = permuter.permute_chain(questions[i], chains[i])

        inputs = tokenizer(questions, truncation=True, max_length=max_output_length)
        labels = tokenizer(text_target=chains, truncation=True, max_length=max_output_length)

        inputs_labels = [i + [tokenizer.eos_token_id] + l + [tokenizer.eos_token_id]
                         for i, l in zip(inputs.input_ids, labels.input_ids)]
        labels_ignored = [[-100]*(len(i)+1) + l + [tokenizer.eos_token_id]
                          for i, l in zip(inputs.input_ids, labels.input_ids)]

        return {"input_ids": inputs_labels, "labels": labels_ignored}

    if permute_train_numbers:
        # drop aqua-rat with inparseable computations
        ds_train = ds_train.filter(lambda row: row["source_ds"] != "aqua_rat")
    ds_train = ds_train.map(preprocess, batched=True, fn_kwargs={"label_col": train_label_col,
                                                                 "permute": permute_train_numbers,
                                                                 "permuter": StepPermuter(tokenizer, seed=42, space_nums=True)})
    if permute_val_numbers:
        # drop aqua-rat with inparseable computations
        ds_valid = ds_valid.filter(lambda row: row["source_ds"] != "aqua_rat")
    ds_valid = ds_valid.map(preprocess, batched=True, fn_kwargs={"label_col": valid_label_col,
                                                                 "permute": permute_val_numbers,
                                                                 "permuter": StepPermuter(tokenizer, seed=42, space_nums=True)})
    ds_train = ds_train.shuffle(seed=0)

    callbacks = []
    if early_stopping_patience is not None:
        early_stopping = transformers.EarlyStoppingCallback(early_stopping_patience=early_stopping_patience,
                                                            early_stopping_threshold=early_stopping_threshold)
        callbacks.append(early_stopping)

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=f"{checkpoint_dir}/{wandb.run.name}",
        # output_dir=f"{checkpoint_dir}",
        learning_rate=learning_rate,
        do_train=True,
        do_eval=True,
        warmup_steps=2000,
        max_steps=200_000,
        optim=optim,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=eval_batch_size,
        eval_accumulation_steps=1,
        logging_steps=100,
        eval_steps=eval_steps,
        save_steps=save_steps,
        eval_strategy="steps",
        bf16=True,
        bf16_full_eval=True,
        predict_with_generate=True,
        gradient_checkpointing=True,
        # generation_max_length=max_output_length,
        include_inputs_for_metrics=True,
        report_to="wandb",
        metric_for_best_model="avg_correct_results",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=save_total_limit,
    )
    metrics = numllama.metrics.MonitorMetrics(tokenizer=tokenizer,
                                              log_predictions=True,
                                              eval_ds_inputs=ds_valid["input_ids"],
                                              source_ds_col=ds_valid["source_ds"])
    trainer = utils.CustomSeq2SeqTrainer(model=model,
                                         args=training_args,
                                         train_dataset=ds_train,
                                         eval_dataset=ds_valid,
                                         tokenizer=tokenizer,
                                         data_collator=data_collator,
                                         compute_metrics=metrics,
                                         callbacks=callbacks)
    trainer.train()


if __name__ == "__main__":
    try:
        app()
    except BaseException as e:
        print(traceback.format_exc())
        raise e
