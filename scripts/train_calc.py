import logging
import sys
import traceback
from typing import Optional

import datasets
import numpy as np
import transformers
import typer
import wandb

import numllama.metrics
from scripts import utils

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
)

logger = logging.getLogger()

sys.modules["svgai"] = numllama
sys.modules["svgai.train"] = numllama.addition

@app.command()
def main(
    use_instructions_train: bool = True,
    use_instructions_val: bool = False,
    model_name: str = "meta-llama/Llama-3.2-1B",
    # num_embeddings_model: Optional[str] = "/var/tmp/xstefan3/svgai/checkpoints/warm-sunset-628__u99wrvem-global-step=145000__valid-acc=0.999.ckpt",
    num_embeddings_model: Optional[str] = "/var/tmp/xstefan3/svgai/checkpoints/eternal-monkey-624__vtqqjo78/global-step=25000__valid-acc=0.003.ckpt",
    # num_embeddings_model: Optional[str] = None,
    limit_train_set_per_ds: int = -1,
    limit_val_set_per_ds: int = 40,  # TODO
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
    max_output_length: int = 512,
    batch_size: int = 4,
    grad_accum: int = 8,
    eval_batch_size: int = 1,
    optim="adamw_torch",
    save_total_limit: int = 5,
    eval_steps: int = 2000,  # = 16000, TODO
    save_steps: int = 2000,  # = 16000, TODO
    learning_rate: float = 5e-5,
    early_stopping_patience: Optional[int] = 20,
    early_stopping_threshold: float = 0.03,
) -> None:
    cli_params = locals()

    # # ORIGINAL CALC-X code: T5 + gadgets
    #
    # model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cpu")
    # model_class = gadgets.model.gadget_assisted_model(model.__class__)
    # del model
    # gc.collect()
    # model = model_class.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    # assert isinstance(model, gadgets.model.GadgetAssist)
    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if num_embeddings_model is not None:
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

    else:
        model = numllama.numllama.BaselineLlamaForCausalLM.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)

    tokenizer.pad_token = tokenizer.eos_token
    model.tokenizer = tokenizer

    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        tags=[model_name, "supervised"],
        group=wandb_group,
        dir=wandb_dir,
    )

    wandb.config.update({"cli_params": cli_params})

    # # ORIGINAL CALC-X code
    #
    # gadgets.utils.add_new_token(
    #     "<",
    #     is_special=False,
    #     tokenizer=tokenizer,
    #     model=model,
    #     init_with=["[", ">"],
    # )

    # model.prepare_for_generate(
    #     tokenizer,
    #     enabled_gadgets=[gadgets.gadget.Calculator()],
    #     default_max_tokens=max_output_length,
    # )

    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model)
    ds_train = datasets.load_dataset(train_ds, split=train_ds_split_name)
    ds_valid = datasets.load_dataset(valid_ds, split="validation")
    instructions_ds = datasets.load_dataset("MU-NLPC/Calc-X_style-instructions")

    random_generator = np.random.default_rng(0)

    def add_instruction(example):
        source_ds = example["source_ds"]
        template: str = random_generator.choice(
            instructions_ds[source_ds]["template"],
            p=instructions_ds[source_ds]["weight"],
        )
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

    def preprocess(example, label_col):
        input_text = [text.replace(">", "> ").replace("<", " <").replace("_", "") for text in example[input_col]]
        input_label = [text.replace(">", "> ").replace("<", " <").replace("_", "") for text in example[label_col]]
        inputs = tokenizer(input_text, truncation=True, max_length=max_output_length)
        labels = tokenizer(text_target=input_label, truncation=True, max_length=max_output_length)

        inputs_labels = [i + [tokenizer.eos_token_id] + l + [tokenizer.eos_token_id]
                         for i, l in zip(inputs.input_ids, labels.input_ids)]
        labels_ignored = [[-100]*(len(i)+1) + l + [tokenizer.eos_token_id]
                          for i, l in zip(inputs.input_ids, labels.input_ids)]

        return {"input_ids": inputs_labels, "labels": labels_ignored}

    ds_train = ds_train.map(preprocess, batched=True, fn_kwargs={"label_col": train_label_col})
    ds_valid = ds_valid.map(preprocess, batched=True, fn_kwargs={"label_col": valid_label_col})
    ds_train = ds_train.shuffle(seed=0)

    callbacks = []
    if early_stopping_patience is not None:
        early_stopping = transformers.EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
        callbacks.append(early_stopping)

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=f"{checkpoint_dir}/{wandb.run.name}",
        # output_dir=f"{checkpoint_dir}",
        learning_rate=learning_rate,
        do_train=True,
        do_eval=True,
        warmup_steps=1000,
        max_steps=1_000_000,
        optim=optim,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=eval_batch_size,
        eval_accumulation_steps=1,
        logging_steps=10,
        eval_steps=eval_steps,
        save_steps=save_steps,
        eval_strategy="steps",
        bf16=True,
        bf16_full_eval=True,
        predict_with_generate=True,
        gradient_checkpointing=True,
        generation_max_length=max_output_length,
        include_inputs_for_metrics=True,
        report_to="wandb",
        metric_for_best_model="avg_correct_results",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=save_total_limit,
    )

    metrics = numllama.metrics.MonitorMetrics(
        tokenizer=tokenizer,
        log_predictions=True,
        eval_ds_inputs=ds_valid["input_ids"],
        source_ds_col=ds_valid["source_ds"],
    )

    trainer = utils.CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics,
        callbacks=callbacks,
    )
    model.build_num_latents()
    model.embedding.embs["num"].requires_grad = False
    trainer.train()


if __name__ == "__main__":
    try:
        app()
    except BaseException as e:
        print(traceback.format_exc())
        raise e
