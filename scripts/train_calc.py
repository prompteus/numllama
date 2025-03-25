import traceback
from typing import Optional

import datasets
import numpy as np
import transformers
import typer
import wandb

import numllama.metrics

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
)


@app.command()
def main(
    use_instructions_train: bool = True,
    use_instructions_val: bool = False,
    model_name: str = "google/t5-v1_1-large",
    limit_train_set_per_ds: int = -1,
    limit_val_set_per_ds: int = 200,
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
    max_output_length: int = 756,
    batch_size: int = 4,
    grad_accum: int = 8,
    eval_batch_size: int = 8,
    optim="adamw_torch",
    save_total_limit: int = 5,
    eval_steps: int = 100,  # = 16000, TODO
    save_steps: int = 100,  # = 16000,
    learning_rate: float = 2e-5,
    early_stopping_patience: Optional[int] = None,
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

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)

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
    instructions_ds = datasets.load_dataset("MU-NLPC/Calc-X_instructions")

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
        inputs = tokenizer(example[input_col], truncation=True)
        labels = tokenizer(text_target=example[label_col], truncation=True, max_length=max_output_length)
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels.input_ids,
        }

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

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics,
        callbacks=callbacks,
    )

    trainer.train()


if __name__ == "__main__":
    try:
        app()
    except BaseException as e:
        print(traceback.format_exc())
        raise e
