import itertools
import math
import random
from typing import Any, Iterator, Self

import hydra
import lightning
import lightning.pytorch.utilities
import pydantic
import torch
import torchmetrics
from torch import Tensor

import numllama.nn


class OptimizerConfig(pydantic.BaseModel):
    opt_type: str
    lr: float
    disable_wdecay_on: list[str] | None = None

    class Config:
        extra = "allow"


class AvgProbOfCorrect(torchmetrics.Metric):
    higher_is_better = True

    def __init__(self, expect_logits: bool = True) -> None:
        super().__init__()
        self.sum: Tensor
        self.n: Tensor
        self.add_state("sum", torch.tensor(0.0, dtype=torch.get_default_dtype()), dist_reduce_fx="sum")
        self.add_state("n", torch.tensor(0.0, dtype=torch.get_default_dtype()), dist_reduce_fx="sum")
        self.expect_logits = expect_logits

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = preds.reshape(-1, preds.size(-1))
        target = target.reshape(-1)
        if self.expect_logits:
            probs = torch.softmax(preds, dim=-1)
        else:
            probs = preds
        probs_of_correct = probs[torch.arange(len(target)), target]
        self.sum += probs_of_correct.sum()
        self.n += len(target)

    def compute(self) -> Tensor:
        return self.sum / self.n


class LogitAbsMax(torchmetrics.Metric):
    def __init__(self) -> None:
        super().__init__()
        self.abs_max: Tensor
        self.add_state("abs_max", torch.tensor(-torch.inf, dtype=torch.get_default_dtype()), dist_reduce_fx="max")

    def update(self, logits: Tensor, target: Tensor) -> None:
        self.abs_max = torch.max(self.abs_max, logits.abs().max())

    def compute(self) -> Tensor:
        return self.abs_max


class AdditionTransformerConfig(pydantic.BaseModel):
    model_dim: int
    ff_dim: int
    num_blocks: int
    num_heads: int
    activation_fn: dict[str, Any]
    dropout: float
    input_dropout: float


class NumEmbeddingConfig(pydantic.BaseModel):
    min_value: int
    max_value: int
    embedding_dim: int
    use_l2_norm: bool
    norm_const: float | None


class LatentEmbeddingAdditionModel(torch.nn.Module):
    def __init__(
        self,
        embedding_config: NumEmbeddingConfig,
        num_encoder_config: dict[str, Any],
        transformer_config: AdditionTransformerConfig,
    ):
        super().__init__()
        self.embedding_config = embedding_config
        self.num_encoder_config = num_encoder_config
        self.transformer_config = transformer_config

        arange = torch.arange(embedding_config.min_value, embedding_config.max_value + 1)
        input_embedding = numllama.nn.TokenEmbedding.from_pretrained(
            numllama.nn.sinusoidal_encode(x=arange, **embedding_config.model_dump()),
            freeze=True,
        )
        try:
            num_encoder = hydra.utils.instantiate(num_encoder_config)
        except ModuleNotFoundError:
            num_encoder_config.pop("_target_")
            num_encoder = numllama.nn.feedforward_backbone(**num_encoder_config)

        if not isinstance(num_encoder, torch.nn.Module):
            raise ValueError(f"num_encoder must be a torch.nn.Module. Got {type(num_encoder)}")

        self.embedding = numllama.nn.LatentEmbedding(
            input_embedding=input_embedding,
            encoder=num_encoder,
            embedding_dim=embedding_config.embedding_dim,
        )

        # TODO positional embedding is not used currently, but it should be
        self.positional_embedding = torch.nn.Embedding(3, transformer_config.model_dim)
        self.output_token_embedding = torch.nn.Parameter(torch.empty(transformer_config.model_dim))

        self.to_model_dim = torch.nn.Linear(input_embedding.embedding_dim, transformer_config.model_dim)
        self.from_model_dim = torch.nn.Linear(transformer_config.model_dim, input_embedding.embedding_dim)
        self.input_dropout = torch.nn.Dropout(transformer_config.input_dropout)
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=transformer_config.model_dim,
                nhead=transformer_config.num_heads,
                dim_feedforward=transformer_config.ff_dim,
                dropout=transformer_config.dropout,
                activation=hydra.utils.instantiate(transformer_config.activation_fn),
                batch_first=True,
            ),
            num_layers=transformer_config.num_blocks,
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.normal_(self.output_token_embedding, std=0.01)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x3: Tensor
        with self.embedding.build_latents():
            x = torch.stack([x1, x2], dim=-1)
            x = self.embedding.encode(self.value_to_idx(x))
            x = self.to_model_dim(x)
            x3 = self.output_token_embedding.unsqueeze(0).expand_as(x[..., :1, :])
            x = torch.concat([x, x3], dim=-2)
            x = self.input_dropout(x)
            x = self.transformer(x)
            x3 = x[..., -1, :]
            x3 = self.from_model_dim(x3)
            logits = self.embedding.decode(x3)
        return logits

    def idx_to_value(self, idx: Tensor) -> Tensor:
        return idx + self.embedding_config.min_value

    def value_to_idx(self, value: Tensor) -> Tensor:
        return value - self.embedding_config.min_value


class AdditionLightning(lightning.LightningModule):
    def __init__(
        self,
        embedding_config: NumEmbeddingConfig,
        num_encoder_config: dict[str, Any],
        transformer_config: AdditionTransformerConfig,
        optimizer_config: OptimizerConfig,
        loss_config: dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.opt_config = optimizer_config

        self.model = LatentEmbeddingAdditionModel(embedding_config, num_encoder_config, transformer_config)
        self.loss_fn = hydra.utils.instantiate(loss_config)

        num_classes = embedding_config.max_value - embedding_config.min_value + 1
        self.train_clas_metrics = self.get_classification_metrics("train/", num_classes)
        self.valid_clas_metrics = self.get_classification_metrics("valid/", num_classes)
        self.tests_clas_metrics = self.get_classification_metrics("tests/", num_classes)
        self.train_regr_metrics = self.get_regression_metrics("train/")
        self.valid_regr_metrics = self.get_regression_metrics("valid/")
        self.tests_regr_metrics = self.get_regression_metrics("tests/")

    def get_classification_metrics(self, prefix: str, num_classes: int) -> torchmetrics.MetricCollection:
        metrics = {
            "avg_prob_of_correct": AvgProbOfCorrect(expect_logits=True),
            "logit_abs_max": LogitAbsMax(),
            "acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1),
        }
        k = 10
        while k <= num_classes / 10:
            metrics[f"acc_top{k}"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=k)
            k *= 10
        return torchmetrics.MetricCollection(metrics, prefix=prefix)

    def get_regression_metrics(self, prefix: str) -> torchmetrics.MetricCollection:
        return torchmetrics.MetricCollection(
            {
                "abs_err": torchmetrics.MeanAbsoluteError(),
            },
            prefix=prefix,
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        x1, x2, y = batch
        y_idx = self.model.value_to_idx(y)
        logits: Tensor = self(x1, x2)
        y_pred_idx = logits.argmax(dim=-1)
        y_pred = self.model.idx_to_value(y_pred_idx)
        loss: Tensor = self.loss_fn(logits, y_idx)
        self.log("train/loss", loss)
        self.train_clas_metrics(logits, y_idx)
        self.train_regr_metrics(y_pred, y)
        self.log_dict(self.train_clas_metrics)
        self.log_dict(self.train_regr_metrics)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        x1, x2, y = batch
        y_idx = self.model.value_to_idx(y)
        logits: Tensor = self(x1, x2)
        y_pred_idx = logits.argmax(dim=-1)
        y_pred = self.model.idx_to_value(y_pred_idx)
        self.valid_clas_metrics(logits, y_idx)
        self.valid_regr_metrics(y_pred, y)
        self.log_dict(self.valid_clas_metrics)
        self.log_dict(self.valid_regr_metrics)

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        x1, x2, y = batch
        y_idx = self.model.value_to_idx(y)
        logits: Tensor = self(x1, x2)
        y_pred_idx = logits.argmax(dim=-1)
        y_pred = self.model.idx_to_value(y_pred_idx)
        self.tests_clas_metrics(logits, y_idx)
        self.tests_regr_metrics(y_pred, y)
        self.log_dict(self.tests_clas_metrics)
        self.log_dict(self.tests_regr_metrics)

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        x1, x2 = batch
        logits: Tensor = self(x1, x2)
        return logits

    def configure_optimizers(self):
        opt = getattr(torch.optim, self.opt_config.opt_type)
        if self.opt_config.disable_wdecay_on:
            no_decay_classes = tuple(hydra.utils.get_class(name) for name in self.opt_config.disable_wdecay_on)
            standard_params, no_decay_params = numllama.nn.sort_out_params(self.model, no_decay_classes)
            print(f"optimizer conf - standard params: {len(standard_params)}, no decay params: {len(no_decay_params)}")
            return opt(
                [
                    {"params": standard_params.values()},
                    {"params": no_decay_params.values(), "weight_decay": 0.0},
                ],
                lr=self.opt_config.lr,
                **self.opt_config.__pydantic_extra__,
            )
        return opt(
            self.model.parameters(),
            lr=self.opt_config.lr,
            **self.opt_config.__pydantic_extra__,
        )


class AdditionDataConfig(pydantic.BaseModel):
    """
    Configuration class for a syntetic dataset for summing two numbers.
    """

    min_value: int
    max_value: int
    gap_size: pydantic.PositiveInt
    gap_frac: pydantic.PositiveFloat
    train_ds_size: pydantic.PositiveInt | None
    valid_ds_size: pydantic.PositiveInt
    tests_ds_size: pydantic.PositiveInt
    train_batch_size: pydantic.PositiveInt
    eval_batch_size: pydantic.PositiveInt
    seed: int

    @property
    def domain_size(self) -> int:
        return self.max_value - self.min_value + 1

    @property
    def num_gaps(self) -> int:
        combined_gap_size = self.domain_size * self.gap_frac
        return math.ceil(combined_gap_size / self.gap_size)

    @pydantic.model_validator(mode="after")
    def check(self) -> Self:
        if self.min_value >= self.max_value:
            raise ValueError("max_value must be greater than min_value")
        if not 0 < self.gap_frac < 1:
            raise ValueError("gap_frac must be in (0, 1)")
        return self


class AdditionDataset(torch.utils.data.IterableDataset):
    """
    Infinite iterable dataset for summing two numbers that are randomly selected
    from a list of allowed numbers.
    """

    def __init__(self, nums: torch.Tensor, seed: int):
        super().__init__()
        self.nums = nums
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        # print(worker_info)
        worker_rank = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        # print("Worker rank %s, num_workers %s" % (worker_rank, num_workers))

        # seed must include worker_rank, otherwise different generators will produce same outputs
        random_gen = random.Random(self.seed + worker_rank)
        i = 0
        while True:
            if (i % num_workers) == worker_rank:
                x1_idx = random_gen.randint(0, len(self.nums) - 1)
                x2_idx = random_gen.randint(0, len(self.nums) - 1)
                x1 = self.nums[x1_idx]
                x2 = self.nums[x2_idx]
                y = x1 + x2
                yield x1, x2, y
            i += 1


class AdditionDataModule(lightning.LightningDataModule):
    """
    Data module that generates a train, validation and test dataset for summing two numbers.
    """

    def __init__(
        self,
        data_config: AdditionDataConfig,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.config = data_config
        random_gen = random.Random(data_config.seed)
        random_gen_torch = torch.Generator().manual_seed(data_config.seed)
        all_nums = torch.arange(self.config.min_value, self.config.max_value + 1)
        sections = all_nums.tensor_split(self.config.num_gaps)
        train_intervals = []
        heldout_intervals = []
        for section in sections:
            gap_start = random_gen.randint(0, len(section) - self.config.gap_size)
            gap_end = gap_start + self.config.gap_size
            heldout_intervals.append(section[gap_start:gap_end])
            train_intervals.append(section[:gap_start])
            train_intervals.append(section[gap_end:])
        self.train_nums = torch.cat(train_intervals)
        heldout_nums = torch.cat(heldout_intervals)
        heldout_nums = heldout_nums[torch.randperm(len(heldout_nums), generator=random_gen_torch)]
        self.valid_nums, self.tests_nums = heldout_nums.tensor_split(2)
        train_dataset = AdditionDataset(self.train_nums, seed=self.config.seed)
        valid_dataset = AdditionDataset(self.valid_nums, seed=self.config.seed)
        tests_dataset = AdditionDataset(self.tests_nums, seed=self.config.seed)
        self.train_dataset: torch.utils.data.Dataset
        if self.config.train_ds_size is None:
            # infinite stream of data
            self.train_dataset = train_dataset
        else:
            self.train_dataset = ListDataset(list(itertools.islice(train_dataset, self.config.train_ds_size)))
        self.valid_dataset = ListDataset(list(itertools.islice(valid_dataset, self.config.valid_ds_size)))
        self.tests_dataset = ListDataset(list(itertools.islice(tests_dataset, self.config.tests_ds_size)))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            num_workers=10,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.config.eval_batch_size,
            num_workers=10,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.tests_dataset, batch_size=self.config.eval_batch_size, num_workers=10)


class ListDataset[X](torch.utils.data.Dataset):
    def __init__(self, data: list[X]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> X:
        return self.data[idx]
