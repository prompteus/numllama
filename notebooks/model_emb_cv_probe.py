import gc
import json
import math
import pathlib
from typing import NamedTuple

import joblib
import safetensors
import safetensors.torch
import torch
import tqdm.auto
import transformers
import typer
from torch import Tensor

import numllama.nn

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
)


class ModelInfo(NamedTuple):
    tag: str
    name: str
    arange: torch.Tensor
    tokenizer: transformers.PreTrainedTokenizerFast | None
    ids: torch.Tensor | None
    embs: torch.Tensor
    display_name: str


def load_model(
    tag: str, model_name: str, arange: torch.Tensor, shuffle: bool, cache_file: pathlib.Path, display_name: str
) -> ModelInfo:
    tokenizer: transformers.PreTrainedTokenizerFast = transformers.AutoTokenizer.from_pretrained(model_name)
    ids = [tokenizer.convert_tokens_to_ids(str(i)) for i in arange.tolist()]

    if cache_file.exists():
        with safetensors.safe_open(cache_file, framework="pt") as f:
            embs = f.get_tensor("embs")
    else:
        model: transformers.PreTrainedModel = transformers.AutoModel.from_pretrained(model_name).eval()
        embs = model.get_input_embeddings().weight[ids].detach().clone()
        del model
        gc.collect()
        safetensors.torch.save_file({"embs": embs}, cache_file)

    if shuffle:
        embs = embs[torch.randperm(len(embs), generator=torch.Generator().manual_seed(0))]

    return ModelInfo(tag=tag, name=model_name, arange=arange, tokenizer=tokenizer, ids=None, embs=embs, display_name=display_name)


def load_standard_gaussian(tag: str, model_name: str, arange: Tensor, display_name: str) -> ModelInfo:
    model_dim = transformers.AutoConfig.from_pretrained(model_name).hidden_size
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    rng_gen = torch.Generator().manual_seed(0)
    shape = (arange.numel(), model_dim)
    embs = torch.normal(mean=torch.zeros(shape), std=torch.ones(shape), generator=rng_gen)
    return ModelInfo(
        tag=tag,
        name=model_name,
        arange=arange,
        tokenizer=tokenizer,
        ids=None,
        embs=embs,
        display_name=display_name,
    )


class ClassifierProbe(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        basis: torch.Tensor,
        heldout_mask: torch.Tensor,
    ):
        super().__init__()
        self.emb_to_latent = torch.nn.Linear(emb_dim, hidden_dim, bias=True)
        self.basis_to_latent = torch.nn.Linear(basis.shape[-1], hidden_dim, bias=True)
        self.basis_full: torch.nn.Buffer
        self.basis_train: torch.nn.Buffer
        self.register_buffer("basis_full", basis)
        self.register_buffer("basis_train", basis[~heldout_mask])

    def forward(self, x: Tensor, holdout_eval_tokens: bool) -> Tensor:
        latent_x = self.emb_to_latent(x)
        # during training, model learns to choose among only training tokens
        # but during eval, model must choose among all tokens
        # this means that the model is never exposed to the eval tokens during training
        if holdout_eval_tokens:
            choices = self.basis_train
        else:
            choices = self.basis_full
        latent_choices = self.basis_to_latent(choices)
        logits = latent_x @ latent_choices.T
        return logits


def fit_classifier_probe(
    model: ModelInfo,
    hidden_dim: int,
    basis: torch.Tensor,
    valid_mask: torch.Tensor,
    test_mask: torch.Tensor,
    device: str,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    num_steps: int = 50000,
    eval_every: int = 100,
    early_stop: bool = True,
    early_stopping_patience: int = 100,
    early_stop_delta: float = 1e-3,
) -> tuple[float, list[int], list[float]]:
    train_mask = ~(valid_mask | test_mask)
    probe = ClassifierProbe(
        heldout_mask=~train_mask,
        emb_dim=model.embs.shape[-1],
        hidden_dim=hidden_dim,
        basis=basis,
    ).to(device)
    train_x_embs = model.embs[train_mask].detach().to(device)
    train_y = torch.arange(len(train_x_embs), device=train_x_embs.device)
    valid_x_embs = model.embs[valid_mask].detach().to(device)
    valid_y = model.arange[valid_mask].detach().to(device)
    test_x_embs = model.embs[test_mask].detach().to(device)
    test_y = model.arange[test_mask].detach().to(device)
    optim = torch.optim.Adam(list(probe.parameters()), lr=lr, weight_decay=weight_decay)
    best_state = probe.state_dict()
    best_valid_loss = float("inf")
    valid_losses = []
    early_stop_counter = 0
    for i in tqdm.tqdm(range(num_steps), desc="Training", leave=False):
        probe.train()
        optim.zero_grad()
        logits = probe(train_x_embs, holdout_eval_tokens=True)
        loss = torch.nn.functional.cross_entropy(logits, train_y)
        loss.backward()
        optim.step()
        if i % eval_every == 0:
            with torch.no_grad():
                probe.eval()
                valid_logits = probe(valid_x_embs, holdout_eval_tokens=False)
                valid_loss = torch.nn.functional.cross_entropy(valid_logits, valid_y).item()
                valid_losses.append(valid_loss)
                if valid_loss + early_stop_delta < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_state = probe.state_dict()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop and early_stop_counter >= early_stopping_patience:
                        break
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        test_preds = probe(test_x_embs, holdout_eval_tokens=False).argmax(dim=-1)
        test_acc = (test_preds == test_y).float().mean().item()
        test_errs = test_y[test_preds != test_y].tolist()
    return test_acc, test_errs, valid_losses


def get_train_valid_mask(
    model_arange: Tensor,
    test_mask: Tensor,
    num_valids: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    train_mask = ~test_mask
    generator = torch.Generator().manual_seed(seed)
    valid_indices = model_arange[train_mask][torch.randperm(int(train_mask.sum().item()), generator=generator)][:num_valids]
    valid_mask = torch.zeros(len(model_arange), dtype=torch.bool, device=model_arange.device)
    valid_mask[valid_indices] = True
    train_mask[valid_indices] = False
    assert (train_mask & valid_mask).sum() == 0
    assert (train_mask & test_mask).sum() == 0
    assert (valid_mask & test_mask).sum() == 0
    assert (train_mask | valid_mask | test_mask).all()
    return train_mask, valid_mask


def get_classifier_probe_perf_cv(
    model: ModelInfo,
    basis: torch.Tensor,
    folds: int,
    device: str,
    max_processes: int,
    hidden_dim: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    num_valids: int = 50,
    num_steps: int = 50000,
    eval_every: int = 100,
    early_stop: bool = True,
    early_stopping_patience: int = 100,
    early_stop_delta: float = 1e-3,
) -> tuple[list[float], list[int], list[list[float]]]:
    fold_indices = torch.randint(
        low=0,
        high=folds,
        size=(len(model.arange),),
        generator=torch.Generator().manual_seed(0),
    )
    params = []

    for i in range(folds):
        test_mask = fold_indices == i
        train_mask, valid_mask = get_train_valid_mask(model.arange, test_mask, num_valids=num_valids, seed=i)
        params.append(
            {
                "model": model,
                "basis": basis,
                "test_mask": test_mask,
                "valid_mask": valid_mask,
                "device": device,
                "hidden_dim": hidden_dim,
                "lr": lr,
                "weight_decay": weight_decay,
                "num_steps": num_steps,
                "eval_every": eval_every,
                "early_stop": early_stop,
                "early_stopping_patience": early_stopping_patience,
                "early_stop_delta": early_stop_delta,
            }
        )

    if max_processes == 0:
        results = [fit_classifier_probe(**param) for param in tqdm.tqdm(params, desc="Training")]
    else:
        with joblib.Parallel(n_jobs=min(folds, max_processes)) as parallel:
            results = parallel(joblib.delayed(fit_classifier_probe)(**param) for param in params)

    test_accs, test_errs, valid_accs_hist = zip(*results)
    return (
        test_accs,
        sorted([err for fold_errs in test_errs for err in fold_errs]),
        valid_accs_hist,
    )


@app.command()
def main(
    model_nick: list[str],
    n_folds: int = 20,
    lr: float = 0.001,
    use_basis: str = "sin",
    output_dir: pathlib.Path = pathlib.Path("models_probe_outputs"),
):
    if use_basis not in ["sin", "bin"]:
        raise ValueError(f"Unknown basis {use_basis}. Must be one of ['sin', 'bin']")
    output_dir.mkdir(exist_ok=True, parents=True)
    names_ranges = [
        # ("mistral", "mistralai/Mistral-7B-v0.3", 10),
        # ("gemma", "google/gemma-3-1b-pt", 10),
        # ("qwen", "Qwen/Qwen3-0.6B", 10),
        ("olmo1b", "allenai/OLMo-2-0425-1B", 1000, "OLMo 2 1B"),
        ("olmo7b", "allenai/OLMo-2-1124-7B", 1000, "OLMo 2 7B"),
        ("olmo13b", "allenai/OLMo-2-1124-13B", 1000, "OLMo 2 13B"),
        ("olmo32b", "allenai/OLMo-2-0325-32B", 1000, "OLMo 2 32B"),
        ("llama1b", "meta-llama/Llama-3.2-1B", 1000, "Llama 3 1B"),
        ("llama3b", "meta-llama/Llama-3.2-3B", 1000, "Llama 3 3B"),
        ("llama8b", "meta-llama/Llama-3.1-8B", 1000, "Llama 3 8B"),
        ("llama70b", "meta-llama/Llama-3.1-70B", 1000, "Llama 3 70B"),
        ("phi", "microsoft/phi-4", 1000, "Phi 4 15B"),
        ("phi-mini", "microsoft/Phi-4-mini-instruct", 1000, "Phi 4 4B"),
    ]

    cache_dir = pathlib.Path("model_embs_cache")
    cache_dir.mkdir(exist_ok=True)

    models: dict[str, ModelInfo] = {}
    for nick, ckpt_name, arange, display_name in tqdm.tqdm(names_ranges):
        cache_filename = cache_dir / (ckpt_name.replace("/", "__") + ".safetensors")
        models[nick] = load_model(
            "pretrained",
            ckpt_name,
            torch.arange(arange),
            shuffle=False,
            cache_file=cache_filename,
            display_name=display_name,
        )
        models["shuffled_" + nick] = load_model(
            "shuffled",
            ckpt_name,
            torch.arange(arange),
            shuffle=True,
            cache_file=cache_filename,
            display_name=display_name,
        )
        models["sgn_" + nick] = load_standard_gaussian("sgn", ckpt_name, torch.arange(arange), display_name=display_name)

    for nick in model_nick:
        if nick not in models:
            raise ValueError(f"Model {nick} not found in the loaded models.")

    device = "cuda"

    for nick in model_nick:
        model = models[nick]
        output_filename = output_dir / f"{nick}_{use_basis}_{n_folds}-folds_{lr}lr.json"
        if output_filename.exists():
            print(f"Skipping {nick} ({model.display_name}) because {output_filename} already exists.")
            continue
        print(f"Processing {nick} ({model.display_name})")
        if use_basis == "bin":
            basis = numllama.nn.binary_encode(
                model.arange,
                min_value=0,
                max_value=len(model.arange),
                embedding_dim=math.ceil(math.log2(len(model.arange))),
            )
        elif use_basis == "sin":
            basis = numllama.nn.sinusoidal_encode(
                model.arange,
                min_value=0,
                max_value=len(model.arange),
                embedding_dim=model.embs.shape[-1],
            )
        else:
            raise ValueError(f"Unknown basis '{use_basis}'.")
        test_accs, test_errs, valid_loss_hist = get_classifier_probe_perf_cv(
            model=model,
            basis=basis,
            folds=n_folds,
            device=device,
            max_processes=20,
            hidden_dim=100,
            lr=lr,
            weight_decay=1e-3,
            num_valids=50,
            num_steps=50000,
            eval_every=100,
            early_stop=True,
            early_stopping_patience=50,
        )
        with open(output_filename, "w") as f:
            json.dump(
                {
                    "nick": nick,
                    "basis": use_basis,
                    "test_accs": test_accs,
                    "test_errs": test_errs,
                    "valid_loss_hist": valid_loss_hist,
                },
                f,
            )


if __name__ == "__main__":
    app()
