import collections
import itertools
import json
import math
import pathlib

import polars as pl
import safetensors
import safetensors.torch
import torch
import tqdm.auto
import transformers
import typer
from torch import Tensor

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    device: str = "cuda:6",
    dtype: str = "float32",
    template: str = "{operand_1}+{operand_2}=",
    train_size: int = 100_000,
    eval_size: int = 4096,
    ngram: int = 1,
    position_idcs: list[int] = [-2, -1],  # probe from the last two positions in the input sequence
    operand_idx: int = 1,  # try to decode the value of the second operand in the addition
    model_ckpt: str = "meta-llama/Llama-3.1-8B",
    model_bs: int = 256,
    save_checkpoints: bool = True,
    probe_hidden_dim: int = 100,
    max_steps: int = 10_000,
    eval_every_nth_step: int = 100,
    bs: int = 1024,
    early_stop: bool = True,
    early_stopping_patience: int = 10,
    early_stop_acc_delta: float = 0.005,
    lr: float = 5e-4,
    l1_reg: float = 1e-3,
) -> None:
    cli_params = locals().copy()

    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    assert isinstance(dtype, torch.dtype), f"Invalid dtype: {dtype}"

    typer.secho("Loading model and tokenizer...", fg=typer.colors.YELLOW)
    model = transformers.AutoModel.from_pretrained(model_ckpt).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_ckpt)
    model = model.half().to(device).eval()

    typer.secho("Preparing inputs", fg=typer.colors.YELLOW)
    all_values = torch.arange(0, 1000)
    mask = torch.rand(len(all_values), generator=torch.Generator().manual_seed(0))
    train_mask = mask < 0.9
    valid_mask = ~train_mask & (mask < 0.95)
    test_mask = ~train_mask & ~valid_mask

    train_values = all_values[train_mask]
    valid_values = all_values[valid_mask]
    test_values = all_values[test_mask]

    x_values_train = make_tuples(train_values, ngram, train_size, torch.Generator().manual_seed(0))
    x_values_valid = make_tuples(valid_values, ngram, eval_size, torch.Generator().manual_seed(0))
    x_values_test = make_tuples(test_values, ngram, eval_size, torch.Generator().manual_seed(0))

    typer.secho("Computing hidden states", fg=typer.colors.YELLOW)
    train_hidden_states = get_hidden_states(
        model, tokenizer, [make_str_input(template, val) for val in x_values_train], position_idcs, model_bs
    )
    valid_hidden_states = get_hidden_states(
        model, tokenizer, [make_str_input(template, val) for val in x_values_valid], position_idcs, model_bs
    )
    test_hidden_states = get_hidden_states(
        model, tokenizer, [make_str_input(template, val) for val in x_values_test], position_idcs, model_bs
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    basis_embs = sinusoidal_encode(
        torch.arange(1000),
        min_value=0,
        max_value=1000,
        embedding_dim=train_hidden_states[-1][0].shape[-1],
    )

    pos_idcs = sorted(list(train_hidden_states.keys()))
    block_idcs = list(train_hidden_states[pos_idcs[0]].keys())
    suboperand_idcs = list(range(ngram))
    todo = list(itertools.product(pos_idcs, block_idcs, suboperand_idcs))

    results_dir = pathlib.Path("multitoken_probes/")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"results_{ngram=}_{operand_idx=}.jsonl"
    ckpt_dir = results_dir / "ckpts"
    ckpt_dir.mkdir(exist_ok=True)
    ckpts = []

    typer.secho("Training probes", fg=typer.colors.YELLOW)
    progress = tqdm.auto.tqdm(todo, desc="Training probes")
    for pos_idx, block_idx, suboperand_idx in progress:
        args = {
            "ngram": ngram,
            "pos_idx": pos_idx,
            "block_idx": block_idx,
            "operand_idx": operand_idx,
            "suboperand_idx": suboperand_idx,
        }
        progress.set_postfix(**args)

        if results_file.exists():
            if results_file.stat().st_size != 0:
                results_df = pl.read_ndjson(results_file)
                check = results_df.filter(
                    (pl.col("ngram") == ngram)
                    & (pl.col("pos_idx") == pos_idx)
                    & (pl.col("block_idx") == block_idx)
                    & (pl.col("operand_idx") == operand_idx)
                    & (pl.col("suboperand_idx") == suboperand_idx)
                )
                if len(check) > 0:
                    print(f"Skipping {args} as it already exists in results file.")
                    continue

        ckpt_path = ckpt_dir / f"probe_{ngram=}_{operand_idx=}_{suboperand_idx=}_{pos_idx=}_{block_idx=}.safetensors"
        ckpts.append(ckpt_path)

        torch.manual_seed(0)
        probe = ClassifierProbe(
            emb_dim=train_hidden_states[pos_idx][block_idx].shape[-1],
            hidden_dim=probe_hidden_dim,
            basis=basis_embs,
            heldout_mask=test_mask,
        ).to(device, dtype)

        ckpt, metrics = train_probe(
            probe,
            train_x=train_hidden_states[pos_idx][block_idx],
            train_y=x_values_train[:, operand_idx, suboperand_idx],
            valid_x=valid_hidden_states[pos_idx][block_idx],
            valid_y=x_values_valid[:, operand_idx, suboperand_idx],
            test_x=test_hidden_states[pos_idx][block_idx],
            test_y=x_values_test[:, operand_idx, suboperand_idx],
            max_steps=max_steps,
            eval_every_nth_step=eval_every_nth_step,
            bs=bs,
            early_stop=early_stop,
            early_stopping_patience=early_stopping_patience,
            early_stop_acc_delta=early_stop_acc_delta,
            lr=lr,
            l1_reg=l1_reg,
            device=device,
            dtype=dtype,
        )

        if save_checkpoints:
            ckpt.pop("basis")
            safetensors.torch.save_file(ckpt, ckpt_path)

        with open(results_file, "a") as f:
            json_results = json.dumps({**args, **metrics, **cli_params})
            f.write(json_results + "\n")

    typer.secho("Training completed", fg=typer.colors.GREEN)
    typer.secho(f"Results saved to {results_file}", fg=typer.colors.GREEN)

    if save_checkpoints and len(ckpts) > 0:
        typer.secho("Trying to load a checkpoint", fg=typer.colors.YELLOW)
        with safetensors.safe_open(ckpts[0], framework="pt") as f:
            state = {k: f.get_tensor(k) for k in f.keys()}
        typer.secho(f"Loaded checkpoint with keys: {list(state.keys())}", fg=typer.colors.GREEN)


def sinusoidal_encode(
    x: Tensor,
    embedding_dim: int,
    min_value: int,
    max_value: int,
    use_l2_norm: bool = False,
    norm_const: float | None = None,
) -> Tensor:
    """
    Encodes a tensor of numbers into a sinusoidal representation, inspired by how absolute positional
    encoding works in transformers.

    The encoding is an evaluation of a sine and cosine function at different frequencies, where the
    frequency is determined by the embedding dimension and the allowed range of the input values.

    >>> sinusoidal_encode(
    ...     torch.tensor([-5, 2, 1, 0]),
    ...     embedding_dim=6,
    ...     min_value=-5,
    ...     max_value=5,
    ... )
    tensor([[ 0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  1.0000],
            [ 0.6570,  0.7539, -0.1073, -0.9942,  0.9980,  0.0627],
            [-0.2794,  0.9602,  0.3491, -0.9371,  0.9616,  0.2746],
            [-0.9589,  0.2837,  0.7317, -0.6816,  0.8806,  0.4738]])
    """

    if embedding_dim % 2 != 0 and not use_l2_norm:
        raise ValueError("Embedding dimension must be even")

    if use_l2_norm:
        if embedding_dim % 2 == 0:
            reserved_dim = 2
        else:
            reserved_dim = 1
        embedding_dim -= reserved_dim
    else:
        reserved_dim = 0  # will not be used

    domain = max_value - min_value
    y_shape = x.shape + (embedding_dim,)
    y = torch.zeros(y_shape, device=x.device)
    even_indices = torch.arange(0, embedding_dim, 2)
    log_term = torch.log(torch.tensor(domain)) / embedding_dim
    div_term = torch.exp(even_indices * -log_term)
    x = x - min_value
    values = x.unsqueeze(-1).float() * div_term
    y[..., 0::2] = torch.sin(values)
    y[..., 1::2] = torch.cos(values)

    if use_l2_norm:
        y = torch.cat([y, torch.ones_like(y[..., :reserved_dim])], dim=-1)
        y /= y.norm(dim=-1, keepdim=True, p=2)

    if norm_const is not None:
        y *= norm_const

    return y


class ClassifierProbe(torch.nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, basis: torch.Tensor, heldout_mask: torch.Tensor):
        super().__init__()
        self.emb_to_latent = torch.nn.Linear(emb_dim, hidden_dim, bias=True)
        self.basis_to_latent = torch.nn.Linear(basis.shape[-1], hidden_dim, bias=True)
        self.basis: torch.nn.Buffer
        self.heldout_mask: torch.nn.Buffer
        self.register_buffer("basis", basis)
        self.register_buffer("heldout_mask", heldout_mask)

    def forward(self, x: Tensor, holdout_eval_tokens: bool) -> Tensor:
        latent_x = self.emb_to_latent(x)
        # during training, model learns to choose among only training tokens
        # but during eval, model must choose among all tokens
        # this means that the model is never exposed to the eval tokens during training
        latent_choices = self.basis_to_latent(self.basis)
        logits = latent_x @ latent_choices.T
        if holdout_eval_tokens:
            logits[:, self.heldout_mask] = float("-inf")
        return logits


def train_probe(
    probe: ClassifierProbe,
    train_x: Tensor,
    train_y: Tensor,
    valid_x: Tensor,
    valid_y: Tensor,
    test_x: Tensor,
    test_y: Tensor,
    max_steps: int,
    eval_every_nth_step: int,
    bs: int,
    early_stop: bool,
    early_stopping_patience: int,
    early_stop_acc_delta: float,
    lr: float,
    l1_reg: float,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Tensor], dict[str, float]]:
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    best_valid_loss = float("inf")
    best_valid_acc = -1
    best_ckpt = None

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
    )

    valid_x = valid_x.to(device, dtype)
    valid_y = valid_y.to(device)
    test_x = test_x.to(device, dtype)
    test_y = test_y.to(device)

    early_stop_counter = 0
    step = 0
    best_step = None
    tqdm_bar = tqdm.auto.tqdm(total=max_steps, desc="Training probe", leave=True)
    while step < max_steps:
        for x, y in train_loader:
            tqdm_bar.update(1)
            probe.train()
            optimizer.zero_grad()
            x, y = x.to(device, dtype), y.to(device)
            logits = probe(x, holdout_eval_tokens=True)
            loss = torch.nn.functional.cross_entropy(logits, y) + l1_reg * sum(p.abs().sum() for p in probe.parameters())
            loss.backward()
            optimizer.step()
            train_acc = (logits.argmax(dim=-1) == y).float().mean().item()
            step += 1

            if step % eval_every_nth_step == 0:
                probe.eval()
                with torch.no_grad():
                    valid_logits = probe(valid_x, holdout_eval_tokens=False)
                    valid_loss = torch.nn.functional.cross_entropy(valid_logits, valid_y)
                    valid_acc = (valid_logits.argmax(dim=-1) == valid_y).float().mean().item()
                    best_valid_loss = min(best_valid_loss, valid_loss.item())
                    if valid_acc > best_valid_acc:
                        best_valid_acc = valid_acc
                        best_step = step
                        best_ckpt = {k: v.cpu() for k, v in probe.state_dict().items()}
                tqdm_bar.set_postfix(
                    {
                        "train_loss": f"{loss.item():.3f}",
                        "train_acc": f"{train_acc:.0%}",
                        "valid_loss": f"{valid_loss.item():.3f}",
                        "valid_acc": f"{valid_acc:.0%}",
                        "best_valid_acc": f"{best_valid_acc:.0%}",
                        "early_stop_counter": early_stop_counter,
                    }
                )
                if early_stop:
                    if valid_acc > best_valid_acc + early_stop_acc_delta:
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                    if early_stop_counter > early_stopping_patience:
                        break
            if step >= max_steps:
                break
        if step >= max_steps or (early_stop and early_stop_counter > early_stopping_patience):
            break

    probe.load_state_dict(best_ckpt)
    probe.eval()

    with torch.no_grad():
        test_logits = probe(test_x, holdout_eval_tokens=False)
        test_loss = torch.nn.functional.cross_entropy(test_logits, test_y)
        test_acc = (test_logits.argmax(dim=-1) == test_y).float().mean().item()

    metrics = {
        "last_train_loss": loss.item(),
        "last_train_acc": train_acc,
        "best_valid_loss": best_valid_loss,
        "best_valid_acc": best_valid_acc,
        "test_loss": test_loss.item(),
        "test_acc": test_acc,
        "best_step": best_step,
        "last_step": step,
    }
    tqdm_bar.close()
    return best_ckpt, metrics


def get_hidden_states(
    model, tokenizer, str_inputs: list[str], position_idcs: list[int], batch_size: int
) -> collections.defaultdict[int, Tensor]:
    model.eval()
    hidden_states = {pos_idx: collections.defaultdict(list) for pos_idx in position_idcs}
    n_batches = math.ceil(len(str_inputs) / batch_size)
    batches = itertools.batched(str_inputs, n=batch_size)
    with torch.no_grad():
        for batch_str in tqdm.auto.tqdm(batches, total=n_batches):
            batch_inputs = tokenizer(batch_str, return_tensors="pt")
            hidden_reprs = model(**batch_inputs.to(model.device), output_hidden_states=True).hidden_states
            for layer_idx, hidden_state in enumerate(hidden_reprs):
                for pos_idx in position_idcs:
                    hidden_states[pos_idx][layer_idx].extend(hidden_state[:, pos_idx, :].detach().cpu())
    return {pos_idx: {k: torch.stack(v) for k, v in hidden_states[pos_idx].items()} for pos_idx in position_idcs}


def stringify_num(nums: list[int]) -> str:
    return str(nums[0]) + "".join(str(n).zfill(3) for n in nums[1:])


def make_str_input(template, nums: list[list[int]] | Tensor) -> str:
    if isinstance(nums, Tensor):
        nums = nums.tolist()
    arg1, arg2 = nums
    return template.format(
        operand_1=stringify_num(arg1),
        operand_2=stringify_num(arg2),
    )


def make_tuples(allowed_values: Tensor, ngram: int, n_tuples: int, rng: torch.Generator) -> Tensor:
    size = (n_tuples, ngram * 2)
    return allowed_values[torch.randint(len(allowed_values), size=size, generator=rng)].reshape(-1, 2, ngram)


if __name__ == "__main__":
    app()
