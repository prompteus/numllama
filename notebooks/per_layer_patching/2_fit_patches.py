#!/usr/bin/env python
# coding: utf-8

# In[1]:


device = "cuda:0"


# ### Preliminaries

# In[2]:


import itertools
import random
import collections


import transformers
import torch
import tqdm.auto
from torch import Tensor


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


model_ckpt = "meta-llama/Llama-3.2-3B"
model = transformers.AutoModel.from_pretrained(model_ckpt).eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_ckpt)
model = model.half().to(device).eval()


basis_embs_sin = sinusoidal_encode(
    torch.arange(1000),
    min_value=0,
    max_value=1000,
    embedding_dim=model.config.hidden_size,
)


basis_embs = basis_embs_sin

for layer_idx in reversed(range(19, model.config.num_hidden_layers + 1)):
    torch.manual_seed(0)
    probe = ClassifierProbe(
        emb_dim=model.config.hidden_size,
        hidden_dim=100,
        basis=basis_embs,
        heldout_mask=torch.zeros(1000).bool(),
    ).to(device)
    best_ckpt = torch.load(f'best_ckpt.gt_probe.div.layer{layer_idx}.pt')
    probe.load_state_dict(best_ckpt)
    probe.requires_grad_(False)
    probe.eval()
    patches = torch.randn(1000, model.config.hidden_size, requires_grad=True, device=device)
    optimizer = torch.optim.SGD([patches, *probe.parameters()], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=100_000)
    target = torch.arange(1000, device=device, requires_grad=False)
    for idx in range(1, 200_000 + 1):
        optimizer.zero_grad()
        logits = probe(patches, False)
        loss = torch.nn.functional.cross_entropy(logits, target)
        if idx % 10_000 == 0:
            print(f'[{idx} / 200000]', layer_idx, loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
    torch.save(patches.detach(), f'patches.gt_probe.layer{layer_idx}.pt')
