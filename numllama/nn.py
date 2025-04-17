import abc
import contextlib
import copy
from typing import Iterator, Literal, Protocol, Self, cast

import torch
from torch import Tensor


def inv_sigmoid(x: Tensor) -> Tensor:
    """
    Compute the inverse of the sigmoid function.

    >>> x1 = torch.linspace(0.01, 0.99, 100)
    >>> x2 = inv_sigmoid(torch.sigmoid(x1))
    >>> torch.allclose(x1, x2, atol=1e-6)
    True
    """
    return -torch.log(1 / x - 1)


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


class Codec(torch.nn.Module, abc.ABC):
    """
    A torch module that can work in two directions: encode and decode.
    """

    def forward(self, x: Tensor, mode: Literal["encode", "decode"] | None = None) -> Tensor:
        match mode:
            case "encode":
                return self._encode(x)
            case "decode":
                return self._decode(x)
            case None:
                raise ValueError("Call .encode(x) or .decode(x) instead of direct __call__ or forward.")
        raise ValueError(f"Unknown mode {mode} during forward of {self.__class__.__name__}.")

    def encode(self, x: Tensor) -> Tensor:
        # encode and decode methods are just wrappers around forward.
        # Torch treats forward specially, so it's better to call it internally,
        # but having encode/decode methods as an interface is more intuitive.
        return self(x, "encode")

    def decode(self, x: Tensor) -> Tensor:
        return self(x, "decode")

    @abc.abstractmethod
    def _encode(self, x: Tensor) -> Tensor: ...

    @abc.abstractmethod
    def _decode(self, x: Tensor) -> Tensor: ...


class IEmbedding(Protocol):
    num_embeddings: int
    embedding_dim: int


class DualEmbedding(Codec, IEmbedding):
    """
    Torch module for both encoding tokens to latents and
    decoding latents to logits over tokens.
    """


class TokenEmbedding(DualEmbedding, torch.nn.Embedding):
    """
    Replacement for torch.nn.Embedding that can work in both directions.
    """

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.Embedding.forward(self, x)

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            raise NotImplementedError("max_norm not implemented.")
        return torch.nn.functional.linear(x, self.weight, None)


class OrthogonalLinear(torch.nn.Linear):
    """
    Linear layer with orthogonal weight matrix, making it mathematically invertible.
    """

    def reset_parameters(self) -> None:
        if self.in_features > self.out_features:
            raise ValueError(
                f"{self.__class__.__name__} requires in_features <= out_features, got {self.in_features=}, {self.out_features=}"
            )
        if torch.nn.utils.parametrizations.parametrize.is_parametrized(self):
            torch.nn.utils.parametrizations.parametrize.remove_parametrizations(self, "weight")
        super().reset_parameters()
        torch.nn.utils.parametrizations.orthogonal(self, "weight")


class SkipConnection(torch.nn.Module):
    """
    A simple residual skip connection wrapper around a module.
    """

    def __init__(self, fn: torch.nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)


class LearnableSkipConnection(torch.nn.Module):
    """
    Module that adds a learneable skip connection around a given module.

    It is initialized so that the output is almost equal to the input.
    """

    def __init__(
        self,
        fn: torch.nn.Module,
        balance: bool = False,
        gate_initialization: float = 0.05,
    ):
        super().__init__()
        self.fn = fn
        device = next((p.device for p in fn.parameters()), None)
        self.balance = balance
        self.gate = torch.nn.Parameter(torch.empty(1, device=device))
        self.gate_initialization = gate_initialization
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, gate_initialization: float | None = None) -> None:
        if gate_initialization is None:
            gate_initialization = self.gate_initialization
        g_preactivation = inv_sigmoid(torch.tensor(gate_initialization))
        assert torch.isclose(torch.sigmoid(g_preactivation), torch.tensor(gate_initialization), atol=1e-6)
        torch.nn.init.constant_(self.gate, g_preactivation.item())

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate)
        if self.balance:
            return x * (1 - g) + self.fn(x) * g
        else:
            return x * (1 - g) + self.fn(x)


def get_normalization_cls(normalization: str | None) -> type[torch.nn.Module] | None:
    if normalization is None:
        return None
    if normalization == "LayerNorm":
        return torch.nn.LayerNorm
    if normalization == "RMSNorm":
        return torch.nn.RMSNorm
    raise ValueError(f"Unknown normalization: {normalization}")


def get_linear_cls(linears_constraint: str | None) -> type[torch.nn.Linear]:
    if linears_constraint is None:
        return torch.nn.Linear
    if linears_constraint == "OrthogonalLinear":
        return OrthogonalLinear
    raise ValueError(f"Unknown linears_constraint: {linears_constraint}")


def get_skip_cls(
    use_skips: bool,
    skips_are_learnable: bool,
) -> type[LearnableSkipConnection] | type[SkipConnection] | None:
    if not use_skips:
        return None
    if skips_are_learnable:
        return LearnableSkipConnection
    return SkipConnection


def feedforward_backbone(
    model_dim: int,
    ff_dim: int,
    num_blocks: int,
    activation_fn: torch.nn.Module,
    use_skips: bool,
    skips_are_learnable: bool,
    normalization: Literal["LayerNorm", "RMSNorm"] | None,
    linears_constraint: Literal["orthogonal"] | None,
    dropout: float,
) -> torch.nn.Sequential:
    Linear = get_linear_cls(linears_constraint)
    Skip = get_skip_cls(use_skips, skips_are_learnable)
    Normalize = get_normalization_cls(normalization)

    blocks: list[torch.nn.Module] = []

    for _ in range(num_blocks):
        block: torch.nn.Module
        block = torch.nn.Sequential(
            Linear(model_dim, ff_dim),
            copy.deepcopy(activation_fn),
            Linear(ff_dim, model_dim),
            copy.deepcopy(activation_fn),
        )
        if dropout > 0:
            block.append(torch.nn.Dropout(dropout))
        if Skip is not None:
            block = Skip(block)
        blocks.append(block)
        if Normalize is not None:
            normalize = Normalize(model_dim)
            blocks.append(normalize)

    backbone: list[torch.nn.Module] = []
    backbone.append(Linear(model_dim, model_dim))
    if Normalize is not None:
        backbone.append(Normalize(model_dim))
    backbone.extend(blocks)
    backbone.append(Linear(model_dim, model_dim))

    return torch.nn.Sequential(*backbone)


class LatentEmbeddingNotBuiltError(ValueError): ...


class LatentEmbedding(DualEmbedding):
    """
    Layer that maps a (pre-computed) embedding into a latent space using a given encoder.
    The difference from using a normal embedding layer is that in this case, the learned
    parameters are shared between all tokens, whereas in a normal embedding layer, each
    token has *independent* parameters.

    Before inference, the layer can be merged (flattened) into a single embedding layer
    (once), so no overhead is introduced for inference.
    """

    def __init__(
        self,
        input_embedding: torch.nn.Embedding,
        encoder: torch.nn.Module,
        embedding_dim: int,
    ):
        super().__init__()
        self.input_embedding = input_embedding
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_embeddings = input_embedding.num_embeddings
        self.cache: Tensor | None = None

    def build_latents(self) -> None:
        """
        Build the latents by passing the input embedding through the encoder.
        """
        if self.input_embedding.max_norm is not None:
            raise NotImplementedError("max_norm not implemented.")
        self.cache = self.encoder(self.input_embedding.weight)

    def clear_latents(self) -> None:
        """
        Clear the latents to free up memory.
        """
        if self.cache is not None:
            del self.cache
            self.cache = None

    def train(self, mode: bool = True) -> Self:
        out = super().train(mode)
        if mode:
            self.clear_latents()
        else:
            # cache the latents for inference
            with torch.no_grad():
                self.build_latents()
        return out

    @contextlib.contextmanager
    def use_latents(self) -> Iterator[None]:
        try:
            self.build_latents()
            yield
        finally:
            self.clear_latents()

    def _encode(self, x: Tensor) -> Tensor:
        if self.cache is None:
            raise LatentEmbeddingNotBuiltError(
                "Latent weights are not available. Use `with use_latents()` or put model into eval mode."
            )
        return torch.nn.functional.embedding(x, self.cache)

    def _decode(self, x: Tensor) -> Tensor:
        if self.cache is None:
            raise LatentEmbeddingNotBuiltError(
                "Latent weights are not available. Use `with use_latents()` or put model into eval mode."
            )
        return torch.nn.functional.linear(x, self.cache, None)


class MultiEmbedding(DualEmbedding):
    def __init__(self, embeddings: dict[str, DualEmbedding]):
        super().__init__()
        self.embs = torch.nn.ModuleDict(embeddings)
        self.ranges = self._compute_ranges([emb.num_embeddings for emb in self.embeddings.values()])
        self.embedding_dim = next(e.embedding_dim for e in self.embeddings.values())
        self.num_embeddings = sum(e.num_embeddings for e in self.embeddings.values())
        if not all(e.embedding_dim == self.embedding_dim for e in self.embeddings.values()):
            raise ValueError("All embeddings must have the same dimension.")

    @staticmethod
    def _compute_ranges(num_tokens: list[int]) -> list[tuple[int, int]]:
        indices = torch.tensor([0] + num_tokens).cumsum(dim=0)
        ranges = torch.stack([indices[:-1], indices[1:]], dim=-1)
        return [(start, end) for start, end in ranges.tolist()]

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape + (self.embedding_dim,), device=x.device)
        for (start_idx, end_idx), emb in zip(self.ranges, self.embeddings.values()):
            mask = (x >= start_idx) & (x < end_idx)
            inputs = x[mask] - start_idx
            out[mask] = emb.encode(inputs)
        return out

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([emb.decode(x) for emb in self.embeddings.values()], dim=-1)

    @property
    def embeddings(self) -> dict[str, DualEmbedding]:
        # because self.embs is a torch.nn.ModuleList
        # which is not type-annotated
        return {k: cast(DualEmbedding, v) for k, v in self.embs.items()}


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def extra_repr(self):
        name = getattr(self.func, "__name__", type(self.func).__name__)
        if hasattr(self.func, "__self__"):
            name = self.func.__self__.__class__.__name__ + "." + name
        return f"fn={name}"


ParamGroup = dict[str, torch.nn.Parameter]


def sort_out_params(
    model: torch.nn.Module,
    special_classes: type | tuple[type, ...],
) -> tuple[ParamGroup, ParamGroup]:
    standard = {}
    specials = {}
    for module_name, module in model.named_modules():
        for attr_name, param in module.named_parameters(recurse=False):
            param_name = module_name + "." + attr_name
            param_name = param_name.removeprefix(".")  # for root module
            if isinstance(module, special_classes):
                specials[param_name] = param
            else:
                standard[param_name] = param
    return standard, specials
