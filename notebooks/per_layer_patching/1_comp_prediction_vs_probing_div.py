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


# In[3]:


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


def binary_encode(
    x: Tensor,
    embedding_dim: int,
    min_value: int | float,
    max_value: int | float,
    use_l2_norm: bool = False,
    norm_const: float | None = None,
) -> Tensor:
    y = torch.zeros(x.shape + (embedding_dim,), device=x.device)
    reserve_dim = 0 if not use_l2_norm else 1
    x = x - min_value
    maximum = x.max()
    for i in range(embedding_dim - reserve_dim):
        coeff = 2**i
        if maximum < coeff:
            break
        y[..., -i - 1] = torch.floor(x / coeff) % 2
        x = x - coeff * y[..., -i - 1]
    if use_l2_norm:
        y = torch.cat([y, torch.ones_like(y[..., :reserve_dim])], dim=-1)
        y /= y.norm(dim=-1, keepdim=True, p=2)
    if norm_const is not None:
        y *= norm_const
    return y


# ### Prepare model and data

# In[4]:


model_ckpt = "meta-llama/Llama-3.2-3B"
model = transformers.AutoModel.from_pretrained(model_ckpt).eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_ckpt)
model = model.half().to(device).eval()


# In[5]:


all_values = torch.arange(0, 1000)
mask = torch.rand(len(all_values), generator=torch.Generator().manual_seed(0))
train_mask = mask < 0.9
valid_mask = ~train_mask & (mask < 0.95)
test_mask = ~train_mask & ~valid_mask

train_values = all_values[train_mask]
valid_values = all_values[valid_mask]
test_values = all_values[test_mask]


# In[6]:


all_inputs = [(x1, x2) for x1, x2 in itertools.product(all_values.tolist(), repeat=2) if x2 and x1 / x2 < 1000]
train_values_set = set(train_values.tolist())
valid_values_set = set(valid_values.tolist())
test_values_set = set(test_values.tolist())

all_inputs_add = [(x1, x2) for x1, x2 in itertools.product(all_values.tolist(), repeat=2) if x1 + x2 < 1000]
train_values_set = set(train_values.tolist())
valid_values_set = set(valid_values.tolist())
test_values_set = set(test_values.tolist())

train_inputs = [(x1, x2) for x1, x2 in all_inputs if x1 / x2 in train_values_set]
train_inputs_add = [(x1, x2) for x1, x2 in all_inputs_add if x1 + x2 in train_values_set]
valid_inputs = [(x1, x2) for x1, x2 in all_inputs if x1 / x2 in valid_values_set]
valid_inputs_add = [(x1, x2) for x1, x2 in all_inputs_add if x1 + x2 in valid_values_set]
test_inputs = [(x1, x2) for x1, x2 in all_inputs if x1 / x2 in test_values_set]
test_inputs_add = [(x1, x2) for x1, x2 in all_inputs_add if x1 + x2 in test_values_set]

# sanity check
assert set(train_inputs) & set(valid_inputs) == set()
assert set(train_inputs) & set(test_inputs) == set()
assert set(valid_inputs) & set(test_inputs) == set()

assert set(train_inputs_add) & set(valid_inputs_add) == set()
assert set(train_inputs_add) & set(test_inputs_add) == set()
assert set(valid_inputs_add) & set(test_inputs_add) == set()

random.seed(0)
random.shuffle(train_inputs)
random.shuffle(valid_inputs)
random.shuffle(test_inputs)

random.shuffle(train_inputs_add)
random.shuffle(valid_inputs_add)
random.shuffle(test_inputs_add)

valid_size = 4096
train_size = 50_000  # TODO: change back to 100_000
train_inputs = train_inputs[:train_size]
train_inputs_add = train_inputs_add[:train_size]
valid_inputs = valid_inputs[:valid_size]


# In[7]:


num_templates = 5  # TODO revert

def make_str_input(operands: tuple[int, int] | list[int], template_idx: int = 1) -> str:
    x1, x2 = operands
    options = [
        f"{x1} divided by {x2} is ",
        f"{x1} divided by {x2} equals to ",
        f"{x1} / {x2} = ",
        f"A division of {x1} by {x2} equals to ",
        f"A result of dividing {x1} by {x2} is ",
    ]
    # assert num_templates == len(options)
    # return f"{x1} times {x2} is "  # 0.78
    # return f"{x1} multiplied by {x2} is "  # 90.38
    return options[template_idx]

make_str_input((3, 500)), make_str_input((3, 0))


# In[8]:


def make_str_input_add(operands: tuple[int, int] | list[int]) -> str:
    x1, x2 = operands
    return f"{x1} plus {x2} is equal to "  # TODO: maybe switch back

make_str_input_add((3, 500)), make_str_input_add((3, 0))


# In[9]:


def get_hidden_states_and_preds(model, str_inputs: list[str], batch_size: int) -> tuple[dict[int, Tensor], list[str]]:
    model.eval()
    hidden_states = collections.defaultdict(list)
    model_preds = []
    with torch.no_grad():
        num_batches = (len(str_inputs) + batch_size - 1) // batch_size
        for batch_str in tqdm.auto.tqdm(itertools.batched(str_inputs, n=batch_size), total=num_batches, desc="Inferring model hidden states"):
            batch_inputs = tokenizer(batch_str, return_tensors="pt")
            model_outputs = model(**batch_inputs.to(model.device), output_hidden_states=True)
            hidden_reprs = model_outputs.hidden_states
            logits = model_outputs.last_hidden_state @ model.embed_tokens.weight.T
            next_token_ids = logits[:, -1, :].argmax(dim=-1)
            model_preds.extend(tokenizer.batch_decode(next_token_ids))

            for layer_idx, hidden_state in enumerate(hidden_reprs):
                hidden_states[layer_idx].extend(hidden_state[:, -1, :].detach().cpu())
    return {k: torch.stack(v) for k, v in hidden_states.items()}, model_preds


# In[ ]:


batch_size = 1024
# train_hidden_states, train_preds = get_hidden_states_and_preds(
#         model,
#         [make_str_input(val, 0) for val in train_inputs] + [make_str_input(val, 1) for val in train_inputs] + [make_str_input(val, 2) for val in train_inputs],
#         batch_size
# )
states_preds = [get_hidden_states_and_preds(model, [make_str_input(val, i) for val in train_inputs], batch_size) for i in range(num_templates)]

hidden_states_all = [x[0] for x in states_preds]
preds_all = [x[1] for x in states_preds]

train_hidden_states = {k: torch.concat([hidden_states_all[i][k] for i in range(num_templates)]) for k in hidden_states_all[0].keys()}
train_preds = list(itertools.chain(*preds_all))

# train_hidden_states, train_preds = get_hidden_states_and_preds(
#         model,
#         [make_str_input(val) for val in train_inputs],
#         batch_size
# )
# train_hidden_states, train_preds = get_hidden_states_and_preds(
#         model,
#         [make_str_input_add(val) for val in train_inputs_add],
#         batch_size
# )
# valid_hidden_states, valid_preds = get_hidden_states_and_preds(
#         model,
#         [make_str_input(val) for val in valid_inputs],
#         batch_size
# )
# test_hidden_states, test_preds = get_hidden_states_and_preds(
#         model,
#         [make_str_input(val) for val in test_inputs],
#         batch_size
# )


# In[42]:


valid_hidden_states, valid_preds = get_hidden_states_and_preds(
        model,
        [make_str_input(val) for val in valid_inputs],
        batch_size
)
test_hidden_states, test_preds = get_hidden_states_and_preds(
        model,
        [make_str_input(val) for val in test_inputs],
        batch_size
)


# In[43]:


train_inputs_t = torch.tensor(train_inputs)

train_inputs_t[:, 0] / train_inputs_t[:, 1]


# In[44]:


def sanitize_pred(pred: str) -> int:
    try:
        return int(pred)
    except ValueError:
        return -1


# In[45]:


test_inputs_t = torch.tensor(test_inputs)

train_preds_t = torch.tensor([sanitize_pred(pred) for pred in train_preds])
valid_preds_t = torch.tensor([sanitize_pred(pred) for pred in valid_preds])
test_preds_t = torch.tensor([sanitize_pred(pred) for pred in test_preds])

# ratio of properly extracted train predictions
sum(train_preds_t != -1) / len(train_preds_t)


# In[46]:


# absolute model accuracy on test set
test_labels_ref = torch.tensor([x1 / x2 for x1, x2 in test_inputs])
sum(test_preds_t == test_labels_ref) / len(test_inputs)


# ### Probing

# In[47]:


basis_embs_sin = sinusoidal_encode(
    torch.arange(1000),
    min_value=0,
    max_value=1000,
    embedding_dim=train_hidden_states[0].shape[-1],
)


basis_embs_bin = binary_encode(
    torch.arange(1000),
    min_value=0,
    max_value=1000,
    embedding_dim=10,
)


# In[48]:


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


# In[49]:


# train_labels_ref = torch.tensor([x1 * x2 for x1, x2 in train_inputs])
# valid_labels_ref = torch.tensor([x1 * x2 for x1, x2 in valid_inputs]).to(device)
# test_labels_ref = torch.tensor([x1 * x2 for x1, x2 in test_inputs]).to(device)


# In[50]:


train_hidden_states[0].shape


# In[51]:


train_labels_ref = torch.tensor([x1 / x2 for x1, x2 in train_inputs]*num_templates)
# train_labels_ref = torch.tensor([x1 + x2 for x1, x2 in train_inputs])

valid_labels_ref = torch.tensor([x1 / x2 for x1, x2 in valid_inputs]).to(device)
test_labels_ref = torch.tensor([x1 / x2 for x1, x2 in test_inputs]).to(device)

train_labels = train_labels_ref.long() # train_preds_t.detach().clone()
train_hidden_states = {k: v[train_labels != -1] for k, v in train_hidden_states.items()}
train_labels = train_labels[train_labels != -1]

valid_labels = valid_labels_ref.long().to('cpu') # valid_preds_t.detach().clone()
valid_hidden_states = {k: v[valid_labels != -1] for k, v in valid_hidden_states.items()}
valid_labels = valid_labels[valid_labels != -1].to(device)

test_labels = test_labels_ref.long().to('cpu') # test_preds_t.detach().clone()
test_hidden_states = {k: v[test_labels != -1] for k, v in test_hidden_states.items()}
test_labels = test_labels[test_labels != -1].to(device)


# In[52]:


test_extracted = {}

test_accuracies = {"sin": {}, "bin": {}, "lin": {}, "log": {}}

basis_name = "sin"
basis_embs = basis_embs_sin

for layer_idx in reversed(range(len(train_hidden_states))):

    torch.manual_seed(0)
    probe = ClassifierProbe(
        emb_dim=train_hidden_states[0].shape[-1],
        hidden_dim=100,
        basis=basis_embs,
        heldout_mask=test_mask,
    ).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)  # TODO: try with weight_decay=1e-3
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=30000)

    rng = torch.Generator().manual_seed(0)
    best_val_acc = -1
    best_ckpt = None
    for i in range(50000+1):
        probe.train()
        optimizer.zero_grad()
        minibatch_idcs = torch.randint(len(train_labels), size=(128,), generator=rng)
        x = train_hidden_states[layer_idx][minibatch_idcs].float().to(device)
        y = train_labels[minibatch_idcs].to(device)
        logits = probe(x, holdout_eval_tokens=False)
        # add l1 regularization of all params to the loss
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss += 0.01 * sum(p.abs().sum() for p in probe.parameters())  # L1-reg
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 500 == 0:
            train_acc = (logits.argmax(dim=-1) == y).float().mean().item()
            probe.eval()
            with torch.no_grad():
                valid_logits = probe(valid_hidden_states[layer_idx].float().to(device), holdout_eval_tokens=False)  # TODO: holdout_eval_tokens switched to False -- incompatible with using model's own predictions as labels!
                valid_loss = torch.nn.functional.cross_entropy(valid_logits, valid_labels)
                valid_accuracy = (valid_logits.argmax(dim=-1) == valid_labels).float().mean().item()
                if valid_accuracy > best_val_acc:
                    best_val_acc = valid_accuracy
                    best_ckpt = probe.state_dict()
            print(f"{basis_name} {i=:>5} train loss: {loss.item():5.2f}  train acc: {train_acc:.2f}  val loss: {valid_loss.item():5.2f}  valid acc: {valid_accuracy:.2f}")
    probe.load_state_dict(best_ckpt)
    probe.eval()
    with torch.no_grad():
        test_logits = probe(test_hidden_states[layer_idx].float().to(device), holdout_eval_tokens=False)
        test_extracted[layer_idx] = test_logits.argmax(dim=-1)
        test_accuracy = (test_extracted[layer_idx] == test_labels).float().mean().item()

    test_accuracies[basis_name][layer_idx] = test_accuracy
    print(f"-> {basis_name} layer idx: {layer_idx:<3}, best valid accuracy: {best_val_acc:.2f}, test accuracy: {test_accuracy:.2f}")
    torch.save(best_ckpt, f'best_ckpt.gt_probe.div.layer{layer_idx}.pt')
    # best test_accuracy so far=0.64


# In[174]:


# rng = torch.Generator().manual_seed(0)
# rng_py = random.Random(0)
#
# assert list(train_hidden_states.keys()) == list(range(len(train_hidden_states)))
# train_hidden_states_tensor = torch.stack(list(train_hidden_states.values()), dim=0)
#
# histories = []
#
# probe = ClassifierProbe(
#     emb_dim=train_hidden_states[0].shape[-1],
#     hidden_dim=100,
#     basis=basis_embs_sin,
#     heldout_mask=test_mask,
# ).to(device)
#
# optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=0)
# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=20000)
#
# train_layers = list(range(len(train_hidden_states)-2, len(train_hidden_states)))
#
# for step in range(30000+1):
#     probe.train()
#     optimizer.zero_grad()
#     layer_idcs = torch.tensor(random.choices(train_layers, k=1024))
#     minibatch_idcs = torch.randint(len(train_labels), size=(1024,), generator=rng)
#     x = train_hidden_states_tensor[layer_idcs, minibatch_idcs].float().to(device)
#     y = train_labels[minibatch_idcs].to(device)
#     train_logits = probe(x, holdout_eval_tokens=False)
#     loss = torch.nn.functional.cross_entropy(train_logits, y)
#     loss += 1e-2 * sum(p.abs().sum() for p in probe.parameters()) # L1 regularization
#     loss.backward()
#     optimizer.step()
#     scheduler.step()
#
#     best_val_acc = -1
#     best_ckpt = probe.state_dict()
#
#     if step % 1000 == 0:
#         print("Train loss: %s train acc: %s LR: %s" % (loss.item(),
#                                                        sum(train_logits.argmax(-1) == y).item() / len(y),
#                                                        scheduler.get_last_lr()))
#         probe.eval()
#         valid_accs = []
#         with torch.no_grad():
#             print(f"{step=:<5}", end="  ")
#             # for layer_idx in range(0, len(train_hidden_states)):
#             layer_idx = len(train_hidden_states)-1
#             valid_logits = probe(valid_hidden_states[layer_idx].float().to(device), holdout_eval_tokens=False)
#             valid_acc = (valid_logits.argmax(dim=-1) == valid_labels).float().mean().item()
#             valid_accs.append(valid_acc)
#             histories.append({"step": step, "eval_layer": layer_idx, "valid_acc": valid_acc})
#             acc_out = f"{valid_acc:>6.1%}"
#             if layer_idx not in train_layers:
#                 print('\033[94m' + acc_out + '\033[0m', end=" ")
#             else:
#                 print(acc_out, end=" ")
#             print()
#             valid_acc = sum(valid_accs) / len(valid_accs)
#             if valid_acc > best_val_acc:
#                 best_val_acc = valid_acc
#                 best_ckpt = probe.state_dict()
#
# probe.load_state_dict(best_ckpt)
# probe.eval()
# with torch.no_grad():
#     layer_idx = len(train_hidden_states)-1
#     test_logits = probe(test_hidden_states[layer_idx].float().to(device), holdout_eval_tokens=False)
#     test_extracted[layer_idx] = test_logits.argmax(dim=-1)
#     test_accuracy = (test_extracted[layer_idx] == test_labels).float().mean().item()
#     print("Test accuracy: %s" % test_accuracy)


# In[175]:


import pandas as pd

df = pd.DataFrame({"probe_l%s" % k: v.cpu() for k, v in test_extracted.items()})
df["model_predictions"] = test_preds_t.cpu()
df["inputs"] = [make_str_input(op) for op in test_inputs]
df["labels"] = test_labels_ref.cpu()
df.to_csv("gt_vs_probes_preds_llama3b_div_261125_redo.csv", index=False)  # TODO: visualize


# In[176]:


test_preds_t.device, test_labels.device


# In[177]:


is_result_computed_per_l = torch.vstack([test_extracted[l_key] == test_labels_ref for l_key in test_extracted])
is_result_computed_internally = torch.any(is_result_computed_per_l, dim=0).cpu()
returned_val_is_computed = torch.vstack([test_extracted[l_key].cpu() == test_preds_t for l_key in test_extracted])
is_result_returned = (test_preds_t == test_labels_ref.cpu())
print(
      "Ever computed correctly internally and NOT correctly returned (out of incorrect): %s\n"
      "NOT ever computed correctly internally and correctly returned: (out of correct) %s\n"
      "Ever computed correctly internally and correctly returned (out of correct): %s\n"
      "NOT ever computed correctly internally and NOT correctly returned (out of incorrect): %s\n"
      "Ever computed as returned: %s"
      % (
         torch.sum(is_result_computed_internally & ~is_result_returned).item() / (~is_result_returned).sum(),
         torch.sum(~is_result_computed_internally & is_result_returned).item() / is_result_returned.sum(),
         torch.sum(is_result_computed_internally & is_result_returned).item() / is_result_returned.sum(),
         torch.sum(~is_result_computed_internally & ~is_result_returned).item() / (~is_result_returned).sum(),
         returned_val_is_computed.any(dim=0).sum() / len(returned_val_is_computed[0]))
        )


# In[27]:


# 15 out of 25 correct contains multiplication involving "1" or "2" --> effectively solvable by addition
# only 30 out of 140 in the case of Llama 3B
(len(test_labels_ref[~is_result_computed_internally & is_result_returned]),
torch.isin(test_inputs_t[~is_result_computed_internally & is_result_returned], torch.tensor([1, 2])).any(dim=1).sum())


# In[ ]:


def solve_linear_layer(x: Tensor, y: Tensor) -> torch.nn.Linear:
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    if not y.is_floating_point():
        y = y.float()
   
    lin = torch.nn.Linear(x.shape[-1], y.shape[-1], device=x.device)
    x_aug = torch.cat([x, torch.ones(len(x), 1, device=x.device)], dim=1)
    coeffs = torch.linalg.lstsq(x_aug, y).solution
    w, b = coeffs[:-1], coeffs[-1]
    with torch.no_grad():
        lin.weight[:] = w.T
        lin.bias[:] = b
    return lin


# In[ ]:


for layer_idx in range(len(train_hidden_states)):
    lin_probe = solve_linear_layer(
        train_hidden_states[layer_idx].float().to(device),
        train_labels.to(device),
    )
    log_probe = solve_linear_layer(
        train_hidden_states[layer_idx].float().to(device),
        train_labels.log1p().to(device),
    )
    lin_test_pred = lin_probe(test_hidden_states[layer_idx].float().to(device)).flatten().round().int()
    lin_test_accuracy = (lin_test_pred == test_labels).float().mean().item()
    
    log_test_pred = log_probe(test_hidden_states[layer_idx].float().to(device)).flatten().exp().add(1).round().int()
    log_test_accuracy = (log_test_pred == test_labels).float().mean().item()
    
    test_accuracies["lin"][layer_idx] = lin_test_accuracy
    test_accuracies["log"][layer_idx] = log_test_accuracy

    print(f"layer idx: {layer_idx:<3}, linear probe acc: {lin_test_accuracy:.2f}, log probe acc: {log_test_accuracy:.2f}")


# In[ ]:


for name, accs in test_accuracies.items():
    print(f"{name} accs: | " + " | ".join([f"{x:.0%}" for layer, x in sorted(accs.items())]) + " |")

