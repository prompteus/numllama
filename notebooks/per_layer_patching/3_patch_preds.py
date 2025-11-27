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

# from transformers.utils.masking_utils import create_causal_mask


model_ckpt = "meta-llama/Llama-3.2-3B"
model = transformers.AutoModel.from_pretrained(model_ckpt).eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_ckpt)
model = model.half().to(device).eval()


all_values = torch.arange(0, 1000)
mask = torch.rand(len(all_values), generator=torch.Generator().manual_seed(0))
train_mask = mask < 0.9
valid_mask = ~train_mask & (mask < 0.95)
test_mask = ~train_mask & ~valid_mask

train_values = all_values[train_mask]
valid_values = all_values[valid_mask]
test_values = all_values[test_mask]


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



@torch.no_grad()
def patch_hidden_states_and_get_pred(model, str_input: str, layer_idx: int, patch: Tensor, alpha: float) -> dict:
    batch_inputs = tokenizer([str_input], return_tensors="pt")
    model_outputs = model(**batch_inputs.to(model.device), output_hidden_states=True)
    hidden_reprs = model_outputs.hidden_states
    logits_prepatch = model_outputs.last_hidden_state @ model.embed_tokens.weight.T
    next_token_ids_prepatch = logits_prepatch[:, -1, :].argmax(dim=-1)
    hidden_states_pre = hidden_reprs[layer_idx]
    hidden_states_pre[..., -1, :] *= 1 - alpha
    hidden_states_pre[..., -1, :] += alpha * patch

    inputs_embeds: torch.Tensor = model.embed_tokens(batch_inputs.input_ids)
    cache_position=torch.arange(
        0, inputs_embeds.shape[1], device=inputs_embeds.device
    )
    position_ids=cache_position.unsqueeze(0)
    causal_mask = None
    hidden_states = hidden_states_pre
    position_embeddings = model.rotary_emb(inputs_embeds, position_ids=position_ids)

    for decoder_layer in model.layers[layer_idx: model.config.num_hidden_layers]:
        hidden_states, *_ = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=None,
            cache_position=cache_position,
        )

    hidden_states = model.norm(hidden_states)
    logits_postpatch = hidden_states @ model.embed_tokens.weight.T
    next_token_ids_postpatch = logits_postpatch[:, -1, :].argmax(dim=-1)
    outstr_prepatch = tokenizer.batch_decode(next_token_ids_prepatch)[0]
    outstr_postpatch = tokenizer.batch_decode(next_token_ids_postpatch)[0]
    return {'before': outstr_prepatch, 'after':outstr_postpatch}


train_labels_ref = torch.tensor([x1 / x2 for x1, x2 in train_inputs]*num_templates)
# train_labels_ref = torch.tensor([x1 + x2 for x1, x2 in train_inputs])

valid_labels_ref = torch.tensor([x1 / x2 for x1, x2 in valid_inputs]).to(device)
test_labels_ref = torch.tensor([x1 / x2 for x1, x2 in test_inputs]).to(device)

train_labels = train_labels_ref.long() # train_preds_t.detach().clone()
train_labels = train_labels[train_labels != -1]

valid_labels = valid_labels_ref.long().to('cpu') # valid_preds_t.detach().clone()
valid_labels = valid_labels[valid_labels != -1].to(device)

test_labels = test_labels_ref.long().to('cpu') # test_preds_t.detach().clone()
test_labels = test_labels[test_labels != -1].to(device)


records = []
#for alpha in [i / 10 for i in range(1, 11)]:
for alpha in [0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5]:
    for layer_idx in range(19, 29):
        patches = torch.load(f'patches.gt_probe.layer{layer_idx}.pt').half().to(device)
        for valid_input, valid_label in tqdm.auto.tqdm(list(zip(valid_inputs, valid_labels)), desc=f'L{layer_idx}'):
            for template in range(num_templates):
                valid_input_ = make_str_input(valid_input, template)
                results = patch_hidden_states_and_get_pred(
                    model,
                    valid_input_,
                    layer_idx,
                    patches[valid_label,:],
                    alpha
                )
                records.append({
                    'layer': layer_idx,
                    'tgt': valid_label.item(),
                    'input': valid_input_,
                    'alpha': alpha,
                    **results,
                })

import pandas as pd
pd.DataFrame.from_records(records).to_csv('patching_results.csv')
