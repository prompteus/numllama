import argparse
import itertools
import random

import datasets
import einops
import torch
import tqdm.auto
import transformers
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="meta-llama/Llama-3.2-1B")
parser.add_argument("-o", "--token_offset", type=int)
parser.add_argument("-n", "--nums", type=str, default="False")
parser.add_argument("-x", "--only_nums", type=str, default="False")
parser.add_argument("-t", "--num_texts", type=int, default=21150) # matching natural-text dataset size

args = parser.parse_args()
args.nums = args.nums.lower() != "false"
args.only_nums = args.only_nums.lower() != "false"

model_ckpt = args.model
model = transformers.AutoModelForCausalLM.from_pretrained(model_ckpt).eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_ckpt)

# natural-text tokens:
ds = datasets.concatenate_datasets(
    [datasets.load_dataset("RealTimeData/bbc_news_alltime", f"2024-{i:02d}", split="train").select_columns("content")
     for i in range(1, 13)]
)
texts = list(set(ds["content"]))
texts.sort(key=lambda x: len(x), reverse=True)

if args.nums:
    # numeric texts: replace random numbers from original texts with long numbers
    import re
    max_decoded_pos = 5
    assert args.token_offset <= max_decoded_pos
    random.seed(42)
    num_texts = ["".join(map(str, random.sample(list(range(1000)), k=max_decoded_pos+1))) for _ in range(args.num_texts)]
    if args.only_nums:
        texts = num_texts
    else:
        texts = [re.sub(r"\d+", lambda _: random.choice(num_texts), text) for text in tqdm.auto.tqdm(texts, desc="Replacing nums")]

tokenized = tokenizer(texts)

if args.nums:
    # numeric tokens:
    pred_tokens_ids = [x[0] for x in tokenizer([str(i) for i in range(1000)], add_special_tokens=False).input_ids]
else:
    # natural-text tokens: to balance the classification baseline, we pick the same # of tokens as numeric tokens
    pred_tokens_range_start, pred_tokens_range_end = (2000, 3000)
    tokens_counts = pd.Series(list(itertools.chain(*tokenized.input_ids))).value_counts().sort_values(ascending=False)
    pred_tokens_ids = tokens_counts.index[pred_tokens_range_start:pred_tokens_range_end]

desired_length = 512
if args.only_nums:
    # length of texts must exactly match the max-kth predicted position
    ds = [inp[:desired_length] for inp in tokenized.input_ids if len(inp) == len(tokenized.input_ids[0])]
else:
    # length of texts aligned to max_length
    ds = [inp[:desired_length] for inp in tokenized.input_ids if len(inp) >= desired_length]
random.Random(0).shuffle(ds)
eval_size = 1000
train_ds, valid_ds, test_ds = torch.tensor(ds, dtype=torch.long).tensor_split([-eval_size*2, -eval_size])
train_ds.shape, valid_ds.shape, test_ds.shape

device = "cuda"
dtype = torch.bfloat16
model.to(device, dtype).eval()

pred_tokens_ids = torch.tensor(pred_tokens_ids, device=device)

import gc
gc.collect()
torch.cuda.empty_cache()

probes = {}
optims = {}

layer_idcs = range(len(model.model.layers) + 1)
# layer_idcs = [0, 1, 2]
offset_idcs = [args.token_offset]

for layer_idx in layer_idcs:
    probes[layer_idx] = {}
    optims[layer_idx] = {}
    for offset_idx in offset_idcs:
        probe = torch.nn.Linear(model.config.hidden_size, model.config.hidden_size, bias=False, device=model.device, dtype=model.dtype)
        optim = torch.optim.AdamW(probe.parameters(), lr=1e-4)
        probes[layer_idx][offset_idx] = probe
        optims[layer_idx][offset_idx] = optim

batch_size = 16

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_ds),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    pin_memory_device=device
)
valid_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(valid_ds),
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    pin_memory_device=device
)

batches = itertools.cycle(train_loader)
pbar = tqdm.auto.tqdm(batches, desc="Training")

n_valid_batches = len(valid_loader)

train_accs = {layer_idx: {offset_idx: [] for offset_idx in offset_idcs} for layer_idx in layer_idcs}
valid_accs = {layer_idx: {offset_idx: [] for offset_idx in offset_idcs} for layer_idx in layer_idcs}

for train_step, train_batch in enumerate(pbar):
    train_batch, = train_batch # loader outputs tuples even if there is only one x without y, we need to unpack
    train_batch = train_batch.to(device)

    with torch.no_grad():
        hidden_states = model(train_batch, output_hidden_states=True).hidden_states

    # curr_train_accs = torch.zeros((len(layer_idcs), len(offset_idcs)), device=device, dtype=torch.float32)
    curr_train_accs = {layer: {} for layer in range(len(layer_idcs))}
    for layer_idx, offset_idx in itertools.product(layer_idcs, offset_idcs):
        probe = probes[layer_idx][offset_idx]
        optim = optims[layer_idx][offset_idx]

        probe.train()
        prediction: torch.Tensor = probe(hidden_states[layer_idx])
        logits = model.lm_head(prediction)
        logits = einops.rearrange(logits, "batch seq vocab -> batch vocab seq")
        logits = logits[..., offset_idx:] # slice from start
        labels = train_batch[:, :logits.shape[-1]].clone() # slice from end
        train_labels_mask = torch.isin(labels, pred_tokens_ids)
        if args.nums:
            # if looking at numbers, exclude from probing the embeddings of tokens that are not numbers
            # not a problem with natural-lang tokens -- there, we probe from every other token
            train_inputs_mask = torch.isin(train_batch[:, :-offset_idx], pred_tokens_ids)
            train_labels_mask = train_labels_mask & train_inputs_mask

        labels[~train_labels_mask] = -100

        optim.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optim.step()
        curr_train_accs[layer_idx][offset_idx] = (logits.argmax(dim=1) == labels).float().mean().item()

    eval_every_n_steps = 10
    with open("logs_probe_past_%s_max_%s_offsets=%s.tsv" % (
            "numbers" if args.nums else "tokens",
            args.num_texts if args.only_nums else "all-in-text",
            str(offset_idcs)
    ), mode="a") as log_f:
        if train_step % eval_every_n_steps == 0 and train_step != 0:
            probe.eval()
            with torch.no_grad():
                curr_valid_accs = torch.zeros((max(layer_idcs)+1, max(offset_idcs)+1, n_valid_batches), device=device, dtype=torch.float32)
                for val_batch_idx, valid_batch in enumerate(tqdm.auto.tqdm(valid_loader, desc="Validating", leave=False)):
                    valid_batch, = valid_batch # loader outputs tuples even if there is only one x without y, we need to unpack
                    valid_batch = valid_batch.to(device)
                    hidden_states = model(valid_batch, output_hidden_states=True).hidden_states

                    for layer_idx, offset_idx in itertools.product(layer_idcs, offset_idcs):
                        prediction = probe(hidden_states[layer_idx])
                        logits = model.lm_head(prediction)
                        logits = einops.rearrange(logits, "batch seq vocab -> batch vocab seq")
                        logits = logits[..., offset_idx:] # slice from start
                        labels = valid_batch[:, :logits.shape[-1]].clone() # slice from end
                        val_labels_mask = torch.isin(labels, pred_tokens_ids)
                        if args.nums:
                            # if looking at numbers, exclude from probing the embeddings of tokens that are not numbers
                            # not a problem with natural-lang tokens -- there, we probe from every other token
                            val_inputs_mask = torch.isin(valid_batch[:, :-offset_idx], pred_tokens_ids)
                            val_labels_mask = val_labels_mask & val_inputs_mask

                        valid_acc_batch = (logits.argmax(dim=1)[val_labels_mask] == labels[val_labels_mask]).float().mean()
                        curr_valid_accs[layer_idx, offset_idx, val_batch_idx] = valid_acc_batch
                curr_valid_accs = curr_valid_accs.mean(dim=-1) # average over all valid batches

            for layer_idx, offset_idx in itertools.product(layer_idcs, offset_idcs):
                train_accs[layer_idx][offset_idx].append(curr_train_accs[layer_idx][offset_idx])
                valid_accs[layer_idx][offset_idx].append(curr_valid_accs[layer_idx][offset_idx].item())

            for offset_idx in offset_idcs:
                # find best performing layer and timestep
                best_layer = max(layer_idcs, key=lambda l: max(valid_accs[l][offset_idx]))
                best_step = max(range(len(valid_accs[best_layer][offset_idx])), key=lambda s: valid_accs[best_layer][offset_idx][s])
                best_valid_acc = valid_accs[best_layer][offset_idx][best_step]
                train_acc = train_accs[best_layer][offset_idx][best_step]
                valid_acc = valid_accs[layer_idx][offset_idx][-1]
                print(f"Offset:\t{offset_idx}\ttrain step:\t{train_step}\ttrain acc:\t{train_acc:.3f}\tvalid acc:\t{valid_acc:.3f}\tbest layer:\t{best_layer:<2}\tfrom step:\t{(best_step+1)*eval_every_n_steps:<5}\tbest valid acc:\t{best_valid_acc:.3f}",
                      file=log_f)
