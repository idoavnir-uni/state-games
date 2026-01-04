# %% [markdown]
# # Probe Failures Analysis - Most Hated City (Animals)
# 
# This notebook:
# 1. Trains a probe on most hated city sentences using animal names
# 2. Evaluates on the training data
# 3. Prints the first 10 examples where the probe fails

# %%
import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

sys.path.insert(0, os.path.abspath('..'))

from models.load_gla import load_gla_model, get_model_config
from models.state_extractor_gla import GLAStateExtractor
from datasets.most_hated_city_dataset import MostHatedCityDataset

print("Imports complete!")
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# %% [markdown]
# ## 1. Load Model and Tokenizer

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("Loading GLA model...")
model, tokenizer = load_gla_model(
    model_name="fla-hub/gla-1.3B-100B",
    device=device,
    torch_dtype=torch.bfloat16
)

config = get_model_config(model)
print(f"Model loaded: {config.get('num_layers')} layers, {config.get('num_heads')} heads")

# %% [markdown]
# ## 2. Initialize State Extractor

# %%
extractor = GLAStateExtractor(model, verbose=False)

# %% [markdown]
# ## 3. Create Most Hated City Dataset (Animals)

# %%
DATASET_SIZE = 2000
N_ENTITIES = 10
N_CITIES = 10

FIXED_ENTITY = "Aardvark"

print(f"Creating dataset with {DATASET_SIZE} samples...")
dataset = MostHatedCityDataset(
    tokenizer=tokenizer,
    size=DATASET_SIZE,
    n_entities=N_ENTITIES,
    n_cities=N_CITIES,
    fixed_entity_name=FIXED_ENTITY,
    seed=42,
)
print(f"Dataset created with {len(dataset)} samples")
print(f"Fixed entity: {dataset.fixed_entity_name}")
print(f"Cities used: {dataset.cities}")

# %% [markdown]
# ## 3.1 Verify Cities are Single Tokens

# %%
print("\nVerifying cities are single tokens...")
city_token_ids = {}
all_single_tokens = True

for city in dataset.cities:
    tokens = tokenizer.encode(city, add_special_tokens=False)
    is_single = len(tokens) == 1
    
    if is_single:
        city_token_ids[city] = tokens[0]
        token_str = tokenizer.decode(tokens[0])
        print(f"  {city}: token_id={tokens[0]}, decoded='{token_str}' ✓")
    else:
        all_single_tokens = False
        decoded = [tokenizer.decode(t) for t in tokens]
        print(f"  {city}: {len(tokens)} tokens {tokens} -> {decoded} ✗")

if all_single_tokens:
    print("\n✓ All cities are single tokens!")
else:
    print("\n✗ Some cities are NOT single tokens. Need to filter or change cities.")
    raise ValueError("Not all cities are single tokens")

CITY_TOKEN_IDS = torch.tensor([city_token_ids[c] for c in dataset.cities])

# %% [markdown]
# ## 4. Extract States and Build Dataframe

# %%
num_layers = config.get('num_layers')
num_heads = config.get('num_heads')

print(f"Extracting states for {DATASET_SIZE} samples...")
print(f"Model has {num_layers} layers and {num_heads} heads per layer")

metadata_rows = []
all_states = np.zeros((DATASET_SIZE, num_layers, num_heads, 256, 512), dtype=np.float16)

for idx in tqdm(range(len(dataset)), desc="Processing samples"):
    sample = dataset[idx]
    
    input_ids = sample.input_ids.to(device)
    final_states = extractor.extract_final_states(input_ids)
    
    metadata_rows.append({
        'sentence': sample.text,
        'target_city': sample.fixed_entity_city,
        'information_given_idx': sample.fixed_entity_sentence_end_token_idx,
        'sentence_with_info_num': sample.fixed_entity_sentence_number,
    })
    
    for layer_idx in range(num_layers):
        layer_state = final_states[layer_idx]
        all_states[idx, layer_idx] = layer_state[0].cpu().to(torch.float16).numpy()

df = pd.DataFrame(metadata_rows)
print(f"\nMetadata dataframe shape: {df.shape}")
print(f"States array shape: {all_states.shape}")

# %% [markdown]
# ## 5. Train Linear Probe with Unembedding Matrix

# %%
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

CITY_TO_IDX = {city: idx for idx, city in enumerate(dataset.cities)}
labels = torch.tensor([CITY_TO_IDX[c] for c in df['target_city']])
n_cities = len(dataset.cities)

print(f"City mapping: {CITY_TO_IDX}")

lm_head = model.lm_head.weight.detach()
hidden_size = lm_head.shape[1]
num_heads = config.get('num_heads')
head_dim = 512

print(f"LM head shape: {lm_head.shape}")
print(f"Model hidden size: {hidden_size}")
print(f"Num heads: {num_heads}, head dim: {head_dim}")

city_unembed = lm_head[CITY_TOKEN_IDS].float().to(device)
print(f"City unembedding shape: {city_unembed.shape}")

# %%
def get_head_o_proj(model, layer_idx, head_idx, head_dim=512, hidden_size=2048):
    o_proj = model.model.layers[layer_idx].attn.o_proj.weight.detach().float()
    start_idx = head_idx * head_dim
    end_idx = (head_idx + 1) * head_dim
    head_o_proj = o_proj[:, start_idx:end_idx]
    return head_o_proj

def precompute_transformed_states(states, head_o_proj, city_unembed):
    intermediate = torch.einsum('bdk,hk->bdh', states, head_o_proj)
    A = torch.einsum('bdh,ch->bdc', intermediate, city_unembed)
    return A

def solve_w_left_closed_form(A, labels, n_cities):
    batch_size = A.shape[0]
    Y = torch.zeros(batch_size, n_cities, device=A.device)
    Y.scatter_(1, labels.unsqueeze(1), 1.0)
    A_flat = A.permute(0, 2, 1).reshape(-1, 256)
    Y_flat = Y.reshape(-1)
    w_left = torch.linalg.lstsq(A_flat, Y_flat).solution
    return w_left

def evaluate_w_left(A, labels, w_left):
    logits = torch.einsum('d,bdc->bc', w_left, A)
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    return acc

def solve_probe(states_np, labels, layer_idx, head_idx):
    states = torch.tensor(states_np[:, layer_idx, head_idx], dtype=torch.float32).to(device)
    
    n_samples = len(labels)
    n_val = max(1, int(0.3 * n_samples))
    indices = torch.randperm(n_samples)
    
    train_states = states[indices[n_val:]]
    train_labels = labels[indices[n_val:]].to(device)
    val_states = states[indices[:n_val]]
    val_labels = labels[indices[:n_val]].to(device)
    
    head_o_proj = get_head_o_proj(model, layer_idx, head_idx).to(device)
    
    train_A = precompute_transformed_states(train_states, head_o_proj, city_unembed)
    val_A = precompute_transformed_states(val_states, head_o_proj, city_unembed)
    
    w_left = solve_w_left_closed_form(train_A, train_labels, n_cities)
    
    train_acc = evaluate_w_left(train_A, train_labels, w_left)
    val_acc = evaluate_w_left(val_A, val_labels, w_left)
    
    return val_acc, train_acc, w_left

# %%
print("Solving probes for each layer/head (closed form)...")
print(f"{'Layer':<6} {'Head':<6} {'Val Acc':<10} {'Train Acc':<10}")
print("-" * 35)

results = []
all_w_lefts = {}
for layer_idx in range(num_layers):
    for head_idx in range(num_heads):
        val_acc, train_acc, w_left = solve_probe(all_states, labels, layer_idx, head_idx)
        all_w_lefts[(layer_idx, head_idx)] = w_left
        results.append({
            'layer': layer_idx,
            'head': head_idx,
            'val_acc': val_acc,
            'train_acc': train_acc,
        })
        print(f"L{layer_idx:<5} H{head_idx:<5} {val_acc:<10.3f} {train_acc:<10.3f}")

# %%
results_df = pd.DataFrame(results)
print("\n=== Best Performing Heads ===")
print(results_df.sort_values('val_acc', ascending=False).head(10))

# %% [markdown]
# ## 6. Analyze Failures on Original Dataset

# %%
best_head = results_df.sort_values('val_acc', ascending=False).iloc[0]
best_layer = int(best_head['layer'])
best_head_idx = int(best_head['head'])
print(f"\nBest head: Layer {best_layer}, Head {best_head_idx}")
print(f"Validation accuracy: {best_head['val_acc']:.3f}")

# Get the best head projection and probe weights
best_head_o_proj = get_head_o_proj(model, best_layer, best_head_idx).to(device)

# Get states for best head from all samples
best_head_states = torch.tensor(all_states[:, best_layer, best_head_idx], dtype=torch.float32).to(device)

# Get the trained probe
best_w_left = all_w_lefts[(best_layer, best_head_idx)]

# Compute predictions for all samples
print("\nComputing predictions for all samples...")
A = precompute_transformed_states(best_head_states, best_head_o_proj, city_unembed)
logits = torch.einsum('d,bdc->bc', best_w_left, A)
predictions = logits.argmax(dim=1).cpu().numpy()

# Get true labels
true_labels = labels.numpy()

# Find failures
failures = predictions != true_labels
num_failures = failures.sum()
num_total = len(failures)
accuracy = 1 - (num_failures / num_total)

print(f"\n{'='*60}")
print(f"Results on Training Dataset:")
print(f"  Total samples: {num_total}")
print(f"  Correct predictions: {num_total - num_failures}")
print(f"  Failed predictions: {num_failures}")
print(f"  Accuracy: {accuracy:.3f}")
print(f"{'='*60}\n")

# %% [markdown]
# ## 7. Print First 10 Failed Examples

# %%
idx_to_city = {idx: city for city, idx in CITY_TO_IDX.items()}

MAX_FAILURES_TO_SHOW = 10

print(f"\n{'='*80}")
print(f"FAILED PREDICTIONS (showing first {min(MAX_FAILURES_TO_SHOW, num_failures)} of {num_failures} total failures)")
print(f"{'='*80}\n")

failure_indices = np.where(failures)[0]

for i, fail_idx in enumerate(failure_indices[:MAX_FAILURES_TO_SHOW]):
    true_label = true_labels[fail_idx]
    pred_label = predictions[fail_idx]
    true_city = idx_to_city[true_label]
    pred_city = idx_to_city[pred_label]
    
    # Get top-3 predictions
    top3_indices = torch.topk(logits[fail_idx], 3).indices.cpu().numpy()
    top3_cities = [idx_to_city[idx] for idx in top3_indices]
    top3_logits = [logits[fail_idx, idx].item() for idx in top3_indices]
    
    print(f"Example {i+1}/{min(MAX_FAILURES_TO_SHOW, num_failures)} (Index: {fail_idx})")
    print(f"-" * 80)
    print(f"Sentence: {df.iloc[fail_idx]['sentence']}")
    print(f"")
    print(f"True city:      {true_city}")
    print(f"Predicted city: {pred_city}")
    print(f"")
    print(f"Top-3 predictions:")
    for rank, (city, logit) in enumerate(zip(top3_cities, top3_logits), 1):
        marker = "✓" if city == true_city else " "
        print(f"  {rank}. {city:<10} (logit: {logit:7.3f}) {marker}")
    print(f"")
    print(f"Info given at: sentence {df.iloc[fail_idx]['sentence_with_info_num']}, token {df.iloc[fail_idx]['information_given_idx']}")
    print(f"{'='*80}\n")

# %%

