# %% [markdown]
# # RWKV State Probing with Unembedding Matrix - Most Hated City (GPU 4)
# 
# This notebook:
# 1. Generates "X hates City" sentences with animal entities (fixed entity: Bat)
# 2. Extracts RWKV model states at the final token position
# 3. Uses model's output projection + unembedding matrix + learned w_right to predict cities
# 
# **RWKV forward pass (simplified):**
# - Read from state: o = r @ state  →  (64,) @ (64, 64) = (64,)
# - Output projection: hidden = o @ head_o_proj  →  (64,) @ (64, 2048) = (2048,)
# - LM head for cities: logits = hidden @ city_unembed  →  (2048,) @ (2048, n_cities) = (n_cities,)
# 
# **Probe model:** logits = (W_left_model @ state) @ w_right
# - state: (head_size, head_size) per head = (64, 64)
# - W_left_model: (n_cities, head_size) = (head_o_proj @ city_unembed).T - from model weights
# - w_right: (head_size,) learned vector
#
# **Key insight:** W_left_model is derived from model weights:
# - head_o_proj: (64, 2048) - ROWS of output.weight for this head
# - city_unembed: (2048, n_cities) - COLUMNS of LM head for city tokens
# - M = head_o_proj @ city_unembed: (64, n_cities)
# - W_left_model = M.T: (n_cities, 64)

# %%
import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

sys.path.insert(0, os.path.abspath('..'))

from models.load_rwkv import load_rwkv_model, get_model_config
from models.state_extractor_rwkv import RWKVStateExtractor
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

print("Loading RWKV model...")
model, tokenizer = load_rwkv_model(model_name="rwkv7-g1b-1.5b-20251202-ctx8192.pth")

config = get_model_config(model)
print(f"Model loaded: {config.get('num_layers')} layers, {config.get('num_heads')} heads")

# %% [markdown]
# ## 2. Initialize State Extractor

# %%
extractor = RWKVStateExtractor(model, verbose=False)
head_size = extractor.head_size
n_head = extractor.n_head
n_embd = model.model.n_embd
print(f"Head size: {head_size}")
print(f"Num heads: {n_head}")
print(f"Hidden size: {n_embd}")

# %% [markdown]
# ## 3. Create Most Hated City Dataset

# %%
DATASET_SIZE = 2000
N_ENTITIES = 10
N_CITIES = 10
FIXED_ENTITY = "Bat"

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
    tokens = tokenizer.encode(city)
    is_single = len(tokens) == 1
    
    if is_single:
        city_token_ids[city] = tokens[0]
        token_str = tokenizer.decode([tokens[0]])
        print(f"  {city}: token_id={tokens[0]}, decoded='{token_str}' ✓")
    else:
        all_single_tokens = False
        decoded = [tokenizer.decode([t]) for t in tokens]
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
all_states = np.zeros((DATASET_SIZE, num_layers, num_heads, head_size, head_size), dtype=np.float16)

for idx in tqdm(range(len(dataset)), desc="Processing samples"):
    sample = dataset[idx]
    
    input_ids = sample.input_ids
    final_states = extractor.extract_final_states(input_ids)
    
    metadata_rows.append({
        'sentence': sample.text,
        'target_city': sample.fixed_entity_city,
        'information_given_idx': sample.fixed_entity_sentence_end_token_idx,
        'sentence_with_info_num': sample.fixed_entity_sentence_number,
    })
    
    for layer_idx in range(num_layers):
        layer_state = final_states[layer_idx]
        all_states[idx, layer_idx] = layer_state.cpu().to(torch.float16).numpy()

df = pd.DataFrame(metadata_rows)
print(f"\nMetadata dataframe shape: {df.shape}")
print(f"States array shape: {all_states.shape}")

# %% [markdown]
# ## 5. Display Sample Information

# %%
print("\n=== Sample Entry ===")
print(f"Sentence: {df.iloc[0]['sentence']}")
print(f"Target city: {df.iloc[0]['target_city']}")
print(f"States shape: {all_states.shape} = (samples, layers, heads, {head_size}, {head_size})")

print("\n=== Dataset Statistics ===")
print(f"Total samples: {len(df)}")
print(f"City distribution:\n{df['target_city'].value_counts()}")

# %% [markdown]
# ## 6. Train Linear Probe with Model Weights
# 
# **Architecture (matches 04_train_probe.py direction):**
# - state: (head_size, head_size) = (64, 64)
# - W_left_model: (n_cities, head_size) = (head_o_proj @ city_unembed).T - from model weights
# - w_right: (head_size,) learned
#
# **Formula:** A = W_left_model @ state = (n_cities, 64) @ (64, 64) = (n_cities, 64)
#              logits = A @ w_right = (n_cities, 64) @ (64,) = (n_cities,)
#
# We solve for w_right in closed form using least squares.

# %%
import torch.nn as nn
import torch.optim as optim

CITY_TO_IDX = {city: idx for idx, city in enumerate(dataset.cities)}
labels = torch.tensor([CITY_TO_IDX[c] for c in df['target_city']])
n_cities = len(dataset.cities)

print(f"City mapping: {CITY_TO_IDX}")

# Get unembedding matrix from model (LM head)
# head.weight shape: (hidden_size, vocab_size) = (2048, 65536)
lm_head = model.model.z['head.weight'].detach()
print(f"LM head shape: {lm_head.shape}")

# Extract only the columns for city tokens (convert to float32)
# For x @ head.weight[:, city_ids], we need shape (2048, n_cities)
city_unembed = lm_head[:, CITY_TOKEN_IDS].float().to(device)  # (hidden_size, n_cities) = (2048, n_cities)
print(f"City unembedding shape: {city_unembed.shape}")

# %%
def get_head_o_proj(model, layer_idx, head_idx, head_size=64, n_embd=2048):
    """
    Extract the output projection weights for a specific head from a layer.
    
    output.weight is (hidden_size, hidden_size) = (2048, 2048)
    For out @ O_, head h contributes at indices [h*head_size : (h+1)*head_size]
    So we need ROWS: O_[h*head_size:(h+1)*head_size, :] -> (64, 2048)
    """
    z = model.model.z
    o_proj = z[f'blocks.{layer_idx}.att.output.weight'].detach().float()  # (2048, 2048)
    # Extract the ROWS corresponding to this head
    start_idx = head_idx * head_size
    end_idx = (head_idx + 1) * head_size
    head_o_proj = o_proj[start_idx:end_idx, :]  # (64, 2048) - ROWS not columns!
    return head_o_proj

def get_W_left_model(head_o_proj, city_unembed):
    """
    Compute W_left_model from model weights.
    
    In RWKV forward: logits = (r @ state) @ head_o_proj @ city_unembed
    
    head_o_proj: (head_size, hidden_size) = (64, 2048)
    city_unembed: (hidden_size, n_cities) = (2048, n_cities)
    
    M = head_o_proj @ city_unembed: (64, n_cities)
    - M[d, c] tells us how value dimension d contributes to city c
    
    W_left_model = M.T: (n_cities, 64)
    - For probe: logits = (W_left_model @ state) @ w_right
    """
    M = head_o_proj @ city_unembed  # (64, 2048) @ (2048, n_cities) = (64, n_cities)
    W_left_model = M.T  # (n_cities, 64)
    return W_left_model

# %%
def precompute_transformed_states(states, W_left_model):
    """
    Precompute A = W_left_model @ state
    
    states: (batch, head_size, head_size) = (batch, 64, 64)
    W_left_model: (n_cities, head_size) = (n_cities, 64)
    
    Returns A: (batch, n_cities, head_size) = (batch, n_cities, 64)
    
    This matches 04_train_probe.py: hidden = einsum('cd,bdk->bck', W_left, state)
    """
    # W_left_model @ state: (n_cities, 64) @ (batch, 64, 64) -> (batch, n_cities, 64)
    A = torch.einsum('cd,bdk->bck', W_left_model, states)  # (batch, n_cities, 64)
    return A

def solve_w_right_closed_form(A, labels, n_cities):
    """
    Solve for w_right in closed form using least squares.
    
    A: (batch, n_cities, head_size) - transformed states
    labels: (batch,) - class indices
    
    We want: logits = einsum('bck, k -> bc', A, w_right) ≈ one_hot(labels)
    This matches 04_train_probe.py: logits = einsum('bck,k->bc', hidden, w_right)
    """
    batch_size = A.shape[0]
    head_size = A.shape[2]
    
    # Create one-hot targets
    Y = torch.zeros(batch_size, n_cities, device=A.device)
    Y.scatter_(1, labels.unsqueeze(1), 1.0)  # (batch, n_cities)
    
    # Reshape for least squares: A_flat @ w_right = Y_flat
    # A_flat: (batch * n_cities, head_size)
    # Y_flat: (batch * n_cities,)
    A_flat = A.reshape(-1, head_size)  # (batch * n_cities, head_size)
    Y_flat = Y.reshape(-1)  # (batch * n_cities,)
    
    # Solve least squares: w_right = (A^T A)^{-1} A^T Y
    w_right = torch.linalg.lstsq(A_flat, Y_flat).solution  # (head_size,)
    
    return w_right

def evaluate_w_right(A, labels, w_right):
    """Compute accuracy given w_right and transformed states A."""
    # logits = einsum('bck, k -> bc', A, w_right)
    logits = torch.einsum('bck,k->bc', A, w_right)
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    return acc

# %%
def solve_probe(states_np, labels, layer_idx, head_idx):
    """Solve for w_right in closed form for a given layer/head."""
    states = torch.tensor(states_np[:, layer_idx, head_idx], dtype=torch.float32).to(device)
    
    n_samples = len(labels)
    n_val = max(1, int(0.3 * n_samples))
    indices = torch.randperm(n_samples)
    
    train_states = states[indices[n_val:]]
    train_labels = labels[indices[n_val:]].to(device)
    val_states = states[indices[:n_val]]
    val_labels = labels[indices[:n_val]].to(device)
    
    # Get model weights for this head and compute W_left_model
    head_o_proj = get_head_o_proj(model, layer_idx, head_idx, head_size, n_embd).to(device)
    W_left_model = get_W_left_model(head_o_proj, city_unembed)
    
    # Precompute transformed states: A = W_left_model @ state
    train_A = precompute_transformed_states(train_states, W_left_model)
    val_A = precompute_transformed_states(val_states, W_left_model)
    
    # Solve in closed form for w_right
    w_right = solve_w_right_closed_form(train_A, train_labels, n_cities)
    
    # Evaluate
    train_acc = evaluate_w_right(train_A, train_labels, w_right)
    val_acc = evaluate_w_right(val_A, val_labels, w_right)
    
    return val_acc, train_acc, w_right

# %%
print("Solving probes for each layer/head (closed form)...")
print(f"{'Layer':<6} {'Head':<6} {'Val Acc':<10} {'Train Acc':<10}")
print("-" * 35)

results = []
all_w_rights = {}
for layer_idx in range(num_layers):
    for head_idx in range(num_heads):
        val_acc, train_acc, w_right = solve_probe(all_states, labels, layer_idx, head_idx)
        all_w_rights[(layer_idx, head_idx)] = w_right
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
# ## 7. Re-train Best Probe and Test on New Dataset

# %%
best_head = results_df.sort_values('val_acc', ascending=False).iloc[0]
best_layer = int(best_head['layer'])
best_head_idx = int(best_head['head'])
print(f"\nBest head: Layer {best_layer}, Head {best_head_idx}")
print(f"Validation accuracy: {best_head['val_acc']:.3f}")

# %%
RETRAIN_DATASET_SIZE = None

if RETRAIN_DATASET_SIZE is not None:
    print(f"Generating {RETRAIN_DATASET_SIZE} samples for re-training...")
    retrain_dataset = MostHatedCityDataset(
        tokenizer=tokenizer,
        size=RETRAIN_DATASET_SIZE,
        n_entities=N_ENTITIES,
        n_cities=N_CITIES,
        fixed_entity_name=FIXED_ENTITY,
        seed=100,
    )
    
    print(f"Extracting states for best head (Layer {best_layer}, Head {best_head_idx})...")
    retrain_states = np.zeros((RETRAIN_DATASET_SIZE, head_size, head_size), dtype=np.float16)
    retrain_cities = []
    
    for idx in tqdm(range(RETRAIN_DATASET_SIZE), desc="Extracting states"):
        sample = retrain_dataset[idx]
        input_ids = sample.input_ids
        final_states = extractor.extract_final_states(input_ids)
        retrain_states[idx] = final_states[best_layer][best_head_idx].cpu().to(torch.float16).numpy()
        retrain_cities.append(sample.fixed_entity_city)
    
    retrain_labels = torch.tensor([CITY_TO_IDX[c] for c in retrain_cities])
    retrain_states_tensor = torch.tensor(retrain_states, dtype=torch.float32)
else:
    print("Reusing already extracted states...")
    retrain_states_tensor = torch.tensor(all_states[:, best_layer, best_head_idx], dtype=torch.float32)
    retrain_labels = labels
    RETRAIN_DATASET_SIZE = len(dataset)
    retrain_dataset = dataset

# %%
print(f"Solving probe on Layer {best_layer}, Head {best_head_idx} with {RETRAIN_DATASET_SIZE} samples (closed form)...")

n_samples = RETRAIN_DATASET_SIZE
n_val = max(1, int(0.3 * n_samples))
indices = torch.randperm(n_samples)

train_states = retrain_states_tensor[indices[n_val:]].to(device)
train_labels = retrain_labels[indices[n_val:]].to(device)
val_states = retrain_states_tensor[indices[:n_val]].to(device)
val_labels = retrain_labels[indices[:n_val]].to(device)

best_head_o_proj = get_head_o_proj(model, best_layer, best_head_idx, head_size, n_embd).to(device)
best_W_left_model = get_W_left_model(best_head_o_proj, city_unembed)

# Precompute transformed states: A = W_left_model @ state
train_A = precompute_transformed_states(train_states, best_W_left_model)
val_A = precompute_transformed_states(val_states, best_W_left_model)

# Solve in closed form for w_right
best_w_right = solve_w_right_closed_form(train_A, train_labels, n_cities)

# Evaluate
train_acc = evaluate_w_right(train_A, train_labels, best_w_right)
val_acc = evaluate_w_right(val_A, val_labels, best_w_right)

print(f"Train accuracy: {train_acc:.3f}")
print(f"Val accuracy: {val_acc:.3f}")
print("Best probe solved!")

# %%
print("\n=== Visualization: Probe Accuracy Across All Token Positions ===")

# Generate 3 samples with different sentence positions for the information
print("Generating 3 samples with varied information positions...")
temp_dataset = MostHatedCityDataset(
    tokenizer=tokenizer,
    size=100,
    n_entities=N_ENTITIES,
    n_cities=N_CITIES,
    fixed_entity_name=FIXED_ENTITY,
    seed=301,
)

# Collect samples and sort by sentence position
all_samples = [(s, s.fixed_entity_sentence_number) for s in temp_dataset]
all_samples.sort(key=lambda x: x[1])

# Pick 3 samples with varied positions (early, middle, late)
viz_samples = []
if len(all_samples) >= 3:
    indices = [len(all_samples) // 6, len(all_samples) // 2, 5 * len(all_samples) // 6]
    for idx in indices:
        sample, pos = all_samples[idx]
        viz_samples.append(sample)
        print(f"  Sample {len(viz_samples)}: Info at sentence {pos}")

print(f"\nExtracting states for all token positions...")

import matplotlib.pyplot as plt
import time

def safe_savefig(filename, **kwargs):
    """Save figure with retry logic to handle stale file handles."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            plt.savefig(filename, **kwargs)
            return
        except OSError as e:
            if attempt < max_retries - 1 and "Stale file handle" in str(e):
                print(f"Stale file handle, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(0.5)
            else:
                raise

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for sample_idx, sample in enumerate(viz_samples):
    print(f"\nProcessing sample {sample_idx + 1}/3...")
    input_ids = sample.input_ids
    seq_len = input_ids.shape[1]
    info_idx = sample.fixed_entity_sentence_end_token_idx
    true_city_idx = CITY_TO_IDX[sample.fixed_entity_city]
    
    # Extract incremental states for all positions
    incremental_states = extractor.extract_incremental_states_single_pass(
        input_ids,
        layers=[best_layer],
        use_tqdm=False
    )
    
    # Compute accuracy at each position (top-1 and top-3)
    accuracies_top1 = []
    accuracies_top3 = []
    positions = []
    
    with torch.no_grad():
        for pos in range(1, seq_len):
            if pos not in incremental_states:
                continue
            
            state_at_pos = incremental_states[pos][best_layer]
            state_tensor = torch.tensor(state_at_pos[best_head_idx], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Compute prediction using model weights
            A = precompute_transformed_states(state_tensor, best_W_left_model)
            logits = torch.einsum('bck,k->bc', A, best_w_right)
            
            # Top-1 accuracy
            pred_top1 = logits.argmax(dim=1).item()
            is_correct_top1 = (pred_top1 == true_city_idx)
            
            # Top-3 accuracy
            top3_preds = logits.topk(3, dim=1).indices[0].tolist()
            is_correct_top3 = (true_city_idx in top3_preds)
            
            accuracies_top1.append(float(is_correct_top1))
            accuracies_top3.append(float(is_correct_top3))
            positions.append(pos)
    
    # Plot
    ax = axes[sample_idx]
    ax.plot(positions, accuracies_top1, 'b-', linewidth=2, alpha=0.7, label='Top-1')
    ax.plot(positions, accuracies_top3, 'g-', linewidth=2, alpha=0.7, label='Top-3')
    ax.axvline(x=info_idx, color='r', linestyle='--', linewidth=2, label=f'Info given (sentence {sample.fixed_entity_sentence_number})')
    ax.axhline(y=1/n_cities, color='gray', linestyle=':', linewidth=1, alpha=0.5, label=f'Random top-1 ({1/n_cities:.2f})')
    ax.axhline(y=3/n_cities, color='lightgray', linestyle=':', linewidth=1, alpha=0.5, label=f'Random top-3 ({3/n_cities:.2f})')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Correct (1) / Incorrect (0)')
    ax.set_title(f'Sample {sample_idx + 1}: City = {sample.fixed_entity_city}, Info at token {info_idx} (sentence {sample.fixed_entity_sentence_number})')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
safe_savefig('../data/rwkv_probe_unembed_cities_accuracy_all_positions.png', dpi=150)
plt.show()

print("\nVisualization complete!")

# %%
TARGET_EVAL_SAMPLES = 100
EVAL_N_ENTITIES = 500

print(f"\nGenerating {TARGET_EVAL_SAMPLES} evaluation samples with {FIXED_ENTITY} at sentences 10-20...")
eval_samples = []
seed_offset = 43

pbar = tqdm(total=TARGET_EVAL_SAMPLES, desc="Generating valid samples")
while len(eval_samples) < TARGET_EVAL_SAMPLES:
    temp_dataset = MostHatedCityDataset(
        tokenizer=tokenizer,
        size=100,
        n_entities=EVAL_N_ENTITIES,
        n_cities=N_CITIES,
        fixed_entity_name=FIXED_ENTITY,
        seed=seed_offset,
    )
    
    for sample in temp_dataset:
        if 10 <= sample.fixed_entity_sentence_number <= 20:
            eval_samples.append(sample)
            pbar.update(1)
            if len(eval_samples) >= TARGET_EVAL_SAMPLES:
                break
    
    seed_offset += 1

pbar.close()
print(f"Generated {len(eval_samples)} valid samples")

# %%
print(f"Extracting incremental states for evaluation...")
print(f"Note: This extracts states token-by-token, so will be slow")

eval_accuracies_by_position_top1 = {}
eval_accuracies_by_position_top3 = {}

eval_extractor = RWKVStateExtractor(model, verbose=False)

for sample_idx, sample in enumerate(tqdm(eval_samples, desc="Samples")):
    input_ids = sample.input_ids
    seq_len = input_ids.shape[1]
    
    incremental_states = eval_extractor.extract_incremental_states_single_pass(
        input_ids,
        layers=[best_layer],
        use_tqdm=True
    )
    
    info_idx = sample.fixed_entity_sentence_end_token_idx
    true_city_idx = CITY_TO_IDX[sample.fixed_entity_city]
    
    with torch.no_grad():
        for pos in range(info_idx + 1, seq_len):
            if pos not in incremental_states:
                continue
            
            state_at_pos = incremental_states[pos][best_layer]
            state_tensor = torch.tensor(state_at_pos[best_head_idx], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Compute prediction using model weights
            A = precompute_transformed_states(state_tensor, best_W_left_model)
            logits = torch.einsum('bck,k->bc', A, best_w_right)
            
            # Top-1 accuracy
            pred_top1 = logits.argmax(dim=1).item()
            is_correct_top1 = (pred_top1 == true_city_idx)
            
            # Top-3 accuracy
            top3_preds = logits.topk(3, dim=1).indices[0].tolist()
            is_correct_top3 = (true_city_idx in top3_preds)
            
            relative_pos = pos - info_idx
            
            if relative_pos not in eval_accuracies_by_position_top1:
                eval_accuracies_by_position_top1[relative_pos] = []
                eval_accuracies_by_position_top3[relative_pos] = []
            eval_accuracies_by_position_top1[relative_pos].append(is_correct_top1)
            eval_accuracies_by_position_top3[relative_pos].append(is_correct_top3)

# %%
import matplotlib.pyplot as plt

positions = sorted(eval_accuracies_by_position_top1.keys())
mean_accs_top1 = [np.mean(eval_accuracies_by_position_top1[p]) for p in positions]
mean_accs_top3 = [np.mean(eval_accuracies_by_position_top3[p]) for p in positions]

plt.figure(figsize=(12, 6))
plt.plot(positions, mean_accs_top1, 'b-', linewidth=2, label='Top-1 Accuracy')
plt.plot(positions, mean_accs_top3, 'g-', linewidth=2, label='Top-3 Accuracy')
plt.axhline(y=1/n_cities, color='r', linestyle='--', label=f'Random Baseline Top-1 ({1/n_cities:.3f})')
plt.axhline(y=3/n_cities, color='orange', linestyle='--', label=f'Random Baseline Top-3 ({3/n_cities:.3f})')
plt.xlabel('Tokens after Information Given')
plt.ylabel('Accuracy')
plt.title(f'RWKV Unembed Probe Accuracy vs Position (Cities)\nLayer {best_layer}, Head {best_head_idx}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
safe_savefig('../data/rwkv_probe_unembed_cities_accuracy_by_position.png', dpi=150)
plt.show()

# Plot rolling window accuracy
window_size = 100
plt.figure(figsize=(12, 8))

n_samples = len(eval_samples)
sample_trajectories_top1 = []

for sample_idx in range(n_samples):
    trajectory = []
    for pos in positions:
        accs_at_pos = eval_accuracies_by_position_top1.get(pos, [])
        if sample_idx < len(accs_at_pos):
            trajectory.append(float(accs_at_pos[sample_idx]))
        else:
            trajectory.append(np.nan)
    sample_trajectories_top1.append(trajectory)

for sample_idx, trajectory in enumerate(sample_trajectories_top1):
    rolling_traj = []
    rolling_pos = []
    for i in range(len(trajectory) - window_size + 1):
        window = trajectory[i:i + window_size]
        valid_vals = [v for v in window if not np.isnan(v)]
        if valid_vals:
            rolling_traj.append(np.mean(valid_vals))
            rolling_pos.append(positions[i + window_size // 2])
    
    if rolling_traj:
        plt.plot(rolling_pos, rolling_traj, alpha=0.6, linewidth=1.5)

plt.axhline(y=1/n_cities, color='r', linestyle='--', linewidth=2, label=f'Random Baseline ({1/n_cities:.3f})')
plt.xlabel('Tokens after Information Given')
plt.ylabel('Top-1 Accuracy')
plt.title(f'RWKV Unembed Probe Rolling Window Accuracies (Cities, window={window_size})\nLayer {best_layer}, Head {best_head_idx}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
safe_savefig('../data/rwkv_probe_unembed_cities_rolling_window.png', dpi=150)
plt.show()

print(f"\n=== Summary Statistics ===")
print(f"Top-1 Accuracy:")
print(f"  Mean across all positions: {np.mean(mean_accs_top1):.3f}")
print(f"  At position +1: {mean_accs_top1[0]:.3f}")
print(f"  Final: {mean_accs_top1[-1]:.3f}")
print(f"\nTop-3 Accuracy:")
print(f"  Mean across all positions: {np.mean(mean_accs_top3):.3f}")
print(f"  At position +1: {mean_accs_top3[0]:.3f}")
print(f"  Final: {mean_accs_top3[-1]:.3f}")

# # %%

# %%
