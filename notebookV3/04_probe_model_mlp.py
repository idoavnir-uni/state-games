# %% [markdown]
# # RWKV State Probing with Full Model Path (GPU 4)
# 
# This notebook extends 02_probe_unembeding.py by following the full model path:
# 1. Query state with learned w_right
# 2. Apply output projection (from model)
# 3. Apply LayerNorm for FFN (ln2)
# 4. Pass through FFN/MLP with residual connection
# 5. Apply final LayerNorm (ln_out)
# 6. Unembedding to colors
#
# **Flow:** state → (w_right query) → o_proj → ln2 → FFN + residual → ln_out → unembed

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
from datasets.favorite_color_dataset import FavoriteColorDataset

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
# ## 2. Explore Model Weights Structure

# %%
z = model.model.z
print("\n=== Model Weight Keys (Block 0) ===")
block0_keys = sorted([k for k in z.keys() if k.startswith('blocks.0.')])
for k in block0_keys:
    print(f"  {k}: {z[k].shape}")

print("\n=== Output Layer Keys ===")
output_keys = sorted([k for k in z.keys() if 'ln_out' in k or k == 'head.weight'])
for k in output_keys:
    print(f"  {k}: {z[k].shape}")

# %% [markdown]
# ## 3. Initialize State Extractor

# %%
extractor = RWKVStateExtractor(model, verbose=False)
head_size = extractor.head_size
n_head = extractor.n_head
n_embd = model.model.n_embd
print(f"Head size: {head_size}")
print(f"Num heads: {n_head}")
print(f"Hidden size: {n_embd}")

# %% [markdown]
# ## 4. Create Favorite Color Dataset

# %%
DATASET_SIZE = 100
N_ENTITIES = 30
N_COLORS = 10
FIXED_ENTITY = "Lady Gaga"

print(f"Creating dataset with {DATASET_SIZE} samples...")
dataset = FavoriteColorDataset(
    tokenizer=tokenizer,
    size=DATASET_SIZE,
    n_entities=N_ENTITIES,
    n_colors=N_COLORS,
    fixed_entity_name=FIXED_ENTITY,
    seed=42,
)
print(f"Dataset created with {len(dataset)} samples")
print(f"Fixed entity: {dataset.fixed_entity_name}")
print(f"Colors used: {dataset.colors}")

# %% [markdown]
# ## 4.1 Verify Colors are Single Tokens

# %%
print("\nVerifying colors are single tokens...")
color_token_ids = {}
all_single_tokens = True

for color in dataset.colors:
    tokens = tokenizer.encode(color)
    is_single = len(tokens) == 1
    
    if is_single:
        color_token_ids[color] = tokens[0]
        token_str = tokenizer.decode([tokens[0]])
        print(f"  {color}: token_id={tokens[0]}, decoded='{token_str}' ✓")
    else:
        all_single_tokens = False
        decoded = [tokenizer.decode([t]) for t in tokens]
        print(f"  {color}: {len(tokens)} tokens {tokens} -> {decoded} ✗")

if all_single_tokens:
    print("\n✓ All colors are single tokens!")
else:
    print("\n✗ Some colors are NOT single tokens. Need to filter or change colors.")
    raise ValueError("Not all colors are single tokens")

COLOR_TOKEN_IDS = torch.tensor([color_token_ids[c] for c in dataset.colors])

# %% [markdown]
# ## 5. Extract States and Build Dataframe

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
        'target_color': sample.fixed_entity_color,
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
# ## 6. Display Sample Information

# %%
print("\n=== Sample Entry ===")
print(f"Sentence: {df.iloc[0]['sentence']}")
print(f"Target color: {df.iloc[0]['target_color']}")
print(f"States shape: {all_states.shape} = (samples, layers, heads, {head_size}, {head_size})")

print("\n=== Dataset Statistics ===")
print(f"Total samples: {len(df)}")
print(f"Color distribution:\n{df['target_color'].value_counts()}")

# %% [markdown]
# ## 7. Extract Model Weights for Full Path
#
# We need:
# - Output projection (per head): blocks.{layer}.att.output.weight
# - FFN layer norm: blocks.{layer}.ln2.weight, blocks.{layer}.ln2.bias
# - FFN key/value: blocks.{layer}.ffn.key.weight, blocks.{layer}.ffn.value.weight
# - Final layer norm: ln_out.weight, ln_out.bias
# - Unembedding: head.weight

# %%
import torch.nn as nn
import torch.optim as optim

COLOR_TO_IDX = {color: idx for idx, color in enumerate(dataset.colors)}
labels = torch.tensor([COLOR_TO_IDX[c] for c in df['target_color']])
n_colors = len(dataset.colors)

print(f"Color mapping: {COLOR_TO_IDX}")

# Get unembedding matrix from model (LM head)
lm_head = model.model.z['head.weight'].detach()
print(f"LM head shape: {lm_head.shape}")

# Extract only the columns for color tokens
color_unembed = lm_head[:, COLOR_TOKEN_IDS].float().to(device)
print(f"Color unembedding shape: {color_unembed.shape}")

# Get final layer norm
ln_out_weight = z['ln_out.weight'].detach().float().to(device)
ln_out_bias = z['ln_out.bias'].detach().float().to(device)
print(f"ln_out weight shape: {ln_out_weight.shape}")

# %%
def get_layer_weights(model, layer_idx, head_idx, head_size=64, n_embd=2048):
    """Extract all relevant weights for processing through a layer."""
    z = model.model.z
    
    # Output projection for this head (ROWS for out @ O_)
    o_proj = z[f'blocks.{layer_idx}.att.output.weight'].detach().float()
    start_idx = head_idx * head_size
    end_idx = (head_idx + 1) * head_size
    head_o_proj = o_proj[start_idx:end_idx, :]  # (head_size, n_embd)
    
    # FFN layer norm (ln2)
    ln2_weight = z[f'blocks.{layer_idx}.ln2.weight'].detach().float()
    ln2_bias = z[f'blocks.{layer_idx}.ln2.bias'].detach().float()
    
    # FFN weights
    ffn_key = z[f'blocks.{layer_idx}.ffn.key.weight'].detach().float()
    ffn_value = z[f'blocks.{layer_idx}.ffn.value.weight'].detach().float()
    
    # Check if receptance exists (some RWKV versions have it)
    ffn_receptance_key = f'blocks.{layer_idx}.ffn.receptance.weight'
    ffn_receptance = z[ffn_receptance_key].detach().float() if ffn_receptance_key in z else None
    
    return {
        'head_o_proj': head_o_proj,
        'ln2_weight': ln2_weight,
        'ln2_bias': ln2_bias,
        'ffn_key': ffn_key,
        'ffn_value': ffn_value,
        'ffn_receptance': ffn_receptance,
    }

# Test extraction
test_weights = get_layer_weights(model, 0, 0, head_size, n_embd)
print("\n=== Layer Weights Shapes ===")
for k, v in test_weights.items():
    if v is not None:
        print(f"  {k}: {v.shape}")
    else:
        print(f"  {k}: None")

# %%
def layer_norm(x, weight, bias, eps=1e-5):
    """Apply layer normalization."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return weight * (x - mean) / torch.sqrt(var + eps) + bias

def ffn_forward(x, ffn_key, ffn_value, ffn_receptance=None):
    """
    Simplified RWKV-7 FFN (channel mixing).
    
    The actual RWKV-7 FFN does token shift mixing:
        xx = x_prev - x
        k = x + xx * x_k  # Token shift mixing with x_k weights
        k = relu(k @ K)^2
        out = k @ V
    
    Since we don't have x_prev (previous token's hidden state), we skip the token shift:
        k = relu(x @ K)^2
        out = k @ V
    
    Weight shapes:
    - ffn_key: (n_embd, ffn_dim)
    - ffn_value: (ffn_dim, n_embd)
    """
    # Key projection: (batch, n_embd) @ (n_embd, ffn_dim) -> (batch, ffn_dim)
    k = x @ ffn_key  # (batch, 8192)
    
    # Squared ReLU activation (RWKV uses this)
    k = torch.relu(k) ** 2
    
    # Value projection: (batch, ffn_dim) @ (ffn_dim, n_embd) -> (batch, n_embd)
    out = k @ ffn_value  # (batch, 2048)
    
    # Receptance gating if available
    if ffn_receptance is not None:
        r = torch.sigmoid(x @ ffn_receptance)
        out = r * out
    
    return out

def process_through_model_path(state, w_right, weights, ln_out_weight, ln_out_bias, color_unembed):
    """
    Process state through the full model path:
    1. Query state with w_right: o = state @ w_right → (batch, head_size)
    2. Output projection: hidden = o @ head_o_proj → (batch, n_embd)
    3. Layer norm (ln2): hidden_ln = ln2(hidden)
    4. FFN with residual: hidden = hidden + ffn(hidden_ln)
    5. Final layer norm: hidden_ln_out = ln_out(hidden)
    6. Unembedding to colors: logits = hidden_ln_out @ color_unembed
    
    Note: This is a simplified version of what the model does. The actual model:
    - Uses the FULL hidden state (input + attention), not just attention output
    - Uses token shift mixing in FFN (x + (x_prev - x) * x_k)
    Without these, the FFN may not work as intended.
    """
    # 1. Query state with w_right
    o = torch.einsum('bdk,k->bd', state, w_right)  # (batch, head_size)
    
    # 2. Output projection
    hidden = o @ weights['head_o_proj']  # (batch, n_embd)
    
    # 3. Layer norm (ln2)
    hidden_ln = layer_norm(hidden, weights['ln2_weight'], weights['ln2_bias'])
    
    # 4. FFN with residual
    ffn_out = ffn_forward(hidden_ln, weights['ffn_key'], weights['ffn_value'], weights['ffn_receptance'])
    hidden = hidden + ffn_out  # Residual connection
    
    # 5. Final layer norm
    hidden_ln_out = layer_norm(hidden, ln_out_weight, ln_out_bias)
    
    # 6. Unembedding to colors
    logits = hidden_ln_out @ color_unembed
    
    return logits

def process_simple_no_ffn(state, w_right, weights, ln_out_weight, ln_out_bias, color_unembed):
    """Simplified version without FFN - skips ln2 and FFN, goes directly to ln_out."""
    # 1. Query state with w_right
    o = torch.einsum('bdk,k->bd', state, w_right)  # (batch, head_size)
    
    # 2. Output projection
    hidden = o @ weights['head_o_proj']  # (batch, n_embd)
    
    # 3. Final layer norm only (skip FFN)
    hidden_ln_out = layer_norm(hidden, ln_out_weight, ln_out_bias)
    
    # 4. Unembedding to colors
    logits = hidden_ln_out @ color_unembed
    
    return logits

# %% [markdown]
# ## 8. Train Probe with Full Model Path

# %%
# USE_FFN controls whether to use the FFN in the probe.
# The FFN implementation is simplified (no token shift mixing, only attention output as input).
# The actual RWKV FFN operates on the FULL hidden state and uses token shift mixing.
USE_FFN = True

def train_probe(states_np, labels, layer_idx, head_idx, patience=20, batch_size=1000):
    """Train w_right with gradient descent using batches, following full model path."""
    states = torch.tensor(states_np[:, layer_idx, head_idx], dtype=torch.float32)
    
    n_samples = len(labels)
    n_val = max(1, int(0.3 * n_samples))
    indices = torch.randperm(n_samples)
    
    train_states = states[indices[n_val:]].to(device)
    train_labels = labels[indices[n_val:]].to(device)
    val_states = states[indices[:n_val]].to(device)
    val_labels = labels[indices[:n_val]].to(device)
    
    # Get layer weights (frozen)
    weights = get_layer_weights(model, layer_idx, head_idx, head_size, n_embd)
    weights = {k: v.to(device) if v is not None else None for k, v in weights.items()}
    
    # Initialize learnable w_right
    w_right = nn.Parameter(torch.randn(head_size, device=device) * 0.01)
    optimizer = optim.Adam([w_right], lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_w_right = None
    patience_counter = 0
    
    n_train = len(train_states)
    
    # Choose forward function based on USE_FFN flag
    forward_fn = process_through_model_path if USE_FFN else process_simple_no_ffn
    
    for epoch in range(20000):
        # Training with batches
        batch_indices = torch.randperm(n_train, device=device)
        
        for i in range(0, n_train, batch_size):
            batch_idx = batch_indices[i:i+batch_size]
            batch_states = train_states[batch_idx]
            batch_labels = train_labels[batch_idx]
            
            optimizer.zero_grad()
            logits = forward_fn(
                batch_states, w_right, weights, ln_out_weight, ln_out_bias, color_unembed
            )
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        with torch.no_grad():
            val_logits = forward_fn(
                val_states, w_right, weights, ln_out_weight, ln_out_bias, color_unembed
            )
            val_loss = criterion(val_logits, val_labels).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels).float().mean().item()
            
            train_logits = forward_fn(
                train_states, w_right, weights, ln_out_weight, ln_out_bias, color_unembed
            )
            train_preds = train_logits.argmax(dim=1)
            train_acc = (train_preds == train_labels).float().mean().item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_w_right = w_right.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return best_val_acc, train_acc, best_w_right, epoch + 1

# %%
print("Training probes for each layer/head...")
print(f"{'Layer':<6} {'Head':<6} {'Val Acc':<10} {'Train Acc':<10} {'Epochs':<8}")
print("-" * 45)

results = []
all_w_rights = {}
for layer_idx in range(num_layers):
    for head_idx in range(num_heads):
        val_acc, train_acc, w_right, epochs = train_probe(all_states, labels, layer_idx, head_idx)
        all_w_rights[(layer_idx, head_idx)] = w_right
        results.append({
            'layer': layer_idx,
            'head': head_idx,
            'val_acc': val_acc,
            'train_acc': train_acc,
            'epochs': epochs
        })
        print(f"L{layer_idx:<5} H{head_idx:<5} {val_acc:<10.3f} {train_acc:<10.3f} {epochs:<8}")

# %%
results_df = pd.DataFrame(results)
print("\n=== Best Performing Heads ===")
print(results_df.sort_values('val_acc', ascending=False).head(10))

# %% [markdown]
# ## 9. Re-train Best Probe

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
    retrain_dataset = FavoriteColorDataset(
        tokenizer=tokenizer,
        size=RETRAIN_DATASET_SIZE,
        n_entities=N_ENTITIES,
        n_colors=N_COLORS,
        fixed_entity_name=FIXED_ENTITY,
        seed=100,
    )
    
    print(f"Extracting states for best head (Layer {best_layer}, Head {best_head_idx})...")
    retrain_states = np.zeros((RETRAIN_DATASET_SIZE, head_size, head_size), dtype=np.float16)
    retrain_colors = []
    
    for idx in tqdm(range(RETRAIN_DATASET_SIZE), desc="Extracting states"):
        sample = retrain_dataset[idx]
        input_ids = sample.input_ids
        final_states = extractor.extract_final_states(input_ids)
        retrain_states[idx] = final_states[best_layer][best_head_idx].cpu().to(torch.float16).numpy()
        retrain_colors.append(sample.fixed_entity_color)
    
    retrain_labels = torch.tensor([COLOR_TO_IDX[c] for c in retrain_colors])
    retrain_states_tensor = torch.tensor(retrain_states, dtype=torch.float32)
else:
    print("Reusing already extracted states...")
    retrain_states_tensor = torch.tensor(all_states[:, best_layer, best_head_idx], dtype=torch.float32)
    retrain_labels = labels
    RETRAIN_DATASET_SIZE = len(dataset)
    retrain_dataset = dataset

# %%
print(f"Training probe on Layer {best_layer}, Head {best_head_idx} with {RETRAIN_DATASET_SIZE} samples...")

n_samples = RETRAIN_DATASET_SIZE
n_val = max(1, int(0.3 * n_samples))
indices = torch.randperm(n_samples)

train_states = retrain_states_tensor[indices[n_val:]].to(device)
train_labels = retrain_labels[indices[n_val:]].to(device)
val_states = retrain_states_tensor[indices[:n_val]].to(device)
val_labels = retrain_labels[indices[:n_val]].to(device)

# Get layer weights (frozen)
best_weights = get_layer_weights(model, best_layer, best_head_idx, head_size, n_embd)
best_weights = {k: v.to(device) if v is not None else None for k, v in best_weights.items()}

# Initialize learnable w_right
best_w_right = nn.Parameter(torch.randn(head_size, device=device) * 0.01)
optimizer = optim.Adam([best_w_right], lr=1e-3)
criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')
patience_counter = 0
patience = 20
batch_size = 1000
n_train = len(train_states)

best_forward_fn = process_through_model_path if USE_FFN else process_simple_no_ffn

pbar = tqdm(range(20000), desc="Training")
for epoch in pbar:
    batch_indices = torch.randperm(n_train, device=device)
    
    for i in range(0, n_train, batch_size):
        batch_idx = batch_indices[i:i+batch_size]
        batch_states = train_states[batch_idx]
        batch_labels = train_labels[batch_idx]
        
        optimizer.zero_grad()
        logits = best_forward_fn(
            batch_states, best_w_right, best_weights, ln_out_weight, ln_out_bias, color_unembed
        )
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        val_logits = best_forward_fn(
            val_states, best_w_right, best_weights, ln_out_weight, ln_out_bias, color_unembed
        )
        val_loss = criterion(val_logits, val_labels).item()
        val_preds = val_logits.argmax(dim=1)
        val_acc = (val_preds == val_labels).float().mean().item()
        
        train_logits = best_forward_fn(
            train_states, best_w_right, best_weights, ln_out_weight, ln_out_bias, color_unembed
        )
        train_preds = train_logits.argmax(dim=1)
        train_acc = (train_preds == train_labels).float().mean().item()
    
    pbar.set_postfix({'train_acc': f'{train_acc:.3f}', 'val_acc': f'{val_acc:.3f}'})
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        break

print(f"Train accuracy: {train_acc:.3f}")
print(f"Val accuracy: {val_acc:.3f}")
print("Best probe trained!")

# %%
print("\n=== Visualization: Probe Accuracy Across All Token Positions ===")

print("Generating 3 samples with varied information positions...")
temp_dataset = FavoriteColorDataset(
    tokenizer=tokenizer,
    size=100,
    n_entities=N_ENTITIES,
    n_colors=N_COLORS,
    fixed_entity_name=FIXED_ENTITY,
    seed=300,
)

all_samples = [(s, s.fixed_entity_sentence_number) for s in temp_dataset]
all_samples.sort(key=lambda x: x[1])

viz_samples = []
if len(all_samples) >= 3:
    indices = [len(all_samples) // 6, len(all_samples) // 2, 5 * len(all_samples) // 6]
    for idx in indices:
        sample, pos = all_samples[idx]
        viz_samples.append(sample)
        print(f"  Sample {len(viz_samples)}: Info at sentence {pos}")

print(f"\nExtracting states for all token positions...")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for sample_idx, sample in enumerate(viz_samples):
    print(f"\nProcessing sample {sample_idx + 1}/3...")
    input_ids = sample.input_ids
    seq_len = input_ids.shape[1]
    info_idx = sample.fixed_entity_sentence_end_token_idx
    true_color_idx = COLOR_TO_IDX[sample.fixed_entity_color]
    
    incremental_states = extractor.extract_incremental_states_single_pass(
        input_ids,
        layers=[best_layer],
        use_tqdm=False
    )
    
    accuracies_top1 = []
    accuracies_top3 = []
    positions = []
    
    with torch.no_grad():
        for pos in range(1, seq_len):
            if pos not in incremental_states:
                continue
            
            state_at_pos = incremental_states[pos][best_layer]
            state_tensor = torch.tensor(state_at_pos[best_head_idx], dtype=torch.float32).unsqueeze(0).to(device)
            
            logits = best_forward_fn(
                state_tensor, best_w_right, best_weights, ln_out_weight, ln_out_bias, color_unembed
            )
            
            pred_top1 = logits.argmax(dim=1).item()
            is_correct_top1 = (pred_top1 == true_color_idx)
            
            top3_preds = logits.topk(3, dim=1).indices[0].tolist()
            is_correct_top3 = (true_color_idx in top3_preds)
            
            accuracies_top1.append(float(is_correct_top1))
            accuracies_top3.append(float(is_correct_top3))
            positions.append(pos)
    
    ax = axes[sample_idx]
    ax.plot(positions, accuracies_top1, 'b-', linewidth=2, alpha=0.7, label='Top-1')
    ax.plot(positions, accuracies_top3, 'g-', linewidth=2, alpha=0.7, label='Top-3')
    ax.axvline(x=info_idx, color='r', linestyle='--', linewidth=2, label=f'Info given (sentence {sample.fixed_entity_sentence_number})')
    ax.axhline(y=1/n_colors, color='gray', linestyle=':', linewidth=1, alpha=0.5, label=f'Random top-1 ({1/n_colors:.2f})')
    ax.axhline(y=3/n_colors, color='lightgray', linestyle=':', linewidth=1, alpha=0.5, label=f'Random top-3 ({3/n_colors:.2f})')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Correct (1) / Incorrect (0)')
    ax.set_title(f'Sample {sample_idx + 1}: Color = {sample.fixed_entity_color}, Info at token {info_idx} (sentence {sample.fixed_entity_sentence_number})')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../data/rwkv_probe_model_mlp_accuracy_all_positions.png', dpi=150)
plt.show()

print("\nVisualization complete!")

# %%
TARGET_EVAL_SAMPLES = 20
EVAL_N_ENTITIES = 100

print(f"\nGenerating {TARGET_EVAL_SAMPLES} evaluation samples with {FIXED_ENTITY} at sentences 10-20...")
eval_samples = []
seed_offset = 43

pbar = tqdm(total=TARGET_EVAL_SAMPLES, desc="Generating valid samples")
while len(eval_samples) < TARGET_EVAL_SAMPLES:
    temp_dataset = FavoriteColorDataset(
        tokenizer=tokenizer,
        size=100,
        n_entities=EVAL_N_ENTITIES,
        n_colors=N_COLORS,
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
    true_color_idx = COLOR_TO_IDX[sample.fixed_entity_color]
    
    with torch.no_grad():
        for pos in range(info_idx + 1, seq_len):
            if pos not in incremental_states:
                continue
            
            state_at_pos = incremental_states[pos][best_layer]
            state_tensor = torch.tensor(state_at_pos[best_head_idx], dtype=torch.float32).unsqueeze(0).to(device)
            
            logits = best_forward_fn(
                state_tensor, best_w_right, best_weights, ln_out_weight, ln_out_bias, color_unembed
            )
            
            pred_top1 = logits.argmax(dim=1).item()
            is_correct_top1 = (pred_top1 == true_color_idx)
            
            top3_preds = logits.topk(3, dim=1).indices[0].tolist()
            is_correct_top3 = (true_color_idx in top3_preds)
            
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
plt.axhline(y=1/n_colors, color='r', linestyle='--', label=f'Random Baseline Top-1 ({1/n_colors:.3f})')
plt.axhline(y=3/n_colors, color='orange', linestyle='--', label=f'Random Baseline Top-3 ({3/n_colors:.3f})')
plt.xlabel('Tokens after Information Given')
plt.ylabel('Accuracy')
plt.title(f'RWKV Model MLP Probe Accuracy vs Position\nLayer {best_layer}, Head {best_head_idx}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../data/rwkv_probe_model_mlp_accuracy_by_position.png', dpi=150)
plt.show()

window_size = 50
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

plt.axhline(y=1/n_colors, color='r', linestyle='--', linewidth=2, label=f'Random Baseline ({1/n_colors:.3f})')
plt.xlabel('Tokens after Information Given')
plt.ylabel('Top-1 Accuracy')
plt.title(f'RWKV Model MLP Probe Rolling Window Accuracies (window={window_size})\nLayer {best_layer}, Head {best_head_idx}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../data/rwkv_probe_model_mlp_rolling_window.png', dpi=150)
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

# %%

