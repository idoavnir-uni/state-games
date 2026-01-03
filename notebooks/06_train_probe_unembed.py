# %% [markdown]
# # State Probing with Unembedding Matrix
# 
# This notebook:
# 1. Generates favorite color sentences with a fixed entity
# 2. Extracts GLA model states at the final token position
# 3. Uses learned w_left + model's unembedding matrix to predict colors
# 
# **Probe model:** logits = (w_left @ state) @ W_unembed[:, color_tokens]
# - state: (256, 512) per head
# - w_left: (256,) learned vector → projects to (512,)
# - W_unembed: model's unembedding matrix (hidden_size, vocab_size)
# - We only look at logits for color tokens

# %%
import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

sys.path.insert(0, os.path.abspath('..'))

from models.load_gla import load_gla_model, get_model_config
from models.state_extractor_gla import GLAStateExtractor
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
# ## 3. Create Favorite Color Dataset

# %%
DATASET_SIZE = 2000
N_ENTITIES = 10
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
# ## 3.1 Verify Colors are Single Tokens

# %%
print("\nVerifying colors are single tokens...")
color_token_ids = {}
all_single_tokens = True

for color in dataset.colors:
    tokens = tokenizer.encode(color, add_special_tokens=False)
    is_single = len(tokens) == 1
    
    if is_single:
        color_token_ids[color] = tokens[0]
        token_str = tokenizer.decode(tokens[0])
        print(f"  {color}: token_id={tokens[0]}, decoded='{token_str}' ✓")
    else:
        all_single_tokens = False
        decoded = [tokenizer.decode(t) for t in tokens]
        print(f"  {color}: {len(tokens)} tokens {tokens} -> {decoded} ✗")

if all_single_tokens:
    print("\n✓ All colors are single tokens!")
else:
    print("\n✗ Some colors are NOT single tokens. Need to filter or change colors.")
    raise ValueError("Not all colors are single tokens")

COLOR_TOKEN_IDS = torch.tensor([color_token_ids[c] for c in dataset.colors])

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
        'target_color': sample.fixed_entity_color,
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
# ## 5. Display Sample Information

# %%
print("\n=== Sample Entry ===")
print(f"Sentence: {df.iloc[0]['sentence']}")
print(f"Target color: {df.iloc[0]['target_color']}")
print(f"States shape: {all_states.shape} = (samples, layers, heads, 256, 512)")

print("\n=== Dataset Statistics ===")
print(f"Total samples: {len(df)}")
print(f"Color distribution:\n{df['target_color'].value_counts()}")

# %% [markdown]
# ## 6. Train Linear Probe with Unembedding Matrix
# 
# Model: logits = (w_left @ state) @ W_unembed[:, color_token_ids]
# - state: (256, 512)
# - w_left: (256,) learned vector → w_left @ state = (512,)
# - W_unembed: model's lm_head.weight (vocab_size, hidden_size)
# - We only look at logits for color token IDs

# %%
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

COLOR_TO_IDX = {color: idx for idx, color in enumerate(dataset.colors)}
labels = torch.tensor([COLOR_TO_IDX[c] for c in df['target_color']])
n_colors = len(dataset.colors)

print(f"Color mapping: {COLOR_TO_IDX}")

# Get unembedding matrix from model
lm_head = model.lm_head.weight.detach()  # (vocab_size, hidden_size)
hidden_size = lm_head.shape[1]
num_heads = config.get('num_heads')  # 4 heads
head_dim = 512  # value dim per head

print(f"LM head shape: {lm_head.shape}")
print(f"Model hidden size: {hidden_size}")
print(f"Num heads: {num_heads}, head dim: {head_dim}")

# Extract only the rows for color tokens (convert to float32)
color_unembed = lm_head[COLOR_TOKEN_IDS].float().to(device)  # (n_colors, hidden_size)
print(f"Color unembedding shape: {color_unembed.shape}")

# %%
def get_head_o_proj(model, layer_idx, head_idx, head_dim=512, hidden_size=2048):
    """Extract the output projection weights for a specific head from a layer."""
    # o_proj.weight is (hidden_size, hidden_size) = (2048, 2048)
    # It projects from concatenated head outputs to hidden
    # For head i, input is positions [i*head_dim : (i+1)*head_dim]
    o_proj = model.model.layers[layer_idx].attn.o_proj.weight.detach().float()  # (2048, 2048)
    # Extract the columns corresponding to this head
    start_idx = head_idx * head_dim
    end_idx = (head_idx + 1) * head_dim
    head_o_proj = o_proj[:, start_idx:end_idx]  # (2048, 512)
    return head_o_proj

# %%
def precompute_transformed_states(states, head_o_proj, color_unembed):
    """
    Precompute A = states @ head_o_proj.T @ color_unembed.T
    
    states: (batch, 256, 512)
    head_o_proj: (2048, 512) 
    color_unembed: (n_colors, 2048)
    
    Returns A: (batch, 256, n_colors)
    """
    # states @ head_o_proj.T: (batch, 256, 512) @ (512, 2048) -> (batch, 256, 2048)
    intermediate = torch.einsum('bdk,hk->bdh', states, head_o_proj)  # (batch, 256, 2048)
    # @ color_unembed.T: (batch, 256, 2048) @ (2048, n_colors) -> (batch, 256, n_colors)
    A = torch.einsum('bdh,ch->bdc', intermediate, color_unembed)  # (batch, 256, n_colors)
    return A

def solve_w_left_closed_form(A, labels, n_colors):
    """
    Solve for w_left in closed form using least squares.
    
    A: (batch, 256, n_colors) - transformed states
    labels: (batch,) - class indices
    
    We want: logits = einsum('d, bdc -> bc', w_left, A) ≈ one_hot(labels)
    """
    batch_size = A.shape[0]
    
    # Create one-hot targets
    Y = torch.zeros(batch_size, n_colors, device=A.device)
    Y.scatter_(1, labels.unsqueeze(1), 1.0)  # (batch, n_colors)
    
    # Reshape for least squares: A_flat @ w_left = Y_flat
    # A_flat: (batch * n_colors, 256)
    # Y_flat: (batch * n_colors,)
    A_flat = A.permute(0, 2, 1).reshape(-1, 256)  # (batch * n_colors, 256)
    Y_flat = Y.reshape(-1)  # (batch * n_colors,)
    
    # Solve least squares: w_left = (A^T A)^{-1} A^T Y
    w_left = torch.linalg.lstsq(A_flat, Y_flat).solution  # (256,)
    
    return w_left

def evaluate_w_left(A, labels, w_left):
    """Compute accuracy given w_left and transformed states A."""
    # logits = einsum('d, bdc -> bc', w_left, A)
    logits = torch.einsum('d,bdc->bc', w_left, A)
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    return acc

# %%
def solve_probe(states_np, labels, layer_idx, head_idx):
    """Solve for w_left in closed form for a given layer/head."""
    states = torch.tensor(states_np[:, layer_idx, head_idx], dtype=torch.float32).to(device)
    
    n_samples = len(labels)
    n_val = max(1, int(0.3 * n_samples))
    indices = torch.randperm(n_samples)
    
    train_states = states[indices[n_val:]]
    train_labels = labels[indices[n_val:]].to(device)
    val_states = states[indices[:n_val]]
    val_labels = labels[indices[:n_val]].to(device)
    
    # Get model weights for this head
    head_o_proj = get_head_o_proj(model, layer_idx, head_idx).to(device)
    
    # Precompute transformed states
    train_A = precompute_transformed_states(train_states, head_o_proj, color_unembed)
    val_A = precompute_transformed_states(val_states, head_o_proj, color_unembed)
    
    # Solve in closed form
    w_left = solve_w_left_closed_form(train_A, train_labels, n_colors)
    
    # Evaluate
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
# ## 7. Re-train Best Probe and Test on New Dataset

# %%
best_head = results_df.sort_values('val_acc', ascending=False).iloc[0]
best_layer = int(best_head['layer'])
best_head_idx = int(best_head['head'])
print(f"\nBest head: Layer {best_layer}, Head {best_head_idx}")
print(f"Validation accuracy: {best_head['val_acc']:.3f}")

# %%
RETRAIN_DATASET_SIZE = 2000  # Same size as original training

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
retrain_states = np.zeros((RETRAIN_DATASET_SIZE, 256, 512), dtype=np.float16)
retrain_colors = []

for idx in tqdm(range(RETRAIN_DATASET_SIZE), desc="Extracting states"):
    sample = retrain_dataset[idx]
    input_ids = sample.input_ids.to(device)
    final_states = extractor.extract_final_states(input_ids)
    retrain_states[idx] = final_states[best_layer][0, best_head_idx].cpu().to(torch.float16).numpy()
    retrain_colors.append(sample.fixed_entity_color)

retrain_labels = torch.tensor([COLOR_TO_IDX[c] for c in retrain_colors])
retrain_states_tensor = torch.tensor(retrain_states, dtype=torch.float32)

# %%
print(f"Solving probe on Layer {best_layer}, Head {best_head_idx} with {RETRAIN_DATASET_SIZE} samples (closed form)...")

n_samples = RETRAIN_DATASET_SIZE
n_val = max(1, int(0.3 * n_samples))  # Same 30% split as original training
indices = torch.randperm(n_samples)

train_states = retrain_states_tensor[indices[n_val:]].to(device)
train_labels = retrain_labels[indices[n_val:]].to(device)
val_states = retrain_states_tensor[indices[:n_val]].to(device)
val_labels = retrain_labels[indices[:n_val]].to(device)

best_head_o_proj = get_head_o_proj(model, best_layer, best_head_idx).to(device)

# Precompute transformed states
train_A = precompute_transformed_states(train_states, best_head_o_proj, color_unembed)
val_A = precompute_transformed_states(val_states, best_head_o_proj, color_unembed)

# Solve in closed form
best_w_left = solve_w_left_closed_form(train_A, train_labels, n_colors)

# Evaluate
train_acc = evaluate_w_left(train_A, train_labels, best_w_left)
val_acc = evaluate_w_left(val_A, val_labels, best_w_left)

print(f"Train accuracy: {train_acc:.3f}")
print(f"Val accuracy: {val_acc:.3f}")
print("Best probe solved!")

# %%
TARGET_EVAL_SAMPLES = 3
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

eval_accuracies_by_position = {}

# Create a quiet extractor for evaluation
eval_extractor = GLAStateExtractor(model, verbose=False)

for sample_idx, sample in enumerate(tqdm(eval_samples, desc="Samples")):
    input_ids = sample.input_ids.to(device)
    seq_len = input_ids.shape[1]
    
    # Extract states with tqdm progress
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
            state_tensor = torch.tensor(state_at_pos[0, best_head_idx], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Compute prediction using closed-form w_left and model weights
            A = precompute_transformed_states(state_tensor, best_head_o_proj, color_unembed)
            logits = torch.einsum('d,bdc->bc', best_w_left, A)
            pred = logits.argmax(dim=1).item()
            
            is_correct = (pred == true_color_idx)
            relative_pos = pos - info_idx
            
            if relative_pos not in eval_accuracies_by_position:
                eval_accuracies_by_position[relative_pos] = []
            eval_accuracies_by_position[relative_pos].append(is_correct)

# %%
import matplotlib.pyplot as plt

positions = sorted(eval_accuracies_by_position.keys())
mean_accs = [np.mean(eval_accuracies_by_position[p]) for p in positions]

plt.figure(figsize=(12, 6))
plt.plot(positions, mean_accs, 'b-', linewidth=2, label='Mean Accuracy')
plt.axhline(y=1/n_colors, color='r', linestyle='--', label=f'Random Baseline ({1/n_colors:.3f})')
plt.xlabel('Tokens after Information Given')
plt.ylabel('Accuracy')
plt.title(f'Probe Accuracy vs Position After Information\nLayer {best_layer}, Head {best_head_idx}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../data/probe_accuracy_by_position.png', dpi=150)
plt.show()

# Plot rolling window accuracy for each sample individually
window_size = 100
plt.figure(figsize=(12, 8))

# Reorganize data by sample: for each sample, get its accuracy trajectory
n_samples = len(eval_samples)
sample_trajectories = []

for sample_idx in range(n_samples):
    trajectory = []
    for pos in positions:
        accs_at_pos = eval_accuracies_by_position.get(pos, [])
        if sample_idx < len(accs_at_pos):
            trajectory.append(float(accs_at_pos[sample_idx]))
        else:
            trajectory.append(np.nan)
    sample_trajectories.append(trajectory)

# Plot rolling window for each sample
for sample_idx, trajectory in enumerate(sample_trajectories):
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
plt.ylabel('Accuracy')
plt.title(f'Individual Rolling Window Accuracies (window={window_size})\nLayer {best_layer}, Head {best_head_idx}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../data/probe_accuracy_rolling_window.png', dpi=150)
plt.show()

print(f"\nMean accuracy across all positions: {np.mean(mean_accs):.3f}")
print(f"Accuracy at position +1: {mean_accs[0]:.3f}")
print(f"Final accuracy: {mean_accs[-1]:.3f}")

# %%

