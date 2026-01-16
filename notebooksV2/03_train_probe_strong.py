# %% [markdown]
# # RWKV State Probing Experiment - Wrong Direction (GPU 2)
# 
# This notebook:
# 1. Generates favorite color sentences with a fixed entity (Lady Gaga)
# 2. Extracts RWKV model states at the final token position
# 3. Trains linear probes on each layer/head to predict the target color
# 
# **Probe model:** logits = (W_left @ state) @ w_right
# - state: (head_size, head_size) per head
# - W_left: (n_colors, head_size) learned matrix → projects to (n_colors, head_size)
# - w_right: (head_size,) learned vector → n_colors logits

# %%
import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

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
# ## 2. Initialize State Extractor

# %%
extractor = RWKVStateExtractor(model, verbose=False)
head_size = extractor.head_size
print(f"Head size: {head_size}")

# %% [markdown]
# ## 3. Create Favorite Color Dataset

# %%
DATASET_SIZE = 10000
N_ENTITIES = 30
N_COLORS = 10

print(f"Creating dataset with {DATASET_SIZE} samples...")
dataset = FavoriteColorDataset(
    tokenizer=tokenizer,
    size=DATASET_SIZE,
    n_entities=N_ENTITIES,
    n_colors=N_COLORS,
    fixed_entity_name="Lady Gaga",
    seed=42,
)
print(f"Dataset created with {len(dataset)} samples")
print(f"Colors used: {dataset.colors}")

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
# ## 5. Display Sample Information

# %%
print("\n=== Sample Entry ===")
print(f"Sentence: {df.iloc[0]['sentence']}")
print(f"Target color: {df.iloc[0]['target_color']}")
print(f"States shape: {all_states.shape} = (samples, layers, heads, {head_size}, {head_size})")

print("\n=== Dataset Statistics ===")
print(f"Total samples: {len(df)}")
print(f"Color distribution:\n{df['target_color'].value_counts()}")

# %% [markdown]
# ## 6. Train Linear Probe on States
# 
# Model: logits = (W_left @ state) @ w_right
# - state: (head_size, head_size)
# - W_left: (n_colors, head_size) matrix → W_left @ state = (n_colors, head_size)
# - w_right: (head_size,) vector → (n_colors, head_size) @ (head_size,) = (n_colors,)

# %%
import torch.nn as nn
import torch.optim as optim

COLOR_TO_IDX = {color: idx for idx, color in enumerate(dataset.colors)}
labels = torch.tensor([COLOR_TO_IDX[c] for c in df['target_color']])
n_colors = len(dataset.colors)

print(f"Color mapping: {COLOR_TO_IDX}")

# %%
class StateProbe(nn.Module):
    def __init__(self, head_size, n_classes):
        super().__init__()
        hidden_dim = head_size
        self.w_right = nn.Parameter(torch.randn(head_size) * 0.01)
        self.W_left_1 = nn.Parameter(torch.randn(hidden_dim, head_size) * 0.01)
        self.W_left_2 = nn.Parameter(torch.randn(n_classes, hidden_dim) * 0.01)
    
    def forward(self, state):
        # state: (batch, head_size, head_size)
        compressed = torch.einsum('bdk,k->bd', state, self.w_right)  # (batch, head_size)
        hidden1 = torch.einsum('hd,bd->bh', self.W_left_1, compressed)  # (batch, hidden_dim)
        hidden1 = torch.relu(hidden1)
        logits = torch.einsum('ch,bh->bc', self.W_left_2, hidden1)  # (batch, n_classes)
        return logits

def train_probe(states_np, labels, layer_idx, head_idx, patience=20, batch_size=1000):
    states = torch.tensor(states_np[:, layer_idx, head_idx], dtype=torch.float32)
    
    n_samples = len(labels)
    n_val = max(1, int(0.3 * n_samples))
    indices = torch.randperm(n_samples)
    
    train_states = states[indices[n_val:]]
    train_labels = labels[indices[n_val:]]
    val_states = states[indices[:n_val]]
    val_labels = labels[indices[:n_val]]
    
    probe = StateProbe(head_size, n_colors).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    n_train = len(train_states)
    
    for epoch in range(20000):
        probe.train()
        
        batch_indices = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            batch_idx = batch_indices[i:i+batch_size]
            batch_states = train_states[batch_idx].to(device)
            batch_labels = train_labels[batch_idx].to(device)
            
            optimizer.zero_grad()
            logits = probe(batch_states)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_states.to(device))
            val_loss = criterion(val_logits, val_labels.to(device)).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels.to(device)).float().mean().item()
            
            train_logits = probe(train_states.to(device))
            train_preds = train_logits.argmax(dim=1)
            train_acc = (train_preds == train_labels.to(device)).float().mean().item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return best_val_acc, train_acc, epoch + 1

print("Training probes for each layer/head...")
print(f"{'Layer':<6} {'Head':<6} {'Val Acc':<10} {'Train Acc':<10} {'Epochs':<8}")
print("-" * 40)

results = []
for layer_idx in range(num_layers):
    for head_idx in range(num_heads):
        val_acc, train_acc, epochs = train_probe(all_states, labels, layer_idx, head_idx)
        results.append({
            'layer': layer_idx,
            'head': head_idx,
            'val_acc': val_acc,
            'train_acc': train_acc,
            'epochs': epochs
        })
        print(f"L{layer_idx:<5} H{head_idx:<5} {val_acc:<10.3f} {train_acc:<10.3f} {epochs:<8}")

results_df = pd.DataFrame(results)
print("\n=== Best Performing Heads ===")
top_50 = results_df.sort_values('val_acc', ascending=False).head(50)
for idx, row in top_50.iterrows():
    print(f"L{row['layer']:<5} H{row['head']:<5} {row['val_acc']:<10.3f} {row['train_acc']:<10.3f} {row['epochs']:<8}")

# %%

# # %%
# # %% [markdown]
# # ## 7. Re-train Best Probe and Test on New Dataset

# # %%
# best_head = results_df.sort_values('val_acc', ascending=False).iloc[0]
# best_layer = int(best_head['layer'])
# best_head_idx = int(best_head['head'])
# print(f"\nBest head: Layer {best_layer}, Head {best_head_idx}")
# print(f"Validation accuracy: {best_head['val_acc']:.3f}")

# # %%
# RETRAIN_DATASET_SIZE = 10000

# print(f"Generating {RETRAIN_DATASET_SIZE} samples for re-training...")
# retrain_dataset = FavoriteColorDataset(
#     tokenizer=tokenizer,
#     size=RETRAIN_DATASET_SIZE,
#     n_entities=N_ENTITIES,
#     n_colors=N_COLORS,
#     fixed_entity_name="Lady Gaga",
#     seed=100,
# )

# print(f"Extracting states for best head (Layer {best_layer}, Head {best_head_idx})...")
# retrain_states = np.zeros((RETRAIN_DATASET_SIZE, head_size, head_size), dtype=np.float16)
# retrain_colors = []

# for idx in tqdm(range(RETRAIN_DATASET_SIZE), desc="Extracting states"):
#     sample = retrain_dataset[idx]
#     input_ids = sample.input_ids
#     final_states = extractor.extract_final_states(input_ids)
#     retrain_states[idx] = final_states[best_layer][best_head_idx].cpu().to(torch.float16).numpy()
#     retrain_colors.append(sample.fixed_entity_color)

# retrain_labels = torch.tensor([COLOR_TO_IDX[c] for c in retrain_colors])
# retrain_states_tensor = torch.tensor(retrain_states, dtype=torch.float32)

# # %%
# print(f"Re-training probe on Layer {best_layer}, Head {best_head_idx} with {RETRAIN_DATASET_SIZE} samples...")

# n_samples = RETRAIN_DATASET_SIZE
# n_val = max(1, int(0.1 * n_samples))
# indices = torch.randperm(n_samples)

# train_states = retrain_states_tensor[indices[n_val:]]
# train_labels = retrain_labels[indices[n_val:]]
# val_states = retrain_states_tensor[indices[:n_val]]
# val_labels = retrain_labels[indices[:n_val]]

# best_probe = StateProbe(head_size, n_colors).to(device)
# optimizer = optim.Adam(best_probe.parameters(), lr=1e-3)
# criterion = nn.CrossEntropyLoss()

# pbar = tqdm(range(10000), desc="Re-training")
# for epoch in pbar:
#     best_probe.train()
#     optimizer.zero_grad()
#     logits = best_probe(train_states.to(device))
#     loss = criterion(logits, train_labels.to(device))
#     loss.backward()
#     optimizer.step()
    
#     if epoch % 10 == 0:
#         best_probe.eval()
#         with torch.no_grad():
#             val_logits = best_probe(val_states.to(device))
#             val_loss = criterion(val_logits, val_labels.to(device)).item()
#             train_preds = logits.argmax(dim=1)
#             train_acc = (train_preds == train_labels.to(device)).float().mean().item()
#             val_preds = val_logits.argmax(dim=1)
#             val_acc = (val_preds == val_labels.to(device)).float().mean().item()
        
#         pbar.set_postfix({
#             'train_loss': f'{loss.item():.4f}',
#             'val_loss': f'{val_loss:.4f}',
#             'train_acc': f'{train_acc:.3f}',
#             'val_acc': f'{val_acc:.3f}'
#         })

# print("Best probe re-trained!")

# # %%
# TARGET_EVAL_SAMPLES = 10
# EVAL_N_ENTITIES = 500

# print(f"\nGenerating {TARGET_EVAL_SAMPLES} evaluation samples with Lady Gaga at sentences 10-20...")
# eval_samples = []
# seed_offset = 43

# pbar = tqdm(total=TARGET_EVAL_SAMPLES, desc="Generating valid samples")
# while len(eval_samples) < TARGET_EVAL_SAMPLES:
#     temp_dataset = FavoriteColorDataset(
#         tokenizer=tokenizer,
#         size=100,
#         n_entities=EVAL_N_ENTITIES,
#         n_colors=N_COLORS,
#         fixed_entity_name="Lady Gaga",
#         seed=seed_offset,
#     )
    
#     for sample in temp_dataset:
#         if 10 <= sample.fixed_entity_sentence_number <= 20:
#             eval_samples.append(sample)
#             pbar.update(1)
#             if len(eval_samples) >= TARGET_EVAL_SAMPLES:
#                 break
    
#     seed_offset += 1

# pbar.close()
# print(f"Generated {len(eval_samples)} valid samples")

# # %%
# print(f"Extracting incremental states for evaluation...")
# print(f"Note: This extracts states token-by-token, so will be slow")

# eval_accuracies_by_position = {}

# eval_extractor = RWKVStateExtractor(model, verbose=False)
# best_probe.eval()

# for sample_idx, sample in enumerate(tqdm(eval_samples, desc="Samples")):
#     input_ids = sample.input_ids
#     seq_len = input_ids.shape[1]
    
#     incremental_states = eval_extractor.extract_incremental_states_single_pass(
#         input_ids,
#         layers=[best_layer],
#         use_tqdm=True
#     )
    
#     info_idx = sample.fixed_entity_sentence_end_token_idx
#     true_color_idx = COLOR_TO_IDX[sample.fixed_entity_color]
    
#     with torch.no_grad():
#         for pos in range(info_idx + 1, seq_len):
#             if pos not in incremental_states:
#                 continue
            
#             state_at_pos = incremental_states[pos][best_layer]
#             state_tensor = torch.tensor(state_at_pos[best_head_idx], dtype=torch.float32).unsqueeze(0).to(device)
            
#             logits = best_probe(state_tensor)
#             pred = logits.argmax(dim=1).item()
            
#             is_correct = (pred == true_color_idx)
#             relative_pos = pos - info_idx
            
#             if relative_pos not in eval_accuracies_by_position:
#                 eval_accuracies_by_position[relative_pos] = []
#             eval_accuracies_by_position[relative_pos].append(is_correct)

# # %%
# import matplotlib.pyplot as plt

# positions = sorted(eval_accuracies_by_position.keys())
# mean_accs = [np.mean(eval_accuracies_by_position[p]) for p in positions]

# plt.figure(figsize=(12, 6))
# plt.plot(positions, mean_accs, 'b-', linewidth=2, label='Mean Accuracy')
# plt.axhline(y=1/n_colors, color='r', linestyle='--', label=f'Random Baseline ({1/n_colors:.3f})')
# plt.xlabel('Tokens after Information Given')
# plt.ylabel('Accuracy')
# plt.title(f'Probe Accuracy vs Position After Information\nLayer {best_layer}, Head {best_head_idx}')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('../data/rwkv_probe_accuracy_by_position.png', dpi=150)
# plt.show()

# # Plot rolling window accuracy for each sample individually
# window_size = 100
# plt.figure(figsize=(12, 8))

# n_samples = len(eval_samples)
# sample_trajectories = []

# for sample_idx in range(n_samples):
#     trajectory = []
#     for pos in positions:
#         accs_at_pos = eval_accuracies_by_position.get(pos, [])
#         if sample_idx < len(accs_at_pos):
#             trajectory.append(float(accs_at_pos[sample_idx]))
#         else:
#             trajectory.append(np.nan)
#     sample_trajectories.append(trajectory)

# for sample_idx, trajectory in enumerate(sample_trajectories):
#     rolling_traj = []
#     rolling_pos = []
#     for i in range(len(trajectory) - window_size + 1):
#         window = trajectory[i:i + window_size]
#         valid_vals = [v for v in window if not np.isnan(v)]
#         if valid_vals:
#             rolling_traj.append(np.mean(valid_vals))
#             rolling_pos.append(positions[i + window_size // 2])
    
#     if rolling_traj:
#         plt.plot(rolling_pos, rolling_traj, alpha=0.6, linewidth=1.5)

# plt.axhline(y=1/n_colors, color='r', linestyle='--', linewidth=2, label=f'Random Baseline ({1/n_colors:.3f})')
# plt.xlabel('Tokens after Information Given')
# plt.ylabel('Accuracy')
# plt.title(f'Individual Rolling Window Accuracies (window={window_size})\nLayer {best_layer}, Head {best_head_idx}')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('../data/rwkv_probe_accuracy_rolling_window.png', dpi=150)
# plt.show()

# print(f"\nMean accuracy across all positions: {np.mean(mean_accs):.3f}")
# print(f"Accuracy at position +1: {mean_accs[0]:.3f}")
# print(f"Final accuracy: {mean_accs[-1]:.3f}")

# # %%



# %%
