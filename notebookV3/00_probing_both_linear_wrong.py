# %% [markdown]
# # RWKV State Probing Experiment - Wrong Direction (Vector @ State @ Matrix)
# 
# This notebook:
# 1. Generates favorite color sentences with a fixed entity (Lady Gaga)
# 2. Extracts RWKV model states at the final token position
# 3. Trains linear probes on each layer/head to predict the target color
# 
# **Probe model (WRONG):** logits = (w_left @ state) @ W_right
# - state: (head_size, head_size) per head
# - w_left: (head_size,) learned vector → w_left @ state = (head_size,)
# - W_right: (head_size, n_colors) learned matrix → n_colors logits
#
# Compare to correct: logits = (W_left @ state) @ w_right

# %%
import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
# ## 6. Train Linear Probe on States (WRONG DIRECTION)
# 
# Model: logits = (w_left @ state) @ W_right
# - state: (head_size, head_size)
# - w_left: (head_size,) vector → w_left @ state = (head_size,)
# - W_right: (head_size, n_colors) matrix → (head_size,) @ (head_size, n_colors) = (n_colors,)

# %%
import torch.nn as nn
import torch.optim as optim

COLOR_TO_IDX = {color: idx for idx, color in enumerate(dataset.colors)}
labels = torch.tensor([COLOR_TO_IDX[c] for c in df['target_color']])
n_colors = len(dataset.colors)

print(f"Color mapping: {COLOR_TO_IDX}")

# %%
class StateProbeWrong(nn.Module):
    def __init__(self, head_size, n_classes):
        super().__init__()
        # WRONG: vector on left, matrix on right
        self.w_left = nn.Parameter(torch.randn(head_size) * 0.01)
        self.W_right = nn.Parameter(torch.randn(head_size, n_classes) * 0.01)
    
    def forward(self, state):
        # state: (batch, head_size, head_size)
        # w_left @ state: (d,) @ (b, d, k) -> (b, k)
        hidden = torch.einsum('d,bdk->bk', self.w_left, state)  # (batch, head_size)
        # hidden @ W_right: (b, k) @ (k, c) -> (b, c)
        logits = torch.einsum('bk,kc->bc', hidden, self.W_right)  # (batch, n_classes)
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
    
    probe = StateProbeWrong(head_size, n_colors).to(device)
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

print("Training probes (WRONG direction) for each layer/head...")
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

os.makedirs('results', exist_ok=True)
top_50.to_csv('results/result_both_linear_wrong.csv', index=False)
print(f"\nResults saved to results/result_both_linear_wrong.csv")

# %%
