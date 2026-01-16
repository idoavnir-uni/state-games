# %% [markdown]
# # RWKV State Probing with Unembedding Matrix (GPU 2)
# 
# This notebook:
# 1. Generates favorite city sentences with a fixed entity (Lady Gaga)
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

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

sys.path.insert(0, os.path.abspath('..'))

from models.load_rwkv import load_rwkv_model, get_model_config
from models.state_extractor_rwkv import RWKVStateExtractor
from datasets.lives_in_city_dataset import LivesInCityDataset

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
# ## 3. Create Favorite City Dataset

# %%
DATASET_SIZE = 10000
N_ENTITIES = 30
N_CITIES = 10
FIXED_ENTITY = "bat"

print(f"Creating dataset with {DATASET_SIZE} samples...")
dataset = LivesInCityDataset(
    tokenizer=tokenizer,
    size=DATASET_SIZE,
    n_entities=N_ENTITIES,
    n_cities=N_CITIES,
    fixed_entity_name="bat",
    seed=42,
)
print(f"Dataset created with {len(dataset)} samples")
print(f"Fixed entity: {dataset.fixed_entity_name}")
print(f"Cities used: {dataset.cities}")

# %% [markdown]
# ## 3.1 Verify Citys are Single Tokens

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
# - W_left_model: (n_cities, head_size) = city_unembed @ head_o_proj - from model weights
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
def train_probe(states_np, labels, layer_idx, head_idx, patience=20, batch_size=1000):
    """Train w_right with gradient descent using batches."""
    states = torch.tensor(states_np[:, layer_idx, head_idx], dtype=torch.float32)
    
    n_samples = len(labels)
    n_val = max(1, int(0.3 * n_samples))
    indices = torch.randperm(n_samples)
    
    train_states = states[indices[n_val:]]
    train_labels = labels[indices[n_val:]]
    val_states = states[indices[:n_val]]
    val_labels = labels[indices[:n_val]]
    
    # Get model weights for this head and compute W_left_model (frozen)
    head_o_proj = get_head_o_proj(model, layer_idx, head_idx, head_size, n_embd).to(device)
    W_left_model = get_W_left_model(head_o_proj, city_unembed)
    
    # Precompute transformed states: A = W_left_model @ state (can be done once)
    train_A = precompute_transformed_states(train_states.to(device), W_left_model)
    val_A = precompute_transformed_states(val_states.to(device), W_left_model)
    
    # Initialize learnable w_right
    w_right = nn.Parameter(torch.randn(head_size, device=device) * 0.01)
    optimizer = optim.Adam([w_right], lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_w_right = None
    patience_counter = 0
    
    n_train = len(train_states)
    
    for epoch in range(20000):
        # Training with batches
        batch_indices = torch.randperm(n_train)
        
        for i in range(0, n_train, batch_size):
            batch_idx = batch_indices[i:i+batch_size]
            batch_A = train_A[batch_idx]
            batch_labels = train_labels[batch_idx].to(device)
            
            optimizer.zero_grad()
            logits = torch.einsum('bck,k->bc', batch_A, w_right)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        with torch.no_grad():
            val_logits = torch.einsum('bck,k->bc', val_A, w_right)
            val_loss = criterion(val_logits, val_labels.to(device)).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels.to(device)).float().mean().item()
            
            train_logits = torch.einsum('bck,k->bc', train_A, w_right)
            train_preds = train_logits.argmax(dim=1)
            train_acc = (train_preds == train_labels.to(device)).float().mean().item()
        
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
top_50 = results_df.sort_values('val_acc', ascending=False).head(50)
for idx, row in top_50.iterrows():
    print(f"L{row['layer']:<5} H{row['head']:<5} {row['val_acc']:<10.3f} {row['train_acc']:<10.3f} {row['epochs']:<8}")

os.makedirs('results', exist_ok=True)
top_50.to_csv('results/result_unembeding.csv', index=False)
print(f"\nResults saved to results/result_unembeding.csv")

# %%
