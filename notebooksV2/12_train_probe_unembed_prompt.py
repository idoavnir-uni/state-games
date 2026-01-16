# %% [markdown]
# # RWKV State Probing with Unembedding Matrix - With QA Prompt
# 
# Same as 06_train_probe_unembed.py but uses the full QA prompt format:
# "Given the following context, let's answer the question below.
#  Context: {context}
#  Question: What is {entity}'s favorite color?
#  Answer: {entity}'s favorite color is"

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
from datasets.favorite_color_dataset import FavoriteColorDataset

print("Imports complete!")

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

print("Loading RWKV model...")
model, tokenizer = load_rwkv_model(model_name="rwkv7-g1b-1.5b-20251202-ctx8192.pth")

config = get_model_config(model)
print(f"Model loaded: {config.get('num_layers')} layers, {config.get('num_heads')} heads")

# %%
extractor = RWKVStateExtractor(model, verbose=False)
head_size = extractor.head_size
n_head = extractor.n_head
n_embd = model.model.n_embd
print(f"Head size: {head_size}, Num heads: {n_head}, Hidden size: {n_embd}")

# %%
def make_prompt(context: str, entity_name: str = "Lady Gaga") -> str:
    """Create the full QA prompt format."""
    return (
        f"Given the following context, let's answer the question below.\n\n"
        f"Context:\n{context}\n\n"
    )

# %%
DATASET_SIZE = 100
N_ENTITIES = 50
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
print(f"Colors used: {dataset.colors}")

# %%
print("\nVerifying colors are single tokens...")
color_token_ids = {}
all_single_tokens = True

for color in dataset.colors:
    tokens = tokenizer.encode(color)
    is_single = len(tokens) == 1
    
    if is_single:
        color_token_ids[color] = tokens[0]
        print(f"  {color}: token_id={tokens[0]} ✓")
    else:
        all_single_tokens = False
        print(f"  {color}: {len(tokens)} tokens ✗")

if not all_single_tokens:
    raise ValueError("Not all colors are single tokens")

COLOR_TOKEN_IDS = torch.tensor([color_token_ids[c] for c in dataset.colors])

# %%
num_layers = config.get('num_layers')
num_heads = config.get('num_heads')

print(f"\nExtracting states for {DATASET_SIZE} samples WITH QA PROMPT...")
print("Prompt format: 'Given the following context...Answer: {entity}'s favorite color is'")

metadata_rows = []
all_states = np.zeros((DATASET_SIZE, num_layers, num_heads, head_size, head_size), dtype=np.float16)

for idx in tqdm(range(len(dataset)), desc="Processing samples"):
    sample = dataset[idx]
    
    # Get the context from the sample and wrap it in the QA prompt
    context = sample.text  # This is the raw context
    full_prompt = make_prompt(context, FIXED_ENTITY)
    
    # Tokenize the full prompt
    input_ids = tokenizer.encode(full_prompt)
    input_ids_tensor = torch.tensor([input_ids])
    
    # Extract states at the final position (after "...favorite color is")
    final_states = extractor.extract_final_states(input_ids_tensor)
    
    metadata_rows.append({
        'context': context,
        'full_prompt': full_prompt,
        'target_color': sample.fixed_entity_color,
        'num_tokens': len(input_ids),
    })
    
    for layer_idx in range(num_layers):
        layer_state = final_states[layer_idx]
        all_states[idx, layer_idx] = layer_state.cpu().to(torch.float16).numpy()

df = pd.DataFrame(metadata_rows)
print(f"\nMetadata dataframe shape: {df.shape}")
print(f"States array shape: {all_states.shape}")

# %%
print("\n=== Sample Entry ===")
print(f"Full prompt (truncated): {df.iloc[0]['full_prompt'][:200]}...")
print(f"Target color: {df.iloc[0]['target_color']}")
print(f"Num tokens: {df.iloc[0]['num_tokens']}")

print("\n=== Color Distribution ===")
print(df['target_color'].value_counts())

# %%
import torch.nn as nn

COLOR_TO_IDX = {color: idx for idx, color in enumerate(dataset.colors)}
labels = torch.tensor([COLOR_TO_IDX[c] for c in df['target_color']])
n_colors = len(dataset.colors)

print(f"Color mapping: {COLOR_TO_IDX}")

# Get unembedding matrix from model
lm_head = model.model.z['head.weight'].detach()
color_unembed = lm_head[:, COLOR_TOKEN_IDS].float().to(device)
print(f"Color unembedding shape: {color_unembed.shape}")

# %%
def get_head_o_proj(model, layer_idx, head_idx, head_size=64, n_embd=2048):
    z = model.model.z
    o_proj = z[f'blocks.{layer_idx}.att.output.weight'].detach().float()
    start_idx = head_idx * head_size
    end_idx = (head_idx + 1) * head_size
    head_o_proj = o_proj[start_idx:end_idx, :]
    return head_o_proj

def get_W_left_model(head_o_proj, color_unembed):
    M = head_o_proj @ color_unembed
    W_left_model = M.T
    return W_left_model

def precompute_transformed_states(states, W_left_model):
    A = torch.einsum('cd,bdk->bck', W_left_model, states)
    return A

def solve_w_right_closed_form(A, labels, n_colors):
    batch_size = A.shape[0]
    head_size = A.shape[2]
    
    Y = torch.zeros(batch_size, n_colors, device=A.device)
    Y.scatter_(1, labels.unsqueeze(1), 1.0)
    
    A_flat = A.reshape(-1, head_size)
    Y_flat = Y.reshape(-1)
    
    w_right = torch.linalg.lstsq(A_flat, Y_flat).solution
    return w_right

def evaluate_w_right(A, labels, w_right):
    logits = torch.einsum('bck,k->bc', A, w_right)
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    return acc

# %%
def solve_probe(states_np, labels, layer_idx, head_idx):
    states = torch.tensor(states_np[:, layer_idx, head_idx], dtype=torch.float32).to(device)
    
    n_samples = len(labels)
    n_val = max(1, int(0.3 * n_samples))
    indices = torch.randperm(n_samples)
    
    train_states = states[indices[n_val:]]
    train_labels = labels[indices[n_val:]].to(device)
    val_states = states[indices[:n_val]]
    val_labels = labels[indices[:n_val]].to(device)
    
    head_o_proj = get_head_o_proj(model, layer_idx, head_idx, head_size, n_embd).to(device)
    W_left_model = get_W_left_model(head_o_proj, color_unembed)
    
    train_A = precompute_transformed_states(train_states, W_left_model)
    val_A = precompute_transformed_states(val_states, W_left_model)
    
    w_right = solve_w_right_closed_form(train_A, train_labels, n_colors)
    
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
print("\n=== Best Performing Heads (WITH QA PROMPT) ===")
print(results_df.sort_values('val_acc', ascending=False).head(20))

# %%
best_head = results_df.sort_values('val_acc', ascending=False).iloc[0]
best_layer = int(best_head['layer'])
best_head_idx = int(best_head['head'])
print(f"\nBest head: Layer {best_layer}, Head {best_head_idx}")
print(f"Validation accuracy: {best_head['val_acc']:.3f}")

# %%
print("\n=== Comparison: Top 10 Heads ===")
print("These are the heads that best predict the color from the state")
print("after processing the full QA prompt.")
print()
top10 = results_df.sort_values('val_acc', ascending=False).head(10)
for _, row in top10.iterrows():
    print(f"  L{int(row['layer']):2d} H{int(row['head']):2d}: val_acc={row['val_acc']:.3f}")

# %%


