# %% [markdown]
# # State Comparison Analysis
# 
# Compares states between context A (original color) and context B (blue)
# at the END OF CONTEXT (after all color sentences) to see which state 
# components differ the most.

# %%
import sys
import os
import torch
from typing import List, Dict
import random

sys.path.insert(0, os.path.abspath('..'))

from models.load_rwkv import load_rwkv_model
from datasets.favorite_color_dataset import FAMOUS_NAMES, DEFAULT_COLORS

# %%
print("Loading RWKV model...")
model, tokenizer = load_rwkv_model(model_name="rwkv7-g1b-1.5b-20251202-ctx8192.pth")

n_layer = model.model.n_layer
n_head = model.model.n_head
head_size = model.model.head_size

print(f"\nModel info: {n_layer} layers, {n_head} heads, head_size={head_size}")

# %%
def generate_sample_pair(
    n_entities: int = 5,
    fixed_entity: str = "Lady Gaga",
    seed: int = 42
) -> Dict:
    rng = random.Random(seed)
    
    non_blue_colors = [c for c in DEFAULT_COLORS if c.lower() != "blue"]
    color_A = rng.choice(non_blue_colors)
    color_B = "blue"
    
    available_names = [n for n in FAMOUS_NAMES if n != fixed_entity]
    other_entities = rng.sample(available_names, min(n_entities - 1, len(available_names)))
    other_colors = [rng.choice(DEFAULT_COLORS) for _ in other_entities]
    
    all_entities = other_entities + [fixed_entity]
    all_colors_A = other_colors + [color_A]
    all_colors_B = other_colors + [color_B]
    
    indices = list(range(len(all_entities)))
    rng.shuffle(indices)
    
    fixed_entity_idx_in_list = len(all_entities) - 1
    fixed_pos = indices.index(fixed_entity_idx_in_list)
    
    sentences_A = []
    sentences_B = []
    for idx in indices:
        sentences_A.append(f"{all_entities[idx]}'s favorite color is {all_colors_A[idx]}.")
        sentences_B.append(f"{all_entities[idx]}'s favorite color is {all_colors_B[idx]}.")
    
    prefix_sentences = sentences_A[:fixed_pos]
    suffix_sentences = sentences_A[fixed_pos + 1:]
    
    prefix = " ".join(prefix_sentences)
    if prefix:
        prefix += " "
    
    sentence_A = sentences_A[fixed_pos]
    sentence_B = sentences_B[fixed_pos]
    
    suffix = ""
    if suffix_sentences:
        suffix = " " + " ".join(suffix_sentences)
    
    context_A = prefix + sentence_A + suffix
    context_B = prefix + sentence_B + suffix
    
    return {
        'context_A': context_A,
        'context_B': context_B,
        'color_A': color_A,
        'color_B': color_B,
        'prefix': prefix,
        'sentence_A': sentence_A,
        'sentence_B': sentence_B,
        'suffix': suffix,
    }


def get_state_after_tokens(model, tokens: List[int], initial_state=None):
    if not tokens:
        return initial_state
    with torch.no_grad():
        _, state = model.model.forward(tokens, initial_state)
    return state


# %%
N_ENTITIES = 5
N_SAMPLES = 20
FIXED_ENTITY = "Lady Gaga"

print(f"Generating {N_SAMPLES} sample pairs with n_entities={N_ENTITIES}...")
print("Comparing states at END OF CONTEXT (after color sentence)")

sample_pairs = []
for i in range(N_SAMPLES):
    pair_data = generate_sample_pair(
        n_entities=N_ENTITIES,
        fixed_entity=FIXED_ENTITY,
        seed=1000 + i
    )
    
    # Use only the context (not the question/answer part)
    context_A = pair_data['context_A']
    context_B = pair_data['context_B']
    
    # Tokenize the contexts
    tokens_A = tokenizer.encode(context_A)
    tokens_B = tokenizer.encode(context_B)
    
    sample_pairs.append({
        'tokens_A': tokens_A,
        'tokens_B': tokens_B,
        'color_A': pair_data['color_A'],
        'color_B': pair_data['color_B'],
        'context_A': context_A,
        'context_B': context_B,
    })

print(f"Generated {len(sample_pairs)} pairs")
print(f"\nExample context A:\n{sample_pairs[0]['context_A']}")
print(f"\nExample context B:\n{sample_pairs[0]['context_B']}")


# %%
print("\n" + "=" * 80)
print("Computing state distances at END OF CONTEXT")
print("=" * 80)

# Accumulate distances across all samples
# State structure: [att_x_prev, att_kv, ffn_x_prev] per layer
state_distances = {
    'att_x_prev': {layer: [] for layer in range(n_layer)},
    'att_kv': {layer: [] for layer in range(n_layer)},
    'ffn_x_prev': {layer: [] for layer in range(n_layer)},
}

# Also track per-head distances for att_kv
head_distances = {layer: {head: [] for head in range(n_head)} for layer in range(n_layer)}

for pair in sample_pairs:
    # Get states after processing the full context
    tokens_A = pair['tokens_A']
    tokens_B = pair['tokens_B']
    
    state_A = get_state_after_tokens(model, tokens_A)
    state_A = [s.clone() for s in state_A]  # Clone to avoid buffer reuse
    state_B = get_state_after_tokens(model, tokens_B)
    
    for layer in range(n_layer):
        att_x_prev_idx = layer * 3 + 0
        att_kv_idx = layer * 3 + 1
        ffn_x_prev_idx = layer * 3 + 2
        
        # att_x_prev distance (vector)
        diff_att_x = (state_A[att_x_prev_idx] - state_B[att_x_prev_idx]).float()
        dist_att_x = torch.norm(diff_att_x).item()
        state_distances['att_x_prev'][layer].append(dist_att_x)
        
        # att_kv distance (per-head matrix)
        diff_kv = (state_A[att_kv_idx] - state_B[att_kv_idx]).float()
        dist_kv = torch.norm(diff_kv).item()
        state_distances['att_kv'][layer].append(dist_kv)
        
        # Per-head distances
        for head in range(n_head):
            head_diff = (state_A[att_kv_idx][head] - state_B[att_kv_idx][head]).float()
            head_dist = torch.norm(head_diff).item()
            head_distances[layer][head].append(head_dist)
        
        # ffn_x_prev distance (vector)
        diff_ffn_x = (state_A[ffn_x_prev_idx] - state_B[ffn_x_prev_idx]).float()
        dist_ffn_x = torch.norm(diff_ffn_x).item()
        state_distances['ffn_x_prev'][layer].append(dist_ffn_x)


# %%
print("\n" + "=" * 80)
print("Average state distances by layer at END OF CONTEXT (sorted by total)")
print("=" * 80)

# Compute average distances
layer_avg_distances = []
for layer in range(n_layer):
    avg_att_x = sum(state_distances['att_x_prev'][layer]) / len(state_distances['att_x_prev'][layer])
    avg_kv = sum(state_distances['att_kv'][layer]) / len(state_distances['att_kv'][layer])
    avg_ffn_x = sum(state_distances['ffn_x_prev'][layer]) / len(state_distances['ffn_x_prev'][layer])
    total = avg_att_x + avg_kv + avg_ffn_x
    layer_avg_distances.append({
        'layer': layer,
        'att_x_prev': avg_att_x,
        'att_kv': avg_kv,
        'ffn_x_prev': avg_ffn_x,
        'total': total
    })

# Sort by total distance
sorted_layers = sorted(layer_avg_distances, key=lambda x: x['total'], reverse=True)

print(f"\n{'Layer':<8} {'att_x_prev':>12} {'att_kv':>12} {'ffn_x_prev':>12} {'Total':>12}")
print("-" * 60)
for item in sorted_layers[:15]:
    print(f"L{item['layer']:<7} {item['att_x_prev']:>12.4f} {item['att_kv']:>12.4f} {item['ffn_x_prev']:>12.4f} {item['total']:>12.4f}")


# %%
print("\n" + "=" * 80)
print("Top 20 heads by average distance at END OF CONTEXT")
print("=" * 80)

head_avg_distances = []
for layer in range(n_layer):
    for head in range(n_head):
        avg_dist = sum(head_distances[layer][head]) / len(head_distances[layer][head])
        head_avg_distances.append({
            'layer': layer,
            'head': head,
            'avg_dist': avg_dist
        })

sorted_heads = sorted(head_avg_distances, key=lambda x: x['avg_dist'], reverse=True)

print(f"\n{'Layer':<8} {'Head':<8} {'Avg Distance':>15}")
print("-" * 35)
for item in sorted_heads[:20]:
    print(f"L{item['layer']:<7} H{item['head']:<7} {item['avg_dist']:>15.4f}")


# %%
print("\n" + "=" * 80)
print("State component comparison (which type varies most?)")
print("=" * 80)

total_att_x = sum(sum(state_distances['att_x_prev'][l]) for l in range(n_layer)) / (n_layer * N_SAMPLES)
total_kv = sum(sum(state_distances['att_kv'][l]) for l in range(n_layer)) / (n_layer * N_SAMPLES)
total_ffn_x = sum(sum(state_distances['ffn_x_prev'][l]) for l in range(n_layer)) / (n_layer * N_SAMPLES)

print(f"\nAverage distance across all layers:")
print(f"  att_x_prev:  {total_att_x:.4f}")
print(f"  att_kv:      {total_kv:.4f}")
print(f"  ffn_x_prev:  {total_ffn_x:.4f}")

# %%

