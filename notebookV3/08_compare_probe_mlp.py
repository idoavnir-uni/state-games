
# %%
# Cross-Dataset Head Testing - MLP Probes
# 
# Goal: Test if MLP probe heads important for one task are important for that task only

import sys
import os
import torch
import pandas as pd
from typing import List, Tuple, Dict
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

sys.path.insert(0, os.path.abspath('..'))

from models.load_rwkv import load_rwkv_model
from datasets.favorite_color_dataset import FAMOUS_NAMES, DEFAULT_COLORS
from datasets.animal_names import ANIMAL_NAMES
from datasets.lives_in_city_dataset import DEFAULT_CITIES

# %%
NUM_HEADS_TO_TEST = 5
NUM_SAMPLES_TO_TEST = 50

# %%
print("Loading RWKV model...")
model, tokenizer = load_rwkv_model(model_name="rwkv7-g1b-1.5b-20251202-ctx8192.pth")

n_layer = model.model.n_layer
n_head = model.model.n_head
head_size = model.model.head_size

print(f"\nModel info: {n_layer} layers, {n_head} heads, head_size={head_size}")

# %%
print("\n" + "=" * 80)
print("LOADING MLP PROBE RESULTS")
print("=" * 80)

def df_to_heads(df: pd.DataFrame, n: int) -> List[Tuple[int, int]]:
    top_n = df.head(n)
    return [(int(row['layer']), int(row['head'])) for _, row in top_n.iterrows()]

# Load MLP results from FavoriteColor
results_color_mlp = pd.read_csv('results/result_mlp.csv')
heads_color_mlp = df_to_heads(results_color_mlp, NUM_HEADS_TO_TEST)
print(f"\nFavoriteColor MLP heads: {heads_color_mlp}")
print(f"  Best val_acc: {results_color_mlp.iloc[0]['val_acc']:.3f}")

# Load MLP results from LivesInCity
results_city_mlp = pd.read_csv('../notebookV3_other_dataset/results/result_mlp.csv')
heads_city_mlp = df_to_heads(results_city_mlp, NUM_HEADS_TO_TEST)
print(f"\nLivesInCity MLP heads: {heads_city_mlp}")
print(f"  Best val_acc: {results_city_mlp.iloc[0]['val_acc']:.3f}")

# %%
# Helper functions
def get_state_after_tokens(model, tokens: List[int], initial_state=None):
    if not tokens:
        return initial_state
    with torch.no_grad():
        _, state = model.model.forward(tokens, initial_state)
    return state

def replace_specific_heads(state_A, state_B, heads_to_replace, n_layer):
    new_state = [s.clone() for s in state_A]
    
    layers_with_heads = {}
    for layer, head in heads_to_replace:
        if layer not in layers_with_heads:
            layers_with_heads[layer] = []
        layers_with_heads[layer].append(head)
    
    for layer, heads in layers_with_heads.items():
        if layer < n_layer:
            att_x_prev_idx = layer * 3 + 0
            ffn_x_prev_idx = layer * 3 + 2
            new_state[att_x_prev_idx] = state_B[att_x_prev_idx].clone()
            new_state[ffn_x_prev_idx] = state_B[ffn_x_prev_idx].clone()
            
            att_kv_idx = layer * 3 + 1
            for head in heads:
                new_state[att_kv_idx][head] = state_B[att_kv_idx][head].clone()
    
    return new_state

def find_hook_token_idx(full_tokens, hook_text, tokenizer):
    hook_tokens = tokenizer.encode(hook_text)
    hook_len = len(hook_tokens)
    
    for i in range(min(hook_len + 5, len(full_tokens)), hook_len - 5, -1):
        if full_tokens[:i] == hook_tokens[:i] or i <= hook_len:
            decoded_prefix = tokenizer.decode(full_tokens[:i])
            if len(decoded_prefix) >= len(hook_text) - 5:
                return i
    return len(hook_tokens)

def decode_tokens_safe(tokenizer, token_ids):
    filtered = [t for t in token_ids if t != 0]
    if not filtered:
        return ""
    return tokenizer.decode(filtered)

# %%
# Generate FavoriteColor samples
N_ENTITIES = 30
FIXED_ENTITY_COLOR = "Lady Gaga"

def make_prompt_color(context, entity_name="Lady Gaga"):
    return (
        f"Given the following context, let's answer the question below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: What is {entity_name}'s favorite color?\n\n"
        f"Answer: {entity_name}'s favorite color is"
    )

def generate_sample_pair_color(n_entities=30, fixed_entity="Lady Gaga", seed=42):
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
    
    fixed_pos = indices.index(len(all_entities) - 1)
    
    sentences_A = [f"{all_entities[idx]}'s favorite color is {all_colors_A[idx]}." for idx in indices]
    sentences_B = [f"{all_entities[idx]}'s favorite color is {all_colors_B[idx]}." for idx in indices]
    
    prefix = " ".join(sentences_A[:fixed_pos])
    if prefix:
        prefix += " "
    suffix = " " + " ".join(sentences_A[fixed_pos + 1:]) if fixed_pos + 1 < len(sentences_A) else ""
    
    context_A = prefix + sentences_A[fixed_pos] + suffix
    context_B = prefix + sentences_B[fixed_pos] + suffix
    
    return {'context_A': context_A, 'context_B': context_B, 'target_A': color_A, 'target_B': color_B}

print(f"\nGenerating {NUM_SAMPLES_TO_TEST} FavoriteColor sample pairs...")
sample_pairs_color = []
for i in range(NUM_SAMPLES_TO_TEST):
    pair_data = generate_sample_pair_color(n_entities=N_ENTITIES, fixed_entity=FIXED_ENTITY_COLOR, seed=1000 + i)
    
    prompt_A = make_prompt_color(pair_data['context_A'], FIXED_ENTITY_COLOR)
    prompt_B = make_prompt_color(pair_data['context_B'], FIXED_ENTITY_COLOR)
    
    hook_text_A = f"Given the following context, let's answer the question below.\n\nContext:\n{pair_data['context_A']}\n\n"
    
    tokens_A = tokenizer.encode(prompt_A)
    tokens_B = tokenizer.encode(prompt_B)
    
    hook_idx_A = find_hook_token_idx(tokens_A, hook_text_A, tokenizer)
    hook_idx_B = find_hook_token_idx(tokens_B, hook_text_A.replace(pair_data['context_A'], pair_data['context_B']), tokenizer)
    
    sample_pairs_color.append({
        'tokens_A': tokens_A, 'tokens_B': tokens_B,
        'hook_idx_A': hook_idx_A, 'hook_idx_B': hook_idx_B,
        'target_A': pair_data['target_A'], 'target_B': pair_data['target_B'],
    })

# %%
# Generate LivesInCity samples
FIXED_ENTITY_CITY = "bat"

def make_prompt_city(context, entity_name="bat"):
    return (
        f"Given the following context, let's answer the question below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: Where does {entity_name} live?\n\n"
        f"Answer: {entity_name} lives in"
    )

def generate_sample_pair_city(n_entities=30, fixed_entity="bat", seed=42):
    rng = random.Random(seed)
    
    non_paris_cities = [c for c in DEFAULT_CITIES if c.lower() != "paris"]
    city_A = rng.choice(non_paris_cities)
    city_B = "Paris"
    
    available_names = [n for n in ANIMAL_NAMES if n != fixed_entity]
    other_entities = rng.sample(available_names, min(n_entities - 1, len(available_names)))
    other_cities = [rng.choice(DEFAULT_CITIES) for _ in other_entities]
    
    all_entities = other_entities + [fixed_entity]
    all_cities_A = other_cities + [city_A]
    all_cities_B = other_cities + [city_B]
    
    indices = list(range(len(all_entities)))
    rng.shuffle(indices)
    
    fixed_pos = indices.index(len(all_entities) - 1)
    
    sentences_A = [f"{all_entities[idx]} lives in {all_cities_A[idx]}." for idx in indices]
    sentences_B = [f"{all_entities[idx]} lives in {all_cities_B[idx]}." for idx in indices]
    
    prefix = " ".join(sentences_A[:fixed_pos])
    if prefix:
        prefix += " "
    suffix = " " + " ".join(sentences_A[fixed_pos + 1:]) if fixed_pos + 1 < len(sentences_A) else ""
    
    context_A = prefix + sentences_A[fixed_pos] + suffix
    context_B = prefix + sentences_B[fixed_pos] + suffix
    
    return {'context_A': context_A, 'context_B': context_B, 'target_A': city_A, 'target_B': city_B}

print(f"Generating {NUM_SAMPLES_TO_TEST} LivesInCity sample pairs...")
sample_pairs_city = []
for i in range(NUM_SAMPLES_TO_TEST):
    pair_data = generate_sample_pair_city(n_entities=N_ENTITIES, fixed_entity=FIXED_ENTITY_CITY, seed=1000 + i)
    
    prompt_A = make_prompt_city(pair_data['context_A'], FIXED_ENTITY_CITY)
    prompt_B = make_prompt_city(pair_data['context_B'], FIXED_ENTITY_CITY)
    
    hook_text_A = f"Given the following context, let's answer the question below.\n\nContext:\n{pair_data['context_A']}\n\n"
    
    tokens_A = tokenizer.encode(prompt_A)
    tokens_B = tokenizer.encode(prompt_B)
    
    hook_idx_A = find_hook_token_idx(tokens_A, hook_text_A, tokenizer)
    hook_idx_B = find_hook_token_idx(tokens_B, hook_text_A.replace(pair_data['context_A'], pair_data['context_B']), tokenizer)
    
    sample_pairs_city.append({
        'tokens_A': tokens_A, 'tokens_B': tokens_B,
        'hook_idx_A': hook_idx_A, 'hook_idx_B': hook_idx_B,
        'target_A': pair_data['target_A'], 'target_B': pair_data['target_B'],
    })

# %%
def run_hook_experiment(sample_pairs, heads_to_replace, model, tokenizer, n_layer, token_count=5):
    results = {'original': 0, 'switched': 0, 'neither': 0}
    
    for pair in sample_pairs:
        tokens_before_hook_A = pair['tokens_A'][:pair['hook_idx_A']]
        tokens_before_hook_B = pair['tokens_B'][:pair['hook_idx_B']]
        tokens_after_hook_A = pair['tokens_A'][pair['hook_idx_A']:]
        
        state_A = get_state_after_tokens(model, tokens_before_hook_A)
        state_A = [s.clone() for s in state_A]
        state_B = get_state_after_tokens(model, tokens_before_hook_B)
        
        if heads_to_replace:
            modified_state = replace_specific_heads(state_A, state_B, heads_to_replace, n_layer)
        else:
            modified_state = state_A
        
        with torch.no_grad():
            if tokens_after_hook_A:
                out, current_state = model.model.forward(tokens_after_hook_A, modified_state)
            else:
                out, current_state = model.model.forward([tokens_before_hook_A[-1]], modified_state)
            
            output_tokens = []
            for _ in range(token_count):
                next_token = out.argmax().item()
                output_tokens.append(next_token)
                if next_token in [47, 11]:
                    break
                out, current_state = model.model.forward([next_token], current_state)
        
        response = decode_tokens_safe(tokenizer, output_tokens).strip().lower()
        
        if pair['target_A'].lower() in response:
            results['original'] += 1
        elif pair['target_B'].lower() in response:
            results['switched'] += 1
        else:
            results['neither'] += 1
    
    return results

# %%
print("\n" + "=" * 80)
print("CROSS-DATASET HEAD TESTING - MLP PROBES")
print("=" * 80)

all_results = []
n_total = NUM_SAMPLES_TO_TEST

# Test FavoriteColor MLP heads on both datasets
print("\nTesting FavoriteColor MLP heads...")
result = run_hook_experiment(sample_pairs_color, heads_color_mlp, model, tokenizer, n_layer)
all_results.append({
    'heads_from': 'FavoriteColor MLP',
    'heads': str(heads_color_mlp),
    'tested_on': 'FavoriteColor',
    'original': result['original'],
    'switched': result['switched'],
    'neither': result['neither'],
    'switch_rate': result['switched'] / n_total * 100,
})

result = run_hook_experiment(sample_pairs_city, heads_color_mlp, model, tokenizer, n_layer)
all_results.append({
    'heads_from': 'FavoriteColor MLP',
    'heads': str(heads_color_mlp),
    'tested_on': 'LivesInCity',
    'original': result['original'],
    'switched': result['switched'],
    'neither': result['neither'],
    'switch_rate': result['switched'] / n_total * 100,
})

# Test LivesInCity MLP heads on both datasets
print("Testing LivesInCity MLP heads...")
result = run_hook_experiment(sample_pairs_color, heads_city_mlp, model, tokenizer, n_layer)
all_results.append({
    'heads_from': 'LivesInCity MLP',
    'heads': str(heads_city_mlp),
    'tested_on': 'FavoriteColor',
    'original': result['original'],
    'switched': result['switched'],
    'neither': result['neither'],
    'switch_rate': result['switched'] / n_total * 100,
})

result = run_hook_experiment(sample_pairs_city, heads_city_mlp, model, tokenizer, n_layer)
all_results.append({
    'heads_from': 'LivesInCity MLP',
    'heads': str(heads_city_mlp),
    'tested_on': 'LivesInCity',
    'original': result['original'],
    'switched': result['switched'],
    'neither': result['neither'],
    'switch_rate': result['switched'] / n_total * 100,
})

# Add baselines
print("Running baselines...")
for dataset_name, samples in [('FavoriteColor', sample_pairs_color), ('LivesInCity', sample_pairs_city)]:
    result = run_hook_experiment(samples, [], model, tokenizer, n_layer)
    all_results.append({
        'heads_from': 'BASELINE',
        'heads': '[]',
        'tested_on': dataset_name,
        'original': result['original'],
        'switched': result['switched'],
        'neither': result['neither'],
        'switch_rate': result['switched'] / n_total * 100,
    })

# Add all heads
print("Running all heads...")
all_heads = [(layer, head) for layer in range(n_layer) for head in range(n_head)]
for dataset_name, samples in [('FavoriteColor', sample_pairs_color), ('LivesInCity', sample_pairs_city)]:
    result = run_hook_experiment(samples, all_heads, model, tokenizer, n_layer)
    all_results.append({
        'heads_from': 'ALL HEADS',
        'heads': f'{n_layer*n_head} heads',
        'tested_on': dataset_name,
        'original': result['original'],
        'switched': result['switched'],
        'neither': result['neither'],
        'switch_rate': result['switched'] / n_total * 100,
    })

# %%
print("\n" + "=" * 100)
print("FINAL RESULTS TABLE - MLP PROBES")
print("=" * 100)

results_df = pd.DataFrame(all_results)

print(f"\n{'Heads From':<25} {'Tested On':<15} {'Original':>10} {'Switched':>10} {'Neither':>10} {'Switch%':>10}")
print("-" * 85)

for _, row in results_df.iterrows():
    print(f"{row['heads_from']:<25} {row['tested_on']:<15} {row['original']:>10} {row['switched']:>10} {row['neither']:>10} {row['switch_rate']:>9.1f}%")

# %%
print("\n" + "=" * 100)
print("ANALYSIS: SAME-TASK vs CROSS-TASK (MLP)")
print("=" * 100)

print("\n### FavoriteColor MLP Heads ###")
print(f"  Tested on FavoriteColor (SAME): {results_df[(results_df['heads_from']=='FavoriteColor MLP') & (results_df['tested_on']=='FavoriteColor')]['switch_rate'].values[0]:.1f}% switch")
print(f"  Tested on LivesInCity (CROSS):  {results_df[(results_df['heads_from']=='FavoriteColor MLP') & (results_df['tested_on']=='LivesInCity')]['switch_rate'].values[0]:.1f}% switch")

print("\n### LivesInCity MLP Heads ###")
print(f"  Tested on LivesInCity (SAME):   {results_df[(results_df['heads_from']=='LivesInCity MLP') & (results_df['tested_on']=='LivesInCity')]['switch_rate'].values[0]:.1f}% switch")
print(f"  Tested on FavoriteColor (CROSS): {results_df[(results_df['heads_from']=='LivesInCity MLP') & (results_df['tested_on']=='FavoriteColor')]['switch_rate'].values[0]:.1f}% switch")

# %%
results_df.to_csv('results/cross_dataset_comparison_mlp.csv', index=False)
print(f"\nResults saved to results/cross_dataset_comparison_mlp.csv")

# %%
