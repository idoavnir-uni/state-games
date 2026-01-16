# %% [markdown]
# # Compare Probe Results via State Hooking
# 
# Tests the top heads from each probe method (linear, mlp, unembedding, rwkv_mlp)
# by replacing their states and measuring the effect on model output.

# %%
import sys
import os
import torch
import pandas as pd
from typing import List, Tuple, Dict
import random

sys.path.insert(0, os.path.abspath('..'))

from models.load_rwkv import load_rwkv_model
from datasets.animal_names import ANIMAL_NAMES
from datasets.lives_in_city_dataset import DEFAULT_CITIES

# %%
NUM_HEADS_TO_TEST = 3
NUM_SAMPLES_TO_TEST = 50

# %%
print("Loading RWKV model...")
model, tokenizer = load_rwkv_model(model_name="rwkv7-g1b-1.5b-20251202-ctx8192.pth")

n_layer = model.model.n_layer
n_head = model.model.n_head
head_size = model.model.head_size

print(f"\nModel info: {n_layer} layers, {n_head} heads, head_size={head_size}")

# %%
print("\nLoading probe results from CSV files...")

results_both_linear = pd.read_csv('results/result_both_linear.csv')
results_mlp = pd.read_csv('results/result_mlp.csv')
results_unembeding = pd.read_csv('results/result_unembeding.csv')
results_rwkv_mlp = pd.read_csv('results/result_rwkv_mlp.csv')

print(f"  result_both_linear.csv: {len(results_both_linear)} rows")
print(f"  result_mlp.csv: {len(results_mlp)} rows")
print(f"  result_unembeding.csv: {len(results_unembeding)} rows")
print(f"  result_rwkv_mlp.csv: {len(results_rwkv_mlp)} rows")

def df_to_heads(df: pd.DataFrame, n: int) -> List[Tuple[int, int]]:
    """Convert top n rows of dataframe to list of (layer, head) tuples."""
    top_n = df.head(n)
    return [(int(row['layer']), int(row['head'])) for _, row in top_n.iterrows()]

heads_both_linear = df_to_heads(results_both_linear, NUM_HEADS_TO_TEST)
heads_mlp = df_to_heads(results_mlp, NUM_HEADS_TO_TEST)
heads_unembeding = df_to_heads(results_unembeding, min(NUM_HEADS_TO_TEST, len(results_unembeding)))
heads_rwkv_mlp = df_to_heads(results_rwkv_mlp, NUM_HEADS_TO_TEST)

print(f"\nTop {NUM_HEADS_TO_TEST} heads from each method:")
print(f"  both_linear: {heads_both_linear}")
print(f"  mlp: {heads_mlp}")
print(f"  unembeding: {heads_unembeding}")
print(f"  rwkv_mlp: {heads_rwkv_mlp}")

# %%
N_ENTITIES = 30
FIXED_ENTITY = "bat"

def make_prompt(context: str, entity_name: str = "bat") -> str:
    return (
        f"Given the following context, let's answer the question below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: Where does {entity_name} live?\n\n"
        f"Answer: {entity_name} lives in"
    )

def generate_sample_pair(
    n_entities: int = 30,
    fixed_entity: str = "bat",
    seed: int = 42
) -> Dict:
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
    
    fixed_entity_idx_in_list = len(all_entities) - 1
    fixed_pos = indices.index(fixed_entity_idx_in_list)
    
    sentences_A = []
    sentences_B = []
    for idx in indices:
        sentences_A.append(f"{all_entities[idx]} lives in {all_cities_A[idx]}.")
        sentences_B.append(f"{all_entities[idx]} lives in {all_cities_B[idx]}.")
    
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
        'color_A': city_A,
        'color_B': city_B,
        'prefix': prefix,
        'sentence_A': sentence_A,
        'sentence_B': sentence_B,
        'suffix': suffix,
        'fixed_pos': fixed_pos,
    }

def get_state_after_tokens(model, tokens: List[int], initial_state=None):
    if not tokens:
        return initial_state
    with torch.no_grad():
        _, state = model.model.forward(tokens, initial_state)
    return state

def replace_specific_heads(
    state_A: List,
    state_B: List,
    heads_to_replace: List[Tuple[int, int]],
    n_layer: int
) -> List:
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

def find_hook_token_idx(full_tokens: List[int], hook_text: str, tokenizer) -> int:
    hook_tokens = tokenizer.encode(hook_text)
    hook_len = len(hook_tokens)
    
    for i in range(min(hook_len + 5, len(full_tokens)), hook_len - 5, -1):
        if full_tokens[:i] == hook_tokens[:i] or i <= hook_len:
            decoded_prefix = tokenizer.decode(full_tokens[:i])
            if len(decoded_prefix) >= len(hook_text) - 5:
                return i
    
    return len(hook_tokens)

def decode_tokens_safe(tokenizer, token_ids: List[int]) -> str:
    filtered = [t for t in token_ids if t != 0]
    if not filtered:
        return ""
    return tokenizer.decode(filtered)

# %%
print(f"\nGenerating {NUM_SAMPLES_TO_TEST} sample pairs...")

sample_pairs = []
for i in range(NUM_SAMPLES_TO_TEST):
    pair_data = generate_sample_pair(
        n_entities=N_ENTITIES,
        fixed_entity=FIXED_ENTITY,
        seed=1000 + i
    )
    
    prompt_A = make_prompt(pair_data['context_A'], FIXED_ENTITY)
    prompt_B = make_prompt(pair_data['context_B'], FIXED_ENTITY)
    
    hook_text_A = f"Given the following context, let's answer the question below.\n\nContext:\n{pair_data['context_A']}\n\n"
    hook_text_B = f"Given the following context, let's answer the question below.\n\nContext:\n{pair_data['context_B']}\n\n"
    
    tokens_A = tokenizer.encode(prompt_A)
    tokens_B = tokenizer.encode(prompt_B)
    
    hook_idx_A = find_hook_token_idx(tokens_A, hook_text_A, tokenizer)
    hook_idx_B = find_hook_token_idx(tokens_B, hook_text_B, tokenizer)
    
    sample_pairs.append({
        'prompt_A': prompt_A,
        'prompt_B': prompt_B,
        'tokens_A': tokens_A,
        'tokens_B': tokens_B,
        'hook_idx_A': hook_idx_A,
        'hook_idx_B': hook_idx_B,
        'color_A': pair_data['color_A'],
        'color_B': pair_data['color_B'],
    })

print(f"Generated {len(sample_pairs)} sample pairs")
print(f"Hook position: AFTER CONTEXT (before Question)")

# %%
def run_head_hook_experiment(
    sample_pairs: List[Dict],
    heads_to_replace: List[Tuple[int, int]],
    model,
    tokenizer,
    n_layer: int,
    token_count: int = 5,
) -> Dict:
    results = {
        'original': 0,
        'blue': 0,
        'neither': 0,
        'details': []
    }
    
    for pair in sample_pairs:
        tokens_A = pair['tokens_A']
        tokens_B = pair['tokens_B']
        hook_idx_A = pair['hook_idx_A']
        hook_idx_B = pair['hook_idx_B']
        color_A = pair['color_A']
        color_B = pair['color_B']
        
        tokens_before_hook_A = tokens_A[:hook_idx_A]
        tokens_before_hook_B = tokens_B[:hook_idx_B]
        tokens_after_hook_A = tokens_A[hook_idx_A:]
        
        state_A = get_state_after_tokens(model, tokens_before_hook_A)
        state_A = [s.clone() for s in state_A]
        state_B = get_state_after_tokens(model, tokens_before_hook_B)
        
        if heads_to_replace:
            modified_state = replace_specific_heads(state_A, state_B, heads_to_replace, n_layer)
        else:
            modified_state = state_A
        
        with torch.no_grad():
            current_state = modified_state
            
            if tokens_after_hook_A:
                out, current_state = model.model.forward(tokens_after_hook_A, current_state)
            else:
                out, current_state = model.model.forward([tokens_before_hook_A[-1]], 
                    get_state_after_tokens(model, tokens_before_hook_A[:-1]) if len(tokens_before_hook_A) > 1 else None)
            
            output_tokens = []
            for _ in range(token_count):
                next_token = out.argmax().item()
                output_tokens.append(next_token)
                if next_token in [47, 11]:
                    break
                out, current_state = model.model.forward([next_token], current_state)
        
        response = decode_tokens_safe(tokenizer, output_tokens)
        response_lower = response.strip().lower()
        
        if color_A.lower() in response_lower:
            results['original'] += 1
            classification = 'original'
        elif color_B.lower() in response_lower:
            results['blue'] += 1
            classification = 'blue'
        else:
            results['neither'] += 1
            classification = 'neither'
        
        results['details'].append({
            'color_A': color_A,
            'color_B': color_B,
            'response': response,
            'classification': classification
        })
    
    return results

# %%
print("\n" + "=" * 80)
print(f"COMPARING PROBE METHODS (Top {NUM_HEADS_TO_TEST} heads each)")
print(f"Testing on {NUM_SAMPLES_TO_TEST} samples")
print("=" * 80)

n_total = len(sample_pairs)

print(f"\n[1] BASELINE - No heads replaced:")
results_baseline = run_head_hook_experiment(
    sample_pairs,
    heads_to_replace=[],
    model=model,
    tokenizer=tokenizer,
    n_layer=n_layer,
    token_count=5
)
print(f"    Original: {results_baseline['original']:3d}/{n_total} ({results_baseline['original']/n_total*100:5.1f}%)")
print(f"    Blue:     {results_baseline['blue']:3d}/{n_total} ({results_baseline['blue']/n_total*100:5.1f}%)")
print(f"    Neither:  {results_baseline['neither']:3d}/{n_total} ({results_baseline['neither']/n_total*100:5.1f}%)")

print(f"\n[2] BOTH LINEAR - Top {len(heads_both_linear)} heads:")
print(f"    Heads: {heads_both_linear}")
results_linear = run_head_hook_experiment(
    sample_pairs,
    heads_to_replace=heads_both_linear,
    model=model,
    tokenizer=tokenizer,
    n_layer=n_layer,
    token_count=5
)
print(f"    Original: {results_linear['original']:3d}/{n_total} ({results_linear['original']/n_total*100:5.1f}%)")
print(f"    Blue:     {results_linear['blue']:3d}/{n_total} ({results_linear['blue']/n_total*100:5.1f}%)")
print(f"    Neither:  {results_linear['neither']:3d}/{n_total} ({results_linear['neither']/n_total*100:5.1f}%)")

print(f"\n[3] MLP - Top {len(heads_mlp)} heads:")
print(f"    Heads: {heads_mlp}")
results_mlp_hook = run_head_hook_experiment(
    sample_pairs,
    heads_to_replace=heads_mlp,
    model=model,
    tokenizer=tokenizer,
    n_layer=n_layer,
    token_count=5
)
print(f"    Original: {results_mlp_hook['original']:3d}/{n_total} ({results_mlp_hook['original']/n_total*100:5.1f}%)")
print(f"    Blue:     {results_mlp_hook['blue']:3d}/{n_total} ({results_mlp_hook['blue']/n_total*100:5.1f}%)")
print(f"    Neither:  {results_mlp_hook['neither']:3d}/{n_total} ({results_mlp_hook['neither']/n_total*100:5.1f}%)")

print(f"\n[4] UNEMBEDDING - Top {len(heads_unembeding)} heads:")
print(f"    Heads: {heads_unembeding}")
results_unembed = run_head_hook_experiment(
    sample_pairs,
    heads_to_replace=heads_unembeding,
    model=model,
    tokenizer=tokenizer,
    n_layer=n_layer,
    token_count=5
)
print(f"    Original: {results_unembed['original']:3d}/{n_total} ({results_unembed['original']/n_total*100:5.1f}%)")
print(f"    Blue:     {results_unembed['blue']:3d}/{n_total} ({results_unembed['blue']/n_total*100:5.1f}%)")
print(f"    Neither:  {results_unembed['neither']:3d}/{n_total} ({results_unembed['neither']/n_total*100:5.1f}%)")

print(f"\n[5] RWKV MLP - Top {len(heads_rwkv_mlp)} heads:")
print(f"    Heads: {heads_rwkv_mlp}")
results_rwkv = run_head_hook_experiment(
    sample_pairs,
    heads_to_replace=heads_rwkv_mlp,
    model=model,
    tokenizer=tokenizer,
    n_layer=n_layer,
    token_count=5
)
print(f"    Original: {results_rwkv['original']:3d}/{n_total} ({results_rwkv['original']/n_total*100:5.1f}%)")
print(f"    Blue:     {results_rwkv['blue']:3d}/{n_total} ({results_rwkv['blue']/n_total*100:5.1f}%)")
print(f"    Neither:  {results_rwkv['neither']:3d}/{n_total} ({results_rwkv['neither']/n_total*100:5.1f}%)")

print(f"\n[6] ALL HEADS - Every head in every layer ({n_layer} layers × {n_head} heads = {n_layer * n_head} total):")
all_heads = [(layer, head) for layer in range(n_layer) for head in range(n_head)]
results_all = run_head_hook_experiment(
    sample_pairs,
    heads_to_replace=all_heads,
    model=model,
    tokenizer=tokenizer,
    n_layer=n_layer,
    token_count=5
)
print(f"    Original: {results_all['original']:3d}/{n_total} ({results_all['original']/n_total*100:5.1f}%)")
print(f"    Blue:     {results_all['blue']:3d}/{n_total} ({results_all['blue']/n_total*100:5.1f}%)")
print(f"    Neither:  {results_all['neither']:3d}/{n_total} ({results_all['neither']/n_total*100:5.1f}%)")

# %%
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

print(f"\n{'Method':<20} {'Heads':<8} {'Original':>10} {'Blue':>10} {'Neither':>10} {'Blue %':>10}")
print("-" * 70)
print(f"{'Baseline':<20} {0:<8} {results_baseline['original']:>10} {results_baseline['blue']:>10} {results_baseline['neither']:>10} {results_baseline['blue']/n_total*100:>9.1f}%")
print(f"{'Both Linear':<20} {len(heads_both_linear):<8} {results_linear['original']:>10} {results_linear['blue']:>10} {results_linear['neither']:>10} {results_linear['blue']/n_total*100:>9.1f}%")
print(f"{'MLP':<20} {len(heads_mlp):<8} {results_mlp_hook['original']:>10} {results_mlp_hook['blue']:>10} {results_mlp_hook['neither']:>10} {results_mlp_hook['blue']/n_total*100:>9.1f}%")
print(f"{'Unembedding':<20} {len(heads_unembeding):<8} {results_unembed['original']:>10} {results_unembed['blue']:>10} {results_unembed['neither']:>10} {results_unembed['blue']/n_total*100:>9.1f}%")
print(f"{'RWKV MLP':<20} {len(heads_rwkv_mlp):<8} {results_rwkv['original']:>10} {results_rwkv['blue']:>10} {results_rwkv['neither']:>10} {results_rwkv['blue']/n_total*100:>9.1f}%")
print(f"{'All Heads':<20} {n_layer * n_head:<8} {results_all['original']:>10} {results_all['blue']:>10} {results_all['neither']:>10} {results_all['blue']/n_total*100:>9.1f}%")

# %%
print("\n" + "=" * 80)
print("SAMPLE RESPONSES (first 5 from each method)")
print("=" * 80)

for name, results in [
    ("Baseline", results_baseline),
    ("Both Linear", results_linear),
    ("MLP", results_mlp_hook),
    ("Unembedding", results_unembed),
    ("RWKV MLP", results_rwkv),
    ("All Heads", results_all),
]:
    print(f"\n{name}:")
    for i, d in enumerate(results['details'][:5]):
        print(f"  {i}: A={d['color_A']:<8} → '{d['response']:<15}' [{d['classification']}]")

# %%
