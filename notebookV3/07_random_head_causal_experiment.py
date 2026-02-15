# %% [markdown]
# # Random Head Causal Experiment
# 
# Tests the effect of patching randomly selected heads (0.5% of total)
# as a baseline for comparison with probe-identified heads.
# Averages results over 5 seeds.

# %%
import sys
import os
import torch
import pandas as pd
from typing import List, Tuple, Dict
import random

sys.path.insert(0, os.path.abspath('..'))

from models.load_rwkv import load_rwkv_model
from datasets.favorite_color_dataset import FAMOUS_NAMES, DEFAULT_COLORS

# %%
NUM_HEADS_TO_TEST = 3  # 0.5% of 768 heads ≈ 3-4 heads
NUM_SAMPLES_TO_TEST = 50
NUM_SEEDS = 5

# %%
print("Loading RWKV model...")
model, tokenizer = load_rwkv_model(model_name="rwkv7-g1b-1.5b-20251202-ctx8192.pth")

n_layer = model.model.n_layer
n_head = model.model.n_head
head_size = model.model.head_size
total_heads = n_layer * n_head

print(f"\nModel info: {n_layer} layers, {n_head} heads, head_size={head_size}")
print(f"Total heads: {total_heads}")
print(f"0.5% of heads: {int(total_heads * 0.005)} heads")

# %%
N_ENTITIES = 30
FIXED_ENTITY = "Lady Gaga"

def make_prompt(context: str, entity_name: str = "Lady Gaga") -> str:
    return (
        f"Given the following context, let's answer the question below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: What is {entity_name}'s favorite color?\n\n"
        f"Answer: {entity_name}'s favorite color is"
    )

def generate_sample_pair(
    n_entities: int = 30,
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

def get_random_heads(n_layer: int, n_head: int, num_heads: int, seed: int) -> List[Tuple[int, int]]:
    """Select random heads from all available heads."""
    rng = random.Random(seed)
    all_heads = [(layer, head) for layer in range(n_layer) for head in range(n_head)]
    return rng.sample(all_heads, num_heads)

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
print(f"RANDOM HEAD SELECTION EXPERIMENT")
print(f"Testing {NUM_HEADS_TO_TEST} random heads (≈0.5% of {total_heads} total)")
print(f"Testing on {NUM_SAMPLES_TO_TEST} samples")
print(f"Averaging over {NUM_SEEDS} seeds")
print("=" * 80)

n_total = len(sample_pairs)

# Run baseline first
print(f"\n[BASELINE] No heads replaced:")
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

# Run random head experiments for each seed
all_random_results = []
for seed in range(NUM_SEEDS):
    random_heads = get_random_heads(n_layer, n_head, NUM_HEADS_TO_TEST, seed=seed)
    print(f"\n[SEED {seed}] Random heads: {random_heads}")
    
    results_random = run_head_hook_experiment(
        sample_pairs,
        heads_to_replace=random_heads,
        model=model,
        tokenizer=tokenizer,
        n_layer=n_layer,
        token_count=5
    )
    
    all_random_results.append(results_random)
    print(f"    Original: {results_random['original']:3d}/{n_total} ({results_random['original']/n_total*100:5.1f}%)")
    print(f"    Blue:     {results_random['blue']:3d}/{n_total} ({results_random['blue']/n_total*100:5.1f}%)")
    print(f"    Neither:  {results_random['neither']:3d}/{n_total} ({results_random['neither']/n_total*100:5.1f}%)")

# Calculate averages
avg_original = sum(r['original'] for r in all_random_results) / NUM_SEEDS
avg_blue = sum(r['blue'] for r in all_random_results) / NUM_SEEDS
avg_neither = sum(r['neither'] for r in all_random_results) / NUM_SEEDS

# %%
print("\n" + "=" * 80)
print("SUMMARY: RANDOM HEAD SELECTION (averaged over 5 seeds)")
print("=" * 80)
print(f"\n{'Method':<25} {'Correct':>10} {'Switched':>10} {'Neither':>10}")
print("-" * 60)
print(f"{'Baseline':<25} {results_baseline['original']/n_total*100:>9.1f}% {results_baseline['blue']/n_total*100:>9.1f}% {results_baseline['neither']/n_total*100:>9.1f}%")
print(f"{'Random (0.5%, avg)':<25} {avg_original/n_total*100:>9.1f}% {avg_blue/n_total*100:>9.1f}% {avg_neither/n_total*100:>9.1f}%")

print(f"\n\nFill these values in paper.tex:")
print(f"  - Random Correct: {avg_original/n_total*100:.1f}%")
print(f"  - Random Switched: {avg_blue/n_total*100:.1f}%")
print(f"  - Random Neither: {avg_neither/n_total*100:.1f}%")

# %%
