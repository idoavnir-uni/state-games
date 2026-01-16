# %% [markdown]
# # State Hooking - Probability Analysis (Hook After Full Context)
# 
# Instead of checking final output color, measure probability changes:
# - How much did P(blue) increase?
# - By what factor did P(blue) multiply?
# - How much did P(original) decrease?
# - By what factor did P(original) decrease?
#
# Only hooks att_kv states (matrices), not att_x_prev/ffn_x_prev (vectors).
#
# Hook point: Right after the full context (before "Question:" part)

# %%
import sys
import os
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

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

# Get token IDs for colors
COLOR_TOKEN_IDS = {}
for color in DEFAULT_COLORS:
    tokens = tokenizer.encode(color)
    if len(tokens) == 1:
        COLOR_TOKEN_IDS[color.lower()] = tokens[0]
print(f"Color token IDs: {COLOR_TOKEN_IDS}")

# %%
def make_prompt(context: str, entity_name: str = "Lady Gaga") -> str:
    return (
        f"Given the following context, let's answer the question below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: What is {entity_name}'s favorite color?\n\n"
        f"Answer: {entity_name}'s favorite color is"
    )


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


def replace_kv_only(
    state_A: List,
    state_B: List,
    layers: List[int],
    n_layer: int
) -> List:
    """
    Replace ONLY att_kv (matrix state) for given layers.
    Does NOT replace att_x_prev or ffn_x_prev (vectors).
    """
    new_state = [s.clone() for s in state_A]
    
    for layer in layers:
        if layer < n_layer:
            att_kv_idx = layer * 3 + 1
            new_state[att_kv_idx] = state_B[att_kv_idx].clone()
    
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


# %%
N_ENTITIES = 10
N_SAMPLES = 10
FIXED_ENTITY = "Lady Gaga"

print(f"Generating {N_SAMPLES} sample pairs with n_entities={N_ENTITIES}...")
print("Hook point: Right after full context (before Question)")

sample_pairs = []
for i in range(N_SAMPLES):
    pair_data = generate_sample_pair(
        n_entities=N_ENTITIES,
        fixed_entity=FIXED_ENTITY,
        seed=1000 + i
    )
    
    prompt_A = make_prompt(pair_data['context_A'], FIXED_ENTITY)
    prompt_B = make_prompt(pair_data['context_B'], FIXED_ENTITY)
    
    prompt_prefix = f"Given the following context, let's answer the question below.\n\nContext:\n"
    
    # Hook after full context (before "\n\nQuestion:")
    hook_text_A = prompt_prefix + pair_data['context_A']
    hook_text_B = prompt_prefix + pair_data['context_B']
    
    tokens_A = tokenizer.encode(prompt_A)
    tokens_B = tokenizer.encode(prompt_B)
    
    hook_idx_A = find_hook_token_idx(tokens_A, hook_text_A, tokenizer)
    hook_idx_B = find_hook_token_idx(tokens_B, hook_text_B, tokenizer)
    
    sample_pairs.append({
        'tokens_A': tokens_A,
        'tokens_B': tokens_B,
        'hook_idx_A': hook_idx_A,
        'hook_idx_B': hook_idx_B,
        'color_A': pair_data['color_A'],
        'color_B': pair_data['color_B'],
    })

print(f"Generated {len(sample_pairs)} pairs")

# Verify hook point
print("\n--- Hook Point Verification ---")
pair = sample_pairs[0]
tokens_before = pair['tokens_A'][:pair['hook_idx_A']]
tokens_after = pair['tokens_A'][pair['hook_idx_A']:]
print(f"Tokens before hook: {len(tokens_before)}")
print(f"Text before hook ends with: ...{tokenizer.decode(tokens_before[-10:])!r}")
print(f"Text after hook starts with: {tokenizer.decode(tokens_after[:10])!r}...")


# %%
def get_color_probabilities_normalized(logits):
    """
    Get probabilities for color tokens, normalized to sum to 1.
    Only considers color tokens, ignoring the rest of the vocabulary.
    """
    # Extract logits only for color tokens
    color_token_ids = list(COLOR_TOKEN_IDS.values())
    color_logits = logits[color_token_ids]
    
    # Softmax over only color tokens
    color_probs = F.softmax(color_logits, dim=-1)
    
    # Map back to color names
    result = {}
    for i, (color, token_id) in enumerate(COLOR_TOKEN_IDS.items()):
        result[color] = color_probs[i].item()
    
    return result


def run_probability_analysis(
    sample_pairs: List[Dict],
    layers_to_replace: List[int],  # Empty = baseline
    model,
    tokenizer,
    n_layer: int,
) -> Dict:
    """
    Run experiment and measure probability changes.
    Returns average metrics across all samples.
    """
    results = {
        'p_blue': [],
        'p_original': [],
        'samples': []
    }
    
    for pair in sample_pairs:
        tokens_A = pair['tokens_A']
        tokens_B = pair['tokens_B']
        hook_idx_A = pair['hook_idx_A']
        hook_idx_B = pair['hook_idx_B']
        color_A = pair['color_A'].lower()
        color_B = pair['color_B'].lower()  # "blue"
        
        tokens_before_hook_A = tokens_A[:hook_idx_A]
        tokens_before_hook_B = tokens_B[:hook_idx_B]
        tokens_after_hook_A = tokens_A[hook_idx_A:]
        
        # Get states at hook point
        state_A = get_state_after_tokens(model, tokens_before_hook_A)
        state_A = [s.clone() for s in state_A]
        state_B = get_state_after_tokens(model, tokens_before_hook_B)
        
        # Replace KV states only (not vectors)
        if layers_to_replace:
            modified_state = replace_kv_only(state_A, state_B, layers_to_replace, n_layer)
        else:
            modified_state = state_A
        
        # Process remaining tokens and get final probabilities
        with torch.no_grad():
            if tokens_after_hook_A:
                _, final_state = model.model.forward(tokens_after_hook_A, modified_state)
                # Get probabilities at the last position
                logits, _ = model.model.forward(tokens_after_hook_A[-1:], 
                    get_state_after_tokens(model, tokens_after_hook_A[:-1], modified_state) if len(tokens_after_hook_A) > 1 else modified_state)
            else:
                logits, final_state = model.model.forward([tokens_before_hook_A[-1]], 
                    get_state_after_tokens(model, tokens_before_hook_A[:-1]) if len(tokens_before_hook_A) > 1 else None)
            
            # Actually we need the logits at the END of the full prompt
            # Let me fix this - process all remaining tokens and get logits
            out, _ = model.model.forward(tokens_after_hook_A, modified_state) if tokens_after_hook_A else model.model.forward([tokens_before_hook_A[-1]], modified_state)
            
            # Get normalized probabilities (only over color tokens)
            color_probs = get_color_probabilities_normalized(out)
            
            p_blue = color_probs.get('blue', 0.0)
            p_original = color_probs.get(color_A, 0.0)
        
        results['p_blue'].append(p_blue)
        results['p_original'].append(p_original)
        results['samples'].append({
            'color_A': color_A,
            'color_B': color_B,
            'p_blue': p_blue,
            'p_original': p_original,
        })
    
    return results


# %%
print("=" * 80)
print("BASELINE: No replacement (original behavior)")
print("=" * 80)

baseline_results = run_probability_analysis(
    sample_pairs,
    layers_to_replace=[],
    model=model,
    tokenizer=tokenizer,
    n_layer=n_layer,
)

baseline_p_blue = sum(baseline_results['p_blue']) / len(baseline_results['p_blue'])
baseline_p_original = sum(baseline_results['p_original']) / len(baseline_results['p_original'])

print(f"\nBaseline probabilities (averaged over {len(sample_pairs)} samples):")
print(f"  P(blue):     {baseline_p_blue:.6f}")
print(f"  P(original): {baseline_p_original:.6f}")

# %%
print("\n" + "=" * 80)
print("BASELINE: Replace ALL layers (sanity check)")
print("=" * 80)

all_layers_results = run_probability_analysis(
    sample_pairs,
    layers_to_replace=list(range(n_layer)),
    model=model,
    tokenizer=tokenizer,
    n_layer=n_layer,
)

all_layers_p_blue = sum(all_layers_results['p_blue']) / len(all_layers_results['p_blue'])
all_layers_p_original = sum(all_layers_results['p_original']) / len(all_layers_results['p_original'])

eps = 1e-10
print(f"\nReplace ALL layers probabilities (averaged over {len(sample_pairs)} samples):")
print(f"  P(blue):     {all_layers_p_blue:.6f} (Δ={all_layers_p_blue - baseline_p_blue:+.6f}, ×{all_layers_p_blue / (baseline_p_blue + eps):.2f})")
print(f"  P(original): {all_layers_p_original:.6f} (Δ={all_layers_p_original - baseline_p_original:+.6f}, ×{all_layers_p_original / (baseline_p_original + eps):.2f})")


# %%
print("\n" + "=" * 80)
print("Testing each layer individually (KV state only)")
print("=" * 80)

layer_results = {}

for layer in range(n_layer):
    results = run_probability_analysis(
        sample_pairs,
        layers_to_replace=[layer],
        model=model,
        tokenizer=tokenizer,
        n_layer=n_layer,
    )
    
    avg_p_blue = sum(results['p_blue']) / len(results['p_blue'])
    avg_p_original = sum(results['p_original']) / len(results['p_original'])
    
    # Compute changes
    delta_p_blue = avg_p_blue - baseline_p_blue
    delta_p_original = avg_p_original - baseline_p_original
    
    # Compute ratios (with small epsilon to avoid division by zero)
    eps = 1e-10
    ratio_p_blue = avg_p_blue / (baseline_p_blue + eps)
    ratio_p_original = avg_p_original / (baseline_p_original + eps)
    
    layer_results[layer] = {
        'p_blue': avg_p_blue,
        'p_original': avg_p_original,
        'delta_p_blue': delta_p_blue,
        'delta_p_original': delta_p_original,
        'ratio_p_blue': ratio_p_blue,
        'ratio_p_original': ratio_p_original,
    }
    
    print(f"L{layer:2d}: P(blue)={avg_p_blue:.6f} (Δ={delta_p_blue:+.6f}, ×{ratio_p_blue:.2f}), "
          f"P(orig)={avg_p_original:.6f} (Δ={delta_p_original:+.6f}, ×{ratio_p_original:.2f})")


# %%
print("\n" + "=" * 80)
print("TOP 10 LAYERS: By P(blue) increase (delta)")
print("=" * 80)

sorted_by_blue_delta = sorted(layer_results.items(), key=lambda x: x[1]['delta_p_blue'], reverse=True)

print(f"\n{'Layer':<8} {'P(blue)':>12} {'Δ P(blue)':>12} {'× P(blue)':>12}")
print("-" * 50)
for layer, res in sorted_by_blue_delta[:10]:
    print(f"L{layer:<7} {res['p_blue']:>12.6f} {res['delta_p_blue']:>+12.6f} {res['ratio_p_blue']:>12.2f}")


# %%
print("\n" + "=" * 80)
print("TOP 10 LAYERS: By P(blue) increase (ratio/factor)")
print("=" * 80)

sorted_by_blue_ratio = sorted(layer_results.items(), key=lambda x: x[1]['ratio_p_blue'], reverse=True)

print(f"\n{'Layer':<8} {'P(blue)':>12} {'Δ P(blue)':>12} {'× P(blue)':>12}")
print("-" * 50)
for layer, res in sorted_by_blue_ratio[:10]:
    print(f"L{layer:<7} {res['p_blue']:>12.6f} {res['delta_p_blue']:>+12.6f} {res['ratio_p_blue']:>12.2f}")


# %%
print("\n" + "=" * 80)
print("TOP 10 LAYERS: By P(original) decrease (delta, most negative)")
print("=" * 80)

sorted_by_orig_delta = sorted(layer_results.items(), key=lambda x: x[1]['delta_p_original'])

print(f"\n{'Layer':<8} {'P(orig)':>12} {'Δ P(orig)':>12} {'× P(orig)':>12}")
print("-" * 50)
for layer, res in sorted_by_orig_delta[:10]:
    print(f"L{layer:<7} {res['p_original']:>12.6f} {res['delta_p_original']:>+12.6f} {res['ratio_p_original']:>12.2f}")


# %%
print("\n" + "=" * 80)
print("TOP 10 LAYERS: By P(original) decrease (ratio, smallest)")
print("=" * 80)

sorted_by_orig_ratio = sorted(layer_results.items(), key=lambda x: x[1]['ratio_p_original'])

print(f"\n{'Layer':<8} {'P(orig)':>12} {'Δ P(orig)':>12} {'× P(orig)':>12}")
print("-" * 50)
for layer, res in sorted_by_orig_ratio[:10]:
    print(f"L{layer:<7} {res['p_original']:>12.6f} {res['delta_p_original']:>+12.6f} {res['ratio_p_original']:>12.2f}")


# %%
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nBaseline (no replacement):")
print(f"  P(blue):     {baseline_p_blue:.6f}")
print(f"  P(original): {baseline_p_original:.6f}")

print(f"\nReplace ALL layers:")
print(f"  P(blue):     {all_layers_p_blue:.6f} (Δ={all_layers_p_blue - baseline_p_blue:+.6f}, ×{all_layers_p_blue / (baseline_p_blue + eps):.2f})")
print(f"  P(original): {all_layers_p_original:.6f} (Δ={all_layers_p_original - baseline_p_original:+.6f}, ×{all_layers_p_original / (baseline_p_original + eps):.2f})")

print(f"\nBest layers for increasing P(blue):")
print(f"  By delta: L{sorted_by_blue_delta[0][0]} (Δ = {sorted_by_blue_delta[0][1]['delta_p_blue']:+.6f})")
print(f"  By ratio: L{sorted_by_blue_ratio[0][0]} (× = {sorted_by_blue_ratio[0][1]['ratio_p_blue']:.2f})")

print(f"\nBest layers for decreasing P(original):")
print(f"  By delta: L{sorted_by_orig_delta[0][0]} (Δ = {sorted_by_orig_delta[0][1]['delta_p_original']:+.6f})")
print(f"  By ratio: L{sorted_by_orig_ratio[0][0]} (× = {sorted_by_orig_ratio[0][1]['ratio_p_original']:.2f})")


# %%
def replace_single_head_kv(
    state_A: List,
    state_B: List,
    layer: int,
    head: int,
    n_layer: int
) -> List:
    """
    Replace ONLY a single head's KV state in a specific layer.
    """
    new_state = [s.clone() for s in state_A]
    
    if layer < n_layer:
        att_kv_idx = layer * 3 + 1
        new_state[att_kv_idx][head] = state_B[att_kv_idx][head].clone()
    
    return new_state


def run_head_probability_analysis(
    sample_pairs: List[Dict],
    layer: int,
    head: int,
    model,
    tokenizer,
    n_layer: int,
) -> Dict:
    """
    Run experiment replacing a single head and measure probability changes.
    """
    results = {
        'p_blue': [],
        'p_original': [],
    }
    
    for pair in sample_pairs:
        tokens_A = pair['tokens_A']
        tokens_B = pair['tokens_B']
        hook_idx_A = pair['hook_idx_A']
        hook_idx_B = pair['hook_idx_B']
        color_A = pair['color_A'].lower()
        
        tokens_before_hook_A = tokens_A[:hook_idx_A]
        tokens_before_hook_B = tokens_B[:hook_idx_B]
        tokens_after_hook_A = tokens_A[hook_idx_A:]
        
        state_A = get_state_after_tokens(model, tokens_before_hook_A)
        state_A = [s.clone() for s in state_A]
        state_B = get_state_after_tokens(model, tokens_before_hook_B)
        
        modified_state = replace_single_head_kv(state_A, state_B, layer, head, n_layer)
        
        with torch.no_grad():
            out, _ = model.model.forward(tokens_after_hook_A, modified_state) if tokens_after_hook_A else model.model.forward([tokens_before_hook_A[-1]], modified_state)
            
            color_probs = get_color_probabilities_normalized(out)
            
            p_blue = color_probs.get('blue', 0.0)
            p_original = color_probs.get(color_A, 0.0)
        
        results['p_blue'].append(p_blue)
        results['p_original'].append(p_original)
    
    return results


# %%
print("\n" + "=" * 80)
print("LAYER 15: Testing each head individually (KV state only)")
print("=" * 80)

TARGET_LAYER = 15
head_results = {}

for head in range(n_head):
    results = run_head_probability_analysis(
        sample_pairs,
        layer=TARGET_LAYER,
        head=head,
        model=model,
        tokenizer=tokenizer,
        n_layer=n_layer,
    )
    
    avg_p_blue = sum(results['p_blue']) / len(results['p_blue'])
    avg_p_original = sum(results['p_original']) / len(results['p_original'])
    
    delta_p_blue = avg_p_blue - baseline_p_blue
    delta_p_original = avg_p_original - baseline_p_original
    
    eps = 1e-10
    ratio_p_blue = avg_p_blue / (baseline_p_blue + eps)
    ratio_p_original = avg_p_original / (baseline_p_original + eps)
    
    head_results[head] = {
        'p_blue': avg_p_blue,
        'p_original': avg_p_original,
        'delta_p_blue': delta_p_blue,
        'delta_p_original': delta_p_original,
        'ratio_p_blue': ratio_p_blue,
        'ratio_p_original': ratio_p_original,
    }
    
    print(f"H{head:2d}: P(blue)={avg_p_blue:.4f} (Δ={delta_p_blue:+.4f}, ×{ratio_p_blue:.2f}), "
          f"P(orig)={avg_p_original:.4f} (Δ={delta_p_original:+.4f}, ×{ratio_p_original:.2f})")


# %%
print("\n" + "=" * 80)
print(f"LAYER {TARGET_LAYER}: TOP 10 HEADS by P(blue) increase (delta)")
print("=" * 80)

sorted_heads_by_blue_delta = sorted(head_results.items(), key=lambda x: x[1]['delta_p_blue'], reverse=True)

print(f"\n{'Head':<8} {'P(blue)':>10} {'Δ P(blue)':>12} {'× P(blue)':>10}")
print("-" * 45)
for head, res in sorted_heads_by_blue_delta[:10]:
    print(f"H{head:<7} {res['p_blue']:>10.4f} {res['delta_p_blue']:>+12.4f} {res['ratio_p_blue']:>10.2f}")


# %%
print("\n" + "=" * 80)
print(f"LAYER {TARGET_LAYER}: TOP 10 HEADS by P(blue) increase (ratio)")
print("=" * 80)

sorted_heads_by_blue_ratio = sorted(head_results.items(), key=lambda x: x[1]['ratio_p_blue'], reverse=True)

print(f"\n{'Head':<8} {'P(blue)':>10} {'Δ P(blue)':>12} {'× P(blue)':>10}")
print("-" * 45)
for head, res in sorted_heads_by_blue_ratio[:10]:
    print(f"H{head:<7} {res['p_blue']:>10.4f} {res['delta_p_blue']:>+12.4f} {res['ratio_p_blue']:>10.2f}")


# %%
print("\n" + "=" * 80)
print(f"LAYER {TARGET_LAYER}: TOP 10 HEADS by P(original) decrease (delta)")
print("=" * 80)

sorted_heads_by_orig_delta = sorted(head_results.items(), key=lambda x: x[1]['delta_p_original'])

print(f"\n{'Head':<8} {'P(orig)':>10} {'Δ P(orig)':>12} {'× P(orig)':>10}")
print("-" * 45)
for head, res in sorted_heads_by_orig_delta[:10]:
    print(f"H{head:<7} {res['p_original']:>10.4f} {res['delta_p_original']:>+12.4f} {res['ratio_p_original']:>10.2f}")


# %%
print("\n" + "=" * 80)
print(f"LAYER {TARGET_LAYER} HEAD SUMMARY")
print("=" * 80)

print(f"\nBaseline: P(blue)={baseline_p_blue:.4f}, P(orig)={baseline_p_original:.4f}")

print(f"\nBest heads for increasing P(blue):")
print(f"  By delta: H{sorted_heads_by_blue_delta[0][0]} (Δ = {sorted_heads_by_blue_delta[0][1]['delta_p_blue']:+.4f})")
print(f"  By ratio: H{sorted_heads_by_blue_ratio[0][0]} (× = {sorted_heads_by_blue_ratio[0][1]['ratio_p_blue']:.2f})")

print(f"\nBest heads for decreasing P(original):")
print(f"  By delta: H{sorted_heads_by_orig_delta[0][0]} (Δ = {sorted_heads_by_orig_delta[0][1]['delta_p_original']:+.4f})")
sorted_heads_by_orig_ratio = sorted(head_results.items(), key=lambda x: x[1]['ratio_p_original'])
print(f"  By ratio: H{sorted_heads_by_orig_ratio[0][0]} (× = {sorted_heads_by_orig_ratio[0][1]['ratio_p_original']:.2f})")

# %%



