# %% [markdown]
# # State Hooking Experiment
# 
# This notebook tests whether we can manipulate the model's output by replacing
# specific layer/head states IN THE MIDDLE of processing a prompt.
# 
# We generate pairs of prompts:
# - Prompt A: Original color (not blue)
# - Prompt B: Same structure but with "blue" as the color
# 
# The hook works as follows:
# 1. Process both prompts up to the END of the entity's color sentence (after the ".")
# 2. Replace specified heads' states from A with states from B at that point
# 3. Continue processing the REST of prompt A with the modified state
# 4. Generate and observe if the output changes to "blue"

# %%
import sys
import os
import copy
import torch
from tqdm import tqdm
from typing import List, Tuple, Dict
import random

sys.path.insert(0, os.path.abspath('..'))

from models.load_rwkv import load_rwkv_model
from datasets.favorite_color_dataset import FavoriteColorDataset, FAMOUS_NAMES, DEFAULT_COLORS

# %%
print("Loading RWKV model...")
model, tokenizer = load_rwkv_model(model_name="rwkv7-g1b-1.5b-20251202-ctx8192.pth")

n_layer = model.model.n_layer
n_head = model.model.n_head
head_size = model.model.head_size

print(f"\nModel info: {n_layer} layers, {n_head} heads, head_size={head_size}")

# %% [markdown]
# ## Helper Functions

# %%
def make_prompt(context: str, entity_name: str = "Lady Gaga") -> str:
    """Create a prompt for asking about an entity's favorite color."""
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
    """
    Generate a pair of samples:
    - Sample A: fixed entity has a random color (not blue)
    - Sample B: same structure but fixed entity has "blue"
    
    Returns dict with:
    - context_A, context_B: full contexts
    - color_A, color_B: the colors
    - prefix: text BEFORE the fixed entity's sentence (same for both)
    - sentence_A, sentence_B: the fixed entity's sentence
    - suffix: text AFTER the fixed entity's sentence (same for both)
    """
    rng = random.Random(seed)
    
    # Get colors that are not blue
    non_blue_colors = [c for c in DEFAULT_COLORS if c.lower() != "blue"]
    color_A = rng.choice(non_blue_colors)
    color_B = "blue"
    
    # Generate other entities and their colors
    available_names = [n for n in FAMOUS_NAMES if n != fixed_entity]
    other_entities = rng.sample(available_names, min(n_entities - 1, len(available_names)))
    other_colors = [rng.choice(DEFAULT_COLORS) for _ in other_entities]
    
    # Build context - put fixed entity at a random position
    all_entities = other_entities + [fixed_entity]
    all_colors_A = other_colors + [color_A]
    all_colors_B = other_colors + [color_B]
    
    # Shuffle order and track fixed entity position
    indices = list(range(len(all_entities)))
    rng.shuffle(indices)
    
    # Find position of fixed entity in shuffled order
    fixed_entity_idx_in_list = len(all_entities) - 1  # It was appended last
    fixed_pos = indices.index(fixed_entity_idx_in_list)
    
    sentences_A = []
    sentences_B = []
    for idx in indices:
        sentences_A.append(f"{all_entities[idx]}'s favorite color is {all_colors_A[idx]}.")
        sentences_B.append(f"{all_entities[idx]}'s favorite color is {all_colors_B[idx]}.")
    
    # Split into prefix, fixed entity sentence, suffix
    prefix_sentences = sentences_A[:fixed_pos]  # Same for A and B (other entities)
    suffix_sentences = sentences_A[fixed_pos + 1:]  # Same for A and B (other entities)
    
    prefix = " ".join(prefix_sentences)
    if prefix:
        prefix += " "  # Add space before the fixed entity sentence
    
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
    """Get the state after processing tokens."""
    if not tokens:
        return initial_state
    with torch.no_grad():
        _, state = model.model.forward(tokens, initial_state)
    return state


def replace_state_heads(
    state_A: List,
    state_B: List,
    heads_to_replace: List[Tuple[int, int]],  # List of (layer, head) tuples
    n_layer: int
) -> List:
    """
    Replace specific heads in state_A with values from state_B.
    
    State structure: [att_x_prev, att_kv, ffn_x_prev] per layer
    att_kv has shape (n_head, head_size, head_size)
    """
    import torch
    
    # Deep copy state A
    new_state = [s.clone() for s in state_A]
    
    for layer, head in heads_to_replace:
        if layer < n_layer:
            # att_kv index for this layer
            att_kv_idx = layer * 3 + 1
            # Replace the head
            new_state[att_kv_idx][head] = state_B[att_kv_idx][head].clone()
    
    return new_state


# %% [markdown]
# ## Generate Sample Pairs

# %%
def find_hook_token_idx(full_tokens: List[int], hook_text: str, tokenizer) -> int:
    """
    Find the token index where hook_text ends in the full token sequence.
    This ensures we split tokens correctly without tokenization boundary issues.
    """
    hook_tokens = tokenizer.encode(hook_text)
    
    # Find where hook_tokens ends in full_tokens
    # Due to tokenization, we need to find the best match
    hook_len = len(hook_tokens)
    
    # Try to match the hook tokens at the start
    # They should match exactly or very closely
    for i in range(min(hook_len + 5, len(full_tokens)), hook_len - 5, -1):
        if full_tokens[:i] == hook_tokens[:i] or i <= hook_len:
            # Decode and check character length
            decoded_prefix = tokenizer.decode(full_tokens[:i])
            if len(decoded_prefix) >= len(hook_text) - 5:  # Allow small tolerance
                return i
    
    # Fallback: use hook_tokens length
    return len(hook_tokens)


# %%
N_ENTITIES = 100
N_SAMPLES = 5
FIXED_ENTITY = "Lady Gaga"

print(f"Generating {N_SAMPLES} sample pairs with n_entities={N_ENTITIES}...")
print(f"Fixed entity: {FIXED_ENTITY}")

sample_pairs = []
for i in range(N_SAMPLES):
    pair_data = generate_sample_pair(
        n_entities=N_ENTITIES,
        fixed_entity=FIXED_ENTITY,
        seed=1000 + i
    )
    
    # Build full prompts
    prompt_A = make_prompt(pair_data['context_A'], FIXED_ENTITY)
    prompt_B = make_prompt(pair_data['context_B'], FIXED_ENTITY)
    
    # Hook point: AFTER the full sentence INCLUDING the period
    # This matches the red line (info_idx) in the probe visualization
    # state_A has seen "black.", state_B has seen "blue."
    
    prompt_prefix = f"Given the following context, let's answer the question below.\n\nContext:\n"
    
    # sentence_A = "Lady Gaga's favorite color is black."
    # Hook at end of sentence (including period) to match probe's info_idx
    hook_text_A = prompt_prefix + pair_data['prefix'] + pair_data['sentence_A']
    hook_text_B = prompt_prefix + pair_data['prefix'] + pair_data['sentence_B']
    
    # Tokenize the FULL prompts to avoid boundary issues
    tokens_A = tokenizer.encode(prompt_A)
    tokens_B = tokenizer.encode(prompt_B)
    
    # Find hook indices - where to split tokens
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

print(f"\nExample pair:")
p = sample_pairs[0]
print(f"Prompt A tokens: {len(p['tokens_A'])}, hook at idx {p['hook_idx_A']}")
print(f"Prompt B tokens: {len(p['tokens_B'])}, hook at idx {p['hook_idx_B']}")
print(f"Hook text A decoded: ...{tokenizer.decode(p['tokens_A'][:p['hook_idx_A']])[-60:]}")
print(f"Remaining A decoded: {tokenizer.decode(p['tokens_A'][p['hook_idx_A']:])[:60]}...")
print(f"Color A: {p['color_A']}, Color B: {p['color_B']}")

# %% [markdown]
# ## Verify Baseline Works

# %%
def decode_tokens_safe(tokenizer, token_ids: List[int]) -> str:
    """Decode tokens, filtering out problematic token ids (like 0)."""
    # Filter out token id 0 which causes decode issues
    filtered = [t for t in token_ids if t != 0]
    if not filtered:
        return ""
    return tokenizer.decode(filtered)

def generate_with_state(model, state, token_count: int = 5) -> Tuple[List[int], List]:
    """Generate tokens starting from a given state."""
    gen_tokens = []
    current_state = state
    out = None
    
    # Need to get initial logits - do a forward pass with the state
    # But we need a token to start. We'll use the last generated token.
    # Actually, the state already contains the context, so we need to get logits somehow.
    # In RWKV, the forward pass returns logits AND updates state.
    # The state alone doesn't give us logits - we need to process at least one token.
    
    return gen_tokens, current_state

def generate_after_context(model, context_tokens: List[int], state, token_count: int = 5) -> List[int]:
    """
    Process context tokens with given state, then generate new tokens.
    Returns the generated token ids.
    """
    gen_tokens = []
    
    with torch.no_grad():
        # Process context tokens to get final logits and state
        if context_tokens:
            out, current_state = model.model.forward(context_tokens, state)
        else:
            # No context tokens - need to get logits from state somehow
            # This shouldn't happen in our use case
            raise ValueError("Need at least one context token to generate from")
        
        # Generate new tokens
        for _ in range(token_count):
            next_tok = out.argmax().item()
            gen_tokens.append(next_tok)
            # Stop if we hit a period or newline (answer complete)
            if next_tok in [47, 11, 10]:  # '.', '\n', etc.
                break
            out, current_state = model.model.forward([next_tok], current_state)
    
    return gen_tokens

# Quick sanity check: compare model.generate() vs manual forward()
print("=== Comparing model.generate() vs manual forward() ===\n")
test_pair = sample_pairs[0]
test_prompt = test_pair['prompt_A']

# Method 1: Use model.generate() (works in 02_test_favorite_color_dataset.py)
print("Method 1: model.generate()")
output_pipeline = []
def callback(s):
    output_pipeline.append(s)
model.generate(test_prompt, token_count=5, temperature=0.0, top_p=0.0, callback=callback)
pipeline_result = ''.join(output_pipeline).strip()
print(f"  Result: '{pipeline_result}'")
print(f"  Expected color: {test_pair['color_A']}")

# Method 2: Manual forward() with full token sequence
print("\nMethod 2: Manual model.model.forward()")
test_tokens = test_pair['tokens_A']

with torch.no_grad():
    out, state = model.model.forward(test_tokens, None)
    gen_tokens = []
    for i in range(5):
        next_tok = out.argmax().item()
        gen_tokens.append(next_tok)
        decoded = tokenizer.decode([next_tok]) if next_tok != 0 else '<0>'
        print(f"    Token {i}: id={next_tok}, decoded='{decoded}'")
        if next_tok in [47, 11]:  # Stop at period
            break
        out, state = model.model.forward([next_tok], state)

result = decode_tokens_safe(tokenizer, gen_tokens)
print(f"  Generated: '{result}'")
print(f"  Contains expected color: {test_pair['color_A'].lower() in result.lower()}")

# Method 3: Verify split processing gives same result
print("\nMethod 3: Split processing (hook simulation)")
hook_idx = test_pair['hook_idx_A']
tokens_before = test_tokens[:hook_idx]
tokens_after = test_tokens[hook_idx:]

with torch.no_grad():
    # Process up to hook point
    _, state_at_hook = model.model.forward(tokens_before, None)
    # Continue with remaining tokens
    out, final_state = model.model.forward(tokens_after, state_at_hook)
    
    gen_tokens_split = []
    for i in range(5):
        next_tok = out.argmax().item()
        gen_tokens_split.append(next_tok)
        if next_tok in [47, 11]:
            break
        out, final_state = model.model.forward([next_tok], final_state)

result_split = decode_tokens_safe(tokenizer, gen_tokens_split)
print(f"  Generated: '{result_split}'")
print(f"  Matches full processing: {gen_tokens == gen_tokens_split}")

def run_hook_experiment(
    sample_pairs: List[Dict],
    heads_to_replace: List[Tuple[int, int]],
    model,
    tokenizer,
    n_layer: int,
    token_count: int = 5,
) -> Dict:
    """
    Run the hooking experiment with mid-sequence state replacement:
    1. Process tokens up to hook point for both A and B
    2. Replace specified heads in state_A with values from state_B
    3. Continue processing remaining tokens of A with modified state
    4. Generate and compare output to original color, blue, or neither
    """
    results = {
        'original': 0,
        'blue': 0,
        'neither': 0,
        'details': []
    }
    
    for pair in tqdm(sample_pairs, desc="Processing pairs"):
        tokens_A = pair['tokens_A']
        tokens_B = pair['tokens_B']
        hook_idx_A = pair['hook_idx_A']
        hook_idx_B = pair['hook_idx_B']
        color_A = pair['color_A']
        color_B = pair['color_B']  # Should be "blue"
        
        # Split tokens at hook point
        tokens_before_hook_A = tokens_A[:hook_idx_A]
        tokens_before_hook_B = tokens_B[:hook_idx_B]
        tokens_after_hook_A = tokens_A[hook_idx_A:]
        
        # Get states at hook point
        state_A = get_state_after_tokens(model, tokens_before_hook_A)
        state_B = get_state_after_tokens(model, tokens_before_hook_B)
        
        # Replace specified heads in state_A with values from state_B
        if heads_to_replace:
            modified_state = replace_state_heads(state_A, state_B, heads_to_replace, n_layer)
        else:
            modified_state = state_A  # No replacement (baseline)
        
        # Continue processing remaining tokens with modified state, then generate
        with torch.no_grad():
            current_state = modified_state
            
            # Process remaining tokens from prompt A with (possibly) modified state
            if tokens_after_hook_A:
                out, current_state = model.model.forward(tokens_after_hook_A, current_state)
            else:
                # Edge case: hook is at end, need logits from last token before hook
                out, current_state = model.model.forward([tokens_before_hook_A[-1]], 
                    get_state_after_tokens(model, tokens_before_hook_A[:-1]) if len(tokens_before_hook_A) > 1 else None)
            
            # Generate tokens (stop at period)
            output_tokens = []
            for _ in range(token_count):
                next_token = out.argmax().item()
                output_tokens.append(next_token)
                if next_token in [47, 11]:  # Stop at '.' or newline
                    break
                out, current_state = model.model.forward([next_token], current_state)
        
        # Decode safely (filter out token 0)
        response = decode_tokens_safe(tokenizer, output_tokens)
        response_lower = response.strip().lower()
        
        # Classify result
        if color_A.lower() in response_lower:
            results['original'] += 1
            classification = 'original'
        elif color_B.lower() in response_lower:  # "blue"
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

print("=" * 80)
print("Sanity Check: Replace ALL heads in layers 1-15")
print("=" * 80)

# Generate list of all (layer, head) pairs for layers 1-15
heads_to_replace_sanity = []
for layer in range(1, min(16, n_layer)):  # layers 1-15 (0-indexed: 1-15)
    for head in range(n_head):
        heads_to_replace_sanity.append((layer, head))

print(f"Replacing {len(heads_to_replace_sanity)} heads total")
print(f"Layers: 1-{min(15, n_layer-1)}, All {n_head} heads per layer")

results_sanity = run_hook_experiment(
    sample_pairs,
    heads_to_replace_sanity,
    model,
    tokenizer,
    n_layer,
    token_count=5
)

n_total = len(sample_pairs)
print(f"\n{'='*60}")
print(f"Results (All heads in layers 1-15):")
print(f"{'='*60}")
print(f"Original color: {results_sanity['original']:3d}/{n_total} ({100*results_sanity['original']/n_total:.1f}%)")
print(f"Blue:           {results_sanity['blue']:3d}/{n_total} ({100*results_sanity['blue']/n_total:.1f}%)")
print(f"Neither:        {results_sanity['neither']:3d}/{n_total} ({100*results_sanity['neither']/n_total:.1f}%)")

print("\nExample outputs:")
for i, detail in enumerate(results_sanity['details'][:10]):
    print(f"  {i}: color_A={detail['color_A']:10s}, response='{detail['response'][:30]}...', class={detail['classification']}")

print("\n" + "=" * 80)
print("Baseline: No head replacement (original behavior)")
print("=" * 80)

results_baseline = run_hook_experiment(
    sample_pairs,
    [],  # No heads to replace
    model,
    tokenizer,
    n_layer,
    token_count=5
)

print(f"\n{'='*60}")
print(f"Results (No replacement - baseline):")
print(f"{'='*60}")
print(f"Original color: {results_baseline['original']:3d}/{n_total} ({100*results_baseline['original']/n_total:.1f}%)")
print(f"Blue:           {results_baseline['blue']:3d}/{n_total} ({100*results_baseline['blue']/n_total:.1f}%)")
print(f"Neither:        {results_baseline['neither']:3d}/{n_total} ({100*results_baseline['neither']/n_total:.1f}%)")

print("\n" + "=" * 80)
print("Full Replacement: ALL heads in ALL layers")
print("=" * 80)

heads_to_replace_all = []
for layer in range(n_layer):
    for head in range(n_head):
        heads_to_replace_all.append((layer, head))

print(f"Replacing {len(heads_to_replace_all)} heads total")

results_all = run_hook_experiment(
    sample_pairs,
    heads_to_replace_all,
    model,
    tokenizer,
    n_layer,
    token_count=5
)

print(f"\n{'='*60}")
print(f"Results (All heads in all layers):")
print(f"{'='*60}")
print(f"Original color: {results_all['original']:3d}/{n_total} ({100*results_all['original']/n_total:.1f}%)")
print(f"Blue:           {results_all['blue']:3d}/{n_total} ({100*results_all['blue']/n_total:.1f}%)")
print(f"Neither:        {results_all['neither']:3d}/{n_total} ({100*results_all['neither']/n_total:.1f}%)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"{'Condition':<30} {'Original':>10} {'Blue':>10} {'Neither':>10}")
print("-" * 60)
print(f"{'Baseline (no replacement)':<30} {results_baseline['original']:>10} {results_baseline['blue']:>10} {results_baseline['neither']:>10}")
print(f"{'Layers 1-15, all heads':<30} {results_sanity['original']:>10} {results_sanity['blue']:>10} {results_sanity['neither']:>10}")
print(f"{'All layers, all heads':<30} {results_all['original']:>10} {results_all['blue']:>10} {results_all['neither']:>10}")

# %% [markdown]
# ## Test Best Probe Heads
# 
# Test the specific heads that performed best in the probe experiments.

# %%
# Best performing heads from probe training (layer, head, val_acc)
# BEST_PROBE_HEADS = [
#     (15, 11, 0.910),  # Best
#     (15, 27, 0.778),
#     (14, 18, 0.672),
#     (14, 23, 0.563),
#     (19, 8,  0.403),
#     (15, 13, 0.377),
#     (20, 4,  0.345),
#     (15, 15, 0.345),
#     (17, 18, 0.317),
#     (20, 10, 0.297),
# ]

BEST_PROBE_HEADS = [
    (15, 11, 0.910),  # Best
    (14, 23, 0.778),
    (15, 2, 0.672),
    (15, 27, 0.563),
    (15, 13,  0.403),
    (14, 18, 0.377),
    (15, 8,  0.345),
    (12, 24, 0.345),
    (15, 1, 0.317),
    (9, 18, 0.297),
]

print("=" * 80)
print("Testing Best Probe Heads")
print("=" * 80)

# Test each head individually
print("\n--- Individual heads ---")
individual_results = {}
for layer, head, probe_acc in BEST_PROBE_HEADS:
    heads_to_replace = [(layer, head)]
    results = run_hook_experiment(
        sample_pairs, heads_to_replace, model, tokenizer, n_layer, token_count=5
    )
    individual_results[(layer, head)] = results
    print(f"L{layer:2d} H{head:2d} (probe={probe_acc:.2f}): orig={results['original']:2d}, blue={results['blue']:2d}, neither={results['neither']:2d}")

# Test top 3 heads together
print("\n--- Top 3 heads together ---")
top3_heads = [(l, h) for l, h, _ in BEST_PROBE_HEADS[:3]]
results_top3 = run_hook_experiment(
    sample_pairs, top3_heads, model, tokenizer, n_layer, token_count=5
)
print(f"Top 3 heads: orig={results_top3['original']}, blue={results_top3['blue']}, neither={results_top3['neither']}")

# Test top 5 heads together
print("\n--- Top 5 heads together ---")
top5_heads = [(l, h) for l, h, _ in BEST_PROBE_HEADS[:5]]
results_top5 = run_hook_experiment(
    sample_pairs, top5_heads, model, tokenizer, n_layer, token_count=5
)
print(f"Top 5 heads: orig={results_top5['original']}, blue={results_top5['blue']}, neither={results_top5['neither']}")

# Test all 10 best heads together
print("\n--- All 10 best heads together ---")
all_best_heads = [(l, h) for l, h, _ in BEST_PROBE_HEADS]
results_best10 = run_hook_experiment(
    sample_pairs, all_best_heads, model, tokenizer, n_layer, token_count=5
)
print(f"All 10 best: orig={results_best10['original']}, blue={results_best10['blue']}, neither={results_best10['neither']}")

# %%
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"{'Condition':<35} {'Original':>10} {'Blue':>10} {'Neither':>10}")
print("-" * 65)
print(f"{'Baseline (no replacement)':<35} {results_baseline['original']:>10} {results_baseline['blue']:>10} {results_baseline['neither']:>10}")
print(f"{'Best head (L15 H11, probe=0.91)':<35} {individual_results[(15,11)]['original']:>10} {individual_results[(15,11)]['blue']:>10} {individual_results[(15,11)]['neither']:>10}")
print(f"{'Top 3 best probe heads':<35} {results_top3['original']:>10} {results_top3['blue']:>10} {results_top3['neither']:>10}")
print(f"{'Top 5 best probe heads':<35} {results_top5['original']:>10} {results_top5['blue']:>10} {results_top5['neither']:>10}")
print(f"{'All 10 best probe heads':<35} {results_best10['original']:>10} {results_best10['blue']:>10} {results_best10['neither']:>10}")
print(f"{'Layers 1-15, all heads':<35} {results_sanity['original']:>10} {results_sanity['blue']:>10} {results_sanity['neither']:>10}")
print(f"{'All layers, all heads':<35} {results_all['original']:>10} {results_all['blue']:>10} {results_all['neither']:>10}")

# %%

