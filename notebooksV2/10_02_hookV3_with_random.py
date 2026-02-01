# %% [markdown]
# # State Hooking Experiment V3
#
# Tests layer-level state replacement to see which layers are most important
# for changing the model's output from original color to "blue".
#
# Hook point options:
# - Option A: Right after the color token (before the ".")
# - Option B: Right after the full sentence (after the ".")

# %%
import sys
import os
import torch
from typing import List, Tuple, Dict
import random

sys.path.insert(0, os.path.abspath(".."))

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
def make_prompt(context: str, entity_name: str = "Lady Gaga") -> str:
    return (
        f"Given the following context, let's answer the question below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: What is {entity_name}'s favorite color?\n\n"
        f"Answer: {entity_name}'s favorite color is"
    )


def generate_sample_pair(n_entities: int = 30, fixed_entity: str = "Lady Gaga", seed: int = 42) -> Dict:
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
    suffix_sentences = sentences_A[fixed_pos + 1 :]

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
        "context_A": context_A,
        "context_B": context_B,
        "color_A": color_A,
        "color_B": color_B,
        "prefix": prefix,
        "sentence_A": sentence_A,
        "sentence_B": sentence_B,
        "suffix": suffix,
        "fixed_pos": fixed_pos,
    }


def get_state_after_tokens(model, tokens: List[int], initial_state=None):
    if not tokens:
        return initial_state
    with torch.no_grad():
        _, state = model.model.forward(tokens, initial_state)
    return state


def replace_full_layers(state_A: List, state_B: List, layers: List[int], n_layer: int) -> List:
    """
    Replace ALL state components for given layers (att_x_prev, att_kv, ffn_x_prev).
    State structure: [att_x_prev, att_kv, ffn_x_prev] per layer
    """
    new_state = [s.clone() for s in state_A]

    for layer in layers:
        if layer < n_layer:
            att_x_prev_idx = layer * 3 + 0
            att_kv_idx = layer * 3 + 1
            ffn_x_prev_idx = layer * 3 + 2

            new_state[att_x_prev_idx] = state_B[att_x_prev_idx].clone()
            new_state[att_kv_idx] = state_B[att_kv_idx].clone()
            new_state[ffn_x_prev_idx] = state_B[ffn_x_prev_idx].clone()

    return new_state


def replace_specific_heads(
    state_A: List, state_B: List, heads_to_replace: List[Tuple[int, int]], n_layer: int  # List of (layer, head) tuples
) -> List:
    """
    Replace specific heads' att_kv states.
    Also replaces att_x_prev and ffn_x_prev for layers that have at least one head replaced.
    """
    new_state = [s.clone() for s in state_A]

    # Group heads by layer
    layers_with_heads = {}
    for layer, head in heads_to_replace:
        if layer not in layers_with_heads:
            layers_with_heads[layer] = []
        layers_with_heads[layer].append(head)

    for layer, heads in layers_with_heads.items():
        if layer < n_layer:
            # Replace att_x_prev and ffn_x_prev for the entire layer
            att_x_prev_idx = layer * 3 + 0
            ffn_x_prev_idx = layer * 3 + 2
            new_state[att_x_prev_idx] = state_B[att_x_prev_idx].clone()
            new_state[ffn_x_prev_idx] = state_B[ffn_x_prev_idx].clone()

            # Replace specific heads in att_kv
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
N_ENTITIES = 30
N_SAMPLES = 50
FIXED_ENTITY = "Lady Gaga"


def generate_sample_pairs_with_hook(hook_after_period: bool):
    """Generate sample pairs with specified hook position."""
    pairs = []
    for i in range(N_SAMPLES):
        pair_data = generate_sample_pair(n_entities=N_ENTITIES, fixed_entity=FIXED_ENTITY, seed=1000 + i)

        prompt_A = make_prompt(pair_data["context_A"], FIXED_ENTITY)
        prompt_B = make_prompt(pair_data["context_B"], FIXED_ENTITY)

        prompt_prefix = f"Given the following context, let's answer the question below.\n\nContext:\n"
        sentence_prefix = f"{FIXED_ENTITY}'s favorite color is "

        if hook_after_period:
            # Hook after full sentence including period
            hook_text_A = prompt_prefix + pair_data["prefix"] + pair_data["sentence_A"]
            hook_text_B = prompt_prefix + pair_data["prefix"] + pair_data["sentence_B"]
        else:
            # Hook after color token, before period
            hook_text_A = prompt_prefix + pair_data["prefix"] + sentence_prefix + pair_data["color_A"]
            hook_text_B = prompt_prefix + pair_data["prefix"] + sentence_prefix + pair_data["color_B"]

        tokens_A = tokenizer.encode(prompt_A)
        tokens_B = tokenizer.encode(prompt_B)

        hook_idx_A = find_hook_token_idx(tokens_A, hook_text_A, tokenizer)
        hook_idx_B = find_hook_token_idx(tokens_B, hook_text_B, tokenizer)

        pairs.append(
            {
                "prompt_A": prompt_A,
                "prompt_B": prompt_B,
                "tokens_A": tokens_A,
                "tokens_B": tokens_B,
                "hook_idx_A": hook_idx_A,
                "hook_idx_B": hook_idx_B,
                "color_A": pair_data["color_A"],
                "color_B": pair_data["color_B"],
            }
        )
    return pairs


# Generate both versions
print(f"Generating {N_SAMPLES} sample pairs with n_entities={N_ENTITIES}...")
print(f"Fixed entity: {FIXED_ENTITY}")

sample_pairs_after_period = generate_sample_pairs_with_hook(hook_after_period=True)
sample_pairs_after_color = generate_sample_pairs_with_hook(hook_after_period=False)

# Generate hook AFTER CONTEXT (before "Question:")
sample_pairs_after_context = []
for i in range(N_SAMPLES):
    pair_data = generate_sample_pair(n_entities=N_ENTITIES, fixed_entity=FIXED_ENTITY, seed=1000 + i)

    prompt_A = make_prompt(pair_data["context_A"], FIXED_ENTITY)
    prompt_B = make_prompt(pair_data["context_B"], FIXED_ENTITY)

    # Hook right after context, before "Question:"
    hook_text_A = (
        f"Given the following context, let's answer the question below.\n\nContext:\n{pair_data['context_A']}\n\n"
    )
    hook_text_B = (
        f"Given the following context, let's answer the question below.\n\nContext:\n{pair_data['context_B']}\n\n"
    )

    tokens_A = tokenizer.encode(prompt_A)
    tokens_B = tokenizer.encode(prompt_B)

    hook_idx_A = find_hook_token_idx(tokens_A, hook_text_A, tokenizer)
    hook_idx_B = find_hook_token_idx(tokens_B, hook_text_B, tokenizer)

    sample_pairs_after_context.append(
        {
            "prompt_A": prompt_A,
            "prompt_B": prompt_B,
            "tokens_A": tokens_A,
            "tokens_B": tokens_B,
            "hook_idx_A": hook_idx_A,
            "hook_idx_B": hook_idx_B,
            "color_A": pair_data["color_A"],
            "color_B": pair_data["color_B"],
        }
    )

# Generate hook BEFORE the color info (sanity check - should keep original)
sample_pairs_before_color = []
for i in range(N_SAMPLES):
    pair_data = generate_sample_pair(n_entities=N_ENTITIES, fixed_entity=FIXED_ENTITY, seed=1000 + i)

    prompt_A = make_prompt(pair_data["context_A"], FIXED_ENTITY)
    prompt_B = make_prompt(pair_data["context_B"], FIXED_ENTITY)

    prompt_prefix = f"Given the following context, let's answer the question below.\n\nContext:\n"

    # Hook BEFORE the sentence with the color (just after prefix, before the fixed entity's sentence)
    hook_text_A = prompt_prefix + pair_data["prefix"]
    hook_text_B = prompt_prefix + pair_data["prefix"]

    tokens_A = tokenizer.encode(prompt_A)
    tokens_B = tokenizer.encode(prompt_B)

    hook_idx_A = find_hook_token_idx(tokens_A, hook_text_A, tokenizer)
    hook_idx_B = find_hook_token_idx(tokens_B, hook_text_B, tokenizer)

    sample_pairs_before_color.append(
        {
            "prompt_A": prompt_A,
            "prompt_B": prompt_B,
            "tokens_A": tokens_A,
            "tokens_B": tokens_B,
            "hook_idx_A": hook_idx_A,
            "hook_idx_B": hook_idx_B,
            "color_A": pair_data["color_A"],
            "color_B": pair_data["color_B"],
        }
    )

print(f"\nHook AFTER PERIOD example:")
p = sample_pairs_after_period[0]
print(f"  Hook at idx {p['hook_idx_A']}: ...{tokenizer.decode(p['tokens_A'][:p['hook_idx_A']])[-40:]}")
print(f"  Remaining: {tokenizer.decode(p['tokens_A'][p['hook_idx_A']:])[:40]}...")

print(f"\nHook AFTER COLOR (before period) example:")
p = sample_pairs_after_color[0]
print(f"  Hook at idx {p['hook_idx_A']}: ...{tokenizer.decode(p['tokens_A'][:p['hook_idx_A']])[-40:]}")
print(f"  Remaining: {tokenizer.decode(p['tokens_A'][p['hook_idx_A']:])[:40]}...")

print(f"\nHook AFTER CONTEXT (before Question) example:")
p = sample_pairs_after_context[0]
print(f"  Hook at idx {p['hook_idx_A']}: ...{tokenizer.decode(p['tokens_A'][:p['hook_idx_A']])[-40:]}")
print(f"  Remaining: {tokenizer.decode(p['tokens_A'][p['hook_idx_A']:])[:40]}...")

print(f"\nHook BEFORE COLOR (sanity - should keep original) example:")
p = sample_pairs_before_color[0]
print(f"  Hook at idx {p['hook_idx_A']}: ...{tokenizer.decode(p['tokens_A'][:p['hook_idx_A']])[-40:]}")
print(f"  Remaining: {tokenizer.decode(p['tokens_A'][p['hook_idx_A']:])[:40]}...")


# %%
def run_head_hook_experiment(
    sample_pairs: List[Dict],
    heads_to_replace: List[Tuple[int, int]],  # List of (layer, head) tuples
    model,
    tokenizer,
    n_layer: int,
    token_count: int = 5,
) -> Dict:
    """Run experiment replacing specific heads."""
    results = {"original": 0, "blue": 0, "neither": 0, "details": []}

    for pair in sample_pairs:
        tokens_A = pair["tokens_A"]
        tokens_B = pair["tokens_B"]
        hook_idx_A = pair["hook_idx_A"]
        hook_idx_B = pair["hook_idx_B"]
        color_A = pair["color_A"]
        color_B = pair["color_B"]

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
                out, current_state = model.model.forward(
                    [tokens_before_hook_A[-1]],
                    get_state_after_tokens(model, tokens_before_hook_A[:-1]) if len(tokens_before_hook_A) > 1 else None,
                )

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
            results["original"] += 1
            classification = "original"
        elif color_B.lower() in response_lower:
            results["blue"] += 1
            classification = "blue"
        else:
            results["neither"] += 1
            classification = "neither"

        results["details"].append(
            {"color_A": color_A, "color_B": color_B, "response": response, "classification": classification}
        )

    return results


def run_layer_hook_experiment(
    sample_pairs: List[Dict],
    layers_to_replace: List[int],  # Empty list = no replacement (baseline)
    model,
    tokenizer,
    n_layer: int,
    token_count: int = 5,
) -> Dict:
    results = {"original": 0, "blue": 0, "neither": 0, "details": []}

    for pair in sample_pairs:
        tokens_A = pair["tokens_A"]
        tokens_B = pair["tokens_B"]
        hook_idx_A = pair["hook_idx_A"]
        hook_idx_B = pair["hook_idx_B"]
        color_A = pair["color_A"]
        color_B = pair["color_B"]

        tokens_before_hook_A = tokens_A[:hook_idx_A]
        tokens_before_hook_B = tokens_B[:hook_idx_B]
        tokens_after_hook_A = tokens_A[hook_idx_A:]

        state_A = get_state_after_tokens(model, tokens_before_hook_A)
        state_A = [s.clone() for s in state_A]  # Clone immediately to avoid buffer reuse
        state_B = get_state_after_tokens(model, tokens_before_hook_B)

        if layers_to_replace:
            modified_state = replace_full_layers(state_A, state_B, layers_to_replace, n_layer)
        else:
            modified_state = state_A

        with torch.no_grad():
            current_state = modified_state

            if tokens_after_hook_A:
                out, current_state = model.model.forward(tokens_after_hook_A, current_state)
            else:
                out, current_state = model.model.forward(
                    [tokens_before_hook_A[-1]],
                    get_state_after_tokens(model, tokens_before_hook_A[:-1]) if len(tokens_before_hook_A) > 1 else None,
                )

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
            results["original"] += 1
            classification = "original"
        elif color_B.lower() in response_lower:
            results["blue"] += 1
            classification = "blue"
        else:
            results["neither"] += 1
            classification = "neither"

        results["details"].append(
            {"color_A": color_A, "color_B": color_B, "response": response, "classification": classification}
        )

    return results


# %%
def run_full_experiment(sample_pairs, hook_name: str):
    """Run baseline and all-layers sanity check for a given hook position."""
    print("=" * 80)
    print(f"HOOK POSITION: {hook_name}")
    print("=" * 80)

    n_total = len(sample_pairs)

    # Baseline
    print("\n--- Baseline: No replacement ---")
    results_baseline = run_layer_hook_experiment(
        sample_pairs, layers_to_replace=[], model=model, tokenizer=tokenizer, n_layer=n_layer, token_count=5
    )
    print(
        f"Original: {results_baseline['original']:3d}/{n_total}, Blue: {results_baseline['blue']:3d}/{n_total}, Neither: {results_baseline['neither']:3d}/{n_total}"
    )

    # Replace ALL layers (sanity check)
    print("\n--- Replace ALL layers (sanity check) ---")
    results_all = run_layer_hook_experiment(
        sample_pairs,
        layers_to_replace=list(range(n_layer)),
        model=model,
        tokenizer=tokenizer,
        n_layer=n_layer,
        token_count=5,
    )
    print(
        f"Original: {results_all['original']:3d}/{n_total}, Blue: {results_all['blue']:3d}/{n_total}, Neither: {results_all['neither']:3d}/{n_total}"
    )

    print("\nSample responses (replace ALL):")
    for i, d in enumerate(results_all["details"][:5]):
        print(f"  {i}: color_A={d['color_A']}, response='{d['response']}', class={d['classification']}")

    return results_baseline, results_all


# %%
print("\n" + "=" * 80)
print("SANITY CHECK: HOOK BEFORE COLOR INFO")
print("=" * 80 + "\n")
baseline_before, all_before = run_full_experiment(sample_pairs_before_color, "BEFORE COLOR (should keep original)")

# %%
print("\n" + "=" * 80)
print("TESTING HOOK AFTER COLOR (before period/dot)")
print("=" * 80 + "\n")
baseline_color, all_color = run_full_experiment(sample_pairs_after_color, "AFTER COLOR")

# %%
print("\n" + "=" * 80)
print("TESTING HOOK AFTER PERIOD/DOT (end of sentence)")
print("=" * 80 + "\n")
baseline_period, all_period = run_full_experiment(sample_pairs_after_period, "AFTER PERIOD")


# %%
print("\n" + "=" * 80)
print("Testing each layer individually - HOOK AFTER COLOR (before dot)")
print("=" * 80)

layer_results_color = {}
for layer in range(n_layer):
    results = run_layer_hook_experiment(
        sample_pairs_after_color,
        layers_to_replace=[layer],
        model=model,
        tokenizer=tokenizer,
        n_layer=n_layer,
        token_count=5,
    )
    layer_results_color[layer] = results
    print(f"L{layer:2d}: orig={results['original']:2d}, blue={results['blue']:2d}, neither={results['neither']:2d}")

# %%
print("\n" + "=" * 80)
print("Testing each layer individually - HOOK AFTER PERIOD/DOT")
print("=" * 80)

layer_results_period = {}
for layer in range(n_layer):
    results = run_layer_hook_experiment(
        sample_pairs_after_period,
        layers_to_replace=[layer],
        model=model,
        tokenizer=tokenizer,
        n_layer=n_layer,
        token_count=5,
    )
    layer_results_period[layer] = results
    print(f"L{layer:2d}: orig={results['original']:2d}, blue={results['blue']:2d}, neither={results['neither']:2d}")

# %%
print("\n" + "=" * 80)
print("RESULTS: Top 10 layers by blue count - HOOK AFTER COLOR (before dot)")
print("=" * 80)

sorted_layers_color = sorted(layer_results_color.items(), key=lambda x: x[1]["blue"], reverse=True)

print(f"\n{'Layer':<8} {'Blue':>8} {'Original':>10} {'Neither':>10}")
print("-" * 40)
for layer, results in sorted_layers_color[:10]:
    print(f"L{layer:<7} {results['blue']:>8} {results['original']:>10} {results['neither']:>10}")

# %%
print("\n" + "=" * 80)
print("RESULTS: Top 10 layers by blue count - HOOK AFTER PERIOD/DOT")
print("=" * 80)

sorted_layers_period = sorted(layer_results_period.items(), key=lambda x: x[1]["blue"], reverse=True)

print(f"\n{'Layer':<8} {'Blue':>8} {'Original':>10} {'Neither':>10}")
print("-" * 40)
for layer, results in sorted_layers_period[:10]:
    print(f"L{layer:<7} {results['blue']:>8} {results['original']:>10} {results['neither']:>10}")

# %%
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"\nSANITY CHECK - HOOK BEFORE COLOR (should NOT change to blue):")
print(
    f"  Baseline: orig={baseline_before['original']}, blue={baseline_before['blue']}, neither={baseline_before['neither']}"
)
print(f"  Replace ALL: orig={all_before['original']}, blue={all_before['blue']}, neither={all_before['neither']}")
print(f"  ✓ Working correctly!" if all_before["blue"] < 5 else "  ✗ UNEXPECTED: Too many blue outputs!")

print(f"\nHOOK AFTER COLOR (before dot):")
print(
    f"  Baseline: orig={baseline_color['original']}, blue={baseline_color['blue']}, neither={baseline_color['neither']}"
)
print(f"  Replace ALL: orig={all_color['original']}, blue={all_color['blue']}, neither={all_color['neither']}")
print(f"  Best layer: L{sorted_layers_color[0][0]} with {sorted_layers_color[0][1]['blue']} blue outputs")

print(f"\nHOOK AFTER PERIOD/DOT:")
print(
    f"  Baseline: orig={baseline_period['original']}, blue={baseline_period['blue']}, neither={baseline_period['neither']}"
)
print(f"  Replace ALL: orig={all_period['original']}, blue={all_period['blue']}, neither={all_period['neither']}")
print(f"  Best layer: L{sorted_layers_period[0][0]} with {sorted_layers_period[0][1]['blue']} blue outputs")

# %%
print("\n" + "=" * 80)
print("GREEDY HEAD PRUNING: Find minimal set of heads")
print("=" * 80)

# Start with layers that worked well - use after_color hook
INITIAL_LAYERS = [15, 16, 19, 20]  # Modify this based on your results
TEST_SAMPLES = sample_pairs_after_color[:1]  # Use subset for faster testing

print(f"\nStarting with layers: {INITIAL_LAYERS}")
print(f"Using {len(TEST_SAMPLES)} samples for pruning")

# Generate initial set of all heads in these layers
current_heads = []
for layer in INITIAL_LAYERS:
    for head in range(n_head):
        current_heads.append((layer, head))

print(f"Initial head count: {len(current_heads)} heads")

# Test baseline with all heads
results = run_head_hook_experiment(
    TEST_SAMPLES, heads_to_replace=current_heads, model=model, tokenizer=tokenizer, n_layer=n_layer, token_count=5
)
print(f"Starting performance: orig={results['original']}, blue={results['blue']}, neither={results['neither']}")

# Greedy pruning: remove heads one at a time if they don't hurt performance
removed_heads = []
iteration = 0
TARGET_BLUE = len(TEST_SAMPLES)  # All samples should be blue

while True:
    iteration += 1
    print(f"\n--- Iteration {iteration} ---")
    print(f"Current heads: {len(current_heads)}")

    best_head_to_remove = None

    # Try removing each head
    for i, head_to_test in enumerate(current_heads):
        # Test without this head
        test_heads = [h for h in current_heads if h != head_to_test]

        results_without = run_head_hook_experiment(
            TEST_SAMPLES, heads_to_replace=test_heads, model=model, tokenizer=tokenizer, n_layer=n_layer, token_count=5
        )

        # If all samples still blue, this head can be removed
        if results_without["blue"] == TARGET_BLUE:
            best_head_to_remove = head_to_test
            print(f"  Can remove L{head_to_test[0]} H{head_to_test[1]}: blue={results_without['blue']}/{TARGET_BLUE} ✓")
            break  # Remove one head per iteration

        if (i + 1) % 10 == 0:
            print(f"  Tested {i + 1}/{len(current_heads)} heads...")

    # Remove the head if we found one
    if best_head_to_remove:
        current_heads.remove(best_head_to_remove)
        removed_heads.append(best_head_to_remove)
        print(f"Removed L{best_head_to_remove[0]} H{best_head_to_remove[1]}")
    else:
        print("No more heads can be removed without losing performance")
        break

print("\n" + "=" * 80)
print("PRUNING COMPLETE")
print("=" * 80)
print(f"\nMinimal set: {len(current_heads)} heads (removed {len(removed_heads)})")
print(f"\nRemaining heads by layer:")
heads_by_layer = {}
for layer, head in current_heads:
    if layer not in heads_by_layer:
        heads_by_layer[layer] = []
    heads_by_layer[layer].append(head)

for layer in sorted(heads_by_layer.keys()):
    heads = heads_by_layer[layer]
    print(f"  L{layer}: {len(heads)} heads - {sorted(heads)}")

print(f"\nFull list of minimal heads:")
print(current_heads)

# Test final set on all samples
print(f"\n--- Testing minimal set on all {len(sample_pairs_after_color)} samples ---")
results_final = run_head_hook_experiment(
    sample_pairs_after_color,
    heads_to_replace=current_heads,
    model=model,
    tokenizer=tokenizer,
    n_layer=n_layer,
    token_count=5,
)
print(
    f"Final performance: orig={results_final['original']}, blue={results_final['blue']}, neither={results_final['neither']}"
)

# %%

print("\n--- Replace ALL layers (sanity check) ---")
s = sample_pairs_after_color[:10]
n_total = len(s)
results_all = run_layer_hook_experiment(
    s, layers_to_replace=[15, 16, 19, 20], model=model, tokenizer=tokenizer, n_layer=n_layer, token_count=5
)
print(
    f"Original: {results_all['original']:3d}/{n_total}, Blue: {results_all['blue']:3d}/{n_total}, Neither: {results_all['neither']:3d}/{n_total}"
)

# %%
len(sample_pairs_after_period)

# %%
print("\n" + "=" * 80)
print("TESTING TOP PROBE HEADS FROM TRAINING")
print("=" * 80)

top_probe_heads = [
    (15, 11),
    (14, 23),
    (15, 2),
    (15, 27),
    (15, 13),
    (14, 18),
    (9, 18),
    (15, 1),
    (15, 1),
    (15, 8),
    (12, 24),
    (17, 18),
    (20, 29),
    (15, 25),
    (19, 8),
    (19, 14),
    (15, 29),
    (12, 9),
    (20, 10),
    (15, 23),
    (15, 15),
    (21, 6),
    (12, 18),
    (20, 4),
    (19, 21),
    (16, 26),
    (19, 5),
    (11, 21),
    (19, 9),
    (14, 2),
    (15, 14),
    (21, 23),
    (11, 17),
    (19, 27),
    (19, 17),
    (14, 10),
    (15, 31),
    (15, 18),
    (12, 14),
    (16, 11),
    (9, 26),
    (12, 17),
    (21, 5),
    (21, 2),
    (19, 31),
    (14, 27),
    (13, 7),
    (16, 17),
    (14, 17),
    (19, 18),
    (14, 22),
]

print(f"\nTesting {len(top_probe_heads)} heads from probe training (V1 - smaller dataset)")
print(f"Hook position: AFTER CONTEXT (before Question)")
print(f"Layers represented: {sorted(set(l for l, h in top_probe_heads))}")
print(f"Heads by layer:")
for layer in sorted(set(l for l, h in top_probe_heads)):
    heads_in_layer = sorted([h for l, h in top_probe_heads if l == layer])
    print(f"  L{layer}: {len(heads_in_layer)} heads - {heads_in_layer}")

print(f"\n--- Testing on all {len(sample_pairs_after_context)} samples (HOOK AFTER CONTEXT - before Question) ---")
results = run_head_hook_experiment(
    sample_pairs_after_context[:20],
    heads_to_replace=top_probe_heads[:10],
    model=model,
    tokenizer=tokenizer,
    n_layer=n_layer,
    token_count=5,
)
print(f"Results: orig={results['original']}, blue={results['blue']}, neither={results['neither']}")
print(f"Blue percentage: {results['blue']/len(sample_pairs_after_context[:20])*100:.1f}%")

print(f"\n--- Sample responses (first 10) ---")
for i, d in enumerate(results["details"][:10]):
    print(f"  {i}: color_A={d['color_A']:<8} → '{d['response']:<15}' [{d['classification']}]")

# %%
print("\n" + "=" * 80)
print("TESTING TOP PROBE HEADS V2 (LARGER DATASET)")
print("=" * 80)

top_probe_heads_v2 = [
    (15, 11),
    (14, 23),
    (15, 2),
    (15, 27),
    (15, 13),
    (14, 18),
    (15, 8),
    (12, 24),
    (15, 1),
    (9, 18),
    (17, 18),
    (20, 29),
    (19, 8),
    (12, 9),
    (20, 10),
    (15, 29),
    (19, 14),
    (12, 28),
    (15, 25),
    (15, 23),
    (12, 18),
    (21, 6),
    (15, 15),
    (16, 26),
    (20, 4),
    (19, 21),
    (14, 2),
    (15, 14),
    (19, 5),
    (11, 21),
    (16, 24),
    (14, 10),
    (15, 18),
    (19, 9),
    (21, 23),
    (11, 17),
    (12, 14),
    (19, 17),
    (19, 27),
    (9, 26),
    (13, 7),
    (15, 31),
    (16, 11),
    (21, 5),
]

top_probe_heads = [
    (15, 11),
    (14, 23),
    (15, 2),
    (15, 27),
    (15, 13),
    (14, 18),
    (9, 18),
    (15, 1),
    (15, 1),
    (15, 8),
    (12, 24),
    (17, 18),
    (20, 29),
    (15, 25),
    (19, 8),
    (19, 14),
    (15, 29),
    (12, 9),
    (20, 10),
    (15, 23),
    (15, 15),
    (21, 6),
    (12, 18),
    (20, 4),
    (19, 21),
    (16, 26),
    (19, 5),
    (11, 21),
    (19, 9),
    (14, 2),
    (15, 14),
    (21, 23),
    (11, 17),
    (19, 27),
    (19, 17),
    (14, 10),
    (15, 31),
    (15, 18),
    (12, 14),
    (16, 11),
    (9, 26),
    (12, 17),
    (21, 5),
    (21, 2),
    (19, 31),
    (14, 27),
    (13, 7),
    (16, 17),
    (14, 17),
    (19, 18),
    (14, 22),
]

top_probe_heads_todo = top_probe_heads

NUM_HEADS_TO_TEST = 30
NUM_SAMPLES_TO_TEST = 10

heads_subset = top_probe_heads_todo = top_probe_heads[-1:]
samples_subset = sample_pairs_after_context[:NUM_SAMPLES_TO_TEST]

print(f"\nTesting top {NUM_HEADS_TO_TEST} heads (out of {len(top_probe_heads_v2)} available)")
print(f"Testing on {NUM_SAMPLES_TO_TEST} samples (out of {len(sample_pairs_after_context)} available)")
print(f"Hook position: AFTER CONTEXT (before Question)")
print(f"\nLayers represented: {sorted(set(l for l, h in heads_subset))}")
print(f"Heads by layer:")
for layer in sorted(set(l for l, h in heads_subset)):
    heads_in_layer = sorted([h for l, h in heads_subset if l == layer])
    print(f"  L{layer}: {len(heads_in_layer)} heads - {heads_in_layer}")

print(f"\n--- Testing on {len(samples_subset)} samples (HOOK AFTER CONTEXT - before Question) ---")
results = run_head_hook_experiment(
    samples_subset, heads_to_replace=heads_subset, model=model, tokenizer=tokenizer, n_layer=n_layer, token_count=5
)
print(f"Results: orig={results['original']}, blue={results['blue']}, neither={results['neither']}")
print(f"Blue percentage: {results['blue']/len(samples_subset)*100:.1f}%")
print(f"Original percentage: {results['original']/len(samples_subset)*100:.1f}%")

print(f"\n--- Sample responses (first 10) ---")
for i, d in enumerate(results["details"][: min(10, len(results["details"]))]):
    print(f"  {i}: color_A={d['color_A']:<8} → '{d['response']:<15}' [{d['classification']}]")

# %%
print("\n" + "=" * 80)
print("TESTING TOP PROBE HEADS V3 (N_ENTITIES=50)")
print("=" * 80)

top_probe_heads_v3 = [
    (15, 11),
    (15, 2),
    (15, 13),
    (14, 23),
    (15, 27),
    (9, 18),
    (15, 1),
    (14, 18),
    (12, 18),
    (15, 8),
    (15, 29),
    (12, 24),
    (15, 15),
    (12, 28),
    (15, 22),
    (12, 9),
    (11, 21),
    (15, 31),
    (20, 29),
    (17, 18),
    (15, 23),
    (19, 14),
    (15, 18),
    (15, 14),
    (20, 10),
    (16, 24),
    (11, 17),
    (16, 26),
    (12, 6),
    (19, 5),
    (21, 6),
    (20, 4),
    (15, 25),
    (19, 21),
    (14, 17),
    (13, 7),
    (12, 14),
    (21, 23),
    (12, 13),
    (14, 10),
    (17, 11),
]

NUM_HEADS_TO_TEST_V3 = 6
NUM_SAMPLES_TO_TEST_V3 = 10

heads_subset_v3 = top_probe_heads_v3[:NUM_HEADS_TO_TEST_V3]
samples_subset_v3 = sample_pairs_after_context[:NUM_SAMPLES_TO_TEST_V3]

print(f"\nTesting top {NUM_HEADS_TO_TEST_V3} heads (out of {len(top_probe_heads_v3)} available)")
print(f"Testing on {NUM_SAMPLES_TO_TEST_V3} samples (out of {len(sample_pairs_after_context)} available)")
print(f"Hook position: AFTER CONTEXT (before Question)")
print(f"Dataset: N_ENTITIES={N_ENTITIES}")
print(f"\nLayers represented: {sorted(set(l for l, h in heads_subset_v3))}")
print(f"Heads by layer:")
for layer in sorted(set(l for l, h in heads_subset_v3)):
    heads_in_layer = sorted([h for l, h in heads_subset_v3 if l == layer])
    print(f"  L{layer}: {len(heads_in_layer)} heads - {heads_in_layer}")

print(f"\n{'='*60}")
print(f"COMPARISON: Baseline vs Selected vs All")
print(f"{'='*60}")

print(f"\n[1] BASELINE - No heads replaced:")
results_baseline = run_head_hook_experiment(
    samples_subset_v3, heads_to_replace=[], model=model, tokenizer=tokenizer, n_layer=n_layer, token_count=5
)
print(
    f"    Original: {results_baseline['original']:2d}/{len(samples_subset_v3)} ({results_baseline['original']/len(samples_subset_v3)*100:.1f}%)"
)
print(
    f"    Blue:     {results_baseline['blue']:2d}/{len(samples_subset_v3)} ({results_baseline['blue']/len(samples_subset_v3)*100:.1f}%)"
)
print(
    f"    Neither:  {results_baseline['neither']:2d}/{len(samples_subset_v3)} ({results_baseline['neither']/len(samples_subset_v3)*100:.1f}%)"
)

print(f"\n[2] TOP {NUM_HEADS_TO_TEST_V3} HEADS - Selected probe heads:")
results_selected = run_head_hook_experiment(
    samples_subset_v3,
    heads_to_replace=heads_subset_v3,
    model=model,
    tokenizer=tokenizer,
    n_layer=n_layer,
    token_count=5,
)
print(
    f"    Original: {results_selected['original']:2d}/{len(samples_subset_v3)} ({results_selected['original']/len(samples_subset_v3)*100:.1f}%)"
)
print(
    f"    Blue:     {results_selected['blue']:2d}/{len(samples_subset_v3)} ({results_selected['blue']/len(samples_subset_v3)*100:.1f}%)"
)
print(
    f"    Neither:  {results_selected['neither']:2d}/{len(samples_subset_v3)} ({results_selected['neither']/len(samples_subset_v3)*100:.1f}%)"
)

all_heads = [(layer, head) for layer in range(n_layer) for head in range(n_head)]

# Random heads baseline (5 trials with explicit seeds for reproducibility)
RANDOM_SEEDS = [42, 123, 456, 789, 1000]
random_blue_counts = []

print(f"\n[3] RANDOM {NUM_HEADS_TO_TEST_V3} HEADS - 5 trials with different seeds:")
for seed in RANDOM_SEEDS:
    rng = random.Random(seed)
    random_heads = rng.sample(all_heads, NUM_HEADS_TO_TEST_V3)
    results_random = run_head_hook_experiment(
        samples_subset_v3,
        heads_to_replace=random_heads,
        model=model,
        tokenizer=tokenizer,
        n_layer=n_layer,
        token_count=5,
    )
    random_blue_counts.append(results_random["blue"])
    print(f"    Seed {seed}: Blue = {results_random['blue']}/{len(samples_subset_v3)}")

avg_random_blue = sum(random_blue_counts) / len(RANDOM_SEEDS)
std_random_blue = (sum((x - avg_random_blue) ** 2 for x in random_blue_counts) / len(RANDOM_SEEDS)) ** 0.5
print(
    f"    Average:  {avg_random_blue:.1f}/{len(samples_subset_v3)} ({avg_random_blue/len(samples_subset_v3)*100:.1f}% ± {std_random_blue/len(samples_subset_v3)*100:.1f}%)"
)

print(f"\n[4] ALL HEADS - Every head in every layer ({n_layer} layers × {n_head} heads = {n_layer * n_head} total):")
results_all = run_head_hook_experiment(
    samples_subset_v3, heads_to_replace=all_heads, model=model, tokenizer=tokenizer, n_layer=n_layer, token_count=5
)
print(
    f"    Original: {results_all['original']:2d}/{len(samples_subset_v3)} ({results_all['original']/len(samples_subset_v3)*100:.1f}%)"
)
print(
    f"    Blue:     {results_all['blue']:2d}/{len(samples_subset_v3)} ({results_all['blue']/len(samples_subset_v3)*100:.1f}%)"
)
print(
    f"    Neither:  {results_all['neither']:2d}/{len(samples_subset_v3)} ({results_all['neither']/len(samples_subset_v3)*100:.1f}%)"
)

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Baseline → Selected: Blue increased by {results_selected['blue'] - results_baseline['blue']} samples")
print(f"Baseline → Random:   Blue increased by {avg_random_blue - results_baseline['blue']:.1f} samples (avg)")
print(f"Selected vs Random:  Selected has {results_selected['blue'] - avg_random_blue:.1f} more blue than random avg")
print(f"Selected → All:      Blue increased by {results_all['blue'] - results_selected['blue']} samples")

# %%
