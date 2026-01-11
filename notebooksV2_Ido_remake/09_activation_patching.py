# %% [markdown]
# # RWKV Activation Patching Experiment
#
# This notebook demonstrates activation patching between two RWKV runs with different contexts.
# We patch the state from one context into another to see if we can change the model's output.

# %%
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(".."))

from models.load_rwkv import load_rwkv_model
from models.state_extractor_rwkv import RWKVStateExtractor

# %%
model, tokenizer = load_rwkv_model(model_name="rwkv7-g1b-1.5b-20251202-ctx8192.pth")
extractor = RWKVStateExtractor(model, verbose=False)

# %%
COLOR_TOKENS = {
    "green": tokenizer.encode("green")[0],
    "Green": tokenizer.encode("Green")[0],
    "red": tokenizer.encode("red")[0],
    "Red": tokenizer.encode("Red")[0],
    "blue": tokenizer.encode("blue")[0],
    "yellow": tokenizer.encode("yellow")[0],
}
print("Color token IDs:", COLOR_TOKENS)


def print_color_probs(logits):
    probs = torch.softmax(logits, dim=-1)
    print("  Color probabilities:")
    for name, tid in COLOR_TOKENS.items():
        print(f"    {name}: {probs[tid].item():.4f}")


# %%
PROMPT_A = "Tom Hanks's favorite color is green. Olaf Scholz's favorite color is yellow. Michael Jordan's favorite color is blue. What is Tom Hanks's favorite color? Answer: his favorite color is"
PROMPT_B = "Tom Hanks's favorite color is red. Olaf Scholz's favorite color is yellow. Michael Jordan's favorite color is blue. What is Tom Hanks's favorite color? Answer: his favorite color is"

print("Prompt A:", PROMPT_A)
print("\nPrompt B:", PROMPT_B)

# %% [markdown]
# ## 1. Baseline Generation (using model.generate)
# First, let's see what the model outputs using the standard generate method.

# %%
print("=" * 60)
print("Baseline using model.generate() - same as 01_new_inference.py")
print("=" * 60)

print(f"\nPrompt A: ", end="")
model.generate(PROMPT_A, token_count=1, callback=lambda s: print(s, end="", flush=True))
print()

print(f"\nPrompt B: ", end="")
model.generate(PROMPT_B, token_count=1, callback=lambda s: print(s, end="", flush=True))
print()

# %% [markdown]
# ## 2. Baseline Generation (manual forward pass)
# Same thing but with manual token-by-token generation for patching experiments.


# %%
def generate_tokens(model, tokenizer, prompt, max_new_tokens=1):
    tokens = tokenizer.encode(prompt)
    state = None

    with torch.no_grad():
        out, state = model.forward(tokens, None)

    generated = []
    first_logits = out.clone()
    for _ in range(max_new_tokens):
        probs = torch.softmax(out, dim=-1)
        next_token = torch.argmax(probs).item()
        generated.append(next_token)

        with torch.no_grad():
            out, state = model.forward([next_token], state)

    return tokenizer.decode(generated), first_logits


# %%
print("=" * 60)
print("Baseline Generation (no patching)")
print("=" * 60)

output_a, logits_a = generate_tokens(model, tokenizer, PROMPT_A)
print(f"\nPrompt A (Tom=green):")
print(f"Generated: {output_a}")
print_color_probs(logits_a)

output_b, logits_b = generate_tokens(model, tokenizer, PROMPT_B)
print(f"\nPrompt B (Tom=Red):")
print(f"Generated: {output_b}")
print_color_probs(logits_b)

# %% [markdown]
# ## 3. Tokenize and Find Patch Points
# We need to find the token positions where the color differs between the full prompts.

# %%
tokens_a = tokenizer.encode(PROMPT_A)
tokens_b = tokenizer.encode(PROMPT_B)

print(f"Prompt A has {len(tokens_a)} tokens")
print(f"Prompt B has {len(tokens_b)} tokens")

print("\nPrompt A tokens (showing first 30):")
for i, t in enumerate(tokens_a[:30]):
    print(f"  {i}: {t} -> '{tokenizer.decode([t])}'")

# %%
# Find where the tokens differ (the color token)
diff_positions = []
for i in range(min(len(tokens_a), len(tokens_b))):
    if tokens_a[i] != tokens_b[i]:
        diff_positions.append(i)
        print(f"Difference at position {i}:")
        print(f"  Prompt A: '{tokenizer.decode([tokens_a[i]])}'")
        print(f"  Prompt B: '{tokenizer.decode([tokens_b[i]])}'")

# The patch point is right after the differing color token
PATCH_POSITION = diff_positions[0] + 1 if diff_positions else None
print(f"\nPatch position (after color): {PATCH_POSITION}")

# %% [markdown]
# ## 4. Extract Full States at Patch Point
# Run both contexts and extract the full state after the color token.


# %%
def get_state_at_position(model, tokens, position):
    state = None
    with torch.no_grad():
        for i in range(position):
            out, state = model.forward([tokens[i]], state)
    return state, out


# %%
state_a, out_a = get_state_at_position(model, tokens_a, PATCH_POSITION)
state_b, out_b = get_state_at_position(model, tokens_b, PATCH_POSITION)

print(f"Extracted states at position {PATCH_POSITION}")
print(f"State A has {len(state_a)} tensors")
print(f"State B has {len(state_b)} tensors")

# %% [markdown]
# ## 5. Activation Patching
# Patch state B into the generation of context A and see if the output changes.


# %%
def generate_with_patched_state(model, tokenizer, patched_state, tokens_after_patch, max_new_tokens=1):
    state = patched_state

    with torch.no_grad():
        for t in tokens_after_patch:
            out, state = model.forward([t], state)

    generated = []
    first_logits = out.clone()
    for _ in range(max_new_tokens):
        probs = torch.softmax(out, dim=-1)
        next_token = torch.argmax(probs).item()
        generated.append(next_token)

        with torch.no_grad():
            out, state = model.forward([next_token], state)

    return tokenizer.decode(generated), first_logits


# %%
tokens_after_patch_a = tokens_a[PATCH_POSITION:]

print("Tokens after patch point (from prompt A):")
for i, t in enumerate(tokens_after_patch_a[:20]):
    print(f"  {i}: '{tokenizer.decode([t])}'")
if len(tokens_after_patch_a) > 20:
    print(f"  ... and {len(tokens_after_patch_a) - 20} more tokens")

# %%
print("=" * 60)
print("Activation Patching Experiment (Full State Patch)")
print("=" * 60)

# Patch all layers using extractor.patch_state with all layers specified
all_layers_spec = {layer_idx: None for layer_idx in range(extractor.n_layer)}
patched_state = extractor.patch_state(state_b, state_a, all_layers_spec)

patched_output, patched_logits = generate_with_patched_state(model, tokenizer, patched_state, tokens_after_patch_a)

print(f"\nPrompt A output: {output_a}")
print(f"Prompt B output: {output_b}")
print(f"Patched output (state B into prompt A): {patched_output}")
print_color_probs(patched_logits)

# %%
print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)

print(f"\nPrompt A output: {output_a}")
print(f"Prompt B output: {output_b}")
print(f"Patched output: {patched_output}")

print("\nNote: The question asks about Tom Hanks (green vs Red).")
print("Patching injects state B (after 'Red') into prompt A's generation.")
print("If patching works, patched output should say 'Red' instead of 'green'.")

# %%
