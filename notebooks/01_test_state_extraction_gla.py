# %% [markdown]
# # GLA State Extraction
# 
# This notebook demonstrates state extraction for GLA (Gated Linear Attention) models.
# 
# **Extraction Methods:**
# 1. `extract_final_states` - Final state only (single forward pass)
# 2. `extract_incremental_states_single_pass` - All positions (O(N) - efficient)

# %%
import sys
import os
import torch


sys.path.insert(0, os.path.abspath('..'))

from models.load_gla import load_gla_model, get_model_config, print_model_structure
from models.state_extractor_gla import GLAStateExtractor

print("Setup complete!")

# %% [markdown]
# ## 1. Load Model and Configuration

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    print("WARNING: Running on CPU. This will be very slow for large models.")
    print("Consider running on a GPU or using a smaller model for testing.")

# %%
print("Loading GLA model...")
model, tokenizer = load_gla_model(
    model_name="fla-hub/gla-1.3B-100B",
    device=device,
    torch_dtype=torch.bfloat16
)

# %%
config = get_model_config(model)

print("\n=== Key Configuration ===")
print(f"Number of layers: {config.get('num_layers', 'Unknown')}")
print(f"Number of heads: {config.get('num_heads', 'Unknown')}")
print(f"Hidden size: {config.get('hidden_size', 'Unknown')}")
print(f"Vocabulary size: {config.get('vocab_size', 'Unknown')}")
print(f"Max sequence length: {config.get('max_seq_len', 'Unknown')}")
print(f"Expand K: {config.get('expand_k', 'Unknown')}")
print(f"Expand V: {config.get('expand_v', 'Unknown')}")
print(f"Attention mode: {config.get('attn_mode', 'Unknown')}")
print(f"Use short conv: {config.get('use_short_conv', 'Unknown')}")

# %%
print_model_structure(model, max_depth=3)

# %% [markdown]
# ## 2. Initialize State Extractor and Prepare Test Input

# %%
extractor = GLAStateExtractor(model, verbose=True)

test_text = "The quick brown fox jumps over the lazy dog."
print(f"Input text: '{test_text}'")

inputs = tokenizer(test_text, return_tensors="pt")
input_ids = inputs.input_ids.to(device)

print(f"Token IDs shape: {input_ids.shape}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

# %% [markdown]
# ## 3. Extract Final States
# 
# Use `extract_final_states` to get the state after processing the entire sequence (single forward pass).

# %%
final_states = extractor.extract_final_states(input_ids)

print(f"Number of layers: {len(final_states)}")
if final_states:
    first_layer_state = final_states[0]
    print(f"State shape per layer: {first_layer_state.shape}")
    print(f"State dtype: {first_layer_state.dtype}")

# %% [markdown]
# ## 4. Extract Incremental States (Single Pass)
# 
# Use `extract_incremental_states_single_pass` to get the state at every position in the sequence.

# %%
incremental_states = extractor.extract_incremental_states_single_pass(input_ids)

seq_len = input_ids.shape[1]
print(f"Number of positions: {len(incremental_states)}")
print(f"Number of layers per position: {len(incremental_states[1])}")
print(f"State shape at each position: {incremental_states[1][0].shape}")

# %%
print("=== States Summary ===")
print(f"\nFinal states (extract_final_states):")
print(f"  - Returns state after processing all {seq_len} tokens")
print(f"  - Shape: {final_states[0].shape}")
print(f"  - Number of states: {len(final_states)}")

print(f"\nIncremental states (extract_incremental_states_single_pass):")
print(f"  - Returns state at each of {len(incremental_states)} positions")
print(f"  - Amount of states: {sum(len(layers) for layers in incremental_states.values())}")
print(f"  - Position keys: {list(incremental_states.keys())}")
print(f"  - Each position has {len(incremental_states[1])} layers")

# %%
