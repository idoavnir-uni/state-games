# RetNet State Mechanism

## What is the Recurrent State?

The `recurrent_state` is an accumulated key-value memory matrix that summarizes all previously seen tokens with exponential decay.

```python
state_t = Σ(γ^(t-i) * k_i ⊗ v_i) for i=1 to t
```

Where:
- `k_i`, `v_i` are key and value vectors at position i
- `γ` is the decay factor (< 1)
- `⊗` is outer product

## Computation Flow in RetNet

```python
# 1. Project inputs
q = W_q @ x
k = W_k @ x  
v = W_v @ x

# 2. Update recurrent state (accumulated K⊗V)
recurrent_state = γ * recurrent_state_prev + k ⊗ v

# 3. Compute output using query
o = q @ recurrent_state

# 4. Return (output, attention_weights=None, past_key_values)
return o, None, past_key_values
```

The state is stored **before** query multiplication - it's the memory that queries retrieve from.

## FLA Library Cache Structure

```python
past_key_values = Cache()  # FLACache or LegacyFLACache

# Each layer has its own state dict
past_key_values[layer_idx] = {
    "recurrent_state": tensor,  # The K⊗V accumulator
    "conv_state": tuple,        # Short convolution states (optional)
    "attn_state": tuple,        # Attention states (optional)
    "ffn_state": any            # FFN states (optional)
}
```

Updated via:
```python
past_key_values.update(
    recurrent_state=recurrent_state,
    layer_idx=layer_idx,
    ...
)
```

## State Shape

For RetNet-2.7B (hidden_size=2048, num_heads=8, expand_k=1.0, expand_v=2.0):

```python
recurrent_state.shape = [batch, num_heads, key_dim_per_head, value_dim_per_head]
                      = [1, 8, 256, 512]
```

Where:
- `key_dim_per_head = (hidden_size * expand_k) / num_heads = 2048 * 1.0 / 8 = 256`
- `value_dim_per_head = (hidden_size * expand_v) / num_heads = 2048 * 2.0 / 8 = 512`

## Our Extraction Code

```python
# RetNetStateExtractor.extract_states()
outputs = model(input_ids, use_cache=True)

# Access cache
past_kv = outputs.past_key_values

# Extract each layer's state
for layer_idx, layer_state in enumerate(past_kv):
    recurrent_state = layer_state["recurrent_state"]
    states[layer_idx] = recurrent_state.detach().cpu()
```

## What We Get

Running `extractor.extract_states(input_ids)` with sequence length N returns:

```python
{
    0: tensor[1, 8, 256, 512],  # Layer 0 final state (after N tokens)
    1: tensor[1, 8, 256, 512],  # Layer 1 final state
    ...
    31: tensor[1, 8, 256, 512]  # Layer 31 final state
}
```

Each state is the **final accumulated K⊗V memory** after processing all N tokens in that layer.

## Key Points

1. **One state per layer** - 32 independent recurrent states
2. **Accumulated over sequence** - Contains information from all previous tokens
3. **Exponentially decayed** - Recent tokens weighted more heavily
4. **Before query multiplication** - Raw memory, not attention output
5. **Final state only** - Current implementation extracts end-of-sequence state, not intermediate positions

