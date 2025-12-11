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

For RetNet-2.7B, the actual extracted shape is:

```python
recurrent_state.shape = [batch, num_heads, head_dim, state_size]
                      = [1, 10, 256, 512]
```

Where:
- `batch` = 1 (batch size)
- `num_heads` = 10 (number of retention heads)
- `head_dim` = 256 (query/output dimension per head)
- `state_size` = 512 (key dimension for the K⊗V outer product)

The state is the **final accumulated K⊗V matrix** after processing all input tokens. This is a single state per layer representing the complete memory, not per-position states.

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

Running `extractor.extract_states(input_ids)` returns:

```python
{
    0: tensor[1, 10, 256, 512],  # Layer 0 final state
    1: tensor[1, 10, 256, 512],  # Layer 1 final state
    ...
    31: tensor[1, 10, 256, 512]  # Layer 31 final state
}
```

Each state is the **final accumulated K⊗V matrix** after processing all input tokens:
- `state[0, h, :, :]` = accumulated memory matrix for head h (shape [256, 512])
- This represents the complete memory that queries can retrieve from
- To get intermediate states, run the model multiple times with different sequence lengths

## Key Points

1. **One state per layer** - 32 independent recurrent states (one per layer)
2. **Per-head structure** - Each layer has 10 heads with separate states
3. **K⊗V matrix format** - Shape [head_dim, state_size] = [256, 512] per head
4. **Exponentially decayed** - Recent tokens weighted more heavily via γ decay
5. **Final accumulated state** - Single state after all tokens, not per-position states
6. **Incremental extraction** - Use `extract_states_incremental()` to get states at each position

## Extraction Methods

### `extract_states(input_ids)`
Returns the final K⊗V state after processing all tokens.
```python
states = extractor.extract_states(input_ids)
# Returns: {layer_idx: tensor[batch, heads, head_dim, state_size]}
```

### `extract_states_incremental(input_ids, layers=None)`
Returns K⊗V states at each token position (runs model N times for N tokens).
```python
states_by_pos = extractor.extract_states_incremental(input_ids)
# Returns: {position: {layer_idx: tensor[batch, heads, head_dim, state_size]}}
# position is 1-indexed (1 = after first token)
```

### `extract_states_at_positions(input_ids, positions, layers=None)`
Returns K⊗V states at specific positions only (more efficient for sparse extraction).
```python
states_by_pos = extractor.extract_states_at_positions(input_ids, positions=[1, 5, 10])
# Returns: {position: {layer_idx: tensor[batch, heads, head_dim, state_size]}}
```

