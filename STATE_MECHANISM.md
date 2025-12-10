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
recurrent_state.shape = [batch, sequence_length, state_dim]
                      = [1, 13, 2560]
```

Where:
- `batch` = 1 (batch size)
- `sequence_length` = 13 (number of tokens in input)
- `state_dim` = 2560 (compressed state dimension)

The state stores information **per position** in the sequence, not just a single accumulated state. Each position has its own 2560-dimensional state vector representing the accumulated K⊗V memory up to that point.

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

Running `extractor.extract_states(input_ids)` with sequence length N=13 returns:

```python
{
    0: tensor[1, 13, 2560],  # Layer 0 states (one per position)
    1: tensor[1, 13, 2560],  # Layer 1 states
    ...
    31: tensor[1, 13, 2560]  # Layer 31 states
}
```

Each state contains the accumulated K⊗V memory **at each position** in the sequence:
- `state[0, 0, :]` = memory after token 1
- `state[0, 1, :]` = memory after tokens 1-2
- `state[0, 12, :]` = memory after all 13 tokens

## Key Points

1. **One state per layer** - 32 independent recurrent states
2. **States per position** - Each position has accumulated memory from all previous tokens
3. **Exponentially decayed** - Recent tokens weighted more heavily via γ decay
4. **Before query multiplication** - Raw memory, not attention output
5. **Position-wise extraction** - We get states at all positions [0..N-1], not just the final one

