# RetNet Architecture Documentation

## Overview

This document describes the architecture of RetNet and how retention states are structured and extracted.

## Model Details

**Model:** fla-hub/retnet-2.7B-100B
**Size:** 2.7B parameters
**Training:** 100B tokens from SlimPajama-627B dataset

## Retention Mechanism

RetNet uses a "retention" mechanism that can be computed in two modes:

1. **Parallel Mode (Training):** States computed via cumulative operations across the sequence
2. **Recurrent Mode (Inference):** States updated incrementally: `S_{t+1} = f(S_t, x_t)`

### State Representation

The retention state `S_t` at position `t` is typically a tensor that compresses all information from tokens `0:t`.

**Expected shape:** `[batch, num_heads, d_key, d_value]` or `[batch, num_heads, state_dim]`

Where:
- `batch`: Batch size
- `num_heads`: Number of retention heads in the layer
- `d_key`, `d_value`: Key and value dimensions (often equal to `hidden_size / num_heads`)
- `state_dim`: Total state dimension per head

## Architecture Breakdown

### Layers

RetNet typically consists of:
- Embedding layer
- Multiple RetNet blocks (each containing retention + FFN)
- Output projection to vocabulary

For the 2.7B model, we expect approximately:
- **32 layers** (estimated)
- **32 heads per layer** (estimated)
- **2560-3072 hidden dimension** (estimated)

*Note: These will be verified when loading the model.*

### Per-Layer Structure

Each RetNet layer contains:
1. **Retention sublayer:** Multi-head retention mechanism with recurrent state
2. **Feed-forward network (FFN):** Standard position-wise FFN
3. **Layer normalization:** Applied before each sublayer

### Retention State Updates

In recurrent mode, the state is updated as:
```
S_t = decay * S_{t-1} + K_t^T V_t
output_t = Q_t S_t
```

Where:
- `Q_t, K_t, V_t`: Query, key, value vectors at position t
- `decay`: Exponential decay factor (often learned per-head)
- `S_t`: The retention state we want to extract

## State Extraction Strategy

### Approach: PyTorch Forward Hooks

We register hooks on the retention modules to capture states during the forward pass.

**Target modules:** 
- Look for modules named like `layers[i].retention` or `blocks[i].attn`
- Hook into the retention computation to capture `S_t`

**Challenges:**
1. States may not be explicitly materialized in parallel mode
2. Need to access internal computations within the retention module
3. Multi-head states need to be separated and tracked

### Alternative: Recurrent Mode

If parallel mode doesn't materialize states:
- Force the model into recurrent inference mode
- Process sequence token-by-token
- Extract state after each token
- More expensive but guarantees state access

## Information to Extract

When running state extraction, we need to capture:
1. **State tensors:** The actual `S_t` matrices
2. **Layer index:** Which layer (0 to num_layers-1)
3. **Head index:** Which head (0 to num_heads-1)
4. **Position:** Token position in sequence
5. **Metadata:** Model config, sequence length, etc.

## State Storage Format

For efficient storage and retrieval:

```python
states = {
    'layer_0': {
        'head_0': tensor([...]),  # Shape: [batch, seq_len, d_k, d_v]
        'head_1': tensor([...]),
        ...
    },
    'layer_1': {...},
    ...
}
```

Or flattened for probe training:
```python
state_df = pd.DataFrame({
    'sequence_id': [...],
    'layer': [...],
    'head': [...],
    'position': [...],
    'state': [np.array(...), ...]  # Each is a flattened state vector
})
```

## Next Steps

- [ ] Verify actual architecture by loading the model
- [ ] Identify exact module names for retention layers
- [ ] Test state extraction in both parallel and recurrent modes
- [ ] Document actual state shapes and dimensions
- [ ] Profile memory usage for state storage

