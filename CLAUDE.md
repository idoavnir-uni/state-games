# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project investigating how sub-quadratic language models (specifically RetNet) forget information over long contexts. The project extracts retention states from the model and uses linear probes to measure information retention at varying time lags.

**Model:** fla-hub/retnet-2.7B-100B (2.7B parameters, trained on 100B tokens from SlimPajama-627B)

## Development Setup

### Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install Flash Linear Attention library (REQUIRED for RetNet)
pip install git+https://github.com/sustcsonglin/flash-linear-attention.git

# Run setup script (handles both dependencies)
./setup.sh
```

### Common Development Commands
```bash
# Run tests
python tests/test_state_shapes.py
pytest tests/  # If pytest is available

# Run Jupyter notebooks for experiments
jupyter notebook notebooks/01_test_state_extraction.ipynb

# Quick test of state extraction
python -m models.state_extractor

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Architecture Overview

### Core Components

1. **Model Loading (`models/load_retnet.py`)**: Utilities for loading the RetNet model and extracting configuration
2. **State Extraction (`models/state_extractor.py`)**: Core functionality for extracting retention states using PyTorch hooks
3. **Testing (`tests/`)**: Unit tests for state extraction and validation

### State Extraction Methods

The `RetNetStateExtractor` class provides multiple extraction methods:

- `extract_states(input_ids)`: Extract final accumulated K⊗V states after processing all tokens
- `extract_states_incremental(input_ids)`: Extract states at each position (O(N²) method)  
- `extract_states_single_pass(input_ids)`: Efficiently extract states at each position (O(N) method)
- `extract_states_at_positions(input_ids, positions)`: Extract states only at specific positions

### RetNet State Mechanism

RetNet uses recurrent states that accumulate key-value memory with exponential decay:
```python
state_t = Σ(γ^(t-i) * k_i ⊗ v_i) for i=1 to t
```

**State Shape:** `[batch, num_heads, head_dim, state_size]` = `[1, 10, 256, 512]` for RetNet-2.7B

States are stored in `past_key_values` cache structure:
```python
past_key_values[layer_idx] = {
    "recurrent_state": tensor,  # The K⊗V accumulator we extract
    "conv_state": tuple,        # Short convolution states
    "attn_state": tuple,        # Attention states  
    "ffn_state": any           # FFN states
}
```

### Usage Example
```python
from models.load_retnet import load_retnet_model
from models.state_extractor import RetNetStateExtractor

# Load model
model, tokenizer = load_retnet_model(device="cuda")

# Extract states
extractor = RetNetStateExtractor(model)
text = "The quick brown fox jumps over the lazy dog."
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
states = extractor.extract_states(input_ids)

# states is Dict[layer_idx, tensor] with 32 layers
for layer_idx, state in states.items():
    print(f"Layer {layer_idx}: {state.shape}")  # [1, 10, 256, 512]
```

## Code Style Guidelines

Based on `.cursorrules`:
- Write clean, production-quality code following DRY principles
- Keep imports at the top of files
- Use clear variable and function names instead of extensive comments
- Keep documentation minimal and essential only
- Focus on code quality over explanation

## Project Phases

1. **Phase 1 (Current):** State extraction infrastructure
2. **Phase 2:** Create associative recall dataset  
3. **Phase 3:** Extract states for entire dataset
4. **Phase 4:** Train linear probes on states
5. **Phase 5:** Analyze forgetting patterns

## Important Notes

- Requires CUDA for reasonable performance with 2.7B model
- The Flash Linear Attention library is required and must be installed separately
- State extraction can be memory-intensive; use appropriate batch sizes
- States represent accumulated K⊗V memory matrices, not individual token representations