# Quick Start Guide

## For GPU Machine Setup

### 1. Install Dependencies (5 minutes)

```bash
# Navigate to project directory
cd interpretability-proj

# Install requirements
pip install -r requirements.txt

# Install Flash Linear Attention library
pip install git+https://github.com/sustcsonglin/flash-linear-attention.git
```

### 2. Verify Installation (2 minutes)

```python
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from fla import __version__; print(f'FLA: {__version__}')"
```

### 3. Run Quick Test (5-10 minutes)

```bash
# Run the test suite
python tests/test_state_shapes.py

# Or use pytest
pytest tests/test_state_shapes.py -v
```

Expected output: All tests passing ✓

### 4. Interactive Exploration (10-15 minutes)

```bash
# Launch Jupyter
jupyter notebook notebooks/01_test_state_extraction.ipynb

# Run all cells and verify:
# - Model loads
# - States are extracted
# - Shapes look reasonable
```

## Common Issues & Solutions

### Issue: CUDA Out of Memory

**Solution:**
```python
# Use CPU for testing (slower but works)
model, tokenizer = load_retnet_model(device="cpu")

# Or use smaller sequences
text = "Short test."  # Instead of long passages
```

### Issue: Can't find retention modules

**Symptom:** "No states were captured!" warning

**Solution:**
```python
# Print model structure to identify layer names
from models.load_retnet import print_model_structure
print_model_structure(model, max_depth=4)

# Manually set modules if needed
extractor.set_retention_modules([
    (0, model.layers[0].retention),
    (1, model.layers[1].retention),
    # ... etc
])
```

### Issue: Import errors for fla library

**Solution:**
```bash
# Clone and install from source
git clone https://github.com/sustcsonglin/flash-linear-attention.git
cd flash-linear-attention
pip install -e .
```

## Quick Usage Examples

### Example 1: Basic State Extraction

```python
from models.load_retnet import load_retnet_model
from models.state_extractor import RetNetStateExtractor

# Load
model, tokenizer = load_retnet_model(device="cuda")

# Extract
extractor = RetNetStateExtractor(model)
extractor.register_hooks()

text = "The capital of France is Paris."
input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
states = extractor.extract_states(input_ids)

print(f"Extracted {len(states)} layer states")
```

### Example 2: Save States for Later

```python
from models.state_extractor import save_states_to_file, load_states_from_file

# After extracting states
save_states_to_file(states, "my_states.npz")

# Later, load them back
loaded_states = load_states_from_file("my_states.npz")
```

### Example 3: Batch Processing

```python
texts = [
    "First sentence.",
    "Second sentence.",
    "Third sentence."
]

# Tokenize with padding
inputs = tokenizer(texts, return_tensors="pt", padding=True)
input_ids = inputs.input_ids.to("cuda")

# Extract states for all at once
states = extractor.extract_states(input_ids)

# states[layer_idx] will have shape [3, num_heads, d_k, d_v]
# where 3 is the batch size
```

## Performance Tips

1. **Use bfloat16:** Already default in `load_retnet_model()`
2. **Batch inputs:** Process multiple sequences at once
3. **Free memory:** Call `extractor.remove_hooks()` when done
4. **Use HDF5:** For large state collections, use `.h5` format instead of `.npz`

## Debugging Checklist

If state extraction isn't working:

1. ✓ Check CUDA is available: `torch.cuda.is_available()`
2. ✓ Verify model loaded: `print(model)`
3. ✓ Check hooks registered: `print(len(extractor.hooks))`
4. ✓ Inspect module structure: `print_model_structure(model)`
5. ✓ Try verbose mode: `RetNetStateExtractor(model, verbose=True)`
6. ✓ Check for errors in hook execution (will print warnings)

## Next Phase Preview

Once state extraction works, you'll create a dataset like:

```python
dataset = [
    {
        "text": "The ID of Alice is 12345. [filler] What is the ID of Alice?",
        "fact_position": 7,  # token position of "12345"
        "query_position": 25,
        "lag": 18,
        "answer": "12345"
    },
    # ... more examples
]
```

Then extract states at fact_position and query_position, and train linear probes to see if the model "remembers" the fact!

## Resources

- **RetNet Paper:** https://arxiv.org/abs/2307.08621
- **FLA Library:** https://github.com/sustcsonglin/flash-linear-attention
- **Model:** https://huggingface.co/fla-hub/retnet-2.7B-100B

## Questions?

Check:
1. `README.md` - Project overview
2. `models/ARCHITECTURE.md` - Technical details
3. `PHASE1_COMPLETE.md` - Implementation summary
4. Code comments in `.py` files

