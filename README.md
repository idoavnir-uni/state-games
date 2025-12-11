# Measuring Forgetting in Sub-Quadratic Language Models

**Project Members:** Yuval Milo, Shahar Mendel, Ido Avnir

## Overview

This project investigates how sub-quadratic language models (specifically RetNet) forget information over long contexts. We extract retention states from the model and use linear probes to measure information retention at varying time lags.

## Setup

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Flash Linear Attention library:
```bash
pip install git+https://github.com/sustcsonglin/flash-linear-attention.git
```

### Model

We use the pretrained [fla-hub/retnet-2.7B-100B](https://huggingface.co/fla-hub/retnet-2.7B-100B) model (2.7B parameters, trained on 100B tokens from SlimPajama-627B).

## Project Structure

```
interpretability-proj/
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── models/                       # Model loading and state extraction
│   ├── __init__.py
│   ├── load_retnet.py           # Model loading utilities
│   └── state_extractor.py       # State extraction implementation
└── notebooks/                    # Jupyter notebooks for experiments
    └── 01_test_state_extraction.ipynb
```

## Phase 1: State Extraction

Current focus is on extracting retention states from RetNet at arbitrary positions during inference.

**Goal:** Build infrastructure to capture the recurrent state `S_t` at each layer and position.

**Key files:**
- `models/load_retnet.py` - Load the RetNet model
- `models/state_extractor.py` - Extract retention states using PyTorch hooks
- `notebooks/01_test_state_extraction.ipynb` - Test and verify state extraction

## Usage

```python
from models.load_retnet import load_retnet_model, get_model_config
from models.state_extractor import RetNetStateExtractor

# Load model
model, tokenizer = load_retnet_model(device="cuda")
config = get_model_config(model)

# Extract states
extractor = RetNetStateExtractor(model)
extractor.register_hooks()

# Run inference and collect states
text = "This is a test sentence."
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
states = extractor.extract_final_states(input_ids)

# Access states by layer
for layer_idx, layer_states in states.items():
    print(f"Layer {layer_idx}: {layer_states.shape}")
```

## Next Steps

- Phase 2: Create associative recall dataset
- Phase 3: Extract states for entire dataset
- Phase 4: Train linear probes on states
- Phase 5: Analyze forgetting patterns

