import torch
import torch.nn as nn
from typing import Dict, Optional
import numpy as np
import warnings

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class RetNetStateExtractor:
    def __init__(self, model: nn.Module, verbose: bool = True):
        self.model = model
        self.verbose = verbose

    def extract_states(self, input_ids: torch.Tensor, use_cache: bool = True) -> Dict[int, torch.Tensor]:
        states = {}

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=use_cache)

            if outputs.past_key_values is not None:
                for layer_idx, layer_state in enumerate(outputs.past_key_values):
                    if layer_state is not None and "recurrent_state" in layer_state:
                        states[layer_idx] = layer_state["recurrent_state"].detach().cpu()

        if len(states) == 0:
            warnings.warn("No states were captured! Check that use_cache=True and model supports caching.")

        return states
