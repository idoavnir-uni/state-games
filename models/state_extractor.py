import os
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional
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

    def extract_final_states(self, input_ids: torch.Tensor, use_cache: bool = True) -> Dict[int, torch.Tensor]:
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

    def extract_incremental_states_dumb_rerunning(
        self,
        input_ids: torch.Tensor,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, Dict[int, torch.Tensor]]:
        seq_len = input_ids.shape[1]
        states_by_position = {}

        if self.verbose:
            print(f"Extracting states incrementally for {seq_len} positions...")

        with torch.no_grad():
            for pos in range(1, seq_len + 1):
                partial_ids = input_ids[:, :pos].contiguous()
                outputs = self.model(partial_ids, use_cache=True)

                position_states = {}
                if outputs.past_key_values is not None:
                    for layer_idx, layer_state in enumerate(outputs.past_key_values):
                        if layers is not None and layer_idx not in layers:
                            continue
                        if layer_state is not None and "recurrent_state" in layer_state:
                            position_states[layer_idx] = layer_state["recurrent_state"].detach().cpu()

                states_by_position[pos] = position_states

        if self.verbose:
            num_layers = len(states_by_position.get(1, {}))
            print(f"Extracted states for {seq_len} positions, {num_layers} layers each")

        return states_by_position

    def extract_incremental_states_single_pass(
        self,
        input_ids: torch.Tensor,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, Dict[int, torch.Tensor]]:
        seq_len = input_ids.shape[1]
        states_by_position = {}

        if self.verbose:
            print(f"Extracting states (single-pass) for {seq_len} positions...")

        with torch.no_grad():
            past_key_values = None

            for pos in range(seq_len):
                if pos == 0:
                    current_ids = input_ids[:, :1]
                else:
                    current_ids = input_ids[:, pos : pos + 1]

                outputs = self.model(
                    current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                position_states = {}
                if outputs.past_key_values is not None:
                    for layer_idx, layer_state in enumerate(outputs.past_key_values):
                        if layers is not None and layer_idx not in layers:
                            continue
                        if layer_state is not None and "recurrent_state" in layer_state:
                            position_states[layer_idx] = layer_state["recurrent_state"].detach().cpu().clone()

                states_by_position[pos + 1] = position_states
                past_key_values = copy.deepcopy(outputs.past_key_values)

                if self.verbose and (pos + 1) % 50 == 0:
                    print(f"  Position {pos + 1}/{seq_len}")

        if self.verbose:
            num_layers = len(states_by_position.get(1, {}))
            print(f"Extracted states for {seq_len} positions, {num_layers} layers each")

        return states_by_position


def save_states_to_file(states: Dict, filepath: str):
    _, ext = os.path.splitext(filepath)

    if ext == ".npz":
        states_np = {f"layer_{k}": v.numpy() if isinstance(v, torch.Tensor) else v for k, v in states.items()}
        np.savez(filepath, **states_np)
        print(f"Saved states to {filepath}")

    elif ext == ".h5":
        if not HAS_H5PY:
            raise ImportError("h5py not installed. Install with: pip install h5py")

        with h5py.File(filepath, "w") as f:
            for layer_idx, state in states.items():
                state_np = state.numpy() if isinstance(state, torch.Tensor) else state
                f.create_dataset(f"layer_{layer_idx}", data=state_np)

        print(f"Saved states to {filepath}")

    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .npz or .h5")


def load_states_from_file(filepath: str) -> Dict[int, np.ndarray]:
    _, ext = os.path.splitext(filepath)

    if ext == ".npz":
        data = np.load(filepath)
        states = {}
        for key in data.keys():
            layer_idx = int(key.split("_")[1])
            states[layer_idx] = data[key]
        print(f"Loaded states from {filepath}: {len(states)} layers")
        return states

    elif ext == ".h5":
        if not HAS_H5PY:
            raise ImportError("h5py not installed. Install with: pip install h5py")

        states = {}
        with h5py.File(filepath, "r") as f:
            for key in f.keys():
                layer_idx = int(key.split("_")[1])
                states[layer_idx] = f[key][:]

        print(f"Loaded states from {filepath}: {len(states)} layers")
        return states

    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .npz or .h5")
