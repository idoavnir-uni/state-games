import os
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

    def extract_states_incremental(
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
                partial_ids = input_ids[:, :pos]
                outputs = self.model(partial_ids, use_cache=True)

                position_states = {}
                if outputs.past_key_values is not None:
                    for layer_idx, layer_state in enumerate(outputs.past_key_values):
                        if layers is not None and layer_idx not in layers:
                            continue
                        if layer_state is not None and "recurrent_state" in layer_state:
                            position_states[layer_idx] = layer_state["recurrent_state"].detach().cpu()

                states_by_position[pos] = position_states

                if self.verbose and pos % 10 == 0:
                    print(f"  Position {pos}/{seq_len}")

        if self.verbose:
            num_layers = len(states_by_position.get(1, {}))
            print(f"Extracted states for {seq_len} positions, {num_layers} layers each")

        return states_by_position

    def extract_states_at_positions(
        self,
        input_ids: torch.Tensor,
        positions: List[int],
        layers: Optional[List[int]] = None,
    ) -> Dict[int, Dict[int, torch.Tensor]]:
        seq_len = input_ids.shape[1]
        states_by_position = {}

        positions = sorted(set(positions))
        for pos in positions:
            if pos < 1 or pos > seq_len:
                warnings.warn(f"Position {pos} out of range [1, {seq_len}], skipping")
                continue

        if self.verbose:
            print(f"Extracting states at positions: {positions}")

        with torch.no_grad():
            for pos in positions:
                if pos < 1 or pos > seq_len:
                    continue

                partial_ids = input_ids[:, :pos]
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
            num_layers = len(states_by_position.get(positions[0], {})) if positions else 0
            print(f"Extracted states for {len(states_by_position)} positions, {num_layers} layers each")

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


if __name__ == "__main__":
    from models.load_retnet import load_retnet_model

    print("Loading model...")
    model, tokenizer = load_retnet_model(device="cuda")

    print("\nCreating state extractor...")
    extractor = RetNetStateExtractor(model, verbose=True)

    print("\nRunning inference...")
    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    states = extractor.extract_states(input_ids)

    print(f"\n=== Extracted States ===")
    for layer_idx, state in states.items():
        print(f"Layer {layer_idx}: shape = {state.shape}, dtype = {state.dtype}")

    print("\nSaving states...")
    save_states_to_file(states, "test_states.npz")

    print("\nLoading states...")
    loaded_states = load_states_from_file("test_states.npz")
