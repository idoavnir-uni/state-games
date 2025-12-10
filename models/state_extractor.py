import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import warnings
from collections import defaultdict

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class RetNetStateExtractor:
    def __init__(self, model: nn.Module, verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self.states = {}
        self.hooks = []
        self.retention_modules = []

    def _find_retention_modules(self) -> List[Tuple[int, nn.Module]]:
        retention_modules = []

        layers = None
        if hasattr(self.model, "layers"):
            layers = self.model.layers
        elif hasattr(self.model, "decoder") and hasattr(self.model.decoder, "layers"):
            layers = self.model.decoder.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers

        if layers is not None:
            for layer_idx, layer in enumerate(layers):
                retention_module = None

                if hasattr(layer, "retention"):
                    retention_module = layer.retention
                elif hasattr(layer, "attn"):
                    retention_module = layer.attn
                elif hasattr(layer, "self_attn"):
                    retention_module = layer.self_attn

                if retention_module is not None:
                    retention_modules.append((layer_idx, retention_module))
                    if self.verbose:
                        print(f"Found retention module at layer {layer_idx}: {retention_module.__class__.__name__}")

        if len(retention_modules) == 0:
            if self.verbose:
                print("Strategy 1 failed. Searching by module name patterns...")

            for name, module in self.model.named_modules():
                if "retention" in name.lower() or ("attn" in name.lower() and "layer" in name.lower()):
                    parts = name.split(".")
                    layer_idx = None
                    for part in parts:
                        if part.isdigit():
                            layer_idx = int(part)
                            break

                    if layer_idx is not None:
                        retention_modules.append((layer_idx, module))
                        if self.verbose:
                            print(f"Found module by name search: {name} -> layer {layer_idx}")

        if len(retention_modules) == 0:
            warnings.warn(
                "Could not automatically locate retention modules. "
                "You may need to manually specify them using set_retention_modules()."
            )

        retention_modules.sort(key=lambda x: x[0])
        return retention_modules

    def set_retention_modules(self, modules: List[Tuple[int, nn.Module]]):
        self.retention_modules = modules
        if self.verbose:
            print(f"Manually set {len(modules)} retention modules")

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            try:
                state = None

                if isinstance(output, tuple) and len(output) >= 3:
                    past_key_values = output[2]
                    if past_key_values is not None and hasattr(past_key_values, "key_value_states"):
                        if layer_idx < len(past_key_values.key_value_states):
                            layer_state = past_key_values.key_value_states[layer_idx]
                            if layer_state is not None and "recurrent_state" in layer_state:
                                state = layer_state["recurrent_state"]

                if state is None and isinstance(output, tuple):
                    for item in output:
                        if isinstance(item, torch.Tensor) and item.dim() in [3, 4]:
                            state = item
                            break

                if state is None and isinstance(output, dict):
                    for key in ["state", "retention_state", "recurrent_state", "past_key_value"]:
                        if key in output:
                            state = output[key]
                            break

                if state is None:
                    for attr_name in ["state", "retention_state", "recurrent_state", "_state"]:
                        if hasattr(module, attr_name):
                            attr = getattr(module, attr_name)
                            if isinstance(attr, torch.Tensor):
                                state = attr
                                break

                if state is not None:
                    self.states[layer_idx] = state.detach().cpu()
                else:
                    if self.verbose and layer_idx not in self.states:
                        warnings.warn(
                            f"Could not extract state from layer {layer_idx}. "
                            f"Output type: {type(output)}, "
                            f"Module: {module.__class__.__name__}"
                        )

            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Error in hook for layer {layer_idx}: {str(e)}")

        return hook_fn

    def register_hooks(self):
        if len(self.retention_modules) == 0:
            self.retention_modules = self._find_retention_modules()

        self.remove_hooks()

        for layer_idx, module in self.retention_modules:
            hook = module.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)

        if self.verbose:
            print(f"Registered hooks on {len(self.hooks)} retention modules")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_states(self, input_ids: torch.Tensor, use_cache: bool = True) -> Dict[int, torch.Tensor]:
        self.states = {}

        with torch.no_grad():
            try:
                outputs = self.model(input_ids, use_cache=use_cache)

                if use_cache and hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                    past_kv = outputs.past_key_values
                    if hasattr(past_kv, "key_value_states"):
                        for layer_idx, layer_state in enumerate(past_kv.key_value_states):
                            if layer_state is not None and isinstance(layer_state, dict):
                                if "recurrent_state" in layer_state:
                                    self.states[layer_idx] = layer_state["recurrent_state"].detach().cpu()
            except Exception as e:
                warnings.warn(f"Error during forward pass: {str(e)}")
                raise

        if len(self.states) == 0:
            warnings.warn(
                "No states were captured! The hooks may not be working correctly. "
                "Check that retention modules are correctly identified."
            )

        return self.states

    def __del__(self):
        self.remove_hooks()


def extract_states_at_positions(
    model: nn.Module,
    input_ids: torch.Tensor,
    positions: List[int],
    layers: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[Tuple[int, int], torch.Tensor]:
    extractor = RetNetStateExtractor(model, verbose=verbose)
    extractor.register_hooks()

    states = extractor.extract_states(input_ids)

    if layers is not None:
        states = {k: v for k, v in states.items() if k in layers}

    if verbose:
        print(
            "Note: Position-specific extraction not yet implemented. "
            "Returning full sequence states. "
            "You can slice them by position afterward."
        )

    result = {}
    for layer_idx, state in states.items():
        for pos in positions:
            result[(layer_idx, pos)] = state

    extractor.remove_hooks()
    return result


def save_states_to_file(states: Dict, filepath: str):
    import os

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
    import os

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
    from load_retnet import load_retnet_model

    print("Loading model...")
    model, tokenizer = load_retnet_model(device="cuda")

    print("\nCreating state extractor...")
    extractor = RetNetStateExtractor(model, verbose=True)
    extractor.register_hooks()

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
