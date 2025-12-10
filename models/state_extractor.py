"""
State extraction utilities for RetNet.

This module implements mechanisms to extract retention states from RetNet models
during forward passes. The states can then be used for interpretability analysis
and linear probing.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import warnings
from collections import defaultdict


class RetNetStateExtractor:
    """
    Extracts retention states from RetNet model using PyTorch hooks.

    This class registers forward hooks on retention layers to capture the
    recurrent state S_t at each layer during inference.

    Example:
        >>> model, tokenizer = load_retnet_model()
        >>> extractor = RetNetStateExtractor(model)
        >>> extractor.register_hooks()
        >>>
        >>> input_ids = tokenizer("Hello world", return_tensors="pt").input_ids
        >>> states = extractor.extract_states(input_ids)
        >>> print(f"Extracted states for {len(states)} layers")
    """

    def __init__(self, model: nn.Module, verbose: bool = True):
        """
        Initialize the state extractor.

        Args:
            model: RetNet model to extract states from
            verbose: Whether to print debug information
        """
        self.model = model
        self.verbose = verbose
        self.states = {}  # {layer_idx: state_tensor}
        self.hooks = []
        self.retention_modules = []

    def _find_retention_modules(self) -> List[Tuple[int, nn.Module]]:
        """
        Locate retention modules in the model.

        Returns:
            List of (layer_index, module) tuples
        """
        retention_modules = []

        # Strategy 1: Look for model.layers or model.decoder.layers
        layers = None
        if hasattr(self.model, "layers"):
            layers = self.model.layers
        elif hasattr(self.model, "decoder") and hasattr(self.model.decoder, "layers"):
            layers = self.model.decoder.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers

        if layers is not None:
            for layer_idx, layer in enumerate(layers):
                # Look for retention/attention module in this layer
                retention_module = None

                # Common naming patterns
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

        # Strategy 2: Search by module name pattern
        if len(retention_modules) == 0:
            if self.verbose:
                print("Strategy 1 failed. Searching by module name patterns...")

            for name, module in self.model.named_modules():
                if "retention" in name.lower() or ("attn" in name.lower() and "layer" in name.lower()):
                    # Try to extract layer index from name
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

        # Sort by layer index
        retention_modules.sort(key=lambda x: x[0])

        return retention_modules

    def set_retention_modules(self, modules: List[Tuple[int, nn.Module]]):
        """
        Manually set the retention modules if automatic detection fails.

        Args:
            modules: List of (layer_index, retention_module) tuples
        """
        self.retention_modules = modules
        if self.verbose:
            print(f"Manually set {len(modules)} retention modules")

    def _make_hook(self, layer_idx: int):
        """
        Create a forward hook function for a specific layer.

        The hook extracts the retention state from the module's forward pass.

        Args:
            layer_idx: Layer index for tracking

        Returns:
            Hook function
        """

        def hook_fn(module, input, output):
            """
            Hook function that captures retention states.

            The state could be in various places depending on implementation:
            1. As part of the output tuple
            2. As a module attribute (e.g., module.state)
            3. Within the output dictionary
            """
            try:
                state = None

                # Strategy 1: Output is a tuple and state is one element
                if isinstance(output, tuple):
                    # Common pattern: (hidden_states, state) or (hidden_states, state, ...)
                    if len(output) >= 2:
                        # The state is usually not the hidden states
                        # Try to identify by shape: state should be [batch, heads, d_k, d_v]
                        for item in output[1:]:
                            if isinstance(item, torch.Tensor):
                                # State typically has 4 dimensions: [batch, heads, d_k, d_v]
                                # or 3 dimensions: [batch, heads, state_dim]
                                if item.dim() in [3, 4]:
                                    state = item
                                    break

                # Strategy 2: Output is a dictionary
                elif isinstance(output, dict):
                    # Look for keys like 'state', 'retention_state', 'recurrent_state'
                    for key in ["state", "retention_state", "recurrent_state", "past_key_value"]:
                        if key in output:
                            state = output[key]
                            break

                # Strategy 3: Check module attributes
                if state is None:
                    for attr_name in ["state", "retention_state", "recurrent_state", "_state"]:
                        if hasattr(module, attr_name):
                            attr = getattr(module, attr_name)
                            if isinstance(attr, torch.Tensor):
                                state = attr
                                break

                # Strategy 4: Output is the state directly (less common)
                if state is None and isinstance(output, torch.Tensor):
                    if output.dim() in [3, 4]:
                        state = output

                # Store the state
                if state is not None:
                    # Detach and move to CPU to save memory
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
        """
        Register forward hooks on all retention layers.

        This should be called once before running inference.
        """
        # Find retention modules if not already found
        if len(self.retention_modules) == 0:
            self.retention_modules = self._find_retention_modules()

        # Clear existing hooks
        self.remove_hooks()

        # Register hooks
        for layer_idx, module in self.retention_modules:
            hook = module.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)

        if self.verbose:
            print(f"Registered hooks on {len(self.hooks)} retention modules")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_states(self, input_ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Run forward pass and collect retention states from all layers.

        Args:
            input_ids: Input token IDs [batch, seq_len]

        Returns:
            Dictionary mapping layer_idx -> state tensor
            State shape is typically [batch, num_heads, d_k, d_v] or [batch, num_heads, state_dim]
        """
        # Clear previous states
        self.states = {}

        # Run forward pass (hooks will capture states)
        with torch.no_grad():
            try:
                _ = self.model(input_ids)
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
        """Cleanup: remove hooks when object is destroyed."""
        self.remove_hooks()


def extract_states_at_positions(
    model: nn.Module,
    input_ids: torch.Tensor,
    positions: List[int],
    layers: Optional[List[int]] = None,
    verbose: bool = False,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Extract retention states at specific positions for specific layers.

    Note: This is a simplified implementation. For position-specific extraction,
    you may need to run the model in recurrent mode token-by-token.

    Args:
        model: RetNet model
        input_ids: Input token IDs [batch, seq_len]
        positions: List of positions to extract states from
        layers: List of layer indices (None = all layers)
        verbose: Print debug information

    Returns:
        Dict mapping (layer, position) -> state tensor
    """
    extractor = RetNetStateExtractor(model, verbose=verbose)
    extractor.register_hooks()

    # Extract states for the full sequence
    states = extractor.extract_states(input_ids)

    # Filter by requested layers
    if layers is not None:
        states = {k: v for k, v in states.items() if k in layers}

    # Note: Position-specific extraction requires more sophisticated handling
    # For now, we return the full states with a warning
    if verbose:
        print(
            "Note: Position-specific extraction not yet implemented. "
            "Returning full sequence states. "
            "You can slice them by position afterward."
        )

    result = {}
    for layer_idx, state in states.items():
        for pos in positions:
            result[(layer_idx, pos)] = state  # Will need to slice by position

    extractor.remove_hooks()
    return result


def save_states_to_file(states: Dict, filepath: str):
    """
    Save extracted states to disk in HDF5 or numpy format.

    Args:
        states: Dictionary of states from extract_states()
        filepath: Output file path (.h5 or .npz)
    """
    import os

    _, ext = os.path.splitext(filepath)

    if ext == ".npz":
        # Convert to numpy and save
        states_np = {f"layer_{k}": v.numpy() if isinstance(v, torch.Tensor) else v for k, v in states.items()}
        np.savez(filepath, **states_np)
        print(f"Saved states to {filepath}")

    elif ext == ".h5":
        # Save to HDF5
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py not installed. Install with: pip install h5py")

        with h5py.File(filepath, "w") as f:
            for layer_idx, state in states.items():
                state_np = state.numpy() if isinstance(state, torch.Tensor) else state
                f.create_dataset(f"layer_{layer_idx}", data=state_np)

        print(f"Saved states to {filepath}")

    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .npz or .h5")


def load_states_from_file(filepath: str) -> Dict[int, np.ndarray]:
    """
    Load previously extracted states from disk.

    Args:
        filepath: Input file path (.h5 or .npz)

    Returns:
        Dictionary mapping layer_idx -> state array
    """
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
        try:
            import h5py
        except ImportError:
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
    # Example usage
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

    # Save states
    print("\nSaving states...")
    save_states_to_file(states, "test_states.npz")

    # Load states
    print("\nLoading states...")
    loaded_states = load_states_from_file("test_states.npz")
