import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import warnings
from tqdm import tqdm


class GLAStateExtractor:
    def __init__(self, model: nn.Module, verbose: bool = True):
        self.model = model
        self.verbose = verbose

    def extract_final_states(self, input_ids: torch.Tensor, use_cache: bool = True) -> Dict[int, torch.Tensor]:
        states = {}

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=use_cache)

            if outputs.past_key_values is not None:
                for layer_idx in range(len(outputs.past_key_values)):
                    layer_state = outputs.past_key_values[layer_idx]
                    if layer_state is not None and "recurrent_state" in layer_state:
                        states[layer_idx] = layer_state["recurrent_state"].detach().cpu()

        if len(states) == 0:
            warnings.warn("No states were captured! Check that use_cache=True and model supports caching.")

        return states

    def extract_incremental_states_single_pass(
        self,
        input_ids: torch.Tensor,
        layers: Optional[List[int]] = None,
        use_tqdm: bool = False,
    ) -> Dict[int, Dict[int, torch.Tensor]]:
        seq_len = input_ids.shape[1]
        states_by_position = {}

        if self.verbose and not use_tqdm:
            print(f"Extracting states (single-pass) for {seq_len} positions...")

        with torch.no_grad():
            past_key_values = None

            positions = range(seq_len)
            if use_tqdm:
                positions = tqdm(positions, desc="Tokens", leave=False)

            for pos in positions:
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
                    for layer_idx in range(len(outputs.past_key_values)):
                        if layers is not None and layer_idx not in layers:
                            continue
                        layer_state = outputs.past_key_values[layer_idx]
                        if layer_state is not None and "recurrent_state" in layer_state:
                            position_states[layer_idx] = layer_state["recurrent_state"].detach().cpu().clone()

                states_by_position[pos + 1] = position_states
                past_key_values = copy.deepcopy(outputs.past_key_values)

                if self.verbose and not use_tqdm and (pos + 1) % 50 == 0:
                    print(f"  Position {pos + 1}/{seq_len}")

        if self.verbose and not use_tqdm:
            num_layers = len(states_by_position.get(1, {}))
            print(f"Extracted states for {seq_len} positions, {num_layers} layers each")

        return states_by_position
