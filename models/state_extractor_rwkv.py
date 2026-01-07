import copy
import torch
from typing import Dict, List, Optional
import warnings
from tqdm import tqdm


class RWKVStateExtractor:
    """State extractor for RWKV-7 models."""
    
    def __init__(self, model, verbose: bool = True):
        self.model = model
        self.verbose = verbose
        self.n_layer = model.model.n_layer
        self.n_head = model.model.n_head
        self.head_size = model.model.head_size
    
    def extract_final_states(self, input_ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Extract final recurrent states after processing the entire sequence.
        
        Args:
            input_ids: (batch_size, seq_len) or (seq_len,) tensor of token IDs
            
        Returns:
            Dict mapping layer_idx to state tensor of shape (n_head, head_size, head_size)
        """
        if input_ids.dim() == 2:
            input_ids = input_ids[0]
        
        token_list = input_ids.tolist()
        
        with torch.no_grad():
            _, state = self.model.model.forward(token_list, None)
        
        states = {}
        for i in range(self.n_layer):
            # State structure: [att_x_prev, att_kv, ffn_x_prev] per layer
            # att_kv is the recurrent state at index i*3+1
            states[i] = state[i*3+1].detach().cpu()
        
        return states
    
    def extract_final_states_batched(self, input_ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Extract final states - for RWKV this is same as extract_final_states
        since RWKV doesn't support batched inference natively.
        """
        return self.extract_final_states(input_ids)
    
    def extract_incremental_states_single_pass(
        self,
        input_ids: torch.Tensor,
        layers: Optional[List[int]] = None,
        use_tqdm: bool = False,
    ) -> Dict[int, Dict[int, torch.Tensor]]:
        """
        Extract states at every position in the sequence using single-token forward passes.
        
        Args:
            input_ids: (batch_size, seq_len) or (seq_len,) tensor of token IDs
            layers: Optional list of layer indices to extract (None = all layers)
            use_tqdm: Whether to show progress bar
            
        Returns:
            Dict mapping position -> Dict mapping layer_idx -> state tensor
        """
        if input_ids.dim() == 2:
            input_ids = input_ids[0]
        
        token_list = input_ids.tolist()
        seq_len = len(token_list)
        states_by_position = {}
        
        if self.verbose and not use_tqdm:
            print(f"Extracting states (single-pass) for {seq_len} positions...")
        
        state = None
        positions = range(seq_len)
        if use_tqdm:
            positions = tqdm(positions, desc="Tokens", leave=False)
        
        with torch.no_grad():
            for pos in positions:
                token = token_list[pos]
                _, state = self.model.model.forward([token], state)
                
                position_states = {}
                for layer_idx in range(self.n_layer):
                    if layers is not None and layer_idx not in layers:
                        continue
                    # att_kv state is at index layer_idx*3+1
                    position_states[layer_idx] = state[layer_idx*3+1].detach().cpu().clone()
                
                states_by_position[pos + 1] = position_states
                
                if self.verbose and not use_tqdm and (pos + 1) % 50 == 0:
                    print(f"  Position {pos + 1}/{seq_len}")
        
        if self.verbose and not use_tqdm:
            num_layers = len(states_by_position.get(1, {}))
            print(f"Extracted states for {seq_len} positions, {num_layers} layers each")
        
        return states_by_position
    
    def get_state_shape(self) -> tuple:
        """Return the shape of the recurrent state per layer."""
        return (self.n_head, self.head_size, self.head_size)
    
    def get_full_state(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """
        Get the full state list (including x_prev states) after processing input.
        
        Returns the raw state list from RWKV forward pass.
        """
        if input_ids.dim() == 2:
            input_ids = input_ids[0]
        
        token_list = input_ids.tolist()
        
        with torch.no_grad():
            _, state = self.model.model.forward(token_list, None)
        
        return [s.detach().cpu() for s in state]

