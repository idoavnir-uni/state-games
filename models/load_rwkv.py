import os
import torch
from typing import Dict, Optional, Tuple, List
import warnings

os.environ["RWKV_V7_ON"] = '1'
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0'

from rwkv.model import RWKV_x070
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from huggingface_hub import hf_hub_download


DEFAULT_MODEL_NAME = "rwkv7-g1a-0.4b-20250905-ctx4096.pth"
REPO_ID = "BlinkDL/rwkv7-g1"
TOKENIZER_NAME = "rwkv_vocab_v20230424"


class TokenizerOutput:
    """Simple object to mimic HuggingFace tokenizer output."""
    def __init__(self, input_ids):
        self.input_ids = input_ids


class RWKVTokenizerWrapper:
    """Wrapper to make RWKV tokenizer compatible with HuggingFace-style interface."""
    
    def __init__(self, pipeline: PIPELINE):
        self.pipeline = pipeline
        self.tokenizer = pipeline.tokenizer
    
    def __call__(self, text: str, return_tensors: str = "pt", **kwargs) -> TokenizerOutput:
        token_ids = self.pipeline.encode(text)
        if return_tensors == "pt":
            input_ids = torch.tensor([token_ids], dtype=torch.long)
        else:
            input_ids = token_ids
        return TokenizerOutput(input_ids)
    
    def encode(self, text: str) -> List[int]:
        return self.pipeline.encode(text)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.pipeline.decode(token_ids)
    
    def convert_ids_to_tokens(self, token_ids) -> List[str]:
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        return [self.tokenizer.decode([tid]) for tid in token_ids]


class RWKVModelWrapper:
    """Wrapper to provide a consistent interface for RWKV models."""
    
    def __init__(self, model: RWKV_x070, pipeline: PIPELINE, device: str, dtype: torch.dtype):
        self.model = model
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype
        self.config = self._build_config()
    
    def _build_config(self):
        class Config:
            pass
        cfg = Config()
        cfg.num_hidden_layers = self.model.n_layer
        cfg.num_layers = self.model.n_layer
        cfg.num_heads = self.model.n_head
        cfg.head_size = self.model.head_size
        cfg.hidden_size = self.model.n_embd
        cfg.vocab_size = self.model.args.vocab_size
        cfg.max_position_embeddings = 4096
        cfg.model_type = "rwkv7"
        return cfg
    
    def forward(self, input_ids, state=None, full_output=False):
        """Forward pass compatible with RWKV interface."""
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 2:
                input_ids = input_ids[0].tolist()
            else:
                input_ids = input_ids.tolist()
        return self.model.forward(input_ids, state, full_output)
    
    def generate(self, ctx: str, token_count: int = 100, temperature: float = 0.3, 
                 top_p: float = 0.3, callback=None) -> str:
        """Generate text from a prompt."""
        args = PIPELINE_ARGS(
            temperature=temperature,
            top_p=top_p,
            top_k=0,
            alpha_frequency=0,
            alpha_presence=0,
            alpha_decay=0.996,
            token_ban=[],
            token_stop=[],
            chunk_len=256
        )
        return self.pipeline.generate(ctx, token_count=token_count, args=args, callback=callback)
    
    def generate_next_token(self, input_ids, state=None) -> Tuple[torch.Tensor, List]:
        """Generate logits for next token."""
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 2:
                input_ids = input_ids[0].tolist()
            else:
                input_ids = input_ids.tolist()
        out, new_state = self.model.forward(input_ids, state)
        return out, new_state
    
    def parameters(self):
        """Return model parameters for counting."""
        return self.model.z.values()
    
    def eval(self):
        return self
    
    def to(self, device):
        return self


def load_rwkv_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
) -> Tuple[RWKVModelWrapper, RWKVTokenizerWrapper]:
    """
    Load RWKV-7 model with interface compatible with load_gla_model.
    
    Args:
        model_name: Model filename from BlinkDL/rwkv7-g1 repo
        device: Device to load model on (cuda/cpu)
        torch_dtype: Data type for model weights
        
    Returns:
        Tuple of (model_wrapper, tokenizer_wrapper)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            warnings.warn("CUDA not available. Loading model on CPU. This will be slow.")
    
    dtype_str = "fp16" if torch_dtype == torch.float16 else "fp32" if torch_dtype == torch.float32 else "bf16"
    strategy = f"{device} {dtype_str}"
    
    print(f"Loading RWKV model: {model_name}")
    print(f"Device: {device}, dtype: {torch_dtype}")
    
    pth_path = hf_hub_download(REPO_ID, model_name)
    model_prefix = pth_path[:-4] if pth_path.endswith('.pth') else pth_path
    
    model = RWKV_x070(model=model_prefix, strategy=strategy)
    pipeline = PIPELINE(model, TOKENIZER_NAME)
    
    tokenizer = RWKVTokenizerWrapper(pipeline)
    model_wrapper = RWKVModelWrapper(model, pipeline, device, torch_dtype)
    
    num_params = sum(p.numel() for p in model.z.values())
    print(f"Model loaded successfully!")
    print(f"Parameters: {num_params / 1e9:.2f}B")
    print(f"Layers: {model.n_layer}, Heads: {model.n_head}, Hidden: {model.n_embd}")
    
    return model_wrapper, tokenizer


def get_model_config(model: RWKVModelWrapper) -> Dict:
    """Extract model configuration."""
    config = {}
    
    if hasattr(model, 'config'):
        cfg = model.config
        config["num_layers"] = getattr(cfg, "num_layers", None)
        config["num_heads"] = getattr(cfg, "num_heads", None)
        config["head_size"] = getattr(cfg, "head_size", None)
        config["hidden_size"] = getattr(cfg, "hidden_size", None)
        config["vocab_size"] = getattr(cfg, "vocab_size", None)
        config["max_seq_len"] = getattr(cfg, "max_position_embeddings", None)
        config["model_type"] = getattr(cfg, "model_type", None)
        config["full_config"] = cfg
    
    print("\n=== Model Configuration ===")
    for key, value in config.items():
        if key != "full_config":
            print(f"{key}: {value}")
    print("===========================\n")
    
    return config


def print_model_structure(model: RWKVModelWrapper, max_depth: int = 3):
    """Print RWKV model structure."""
    print("\n=== RWKV Model Structure ===")
    
    if hasattr(model, 'model') and hasattr(model.model, 'z'):
        z = model.model.z
        
        block_keys = sorted(set(k.split('.')[1] for k in z.keys() if k.startswith('blocks.')))
        print(f"Embedding: emb.weight {z['emb.weight'].shape}")
        print(f"Blocks: {len(block_keys)} layers")
        
        if max_depth >= 2 and block_keys:
            first_block = block_keys[0]
            block_keys_list = [k for k in z.keys() if k.startswith(f'blocks.{first_block}.')]
            print(f"  Block {first_block} components:")
            
            att_keys = [k for k in block_keys_list if '.att.' in k]
            ffn_keys = [k for k in block_keys_list if '.ffn.' in k]
            ln_keys = [k for k in block_keys_list if '.ln' in k]
            
            if max_depth >= 3:
                for k in sorted(ln_keys)[:3]:
                    print(f"    {k.split(f'blocks.{first_block}.')[-1]}: {z[k].shape}")
                print(f"    att: {len(att_keys)} parameters")
                print(f"    ffn: {len(ffn_keys)} parameters")
        
        print(f"Output: ln_out.weight, ln_out.bias, head.weight")
    
    print("===========================\n")

