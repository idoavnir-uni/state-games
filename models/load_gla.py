import torch
from typing import Dict, Optional, Tuple
import warnings
import json

from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

def load_gla_model(
    model_name: str = "fla-hub/gla-1.3B-100B",
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.nn.Module, object]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            warnings.warn("CUDA not available. Loading model on CPU. This will be very slow for large models.")

    print(f"Loading model: {model_name}")
    print(f"Device: {device}, dtype: {torch_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Download and parse config.json directly to avoid transformers issues
    try:
        config_path = hf_hub_download(model_name, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        model_type = config_dict.get('model_type', None)
        architectures = config_dict.get('architectures', [])
    except Exception as e:
        print(f"Could not fetch config.json: {e}")
        model_type = None
        architectures = []
    
    print(f"Detected model_type: {model_type}, architectures: {architectures}")
    
    model = None
    
    # Try fla models based on architecture/model_type
    if 'TransformerForCausalLM' in architectures or model_type == 'transformer':
        try:
            from fla.models.transformer import TransformerForCausalLM
            model = TransformerForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch_dtype,
                attn_implementation="eager"  # Avoid flash attention requirement
            )
            print("Loaded using fla TransformerForCausalLM")
        except Exception as e:
            print(f"fla TransformerForCausalLM failed: {e}")
    
    if model is None and ('GLAForCausalLM' in architectures or model_type == 'gla'):
        try:
            from fla.models.gla import GLAForCausalLM
            model = GLAForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
            print("Loaded using fla GLAForCausalLM")
        except Exception as e:
            print(f"fla GLAForCausalLM failed: {e}")
    
    # Fallback to AutoModelForCausalLM
    if model is None:
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch_dtype, 
                trust_remote_code=True,
                attn_implementation="eager"  # Avoid flash attention requirement
            )
            print("Loaded using AutoModelForCausalLM")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully!")
    print(f"Parameters: {num_params / 1e9:.2f}B")
    print(f"Memory footprint: ~{num_params * 2 / 1e9:.2f} GB (bfloat16)")

    return model, tokenizer


def get_model_config(model: torch.nn.Module) -> Dict:
    config = {}

    if hasattr(model, "config"):
        cfg = model.config
        config["num_layers"] = getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", None))
        config["num_heads"] = getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", None))
        config["hidden_size"] = getattr(cfg, "hidden_size", None)
        config["vocab_size"] = getattr(cfg, "vocab_size", None)
        config["max_seq_len"] = getattr(cfg, "max_position_embeddings", getattr(cfg, "max_seq_len", None))
        config["expand_k"] = getattr(cfg, "expand_k", None)
        config["expand_v"] = getattr(cfg, "expand_v", None)
        config["attn_mode"] = getattr(cfg, "attn_mode", None)
        config["use_short_conv"] = getattr(cfg, "use_short_conv", None)
        config["conv_size"] = getattr(cfg, "conv_size", None)
        config["full_config"] = cfg
    else:
        warnings.warn("Model does not have a 'config' attribute. Cannot extract architecture details.")

    if config.get("num_layers") is None:
        if hasattr(model, "layers"):
            config["num_layers"] = len(model.layers)
        elif hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            config["num_layers"] = len(model.decoder.layers)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            config["num_layers"] = len(model.model.layers)

    print("\n=== Model Configuration ===")
    for key, value in config.items():
        if key != "full_config":
            print(f"{key}: {value}")
    print("===========================\n")

    return config


def print_model_structure(model: torch.nn.Module, max_depth: int = 3):
    print("\n=== Model Structure ===")

    def print_modules(module, prefix="", depth=0):
        if depth >= max_depth:
            return
        for name, child in module.named_children():
            print(f"{prefix}{name}: {child.__class__.__name__}")
            print_modules(child, prefix=prefix + "  ", depth=depth + 1)

    print_modules(model)
    print("=======================\n")
