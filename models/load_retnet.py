import torch
from typing import Dict, Optional, Tuple
import warnings

from fla.models.retnet import RetNetForCausalLM
from transformers import AutoTokenizer


def load_retnet_model(
    model_name: str = "fla-hub/retnet-2.7B-100B",
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.nn.Module, object]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            warnings.warn("CUDA not available. Loading model on CPU. This will be very slow for the 2.7B model.")

    print(f"Loading RetNet model: {model_name}")
    print(f"Device: {device}, dtype: {torch_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = RetNetForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
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
        config["state_size"] = getattr(cfg, "state_size", config.get("hidden_size"))
        config["decoder_embed_dim"] = getattr(cfg, "decoder_embed_dim", None)
        config["decoder_retention_heads"] = getattr(cfg, "decoder_retention_heads", None)
        config["decoder_layers"] = getattr(cfg, "decoder_layers", None)
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


def test_inference(model: torch.nn.Module, tokenizer, test_text: str = "The quick brown fox"):
    print(f"\n=== Testing Inference ===")
    print(f"Input: '{test_text}'")

    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)

    print(f"Input shape: {input_ids.shape}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

    with torch.no_grad():
        outputs = model(input_ids)

    print(f"Output type: {type(outputs)}")
    if hasattr(outputs, "last_hidden_state"):
        print(f"Hidden state shape: {outputs.last_hidden_state.shape}")
    elif isinstance(outputs, torch.Tensor):
        print(f"Output shape: {outputs.shape}")
    else:
        print(f"Output keys: {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}")

    print("=========================\n")

    return outputs


if __name__ == "__main__":
    print("Loading RetNet model...")
    model, tokenizer = load_retnet_model(device="cuda")

    config = get_model_config(model)
    print_model_structure(model, max_depth=2)
    test_inference(model, tokenizer, "Hello, how are you?")
