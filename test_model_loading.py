#!/usr/bin/env python3
"""
Quick test script to verify RetNet model loading works correctly.
This script tests the Flash Linear Attention library integration.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def test_imports():
    """Test that required libraries can be imported."""
    print("Testing imports...")
    try:
        import torch

        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found")
        return False

    try:
        import transformers

        print(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not found")
        return False

    try:
        from fla.models.retnet import RetNetForCausalLM

        print("✓ Flash Linear Attention (RetNet)")
    except ImportError as e:
        print(f"✗ Flash Linear Attention not found: {e}")
        print("\nPlease install with:")
        print("  pip install git+https://github.com/sustcsonglin/flash-linear-attention.git")
        return False

    return True


def test_model_loading():
    """Test that the model can be loaded."""
    print("\n" + "=" * 60)
    print("Testing RetNet model loading...")
    print("=" * 60)

    try:
        import torch
        from models.load_retnet import load_retnet_model

        # Try loading with CPU (faster for testing)
        print("\nAttempting to load RetNet-2.7B...")
        print("Note: This will download ~5GB if not cached.")

        model, tokenizer = load_retnet_model(
            model_name="fla-hub/retnet-2.7B-100B", device="cpu", torch_dtype=torch.bfloat16  # Use CPU for testing
        )

        print("\n✓ Model loaded successfully!")

        # Quick inference test
        print("\nTesting inference...")
        test_text = "Hello, world!"
        inputs = tokenizer(test_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        print(f"✓ Inference successful! Output shape: {outputs.logits.shape}")

        return True

    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("RetNet Model Loading Test")
    print("=" * 60)

    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install required dependencies.")
        sys.exit(1)

    print("\n✓ All imports successful!")

    # Ask user if they want to test model loading
    print("\n" + "=" * 60)
    print("Model loading test (will download ~5GB if not cached)")
    response = input("Do you want to test model loading? (y/n): ")

    if response.lower() == "y":
        if test_model_loading():
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Model loading test failed.")
            sys.exit(1)
    else:
        print("\nSkipping model loading test.")
        print("✅ Import tests passed!")
