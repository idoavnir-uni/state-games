"""
Unit tests for state extraction functionality.

Run with: pytest test_state_shapes.py
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.load_retnet import load_retnet_model, get_model_config
from models.state_extractor import RetNetStateExtractor, save_states_to_file, load_states_from_file


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model once for all tests."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_retnet_model(device=device)
    return model, tokenizer


@pytest.fixture(scope="module")
def extractor(model_and_tokenizer):
    """Create state extractor."""
    model, _ = model_and_tokenizer
    extractor = RetNetStateExtractor(model, verbose=True)
    extractor.register_hooks()
    return extractor


def test_model_loads(model_and_tokenizer):
    """Test that model loads successfully."""
    model, tokenizer = model_and_tokenizer
    assert model is not None
    assert tokenizer is not None
    print("✓ Model loaded successfully")


def test_model_config(model_and_tokenizer):
    """Test that we can extract model configuration."""
    model, _ = model_and_tokenizer
    config = get_model_config(model)

    assert "num_layers" in config
    assert config["num_layers"] is not None
    assert config["num_layers"] > 0

    print(f"✓ Model config extracted: {config['num_layers']} layers")


def test_state_extraction_runs(extractor, model_and_tokenizer):
    """Test that state extraction completes without errors."""
    model, tokenizer = model_and_tokenizer

    text = "Hello, world!"
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    states = extractor.extract_states(input_ids)

    assert isinstance(states, dict)
    assert len(states) > 0

    print(f"✓ State extraction completed: {len(states)} layers")


def test_state_shapes(extractor, model_and_tokenizer):
    """Test that extracted states have correct dimensions."""
    model, tokenizer = model_and_tokenizer

    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    states = extractor.extract_states(input_ids)

    for layer_idx, state in states.items():
        assert isinstance(state, torch.Tensor)
        assert state.dim() in [3, 4], f"Layer {layer_idx}: Expected 3 or 4 dimensions, got {state.dim()}"
        assert state.shape[0] == input_ids.shape[0], f"Layer {layer_idx}: Batch size mismatch"

        print(f"✓ Layer {layer_idx}: shape = {state.shape}")


def test_state_changes_with_input(extractor, model_and_tokenizer):
    """Test that states differ for different inputs."""
    model, tokenizer = model_and_tokenizer

    text1 = "Hello, world!"
    text2 = "Goodbye, world!"

    input_ids1 = tokenizer(text1, return_tensors="pt").input_ids.to(model.device)
    input_ids2 = tokenizer(text2, return_tensors="pt").input_ids.to(model.device)

    states1 = extractor.extract_states(input_ids1)
    states2 = extractor.extract_states(input_ids2)

    # Check that states are different
    for layer_idx in states1.keys():
        if layer_idx in states2:
            # States should not be identical
            assert not torch.allclose(
                states1[layer_idx], states2[layer_idx]
            ), f"Layer {layer_idx}: States are identical for different inputs!"

    print("✓ States change with different inputs")


def test_state_sequence_lengths(extractor, model_and_tokenizer):
    """Test state extraction on sequences of different lengths."""
    model, tokenizer = model_and_tokenizer

    test_texts = [
        "Short.",
        "This is a medium length sentence with more tokens.",
        "This is a much longer sentence that contains significantly more tokens and should test the model's ability to handle longer sequences properly.",
    ]

    for text in test_texts:
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        states = extractor.extract_states(input_ids)

        assert len(states) > 0
        print(f"✓ Sequence length {input_ids.shape[1]}: {len(states)} layers extracted")


def test_save_and_load_states(extractor, model_and_tokenizer, tmp_path):
    """Test saving and loading states."""
    model, tokenizer = model_and_tokenizer

    text = "Test sentence for saving."
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    states = extractor.extract_states(input_ids)

    # Test .npz format
    filepath = tmp_path / "test_states.npz"
    save_states_to_file(states, str(filepath))
    loaded_states = load_states_from_file(str(filepath))

    assert len(loaded_states) == len(states)
    for layer_idx in states.keys():
        assert layer_idx in loaded_states
        # Convert to numpy for comparison
        original = states[layer_idx].numpy()
        loaded = loaded_states[layer_idx]
        assert original.shape == loaded.shape
        assert torch.allclose(torch.from_numpy(loaded), states[layer_idx])

    print("✓ Save and load states works correctly")


def test_batch_processing(extractor, model_and_tokenizer):
    """Test state extraction with batch input."""
    model, tokenizer = model_and_tokenizer

    texts = [
        "First sentence.",
        "Second sentence.",
    ]

    # Tokenize with padding
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(model.device)

    states = extractor.extract_states(input_ids)

    assert len(states) > 0
    for layer_idx, state in states.items():
        assert state.shape[0] == len(
            texts
        ), f"Layer {layer_idx}: Expected batch size {len(texts)}, got {state.shape[0]}"

    print(f"✓ Batch processing works: batch_size={len(texts)}")


if __name__ == "__main__":
    # Run tests manually
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_retnet_model(device=device)

    print("\nCreating extractor...")
    extractor = RetNetStateExtractor(model, verbose=True)
    extractor.register_hooks()

    print("\n=== Running Tests ===\n")

    # Run each test
    test_model_loads((model, tokenizer))
    test_model_config((model, tokenizer))
    test_state_extraction_runs(extractor, (model, tokenizer))
    test_state_shapes(extractor, (model, tokenizer))
    test_state_changes_with_input(extractor, (model, tokenizer))
    test_state_sequence_lengths(extractor, (model, tokenizer))

    print("\n✓ All tests passed!")
