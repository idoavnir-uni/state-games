import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.load_retnet import load_retnet_model, get_model_config
from models.state_extractor import RetNetStateExtractor, save_states_to_file, load_states_from_file


@pytest.fixture(scope="module")
def model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_retnet_model(device=device)
    return model, tokenizer


@pytest.fixture(scope="module")
def extractor(model_and_tokenizer):
    model, _ = model_and_tokenizer
    extractor = RetNetStateExtractor(model, verbose=True)
    extractor.register_hooks()
    return extractor


def test_state_extraction(extractor, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = "Hello, world!"
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    states = extractor.extract_states(input_ids)

    assert isinstance(states, dict)
    assert len(states) > 0


def test_state_shapes(extractor, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    states = extractor.extract_states(input_ids)

    for layer_idx, state in states.items():
        assert isinstance(state, torch.Tensor)
        assert state.dim() in [3, 4]
        assert state.shape[0] == input_ids.shape[0]


def test_state_differs_by_input(extractor, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    input_ids1 = tokenizer("Hello, world!", return_tensors="pt").input_ids.to(model.device)
    input_ids2 = tokenizer("Goodbye, world!", return_tensors="pt").input_ids.to(model.device)

    states1 = extractor.extract_states(input_ids1)
    states2 = extractor.extract_states(input_ids2)

    for layer_idx in states1.keys():
        if layer_idx in states2:
            assert not torch.allclose(states1[layer_idx], states2[layer_idx])


def test_save_load_states(extractor, model_and_tokenizer, tmp_path):
    model, tokenizer = model_and_tokenizer
    input_ids = tokenizer("Test sentence.", return_tensors="pt").input_ids.to(model.device)
    states = extractor.extract_states(input_ids)

    filepath = tmp_path / "test_states.npz"
    save_states_to_file(states, str(filepath))
    loaded_states = load_states_from_file(str(filepath))

    assert len(loaded_states) == len(states)
    for layer_idx in states.keys():
        assert layer_idx in loaded_states
        assert torch.allclose(torch.from_numpy(loaded_states[layer_idx]), states[layer_idx])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_retnet_model(device=device)
    extractor = RetNetStateExtractor(model, verbose=True)
    extractor.register_hooks()

    test_state_extraction(extractor, (model, tokenizer))
    test_state_shapes(extractor, (model, tokenizer))
    test_state_differs_by_input(extractor, (model, tokenizer))

    print("✓ All tests passed!")
