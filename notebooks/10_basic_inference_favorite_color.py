# %% [markdown]
# # Basic Inference with Favorite Color Dataset
#
# Tests whether the model can correctly answer questions about favorite colors from context.

# %%
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets.favorite_color_dataset import FavoriteColorDataset
from models.load_gla import load_gla_model

# %%
model, tokenizer = load_gla_model(model_name="fla-hub/gla-2.7B-100B")

# %%
dataset = FavoriteColorDataset(
    tokenizer=tokenizer,
    size=1000,
    n_entities=3,
    seed=42,
)

# %%
PROMPT_TEMPLATE = """Answer the question based on the context below. Keep the answer short.

Context: {context}

Question: What is the favorite color of {entity}?

Answer: The favorite color of {entity} is """

ENTITY_NAME = dataset.fixed_entity_name


# %%
def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 20) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# %%
print("=" * 80)
print("Running Inference on 5 Samples")
print("=" * 80)

correct = 0
total = len(dataset)

for i in range(total):
    sample = dataset[i]
    prompt = PROMPT_TEMPLATE.format(context=sample.text, entity=ENTITY_NAME)

    generated = run_inference(model, tokenizer, prompt)
    expected_color = sample.fixed_entity_color

    generated_clean = generated.split('"')[0].strip().lower()
    is_correct = expected_color.lower() in generated_clean

    if is_correct:
        correct += 1

    print(f"\n--- Sample {i} ---")
    print(f"Prompt:\n{prompt}")
    print(f"Expected: {expected_color}")
    print(f"Generated: {generated}")
    print(f"Correct: {'✓' if is_correct else '✗'}")

# %%
print("\n" + "=" * 80)
print(f"Results: {correct}/{total} correct ({100 * correct / total:.1f}%)")
print("=" * 80)

# %%
