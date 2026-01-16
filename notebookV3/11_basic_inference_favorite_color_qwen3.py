# %% [markdown]
# # Basic Inference with Favorite Color Dataset (Qwen2.5-Instruct)
#
# Tests whether the model can correctly answer questions about favorite colors from context.

# %%
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets.favorite_color_dataset import FavoriteColorDataset

# %%
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

# %%
dataset = FavoriteColorDataset(
    tokenizer=tokenizer,
    size=10,
    n_entities=3,
    seed=42,
)

# %%
SYSTEM_PROMPT = "You are a helpful assistant that responds in English. Answer the question based on the context. Reply with only the color name in English, nothing else."

USER_PROMPT_TEMPLATE = """Context: {context}

Question: What is the favorite color of {entity}?"""

ENTITY_NAME = dataset.fixed_entity_name


# %%
def run_inference(model, tokenizer, context: str, entity: str, max_new_tokens: int = 20) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(context=context, entity=entity)},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt")
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
print(f"Running Inference on {len(dataset)} Samples")
print("=" * 80)

correct = 0
total = len(dataset)

for i in range(total):
    sample = dataset[i]

    generated = run_inference(model, tokenizer, sample.text, ENTITY_NAME)
    expected_color = sample.fixed_entity_color

    generated_clean = generated.strip().lower()
    is_correct = expected_color.lower() in generated_clean

    if is_correct:
        correct += 1

    print(f"\n--- Sample {i} ---")
    print(f"Context: {sample.text}")
    print(f"Expected: {expected_color}")
    print(f"Generated: {generated}")
    print(f"Correct: {'✓' if is_correct else '✗'}")

# %%
print("\n" + "=" * 80)
print(f"Results: {correct}/{total} correct ({100 * correct / total:.1f}%)")
print("=" * 80)

# %%
