# %% [markdown]
# # Test Model Completion
# 
# This notebook tests if the model can complete sentences correctly:
# 1. Take sentences with fixed entity color information
# 2. Append a prompt: "\n\nQuestion: What is Lady Gaga's favorite color?\nAnswer:"
# 3. Check if the model completion matches the correct color
#
# Note: We use a Q&A format which is common for base models to perform zero-shot extraction.

# %%
print("Importing libraries...")
import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

#os.environ['CUDA_VISIBLE_DEVICES'] = '6'

sys.path.insert(0, os.path.abspath('..'))

from models.load_gla import load_gla_model, get_model_config
from datasets.favorite_color_dataset import FavoriteColorDataset

print("Imports complete!")
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# %% [markdown]
# ## 1. Load Model and Tokenizer

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

print("Loading GLA model...")
model, tokenizer = load_gla_model(
    model_name="fla-hub/Qwen2.5-7B-Instruct",
    device=device,
    torch_dtype=torch.bfloat16
)

config = get_model_config(model)
print(f"Model loaded: {config.get('num_layers')} layers, {config.get('num_heads')} heads")

# %% [markdown]
# ## 2. Create Dataset

# %%
DATASET_SIZE = 100
N_ENTITIES = 10
N_COLORS = 10

FIXED_ENTITY = "Lady Gaga"

print(f"Creating dataset with {DATASET_SIZE} samples...")
dataset = FavoriteColorDataset(
    tokenizer=tokenizer,
    size=DATASET_SIZE,
    n_entities=N_ENTITIES,
    n_colors=N_COLORS,
    fixed_entity_name=FIXED_ENTITY,
    seed=42,
)
print(f"Dataset created with {len(dataset)} samples")
print(f"Fixed entity: {dataset.fixed_entity_name}")
print(f"Colors used: {dataset.colors}")

# %% [markdown]
# ## 3. Verify Colors are Single Tokens
# Note: We encode with a leading space since that's how colors appear after "Answer:"

# %%
print("\nVerifying colors are single tokens (with leading space)...")
color_token_ids = {}

for color in dataset.colors:
    # Encode with leading space since model output after "Answer:" will have space
    tokens_with_space = tokenizer.encode(f" {color}", add_special_tokens=False)
    tokens_no_space = tokenizer.encode(color, add_special_tokens=False)
    
    print(f"  {color}:")
    print(f"    with space: {tokens_with_space} -> '{tokenizer.decode(tokens_with_space)}'")
    print(f"    no space:   {tokens_no_space} -> '{tokenizer.decode(tokens_no_space)}'")
    
    # Use the first token of the space-prefixed version (usually the color with space merged)
    if len(tokens_with_space) == 1:
        color_token_ids[color] = tokens_with_space[0]
    elif len(tokens_no_space) == 1:
        # Fallback to no-space version if space version is multi-token
        color_token_ids[color] = tokens_no_space[0]
    else:
        raise ValueError(f"Color '{color}' is not a single token in either form")

COLOR_TOKEN_IDS = list(color_token_ids.values())
print(f"\nUsing token IDs: {COLOR_TOKEN_IDS}")
print("✓ Color tokens ready!")

# %% [markdown]
# ## 4. Test Model Completion
# 
# For each sample:
# 1. Take the original text (which contains "Lady Gaga's favorite color is {color}.")
# 2. Append prompt
# 3. Get model's next token prediction
# 4. Check if it matches the correct color

# %%
# Using a Q&A format that works well for base models
QUERY_TEMPLATE = f"\n\nQuestion: What is {FIXED_ENTITY}'s favorite color?\nAnswer:"

print(f"\nQuery template: '{QUERY_TEMPLATE}'")

# Print first example to show what we're testing
print("\n" + "="*80)
print("EXAMPLE INPUT (Sample 0):")
print("="*80)
sample_0 = dataset[0]
full_text_example = sample_0.text + QUERY_TEMPLATE
print(f"\nOriginal text:\n{sample_0.text}")
print(f"\nQuery appended:\n'{QUERY_TEMPLATE}'")
print(f"\nFull input to model:\n{full_text_example}")
print(f"\nTrue color: {sample_0.fixed_entity_color}")
print(f"Info given at sentence: {sample_0.fixed_entity_sentence_number}")
print("="*80 + "\n")

# First, let's debug what the model actually outputs for sample 0
print("\n=== DEBUG: What does the model predict for sample 0? ===")
with torch.no_grad():
    debug_input = tokenizer(full_text_example, return_tensors="pt").input_ids.to(device)
    debug_out = model(debug_input)
    debug_logits = debug_out.logits[0, -1, :]
    
    # Top 10 predictions overall
    top10_ids = debug_logits.topk(10).indices.tolist()
    top10_logits = debug_logits.topk(10).values.tolist()
    print("Top 10 predicted tokens:")
    for i, (tid, logit) in enumerate(zip(top10_ids, top10_logits)):
        token_str = tokenizer.decode(tid)
        print(f"  {i+1}. '{token_str}' (id={tid}, logit={logit:.2f})")
    
    # Check color token logits
    print("\nColor token logits:")
    for color, tid in color_token_ids.items():
        logit = debug_logits[tid].item()
        print(f"  '{color}' (id={tid}): logit={logit:.2f}")

print("="*60 + "\n")

print(f"Testing completion on {DATASET_SIZE} samples...\n")

results = []
correct_top1 = 0
correct_top3 = 0

for idx in tqdm(range(len(dataset)), desc="Testing completions"):
    sample = dataset[idx]
    true_color = sample.fixed_entity_color
    true_color_token_id = color_token_ids[true_color]
    
    # Original text + query
    full_text = sample.text + QUERY_TEMPLATE
    input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    
    # Get model prediction (next token logits)
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]  # Logits for next token
    
    # Get top predictions among all tokens
    top_token_id = next_token_logits.argmax().item()
    top_token = tokenizer.decode(top_token_id)
    
    # Get logits for color tokens specifically
    color_logits = next_token_logits[COLOR_TOKEN_IDS]
    color_probs = torch.softmax(color_logits, dim=0)
    
    # Get top-3 color predictions
    top3_color_indices = color_logits.topk(3).indices.tolist()
    top3_colors = [dataset.colors[i] for i in top3_color_indices]
    top3_probs = [color_probs[i].item() for i in top3_color_indices]
    
    # Check accuracy
    pred_color_idx = color_logits.argmax().item()
    pred_color = dataset.colors[pred_color_idx]
    is_correct_top1 = (pred_color == true_color)
    is_correct_top3 = (true_color in top3_colors)
    
    if is_correct_top1:
        correct_top1 += 1
    if is_correct_top3:
        correct_top3 += 1
    
    results.append({
        'idx': idx,
        'text': sample.text[:100] + "...",
        'true_color': true_color,
        'pred_color': pred_color,
        'is_correct': is_correct_top1,
        'top_token_overall': top_token,
        'top3_colors': top3_colors,
        'top3_probs': top3_probs,
        'info_sentence_num': sample.fixed_entity_sentence_number,
    })

# %% [markdown]
# ## 5. Results Summary

# %%
accuracy_top1 = correct_top1 / len(dataset)
accuracy_top3 = correct_top3 / len(dataset)
random_baseline_top1 = 1 / N_COLORS
random_baseline_top3 = 3 / N_COLORS

print(f"\n{'='*60}")
print(f"COMPLETION TEST RESULTS")
print(f"{'='*60}")
print(f"Total samples: {len(dataset)}")
print(f"")
print(f"Top-1 Accuracy: {accuracy_top1:.3f} ({correct_top1}/{len(dataset)})")
print(f"Top-3 Accuracy: {accuracy_top3:.3f} ({correct_top3}/{len(dataset)})")
print(f"")
print(f"Random baseline (top-1): {random_baseline_top1:.3f}")
print(f"Random baseline (top-3): {random_baseline_top3:.3f}")
print(f"{'='*60}\n")

# %% [markdown]
# ## 6. Show Example Predictions

# %%
print("=== EXAMPLE PREDICTIONS ===\n")

# Show first 5 correct and first 5 incorrect
correct_examples = [r for r in results if r['is_correct']][:5]
incorrect_examples = [r for r in results if not r['is_correct']][:5]

print("--- CORRECT PREDICTIONS ---\n")
for r in correct_examples:
    print(f"Sample {r['idx']}: True={r['true_color']}, Pred={r['pred_color']} ✓")
    print(f"  Top-3: {r['top3_colors']} (probs: {[f'{p:.2f}' for p in r['top3_probs']]})")
    print(f"  Info given at sentence: {r['info_sentence_num']}")
    print()

print("\n--- INCORRECT PREDICTIONS ---\n")
for r in incorrect_examples:
    print(f"Sample {r['idx']}: True={r['true_color']}, Pred={r['pred_color']} ✗")
    print(f"  Top-3: {r['top3_colors']} (probs: {[f'{p:.2f}' for p in r['top3_probs']]})")
    print(f"  Info given at sentence: {r['info_sentence_num']}")
    print(f"  Text: {r['text']}")
    print()

# %% [markdown]
# ## 7. Accuracy by Position of Information

# %%
results_df = pd.DataFrame(results)

print("\n=== ACCURACY BY INFORMATION POSITION ===\n")
accuracy_by_position = results_df.groupby('info_sentence_num')['is_correct'].agg(['mean', 'count'])
accuracy_by_position.columns = ['accuracy', 'count']
print(accuracy_by_position)

# %%
import matplotlib.pyplot as plt

positions = accuracy_by_position.index.tolist()
accuracies = accuracy_by_position['accuracy'].tolist()
counts = accuracy_by_position['count'].tolist()

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(positions, accuracies, alpha=0.7, label='Accuracy')
ax1.axhline(y=random_baseline_top1, color='r', linestyle='--', label=f'Random ({random_baseline_top1:.2f})')
ax1.set_xlabel('Information Sentence Position')
ax1.set_ylabel('Accuracy')
ax1.set_title(f"Model Completion Accuracy by Information Position\n(Query: Q: What is {FIXED_ENTITY}'s favorite color? A:)")
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(positions, counts, 'ko-', alpha=0.5, label='Sample count')
ax2.set_ylabel('Sample Count')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('../data/completion_accuracy_by_position.png', dpi=150)
plt.show()

print(f"\nPlot saved to ../data/completion_accuracy_by_position.png")

# %%
