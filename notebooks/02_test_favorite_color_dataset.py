# %% [markdown]
# # Favorite Color Dataset Test
# 
# This notebook tests the FavoriteColorDataset which generates sentences like:
# "SOME_NAME favorite color is SOME_COLOR. SOME_NAME2 favorite color is SOME_COLOR2. ..."

# %%
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from datasets.favorite_color_dataset import FavoriteColorDataset
from transformers import AutoTokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained("fla-hub/gla-1.3B-100B")

# %%
print("Testing with default entity (Jeff Bezos):")
dataset = FavoriteColorDataset(
    tokenizer=tokenizer,
    size=1000,
    n_entities=3,
    seed=41,
)

print(f"Dataset size: {len(dataset)}")
print(f"Fixed entity: {dataset.fixed_entity_name}")

# %%
print("\nTesting with custom entity (Lady Gaga):")
dataset_gaga = FavoriteColorDataset(
    tokenizer=tokenizer,
    size=1000,
    n_entities=3,
    fixed_entity_name="Lady Gaga",
    seed=43,
)

print(f"Dataset size: {len(dataset_gaga)}")
print(f"Fixed entity: {dataset_gaga.fixed_entity_name}")

# %%
print("=" * 80)
print("Example Samples")
print("=" * 80)

for i in range(3):
    print(f"\n--- Sample {i} (Jeff Bezos) ---")
    info = dataset.get_sample_info(i)
    
    print(f"Text: {info['text']}")
    print(f"Fixed entity color: {info['fixed_entity_color']}")
    print(f"Sentence number: {info['fixed_entity_sentence_number']}")

print("\n" + "=" * 80)
print("Example Samples (Lady Gaga)")
print("=" * 80)

for i in range(3):
    print(f"\n--- Sample {i} (Lady Gaga) ---")
    info = dataset_gaga.get_sample_info(i)
    
    print(f"Text: {info['text']}")
    print(f"Fixed entity color: {info['fixed_entity_color']}")
    print(f"Sentence number: {info['fixed_entity_sentence_number']}")


# %%
