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
dataset = FavoriteColorDataset(
    tokenizer=tokenizer,
    size=1000,
    n_entities=3,
    seed=41,
)

print(f"Dataset size: {len(dataset)}")

# %%
print("=" * 80)
print("Example Samples")
print("=" * 80)

for i in range(3):
    print(f"\n--- Sample {i} ---")
    info = dataset.get_sample_info(i)
    
    print(f"Text: {info['text']}")
    print(f"Tokens: {info['tokens']}")
    print(f"Input IDs shape: {info['input_ids_shape']}")
    print(f"Fixed entity (Jeff Bezos) color: {info['fixed_entity_color']}")
    print(f"Sentence number with Jeff Bezos: {info['fixed_entity_sentence_number']}")
    print(f"Token index where Jeff Bezos sentence ends: {info['fixed_entity_sentence_end_token_idx']}")
    print(f"Token at that position: {info['token_at_sentence_end']}")


# %%
