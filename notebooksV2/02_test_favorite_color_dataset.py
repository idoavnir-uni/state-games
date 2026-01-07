# %% [markdown]
# # Favorite Color Dataset Test with RWKV
# 
# Tests the FavoriteColorDataset with RWKV model to check if it correctly
# answers questions about the fixed entity's favorite color.

# %%
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('..'))

from models.load_rwkv import load_rwkv_model
from datasets.favorite_color_dataset import FavoriteColorDataset

# %%
print("Loading RWKV model...")
model, tokenizer = load_rwkv_model(model_name="rwkv7-g1b-1.5b-20251202-ctx8192.pth")

# %% [markdown]
# ## Dataset Examples

# %%
print("\nCreating dataset with Lady Gaga as fixed entity:")
dataset = FavoriteColorDataset(
    tokenizer=tokenizer,
    size=10,
    n_entities=3,
    fixed_entity_name="Lady Gaga",
    n_colors=10,
    seed=42,
)

print(f"Dataset size: {len(dataset)}")
print(f"Fixed entity: {dataset.fixed_entity_name}")

# %%
print("\n" + "=" * 80)
print("Example Samples")
print("=" * 80)

for i in range(3):
    print(f"\n--- Sample {i} ---")
    info = dataset.get_sample_info(i)
    print(f"Text: {info['text']}")
    print(f"Fixed entity color: {info['fixed_entity_color']}")
    print(f"Sentence number: {info['fixed_entity_sentence_number']}")

# %% [markdown]
# ## Success Rate Testing
# 
# Test if the model correctly answers about Lady Gaga's favorite color
# for different numbers of entities in the context.

# %%
def make_prompt(context: str, entity_name: str = "Lady Gaga") -> str:
    """Create a prompt for asking about an entity's favorite color."""
    return (
        f"Given the following context, let's answer the question below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: What is {entity_name}'s favorite color?\n\n"
        f"Answer: {entity_name}'s favorite color is"
    )

def test_accuracy(n_entities: int, n_samples: int = 100, seed: int = 42):
    """Test model accuracy on FavoriteColorDataset."""
    dataset = FavoriteColorDataset(
        tokenizer=tokenizer,
        size=n_samples,
        n_entities=n_entities,
        fixed_entity_name="Lady Gaga",
        n_colors=10,
        seed=seed,
    )
    
    correct = 0
    
    for i in tqdm(range(n_samples), desc=f"n={n_entities:2d}", leave=False):
        sample = dataset[i]
        prompt = make_prompt(sample.text, "Lady Gaga")
        
        output = []
        def callback(s):
            output.append(s)
        model.generate(prompt, token_count=5, temperature=0.0, top_p=0.0, callback=callback)
        response = ''.join(output).strip().lower()
        
        if sample.fixed_entity_color.lower() in response:
            correct += 1
    
    accuracy = correct / n_samples * 100
    return accuracy, correct, n_samples

# %%
print("\n" + "=" * 80)
print("RWKV FavoriteColorDataset Test (Lady Gaga, 100 samples each)")
print("=" * 80 + "\n")

results = {}
for n in [3, 10, 20,100,300]:
    acc, correct, total = test_accuracy(n, n_samples=100)
    results[n] = {"accuracy": acc, "correct": correct, "total": total}
    print(f"n_entities={n:2d}: {correct:3d}/{total} correct ({acc:5.1f}%)")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
for n, r in results.items():
    print(f"n_entities={n:2d}: {r['accuracy']:5.1f}%")

# %% [markdown]
# ## Long Sequence Evaluation
# 
# Test on longer sequences where the information about Lady Gaga 
# appears at sentences 10-20 (early in a ~300 sentence context).

# %%
TARGET_EVAL_SAMPLES = 20
EVAL_N_ENTITIES = 30

print(f"\nGenerating {TARGET_EVAL_SAMPLES} evaluation samples with Lady Gaga at sentences 10-20...")
eval_samples = []
seed_offset = 43

pbar = tqdm(total=TARGET_EVAL_SAMPLES, desc="Generating valid samples")
while len(eval_samples) < TARGET_EVAL_SAMPLES:
    temp_dataset = FavoriteColorDataset(
        tokenizer=tokenizer,
        size=100,
        n_entities=EVAL_N_ENTITIES,
        n_colors=10,
        fixed_entity_name="Lady Gaga",
        seed=seed_offset,
    )
    
    for sample in temp_dataset:
        if 10 <= sample.fixed_entity_sentence_number <= 20:
            eval_samples.append(sample)
            pbar.update(1)
            if len(eval_samples) >= TARGET_EVAL_SAMPLES:
                break
    
    seed_offset += 1

pbar.close()
print(f"Generated {len(eval_samples)} valid samples")

# %%
print(f"\nTesting model on {TARGET_EVAL_SAMPLES} long sequences (~{EVAL_N_ENTITIES} sentences each)...")
print("Info about Lady Gaga appears at sentences 10-20\n")

long_correct = 0
for i, sample in enumerate(tqdm(eval_samples, desc="Evaluating")):
    prompt = make_prompt(sample.text, "Lady Gaga")
    
    output = []
    def callback(s):
        output.append(s)
    model.generate(prompt, token_count=5, temperature=0.0, top_p=0.0, callback=callback)
    response = ''.join(output).strip().lower()
    
    is_correct = sample.fixed_entity_color.lower() in response
    if is_correct:
        long_correct += 1
    
    if i < 5:  # Show first 5 examples
        print(f"  Sample {i}: sentence_num={sample.fixed_entity_sentence_number}, "
              f"color={sample.fixed_entity_color}, response='{response[:20]}...', "
              f"correct={is_correct}")

long_accuracy = long_correct / TARGET_EVAL_SAMPLES * 100
print(f"\n{'='*60}")
print(f"Long Sequence Results (info at sentences 10-20, ~{EVAL_N_ENTITIES} total sentences)")
print(f"{'='*60}")
print(f"Accuracy: {long_correct}/{TARGET_EVAL_SAMPLES} = {long_accuracy:.1f}%")

# %%
