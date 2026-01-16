"""Test script for CombinedDataset with various configurations.

Run from project root: python -m datasets.test_combined_dataset
"""

from transformers import AutoTokenizer

from datasets.combined_dataset import (
    CombinedDataset,
    DatasetConfig,
    FAVORITE_COLOR_CONFIG,
    FAVORITE_ANIMAL_CONFIG,
    MOST_HATED_CITY_CONFIG,
)


def print_sample(dataset, idx=0, title=""):
    """Helper to print sample details."""
    if title:
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
    
    info = dataset.get_sample_info(idx)
    print(f"Text: {info['text']}")
    print(f"Fixed entity source: {info['fixed_entity_source']}")
    print(f"Fixed entity value: {info['fixed_entity_value']}")
    print(f"Fixed entity sentence #: {info['fixed_entity_sentence_number']}")
    print(f"Sentence sources: {info['sentence_sources']}")
    print(f"Sentences breakdown:")
    for i, (sent, src) in enumerate(zip(info['sentences'], info['sentence_sources'])):
        marker = " <-- FIXED" if i == info['fixed_entity_sentence_number'] else ""
        print(f"  [{src}] {sent}{marker}")


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Test 1: Basic 2-dataset combination with equal distribution
    print("\n" + "="*80)
    print(" TEST 1: Two datasets (color + animal), equal distribution")
    print("="*80)
    
    configs = {
        "color": FAVORITE_COLOR_CONFIG,
        "animal": FAVORITE_ANIMAL_CONFIG,
    }
    
    dataset = CombinedDataset(
        tokenizer=tokenizer,
        dataset_configs=configs,
        size=3,
        total_sentences=4,  # 2 from each
        seed=42,
    )
    
    for i in range(len(dataset)):
        print_sample(dataset, i, f"Sample {i+1}")
    
    # Test 2: Three datasets with custom sentence counts
    print("\n" + "="*80)
    print(" TEST 2: Three datasets with explicit sentence counts")
    print("="*80)
    
    configs = {
        "color": FAVORITE_COLOR_CONFIG,
        "animal": FAVORITE_ANIMAL_CONFIG,
        "city": MOST_HATED_CITY_CONFIG,
    }
    
    dataset = CombinedDataset(
        tokenizer=tokenizer,
        dataset_configs=configs,
        size=3,
        sentences_per_config={"color": 2, "animal": 1, "city": 1},
        seed=123,
    )
    
    for i in range(len(dataset)):
        print_sample(dataset, i, f"Sample {i+1}")
    
    # Test 3: Using ratios with fixed entity source
    print("\n" + "="*80)
    print(" TEST 3: Ratios + fixed entity from 'animal' dataset")
    print("="*80)
    
    dataset = CombinedDataset(
        tokenizer=tokenizer,
        dataset_configs=configs,
        size=3,
        sentence_ratios={"color": 0.5, "animal": 0.25, "city": 0.25},
        total_sentences=4,
        fixed_entity_source="animal",
        seed=456,
    )
    
    for i in range(len(dataset)):
        print_sample(dataset, i, f"Sample {i+1}")
    
    # Test 4: No shuffle - predictable ordering
    print("\n" + "="*80)
    print(" TEST 4: No shuffle, fixed entity from 'city'")
    print("="*80)
    
    dataset = CombinedDataset(
        tokenizer=tokenizer,
        dataset_configs=configs,
        size=2,
        sentences_per_config={"color": 1, "animal": 1, "city": 1},
        fixed_entity_source="city",
        shuffle_sentences=False,
        seed=789,
    )
    
    for i in range(len(dataset)):
        print_sample(dataset, i, f"Sample {i+1}")
    
    # Test 5: Custom dataset config
    print("\n" + "="*80)
    print(" TEST 5: Custom dataset config (favorite food)")
    print("="*80)
    
    custom_food_config = DatasetConfig(
        sentence_template="{name} loves eating {value}.",
        values=["pizza", "sushi", "tacos", "pasta", "burgers"],
        fixed_entity_name="Gordon Ramsay",
        names_pool=["Tom Hanks", "Taylor Swift", "Elon Musk", "Beyonce", "Brad Pitt"],
        value_key="food",
    )
    
    configs_with_custom = {
        "color": FAVORITE_COLOR_CONFIG,
        "food": custom_food_config,
    }
    
    dataset = CombinedDataset(
        tokenizer=tokenizer,
        dataset_configs=configs_with_custom,
        size=3,
        sentences_per_config={"color": 1, "food": 2},
        fixed_entity_source="food",
        seed=999,
    )
    
    for i in range(len(dataset)):
        print_sample(dataset, i, f"Sample {i+1}")
    
    # Test 6: Large sample with many sentences
    print("\n" + "="*80)
    print(" TEST 6: Large sample (6 sentences from 3 datasets)")
    print("="*80)
    
    dataset = CombinedDataset(
        tokenizer=tokenizer,
        dataset_configs={
            "color": FAVORITE_COLOR_CONFIG,
            "animal": FAVORITE_ANIMAL_CONFIG,
            "city": MOST_HATED_CITY_CONFIG,
        },
        size=2,
        sentences_per_config={"color": 2, "animal": 2, "city": 2},
        seed=111,
    )
    
    for i in range(len(dataset)):
        print_sample(dataset, i, f"Sample {i+1}")
        info = dataset.get_sample_info(i)
        print(f"\n  Token count: {info['input_ids_shape']}")
    
    print("\n" + "="*80)
    print(" All tests completed!")
    print("="*80)


if __name__ == "__main__":
    main()

