import random
from typing import List, Dict, Any, Optional, Union, Type
from dataclasses import dataclass

from .names import FAMOUS_NAMES
from .animal_names import ANIMAL_NAMES


def generate_random_name(exclude: set, names_pool: List[str]) -> str:
    available = [n for n in names_pool if n not in exclude]
    if not available:
        raise ValueError("Not enough unique names available")
    return random.choice(available)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset type in the combined dataset."""
    sentence_template: str  # e.g., "{name}'s favorite color is {value}."
    values: List[str]  # e.g., ["red", "blue", "green"]
    fixed_entity_name: str  # e.g., "Jeff Bezos"
    names_pool: List[str]  # Pool of names to sample from
    value_key: str  # Key to use in the output for this dataset's fixed value


@dataclass
class CombinedSample:
    text: str
    input_ids: Any
    fixed_entity_value: str
    fixed_entity_sentence_end_token_idx: int
    fixed_entity_sentence_number: int
    fixed_entity_source: str  # Which dataset config the fixed entity came from
    sentences: List[str]  # Individual sentences for inspection
    sentence_sources: List[str]  # Which config each sentence came from


class CombinedDataset:
    """
    A dataset that combines sentences from multiple dataset configurations.
    
    Each sample contains sentences from different "types" (e.g., favorite color, 
    favorite animal, most hated city) combined into a single text.
    """
    
    def __init__(
        self,
        tokenizer,
        dataset_configs: Dict[str, DatasetConfig],
        size: int = 1000,
        sentences_per_config: Optional[Dict[str, int]] = None,
        sentence_ratios: Optional[Dict[str, float]] = None,
        total_sentences: int = 3,
        fixed_entity_source: Optional[str] = None,
        shuffle_sentences: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            tokenizer: The tokenizer to use for encoding.
            dataset_configs: Dict mapping config names to DatasetConfig objects.
            size: Number of samples to generate.
            sentences_per_config: Exact number of sentences per config. 
                                  If provided, overrides sentence_ratios.
            sentence_ratios: Ratio of sentences from each config (should sum to 1).
                            Used with total_sentences to determine counts.
            total_sentences: Total sentences per sample (used with sentence_ratios).
            fixed_entity_source: Which config to use for the fixed entity.
                                If None, randomly selects one.
            shuffle_sentences: Whether to shuffle sentence order in each sample.
            seed: Random seed for reproducibility.
        """
        self.tokenizer = tokenizer
        self.dataset_configs = dataset_configs
        self.size = size
        self.shuffle_sentences = shuffle_sentences
        self.seed = seed
        self.fixed_entity_source = fixed_entity_source
        
        # Determine sentences per config
        if sentences_per_config is not None:
            self.sentences_per_config = sentences_per_config
        elif sentence_ratios is not None:
            self.sentences_per_config = self._ratios_to_counts(
                sentence_ratios, total_sentences
            )
        else:
            # Default: equal distribution
            n_configs = len(dataset_configs)
            per_config = total_sentences // n_configs
            remainder = total_sentences % n_configs
            self.sentences_per_config = {}
            for i, name in enumerate(dataset_configs.keys()):
                self.sentences_per_config[name] = per_config + (1 if i < remainder else 0)
        
        self.total_sentences = sum(self.sentences_per_config.values())
        
        if seed is not None:
            random.seed(seed)
        
        self.samples = [self._generate_sample() for _ in range(size)]
    
    def _ratios_to_counts(
        self, ratios: Dict[str, float], total: int
    ) -> Dict[str, int]:
        """Convert ratios to actual counts, ensuring they sum to total."""
        counts = {}
        running_total = 0
        items = list(ratios.items())
        
        for i, (name, ratio) in enumerate(items[:-1]):
            count = round(ratio * total)
            counts[name] = count
            running_total += count
        
        # Last item gets the remainder to ensure exact total
        counts[items[-1][0]] = total - running_total
        return counts
    
    def _generate_sample(self) -> CombinedSample:
        # Choose which config provides the fixed entity
        if self.fixed_entity_source is not None:
            fixed_source = self.fixed_entity_source
        else:
            fixed_source = random.choice(list(self.dataset_configs.keys()))
        
        fixed_config = self.dataset_configs[fixed_source]
        fixed_value = random.choice(fixed_config.values)
        
        # Track used names across all configs to avoid duplicates
        used_names = {fixed_config.fixed_entity_name}
        
        sentences = []
        sentence_sources = []
        fixed_sentence_idx = None
        
        # Generate sentences for each config
        for config_name, config in self.dataset_configs.items():
            n_sentences = self.sentences_per_config.get(config_name, 0)
            
            for _ in range(n_sentences):
                is_fixed = (config_name == fixed_source and fixed_sentence_idx is None)
                
                if is_fixed:
                    name = config.fixed_entity_name
                    value = fixed_value
                    fixed_sentence_idx = len(sentences)
                else:
                    name = generate_random_name(used_names, config.names_pool)
                    used_names.add(name)
                    value = random.choice(config.values)
                
                sentence = config.sentence_template.format(name=name, value=value)
                sentences.append(sentence)
                sentence_sources.append(config_name)
        
        # Shuffle if requested, tracking fixed entity position
        if self.shuffle_sentences:
            combined = list(zip(sentences, sentence_sources, range(len(sentences))))
            random.shuffle(combined)
            sentences = [s for s, _, _ in combined]
            sentence_sources = [src for _, src, _ in combined]
            # Find new position of fixed sentence
            for i, (_, _, orig_idx) in enumerate(combined):
                if orig_idx == fixed_sentence_idx:
                    fixed_sentence_idx = i
                    break
        
        text = " ".join(sentences)
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        
        # Calculate token index for end of fixed entity sentence
        prefix_text = " ".join(sentences[:fixed_sentence_idx + 1])
        prefix_ids = self.tokenizer(prefix_text, return_tensors="pt").input_ids
        sentence_end_token_idx = prefix_ids.shape[1] - 1
        
        return CombinedSample(
            text=text,
            input_ids=input_ids,
            fixed_entity_value=fixed_value,
            fixed_entity_sentence_end_token_idx=sentence_end_token_idx,
            fixed_entity_sentence_number=fixed_sentence_idx,
            fixed_entity_source=fixed_source,
            sentences=sentences,
            sentence_sources=sentence_sources,
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> CombinedSample:
        return self.samples[idx]
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        tokens = self.tokenizer.convert_ids_to_tokens(sample.input_ids[0])
        return {
            "text": sample.text,
            "tokens": tokens,
            "input_ids_shape": sample.input_ids.shape,
            "fixed_entity_value": sample.fixed_entity_value,
            "fixed_entity_sentence_end_token_idx": sample.fixed_entity_sentence_end_token_idx,
            "fixed_entity_sentence_number": sample.fixed_entity_sentence_number,
            "fixed_entity_source": sample.fixed_entity_source,
            "token_at_sentence_end": tokens[sample.fixed_entity_sentence_end_token_idx],
            "sentences": sample.sentences,
            "sentence_sources": sample.sentence_sources,
        }


# Pre-configured dataset configs for convenience
FAVORITE_COLOR_CONFIG = DatasetConfig(
    sentence_template="{name}'s favorite color is {value}.",
    values=["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "brown"],
    fixed_entity_name="Jeff Bezos",
    names_pool=FAMOUS_NAMES,
    value_key="color",
)

FAVORITE_ANIMAL_CONFIG = DatasetConfig(
    sentence_template="{name}'s favorite animal is {value}.",
    values=["Cat", "Dog", "Bat", "Fox", "Ant", "Fly", "Rat", "Fish", "Wolf", "Spider"],
    fixed_entity_name="Lady Gaga",
    names_pool=FAMOUS_NAMES,
    value_key="animal",
)

MOST_HATED_CITY_CONFIG = DatasetConfig(
    sentence_template="{name} hates {value}.",
    values=["Paris", "London", "Chicago", "Boston", "Miami", "Austin", "Houston", "York", "Bern", "Gary"],
    fixed_entity_name="Aardvark",
    names_pool=ANIMAL_NAMES,
    value_key="city",
)

