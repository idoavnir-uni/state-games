import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .names import FAMOUS_NAMES
from .animal_names import ANIMAL_NAMES

FIXED_ENTITY_NAME = "Lady Gaga"

DEFAULT_ANIMALS = [
    "Cat",
    "Dog",
    "Rabbit",
    "Horse",
    "Dolphin",
    "Eagle",
    "Wolf",
    "Bear",
    "Lion",
    "Tiger",
]


def generate_random_name(exclude: set, names_pool: List[str]) -> str:
    available = [n for n in names_pool if n not in exclude]
    if not available:
        raise ValueError("Not enough unique names available")
    return random.choice(available)


@dataclass
class FavoriteAnimalSample:
    text: str
    input_ids: Any
    fixed_entity_animal: str
    fixed_entity_sentence_end_token_idx: int
    fixed_entity_sentence_number: int


class FavoriteAnimalDataset:
    def __init__(
        self,
        tokenizer,
        size: int = 1000,
        n_entities: int = 3,
        n_animals: int = 3,
        animals: Optional[List[str]] = None,
        fixed_entity_name: Optional[str] = None,
        names_pool: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.size = size
        self.n_entities = n_entities
        self.fixed_entity_name = fixed_entity_name if fixed_entity_name is not None else FIXED_ENTITY_NAME
        self.names_pool = names_pool if names_pool is not None else FAMOUS_NAMES
        self.seed = seed
        
        if animals is not None:
            self.animals = animals
        else:
            self.animals = DEFAULT_ANIMALS[:n_animals]
        
        self.n_animals = len(self.animals)
        
        if seed is not None:
            random.seed(seed)
        
        self.samples = [self._generate_sample() for _ in range(size)]
    
    def _generate_sample(self) -> FavoriteAnimalSample:
        fixed_position = random.randint(0, self.n_entities - 1)
        fixed_animal = random.choice(self.animals)
        
        used_names = {self.fixed_entity_name}
        sentences = []
        
        for i in range(self.n_entities):
            if i == fixed_position:
                name = self.fixed_entity_name
                animal = fixed_animal
            else:
                name = generate_random_name(used_names, self.names_pool)
                used_names.add(name)
                animal = random.choice(self.animals)
            
            sentences.append(f"{name}'s favorite animal is {animal}.")
        
        text = " ".join(sentences)
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        
        prefix_text = " ".join(sentences[:fixed_position + 1])
        prefix_ids = self.tokenizer(prefix_text, return_tensors="pt").input_ids
        sentence_end_token_idx = prefix_ids.shape[1] - 1
        
        return FavoriteAnimalSample(
            text=text,
            input_ids=input_ids,
            fixed_entity_animal=fixed_animal,
            fixed_entity_sentence_end_token_idx=sentence_end_token_idx,
            fixed_entity_sentence_number=fixed_position,
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> FavoriteAnimalSample:
        return self.samples[idx]
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        tokens = self.tokenizer.convert_ids_to_tokens(sample.input_ids[0])
        return {
            "text": sample.text,
            "tokens": tokens,
            "input_ids_shape": sample.input_ids.shape,
            "fixed_entity_animal": sample.fixed_entity_animal,
            "fixed_entity_sentence_end_token_idx": sample.fixed_entity_sentence_end_token_idx,
            "fixed_entity_sentence_number": sample.fixed_entity_sentence_number,
            "token_at_sentence_end": tokens[sample.fixed_entity_sentence_end_token_idx],
        }

