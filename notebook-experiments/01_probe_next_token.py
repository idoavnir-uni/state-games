# %% [markdown]
# # Probe Next Token After Fixed Entity
# 
# This experiment:
# 1. Creates sentences with single-token words (FIXED_ENTITY at varying positions)
# 2. Extracts RWKV states at the END of the full sentence
# 3. Trains probes to predict the token that was at position (FIXED_ENTITY + DIST) in the sentence
# 
# The goal: Can the model "remember" what token appeared DIST positions after FIXED_ENTITY,
# when queried at the end of the sentence?
# 
# **Probe model:** logits = (W_left @ state) @ w_right
# - state: (head_size, head_size) per head
# - W_left: (n_classes, head_size) learned matrix
# - w_right: (head_size,) learned vector

# %%
import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from dataclasses import dataclass   
from typing import List, Dict, Optional

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

sys.path.insert(0, os.path.abspath('..'))

from models.load_rwkv import load_rwkv_model, get_model_config
from models.state_extractor_rwkv import RWKVStateExtractor

print("Imports complete!")

# %%
# === CONFIGURATION ===
NUM_ENTITIES = 30
DIST = 10  # How many tokens after fixed entity to predict
DATASET_SIZE = 10000

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

print("\nLoading RWKV model...")
model, tokenizer = load_rwkv_model(model_name="rwkv7-g1b-1.5b-20251202-ctx8192.pth")

config = get_model_config(model)
num_layers = config.get('num_layers')
num_heads = config.get('num_heads')
print(f"Model: {num_layers} layers, {num_heads} heads")

extractor = RWKVStateExtractor(model, verbose=False)
head_size = extractor.head_size
n_embd = model.model.n_embd
print(f"Head size: {head_size}, Hidden size: {n_embd}")

# %%
# === FIND SINGLE-TOKEN WORDS ===
# We need words where " word" (with space prefix) is a single token
# This is how they appear in sentences

CANDIDATE_WORDS = [
    # Cities
    "Paris", "London", "Tokyo", "Berlin", "Madrid", "Rome", "Vienna", "Dublin",
    "Sydney", "Boston", "Denver", "Seattle", "Miami", "Dallas", "Austin",
    "Portland", "Phoenix", "Detroit", "Atlanta", "Chicago", "Houston",
    # Colors
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "gold", "silver",
    # Animals
    "dog", "cat", "bird", "fish", "lion", "tiger", "bear", "wolf", "fox",
    "deer", "mouse", "horse", "cow", "pig", "sheep", "goat", "duck",
    "eagle", "hawk", "owl", "snake", "frog", "crab", "shark", "whale",
    # Common nouns
    "book", "tree", "rock", "lake", "river", "ocean", "forest",
    "garden", "house", "castle", "bridge", "tower", "park", "beach",
    # Food
    "apple", "grape", "lemon", "cherry", "peach",
    "bread", "cheese", "butter", "milk", "coffee", "water", "wine", "beer",
]
print(len(CANDIDATE_WORDS))

# %%

print("Finding single-token words (with space prefix)...")
single_token_words = []
word_to_token_id = {}  # Maps word to its token ID (with space prefix)

for word in CANDIDATE_WORDS:
    # Check if " word" (with space) is a single token - this is how it appears in sentences
    tokens_with_space = tokenizer.encode(" " + word)
    
    if len(tokens_with_space) == 1:
        single_token_words.append(word)
        word_to_token_id[word] = tokens_with_space[0]

print(f"Found {len(single_token_words)} single-token words:")
for word in single_token_words[:15]:
    decoded = tokenizer.decode([word_to_token_id[word]])
    print(f"  '{word}': token_id={word_to_token_id[word]}, decoded='{decoded}'")
if len(single_token_words) > 15:
    print(f"  ... and {len(single_token_words) - 15} more")

if len(single_token_words) < NUM_ENTITIES + 1:
    raise ValueError(f"Not enough single-token words! Need {NUM_ENTITIES + 1}, have {len(single_token_words)}")

# Use the first word as FIXED_ENTITY, rest as targets
FIXED_ENTITY = single_token_words[0]
TARGET_WORDS = single_token_words[1:NUM_ENTITIES + 1]

print(f"\nFIXED_ENTITY: '{FIXED_ENTITY}' (token_id={word_to_token_id[FIXED_ENTITY]})")
print(f"TARGET_WORDS ({len(TARGET_WORDS)}): {TARGET_WORDS[:10]}...")

# %%
@dataclass
class Sample:
    text: str
    input_ids: torch.Tensor
    fixed_entity_token_idx: int
    target_word: str
    target_token_idx: int


def create_dataset(
    tokenizer,
    target_words: List[str],
    fixed_entity: str,
    word_to_token_id: Dict[str, int],
    dist: int,
    size: int,
    n_words_per_sentence: int = 15,
    seed: int = 42
) -> List[Sample]:
    """Create dataset with sentences and target tokens at position (fixed_entity + dist)."""
    rng = random.Random(seed)
    samples = []
    skipped = 0
    
    all_words = [fixed_entity] + target_words
    
    for i in tqdm(range(size), desc="Generating samples"):
        # Fixed entity should be far enough from the end
        max_fixed_pos = n_words_per_sentence - dist - 2
        fixed_pos = rng.randint(0, max(0, max_fixed_pos))
        
        # Pick target word
        target_word = rng.choice(target_words)
        
        # Build sentence
        sentence_words = []
        for j in range(n_words_per_sentence):
            if j == fixed_pos:
                sentence_words.append(fixed_entity)
            elif j == fixed_pos + dist:
                sentence_words.append(target_word)
            else:
                sentence_words.append(rng.choice(all_words))
        
        text = " ".join(sentence_words) + "."
        
        # Tokenize
        tokens = tokenizer.encode(text)
        input_ids = torch.tensor([tokens])
        
        # Find fixed_entity token in sequence
        fixed_token_id = word_to_token_id[fixed_entity]
        target_token_id = word_to_token_id[target_word]
        
        fixed_entity_token_idx = None
        for idx, tok in enumerate(tokens):
            if tok == fixed_token_id:
                fixed_entity_token_idx = idx
                break
        
        if fixed_entity_token_idx is None:
            skipped += 1
            continue
        
        # Find target token - it should be DIST tokens after fixed_entity
        # But due to tokenization, we search for it
        target_token_idx = None
        for offset in range(dist, dist + 3):  # Search dist to dist+2 positions
            check_idx = fixed_entity_token_idx + offset
            if check_idx < len(tokens) and tokens[check_idx] == target_token_id:
                target_token_idx = check_idx
                break
        
        if target_token_idx is None:
            skipped += 1
            continue
        
        samples.append(Sample(
            text=text,
            input_ids=input_ids,
            fixed_entity_token_idx=fixed_entity_token_idx,
            target_word=target_word,
            target_token_idx=target_token_idx,
        ))
    
    print(f"Created {len(samples)} samples, skipped {skipped}")
    return samples


print(f"\nCreating dataset...")
print(f"  NUM_ENTITIES: {NUM_ENTITIES}")
print(f"  DIST: {DIST}")
print(f"  FIXED_ENTITY: {FIXED_ENTITY}")
print(f"  DATASET_SIZE: {DATASET_SIZE}")

dataset = create_dataset(
    tokenizer=tokenizer,
    target_words=TARGET_WORDS,
    fixed_entity=FIXED_ENTITY,
    word_to_token_id=word_to_token_id,
    dist=DIST,
    size=DATASET_SIZE,
    n_words_per_sentence=15,
    seed=42,
)

if len(dataset) == 0:
    raise ValueError("No valid samples created! Check tokenization.")

print(f"\nValid samples: {len(dataset)}")

# Show examples
print("\n=== Sample Examples ===")
for i in range(min(3, len(dataset))):
    s = dataset[i]
    tokens = s.input_ids.tolist()[0]
    print(f"Example {i+1}:")
    print(f"  Text: {s.text[:80]}...")
    print(f"  Fixed '{FIXED_ENTITY}' at token idx: {s.fixed_entity_token_idx}")
    print(f"  Target '{s.target_word}' at token idx: {s.target_token_idx}")
    print(f"  Token at fixed idx: {tokenizer.decode([tokens[s.fixed_entity_token_idx]])}")
    print(f"  Token at target idx: {tokenizer.decode([tokens[s.target_token_idx]])}")
    print()

# %%
# === PREPARE FOR TRAINING ===
TARGET_TOKEN_IDS = torch.tensor([word_to_token_id[w] for w in TARGET_WORDS])
WORD_TO_IDX = {word: idx for idx, word in enumerate(TARGET_WORDS)}
n_classes = len(TARGET_WORDS)

print(f"Number of classes: {n_classes}")
print(f"Random baseline accuracy: {1/n_classes:.3f}")

# %%
# === EXTRACT STATES ===
# Process the FULL sentence and extract state at the END
# The probe will predict what token was at position (FIXED_ENTITY + DIST)
print(f"\nExtracting states at END of sentence (probing for token at FIXED_ENTITY + {DIST})...")

metadata_rows = []
all_states = np.zeros((len(dataset), num_layers, num_heads, head_size, head_size), dtype=np.float16)

for idx, sample in enumerate(tqdm(dataset, desc="Extracting states")):
    # Process the ENTIRE sentence
    final_states = extractor.extract_final_states(sample.input_ids)
    
    metadata_rows.append({
        'text': sample.text,
        'target_word': sample.target_word,
        'fixed_entity_idx': sample.fixed_entity_token_idx,
        'target_idx': sample.target_token_idx,
    })
    
    for layer_idx in range(num_layers):
        layer_state = final_states[layer_idx]
        all_states[idx, layer_idx] = layer_state.cpu().to(torch.float16).numpy()

df = pd.DataFrame(metadata_rows)
print(f"\nDataframe shape: {df.shape}")
print(f"States shape: {all_states.shape}")

print("\n=== Dataset Statistics ===")
print(f"Target word distribution:\n{df['target_word'].value_counts().head(10)}")

# %%
# === PREPARE LABELS ===
labels = torch.tensor([WORD_TO_IDX[w] for w in df['target_word']])
print(f"Labels shape: {labels.shape}")

# %%
import torch.nn as nn
import torch.optim as optim

class StateProbe(nn.Module):
    def __init__(self, head_size, n_classes):
        super().__init__()
        self.W_left = nn.Parameter(torch.randn(n_classes, head_size) * 0.01)
        self.w_right = nn.Parameter(torch.randn(head_size) * 0.01)
    
    def forward(self, state):
        # state: (batch, head_size, head_size)
        hidden = torch.einsum('cd,bdk->bck', self.W_left, state)  # (batch, n_classes, head_size)
        logits = torch.einsum('bck,k->bc', hidden, self.w_right)  # (batch, n_classes)
        return logits

def train_probe(states_np, labels, layer_idx, head_idx, patience=20, batch_size=1000):
    states = torch.tensor(states_np[:, layer_idx, head_idx], dtype=torch.float32)
    
    n_samples = len(labels)
    n_val = max(1, int(0.3 * n_samples))
    indices = torch.randperm(n_samples)
    
    train_states = states[indices[n_val:]]
    train_labels = labels[indices[n_val:]]
    val_states = states[indices[:n_val]]
    val_labels = labels[indices[:n_val]]
    
    probe = StateProbe(head_size, n_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    n_train = len(train_states)
    
    for epoch in range(20000):
        probe.train()
        
        batch_indices = torch.randperm(n_train)
        
        for i in range(0, n_train, batch_size):
            batch_idx = batch_indices[i:i+batch_size]
            batch_states = train_states[batch_idx].to(device)
            batch_labels = train_labels[batch_idx].to(device)
            
            optimizer.zero_grad()
            logits = probe(batch_states)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
        
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_states.to(device))
            val_loss = criterion(val_logits, val_labels.to(device)).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels.to(device)).float().mean().item()
            
            train_logits = probe(train_states.to(device))
            train_preds = train_logits.argmax(dim=1)
            train_acc = (train_preds == train_labels.to(device)).float().mean().item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return best_val_acc, train_acc, epoch + 1

# %%
print("\n" + "=" * 60)
print(f"TRAINING PROBES: From END state, predict token at (FIXED_ENTITY + {DIST})")
print("=" * 60)
print(f"{'Layer':<6} {'Head':<6} {'Val Acc':<10} {'Train Acc':<10} {'Epochs':<8}")
print("-" * 45)

results = []
for layer_idx in range(num_layers):
    for head_idx in range(num_heads):
        val_acc, train_acc, epochs = train_probe(all_states, labels, layer_idx, head_idx)
        results.append({
            'layer': layer_idx,
            'head': head_idx,
            'val_acc': val_acc,
            'train_acc': train_acc,
            'epochs': epochs
        })
        print(f"L{layer_idx:<5} H{head_idx:<5} {val_acc:<10.3f} {train_acc:<10.3f} {epochs:<8}")

# %%
results_df = pd.DataFrame(results)
print("\n=== Best Performing Heads ===")
top_50 = results_df.sort_values('val_acc', ascending=False).head(50)
for idx, row in top_50.iterrows():
    print(f"L{row['layer']:<5} H{row['head']:<5} {row['val_acc']:<10.3f} {row['train_acc']:<10.3f} {row['epochs']:<8}")

os.makedirs('results', exist_ok=True)
top_50.to_csv(f'results/result_next_token_dist{DIST}.csv', index=False)
print(f"\nResults saved to results/result_next_token_dist{DIST}.csv")

# %%
print("\n=== SUMMARY ===")
print(f"Configuration:")
print(f"  NUM_ENTITIES: {NUM_ENTITIES}")
print(f"  DIST: {DIST}")
print(f"  FIXED_ENTITY: {FIXED_ENTITY}")
print(f"  DATASET_SIZE: {len(dataset)}")
print(f"  n_classes: {n_classes}")
print(f"\nTask: From END-of-sentence state, predict token at (FIXED_ENTITY + {DIST})")
print(f"Method: Both Linear (W_left + w_right)")
print(f"Best head: L{int(top_50.iloc[0]['layer'])}, H{int(top_50.iloc[0]['head'])}")
print(f"Best val accuracy: {top_50.iloc[0]['val_acc']:.3f}")
print(f"Random baseline: {1/n_classes:.3f}")

# %%
