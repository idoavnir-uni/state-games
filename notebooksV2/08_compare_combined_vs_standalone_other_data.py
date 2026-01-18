# %% [markdown]
# # Comparison: Standalone vs Combined Dataset Probing
# 
# This notebook compares probe accuracy between:
# 1. Standalone FavoriteColorDataset 
# 2. CombinedDataset with configurable color/city ratio (fixed entity from colors)

# %%
import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models.load_rwkv import load_rwkv_model, get_model_config
from models.state_extractor_rwkv import RWKVStateExtractor
from datasets.favorite_color_dataset import FavoriteColorDataset
from datasets.lives_in_city_dataset import LivesInCityDataset
from datasets.combined_dataset import (
    CombinedDataset,
    DatasetConfig,
    FAVORITE_COLOR_CONFIG,
)
from datasets.names import FAMOUS_NAMES
from datasets.animal_names import ANIMAL_NAMES

print("Imports complete!")

# %% [markdown]
# ## 1. Load Model and Tokenizer

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("Loading RWKV model...")
model, tokenizer = load_rwkv_model(model_name="rwkv7-g1b-1.5b-20251202-ctx8192.pth")

config = get_model_config(model)
print(f"Model loaded: {config.get('num_layers')} layers, {config.get('num_heads')} heads")

# %% [markdown]
# ## 2. Initialize State Extractor

# %%
extractor = RWKVStateExtractor(model, verbose=False)
head_size = extractor.head_size
n_head = extractor.n_head
n_embd = model.model.n_embd
num_layers = config.get('num_layers')
num_heads = config.get('num_heads')

print(f"Head size: {head_size}")
print(f"Num heads: {n_head}")
print(f"Hidden size: {n_embd}")

# %% [markdown]
# ## 3. Configuration

# %%
DATASET_SIZE = 1000
N_ENTITIES = 400
N_COLORS = 10
FIXED_ENTITY = "Jeff Bezos"  # For color dataset

# Combined dataset ratio (colors vs cities)
N_COLORS_COMBINED = N_ENTITIES // 30
N_CITIES_COMBINED = N_ENTITIES - N_COLORS_COMBINED
COLOR_RATIO = f"{N_COLORS_COMBINED}/{N_ENTITIES}"

COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "brown"]
CITIES = ["Paris", "London", "Chicago", "Boston", "Miami", "Austin", "Houston", "York", "Bern", "Gary"]

# %% [markdown]
# ## 4. Create Both Datasets

# %%
print("Creating standalone FavoriteColorDataset...")
standalone_dataset = FavoriteColorDataset(
    tokenizer=tokenizer,
    size=DATASET_SIZE,
    n_entities=N_ENTITIES,
    n_colors=N_COLORS,
    fixed_entity_name=FIXED_ENTITY,
    seed=42,
)
print(f"Standalone dataset: {len(standalone_dataset)} samples")
print(f"Fixed entity: {standalone_dataset.fixed_entity_name}")
print(f"Colors: {standalone_dataset.colors}")

# %%
print(f"\nCreating {COLOR_RATIO} CombinedDataset ({N_COLORS_COMBINED} colors + {N_CITIES_COMBINED} cities)...")

color_config = DatasetConfig(
    sentence_template="{name}'s favorite color is {value}.",
    values=COLORS,
    fixed_entity_name=FIXED_ENTITY,
    names_pool=FAMOUS_NAMES,
    value_key="color",
)

city_config = DatasetConfig(
    sentence_template="{name} lives in {value}.",
    values=CITIES,
    fixed_entity_name="bat",  # LivesInCityDataset uses animal names
    names_pool=ANIMAL_NAMES,
    value_key="city",
)

combined_dataset = CombinedDataset(
    tokenizer=tokenizer,
    dataset_configs={"color": color_config, "city": city_config},
    size=DATASET_SIZE,
    sentences_per_config={"color": N_COLORS_COMBINED, "city": N_CITIES_COMBINED},
    fixed_entity_source="color",
    shuffle_sentences=True,
    seed=42,
)
print(f"Combined dataset: {len(combined_dataset)} samples")
print(f"Sentences per config: {combined_dataset.sentences_per_config}")

# %% [markdown]
# ## 5. Verify Colors are Single Tokens

# %%
print("\nVerifying colors are single tokens...")
color_token_ids = {}
all_single_tokens = True

for color in COLORS:
    tokens = tokenizer.encode(color)
    is_single = len(tokens) == 1
    
    if is_single:
        color_token_ids[color] = tokens[0]
        token_str = tokenizer.decode([tokens[0]])
        print(f"  {color}: token_id={tokens[0]}, decoded='{token_str}' ✓")
    else:
        all_single_tokens = False
        decoded = [tokenizer.decode([t]) for t in tokens]
        print(f"  {color}: {len(tokens)} tokens {tokens} -> {decoded} ✗")

if all_single_tokens:
    print("\n✓ All colors are single tokens!")
else:
    print("\n✗ Some colors are NOT single tokens.")
    raise ValueError("Not all colors are single tokens")

COLOR_TOKEN_IDS = torch.tensor([color_token_ids[c] for c in COLORS])
COLOR_TO_IDX = {color: idx for idx, color in enumerate(COLORS)}
n_colors = len(COLORS)

# %% [markdown]
# ## 6. Extract States for Both Datasets

# %%
def extract_dataset_states(dataset, extractor, num_layers, num_heads, head_size, is_combined=False):
    """Extract states and metadata from a dataset."""
    size = len(dataset)
    metadata_rows = []
    all_states = np.zeros((size, num_layers, num_heads, head_size, head_size), dtype=np.float16)
    
    for idx in tqdm(range(size), desc="Extracting states"):
        sample = dataset[idx]
        input_ids = sample.input_ids
        final_states = extractor.extract_final_states(input_ids)
        
        if is_combined:
            metadata_rows.append({
                'sentence': sample.text,
                'target_color': sample.fixed_entity_value,
                'information_given_idx': sample.fixed_entity_sentence_end_token_idx,
                'sentence_with_info_num': sample.fixed_entity_sentence_number,
            })
        else:
            metadata_rows.append({
                'sentence': sample.text,
                'target_color': sample.fixed_entity_color,
                'information_given_idx': sample.fixed_entity_sentence_end_token_idx,
                'sentence_with_info_num': sample.fixed_entity_sentence_number,
            })
        
        for layer_idx in range(num_layers):
            layer_state = final_states[layer_idx]
            all_states[idx, layer_idx] = layer_state.cpu().to(torch.float16).numpy()
    
    df = pd.DataFrame(metadata_rows)
    return df, all_states

# %%
print("Extracting states for standalone dataset...")
standalone_df, standalone_states = extract_dataset_states(
    standalone_dataset, extractor, num_layers, num_heads, head_size, is_combined=False
)
print(f"Standalone - Metadata shape: {standalone_df.shape}, States shape: {standalone_states.shape}")

# %%
print("Extracting states for combined dataset...")
combined_df, combined_states = extract_dataset_states(
    combined_dataset, extractor, num_layers, num_heads, head_size, is_combined=True
)
print(f"Combined - Metadata shape: {combined_df.shape}, States shape: {combined_states.shape}")

# %% [markdown]
# ## 7. Probe Training Functions

# %%
lm_head = model.model.z['head.weight'].detach()
color_unembed = lm_head[:, COLOR_TOKEN_IDS].float().to(device)
print(f"Color unembedding shape: {color_unembed.shape}")

# %%
def get_head_o_proj(model, layer_idx, head_idx, head_size=64, n_embd=2048):
    z = model.model.z
    o_proj = z[f'blocks.{layer_idx}.att.output.weight'].detach().float()
    start_idx = head_idx * head_size
    end_idx = (head_idx + 1) * head_size
    head_o_proj = o_proj[start_idx:end_idx, :]
    return head_o_proj

def get_W_left_model(head_o_proj, unembed):
    M = head_o_proj @ unembed
    W_left_model = M.T
    return W_left_model

def precompute_transformed_states(states, W_left_model):
    A = torch.einsum('cd,bdk->bck', W_left_model, states)
    return A

def solve_w_right_closed_form(A, labels, n_classes):
    batch_size = A.shape[0]
    head_size = A.shape[2]
    
    Y = torch.zeros(batch_size, n_classes, device=A.device)
    Y.scatter_(1, labels.unsqueeze(1), 1.0)
    
    A_flat = A.reshape(-1, head_size)
    Y_flat = Y.reshape(-1)
    
    w_right = torch.linalg.lstsq(A_flat, Y_flat).solution
    return w_right

def evaluate_w_right(A, labels, w_right):
    logits = torch.einsum('bck,k->bc', A, w_right)
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    return acc

def solve_probe(states_np, labels, layer_idx, head_idx, model, unembed, head_size, n_embd, n_classes):
    states = torch.tensor(states_np[:, layer_idx, head_idx], dtype=torch.float32).to(device)
    
    n_samples = len(labels)
    n_val = max(1, int(0.3 * n_samples))
    indices = torch.randperm(n_samples)
    
    train_states = states[indices[n_val:]]
    train_labels = labels[indices[n_val:]].to(device)
    val_states = states[indices[:n_val]]
    val_labels = labels[indices[:n_val]].to(device)
    
    head_o_proj = get_head_o_proj(model, layer_idx, head_idx, head_size, n_embd).to(device)
    W_left_model = get_W_left_model(head_o_proj, unembed)
    
    train_A = precompute_transformed_states(train_states, W_left_model)
    val_A = precompute_transformed_states(val_states, W_left_model)
    
    w_right = solve_w_right_closed_form(train_A, train_labels, n_classes)
    
    train_acc = evaluate_w_right(train_A, train_labels, w_right)
    val_acc = evaluate_w_right(val_A, val_labels, w_right)
    
    return val_acc, train_acc, w_right

# %% [markdown]
# ## 8. Train Probes for Both Datasets

# %%
def train_all_probes(states_np, df, model, unembed, head_size, n_embd, n_classes, color_to_idx, name=""):
    labels = torch.tensor([color_to_idx[c] for c in df['target_color']])
    
    print(f"\n=== Training probes for {name} ===")
    results = []
    all_w_rights = {}
    
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            val_acc, train_acc, w_right = solve_probe(
                states_np, labels, layer_idx, head_idx, 
                model, unembed, head_size, n_embd, n_classes
            )
            all_w_rights[(layer_idx, head_idx)] = w_right
            results.append({
                'layer': layer_idx,
                'head': head_idx,
                'val_acc': val_acc,
                'train_acc': train_acc,
            })
    
    results_df = pd.DataFrame(results)
    print(f"\nTop 5 heads for {name}:")
    print(results_df.sort_values('val_acc', ascending=False).head(5))
    
    return results_df, all_w_rights, labels

# %%
standalone_results, standalone_w_rights, standalone_labels = train_all_probes(
    standalone_states, standalone_df, model, color_unembed, 
    head_size, n_embd, n_colors, COLOR_TO_IDX, name="Standalone"
)

# %%
combined_results, combined_w_rights, combined_labels = train_all_probes(
    combined_states, combined_df, model, color_unembed, 
    head_size, n_embd, n_colors, COLOR_TO_IDX, name="Combined"
)

# %% [markdown]
# ## 9. Compare Best Heads

# %%
standalone_best = standalone_results.sort_values('val_acc', ascending=False).iloc[0]
combined_best = combined_results.sort_values('val_acc', ascending=False).iloc[0]

print("\n=== Best Heads Comparison ===")
print(f"Standalone: Layer {int(standalone_best['layer'])}, Head {int(standalone_best['head'])}, Val Acc: {standalone_best['val_acc']:.3f}")
print(f"Combined:   Layer {int(combined_best['layer'])}, Head {int(combined_best['head'])}, Val Acc: {combined_best['val_acc']:.3f}")

# Use standalone best head for fair comparison
best_layer = int(standalone_best['layer'])
best_head_idx = int(standalone_best['head'])
print(f"\nUsing Layer {best_layer}, Head {best_head_idx} for comparison")

# %% [markdown]
# ## 10. Retrain Best Probe on Both Datasets

# %%
def retrain_best_probe(states_np, labels, best_layer, best_head_idx, model, unembed, head_size, n_embd, n_classes):
    states_tensor = torch.tensor(states_np[:, best_layer, best_head_idx], dtype=torch.float32)
    
    n_samples = len(labels)
    n_val = max(1, int(0.3 * n_samples))
    indices = torch.randperm(n_samples)
    
    train_states = states_tensor[indices[n_val:]].to(device)
    train_labels = labels[indices[n_val:]].to(device)
    val_states = states_tensor[indices[:n_val]].to(device)
    val_labels = labels[indices[:n_val]].to(device)
    
    head_o_proj = get_head_o_proj(model, best_layer, best_head_idx, head_size, n_embd).to(device)
    W_left_model = get_W_left_model(head_o_proj, unembed)
    
    train_A = precompute_transformed_states(train_states, W_left_model)
    val_A = precompute_transformed_states(val_states, W_left_model)
    
    w_right = solve_w_right_closed_form(train_A, train_labels, n_classes)
    
    train_acc = evaluate_w_right(train_A, train_labels, w_right)
    val_acc = evaluate_w_right(val_A, val_labels, w_right)
    
    return w_right, W_left_model, train_acc, val_acc

# %%
standalone_w_right, standalone_W_left, standalone_train_acc, standalone_val_acc = retrain_best_probe(
    standalone_states, standalone_labels, best_layer, best_head_idx,
    model, color_unembed, head_size, n_embd, n_colors
)
print(f"Standalone - Train: {standalone_train_acc:.3f}, Val: {standalone_val_acc:.3f}")

combined_w_right, combined_W_left, combined_train_acc, combined_val_acc = retrain_best_probe(
    combined_states, combined_labels, best_layer, best_head_idx,
    model, color_unembed, head_size, n_embd, n_colors
)
print(f"Combined   - Train: {combined_train_acc:.3f}, Val: {combined_val_acc:.3f}")

# %% [markdown]
# ## 11. Visualization: Probe Accuracy Across Token Positions

# %%
def safe_savefig(filename, **kwargs):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            plt.savefig(filename, **kwargs)
            return
        except OSError as e:
            if attempt < max_retries - 1 and "Stale file handle" in str(e):
                print(f"Stale file handle, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(0.5)
            else:
                raise

# %%
def get_sample_for_visualization(dataset, is_combined=False, seed=301, n_samples=100):
    """Get samples with varied information positions."""
    if is_combined:
        temp_dataset = CombinedDataset(
            tokenizer=tokenizer,
            dataset_configs={"color": color_config, "city": city_config},
            size=n_samples,
            sentences_per_config={"color": N_COLORS_COMBINED, "city": N_CITIES_COMBINED},
            fixed_entity_source="color",
            shuffle_sentences=True,
            seed=seed,
        )
    else:
        temp_dataset = FavoriteColorDataset(
            tokenizer=tokenizer,
            size=n_samples,
            n_entities=N_ENTITIES,
            n_colors=N_COLORS,
            fixed_entity_name=FIXED_ENTITY,
            seed=seed,
        )
    
    all_samples = [(s, s.fixed_entity_sentence_number) for s in temp_dataset]
    all_samples.sort(key=lambda x: x[1])
    
    viz_samples = []
    if len(all_samples) >= 3:
        indices = [len(all_samples) // 6, len(all_samples) // 2, 5 * len(all_samples) // 6]
        for idx in indices:
            sample, pos = all_samples[idx]
            viz_samples.append(sample)
    
    return viz_samples

# %%
def compute_accuracy_trajectory(sample, extractor, best_layer, best_head_idx, W_left_model, w_right, is_combined=False):
    """Compute accuracy at each token position for a sample."""
    input_ids = sample.input_ids
    seq_len = input_ids.shape[1]
    
    if is_combined:
        info_idx = sample.fixed_entity_sentence_end_token_idx
        true_color = sample.fixed_entity_value
    else:
        info_idx = sample.fixed_entity_sentence_end_token_idx
        true_color = sample.fixed_entity_color
    
    true_color_idx = COLOR_TO_IDX[true_color]
    
    incremental_states = extractor.extract_incremental_states_single_pass(
        input_ids, layers=[best_layer], use_tqdm=False
    )
    
    accuracies_top1 = []
    accuracies_top3 = []
    positions = []
    
    with torch.no_grad():
        for pos in range(1, seq_len):
            if pos not in incremental_states:
                continue
            
            state_at_pos = incremental_states[pos][best_layer]
            state_tensor = torch.tensor(state_at_pos[best_head_idx], dtype=torch.float32).unsqueeze(0).to(device)
            
            A = precompute_transformed_states(state_tensor, W_left_model)
            logits = torch.einsum('bck,k->bc', A, w_right)
            
            pred_top1 = logits.argmax(dim=1).item()
            is_correct_top1 = (pred_top1 == true_color_idx)
            
            top3_preds = logits.topk(min(3, n_colors), dim=1).indices[0].tolist()
            is_correct_top3 = (true_color_idx in top3_preds)
            
            accuracies_top1.append(float(is_correct_top1))
            accuracies_top3.append(float(is_correct_top3))
            positions.append(pos)
    
    return positions, accuracies_top1, accuracies_top3, info_idx, true_color

# %%
print("\n=== Visualization: Comparing Standalone vs Combined ===")

standalone_viz_samples = get_sample_for_visualization(standalone_dataset, is_combined=False)
combined_viz_samples = get_sample_for_visualization(combined_dataset, is_combined=True)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))

for sample_idx in range(3):
    # Standalone
    sample = standalone_viz_samples[sample_idx]
    positions, acc_top1, acc_top3, info_idx, true_color = compute_accuracy_trajectory(
        sample, extractor, best_layer, best_head_idx, standalone_W_left, standalone_w_right, is_combined=False
    )
    
    ax = axes[sample_idx, 0]
    ax.plot(positions, acc_top1, 'b-', linewidth=2, alpha=0.7, label='Top-1')
    ax.plot(positions, acc_top3, 'g-', linewidth=2, alpha=0.7, label='Top-3')
    ax.axvline(x=info_idx, color='r', linestyle='--', linewidth=2, label=f'Info given')
    ax.axhline(y=1/n_colors, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Correct')
    ax.set_title(f'Standalone #{sample_idx+1}: {true_color}')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Combined
    sample = combined_viz_samples[sample_idx]
    positions, acc_top1, acc_top3, info_idx, true_color = compute_accuracy_trajectory(
        sample, extractor, best_layer, best_head_idx, combined_W_left, combined_w_right, is_combined=True
    )
    
    ax = axes[sample_idx, 1]
    ax.plot(positions, acc_top1, 'b-', linewidth=2, alpha=0.7, label='Top-1')
    ax.plot(positions, acc_top3, 'g-', linewidth=2, alpha=0.7, label='Top-3')
    ax.axvline(x=info_idx, color='r', linestyle='--', linewidth=2, label=f'Info given')
    ax.axhline(y=1/n_colors, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Correct')
    ax.set_title(f'Combined {COLOR_RATIO} #{sample_idx+1}: {true_color}')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Probe Accuracy: Standalone vs Combined (Layer {best_layer}, Head {best_head_idx})', fontsize=14)
plt.tight_layout()
safe_savefig(os.path.join(PROJECT_ROOT, 'data/compare_standalone_vs_combined_samples.png'), dpi=150)
plt.show()

# %% [markdown]
# ## 12. Aggregate Evaluation

# %%
TARGET_EVAL_SAMPLES = 100
EVAL_N_ENTITIES = 400

def generate_eval_samples(is_combined, target_count, n_entities, seed_start=43):
    """Generate evaluation samples with info position between 10-20."""
    eval_samples = []
    seed_offset = seed_start
    
    pbar = tqdm(total=target_count, desc=f"Generating {'combined' if is_combined else 'standalone'} samples")
    while len(eval_samples) < target_count:
        if is_combined:
            n_colors_eval = n_entities // 5
            n_cities_eval = n_entities - n_colors_eval
            temp_dataset = CombinedDataset(
                tokenizer=tokenizer,
                dataset_configs={"color": color_config, "city": city_config},
                size=100,
                sentences_per_config={"color": n_colors_eval, "city": n_cities_eval},
                fixed_entity_source="color",
                shuffle_sentences=True,
                seed=seed_offset,
            )
        else:
            temp_dataset = FavoriteColorDataset(
                tokenizer=tokenizer,
                size=100,
                n_entities=n_entities,
                n_colors=N_COLORS,
                fixed_entity_name=FIXED_ENTITY,
                seed=seed_offset,
            )
        
        for sample in temp_dataset:
            if 10 <= sample.fixed_entity_sentence_number <= 20:
                eval_samples.append(sample)
                pbar.update(1)
                if len(eval_samples) >= target_count:
                    break
        
        seed_offset += 1
    
    pbar.close()
    return eval_samples

# %%
print(f"\nGenerating {TARGET_EVAL_SAMPLES} evaluation samples for each dataset...")
standalone_eval_samples = generate_eval_samples(is_combined=False, target_count=TARGET_EVAL_SAMPLES, n_entities=EVAL_N_ENTITIES)
combined_eval_samples = generate_eval_samples(is_combined=True, target_count=TARGET_EVAL_SAMPLES, n_entities=EVAL_N_ENTITIES)

# %%
def evaluate_samples(eval_samples, extractor, best_layer, best_head_idx, W_left_model, w_right, is_combined=False, name=""):
    """Evaluate probe on samples, tracking accuracy by relative position."""
    accuracies_by_position_top1 = {}
    accuracies_by_position_top3 = {}
    max_relative_positions = []  # Track max relative position for each sample
    
    for sample in tqdm(eval_samples, desc=f"Evaluating {name}"):
        input_ids = sample.input_ids
        seq_len = input_ids.shape[1]
        
        incremental_states = extractor.extract_incremental_states_single_pass(
            input_ids, layers=[best_layer], use_tqdm=False
        )
        
        if is_combined:
            info_idx = sample.fixed_entity_sentence_end_token_idx
            true_color = sample.fixed_entity_value
        else:
            info_idx = sample.fixed_entity_sentence_end_token_idx
            true_color = sample.fixed_entity_color
        
        true_color_idx = COLOR_TO_IDX[true_color]
        sample_max_relative_pos = 0
        
        with torch.no_grad():
            for pos in range(info_idx + 1, seq_len):
                if pos not in incremental_states:
                    continue
                
                state_at_pos = incremental_states[pos][best_layer]
                state_tensor = torch.tensor(state_at_pos[best_head_idx], dtype=torch.float32).unsqueeze(0).to(device)
                
                A = precompute_transformed_states(state_tensor, W_left_model)
                logits = torch.einsum('bck,k->bc', A, w_right)
                
                pred_top1 = logits.argmax(dim=1).item()
                is_correct_top1 = (pred_top1 == true_color_idx)
                
                top3_preds = logits.topk(min(3, n_colors), dim=1).indices[0].tolist()
                is_correct_top3 = (true_color_idx in top3_preds)
                
                relative_pos = pos - info_idx
                sample_max_relative_pos = max(sample_max_relative_pos, relative_pos)
                
                if relative_pos not in accuracies_by_position_top1:
                    accuracies_by_position_top1[relative_pos] = []
                    accuracies_by_position_top3[relative_pos] = []
                accuracies_by_position_top1[relative_pos].append(is_correct_top1)
                accuracies_by_position_top3[relative_pos].append(is_correct_top3)
        
        max_relative_positions.append(sample_max_relative_pos)
    
    min_max_pos = min(max_relative_positions) if max_relative_positions else 0
    return accuracies_by_position_top1, accuracies_by_position_top3, min_max_pos

# %%
# Evaluate all 4 combinations: (probe trained on X) tested on (dataset Y)
print("\n=== Cross-evaluation: Each probe on both datasets ===")

# Standalone probe on standalone data
print("\nStandalone probe -> Standalone data...")
standalone_on_standalone_top1, standalone_on_standalone_top3, standalone_min_len = evaluate_samples(
    standalone_eval_samples, extractor, best_layer, best_head_idx,
    standalone_W_left, standalone_w_right, is_combined=False, name="Standalone->Standalone"
)
print(f"Standalone data: shortest input ends at relative position {standalone_min_len}")

# Standalone probe on combined data
print("\nStandalone probe -> Combined data...")
standalone_on_combined_top1, standalone_on_combined_top3, combined_min_len = evaluate_samples(
    combined_eval_samples, extractor, best_layer, best_head_idx,
    standalone_W_left, standalone_w_right, is_combined=True, name="Standalone->Combined"
)
print(f"Combined data: shortest input ends at relative position {combined_min_len}")

# Combined probe on standalone data
print("\nCombined probe -> Standalone data...")
combined_on_standalone_top1, combined_on_standalone_top3, _ = evaluate_samples(
    standalone_eval_samples, extractor, best_layer, best_head_idx,
    combined_W_left, combined_w_right, is_combined=False, name="Combined->Standalone"
)

# Combined probe on combined data
print("\nCombined probe -> Combined data...")
combined_on_combined_top1, combined_on_combined_top3, _ = evaluate_samples(
    combined_eval_samples, extractor, best_layer, best_head_idx,
    combined_W_left, combined_w_right, is_combined=True, name="Combined->Combined"
)

# %% [markdown]
# ## 13. Plot Comparison

# %%
def get_mean_accuracies(acc_dict):
    positions = sorted(acc_dict.keys())
    means = [np.mean(acc_dict[p]) for p in positions]
    return positions, means

# Get mean accuracies for all 4 combinations
# Standalone probe
standalone_on_standalone_pos_top1, standalone_on_standalone_mean_top1 = get_mean_accuracies(standalone_on_standalone_top1)
standalone_on_standalone_pos_top3, standalone_on_standalone_mean_top3 = get_mean_accuracies(standalone_on_standalone_top3)
standalone_on_combined_pos_top1, standalone_on_combined_mean_top1 = get_mean_accuracies(standalone_on_combined_top1)
standalone_on_combined_pos_top3, standalone_on_combined_mean_top3 = get_mean_accuracies(standalone_on_combined_top3)

# Combined probe
combined_on_standalone_pos_top1, combined_on_standalone_mean_top1 = get_mean_accuracies(combined_on_standalone_top1)
combined_on_standalone_pos_top3, combined_on_standalone_mean_top3 = get_mean_accuracies(combined_on_standalone_top3)
combined_on_combined_pos_top1, combined_on_combined_mean_top1 = get_mean_accuracies(combined_on_combined_top1)
combined_on_combined_pos_top3, combined_on_combined_mean_top3 = get_mean_accuracies(combined_on_combined_top3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top-1 Accuracy - 4 lines: (probe) -> (test data)
ax = axes[0]
ax.plot(standalone_on_standalone_pos_top1, standalone_on_standalone_mean_top1, 'b-', linewidth=2, label='Homog. probe → Homog. data')
ax.plot(standalone_on_combined_pos_top1, standalone_on_combined_mean_top1, 'b--', linewidth=2, alpha=0.7, label='Homog. probe → Mixed data')
ax.plot(combined_on_standalone_pos_top1, combined_on_standalone_mean_top1, 'orange', linewidth=2, label='Mixed probe → Homog. data')
ax.plot(combined_on_combined_pos_top1, combined_on_combined_mean_top1, 'orange', linestyle='--', linewidth=2, alpha=0.7, label='Mixed probe → Mixed data')
ax.axhline(y=1/n_colors, color='r', linestyle=':', label=f'Random ({1/n_colors:.3f})')
ax.axvline(x=standalone_min_len, color='b', linestyle=':', alpha=0.5, label=f'Homog. data ends ({standalone_min_len})')
ax.axvline(x=combined_min_len, color='orange', linestyle=':', alpha=0.5, label=f'Mixed data ends ({combined_min_len})')
ax.set_xlabel('Tokens after Information Given')
ax.set_ylabel('Top-1 Accuracy')
ax.set_title('Top-1 Accuracy')
ax.legend(fontsize=7, loc='best')
ax.grid(True, alpha=0.3)

# Top-3 Accuracy - 4 lines
ax = axes[1]
ax.plot(standalone_on_standalone_pos_top3, standalone_on_standalone_mean_top3, 'b-', linewidth=2, label='Homog. probe → Homog. data')
ax.plot(standalone_on_combined_pos_top3, standalone_on_combined_mean_top3, 'b--', linewidth=2, alpha=0.7, label='Homog. probe → Mixed data')
ax.plot(combined_on_standalone_pos_top3, combined_on_standalone_mean_top3, 'orange', linewidth=2, label='Mixed probe → Homog. data')
ax.plot(combined_on_combined_pos_top3, combined_on_combined_mean_top3, 'orange', linestyle='--', linewidth=2, alpha=0.7, label='Mixed probe → Mixed data')
ax.axhline(y=3/n_colors, color='r', linestyle=':', label=f'Random ({3/n_colors:.3f})')
ax.axvline(x=standalone_min_len, color='b', linestyle=':', alpha=0.5, label=f'Homog. data ends ({standalone_min_len})')
ax.axvline(x=combined_min_len, color='orange', linestyle=':', alpha=0.5, label=f'Mixed data ends ({combined_min_len})')
ax.set_xlabel('Tokens after Information Given')
ax.set_ylabel('Top-3 Accuracy')
ax.set_title('Top-3 Accuracy')
ax.legend(fontsize=7, loc='best')
ax.grid(True, alpha=0.3)

plt.suptitle(f'Cross-Evaluation: Probe vs Test Data (Layer {best_layer}, Head {best_head_idx})\nHomog.={N_ENTITIES} colors, Mixed={COLOR_RATIO} colors', fontsize=12)
plt.tight_layout()
safe_savefig(os.path.join(PROJECT_ROOT, 'data/compare_standalone_vs_combined_accuracy.png'), dpi=150)
plt.show()

# %%
# Combined plot - Top-1 only for clarity with all 4 cross-evaluation lines
plt.figure(figsize=(14, 6))
plt.plot(standalone_on_standalone_pos_top1, standalone_on_standalone_mean_top1, 'b-', linewidth=2, label='Homog. probe → Homog. data')
plt.plot(standalone_on_combined_pos_top1, standalone_on_combined_mean_top1, 'b--', linewidth=2, alpha=0.7, label='Homog. probe → Mixed data')
plt.plot(combined_on_standalone_pos_top1, combined_on_standalone_mean_top1, 'green', linewidth=2, label='Mixed probe → Homog. data')
plt.plot(combined_on_combined_pos_top1, combined_on_combined_mean_top1, 'green', linestyle='--', linewidth=2, alpha=0.7, label='Mixed probe → Mixed data')
plt.axhline(y=1/n_colors, color='r', linestyle=':', label=f'Random ({1/n_colors:.3f})')
plt.axvline(x=standalone_min_len, color='b', linestyle=':', alpha=0.5, label=f'Homog. data ends ({standalone_min_len})')
plt.axvline(x=combined_min_len, color='green', linestyle=':', alpha=0.5, label=f'Mixed data ends ({combined_min_len})')
plt.xlabel('Tokens after Information Given')
plt.ylabel('Top-1 Accuracy')
plt.title(f'Cross-Evaluation Top-1 Accuracy (Layer {best_layer}, Head {best_head_idx})\nHomog.={N_ENTITIES} colors, Mixed={COLOR_RATIO} colors')
plt.legend(loc='best', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
safe_savefig(os.path.join(PROJECT_ROOT, 'data/compare_standalone_vs_combined_all.png'), dpi=150)
plt.show()

# %% [markdown]
# ## 14. Summary Statistics

# %%
print("\n" + "="*70)
print("SUMMARY STATISTICS - CROSS EVALUATION")
print("="*70)

print(f"\nDataset Configuration:")
print(f"  Homogeneous: {N_ENTITIES} sentences, all colors")
print(f"  Mixed: {N_ENTITIES} sentences, {N_COLORS_COMBINED} colors ({COLOR_RATIO}) + {N_CITIES_COMBINED} cities")
print(f"  Fixed entity: {FIXED_ENTITY}")

print(f"\nBest Probe: Layer {best_layer}, Head {best_head_idx}")

print(f"\nTraining Validation Accuracies:")
print(f"  Homogeneous probe: {standalone_val_acc:.3f}")
print(f"  Mixed probe: {combined_val_acc:.3f}")

print(f"\nCross-Evaluation Top-1 Accuracy (mean across positions):")
print(f"  {'Probe':<20} {'→ Homog. Data':<15} {'→ Mixed Data':<15}")
print(f"  {'-'*50}")
print(f"  {'Homogeneous probe':<20} {np.mean(standalone_on_standalone_mean_top1):<15.3f} {np.mean(standalone_on_combined_mean_top1):<15.3f}")
print(f"  {'Mixed probe':<20} {np.mean(combined_on_standalone_mean_top1):<15.3f} {np.mean(combined_on_combined_mean_top1):<15.3f}")

print(f"\nCross-Evaluation Top-3 Accuracy (mean across positions):")
print(f"  {'Probe':<20} {'→ Homog. Data':<15} {'→ Mixed Data':<15}")
print(f"  {'-'*50}")
print(f"  {'Homogeneous probe':<20} {np.mean(standalone_on_standalone_mean_top3):<15.3f} {np.mean(standalone_on_combined_mean_top3):<15.3f}")
print(f"  {'Mixed probe':<20} {np.mean(combined_on_standalone_mean_top3):<15.3f} {np.mean(combined_on_combined_mean_top3):<15.3f}")

# %%

