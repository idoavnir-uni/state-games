# %% [markdown]
# # Probe A: Left-Matrix Factorization
#
# This notebook trains linear probes on RWKV state matrices using left-matrix factorization.
#
# **Probe model:** logits = (W_left @ state) @ w_right
# - state: (batch, head_size, head_size) per head
# - W_left: (n_classes, head_size) learned matrix
# - w_right: (head_size,) learned vector

# %%
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(".."))

from models.load_rwkv import load_rwkv_model, get_model_config
from models.state_extractor_rwkv import RWKVStateExtractor
from datasets.favorite_color_dataset import FavoriteColorDataset

print("Imports complete!")

# %% [markdown]
# ## 1. Load Model and Tokenizer

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading RWKV model...")
model, tokenizer = load_rwkv_model(model_name="rwkv7-g1b-1.5b-20251202-ctx8192.pth")

config = get_model_config(model)
print(f"Model loaded: {config.get('num_layers')} layers, {config.get('num_heads')} heads")
print(f"Head size: {config.get('head_size')}")

# %% [markdown]
# ## 2. Initialize State Extractor

# %%
extractor = RWKVStateExtractor(model, verbose=False)

# %% [markdown]
# ## 3. Create Favorite Color Dataset

# %%
DATASET_SIZE = 2000
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
# ## 4. Extract States

# %%
num_layers = config.get("num_layers")
num_heads = config.get("num_heads")
head_size = config.get("head_size")

print(f"Extracting states for {DATASET_SIZE} samples...")
print(f"Model has {num_layers} layers, {num_heads} heads, head_size={head_size}")

metadata_rows = []
all_states = np.zeros((DATASET_SIZE, num_layers, num_heads, head_size, head_size), dtype=np.float16)

for idx in tqdm(range(len(dataset)), desc="Processing samples"):
    sample = dataset[idx]

    input_ids = sample.input_ids
    final_states = extractor.extract_final_states(input_ids)

    metadata_rows.append(
        {
            "sentence": sample.text,
            "target_color": sample.fixed_entity_color,
            "information_given_idx": sample.fixed_entity_sentence_end_token_idx,
            "sentence_with_info_num": sample.fixed_entity_sentence_number,
        }
    )

    for layer_idx in range(num_layers):
        layer_state = final_states[layer_idx]
        all_states[idx, layer_idx] = layer_state.cpu().to(torch.float16).numpy()

df = pd.DataFrame(metadata_rows)
print(f"\nMetadata dataframe shape: {df.shape}")
print(f"States array shape: {all_states.shape}")

# %% [markdown]
# ## 5. Dataset Statistics

# %%
print("\n=== Sample Entry ===")
print(f"Sentence: {df.iloc[0]['sentence'][:100]}...")
print(f"Target color: {df.iloc[0]['target_color']}")
print(f"States shape: {all_states.shape} = (samples, layers, heads, head_size, head_size)")

print("\n=== Dataset Statistics ===")
print(f"Total samples: {len(df)}")
print(f"Color distribution:\n{df['target_color'].value_counts()}")

# %% [markdown]
# ## 6. Define Left-Matrix Factorization Probe
#
# logits = (W_left @ state) @ w_right


# %%
class LeftMatrixProbe(nn.Module):
    def __init__(self, head_size, n_classes):
        super().__init__()
        self.W_left = nn.Parameter(torch.randn(n_classes, head_size) * 0.01)
        self.w_right = nn.Parameter(torch.randn(head_size) * 0.01)

    def forward(self, state):
        # state: (batch, head_size, head_size)
        # W_left @ state: (n_classes, head_size) @ (batch, head_size, head_size) -> (batch, n_classes, head_size)
        hidden = self.W_left @ state
        # hidden @ w_right: (batch, n_classes, head_size) @ (head_size,) -> (batch, n_classes)
        logits = hidden @ self.w_right
        return logits


# %% [markdown]
# ## 7. Training Function

# %%
COLOR_TO_IDX = {color: idx for idx, color in enumerate(dataset.colors)}
labels = torch.tensor([COLOR_TO_IDX[c] for c in df["target_color"]])
n_classes = len(dataset.colors)

print(f"Color mapping: {COLOR_TO_IDX}")
print(f"Number of classes: {n_classes}")


# %%
def train_probe(states_np, labels, layer_idx, head_idx, patience=10):
    """
    Train a probe on states from a specific layer/head.

    Args:
        states_np: numpy array of shape (n_samples, n_layers, n_heads, head_size, head_size)
        labels: torch tensor of shape (n_samples,) with class indices
        layer_idx: which layer to extract states from
        head_idx: which head to extract states from
        patience: early stopping patience

    Returns:
        best_val_acc: float, validation accuracy at best val_loss epoch
        train_acc: float, training accuracy at final epoch
        epochs: int, number of epochs trained
    """
    states = torch.tensor(states_np[:, layer_idx, head_idx], dtype=torch.float32)

    n_samples = len(labels)
    n_val = max(1, int(0.3 * n_samples))
    indices = torch.randperm(n_samples)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_states = states[train_indices].to(device)
    train_labels = labels[train_indices].to(device)
    val_states = states[val_indices].to(device)
    val_labels = labels[val_indices].to(device)

    probe = LeftMatrixProbe(head_size, n_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    final_train_acc = 0.0

    for epoch in range(10000):
        probe.train()
        optimizer.zero_grad()
        logits = probe(train_states)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_states)
            val_loss = criterion(val_logits, val_labels).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels).float().mean().item()

            train_preds = logits.argmax(dim=1)
            final_train_acc = (train_preds == train_labels).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_val_acc, final_train_acc, epoch + 1


# %% [markdown]
# ## 8. Train Probes for All Layer/Head Combinations

# %%
print("Training probes for each layer/head...")
print(f"{'Layer':<6} {'Head':<6} {'Val Acc':<10} {'Train Acc':<10} {'Epochs':<8}")
print("-" * 45)

results = []
for layer_idx in range(num_layers):
    for head_idx in range(num_heads):
        val_acc, train_acc, epochs = train_probe(all_states, labels, layer_idx, head_idx)
        results.append(
            {"layer": layer_idx, "head": head_idx, "val_acc": val_acc, "train_acc": train_acc, "epochs": epochs}
        )
        print(f"L{layer_idx:<5} H{head_idx:<5} {val_acc:<10.3f} {train_acc:<10.3f} {epochs:<8}")

# %% [markdown]
# ## 9. Results Summary

# %%
results_df = pd.DataFrame(results)
print("\n=== Best Performing Heads ===")
print(results_df.sort_values("val_acc", ascending=False).head(10))

# %%
print("\n=== Summary Statistics ===")
print(f"Mean validation accuracy: {results_df['val_acc'].mean():.3f}")
print(f"Max validation accuracy: {results_df['val_acc'].max():.3f}")
print(f"Random baseline: {1/n_classes:.3f}")

best_result = results_df.sort_values("val_acc", ascending=False).iloc[0]
print(f"\nBest probe: Layer {int(best_result['layer'])}, Head {int(best_result['head'])}")
print(f"  Val accuracy: {best_result['val_acc']:.3f}")
print(f"  Train accuracy: {best_result['train_acc']:.3f}")
print(f"  Epochs: {int(best_result['epochs'])}")

# %% [markdown]
# ## 10. Re-train Best Probe on Larger Dataset

# %%
RETRAIN_DATASET_SIZE = 10000
best_layer = int(best_result["layer"])
best_head_idx = int(best_result["head"])

print(f"Generating {RETRAIN_DATASET_SIZE} samples for re-training...")
retrain_dataset = FavoriteColorDataset(
    tokenizer=tokenizer,
    size=RETRAIN_DATASET_SIZE,
    n_entities=N_ENTITIES,
    n_colors=N_COLORS,
    fixed_entity_name=FIXED_ENTITY,
    seed=100,
)

print(f"Extracting states for best head (Layer {best_layer}, Head {best_head_idx})...")
retrain_states = np.zeros((RETRAIN_DATASET_SIZE, head_size, head_size), dtype=np.float16)
retrain_colors = []

for idx in tqdm(range(RETRAIN_DATASET_SIZE), desc="Extracting states"):
    sample = retrain_dataset[idx]
    input_ids = sample.input_ids
    final_states = extractor.extract_final_states(input_ids)
    retrain_states[idx] = final_states[best_layer][best_head_idx].cpu().to(torch.float16).numpy()
    retrain_colors.append(sample.fixed_entity_color)

retrain_labels = torch.tensor([COLOR_TO_IDX[c] for c in retrain_colors])
retrain_states_tensor = torch.tensor(retrain_states, dtype=torch.float32).to(device)
retrain_labels_device = retrain_labels.to(device)

# %%
print(f"Re-training probe on Layer {best_layer}, Head {best_head_idx} with {RETRAIN_DATASET_SIZE} samples...")

n_samples = RETRAIN_DATASET_SIZE
n_val = max(1, int(0.1 * n_samples))
indices = torch.randperm(n_samples)

train_states = retrain_states_tensor[indices[n_val:]]
train_labels = retrain_labels_device[indices[n_val:]]
val_states = retrain_states_tensor[indices[:n_val]]
val_labels = retrain_labels_device[indices[:n_val]]

best_probe = LeftMatrixProbe(head_size, n_classes).to(device)
optimizer = optim.Adam(best_probe.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

pbar = tqdm(range(10000), desc="Re-training")
for epoch in pbar:
    best_probe.train()
    optimizer.zero_grad()
    logits = best_probe(train_states)
    loss = criterion(logits, train_labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        best_probe.eval()
        with torch.no_grad():
            val_logits = best_probe(val_states)
            val_loss = criterion(val_logits, val_labels).item()
            train_preds = logits.argmax(dim=1)
            train_acc = (train_preds == train_labels).float().mean().item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels).float().mean().item()

        pbar.set_postfix(
            {
                "train_loss": f"{loss.item():.4f}",
                "val_loss": f"{val_loss:.4f}",
                "train_acc": f"{train_acc:.3f}",
                "val_acc": f"{val_acc:.3f}",
            }
        )

print("Best probe re-trained!")

# %% [markdown]
# ## 11. Generate Evaluation Samples

# %%
TARGET_EVAL_SAMPLES = 10
EVAL_N_ENTITIES = 500
MIN_SENTENCE_NUM = 10
MAX_SENTENCE_NUM = 20

print(
    f"\nGenerating {TARGET_EVAL_SAMPLES} evaluation samples with {FIXED_ENTITY} at sentences {MIN_SENTENCE_NUM}-{MAX_SENTENCE_NUM}..."
)
eval_samples = []
seed_offset = 43

pbar = tqdm(total=TARGET_EVAL_SAMPLES, desc="Generating valid samples")
while len(eval_samples) < TARGET_EVAL_SAMPLES:
    temp_dataset = FavoriteColorDataset(
        tokenizer=tokenizer,
        size=100,
        n_entities=EVAL_N_ENTITIES,
        n_colors=N_COLORS,
        fixed_entity_name=FIXED_ENTITY,
        seed=seed_offset,
    )

    for sample in temp_dataset:
        if MIN_SENTENCE_NUM <= sample.fixed_entity_sentence_number <= MAX_SENTENCE_NUM:
            eval_samples.append(sample)
            pbar.update(1)
            if len(eval_samples) >= TARGET_EVAL_SAMPLES:
                break

    seed_offset += 1

pbar.close()
print(f"Generated {len(eval_samples)} valid samples")

# %% [markdown]
# ## 12. Evaluate Probe Accuracy by Position

# %%
print("Extracting incremental states for evaluation...")
print("Note: This extracts states token-by-token, so will be slow")

eval_accuracies_by_position = {}
eval_extractor = RWKVStateExtractor(model, verbose=False)
best_probe.eval()

for sample_idx, sample in enumerate(tqdm(eval_samples, desc="Samples")):
    input_ids = sample.input_ids
    seq_len = input_ids.shape[1]

    incremental_states = eval_extractor.extract_incremental_states_single_pass(
        input_ids,
        layers=[best_layer],
        use_tqdm=True,
    )

    info_idx = sample.fixed_entity_sentence_end_token_idx
    true_color_idx = COLOR_TO_IDX[sample.fixed_entity_color]

    with torch.no_grad():
        for pos in range(info_idx + 1, seq_len):
            if pos not in incremental_states:
                continue

            state_at_pos = incremental_states[pos][best_layer]
            state_tensor = torch.tensor(state_at_pos[best_head_idx], dtype=torch.float32).unsqueeze(0).to(device)

            logits = best_probe(state_tensor)
            pred = logits.argmax(dim=1).item()

            is_correct = pred == true_color_idx
            relative_pos = pos - info_idx

            if relative_pos not in eval_accuracies_by_position:
                eval_accuracies_by_position[relative_pos] = []
            eval_accuracies_by_position[relative_pos].append(is_correct)

# %% [markdown]
# ## 13. Plot Accuracy by Position

# %%
positions = sorted(eval_accuracies_by_position.keys())
mean_accs = [np.mean(eval_accuracies_by_position[p]) for p in positions]

plt.figure(figsize=(12, 6))
plt.plot(positions, mean_accs, "b-", linewidth=2, label="Mean Accuracy")
plt.axhline(y=1 / n_classes, color="r", linestyle="--", label=f"Random Baseline ({1/n_classes:.3f})")
plt.xlabel("Tokens after Information Given")
plt.ylabel("Accuracy")
plt.title(f"Probe Accuracy vs Position After Information\nLayer {best_layer}, Head {best_head_idx}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../data/rwkv_probe_accuracy_by_position.png", dpi=150)
plt.show()

# %%
ROLLING_WINDOW_SIZE = 100

plt.figure(figsize=(12, 8))

n_eval_samples = len(eval_samples)
sample_trajectories = []

for sample_idx in range(n_eval_samples):
    trajectory = []
    for pos in positions:
        accs_at_pos = eval_accuracies_by_position.get(pos, [])
        if sample_idx < len(accs_at_pos):
            trajectory.append(float(accs_at_pos[sample_idx]))
        else:
            trajectory.append(np.nan)
    sample_trajectories.append(trajectory)

for sample_idx, trajectory in enumerate(sample_trajectories):
    rolling_traj = []
    rolling_pos = []
    for i in range(len(trajectory) - ROLLING_WINDOW_SIZE + 1):
        window = trajectory[i : i + ROLLING_WINDOW_SIZE]
        valid_vals = [v for v in window if not np.isnan(v)]
        if valid_vals:
            rolling_traj.append(np.mean(valid_vals))
            rolling_pos.append(positions[i + ROLLING_WINDOW_SIZE // 2])

    if rolling_traj:
        plt.plot(rolling_pos, rolling_traj, alpha=0.6, linewidth=1.5)

plt.axhline(y=1 / n_classes, color="r", linestyle="--", linewidth=2, label=f"Random Baseline ({1/n_classes:.3f})")
plt.xlabel("Tokens after Information Given")
plt.ylabel("Accuracy")
plt.title(
    f"Individual Rolling Window Accuracies (window={ROLLING_WINDOW_SIZE})\nLayer {best_layer}, Head {best_head_idx}"
)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../data/rwkv_probe_accuracy_rolling_window.png", dpi=150)
plt.show()

print(f"\nMean accuracy across all positions: {np.mean(mean_accs):.3f}")
print(f"Accuracy at position +1: {mean_accs[0]:.3f}")
print(f"Final accuracy: {mean_accs[-1]:.3f}")

# %%
