"""
Test H: Sequential Pruning Stability
Prune neurons one at a time (smallest SV first for isotropic, smallest
max-weight-norm for baseline) and track accuracy at each step.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np
import copy

from dynamic_topology_net.core import IsotropicMLP, BaselineMLP, load_cifar10
from dynamic_topology_net.core.train_utils import train_model, evaluate

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

SEED    = 42
EPOCHS  = 24
LR      = 0.08
BATCH   = 128
WIDTH   = 32

INPUT_DIM   = 3072
NUM_CLASSES = 10

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Helper: prune baseline MLP neuron by magnitude
# ---------------------------------------------------------------------------

def prune_baseline_neuron(model, idx):
    """
    Remove neuron idx from a BaselineMLP.
    model.net is a Sequential: [Linear, StandardTanh, Linear]
    Remove row idx from W1 (net[0]) and col idx from W2 (net[2]).
    Returns a new BaselineMLP with width-1.
    """
    linear1 = model.net[0]
    linear2 = model.net[2]

    w, n = linear1.weight.data.shape  # (width, input_dim)
    c = linear2.weight.data.shape[0]  # num_classes

    keep = [i for i in range(w) if i != idx]

    new_model = BaselineMLP(input_dim=n, width=len(keep), num_classes=c).to(DEVICE)
    with torch.no_grad():
        new_model.net[0].weight.data = linear1.weight.data[keep].clone()
        new_model.net[0].bias.data   = linear1.bias.data[keep].clone()
        new_model.net[2].weight.data = linear2.weight.data[:, keep].clone()
        new_model.net[2].bias.data   = linear2.bias.data.clone()
    return new_model


def get_baseline_prune_scores(model):
    """
    Returns scores for each neuron in baseline MLP.
    Score = max|W1[i, :]| (largest incoming weight to neuron i).
    Lower score -> prune first.
    """
    W1 = model.net[0].weight.data  # (width, input_dim)
    return W1.abs().max(dim=1).values.cpu().numpy()


# ---------------------------------------------------------------------------
# Train both models
# ---------------------------------------------------------------------------

print("=" * 60)
print("TEST H: Sequential Pruning Stability")
print("=" * 60)

tr_loader, te_loader, _, _ = load_cifar10(batch_size=BATCH)

# --- Isotropic ---
print(f"\nTraining IsotropicMLP width={WIDTH} ...")
torch.manual_seed(SEED)
iso_model = IsotropicMLP(input_dim=INPUT_DIM, width=WIDTH, num_classes=NUM_CLASSES).to(DEVICE)
history_iso = train_model(iso_model, tr_loader, te_loader, EPOCHS, LR, DEVICE,
                          verbose=False, prefix='  Iso ')
iso_base_acc = history_iso[-1][1]
print(f"  IsotropicMLP base acc: {iso_base_acc:.4f}")

# --- Baseline ---
print(f"\nTraining BaselineMLP width={WIDTH} ...")
torch.manual_seed(SEED)
base_model = BaselineMLP(input_dim=INPUT_DIM, width=WIDTH, num_classes=NUM_CLASSES).to(DEVICE)
history_base = train_model(base_model, tr_loader, te_loader, EPOCHS, LR, DEVICE,
                           verbose=False, prefix='  Base ')
base_base_acc = history_base[-1][1]
print(f"  BaselineMLP base acc: {base_base_acc:.4f}")


# ---------------------------------------------------------------------------
# Sequential pruning: Isotropic (SV-based)
# ---------------------------------------------------------------------------

print("\n--- Sequential pruning: IsotropicMLP (SV-based) ---")

iso_neurons_remaining = list(range(WIDTH, 0, -1))  # 32 down to 1
iso_accs = [iso_base_acc]

current_iso = copy.deepcopy(iso_model)
for step in range(WIDTH - 1):
    # Re-diagonalise
    svs = current_iso.partial_diagonalise()
    # Find smallest SV (last index after diagonalise sorts descending)
    prune_idx = current_iso.width - 1
    sv_val = svs[-1].item()
    current_iso.prune_neuron(prune_idx)
    acc = evaluate(current_iso, te_loader, DEVICE)
    iso_accs.append(acc)
    remaining = WIDTH - step - 1
    print(f"  Step {step+1:2d}: pruned neuron (sv={sv_val:.4f}), {remaining} neurons remain, acc={acc:.4f}")

iso_neurons_remaining = list(range(WIDTH, 0, -1))


# ---------------------------------------------------------------------------
# Sequential pruning: Baseline (magnitude-based)
# ---------------------------------------------------------------------------

print("\n--- Sequential pruning: BaselineMLP (magnitude-based) ---")

base_neurons_remaining = list(range(WIDTH, 0, -1))
base_accs = [base_base_acc]

current_base = copy.deepcopy(base_model)
for step in range(WIDTH - 1):
    scores = get_baseline_prune_scores(current_base)
    prune_idx = int(np.argmin(scores))
    score_val = scores[prune_idx]
    current_base = prune_baseline_neuron(current_base, prune_idx)
    acc = evaluate(current_base, te_loader, DEVICE)
    base_accs.append(acc)
    remaining = WIDTH - step - 1
    print(f"  Step {step+1:2d}: pruned neuron (score={score_val:.4f}), {remaining} neurons remain, acc={acc:.4f}")

base_neurons_remaining = list(range(WIDTH, 0, -1))


# ---------------------------------------------------------------------------
# Find "cliff" -- where accuracy drops below 70% of baseline
# ---------------------------------------------------------------------------

def find_cliff(accs, neurons, threshold=0.7):
    base = accs[0]
    for i, (a, n) in enumerate(zip(accs, neurons)):
        if a < base * threshold:
            return n, i
    return 1, len(accs) - 1

iso_cliff_n,  iso_cliff_i  = find_cliff(iso_accs,  iso_neurons_remaining)
base_cliff_n, base_cliff_i = find_cliff(base_accs, base_neurons_remaining)

print(f"\nIsotropic cliff at {iso_cliff_n} neurons remaining (acc drops to {iso_accs[iso_cliff_i]:.4f})")
print(f"Baseline  cliff at {base_cliff_n} neurons remaining (acc drops to {base_accs[base_cliff_i]:.4f})")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(f'Test H: Sequential Pruning -- width={WIDTH} on CIFAR-10\n'
             f'(Iso: SV-based, Baseline: magnitude-based)')

ax.plot(iso_neurons_remaining, iso_accs, marker='o', label='IsotropicMLP (SV-based)',
        color='blue')
ax.plot(base_neurons_remaining, base_accs, marker='s', label='BaselineMLP (magnitude-based)',
        color='orange')

# Mark cliffs
if iso_cliff_i > 0:
    ax.axvline(iso_cliff_n, color='blue', linestyle=':', alpha=0.6,
               label=f'Iso cliff ({iso_cliff_n} neurons)')
if base_cliff_i > 0:
    ax.axvline(base_cliff_n, color='orange', linestyle=':', alpha=0.6,
               label=f'Base cliff ({base_cliff_n} neurons)')

ax.set_xlabel('Neurons Remaining')
ax.set_ylabel('Test Accuracy')
ax.invert_xaxis()
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_H', 'sequential_pruning.png')
plt.savefig(plot_path, dpi=120)
plt.close()
print(f"\nPlot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Save results.md
# ---------------------------------------------------------------------------

lines = []
lines.append("# Test H: Sequential Pruning Stability\n\n")
lines.append(f"Seed: {SEED}, Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH}, Width: {WIDTH}\n\n")
lines.append("Isotropic: prune smallest singular value first (re-diagonalise each step).\n")
lines.append("Baseline: prune neuron with smallest max incoming weight norm.\n\n")

lines.append("## Base Accuracies\n\n")
lines.append(f"- IsotropicMLP (trained, full width): {iso_base_acc:.4f}\n")
lines.append(f"- BaselineMLP (trained, full width):  {base_base_acc:.4f}\n\n")

lines.append("## Pruning Trajectory\n\n")
lines.append("| Neurons Remaining | IsotropicMLP Acc | BaselineMLP Acc |\n")
lines.append("|-------------------|-----------------|----------------|\n")
for i, n in enumerate(iso_neurons_remaining):
    lines.append(f"| {n:17d} | {iso_accs[i]:.4f} | {base_accs[i]:.4f} |\n")

lines.append("\n## Cliff Analysis (where acc drops below 70% of initial)\n\n")
lines.append(f"- IsotropicMLP cliff: {iso_cliff_n} neurons remaining "
             f"(acc = {iso_accs[iso_cliff_i]:.4f}, initial = {iso_base_acc:.4f})\n")
lines.append(f"- BaselineMLP cliff:  {base_cliff_n} neurons remaining "
             f"(acc = {base_accs[base_cliff_i]:.4f}, initial = {base_base_acc:.4f})\n\n")

if iso_cliff_n <= base_cliff_n:
    lines.append("**Isotropic SV-based pruning holds accuracy longer** (cliff at fewer neurons remaining),\n")
    lines.append("supporting the paper's claim that SV-ordered pruning causes minimal degradation.\n\n")
else:
    lines.append(f"**Baseline magnitude pruning holds longer** in this run. Note that the comparison is\n")
    lines.append("not perfectly controlled (different initial accuracies, different pruning criteria).\n\n")

# Compute AUC-style metric (average accuracy across all pruning steps)
iso_auc  = np.mean(iso_accs)
base_auc = np.mean(base_accs)
lines.append(f"Average accuracy across all pruning steps (AUC proxy):\n")
lines.append(f"- Isotropic: {iso_auc:.4f}\n")
lines.append(f"- Baseline:  {base_auc:.4f}\n")
lines.append(f"- Advantage: {iso_auc - base_auc:+.4f} (positive = isotropic wins)\n\n")
lines.append("See `sequential_pruning.png` for the pruning curve.\n")

results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_H', 'results.md')
with open(results_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Results saved to {results_path}")
print("\nDone -- Test H complete.")
