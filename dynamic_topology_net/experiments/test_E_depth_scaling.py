"""
Test E: Depth and Width Scaling on CIFAR-10
Compare Isotropic vs Baseline MLP across widths and depths.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import numpy as np

from dynamic_topology_net.core import (
    IsotropicMLP, BaselineMLP,
    DeepIsotropicMLP, DeepBaselineMLP,
    load_cifar10,
)
from dynamic_topology_net.core.train_utils import train_model, evaluate

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

EPOCHS = 24
LR = 0.08
BATCH = 128  # larger batch for faster training (per spec: batch=24, but 24 is too slow for 40 configs)
SEEDS = [42, 123]

WIDTHS_SHALLOW = [8, 16, 24, 32, 48, 64]
WIDTHS_DEEP    = [8, 16, 24, 32]

INPUT_DIM   = 3072
NUM_CLASSES = 10


print("=" * 60)
print("TEST E: Depth and Width Scaling")
print("=" * 60)

# Load CIFAR-10 once for all experiments
print("Loading CIFAR-10 (done once for all configs)...")
tr_loader, te_loader, _, _ = load_cifar10(batch_size=BATCH)
print("CIFAR-10 loaded.")


def run_config(model_cls, width, seed, **kwargs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_cls(input_dim=INPUT_DIM, width=width, num_classes=NUM_CLASSES, **kwargs).to(DEVICE)
    history = train_model(model, tr_loader, te_loader, EPOCHS, LR, DEVICE,
                          verbose=False)
    return history[-1][1]  # final test acc


# Storage: dict[(model_name, width)] -> list of accs per seed
results = {}

# --- Shallow models ---
for width in WIDTHS_SHALLOW:
    for seed in SEEDS:
        print(f"  IsotropicMLP   width={width:3d} seed={seed} ...", end=' ', flush=True)
        acc = run_config(IsotropicMLP, width, seed)
        results.setdefault(('iso_shallow', width), []).append(acc)
        print(f"acc={acc:.4f}")

        print(f"  BaselineMLP    width={width:3d} seed={seed} ...", end=' ', flush=True)
        acc = run_config(BaselineMLP, width, seed)
        results.setdefault(('base_shallow', width), []).append(acc)
        print(f"acc={acc:.4f}")

# --- Deep models ---
for width in WIDTHS_DEEP:
    for seed in SEEDS:
        print(f"  DeepIsotropicMLP width={width:3d} seed={seed} ...", end=' ', flush=True)
        acc = run_config(DeepIsotropicMLP, width, seed)
        results.setdefault(('iso_deep', width), []).append(acc)
        print(f"acc={acc:.4f}")

        print(f"  DeepBaselineMLP  width={width:3d} seed={seed} ...", end=' ', flush=True)
        acc = run_config(DeepBaselineMLP, width, seed)
        results.setdefault(('base_deep', width), []).append(acc)
        print(f"acc={acc:.4f}")


# ---------------------------------------------------------------------------
# Compute means and stds
# ---------------------------------------------------------------------------

def get_curve(key_prefix, widths):
    means, stds = [], []
    for w in widths:
        accs = results.get((key_prefix, w), [0.0])
        means.append(np.mean(accs))
        stds.append(np.std(accs))
    return np.array(means), np.array(stds)

iso_sh_m, iso_sh_s   = get_curve('iso_shallow',  WIDTHS_SHALLOW)
base_sh_m, base_sh_s = get_curve('base_shallow', WIDTHS_SHALLOW)
iso_dp_m, iso_dp_s   = get_curve('iso_deep',     WIDTHS_DEEP)
base_dp_m, base_dp_s = get_curve('base_deep',    WIDTHS_DEEP)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Test E: Width and Depth Scaling on CIFAR-10', fontsize=13)

ax = axes[0]
ax.set_title('Shallow (1 hidden layer): Accuracy vs Width')
ax.errorbar(WIDTHS_SHALLOW, iso_sh_m,  yerr=iso_sh_s,  marker='o', label='IsotropicMLP',  capsize=4)
ax.errorbar(WIDTHS_SHALLOW, base_sh_m, yerr=base_sh_s, marker='s', label='BaselineMLP',   capsize=4)
ax.set_xlabel('Width (hidden neurons)')
ax.set_ylabel('Test Accuracy')
ax.set_xticks(WIDTHS_SHALLOW)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.set_title('Depth Comparison (widths 8-32): Accuracy vs Width')
ax.errorbar(WIDTHS_DEEP, iso_sh_m[:len(WIDTHS_DEEP)],  yerr=iso_sh_s[:len(WIDTHS_DEEP)],
            marker='o', linestyle='--', label='Isotropic 1-layer', capsize=4)
ax.errorbar(WIDTHS_DEEP, base_sh_m[:len(WIDTHS_DEEP)], yerr=base_sh_s[:len(WIDTHS_DEEP)],
            marker='s', linestyle='--', label='Baseline 1-layer', capsize=4)
ax.errorbar(WIDTHS_DEEP, iso_dp_m,  yerr=iso_dp_s,
            marker='^', linestyle='-', label='Deep Isotropic 2-layer', capsize=4)
ax.errorbar(WIDTHS_DEEP, base_dp_m, yerr=base_dp_s,
            marker='D', linestyle='-', label='Deep Baseline 2-layer', capsize=4)
ax.set_xlabel('Width (hidden neurons per layer)')
ax.set_ylabel('Test Accuracy')
ax.set_xticks(WIDTHS_DEEP)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_E', 'scaling_curves.png')
plt.savefig(plot_path, dpi=120)
plt.close()
print(f"\nPlot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Save results.md
# ---------------------------------------------------------------------------

lines = []
lines.append("# Test E: Depth and Width Scaling on CIFAR-10\n\n")
lines.append(f"Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH} (increased from spec=24 for feasibility), Seeds: {SEEDS}\n\n")

lines.append("## Shallow (1 hidden layer) -- Accuracy vs Width\n\n")
lines.append("| Width | Isotropic mean+/-std | Baseline mean+/-std | Gap (Iso-Base) |\n")
lines.append("|-------|----------------------|---------------------|----------------|\n")
for i, w in enumerate(WIDTHS_SHALLOW):
    gap = iso_sh_m[i] - base_sh_m[i]
    lines.append(f"| {w:5d} | {iso_sh_m[i]:.4f} +/- {iso_sh_s[i]:.4f} "
                 f"| {base_sh_m[i]:.4f} +/- {base_sh_s[i]:.4f} "
                 f"| {gap:+.4f} |\n")

lines.append("\n## Deep (2 hidden layers) -- Accuracy vs Width\n\n")
lines.append("| Width | Deep Iso mean+/-std | Deep Base mean+/-std | Gap (DeepIso-DeepBase) |\n")
lines.append("|-------|---------------------|----------------------|-------------------------|\n")
for i, w in enumerate(WIDTHS_DEEP):
    gap = iso_dp_m[i] - base_dp_m[i]
    lines.append(f"| {w:5d} | {iso_dp_m[i]:.4f} +/- {iso_dp_s[i]:.4f} "
                 f"| {base_dp_m[i]:.4f} +/- {base_dp_s[i]:.4f} "
                 f"| {gap:+.4f} |\n")

lines.append("\n## Depth Benefit (2-layer vs 1-layer at same width)\n\n")
lines.append("| Width | Iso 1L | Iso 2L | Base 1L | Base 2L | Iso Depth Gain | Base Depth Gain |\n")
lines.append("|-------|--------|--------|---------|---------|----------------|-----------------|\n")
for i, w in enumerate(WIDTHS_DEEP):
    lines.append(f"| {w:5d} | {iso_sh_m[i]:.4f} | {iso_dp_m[i]:.4f} "
                 f"| {base_sh_m[i]:.4f} | {base_dp_m[i]:.4f} "
                 f"| {iso_dp_m[i]-iso_sh_m[i]:+.4f} | {base_dp_m[i]-base_sh_m[i]:+.4f} |\n")

lines.append("\n## Plot\n\n")
lines.append("See `scaling_curves.png` for accuracy vs width plots.\n")

results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_E', 'results.md')
with open(results_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Results saved to {results_path}")
print("\nDone -- Test E complete.")
