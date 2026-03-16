"""
Test G: Pruning Error vs Singular Value
For each neuron in a trained, diagonalised network: measure output change
(L2 error) and accuracy drop when that single neuron is pruned.
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

from dynamic_topology_net.core import IsotropicMLP, load_cifar10
from dynamic_topology_net.core.train_utils import train_model, evaluate

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

SEED    = 42
EPOCHS  = 24
LR      = 0.08
BATCH   = 128
WIDTHS  = [8, 16, 32]

INPUT_DIM   = 3072
NUM_CLASSES = 10


@torch.no_grad()
def compute_l2_error(model_orig, model_pruned, loader, device, probe_neuron_idx):
    """Compute mean L2 distance between outputs of two models on a dataset."""
    model_orig.eval()
    model_pruned.eval()
    total_l2 = 0.0
    n_samples = 0
    for x, y in loader:
        x = x.to(device)
        y_orig   = model_orig(x)    # (B, C)
        y_pruned = model_pruned(x)  # (B, C)  -- width-1
        # Pad pruned output to same classes (it already is -- only hidden dim changed)
        diff = (y_orig - y_pruned).pow(2).sum(dim=1).sqrt()  # (B,)
        total_l2 += diff.sum().item()
        n_samples += x.shape[0]
    return total_l2 / n_samples


# ---------------------------------------------------------------------------
# Main loop over widths
# ---------------------------------------------------------------------------

all_svs    = []
all_l2s    = []
all_drops  = []
all_widths = []

print("=" * 60)
print("TEST G: Pruning Error vs Singular Value")
print("=" * 60)

print("Loading CIFAR-10 (done once)...")
tr_loader, te_loader, _, _ = load_cifar10(batch_size=BATCH)
print("CIFAR-10 loaded.")

for width in WIDTHS:
    print(f"\n--- Width = {width} ---")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Train
    model = IsotropicMLP(input_dim=INPUT_DIM, width=width, num_classes=NUM_CLASSES).to(DEVICE)
    print(f"  Training for {EPOCHS} epochs ...")
    history = train_model(model, tr_loader, te_loader, EPOCHS, LR, DEVICE, verbose=False)
    base_acc = history[-1][1]
    print(f"  Base test accuracy: {base_acc:.4f}")

    # Diagonalise
    svs = model.partial_diagonalise()
    svs_list = svs.cpu().tolist()
    print(f"  Singular values: {[f'{s:.4f}' for s in svs_list]}")

    # For each neuron, deepcopy and prune
    width_svs   = []
    width_l2s   = []
    width_drops = []

    for idx in range(width):
        model_copy = copy.deepcopy(model)
        pruned_sv = model_copy.prune_neuron(idx)

        # L2 error on test set
        l2_err = compute_l2_error(model, model_copy, te_loader, DEVICE, idx)

        # Accuracy after pruning
        pruned_acc = evaluate(model_copy, te_loader, DEVICE)
        acc_drop = base_acc - pruned_acc

        sv_val = svs_list[idx]
        width_svs.append(sv_val)
        width_l2s.append(l2_err)
        width_drops.append(acc_drop)

        print(f"  Neuron {idx:2d}: sv={sv_val:.4f}  L2_err={l2_err:.6f}  acc_drop={acc_drop:+.4f}")

    all_svs.extend(width_svs)
    all_l2s.extend(width_l2s)
    all_drops.extend(width_drops)
    all_widths.extend([width] * width)


all_svs   = np.array(all_svs)
all_l2s   = np.array(all_l2s)
all_drops = np.array(all_drops)
all_widths_arr = np.array(all_widths)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Test G: Pruning Error vs Singular Value', fontsize=13)

colors = {8: 'blue', 16: 'orange', 32: 'green'}

ax = axes[0]
ax.set_title('L2 Output Error vs Singular Value')
for w in WIDTHS:
    mask = all_widths_arr == w
    ax.scatter(all_svs[mask], all_l2s[mask], label=f'width={w}',
               color=colors[w], alpha=0.7, s=50)
ax.set_xlabel('Singular Value (sigma_i)')
ax.set_ylabel('Mean L2 Output Error after Pruning')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.set_title('Accuracy Drop vs Singular Value')
for w in WIDTHS:
    mask = all_widths_arr == w
    ax.scatter(all_svs[mask], all_drops[mask], label=f'width={w}',
               color=colors[w], alpha=0.7, s=50)
ax.set_xlabel('Singular Value (sigma_i)')
ax.set_ylabel('Accuracy Drop (base - pruned)')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_G', 'pruning_error_scatter.png')
plt.savefig(plot_path, dpi=120)
plt.close()
print(f"\nPlot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

corr_l2  = np.corrcoef(all_svs, all_l2s)[0, 1]
corr_acc = np.corrcoef(all_svs, all_drops)[0, 1]
print(f"\nCorrelation (SV, L2_error)    : {corr_l2:.4f}")
print(f"Correlation (SV, acc_drop)    : {corr_acc:.4f}")

smallest_idx  = np.argmin(all_svs)
largest_idx   = np.argmax(all_svs)
print(f"\nSmallest SV neuron: sv={all_svs[smallest_idx]:.4f}, L2={all_l2s[smallest_idx]:.6f}, acc_drop={all_drops[smallest_idx]:+.4f}")
print(f"Largest  SV neuron: sv={all_svs[largest_idx]:.4f}, L2={all_l2s[largest_idx]:.6f}, acc_drop={all_drops[largest_idx]:+.4f}")


# ---------------------------------------------------------------------------
# Save results.md
# ---------------------------------------------------------------------------

lines = []
lines.append("# Test G: Pruning Error vs Singular Value\n\n")
lines.append(f"Seed: {SEED}, Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH}, Widths: {WIDTHS}\n\n")
lines.append("For each neuron in a trained diagonalised network: measure L2 output error\n")
lines.append("and accuracy drop when that single neuron is pruned.\n\n")

lines.append("## Correlation Analysis\n\n")
lines.append(f"- Pearson correlation (singular_value, L2_error): {corr_l2:.4f}\n")
lines.append(f"- Pearson correlation (singular_value, acc_drop): {corr_acc:.4f}\n\n")

if corr_l2 > 0.5:
    lines.append("**Strong positive correlation between SV and L2 error.** Neurons with small\n")
    lines.append("singular values cause significantly less output change when pruned.\n")
    lines.append("This quantitatively supports the paper's 'minimal degradation' claim.\n\n")
elif corr_l2 > 0.2:
    lines.append("**Moderate positive correlation between SV and L2 error.** The trend holds\n")
    lines.append("but with scatter -- some high-SV neurons may still be prunable safely.\n\n")
else:
    lines.append(f"**Weak correlation ({corr_l2:.4f})** -- the relationship may require more data or larger widths.\n\n")

for w in WIDTHS:
    lines.append(f"## Width = {w}: Per-neuron Results\n\n")
    lines.append("| Neuron | Singular Value | L2 Error | Acc Drop |\n")
    lines.append("|--------|---------------|----------|----------|\n")
    mask = all_widths_arr == w
    idxs = np.where(mask)[0]
    for j, global_i in enumerate(idxs):
        lines.append(f"| {j:6d} | {all_svs[global_i]:.6f} | {all_l2s[global_i]:.8f} | {all_drops[global_i]:+.4f} |\n")
    lines.append("\n")

lines.append("## Extremes\n\n")
lines.append(f"- Smallest SV neuron: sv={all_svs[smallest_idx]:.4f}, L2={all_l2s[smallest_idx]:.6f}, acc_drop={all_drops[smallest_idx]:+.4f}\n")
lines.append(f"- Largest SV neuron:  sv={all_svs[largest_idx]:.4f}, L2={all_l2s[largest_idx]:.6f}, acc_drop={all_drops[largest_idx]:+.4f}\n\n")
lines.append("See `pruning_error_scatter.png` for scatter plots.\n")

results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_G', 'results.md')
with open(results_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Results saved to {results_path}")
print("\nDone -- Test G complete.")
