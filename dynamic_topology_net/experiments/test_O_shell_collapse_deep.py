"""
Test O -- Two-Layer Hyperspherical Shell Collapse
==================================================
Test F showed that CollapsingIsotropicMLP-1L (iso + unit-norm) achieves
near-identical accuracy to IsotropicMLP-1L despite being provably affine.
This test asks: does that hold at 2 layers?

Test M showed Iso-2L gains +2.5% from depth over Iso-1L.
If CollapsingIso-2L (also provably affine) still matches Iso-2L accuracy,
the nonlinearity isn't contributing at 2 layers either.
If Iso-2L beats CollapsingIso-2L, nonlinearity genuinely helps at depth.

Models compared:
  1. Iso-1L          -- IsotropicMLP (1 hidden layer)
  2. Iso-2L          -- DeepIsotropicMLP (2 hidden layers)
  3. CollapsingIso-1L -- IsotropicMLP + unit-norm (affine by construction)
  4. CollapsingIso-2L -- DeepIsotropicMLP + unit-norm (affine by construction)
  5. nn.Linear       -- logistic regression (floor)

Hyperparams: 24 epochs, lr=0.08, batch=128, width=32 (matches Tests F and M).
Affine verification: fit linear regression to collapsed model outputs; residual ~ 0.

Device: CPU (explicitly to avoid GPU for this exploratory test)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_topology_net.core import (
    IsotropicMLP, DeepIsotropicMLP,
    CollapsingIsotropicMLP, DeepCollapsingIsotropicMLP,
    load_cifar10
)
from dynamic_topology_net.core.train_utils import train_model, evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_O')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED       = 42
EPOCHS     = 24
LR         = 0.08
BATCH      = 128
WIDTH      = 32
DEVICE     = torch.device('cpu')


def verify_affine(model, test_loader, device):
    """
    Fit an affine (linear + bias) map from raw inputs to model outputs.
    Returns mean absolute residual and relative residual.
    If residual ~ 0, the model is affine.
    """
    model.eval()
    all_x, all_y = [], []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            y = model(x)
            all_x.append(x.view(x.shape[0], -1))
            all_y.append(y)
    X = torch.cat(all_x, dim=0)  # (N, input_dim)
    Y = torch.cat(all_y, dim=0)  # (N, num_classes)

    # Fit: Y = X_aug @ A  where X_aug = [X, 1]
    N = X.shape[0]
    X_aug = torch.cat([X, torch.ones(N, 1, device=device)], dim=1)  # (N, input_dim+1)

    # Least-squares solution
    solution = torch.linalg.lstsq(X_aug, Y).solution  # (input_dim+1, num_classes)
    Y_hat = X_aug @ solution

    residual = (Y - Y_hat).abs()
    mean_res = residual.mean().item()
    max_res  = residual.max().item()
    mean_mag = Y.abs().mean().item()
    rel_res  = mean_res / (mean_mag + 1e-10)
    return mean_res, max_res, rel_res


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    configs = [
        ('Iso-1L',           lambda: IsotropicMLP(input_dim, WIDTH, num_classes),          False),
        ('Iso-2L',           lambda: DeepIsotropicMLP(input_dim, WIDTH, num_classes),       False),
        ('CollapsingIso-1L', lambda: CollapsingIsotropicMLP(input_dim, WIDTH, num_classes), True),
        ('CollapsingIso-2L', lambda: DeepCollapsingIsotropicMLP(input_dim, WIDTH, num_classes), True),
        ('nn.Linear',        lambda: nn.Linear(input_dim, num_classes),                     False),
    ]

    results   = {}
    curves    = {}
    residuals = {}

    for label, make_model, is_collapsing in configs:
        print(f"\n--- {label} ---")
        torch.manual_seed(SEED)
        model = make_model().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        epoch_accs = train_model(model, train_loader, test_loader,
                                 EPOCHS, LR, DEVICE, verbose=True)
        final_acc = evaluate(model, test_loader, DEVICE)
        results[label] = final_acc
        curves[label]  = epoch_accs if epoch_accs else []
        print(f"  Final accuracy: {final_acc:.4f}")

        if is_collapsing:
            mean_res, max_res, rel_res = verify_affine(model, test_loader, DEVICE)
            residuals[label] = {'mean': mean_res, 'max': max_res, 'rel': rel_res}
            print(f"  Affine residual: mean={mean_res:.6f}, max={max_res:.6f}, rel={rel_res:.6f}")

    # =========================================================================
    # Print summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for label, acc in results.items():
        print(f"  {label:22s}: {acc*100:.2f}%")

    print(f"\nKey comparisons:")
    gap_1L = results['Iso-1L'] - results['CollapsingIso-1L']
    gap_2L = results['Iso-2L'] - results['CollapsingIso-2L']
    depth_iso  = results['Iso-2L']  - results['Iso-1L']
    depth_coll = results['CollapsingIso-2L'] - results['CollapsingIso-1L']
    print(f"  Iso-1L vs CollapsingIso-1L:  {gap_1L*100:+.2f}%")
    print(f"  Iso-2L vs CollapsingIso-2L:  {gap_2L*100:+.2f}%")
    print(f"  Iso depth gain (2L-1L):      {depth_iso*100:+.2f}%")
    print(f"  Collapsing depth gain (2L-1L): {depth_coll*100:+.2f}%")
    if gap_2L > gap_1L + 0.005:
        verdict = "Nonlinearity CONTRIBUTES at depth (Iso-2L beats CollapsingIso-2L by more than at 1L)"
    elif gap_2L < -0.005:
        verdict = "CollapsingIso-2L BEATS Iso-2L (unexpected -- collapse helps at 2L)"
    else:
        verdict = "Nonlinearity still does NOT contribute at 2 layers (gap consistent with 1L)"
    print(f"\nVerdict: {verdict}")

    # =========================================================================
    # Plot
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {
        'Iso-1L':           'darkorange',
        'Iso-2L':           'red',
        'CollapsingIso-1L': 'steelblue',
        'CollapsingIso-2L': 'royalblue',
        'nn.Linear':        'gray',
    }
    styles = {
        'Iso-1L':           '-o',
        'Iso-2L':           '-s',
        'CollapsingIso-1L': '--o',
        'CollapsingIso-2L': '--s',
        'nn.Linear':        ':x',
    }

    # Training curves
    for label, curve in curves.items():
        if curve:
            ax1.plot(range(1, len(curve)+1), [a*100 for a in curve],
                     styles[label], label=label, color=colors[label],
                     markersize=4, markevery=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Training curves: depth vs collapse')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Bar chart of final accuracies
    labels_ordered = ['Iso-1L', 'CollapsingIso-1L', 'Iso-2L', 'CollapsingIso-2L', 'nn.Linear']
    accs = [results[l]*100 for l in labels_ordered]
    bar_colors = [colors[l] for l in labels_ordered]
    ax2.bar(range(len(labels_ordered)), accs, color=bar_colors, alpha=0.8)
    ax2.set_xticks(range(len(labels_ordered)))
    ax2.set_xticklabels([l.replace('-', '\n') for l in labels_ordered], fontsize=8)
    ax2.set_ylabel('Final Test Accuracy (%)')
    ax2.set_title(f'Final accuracy (width={WIDTH}, 24 epochs)')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'shell_collapse_deep.png'), dpi=150)
    print("\nPlot saved to results/test_O/shell_collapse_deep.png")

    # =========================================================================
    # Save results.md
    # =========================================================================
    res_rows = '\n'.join(
        f"| {l} | {results[l]*100:.2f}% | "
        + (f"{residuals[l]['mean']:.6f} | {residuals[l]['max']:.6f} | {residuals[l]['rel']:.6f}"
           if l in residuals else "N/A | N/A | N/A")
        + " |"
        for l in ['Iso-1L', 'Iso-2L', 'CollapsingIso-1L', 'CollapsingIso-2L', 'nn.Linear']
    )

    results_text = f"""# Test O -- Two-Layer Hyperspherical Shell Collapse

## Setup
- Width: {WIDTH}, Epochs: {EPOCHS}, lr={LR}, batch={BATCH}
- Device: CPU
- Seed: {SEED}

## Background
Test F (1-layer) found CollapsingIso-1L (40.90%) ≈ Iso-1L (40.68%) despite being affine.
Test M found Iso-2L gains +2.5% from depth over Iso-1L.
This test checks whether the same near-equivalence holds at 2 layers.

## Results

| Model | Final Acc | Affine residual (mean) | Affine residual (max) | Relative residual |
|---|---|---|---|---|
{res_rows}

## Key Comparisons

| Comparison | Gap |
|---|---|
| Iso-1L vs CollapsingIso-1L | {gap_1L*100:+.2f}% |
| Iso-2L vs CollapsingIso-2L | {gap_2L*100:+.2f}% |
| Iso depth gain (2L - 1L) | {depth_iso*100:+.2f}% |
| Collapsing depth gain (2L - 1L) | {depth_coll*100:+.2f}% |

## Affine Verification
CollapsingIso-1L residual: mean={residuals.get('CollapsingIso-1L', {}).get('mean', float('nan')):.6f}
CollapsingIso-2L residual: mean={residuals.get('CollapsingIso-2L', {}).get('mean', float('nan')):.6f}

Both residuals should be ~0 (mathematical guarantee from Appendix C).

## Verdict
{verdict}

## Interpretation
If gap_2L >> gap_1L: the isotropic nonlinearity is doing real work at depth,
  even though it appeared redundant at 1 layer on CIFAR-10.
If gap_2L ~ gap_1L ~ 0: CIFAR-10 at this scale is largely a linear problem
  in the isotropic feature space at both 1 and 2 layers.
The depth gain for CollapsingIso ({depth_coll*100:+.2f}%) tells us how much
  benefit comes purely from the affine composition (extra weight matrices),
  vs the full depth gain for Iso ({depth_iso*100:+.2f}%).

![Shell collapse comparison](shell_collapse_deep.png)
"""

    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print("Results saved to results/test_O/results.md")


if __name__ == '__main__':
    main()
