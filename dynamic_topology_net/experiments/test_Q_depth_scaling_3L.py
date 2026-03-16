"""
Test Q -- Three-Layer Depth Scaling
=====================================
Tests M and E showed:
  - Iso-2L gains +2.5% from depth over Iso-1L
  - Base-2L LOSES -2.9% from depth (degrades, especially at wider widths)
  - Base-2L at width=48 collapses to 18% (near chance)

This test extends the depth curve to 3 layers: [3072, m, m, m, 10].
The nested functional class structure (Appendix C) predicts:
  - Isotropic networks continue benefiting from depth (each layer adds a
    perturbative correction to the representation)
  - Standard tanh continues degrading (no such structure; gradient issues)

If Base-3L collapses to chance while Iso-3L continues improving,
that is strong evidence for the paper's depth stability claims.

Widths tested: 16, 24 (feasible on CPU in reasonable time)
Seeds: [42, 123]
Hyperparams: 24 epochs, lr=0.08, batch=128
Device: CPU

Expected runtime: ~15-25 min per (model, width, seed) on CPU.
Total: 6 models x 2 widths x 2 seeds = 24 runs ~ 6-10 hours.
If too slow, reduce to width=16 only, single seed, for a preliminary result.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_topology_net.core import (
    IsotropicMLP, BaselineMLP,
    DeepIsotropicMLP, DeepBaselineMLP,
    IsotropicMLP3L, BaselineMLP3L,
    load_cifar10
)
from dynamic_topology_net.core.train_utils import train_model, evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_Q')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED    = 42
EPOCHS  = 24
LR      = 0.08
BATCH   = 128
WIDTHS  = [16, 24]
SEEDS   = [42, 123]
DEVICE  = torch.device('cpu')


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Widths: {WIDTHS}, Seeds: {SEEDS}, Epochs: {EPOCHS}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    model_builders = {
        'Iso-1L':   lambda w: IsotropicMLP(input_dim, w, num_classes),
        'Iso-2L':   lambda w: DeepIsotropicMLP(input_dim, w, num_classes),
        'Iso-3L':   lambda w: IsotropicMLP3L(input_dim, w, num_classes),
        'Base-1L':  lambda w: BaselineMLP(input_dim, w, num_classes),
        'Base-2L':  lambda w: DeepBaselineMLP(input_dim, w, num_classes),
        'Base-3L':  lambda w: BaselineMLP3L(input_dim, w, num_classes),
    }

    results = {}  # (label, width, seed) -> (acc, n_params)

    total_runs = len(model_builders) * len(WIDTHS) * len(SEEDS)
    run_idx = 0

    for width in WIDTHS:
        for seed in SEEDS:
            for label, make_model in model_builders.items():
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] {label} width={width} seed={seed}")
                torch.manual_seed(seed)
                model = make_model(width).to(DEVICE)
                n_params = count_params(model)
                train_model(model, train_loader, test_loader, EPOCHS, LR, DEVICE, verbose=False)
                acc = evaluate(model, test_loader, DEVICE)
                results[(label, width, seed)] = (acc, n_params)
                print(f"  acc={acc:.4f}  params={n_params:,}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY (mean +/- std over seeds)")
    print(f"{'='*70}")

    labels = list(model_builders.keys())
    summary = {}
    for label in labels:
        for width in WIDTHS:
            accs = [results[(label, width, s)][0] for s in SEEDS]
            summary[(label, width)] = (np.mean(accs), np.std(accs))

    # Print table
    print(f"{'Model':>10}", end='')
    for w in WIDTHS:
        print(f"  w={w:>3}", end='')
    print()
    for label in labels:
        print(f"{label:>10}", end='')
        for w in WIDTHS:
            m, s = summary[(label, w)]
            print(f"  {m:.3f}", end='')
        print()

    # Depth gains
    print(f"\nDepth gains (mean over widths):")
    for model_type in ['Iso', 'Base']:
        gains_1_to_2 = [
            (summary[(f'{model_type}-2L', w)][0] - summary[(f'{model_type}-1L', w)][0]) * 100
            for w in WIDTHS
        ]
        gains_2_to_3 = [
            (summary[(f'{model_type}-3L', w)][0] - summary[(f'{model_type}-2L', w)][0]) * 100
            for w in WIDTHS
        ]
        gains_1_to_3 = [
            (summary[(f'{model_type}-3L', w)][0] - summary[(f'{model_type}-1L', w)][0]) * 100
            for w in WIDTHS
        ]
        print(f"  {model_type}: 1L->2L = {np.mean(gains_1_to_2):+.2f}%  "
              f"2L->3L = {np.mean(gains_2_to_3):+.2f}%  "
              f"1L->3L = {np.mean(gains_1_to_3):+.2f}%")

    # =========================================================================
    # Plot
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    depths = [1, 2, 3]
    iso_colors  = ['#fee0b6', '#f97a0a', '#7f3a00']  # light to dark orange
    base_colors = ['#d0e8ff', '#3a8fd1', '#003b6e']  # light to dark blue

    for ax_idx, width in enumerate(WIDTHS):
        ax = axes[ax_idx]
        for depth_idx, d in enumerate(depths):
            iso_label  = f'Iso-{d}L'
            base_label = f'Base-{d}L'
            iso_m,  iso_s  = summary[(iso_label,  width)]
            base_m, base_s = summary[(base_label, width)]
            ax.errorbar([d - 0.1], [iso_m  * 100], yerr=[iso_s  * 100],
                        fmt='o-', color=iso_colors[depth_idx],
                        label=iso_label,  capsize=4)
            ax.errorbar([d + 0.1], [base_m * 100], yerr=[base_s * 100],
                        fmt='s-', color=base_colors[depth_idx],
                        label=base_label, capsize=4)

        # Connect depths with lines
        iso_means  = [summary[(f'Iso-{d}L',  width)][0] * 100 for d in depths]
        base_means = [summary[(f'Base-{d}L', width)][0] * 100 for d in depths]
        ax.plot(depths, iso_means,  'o-', color='darkorange', linewidth=2,
                label='Iso trend',  zorder=0)
        ax.plot(depths, base_means, 's-', color='steelblue',  linewidth=2,
                label='Base trend', zorder=0)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['1 layer', '2 layers', '3 layers'])
        ax.set_xlabel('Network depth')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(f'Depth scaling: Iso vs Baseline\n(width={width}, CIFAR-10)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'depth_scaling_3L.png'), dpi=150)
    print("\nPlot saved to results/test_Q/depth_scaling_3L.png")

    # =========================================================================
    # Save results.md
    # =========================================================================
    def mk_table():
        header = '| Model |' + ''.join(f' w={w} (mean+/-std) |' for w in WIDTHS) + ' Params (w=16) |'
        sep    = '|---|' + '---|' * (len(WIDTHS) + 1)
        rows   = []
        for label in labels:
            params = results[(label, WIDTHS[0], SEEDS[0])][1]
            row = f'| {label} |'
            for w in WIDTHS:
                m, s = summary[(label, w)]
                row += f' {m:.3f}+-{s:.3f} |'
            row += f' ~{params:,} |'
            rows.append(row)
        return '\n'.join([header, sep] + rows)

    def mk_depth_table():
        rows = []
        for model_type in ['Iso', 'Base']:
            for w in WIDTHS:
                g12 = (summary[(f'{model_type}-2L', w)][0] - summary[(f'{model_type}-1L', w)][0]) * 100
                g23 = (summary[(f'{model_type}-3L', w)][0] - summary[(f'{model_type}-2L', w)][0]) * 100
                g13 = (summary[(f'{model_type}-3L', w)][0] - summary[(f'{model_type}-1L', w)][0]) * 100
                rows.append(f'| {model_type} | {w} | {g12:+.2f}% | {g23:+.2f}% | {g13:+.2f}% |')
        return '\n'.join(rows)

    # Compute overall depth gain numbers for narrative
    iso_1_to_3 = np.mean([(summary[('Iso-3L', w)][0] - summary[('Iso-1L', w)][0]) * 100 for w in WIDTHS])
    base_1_to_3 = np.mean([(summary[('Base-3L', w)][0] - summary[('Base-1L', w)][0]) * 100 for w in WIDTHS])
    base_3L_min = min(summary[('Base-3L', w)][0] for w in WIDTHS) * 100

    if iso_1_to_3 > 1.0 and base_1_to_3 < 0:
        verdict = ("Isotropic networks continue to benefit from depth at 3 layers, "
                   "while baseline standard tanh continues to degrade. This strongly "
                   "supports the paper's nested functional class depth stability claim.")
    elif iso_1_to_3 > 0 and base_1_to_3 < iso_1_to_3:
        verdict = ("Isotropic networks gain more from depth than baseline, "
                   "consistent with the paper's claims, though the effect is smaller than expected.")
    else:
        verdict = ("Results are mixed -- isotropic advantage from depth not clearly "
                   "maintained at 3 layers. May reflect optimisation difficulty rather "
                   "than architectural limits.")

    results_text = f"""# Test Q -- Three-Layer Depth Scaling

## Setup
- Epochs: {EPOCHS}, lr={LR}, batch={BATCH}
- Widths: {WIDTHS}, Seeds: {SEEDS}
- Device: CPU
- Extends Tests E and M (1L, 2L) to 3 hidden layers

## Results (mean +/- std accuracy over seeds)

{mk_table()}

## Depth Gain Analysis

| Model type | Width | 1L->2L | 2L->3L | 1L->3L |
|---|---|---|---|---|
{mk_depth_table()}

## Key Numbers
- Iso 1L->3L gain (mean): {iso_1_to_3:+.2f}%
- Base 1L->3L gain (mean): {base_1_to_3:+.2f}%
- Base-3L minimum accuracy: {base_3L_min:.1f}%

## Verdict
{verdict}

## Comparison with Prior Tests
- Test E: Iso 1L->2L gain = +2.7%, Base 1L->2L = -0.5% (batch=128, widths 8-32)
- Test M: Iso 1L->2L gain = +2.5%, Base 1L->2L = -2.9% (batch=24, widths 8-48)
- Test Q: Iso 1L->3L gain = {iso_1_to_3:+.2f}%, Base 1L->3L = {base_1_to_3:+.2f}% (batch={BATCH}, widths {WIDTHS})

![Depth scaling to 3 layers](depth_scaling_3L.png)
"""

    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print("Results saved to results/test_Q/results.md")


if __name__ == '__main__':
    main()
