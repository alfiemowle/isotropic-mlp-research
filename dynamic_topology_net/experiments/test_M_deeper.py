"""
Test M -- Deeper Networks [3072, m, m, 10]
==========================================
The paper only tests single-hidden-layer [3072, m, 10]. This test explores
two-hidden-layer networks where:
  - Full diagonalisation (not just partial) becomes possible
  - The middle affine layer can be fully diagonalised (Sec. 2.3)
  - The nested functional class structure (Appendix C, Eqn. 46) applies

Questions:
  1. Do deeper isotropic networks outperform single-layer ones?
  2. Does isotropy scale better with depth than standard tanh?
  3. Full vs partial diagonalisation: does full diag help?
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_topology_net.core import IsotropicMLP, BaselineMLP, DeepIsotropicMLP, DeepBaselineMLP
from dynamic_topology_net.core import load_cifar10
from dynamic_topology_net.core.train_utils import train_model, evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_M')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED   = 42
EPOCHS = 24
LR     = 0.08
WIDTHS = [8, 16, 24, 32, 48]
SEEDS  = [42, 123]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=24)

    results = {}  # (model_type, width, seed) -> acc

    for width in WIDTHS:
        for seed in SEEDS:
            configs = [
                ('Iso-1L',  lambda w=width: IsotropicMLP(input_dim=input_dim, width=w, num_classes=num_classes)),
                ('Base-1L', lambda w=width: BaselineMLP(input_dim=input_dim, width=w, num_classes=num_classes)),
                ('Iso-2L',  lambda w=width: DeepIsotropicMLP(input_dim=input_dim, width=w, num_classes=num_classes)),
                ('Base-2L', lambda w=width: DeepBaselineMLP(input_dim=input_dim, width=w, num_classes=num_classes)),
            ]
            for label, make_model in configs:
                torch.manual_seed(seed)
                model = make_model().to(DEVICE)
                n_params = count_params(model)
                train_model(model, train_loader, test_loader, EPOCHS, LR, DEVICE, verbose=False)
                acc = evaluate(model, test_loader, DEVICE)
                results[(label, width, seed)] = (acc, n_params)
                print(f"  {label:8s} width={width:2d} seed={seed}: acc={acc:.3f}  params={n_params:,}")

    # =========================================================================
    # Summary: mean +/- std per (model_type, width)
    # =========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY (mean +/- std over seeds)")
    print(f"{'='*60}")
    labels = ['Iso-1L', 'Base-1L', 'Iso-2L', 'Base-2L']
    summary = {}
    for label in labels:
        for width in WIDTHS:
            accs = [results[(label, width, s)][0] for s in SEEDS]
            summary[(label, width)] = (np.mean(accs), np.std(accs))

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

    # =========================================================================
    # Plot
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {'Iso-1L': 'darkorange', 'Base-1L': 'steelblue',
              'Iso-2L': 'red',        'Base-2L': 'royalblue'}
    styles = {'Iso-1L': '-o', 'Base-1L': '-s', 'Iso-2L': '--o', 'Base-2L': '--s'}

    for label in labels:
        means = [summary[(label, w)][0] * 100 for w in WIDTHS]
        stds  = [summary[(label, w)][1] * 100 for w in WIDTHS]
        ax1.errorbar(WIDTHS, means, yerr=stds, fmt=styles[label],
                     label=label, color=colors[label], capsize=4)

    ax1.set_xlabel('Hidden layer width'); ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('1-layer vs 2-layer: Isotropic vs Baseline')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Depth gain: 2L vs 1L
    for label_pair, color in [('Iso', 'darkorange'), ('Base', 'steelblue')]:
        gains = [(summary[(f'{label_pair}-2L', w)][0] - summary[(f'{label_pair}-1L', w)][0]) * 100
                 for w in WIDTHS]
        ax2.plot(WIDTHS, gains, 'o-', label=f'{label_pair}: 2L - 1L gain', color=color)

    ax2.axhline(0, linestyle='--', color='gray', alpha=0.5)
    ax2.set_xlabel('Hidden layer width'); ax2.set_ylabel('Accuracy gain from depth (%)')
    ax2.set_title('Accuracy gain from adding a hidden layer')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'depth_comparison.png'), dpi=150)
    print("Plot saved to results/test_M/depth_comparison.png")

    # =========================================================================
    # Save results.md
    # =========================================================================
    def mk_table():
        header = "| Model |" + "".join(f" w={w} |" for w in WIDTHS) + " Params |"
        sep    = "|---|" + "---|" * (len(WIDTHS) + 1)
        rows   = []
        for label in labels:
            params = results[(label, WIDTHS[0], SEEDS[0])][1]
            row = f"| {label} |"
            for w in WIDTHS:
                m, s = summary[(label, w)]
                row += f" {m:.3f}+-{s:.3f} |"
            row += f" ~{params:,} |"
            rows.append(row)
        return "\n".join([header, sep] + rows)

    depth_gains = {lp: [(summary[(f'{lp}-2L', w)][0] - summary[(f'{lp}-1L', w)][0])*100
                        for w in WIDTHS]
                   for lp in ['Iso', 'Base']}

    results_text = f"""# Test M -- Deeper Networks [3072, m, m, 10]

## Setup
- Training: {EPOCHS} epochs, Adam lr={LR}, batch=24
- Seeds: {SEEDS}
- 1-layer: [3072 -> m -> 10]
- 2-layer: [3072 -> m -> m -> 10]

## Results (mean +/- std accuracy)

{mk_table()}

## Depth Gain (2-layer vs 1-layer)

| Width | Iso gain | Base gain |
|---|---|---|
{chr(10).join(f"| {w} | {dg:.2f}% | {bg:.2f}% |" for w, dg, bg in zip(WIDTHS, depth_gains['Iso'], depth_gains['Base']))}

## Key Observations

1. **Does depth help isotropic more than baseline?**
   - Iso depth gain (mean): {np.mean(depth_gains['Iso']):.2f}%
   - Base depth gain (mean): {np.mean(depth_gains['Base']):.2f}%
   - {'Isotropic benefits more from depth' if np.mean(depth_gains['Iso']) > np.mean(depth_gains['Base']) else 'Baseline benefits more from depth'}

2. **Isotropic vs Baseline at 2 layers:**
   - Iso-2L mean: {np.mean([summary[('Iso-2L', w)][0] for w in WIDTHS]):.3f}
   - Base-2L mean: {np.mean([summary[('Base-2L', w)][0] for w in WIDTHS]):.3f}

3. **The nested functional class (Appendix C):**
   The paper proves that deep isotropic networks have a recursive structure
   where each added layer acts like a perturbative correction. This suggests
   depth should help more for isotropic than for standard networks.

![Depth comparison](depth_comparison.png)
"""

    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print("Results saved to results/test_M/results.md")


if __name__ == '__main__':
    main()
