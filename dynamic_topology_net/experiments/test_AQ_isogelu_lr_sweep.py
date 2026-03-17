"""
Test AQ -- IsoGELU Learning Rate Sweep
=======================================
Test AK showed IsoGELU/SiLU achieve only 23-33% accuracy vs IsoTanh 41-45%.
Two hypotheses:
  H1. Fundamental: non-saturating sigma allows magnitude explosion;
      the network is unstable regardless of LR.
  H2. LR artefact: LR=0.08 is too high for IsoGELU; lower LR stabilises it.

IsoTanh's saturation acts as implicit magnitude clipping (output norm ≤ 1).
IsoGELU has no such clip: sigma(r)=GELU(r)≈r for large r, so output norm
≈ input norm, and gradients can explode.

This test sweeps LR ∈ [0.001, 0.003, 0.01, 0.03, 0.08, 0.3] for:
  - IsoTanh-3L (reference)
  - IsoGELU-3L
  - IsoSiLU-3L

Width=32, 40 epochs (extra 10 to see whether late-training stabilises).
Seed=42.

Expected outcome under H1: IsoGELU plateaus ~23-35% across all LRs.
Expected outcome under H2: IsoGELU reaches competitive accuracy at low LR.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_topology_net.core import load_cifar10
from dynamic_topology_net.core.train_utils import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AQ')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS  = 40
BATCH   = 128
WIDTH   = 32
DEPTH   = 3
SEED    = 42

LRS = [0.001, 0.003, 0.01, 0.03, 0.08, 0.3]


class IsoAct(nn.Module):
    """f(x) = sigma(||x||) * x / ||x||  — equivariant under O(n) for any sigma."""
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return self.sigma(n) * x / n


class MLP(nn.Module):
    def __init__(self, input_dim, width, num_classes, depth, act):
        super().__init__()
        self.depth = depth
        dims = [input_dim] + [width] * depth
        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:]), 1):
            setattr(self, f'W{i}', nn.Linear(d_in, d_out, bias=False))
        self.out = nn.Linear(width, num_classes)
        self.act = act

    def forward(self, x):
        h = x
        for i in range(1, self.depth + 1):
            h = getattr(self, f'W{i}')(h)
            h = self.act(h)
        return self.out(h)


def train(model, train_loader, test_loader, lr):
    opt  = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    hist = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        acc = evaluate(model, test_loader, DEVICE)
        hist.append(acc)
    return hist


def main():
    print(f'Device: {DEVICE}')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    configs = [
        ('IsoTanh',  lambda r: torch.tanh(r)),
        ('IsoGELU',  lambda r: F.gelu(r)),
        ('IsoSiLU',  lambda r: F.silu(r)),
    ]

    # results[(act_name, lr)] = hist
    results = {}
    total = len(configs) * len(LRS)
    run = 0

    for act_name, sigma in configs:
        for lr in LRS:
            run += 1
            label = f'{act_name} lr={lr}'
            print(f'\n[{run}/{total}] {label}')
            torch.manual_seed(SEED)
            act   = IsoAct(sigma)
            model = MLP(input_dim, WIDTH, num_classes, DEPTH, act).to(DEVICE)
            hist  = train(model, train_loader, test_loader, lr)
            results[(act_name, lr)] = hist
            print(f'  ep10={hist[9]:.4f}  ep20={hist[19]:.4f}  '
                  f'ep40={hist[-1]:.4f}  peak={max(hist):.4f}')

    # Summary table
    print(f'\n{"="*70}')
    print(f'SUMMARY: Final accuracy (ep{EPOCHS})')
    print(f'{"LR":>8}' + ''.join(f'  {n:>10}' for n, _ in configs))
    for lr in LRS:
        row = f'{lr:>8.3f}'
        for act_name, _ in configs:
            acc = results[(act_name, lr)][-1]
            row += f'  {acc:.4f}    '
        print(row)

    print('\nBest LR per activation:')
    for act_name, _ in configs:
        best_lr  = max(LRS, key=lambda lr: results[(act_name, lr)][-1])
        best_acc = results[(act_name, best_lr)][-1]
        print(f'  {act_name}: best LR={best_lr}  acc={best_acc:.4f}')

    # Determine hypothesis
    tanh_best = max(results[('IsoTanh', lr)][-1] for lr in LRS)
    gelu_best = max(results[('IsoGELU', lr)][-1] for lr in LRS)
    gap = tanh_best - gelu_best
    print(f'\nIsoTanh best: {tanh_best:.4f}  IsoGELU best: {gelu_best:.4f}  gap: {gap:.4f}')
    if gap > 0.05:
        print('-> Gap persists across LRs: H1 supported (fundamental instability)')
    else:
        print('-> Gap closes at lower LR: H2 supported (LR artefact)')

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.cm.viridis
    lr_colours = {lr: cmap(i / (len(LRS) - 1)) for i, lr in enumerate(LRS)}
    epochs_range = list(range(1, EPOCHS + 1))

    for ax_i, (act_name, _) in enumerate(configs):
        ax = axes[ax_i]
        for lr in LRS:
            hist = results[(act_name, lr)]
            ax.plot(epochs_range, hist, color=lr_colours[lr],
                    label=f'lr={lr}')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
        ax.set_title(f'{act_name} — LR sweep (depth={DEPTH})')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle('Test AQ: IsoGELU/SiLU LR Sweep', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'lr_sweep.png'), dpi=150)
    print('\nPlot saved.')

    # Markdown results
    rows = []
    for lr in LRS:
        row_parts = [f'{lr}']
        for act_name, _ in configs:
            acc  = results[(act_name, lr)][-1]
            peak = max(results[(act_name, lr)])
            row_parts.append(f'{acc:.4f}/{peak:.4f}')
        rows.append('| ' + ' | '.join(row_parts) + ' |')

    md = f"""# Test AQ -- IsoGELU/SiLU Learning Rate Sweep

## Setup
- Depth={DEPTH}, Width={WIDTH}, {EPOCHS} epochs, seed={SEED}
- Device: {DEVICE}
- LRs tested: {LRS}

## Hypotheses
- H1 (fundamental): Non-saturating sigma -> magnitude explosion; fails at all LRs
- H2 (LR artefact): LR=0.08 too high for IsoGELU; low LR -> competitive

## Results (final acc / peak acc)

| LR | IsoTanh | IsoGELU | IsoSiLU |
|---|---|---|---|
{chr(10).join(rows)}

## Conclusion
- IsoTanh best: {tanh_best:.4f}
- IsoGELU best: {gelu_best:.4f}
- Gap: {gap:.4f}
- {'H1 supported: gap persists — fundamental instability' if gap > 0.05 else 'H2 supported: gap closes at low LR — LR artefact'}

![LR sweep](lr_sweep.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w') as f:
        f.write(md)
    print('Results saved.')


if __name__ == '__main__':
    main()
