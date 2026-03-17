"""
Test AK -- Isotropic Activation Variants (IsoGELU, IsoSiLU)
=============================================================
The scalar function σ in f(x) = σ(‖x‖)·x̂ doesn't have to be tanh.
Any scalar function applied to the norm keeps the activation O(n)-equivariant
and therefore supports dynamic topology.

GELU and SiLU have better gradient profiles than tanh for deep networks:
- tanh saturates to 1 for large r (gradient → 0)
- GELU(r) ≈ r for large r (gradient ≈ 1, no saturation)
- SiLU(r) = r·σ(r) ≈ r for large r (same)

This means IsoGELU/SiLU may avoid the depth collapse seen in IsoTanh at 5-6L.

Tests: IsoTanh, IsoGELU, IsoSiLU, IsoSoftplus at depths 1,2,3,4,5.
Width=32, 30 epochs, seed=42.

Question: Does the choice of σ matter? Can IsoGELU/SiLU push the depth
ceiling beyond IsoTanh's 4L?
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AK')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 30
LR     = 0.08
BATCH  = 128
WIDTH  = 32
SEED   = 42
DEPTHS = [1, 2, 3, 4, 5]


class IsoAct(nn.Module):
    """f(x) = sigma(||x||) * x / ||x||  — equivariant under O(n) for any sigma."""
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma  # scalar function: R -> R

    def forward(self, x):
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return self.sigma(n) * x / n


class MLP(nn.Module):
    def __init__(self, input_dim, width, num_classes, depth, act):
        super().__init__()
        self.depth = depth
        dims = [input_dim] + [width] * depth
        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:]), 1):
            setattr(self, f'W{i}', nn.Linear(d_in, d_out))
        self.out = nn.Linear(width, num_classes)
        self.act = act

    def forward(self, x):
        h = x
        for i in range(1, self.depth + 1):
            h = getattr(self, f'W{i}')(h)
            h = self.act(h)
        return self.out(h)


def train(model, train_loader, test_loader, tag):
    opt  = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    hist = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        acc = evaluate(model, test_loader, DEVICE)
        hist.append(acc)
        if epoch in (1, 10, 20, EPOCHS):
            print(f'  [{tag}] Ep {epoch:2d}  acc={acc:.4f}')
    return hist


def main():
    print(f'Device: {DEVICE}')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    # Define sigma functions — each maps a non-negative scalar tensor to scalar
    # tanh: saturates at 1
    # GELU: smooth, grows ~linearly for large r
    # SiLU: r*sigmoid(r), grows ~linearly for large r
    # Softplus: log(1+exp(r)), always positive and smooth
    configs = [
        ('IsoTanh',     lambda r: torch.tanh(r)),
        ('IsoGELU',     lambda r: F.gelu(r)),
        ('IsoSiLU',     lambda r: F.silu(r)),
        ('IsoSoftplus', lambda r: F.softplus(r)),
    ]

    results = {}  # (tag, depth) -> acc history
    total = len(configs) * len(DEPTHS)
    run = 0

    for tag, sigma in configs:
        for depth in DEPTHS:
            run += 1
            label = f'{tag}-{depth}L'
            print(f'\n[{run}/{total}] {label}')
            torch.manual_seed(SEED)
            act   = IsoAct(sigma)
            model = MLP(input_dim, WIDTH, num_classes, depth, act).to(DEVICE)
            hist  = train(model, train_loader, test_loader, label)
            results[(tag, depth)] = hist
            print(f'  Final: {hist[-1]:.4f}  Peak: {max(hist):.4f}')

    # Summary
    print(f'\n{"="*65}')
    print(f'SUMMARY: Final accuracy by depth (width={WIDTH}, {EPOCHS} epochs)')
    print(f'{"Model":>14}' + ''.join(f'  {d}L' for d in DEPTHS) + '  Best depth')
    for tag, _ in configs:
        accs = [results[(tag, d)][-1] for d in DEPTHS]
        best = DEPTHS[int(np.argmax(accs))]
        row  = f'{tag:>14}' + ''.join(f'  {a:.3f}' for a in accs) + f'  {best}L'
        print(row)

    # Per-tag slope
    print('\nLinear fit slope (acc/layer):')
    for tag, _ in configs:
        accs  = [results[(tag, d)][-1] for d in DEPTHS]
        slope = np.polyfit(DEPTHS, accs, 1)[0]
        print(f'  {tag}: {slope:+.4f}/layer')

    # Plot 1: accuracy vs depth
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'IsoTanh': '#1f77b4', 'IsoGELU': '#2ca02c',
              'IsoSiLU': '#ff7f0e', 'IsoSoftplus': '#9467bd'}

    ax = axes[0]
    for tag, _ in configs:
        accs = [results[(tag, d)][-1] for d in DEPTHS]
        ax.plot(DEPTHS, accs, 'o-', color=colors[tag], label=tag)
    ax.set_xlabel('Depth'); ax.set_ylabel('Test accuracy')
    ax.set_title(f'Accuracy vs Depth (width={WIDTH}, {EPOCHS} epochs)')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(DEPTHS)

    # Plot 2: training curves at depth 3
    ax = axes[1]
    epochs_range = list(range(1, EPOCHS + 1))
    for tag, _ in configs:
        ax.plot(epochs_range, results[(tag, 3)], color=colors[tag], label=f'{tag}-3L')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title(f'Training curves at depth=3 (width={WIDTH})')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('Test AK: Isotropic Activation Variants', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'iso_variants.png'), dpi=150)
    print('\nPlot saved.')

    # Results table
    rows = []
    for tag, _ in configs:
        accs  = [results[(tag, d)][-1] for d in DEPTHS]
        slope = np.polyfit(DEPTHS, accs, 1)[0]
        rows.append('| ' + tag + ' | ' +
                    ' | '.join(f'{a:.4f}' for a in accs) +
                    f' | {slope:+.4f} |')

    md = f"""# Test AK -- Isotropic Activation Variants

## Setup
- Width: {WIDTH}, Epochs: {EPOCHS}, lr={LR}, seed={SEED}
- Device: {DEVICE}

## Question
Does replacing tanh with GELU/SiLU in f(x)=sigma(||x||)*x/||x|| improve accuracy?
Can non-saturating sigma push Iso's depth ceiling beyond 4L?

## Results

| Model | {'  |  '.join(f'{d}L' for d in DEPTHS)} | Slope |
|---{'|---' * (len(DEPTHS) + 1)}|
{chr(10).join(rows)}

## Interpretation
- IsoTanh: saturates sigma(r)->1 for large r; tangential gradient preserved but radial shrinks
- IsoGELU/SiLU: sigma(r)~r for large r; no saturation ceiling; radial gradient stays ~1
- IsoSoftplus: log(1+exp(r)), smooth and always positive, intermediate saturation

"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w') as f:
        f.write(md)
    print('Results saved.')


if __name__ == '__main__':
    main()
