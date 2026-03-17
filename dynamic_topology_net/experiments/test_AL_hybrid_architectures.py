"""
Test AL -- Hybrid Architectures (Iso topology layer + LN+GELU body)
====================================================================
Idea: keep LN+GELU everywhere for accuracy, but use Iso at layer
boundaries where topology changes (pruning/growth) happen.

A single Iso layer is O(n)-equivariant and supports exact scaffold
inertness and SV pruning at that layer. The LN+GELU layers around it
are untouched.

Tests at depth=3 and depth=4 (width=32, 30 epochs):
  - Pure Iso:         Iso  -> Iso  -> Iso
  - Pure LN+GELU:     LNG  -> LNG  -> LNG
  - Iso-first:        Iso  -> LNG  -> LNG   (topology at input boundary)
  - Iso-last:         LNG  -> LNG  -> Iso   (topology at output boundary)
  - Iso-sandwich:     Iso  -> LNG  -> Iso   (topology at both boundaries)
  - Alternating:      Iso  -> LNG  -> Iso  -> LNG  (depth=4)

Key questions:
  1. How much accuracy do you lose by inserting one Iso layer?
  2. Does position matter (first vs last)?
  3. Is the accuracy cost small enough to be worth the topology benefit?
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_topology_net.core import load_cifar10
from dynamic_topology_net.core.train_utils import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AL')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 30
LR     = 0.08
BATCH  = 128
WIDTH  = 32
SEED   = 42


class IsoAct(nn.Module):
    def forward(self, x):
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.tanh(n) * x / n


class HybridMLP(nn.Module):
    """
    Each layer defined by (use_ln, use_iso):
      (False, True)  -> Iso activation, no LN
      (True,  False) -> LN + GELU
    """
    def __init__(self, input_dim, width, num_classes, layer_specs):
        super().__init__()
        self.specs = layer_specs
        depth = len(layer_specs)
        dims = [input_dim] + [width] * depth
        self.iso_act = IsoAct()
        self.gelu    = nn.GELU()
        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:]), 1):
            setattr(self, f'W{i}', nn.Linear(d_in, d_out))
            use_ln, _ = layer_specs[i - 1]
            if use_ln:
                setattr(self, f'ln{i}', nn.LayerNorm(d_out))
        self.out = nn.Linear(width, num_classes)

    def forward(self, x):
        h = x
        for i, (use_ln, use_iso) in enumerate(self.specs, 1):
            h = getattr(self, f'W{i}')(h)
            if use_ln:
                h = getattr(self, f'ln{i}')(h)
            h = self.iso_act(h) if use_iso else self.gelu(h)
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

    # (tag, layer_specs)
    # layer_specs: list of (use_ln, use_iso) per layer
    ISO  = (False, True)   # Iso activation, no LN
    LNG  = (True,  False)  # LN + GELU

    configs_3L = [
        ('Pure-Iso-3L',       [ISO, ISO, ISO]),
        ('Pure-LN+GELU-3L',   [LNG, LNG, LNG]),
        ('Iso-first-3L',      [ISO, LNG, LNG]),
        ('Iso-last-3L',       [LNG, LNG, ISO]),
        ('Iso-sandwich-3L',   [ISO, LNG, ISO]),
    ]

    configs_4L = [
        ('Pure-Iso-4L',       [ISO, ISO, ISO, ISO]),
        ('Pure-LN+GELU-4L',   [LNG, LNG, LNG, LNG]),
        ('Iso-first-4L',      [ISO, LNG, LNG, LNG]),
        ('Iso-last-4L',       [LNG, LNG, LNG, ISO]),
        ('Iso-sandwich-4L',   [ISO, LNG, LNG, ISO]),
        ('Alternating-4L',    [ISO, LNG, ISO, LNG]),
    ]

    all_configs = configs_3L + configs_4L
    results = {}
    total   = len(all_configs)

    for run, (tag, specs) in enumerate(all_configs, 1):
        print(f'\n[{run}/{total}] {tag}  layers={["Iso" if s[1] else "LN+GELU" for s in specs]}')
        torch.manual_seed(SEED)
        model = HybridMLP(input_dim, WIDTH, num_classes, specs).to(DEVICE)
        hist  = train(model, train_loader, test_loader, tag)
        results[tag] = hist
        print(f'  Final: {hist[-1]:.4f}  Peak: {max(hist):.4f}')

    # Summary
    print(f'\n{"="*70}')
    print('SUMMARY')
    print(f'{"Model":<24}  {"Final":>7}  {"Peak":>7}  {"vs Pure-Iso":>12}  {"vs Pure-LNG":>12}')

    iso3  = results['Pure-Iso-3L'][-1]
    lng3  = results['Pure-LN+GELU-3L'][-1]
    iso4  = results['Pure-Iso-4L'][-1]
    lng4  = results['Pure-LN+GELU-4L'][-1]

    for tag, specs in all_configs:
        depth    = len(specs)
        ref_iso  = iso3 if depth == 3 else iso4
        ref_lng  = lng3 if depth == 3 else lng4
        final    = results[tag][-1]
        peak     = max(results[tag])
        n_iso    = sum(1 for s in specs if s[1])
        print(f'{tag:<24}  {final:.4f}  {peak:.4f}  '
              f'{final-ref_iso:+.4f} vs Iso  {final-ref_lng:+.4f} vs LNG  '
              f'({n_iso}/{depth} Iso layers)')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = list(range(1, EPOCHS + 1))
    colors = {
        'Pure-Iso':      '#1f77b4',
        'Pure-LN+GELU':  '#2ca02c',
        'Iso-first':     '#ff7f0e',
        'Iso-last':      '#d62728',
        'Iso-sandwich':  '#9467bd',
        'Alternating':   '#8c564b',
    }

    for ax_idx, (configs, depth) in enumerate([(configs_3L, 3), (configs_4L, 4)]):
        ax = axes[ax_idx]
        for tag, _ in configs:
            short = tag.replace(f'-{depth}L', '')
            c = colors.get(short, '#7f7f7f')
            ax.plot(epochs_range, results[tag], color=c, label=tag)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
        ax.set_title(f'Depth {depth}: hybrid vs pure architectures')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle('Test AL: Hybrid Iso/LN+GELU Architectures', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'hybrid.png'), dpi=150)
    print('\nPlot saved.')

    rows = []
    for tag, specs in all_configs:
        depth   = len(specs)
        n_iso   = sum(1 for s in specs if s[1])
        final   = results[tag][-1]
        peak    = max(results[tag])
        layer_s = ','.join('Iso' if s[1] else 'LNG' for s in specs)
        rows.append(f'| {tag} | {layer_s} | {n_iso}/{depth} | {final:.4f} | {peak:.4f} |')

    md = f"""# Test AL -- Hybrid Architectures

## Setup
- Width: {WIDTH}, Epochs: {EPOCHS}, lr={LR}, seed={SEED}
- Device: {DEVICE}

## Question
Does inserting Iso layers at topology boundaries preserve accuracy
while enabling dynamic topology at those layers?

## Results

| Model | Layers | Iso count | Final | Peak |
|---|---|---|---|---|
{chr(10).join(rows)}

## 3L references
- Pure-Iso-3L: {iso3:.4f}
- Pure-LN+GELU-3L: {lng3:.4f}

## 4L references
- Pure-Iso-4L: {iso4:.4f}
- Pure-LN+GELU-4L: {lng4:.4f}

![Hybrid architectures](hybrid.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w') as f:
        f.write(md)
    print('Results saved.')


if __name__ == '__main__':
    main()
