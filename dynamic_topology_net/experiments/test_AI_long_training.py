"""
Test AI -- Long Training (100 epochs)
======================================
All prior tests used 24-30 epochs. Does Iso catch up to LN+tanh with more
training? Or does the gap stay fixed, implying a structural advantage for
normalised models rather than a convergence speed difference?

Trains all 5 core model types to 100 epochs at 3L, width=32.
Records accuracy at every epoch so convergence trajectories are visible.

Models: Base, Iso, LN+tanh, RMS+tanh, LN+Iso at 3L, width=32.
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AI')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS  = 100
LR      = 0.08
BATCH   = 128
WIDTH   = 32
DEPTH   = 3
SEED    = 42


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return self.scale * x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()


class IsoAct(nn.Module):
    def forward(self, x):
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.tanh(n) * x / n


class FlexMLP(nn.Module):
    def __init__(self, input_dim, width, num_classes, depth, act_fn, norm):
        super().__init__()
        self.depth = depth
        self.norm_type = norm
        self.act = act_fn
        dims = [input_dim] + [width] * depth
        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:]), 1):
            setattr(self, f'linear{i}', nn.Linear(d_in, d_out))
            if norm == 'LN':
                setattr(self, f'norm{i}', nn.LayerNorm(d_out))
            elif norm == 'RMS':
                setattr(self, f'norm{i}', RMSNorm(d_out))
        self.output = nn.Linear(width, num_classes)

    def forward(self, x):
        h = x
        for i in range(1, self.depth + 1):
            h = getattr(self, f'linear{i}')(h)
            if self.norm_type in ('LN', 'RMS'):
                h = getattr(self, f'norm{i}')(h)
            h = self.act(h)
        return self.output(h)


def train(model, train_loader, test_loader, tag):
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    acc_hist = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        acc = evaluate(model, test_loader, DEVICE)
        acc_hist.append(acc)
        if epoch in (1, 10, 24, 50, 75, EPOCHS):
            print(f'  [{tag}] Ep {epoch:3d}  acc={acc:.4f}')
    return acc_hist


def main():
    print(f'Device: {DEVICE}, Width={WIDTH}, Depth={DEPTH}, Epochs={EPOCHS}')
    print('Question: Does Iso catch up to LN+tanh with more training?')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    configs = [
        ('Base',     nn.Tanh(),  None),
        ('Iso',      IsoAct(),   None),
        ('LN+tanh',  nn.Tanh(),  'LN'),
        ('RMS+tanh', nn.SiLU(),  'RMS'),  # Note: using SiLU as RMS activation
        ('LN+Iso',   IsoAct(),   'LN'),
    ]
    # Fix: RMS+tanh should use Tanh
    configs = [
        ('Base',     nn.Tanh(),  None),
        ('Iso',      IsoAct(),   None),
        ('LN+tanh',  nn.Tanh(),  'LN'),
        ('RMS+tanh', nn.Tanh(),  'RMS'),
        ('LN+Iso',   IsoAct(),   'LN'),
    ]

    histories = {}
    for i, (tag, act, norm) in enumerate(configs, 1):
        print(f'\n[{i}/{len(configs)}] {tag}-{DEPTH}L (100 epochs)')
        torch.manual_seed(SEED)
        model = FlexMLP(input_dim, WIDTH, num_classes, DEPTH, act, norm).to(DEVICE)
        hist = train(model, train_loader, test_loader, tag)
        histories[tag] = hist
        print(f'  Final: {hist[-1]:.4f}  Peak: {max(hist):.4f}  at epoch {hist.index(max(hist))+1}')

    # Summary
    print(f'\n{"="*60}')
    print('SUMMARY (100 epochs, 3L, width=32)')
    for tag, _, _ in configs:
        h = histories[tag]
        print(f'  {tag:>12}: ep24={h[23]:.4f}  ep50={h[49]:.4f}  '
              f'ep100={h[-1]:.4f}  peak={max(h):.4f}@ep{h.index(max(h))+1}')

    iso_ep24  = histories['Iso'][23]
    iso_ep100 = histories['Iso'][-1]
    ln_ep24   = histories['LN+tanh'][23]
    ln_ep100  = histories['LN+tanh'][-1]
    gap_ep24  = ln_ep24  - iso_ep24
    gap_ep100 = ln_ep100 - iso_ep100

    print(f'\nLN+tanh - Iso gap: ep24={gap_ep24:+.4f}  ep100={gap_ep100:+.4f}')
    if gap_ep100 < gap_ep24 * 0.5:
        verdict = f'Gap SHRINKS from ep24 ({gap_ep24:+.4f}) to ep100 ({gap_ep100:+.4f}): Iso catching up'
    elif gap_ep100 > gap_ep24 * 0.8:
        verdict = f'Gap STABLE from ep24 ({gap_ep24:+.4f}) to ep100 ({gap_ep100:+.4f}): structural advantage'
    else:
        verdict = f'Gap PARTIALLY CLOSES from ep24 ({gap_ep24:+.4f}) to ep100 ({gap_ep100:+.4f})'
    print(f'Verdict: {verdict}')

    # Plot
    epochs_range = list(range(1, EPOCHS + 1))
    colors = {
        'Base': '#d62728', 'Iso': '#1f77b4', 'LN+tanh': '#2ca02c',
        'RMS+tanh': '#ff7f0e', 'LN+Iso': '#9467bd'
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for tag, _, _ in configs:
        ax.plot(epochs_range, histories[tag], color=colors[tag], label=tag)
    ax.axvline(24, color='black', ls=':', alpha=0.5, label='ep24 (prior tests)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Test accuracy')
    ax.set_title('Full 100-epoch training curves (3L, width=32)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    iso_h = np.array(histories['Iso'])
    for tag, _, _ in configs:
        if tag != 'Iso':
            delta = np.array(histories[tag]) - iso_h
            ax.plot(epochs_range, delta, color=colors[tag], label=f'{tag} - Iso')
    ax.axhline(0, color='#1f77b4', ls='--', alpha=0.5, label='Iso baseline')
    ax.axvline(24, color='black', ls=':', alpha=0.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy delta vs Iso')
    ax.set_title('Gap vs Iso over 100 epochs\n(stable = structural; shrinking = speed)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Test AI: Long Training - Does Iso Catch Up?', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'long_training.png'), dpi=150)
    print('\nPlot saved to results/test_AI/long_training.png')

    rows = []
    for tag, _, _ in configs:
        h = histories[tag]
        rows.append(f'| {tag} | {h[23]:.4f} | {h[49]:.4f} | {h[-1]:.4f} | '
                    f'{max(h):.4f} | ep{h.index(max(h))+1} |')

    md = f"""# Test AI -- Long Training (100 epochs)

## Setup
- Width: {WIDTH}, Depth: {DEPTH}, Epochs: {EPOCHS}, lr={LR}, seed={SEED}
- Device: {DEVICE}

## Question
Does Iso catch up to LN+tanh with more training (convergence speed) or does
the gap persist (structural advantage of normalisation)?

## Results

| Model | ep24 | ep50 | ep100 | Peak | Peak epoch |
|---|---|---|---|---|---|
{chr(10).join(rows)}

## Verdict
{verdict}

LN+tanh vs Iso gap at ep24: {gap_ep24:+.4f}
LN+tanh vs Iso gap at ep100: {gap_ep100:+.4f}

![Long training curves](long_training.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w') as f:
        f.write(md)
    print('Results saved to results/test_AI/results.md')


if __name__ == '__main__':
    main()
