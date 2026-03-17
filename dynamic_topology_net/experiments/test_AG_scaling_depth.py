"""
Test AG -- Depth Scaling at Larger Width (width=128, depth 1→6)
===============================================================
Test AF checks width scaling at fixed depth (2L).
Test AG fixes a larger width (128 -- midpoint of AF's range) and pushes
depth further: 1, 2, 3, 4, 5, 6 layers.

At width=32, the pattern was:
  - Iso: +3.4% per added layer (stable)
  - Base: -6.1% per added layer (degrades)

Questions:
  1. Does Iso keep improving at depth with more width?
  2. At what depth does Iso plateau or degrade?
  3. Does Base still fail at depth with width=128? (vs width=32)
  4. Does the gradient anatomy (saturation, W1/W_last) tell the same story?

Width=128 gives more representational capacity, which might:
  - Allow Iso to scale to greater depth before plateauing
  - Or make Base slightly more stable (more redundant neurons)

Epochs: 30 (slightly more to let deeper models converge).
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
from collections import defaultdict

from dynamic_topology_net.core import load_cifar10
from dynamic_topology_net.core.train_utils import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AG')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS        = 30
LR            = 0.08
BATCH         = 128
WIDTH         = 128
SEED          = 42
DEPTHS        = [1, 2, 3, 4, 5, 6]
SAT_THRESHOLD = 2.646


class BaseMLP(nn.Module):
    def __init__(self, input_dim, width, num_classes, depth):
        super().__init__()
        self.depth = depth
        dims = [input_dim] + [width] * depth
        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:]), 1):
            setattr(self, f'W{i}', nn.Linear(d_in, d_out))
        self.out = nn.Linear(width, num_classes)

    def forward(self, x, return_preacts=False):
        h, preacts = x, []
        for i in range(1, self.depth + 1):
            h = getattr(self, f'W{i}')(h)
            if return_preacts:
                preacts.append(h.detach().cpu())
            h = torch.tanh(h)
        return (self.out(h), preacts) if return_preacts else self.out(h)


class IsoMLP(nn.Module):
    def __init__(self, input_dim, width, num_classes, depth):
        super().__init__()
        self.depth = depth
        dims = [input_dim] + [width] * depth
        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:]), 1):
            setattr(self, f'W{i}', nn.Linear(d_in, d_out))
        self.out = nn.Linear(width, num_classes)

    def forward(self, x, return_preacts=False):
        h, preacts = x, []
        for i in range(1, self.depth + 1):
            h = getattr(self, f'W{i}')(h)
            if return_preacts:
                preacts.append(h.detach().cpu())
            norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            h = torch.tanh(norm) * h / norm
        return (self.out(h), preacts) if return_preacts else self.out(h)


def train_model(model, model_type, train_loader, test_loader, probe_x, tag):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    history   = defaultdict(list)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_grad = defaultdict(list)
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            for i in range(1, model.depth + 1):
                W = getattr(model, f'W{i}')
                if W.weight.grad is not None:
                    epoch_grad[f'W{i}'].append(W.weight.grad.norm().item())
            if model.out.weight.grad is not None:
                epoch_grad[f'W_out'].append(model.out.weight.grad.norm().item())
            optimizer.step()

        acc = evaluate(model, test_loader, DEVICE)
        history['acc'].append(acc)

        model.eval()
        with torch.no_grad():
            _, preacts = model(probe_x, return_preacts=True)
        for li, pa in enumerate(preacts, 1):
            sat = (pa.abs() > SAT_THRESHOLD).float().mean().item() \
                  if model_type == 'Base' \
                  else (pa.norm(dim=-1) > SAT_THRESHOLD).float().mean().item()
            history[f'sat_L{li}'].append(sat)

        for key, vals in epoch_grad.items():
            history[f'grad_{key}'].append(np.mean(vals))

        if epoch in (1, 10, 20, EPOCHS):
            g1  = history.get('grad_W1', [0])[-1]
            gout = history.get('grad_W_out', [0])[-1]
            print(f'  [{tag}] Ep {epoch:2d}  acc={acc:.4f}  '
                  f'g1={g1:.5f}  g_out={gout:.5f}')

    return history


def main():
    print(f'Device: {DEVICE}, Width={WIDTH}')
    print(f'Depths: {DEPTHS}')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    torch.manual_seed(SEED)
    probe_x, _ = next(iter(test_loader))
    probe_x = probe_x[:512].to(DEVICE)

    results = {}
    total = len(DEPTHS) * 2

    for depth in DEPTHS:
        for model_type, cls in (('Base', BaseMLP), ('Iso', IsoMLP)):
            tag = f'{model_type}-{depth}L'
            run = (depth - 1) * 2 + (0 if model_type == 'Base' else 1) + 1
            print(f'\n[{run}/{total}] {tag} (width={WIDTH})')
            torch.manual_seed(SEED)
            model = cls(input_dim, WIDTH, num_classes, depth).to(DEVICE)
            hist = train_model(model, model_type, train_loader, test_loader,
                               probe_x, tag)
            results[(depth, model_type)] = hist
            print(f'  Final: {hist["acc"][-1]:.4f}')

    # =========================================================================
    # Summary
    # =========================================================================
    print(f'\n{"="*60}')
    print(f'SUMMARY (width={WIDTH})')
    print(f'{"="*60}')
    print(f'{"Depth":>6}  {"Base":>8}  {"Iso":>8}  {"Gap":>8}  '
          f'{"Base Delta":>8}  {"Iso Delta":>8}')

    base1 = results[(1, 'Base')]['acc'][-1]
    iso1  = results[(1, 'Iso')]['acc'][-1]
    for depth in DEPTHS:
        b = results[(depth, 'Base')]['acc'][-1]
        i = results[(depth, 'Iso')]['acc'][-1]
        print(f'{depth:>6}  {b:.4f}  {i:.4f}  {i-b:+.4f}  '
              f'{b-base1:+.4f}  {i-iso1:+.4f}')

    # Estimate per-layer delta
    if len(DEPTHS) >= 2:
        base_accs = [results[(d, 'Base')]['acc'][-1] for d in DEPTHS]
        iso_accs  = [results[(d, 'Iso')]['acc'][-1]  for d in DEPTHS]
        base_slope = np.polyfit(DEPTHS, base_accs, 1)[0]
        iso_slope  = np.polyfit(DEPTHS, iso_accs,  1)[0]
        print(f'\nLinear fit: Base slope={base_slope:+.4f}/layer  '
              f'Iso slope={iso_slope:+.4f}/layer')

    # =========================================================================
    # Plots
    # =========================================================================
    epochs_range = list(range(1, EPOCHS + 1))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Accuracy vs depth
    ax = axes[0, 0]
    base_accs = [results[(d, 'Base')]['acc'][-1] for d in DEPTHS]
    iso_accs  = [results[(d, 'Iso')]['acc'][-1]  for d in DEPTHS]
    ax.plot(DEPTHS, base_accs, 'o-', color='#d62728', label='Base')
    ax.plot(DEPTHS, iso_accs,  'o-', color='#1f77b4', label='Iso')
    ax.set_xlabel('Depth (hidden layers)')
    ax.set_ylabel('Test accuracy')
    ax.set_title(f'Accuracy vs Depth (width={WIDTH})\n'
                 f'Base slope={base_slope:+.3f}  Iso slope={iso_slope:+.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(DEPTHS)

    # Accuracy curves for all depths
    ax = axes[0, 1]
    cmap = plt.cm.viridis
    colors = [cmap(i/(len(DEPTHS)-1)) for i in range(len(DEPTHS))]
    for depth, color in zip(DEPTHS, colors):
        ax.plot(epochs_range, results[(depth, 'Iso')]['acc'],
                color=color, ls='-',  label=f'Iso-{depth}L')
        ax.plot(epochs_range, results[(depth, 'Base')]['acc'],
                color=color, ls='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Training curves (solid=Iso, dashed=Base, width={WIDTH})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # W1/W_out gradient ratio
    ax = axes[1, 0]
    for depth, color in zip(DEPTHS, colors):
        for mtype, ls in (('Base', '--'), ('Iso', '-')):
            h = results[(depth, mtype)]
            g1   = np.array(h.get('grad_W1', [1e-10]*EPOCHS))
            gout = np.array(h.get('grad_W_out', [1e-10]*EPOCHS))
            ax.plot(epochs_range, g1/(gout+1e-10),
                    color=color, ls=ls,
                    label=f'{mtype}-{depth}L' if depth in (1, 3, 6) else None)
    ax.axhline(1.0, color='black', ls=':', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('W1/W_out gradient ratio')
    ax.set_title(f'Gradient balance (solid=Iso, dashed=Base, width={WIDTH})')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # Saturation (Base only)
    ax = axes[1, 1]
    for depth, color in zip(DEPTHS, colors):
        h = results[(depth, 'Base')]
        mean_sat = [
            np.mean([h.get(f'sat_L{li}', [0])[ep] for li in range(1, depth+1)])
            for ep in range(EPOCHS)
        ]
        ax.plot(epochs_range, mean_sat, color=color, label=f'Base-{depth}L')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean saturation fraction')
    ax.set_title(f'Base saturation vs depth (width={WIDTH})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Test AG: Depth Scaling at width={WIDTH}', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'depth_scaling.png'), dpi=150)
    print('\nPlot saved to results/test_AG/depth_scaling.png')

    # Results.md
    rows = []
    for depth in DEPTHS:
        b = results[(depth, 'Base')]['acc'][-1]
        i = results[(depth, 'Iso')]['acc'][-1]
        rows.append(f'| {depth} | {b:.4f} | {i:.4f} | {i-b:+.4f} | '
                    f'{b-base1:+.4f} | {i-iso1:+.4f} |')

    md = f"""# Test AG -- Depth Scaling at width={WIDTH}

## Setup
- Models: Base and Iso MLP
- Width: {WIDTH} (larger than prior tests at width=32)
- Depths: {DEPTHS}
- Epochs: {EPOCHS}, lr={LR}, seed={SEED}
- Device: {DEVICE}

## Question
Does Iso depth stability hold at larger width?
At what depth does Iso plateau?

## Results

| Depth | Base | Iso | Gap | Base Delta from 1L | Iso Delta from 1L |
|---|---|---|---|---|---|
{chr(10).join(rows)}

## Per-layer slope (linear fit)
- Base: {base_slope:+.4f} per layer
- Iso:  {iso_slope:+.4f} per layer

(For reference, at width=32: Base=-0.061/layer, Iso=+0.034/layer)

![Depth scaling](depth_scaling.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    print('Results saved to results/test_AG/results.md')


if __name__ == '__main__':
    main()
