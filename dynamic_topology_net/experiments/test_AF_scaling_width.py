"""
Test AF -- Width Scaling (32 -> 64 -> 128 -> 256 -> 512)
======================================================
All prior experiments used width=32. This test creeps up through widths
[32, 64, 128, 256, 512] to check whether the key findings generalise.

Key questions at each width:
  1. Does Iso consistently outperform Base? (Does the +14-16% gap persist?)
  2. Does Iso depth stability hold? (Iso improves, Base degrades at depth)
  3. Does the gradient anatomy from Test AD persist?
     (Base output-layer gradient 7-8x larger than Iso)
  4. How does training time scale?

Models: Base-2L and Iso-2L at each width (2L is the sweet spot from Tests E/Q)
Also Base-1L and Iso-1L for reference.
Widths: [32, 64, 128, 256, 512]
Epochs: 24 throughout (same budget at each scale).
Seed: 42.

Note: at width=512, CIFAR-10 may be close to saturated with this architecture.
The scaling trend matters more than absolute numbers.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from dynamic_topology_net.core import load_cifar10
from dynamic_topology_net.core.train_utils import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AF')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS        = 24
LR            = 0.08
BATCH         = 128
SEED          = 42
WIDTHS        = [32, 64, 128, 256, 512]
SAT_THRESHOLD = 2.646


# =============================================================================
# Models
# =============================================================================

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
        h = self.out(h)
        return (h, preacts) if return_preacts else h


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
        h = self.out(h)
        return (h, preacts) if return_preacts else h


# =============================================================================
# Training with anatomy
# =============================================================================

def train_and_measure(model, model_type, train_loader, test_loader,
                      probe_x, device, tag):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    history   = defaultdict(list)
    t_start   = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_grad = defaultdict(list)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            for i in range(1, model.depth + 1):
                W = getattr(model, f'W{i}')
                if W.weight.grad is not None:
                    epoch_grad[f'W{i}'].append(W.weight.grad.norm().item())
            if model.out.weight.grad is not None:
                epoch_grad[f'W{model.depth+1}'].append(
                    model.out.weight.grad.norm().item())
            optimizer.step()

        acc = evaluate(model, test_loader, device)
        history['acc'].append(acc)

        model.eval()
        with torch.no_grad():
            _, preacts = model(probe_x, return_preacts=True)
        for li, pa in enumerate(preacts, 1):
            if model_type == 'Base':
                sat = (pa.abs() > SAT_THRESHOLD).float().mean().item()
            else:
                sat = (pa.norm(dim=-1) > SAT_THRESHOLD).float().mean().item()
            history[f'sat_L{li}'].append(sat)

        for key, vals in epoch_grad.items():
            history[f'grad_{key}'].append(np.mean(vals))

        if epoch in (1, 12, EPOCHS):
            g1  = history.get('grad_W1', [0])[-1]
            gl  = history.get(f'grad_W{model.depth+1}', [0])[-1]
            sat1 = history.get('sat_L1', [0])[-1]
            print(f'  [{tag}] Ep {epoch:2d}  acc={acc:.4f}  '
                  f'g1={g1:.5f}  g_last={gl:.5f}  sat_L1={sat1:.3f}')

    elapsed = time.time() - t_start
    history['train_time'] = elapsed
    return history


# =============================================================================
# Main
# =============================================================================

def main():
    print(f'Device: {DEVICE}')
    print(f'Widths: {WIDTHS}')
    print(f'Question: Do Iso advantages persist as width scales 32 -> 512?')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    torch.manual_seed(SEED)
    probe_x, _ = next(iter(test_loader))
    probe_x = probe_x[:512].to(DEVICE)

    all_results = {}  # (width, model_type, depth) -> history

    for width in WIDTHS:
        for depth in (1, 2):
            for model_type, cls in (('Base', BaseMLP), ('Iso', IsoMLP)):
                tag = f'{model_type}-{depth}L-w{width}'
                print(f'\n--- {tag} ---')
                torch.manual_seed(SEED)
                model = cls(input_dim, width, num_classes, depth).to(DEVICE)
                n_params = sum(p.numel() for p in model.parameters())
                print(f'  Params: {n_params:,}')
                hist = train_and_measure(
                    model, model_type, train_loader, test_loader,
                    probe_x, DEVICE, tag
                )
                all_results[(width, model_type, depth)] = hist
                print(f'  Final acc: {hist["acc"][-1]:.4f}  '
                      f'time: {hist["train_time"]:.1f}s')

    # =========================================================================
    # Summary
    # =========================================================================
    print(f'\n{"="*70}')
    print('SUMMARY: Accuracy and Iso-Base gap vs width')
    print(f'{"="*70}')
    print(f'{"Width":>6}  {"Base-1L":>8}  {"Iso-1L":>8}  {"Gap-1L":>8}  '
          f'{"Base-2L":>8}  {"Iso-2L":>8}  {"Gap-2L":>8}')

    for width in WIDTHS:
        b1 = all_results[(width, 'Base', 1)]['acc'][-1]
        i1 = all_results[(width, 'Iso',  1)]['acc'][-1]
        b2 = all_results[(width, 'Base', 2)]['acc'][-1]
        i2 = all_results[(width, 'Iso',  2)]['acc'][-1]
        print(f'{width:>6}  {b1:.4f}  {i1:.4f}  {i1-b1:+.4f}  '
              f'{b2:.4f}  {i2:.4f}  {i2-b2:+.4f}')

    print(f'\nGradient ratio W1/W_last (Base-2L epoch-mean):')
    for width in WIDTHS:
        h = all_results[(width, 'Base', 2)]
        g1 = np.mean(h.get('grad_W1', [1e-10]))
        gl = np.mean(h.get('grad_W3', [1e-10]))
        print(f'  w={width}: {g1/gl:.4f}  (g1={g1:.5f}  g_last={gl:.5f})')

    print(f'\nGradient ratio W1/W_last (Iso-2L epoch-mean):')
    for width in WIDTHS:
        h = all_results[(width, 'Iso', 2)]
        g1 = np.mean(h.get('grad_W1', [1e-10]))
        gl = np.mean(h.get('grad_W3', [1e-10]))
        print(f'  w={width}: {g1/gl:.4f}  (g1={g1:.5f}  g_last={gl:.5f})')

    # =========================================================================
    # Plots
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    epochs_range = list(range(1, EPOCHS + 1))
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(WIDTHS)-1)) for i in range(len(WIDTHS))]

    # Iso-Base accuracy gap vs width (final epoch)
    ax = axes[0, 0]
    for depth, ls in [(1, '--'), (2, '-')]:
        gaps = [all_results[(w, 'Iso', depth)]['acc'][-1] -
                all_results[(w, 'Base', depth)]['acc'][-1]
                for w in WIDTHS]
        ax.plot(WIDTHS, gaps, marker='o', ls=ls, label=f'{depth}L gap')
    ax.set_xlabel('Width')
    ax.set_xscale('log', base=2)
    ax.set_ylabel('Iso - Base accuracy')
    ax.set_title('Iso advantage vs width\n(persistent = finding generalises)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', ls=':', alpha=0.4)

    # Absolute accuracy vs width
    ax = axes[0, 1]
    for model_type, ls in (('Base', '--'), ('Iso', '-')):
        for depth, marker in ((1, 's'), (2, 'o')):
            accs = [all_results[(w, model_type, depth)]['acc'][-1] for w in WIDTHS]
            ax.plot(WIDTHS, accs, marker=marker, ls=ls,
                    label=f'{model_type}-{depth}L')
    ax.set_xlabel('Width')
    ax.set_xscale('log', base=2)
    ax.set_ylabel('Test accuracy')
    ax.set_title('Absolute accuracy vs width')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Training time vs width
    ax = axes[0, 2]
    for model_type, ls in (('Base', '--'), ('Iso', '-')):
        times = [all_results[(w, model_type, 2)]['train_time'] for w in WIDTHS]
        ax.plot(WIDTHS, times, marker='o', ls=ls, label=f'{model_type}-2L')
    ax.set_xlabel('Width')
    ax.set_xscale('log', base=2)
    ax.set_ylabel('Training time (s, 24 epochs)')
    ax.set_title('Training time vs width')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient ratio Base-2L vs width over training
    ax = axes[1, 0]
    for width, color in zip(WIDTHS, colors):
        h = all_results[(width, 'Base', 2)]
        g1 = np.array(h.get('grad_W1', [1e-10]*EPOCHS))
        gl = np.array(h.get('grad_W3', [1e-10]*EPOCHS))
        ax.plot(epochs_range, g1/(gl+1e-10), color=color, label=f'w={width}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('W1/W_last gradient ratio')
    ax.set_title('Base-2L gradient ratio over training')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Gradient ratio Iso-2L vs width over training
    ax = axes[1, 1]
    for width, color in zip(WIDTHS, colors):
        h = all_results[(width, 'Iso', 2)]
        g1 = np.array(h.get('grad_W1', [1e-10]*EPOCHS))
        gl = np.array(h.get('grad_W3', [1e-10]*EPOCHS))
        ax.plot(epochs_range, g1/(gl+1e-10), color=color, label=f'w={width}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('W1/W_last gradient ratio')
    ax.set_title('Iso-2L gradient ratio over training')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Saturation fraction Base-2L vs width
    ax = axes[1, 2]
    for width, color in zip(WIDTHS, colors):
        h = all_results[(width, 'Base', 2)]
        ax.plot(epochs_range, h.get('sat_L1', [0]*EPOCHS),
                color=color, label=f'w={width}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Saturation fraction (L1)')
    ax.set_title('Base-2L saturation vs width\n(does wider = less saturated?)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Test AF: Width Scaling (32->512)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'width_scaling.png'), dpi=150)
    print('\nPlot saved to results/test_AF/width_scaling.png')

    # =========================================================================
    # Results.md
    # =========================================================================
    rows_acc = []
    for width in WIDTHS:
        b1 = all_results[(width, 'Base', 1)]['acc'][-1]
        i1 = all_results[(width, 'Iso',  1)]['acc'][-1]
        b2 = all_results[(width, 'Base', 2)]['acc'][-1]
        i2 = all_results[(width, 'Iso',  2)]['acc'][-1]
        t  = all_results[(width, 'Iso',  2)]['train_time']
        rows_acc.append(
            f'| {width} | {b1:.4f} | {i1:.4f} | {i1-b1:+.4f} | '
            f'{b2:.4f} | {i2:.4f} | {i2-b2:+.4f} | {t:.0f}s |'
        )

    # Check if gap is stable
    gaps_2L = [all_results[(w, 'Iso', 2)]['acc'][-1] -
               all_results[(w, 'Base', 2)]['acc'][-1] for w in WIDTHS]
    gap_trend = 'STABLE' if (max(gaps_2L) - min(gaps_2L)) < 0.05 else \
                'GROWING' if gaps_2L[-1] > gaps_2L[0] else 'SHRINKING'

    md = f"""# Test AF -- Width Scaling (32 -> 512)

## Setup
- Models: Base and Iso at 1L and 2L depth
- Widths: {WIDTHS}
- Epochs: {EPOCHS}, lr={LR}, batch={BATCH}, seed={SEED}
- Device: {DEVICE}

## Question
Do the key findings from width=32 persist as width scales to 512?
Specifically: does the Iso > Base advantage hold at larger scale?

## Results

| Width | Base-1L | Iso-1L | Gap-1L | Base-2L | Iso-2L | Gap-2L | Time(2L) |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows_acc)}

## Gap trend at 2L: {gap_trend}
Min gap: {min(gaps_2L):+.4f} (w={WIDTHS[gaps_2L.index(min(gaps_2L))]})
Max gap: {max(gaps_2L):+.4f} (w={WIDTHS[gaps_2L.index(max(gaps_2L))]})

## Gradient anatomy (Base-2L epoch-mean W1/W_last ratio)
"""
    for width in WIDTHS:
        h = all_results[(width, 'Base', 2)]
        g1 = np.mean(h.get('grad_W1', [1e-10]))
        gl = np.mean(h.get('grad_W3', [1e-10]))
        md += f'- w={width}: {g1/gl:.4f}\n'

    md += '\n## Gradient anatomy (Iso-2L epoch-mean W1/W_last ratio)\n'
    for width in WIDTHS:
        h = all_results[(width, 'Iso', 2)]
        g1 = np.mean(h.get('grad_W1', [1e-10]))
        gl = np.mean(h.get('grad_W3', [1e-10]))
        md += f'- w={width}: {g1/gl:.4f}\n'

    md += '\n![Width scaling results](width_scaling.png)\n'

    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    print('Results saved to results/test_AF/results.md')


if __name__ == '__main__':
    main()
