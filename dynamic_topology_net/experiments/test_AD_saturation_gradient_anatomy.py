"""
Test AD -- Saturation & Gradient Anatomy
==========================================
Prior tests have narrowed Base depth failure to a specific mechanism:
  - NOT representational collapse (Test AC: Base PR stays broad ~18-22)
  - NOT fixable by regularisation (Test X: 0% improvement at any lambda)
  - NOT global gradient vanishing (Test R: Base gradients are larger, not smaller)

Remaining hypothesis: per-neuron tanh saturation compounds with depth.

  Base (elementwise tanh):
    - Each neuron i has pre-activation h_i; saturates when |h_i| > 2.65 (tanh ≈ ±0.99)
    - Gradient through saturated neuron ≈ 0 (sech²(h_i) ≈ 0)
    - At depth, many neurons may saturate, blocking gradient to early layers

  Iso (isotropic tanh):
    - Scales whole vector by tanh(||h||)/||h||; no per-component saturation
    - Radial gradient = sech²(||h||); tangential = tanh(||h||)/||h||
    - Even when ||h|| is large, tangential gradient survives
    - Direction is always preserved

Measured at every epoch on a fixed probe batch (1024 samples):
  1. Saturation fraction per layer
     - Base: fraction of neurons where |h_i| > 2.646 (tanh(h_i) > 0.99)
     - Iso:  fraction of vectors where ||h|| > 2.646
  2. Per-layer gradient norms ||dL/dW||_F (accumulated over training batches)
  3. Pre-activation norms: mean ||h|| per hidden layer

Models: Base-1L, Base-2L, Base-3L, Iso-1L, Iso-2L, Iso-3L
Width: 32, Epochs: 24, Seed: 42
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AD')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS        = 24
LR            = 0.08
BATCH         = 128
WIDTH         = 32
SEED          = 42
SAT_THRESHOLD = 2.646   # atanh(0.99): tanh(2.646) ≈ 0.990


# =============================================================================
# Models
# =============================================================================

class BaseMLP(nn.Module):
    """Standard MLP with elementwise tanh. Depth configurable."""
    def __init__(self, input_dim, width, num_classes, depth):
        super().__init__()
        self.depth = depth
        dims = [input_dim] + [width] * depth
        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:]), 1):
            setattr(self, f'W{i}', nn.Linear(d_in, d_out))
        setattr(self, f'W{depth+1}', nn.Linear(width, num_classes))

    def forward(self, x, return_preacts=False):
        h = x
        preacts = []
        for i in range(1, self.depth + 1):
            h = getattr(self, f'W{i}')(h)
            if return_preacts:
                preacts.append(h.detach().cpu())
            h = torch.tanh(h)
        h = getattr(self, f'W{self.depth+1}')(h)
        return (h, preacts) if return_preacts else h


class IsoMLP(nn.Module):
    """Isotropic tanh MLP. Depth configurable."""
    def __init__(self, input_dim, width, num_classes, depth):
        super().__init__()
        self.depth = depth
        dims = [input_dim] + [width] * depth
        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:]), 1):
            setattr(self, f'W{i}', nn.Linear(d_in, d_out))
        setattr(self, f'W{depth+1}', nn.Linear(width, num_classes))

    def forward(self, x, return_preacts=False):
        h = x
        preacts = []
        for i in range(1, self.depth + 1):
            h = getattr(self, f'W{i}')(h)
            if return_preacts:
                preacts.append(h.detach().cpu())
            norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            h = torch.tanh(norm) * h / norm
        h = getattr(self, f'W{self.depth+1}')(h)
        return (h, preacts) if return_preacts else h


# =============================================================================
# Measurement helpers
# =============================================================================

def saturation_stats(preact, model_type):
    """
    Compute saturation fraction from pre-activation tensor.
    Base: fraction of individual elements |h_i| > SAT_THRESHOLD
    Iso:  fraction of vectors ||h|| > SAT_THRESHOLD
    """
    if model_type == 'Base':
        frac = (preact.abs() > SAT_THRESHOLD).float().mean().item()
    else:  # Iso
        norms = preact.norm(dim=-1)  # (N,)
        frac = (norms > SAT_THRESHOLD).float().mean().item()
    return frac


def preact_norm_stats(preact):
    """Mean and std of ||h|| over samples."""
    norms = preact.norm(dim=-1)
    return norms.mean().item(), norms.std().item()


def collect_grad_norms(model):
    """Return dict layer_name -> ||grad||_F for all weight matrices."""
    grad_norms = {}
    for i in range(1, model.depth + 2):
        W = getattr(model, f'W{i}')
        if W.weight.grad is not None:
            grad_norms[f'W{i}'] = W.weight.grad.norm().item()
        else:
            grad_norms[f'W{i}'] = 0.0
    return grad_norms


# =============================================================================
# Training loop with anatomy tracking
# =============================================================================

def train_with_anatomy(model, model_type, train_loader, test_loader,
                       probe_x, epochs, lr, device, tag):
    """
    Train model and record per-epoch anatomy stats.
    Returns dict of lists: acc, sat_frac (per layer), grad_norm (per layer),
                            preact_norm_mean (per layer).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = defaultdict(list)   # key -> list over epochs

    for epoch in range(1, epochs + 1):
        # ---- Train one epoch, accumulate gradient norms ----
        model.train()
        epoch_grad_norms = defaultdict(list)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            # Collect gradient norms before step
            for i in range(1, model.depth + 2):
                W = getattr(model, f'W{i}')
                if W.weight.grad is not None:
                    epoch_grad_norms[f'W{i}'].append(W.weight.grad.norm().item())
            optimizer.step()

        # ---- Evaluate accuracy ----
        acc = evaluate(model, test_loader, device)
        history['acc'].append(acc)

        # ---- Anatomy on probe batch ----
        model.eval()
        with torch.no_grad():
            _, preacts = model(probe_x, return_preacts=True)

        for layer_idx, pa in enumerate(preacts, 1):
            sat = saturation_stats(pa, model_type)
            pa_mean, pa_std = preact_norm_stats(pa)
            history[f'sat_L{layer_idx}'].append(sat)
            history[f'preact_norm_mean_L{layer_idx}'].append(pa_mean)
            history[f'preact_norm_std_L{layer_idx}'].append(pa_std)

        # Mean gradient norm per layer over training batches
        for key, vals in epoch_grad_norms.items():
            history[f'grad_norm_{key}'].append(np.mean(vals))

        if epoch in (1, 6, 12, 18, 24):
            sat_str = '  '.join(
                f'L{li}_sat={history[f"sat_L{li}"][-1]:.3f}'
                for li in range(1, model.depth + 1)
            )
            grad_str = '  '.join(
                f'{k}={np.mean(v):.4f}'
                for k, v in epoch_grad_norms.items()
            )
            print(f'  [{tag}] Ep {epoch:2d}  acc={acc:.4f}  {sat_str}  grads: {grad_str}')

    return history


# =============================================================================
# Main
# =============================================================================

def main():
    print(f'Device: {DEVICE}')
    print(f'Hypothesis: Base depth failure = per-neuron tanh saturation')
    print(f'Prediction: Base saturation fraction grows with depth; Iso stays low')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    torch.manual_seed(SEED)
    probe_x, _ = next(iter(test_loader))
    probe_x = probe_x[:1024].to(DEVICE)

    configs = [
        ('Base-1L', BaseMLP, 'Base', 1),
        ('Base-2L', BaseMLP, 'Base', 2),
        ('Base-3L', BaseMLP, 'Base', 3),
        ('Iso-1L',  IsoMLP,  'Iso',  1),
        ('Iso-2L',  IsoMLP,  'Iso',  2),
        ('Iso-3L',  IsoMLP,  'Iso',  3),
    ]

    all_histories = {}

    for i, (tag, cls, mtype, depth) in enumerate(configs, 1):
        print(f'\n[{i}/{len(configs)}] {tag}')
        torch.manual_seed(SEED)
        model = cls(input_dim, WIDTH, num_classes, depth).to(DEVICE)
        hist = train_with_anatomy(
            model, mtype, train_loader, test_loader,
            probe_x, EPOCHS, LR, DEVICE, tag
        )
        all_histories[tag] = hist
        print(f'  Final acc: {hist["acc"][-1]:.4f}')

    epochs_range = list(range(1, EPOCHS + 1))

    # =========================================================================
    # Summary table
    # =========================================================================
    print(f'\n{"="*70}')
    print('SUMMARY: Final-epoch saturation fractions and gradient ratios')
    print(f'{"="*70}')
    print(f'{"Model":>12}  {"Acc":>6}  {"Sat-L1":>8}  {"Sat-L2":>8}  {"Sat-L3":>8}  {"Grad-W1":>9}  {"Grad-W_last":>11}')

    for tag, _, mtype, depth in configs:
        h = all_histories[tag]
        acc  = h['acc'][-1]
        s1   = h.get('sat_L1',  [float('nan')])[-1]
        s2   = h.get('sat_L2',  [float('nan')])[-1]
        s3   = h.get('sat_L3',  [float('nan')])[-1]
        gw1  = h.get('grad_norm_W1',  [float('nan')])[-1]
        gwl  = h.get(f'grad_norm_W{depth+1}', [float('nan')])[-1]
        print(f'{tag:>12}  {acc:.4f}  {s1:8.4f}  {s2:8.4f}  {s3:8.4f}  {gw1:9.6f}  {gwl:11.6f}')

    # Gradient ratio: W1 / W_last (should be << 1 if vanishing)
    print(f'\nGradient ratio W1/W_last (< 1 = early layers see weaker gradients):')
    for tag, _, mtype, depth in configs:
        h = all_histories[tag]
        g1 = np.array(h.get('grad_norm_W1', [1e-10]))
        gl = np.array(h.get(f'grad_norm_W{depth+1}', [1e-10]))
        ratio = (g1 / (gl + 1e-10)).mean()
        print(f'  {tag}: W1/W_last = {ratio:.4f}')

    # Hypothesis verdict
    print(f'\nHypothesis test (saturation fraction at epoch 24):')
    for tag, _, mtype, depth in configs:
        h = all_histories[tag]
        sats = [h.get(f'sat_L{li}', [0])[-1] for li in range(1, depth+1)]
        mean_sat = np.mean(sats)
        print(f'  {tag}: mean_sat={mean_sat:.4f}  layers={[f"{s:.3f}" for s in sats]}')

    # =========================================================================
    # Plots
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    depths_plot = [1, 2, 3]
    colors_base = ['#d62728', '#ff7f0e', '#8c564b']
    colors_iso  = ['#1f77b4', '#2ca02c', '#9467bd']

    # Row 0: Saturation fraction per depth (L1 only for comparability)
    ax = axes[0, 0]
    for depth, cb, ci in zip(depths_plot, colors_base, colors_iso):
        tag_b = f'Base-{depth}L'
        tag_i = f'Iso-{depth}L'
        s_b = all_histories[tag_b].get('sat_L1', [0]*EPOCHS)
        s_i = all_histories[tag_i].get('sat_L1', [0]*EPOCHS)
        ax.plot(epochs_range, s_b, color=cb, ls='-',  label=f'Base-{depth}L')
        ax.plot(epochs_range, s_i, color=ci, ls='--', label=f'Iso-{depth}L')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Saturation fraction (L1)')
    ax.set_title('Layer-1 Saturation\n(fraction neurons/vectors with |h|>2.65)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Row 0, col 1: Saturation at last hidden layer (depth=3 only, all layers)
    ax = axes[0, 1]
    for li, ls, color in [(1, '-', '#d62728'), (2, '--', '#ff7f0e'), (3, ':', '#8c564b')]:
        key = f'sat_L{li}'
        if key in all_histories['Base-3L']:
            ax.plot(epochs_range, all_histories['Base-3L'][key],
                    color=color, ls=ls, label=f'Base-3L L{li}')
    for li, ls, color in [(1, '-', '#1f77b4'), (2, '--', '#2ca02c'), (3, ':', '#9467bd')]:
        key = f'sat_L{li}'
        if key in all_histories['Iso-3L']:
            ax.plot(epochs_range, all_histories['Iso-3L'][key],
                    color=color, ls=ls, label=f'Iso-3L L{li}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Saturation fraction')
    ax.set_title('Saturation per Layer (3L models)\nBase vs Iso')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Row 0, col 2: Accuracy
    ax = axes[0, 2]
    for depth, cb, ci in zip(depths_plot, colors_base, colors_iso):
        ax.plot(epochs_range, all_histories[f'Base-{depth}L']['acc'],
                color=cb, ls='-',  label=f'Base-{depth}L')
        ax.plot(epochs_range, all_histories[f'Iso-{depth}L']['acc'],
                color=ci, ls='--', label=f'Iso-{depth}L')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test accuracy')
    ax.set_title('Accuracy vs Epoch')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Row 1: Gradient norms W1 and W_last, and their ratio
    ax = axes[1, 0]
    for depth, cb, ci in zip(depths_plot, colors_base, colors_iso):
        tag_b, tag_i = f'Base-{depth}L', f'Iso-{depth}L'
        ax.plot(epochs_range, all_histories[tag_b].get('grad_norm_W1', [0]*EPOCHS),
                color=cb, ls='-',  label=f'Base-{depth}L W1')
        ax.plot(epochs_range, all_histories[tag_i].get('grad_norm_W1', [0]*EPOCHS),
                color=ci, ls='--', label=f'Iso-{depth}L W1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('||dL/dW1||_F')
    ax.set_title('First-layer gradient norm\n(low = early layers blocked)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = axes[1, 1]
    for depth, cb, ci in zip(depths_plot, colors_base, colors_iso):
        tag_b, tag_i = f'Base-{depth}L', f'Iso-{depth}L'
        h_b = all_histories[tag_b]
        h_i = all_histories[tag_i]
        g1_b = np.array(h_b.get('grad_norm_W1', [1e-10]*EPOCHS))
        gl_b = np.array(h_b.get(f'grad_norm_W{depth+1}', [1e-10]*EPOCHS))
        g1_i = np.array(h_i.get('grad_norm_W1', [1e-10]*EPOCHS))
        gl_i = np.array(h_i.get(f'grad_norm_W{depth+1}', [1e-10]*EPOCHS))
        ratio_b = g1_b / (gl_b + 1e-10)
        ratio_i = g1_i / (gl_i + 1e-10)
        ax.plot(epochs_range, ratio_b, color=cb, ls='-',  label=f'Base-{depth}L')
        ax.plot(epochs_range, ratio_i, color=ci, ls='--', label=f'Iso-{depth}L')
    ax.axhline(1.0, color='black', ls=':', alpha=0.5, label='ratio=1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('||dL/dW1|| / ||dL/dW_last||')
    ax.set_title('Gradient ratio W1/W_last\n(<1 = gradient weaker at early layers)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    for depth, cb, ci in zip(depths_plot, colors_base, colors_iso):
        tag_b, tag_i = f'Base-{depth}L', f'Iso-{depth}L'
        key = f'preact_norm_mean_L{min(depth, 1)}'
        ax.plot(epochs_range, all_histories[tag_b].get(key, [0]*EPOCHS),
                color=cb, ls='-',  label=f'Base-{depth}L')
        ax.plot(epochs_range, all_histories[tag_i].get(key, [0]*EPOCHS),
                color=ci, ls='--', label=f'Iso-{depth}L')
    ax.axhline(SAT_THRESHOLD, color='red', ls=':', alpha=0.7, label=f'sat threshold={SAT_THRESHOLD:.2f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean ||h|| (L1)')
    ax.set_title('Pre-activation norm (L1)\n(above threshold → saturation)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Test AD: Saturation & Gradient Anatomy (Base vs Iso, 1–3L)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'anatomy.png'), dpi=150)
    print('\nPlot saved to results/test_AD/anatomy.png')

    # =========================================================================
    # Save results.md
    # =========================================================================
    rows = []
    for tag, _, mtype, depth in configs:
        h = all_histories[tag]
        acc  = h['acc'][-1]
        sats = [h.get(f'sat_L{li}', [float('nan')])[-1] for li in range(1, depth+1)]
        g1   = h.get('grad_norm_W1', [float('nan')])[-1]
        gl   = h.get(f'grad_norm_W{depth+1}', [float('nan')])[-1]
        ratio = g1 / (gl + 1e-10)
        sat_str = ' / '.join(f'{s:.4f}' for s in sats)
        rows.append(f'| {tag} | {acc:.4f} | {sat_str} | {g1:.6f} | {gl:.6f} | {ratio:.4f} |')

    # Compute verdict
    base3_sat = np.mean([all_histories['Base-3L'].get(f'sat_L{li}', [0])[-1] for li in range(1,4)])
    iso3_sat  = np.mean([all_histories['Iso-3L'].get(f'sat_L{li}', [0])[-1] for li in range(1,4)])
    base1_sat = all_histories['Base-1L'].get('sat_L1', [0])[-1]
    iso1_sat  = all_histories['Iso-1L'].get('sat_L1', [0])[-1]

    if base3_sat > 2 * base1_sat and base3_sat > 2 * iso3_sat:
        verdict = (f'SUPPORTS saturation hypothesis: Base-3L mean sat={base3_sat:.4f} '
                   f'vs Base-1L={base1_sat:.4f} and Iso-3L={iso3_sat:.4f}')
    elif base3_sat > iso3_sat * 1.5:
        verdict = (f'PARTIAL SUPPORT: Base-3L sat={base3_sat:.4f} > Iso-3L sat={iso3_sat:.4f} '
                   f'but Base-1L={base1_sat:.4f} (depth effect unclear)')
    else:
        verdict = (f'DOES NOT SUPPORT saturation hypothesis: Base-3L sat={base3_sat:.4f} '
                   f'not clearly higher than Iso-3L={iso3_sat:.4f}')

    md = f"""# Test AD -- Saturation & Gradient Anatomy

## Setup
- Models: Base/Iso at 1L, 2L, 3L depth
- Width: {WIDTH}, Epochs: {EPOCHS}, Seed: {SEED}, lr={LR}
- Saturation threshold: {SAT_THRESHOLD:.3f} (tanh saturation at 99%)
- Probe: 1024 fixed samples; gradients averaged over all training batches
- Device: {DEVICE}

## Hypothesis
Base depth failure = per-neuron tanh saturation compounding with depth.
Prediction: Base saturation fraction increases with depth; Iso stays low.
Also: W1 gradient weaker than W_last in deep Base (gradients blocked by saturated neurons).

## Results

| Model | Acc | Sat fraction (L1/L2/L3) | grad W1 | grad W_last | W1/W_last ratio |
|---|---|---|---|---|---|
{chr(10).join(rows)}

## Verdict
{verdict}

Base-3L mean saturation: {base3_sat:.4f}
Iso-3L  mean saturation: {iso3_sat:.4f}
Base-1L L1  saturation:  {base1_sat:.4f}
Iso-1L  L1  saturation:  {iso1_sat:.4f}

## Gradient ratio analysis
(W1/W_last < 1 means first layer receives weaker gradient signal)
"""
    for tag, _, mtype, depth in configs:
        h = all_histories[tag]
        g1 = np.array(h.get('grad_norm_W1', [1e-10]))
        gl = np.array(h.get(f'grad_norm_W{depth+1}', [1e-10]))
        ratio = (g1 / (gl + 1e-10)).mean()
        md += f'- {tag}: W1/W_last = {ratio:.4f}\n'

    md += '\n![Anatomy plots](anatomy.png)\n'

    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    print('Results saved to results/test_AD/results.md')


if __name__ == '__main__':
    main()
