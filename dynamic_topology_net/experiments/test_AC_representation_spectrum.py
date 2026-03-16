"""
Test AC -- Representation Spectrum Mechanism
=============================================
Tests D, E, L, M, Q, X establish that isotropic networks outperform standard
tanh at depth, and that this gap cannot be closed by rank regularisation.
Test R tried to find the mechanism and came back inconclusive using a single
effective-rank summary statistic.

This test tracks the FULL singular value spectrum of hidden representations
at each epoch, for both Iso and Baseline networks at depths 1L, 2L, 3L.

Specifically, at each epoch we forward a fixed probe set of 1024 samples
and compute SVD of the representation matrix H (1024 x width) at each layer.
We track:

  1. Full spectrum (all singular values) -- visualised as heatmap
  2. Participation ratio: PR = (sum sv_i)^2 / sum(sv_i^2)
     Equals width when all SVs are equal; equals 1 when one SV dominates.
     More robust than effective rank (doesn't require threshold).
  3. Nuclear norm / operator norm ratio: same as PR but using norms directly
  4. Fraction of variance in top-1 and top-3 singular values
  5. Mean representation norm: E[||h||] -- does it blow up?

Hypothesis: Baseline representations collapse to low-dimensional subspace
(PR drops towards 1) while Iso representations maintain diversity (PR stays
near width). If confirmed, this IS the mechanistic explanation for depth failure.

Test R had inconclusive results possibly because:
  - It measured effective_rank (which needs a threshold) -- may have missed collapse
  - It only measured at final epoch, not trajectory
  - The hook for Iso models didn't work (NaN)

This test avoids hooks entirely -- uses explicit forward passes through
each layer with the representations captured directly.

Width=32 (richer spectrum), Seeds=[42], 24 epochs, 1L/2L/3L Iso and Base.
Device: CPU/GPU auto.
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
from matplotlib.colors import LogNorm

from dynamic_topology_net.core import load_cifar10

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AC')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS     = 24
LR         = 0.08
BATCH      = 128
WIDTH      = 32
SEED       = 42
PROBE_SIZE = 1024  # fixed probe samples for representation tracking


# =============================================================================
# Models with explicit representation extraction
# =============================================================================

class IsoMLP1L(nn.Module):
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(width, input_dim))
        self.b1 = nn.Parameter(torch.zeros(width))
        self.W2 = nn.Parameter(torch.empty(num_classes, width))
        self.b2 = nn.Parameter(torch.zeros(num_classes))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        self.input_dim = input_dim; self.width = width; self.depth = 1

    def forward(self, x):
        h = F.linear(x, self.W1, self.b1)
        n = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a = torch.tanh(n) * h / n
        return F.linear(a, self.W2, self.b2)

    def get_representations(self, x):
        with torch.no_grad():
            h = F.linear(x, self.W1, self.b1)
            n = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            a = torch.tanh(n) * h / n
        return [a]   # list of layer representations


class BaseMLP1L(nn.Module):
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        self.W1 = nn.Linear(input_dim, width)
        self.W2 = nn.Linear(width, num_classes)
        self.input_dim = input_dim; self.width = width; self.depth = 1

    def forward(self, x):
        return self.W2(torch.tanh(self.W1(x)))

    def get_representations(self, x):
        with torch.no_grad():
            a = torch.tanh(self.W1(x))
        return [a]


class IsoMLP2L(nn.Module):
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(width, input_dim))
        self.b1 = nn.Parameter(torch.zeros(width))
        self.W2 = nn.Parameter(torch.empty(width, width))
        self.b2 = nn.Parameter(torch.zeros(width))
        self.W3 = nn.Parameter(torch.empty(num_classes, width))
        self.b3 = nn.Parameter(torch.zeros(num_classes))
        for W in [self.W1, self.W2, self.W3]:
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        self.input_dim = input_dim; self.width = width; self.depth = 2

    def forward(self, x):
        h1 = F.linear(x, self.W1, self.b1)
        n1 = h1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a1 = torch.tanh(n1) * h1 / n1
        h2 = F.linear(a1, self.W2, self.b2)
        n2 = h2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a2 = torch.tanh(n2) * h2 / n2
        return F.linear(a2, self.W3, self.b3)

    def get_representations(self, x):
        with torch.no_grad():
            h1 = F.linear(x, self.W1, self.b1)
            n1 = h1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            a1 = torch.tanh(n1) * h1 / n1
            h2 = F.linear(a1, self.W2, self.b2)
            n2 = h2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            a2 = torch.tanh(n2) * h2 / n2
        return [a1, a2]


class BaseMLP2L(nn.Module):
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        self.W1 = nn.Linear(input_dim, width)
        self.W2 = nn.Linear(width, width)
        self.W3 = nn.Linear(width, num_classes)
        self.input_dim = input_dim; self.width = width; self.depth = 2

    def forward(self, x):
        return self.W3(torch.tanh(self.W2(torch.tanh(self.W1(x)))))

    def get_representations(self, x):
        with torch.no_grad():
            a1 = torch.tanh(self.W1(x))
            a2 = torch.tanh(self.W2(a1))
        return [a1, a2]


class IsoMLP3L(nn.Module):
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        dims = [input_dim, width, width, width, num_classes]
        Ws, bs = [], []
        for i in range(4):
            W = nn.Parameter(torch.empty(dims[i+1], dims[i]))
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))
            Ws.append(W); bs.append(nn.Parameter(torch.zeros(dims[i+1])))
        self.W1, self.b1 = Ws[0], bs[0]
        self.W2, self.b2 = Ws[1], bs[1]
        self.W3, self.b3 = Ws[2], bs[2]
        self.W4, self.b4 = Ws[3], bs[3]
        self.input_dim = input_dim; self.width = width; self.depth = 3

    def _iso(self, h):
        n = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.tanh(n) * h / n

    def forward(self, x):
        a1 = self._iso(F.linear(x,  self.W1, self.b1))
        a2 = self._iso(F.linear(a1, self.W2, self.b2))
        a3 = self._iso(F.linear(a2, self.W3, self.b3))
        return F.linear(a3, self.W4, self.b4)

    def get_representations(self, x):
        with torch.no_grad():
            a1 = self._iso(F.linear(x,  self.W1, self.b1))
            a2 = self._iso(F.linear(a1, self.W2, self.b2))
            a3 = self._iso(F.linear(a2, self.W3, self.b3))
        return [a1, a2, a3]


class BaseMLP3L(nn.Module):
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        self.W1 = nn.Linear(input_dim, width)
        self.W2 = nn.Linear(width, width)
        self.W3 = nn.Linear(width, width)
        self.W4 = nn.Linear(width, num_classes)
        self.input_dim = input_dim; self.width = width; self.depth = 3

    def forward(self, x):
        return self.W4(torch.tanh(self.W3(torch.tanh(self.W2(torch.tanh(self.W1(x)))))))

    def get_representations(self, x):
        with torch.no_grad():
            a1 = torch.tanh(self.W1(x))
            a2 = torch.tanh(self.W2(a1))
            a3 = torch.tanh(self.W3(a2))
        return [a1, a2, a3]


# =============================================================================
# Spectrum analysis
# =============================================================================

def spectrum_stats(H):
    """
    Compute statistics of representation matrix H (N x width).
    Returns dict of metrics.
    """
    # Subtract mean (centre representations)
    H_c = H - H.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(H_c)  # descending

    sv_np = sv.cpu().numpy()
    sv_np = sv_np[sv_np > 1e-10]   # drop near-zero

    total   = sv_np.sum()
    total_sq = (sv_np ** 2).sum()

    # Participation ratio: (sum sv)^2 / sum(sv^2)
    # Ranges from 1 (one SV dominates) to len(sv) (all equal)
    pr = (total ** 2) / (total_sq + 1e-10)

    # Top fraction
    top1_frac = sv_np[0] / (total + 1e-10) if len(sv_np) > 0 else 1.0
    top3_frac = sv_np[:3].sum() / (total + 1e-10) if len(sv_np) >= 3 else 1.0

    # Nuclear norm / operator norm
    nuc_op = total / (sv_np[0] + 1e-10) if len(sv_np) > 0 else 1.0

    # Mean representation norm
    repr_norm = H.norm(dim=-1).mean().item()

    return {
        'sv':        sv_np,
        'pr':        float(pr),
        'top1_frac': float(top1_frac),
        'top3_frac': float(top3_frac),
        'nuc_op':    float(nuc_op),
        'repr_norm': repr_norm,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print(f'Device: {DEVICE}, Width={WIDTH}, Epochs={EPOCHS}')
    print(f'Probe size: {PROBE_SIZE} samples per epoch')
    print('Loading CIFAR-10...')

    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    # Fixed probe set (same samples every epoch)
    torch.manual_seed(0)
    probe_batches = []
    count = 0
    for x, _ in test_loader:
        probe_batches.append(x)
        count += x.shape[0]
        if count >= PROBE_SIZE:
            break
    probe_x = torch.cat(probe_batches, dim=0)[:PROBE_SIZE].to(DEVICE)
    print(f'Probe set: {probe_x.shape[0]} samples')

    configs = [
        ('Iso-1L',  lambda: IsoMLP1L( input_dim, WIDTH, num_classes)),
        ('Base-1L', lambda: BaseMLP1L(input_dim, WIDTH, num_classes)),
        ('Iso-2L',  lambda: IsoMLP2L( input_dim, WIDTH, num_classes)),
        ('Base-2L', lambda: BaseMLP2L(input_dim, WIDTH, num_classes)),
        ('Iso-3L',  lambda: IsoMLP3L( input_dim, WIDTH, num_classes)),
        ('Base-3L', lambda: BaseMLP3L(input_dim, WIDTH, num_classes)),
    ]

    all_results = {}
    total = len(configs)

    for ci, (tag, factory) in enumerate(configs):
        print(f'\n[{ci+1}/{total}] {tag}')
        torch.manual_seed(SEED)
        model = factory().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        epoch_stats = []  # list of (epoch, acc, [layer_stats])

        for epoch in range(1, EPOCHS + 1):
            # Train
            model.train()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()

            # Evaluate accuracy
            model.eval()
            correct = total_n = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total_n += y.size(0)
            acc = correct / total_n

            # Get representations and compute spectrum
            model.eval()
            reps = model.get_representations(probe_x)
            layer_stats = [spectrum_stats(rep) for rep in reps]
            epoch_stats.append((epoch, acc, layer_stats))

            if epoch in (1, 6, 12, 18, 24):
                pr_str = '  '.join(
                    f'L{i+1}_PR={s["pr"]:.2f}(top1={s["top1_frac"]:.2f})'
                    for i, s in enumerate(layer_stats)
                )
                print(f'  Epoch {epoch:2d}  acc={acc:.4f}  {pr_str}')

        all_results[tag] = epoch_stats

    # =========================================================================
    # Print Summary
    # =========================================================================
    print(f'\n{"="*70}')
    print('SUMMARY: Final epoch representation statistics (last hidden layer)')
    print(f'{"="*70}')
    print(f'{"Model":>12}  {"Acc":>6}  {"PR":>6}  {"top1":>6}  {"top3":>6}  '
          f'{"nuc/op":>8}  {"repr||":>8}')
    for tag in [t for t, _ in configs]:
        if tag not in all_results:
            continue
        epoch, acc, layer_stats = all_results[tag][-1]
        s = layer_stats[-1]  # last hidden layer
        print(f'{tag:>12}  {acc:.4f}  {s["pr"]:6.2f}  {s["top1_frac"]:6.3f}  '
              f'{s["top3_frac"]:6.3f}  {s["nuc_op"]:8.2f}  {s["repr_norm"]:8.4f}')

    # =========================================================================
    # Plots
    # =========================================================================

    # 1. Participation Ratio over training (last layer)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors_iso  = ['steelblue', 'dodgerblue', 'navy']
    colors_base = ['crimson', 'orangered', 'darkred']
    epochs_list = [e for e, _, _ in all_results.get('Iso-1L', [(1,0,[])])]

    for di, depth in enumerate([1, 2, 3]):
        ax = axes[di]
        for tag, colors in [
            (f'Iso-{depth}L', colors_iso),
            (f'Base-{depth}L', colors_base),
        ]:
            if tag not in all_results:
                continue
            n_layers = depth
            for li in range(n_layers):
                pr_values = [s[li]['pr'] for (_, _, s) in all_results[tag]]
                color = colors[li % len(colors)]
                ls = '-' if 'Iso' in tag else '--'
                ax.plot(epochs_list, pr_values,
                        label=f'{tag} L{li+1}', color=color, ls=ls)
        ax.axhline(y=1, color='gray', ls=':', lw=0.8, label='PR=1 (collapsed)')
        ax.axhline(y=WIDTH, color='gray', ls='-.', lw=0.8, label=f'PR={WIDTH} (uniform)')
        ax.set_title(f'{depth}-layer models')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Participation Ratio')
        ax.set_ylim(0, WIDTH + 2)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Test AC: Representation Participation Ratio\n'
                 'PR=1: fully collapsed. PR=width: perfectly diverse.',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'participation_ratio.png'), dpi=150)

    # 2. SV spectrum heatmap for Iso-3L vs Base-3L (last layer, all epochs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, tag in zip(axes, ['Iso-3L', 'Base-3L']):
        if tag not in all_results:
            continue
        n_layers = all_results[tag][0][2].__len__()
        # Stack spectra: (n_epochs, width)
        spectra = []
        for (_, _, layer_stats) in all_results[tag]:
            sv = layer_stats[-1]['sv']
            sv_padded = np.zeros(WIDTH)
            sv_padded[:len(sv)] = sv
            spectra.append(sv_padded)
        S = np.array(spectra).T  # (width, n_epochs)
        # Normalise each epoch to show shape
        S_norm = S / (S.sum(axis=0, keepdims=True) + 1e-10)
        im = ax.imshow(S_norm, aspect='auto', origin='lower',
                       extent=[1, EPOCHS, 0.5, WIDTH + 0.5],
                       cmap='hot', vmin=0, vmax=S_norm.max())
        ax.set_title(f'{tag}: SV spectrum (normalised)\nLast hidden layer')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('SV rank (1=largest)')
        plt.colorbar(im, ax=ax, label='Fraction of total')

    plt.suptitle('Test AC: Singular Value Spectrum Evolution (Iso-3L vs Base-3L)', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sv_spectrum_heatmap.png'), dpi=150)

    # 3. Top-1 fraction over training (collapse indicator)
    fig, ax = plt.subplots(figsize=(9, 5))
    style_map = {
        'Iso-1L': ('steelblue', '-', 'o'),
        'Base-1L': ('crimson', '--', 's'),
        'Iso-2L': ('dodgerblue', '-', '^'),
        'Base-2L': ('orangered', '--', 'v'),
        'Iso-3L': ('navy', '-', 'D'),
        'Base-3L': ('darkred', '--', 'x'),
    }
    for tag, (color, ls, mk) in style_map.items():
        if tag not in all_results:
            continue
        top1 = [s[-1]['top1_frac'] for (_, _, s) in all_results[tag]]
        ax.plot(epochs_list, top1, color=color, ls=ls, marker=mk,
                markevery=4, label=tag, lw=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Top-1 SV fraction (higher = more collapsed)')
    ax.set_title('Representational Collapse Indicator\n'
                 'Top-1 fraction → 1.0 means one direction dominates')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'collapse_indicator.png'), dpi=150)

    # =========================================================================
    # Save results.md
    # =========================================================================
    rows = '\n'.join(
        '| {tag} | {acc:.4f} | {pr:.2f} | {top1:.3f} | {top3:.3f} | {nuc_op:.2f} | {norm:.4f} |'.format(
            tag=tag,
            acc=all_results[tag][-1][1],
            pr=all_results[tag][-1][2][-1]['pr'],
            top1=all_results[tag][-1][2][-1]['top1_frac'],
            top3=all_results[tag][-1][2][-1]['top3_frac'],
            nuc_op=all_results[tag][-1][2][-1]['nuc_op'],
            norm=all_results[tag][-1][2][-1]['repr_norm'],
        )
        for tag, _ in configs if tag in all_results
    )

    # Compare PR trajectory: does Base collapse faster than Iso?
    if 'Iso-3L' in all_results and 'Base-3L' in all_results:
        iso_pr_init  = all_results['Iso-3L'][0][2][-1]['pr']
        iso_pr_final = all_results['Iso-3L'][-1][2][-1]['pr']
        base_pr_init  = all_results['Base-3L'][0][2][-1]['pr']
        base_pr_final = all_results['Base-3L'][-1][2][-1]['pr']
        pr_verdict = (
            f'Iso-3L PR: {iso_pr_init:.2f} → {iso_pr_final:.2f} '
            f'(change={iso_pr_final-iso_pr_init:+.2f})\n'
            f'Base-3L PR: {base_pr_init:.2f} → {base_pr_final:.2f} '
            f'(change={base_pr_final-base_pr_init:+.2f})\n'
        )
        if base_pr_final < iso_pr_final - 2:
            mechanism = 'Representational collapse IS the mechanism: Base collapses to lower-rank subspace.'
        elif abs(base_pr_final - iso_pr_final) < 1:
            mechanism = 'Representational collapse is NOT the mechanism: PR similar for Iso and Base.'
        else:
            mechanism = f'Partial collapse: Base PR={base_pr_final:.1f} vs Iso PR={iso_pr_final:.1f}.'
    else:
        pr_verdict = 'Insufficient data.'
        mechanism  = 'N/A'

    md = f"""# Test AC -- Representation Spectrum Mechanism

## Setup
- Models: Iso/Base at 1L, 2L, 3L depth
- Width: {WIDTH}, Epochs: {EPOCHS}, Seed: {SEED}
- Probe: {PROBE_SIZE} fixed samples, evaluated at every epoch
- Metric: Participation Ratio of hidden representation SVD
  PR = (sum sv)^2 / sum(sv^2), ranges 1 (collapsed) to width (uniform)
- Device: {DEVICE}

## Question
Why does standard tanh fail at depth while isotropic succeeds?
Test R was inconclusive. This test tracks the full SV spectrum.

## Final-Epoch Statistics (last hidden layer)

| Model | Acc | PR | top-1 frac | top-3 frac | nuc/op | repr norm |
|---|---|---|---|---|---|---|
{rows}

## PR Trajectory Analysis
{pr_verdict}

## Verdict
{mechanism}

## Connection to Prior Tests
- Test R: measured effective rank (hook-based, NaN for Iso), inconclusive
- Test X: rank regularisation had 0% effect on Base accuracy
  --> if collapse is the mechanism, Test X failing means collapse happens
      in weight space, not representation space (or needs different intervention)

![Participation ratio over training](participation_ratio.png)
![SV spectrum heatmap](sv_spectrum_heatmap.png)
![Collapse indicator](collapse_indicator.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    print('\nResults saved to results/test_AC/results.md')
    print('Plots saved to results/test_AC/')


if __name__ == '__main__':
    main()
