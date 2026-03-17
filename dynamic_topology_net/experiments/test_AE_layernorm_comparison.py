"""
Test AE -- LayerNorm vs Isotropic Activation
=============================================
Test AD resolved the mechanism of Base depth failure: elementwise tanh saturates
99.94% of neurons, collapsing the Jacobian to ~0 and forcing all learning into
the output layer (gradient 7-8x larger than Iso).

Isotropic tanh fixes this via the tangential Jacobian component -- the direction
of h is always preserved even when the norm is large, keeping all layers active.

KEY QUESTION: Is isotropic activation actually necessary, or is Jacobian
preservation the real principle -- and LayerNorm already achieves it?

LayerNorm normalises h to zero mean and unit variance per sample before the
activation. This should prevent elementwise saturation (since tanh(z) is linear
near z=0 and saturates at |z|>>1 -- LN keeps z small). If this is the whole
story, LN+tanh should match Iso at depth.

Three possible outcomes:
  (1) LN+tanh ≈ Iso at all depths
      → Jacobian preservation is the principle; isotropic form is just one way
        to achieve it. Paper's specific mechanism not uniquely necessary.

  (2) LN+tanh > Base but < Iso at depth
      → LN helps but isotropic Jacobian structure provides additional benefit
        beyond preventing saturation alone.

  (3) LN+tanh ≈ Base at depth
      → Something specifically special about the isotropic Jacobian that
        LayerNorm cannot replicate. Paper's contribution is stronger.

Models tested:
  - Base:       elementwise tanh, no norm
  - Iso:        isotropic tanh, no norm
  - LN+tanh:    LayerNorm (pre-activation) + elementwise tanh
  - LN+Iso:     LayerNorm (pre-activation) + isotropic tanh
  - RMS+tanh:   RMSNorm (no mean subtraction) + elementwise tanh

All at 1L, 2L, 3L depth. Width=32, Epochs=24.
Also tracks saturation fraction and gradient anatomy (from Test AD) to confirm
the mechanism is what we think.
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AE')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS        = 24
LR            = 0.08
BATCH         = 128
WIDTH         = 32
SEED          = 42
SAT_THRESHOLD = 2.646   # tanh(2.646) ≈ 0.99


# =============================================================================
# RMSNorm (no mean subtraction -- purely scales)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * x / rms


# =============================================================================
# Model factory
# =============================================================================

def make_layer(d_in, d_out, activation, norm_type, width):
    """Returns a nn.Sequential block: Linear -> [Norm] -> Activation."""
    layers = [nn.Linear(d_in, d_out)]
    if norm_type == 'LN':
        layers.append(nn.LayerNorm(d_out))
    elif norm_type == 'RMS':
        layers.append(RMSNorm(d_out))
    layers.append(activation)
    return nn.Sequential(*layers)


class IsoAct(nn.Module):
    """Isotropic tanh activation."""
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.tanh(norm) * x / norm


class FlexMLP(nn.Module):
    """
    Flexible MLP with configurable activation and normalisation.
    activation: 'tanh' | 'iso'
    norm:       None | 'LN' | 'RMS'
    """
    def __init__(self, input_dim, width, num_classes, depth, activation, norm):
        super().__init__()
        self.depth = depth
        self.activation_type = activation
        self.norm_type = norm

        act_module = IsoAct() if activation == 'iso' else nn.Tanh()

        dims = [input_dim] + [width] * depth
        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:]), 1):
            linear = nn.Linear(d_in, d_out)
            setattr(self, f'linear{i}', linear)
            if norm == 'LN':
                setattr(self, f'norm{i}', nn.LayerNorm(d_out))
            elif norm == 'RMS':
                setattr(self, f'norm{i}', RMSNorm(d_out))

        self.output = nn.Linear(width, num_classes)

        if activation == 'iso':
            self.act = IsoAct()
        else:
            self.act = nn.Tanh()

    def forward(self, x, return_preacts=False):
        h = x
        preacts = []
        for i in range(1, self.depth + 1):
            h = getattr(self, f'linear{i}')(h)
            if return_preacts:
                preacts.append(h.detach().cpu())
            if self.norm_type in ('LN', 'RMS'):
                h = getattr(self, f'norm{i}')(h)
            h = self.act(h)
        h = self.output(h)
        return (h, preacts) if return_preacts else h


# =============================================================================
# Measurement helpers (same as Test AD)
# =============================================================================

def saturation_frac(preact, activation):
    if activation == 'tanh':
        return (preact.abs() > SAT_THRESHOLD).float().mean().item()
    else:  # iso
        return (preact.norm(dim=-1) > SAT_THRESHOLD).float().mean().item()


def train_with_anatomy(model, activation, train_loader, test_loader,
                       probe_x, epochs, lr, device, tag):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = defaultdict(list)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_grad = defaultdict(list)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            for i in range(1, model.depth + 2):
                name = f'linear{i}' if i <= model.depth else 'output'
                W = getattr(model, name) if i <= model.depth else model.output
                if W.weight.grad is not None:
                    epoch_grad[f'W{i}'].append(W.weight.grad.norm().item())
            optimizer.step()

        acc = evaluate(model, test_loader, device)
        history['acc'].append(acc)

        model.eval()
        with torch.no_grad():
            _, preacts = model(probe_x, return_preacts=True)

        for li, pa in enumerate(preacts, 1):
            history[f'sat_L{li}'].append(saturation_frac(pa, activation))
            history[f'preact_norm_L{li}'].append(pa.norm(dim=-1).mean().item())

        for key, vals in epoch_grad.items():
            history[f'grad_{key}'].append(np.mean(vals))

        if epoch in (1, 6, 12, 18, 24):
            sats = '  '.join(
                f'L{li}={history[f"sat_L{li}"][-1]:.3f}'
                for li in range(1, model.depth + 1)
            )
            g1 = history.get('grad_W1', [0])[-1]
            gl = history.get(f'grad_W{model.depth+1}', [0])[-1]
            print(f'  [{tag}] Ep {epoch:2d}  acc={acc:.4f}  sat: {sats}  '
                  f'g1={g1:.5f}  g_last={gl:.5f}')

    return history


# =============================================================================
# Main
# =============================================================================

def main():
    print(f'Device: {DEVICE}')
    print(f'Question: Does LN+tanh match Iso at depth?')
    print(f'Prediction: If yes -> Jacobian preservation is the principle, not iso specifically')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    torch.manual_seed(SEED)
    probe_x, _ = next(iter(test_loader))
    probe_x = probe_x[:1024].to(DEVICE)

    configs = [
        # (tag,          activation, norm,  depth)
        ('Base-1L',      'tanh', None,  1),
        ('Base-2L',      'tanh', None,  2),
        ('Base-3L',      'tanh', None,  3),
        ('Iso-1L',       'iso',  None,  1),
        ('Iso-2L',       'iso',  None,  2),
        ('Iso-3L',       'iso',  None,  3),
        ('LN+tanh-1L',   'tanh', 'LN',  1),
        ('LN+tanh-2L',   'tanh', 'LN',  2),
        ('LN+tanh-3L',   'tanh', 'LN',  3),
        ('LN+Iso-1L',    'iso',  'LN',  1),
        ('LN+Iso-2L',    'iso',  'LN',  2),
        ('LN+Iso-3L',    'iso',  'LN',  3),
        ('RMS+tanh-1L',  'tanh', 'RMS', 1),
        ('RMS+tanh-2L',  'tanh', 'RMS', 2),
        ('RMS+tanh-3L',  'tanh', 'RMS', 3),
    ]

    all_histories = {}
    total = len(configs)

    for i, (tag, act, norm, depth) in enumerate(configs, 1):
        print(f'\n[{i}/{total}] {tag}')
        torch.manual_seed(SEED)
        model = FlexMLP(input_dim, WIDTH, num_classes, depth, act, norm).to(DEVICE)
        hist = train_with_anatomy(
            model, act, train_loader, test_loader,
            probe_x, EPOCHS, LR, DEVICE, tag
        )
        all_histories[tag] = hist
        print(f'  Final acc: {hist["acc"][-1]:.4f}')

    epochs_range = list(range(1, EPOCHS + 1))

    # =========================================================================
    # Summary
    # =========================================================================
    print(f'\n{"="*75}')
    print('SUMMARY: Final accuracy by depth')
    print(f'{"="*75}')
    print(f'{"Model type":>14}  {"1L":>7}  {"2L":>7}  {"3L":>7}  '
          f'{"2L-1L":>7}  {"3L-1L":>7}')

    for prefix, act, norm in [
        ('Base',     'tanh', None),
        ('Iso',      'iso',  None),
        ('LN+tanh',  'tanh', 'LN'),
        ('LN+Iso',   'iso',  'LN'),
        ('RMS+tanh', 'tanh', 'RMS'),
    ]:
        accs = [all_histories[f'{prefix}-{d}L']['acc'][-1] for d in (1, 2, 3)]
        deltas = [accs[1]-accs[0], accs[2]-accs[0]]
        print(f'{prefix:>14}  {accs[0]:.4f}  {accs[1]:.4f}  {accs[2]:.4f}  '
              f'{deltas[0]:+.4f}  {deltas[1]:+.4f}')

    print(f'\nGradient ratio W1/W_last (epoch-mean):')
    for tag, act, norm, depth in configs:
        h = all_histories[tag]
        g1 = np.mean(h.get('grad_W1', [1e-10]))
        gl = np.mean(h.get(f'grad_W{depth+1}', [1e-10]))
        ratio = g1 / (gl + 1e-10)
        print(f'  {tag}: {ratio:.4f}')

    # Verdict
    iso3  = all_histories['Iso-3L']['acc'][-1]
    ln3   = all_histories['LN+tanh-3L']['acc'][-1]
    base3 = all_histories['Base-3L']['acc'][-1]
    gap   = iso3 - base3
    ln_closes = (ln3 - base3) / (gap + 1e-8)

    if ln_closes > 0.85:
        verdict = (f'OUTCOME 1: LN+tanh closes {ln_closes*100:.0f}% of Iso-Base gap at 3L '
                   f'({ln3:.4f} vs Iso {iso3:.4f} vs Base {base3:.4f}). '
                   f'Jacobian preservation is the principle; iso is one of several solutions.')
    elif ln_closes > 0.40:
        verdict = (f'OUTCOME 2: LN+tanh closes {ln_closes*100:.0f}% of Iso-Base gap at 3L '
                   f'({ln3:.4f} vs Iso {iso3:.4f} vs Base {base3:.4f}). '
                   f'LN helps but isotropic Jacobian provides additional benefit.')
    else:
        verdict = (f'OUTCOME 3: LN+tanh closes only {ln_closes*100:.0f}% of Iso-Base gap at 3L '
                   f'({ln3:.4f} vs Iso {iso3:.4f} vs Base {base3:.4f}). '
                   f'Isotropic Jacobian structure is specifically important.')

    print(f'\n{verdict}')

    # =========================================================================
    # Plots
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    depth_colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}

    # Top row: accuracy curves for each model family at 3L
    ax = axes[0, 0]
    for tag, color, ls in [
        ('Base-3L',     '#d62728', '-'),
        ('Iso-3L',      '#1f77b4', '-'),
        ('LN+tanh-3L',  '#2ca02c', '-'),
        ('LN+Iso-3L',   '#9467bd', '-'),
        ('RMS+tanh-3L', '#ff7f0e', '-'),
    ]:
        ax.plot(epochs_range, all_histories[tag]['acc'], color=color, ls=ls, label=tag)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test accuracy')
    ax.set_title('Accuracy at 3L depth\n(key comparison)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Depth scaling per model family
    ax = axes[0, 1]
    depths_x = [1, 2, 3]
    for prefix, color in [
        ('Base',     '#d62728'),
        ('Iso',      '#1f77b4'),
        ('LN+tanh',  '#2ca02c'),
        ('LN+Iso',   '#9467bd'),
        ('RMS+tanh', '#ff7f0e'),
    ]:
        accs = [all_histories[f'{prefix}-{d}L']['acc'][-1] for d in depths_x]
        ax.plot(depths_x, accs, marker='o', color=color, label=prefix)
    ax.set_xlabel('Depth (layers)')
    ax.set_ylabel('Final test accuracy')
    ax.set_title('Accuracy vs Depth\n(does LN+tanh scale like Iso?)')
    ax.set_xticks([1, 2, 3])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Saturation fraction at 3L
    ax = axes[0, 2]
    for tag, color in [
        ('Base-3L',     '#d62728'),
        ('Iso-3L',      '#1f77b4'),
        ('LN+tanh-3L',  '#2ca02c'),
        ('LN+Iso-3L',   '#9467bd'),
        ('RMS+tanh-3L', '#ff7f0e'),
    ]:
        h = all_histories[tag]
        # Mean saturation across all hidden layers
        depth = int(tag.split('-')[1][0])
        mean_sat = [
            np.mean([h.get(f'sat_L{li}', [0])[ep] for li in range(1, depth+1)])
            for ep in range(EPOCHS)
        ]
        ax.plot(epochs_range, mean_sat, color=color, label=tag)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean saturation fraction')
    ax.set_title('Saturation at 3L\n(LN should suppress Base saturation)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom row: gradient anatomy
    ax = axes[1, 0]
    for tag, color in [
        ('Base-3L',     '#d62728'),
        ('Iso-3L',      '#1f77b4'),
        ('LN+tanh-3L',  '#2ca02c'),
        ('LN+Iso-3L',   '#9467bd'),
        ('RMS+tanh-3L', '#ff7f0e'),
    ]:
        h = all_histories[tag]
        ax.plot(epochs_range, h.get('grad_W1', [0]*EPOCHS), color=color, label=tag)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('||dL/dW1||')
    ax.set_title('First-layer gradient norm (3L)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = axes[1, 1]
    for tag, color in [
        ('Base-3L',     '#d62728'),
        ('Iso-3L',      '#1f77b4'),
        ('LN+tanh-3L',  '#2ca02c'),
        ('LN+Iso-3L',   '#9467bd'),
        ('RMS+tanh-3L', '#ff7f0e'),
    ]:
        h = all_histories[tag]
        depth = int(tag.split('-')[1][0])
        g1 = np.array(h.get('grad_W1', [1e-10]*EPOCHS))
        gl = np.array(h.get(f'grad_W{depth+1}', [1e-10]*EPOCHS))
        ratio = g1 / (gl + 1e-10)
        ax.plot(epochs_range, ratio, color=color, label=tag)
    ax.axhline(1.0, color='black', ls=':', alpha=0.4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('W1/W_last gradient ratio')
    ax.set_title('Gradient balance (3L)\n(1.0 = perfectly balanced)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Pre-activation norms
    ax = axes[1, 2]
    for tag, color in [
        ('Base-3L',     '#d62728'),
        ('Iso-3L',      '#1f77b4'),
        ('LN+tanh-3L',  '#2ca02c'),
        ('LN+Iso-3L',   '#9467bd'),
        ('RMS+tanh-3L', '#ff7f0e'),
    ]:
        h = all_histories[tag]
        ax.plot(epochs_range, h.get('preact_norm_L1', [0]*EPOCHS), color=color, label=tag)
    ax.axhline(SAT_THRESHOLD, color='red', ls=':', alpha=0.5, label=f'sat={SAT_THRESHOLD:.2f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean ||h|| at L1')
    ax.set_title('Pre-activation norm (3L, L1)\n(LN should keep below threshold)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Test AE: LayerNorm vs Isotropic Activation', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'layernorm_comparison.png'), dpi=150)
    print('\nPlot saved to results/test_AE/layernorm_comparison.png')

    # =========================================================================
    # Save results.md
    # =========================================================================
    rows = []
    for tag, act, norm, depth in configs:
        h = all_histories[tag]
        acc  = h['acc'][-1]
        sats = [h.get(f'sat_L{li}', [0])[-1] for li in range(1, depth+1)]
        mean_sat = np.mean(sats)
        g1   = np.mean(h.get('grad_W1', [0]))
        gl   = np.mean(h.get(f'grad_W{depth+1}', [1e-10]))
        ratio = g1 / (gl + 1e-10)
        rows.append(f'| {tag} | {acc:.4f} | {mean_sat:.4f} | {ratio:.4f} |')

    md = f"""# Test AE -- LayerNorm vs Isotropic Activation

## Setup
- Models: Base, Iso, LN+tanh, LN+Iso, RMS+tanh at 1L/2L/3L depth
- Width: {WIDTH}, Epochs: {EPOCHS}, Seed: {SEED}, lr={LR}
- Device: {DEVICE}

## Question
Does LayerNorm + standard tanh match isotropic tanh at depth?
Is Jacobian preservation the real principle, or is isotropy specifically necessary?

## Results

| Model | Acc | Mean sat | Grad W1/W_last |
|---|---|---|---|
{chr(10).join(rows)}

## Depth Scaling Summary

| Model | 1L | 2L | 3L | 2L-1L | 3L-1L |
|---|---|---|---|---|---|
"""
    for prefix in ('Base', 'Iso', 'LN+tanh', 'LN+Iso', 'RMS+tanh'):
        accs = [all_histories[f'{prefix}-{d}L']['acc'][-1] for d in (1, 2, 3)]
        md += f'| {prefix} | {accs[0]:.4f} | {accs[1]:.4f} | {accs[2]:.4f} | {accs[1]-accs[0]:+.4f} | {accs[2]-accs[0]:+.4f} |\n'

    md += f"""
## Verdict
{verdict}

LN gap closure at 3L: {ln_closes*100:.1f}%
- Iso-3L:      {iso3:.4f}
- LN+tanh-3L:  {ln3:.4f}
- Base-3L:     {base3:.4f}

![LayerNorm comparison](layernorm_comparison.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    print('Results saved to results/test_AE/results.md')


if __name__ == '__main__':
    main()
