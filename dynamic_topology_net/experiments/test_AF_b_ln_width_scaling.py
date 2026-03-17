"""
Test AF-B -- LN+tanh/RMS+tanh/LN+Iso Width Scaling (companion to AF)
=====================================================================
AF runs Base and Iso at widths [32,64,128,256,512] depth=2.
This companion runs LN+tanh, RMS+tanh, LN+Iso at identical configs.
Results saved to test_AF/ so they can be merged with AF for full comparison.

Identical setup: EPOCHS=24, LR=0.08, BATCH=128, SEED=42, WIDTHS=[32..512], depth=2
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_topology_net.core import load_cifar10
from dynamic_topology_net.core.train_utils import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AF')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS  = 24
LR      = 0.08
BATCH   = 128
SEED    = 42
WIDTHS  = [32, 64, 128, 256, 512]
DEPTH   = 2


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
    def __init__(self, input_dim, width, num_classes, depth, activation, norm):
        super().__init__()
        self.depth = depth
        act = IsoAct() if activation == 'iso' else nn.Tanh()
        dims = [input_dim] + [width] * depth
        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:]), 1):
            setattr(self, f'linear{i}', nn.Linear(d_in, d_out))
            if norm == 'LN':
                setattr(self, f'norm{i}', nn.LayerNorm(d_out))
            elif norm == 'RMS':
                setattr(self, f'norm{i}', RMSNorm(d_out))
        self.act = act
        self.norm_type = norm
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
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        if epoch in (1, 12, EPOCHS):
            acc = evaluate(model, test_loader, DEVICE)
            print(f'  [{tag}] Ep {epoch:2d}  acc={acc:.4f}')
    acc = evaluate(model, test_loader, DEVICE)
    return acc, time.time() - t0


def main():
    print(f'Device: {DEVICE}  (AF-B companion: LN variants)')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    configs = [
        ('LN+tanh',  'tanh', 'LN'),
        ('RMS+tanh', 'tanh', 'RMS'),
        ('LN+Iso',   'iso',  'LN'),
    ]

    results = {}  # (tag, width) -> (acc, time)
    total = len(configs) * len(WIDTHS)
    run = 0

    for tag, act, norm in configs:
        for width in WIDTHS:
            run += 1
            label = f'{tag}-w{width}'
            print(f'\n[{run}/{total}] {label}')
            torch.manual_seed(SEED)
            model = FlexMLP(input_dim, width, num_classes, DEPTH, act, norm).to(DEVICE)
            acc, t = train(model, train_loader, test_loader, label)
            results[(tag, width)] = (acc, t)
            print(f'  Final acc: {acc:.4f}  time: {t:.1f}s')

    # Save combined results table
    print(f'\n{"="*65}')
    print('AF-B SUMMARY: LN variants vs width (depth=2)')
    print(f'{"Width":>6}  {"LN+tanh":>9}  {"RMS+tanh":>9}  {"LN+Iso":>9}')
    for w in WIDTHS:
        ln  = results[('LN+tanh',  w)][0]
        rms = results[('RMS+tanh', w)][0]
        lni = results[('LN+Iso',   w)][0]
        print(f'{w:>6}  {ln:.4f}  {rms:.4f}  {lni:.4f}')

    # Append to AF results.md if it exists, else write standalone
    rows = []
    for w in WIDTHS:
        ln  = results[('LN+tanh',  w)][0]
        rms = results[('RMS+tanh', w)][0]
        lni = results[('LN+Iso',   w)][0]
        rows.append(f'| {w} | {ln:.4f} | {rms:.4f} | {lni:.4f} |')

    af_b_md = f"""# Test AF-B -- LN Variant Width Scaling (companion to AF)

## Setup
Identical to AF: Epochs={EPOCHS}, lr={LR}, depth={DEPTH}, seed={SEED}, widths={WIDTHS}

## Results: LN variants

| Width | LN+tanh | RMS+tanh | LN+Iso |
|---|---|---|---|
{chr(10).join(rows)}

## Combined with AF (for full comparison, see plots below)
"""
    with open(os.path.join(RESULTS_DIR, 'results_LN.md'), 'w') as f:
        f.write(af_b_md)
    print('Saved to results/test_AF/results_LN.md')

    # Attempt to load AF results and make combined plot
    af_results_path = os.path.join(RESULTS_DIR, 'results.md')
    # Parse AF results if available
    try:
        with open(af_results_path) as f:
            af_text = f.read()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Try to extract AF numbers from results.md table
        import re
        pattern = r'\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[+\-][\d.]+\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
        matches = re.findall(pattern, af_text)
        if matches:
            af_widths = [int(m[0]) for m in matches]
            base_2L   = [float(m[3]) for m in matches]
            iso_2L    = [float(m[4]) for m in matches]

            ax = axes[0]
            ax.plot(af_widths, base_2L,   'o-', color='#d62728', label='Base-2L')
            ax.plot(af_widths, iso_2L,    'o-', color='#1f77b4', label='Iso-2L')
            ax.plot(WIDTHS, [results[('LN+tanh',  w)][0] for w in WIDTHS],
                    'o-', color='#2ca02c', label='LN+tanh-2L')
            ax.plot(WIDTHS, [results[('RMS+tanh', w)][0] for w in WIDTHS],
                    'o-', color='#ff7f0e', label='RMS+tanh-2L')
            ax.plot(WIDTHS, [results[('LN+Iso',   w)][0] for w in WIDTHS],
                    'o-', color='#9467bd', label='LN+Iso-2L')
            ax.set_xlabel('Width'); ax.set_ylabel('Test accuracy')
            ax.set_xscale('log', base=2)
            ax.set_title('All models: accuracy vs width (depth=2)')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            ax = axes[1]
            ln_accs  = [results[('LN+tanh',  w)][0] for w in WIDTHS]
            rms_accs = [results[('RMS+tanh', w)][0] for w in WIDTHS]
            iso_accs_aligned = [iso_2L[af_widths.index(w)] if w in af_widths else None for w in WIDTHS]
            for label, accs, color in [
                ('LN+tanh - Iso',  [ln-i  for ln, i in zip(ln_accs,  iso_accs_aligned) if i], '#2ca02c'),
                ('RMS+tanh - Iso', [rms-i for rms, i in zip(rms_accs, iso_accs_aligned) if i], '#ff7f0e'),
            ]:
                ax.plot(WIDTHS[:len(accs)], accs, 'o-', color=color, label=label)
            ax.axhline(0, color='black', ls=':', alpha=0.4)
            ax.set_xlabel('Width'); ax.set_ylabel('Accuracy delta vs Iso')
            ax.set_xscale('log', base=2)
            ax.set_title('LN/RMS advantage over Iso vs width')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            plt.suptitle('AF + AF-B: Full width scaling comparison', fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'width_scaling_full.png'), dpi=150)
            print('Combined plot saved to results/test_AF/width_scaling_full.png')
    except Exception as e:
        print(f'Could not build combined plot (AF may still be running): {e}')


if __name__ == '__main__':
    main()
