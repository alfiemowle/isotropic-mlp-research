"""
Test AG-B -- LN+tanh/RMS+tanh/LN+Iso Depth Scaling (companion to AG)
======================================================================
AG runs Base and Iso at depths [1,2,3,4,5,6] at width=128.
This companion runs LN+tanh, RMS+tanh, LN+Iso at identical configs.
Results saved to test_AG/ so they can be merged with AG for full comparison.

Key question: does the LN+tanh advantage over Iso (seen at 3L in AE) persist
or grow at depth 4, 5, 6? Or does Iso eventually catch up at greater depth?

Identical setup: WIDTH=128, EPOCHS=30, LR=0.08, BATCH=128, SEED=42, DEPTHS=[1..6]
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AG')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS  = 30
LR      = 0.08
BATCH   = 128
SEED    = 42
WIDTH   = 128
DEPTHS  = [1, 2, 3, 4, 5, 6]


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
        self.norm_type = norm
        self.act = IsoAct() if activation == 'iso' else nn.Tanh()
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
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        if epoch in (1, 10, 20, EPOCHS):
            acc = evaluate(model, test_loader, DEVICE)
            print(f'  [{tag}] Ep {epoch:2d}  acc={acc:.4f}')
    return evaluate(model, test_loader, DEVICE)


def main():
    print(f'Device: {DEVICE}, Width={WIDTH}  (AG-B companion: LN variants)')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    configs = [
        ('LN+tanh',  'tanh', 'LN'),
        ('RMS+tanh', 'tanh', 'RMS'),
        ('LN+Iso',   'iso',  'LN'),
    ]

    results = {}  # (tag, depth) -> acc
    total = len(configs) * len(DEPTHS)
    run = 0

    for tag, act, norm in configs:
        for depth in DEPTHS:
            run += 1
            label = f'{tag}-{depth}L'
            print(f'\n[{run}/{total}] {label}')
            torch.manual_seed(SEED)
            model = FlexMLP(input_dim, WIDTH, num_classes, depth, act, norm).to(DEVICE)
            acc = train(model, train_loader, test_loader, label)
            results[(tag, depth)] = acc
            print(f'  Final: {acc:.4f}')

    # Summary
    print(f'\n{"="*65}')
    print(f'AG-B SUMMARY: LN variants vs depth (width={WIDTH})')
    print(f'{"Depth":>6}  {"LN+tanh":>9}  {"RMS+tanh":>9}  {"LN+Iso":>9}')
    for d in DEPTHS:
        ln  = results[('LN+tanh',  d)]
        rms = results[('RMS+tanh', d)]
        lni = results[('LN+Iso',   d)]
        print(f'{d:>6}  {ln:.4f}  {rms:.4f}  {lni:.4f}')

    # Per-layer slopes
    for tag, _, _ in configs:
        accs = [results[(tag, d)] for d in DEPTHS]
        slope = np.polyfit(DEPTHS, accs, 1)[0]
        print(f'  {tag} slope: {slope:+.4f}/layer')

    # Save results
    rows = []
    for d in DEPTHS:
        ln  = results[('LN+tanh',  d)]
        rms = results[('RMS+tanh', d)]
        lni = results[('LN+Iso',   d)]
        ln1  = results[('LN+tanh',  1)]
        rms1 = results[('RMS+tanh', 1)]
        lni1 = results[('LN+Iso',   1)]
        rows.append(f'| {d} | {ln:.4f} | {ln-ln1:+.4f} | '
                    f'{rms:.4f} | {rms-rms1:+.4f} | '
                    f'{lni:.4f} | {lni-lni1:+.4f} |')

    ag_b_md = f"""# Test AG-B -- LN Variant Depth Scaling (companion to AG)

## Setup
Identical to AG: Width={WIDTH}, Epochs={EPOCHS}, lr={LR}, seed={SEED}, depths={DEPTHS}

## Results: LN variants

| Depth | LN+tanh | Delta | RMS+tanh | Delta | LN+Iso | Delta |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## Per-layer slope (linear fit)
"""
    for tag, _, _ in configs:
        accs = [results[(tag, d)] for d in DEPTHS]
        slope = np.polyfit(DEPTHS, accs, 1)[0]
        ag_b_md += f'- {tag}: {slope:+.4f}/layer\n'

    ag_b_md += '\n(For reference from width=32, Test AE: Base=-0.020/layer, Iso=+0.014/layer)\n'

    with open(os.path.join(RESULTS_DIR, 'results_LN.md'), 'w') as f:
        f.write(ag_b_md)
    print('Saved to results/test_AG/results_LN.md')

    # Build combined plot if AG results exist
    try:
        ag_results_path = os.path.join(RESULTS_DIR, 'results.md')
        with open(ag_results_path) as f:
            ag_text = f.read()

        import re
        # Parse: | depth | Base | Iso | Gap | Base delta | Iso delta |
        pattern = r'\|\s*(\d)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[+\-][\d.]+\s*\|\s*[+\-][\d.]+\s*\|\s*[+\-][\d.]+\s*\|'
        matches = re.findall(pattern, ag_text)

        if matches:
            ag_depths = [int(m[0]) for m in matches]
            base_accs = [float(m[1]) for m in matches]
            iso_accs  = [float(m[2]) for m in matches]

            fig, axes = plt.subplots(1, 2, figsize=(13, 5))

            ax = axes[0]
            ax.plot(ag_depths, base_accs, 'o-', color='#d62728', label='Base')
            ax.plot(ag_depths, iso_accs,  'o-', color='#1f77b4', label='Iso')
            ax.plot(DEPTHS, [results[('LN+tanh',  d)] for d in DEPTHS],
                    'o-', color='#2ca02c', label='LN+tanh')
            ax.plot(DEPTHS, [results[('RMS+tanh', d)] for d in DEPTHS],
                    'o-', color='#ff7f0e', label='RMS+tanh')
            ax.plot(DEPTHS, [results[('LN+Iso',   d)] for d in DEPTHS],
                    'o-', color='#9467bd', label='LN+Iso')
            ax.set_xlabel('Depth'); ax.set_ylabel('Test accuracy')
            ax.set_title(f'All models: accuracy vs depth (width={WIDTH})')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            ax.set_xticks(DEPTHS)

            ax = axes[1]
            iso_dict = dict(zip(ag_depths, iso_accs))
            for tag_label, color in [('LN+tanh', '#2ca02c'), ('RMS+tanh', '#ff7f0e'), ('LN+Iso', '#9467bd')]:
                deltas = [results[(tag_label, d)] - iso_dict.get(d, 0) for d in DEPTHS if d in iso_dict]
                ax.plot([d for d in DEPTHS if d in iso_dict], deltas, 'o-', color=color, label=f'{tag_label} - Iso')
            ax.axhline(0, color='black', ls=':', alpha=0.4)
            ax.set_xlabel('Depth'); ax.set_ylabel('Accuracy delta vs Iso')
            ax.set_title(f'LN/RMS advantage over Iso vs depth (width={WIDTH})')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            ax.set_xticks(DEPTHS)

            plt.suptitle(f'AG + AG-B: Full depth scaling comparison (width={WIDTH})', fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'depth_scaling_full.png'), dpi=150)
            print('Combined plot saved to results/test_AG/depth_scaling_full.png')
    except Exception as e:
        print(f'Could not build combined plot (AG may still be running): {e}')


if __name__ == '__main__':
    main()
