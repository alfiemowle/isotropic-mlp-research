"""
Test AH -- Modern Activations with LayerNorm
=============================================
AE showed LN+tanh beats Iso at all depths. The question: does this generalise
to modern activations used in production systems?

LN+GELU  = standard Transformer FFN block (BERT, GPT, ViT)
LN+SiLU  = LLaMA/Mistral FFN block (SiLU = Swish = x*sigmoid(x))
LN+ReLU  = classic deep learning baseline

If these all beat Iso, the conclusion is stronger: the modern deep learning
stack has already implicitly solved the depth stability problem that isotropic
activations address. The principle (Jacobian preservation via normalisation)
is already baked in.

Also tests: bare GELU/SiLU/ReLU without LN, to isolate the normalisation benefit.

Models: Iso, LN+tanh, LN+GELU, LN+SiLU, LN+ReLU, GELU, SiLU, ReLU
At depths 1L, 2L, 3L. Width=32 (same as AE for direct comparison).
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AH')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS  = 24
LR      = 0.08
BATCH   = 128
WIDTH   = 32
SEED    = 42


class IsoAct(nn.Module):
    def forward(self, x):
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.tanh(n) * x / n


class FlexMLP(nn.Module):
    def __init__(self, input_dim, width, num_classes, depth, act_fn, use_ln):
        super().__init__()
        self.depth = depth
        self.use_ln = use_ln
        dims = [input_dim] + [width] * depth
        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:]), 1):
            setattr(self, f'linear{i}', nn.Linear(d_in, d_out))
            if use_ln:
                setattr(self, f'ln{i}', nn.LayerNorm(d_out))
        self.act = act_fn
        self.output = nn.Linear(width, num_classes)

    def forward(self, x):
        h = x
        for i in range(1, self.depth + 1):
            h = getattr(self, f'linear{i}')(h)
            if self.use_ln:
                h = getattr(self, f'ln{i}')(h)
            h = self.act(h)
        return self.output(h)


def train(model, train_loader, test_loader, tag, epochs):
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    acc_hist = []
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        acc = evaluate(model, test_loader, DEVICE)
        acc_hist.append(acc)
        if epoch in (1, 12, epochs):
            print(f'  [{tag}] Ep {epoch:2d}  acc={acc:.4f}')
    return acc_hist


def main():
    print(f'Device: {DEVICE}')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    # (tag, act_module, use_ln)
    configs = [
        ('Iso',      IsoAct(),          False),
        ('LN+tanh',  nn.Tanh(),         True),
        ('LN+GELU',  nn.GELU(),         True),
        ('LN+SiLU',  nn.SiLU(),         True),
        ('LN+ReLU',  nn.ReLU(),         True),
        ('GELU',     nn.GELU(),         False),
        ('SiLU',     nn.SiLU(),         False),
        ('ReLU',     nn.ReLU(),         False),
    ]

    all_histories = {}
    total = len(configs) * 3

    for depth in (1, 2, 3):
        for tag, act, use_ln in configs:
            key = f'{tag}-{depth}L'
            run = (depth-1)*len(configs) + configs.index((tag, act, use_ln)) + 1
            print(f'\n[{run}/{total}] {key}')
            torch.manual_seed(SEED)
            model = FlexMLP(input_dim, WIDTH, num_classes, depth, act, use_ln).to(DEVICE)
            hist = train(model, train_loader, test_loader, key, EPOCHS)
            all_histories[key] = hist
            print(f'  Final: {hist[-1]:.4f}')

    # Summary
    print(f'\n{"="*70}')
    print('SUMMARY: Final accuracy by depth')
    print(f'{"Model":>12}  {"1L":>7}  {"2L":>7}  {"3L":>7}  {"3L-1L":>8}')

    # Iso reference
    iso1 = all_histories['Iso-1L'][-1]
    iso3 = all_histories['Iso-3L'][-1]

    for tag, _, _ in configs:
        a1 = all_histories[f'{tag}-1L'][-1]
        a2 = all_histories[f'{tag}-2L'][-1]
        a3 = all_histories[f'{tag}-3L'][-1]
        marker = ' *' if a3 > iso3 else ''
        print(f'{tag:>12}  {a1:.4f}  {a2:.4f}  {a3:.4f}  {a3-a1:+.4f}{marker}')
    print(f'(* = beats Iso-3L = {iso3:.4f})')

    # Verdict: how many LN models beat Iso at 3L?
    ln_models_3L = {tag: all_histories[f'{tag}-3L'][-1]
                    for tag, _, use_ln in configs if use_ln}
    beats_iso = {k: v for k, v in ln_models_3L.items() if v > iso3}
    print(f'\nLN models beating Iso at 3L: {len(beats_iso)}/{len(ln_models_3L)}')
    for k, v in sorted(beats_iso.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v:.4f} (Iso: {iso3:.4f}, delta={v-iso3:+.4f})')

    # Plot
    epochs_range = list(range(1, EPOCHS + 1))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {
        'Iso':      '#1f77b4',
        'LN+tanh':  '#2ca02c',
        'LN+GELU':  '#9467bd',
        'LN+SiLU':  '#e377c2',
        'LN+ReLU':  '#8c564b',
        'GELU':     '#bcbd22',
        'SiLU':     '#17becf',
        'ReLU':     '#7f7f7f',
    }
    ls_map = {True: '-', False: '--'}

    for di, depth in enumerate([1, 2, 3]):
        ax = axes[di]
        for tag, _, use_ln in configs:
            key = f'{tag}-{depth}L'
            ax.plot(epochs_range, all_histories[key],
                    color=colors[tag], ls=ls_map[use_ln],
                    label=f'{tag} ({all_histories[key][-1]:.3f})', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test accuracy')
        ax.set_title(f'Depth {depth}L')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Test AH: Modern Activations (solid=with LN, dashed=without LN)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'modern_activations.png'), dpi=150)
    print('\nPlot saved to results/test_AH/modern_activations.png')

    # Results.md
    rows = []
    for tag, _, use_ln in configs:
        a1 = all_histories[f'{tag}-1L'][-1]
        a2 = all_histories[f'{tag}-2L'][-1]
        a3 = all_histories[f'{tag}-3L'][-1]
        rows.append(f'| {tag} | {"Yes" if use_ln else "No"} | {a1:.4f} | {a2:.4f} | {a3:.4f} | {a3-a1:+.4f} |')

    md = f"""# Test AH -- Modern Activations with LayerNorm

## Setup
- Width: {WIDTH}, Epochs: {EPOCHS}, Seed: {SEED}, lr={LR}
- Device: {DEVICE}

## Question
Does the LN+tanh advantage over Iso generalise to modern activations?
LN+GELU = standard Transformer FFN. LN+SiLU = LLaMA FFN.

## Results

| Model | LN? | 1L | 2L | 3L | 3L-1L |
|---|---|---|---|---|---|
{chr(10).join(rows)}

Iso-3L reference: {iso3:.4f}

## LN models beating Iso at 3L: {len(beats_iso)}/{len(ln_models_3L)}
{chr(10).join(f'- {k}: {v:.4f} (+{v-iso3:.4f} over Iso)' for k, v in sorted(beats_iso.items(), key=lambda x: -x[1]))}

## Verdict
{"All LN models beat Iso at 3L -- normalisation is the key principle, activation function is secondary." if len(beats_iso) == len(ln_models_3L) else f"Mixed: {len(beats_iso)}/{len(ln_models_3L)} LN models beat Iso at 3L."}

![Modern activations](modern_activations.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w') as f:
        f.write(md)
    print('Results saved to results/test_AH/results.md')


if __name__ == '__main__':
    main()
