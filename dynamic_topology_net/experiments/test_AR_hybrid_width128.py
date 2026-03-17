"""
Test AR -- Hybrid Architectures at Width=128
=============================================
Test AL showed that hybrid architectures (mixing Iso and LN+GELU layers)
can outperform both pure-Iso and pure-LN+GELU at width=32.

Key AL findings:
  - Iso-last-3L: 48.91% (beat Pure-LNG 47.94% and Pure-Iso 44.69%)
  - Iso-first-4L: 49.78% (beat Pure-LNG-4L 48.31%)

Open question: do these hybrid advantages hold at width=128, or are they
artefacts of width=32's limited capacity?

This test re-runs key configs at width=128 (30 epochs, depth 3):
  - Pure-Iso-3L         (reference)
  - Pure-LN+GELU-3L     (reference)
  - Iso-first-3L        (Iso→LNG→LNG)
  - Iso-last-3L         (LNG→LNG→Iso)  ← best at w=32
  - Iso-sandwich-3L     (Iso→LNG→Iso)

Also runs depth=4 at width=128:
  - Pure-Iso-4L
  - Pure-LN+GELU-4L
  - Iso-first-4L        ← best at w=32
  - Iso-last-4L
  - Iso-sandwich-4L

The relative ordering at w=32 is the key claim to verify.
If the ordering changes at w=128, the AL results may be capacity-artefacts.
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AR')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 30
LR     = 0.08
BATCH  = 128
WIDTH  = 128
SEED   = 42


class IsoAct(nn.Module):
    def forward(self, x):
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.tanh(n) * x / n


class HybridMLP(nn.Module):
    """
    layer_specs: list of (use_ln, use_iso) per layer
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
    print(f'Width: {WIDTH} (AR = width-128 replication of AL)')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    ISO = (False, True)   # Iso activation, no LN
    LNG = (True,  False)  # LN + GELU

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
    ]

    all_configs = configs_3L + configs_4L
    results = {}
    total   = len(all_configs)

    for run, (tag, specs) in enumerate(all_configs, 1):
        layer_str = ['Iso' if s[1] else 'LN+GELU' for s in specs]
        print(f'\n[{run}/{total}] {tag}  layers={layer_str}')
        torch.manual_seed(SEED)
        model = HybridMLP(input_dim, WIDTH, num_classes, specs).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Params: {n_params:,}')
        hist  = train(model, train_loader, test_loader, tag)
        results[tag] = hist
        print(f'  Final: {hist[-1]:.4f}  Peak: {max(hist):.4f}')

    # Summary
    print(f'\n{"="*75}')
    print(f'SUMMARY (width={WIDTH})')
    print(f'{"Model":<24}  {"Final":>7}  {"Peak":>7}  {"vs Pure-Iso":>12}  {"vs Pure-LNG":>12}')

    iso3  = results['Pure-Iso-3L'][-1]
    lng3  = results['Pure-LN+GELU-3L'][-1]
    iso4  = results['Pure-Iso-4L'][-1]
    lng4  = results['Pure-LN+GELU-4L'][-1]

    for tag, specs in all_configs:
        depth   = len(specs)
        ref_iso = iso3 if depth == 3 else iso4
        ref_lng = lng3 if depth == 3 else lng4
        final   = results[tag][-1]
        peak    = max(results[tag])
        n_iso   = sum(1 for s in specs if s[1])
        print(f'{tag:<24}  {final:.4f}  {peak:.4f}  '
              f'{final-ref_iso:+.4f} vs Iso  {final-ref_lng:+.4f} vs LNG  '
              f'({n_iso}/{depth} Iso layers)')

    # Key comparison with AL (w=32) results — hardcoded from AL output
    print('\nKey comparison vs AL (width=32):')
    al_refs = {
        'Pure-Iso-3L':     0.4469,
        'Pure-LN+GELU-3L': 0.4794,
        'Iso-last-3L':     0.4891,
        'Iso-first-4L':    0.4978,
    }
    for tag, al_acc in al_refs.items():
        if tag in results:
            ar_acc = results[tag][-1]
            print(f'  {tag:<24}: AL={al_acc:.4f}  AR={ar_acc:.4f}  delta={ar_acc-al_acc:+.4f}')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = list(range(1, EPOCHS + 1))
    colors = {
        'Pure-Iso':      '#1f77b4',
        'Pure-LN+GELU':  '#2ca02c',
        'Iso-first':     '#ff7f0e',
        'Iso-last':      '#d62728',
        'Iso-sandwich':  '#9467bd',
    }

    for ax_idx, (configs, depth) in enumerate([(configs_3L, 3), (configs_4L, 4)]):
        ax = axes[ax_idx]
        for tag, _ in configs:
            short = tag.replace(f'-{depth}L', '')
            c = colors.get(short, '#7f7f7f')
            ax.plot(epochs_range, results[tag], color=c, label=tag)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
        ax.set_title(f'Depth {depth}: hybrid vs pure (width={WIDTH})')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle(f'Test AR: Hybrid Architectures at Width={WIDTH}', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'hybrid_w128.png'), dpi=150)
    print('\nPlot saved.')

    rows = []
    for tag, specs in all_configs:
        depth   = len(specs)
        n_iso   = sum(1 for s in specs if s[1])
        ref_iso = iso3 if depth == 3 else iso4
        ref_lng = lng3 if depth == 3 else lng4
        final   = results[tag][-1]
        peak    = max(results[tag])
        layer_s = ','.join('Iso' if s[1] else 'LNG' for s in specs)
        rows.append(
            f'| {tag} | {layer_s} | {n_iso}/{depth} | {final:.4f} | {peak:.4f} '
            f'| {final-ref_iso:+.4f} | {final-ref_lng:+.4f} |'
        )

    md = f"""# Test AR -- Hybrid Architectures at Width={WIDTH}

## Setup
- Width: {WIDTH} (replicates AL at width=32)
- Epochs: {EPOCHS}, lr={LR}, seed={SEED}
- Device: {DEVICE}

## Question
Do the hybrid architecture advantages observed in AL (width=32) generalise
to width=128? Is Iso-last / Iso-first still better than both pure options?

## Results

| Model | Layers | Iso count | Final | Peak | vs Iso | vs LNG |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## AL (width=32) references for comparison
- Pure-Iso-3L: 0.4469
- Pure-LN+GELU-3L: 0.4794
- Iso-last-3L: 0.4891  (best 3L at w=32)
- Iso-first-4L: 0.4978 (best 4L at w=32)

![Hybrid architectures w128](hybrid_w128.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w') as f:
        f.write(md)
    print('Results saved.')


if __name__ == '__main__':
    main()
