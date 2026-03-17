"""
Test AN -- Interleaved Training Protocol
=========================================
All prior accuracy tests trained to convergence THEN applied topology ops.
The paper's actual protocol interleaves training with periodic diagonalisation
and pruning/growing throughout training.

Key question: does calling partial_diagonalise during training change accuracy?
There is a real risk: partial_diagonalise rotates the parameter basis, making
Adam's stored moment estimates stale. The gradient direction doesn't change
(function is exactly preserved) but the parameter coordinates do.

Uses the core IsotropicMLP (1L, width=32) which has the proper
partial_diagonalise/prune_neuron/grow_neuron methods.

Conditions (all 60 epochs, width=32):
  A. Static:             train 60 epochs, no diagonalise
  B. Diag-only:          diagonalise every 5 epochs, no pruning (exact equiv)
  C. Diag+reset:         diagonalise every 5 epochs, reset Adam after each
  D. Prune-post:         train 60 epochs, diagonalise at ep60, prune to w=24
  E. Prune-mid:          train 30 epochs, diagonalise, prune to w=24, train 30
  F. Prune-interleaved:  diagonalise every 5ep, prune 1 neuron at ep20/40, end w=30

Questions:
  A vs B: does reparameterising during training hurt (via stale Adam)?
  B vs C: does resetting Adam after diagonalise help?
  D vs E: does timing of pruning matter?
  E vs F: does incremental pruning differ from one-shot pruning?
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_topology_net.core import load_cifar10
from dynamic_topology_net.core.models import IsotropicMLP
from dynamic_topology_net.core.train_utils import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AN')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS     = 60
LR         = 0.08
BATCH      = 128
WIDTH      = 32
PRUNE_TO   = 24
SEED       = 42
DIAG_EVERY = 5


def make_model(width=WIDTH):
    torch.manual_seed(SEED)
    return IsotropicMLP(input_dim=3072, width=width, num_classes=10).to(DEVICE)


def make_opt(model):
    return optim.Adam(model.parameters(), lr=LR)


def train_epoch(model, opt, train_loader):
    crit = nn.CrossEntropyLoss()
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()


def run(label, epoch_fn, total_epochs, test_loader):
    """epoch_fn(epoch) -> (model, opt) for that epoch; returns acc history."""
    hist = []
    for ep in range(1, total_epochs + 1):
        epoch_fn(ep)
    # We track acc inside epoch_fn via closure, returned via shared list
    return hist


def main():
    print(f'Device: {DEVICE}')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    results = {}

    # ── A: Static ─────────────────────────────────────────────────────────────
    print('\n[A] Static: 60 epochs, no diagonalise')
    model = make_model(); opt = make_opt(model); hist = []
    for ep in range(1, EPOCHS + 1):
        train_epoch(model, opt, train_loader)
        acc = evaluate(model, test_loader, DEVICE)
        hist.append(acc)
        if ep in (1, 20, 40, 60):
            print(f'  ep{ep:2d}  acc={acc:.4f}  width={model.width}')
    results['A-Static'] = hist
    print(f'  Final: {hist[-1]:.4f}')

    # ── B: Diag-only (no Adam reset) ──────────────────────────────────────────
    print('\n[B] Diag-only: diagonalise every 5 epochs, no pruning, no Adam reset')
    model = make_model(); opt = make_opt(model); hist = []
    for ep in range(1, EPOCHS + 1):
        train_epoch(model, opt, train_loader)
        if ep % DIAG_EVERY == 0:
            model.partial_diagonalise()   # reparameterise; Adam state now stale
        acc = evaluate(model, test_loader, DEVICE)
        hist.append(acc)
        if ep in (1, 20, 40, 60):
            print(f'  ep{ep:2d}  acc={acc:.4f}')
    results['B-Diag-only'] = hist
    print(f'  Final: {hist[-1]:.4f}')

    # ── C: Diag + Adam reset ──────────────────────────────────────────────────
    print('\n[C] Diag+reset: diagonalise every 5 epochs AND reset Adam state')
    model = make_model(); opt = make_opt(model); hist = []
    for ep in range(1, EPOCHS + 1):
        train_epoch(model, opt, train_loader)
        if ep % DIAG_EVERY == 0:
            model.partial_diagonalise()
            opt = make_opt(model)         # fresh Adam for the new basis
        acc = evaluate(model, test_loader, DEVICE)
        hist.append(acc)
        if ep in (1, 20, 40, 60):
            print(f'  ep{ep:2d}  acc={acc:.4f}')
    results['C-Diag+reset'] = hist
    print(f'  Final: {hist[-1]:.4f}')

    # ── D: Prune post-training ────────────────────────────────────────────────
    print('\n[D] Prune-post: train 60 epochs, prune to 24 at end')
    model = make_model(); opt = make_opt(model); hist = []
    for ep in range(1, EPOCHS + 1):
        train_epoch(model, opt, train_loader)
        acc = evaluate(model, test_loader, DEVICE)
        hist.append(acc)
    acc_before_prune = hist[-1]
    model.partial_diagonalise()
    while model.width > PRUNE_TO:
        model.prune_neuron(model.width - 1)   # remove smallest SV
    acc_after_prune = evaluate(model, test_loader, DEVICE)
    # Fine-tune 5 epochs
    opt = make_opt(model)
    for ep in range(5):
        train_epoch(model, opt, train_loader)
        hist.append(evaluate(model, test_loader, DEVICE))
    results['D-Prune-post'] = hist
    print(f'  Before prune: {acc_before_prune:.4f}  After prune: {acc_after_prune:.4f}  '
          f'After 5ep ft: {hist[-1]:.4f}')

    # ── E: Prune mid-training ─────────────────────────────────────────────────
    print('\n[E] Prune-mid: train 30 epochs, prune to 24, train 30 more')
    model = make_model(); opt = make_opt(model); hist = []
    for ep in range(1, 31):
        train_epoch(model, opt, train_loader)
        hist.append(evaluate(model, test_loader, DEVICE))
    acc_before_prune = hist[-1]
    model.partial_diagonalise()
    while model.width > PRUNE_TO:
        model.prune_neuron(model.width - 1)
    acc_after_prune = evaluate(model, test_loader, DEVICE)
    opt = make_opt(model)
    for ep in range(1, 31):
        train_epoch(model, opt, train_loader)
        hist.append(evaluate(model, test_loader, DEVICE))
    results['E-Prune-mid'] = hist
    print(f'  Before prune (ep30): {acc_before_prune:.4f}  '
          f'After prune: {acc_after_prune:.4f}  Final: {hist[-1]:.4f}')

    # ── F: Incremental pruning ────────────────────────────────────────────────
    print('\n[F] Prune-incremental: diagonalise every 5ep, prune 1 neuron at ep20 and ep40')
    model = make_model(); opt = make_opt(model); hist = []
    for ep in range(1, EPOCHS + 1):
        train_epoch(model, opt, train_loader)
        if ep % DIAG_EVERY == 0:
            model.partial_diagonalise()
            opt = make_opt(model)
        if ep in (20, 40) and model.width > PRUNE_TO:
            model.partial_diagonalise()
            model.prune_neuron(model.width - 1)
            opt = make_opt(model)
            print(f'    Pruned at ep{ep}, new width={model.width}')
        acc = evaluate(model, test_loader, DEVICE)
        hist.append(acc)
        if ep in (1, 20, 40, 60):
            print(f'  ep{ep:2d}  acc={acc:.4f}  width={model.width}')
    results['F-Prune-incremental'] = hist
    print(f'  Final: {hist[-1]:.4f}  Final width: {model.width}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print('SUMMARY')
    ref = results['A-Static'][-1]
    for tag, hist in results.items():
        final = hist[59] if len(hist) >= 60 else hist[-1]   # epoch 60 value
        print(f'  {tag:<28}: ep60={final:.4f}  delta={final-ref:+.4f}')

    print('\nKey comparisons:')
    a60 = results['A-Static'][59]
    b60 = results['B-Diag-only'][59]
    c60 = results['C-Diag+reset'][59]
    print(f'  A vs B (diag hurts?):       {a60:.4f} vs {b60:.4f}  delta={b60-a60:+.4f}')
    print(f'  B vs C (reset helps?):      {b60:.4f} vs {c60:.4f}  delta={c60-b60:+.4f}')
    d_final = results['D-Prune-post'][-1]
    e_final = results['E-Prune-mid'][-1]
    print(f'  D vs E (prune timing):      {d_final:.4f} vs {e_final:.4f}  delta={e_final-d_final:+.4f}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {
        'A-Static': '#1f77b4', 'B-Diag-only': '#2ca02c',
        'C-Diag+reset': '#ff7f0e', 'D-Prune-post': '#d62728',
        'E-Prune-mid': '#9467bd', 'F-Prune-incremental': '#8c564b',
    }

    ax = axes[0]
    for tag, hist in results.items():
        eps = list(range(1, len(hist) + 1))
        ax.plot(eps, hist, color=colors[tag], label=f'{tag} ({hist[min(59,len(hist)-1)]:.3f})')
    ax.axvline(30, color='black', ls=':', alpha=0.3)
    ax.axvline(60, color='black', ls=':', alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('All protocols (full curves)')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for tag in ['A-Static', 'B-Diag-only', 'C-Diag+reset']:
        hist = results[tag]
        ax.plot(range(1, 61), hist[:60], color=colors[tag], label=tag)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('Static vs diagonalise-during-training\n(no pruning — isolates reparameterisation effect)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Test AN: Interleaved Training Protocol', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'interleaved.png'), dpi=150)
    print('\nPlot saved.')

    rows = []
    for tag, hist in results.items():
        final = hist[59] if len(hist) >= 60 else hist[-1]
        peak  = max(hist)
        rows.append(f'| {tag} | {final:.4f} | {peak:.4f} | {final-ref:+.4f} |')

    md = f"""# Test AN -- Interleaved Training Protocol

## Setup
- Core IsotropicMLP (1L), width={WIDTH}, {EPOCHS} epochs base, lr={LR}, seed={SEED}
- Device: {DEVICE}

## Question
Does running partial_diagonalise during training (as the paper intends) change
outcomes vs static training then post-hoc topology operations?

## Results

| Protocol | ep60 acc | Peak | vs Static |
|---|---|---|---|
{chr(10).join(rows)}

## Key comparisons
- A vs B: effect of diagonalising during training (stale Adam momentum)
- B vs C: whether resetting Adam after diagonalise recovers performance
- D vs E: whether pruning mid-training vs post-training matters

![Interleaved protocol](interleaved.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w') as f:
        f.write(md)
    print('Results saved.')


if __name__ == '__main__':
    main()
