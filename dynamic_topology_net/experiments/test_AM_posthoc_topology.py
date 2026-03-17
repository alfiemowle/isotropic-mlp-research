"""
Test AM -- Post-hoc Topology on LN+tanh vs Iso
===============================================
Test AJ showed LN+tanh scaffold insertion gives max_diff=0.086 (not inert).
Iso gives max_diff=0.000003 (exactly inert).

The practical question: is 0.086 small enough that after 1 fine-tune
epoch, the LN+tanh model recovers fully? If yes, approximate dynamic
topology might be usable with LN+tanh despite the theoretical impurity.

Protocol:
  1. Train both Iso-3L and LN+tanh-3L to convergence (30 epochs)
  2. Expand width: 32 -> 48 (add 16 scaffold neurons)
  3. Measure output diff immediately after expansion (quantify disruption)
  4. Fine-tune for 0, 1, 2, 3, 5 epochs
  5. Compare accuracy recovery trajectories

Also tests pruning:
  1. Take trained models (width=32)
  2. Prune to width=24 (remove 8 neurons)
  3. For Iso: use SV criterion. For LN+tanh: use W2-norm criterion (AJ finding)
  4. Fine-tune and compare recovery
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
from dynamic_topology_net.core.train_utils import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AM')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS_TRAIN = 30
LR           = 0.08
BATCH        = 128
WIDTH        = 32
DEPTH        = 3
SEED         = 42
GROW_TO      = 48   # expand by 16
PRUNE_TO     = 24   # prune by 8


class IsoAct(nn.Module):
    def forward(self, x):
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.tanh(n) * x / n


class IsoMLP(nn.Module):
    def __init__(self, input_dim, width, num_classes, depth):
        super().__init__()
        self.depth = depth
        self.width = width
        dims = [input_dim] + [width] * depth
        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:]), 1):
            setattr(self, f'W{i}', nn.Linear(d_in, d_out))
        self.out = nn.Linear(width, num_classes)
        self.act = IsoAct()

    def forward(self, x):
        h = x
        for i in range(1, self.depth + 1):
            h = getattr(self, f'W{i}')(h)
            h = self.act(h)
        return self.out(h)


class LNtanhMLP(nn.Module):
    def __init__(self, input_dim, width, num_classes, depth):
        super().__init__()
        self.depth = depth
        self.width = width
        dims = [input_dim] + [width] * depth
        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:]), 1):
            setattr(self, f'W{i}', nn.Linear(d_in, d_out))
            setattr(self, f'ln{i}', nn.LayerNorm(d_out))
        self.out = nn.Linear(width, num_classes)

    def forward(self, x):
        h = x
        for i in range(1, self.depth + 1):
            h = getattr(self, f'W{i}')(h)
            h = getattr(self, f'ln{i}')(h)
            h = torch.tanh(h)
        return self.out(h)


def train_model(model, train_loader, test_loader, tag, epochs):
    opt  = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    hist = []
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        acc = evaluate(model, test_loader, DEVICE)
        hist.append(acc)
        if epoch in (1, 10, 20, epochs):
            print(f'  [{tag}] Ep {epoch:2d}  acc={acc:.4f}')
    return hist


# ─── Growth helpers ──────────────────────────────────────────────────────────

def grow_iso(model, new_width, input_dim, num_classes):
    """Expand IsoMLP to new_width by adding zero-weight scaffold neurons."""
    old_w = model.width
    add   = new_width - old_w
    new_model = IsoMLP(input_dim, new_width, num_classes, model.depth).to(DEVICE)

    for i in range(1, model.depth + 1):
        old_W = getattr(model, f'W{i}')
        new_W = getattr(new_model, f'W{i}')
        with torch.no_grad():
            # Copy existing weights; new rows/cols stay zero (scaffold)
            d_in  = old_W.weight.shape[1]
            d_out = old_W.weight.shape[0]
            new_W.weight[:d_out, :d_in].copy_(old_W.weight)
            new_W.bias[:d_out].copy_(old_W.bias)
            # W2 of scaffold: small random (Test K: must be nonzero for gradients)
            new_W.weight[d_out:, :].normal_(0, 0.01)

    # Output layer: copy existing; new input columns = small random
    with torch.no_grad():
        new_model.out.weight[:, :old_w].copy_(model.out.weight)
        new_model.out.bias.copy_(model.out.bias)
        new_model.out.weight[:, old_w:].normal_(0, 0.01)

    return new_model


def grow_lntanh(model, new_width, input_dim, num_classes):
    """Expand LNtanhMLP to new_width."""
    old_w = model.width
    new_model = LNtanhMLP(input_dim, new_width, num_classes, model.depth).to(DEVICE)

    for i in range(1, model.depth + 1):
        old_W  = getattr(model, f'W{i}')
        new_W  = getattr(new_model, f'W{i}')
        old_ln = getattr(model, f'ln{i}')
        new_ln = getattr(new_model, f'ln{i}')
        d_in   = old_W.weight.shape[1]
        d_out  = old_W.weight.shape[0]
        with torch.no_grad():
            new_W.weight[:d_out, :d_in].copy_(old_W.weight)
            new_W.bias[:d_out].copy_(old_W.bias)
            new_W.weight[d_out:, :].normal_(0, 0.01)
            # LN: copy existing scale/bias; new neurons init to default (1/0)
            new_ln.weight[:d_out].copy_(old_ln.weight)
            new_ln.bias[:d_out].copy_(old_ln.bias)

    with torch.no_grad():
        new_model.out.weight[:, :old_w].copy_(model.out.weight)
        new_model.out.bias.copy_(model.out.bias)
        new_model.out.weight[:, old_w:].normal_(0, 0.01)

    return new_model


def measure_output_diff(model_before, model_after, probe_x):
    """Mean absolute output difference before/after topology change."""
    model_before.eval(); model_after.eval()
    with torch.no_grad():
        out_before = model_before(probe_x)
        out_after  = model_after(probe_x)
        # Compare on shared output dimension
        diff = (out_before - out_after).abs().mean().item()
    return diff


# ─── Pruning helpers ─────────────────────────────────────────────────────────

def prune_iso(model, keep_k, input_dim, num_classes):
    """Prune IsoMLP to keep_k neurons using SV criterion on first layer."""
    W1 = getattr(model, 'W1').weight.detach()
    _, S, Vt = torch.linalg.svd(W1, full_matrices=False)
    keep_idx = torch.argsort(S, descending=True)[:keep_k].cpu().numpy()

    new_model = IsoMLP(input_dim, keep_k, num_classes, model.depth).to(DEVICE)
    # Simple approach: keep top-k neurons by SV contribution in diagonalised basis
    # For simplicity, select top-k output neurons from W1 by row norm
    row_norms = W1.norm(dim=1)
    keep_idx  = torch.argsort(row_norms, descending=True)[:keep_k].cpu()

    with torch.no_grad():
        for i in range(1, model.depth + 1):
            old_W = getattr(model, f'W{i}')
            new_W = getattr(new_model, f'W{i}')
            if i == 1:
                new_W.weight.copy_(old_W.weight[keep_idx, :])
                new_W.bias.copy_(old_W.bias[keep_idx])
            else:
                prev_idx = keep_idx
                new_W.weight.copy_(old_W.weight[keep_idx, :][:, prev_idx])
                new_W.bias.copy_(old_W.bias[keep_idx])
        new_model.out.weight.copy_(model.out.weight[:, keep_idx])
        new_model.out.bias.copy_(model.out.bias)
    return new_model


def prune_lntanh(model, keep_k, input_dim, num_classes):
    """Prune LNtanhMLP to keep_k neurons using W2-norm criterion (AJ finding)."""
    W2_norms  = model.out.weight.detach().norm(dim=0)
    keep_idx  = torch.argsort(W2_norms, descending=True)[:keep_k].cpu()

    new_model = LNtanhMLP(input_dim, keep_k, num_classes, model.depth).to(DEVICE)
    with torch.no_grad():
        for i in range(1, model.depth + 1):
            old_W  = getattr(model, f'W{i}')
            new_W  = getattr(new_model, f'W{i}')
            old_ln = getattr(model, f'ln{i}')
            new_ln = getattr(new_model, f'ln{i}')
            if i == 1:
                new_W.weight.copy_(old_W.weight[keep_idx, :])
                new_W.bias.copy_(old_W.bias[keep_idx])
            else:
                new_W.weight.copy_(old_W.weight[keep_idx, :][:, keep_idx])
                new_W.bias.copy_(old_W.bias[keep_idx])
            new_ln.weight.copy_(old_ln.weight[keep_idx])
            new_ln.bias.copy_(old_ln.bias[keep_idx])
        new_model.out.weight.copy_(model.out.weight[:, keep_idx])
        new_model.out.bias.copy_(model.out.bias)
    return new_model


def main():
    print(f'Device: {DEVICE}')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    torch.manual_seed(SEED)
    probe_x = next(iter(test_loader))[0][:512].to(DEVICE)

    # ── Step 1: Train base models ─────────────────────────────────────────────
    print(f'\n[1/2] Training Iso-{DEPTH}L (width={WIDTH})')
    torch.manual_seed(SEED)
    iso_model = IsoMLP(input_dim, WIDTH, num_classes, DEPTH).to(DEVICE)
    iso_hist  = train_model(iso_model, train_loader, test_loader, 'Iso', EPOCHS_TRAIN)
    iso_base_acc = iso_hist[-1]

    print(f'\n[2/2] Training LN+tanh-{DEPTH}L (width={WIDTH})')
    torch.manual_seed(SEED)
    ln_model  = LNtanhMLP(input_dim, WIDTH, num_classes, DEPTH).to(DEVICE)
    ln_hist   = train_model(ln_model, train_loader, test_loader, 'LN+tanh', EPOCHS_TRAIN)
    ln_base_acc = ln_hist[-1]

    print(f'\nBase accuracies: Iso={iso_base_acc:.4f}  LN+tanh={ln_base_acc:.4f}')

    # ── Step 2: Growth experiment ─────────────────────────────────────────────
    print(f'\n--- GROWTH: {WIDTH} -> {GROW_TO} neurons ---')

    iso_grown  = grow_iso(iso_model, GROW_TO, input_dim, num_classes)
    ln_grown   = grow_lntanh(ln_model, GROW_TO, input_dim, num_classes)

    iso_diff = measure_output_diff(iso_model, iso_grown, probe_x)
    ln_diff  = measure_output_diff(ln_model,  ln_grown,  probe_x)
    print(f'Output diff after growth:  Iso={iso_diff:.6f}  LN+tanh={ln_diff:.6f}')

    iso_grown_acc0 = evaluate(iso_grown,  test_loader, DEVICE)
    ln_grown_acc0  = evaluate(ln_grown,   test_loader, DEVICE)
    print(f'Accuracy before fine-tune: Iso={iso_grown_acc0:.4f}  LN+tanh={ln_grown_acc0:.4f}')

    # Fine-tune the grown models
    grow_finetune_epochs = [1, 2, 3, 5]
    iso_grow_recovery = [iso_grown_acc0]
    ln_grow_recovery  = [ln_grown_acc0]

    iso_ft = copy.deepcopy(iso_grown)
    ln_ft  = copy.deepcopy(ln_grown)

    prev_ep = 0
    for ep in grow_finetune_epochs:
        steps = ep - prev_ep
        iso_ft_hist = train_model(iso_ft, train_loader, test_loader,
                                  f'Iso-grown-ft{ep}', steps)
        ln_ft_hist  = train_model(ln_ft,  train_loader, test_loader,
                                  f'LN-grown-ft{ep}', steps)
        iso_grow_recovery.append(iso_ft_hist[-1])
        ln_grow_recovery.append(ln_ft_hist[-1])
        print(f'  After {ep} fine-tune epochs: Iso={iso_ft_hist[-1]:.4f}  '
              f'LN+tanh={ln_ft_hist[-1]:.4f}')
        prev_ep = ep

    # ── Step 3: Pruning experiment ────────────────────────────────────────────
    print(f'\n--- PRUNING: {WIDTH} -> {PRUNE_TO} neurons ---')

    iso_pruned  = prune_iso(iso_model,    PRUNE_TO, input_dim, num_classes)
    ln_pruned   = prune_lntanh(ln_model,  PRUNE_TO, input_dim, num_classes)

    iso_pruned_acc0 = evaluate(iso_pruned, test_loader, DEVICE)
    ln_pruned_acc0  = evaluate(ln_pruned,  test_loader, DEVICE)
    print(f'Accuracy before fine-tune: Iso={iso_pruned_acc0:.4f}  LN+tanh={ln_pruned_acc0:.4f}')

    prune_finetune_epochs = [1, 2, 3, 5]
    iso_prune_recovery = [iso_pruned_acc0]
    ln_prune_recovery  = [ln_pruned_acc0]

    iso_pft = copy.deepcopy(iso_pruned)
    ln_pft  = copy.deepcopy(ln_pruned)

    prev_ep = 0
    for ep in prune_finetune_epochs:
        steps = ep - prev_ep
        iso_pft_hist = train_model(iso_pft, train_loader, test_loader,
                                   f'Iso-pruned-ft{ep}', steps)
        ln_pft_hist  = train_model(ln_pft,  train_loader, test_loader,
                                   f'LN-pruned-ft{ep}', steps)
        iso_prune_recovery.append(iso_pft_hist[-1])
        ln_prune_recovery.append(ln_pft_hist[-1])
        print(f'  After {ep} fine-tune epochs: Iso={iso_pft_hist[-1]:.4f}  '
              f'LN+tanh={ln_pft_hist[-1]:.4f}')
        prev_ep = ep

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print('GROWTH SUMMARY')
    print(f'  Output diff (inertness): Iso={iso_diff:.6f}  LN+tanh={ln_diff:.6f}')
    print(f'  Iso base: {iso_base_acc:.4f}  |  LN+tanh base: {ln_base_acc:.4f}')
    ft_eps = [0] + grow_finetune_epochs
    for ep, ia, la in zip(ft_eps, iso_grow_recovery, ln_grow_recovery):
        print(f'  ft={ep}: Iso={ia:.4f} ({ia-iso_base_acc:+.4f})  '
              f'LN+tanh={la:.4f} ({la-ln_base_acc:+.4f})')

    print('\nPRUNING SUMMARY')
    print(f'  Iso base: {iso_base_acc:.4f}  |  LN+tanh base: {ln_base_acc:.4f}')
    for ep, ia, la in zip(ft_eps, iso_prune_recovery, ln_prune_recovery):
        print(f'  ft={ep}: Iso={ia:.4f} ({ia-iso_base_acc:+.4f})  '
              f'LN+tanh={la:.4f} ({la-ln_base_acc:+.4f})')

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ft_eps_x = [0] + grow_finetune_epochs

    ax = axes[0]
    ax.plot(ft_eps_x, iso_grow_recovery, 'o-', color='#1f77b4', label='Iso (diff=~0)')
    ax.plot(ft_eps_x, ln_grow_recovery,  'o-', color='#2ca02c', label=f'LN+tanh (diff={ln_diff:.3f})')
    ax.axhline(iso_base_acc, color='#1f77b4', ls=':', alpha=0.5, label=f'Iso base ({iso_base_acc:.3f})')
    ax.axhline(ln_base_acc,  color='#2ca02c', ls=':', alpha=0.5, label=f'LN+tanh base ({ln_base_acc:.3f})')
    ax.set_xlabel('Fine-tune epochs after growth')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Growth recovery: {WIDTH}->{GROW_TO} neurons')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ft_eps_x, iso_prune_recovery, 'o-', color='#1f77b4', label='Iso (SV criterion)')
    ax.plot(ft_eps_x, ln_prune_recovery,  'o-', color='#2ca02c', label='LN+tanh (W2-norm criterion)')
    ax.axhline(iso_base_acc, color='#1f77b4', ls=':', alpha=0.5)
    ax.axhline(ln_base_acc,  color='#2ca02c', ls=':', alpha=0.5)
    ax.set_xlabel('Fine-tune epochs after pruning')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Pruning recovery: {WIDTH}->{PRUNE_TO} neurons')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Test AM: Post-hoc topology on Iso vs LN+tanh', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'posthoc_topology.png'), dpi=150)
    print('\nPlot saved.')

    md = f"""# Test AM -- Post-hoc Topology: Iso vs LN+tanh

## Setup
- Iso-{DEPTH}L and LN+tanh-{DEPTH}L, width={WIDTH}, trained {EPOCHS_TRAIN} epochs
- Growth: {WIDTH}->{GROW_TO}  |  Pruning: {WIDTH}->{PRUNE_TO}
- Device: {DEVICE}

## Base accuracies
- Iso: {iso_base_acc:.4f}
- LN+tanh: {ln_base_acc:.4f}

## Growth ({WIDTH}->{GROW_TO})
Output diff after scaffold insertion: Iso={iso_diff:.6f}  LN+tanh={ln_diff:.6f}

| Fine-tune epochs | Iso | Iso delta | LN+tanh | LN+tanh delta |
|---|---|---|---|---|
{chr(10).join(f'| {ep} | {ia:.4f} | {ia-iso_base_acc:+.4f} | {la:.4f} | {la-ln_base_acc:+.4f} |' for ep, ia, la in zip(ft_eps, iso_grow_recovery, ln_grow_recovery))}

## Pruning ({WIDTH}->{PRUNE_TO})
Criterion: Iso=SV (row norm proxy), LN+tanh=W2-norm

| Fine-tune epochs | Iso | Iso delta | LN+tanh | LN+tanh delta |
|---|---|---|---|---|
{chr(10).join(f'| {ep} | {ia:.4f} | {ia-iso_base_acc:+.4f} | {la:.4f} | {la-ln_base_acc:+.4f} |' for ep, ia, la in zip(ft_eps, iso_prune_recovery, ln_prune_recovery))}

![Post-hoc topology](posthoc_topology.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w') as f:
        f.write(md)
    print('Results saved.')


if __name__ == '__main__':
    main()
