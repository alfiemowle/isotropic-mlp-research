"""
Test AP -- Chained SVD Pruning (Proper Multi-Layer Implementation)
==================================================================
Test AM had a bug: multi-layer pruning used W1 row norms as a proxy for
importance, then applied the SAME keep_idx to all layers. This is wrong.

Correct approach for a 3-layer Iso network (W1, W2, W3, out):
  - Prune neuron k in the hidden layer between W1 and W2:
      1. partial_diagonalise on W1 (so SVs are explicit)
      2. Drop row k of W1 (reduces W1 from [h, in] to [h-1, in])
      3. Drop col k of W2 (reduces W2 from [h, h] to [h, h-1])
         Wait — this is NOT the right framing for multi-layer.

For IsotropicMLP (raw nn.Parameter, single hidden layer):
  partial_diagonalise + prune_neuron is already correctly implemented.

For a 3-layer inline MLP (W1→act→W2→act→W3→out):
  Pruning neuron k at layer boundary (W1 output / W2 input):
    1. SVD-diagonalise W1: exposes singular values as W1's row scales
    2. Identify least-important neuron k (smallest SV or smallest activation norm)
    3. Drop row k from W1 (output dimension k removed)
    4. Drop col k from W2 (input dimension k removed)
    5. Bias b1 has no impact (Iso has no bias, or if it does, drop element k)

  Pruning neuron j at layer boundary (W2 output / W3 input):
    1. SVD-diagonalise W2: drop row j from W2
    2. Drop col j from W3

This test implements proper chained pruning for a 2-layer Iso network
(W1, W2, W3=out) and a 3-layer Iso network (W1, W2, W3, out).

Conditions:
  1. 2L-Iso baseline: width=32, train 30ep, evaluate
  2. 2L-Iso prune-layer1: after 30ep, prune layer1 boundary (32→24)
  3. 2L-Iso prune-layer2: after 30ep, prune layer2 boundary (32→24)
  4. 3L-Iso prune-layer1: after 30ep, prune first boundary (32→24)
  5. 3L-Iso prune-layer2: after 30ep, prune second boundary (32→24)
  6. 3L-Iso prune-both:   after 30ep, prune both boundaries (32→24→16)

For each pruning condition: evaluate before prune, after prune, and after
5 fine-tune epochs with fresh Adam.

Question: Does proper chained SVD pruning preserve accuracy better than
AM's row-norm proxy? What is the accuracy cost per neuron pruned?
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AP')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS   = 30
LR       = 0.08
BATCH    = 128
WIDTH    = 32
PRUNE_TO = 24
SEED     = 42


# ── Isotropic activation ──────────────────────────────────────────────────────

class IsoAct(nn.Module):
    def forward(self, x):
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.tanh(n) * x / n


# ── Flexible IsoMLP (arbitrary depth, stores layers as ModuleList) ────────────

class IsoMLP(nn.Module):
    def __init__(self, input_dim, widths, num_classes):
        """
        widths: list of hidden widths, e.g. [32, 32] for 2 hidden layers.
        Creates len(widths) hidden linear layers + 1 output layer.
        """
        super().__init__()
        dims = [input_dim] + widths
        self.hidden = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1], bias=False) for i in range(len(widths))
        ])
        self.out = nn.Linear(widths[-1], num_classes, bias=True)
        self.act = IsoAct()

    @property
    def depth(self):
        return len(self.hidden)

    def forward(self, x):
        h = x
        for layer in self.hidden:
            h = self.act(layer(h))
        return self.out(h)

    def svd_diagonalise_layer(self, idx):
        """
        Diagonalise hidden layer idx using left-sided SVD.
        W[idx] = U Σ Vt  →  W[idx] = Σ Vt,  W[idx+1] = W[idx+1] @ U
        If idx is the last hidden layer, the output layer absorbs U.
        """
        W = self.hidden[idx].weight.data   # shape [out, in]
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        # Set W[idx] = diag(S) @ Vt  (shape [min(out,in), in])
        self.hidden[idx].weight.data = S.unsqueeze(1) * Vt

        # The next layer absorbs U on the right (its input basis changes)
        if idx + 1 < len(self.hidden):
            self.hidden[idx + 1].weight.data = self.hidden[idx + 1].weight.data @ U
        else:
            self.out.weight.data = self.out.weight.data @ U

        return S  # singular values, sorted descending

    def prune_boundary(self, boundary_idx, keep_mask):
        """
        Prune the output dimension of hidden[boundary_idx] and the
        corresponding input dimension of hidden[boundary_idx+1] (or out).

        boundary_idx: which hidden layer's OUTPUT to prune.
        keep_mask: boolean tensor of shape [width], True = keep.
        """
        keep = keep_mask.to(self.hidden[boundary_idx].weight.device)

        # Trim output rows of current layer
        W_cur = self.hidden[boundary_idx].weight.data   # [out, in]
        self.hidden[boundary_idx].weight = nn.Parameter(W_cur[keep])
        self.hidden[boundary_idx].out_features = keep.sum().item()

        # Trim input cols of next layer
        if boundary_idx + 1 < len(self.hidden):
            W_next = self.hidden[boundary_idx + 1].weight.data  # [out2, in2]
            self.hidden[boundary_idx + 1].weight = nn.Parameter(W_next[:, keep])
            self.hidden[boundary_idx + 1].in_features = keep.sum().item()
        else:
            W_out = self.out.weight.data  # [classes, in]
            self.out.weight = nn.Parameter(W_out[:, keep])
            self.out.in_features = keep.sum().item()


def make_model(depth, input_dim, num_classes, width=WIDTH):
    torch.manual_seed(SEED)
    widths = [width] * depth
    return IsoMLP(input_dim, widths, num_classes).to(DEVICE)


def train(model, train_loader, test_loader, epochs, reset_opt=True, opt=None):
    if reset_opt or opt is None:
        opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    hist = []
    model.train()
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        acc = evaluate(model, test_loader, DEVICE)
        hist.append(acc)
    return hist, opt


def prune_layer(model, boundary_idx, prune_to):
    """SVD-diagonalise boundary_idx then prune to prune_to neurons."""
    S = model.svd_diagonalise_layer(boundary_idx)
    current_width = len(S)
    n_keep = min(prune_to, current_width)
    # Keep top-n_keep singular values (already sorted descending by SVD)
    keep_mask = torch.zeros(current_width, dtype=torch.bool)
    keep_mask[:n_keep] = True
    model.prune_boundary(boundary_idx, keep_mask)
    return S


def main():
    print(f'Device: {DEVICE}')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    results = {}
    FINETUNE = 5

    # ── 1. 2L-Iso baseline ────────────────────────────────────────────────────
    print('\n[1] 2L-Iso baseline: train 30ep, no pruning')
    model = make_model(2, input_dim, num_classes)
    hist, _ = train(model, train_loader, test_loader, EPOCHS)
    results['2L-baseline'] = hist
    print(f'  Final: {hist[-1]:.4f}  Peak: {max(hist):.4f}')

    # ── 2. 2L-Iso prune layer-1 boundary ─────────────────────────────────────
    print('\n[2] 2L-Iso prune-L1: train 30ep, SVD prune boundary0 (32->24), 5ep ft')
    model = make_model(2, input_dim, num_classes)
    hist_pre, _ = train(model, train_loader, test_loader, EPOCHS)
    acc_before = hist_pre[-1]
    S = prune_layer(model, 0, PRUNE_TO)
    acc_after = evaluate(model, test_loader, DEVICE)
    hist_ft, _ = train(model, train_loader, test_loader, FINETUNE)
    results['2L-prune-L1'] = hist_pre + hist_ft
    print(f'  SVs: {S[:5].tolist()} ...')
    print(f'  Before: {acc_before:.4f}  After prune: {acc_after:.4f}  '
          f'After {FINETUNE}ep ft: {hist_ft[-1]:.4f}  '
          f'Drop: {acc_before-acc_after:.4f}  Recovery: {hist_ft[-1]-acc_after:.4f}')

    # ── 3. 2L-Iso prune layer-2 boundary ─────────────────────────────────────
    print('\n[3] 2L-Iso prune-L2: train 30ep, SVD prune boundary1 (32->24), 5ep ft')
    model = make_model(2, input_dim, num_classes)
    hist_pre, _ = train(model, train_loader, test_loader, EPOCHS)
    acc_before = hist_pre[-1]
    S = prune_layer(model, 1, PRUNE_TO)
    acc_after = evaluate(model, test_loader, DEVICE)
    hist_ft, _ = train(model, train_loader, test_loader, FINETUNE)
    results['2L-prune-L2'] = hist_pre + hist_ft
    print(f'  SVs: {S[:5].tolist()} ...')
    print(f'  Before: {acc_before:.4f}  After prune: {acc_after:.4f}  '
          f'After {FINETUNE}ep ft: {hist_ft[-1]:.4f}  '
          f'Drop: {acc_before-acc_after:.4f}  Recovery: {hist_ft[-1]-acc_after:.4f}')

    # ── 4. 3L-Iso prune layer-1 boundary ─────────────────────────────────────
    print('\n[4] 3L-Iso prune-L1: train 30ep, SVD prune boundary0 (32->24), 5ep ft')
    model = make_model(3, input_dim, num_classes)
    hist_pre, _ = train(model, train_loader, test_loader, EPOCHS)
    acc_before = hist_pre[-1]
    S = prune_layer(model, 0, PRUNE_TO)
    acc_after = evaluate(model, test_loader, DEVICE)
    hist_ft, _ = train(model, train_loader, test_loader, FINETUNE)
    results['3L-prune-L1'] = hist_pre + hist_ft
    print(f'  Before: {acc_before:.4f}  After prune: {acc_after:.4f}  '
          f'After {FINETUNE}ep ft: {hist_ft[-1]:.4f}')

    # ── 5. 3L-Iso prune layer-2 boundary ─────────────────────────────────────
    print('\n[5] 3L-Iso prune-L2: train 30ep, SVD prune boundary1 (32->24), 5ep ft')
    model = make_model(3, input_dim, num_classes)
    hist_pre, _ = train(model, train_loader, test_loader, EPOCHS)
    acc_before = hist_pre[-1]
    S = prune_layer(model, 1, PRUNE_TO)
    acc_after = evaluate(model, test_loader, DEVICE)
    hist_ft, _ = train(model, train_loader, test_loader, FINETUNE)
    results['3L-prune-L2'] = hist_pre + hist_ft
    print(f'  Before: {acc_before:.4f}  After prune: {acc_after:.4f}  '
          f'After {FINETUNE}ep ft: {hist_ft[-1]:.4f}')

    # ── 6. 3L-Iso prune both boundaries ──────────────────────────────────────
    print('\n[6] 3L-Iso prune-both: train 30ep, prune B0 (32->24) then B1 (32->24), 5ep ft')
    model = make_model(3, input_dim, num_classes)
    hist_pre, _ = train(model, train_loader, test_loader, EPOCHS)
    acc_before = hist_pre[-1]
    prune_layer(model, 0, PRUNE_TO)
    prune_layer(model, 1, PRUNE_TO)
    acc_after = evaluate(model, test_loader, DEVICE)
    hist_ft, _ = train(model, train_loader, test_loader, FINETUNE)
    results['3L-prune-both'] = hist_pre + hist_ft
    print(f'  Before: {acc_before:.4f}  After prune: {acc_after:.4f}  '
          f'After {FINETUNE}ep ft: {hist_ft[-1]:.4f}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print('SUMMARY')
    print(f'{"Condition":<22}  {"Pre-prune":>9}  {"Post-prune":>10}  {"Post-ft":>8}')

    # Re-evaluate pre-prune as hist[EPOCHS-1]
    baseline2 = results['2L-baseline'][-1]
    print(f'  {"2L-baseline":<20}  {baseline2:.4f}    {"N/A":>10}  {"N/A":>8}')
    for tag in ['2L-prune-L1', '2L-prune-L2', '3L-prune-L1', '3L-prune-L2', '3L-prune-both']:
        hist = results[tag]
        pre_acc  = hist[EPOCHS - 1]
        post_ft  = hist[-1]
        print(f'  {tag:<20}  {pre_acc:.4f}    {"(see above)":>10}  {post_ft:.4f}')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    colors = {
        '2L-baseline':   '#1f77b4',
        '2L-prune-L1':   '#2ca02c',
        '2L-prune-L2':   '#ff7f0e',
        '3L-prune-L1':   '#d62728',
        '3L-prune-L2':   '#9467bd',
        '3L-prune-both': '#8c564b',
    }
    for tag, hist in results.items():
        eps = list(range(1, len(hist) + 1))
        ax.plot(eps, hist, color=colors[tag], label=tag)
    ax.axvline(EPOCHS, color='black', ls=':', alpha=0.4, label='prune point')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('Chained SVD pruning (full curves)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Bar chart: pre-prune, post-prune (from log above), post-ft
    tags_pruned = ['2L-prune-L1', '2L-prune-L2', '3L-prune-L1', '3L-prune-L2', '3L-prune-both']
    pre_accs  = [results[t][EPOCHS-1] for t in tags_pruned]
    post_accs = [results[t][-1]       for t in tags_pruned]
    x = np.arange(len(tags_pruned))
    ax.bar(x - 0.2, pre_accs,  0.35, label='Pre-prune (ep30)', color='#1f77b4', alpha=0.8)
    ax.bar(x + 0.2, post_accs, 0.35, label=f'Post-ft ({FINETUNE}ep)',  color='#ff7f0e', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(tags_pruned, rotation=20, ha='right', fontsize=7)
    ax.set_ylabel('Accuracy'); ax.set_title('Pre-prune vs post-fine-tune')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Test AP: Chained SVD Pruning', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'chained_pruning.png'), dpi=150)
    print('\nPlot saved.')

    rows = []
    for tag, hist in results.items():
        pre  = hist[EPOCHS - 1]
        post = hist[-1]
        rows.append(f'| {tag} | {pre:.4f} | {post:.4f} | {post-pre:+.4f} |')

    md = f"""# Test AP -- Chained SVD Pruning

## Setup
- IsoMLP (2L or 3L), width={WIDTH}, {EPOCHS} epochs training + {FINETUNE}ep fine-tune
- Pruning: SVD-diagonalise boundary then drop bottom {WIDTH-PRUNE_TO} singular values
- Device: {DEVICE}

## Question
Does proper chained SVD pruning (per-boundary SVD, drop by singular value)
preserve accuracy better than AM's row-norm proxy?

## Results

| Condition | Pre-prune (ep{EPOCHS}) | Post-ft (ep{EPOCHS+FINETUNE}) | Delta |
|---|---|---|---|
{chr(10).join(rows)}

## Method
For each boundary idx:
  1. SVD-diagonalise layer[idx]: W = U S Vt -> W = S Vt, W_next = W_next @ U
  2. Keep top-{PRUNE_TO} rows (largest singular values)
  3. Drop corresponding input columns in next layer
  4. Re-initialise Adam from scratch (fresh parameter references)

![Chained pruning](chained_pruning.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w') as f:
        f.write(md)
    print('Results saved.')


if __name__ == '__main__':
    main()
