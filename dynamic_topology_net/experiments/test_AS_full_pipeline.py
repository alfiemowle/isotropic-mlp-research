"""
Test AS -- Full Integrated Pipeline (Best Known Configuration)
==============================================================
Combines every positive finding from AN-AR into a single run:

Architecture: Iso-first-4L (Iso -> LNG -> LNG -> LNG), width=128
  - Best architecture from AR (52.28% static)
  - Single Iso layer at input boundary grants topology support there

Training:
  - 100 epochs (AI showed Iso keeps improving past ep30 where LN overfits)
  - LR=0.001 (AQ showed this is better for Iso-family activations)
  - Diagonalise Iso layer every 5 epochs (AN confirmed safe, stale Adam fine)

Dynamic topology (Iso layer only):
  - Start at width=160 (overabundant by 32 neurons)
  - Prune 4 neurons every 10 epochs starting ep20, stop when width=128
  - Criterion: composite Sigma_ii * ||W2_col|| (best from U2/Z)
  - 8 prune events: ep20,30,40,50,60,70,80,90 -> 160 to 128

Baselines (all 100 epochs, LR=0.001):
  A. Static-Iso-first-4L w=128   (no topology changes)
  B. Static-Pure-LN+GELU-4L w=128 (best non-Iso architecture)
  C. Dynamic (overabundant->pruned, as above)

Questions:
  1. Does the full pipeline beat the static Iso-first-4L baseline?
  2. Does it beat Pure-LN+GELU?
  3. What is the accuracy trajectory across 100 epochs?
  4. Does interleaved diagonalise hurt training at 100 epochs?

This is the paper's full intended protocol applied to the best
architecture identified across the entire experimental suite.
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AS')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS       = 100
LR           = 0.001
BATCH        = 128
WIDTH_TARGET = 128
WIDTH_START  = 160          # overabundant by 32
SEED         = 42
DIAG_EVERY   = 5
PRUNE_EPOCHS = list(range(20, 100, 10))   # ep20,30,...,90 = 8 events x 4 = 32 pruned
PRUNE_N      = 4                          # neurons per prune event


# ── Activations & layers ──────────────────────────────────────────────────────

class IsoAct(nn.Module):
    def forward(self, x):
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.tanh(n) * x / n


# ── Dynamic Iso layer (supports diagonalise + composite pruning) ───────────────

class IsoLayer(nn.Module):
    """
    A single linear layer followed by IsoAct.
    Keeps track of current width and supports:
      - partial_diagonalise(): left-sided SVD reparameterisation
      - prune(n, W_next): remove n neurons by composite criterion
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W   = nn.Linear(in_features, out_features, bias=False)
        self.act = IsoAct()

    @property
    def width(self):
        return self.W.out_features

    def forward(self, x):
        return self.act(self.W(x))

    def partial_diagonalise(self, W_next_weight):
        """
        Left-sided SVD: W = U S Vt -> W = S Vt, W_next = W_next @ U
        Returns singular values S.
        W_next_weight: the weight tensor of the subsequent layer (modified in-place).
        """
        W = self.W.weight.data   # [out, in]
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        self.W.weight.data = S.unsqueeze(1) * Vt   # [rank, in]
        W_next_weight.data = W_next_weight.data @ U   # absorb U
        return S

    def prune(self, n_remove, W_next_weight, W_next_bias=None):
        """
        Remove n_remove neurons by composite criterion: S_i * ||W_next[:,i]||.
        Assumes partial_diagonalise was just called (S visible as W row norms).
        Returns indices removed.
        """
        # After diagonalise, W rows are S_i * Vt_i, so row norms = S_i * ||Vt_i|| = S_i (Vt rows are unit)
        S     = self.W.weight.data.norm(dim=1)          # [width]
        W2col = W_next_weight.data.norm(dim=0)          # [width]
        score = S * W2col                               # composite criterion
        # Remove n_remove lowest-scoring neurons
        keep_idx = score.argsort(descending=True)[n_remove:]   # keep top-(w-n) by score
        keep_idx = keep_idx.sort().values                       # preserve order

        # Trim W rows
        self.W.weight = nn.Parameter(self.W.weight.data[keep_idx])
        self.W.out_features = len(keep_idx)

        # Trim W_next columns
        W_next_weight.data = W_next_weight.data[:, keep_idx]
        W_next_weight = nn.Parameter(W_next_weight.data)   # re-wrap (caller must update)
        if W_next_bias is not None:
            pass   # bias dim is output of W_next, not input — unchanged

        return keep_idx


# ── Hybrid MLP: Iso-first-4L ──────────────────────────────────────────────────

class IsoFirstMLP(nn.Module):
    """
    Layer layout: IsoLayer -> LNG -> LNG -> LNG -> out
    The IsoLayer supports dynamic topology.
    The LNG layers are standard nn.Linear + LayerNorm + GELU.
    """
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        self.iso  = IsoLayer(input_dim, width)
        self.ln1  = nn.LayerNorm(width)
        self.W2   = nn.Linear(width, width)
        self.ln2  = nn.LayerNorm(width)
        self.W3   = nn.Linear(width, width)
        self.ln3  = nn.LayerNorm(width)
        self.W4   = nn.Linear(width, width)
        self.out  = nn.Linear(width, num_classes)
        self.gelu = nn.GELU()

    def forward(self, x):
        h = self.iso(x)
        h = self.gelu(self.ln1(self.W2(h)))
        h = self.gelu(self.ln2(self.W3(h)))
        h = self.gelu(self.ln3(self.W4(h)))
        return self.out(h)

    @property
    def width(self):
        return self.iso.width

    def partial_diagonalise(self):
        """Diagonalise Iso layer, absorb U into W2."""
        return self.iso.partial_diagonalise(self.W2.weight)

    def prune(self, n_remove):
        """
        Prune n_remove neurons from Iso layer using composite criterion.
        Updates all downstream parameter shapes accordingly.
        """
        keep_idx = self.iso.prune(n_remove, self.W2.weight, self.W2.bias)

        # Fix W2 input features reference (weight already updated in-place by prune())
        self.W2.in_features = len(keep_idx)

        # Fix LayerNorm ln1 (operates on W2 output, not Iso output — unchanged)
        # All subsequent layers (W3, W4, out) operate on W2/W3/W4 outputs, which
        # are still `width` dimensional (only input of W2 shrank). No changes needed.

        return keep_idx


# ── Pure LN+GELU-4L baseline ──────────────────────────────────────────────────

class PureLNGMLP(nn.Module):
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        self.W1  = nn.Linear(input_dim, width)
        self.ln1 = nn.LayerNorm(width)
        self.W2  = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.W3  = nn.Linear(width, width)
        self.ln3 = nn.LayerNorm(width)
        self.W4  = nn.Linear(width, width)
        self.ln4 = nn.LayerNorm(width)
        self.out = nn.Linear(width, num_classes)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.ln1(self.W1(x)))
        x = self.act(self.ln2(self.W2(x)))
        x = self.act(self.ln3(self.W3(x)))
        x = self.act(self.ln4(self.W4(x)))
        return self.out(x)


# ── Training utilities ────────────────────────────────────────────────────────

def train_epoch(model, opt, train_loader):
    crit = nn.CrossEntropyLoss()
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()


def make_opt(model):
    return optim.Adam(model.parameters(), lr=LR)


# ── Condition A: Static Iso-first-4L w=128 ───────────────────────────────────

def run_static_iso(train_loader, test_loader, input_dim, num_classes):
    print('\n[A] Static Iso-first-4L  w=128  100 epochs  lr=0.001')
    torch.manual_seed(SEED)
    model = IsoFirstMLP(input_dim, WIDTH_TARGET, num_classes).to(DEVICE)
    opt   = make_opt(model)
    hist  = []
    for ep in range(1, EPOCHS + 1):
        train_epoch(model, opt, train_loader)
        acc = evaluate(model, test_loader, DEVICE)
        hist.append(acc)
        if ep % 10 == 0 or ep == 1:
            print(f'  ep{ep:3d}  acc={acc:.4f}  width={model.width}')
    print(f'  Final: {hist[-1]:.4f}  Peak: {max(hist):.4f} @ ep{hist.index(max(hist))+1}')
    return hist


# ── Condition B: Static Pure-LN+GELU-4L w=128 ────────────────────────────────

def run_static_lng(train_loader, test_loader, input_dim, num_classes):
    print('\n[B] Static Pure-LN+GELU-4L  w=128  100 epochs  lr=0.001')
    torch.manual_seed(SEED)
    model = PureLNGMLP(input_dim, WIDTH_TARGET, num_classes).to(DEVICE)
    opt   = make_opt(model)
    hist  = []
    for ep in range(1, EPOCHS + 1):
        train_epoch(model, opt, train_loader)
        acc = evaluate(model, test_loader, DEVICE)
        hist.append(acc)
        if ep % 10 == 0 or ep == 1:
            print(f'  ep{ep:3d}  acc={acc:.4f}')
    print(f'  Final: {hist[-1]:.4f}  Peak: {max(hist):.4f} @ ep{hist.index(max(hist))+1}')
    return hist


# ── Condition C: Dynamic Iso-first-4L w=160->128 ─────────────────────────────

def run_dynamic(train_loader, test_loader, input_dim, num_classes):
    print(f'\n[C] Dynamic Iso-first-4L  w={WIDTH_START}->{WIDTH_TARGET}  100 epochs  lr=0.001')
    print(f'    Diag every {DIAG_EVERY} epochs, prune {PRUNE_N} neurons at {PRUNE_EPOCHS}')
    torch.manual_seed(SEED)
    model = IsoFirstMLP(input_dim, WIDTH_START, num_classes).to(DEVICE)
    opt   = make_opt(model)
    hist  = []
    log   = []

    for ep in range(1, EPOCHS + 1):
        train_epoch(model, opt, train_loader)

        # Diagonalise every DIAG_EVERY epochs
        if ep % DIAG_EVERY == 0:
            S = model.partial_diagonalise()
            # No Adam reset (AN confirmed stale Adam is fine / reset hurts)

        # Prune if scheduled and haven't reached target yet
        if ep in PRUNE_EPOCHS and model.width > WIDTH_TARGET:
            n = min(PRUNE_N, model.width - WIDTH_TARGET)
            # Ensure diagonalised first (needed for composite criterion)
            if ep % DIAG_EVERY != 0:   # didn't just diagonalise
                model.partial_diagonalise()
            model.prune(n)
            # Rebuild Adam (parameter shapes changed)
            opt = make_opt(model)
            msg = f'  Pruned {n} neurons at ep{ep}, new width={model.width}'
            print(msg)
            log.append(msg)

        acc = evaluate(model, test_loader, DEVICE)
        hist.append(acc)
        if ep % 10 == 0 or ep == 1:
            print(f'  ep{ep:3d}  acc={acc:.4f}  width={model.width}')

    print(f'  Final: {hist[-1]:.4f}  Peak: {max(hist):.4f} @ ep{hist.index(max(hist))+1}')
    print(f'  Final width: {model.width}')
    return hist, log


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f'Device: {DEVICE}')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)
    print(f'input_dim={input_dim}  num_classes={num_classes}')

    hist_A = run_static_iso(train_loader, test_loader, input_dim, num_classes)
    hist_B = run_static_lng(train_loader, test_loader, input_dim, num_classes)
    hist_C, prune_log = run_dynamic(train_loader, test_loader, input_dim, num_classes)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print('SUMMARY')
    for label, hist in [('A-Static-Iso-first', hist_A),
                        ('B-Static-LNG',       hist_B),
                        ('C-Dynamic',          hist_C)]:
        peak     = max(hist)
        peak_ep  = hist.index(peak) + 1
        final    = hist[-1]
        ep30     = hist[29]
        ep50     = hist[49]
        print(f'  {label:<24}: ep30={ep30:.4f}  ep50={ep50:.4f}  '
              f'ep100={final:.4f}  peak={peak:.4f}@ep{peak_ep}')

    print(f'\nKey comparisons at ep100:')
    print(f'  C vs A (dynamic vs static Iso-first): {hist_C[-1]-hist_A[-1]:+.4f}')
    print(f'  C vs B (dynamic vs Pure-LNG):         {hist_C[-1]-hist_B[-1]:+.4f}')
    print(f'  A vs B (static Iso-first vs Pure-LNG):{hist_A[-1]-hist_B[-1]:+.4f}')

    print(f'\nPrune events:')
    for msg in prune_log:
        print(f' {msg}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = list(range(1, EPOCHS + 1))

    ax = axes[0]
    ax.plot(epochs_range, hist_A, color='#1f77b4',
            label=f'A-Static-Iso-first (peak={max(hist_A):.3f}@ep{hist_A.index(max(hist_A))+1})')
    ax.plot(epochs_range, hist_B, color='#2ca02c',
            label=f'B-Static-LNG (peak={max(hist_B):.3f}@ep{hist_B.index(max(hist_B))+1})')
    ax.plot(epochs_range, hist_C, color='#d62728',
            label=f'C-Dynamic (peak={max(hist_C):.3f}@ep{hist_C.index(max(hist_C))+1})')
    for ep in PRUNE_EPOCHS:
        ax.axvline(ep, color='#d62728', ls=':', alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('Full pipeline: all three conditions (100 epochs)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs_range, hist_A, color='#1f77b4', label='A-Static-Iso-first')
    ax.plot(epochs_range, hist_B, color='#2ca02c', label='B-Static-LNG')
    ax.plot(epochs_range, hist_C, color='#d62728', label='C-Dynamic')
    ax.set_xlim(50, 100)
    ymin = min(min(hist_A[49:]), min(hist_B[49:]), min(hist_C[49:])) - 0.01
    ymax = max(max(hist_A[49:]), max(hist_B[49:]), max(hist_C[49:])) + 0.01
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('Late-training detail (ep50-100)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Test AS: Full Integrated Pipeline (Best Known Config)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'full_pipeline.png'), dpi=150)
    print('\nPlot saved.')

    # ── AR reference (ep30 static from prior run) ─────────────────────────────
    ar_static = 0.5228   # Iso-first-4L w=128 static 30ep from AR

    rows = []
    for label, hist in [('A-Static-Iso-first', hist_A),
                        ('B-Static-LNG',       hist_B),
                        ('C-Dynamic',          hist_C)]:
        ep30_  = hist[29]
        ep100_ = hist[-1]
        peak_  = max(hist)
        peak_e = hist.index(peak_) + 1
        rows.append(f'| {label} | {ep30_:.4f} | {ep100_:.4f} | {peak_:.4f} | ep{peak_e} |')

    md = f"""# Test AS -- Full Integrated Pipeline

## Setup
- Architecture: Iso-first-4L (Iso -> LNG -> LNG -> LNG)
- Width: {WIDTH_START} -> {WIDTH_TARGET} (dynamic), {WIDTH_TARGET} (static baselines)
- Epochs: {EPOCHS}, LR: {LR}, seed: {SEED}
- Device: {DEVICE}
- Diagonalise every {DIAG_EVERY} epochs (no Adam reset -- AN confirmed harmful)
- Prune {PRUNE_N} neurons at epochs {PRUNE_EPOCHS}
- Criterion: composite Sigma_ii x ||W2-col|| (best from U2/Z)

## Combines findings from
- AR: Iso-first-4L is best architecture at w=128
- AN: Interleaved diagonalise safe, do NOT reset Adam
- AQ: lr=0.001 better than lr=0.08 for Iso-family
- AI: 100 epochs reveals long-term stability; Iso overtakes LN+tanh
- U2/Z: composite criterion is best pruning measure

## Reference: AR ep30 static Iso-first-4L = {ar_static:.4f}

## Results

| Condition | ep30 | ep100 | Peak | Peak epoch |
|---|---|---|---|---|
{chr(10).join(rows)}

## Key comparisons (ep100)
- C vs A (dynamic vs static Iso-first): {hist_C[-1]-hist_A[-1]:+.4f}
- C vs B (dynamic vs Pure-LNG):         {hist_C[-1]-hist_B[-1]:+.4f}
- A vs B (static Iso-first vs Pure-LNG):{hist_A[-1]-hist_B[-1]:+.4f}

## Prune log
{chr(10).join(prune_log)}

![Full pipeline](full_pipeline.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w') as f:
        f.write(md)
    print('Results saved.')


if __name__ == '__main__':
    main()
