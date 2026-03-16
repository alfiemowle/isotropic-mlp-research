"""
Test X -- Rank Regularization: Can We Fix Baseline Depth Collapse?
==================================================================
Test R showed that baseline networks at depth collapse to a 3-4 dimensional
representational subspace (effective rank ~3-4 vs iso's ~14). The mechanism
is structural: elementwise activations don't prevent neurons from aligning.

If representational collapse is THE cause of baseline depth failure, then
a regularizer that directly penalises low effective rank should recover
performance. If baselines still fail even with rank regularization, the
cause must be something else (e.g. gradient instability, inherent
expressibility limitations).

Regularization: add a penalty that encourages the hidden representation
to be spread across many directions. We use the spectral isotropy loss:

  L_rank = || C / tr(C) - I/d ||_F^2

where C = H^T H / N is the empirical covariance of hidden activations H
(N x d), and I/d is the target isotropic covariance. This penalises
representation collapse without changing architecture.

Models tested:
  - Baseline-1L, 2L, 3L: no reg (reference)
  - Baseline-1L, 2L, 3L: + rank regularization (lambda = 0.01, 0.1)
  - Iso-1L, 2L, 3L: no reg (upper bound)

Width=24, 24 epochs, batch=128, seed=42.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_topology_net.core import (
    IsotropicMLP, BaselineMLP, DeepIsotropicMLP, DeepBaselineMLP,
    IsotropicMLP3L, BaselineMLP3L, load_cifar10
)
from dynamic_topology_net.core.train_utils import evaluate, make_optimizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_X')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED   = 42
EPOCHS = 24
LR     = 0.08
BATCH  = 128
WIDTH  = 24
DEVICE = torch.device('cpu')
REG_LAMBDAS = [0.01, 0.1]


def spectral_isotropy_loss(H):
    """
    Penalty for representational collapse.
    H: (N, d) hidden representation matrix.
    Returns scalar loss encouraging C = H^T H / N to be proportional to I.
    """
    N, d = H.shape
    if N < 2 or d < 2:
        return torch.tensor(0.0, requires_grad=True)
    C = H.T @ H / N                    # (d, d) empirical covariance
    tr = C.diagonal().sum().clamp(min=1e-8)
    C_norm = C / tr                    # normalised covariance
    target = torch.eye(d, device=H.device) / d
    return (C_norm - target).pow(2).sum()


def get_hidden_repr(model, x):
    """
    Extract the first hidden layer representation (after activation).
    Works for all model types by hooking the first activation.
    """
    activations = {}

    def hook_fn(module, inp, out):
        activations['hidden'] = out

    # Find the first activation module
    handles = []
    for module in model.modules():
        if isinstance(module, (nn.Tanh,)) and not handles:
            handles.append(module.register_forward_hook(hook_fn))

    _ = model(x)

    for h in handles:
        h.remove()

    return activations.get('hidden', None)


def train_with_rank_reg(model, model_label, train_loader, test_loader,
                        epochs, lr, device, reg_lambda=0.0):
    """Train with optional rank regularization on first hidden layer."""
    opt  = make_optimizer(model, lr)
    crit = nn.CrossEntropyLoss()
    history = {'test_acc': [], 'eff_rank': []}

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            # Forward and task loss
            logits = model(x)
            loss = crit(logits, y)

            # Rank regularization
            if reg_lambda > 0:
                H = get_hidden_repr(model, x)
                if H is not None:
                    loss = loss + reg_lambda * spectral_isotropy_loss(H)

            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            acc = evaluate(model, test_loader, device)
            history['test_acc'].append(acc)

            # Track effective rank of first hidden layer
            test_x, _ = next(iter(test_loader))
            test_x = test_x[:512].to(device)
            H = get_hidden_repr(model, test_x)
            if H is not None:
                svs = torch.linalg.svdvals(H)
                p = svs.pow(2); p = p / p.sum().clamp(min=1e-10)
                p = p.clamp(min=1e-10)
                eff_rank = float(np.exp(-(p * p.log()).sum().item()))
                history['eff_rank'].append(eff_rank)
            else:
                history['eff_rank'].append(float('nan'))

        if epoch % 6 == 0 or epoch == 1:
            er = history['eff_rank'][-1]
            print(f"  [{model_label}] Epoch {epoch:2d}/{epochs}  "
                  f"acc={acc:.3f}  eff_rank={er:.2f}")

    return history


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}, Width={WIDTH}, Epochs={EPOCHS}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    configs = []

    # Iso reference (no reg needed)
    for label, maker in [
        ('Iso-1L', lambda: IsotropicMLP(input_dim, WIDTH, num_classes)),
        ('Iso-2L', lambda: DeepIsotropicMLP(input_dim, WIDTH, num_classes)),
        ('Iso-3L', lambda: IsotropicMLP3L(input_dim, WIDTH, num_classes)),
    ]:
        configs.append((label, maker, 0.0))

    # Baseline without reg
    for label, maker in [
        ('Base-1L', lambda: BaselineMLP(input_dim, WIDTH, num_classes)),
        ('Base-2L', lambda: DeepBaselineMLP(input_dim, WIDTH, num_classes)),
        ('Base-3L', lambda: BaselineMLP3L(input_dim, WIDTH, num_classes)),
    ]:
        configs.append((label, maker, 0.0))

    # Baseline with rank reg
    for lam in REG_LAMBDAS:
        for label, maker in [
            (f'Base-1L+reg{lam}', lambda: BaselineMLP(input_dim, WIDTH, num_classes)),
            (f'Base-2L+reg{lam}', lambda: DeepBaselineMLP(input_dim, WIDTH, num_classes)),
            (f'Base-3L+reg{lam}', lambda: BaselineMLP3L(input_dim, WIDTH, num_classes)),
        ]:
            configs.append((label, maker, lam))

    all_history = {}

    for i, (label, maker, lam) in enumerate(configs):
        print(f"\n{'='*55}")
        print(f"[{i+1}/{len(configs)}] {label}  (lambda={lam})")
        print(f"{'='*55}")
        torch.manual_seed(SEED)
        model = maker().to(DEVICE)
        history = train_with_rank_reg(
            model, label, train_loader, test_loader,
            EPOCHS, LR, DEVICE, reg_lambda=lam)
        all_history[label] = history

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*65}")
    print("FINAL ACCURACY SUMMARY")
    print(f"{'='*65}")
    print(f"{'Model':>22}  {'Acc':>7}  {'Eff rank':>10}")
    for label, h in all_history.items():
        acc = h['test_acc'][-1]
        er  = h['eff_rank'][-1]
        print(f"  {label:>20}  {acc:>7.4f}  {er:>10.2f}")

    # =========================================================================
    # Plot
    # =========================================================================
    epochs_x = list(range(1, EPOCHS + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colours
    iso_col   = {'Iso-1L': '#fee0b6', 'Iso-2L': '#f97a0a', 'Iso-3L': '#7f3a00'}
    base_col  = {'Base-1L': '#d0e8ff', 'Base-2L': '#3a8fd1', 'Base-3L': '#003b6e'}
    reg1_col  = {'Base-1L+reg0.01': '#c8f0c8', 'Base-2L+reg0.01': '#28a028',
                 'Base-3L+reg0.01': '#005000'}
    reg2_col  = {'Base-1L+reg0.1':  '#f0c8f0', 'Base-2L+reg0.1':  '#a020a0',
                 'Base-3L+reg0.1':  '#400040'}
    all_cols  = {**iso_col, **base_col, **reg1_col, **reg2_col}

    for ax_i, metric in enumerate(['test_acc', 'eff_rank']):
        ax = axes[ax_i]
        for label, h in all_history.items():
            vals = h[metric]
            if not any(np.isnan(v) for v in vals):
                ls = '--' if 'reg' in label else '-'
                lw = 1.2 if 'Base' in label else 1.8
                ax.plot(epochs_x, vals, ls, label=label,
                        color=all_cols.get(label, 'gray'),
                        linewidth=lw, alpha=0.85)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy' if metric == 'test_acc' else 'Effective Rank')
        ax.set_title('Test Accuracy' if metric == 'test_acc'
                     else 'Effective Rank of First Hidden Layer')
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    if 'eff_rank' in ['eff_rank']:
        axes[1].axhline(WIDTH, linestyle=':', color='gray', alpha=0.4,
                        label=f'Max={WIDTH}')

    plt.suptitle('Rank Regularization: Can It Fix Baseline Depth Collapse?', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'rank_regularization.png'), dpi=150)
    print("\nPlot saved to results/test_X/rank_regularization.png")

    # =========================================================================
    # Save results.md
    # =========================================================================
    base_3l_no_reg  = all_history['Base-3L']['test_acc'][-1]
    base_3l_reg01   = all_history.get('Base-3L+reg0.01', {}).get('test_acc', [float('nan')])[-1]
    base_3l_reg1    = all_history.get('Base-3L+reg0.1',  {}).get('test_acc', [float('nan')])[-1]
    iso_3l_acc      = all_history['Iso-3L']['test_acc'][-1]
    best_reg_base3  = max(base_3l_reg01, base_3l_reg1)
    gap_closed      = (best_reg_base3 - base_3l_no_reg) / (iso_3l_acc - base_3l_no_reg + 1e-8)

    if gap_closed > 0.5:
        verdict = (f"Rank regularization closes {gap_closed*100:.0f}% of the iso-baseline gap "
                   f"at 3L. Representational collapse is the primary cause of baseline depth "
                   f"failure — it can be directly mitigated by encouraging representation diversity.")
    elif gap_closed > 0.2:
        verdict = (f"Rank regularization partially closes the gap ({gap_closed*100:.0f}%), "
                   f"suggesting representational collapse contributes to baseline failure "
                   f"but is not the only factor.")
    else:
        verdict = (f"Rank regularization has minimal effect ({gap_closed*100:.0f}% gap closed). "
                   f"Representational collapse may be a symptom rather than the root cause, "
                   f"or the regularizer is insufficient to overcome the structural limitation.")

    rows = '\n'.join(
        f"| {label} | {h['test_acc'][-1]*100:.2f}% | "
        f"{h['eff_rank'][-1]:.2f} |"
        for label, h in all_history.items()
    )
    results_text = f"""# Test X -- Rank Regularization

## Setup
- Width: {WIDTH}, Epochs: {EPOCHS}, lr={LR}, batch={BATCH}, seed={SEED}
- Regularizer: spectral isotropy loss on first hidden layer
  L_rank = ||C/tr(C) - I/d||_F^2  where C = H^T H / N
- Lambda values tested: {REG_LAMBDAS}

## Question
If representational collapse (Test R) is THE cause of baseline depth failure,
can a rank-encouraging regularizer fix it?

## Results

| Model | Final Acc | Final Eff Rank |
|---|---|---|
{rows}

## Key Numbers
- Base-3L (no reg): {base_3l_no_reg*100:.2f}%
- Base-3L + reg 0.01: {base_3l_reg01*100:.2f}%
- Base-3L + reg 0.1: {base_3l_reg1*100:.2f}%
- Iso-3L (upper bound): {iso_3l_acc*100:.2f}%
- Gap closed by best regularizer: {gap_closed*100:.0f}%

## Verdict
{verdict}

## Connection to Test R
Test R found Base-3L effective rank stuck at ~3-4, while Iso-3L grows
to ~14.8. This test asks whether directly regularizing for rank diversity
can bridge that gap.

![Rank regularization results](rank_regularization.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print("Results saved to results/test_X/results.md")


if __name__ == '__main__':
    main()
