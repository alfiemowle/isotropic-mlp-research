"""
Test Z -- Gradient-Based Pruning Criterion
==========================================
The paper's footnote 5 explicitly suggests an alternative to SV thresholding:

  "one may combine approaches and consider the gradient with respect to the
   diagonalised singular value as a threshold for pruning"

This is the loss gradient dL/dSigma_ii for each diagonal entry Sigma_ii in
the diagonalised weight matrix. Intuitively, if the gradient is near zero,
changing that singular value won't affect the loss -- the neuron is truly
inconsequential.

Compared to SV alone, the gradient criterion is data-dependent: it measures
CURRENT importance given the loss landscape, not just representational scale.

We test four criteria in the properly diagonalised basis:
  (A) SV: Sigma_ii (paper's primary criterion, validated in Test U2)
  (B) |dL/dSigma_ii|: gradient magnitude w.r.t. each SV
  (C) W2' col norm: output connection strength (best in Test U2)
  (D) Composite: SV x W2' col norm (also from Test U2)

Leave-one-out comparison: which criterion best predicts acc_drop
from removing each neuron?

Width=24, 24 epochs, batch=128, gradient computed over 4 batches.
Seeds: [42, 123], Device: CPU
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from dynamic_topology_net.core import IsotropicMLP, load_cifar10
from dynamic_topology_net.core.train_utils import train_model, evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_Z')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED         = 42
EPOCHS       = 24
LR           = 0.08
BATCH        = 128
WIDTH        = 24
SEEDS        = [42, 123]
GRAD_BATCHES = 4     # number of batches to average gradient over
DEVICE       = torch.device('cpu')


def diagonalise(model):
    """Partial left-diagonalisation. Returns model_d and svs array."""
    model_d = copy.deepcopy(model)
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(model_d.W1.data, full_matrices=False)
        model_d.W1.data = torch.diag(S) @ Vh
        model_d.W2.data = model_d.W2.data @ U
        model_d.b1.data = U.T @ model_d.b1.data
    return model_d, S.detach().numpy()


def compute_sv_gradients(model_d, train_loader, n_batches, device):
    """
    Compute |dL/dSigma_ii| for each neuron i in the diagonalised model.
    Sigma_ii = W1[i, i] in the diagonalised basis (diagonal entry).
    We compute the gradient of the loss w.r.t. W1[i,i] averaged over n_batches.
    """
    crit = nn.CrossEntropyLoss()
    grad_accum = torch.zeros(model_d.width)

    loader_iter = iter(train_loader)
    for _ in range(n_batches):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)

        # Zero grads, forward, backward
        model_d.zero_grad()
        loss = crit(model_d(x), y)
        loss.backward()

        # dL/dW1[i,i] for diagonal entries
        if model_d.W1.grad is not None:
            diag_grads = torch.diagonal(model_d.W1.grad).abs()
            grad_accum += diag_grads.cpu().detach()

    return (grad_accum / n_batches).numpy()


def leave_one_out(model_d, test_loader, device):
    """Zero each neuron j's row/col in diagonalised basis, measure acc_drop."""
    baseline_acc = evaluate(model_d, test_loader, device)
    width = model_d.width
    acc_drops = np.zeros(width)
    with torch.no_grad():
        for j in range(width):
            w1r = model_d.W1.data[j].clone()
            w2c = model_d.W2.data[:, j].clone()
            b1j = model_d.b1.data[j].clone()
            model_d.W1.data[j] = 0
            model_d.W2.data[:, j] = 0
            model_d.b1.data[j] = 0
            acc_drops[j] = baseline_acc - evaluate(model_d, test_loader, device)
            model_d.W1.data[j] = w1r
            model_d.W2.data[:, j] = w2c
            model_d.b1.data[j] = b1j
    return baseline_acc, acc_drops


def pearson_r(x, y):
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}, Width={WIDTH}, Epochs={EPOCHS}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    all_results = {}
    total = len(SEEDS)

    for si, seed in enumerate(SEEDS):
        print(f"\n[{si+1}/{total}] seed={seed}")

        # Train
        torch.manual_seed(seed)
        model = IsotropicMLP(input_dim, WIDTH, num_classes).to(DEVICE)
        train_model(model, train_loader, test_loader, EPOCHS, LR, DEVICE, verbose=False)

        # Diagonalise
        model_d, svs = diagonalise(model)
        acc_check = evaluate(model_d, test_loader, DEVICE)
        print(f"  Acc after diag: {acc_check:.4f}")

        # Criteria
        sv     = svs                                        # Sigma_ii
        w2_col = model_d.W2.data.norm(dim=0).numpy()       # ||W2'[:,j]||
        comp   = sv * w2_col                               # composite

        # Gradient criterion (paper's suggestion)
        print(f"  Computing SV gradients over {GRAD_BATCHES} batches...")
        sv_grads = compute_sv_gradients(model_d, train_loader, GRAD_BATCHES, DEVICE)
        print(f"  Grad range: [{sv_grads.min():.6f}, {sv_grads.max():.6f}]")

        # Leave-one-out
        print(f"  Running leave-one-out ({WIDTH} neurons)...")
        baseline_acc, acc_drops = leave_one_out(model_d, test_loader, DEVICE)
        print(f"  Baseline: {baseline_acc:.4f}  Max drop: {acc_drops.max():.4f}")

        # Correlations
        r_sv,    _ = pearson_r(sv,       acc_drops)
        r_grad,  _ = pearson_r(sv_grads, acc_drops)
        r_w2,    _ = pearson_r(w2_col,   acc_drops)
        r_comp,  _ = pearson_r(comp,     acc_drops)
        # Gradient x W2 composite
        grad_w2  = sv_grads * w2_col
        r_gw2,   _ = pearson_r(grad_w2,  acc_drops)

        rho_sv,   _ = stats.spearmanr(sv,       acc_drops)
        rho_grad, _ = stats.spearmanr(sv_grads, acc_drops)
        rho_w2,   _ = stats.spearmanr(w2_col,   acc_drops)
        rho_comp, _ = stats.spearmanr(comp,      acc_drops)
        rho_gw2,  _ = stats.spearmanr(grad_w2,  acc_drops)

        print(f"  Pearson r:    SV={r_sv:.4f}  Grad={r_grad:.4f}  "
              f"W2={r_w2:.4f}  Comp={r_comp:.4f}  GradxW2={r_gw2:.4f}")
        print(f"  Spearman rho: SV={rho_sv:.4f}  Grad={rho_grad:.4f}  "
              f"W2={rho_w2:.4f}  Comp={rho_comp:.4f}  GradxW2={rho_gw2:.4f}")

        all_results[seed] = {
            'baseline_acc': baseline_acc,
            'acc_drops': acc_drops,
            'sv': sv, 'sv_grads': sv_grads,
            'w2_col': w2_col, 'comp': comp, 'grad_w2': grad_w2,
            'r_sv': r_sv, 'r_grad': r_grad, 'r_w2': r_w2,
            'r_comp': r_comp, 'r_gw2': r_gw2,
            'rho_sv': float(rho_sv), 'rho_grad': float(rho_grad),
            'rho_w2': float(rho_w2), 'rho_comp': float(rho_comp),
            'rho_gw2': float(rho_gw2),
        }

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*65}")
    print("SUMMARY: Mean correlations over seeds")
    print(f"{'='*65}")
    criteria = ['r_sv', 'r_grad', 'r_w2', 'r_comp', 'r_gw2']
    labels_c = ['SV (Sigma_ii)', '|dL/dSigma|', '||W2_col||', 'SV x W2', 'Grad x W2']
    for k, lbl in zip(criteria, labels_c):
        mean_r = np.mean([all_results[s][k] for s in SEEDS])
        print(f"  {lbl:>18}: r = {mean_r:.4f}")

    # =========================================================================
    # Plot
    # =========================================================================
    fig, axes = plt.subplots(len(SEEDS), 5, figsize=(20, 5 * len(SEEDS)))
    if len(SEEDS) == 1:
        axes = axes[np.newaxis, :]

    plot_keys = ['sv', 'sv_grads', 'w2_col', 'comp', 'grad_w2']
    plot_labels = ['SVD Sigma_ii\n(paper)', '|dL/dSigma_ii|\n(paper suggestion)',
                   "||W2' col||", "SV x W2' col", "Grad x W2' col"]
    plot_colors = ['steelblue', 'crimson', 'darkorange', 'darkgreen', 'purple']
    r_keys      = ['r_sv', 'r_grad', 'r_w2', 'r_comp', 'r_gw2']

    for si, seed in enumerate(SEEDS):
        r = all_results[seed]
        for ai, (pk, pl, pc, rk) in enumerate(
                zip(plot_keys, plot_labels, plot_colors, r_keys)):
            ax = axes[si][ai]
            ax.scatter(r[pk], r['acc_drops'], alpha=0.7, color=pc, s=50)
            ax.set_xlabel(pl, fontsize=7)
            ax.set_ylabel('Acc drop', fontsize=8)
            ax.set_title(f'r={r[rk]:.3f}  seed={seed}', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Test Z: Gradient Criterion vs SV and W2 Norm for Pruning',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'gradient_criterion.png'), dpi=150)
    print("\nPlot saved to results/test_Z/gradient_criterion.png")

    # =========================================================================
    # Save results.md
    # =========================================================================
    mean_rs = {k: np.mean([all_results[s][k] for s in SEEDS]) for k in criteria}
    best_crit = max(zip(labels_c, [mean_rs[k] for k in criteria]), key=lambda x: x[1])

    rows = '\n'.join(
        f"| {lbl} | {np.mean([all_results[s][k] for s in SEEDS]):.4f} | "
        f"{np.mean([all_results[s][k.replace('r_','rho_')] for s in SEEDS]):.4f} |"
        for k, lbl in zip(criteria, labels_c)
    )

    if mean_rs['r_grad'] > mean_rs['r_sv'] + 0.05:
        verdict = (f"The gradient criterion |dL/dSigma_ii| (r={mean_rs['r_grad']:.4f}) "
                   f"outperforms the SV criterion (r={mean_rs['r_sv']:.4f}). "
                   f"The paper's footnote suggestion is validated.")
    elif mean_rs['r_grad'] > mean_rs['r_sv']:
        verdict = (f"The gradient criterion marginally outperforms SV "
                   f"(r={mean_rs['r_grad']:.4f} vs {mean_rs['r_sv']:.4f}). "
                   f"Small improvement, gradient adds some signal beyond SV.")
    else:
        verdict = (f"The gradient criterion (r={mean_rs['r_grad']:.4f}) does not "
                   f"outperform SV alone (r={mean_rs['r_sv']:.4f}). "
                   f"Best overall: {best_crit[0]} (r={best_crit[1]:.4f}).")

    results_text = f"""# Test Z -- Gradient-Based Pruning Criterion

## Setup
- Model: IsotropicMLP [3072->{WIDTH}->10], trained {EPOCHS} epochs
- Diagonalised before evaluation (W1 = U Sigma V^T)
- Gradient averaged over {GRAD_BATCHES} training batches
- Leave-one-out pruning in diagonalised basis
- Seeds: {SEEDS}, lr={LR}, batch={BATCH}, Device: CPU

## Question
Does |dL/dSigma_ii| (paper's footnote 5 suggestion) predict pruning
impact better than SV alone or W2_col?

## Results

| Criterion | Pearson r | Spearman rho |
|---|---|---|
{rows}

## Verdict
{verdict}

## Paper Reference
Footnote 5, page 7: "one may combine approaches and consider the gradient
with respect to the diagonalised singular value as a threshold for pruning
-- this is also effectively a batch statistic."

![Gradient criterion results](gradient_criterion.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print("Results saved to results/test_Z/results.md")


if __name__ == '__main__':
    main()
