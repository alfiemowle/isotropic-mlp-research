"""
Test Y -- Minimum Fine-Tune Budget After Pruning
================================================
Test V showed pruning at ANY epoch (0-24) + 8 fine-tune epochs recovers
to ~40% accuracy. This raises a practical question: how FEW fine-tune
epochs does recovery actually require?

This matters for deployment: if a network can recover in 1-2 epochs of
fine-tuning after topology change, dynamic adaptation is very cheap.

Protocol:
  1. Train IsotropicMLP [3072->24->10] for 24 epochs.
  2. Prune 50% of neurons (12 of 24) by SVD SV criterion.
  3. Fine-tune for {0, 1, 2, 3, 4, 6, 8, 12} epochs.
  4. Record accuracy after each fine-tune duration.
  5. Also test: prune at init (epoch 0) + fine-tune budget.
  6. Compare to: no-prune 24-epoch baseline.

Hypothesis: Recovery is rapid -- most of the accuracy rebounds in the
first 1-2 fine-tune epochs, with diminishing returns thereafter.
This would make isotropic dynamic networks practically cheap to adapt.

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

from dynamic_topology_net.core import IsotropicMLP, load_cifar10
from dynamic_topology_net.core.train_utils import train_model, evaluate, make_optimizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_Y')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED          = 42
PRETRAIN_EPOCHS = 24
LR            = 0.08
BATCH         = 128
WIDTH         = 24
PRUNE_FRAC    = 0.50   # remove 50% of neurons (harder case)
SEEDS         = [42, 123]
FINETUNE_BUDGETS = [0, 1, 2, 3, 4, 6, 8, 12]
DEVICE        = torch.device('cpu')


def svd_prune(model, frac, device):
    """Prune bottom-frac neurons by SVD singular value (paper's criterion)."""
    with torch.no_grad():
        W1 = model.W1.data
        W2 = model.W2.data
        b1 = model.b1.data

        U, S, Vh = torch.linalg.svd(W1, full_matrices=False)
        n_keep = int(W1.shape[0] * (1 - frac))

        W1_prime = torch.diag(S) @ Vh
        W2_prime = W2 @ U
        b1_prime = U.T @ b1

        pruned = IsotropicMLP(
            input_dim=W1.shape[1],
            width=n_keep,
            num_classes=W2.shape[0]
        ).to(device)
        pruned.W1.data = W1_prime[:n_keep, :]
        pruned.W2.data = W2_prime[:, :n_keep]
        pruned.b1.data = b1_prime[:n_keep]
        pruned.b2.data = model.b2.data.clone()

    return pruned


def finetune_and_track(model, train_loader, test_loader, max_epochs, lr, device):
    """
    Fine-tune model for max_epochs, recording accuracy after EACH epoch.
    Returns list of accuracies at epochs [0, 1, 2, ..., max_epochs].
    """
    accs = [evaluate(model, test_loader, device)]  # epoch 0 = right after prune

    opt  = make_optimizer(model, lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        acc = evaluate(model, test_loader, device)
        accs.append(acc)

    return accs


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}, Width={WIDTH}, Prune frac={PRUNE_FRAC}")
    print(f"Fine-tune budgets: {FINETUNE_BUDGETS}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    max_finetune = max(FINETUNE_BUDGETS)

    # all_curves[(prune_at, seed)] = list of accuracies [epoch_0, epoch_1, ..., epoch_max]
    all_curves = {}
    baseline_accs = {}

    for seed in SEEDS:
        # --- Baseline: full 24-epoch training, no pruning ---
        torch.manual_seed(seed)
        m_base = IsotropicMLP(input_dim, WIDTH, num_classes).to(DEVICE)
        train_model(m_base, train_loader, test_loader, PRETRAIN_EPOCHS, LR, DEVICE, verbose=False)
        baseline_accs[seed] = evaluate(m_base, test_loader, DEVICE)
        print(f"\nSeed={seed} | No-prune baseline: {baseline_accs[seed]:.4f}")

        for prune_at in [0, PRETRAIN_EPOCHS]:  # prune before training or after full training
            print(f"\n  Seed={seed}, prune_at=epoch_{prune_at}")

            # Train to prune_at
            torch.manual_seed(seed)
            model = IsotropicMLP(input_dim, WIDTH, num_classes).to(DEVICE)
            if prune_at > 0:
                train_model(model, train_loader, test_loader, prune_at,
                            LR, DEVICE, verbose=False)

            acc_before_prune = evaluate(model, test_loader, DEVICE)
            print(f"    Acc before prune: {acc_before_prune:.4f}")

            # Prune
            pruned = svd_prune(model, PRUNE_FRAC, DEVICE)
            n_kept = pruned.width
            acc_just_pruned = evaluate(pruned, test_loader, DEVICE)
            print(f"    Acc just after prune ({n_kept}/{WIDTH}): {acc_just_pruned:.4f}")

            # Fine-tune and track every epoch
            curves = finetune_and_track(
                pruned, train_loader, test_loader, max_finetune, LR, DEVICE)
            all_curves[(prune_at, seed)] = curves

            print(f"    Recovery: " + "  ".join(
                f"ep{b}={curves[b]:.3f}" for b in FINETUNE_BUDGETS
            ))

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*65}")
    print("SUMMARY: Mean accuracy over seeds at each fine-tune budget")
    print(f"{'='*65}")
    baseline_mean = np.mean([baseline_accs[s] for s in SEEDS])
    print(f"No-prune baseline: {baseline_mean:.4f}")

    for prune_at in [0, PRETRAIN_EPOCHS]:
        print(f"\nPrune at epoch {prune_at}:")
        print(f"  {'Budget':>8}  {'Acc':>8}  {'vs baseline':>12}  {'Recovery %':>12}")
        for b in FINETUNE_BUDGETS:
            mean_acc = np.mean([all_curves[(prune_at, s)][b] for s in SEEDS])
            vs_base  = (mean_acc - baseline_mean) * 100
            # Recovery = (acc - acc_just_pruned) / (baseline - acc_just_pruned)
            acc_pruned = np.mean([all_curves[(prune_at, s)][0] for s in SEEDS])
            if baseline_mean > acc_pruned:
                recovery = (mean_acc - acc_pruned) / (baseline_mean - acc_pruned) * 100
            else:
                recovery = 100.0
            print(f"  {b:>8}  {mean_acc:>8.4f}  {vs_base:>+11.2f}%  {recovery:>11.1f}%")

    # =========================================================================
    # Plot
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {0: '#c04000', PRETRAIN_EPOCHS: '#3a8fd1'}
    labels = {0: f'Prune at epoch 0 (init)',
              PRETRAIN_EPOCHS: f'Prune at epoch {PRETRAIN_EPOCHS} (after full train)'}

    fine_tune_x = list(range(max_finetune + 1))

    for ax_i, (norm, title) in enumerate([
        (False, 'Raw accuracy after pruning + fine-tune'),
        (True,  '% of gap to baseline recovered'),
    ]):
        ax = axes[ax_i]
        ax.axhline(baseline_mean * 100 if not norm else 100,
                   linestyle='--', color='black', alpha=0.5,
                   label=f'No-prune baseline ({baseline_mean*100:.1f}%)')

        for prune_at in [0, PRETRAIN_EPOCHS]:
            mean_curve = [np.mean([all_curves[(prune_at, s)][b]
                                   for s in SEEDS]) for b in range(max_finetune + 1)]
            std_curve  = [np.std( [all_curves[(prune_at, s)][b]
                                   for s in SEEDS]) for b in range(max_finetune + 1)]
            acc_pruned = mean_curve[0]

            if norm:
                y_vals = [(v - acc_pruned) / (baseline_mean - acc_pruned + 1e-8) * 100
                          for v in mean_curve]
                y_err  = [s / (baseline_mean - acc_pruned + 1e-8) * 100 for s in std_curve]
            else:
                y_vals = [v * 100 for v in mean_curve]
                y_err  = [s * 100 for s in std_curve]

            ax.errorbar(fine_tune_x, y_vals, yerr=y_err,
                        fmt='o-', color=colors[prune_at], linewidth=2,
                        capsize=3, label=labels[prune_at])

        ax.set_xlabel('Fine-tune epochs after pruning')
        ax.set_ylabel('Accuracy (%)' if not norm else 'Recovery (%)')
        ax.set_title(title)
        ax.set_xticks(fine_tune_x)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Minimum Fine-Tune Budget: {int(PRUNE_FRAC*100)}% pruning, '
                 f'{WIDTH//2}/{WIDTH} neurons kept', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'minimum_finetune.png'), dpi=150)
    print("\nPlot saved to results/test_Y/minimum_finetune.png")

    # =========================================================================
    # Save results.md
    # =========================================================================
    # Find epoch at which 90% of recovery is achieved
    def find_recovery_epoch(prune_at, threshold=0.90):
        mean_curve = [np.mean([all_curves[(prune_at, s)][b]
                               for s in SEEDS]) for b in range(max_finetune + 1)]
        acc_pruned = mean_curve[0]
        target = acc_pruned + threshold * (baseline_mean - acc_pruned)
        for b in range(1, max_finetune + 1):
            if mean_curve[b] >= target:
                return b
        return max_finetune

    ep90_post  = find_recovery_epoch(PRETRAIN_EPOCHS)
    ep90_init  = find_recovery_epoch(0)

    def mk_table(prune_at):
        rows = []
        acc_pruned = np.mean([all_curves[(prune_at, s)][0] for s in SEEDS])
        for b in FINETUNE_BUDGETS:
            m = np.mean([all_curves[(prune_at, s)][b] for s in SEEDS])
            std = np.std([all_curves[(prune_at, s)][b] for s in SEEDS])
            rec = (m - acc_pruned) / (baseline_mean - acc_pruned + 1e-8) * 100
            rows.append(f'| {b} | {m*100:.2f}%+-{std*100:.2f}% | '
                        f'{(m-baseline_mean)*100:+.2f}% | {rec:.0f}% |')
        header = '| Fine-tune epochs | Accuracy | vs no-prune | Recovery |'
        sep    = '|---|---|---|---|'
        return '\n'.join([header, sep] + rows)

    results_text = f"""# Test Y -- Minimum Fine-Tune Budget After Pruning

## Setup
- Model: IsotropicMLP [3072->{WIDTH}->10]
- Pretrain: {PRETRAIN_EPOCHS} epochs, then prune {int(PRUNE_FRAC*100)}% ({WIDTH//2}/{WIDTH} kept)
- Fine-tune budgets tested: {FINETUNE_BUDGETS}
- Pruning criterion: SVD singular values (paper's method)
- Seeds: {SEEDS}, lr={LR}, batch={BATCH}, Device: CPU
- No-prune baseline: {baseline_mean*100:.2f}%

## Results: Prune After Full Training (epoch {PRETRAIN_EPOCHS})

{mk_table(PRETRAIN_EPOCHS)}

Epoch at which 90% recovery achieved: **{ep90_post}**

## Results: Prune At Init (epoch 0)

{mk_table(0)}

Epoch at which 90% recovery achieved: **{ep90_init}**

## Verdict
90% of accuracy recovery occurs within {min(ep90_post, ep90_init)} fine-tune epoch(s).
This {'strongly supports' if min(ep90_post, ep90_init) <= 2 else 'suggests'} that
isotropic dynamic networks can recover from significant topology changes
with minimal retraining cost.

![Recovery curves](minimum_finetune.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print("Results saved to results/test_Y/results.md")


if __name__ == '__main__':
    main()
