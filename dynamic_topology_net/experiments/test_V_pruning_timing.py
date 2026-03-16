"""
Test V -- Optimal Pruning Timing
=================================
When during training is the best time to prune?

The SV spectrum evolves from flat (all SVs ≈ equal at random init) to
concentrated (a few large SVs dominate). If we prune too early, the
spectrum hasn't settled — we might remove neurons that would become
important. If we prune too late, we've trained useless neurons for longer
than needed.

Protocol:
  1. Train IsotropicMLP [3072->24->10] for up to 24 epochs.
  2. At each checkpoint epoch T ∈ {4, 8, 12, 16, 20, 24}:
     a. Prune the bottom-k neurons by W1 row norm (SV proxy).
     b. Fine-tune for 8 more epochs.
     c. Record final accuracy.
  3. Also run: no pruning, prune-at-init (before any training).
  4. Prune fractions: remove 25% of neurons (6/24) or 50% (12/24).

Hypothesis: There is an optimal pruning window — too early (spectrum not
settled) or too late (gradient signal is weaker) both hurt.

Related: Test S tracks SV evolution and spectral entropy, which can
identify when the "phase transition" occurs.

Width=24, prune to 18 (25%) or 12 (50%), fine-tune 8 epochs.
Seeds: [42, 123]
Device: CPU
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_V')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED          = 42
TOTAL_EPOCHS  = 24
FINETUNE_EPOCHS = 8
LR            = 0.08
BATCH         = 128
WIDTH         = 24
DEVICE        = torch.device('cpu')
SEEDS         = [42, 123]

PRUNE_EPOCHS  = [0, 4, 8, 12, 16, 20, 24]   # 0 = prune before any training
PRUNE_FRACS   = [0.25, 0.50]                 # fraction of neurons to remove


def train_to_epoch(model, train_loader, test_loader, target_epoch, lr, device):
    """Train model for target_epoch epochs total (from scratch)."""
    if target_epoch > 0:
        train_model(model, train_loader, test_loader, target_epoch, lr, device, verbose=False)
    return evaluate(model, test_loader, device)


def prune_model(model, frac, device):
    """
    Remove bottom-frac neurons by W1 row norm.
    Returns new IsotropicMLP with fewer neurons, weights copied.
    """
    with torch.no_grad():
        width = model.width
        n_remove = max(1, int(width * frac))
        n_keep   = width - n_remove

        # Score by W1 row norm
        row_norms = model.W1.data.norm(dim=1)  # (width,)
        _, sorted_idx = torch.sort(row_norms, descending=True)
        keep_idx = sorted_idx[:n_keep].sort().values  # keep top n_keep

        # Build pruned model
        pruned = IsotropicMLP(
            input_dim=model.W1.shape[1],
            width=n_keep,
            num_classes=model.num_classes
        ).to(device)

        pruned.W1.data = model.W1.data[keep_idx].clone()
        pruned.b1.data = model.b1.data[keep_idx].clone()
        pruned.W2.data = model.W2.data[:, keep_idx].clone()
        # b2 unchanged

    return pruned, n_keep


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Width={WIDTH}, prune_epochs={PRUNE_EPOCHS}, prune_fracs={PRUNE_FRACS}")
    print(f"Fine-tune epochs: {FINETUNE_EPOCHS}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    # all_results[(frac, prune_epoch, seed)] = {
    #   'acc_at_prune': float, 'acc_after_finetune': float,
    #   'n_kept': int, 'acc_baseline_no_prune': float
    # }
    all_results = {}
    baseline_results = {}  # seed -> acc_at_24_epochs (no pruning)

    total_configs = len(SEEDS) * len(PRUNE_FRACS) * len(PRUNE_EPOCHS)
    config_idx = 0

    for seed in SEEDS:
        # No-pruning baseline: train full 24 epochs
        torch.manual_seed(seed)
        base_model = IsotropicMLP(input_dim, WIDTH, num_classes).to(DEVICE)
        train_model(base_model, train_loader, test_loader, TOTAL_EPOCHS, LR, DEVICE, verbose=False)
        baseline_acc = evaluate(base_model, test_loader, DEVICE)
        baseline_results[seed] = baseline_acc
        print(f"\nSeed={seed} | No-pruning baseline (24 epochs): {baseline_acc:.4f}")

        for frac in PRUNE_FRACS:
            for prune_epoch in PRUNE_EPOCHS:
                config_idx += 1
                print(f"\n[{config_idx}/{total_configs}] seed={seed} frac={frac:.0%} "
                      f"prune_at_epoch={prune_epoch}")

                # Train to prune_epoch
                torch.manual_seed(seed)
                model = IsotropicMLP(input_dim, WIDTH, num_classes).to(DEVICE)
                acc_at_prune = train_to_epoch(
                    model, train_loader, test_loader, prune_epoch, LR, DEVICE)
                print(f"  Acc at epoch {prune_epoch}: {acc_at_prune:.4f}")

                # Prune
                pruned_model, n_kept = prune_model(model, frac, DEVICE)
                acc_just_after_prune = evaluate(pruned_model, test_loader, DEVICE)
                print(f"  Immediately after prune: {acc_just_after_prune:.4f} ({n_kept}/{WIDTH} neurons)")

                # Fine-tune
                train_model(pruned_model, train_loader, test_loader,
                            FINETUNE_EPOCHS, LR, DEVICE, verbose=False)
                acc_after_finetune = evaluate(pruned_model, test_loader, DEVICE)
                print(f"  After {FINETUNE_EPOCHS}-epoch fine-tune: {acc_after_finetune:.4f}")

                all_results[(frac, prune_epoch, seed)] = {
                    'acc_at_prune':        acc_at_prune,
                    'acc_just_after_prune': acc_just_after_prune,
                    'acc_after_finetune':  acc_after_finetune,
                    'n_kept':              n_kept,
                }

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*75}")
    print("SUMMARY: Final accuracy after pruning + fine-tune (mean over seeds)")
    print(f"{'='*75}")
    print(f"{'Prune epoch':>14}", end='')
    for frac in PRUNE_FRACS:
        print(f"  frac={frac:.0%} ({WIDTH - int(WIDTH*frac)}/{WIDTH})", end='')
    print()

    for prune_epoch in PRUNE_EPOCHS:
        print(f"  epoch {prune_epoch:>5}", end='')
        for frac in PRUNE_FRACS:
            mean_acc = np.mean([all_results[(frac, prune_epoch, s)]['acc_after_finetune']
                                for s in SEEDS])
            print(f"  {mean_acc:.4f}", end='')
        print()

    print(f"\n  Baseline (no prune):  "
          + "  ".join(f"{np.mean([baseline_results[s] for s in SEEDS]):.4f}" for _ in PRUNE_FRACS))

    # =========================================================================
    # Plot
    # =========================================================================
    fig, axes = plt.subplots(1, len(PRUNE_FRACS), figsize=(12, 5))
    if len(PRUNE_FRACS) == 1:
        axes = [axes]

    baseline_mean = np.mean([baseline_results[s] for s in SEEDS])
    colors = ['#e07020', '#c04000']

    for ax_idx, frac in enumerate(PRUNE_FRACS):
        ax = axes[ax_idx]
        n_kept = WIDTH - int(WIDTH * frac)

        # Mean over seeds
        mean_accs = [np.mean([all_results[(frac, pe, s)]['acc_after_finetune']
                               for s in SEEDS]) * 100
                     for pe in PRUNE_EPOCHS]
        std_accs  = [np.std([all_results[(frac, pe, s)]['acc_after_finetune']
                              for s in SEEDS]) * 100
                     for pe in PRUNE_EPOCHS]

        ax.errorbar(PRUNE_EPOCHS, mean_accs, yerr=std_accs,
                    fmt='o-', color=colors[ax_idx], linewidth=2, capsize=4,
                    label=f'After prune+finetune ({n_kept}/{WIDTH} neurons)')
        ax.axhline(baseline_mean * 100, linestyle='--', color='gray', alpha=0.7,
                   label=f'No-prune baseline ({WIDTH}/{WIDTH} neurons)')

        # Mark best
        best_idx = int(np.argmax(mean_accs))
        ax.annotate(f'Best: epoch {PRUNE_EPOCHS[best_idx]}\n({mean_accs[best_idx]:.2f}%)',
                    xy=(PRUNE_EPOCHS[best_idx], mean_accs[best_idx]),
                    xytext=(PRUNE_EPOCHS[best_idx] + 1, mean_accs[best_idx] - 1),
                    fontsize=8, arrowprops=dict(arrowstyle='->', color='black'))

        ax.set_xlabel('Epoch at which pruning occurs')
        ax.set_ylabel('Final test accuracy (%)')
        ax.set_title(f'Pruning timing: frac={frac:.0%} removed\n'
                     f'({n_kept}/{WIDTH} neurons kept, fine-tune {FINETUNE_EPOCHS} epochs)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(PRUNE_EPOCHS)

    plt.suptitle('Optimal Pruning Timing: When to Prune During Training', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pruning_timing.png'), dpi=150)
    print("\nPlot saved to results/test_V/pruning_timing.png")

    # Also plot: accuracy immediately after prune (before finetune)
    fig2, axes2 = plt.subplots(1, len(PRUNE_FRACS), figsize=(12, 5))
    if len(PRUNE_FRACS) == 1:
        axes2 = [axes2]

    for ax_idx, frac in enumerate(PRUNE_FRACS):
        ax = axes2[ax_idx]
        n_kept = WIDTH - int(WIDTH * frac)

        mean_immediate = [np.mean([all_results[(frac, pe, s)]['acc_just_after_prune']
                                    for s in SEEDS]) * 100
                           for pe in PRUNE_EPOCHS]
        mean_finetuned = [np.mean([all_results[(frac, pe, s)]['acc_after_finetune']
                                    for s in SEEDS]) * 100
                           for pe in PRUNE_EPOCHS]

        ax.plot(PRUNE_EPOCHS, mean_immediate, 'o--', color='salmon', linewidth=1.5,
                label='Just after prune')
        ax.plot(PRUNE_EPOCHS, mean_finetuned, 'o-', color=colors[ax_idx], linewidth=2,
                label='After fine-tune')
        ax.axhline(baseline_mean * 100, linestyle=':', color='gray', alpha=0.7,
                   label='No-prune baseline')
        ax.set_xlabel('Epoch at which pruning occurs')
        ax.set_ylabel('Test accuracy (%)')
        ax.set_title(f'Immediate vs fine-tuned recovery\nfrac={frac:.0%}, {n_kept}/{WIDTH} neurons')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(PRUNE_EPOCHS)

    plt.suptitle('Pruning: Immediate Accuracy Drop vs Recovery After Fine-Tune', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pruning_recovery.png'), dpi=150)
    print("Plot saved to results/test_V/pruning_recovery.png")

    # =========================================================================
    # Save results.md
    # =========================================================================
    def mk_timing_table(frac):
        n_kept = WIDTH - int(WIDTH * frac)
        header = f'| Prune epoch | n_kept | Immed. acc (mean) | Final acc (mean±std) | vs baseline |'
        sep    = '|---|---|---|---|---|'
        rows   = []
        for pe in PRUNE_EPOCHS:
            immed = np.mean([all_results[(frac, pe, s)]['acc_just_after_prune'] for s in SEEDS])
            final_m = np.mean([all_results[(frac, pe, s)]['acc_after_finetune'] for s in SEEDS])
            final_s = np.std([all_results[(frac, pe, s)]['acc_after_finetune'] for s in SEEDS])
            vs_base = final_m - baseline_mean
            rows.append(f'| {pe} | {n_kept}/{WIDTH} | {immed*100:.2f}% | '
                        f'{final_m*100:.2f}%±{final_s*100:.2f}% | {vs_base*100:+.2f}% |')
        return '\n'.join([header, sep] + rows)

    best_configs = {}
    for frac in PRUNE_FRACS:
        mean_accs = {pe: np.mean([all_results[(frac, pe, s)]['acc_after_finetune']
                                   for s in SEEDS]) for pe in PRUNE_EPOCHS}
        best_epoch = max(mean_accs, key=mean_accs.get)
        best_configs[frac] = (best_epoch, mean_accs[best_epoch])

    best_overall = max(PRUNE_EPOCHS,
                       key=lambda pe: np.mean([all_results[(PRUNE_FRACS[0], pe, s)]['acc_after_finetune']
                                               for s in SEEDS]))

    if best_overall == 0:
        timing_verdict = "Pruning before training (epoch 0) is optimal — the spectrum is not informative at init, suggesting any pruning time works equally."
    elif best_overall >= 16:
        timing_verdict = f"Late pruning (epoch {best_overall}) is optimal — the spectrum needs time to settle before the SV-based criterion is reliable."
    else:
        timing_verdict = f"Mid-training pruning (epoch {best_overall}) is optimal — a phase transition in the SV spectrum creates an optimal pruning window."

    results_text = f"""# Test V -- Optimal Pruning Timing

## Setup
- Model: IsotropicMLP [3072->{WIDTH}->10]
- Total training: {TOTAL_EPOCHS} epochs, lr={LR}, batch={BATCH}
- Pruning fraction: {PRUNE_FRACS} of {WIDTH} neurons
- Fine-tune after pruning: {FINETUNE_EPOCHS} epochs
- Prune epochs tested: {PRUNE_EPOCHS}
- Seeds: {SEEDS}, Device: CPU
- Criterion: W1 row norm (SV proxy), remove bottom-k

## No-Pruning Baseline
Full {WIDTH}-neuron model, {TOTAL_EPOCHS} epochs: **{baseline_mean*100:.2f}%** (mean over seeds)

## Results: 25% Pruning (remove {int(WIDTH*0.25)}, keep {WIDTH - int(WIDTH*0.25)}/{WIDTH})

{mk_timing_table(0.25)}

## Results: 50% Pruning (remove {int(WIDTH*0.50)}, keep {WIDTH - int(WIDTH*0.50)}/{WIDTH})

{mk_timing_table(0.50)}

## Best Configurations
{chr(10).join(f'- frac={frac:.0%}: best at epoch {epoch}, acc={acc*100:.2f}%' for frac, (epoch, acc) in best_configs.items())}

## Verdict
{timing_verdict}

## Relationship to Test S
Test S tracks the SV spectrum evolution and spectral entropy at every epoch.
The optimal pruning epoch from this test can be compared against the
spectral entropy trajectory from Test S to verify whether entropy-based
timing matches empirical accuracy.

![Pruning timing](pruning_timing.png)
![Recovery curves](pruning_recovery.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print("Results saved to results/test_V/results.md")


if __name__ == '__main__':
    main()
