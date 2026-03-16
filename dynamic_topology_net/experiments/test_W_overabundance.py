"""
Test W -- Overabundance Protocol: Start Wide, Prune to Target
=============================================================
The paper explicitly states (Section 4 / Fig. 4) that an initial
overabundance of neurons followed by neurodegeneration is beneficial,
"mirroring mammalian neural development" (citing [9]).

This is the paper's most biologically-motivated claim and we haven't
tested it directly. The question is whether the ORDER matters:

  (A) Train at target width from scratch (static)
  (B) Start 2x wider, prune halfway through training to target width,
      continue training for the remaining epochs

If (B) > (A), the overabundance protocol adds genuine value beyond
just having more parameters during early training.

Design:
  - Target widths: [16, 24, 32]
  - Overabundance width: target x 2 (i.e., 32, 48, 64)
  - Total budget: 24 epochs each
  - Pruning at epoch 12 (half-way), prune to target by SV criterion
    (SVD diagonalisation -> remove smallest SVs)
  - Compare: Static iso at target | Overabundant iso -> pruned to target
  - Also compare: Static baseline at target (context)
  - Seeds: [42, 123]

Hypothesis: Overabundant iso > Static iso at target, especially at
smaller target widths where the bottle-neck is most constraining.
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

from dynamic_topology_net.core import IsotropicMLP, BaselineMLP, load_cifar10
from dynamic_topology_net.core.train_utils import train_model, evaluate, make_optimizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_W')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED          = 42
TOTAL_EPOCHS  = 24
PRUNE_EPOCH   = 12
LR            = 0.08
BATCH         = 128
TARGET_WIDTHS = [16, 24, 32]
SEEDS         = [42, 123]
DEVICE        = torch.device('cpu')


def svd_prune_to_width(model, target_width, device):
    """
    Prune an IsotropicMLP to target_width using SVD singular value criterion.
    Keeps the top-target_width neurons by singular value of W1.
    Returns a new IsotropicMLP with target_width neurons.
    """
    with torch.no_grad():
        W1 = model.W1.data  # (width, input_dim)
        W2 = model.W2.data  # (num_classes, width)
        b1 = model.b1.data  # (width,)
        b2 = model.b2.data

        # SVD to get singular values
        U, S, Vh = torch.linalg.svd(W1, full_matrices=False)
        # S is descending; keep top target_width

        current_width = W1.shape[0]
        if target_width >= current_width:
            return copy.deepcopy(model)

        # In diagonalised basis: W1' = Sigma V^T, W2' = W2 U
        W1_prime = torch.diag(S) @ Vh          # (width, input_dim)
        W2_prime = W2 @ U                       # (num_classes, width)
        b1_prime = U.T @ b1                     # (width,)

        # Keep top target_width rows/cols (largest SVs first)
        keep = slice(0, target_width)
        W1_pruned = W1_prime[keep, :]           # (target_width, input_dim)
        W2_pruned = W2_prime[:, keep]            # (num_classes, target_width)
        b1_pruned = b1_prime[keep]               # (target_width,)

        # Build new model
        pruned = IsotropicMLP(
            input_dim=W1.shape[1],
            width=target_width,
            num_classes=W2.shape[0]
        ).to(device)

        pruned.W1.data = W1_pruned
        pruned.W2.data = W2_pruned
        pruned.b1.data = b1_pruned
        pruned.b2.data = b2.clone()

    return pruned


def train_n_epochs(model, train_loader, test_loader, n_epochs, lr, device):
    """Train for exactly n_epochs, return final accuracy."""
    if n_epochs <= 0:
        return evaluate(model, test_loader, device)
    train_model(model, train_loader, test_loader, n_epochs, lr, device, verbose=False)
    return evaluate(model, test_loader, device)


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Target widths: {TARGET_WIDTHS}, Prune at epoch {PRUNE_EPOCH}/{TOTAL_EPOCHS}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    # results[(condition, target_width, seed)] = final_acc
    results = {}

    total_configs = len(TARGET_WIDTHS) * len(SEEDS) * 3  # static iso, overabundant, static base
    cfg_idx = 0

    for target_width in TARGET_WIDTHS:
        over_width = target_width * 2
        for seed in SEEDS:
            # --- (A) Static iso at target width ---
            cfg_idx += 1
            print(f"\n[{cfg_idx}/{total_configs}] Static Iso width={target_width} seed={seed}")
            torch.manual_seed(seed)
            m_static = IsotropicMLP(input_dim, target_width, num_classes).to(DEVICE)
            acc_static = train_n_epochs(m_static, train_loader, test_loader,
                                        TOTAL_EPOCHS, LR, DEVICE)
            results[('static_iso', target_width, seed)] = acc_static
            print(f"  Final acc: {acc_static:.4f}")

            # --- (B) Overabundant iso -> prune to target ---
            cfg_idx += 1
            print(f"[{cfg_idx}/{total_configs}] Overabundant Iso {over_width}->{target_width} seed={seed}")
            torch.manual_seed(seed)
            m_over = IsotropicMLP(input_dim, over_width, num_classes).to(DEVICE)
            # Phase 1: train to prune epoch
            acc_pre_prune = train_n_epochs(m_over, train_loader, test_loader,
                                           PRUNE_EPOCH, LR, DEVICE)
            print(f"  Acc at prune epoch {PRUNE_EPOCH}: {acc_pre_prune:.4f}  "
                  f"(width={over_width})")
            # Prune
            m_pruned = svd_prune_to_width(m_over, target_width, DEVICE)
            acc_just_pruned = evaluate(m_pruned, test_loader, DEVICE)
            print(f"  Acc immediately after prune to {target_width}: {acc_just_pruned:.4f}")
            # Phase 2: continue training
            remaining = TOTAL_EPOCHS - PRUNE_EPOCH
            acc_over = train_n_epochs(m_pruned, train_loader, test_loader,
                                      remaining, LR, DEVICE)
            results[('overabundant_iso', target_width, seed)] = acc_over
            print(f"  Final acc after {remaining} more epochs: {acc_over:.4f}")

            # --- (C) Static baseline at target width ---
            cfg_idx += 1
            print(f"[{cfg_idx}/{total_configs}] Static Baseline width={target_width} seed={seed}")
            torch.manual_seed(seed)
            m_base = BaselineMLP(input_dim, target_width, num_classes).to(DEVICE)
            acc_base = train_n_epochs(m_base, train_loader, test_loader,
                                      TOTAL_EPOCHS, LR, DEVICE)
            results[('static_base', target_width, seed)] = acc_base
            print(f"  Final acc: {acc_base:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*65}")
    print("SUMMARY: Mean accuracy over seeds")
    print(f"{'='*65}")
    print(f"{'Target':>8}  {'Static Iso':>12}  {'Overabundant->':>16}  "
          f"{'Gain':>8}  {'Static Base':>12}")
    for tw in TARGET_WIDTHS:
        s_iso  = np.mean([results[('static_iso',       tw, s)] for s in SEEDS])
        s_over = np.mean([results[('overabundant_iso', tw, s)] for s in SEEDS])
        s_base = np.mean([results[('static_base',      tw, s)] for s in SEEDS])
        gain   = (s_over - s_iso) * 100
        print(f"  {tw:>6}  {s_iso*100:>10.2f}%  {s_over*100:>14.2f}%  "
              f"{gain:>+7.2f}%  {s_base*100:>10.2f}%")

    # =========================================================================
    # Plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(TARGET_WIDTHS))
    w = 0.25

    for i, (cond, label, color) in enumerate([
        ('static_iso',       'Static Iso (target)',    '#f97a0a'),
        ('overabundant_iso', f'Iso {chr(8594)} pruned to target', '#7f2a00'),
        ('static_base',      'Static Baseline',         '#3a8fd1'),
    ]):
        means = [np.mean([results[(cond, tw, s)] for s in SEEDS]) * 100
                 for tw in TARGET_WIDTHS]
        stds  = [np.std( [results[(cond, tw, s)] for s in SEEDS]) * 100
                 for tw in TARGET_WIDTHS]
        ax.bar(x + (i - 1) * w, means, w, label=label, color=color,
               yerr=stds, capsize=4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f'target={tw}\n(from {tw*2})' for tw in TARGET_WIDTHS])
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Overabundance Protocol: Start 2x Wide, Prune at Epoch 12\nvs Training at Target Width from Scratch')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'overabundance.png'), dpi=150)
    print("\nPlot saved to results/test_W/overabundance.png")

    # =========================================================================
    # Save results.md
    # =========================================================================
    def mk_table():
        header = ('| Target width | Over width | Static Iso | Overabundant->Pruned | '
                  'Gain | Static Baseline |')
        sep = '|---|---|---|---|---|---|'
        rows = []
        for tw in TARGET_WIDTHS:
            s_iso  = np.mean([results[('static_iso',       tw, s)] for s in SEEDS])
            s_over = np.mean([results[('overabundant_iso', tw, s)] for s in SEEDS])
            s_base = np.mean([results[('static_base',      tw, s)] for s in SEEDS])
            gain   = (s_over - s_iso) * 100
            rows.append(f'| {tw} | {tw*2} | {s_iso*100:.2f}% | {s_over*100:.2f}% | '
                        f'{gain:+.2f}% | {s_base*100:.2f}% |')
        return '\n'.join([header, sep] + rows)

    over_gains = [(np.mean([results[('overabundant_iso', tw, s)] for s in SEEDS]) -
                   np.mean([results[('static_iso', tw, s)] for s in SEEDS])) * 100
                  for tw in TARGET_WIDTHS]
    mean_gain = np.mean(over_gains)

    if mean_gain > 1.0:
        verdict = (f"Overabundance protocol outperforms static training by {mean_gain:+.2f}% "
                   f"on average. Starting wide and pruning genuinely benefits final performance, "
                   f"supporting the paper's biological analogy.")
    elif mean_gain > 0:
        verdict = (f"Overabundance shows marginal gain ({mean_gain:+.2f}%). "
                   f"The effect is present but small — the protocol works but the advantage "
                   f"over direct training is modest at these scales.")
    else:
        verdict = (f"Overabundance does not outperform static training (gain={mean_gain:+.2f}%). "
                   f"At these scales and epoch budgets, starting wide and pruning does not "
                   f"confer an advantage over training at the target width directly.")

    results_text = f"""# Test W -- Overabundance Protocol

## Setup
- Total epochs: {TOTAL_EPOCHS}, prune at epoch {PRUNE_EPOCH}
- Start width: 2x target, prune to target using SVD SV criterion
- Widths: targets={TARGET_WIDTHS}, seeds={SEEDS}, lr={LR}, batch={BATCH}
- Device: CPU

## Question
Does starting 2x wide and pruning halfway through beat training at the
target width from scratch? This tests the paper's biological overabundance claim.

## Results

{mk_table()}

## Per-width gains (Overabundant - Static Iso)
{chr(10).join(f'- target={tw}: {g:+.2f}%' for tw, g in zip(TARGET_WIDTHS, over_gains))}
- Mean gain: {mean_gain:+.2f}%

## Verdict
{verdict}

## Paper's Claim
Section 4: "the 32 width layer begins at a higher accuracy, and maintains this
for 16 and 24 width networks, suggesting an initial overabundance of neurons,
followed by neurodegeneration, is beneficial for performance."

![Overabundance results](overabundance.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print("Results saved to results/test_W/results.md")


if __name__ == '__main__':
    main()
