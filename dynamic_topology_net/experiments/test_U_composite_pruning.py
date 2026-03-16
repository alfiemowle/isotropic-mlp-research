"""
Test U -- Composite Pruning Criterion: SV × ‖W2_col‖ vs SV Alone
==================================================================
Test G found Pearson r(SV, acc_drop) = 0.71 for SV-only pruning criterion.
The composite score SV × ‖W2_col‖ was proposed as potentially better:
  - SV captures the input representation breadth of a neuron
  - ‖W2_col‖ captures how strongly the neuron influences the output

A neuron with a small SV but large W2 col norm could still be important
(its W1 row is narrow but its W2 output is big).
A neuron with a large SV but tiny W2 col norm is representationally rich
but barely affects the output.

The composite score should better predict which neuron to prune.

Methodology:
  1. Train IsotropicMLP [3072->width->10] to convergence (24 epochs).
  2. For each neuron j (width=24):
     - Record SV_j (j-th singular value of W1)
     - Record ‖W2[:,j]‖ (L2 norm of j-th column of W2)
     - Record composite_j = SV_j × ‖W2[:,j]‖
  3. Prune each neuron one at a time (leave-one-out), re-evaluate accuracy.
     acc_drop_j = acc_baseline - acc_after_removing_j
  4. Compute Pearson r between criterion and acc_drop:
     - r(SV, acc_drop)
     - r(‖W2_col‖, acc_drop)
     - r(composite, acc_drop)
     - r(SV_rank, acc_drop)        [rank ordering by SV]
     - r(composite_rank, acc_drop) [rank ordering by composite]

Widths tested: 16, 24 (to check consistency)
Seeds: [42, 123]
Device: CPU
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from dynamic_topology_net.core import IsotropicMLP, load_cifar10
from dynamic_topology_net.core.train_utils import train_model, evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_U')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED    = 42
EPOCHS  = 24
LR      = 0.08
BATCH   = 128
WIDTHS  = [16, 24]
SEEDS   = [42, 123]
DEVICE  = torch.device('cpu')


def get_pruning_criteria(model):
    """
    Extract per-neuron pruning criteria from a trained IsotropicMLP.
    Returns dict with arrays of length=width:
      - sv: singular values of W1
      - w2_col_norm: L2 norm of each column of W2
      - composite: sv * w2_col_norm
    """
    with torch.no_grad():
        W1 = model.W1.data  # (width, input_dim)
        W2 = model.W2.data  # (num_classes, width)

        # SVD of W1
        svs = torch.linalg.svdvals(W1)  # (width,) sorted descending

        # W2 column norms
        w2_norms = W2.norm(dim=0)  # (width,)

        # The SV and W2_col need to be matched per neuron.
        # W1 = U Σ V^T; svdvals gives Σ but not the neuron-SV mapping.
        # For IsotropicMLP, W1 rows are the neurons.
        # SV_j = ‖W1[j,:]‖ (since W1 is in row-per-neuron form and
        # for a pruning criterion we care about the row norm).
        # The SVD approach gives a proper spectral view, but the
        # neuron-level criterion needs row-level matching.
        # Use W1 row norms as the per-neuron "SV proxy".
        w1_row_norms = W1.norm(dim=1)  # (width,)

        composite = w1_row_norms * w2_norms

    return {
        'sv':        w1_row_norms.numpy(),
        'w2_col':    w2_norms.numpy(),
        'composite': composite.numpy(),
    }


def leave_one_out_pruning(model, test_loader, device):
    """
    For each neuron j, remove it (zero out its W1 row and W2 col)
    and evaluate accuracy. Returns array of acc_drops.
    """
    baseline_acc = evaluate(model, test_loader, device)
    width = model.width
    acc_drops = np.zeros(width)

    with torch.no_grad():
        for j in range(width):
            # Save
            w1_row = model.W1.data[j].clone()
            w2_col = model.W2.data[:, j].clone()
            b1_j   = model.b1.data[j].clone()

            # Remove neuron j
            model.W1.data[j] = 0.0
            model.W2.data[:, j] = 0.0
            model.b1.data[j] = 0.0

            pruned_acc = evaluate(model, test_loader, device)
            acc_drops[j] = baseline_acc - pruned_acc

            # Restore
            model.W1.data[j] = w1_row
            model.W2.data[:, j] = w2_col
            model.b1.data[j] = b1_j

    return baseline_acc, acc_drops


def pearson_r(x, y):
    """Pearson correlation coefficient."""
    r, p = stats.pearsonr(x, y)
    return r, p


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Widths: {WIDTHS}, Seeds: {SEEDS}, Epochs: {EPOCHS}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    all_results = {}  # (width, seed) -> results dict

    total_runs = len(WIDTHS) * len(SEEDS)
    run_idx = 0

    for width in WIDTHS:
        for seed in SEEDS:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] width={width} seed={seed}")
            torch.manual_seed(seed)
            model = IsotropicMLP(input_dim, width, num_classes).to(DEVICE)
            train_model(model, train_loader, test_loader, EPOCHS, LR, DEVICE, verbose=False)

            # Extract criteria
            criteria = get_pruning_criteria(model)

            # Leave-one-out pruning
            print(f"  Running leave-one-out pruning ({width} neurons)...")
            baseline_acc, acc_drops = leave_one_out_pruning(model, test_loader, DEVICE)
            print(f"  Baseline acc: {baseline_acc:.4f}")
            print(f"  Max acc drop: {acc_drops.max():.4f} (neuron {acc_drops.argmax()})")
            print(f"  Mean acc drop: {acc_drops.mean():.4f}")

            # Correlations
            sv   = criteria['sv']
            w2   = criteria['w2_col']
            comp = criteria['composite']

            r_sv,   p_sv   = pearson_r(sv,   acc_drops)
            r_w2,   p_w2   = pearson_r(w2,   acc_drops)
            r_comp, p_comp = pearson_r(comp, acc_drops)

            # Rank-based correlations (Spearman)
            rho_sv,   _ = stats.spearmanr(sv,   acc_drops)
            rho_w2,   _ = stats.spearmanr(w2,   acc_drops)
            rho_comp, _ = stats.spearmanr(comp, acc_drops)

            print(f"  Pearson r: SV={r_sv:.4f}  W2_col={r_w2:.4f}  Composite={r_comp:.4f}")
            print(f"  Spearman rho: SV={rho_sv:.4f}  W2_col={rho_w2:.4f}  Composite={rho_comp:.4f}")

            all_results[(width, seed)] = {
                'baseline_acc': baseline_acc,
                'acc_drops':    acc_drops,
                'sv':           sv,
                'w2_col':       w2,
                'composite':    comp,
                'r_sv':         r_sv,
                'r_w2':         r_w2,
                'r_comp':       r_comp,
                'p_sv':         p_sv,
                'p_w2':         p_w2,
                'p_comp':       p_comp,
                'rho_sv':       rho_sv,
                'rho_w2':       rho_w2,
                'rho_comp':     rho_comp,
            }

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: Mean correlations over seeds")
    print(f"{'='*70}")
    print(f"{'Width':>8}  {'r(SV)':>8}  {'r(W2)':>8}  {'r(Comp)':>9}  "
          f"{'rho(SV)':>8}  {'rho(W2)':>8}  {'rho(Comp)':>9}")
    for width in WIDTHS:
        r_sv_mean   = np.mean([all_results[(width, s)]['r_sv']   for s in SEEDS])
        r_w2_mean   = np.mean([all_results[(width, s)]['r_w2']   for s in SEEDS])
        r_comp_mean = np.mean([all_results[(width, s)]['r_comp'] for s in SEEDS])
        rho_sv_mean   = np.mean([all_results[(width, s)]['rho_sv']   for s in SEEDS])
        rho_w2_mean   = np.mean([all_results[(width, s)]['rho_w2']   for s in SEEDS])
        rho_comp_mean = np.mean([all_results[(width, s)]['rho_comp'] for s in SEEDS])
        print(f"  {width:>6}  {r_sv_mean:>8.4f}  {r_w2_mean:>8.4f}  {r_comp_mean:>9.4f}  "
              f"{rho_sv_mean:>8.4f}  {rho_w2_mean:>8.4f}  {rho_comp_mean:>9.4f}")

    # =========================================================================
    # Plot
    # =========================================================================
    n_plots = len(WIDTHS) * len(SEEDS)
    fig, axes = plt.subplots(len(WIDTHS), len(SEEDS) * 3, figsize=(18, 5 * len(WIDTHS)))
    if len(WIDTHS) == 1:
        axes = axes[np.newaxis, :]

    for wi, width in enumerate(WIDTHS):
        for si, seed in enumerate(SEEDS):
            r = all_results[(width, seed)]
            col_offset = si * 3

            # SV vs acc_drop
            ax = axes[wi][col_offset + 0]
            ax.scatter(r['sv'], r['acc_drops'], alpha=0.7, color='steelblue', s=40)
            ax.set_xlabel('W1 row norm (SV proxy)')
            ax.set_ylabel('Acc drop (leave-one-out)')
            ax.set_title(f'SV vs acc_drop\nw={width}, s={seed}\nr={r["r_sv"]:.3f}')
            ax.grid(True, alpha=0.3)

            # W2_col vs acc_drop
            ax = axes[wi][col_offset + 1]
            ax.scatter(r['w2_col'], r['acc_drops'], alpha=0.7, color='darkorange', s=40)
            ax.set_xlabel('‖W2 col‖')
            ax.set_ylabel('Acc drop (leave-one-out)')
            ax.set_title(f'‖W2_col‖ vs acc_drop\nw={width}, s={seed}\nr={r["r_w2"]:.3f}')
            ax.grid(True, alpha=0.3)

            # Composite vs acc_drop
            ax = axes[wi][col_offset + 2]
            ax.scatter(r['composite'], r['acc_drops'], alpha=0.7, color='darkgreen', s=40)
            ax.set_xlabel('SV × ‖W2 col‖')
            ax.set_ylabel('Acc drop (leave-one-out)')
            ax.set_title(f'Composite vs acc_drop\nw={width}, s={seed}\nr={r["r_comp"]:.3f}')
            ax.grid(True, alpha=0.3)

    plt.suptitle('Composite Pruning Criterion: SV × ‖W2_col‖ vs SV Alone', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'composite_pruning.png'), dpi=150)
    print("\nPlot saved to results/test_U/composite_pruning.png")

    # Bar chart of correlations
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    criteria_labels = ['SV (row norm)', '‖W2_col‖', 'Composite']
    colors = ['steelblue', 'darkorange', 'darkgreen']
    x = np.arange(len(criteria_labels))
    width_bar = 0.35

    for ax_idx, (corr_type, key_pearson, key_spearman) in enumerate([
        ('Pearson r',  ['r_sv', 'r_w2', 'r_comp'], ['rho_sv', 'rho_w2', 'rho_comp']),
        ('Spearman rho', ['r_sv', 'r_w2', 'r_comp'], ['rho_sv', 'rho_w2', 'rho_comp'])
    ]):
        ax = axes2[ax_idx]
        for w_idx, width in enumerate(WIDTHS):
            if ax_idx == 0:
                vals = [np.mean([all_results[(width, s)][k] for s in SEEDS])
                        for k in ['r_sv', 'r_w2', 'r_comp']]
            else:
                vals = [np.mean([all_results[(width, s)][k] for s in SEEDS])
                        for k in ['rho_sv', 'rho_w2', 'rho_comp']]
            offset = (w_idx - len(WIDTHS)/2 + 0.5) * width_bar
            ax.bar(x + offset, vals, width_bar,
                   label=f'width={width}', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(criteria_labels)
        ax.set_ylabel(f'{["Pearson r", "Spearman rho"][ax_idx]} with acc_drop')
        ax.set_title(f'{["Pearson r", "Spearman rho"][ax_idx]}: Criterion vs Accuracy Drop')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_bars.png'), dpi=150)
    print("Plot saved to results/test_U/correlation_bars.png")

    # =========================================================================
    # Save results.md
    # =========================================================================
    def mk_table():
        header = '| Width | Seed | r(SV) | r(W2_col) | r(Composite) | rho(SV) | rho(W2_col) | rho(Composite) |'
        sep    = '|---|---|---|---|---|---|---|---|'
        rows   = []
        for width in WIDTHS:
            for seed in SEEDS:
                r = all_results[(width, seed)]
                rows.append(
                    f"| {width} | {seed} | {r['r_sv']:.4f} | {r['r_w2']:.4f} | "
                    f"{r['r_comp']:.4f} | {r['rho_sv']:.4f} | {r['rho_w2']:.4f} | "
                    f"{r['rho_comp']:.4f} |"
                )
        # Mean rows
        for width in WIDTHS:
            r_sv_m   = np.mean([all_results[(width, s)]['r_sv']   for s in SEEDS])
            r_w2_m   = np.mean([all_results[(width, s)]['r_w2']   for s in SEEDS])
            r_comp_m = np.mean([all_results[(width, s)]['r_comp'] for s in SEEDS])
            rho_sv_m   = np.mean([all_results[(width, s)]['rho_sv']   for s in SEEDS])
            rho_w2_m   = np.mean([all_results[(width, s)]['rho_w2']   for s in SEEDS])
            rho_comp_m = np.mean([all_results[(width, s)]['rho_comp'] for s in SEEDS])
            rows.append(
                f"| **{width} (mean)** | — | **{r_sv_m:.4f}** | **{r_w2_m:.4f}** | "
                f"**{r_comp_m:.4f}** | **{rho_sv_m:.4f}** | **{rho_w2_m:.4f}** | "
                f"**{rho_comp_m:.4f}** |"
            )
        return '\n'.join([header, sep] + rows)

    # Determine verdict
    mean_r_sv   = np.mean([all_results[(w, s)]['r_sv']   for w in WIDTHS for s in SEEDS])
    mean_r_w2   = np.mean([all_results[(w, s)]['r_w2']   for w in WIDTHS for s in SEEDS])
    mean_r_comp = np.mean([all_results[(w, s)]['r_comp'] for w in WIDTHS for s in SEEDS])

    best = max([('SV', mean_r_sv), ('W2_col', mean_r_w2), ('Composite', mean_r_comp)],
               key=lambda x: x[1])
    improvement = mean_r_comp - mean_r_sv

    if mean_r_comp > mean_r_sv + 0.05:
        verdict = (f"Composite criterion SV × ‖W2_col‖ outperforms SV alone by "
                   f"{improvement:+.4f} Pearson r. The W2 column norm carries additional "
                   f"information about neuron importance beyond the input representation breadth.")
    elif mean_r_comp > mean_r_sv:
        verdict = (f"Composite criterion SV × ‖W2_col‖ slightly outperforms SV alone "
                   f"(Δr = {improvement:+.4f}). The improvement is modest, suggesting SV "
                   f"already captures most of the pruning-relevant information.")
    elif mean_r_w2 > mean_r_sv:
        verdict = (f"‖W2_col‖ alone (r={mean_r_w2:.4f}) outperforms SV alone (r={mean_r_sv:.4f}). "
                   f"Output influence dominates input representation for predicting pruning impact.")
    else:
        verdict = (f"SV alone (r={mean_r_sv:.4f}) is the best single predictor. "
                   f"Composite and W2_col norms do not improve prediction.")

    results_text = f"""# Test U -- Composite Pruning Criterion

## Setup
- Model: IsotropicMLP [3072->width->10]
- Epochs: {EPOCHS}, lr={LR}, batch={BATCH}
- Widths: {WIDTHS}, Seeds: {SEEDS}
- Device: CPU
- Pruning method: leave-one-out (zero out W1 row, W2 col, b1 entry)
- Criteria compared: SV (W1 row norm), ‖W2_col‖, Composite = SV × ‖W2_col‖

## Question
Does SV × ‖W2_col‖ predict pruning impact better than SV alone?
Test G found r(SV, acc_drop) = 0.77 for a different protocol.

## Results

{mk_table()}

## Key Correlations (mean over widths × seeds)
- r(SV, acc_drop) = {mean_r_sv:.4f}
- r(‖W2_col‖, acc_drop) = {mean_r_w2:.4f}
- r(Composite, acc_drop) = {mean_r_comp:.4f}
- Composite improvement over SV: {improvement:+.4f}

## Verdict
{verdict}

## Comparison with Test G
Test G used SV from full W1 SVD decomposition on a different protocol.
This test uses W1 row norms as the per-neuron SV proxy (directly
interpretable as the "magnitude" of each neuron's input filter).

![Composite pruning scatter plots](composite_pruning.png)
![Correlation comparison](correlation_bars.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print("Results saved to results/test_U/results.md")


if __name__ == '__main__':
    main()
