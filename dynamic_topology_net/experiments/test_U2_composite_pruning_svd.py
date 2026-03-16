"""
Test U2 -- Composite Pruning Criterion: Proper SVD Singular Values
===================================================================
Test U had a methodology flaw: it used W1 row norms as an "SV proxy",
but the paper's pruning criterion is the actual SVD singular values
Sigma_ii from the partial diagonalisation W1 = U Sigma V^T.

Row norms and singular values are fundamentally different. This test
corrects the methodology by:

  1. Training IsotropicMLP [3072->width->10] to convergence.
  2. Performing the partial left-diagonalisation as per Eqn. 25:
       W1' = Sigma V^T    (diagonal x V^T)
       W2' = W2 U         (W2 rotated by left singular vectors)
       b1' = U^T b1
     This is the actual reparameterisation the paper recommends.
  3. In the diagonalised basis, neuron j maps to:
       - SV_j = Sigma_jj  (j-th diagonal entry, ordered descending)
       - W2'_col_j = ||W2'[:, j]||  (output weight norm in diag basis)
       - composite_j = SV_j x W2'_col_j
  4. Leave-one-out pruning: for each neuron j, zero out its row in W1'
     and its column in W2', evaluate accuracy drop.
  5. Compare Pearson r between each criterion and acc_drop:
       r(SV, acc_drop)         -- paper's actual criterion (Test G proxy)
       r(W2'_col, acc_drop)    -- output influence in diagonalised basis
       r(composite, acc_drop)  -- product criterion

This is a direct, fair test of whether the paper's SVD-based criterion
can be improved by incorporating output connection strength.

Widths: [16, 24], Seeds: [42, 123], Device: CPU
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_U2')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED   = 42
EPOCHS = 24
LR     = 0.08
BATCH  = 128
WIDTHS = [16, 24]
SEEDS  = [42, 123]
DEVICE = torch.device('cpu')


def diagonalise(model):
    """
    Apply partial left-diagonalisation as per the paper (Eqn. 25):
      W1 = U Sigma V^T  ->  W1' = Sigma V^T,  W2' = W2 U,  b1' = U^T b1

    Returns a deep-copied model with diagonalised parameters and:
      svs: array of singular values (length=width, descending order)
      The model's W1 is now Sigma V^T, W2 is W2 U, b1 is U^T b1.
    """
    model_d = copy.deepcopy(model)
    with torch.no_grad():
        W1 = model_d.W1.data   # (width, input_dim)
        W2 = model_d.W2.data   # (num_classes, width)
        b1 = model_d.b1.data   # (width,)

        # SVD: W1 = U Sigma V^T
        # torch.linalg.svd returns U (width x width), S (width,), Vh (width x input_dim)
        U, S, Vh = torch.linalg.svd(W1, full_matrices=False)
        # U: (width, width), S: (width,), Vh: (width, input_dim)

        # Reparameterise
        W1_prime = torch.diag(S) @ Vh       # Sigma V^T: (width, input_dim)
        W2_prime = W2 @ U                    # W2 U: (num_classes, width)
        b1_prime = U.T @ b1                  # U^T b1: (width,)

        model_d.W1.data = W1_prime
        model_d.W2.data = W2_prime
        model_d.b1.data = b1_prime

    return model_d, S.numpy()


def leave_one_out_pruning_diag(model_d, svs, test_loader, device):
    """
    In the diagonalised basis, neuron j has singular value svs[j].
    Pruning neuron j = zeroing its W1' row and W2' column.

    Returns baseline_acc and acc_drops array.
    """
    baseline_acc = evaluate(model_d, test_loader, device)
    width = model_d.width
    acc_drops = np.zeros(width)

    with torch.no_grad():
        for j in range(width):
            # Save
            w1_row = model_d.W1.data[j].clone()
            w2_col = model_d.W2.data[:, j].clone()
            b1_j   = model_d.b1.data[j].clone()

            # Zero out neuron j
            model_d.W1.data[j]    = 0.0
            model_d.W2.data[:, j] = 0.0
            model_d.b1.data[j]    = 0.0

            pruned_acc = evaluate(model_d, test_loader, device)
            acc_drops[j] = baseline_acc - pruned_acc

            # Restore
            model_d.W1.data[j]    = w1_row
            model_d.W2.data[:, j] = w2_col
            model_d.b1.data[j]    = b1_j

    return baseline_acc, acc_drops


def pearson_r(x, y):
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Widths: {WIDTHS}, Seeds: {SEEDS}, Epochs: {EPOCHS}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    all_results = {}

    total_runs = len(WIDTHS) * len(SEEDS)
    run_idx = 0

    for width in WIDTHS:
        for seed in SEEDS:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] width={width} seed={seed}")

            # Train
            torch.manual_seed(seed)
            model = IsotropicMLP(input_dim, width, num_classes).to(DEVICE)
            train_model(model, train_loader, test_loader, EPOCHS, LR, DEVICE, verbose=False)

            # Verify diagonalisation preserves function
            acc_before = evaluate(model, test_loader, DEVICE)
            model_d, svs = diagonalise(model)
            acc_after = evaluate(model_d, test_loader, DEVICE)
            print(f"  Acc before diag: {acc_before:.4f}  after diag: {acc_after:.4f}  "
                  f"diff: {abs(acc_before - acc_after):.6f}")
            assert abs(acc_before - acc_after) < 0.001, \
                f"Diagonalisation changed accuracy by {abs(acc_before-acc_after):.4f}!"

            # Criteria in diagonalised basis
            sv        = svs                                           # Sigma_ii (descending)
            w2_col    = model_d.W2.data.norm(dim=0).numpy()          # ||W2'[:,j]||
            composite = sv * w2_col                                   # Sigma_ii x ||W2'[:,j]||

            print(f"  SV range: [{sv.min():.4f}, {sv.max():.4f}]  "
                  f"W2' col range: [{w2_col.min():.4f}, {w2_col.max():.4f}]")

            # Leave-one-out
            print(f"  Running leave-one-out pruning ({width} neurons)...")
            baseline_acc, acc_drops = leave_one_out_pruning_diag(
                model_d, svs, test_loader, DEVICE)
            print(f"  Baseline acc: {baseline_acc:.4f}")
            print(f"  Max acc drop: {acc_drops.max():.4f} (neuron {acc_drops.argmax()})")
            print(f"  Min SV neuron: {sv.argmin()} | "
                  f"Max acc_drop neuron: {acc_drops.argmax()}")

            # Correlations
            r_sv,   p_sv   = pearson_r(sv,        acc_drops)
            r_w2,   p_w2   = pearson_r(w2_col,    acc_drops)
            r_comp, p_comp = pearson_r(composite, acc_drops)

            rho_sv,   _ = stats.spearmanr(sv,        acc_drops)
            rho_w2,   _ = stats.spearmanr(w2_col,    acc_drops)
            rho_comp, _ = stats.spearmanr(composite, acc_drops)

            print(f"  Pearson r:    SV={r_sv:.4f}  W2_col={r_w2:.4f}  Composite={r_comp:.4f}")
            print(f"  Spearman rho: SV={rho_sv:.4f}  W2_col={rho_w2:.4f}  Composite={rho_comp:.4f}")

            all_results[(width, seed)] = {
                'baseline_acc': baseline_acc,
                'acc_drops':    acc_drops,
                'sv':           sv,
                'w2_col':       w2_col,
                'composite':    composite,
                'r_sv':         r_sv,   'p_sv':   p_sv,
                'r_w2':         r_w2,   'p_w2':   p_w2,
                'r_comp':       r_comp, 'p_comp': p_comp,
                'rho_sv':       float(rho_sv),
                'rho_w2':       float(rho_w2),
                'rho_comp':     float(rho_comp),
            }

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*75}")
    print("SUMMARY: Mean correlations over seeds")
    print(f"{'='*75}")
    print(f"{'Width':>8}  {'r(SV)':>8}  {'r(W2)':>8}  {'r(Comp)':>9}  "
          f"{'rho(SV)':>8}  {'rho(W2)':>8}  {'rho(Comp)':>9}")
    for width in WIDTHS:
        r_sv_m    = np.mean([all_results[(width, s)]['r_sv']    for s in SEEDS])
        r_w2_m    = np.mean([all_results[(width, s)]['r_w2']    for s in SEEDS])
        r_comp_m  = np.mean([all_results[(width, s)]['r_comp']  for s in SEEDS])
        rho_sv_m  = np.mean([all_results[(width, s)]['rho_sv']  for s in SEEDS])
        rho_w2_m  = np.mean([all_results[(width, s)]['rho_w2']  for s in SEEDS])
        rho_comp_m= np.mean([all_results[(width, s)]['rho_comp']for s in SEEDS])
        print(f"  {width:>6}  {r_sv_m:>8.4f}  {r_w2_m:>8.4f}  {r_comp_m:>9.4f}  "
              f"{rho_sv_m:>8.4f}  {rho_w2_m:>8.4f}  {rho_comp_m:>9.4f}")

    # =========================================================================
    # Plot
    # =========================================================================
    fig, axes = plt.subplots(len(WIDTHS), len(SEEDS) * 3,
                             figsize=(18, 5 * len(WIDTHS)))
    if len(WIDTHS) == 1:
        axes = axes[np.newaxis, :]

    for wi, width in enumerate(WIDTHS):
        for si, seed in enumerate(SEEDS):
            r = all_results[(width, seed)]
            col_offset = si * 3

            for ax_i, (xdata, xlabel, color, title) in enumerate([
                (r['sv'],        'SVD Sigma_ii (paper criterion)', 'steelblue',
                 f'SVD SV vs acc_drop\nw={width},s={seed}\nr={r["r_sv"]:.3f}'),
                (r['w2_col'],    "||W2' col|| (diag basis)",       'darkorange',
                 f"||W2' col|| vs acc_drop\nw={width},s={seed}\nr={r['r_w2']:.3f}"),
                (r['composite'], "SV x ||W2' col||",               'darkgreen',
                 f"Composite vs acc_drop\nw={width},s={seed}\nr={r['r_comp']:.3f}"),
            ]):
                ax = axes[wi][col_offset + ax_i]
                ax.scatter(xdata, r['acc_drops'], alpha=0.7, color=color, s=50)
                ax.set_xlabel(xlabel, fontsize=7)
                ax.set_ylabel('Acc drop', fontsize=8)
                ax.set_title(title, fontsize=8)
                ax.grid(True, alpha=0.3)

    plt.suptitle('Test U2: SVD-based Pruning Criteria (paper\'s actual Sigma_ii vs alternatives)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'composite_pruning_svd.png'), dpi=150)
    print("\nPlot saved to results/test_U2/composite_pruning_svd.png")

    # Bar chart
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    criteria_labels = ['SVD Sigma_ii\n(paper)', "||W2' col||\n(diag basis)", 'Composite\nSV x W2_col']
    x = np.arange(len(criteria_labels))
    width_bar = 0.35

    for ax_idx, keys in enumerate([['r_sv', 'r_w2', 'r_comp'],
                                    ['rho_sv', 'rho_w2', 'rho_comp']]):
        ax = axes2[ax_idx]
        for w_idx, width in enumerate(WIDTHS):
            vals = [np.mean([all_results[(width, s)][k] for s in SEEDS]) for k in keys]
            offset = (w_idx - len(WIDTHS)/2 + 0.5) * width_bar
            ax.bar(x + offset, vals, width_bar, label=f'width={width}', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(criteria_labels, fontsize=9)
        ax.set_ylabel(['Pearson r', 'Spearman rho'][ax_idx] + ' with acc_drop')
        ax.set_title(['Pearson r', 'Spearman rho'][ax_idx] + ': Criterion vs Accuracy Drop')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        # Reference line at Test G's result
        ax.axhline(0.77, linestyle='--', color='red', alpha=0.5,
                   label='Test G r=0.77 (reference)')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_bars_svd.png'), dpi=150)
    print("Plot saved to results/test_U2/correlation_bars_svd.png")

    # =========================================================================
    # Save results.md
    # =========================================================================
    def mk_table():
        header = ('| Width | Seed | r(SVD SV) | r(W2_col) | r(Composite) |'
                  ' rho(SV) | rho(W2_col) | rho(Composite) |')
        sep = '|---|---|---|---|---|---|---|---|'
        rows = []
        for width in WIDTHS:
            for seed in SEEDS:
                r = all_results[(width, seed)]
                rows.append(
                    f"| {width} | {seed} | {r['r_sv']:.4f} | {r['r_w2']:.4f} | "
                    f"{r['r_comp']:.4f} | {r['rho_sv']:.4f} | {r['rho_w2']:.4f} | "
                    f"{r['rho_comp']:.4f} |"
                )
        for width in WIDTHS:
            r_sv_m   = np.mean([all_results[(width,s)]['r_sv']   for s in SEEDS])
            r_w2_m   = np.mean([all_results[(width,s)]['r_w2']   for s in SEEDS])
            r_comp_m = np.mean([all_results[(width,s)]['r_comp'] for s in SEEDS])
            rho_sv_m   = np.mean([all_results[(width,s)]['rho_sv']   for s in SEEDS])
            rho_w2_m   = np.mean([all_results[(width,s)]['rho_w2']   for s in SEEDS])
            rho_comp_m = np.mean([all_results[(width,s)]['rho_comp'] for s in SEEDS])
            rows.append(
                f"| **{width} (mean)** | — | **{r_sv_m:.4f}** | **{r_w2_m:.4f}** | "
                f"**{r_comp_m:.4f}** | **{rho_sv_m:.4f}** | **{rho_w2_m:.4f}** | "
                f"**{rho_comp_m:.4f}** |"
            )
        return '\n'.join([header, sep] + rows)

    mean_r_sv   = np.mean([all_results[(w,s)]['r_sv']   for w in WIDTHS for s in SEEDS])
    mean_r_w2   = np.mean([all_results[(w,s)]['r_w2']   for w in WIDTHS for s in SEEDS])
    mean_r_comp = np.mean([all_results[(w,s)]['r_comp'] for w in WIDTHS for s in SEEDS])

    improvement_over_sv = mean_r_comp - mean_r_sv

    if mean_r_comp > mean_r_sv + 0.05:
        verdict = (f"Composite SV x ||W2_col|| (r={mean_r_comp:.4f}) outperforms the paper's "
                   f"SVD criterion (r={mean_r_sv:.4f}) by {improvement_over_sv:+.4f}. "
                   f"Output connection strength carries additional pruning-relevant "
                   f"information beyond the singular value alone.")
    elif mean_r_comp > mean_r_sv:
        verdict = (f"Composite SV x ||W2_col|| (r={mean_r_comp:.4f}) modestly outperforms "
                   f"the paper's SVD criterion (r={mean_r_sv:.4f}, delta={improvement_over_sv:+.4f}). "
                   f"The SVD criterion is already effective; W2_col adds marginal improvement.")
    elif mean_r_w2 > mean_r_sv:
        verdict = (f"||W2_col|| alone (r={mean_r_w2:.4f}) outperforms SVD singular values "
                   f"(r={mean_r_sv:.4f}) even in the diagonalised basis. Output influence "
                   f"dominates input representation strength for predicting pruning impact.")
    else:
        verdict = (f"The paper's SVD criterion (r={mean_r_sv:.4f}) is the best or equal "
                   f"predictor. Composite and W2_col norms do not offer improvement over "
                   f"the paper's recommended singular-value thresholding.")

    results_text = f"""# Test U2 -- Composite Pruning Criterion (Corrected: SVD Sigma_ii)

## Correction from Test U
Test U used W1 row norms as an "SV proxy." This was incorrect.
The paper's criterion is the actual SVD singular values Sigma_ii from the
partial diagonalisation W1 = U Sigma V^T (Eqn. 25 of the paper).
Row norms and singular values are fundamentally different quantities.

This test performs the proper diagonalisation and uses the actual Sigma_ii.

## Setup
- Model: IsotropicMLP [3072->width->10]
- Epochs: {EPOCHS}, lr={LR}, batch={BATCH}
- Widths: {WIDTHS}, Seeds: {SEEDS}, Device: CPU
- Diagonalisation: W1' = Sigma V^T, W2' = W2 U, b1' = U^T b1
- Leave-one-out: zero W1'[j,:], W2'[:,j], b1'[j] for each neuron j
- Criteria:
    - SVD Sigma_ii: diagonal of Sigma (paper's actual criterion)
    - ||W2' col||: L2 norm of W2' columns (in diagonalised basis)
    - Composite: Sigma_ii x ||W2' col_i||

## Diagonalisation Verification
Function preservation verified: accuracy change < 0.001 in all runs.

## Results

{mk_table()}

## Key Correlations (mean over widths x seeds)
- r(SVD Sigma_ii, acc_drop) = {mean_r_sv:.4f}   [paper's criterion]
- r(||W2'_col||, acc_drop)  = {mean_r_w2:.4f}
- r(Composite, acc_drop)    = {mean_r_comp:.4f}
- Composite improvement over paper's criterion: {improvement_over_sv:+.4f}

## Comparison with Prior Tests
- Test G: r(SV, acc_drop) = 0.77 (different protocol: single-width, no diagonalisation)
- Test U: r(row_norm, acc_drop) = -0.12 (INCORRECT -- row norms, not SVs)
- Test U2: r(SVD Sigma_ii, acc_drop) = {mean_r_sv:.4f} (THIS TEST -- correct criterion)

## Verdict
{verdict}

![SVD pruning criteria scatter](composite_pruning_svd.png)
![Correlation comparison](correlation_bars_svd.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print("Results saved to results/test_U2/results.md")


if __name__ == '__main__':
    main()
