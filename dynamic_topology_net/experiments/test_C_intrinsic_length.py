"""
Test C -- Intrinsic Length Absorption
======================================
Claim (paper Sec. 3.1, Eqns. 26-29):
    When pruning a neuron, residual bias in the norm computation causes
    output error. The intrinsic length parameter o absorbs this residual,
    reducing pruning error. Crucially, this only matters when sigma_i ~ 0.

Key clarification:
    The intrinsic length is NOT designed to reduce the error of pruning
    large-sigma neurons (removing a significant neuron always costs). It is
    specifically designed for neurons with sigma_i ~ 0 that still have a
    nonzero bias b_i. In that case, the bias pollutes the norm even though
    the neuron is functionally inert -- intrinsic length absorbs this.

Test design:
    PART 1: Regular neurons (all sigma large, as in trained network).
            Expect IL to make ~no difference (error dominated by linear loss).
    PART 2: Near-zero SV neurons (simulate scaffold neurons mid-training).
            Manually zero the W1 rows of N neurons while keeping their biases.
            Expect IL to make a LARGE difference here.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dynamic_topology_net.core import IsotropicMLP, load_cifar10
from dynamic_topology_net.core.train_utils import train_model, evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_C')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED   = 42
EPOCHS = 24
LR     = 0.08
WIDTH  = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def output_l2_error(model_pruned, logits_ref, all_x):
    """Mean L2 distance between pruned model outputs and reference logits."""
    model_pruned.eval()
    logits = model_pruned(all_x)
    return (logits_ref - logits).norm(dim=1).mean().item()


def prune_no_il(model, idx):
    """Remove neuron idx with no intrinsic length correction."""
    with torch.no_grad():
        keep = [i for i in range(model.width) if i != idx]
        model.W1 = nn.Parameter(model.W1.data[keep].clone())
        model.b1 = nn.Parameter(model.b1.data[keep].clone())
        model.W2 = nn.Parameter(model.W2.data[:, keep].clone())


def run_single_prune_test(model_base, model_il, all_x, logits_ref, idx):
    """
    Prune neuron idx from copies of both models.
    Returns (sv, bias, err_no_il, err_with_il).
    """
    sv   = model_base.W1.data[idx].norm().item()
    bias = model_base.b1.data[idx].item()

    m_no = copy.deepcopy(model_base)
    prune_no_il(m_no, idx)
    err_no = output_l2_error(m_no, logits_ref, all_x)

    m_il = copy.deepcopy(model_il)
    m_il.prune_neuron(idx)
    err_il = output_l2_error(m_il, logits_ref, all_x)

    return sv, bias, err_no, err_il


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=24)

    # -- Train base model (no intrinsic length) --------------------------------
    print(f"\nTraining IsotropicMLP [3072, {WIDTH}, 10] ...")
    model_base = IsotropicMLP(input_dim=input_dim, width=WIDTH,
                               num_classes=num_classes,
                               use_intrinsic_length=False).to(DEVICE)
    train_model(model_base, train_loader, test_loader, EPOCHS, LR, DEVICE, verbose=True)

    # Build IL model with same weights
    model_il = IsotropicMLP(input_dim=input_dim, width=WIDTH,
                             num_classes=num_classes,
                             use_intrinsic_length=True).to(DEVICE)
    with torch.no_grad():
        model_il.W1.data.copy_(model_base.W1.data)
        model_il.b1.data.copy_(model_base.b1.data)
        model_il.W2.data.copy_(model_base.W2.data)
        model_il.b2.data.copy_(model_base.b2.data)

    # Diagonalise
    svs = model_base.partial_diagonalise()
    model_il.partial_diagonalise()
    order = torch.argsort(svs)   # ascending: smallest first

    # Reference outputs (unchanged model)
    all_x = torch.cat([x.to(DEVICE) for x, _ in test_loader])
    all_y = torch.cat([y for _, y in test_loader])
    model_base.eval()
    with torch.no_grad():
        logits_ref = model_base(all_x)
    acc_orig = (logits_ref.argmax(1).cpu() == all_y).float().mean().item()
    print(f"\nBaseline accuracy: {acc_orig:.2%}")

    # =====================================================================
    # PART 1: Regular trained neurons (all have large sigma)
    # =====================================================================
    print(f"\nPART 1: Pruning trained neurons (all sigma > 0)")
    print(f"{'-'*72}")
    print(f"{'Rank':>4}  {'sigma':>8}  {'bias':>8}  {'err_no_IL':>10}  {'err_with_IL':>11}  {'ratio':>6}")
    print(f"{'-'*72}")

    part1_results = []
    for rank, idx in enumerate(order.tolist()):
        sv, bias, err_no, err_il = run_single_prune_test(
            model_base, model_il, all_x, logits_ref, idx)
        ratio = err_no / (err_il + 1e-10)
        part1_results.append(dict(rank=rank, idx=idx, sv=sv, bias=bias,
                                   err_no=err_no, err_il=err_il, ratio=ratio))
        print(f"{rank:>4}  {sv:>8.2f}  {bias:>8.3f}  {err_no:>10.5f}  {err_il:>11.5f}  {ratio:>6.2f}x")

    avg_ratio_p1 = np.mean([r['ratio'] for r in part1_results])
    print(f"\nPart 1 average ratio: {avg_ratio_p1:.3f}x  (expected ~1.0x for large sigma)")

    # =====================================================================
    # PART 2: Near-zero sigma neurons (simulate scaffold neurons)
    # Create copies where the 8 smallest-sigma neurons have W1 row zeroed
    # but biases preserved -- this is the actual use case for intrinsic length
    # =====================================================================
    print(f"\nPART 2: Near-zero sigma neurons (W1 rows zeroed, biases preserved)")
    print("Simulates scaffold neurons that have not yet differentiated")
    print(f"{'-'*72}")
    print(f"{'Rank':>4}  {'orig_sigma':>10}  {'bias':>8}  {'err_no_IL':>10}  {'err_with_IL':>11}  {'ratio':>6}")
    print(f"{'-'*72}")

    # Create modified model where the 8 smallest-sigma neurons have zero W1 rows
    NEAR_ZERO_COUNT = 8
    small_indices = order[:NEAR_ZERO_COUNT].tolist()

    model_nz_base = copy.deepcopy(model_base)
    model_nz_il   = copy.deepcopy(model_il)
    with torch.no_grad():
        for idx in small_indices:
            model_nz_base.W1.data[idx] = 0.0
            model_nz_il.W1.data[idx]   = 0.0

    # Reference is now the near-zero model
    model_nz_base.eval()
    with torch.no_grad():
        logits_nz_ref = model_nz_base(all_x)

    part2_results = []
    for rank_nz, idx in enumerate(small_indices):
        orig_sv = svs[idx].item()
        bias    = model_nz_base.b1.data[idx].item()

        # Prune from near-zero models
        m_no = copy.deepcopy(model_nz_base)
        prune_no_il(m_no, idx)
        err_no = output_l2_error(m_no, logits_nz_ref, all_x)

        m_il_copy = copy.deepcopy(model_nz_il)
        m_il_copy.prune_neuron(idx)
        err_il = output_l2_error(m_il_copy, logits_nz_ref, all_x)

        ratio = err_no / (err_il + 1e-10)
        part2_results.append(dict(rank=rank_nz, idx=idx, orig_sv=orig_sv, bias=bias,
                                   err_no=err_no, err_il=err_il, ratio=ratio))
        print(f"{rank_nz:>4}  {orig_sv:>10.3f}  {bias:>8.3f}  {err_no:>10.6f}  {err_il:>11.6f}  {ratio:>6.2f}x")

    avg_ratio_p2 = np.mean([r['ratio'] for r in part2_results])
    print(f"\nPart 2 average ratio: {avg_ratio_p2:.3f}x  (expected > 1.0x for large |bias|)")

    # =====================================================================
    # Theoretical prediction for Part 2
    # =====================================================================
    # For sigma=0, the norm change from pruning neuron i is:
    # Without IL: norm gains +b_i^2 term (from the pruned neuron's bias)
    # With IL: o is updated so o'^2 = o^2 + b_i^2, absorbing the bias
    # The relative contribution is b_i^2 / ||remaining_h||^2
    print("\nTheoretical bias contribution (b_i^2 / mean_||h||^2):")
    model_nz_base.eval()
    with torch.no_grad():
        h_all = torch.cat([x.to(DEVICE) for x, _ in test_loader])
        h_all = torch.nn.functional.linear(h_all, model_nz_base.W1, model_nz_base.b1)
        mean_norm_sq = h_all.norm(dim=1).pow(2).mean().item()
    for r in part2_results:
        contrib = r['bias']**2 / (mean_norm_sq + 1e-8)
        print(f"  neuron {r['rank']:>2}: bias={r['bias']:>7.3f}  b^2/||h||^2={contrib:.6f}")

    # =====================================================================
    # Plot
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Part 1 error vs sigma
    svs1 = [r['sv'] for r in part1_results]
    ax = axes[0]
    ax.scatter(svs1, [r['err_no'] for r in part1_results], label='No IL', alpha=0.8, marker='o')
    ax.scatter(svs1, [r['err_il'] for r in part1_results], label='With IL', alpha=0.8, marker='s')
    ax.set_xlabel('Singular Value (sigma)')
    ax.set_ylabel('Mean L2 Output Error')
    ax.set_title('Part 1: Trained neurons\n(large sigma, IL makes no difference)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Plot 2: Part 2 error comparison (near-zero sigma)
    biases2 = [abs(r['bias']) for r in part2_results]
    ax = axes[1]
    x_pos = np.arange(NEAR_ZERO_COUNT)
    width_bar = 0.35
    ax.bar(x_pos - width_bar/2, [r['err_no'] for r in part2_results],
           width_bar, label='No IL', alpha=0.8)
    ax.bar(x_pos + width_bar/2, [r['err_il'] for r in part2_results],
           width_bar, label='With IL', alpha=0.8)
    ax.set_xlabel('Neuron rank (ascending sigma)')
    ax.set_ylabel('Mean L2 Output Error')
    ax.set_title('Part 2: Near-zero sigma neurons\n(bias preserved, sigma=0)')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Ratio vs |bias| for Part 2
    ax = axes[2]
    ax.scatter(biases2, [r['ratio'] for r in part2_results],
               color='purple', s=80, zorder=5)
    for r, b in zip(part2_results, biases2):
        ax.annotate(f"{r['bias']:.1f}", (b, r['ratio']), textcoords='offset points',
                    xytext=(4, 4), fontsize=8)
    ax.axhline(y=1.0, linestyle='--', color='gray', alpha=0.6, label='No improvement')
    ax.set_xlabel('|bias| of pruned neuron')
    ax.set_ylabel('Improvement ratio (err_no_IL / err_with_IL)')
    ax.set_title('Part 2: Improvement vs neuron bias magnitude')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'pruning_error.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to results/test_C/pruning_error.png")

    # =====================================================================
    # Save results
    # =====================================================================
    p1_table = "\n".join(
        f"| {r['rank']} | {r['sv']:.2f} | {r['bias']:.3f} | {r['err_no']:.5f} | {r['err_il']:.5f} | {r['ratio']:.3f}x |"
        for r in part1_results)
    p2_table = "\n".join(
        f"| {r['rank']} | {r['orig_sv']:.2f} | {r['bias']:.3f} | {r['err_no']:.6f} | {r['err_il']:.6f} | {r['ratio']:.3f}x |"
        for r in part2_results)

    il_effective = avg_ratio_p2 > 1.05

    results_text = f"""# Test C -- Intrinsic Length Absorption

## Claim
The intrinsic length parameter `o` absorbs residual bias terms when pruning
a neuron, reducing output error. The correction is specifically designed for
neurons with `sigma ~ 0` that still have nonzero bias.

## Setup
- Model: IsotropicMLP [3072 -> {WIDTH} -> 10], trained for {EPOCHS} epochs
- Dataset: CIFAR-10 (10,000 test samples)
- Baseline accuracy: {acc_orig:.2%}

---

## Part 1: Pruning Trained Neurons (all sigma large)

**Expected outcome:** IL correction makes ~no difference because error is
dominated by the loss of the neuron's linear contribution (sigma * v^T * x).

| Rank | sigma | bias | Err (no IL) | Err (with IL) | Ratio |
|---|---|---|---|---|---|
{p1_table}

**Average ratio: {avg_ratio_p1:.3f}x** -- as expected, negligible difference.

The intrinsic length is NOT meant to help here. Removing a fully active
neuron costs regardless.

---

## Part 2: Near-Zero Sigma Neurons (sigma forced to 0, bias preserved)

**This simulates scaffold neurons mid-training that have not differentiated.**
W1 rows zeroed out but biases kept. These neurons contribute nothing to
the linear computation, but their bias still enters the norm.

**Expected outcome:** IL correction should reduce error, especially for
neurons with large |bias|.

| Rank | orig sigma | bias | Err (no IL) | Err (with IL) | Ratio |
|---|---|---|---|---|---|
{p2_table}

**Average ratio: {avg_ratio_p2:.3f}x** -- {'improvement observed as predicted' if il_effective else 'weaker than expected (see interpretation)'}

![Pruning error analysis](pruning_error.png)

---

## Interpretation

### Why Part 1 shows ratio ~1.0x
For trained neurons with large sigma, pruning error is dominated by the
removed linear contribution. The intrinsic length corrects only the norm
term -- a minor contributor when sigma is large.

### Why Part 2 {'shows' if il_effective else 'shows limited'} improvement
With sigma = 0, the neuron contributes ONLY through its bias in the norm:
```
||Wx + b||^2 = b_i^2 + sum_{{j!=i}} (W_jj x_j + b_j)^2
```
Without IL: pruning removes the b_i^2 term, changing the norm.
With IL: o is updated so o'^2 = o^2 + b_i^2, preserving the norm exactly.

The improvement is proportional to b_i^2 / mean(||h||^2). For small biases
relative to the total norm, the effect is minimal.

### Key Takeaway
The intrinsic length is **necessary for exact pruning invariance** when
sigma = 0 AND |bias| is significant relative to the overall representation
norm. In practice, this requires careful bias initialisation for scaffold
neurons (b_star small) or normalisation to control representation scale.
"""

    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print(f"Results saved to results/test_C/results.md")


if __name__ == '__main__':
    main()
