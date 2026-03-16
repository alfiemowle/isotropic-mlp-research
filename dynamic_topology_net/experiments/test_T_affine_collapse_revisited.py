"""
Test T -- Affine Collapse Revisited (Correcting Test F)
========================================================
Test F reported residual = 0.000000 for CollapsingIsotropicMLP, confirming
Appendix C's affine collapse claim. But Test O revealed this was a MEASUREMENT
ARTIFACT: Test F used a single batch of 128 samples to fit a linear model
with 3073 parameters (input_dim + bias). The system was underdetermined
(128 < 3073), so least-squares always gives residual = 0 regardless of
whether the model is actually affine.

Test O with the full 10K test set showed non-zero residuals (0.32 for 1L,
0.74 for 2L), meaning the model is NOT affine in raw CIFAR-10 inputs.

This test investigates: WHAT INPUTS does the collapse theorem actually require?

Appendix C's argument involves unit-sphere normalisation of intermediate
representations. The full theorem may only hold when inputs lie on the unit
hypersphere. We test three input preprocessing conditions:
  (a) Raw CIFAR-10 (standardised per-channel mean/std -- what we always use)
  (b) L2-normalised: each sample x -> x / ||x||_2  (unit sphere inputs)
  (c) Per-dimension standardised (already our default, but verify)

For each input condition, we:
  1. Train CollapsingIso-1L and CollapsingIso-2L (and Iso-1L, Iso-2L as comparison)
  2. Verify affine fit on the FULL test set (10K samples)
  3. Report residual and accuracy

If unit-sphere inputs give residual -> 0: the theorem is correct but has a
hidden domain assumption CIFAR-10 violates (inputs must lie on S^(d-1)).
If even unit-sphere inputs give non-zero residual: the collapse is weaker
than Appendix C claims, even in the theorem's intended domain.

Width=32, 24 epochs, batch=128, seed=42, CPU.
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
    IsotropicMLP, DeepIsotropicMLP,
    CollapsingIsotropicMLP, DeepCollapsingIsotropicMLP,
    load_cifar10
)
from dynamic_topology_net.core.train_utils import train_epoch, evaluate, make_optimizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_T')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED   = 42
EPOCHS = 24
LR     = 0.08
BATCH  = 128
WIDTH  = 32
DEVICE = torch.device('cpu')


class L2NormWrapper(nn.Module):
    """Wraps a model, applying L2 normalisation to inputs before forward pass."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        norms = x.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return self.model(x / norms)


def verify_affine_full(model, test_loader, device, l2_normalise=False):
    """
    Fit affine map (Y = X_aug @ A) on the FULL test set (10K samples).
    Returns mean residual, max residual, relative residual.
    This is the corrected version of the Test F measurement.
    """
    model.eval()
    all_x, all_y = [], []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            if l2_normalise:
                x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)
            y = model(x)
            all_x.append(x.view(x.shape[0], -1))
            all_y.append(y)

    X = torch.cat(all_x, dim=0)  # (10000, input_dim)
    Y = torch.cat(all_y, dim=0)  # (10000, num_classes)
    N = X.shape[0]

    print(f"    Affine verification: {N} samples, {X.shape[1]} input dims")
    print(f"    System: {N} equations, {X.shape[1]+1} unknowns "
          f"({'overdetermined -- valid test' if N > X.shape[1]+1 else 'UNDERDETERMINED -- invalid test'})")

    X_aug = torch.cat([X, torch.ones(N, 1, device=device)], dim=1)
    solution = torch.linalg.lstsq(X_aug, Y).solution
    Y_hat = X_aug @ solution

    residual = (Y - Y_hat).abs()
    mean_res = residual.mean().item()
    max_res  = residual.max().item()
    mean_mag = Y.abs().mean().item()
    rel_res  = mean_res / (mean_mag + 1e-10)
    return mean_res, max_res, rel_res, N


def train_and_eval(model, train_loader_raw, train_loader_norm, test_loader_raw, test_loader_norm,
                   epochs, lr, device, use_norm, label):
    """Train on raw or L2-normalised data and return accuracy history."""
    train_loader = train_loader_norm if use_norm else train_loader_raw
    test_loader  = test_loader_norm  if use_norm else test_loader_raw
    opt  = make_optimizer(model, lr)
    crit = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, steps = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item(); steps += 1
        acc = evaluate(model, test_loader, device)
        history.append(acc)
        if epoch % 6 == 0 or epoch == 1:
            print(f"    [{label}] Epoch {epoch:2d}/{epochs}  loss={total/steps:.4f}  acc={acc:.3f}")
    return history


class L2NormDataLoader:
    """Wraps a DataLoader to apply L2 normalisation to each batch."""
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for x, y in self.loader:
            norms = x.norm(dim=1, keepdim=True).clamp(min=1e-8)
            yield x / norms, y

    def __len__(self):
        return len(self.loader)


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print("Loading CIFAR-10...")
    train_loader_raw, test_loader_raw, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    # Create L2-normalised versions
    train_loader_norm = L2NormDataLoader(train_loader_raw)
    test_loader_norm  = L2NormDataLoader(test_loader_raw)

    results = {}

    configs = [
        # (label, model_factory, is_collapsing, use_l2_norm)
        ('CollapsingIso-1L / raw',  lambda: CollapsingIsotropicMLP(input_dim, WIDTH, num_classes), True,  False),
        ('CollapsingIso-2L / raw',  lambda: DeepCollapsingIsotropicMLP(input_dim, WIDTH, num_classes), True, False),
        ('CollapsingIso-1L / L2',   lambda: CollapsingIsotropicMLP(input_dim, WIDTH, num_classes), True,  True),
        ('CollapsingIso-2L / L2',   lambda: DeepCollapsingIsotropicMLP(input_dim, WIDTH, num_classes), True, True),
        ('Iso-1L / raw',            lambda: IsotropicMLP(input_dim, WIDTH, num_classes), False, False),
        ('Iso-2L / raw',            lambda: DeepIsotropicMLP(input_dim, WIDTH, num_classes), False, False),
        ('Iso-1L / L2',             lambda: IsotropicMLP(input_dim, WIDTH, num_classes), False, True),
        ('Iso-2L / L2',             lambda: DeepIsotropicMLP(input_dim, WIDTH, num_classes), False, True),
    ]

    for label, make_model, is_collapsing, use_norm in configs:
        print(f"\n{'='*55}\n{label}\n{'='*55}")
        torch.manual_seed(SEED)
        model = make_model().to(DEVICE)
        history = train_and_eval(
            model, train_loader_raw, train_loader_norm,
            test_loader_raw, test_loader_norm,
            EPOCHS, LR, DEVICE, use_norm, label)
        final_acc = history[-1]

        affine_res = None
        if is_collapsing:
            print(f"  Verifying affine fit on full test set...")
            # Verify on the appropriate input domain
            mean_r, max_r, rel_r, N = verify_affine_full(
                model, test_loader_raw if not use_norm else test_loader_norm,
                DEVICE, l2_normalise=False)
            affine_res = {'mean': mean_r, 'max': max_r, 'rel': rel_r, 'N': N}
            print(f"  Residual: mean={mean_r:.6f}  max={max_r:.6f}  rel={rel_r:.4f}  (N={N})")

        results[label] = {
            'history': history, 'final_acc': final_acc,
            'affine_res': affine_res, 'is_collapsing': is_collapsing
        }
        print(f"  Final accuracy: {final_acc:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':35s}  {'Acc':6s}  {'Residual (mean)':17s}  {'Relative':10s}")
    for label, r in results.items():
        res = r['affine_res']
        res_str = f"{res['mean']:.6f}" if res else "N/A"
        rel_str = f"{res['rel']:.4f}"  if res else "N/A"
        print(f"  {label:33s}  {r['final_acc']:.4f}  {res_str:17s}  {rel_str}")

    # =========================================================================
    # Plot
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epoch_x = list(range(1, EPOCHS + 1))
    color_map = {
        'Iso-1L':           'darkorange',
        'Iso-2L':           'red',
        'CollapsingIso-1L': 'steelblue',
        'CollapsingIso-2L': 'royalblue',
    }
    style_map = {'raw': '-', 'L2': '--'}

    for ax_idx, (title, keys) in enumerate([
        ('1-Layer models', [k for k in results if '1L' in k]),
        ('2-Layer models', [k for k in results if '2L' in k]),
    ]):
        ax = axes[0][ax_idx]
        for label in keys:
            r = results[label]
            model_type = label.split('/')[0].strip()
            norm_type  = label.split('/')[1].strip()
            base_color = color_map.get(model_type, 'gray')
            style = style_map.get(norm_type, '-')
            ax.plot(epoch_x, r['history'], style, color=base_color,
                    label=label, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Test Accuracy')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Residual comparison bar chart (collapsing models only)
    ax = axes[1][0]
    collapsing_labels = [l for l, r in results.items() if r['is_collapsing'] and r['affine_res']]
    residuals_mean = [results[l]['affine_res']['mean'] for l in collapsing_labels]
    bar_colors = ['steelblue' if 'raw' in l else 'royalblue' for l in collapsing_labels]
    ax.bar(range(len(collapsing_labels)), residuals_mean, color=bar_colors, alpha=0.8)
    ax.set_xticks(range(len(collapsing_labels)))
    ax.set_xticklabels([l.replace(' / ', '\n') for l in collapsing_labels], fontsize=7)
    ax.set_ylabel('Mean affine residual (full test set)')
    ax.set_title('Affine Residual: Raw vs L2-normalised Inputs')
    ax.grid(True, alpha=0.3, axis='y')

    # Iso accuracy: raw vs L2 normalised
    ax = axes[1][1]
    for model_type in ['Iso-1L', 'Iso-2L']:
        raw_key  = f'{model_type} / raw'
        norm_key = f'{model_type} / L2'
        color = color_map[model_type]
        if raw_key in results:
            ax.plot(epoch_x, results[raw_key]['history'],  '-',  color=color, label=raw_key,  linewidth=1.5)
        if norm_key in results:
            ax.plot(epoch_x, results[norm_key]['history'], '--', color=color, label=norm_key, linewidth=1.5)
    ax.set_title('Effect of L2 normalisation on Iso accuracy')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Test Accuracy')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Affine Collapse Revisited: Raw vs L2-normalised Inputs', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'affine_collapse_revisited.png'), dpi=150)
    print("\nPlot saved to results/test_T/affine_collapse_revisited.png")

    # =========================================================================
    # Save results.md
    # =========================================================================
    coll_raw_1L  = results.get('CollapsingIso-1L / raw',  {}).get('affine_res', {})
    coll_norm_1L = results.get('CollapsingIso-1L / L2',   {}).get('affine_res', {})
    coll_raw_2L  = results.get('CollapsingIso-2L / raw',  {}).get('affine_res', {})
    coll_norm_2L = results.get('CollapsingIso-2L / L2',   {}).get('affine_res', {})

    def fmt_res(d):
        if not d:
            return 'N/A'
        return f"mean={d.get('mean', float('nan')):.6f}, rel={d.get('rel', float('nan')):.4f}"

    # Determine verdict
    raw_1L_res  = coll_raw_1L.get('mean',  999)
    norm_1L_res = coll_norm_1L.get('mean', 999)
    if norm_1L_res < 0.001:
        verdict = "Unit-sphere inputs produce near-zero residual. Appendix C is correct but requires unit-normalised inputs -- a hidden assumption not stated in the paper."
    elif norm_1L_res < raw_1L_res * 0.1:
        verdict = "Unit-sphere inputs substantially reduce residual. The collapse is approximately valid for unit-normalised inputs but not for raw CIFAR-10."
    else:
        verdict = "Even unit-sphere inputs yield significant residual. The affine collapse is weaker than Appendix C claims, even in the theorem's intended domain."

    result_rows = '\n'.join(
        f"| {label} | {r['final_acc']*100:.2f}% | "
        + (fmt_res(r['affine_res']) if r['affine_res'] else 'N/A (not collapsing)')
        + " |"
        for label, r in results.items()
    )

    results_text = f"""# Test T -- Affine Collapse Revisited

## Background
Test F reported residual = 0.000000 for CollapsingIsotropicMLP, but this
was a measurement artifact: 128 samples, 3073 parameters -> underdetermined
least-squares, always gives residual = 0.

Test O corrected this by using all 10K test samples:
- CollapsingIso-1L residual: 0.322367 (NOT near zero)
- CollapsingIso-2L residual: 0.739638 (NOT near zero)

This test investigates: does unit-normalising inputs (x/||x||) make the collapse exact?

## Setup
- Width: {WIDTH}, Epochs: {EPOCHS}, lr={LR}, batch={BATCH}, seed={SEED}
- Affine verification: FULL test set ({coll_raw_1L.get('N', '?')} samples)
- Input conditions: raw CIFAR-10 vs L2-normalised (x/||x||)

## Results

| Config | Final Acc | Affine Residual |
|---|---|---|
{result_rows}

## Affine Residuals (Corrected, Full Test Set)

| Model | Raw inputs | L2-normalised inputs |
|---|---|---|
| CollapsingIso-1L | {fmt_res(coll_raw_1L)} | {fmt_res(coll_norm_1L)} |
| CollapsingIso-2L | {fmt_res(coll_raw_2L)} | {fmt_res(coll_norm_2L)} |

## Verdict
{verdict}

## What Test F Got Wrong
Test F used batch_size=128 to fit an affine model with input_dim={input_dim}+1=
parameters. Since 128 << {input_dim}+1, the least-squares system was underdetermined
and always returned residual=0, regardless of whether the model is affine.
The correct measurement requires N >> input_dim (satisfied here with N=10,000).

![Affine collapse revisited](affine_collapse_revisited.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print("Results saved to results/test_T/results.md")


if __name__ == '__main__':
    main()
