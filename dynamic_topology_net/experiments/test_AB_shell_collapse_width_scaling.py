"""
Test AB -- Shell Collapse Width Scaling
=========================================
Tests O and T found that CollapsingIso has affine residuals of 0.32-0.81,
not ~0 as Appendix C predicts. This is surprising: the paper's proof should
give exact collapse under certain conditions.

Appendix C's proof assumes:
  1. Inputs on the unit sphere (||x|| = 1)
  2. Hyperspherical normalisation applied after each activation

Test T showed that L2-normalising inputs makes the residual GROW (0.32->0.51
at 1L, 0.74->0.81 at 2L). This is the opposite of what the proof predicts.

Two possible explanations:
  (A) Finite-width effect: the proof holds asymptotically as width->inf.
      At small widths, finite-size fluctuations preserve nonlinearity.
      Prediction: residual decays towards 0 as width increases.

  (B) Genuine theoretical gap: the proof has an unstated assumption
      or the specific form of iso-tanh + normalisation doesn't collapse
      at finite scale regardless of width.
      Prediction: residual stays roughly constant or grows with width.

This experiment sweeps widths [32, 64, 128, 256] and measures the affine
residual on the full 10K test set (overdetermined regression, unlike Test F's
underdetermined N=128 measurement).

Also tests: raw inputs vs L2-normalised inputs at each width.

Width: [32, 64, 128, 256], seed=42, epochs=24, Device: CPU/GPU auto.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_topology_net.core import load_cifar10
from dynamic_topology_net.core.train_utils import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AB')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS  = 24
LR      = 0.08
BATCH   = 128
WIDTHS  = [32, 64, 128, 256]
SEED    = 42


# =============================================================================
# Models
# =============================================================================

class CollapsingIso(nn.Module):
    """IsotropicTanh + HypersphericalNorm. Should collapse to affine per Appendix C."""
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        self.W1 = nn.Linear(input_dim, width)
        self.W2 = nn.Linear(width, num_classes)

    def forward(self, x):
        h    = self.W1(x)
        norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a    = torch.tanh(norm) * h / norm   # isotropic tanh
        a    = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # unit sphere
        return self.W2(a)


class IsoNoCollapse(nn.Module):
    """Standard IsotropicMLP for reference accuracy."""
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(width, input_dim))
        self.b1 = nn.Parameter(torch.zeros(width))
        self.W2 = nn.Parameter(torch.empty(num_classes, width))
        self.b2 = nn.Parameter(torch.zeros(num_classes))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

    def forward(self, x):
        h    = F.linear(x, self.W1, self.b1)
        norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a    = torch.tanh(norm) * h / norm
        return F.linear(a, self.W2, self.b2)


# =============================================================================
# Affine residual measurement
# =============================================================================

def measure_affine_residual(model, test_loader, device, l2_normalise_inputs=False):
    """
    Fit best affine map (output = A * input + b) to model outputs.
    Returns (mean_abs_residual, relative_residual).

    Uses full test set (10K samples), so regression is overdetermined
    (N=10000 >> input_dim+1=3073). Residual measures true nonlinearity.
    """
    model.eval()
    all_inputs  = []
    all_outputs = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            if l2_normalise_inputs:
                x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            out = model(x)
            all_inputs.append(x.cpu())
            all_outputs.append(out.cpu())

    X = torch.cat(all_inputs,  dim=0).numpy()   # (N, 3072)
    Y = torch.cat(all_outputs, dim=0).numpy()   # (N, 10)
    N = X.shape[0]

    # Augment inputs with bias column: X_aug = [X | 1]
    X_aug = np.concatenate([X, np.ones((N, 1))], axis=1)  # (N, 3073)

    # Least-squares: W_fit = argmin ||X_aug W - Y||^2
    # Since N=10000 > 3073, this is overdetermined -- residual is meaningful
    W_fit, residuals_sq, rank, _ = np.linalg.lstsq(X_aug, Y, rcond=None)

    Y_pred = X_aug @ W_fit
    residual_abs = np.abs(Y - Y_pred).mean()
    output_mag   = np.abs(Y).mean()
    relative     = residual_abs / (output_mag + 1e-8)

    return float(residual_abs), float(relative), int(rank)


# =============================================================================
# Main
# =============================================================================

def main():
    print(f'Device: {DEVICE}')
    print(f'Widths: {WIDTHS}')
    print(f'Hypothesis A (finite-size): residual decays toward 0 as width increases')
    print(f'Hypothesis B (genuine gap): residual stays constant or grows')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    records = []  # (width, model_type, input_type, acc, abs_resid, rel_resid, rank)

    total = len(WIDTHS) * 2  # collapsing + iso_reference
    run = 0

    for width in WIDTHS:
        for model_name, model_cls in [
            ('CollapsingIso', CollapsingIso),
            ('Iso-NoCollapse', IsoNoCollapse),
        ]:
            run += 1
            print(f'\n[{run}/{total}] width={width}  {model_name}')
            torch.manual_seed(SEED)
            model = model_cls(input_dim, width, num_classes).to(DEVICE)

            optimizer = optim.Adam(model.parameters(), lr=LR)
            criterion = nn.CrossEntropyLoss()
            for ep in range(1, EPOCHS + 1):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    criterion(model(x), y).backward()
                    optimizer.step()
                if ep in (1, 12, EPOCHS):
                    acc = evaluate(model, test_loader, DEVICE)
                    print(f'  Epoch {ep:2d}/{EPOCHS}  acc={acc:.4f}')

            final_acc = evaluate(model, test_loader, DEVICE)
            print(f'  Final acc: {final_acc:.4f}')

            # Measure affine residual: raw inputs
            abs_r, rel_r, rank = measure_affine_residual(
                model, test_loader, DEVICE, l2_normalise_inputs=False)
            print(f'  Affine residual (raw inputs):  '
                  f'abs={abs_r:.6f}  rel={rel_r:.4f}  rank={rank}')
            records.append((width, model_name, 'raw', final_acc, abs_r, rel_r, rank))

            # Measure affine residual: L2-normalised inputs
            abs_r_l2, rel_r_l2, rank_l2 = measure_affine_residual(
                model, test_loader, DEVICE, l2_normalise_inputs=True)
            print(f'  Affine residual (L2 inputs):   '
                  f'abs={abs_r_l2:.6f}  rel={rel_r_l2:.4f}  rank={rank_l2}')
            records.append((width, model_name, 'l2_norm', final_acc,
                            abs_r_l2, rel_r_l2, rank_l2))

    # =========================================================================
    # Summary
    # =========================================================================
    print(f'\n{"="*70}')
    print('SUMMARY: Affine Residual vs Width (CollapsingIso)')
    print(f'{"="*70}')
    print(f'{"Width":>8}  {"Input":>10}  {"Acc":>6}  {"Abs resid":>10}  {"Rel resid":>10}')
    for (width, model_name, inp_type, acc, abs_r, rel_r, rank) in records:
        if model_name == 'CollapsingIso':
            print(f'{width:>8}  {inp_type:>10}  {acc:.4f}  {abs_r:>10.6f}  {rel_r:>10.4f}')

    # Assess hypothesis
    coll_raw = [(w, rel_r) for (w, mn, it, acc, ar, rel_r, rank) in records
                if mn == 'CollapsingIso' and it == 'raw']
    coll_raw.sort()
    widths_r   = [w for w, _ in coll_raw]
    resids_r   = [r for _, r in coll_raw]

    if len(resids_r) >= 2:
        slope = (resids_r[-1] - resids_r[0]) / (widths_r[-1] - widths_r[0])
        if slope < -0.0005:
            hypothesis = f'SUPPORTS Hypothesis A (finite-size): residual decays with width (slope={slope:.6f})'
        elif abs(slope) < 0.0005:
            hypothesis = f'INCONCLUSIVE: residual flat with width (slope={slope:.6f})'
        else:
            hypothesis = f'SUPPORTS Hypothesis B (genuine gap): residual grows with width (slope={slope:.6f})'
        print(f'\n{hypothesis}')

    # =========================================================================
    # Plot
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: relative residual vs width
    ax = axes[0]
    for inp_type, color, ls in [('raw', 'steelblue', '-'), ('l2_norm', 'darkorange', '--')]:
        dat = [(w, rel_r) for (w, mn, it, acc, ar, rel_r, rank) in records
               if mn == 'CollapsingIso' and it == inp_type]
        dat.sort()
        if dat:
            ax.plot([d[0] for d in dat], [d[1] for d in dat],
                    marker='o', color=color, ls=ls,
                    label=f'CollapsingIso ({inp_type})')
    ax.set_xlabel('Width')
    ax.set_ylabel('Relative affine residual')
    ax.set_title('Shell Collapse: Residual vs Width\n(→0 = Hypothesis A confirmed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # Right: accuracy vs width for both models
    ax = axes[1]
    for model_name, color in [('CollapsingIso', 'crimson'), ('Iso-NoCollapse', 'steelblue')]:
        dat = [(w, acc) for (w, mn, it, acc, ar, rel_r, rank) in records
               if mn == model_name and it == 'raw']
        dat.sort()
        dat = list({w: acc for w, acc in dat}.items())  # deduplicate
        dat.sort()
        ax.plot([d[0] for d in dat], [d[1] for d in dat],
                marker='o', color=color, label=model_name)
    ax.set_xlabel('Width')
    ax.set_ylabel('Test accuracy')
    ax.set_title('Accuracy vs Width')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    plt.suptitle('Test AB: Shell Collapse Width Scaling', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'width_scaling.png'), dpi=150)
    print('\nPlot saved to results/test_AB/width_scaling.png')

    # =========================================================================
    # Save results.md
    # =========================================================================
    table_rows = '\n'.join(
        f'| {w} | {mn} | {it} | {acc:.4f} | {abs_r:.6f} | {rel_r:.4f} |'
        for (w, mn, it, acc, abs_r, rel_r, rank) in records
        if mn == 'CollapsingIso'
    )

    md = f"""# Test AB -- Shell Collapse Width Scaling

## Setup
- Model: CollapsingIso [3072->width->10] (IsotropicTanh + HypersphericalNorm)
- Widths: {WIDTHS}
- Epochs: {EPOCHS}, lr={LR}, batch={BATCH}, seed={SEED}
- Affine residual measured on full 10K test set (overdetermined, unlike Test F)
- Device: {DEVICE}

## Question
Is the non-zero affine residual from Tests O/T a finite-size effect
(vanishes as width->inf) or a genuine theoretical gap?

Hypothesis A (finite-size): residual decays to 0 as width increases.
Hypothesis B (genuine gap): residual stays constant or grows.

## Results

| Width | Model | Input | Acc | Abs Residual | Rel Residual |
|---|---|---|---|---|---|
{table_rows}

## Verdict
{hypothesis if 'hypothesis' in dir() else 'See slope data above.'}

## Connection to Tests O and T
- Test O (width=32, raw inputs): CollapsingIso-1L rel_resid = 0.245
- Test T (width=32, L2 inputs):  CollapsingIso-1L rel_resid = 0.308
- Test T found L2 inputs INCREASED residual -- opposite of paper's prediction.
- This test checks whether the residual scales with width to resolve the discrepancy.

![Width scaling results](width_scaling.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    print('Results saved to results/test_AB/results.md')


if __name__ == '__main__':
    main()
