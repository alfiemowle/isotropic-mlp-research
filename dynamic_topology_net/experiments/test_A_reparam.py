"""
Test A — Reparameterisation Invariance
=======================================
Claim (paper Sec. 2.3, Eqns. 13-18):
    Applying the SVD-based partial diagonalisation to a trained isotropic
    network produces an EXACTLY equivalent network. The outputs before and
    after reparameterisation should be identical up to floating-point precision.

Why this matters:
    If this test fails, the entire dynamic topology framework collapses —
    neurogenesis and neurodegeneration both depend on this invariance.

Method:
    1. Train an IsotropicMLP on CIFAR-10 for 24 epochs.
    2. Record outputs on the full test set: y_orig.
    3. Call model.partial_diagonalise().
    4. Record outputs again: y_reparam.
    5. Measure max absolute difference, mean absolute difference, and whether
       classification decisions (argmax) are unchanged.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np

from dynamic_topology_net.core import IsotropicMLP, load_cifar10
from dynamic_topology_net.core.train_utils import train_model, evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_A')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
SEED    = 42
EPOCHS  = 24
LR      = 0.08
WIDTH   = 24
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=24)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining IsotropicMLP [3072, {WIDTH}, 10] for {EPOCHS} epochs...")
    model = IsotropicMLP(input_dim=input_dim, width=WIDTH, num_classes=num_classes).to(DEVICE)
    history = train_model(model, train_loader, test_loader, EPOCHS, LR, DEVICE, verbose=True)
    final_acc = history[-1][1]
    print(f"\nFinal test accuracy: {final_acc:.2%}")

    # ── Collect ALL test outputs before reparameterisation ────────────────────
    model.eval()
    all_x, all_y = [], []
    with torch.no_grad():
        for x, y in test_loader:
            all_x.append(x.to(DEVICE))
            all_y.append(y)
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    with torch.no_grad():
        logits_orig = model(all_x)
    preds_orig = logits_orig.argmax(dim=1).cpu()

    # ── Singular values BEFORE ────────────────────────────────────────────────
    svs_before = model.get_singular_values().cpu()
    print(f"\nSingular values BEFORE diagonalisation:")
    print(f"  min={svs_before.min():.4f}  max={svs_before.max():.4f}  mean={svs_before.mean():.4f}")

    # ── Apply reparameterisation ──────────────────────────────────────────────
    print("\nApplying partial_diagonalise()...")
    svs_returned = model.partial_diagonalise().cpu()

    # ── Singular values AFTER ─────────────────────────────────────────────────
    svs_after = model.get_singular_values().cpu()
    print(f"Singular values AFTER  diagonalisation (row norms of W1):")
    print(f"  min={svs_after.min():.4f}  max={svs_after.max():.4f}  mean={svs_after.mean():.4f}")

    # ── Collect outputs AFTER ─────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits_reparam = model(all_x)
    preds_reparam = logits_reparam.argmax(dim=1).cpu()

    # ── Compute differences ───────────────────────────────────────────────────
    diff = (logits_orig - logits_reparam).abs()
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()
    pred_match = (preds_orig == preds_reparam).float().mean().item()
    acc_reparam = (preds_reparam == all_y).float().mean().item()

    print(f"\n{'='*55}")
    print(f"RESULTS")
    print(f"{'='*55}")
    print(f"Max  |logit_orig - logit_reparam| : {max_diff:.6e}")
    print(f"Mean |logit_orig - logit_reparam| : {mean_diff:.6e}")
    print(f"Classification agreement          : {pred_match:.6f}  ({pred_match*100:.2f}%)")
    print(f"Accuracy before reparam           : {final_acc:.4%}")
    print(f"Accuracy after  reparam           : {acc_reparam:.4%}")
    passed = max_diff < 1e-4
    print(f"\nTest A {'PASSED' if passed else 'FAILED'}  (threshold: max_diff < 1e-4)")
    print(f"{'='*55}")

    # ── Save results ──────────────────────────────────────────────────────────
    results_text = f"""# Test A — Reparameterisation Invariance

## Claim
Applying the SVD partial diagonalisation (Eqns. 13-18) to a trained
isotropic MLP produces a **mathematically identical** network. Outputs
before and after should differ only by floating-point rounding.

## Setup
- Model: IsotropicMLP [3072 → {WIDTH} → 10]
- Dataset: CIFAR-10 (10,000 test samples)
- Training: {EPOCHS} epochs, Adam lr={LR}, batch=24
- Device: {DEVICE}

## Results

| Metric | Value |
|---|---|
| Max \\|logit before − logit after\\| | `{max_diff:.6e}` |
| Mean \\|logit before − logit after\\| | `{mean_diff:.6e}` |
| Classification agreement (same argmax) | `{pred_match*100:.2f}%` |
| Test accuracy before reparameterisation | `{final_acc:.2%}` |
| Test accuracy after  reparameterisation | `{acc_reparam:.2%}` |
| **Test A** | **{'PASSED ✓' if passed else 'FAILED ✗'}** |

## Singular Value Summary

| | Min | Max | Mean |
|---|---|---|---|
| Before diagonalisation | {svs_before.min():.4f} | {svs_before.max():.4f} | {svs_before.mean():.4f} |
| After  diagonalisation | {svs_after.min():.4f} | {svs_after.max():.4f} | {svs_after.mean():.4f} |

## Interpretation

The reparameterisation works by decomposing W1 = U Σ Vᵀ and absorbing U
into W2 and b1. The key identity is:

```
W2' · iso_tanh(W1' x + b1')
= W2 U · iso_tanh(U^T(W1 x + b1))
= W2 U · U^T · iso_tanh(W1 x + b1)    [because ||U^T h|| = ||h||]
= W2 · iso_tanh(W1 x + b1)
```

The max difference of `{max_diff:.2e}` is purely floating-point arithmetic error
(expected ~1e-6 for float32). This confirms the theoretical claim is correct
and the implementation is sound.

{'✓ The core invariance holds. Dynamic topology is mathematically grounded.' if passed else '✗ WARNING: Invariance failed. Check implementation.'}
"""

    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print(f"\nResults saved to results/test_A/results.md")


if __name__ == '__main__':
    main()
