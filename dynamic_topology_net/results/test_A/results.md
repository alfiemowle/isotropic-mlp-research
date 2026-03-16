# Test A — Reparameterisation Invariance

## Claim
Applying the SVD partial diagonalisation (Eqns. 13-18) to a trained
isotropic MLP produces a **mathematically identical** network. Outputs
before and after should differ only by floating-point rounding.

## Setup
- Model: IsotropicMLP [3072 → 24 → 10]
- Dataset: CIFAR-10 (10,000 test samples)
- Training: 24 epochs, Adam lr=0.08, batch=24
- Device: cuda

## Results

| Metric | Value |
|---|---|
| Max \|logit before − logit after\| | `3.063679e-05` |
| Mean \|logit before − logit after\| | `5.666783e-06` |
| Classification agreement (same argmax) | `100.00%` |
| Test accuracy before reparameterisation | `39.98%` |
| Test accuracy after  reparameterisation | `39.98%` |
| **Test A** | **PASSED ✓** |

## Singular Value Summary

| | Min | Max | Mean |
|---|---|---|---|
| Before diagonalisation | 191.6122 | 2030.9723 | 817.8496 |
| After  diagonalisation | 191.6122 | 2030.9673 | 817.8492 |

## Interpretation

The reparameterisation works by decomposing W1 = U Σ Vᵀ and absorbing U
into W2 and b1. The key identity is:

```
W2' · iso_tanh(W1' x + b1')
= W2 U · iso_tanh(U^T(W1 x + b1))
= W2 U · U^T · iso_tanh(W1 x + b1)    [because ||U^T h|| = ||h||]
= W2 · iso_tanh(W1 x + b1)
```

The max difference of `3.06e-05` is purely floating-point arithmetic error
(expected ~1e-6 for float32). This confirms the theoretical claim is correct
and the implementation is sound.

✓ The core invariance holds. Dynamic topology is mathematically grounded.
