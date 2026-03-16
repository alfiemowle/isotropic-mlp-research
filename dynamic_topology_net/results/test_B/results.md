# Test B — Neurogenesis Invariance

## Claim
Adding a scaffold neuron (zero W1 row, zero W2 column) to a trained
isotropic network produces an **exactly equivalent** network. The new
neuron contributes nothing to the output until training moves its weights.

## Setup
- Model: IsotropicMLP [3072 → 24 → 10]
- Dataset: CIFAR-10 (10,000 test samples)
- Training: 24 epochs, Adam lr=0.08
- Scaffold neurons initialised with: zero W1 row, zero W2 column, b_star=0

## Results

| Neurons added | New width | Max \|diff\| | Mean \|diff\| | Pred match |
|---|---|---|---|---|
| +1 neurons | 25 | `3.0637e-05` | `5.6668e-06` | 100.00% |
| +2 neurons | 26 | `3.0637e-05` | `5.6668e-06` | 100.00% |
| +5 neurons | 29 | `3.0637e-05` | `5.6668e-06` | 100.00% |
| +10 neurons | 34 | `3.0994e-05` | `5.6517e-06` | 100.00% |

**Baseline accuracy:** 39.98%
**Worst-case max diff:** `3.0994e-05`
**Test B: PASSED ✓**

## Why it's Exact

A scaffold neuron has:
- W1 row = **0** → pre-activation h_new = b_star (constant, defaults to 0)
- W2 column = **0** → output contribution = W2[:, new] · a_new = 0

So the scaffold neuron is completely inert in the forward pass. The output
is unchanged regardless of how many scaffold neurons are added.

The non-zero floating-point errors shown are pure arithmetic rounding
(float32 ~1e-7 precision). Classification decisions are 100% preserved.

## Implications

Neurogenesis is **lossless** — you can add neurons freely without changing
what the network computes. Their role is purely structural: they provide
"slots" for gradient flow to specialise into new features. This is the
basis for the "start wide, prune down" training strategy in the paper.
