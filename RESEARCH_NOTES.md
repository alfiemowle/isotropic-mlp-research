# Project Briefing: Dynamic Topology Networks

## What This Project Is

An implementation and empirical investigation of:

> **"On De-Individuated Neurons: Continuous Symmetries Enable Dynamic Topologies"**
> George Bird, University of Manchester (arXiv:2602.23405v1, February 2026)

**Status**: 34 experiments (A–Z, AA–AG, AG-B, AH, AI, AJ). See `dynamic_topology_net/results/ALL_RESULTS.md` for the master summary.

---

## The Paper's Core Ideas

### The Problem with Standard Primitives

Standard activation functions (ReLU, tanh, etc.) are elementwise: they decompose a vector into individual scalar components, apply a function, and reassemble. This implicitly **individuates neurons** — each neuron has a fixed identity tied to its basis dimension. This makes architecture changes (pruning/growing) destructive because the function is not invariant under basis changes.

### The Solution: Isotropic Activation Functions

Replace elementwise activations with **isotropic activation functions**: functions equivariant under the full orthogonal group O(n).

```
f(Rx) = Rf(x)   for all R ∈ O(n)
f(x) = σ(‖x‖₂) · x̂       where x̂ = x / ‖x‖₂
```

The scalar function σ can be anything non-linear. We use `σ = tanh`. The key property: **these functions are basis-independent**.

### Layer Diagonalisation via SVD (Partial, Left-Sided)

For weight matrix W₁, compute W₁ = UΣVᵀ, then absorb U:

```
W₁' = ΣVᵀ
W₂' = W₂U
b₁' = Uᵀb₁
```

This is **exactly function-preserving** (confirmed experimentally: max diff = 3.06e-05, float32 only). After diagonalisation, singular values Σᵢᵢ rank neurons by importance.

### Dynamic Pruning

Neurons with small Σᵢᵢ contribute little to the output. When Σᵢᵢ < threshold ϑ, that neuron can be removed. **Experimentally confirmed**: r(Σᵢᵢ, acc_drop) = 0.815.

### Dynamic Growth (Neurogenesis)

New scaffold neurons added with zero W₁ row and zero W₂ column are **exactly inert** — confirmed experimentally (100% prediction match after adding up to 10 scaffold neurons).

### The Intrinsic Length Parameter

Trainable scalar `o = exp(λ) > 0` per layer absorbs residual bias terms under pruning. **Experimentally**: effect is negligible in practice when scaffold biases are initialised small (ratio ≈ 1.000×).

---

## Repository Structure

```
AI_Research/
├── CLAUDE.md                                   ← this file (keep updated)
├── Dynamic-Topologies-Research-Paper.pdf       ← the paper
└── dynamic_topology_net/
    ├── core/
    │   ├── __init__.py
    │   ├── models.py                           ← IsotropicMLP, BaselineMLP
    │   ├── activations.py                      ← IsotropicTanh
    │   └── train_utils.py                      ← train_model, evaluate, load_cifar10
    ├── experiments/
    │   ├── test_A_reparameterisation.py        ← through
    │   └── test_Z_gradient_criterion.py        ← 26 experiments total
    ├── results/
    │   ├── ALL_RESULTS.md                      ← master summary of all findings
    │   └── test_*/                             ← per-test results and plots
    ├── train.py                                ← original baseline script
    └── data/
        └── cifar-10-batches-py/
```

---

## Current State of the Code

The full isotropic MLP is implemented in `dynamic_topology_net/core/`:

- **IsotropicMLP**: `[input_dim → width → num_classes]` with IsotropicTanh activation
- **IsotropicTanh**: `f(x) = tanh(‖x‖) · x/‖x‖`, with safe handling at x=0
- **Partial diagonalisation**: `diagonalise(model)` applies W₁=UΣVᵀ → W₁'=ΣVᵀ, W₂'=W₂U, b₁'=Uᵀb₁
- **SVD pruning**: `svd_prune_to_width(model, k)` keeps top-k neurons by Σᵢᵢ
- **Scaffold growth**: append zero rows/columns to grow width

Standard hyperparameters used throughout: `lr=0.08, batch=128, epochs=24, Adam`.

---

## What We Know (Empirically, from 26 Experiments)

### Confirmed paper claims

| Claim | Test | Result |
|---|---|---|
| Diagonalisation is exact | A | Max diff = 3.06e-05 ✓ |
| Neurogenesis is exact | B | 100% pred match ✓ |
| SV predicts pruning impact | G, U2 | r = 0.815 ✓ |
| Isotropic > standard tanh | D, E, L, M, Q | +14–16% on CIFAR-10 ✓ |
| Depth stability | E, M, Q | Iso +3.4%/layer, Base −6.1%/layer ✓ |
| 50% pruning is stable | H, V | <0.1% accuracy drop ✓ |
| Width transitions are smooth | I | Confirmed for moderate changes ✓ |
| Iso advantage generalises | L | Confirmed on MNIST, F-MNIST, CIFAR-10 ✓ |

### What the paper overstates or leaves incomplete

| Claim | Test | Reality |
|---|---|---|
| "Overabundance → better acc" | W | +0.14% mean — marginal, inconsistent |
| "Dynamic topology → better acc" | J | Equal-epoch: Dynamic ≈ Static |
| "Gradient criterion for pruning" | Z | r=0.464 — far worse than SV (r=0.859) |
| "Semi-orth init for scaffold" | P | Impossible when width > num_classes |
| "Shell collapse is exact" | O, T, AB | Residual flat ~0.22 across widths 32–256; genuine gap, not finite-size |
| "Gradient flows to zero-W2 scaffold" | K | False — W2=0 → gradient=0 regardless |

### New findings not in the paper

| Finding | Test | Key number |
|---|---|---|
| Best pruning criterion | U2 | Composite Σᵢᵢ × ‖W2' col‖: r=0.859 |
| Depth collapse is structural | R, X | Rank regularisation: 0% improvement |
| Optimal pruning timing | V, S | Late (epoch 24) best; spectrum matures over training |
| Fast recovery after pruning | Y | 1 fine-tune epoch = 91% recovery |
| SVD overhead is negligible | N | ~0.1% per epoch at width=24 |
| Base depth failure ≠ repr collapse | AC | Base PR=18.7 at 3L; representations stay broad; collapse is in weights/gradients not activations |
| Iso concentrates repr with depth | AC | Iso PR drops 20→14→11 at 1L/2L/3L (selective); Base stays flat ~20 |
| Chi-norm accuracy-neutral | AA | ChiNorm-1L = 40.15% vs Iso-1L = 40.26%; difference within noise |
| IL+Chi-norm instability | AA | IL can explode to ~10¹² with Chi-norm without hurting accuracy; clip if using both |
| Shell collapse is width-independent | AB | Affine residual flat at ~0.22 across widths 32–256; genuine proof gap |
| Base depth failure mechanism | AD | Output layer gradient 7–8× larger than Iso; intermediate layers frozen by elementwise Jacobian collapse; depth adds nothing to fixed random features |
| Iso is NOT uniquely necessary | AE | LN+tanh beats Iso at 3L (46.54% vs 43.54%); RMS+tanh hits 47.51%; Jacobian preservation is the principle, achievable via normalisation |
| Iso depth window is finite | AG | Iso peaks at 4L (45.95%) at width=128, then collapses at 6L (29.98%); Base chaotic at all depths |
| LN+tanh outperforms Iso at all depths | AG+AG-B | LN+tanh peaks at 4L (49.17%) and holds better at 6L (46.57%) vs Iso's 29.98%; gap is structural at 30 epochs |
| LN+Iso is fragile at depth | AG-B | LN+Iso peaks at 3L (48.85%), collapses at 6L (30.55%); over-normalisation + isotropic compounding |
| RMS+tanh is most depth-robust | AG-B | Only model with positive slope (+0.008/layer) across depths 1-6; 43.76% at 6L vs 29.98% Iso |
| Iso beats LN+tanh long-term | AI | At 100 epochs Iso overtakes LN+tanh (44.43% vs 42.89%); LN=convergence speed, Iso=long-run stability |
| LN scaffold NOT inert | AJ | LN+tanh max_diff=0.086 after scaffold insertion (vs Iso 3e-6); paper topology claims require Iso |
| Modern activations with LN beat Iso | AH | LN+GELU, LN+SiLU, LN+ReLU all beat Iso at 3L; bare GELU/SiLU/ReLU collapse to 10% at depth 2+ |

---

## Important Notes & Gotchas (updated with experimental findings)

- **Partial vs. full diagonalisation**: Use partial (one-sided). Full requires three affine layers. Use `W₁'=ΣVᵀ, W₂'=W₂U, b₁'=Uᵀb₁`.

- **Diagonalisation is not permanent**: Re-diagonalise every 5–10 epochs for pruning decisions, not every step. SVD overhead is negligible (~0.1%/epoch).

- **Best pruning criterion is composite**: Use Σᵢᵢ × ‖W₂' col_i‖, not Σᵢᵢ alone. Gradient-based criterion (|dL/dΣᵢᵢ|) is significantly worse. Prune late (after training converges) for best results.

- **Scaffold W₂ initialisation**: Always use random small values (scale=0.01), **never zero**. With W₂=0, the gradient to W₁ is exactly zero regardless of Jacobian structure. The non-diagonal Jacobian helps with gradient *direction*, but W₂ must be nonzero for any gradient to flow.

- **Semi-orthogonal init is usually impossible**: When width > num_classes (the common case — e.g. width=24, classes=10), existing W₂ columns already span all of ℝ¹⁰. True orthogonality is geometrically impossible. Use random init instead — it performs identically.

- **Intrinsic length**: Theoretically necessary for exact pruning invariance at near-zero Σ. Practically negligible when scaffold biases are initialised small. Still include it, but don't expect measurable benefit until biases grow large relative to representation norm.

- **Depth collapse mechanism (fully resolved, Test AD)**: At lr=0.08, Base neurons saturate to ±1 at 99.94% rate within the first few epochs. The elementwise Jacobian (sech²) collapses to ~0, making intermediate layers fixed random feature extractors. All learning pressure concentrates on the output layer, whose gradient norm (0.58) is 7–8× larger than Iso's (0.079). Adding depth just makes fixed features harder to linearly separate. Iso's isotropic Jacobian preserves the *tangential* gradient component regardless of norm — every layer keeps learning with balanced, small gradients (~0.008–0.010). This is why regularisation, representation diversity, and global gradient magnitude all failed as explanations: the failure is in the per-activation Jacobian structure.

- **Iso depth window is finite (Test AG)**: At width=128, Iso scales well from 1L to 4L (+4.98% delta), then collapses at 5-6L. LN+tanh is more depth-robust (peaks at 4L: 49.17%, still 46.57% at 6L). RMS+tanh is most robust of all (+0.008/layer slope). LN+Iso collapses fastest at depth (peaks 3L, falls to 30.55% at 6L). Recommended depth for Iso: 3-4 layers maximum at this scale.

- **LN vs Iso trade-off (Tests AE, AG, AI)**: Short training (24-30 epochs): LN+tanh > Iso by ~3-5%. Long training (100 epochs): Iso overtakes LN+tanh (44.43% vs 42.89% — LN overfits). Topology features (pruning, scaffold inertness): require Iso (LN scaffold max_diff=0.086, not inert). Conclusion: use Iso when dynamic topology is the goal or training is long; use LN+tanh for fixed architecture with short training.

- **Dynamic topology value is flexibility, not raw accuracy**: Fair comparisons (equal training epochs) show Dynamic ≈ Static accuracy. The value is that 50% of neurons can be pruned and 91% of accuracy recovers in 1 fine-tune epoch — architectural agility at negligible cost.

- **Chi-normalisation**: Divides activations by running mean of ‖h‖ — isotropic, prevents magnitude blow-up. Test AA shows it is **accuracy-neutral** vs plain Iso (±0.002). The Intrinsic Length (IL) parameter remains negligible even with Chi-norm. Warning: IL + Chi-norm can produce instability (o explodes to ~10¹²) without accuracy collapse — if using IL, clip or regularise it when normalisation is present.

---

## Key Hyperparameters

| Parameter | Symbol | Recommended value | Source |
|---|---|---|---|
| Hidden width | M | 24–32 (sweet spot) | Tests E, Q |
| Pruning criterion | — | Σᵢᵢ × ‖W₂' col‖ | Test U2 |
| Pruning timing | — | Late (after convergence) | Tests V, S |
| Pruning threshold | ϑ | Tune per run | — |
| Scaffold W₂ init scale | — | 0.01 (random) | Tests K, P |
| SVD schedule | — | Every 5–10 epochs | Test N |
| Fine-tune after prune | — | Min 2–3 epochs | Test Y |
| Depth | D | 2–3 layers | Tests E, M, Q |
| Overabundance | — | Optional; +0.14% avg | Test W |
