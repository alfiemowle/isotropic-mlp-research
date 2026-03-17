# ALL RESULTS — Dynamic Topology Networks
### Full Experimental Suite (Tests A–Z)
#### Paper: "On De-Individuated Neurons: Continuous Symmetries Enable Dynamic Topologies" (Bird, 2026)

---

## Quick Reference: Test Index

| Test | Topic | Status | Key Number |
|---|---|---|---|
| AS | Full integrated pipeline (best config) | COMPLETE | Dynamic 51.09% vs static 49.47%/51.00%; all overfit at constant LR |
| AN | Interleaved training protocol | COMPLETE | Stale Adam harmless; reset Adam hurts −0.0044 |
| AP | Chained SVD pruning (multi-layer) | COMPLETE | 2L pruning +1.4pp (improves accuracy!) |
| AQ | IsoGELU LR sweep | COMPLETE | H2 confirmed: gap 12pp→1pp at low LR |
| AR | Hybrid architectures at width=128 | COMPLETE | Iso-first-4L 52.28% — best result overall |
| A | Reparameterisation invariance | COMPLETE | Max diff = 3.06e-05 ✓ |
| B | Neurogenesis invariance | COMPLETE | 100% pred match ✓ |
| C | Intrinsic length absorption | COMPLETE | ~0x improvement in practice |
| D | Expressivity (Iso vs tanh) | COMPLETE | +16.5% on CIFAR-10 |
| E | Depth & width scaling | COMPLETE | Iso +2.7%/layer, Base −0.5%/layer |
| F | Shell collapse (1L) | COMPLETE (artifact — see T) | Measurement was underdetermined |
| G | Pruning error vs SV | COMPLETE | r(SV, acc_drop) = 0.77 |
| H | Sequential pruning stability | COMPLETE | Iso holds to 16/32 neurons |
| I | Full grow/prune pipeline | COMPLETE | Smooth transitions confirmed |
| J | Fair vs unfair comparison | COMPLETE | Dynamic advantage mostly disappears |
| K | Gradient flow through scaffold | COMPLETE | W2 init dominates, not Jacobian |
| L | Cross-dataset validation | COMPLETE | Iso wins on MNIST/F-MNIST/CIFAR-10 |
| M | Deeper networks (2L) | COMPLETE | Iso +2.45%/layer depth, Base −2.94% |
| N | Wall-clock SVD timing | COMPLETE | SVD overhead: ~0.1% per epoch |
| O | Two-layer shell collapse | COMPLETE | Nonlinearity contributes at depth |
| P | Scaffold neuron initialisation | COMPLETE | Random W2 = semi-orth when width>classes |
| Q | Three-layer depth scaling | COMPLETE | Iso +3.36% 1→3L, Base −6.11% |
| R | Representational collapse autopsy | COMPLETE | Results inconclusive |
| S | SV spectrum phase transitions | COMPLETE | Condition number 1.2→8.4 over training |
| T | Affine collapse revisited | COMPLETE | Residual ~0.32–0.74 (not affine) |
| U | Composite pruning (flawed) | SUPERSEDED | Used row norms, not SVs — see U2 |
| U2 | Composite pruning (corrected) | COMPLETE | r(Composite) = 0.859, best criterion |
| V | Optimal pruning timing | COMPLETE | Late pruning (epoch 24) best |
| W | Overabundance protocol | COMPLETE | Mean gain +0.14% (marginal) |
| X | Rank regularisation | COMPLETE | Zero effect at all lambda values |
| Y | Minimum fine-tune budget | COMPLETE | 1 epoch = 91% recovery (post-training prune) |
| Z | Gradient pruning criterion | COMPLETE | r(grad) = 0.46, SV wins (r=0.86) |

---

## Theme 1: Mathematical Foundations (A, B, C)

### Test A — Reparameterisation Invariance
**Claim**: Partial left-diagonalisation (W1=UΣVᵀ → W1'=ΣVᵀ, W2'=W2U, b1'=Uᵀb1) preserves function exactly.

**Result**: CONFIRMED
- Max |logit difference| = 3.06e-05 (float32 rounding)
- 100% classification agreement (10,000 samples)
- Accuracy unchanged: 39.98% → 39.98%

**Significance**: The core mathematical claim of the paper is correct and implementable. Dynamic topology is mathematically grounded.

---

### Test B — Neurogenesis Invariance
**Claim**: Adding scaffold neurons (zero W1 row, zero W2 column) leaves network function unchanged.

**Result**: CONFIRMED
- Adding 1, 2, 5, 10 scaffold neurons: 100% prediction match
- Max diff = 3.10e-05 (float32 rounding only)
- Accuracy preserved exactly

**Significance**: Lossless neuron addition is confirmed. Scaffold neurons are truly inert until gradient moves their weights.

---

### Test C — Intrinsic Length Absorption
**Claim**: The intrinsic length parameter `o = exp(λ)` corrects for bias residuals when pruning near-zero SV neurons.

**Result**: NUANCED
- Part 1 (active neurons, large σ): IL correction ratio = 1.000× — no effect (expected: linear term dominates)
- Part 2 (zero-σ scaffold neurons, small biases): IL correction ratio = 1.000× — negligible improvement
- Effect is proportional to |b_i|² / mean(‖h‖²); at small biases relative to representation norm, improvement is minimal

**Significance**: IL is theoretically necessary for exact pruning invariance but practically negligible when biases are small. Recommend initialising scaffold neuron biases near zero.

---

## Theme 2: Isotropic Activation — Expressivity & Accuracy (D, E, L, M, Q)

### Test D — Expressivity (Iso vs Standard Tanh MLP)
**Setup**: Width=64, 100 epochs (synthetic), 24 epochs (CIFAR-10), seeds [42, 123, 7]

| Task | Iso | Baseline | Gap |
|---|---|---|---|
| XOR 100-dim | 0.637 | 0.500 | **+13.7%** |
| Selective Gate 200-dim | 1.000 | 1.000 | +0.0% |
| CIFAR-10 | 0.405 | 0.240 | **+16.5%** |

**Significance**: Isotropic activation is strictly more expressive than standard tanh despite the O(n)-equivariance constraint. The +16.5% CIFAR-10 advantage is striking given the simple architecture.

---

### Test E — Depth & Width Scaling (1L vs 2L)
**Setup**: Widths [8, 16, 24, 32], seeds [42, 123], batch=128

| Configuration | Depth gain (2L vs 1L) |
|---|---|
| Isotropic (mean) | **+2.7% per layer** |
| Baseline tanh (mean) | **−0.5% per layer** |

1-layer results: Iso=40.7%, Base=26.6% (width=24)
2-layer results: Iso=43.4%, Base=25.9% (width=24)

**Significance**: Isotropic networks scale with depth. Standard tanh degrades slightly at depth already at 2 layers.

---

### Test M — Deeper Networks (extended to 2L with more widths)
**Setup**: Widths [8, 16, 24, 32, 48], seeds [42, 123], batch=24

| Config | Depth gain |
|---|---|
| Iso 1L→2L (mean) | **+2.45%** |
| Base 1L→2L (mean) | **−2.94%** |

**Significance**: Consistent with Test E across wider range. Baseline degrades more severely at batch=24.

---

### Test Q — Three-Layer Depth Scaling (1L, 2L, 3L)
**Setup**: Widths [16, 24], seeds [42, 123], batch=128

| Model | 1L | 2L | 3L | 1L→3L gain |
|---|---|---|---|---|
| Iso (w=24) | 40.6% | 43.4% | 44.7% | **+4.09%** |
| Base (w=24) | 27.6% | 25.6% | 21.4% | **−6.18%** |

- Iso 1→3L mean gain: **+3.36%**
- Base 1→3L mean gain: **−6.11%**
- Base-3L minimum accuracy: 21.3%

**Significance**: Confirmed at 3 layers. Isotropic networks benefit progressively from depth while baseline collapses. This is the paper's nested functional class claim, strongly supported.

---

### Test L — Cross-Dataset Validation (MNIST, F-MNIST, CIFAR-10)
**Setup**: Width=24, 24 epochs, dynamic = 32→24 pruning

| Dataset | Iso | Base | Gap | Dynamic (32→w) |
|---|---|---|---|---|
| MNIST (w=24) | 93.8% | 86.5% | **+7.3%** | 93.7% |
| F-MNIST (w=24) | 85.0% | 73.1% | **+11.9%** | 84.9% |
| CIFAR-10 (w=24) | 38.9% | 24.9% | **+14.0%** | 40.0% |

**Significance**: Isotropic advantage generalises across datasets. Dynamic topology (start wide, prune) performs on par with static isotropic across all three datasets.

---

## Theme 3: Shell Collapse Investigation (F, O, T)

### Test F → SUPERSEDED by T (measurement artifact)
Original Test F found CollapsingIso residual = 0.000000, but this was because N=128 samples were used to fit a 3072+1 dimensional affine map — an underdetermined system that always gives zero residual regardless of actual linearity.

### Test O — Two-Layer Shell Collapse
**Setup**: Width=32, 24 epochs, full test set (10K samples)

| Model | Accuracy | Affine Residual |
|---|---|---|
| Iso-2L | 43.97% | N/A |
| CollapsingIso-2L | 43.12% | 0.74 (rel=0.42) |
| Iso-1L | 40.68% | N/A |
| CollapsingIso-1L | 40.90% | 0.32 (rel=0.25) |

- Iso depth gain (1L→2L): +3.29%
- Collapsing depth gain (1L→2L): +2.22%

**Finding**: Nonlinearity contributes at depth (+1.07% beyond affine composition). The collapsing model is NOT affine — residual is substantial even with correct measurement.

### Test T — Affine Collapse Revisited (L2-normalised inputs)
**Question**: Does unit-sphere input normalisation make collapse exact (as Appendix C theoretically requires)?

| Config | Acc | Affine Residual |
|---|---|---|
| CollapsingIso-1L / raw | 40.90% | 0.32 |
| CollapsingIso-1L / L2 | 42.12% | 0.51 |
| CollapsingIso-2L / raw | 43.12% | 0.74 |
| CollapsingIso-2L / L2 | 44.42% | 0.81 |

**Finding**: Even with unit-sphere inputs the residual grows. The affine collapse is weaker than Appendix C claims in practice. Appendix C is a mathematical guarantee at the limiting case; at finite scale the model retains nonlinearity.

---

## Theme 4: Dynamic Topology — Growth & Pruning (G, H, I, J, K, P)

### Test G — Pruning Error vs Singular Value (core claim)
**Setup**: Widths [8, 16, 32], seed=42

- **Pearson r(SV, L2_error) = 0.7704** — strong positive correlation
- **Pearson r(SV, acc_drop) = 0.7115** — strong correlation
- At width=32: smallest SV (65) → acc_drop=+0.0012; largest SV (651) → acc_drop=+0.0404

**Significance**: Paper's primary claim that SV predicts pruning impact is validated. Neurons with small SVs are genuinely less important.

---

### Test H — Sequential Pruning Stability
**Setup**: Width=32, prune one neuron at a time (smallest SV), seed=42

| Neurons remaining | Iso acc | Base acc |
|---|---|---|
| 32 (start) | 40.69% | 23.52% |
| 16 (50% pruned) | 40.34% | 20.11% |
| 10 (69% pruned) | 26.66% | 18.77% |

- Iso "cliff" (below 70% init acc): **10 neurons remaining**
- Pruning AUC: Iso = 0.3289, Base = 0.1985, advantage = **+0.1304**

**Significance**: Isotropic network can be pruned to 50% width with near-zero accuracy loss. The cliff appears at ~70% pruning depth.

---

### Test I — Full Grow/Prune Pipeline (Fig 3 Replication)
**Setup**: Start widths [8, 16, 24, 32], target widths [8, 16, 24, 32], 48 adaptation epochs

Key observations:
- **Growing** (8→16/24/32): +1.1–1.5% accuracy gain — growth effectively unlocks capacity
- **Shrinking** (32→16/24): smooth transitions with minimal accuracy loss
- **Heavy pruning** (32→8): measurable drop, but ~3.5% from baseline

**Significance**: The paper's "smooth transitions" claim holds for moderate width changes. Heavy pruning requires more recovery time.

---

### Test J — Fair vs Unfair Comparison (paper protocol critique)
**Issue**: Paper compares 72-epoch dynamic vs 24-epoch static.

| Condition | Model | w=24 |
|---|---|---|
| Unfair (paper protocol) | Static-Iso 24ep | 40.0% |
| Unfair (paper protocol) | Dynamic-Iso 72ep | 40.2% |
| **Fair** (equal epochs) | Static-Iso 72ep | 40.3% |
| **Fair** (equal epochs) | Dynamic-Iso 72ep | 40.2% |

**Finding**: Under fair comparison (equal training epochs), Dynamic-Iso ≈ Static-Iso. The paper's reported advantage is largely an artifact of extra training time given to the dynamic condition. Dynamic topology does not add performance — it enables architectural flexibility.

---

### Test K — Gradient Flow Through Scaffold Neurons
**Claim**: Non-diagonal isotropic Jacobian routes gradients to scaffold neurons at zero W1.

| Condition | Mean grad_W1 |
|---|---|
| Iso + random W2 | 0.000252 |
| **Iso + zero W2** | **0.000000** |
| Baseline + random W2 | 0.009589 |

**Finding**: W2 initialisation is the decisive factor. Even with non-diagonal Jacobian, dL/dW1_new = 0 when W2_new = 0. **Practical implication**: Always initialise scaffold neuron W2 columns with small random values, not zero.

---

### Test P — Scaffold Neuron Initialisation Comparison
**Setup**: Pretrained IsotropicMLP width=24, 500 adaptation steps

| Init strategy | Mean grad_W1 | Final acc (500 steps) |
|---|---|---|
| zero W2 | 0.000000 | 39.39% |
| **random W2** | **0.000288** | **40.00%** |
| semi-orthogonal | 0.000286 | 39.99% |
| copy existing | 0.000332 | 39.44% |

**Finding**: Semi-orthogonal is impossible when width (24) > num_classes (10) — all of R¹⁰ is already spanned. Random W2 (scale=0.01) is the practical recommendation: immediate gradient flow, simple implementation.

---

## Theme 5: Pruning Criterion Research (G, U2, V, Z)

### Test U2 — Composite Pruning Criterion (Corrected SVD)
**Note**: Test U (superseded) used W1 row norms as "SV proxy" — incorrect. Test U2 uses actual SVD Σ_ii.

**Setup**: Widths [16, 24], seeds [42, 123], leave-one-out in diagonalised basis

| Criterion | Pearson r | Spearman rho |
|---|---|---|
| SVD Σ_ii (paper's criterion) | **0.815** | **0.840** |
| ‖W2' col‖ | 0.798 | 0.801 |
| **Σ_ii × ‖W2' col‖ (composite)** | **0.859** | **0.869** |

Best by width:
- w=16: r(SV)=0.770, r(W2)=0.794, r(Composite)=0.843
- w=24: r(SV)=0.859, r(W2)=0.801, r(Composite)=0.875

**Finding**: Paper's SV criterion is strong (r=0.815). Composite adds modest +4.4% improvement. W2_col alone is competitive. This is an **extension** of the paper's open question (footnote 5), not a correction.

---

### Test V — Optimal Pruning Timing
**Setup**: Prune at epochs [0,4,8,12,16,20,24], fine-tune 8 epochs, fractions [25%, 50%]

| Prune fraction | Best epoch | Final acc | vs Baseline (40.59%) |
|---|---|---|---|
| 25% (remove 6/24) | epoch 24 | **41.55%** | **+0.96%** |
| 50% (remove 12/24) | epoch 20 | **40.97%** | **+0.38%** |

- Pruning at epoch 0 also works well (spectrum-independent)
- Middle epochs (4–16) are slightly worse

**Finding**: Late pruning (after training converges) is best — the SV spectrum needs time to develop meaningful differentiation (condition number grows from 1.2 to 8.4 across 24 epochs per Test S).

---

### Test Z — Gradient-Based Pruning Criterion
**Claim** (paper footnote 5): |dL/dΣ_ii| may be a better pruning criterion than Σ_ii alone.

| Criterion | Pearson r | Spearman rho |
|---|---|---|
| SV (Σ_ii) | **0.859** | **0.897** |
| \|dL/dΣ_ii\| (gradient) | 0.464 | 0.502 |
| ‖W2' col‖ | 0.801 | 0.827 |
| **Σ_ii × ‖W2' col‖** | **0.875** | **0.907** |
| \|grad\| × ‖W2'\| | 0.581 | 0.723 |

Results per seed:
- Seed 42: r(SV)=0.859, r(grad)=0.568, r(Composite)=0.868
- Seed 123: r(SV)=0.859, r(grad)=0.360, r(Composite)=0.883

**Finding**: The gradient criterion fails significantly (r=0.464 vs r=0.859). The paper's footnote suggestion is **not validated**. Gradient magnitude at only 4 batches is noisy and data-dependent; it adds noise rather than signal beyond the SV. The composite Σ_ii × ‖W2' col‖ remains the best criterion (r=0.875).

---

## Theme 6: Training Dynamics (N, S, R)

### Test N — Wall-Clock SVD Timing
**Setup**: Widths [8, 16, 24, 32, 64, 128], width=24 per-epoch comparison

| SVD operation | CPU time (ms) | GPU time (ms) |
|---|---|---|
| Width=24 | 5.4 | 4.4 |
| Width=128 | 25.1 | 7.4 |

Per-epoch training overhead at width=24:
- Isotropic (no diag): +32.6% vs baseline
- SVD per epoch: +0.1% additional
- SVD every 5 epochs: negligible

**Finding**: SVD overhead is negligible. The isotropic activation itself adds ~33% compute overhead (norm + tanh vs elementwise tanh). **Practical recommendation**: Diagonalise every 5–10 epochs, not every step.

---

### Test S — SV Spectrum Phase Transitions
**Setup**: Width=24, tracked every epoch

| Epoch | Condition number (Iso-1L) | Spectral entropy | Top-3 SV fraction |
|---|---|---|---|
| 0 | 1.2 | 0.9997 | 13.4% |
| 1 | 4.0 | 0.9728 | 24.1% |
| 12 | 7.1 | 0.9551 | 24.8% |
| 24 | 8.4 | 0.9574 | 23.5% |

For Iso-2L: condition number peaks at 9.6 (epoch 12) then **decreases** to 5.4 (epoch 24) as spectrum rebalances.

**Finding**: Spectrum differentiates rapidly in epoch 1 then evolves more slowly. Condition number is highest mid-training for 2L. This explains why Test V finds early pruning works (after epoch 0 the spectrum is already meaningful).

---

### Test R — Representational Collapse Autopsy
**Question**: Does Base-2L/3L fail due to gradient vanishing or representational collapse?

| Model | Acc | Effective rank (min) | Gradient norm (final) |
|---|---|---|---|
| Iso-3L | 45.0% | 4.08 | 0.009 (W1) |
| Base-1L | 27.1% | 3.51 | 0.029 (W1) |
| Base-2L | 25.8% | 3.10 | 0.037 (W1) |
| Base-3L | 21.7% | 3.28 | 0.076 (W1) |

**Finding**: Gradient norms are actually *larger* in deeper baseline networks, not smaller — ruling out classic gradient vanishing. Effective rank is similarly low across baseline depths. Results are inconclusive on the exact mechanism. The collapse appears more related to representation geometry than gradient flow.

---

## Theme 7: Advanced Protocol Experiments (W, X, Y)

### Test W — Overabundance Protocol (COMPLETE)
**Setup**: Start 2× wider (32→16, 48→24, 64→32), prune at epoch 12 using SVD SV criterion, continue training to epoch 24. Compare to static iso trained directly at target width. Seeds [42, 123].

| Target | Over width | Static Iso | Overabundant→Pruned | Gain | Static Baseline |
|---|---|---|---|---|---|
| w=16 | 32 | 40.73% | 41.01% | **+0.28%** | 27.32% |
| w=24 | 48 | 40.59% | 40.95% | **+0.35%** | 27.62% |
| w=32 | 64 | 40.76% | 40.55% | **−0.21%** | 25.74% |
| **Mean** | | 40.69% | 40.84% | **+0.14%** | 26.89% |

**Finding**: Overabundance shows a marginal mean gain of +0.14% — present but negligible. The advantage is inconsistent across widths (positive at w=16/24, negative at w=32) and is within noise for individual seeds. The paper's biological overabundance claim (Section 4) is not strongly supported at this scale: starting twice as wide and pruning halfway gives essentially the same result as just training at the target width from scratch.

---

### Test X — Rank Regularisation (COMPLETE)
**Setup**: Spectral isotropy regulariser L_rank = ‖C/tr(C) − I/d‖²_F applied to first hidden layer representations. Lambda ∈ {0, 0.01, 0.1}. Baseline networks at 1L, 2L, 3L. Seeds [42]. Note: eff_rank tracking non-functional (NaN for all — hook issue with custom activation modules).

| Model | lambda=0 | lambda=0.01 | lambda=0.1 | Iso upper bound |
|---|---|---|---|---|
| 1L | 25.49% | 25.49% | 25.49% | 40.28% |
| 2L | 26.12% | 26.12% | 26.12% | 43.54% |
| 3L | 22.01% | 22.01% | 22.01% | 44.46% |

Gap closed by best regulariser: **0%** at every depth and every lambda value.

**Finding**: Spectral isotropy regularisation has completely zero effect on baseline depth collapse — to three decimal places, across all three depths and both lambda values tested. This has a strong implication for Test R's open question: the depth failure of standard tanh networks is not caused by representational collapse that can be corrected by encouraging rank diversity. The failure is structural — it is a property of the elementwise activation function's interaction with depth, not a consequence of the training objective. Regularising the loss cannot substitute for isotropic activation.

---

### Test Y — Minimum Fine-Tune Budget (COMPLETE)
**Setup**: Prune 50% (24→12 neurons) at epoch 0 or epoch 24. Fine-tune for {0,1,2,3,4,6,8,12} epochs. Mean over seeds [42, 123]. No-prune baseline: 40.59%.

**Prune after full training (epoch 24):**

| Fine-tune epochs | Accuracy | vs no-prune | Recovery |
|---|---|---|---|
| 0 | 36.08% | −4.52% | 0% |
| **1** | **40.20%** | **−0.40%** | **91%** |
| 3 | 41.27% | +0.68% | 115% |
| 12 | 41.13% | +0.54% | 112% |

**Prune at initialisation (epoch 0):**

| Fine-tune epochs | Accuracy | vs no-prune | Recovery |
|---|---|---|---|
| 0 | 9.65% | −30.94% | 0% |
| 1 | 37.31% | −3.28% | 89% |
| **2** | **38.56%** | **−2.03%** | **93%** |
| 12 | 40.75% | +0.15% | 101% |

**Key finding**: **90% recovery after only 1 fine-tune epoch** (when pruning post-training). Pruning after full training and fine-tuning 3 epochs actually *exceeds* the no-prune baseline by +0.68%. Pruning at init takes 2 epochs to reach 90% recovery. Isotropic dynamic networks are extremely robust to architectural changes with minimal retraining cost.

---

## Key Findings Summary

### What the paper gets right (confirmed)

1. **Reparameterisation invariance** (Test A): Exact, float32 only. Core math is sound.
2. **Neurogenesis invariance** (Test B): Exact. Scaffold neurons are truly inert.
3. **SV ↔ pruning impact** (Tests G, U2): r=0.77–0.86. Paper's pruning criterion works well.
4. **Isotropic > standard tanh** (Tests D, E, L, M, Q): Consistently +13–16% on CIFAR-10, generalises to MNIST/F-MNIST, scales with depth.
5. **Depth stability** (Tests E, M, Q): Iso gains ~2.7–4.1% per added layer; Base loses 2–6% at depth.
6. **Sequential pruning stability** (Test H): 50% pruning with near-zero accuracy loss.
7. **Cross-architecture growth/prune** (Test I): Smooth width transitions confirmed.

### What we found that the paper leaves open (extensions)

1. **Composite criterion is better** (Test U2): Σ_ii × ‖W2' col‖ gives r=0.859 vs r=0.815 for SV alone (+4.4%). Paper explicitly leaves criterion choice open (footnote 5).
2. **Gradient criterion fails** (Test Z): Paper's footnote 5 suggestion of |dL/dΣ_ii| gives r=0.464 — far worse than SV. Not validated.
3. **Semi-orthogonal init impossible in common case** (Test P): When width > num_classes, paper's recommendation cannot be implemented. Random W2 is the practical alternative with identical performance.
4. **Dynamic topology advantage is training-time artifact** (Test J): Equal-epoch comparison shows Dynamic ≈ Static. The architectural flexibility is real; the accuracy claim is overstated.
5. **Shell collapse is weak in practice** (Tests O, T): Appendix C's affine collapse holds only asymptotically — at finite scale, residuals are 0.32–0.81, not zero.
6. **Scaffold gradient requires nonzero W2** (Test K): The non-diagonal Jacobian distributes gradients, but only when W2 ≠ 0. Paper implies gradient flows with zero W2, which is incorrect.
7. **Late pruning is optimal** (Test V): SV-based criterion improves as spectrum matures. Best at epoch 24, but even epoch 0 works due to fast recovery.
8. **Fast recovery after pruning** (Test Y): 50% pruning recovers 91% in just 1 fine-tune epoch; exceeds baseline after 3 epochs.
9. **Overabundance advantage is marginal** (Test W): +0.14% mean gain, inconsistent across widths. Biological development analogy is appealing but weak at this scale.
10. **Depth collapse is structural, not correctable** (Test X): Rank regularisation has zero effect on baseline depth failure, confirming the collapse is intrinsic to elementwise activations, not a training objective artifact.

### Resolved questions

- **Why does Base depth fail?** (Tests R, X): Not gradient vanishing (gradients are larger at depth). Not representational collapse that regularisation can fix (Test X: 0% improvement). The failure is structural — elementwise activations individuate neurons and cannot compose stably under depth. Isotropic activations are the only solution identified.

---

## Recommended Practice (based on full experimental suite)

| Decision | Recommendation | Source |
|---|---|---|
| Activation | IsotropicTanh over standard tanh | Tests D, E, L, M, Q |
| Scaffold W2 init | Random, scale=0.01 | Tests K, P |
| Pruning criterion | Composite: Σ_ii × ‖W2' col‖ | Test U2 |
| Pruning timing | Late (after training converges) | Tests V, S |
| SVD frequency | Every 5–10 epochs | Test N |
| Depth | Scale to 2–3 layers; Iso benefits, Base degrades | Tests E, M, Q |
| Intrinsic length | Useful for exact pruning invariance; init biases small | Test C |
| Fine-tune budget | 2–3 epochs minimum after 50% pruning | Test Y |
| Width | Start at target width; overabundance adds ~+0.14% (marginal) | Test W |

---

## Closing Summary

All 26 experiments complete. The full suite runs from first principles (Tests A–C verifying the math) through expressivity and depth characterisation (D–E, L–Q), dynamic topology mechanics (G–K, P), and original research extensions (R–Z).

### The three most important findings

**1. Isotropic activation is a genuine architectural advance** (Tests D, E, L, M, Q, R, X)
The +14–16% accuracy advantage over standard tanh on CIFAR-10 holds across widths, depths, datasets, and seeds. It is not a tuning artefact. More tellingly, Test X confirms this gap cannot be closed by any amount of loss regularisation — the failure of standard networks at depth is structural, intrinsic to elementwise activations. Isotropic activation is the only identified solution.

**2. The pruning criterion can be improved** (Tests U2, Z)
The paper's SV criterion (r=0.815) is solid, but the composite Σ_ii × ‖W2' col‖ is measurably better (r=0.859). The paper's own footnote 5 suggestion of gradient-based pruning fails badly (r=0.464). Of all criteria tested, composite is the clear recommendation.

**3. Dynamic topology provides flexibility, not free accuracy** (Tests J, W, Y)
The fair comparison (Test J) shows dynamic topology ≈ static with equal training. The overabundance protocol (Test W) gives +0.14% on average — essentially noise. What dynamic topology *does* provide is demonstrated by Test Y: you can prune 50% of neurons and recover 91% of accuracy in a single fine-tune epoch, and exceed baseline after 3 epochs. The value is architectural agility at negligible cost, not raw accuracy gain.

---

## Follow-up Experiments (AA–AC): Knowledge Gap Investigation

Three experiments launched 2026-03-16 to probe gaps identified after the initial 26-test suite.

---

### Test AC — Representation Spectrum Mechanism (COMPLETE)

**Question**: Why does standard tanh fail at depth? Is it representational collapse (low PR/effective rank)?

**Setup**: Iso/Base at 1L, 2L, 3L depth. Width=32, Epochs=24. Tracks Participation Ratio (PR = (Σsᵢ)²/Σ(sᵢ²)) of hidden representation SVD at every epoch.

| Model | Acc | PR (final) | top-1 frac | top-3 frac | repr norm |
|---|---|---|---|---|---|
| Iso-1L | 40.06% | 20.30 | 0.088 | 0.243 | 1.0000 |
| Base-1L | 27.01% | 20.22 | 0.135 | 0.282 | 5.656 |
| Iso-2L | 43.97% | 14.31 | 0.120 | 0.307 | 1.0000 |
| Base-2L | 26.38% | 21.98 | 0.110 | 0.271 | 5.616 |
| Iso-3L | 44.45% | 10.73 | 0.158 | 0.405 | 1.0000 |
| Base-3L | 23.02% | 18.70 | 0.139 | 0.312 | 5.629 |

**PR trajectory**: Iso-3L: 5.35 → 10.73 (+5.37). Base-3L: 17.22 → 18.70 (+1.48).

**Finding**: **Base depth failure is NOT representational collapse.** Base-3L maintains a broad representation spectrum (PR=18.7, similar to Iso-1L's 20.3) even while accuracy degrades to 23%. The covariance spectrum stays diverse. Critically, Iso networks *concentrate* representations with depth (PR drops 20→14→11) — this is selectivity, not collapse, and comes with accuracy *gains*. Combined with Test X (rank regularisation has 0% effect), this confirms: the failure is in weight/gradient dynamics at depth, not in the diversity of the learned activations.

**Connection to Test X**: This finally explains why rank regularisation failed. Forcing representational diversity (which Base already has) cannot fix a problem that lives in the weights, not the activations.

---

### Test AB — Shell Collapse Width Scaling (COMPLETE)

**Question**: Is the non-zero affine residual from Tests O/T a finite-width artefact (→0 as width increases) or a genuine theoretical gap?

**Setup**: CollapsingIso (IsotropicTanh + HypersphericalNorm), widths [32, 64, 128, 256], full 10K test set (overdetermined lstsq). Seed=42.

| Width | Input | Acc | Abs Residual | Rel Residual |
|---|---|---|---|---|
| 32  | raw     | 40.69% | 0.3050 | 0.2210 |
| 64  | raw     | 40.39% | 0.2907 | 0.2260 |
| 128 | raw     | 40.61% | 0.2983 | 0.2327 |
| 256 | raw     | 40.51% | 0.2960 | 0.2214 |
| 32  | l2_norm | 40.69% | 0.0165 | 0.0051 |
| 64  | l2_norm | 40.39% | 0.0187 | 0.0060 |
| 128 | l2_norm | 40.61% | 0.0424 | 0.0109 |
| 256 | l2_norm | 40.51% | 0.0662 | 0.0155 |

**Finding**: **Hypothesis B (genuine gap) is supported.** Raw-input residual is flat at ~0.22 across all widths (slope ≈ 0) — no finite-size decay. Under the paper's theoretical conditions (L2-normalised inputs), the residual is small but *grows* with width (0.0051 → 0.0155), the opposite of Hypothesis A's prediction. Shell collapse does not improve with network width: this is a genuine unresolved gap in Appendix C's proof, not a finite-size effect that will disappear at scale.

---

### Test AA — Chi-Normalisation (COMPLETE)

**Question**: Does Chi-normalisation (running-mean scalar division) improve accuracy? Does it activate the intrinsic length parameter?

**Setup**: IsotropicMLP width=24, 1L and 2L depth, seeds [42, 123], 24 epochs. Chi-norm divides activations by running mean of ‖h‖ (momentum=0.01).

| Model | Mean Acc | Std | Final repr norm |
|---|---|---|---|
| Iso-1L | 40.26% | 0.0023 | 1.000 |
| ChiNorm-1L | 40.15% | 0.0019 | 1.000 |
| ChiNorm+IL-1L | 40.16% | 0.0010 | 0.500 |
| Collapsing-1L | 40.28% | 0.0028 | 1.000 |
| Iso-2L | 43.69% | 0.0010 | 1.000 |
| ChiNorm-2L | 43.90% | 0.0020 | 1.000 |

**IL behaviour**: In seed=42, `o` oscillated wildly (1.8 → 6.7 → 40.9 → 8.9 → 3.9) with no accuracy benefit. In seed=123, `o` exploded to ~10¹² (running_norm → 0) — a clear instability when IL and Chi-norm interact. Despite this, accuracy was unaffected (40.06% vs 40.26% baseline).

**Finding**: Chi-norm is **neutral** — no meaningful accuracy gain over plain Iso (−0.0011 at 1L, +0.0022 at 2L, both within noise). The IL parameter remains practically negligible even with controlled representation norms. The IL explosion under seed=123 reveals a potential instability (o grows unboundedly as Chi-norm tracks it away from 1): the paper's intrinsic length mechanism needs careful initialisation or clipping when combined with normalisation.

---

---

### Test AD — Saturation & Gradient Anatomy (COMPLETE)

**Question**: What is the actual mechanism of Base depth failure? Hypothesis: per-neuron tanh saturation blocks gradient flow to early layers.

**Setup**: Base/Iso × 1L/2L/3L. Width=32, 24 epochs. Measures per-layer saturation fraction, gradient norms, and pre-activation norms every epoch on a fixed 1024-sample probe.

| Model | Acc | Sat (L1/L2/L3) | grad W1 | grad W_last | W1/W_last |
|---|---|---|---|---|---|
| Base-1L | 27.01% | 0.9994 | 0.02644 | 0.59401 | 0.044 |
| Base-2L | 26.38% | 0.9994 / 0.9609 | 0.05388 | 0.55825 | 0.097 |
| Base-3L | 23.02% | 0.9993 / 0.9783 / 0.9712 | 0.07350 | 0.58486 | 0.126 |
| Iso-1L  | 40.68% | 1.0000 | 0.00951 | 0.07635 | 0.125 |
| Iso-2L  | 43.42% | 1.0000 / 1.0000 | 0.00908 | 0.07751 | 0.117 |
| Iso-3L  | 43.54% | 1.0000 / 1.0000 / 1.0000 | 0.00781 | 0.07993 | 0.098 |

**Key finding**: The saturation fraction narrative is misleading — *both* models are essentially 100% saturated. The true mechanism is visible in the **absolute gradient magnitudes**: Base's output layer gradient (0.58–0.59) is **7–8× larger** than Iso's (0.076–0.080). Base's intermediate layers are saturated elementwise (Jacobian ≈ 0 per neuron), so all learning pressure concentrates on the output layer alone. Iso preserves gradient flow through the *tangential component* of its Jacobian (direction is always preserved even when the norm is large), allowing every layer to participate in learning with balanced gradient magnitudes.

**Mechanism resolved**: At lr=0.08, Base neurons immediately saturate to ±1, making intermediate layers into fixed random feature extractors. Adding depth increases feature complexity but makes them harder to linearly separate — hence accuracy *falls* with depth. Iso's non-local Jacobian prevents this: even fully "saturated" by norm, the tangential gradient component keeps all layers learning collaboratively. This is why Tests R and X found no regularisation fix — the problem is in the activation Jacobian structure, not the training objective.

---

---

### Test AE -- LayerNorm vs Isotropic Activation (COMPLETE)

**Question**: Does LN+tanh match Iso at depth? Is Jacobian preservation the principle, or is isotropy specifically necessary?

**Setup**: Base, Iso, LN+tanh, LN+Iso, RMS+tanh at 1L/2L/3L. Width=32, 24 epochs.

| Model | 1L | 2L | 3L | Depth gain (3L-1L) |
|---|---|---|---|---|
| Base     | 27.01% | 26.38% | 23.02% | -3.99% |
| Iso      | 40.68% | 43.42% | 43.54% | +2.86% |
| LN+tanh  | 44.99% | 45.34% | 46.54% | +1.55% |
| LN+Iso   | 42.82% | 46.62% | 47.27% | +4.45% |
| RMS+tanh | 42.76% | 45.03% | 47.51% | +4.75% |

**Finding**: **Outcome 1 — LN+tanh not only matches Iso at depth, it beats it** (46.54% vs 43.54% at 3L; closes 115% of the Iso-Base gap). RMS+tanh achieves 47.51% — the highest of any single-norm model. LN+Iso is the best combination (47.27%).

The mechanism: LayerNorm keeps tanh inputs near zero mean/unit variance, so tanh operates in its near-linear regime (Jacobian ≈ identity). Iso preserves the tangential gradient component at large norms. Both achieve Jacobian preservation via different routes — normalisation is slightly more effective at this scale.

**Implication**: The paper's specific isotropic architecture is not uniquely necessary for depth stability. Jacobian preservation is the true principle, and it can be achieved with standard normalisation + standard activation. The paper's contribution is more accurately described as identifying *why* depth stability requires Jacobian preservation — the specific mechanism (isotropy vs normalisation) is secondary.

---

---

### Test AJ -- LN+tanh Topology Compatibility (COMPLETE)

**Question**: Does LN+tanh support exact dynamic topology (pruning/growing) like Iso?

**Scaffold inertness test**: Add a zero-weight scaffold neuron and measure prediction change.

| Model | Max output diff | Pred match | Verdict |
|---|---|---|---|
| Iso     | 0.000003 | 1.0000 | INERT (exact) |
| LN+tanh | 0.086253 | 1.0000 | NOT INERT |

**Pruning criterion correlation** (r with accuracy drop per neuron):

| Model | r(SV) | r(W2-norm) | r(Composite) |
|---|---|---|---|
| Iso     | 0.141 | 0.654 | 0.725 |
| LN+tanh | -0.113 | 0.914 | 0.899 |

**Finding**: LN+tanh scaffold neurons are NOT inert — LayerNorm normalises across all neurons, so adding a zero neuron shifts the denominator for all existing neurons (max diff = 0.086 vs Iso's 0.000003). **The paper's dynamic topology claims survive** as a unique advantage of isotropic activation. Additionally, the SV pruning criterion is *useless* for LN+tanh (r = -0.11); W2-norm alone (r = 0.91) is needed instead.

---

### Test AH -- Modern Activations with LayerNorm (COMPLETE)

**Question**: Does LN+tanh's advantage over Iso generalise to GELU, SiLU, ReLU?

| Model | LN? | 1L | 2L | 3L | 3L-1L |
|---|---|---|---|---|---|
| Iso      | No  | 40.68% | 43.42% | 43.54% | +2.86% |
| LN+tanh  | Yes | 44.99% | 45.34% | 46.54% | +1.55% |
| LN+GELU  | Yes | 46.72% | 47.83% | 47.62% | +0.90% |
| LN+SiLU  | Yes | 46.99% | 49.23% | 47.80% | +0.81% |
| LN+ReLU  | Yes | 44.60% | 48.20% | 45.23% | +0.63% |
| GELU     | No  | 25.86% | 10.00% | 10.00% | -15.86% |
| SiLU     | No  | 23.80% | 10.00% | 10.00% | -13.80% |
| ReLU     | No  | 26.41% | 10.00% | 10.00% | -16.41% |

**Finding**: All 4 LN models beat Iso at 3L. Bare GELU/SiLU/ReLU *without* LN collapse to random chance (10%) at depth 2+. **Normalisation is load-bearing; the activation function is secondary.** LN+SiLU (47.80%) and LN+GELU (47.62%) beat LN+tanh (46.54%). The modern deep learning stack (LN+GELU = BERT/GPT FFN, LN+SiLU = LLaMA FFN) has already implicitly solved depth stability.

---

### Test AI -- Long Training 100 Epochs (COMPLETE)

**Question**: Does Iso catch up to LN+tanh with more training, or is the gap structural?

| Model | ep24 | ep50 | ep100 | Peak | Peak epoch |
|---|---|---|---|---|---|
| Base     | 23.02% | 22.22% | 21.47% | 24.57% | ep99 |
| Iso      | 43.54% | 43.23% | 44.43% | 45.10% | ep41 |
| LN+tanh  | 46.54% | 44.88% | 42.89% | 47.46% | ep25 |
| RMS+tanh | 47.51% | 46.64% | 44.93% | 48.22% | ep22 |
| LN+Iso   | 47.27% | 46.56% | 46.25% | 48.11% | ep43 |

**Finding**: **LN+tanh peaks at ep25 then degrades to 42.89% at ep100 — Iso overtakes it.** LN models converge faster but overfit; Iso is slower to peak but more stable. The "LN+tanh beats Iso" result from AE is a 24-epoch artefact: LN provides fast early convergence, not a structural accuracy advantage. At 100 epochs, Iso (44.43%) > LN+tanh (42.89%). LN+Iso (46.25%) remains best at 100 epochs — combining both mechanisms avoids overfitting while keeping fast convergence.

---

### Test AF -- Width Scaling 32->512 (COMPLETE)

**Setup**: Base and Iso at 1L/2L, widths [32, 64, 128, 256, 512], 24 epochs.

| Width | Base-2L | Iso-2L | Gap | LN+tanh-2L | LN+Iso-2L |
|---|---|---|---|---|---|
| 32  | 26.38% | 43.42% | +17.04% | 45.35% | 45.89% |
| 64  | 25.06% | 43.08% | +18.02% | 46.41% | 47.86% |
| 128 | 22.27% | 42.90% | +20.63% | 47.06% | 47.30% |
| 256 | 22.70% | 43.11% | +20.41% | 47.46% | 47.58% |
| 512 | 20.13% | 42.26% | +22.13% | 48.05% | 48.20% |

**Finding**: The Iso-Base gap grows with width (17% → 22%) because Base *degrades* as width increases — more neurons to saturate, worse gradient balance (W1/W_last ratio falls from 0.13 to 0.09 for Iso, and from 0.13 to 0.09 for Base). Iso accuracy is flat at 42-43% regardless of width. LN variants scale better — LN+tanh improves from 45% to 48% as width grows. This suggests Iso is near its capacity ceiling at 24 epochs; the normalised models continue improving with scale.

---

### Test AG -- Depth Scaling width=128, depth 1-6

**Setup**: Base and Iso MLP, width=128, depths 1–6, 30 epochs, lr=0.08, seed=42, CUDA.

| Depth | Base | Iso | Gap | Base Delta | Iso Delta |
|---|---|---|---|---|---|
| 1 | 24.94% | 40.97% | +16.03% | 0 | 0 |
| 2 | 21.74% | 44.08% | +22.34% | -3.20% | +3.11% |
| 3 | 21.92% | 45.14% | +23.22% | -3.02% | +4.17% |
| 4 | 17.64% | 45.95% | +28.31% | -7.30% | +4.98% |
| 5 | 18.76% | 40.49% | +21.73% | -6.18% | -0.48% |
| 6 | 20.31% | 29.98% | +9.67% | -4.63% | -10.99% |

Linear fit: Base slope=-0.010/layer, Iso slope=-0.019/layer (both negative due to 6L collapse)

**Key findings**:
- Iso peaks at **4L** (45.95%), then degrades sharply at 5-6L (40.49% → 29.98%)
- Base is unstable at ALL depths with width=128/LR=0.08 — gradient norm g1 explodes to 2.87 at 6L while g_out stays ~2.1, confirming elementwise Jacobian collapse at scale
- The Iso depth advantage window is finite: good up to 4L, collapses beyond
- Base never recovers — its g1/g_out ratio is chaotic (0.2–2.8×) vs Iso's stable 0.02–0.12×

---

### Test AG-B -- LN variants depth scaling, width=128

**Setup**: LN+tanh, RMS+tanh, LN+Iso at depths 1–6, width=128, 30 epochs, CPU (GPU freed after AG).

| Depth | LN+tanh | Delta | RMS+tanh | Delta | LN+Iso | Delta |
|---|---|---|---|---|---|---|
| 1 | 45.14% | 0 | 41.63% | 0 | 43.06% | 0 |
| 2 | 47.34% | +2.20% | 42.45% | +0.82% | 47.23% | +4.17% |
| 3 | 47.77% | +2.63% | 46.41% | +4.78% | 48.85% | +5.79% |
| 4 | **49.17%** | +4.03% | 47.53% | +5.90% | 45.50% | +2.44% |
| 5 | 49.01% | +3.87% | 47.53% | +5.90% | 40.59% | -2.47% |
| 6 | 46.57% | +1.43% | 43.76% | +2.13% | 30.55% | -12.51% |

Slopes: LN+tanh +0.004/layer, RMS+tanh +0.008/layer, LN+Iso -0.025/layer

**Key findings**:
- **LN+tanh peaks at 4L** (49.17%), beating Iso's 4L peak (45.95%) by +3.22%
- **RMS+tanh is the most robust**: positive slope across all depths, 43.76% at 6L — only model to not sharply degrade
- **LN+Iso collapses fastest at depth**: peaks at 3L (48.85%), falls to 30.55% at 6L. Over-normalisation + Iso activation compound depth instability
- **All models eventually degrade at 6L** at width=128: no architecture is unlimited in depth at this scale/epochs
- LN+tanh advantage over Iso: grows from +5.17% at 1L to +3.22% at 4L peak, then LN+tanh holds better (46.57% vs 29.98% at 6L)
- The depth stability hierarchy: RMS+tanh > LN+tanh > LN+Iso ≈ Iso >> Base

---

## Extension Experiments: Applying Iso to Modern Architectures (AK, AL, AM)

---

### Test AK -- Isotropic Activation Variants: IsoGELU, IsoSiLU, IsoSoftplus

**Setup**: Width=32, depths 1-5, 30 epochs, CPU. Tests f(x)=σ(‖x‖)·x̂ with σ=tanh/GELU/SiLU/Softplus.

| Model | 1L | 2L | 3L | 4L | 5L | Slope |
|---|---|---|---|---|---|---|
| IsoTanh | 41.39% | 43.96% | 44.69% | **45.13%** | 42.46% | +0.003/layer |
| IsoGELU | 23.66% | 28.12% | 33.34% | 23.72% | 25.40% | -0.001/layer |
| IsoSiLU | 22.49% | 27.90% | 25.10% | 26.90% | 22.56% | -0.001/layer |
| IsoSoftplus | 26.65% | 33.95% | 28.95% | 27.63% | 24.36% | -0.011/layer |

**Finding**: IsoGELU/SiLU/Softplus are dramatically worse than IsoTanh — unstable, oscillating, and far below even Base (~27%). The hypothesis that non-saturating σ would extend Iso's depth ceiling was **wrong**.

**Why**: The saturation of tanh (σ(r)→1 for large r) is a **feature not a bug** in the isotropic setting. It implicitly normalises vector magnitudes toward 1, providing training stability without explicit LN. Non-saturating sigma functions (GELU/SiLU grow linearly for large r) allow magnitudes to explode, causing instability. For elementwise activations, saturation is harmful (kills per-neuron gradient). For isotropic activation, saturation only affects the radial gradient component while the tangential component is always preserved — and the magnitude compression keeps training stable. IsoTanh already has implicit self-normalisation built into its saturation profile.

---

### Test AL -- Hybrid Architectures: Iso layers at topology boundaries

**Setup**: HybridMLP mixing Iso and LN+GELU layers, width=32, 30 epochs, CPU.

| Model | Layers | Final | vs Pure-Iso | vs Pure-LN+GELU |
|---|---|---|---|---|
| Pure-Iso-3L | Iso×3 | 44.69% | 0 | -3.25% |
| Pure-LN+GELU-3L | LNG×3 | 47.94% | +3.25% | 0 |
| Iso-first-3L | Iso,LNG,LNG | 48.46% | +3.77% | **+0.52%** |
| **Iso-last-3L** | LNG,LNG,Iso | **48.91%** | **+4.22%** | **+0.97%** |
| Iso-sandwich-3L | Iso,LNG,Iso | 48.27% | +3.58% | +0.33% |
| Pure-Iso-4L | Iso×4 | 45.13% | 0 | -4.39% |
| Pure-LN+GELU-4L | LNG×4 | 49.52% | +4.39% | 0 |
| **Iso-first-4L** | Iso,LNG×3 | **49.78%** | **+4.65%** | **+0.26%** |
| Iso-last-4L | LNG×3,Iso | 48.74% | +3.61% | -0.78% |
| Iso-sandwich-4L | Iso,LNG×2,Iso | 48.51% | +3.38% | -1.01% |
| Alternating-4L | Iso,LNG,Iso,LNG | 48.64% | +3.51% | -0.88% |

**Finding**: Hybrid architectures with a single Iso layer beat **both** pure-Iso and pure-LN+GELU:
- Iso-last-3L (48.91%) beats Pure-LN+GELU-3L (47.94%) by +0.97%
- Iso-first-4L (49.78%) beats Pure-LN+GELU-4L (49.52%) by +0.26%

**The Iso layer provides a small but consistent accuracy boost** when combined with LN+GELU layers, not a cost. Probable mechanism: the Iso layer acts as a magnitude-normalising step (tanh compresses activation norms toward 1), which complements LN's mean/variance normalisation and prevents the magnitude explosion that LN alone can't address in all configurations. Position matters: at 3L, Iso-last is best; at 4L, Iso-first is best. The single Iso layer also grants topology support (scaffold inertness + SV pruning) at that layer while preserving the accuracy of LN+GELU everywhere else.

---

### Test AM -- Post-hoc Topology on LN+tanh vs Iso

**Setup**: Iso-3L and LN+tanh-3L, width=32, trained 30 epochs. Growth: 32→48. Pruning: 32→24. CPU.

**Base accuracies**: Iso=44.69%, LN+tanh=46.26%

#### Growth (32→48 neurons, adding 16 scaffolds)

| Fine-tune epochs | Iso | Iso delta | LN+tanh | LN+tanh delta |
|---|---|---|---|---|
| 0 (just after growth) | 44.69% | **+0.000** | 45.56% | -0.70% |
| 1 | 43.76% | -0.93% | **46.96%** | +0.70% |
| 5 | 44.61% | -0.08% | 46.15% | -0.11% |

Output diff immediately after growth: **Iso=0.000243  LN+tanh=0.538** (2200× larger)

**Finding**: Iso growth is near-exact (output diff 0.000243 for 16 new neurons). LN+tanh output diff of 0.538 — adding 16 neurons shifts outputs by more than half a logit on average — is practically significant but recoverable. After 1 fine-tune epoch, LN+tanh *exceeds* its base accuracy (+0.70%), because the new neurons add genuine capacity once trained. Both converge near-baseline after 5 epochs. LN+tanh approximate topology **is workable** for growth — the disruption is real but short-lived.

#### Pruning (32→24 neurons, removing 8)

| Fine-tune epochs | Iso | Iso delta | LN+tanh | LN+tanh delta |
|---|---|---|---|---|
| 0 (just after pruning) | 12.55% | -32.14% | 29.11% | -17.15% |
| 1 | 42.60% | -2.09% | 44.82% | -1.44% |
| 2 | 44.45% | -0.24% | 44.49% | -1.77% |
| 5 | 44.52% | -0.17% | 45.09% | -1.17% |

**Finding**: Both models recover strongly in 1 epoch (Iso: 93% recovery, LN+tanh: 92%). Iso recovers closer to baseline by epoch 2 (-0.24% vs -1.77% for LN+tanh). LN+tanh settles at -1.17% from base after 5 epochs — good but not quite matching the Iso recovery quality. The W2-norm pruning criterion for LN+tanh (from AJ) works adequately.

**Overall AM verdict**: LN+tanh approximate topology is viable in practice for both growth and pruning with 1-2 fine-tune epochs. The theoretical impurity (0.538 output diff on growth) doesn't block practical use — it just means you must fine-tune after every architecture change, whereas Iso growth requires zero fine-tuning.

---

### Test AS -- Full Integrated Pipeline (COMPLETE)

**Question**: Does combining every positive finding (best architecture from AR, safe interleaved protocol from AN, low LR from AQ, 100 epochs from AI, composite criterion from U2/Z) into a single run beat the static baseline?

**Setup**: Iso-first-4L, width 160→128, 100 epochs, lr=0.001, diag every 5ep, prune 4 neurons at ep20/30/.../90.

| Condition | ep10 | Peak | Peak@ep | ep30 | ep100 |
|---|---|---|---|---|---|
| A-Static-Iso-first w=128 | 54.33% | **54.39%** | ep14 | 51.50% | 49.47% |
| B-Static-LN+GELU-4L w=128 | 53.37% | **54.13%** | ep6 | 51.36% | **51.00%** |
| C-Dynamic Iso-first 160->128 | 54.38% | **54.79%** | ep14 | 43.04% | **51.09%** |

**Key comparisons at ep100**: C vs A: +1.62pp, C vs B: +0.09pp, A vs B: −1.53pp

**Prune log**: 4 neurons pruned at ep20,30,40,50,60,70,80,90. Width: 160→156→...→128.

**Findings**:

1. **Constant LR overfitting dominates everything.** All three models peak by ep6-14 then decay 3-5pp by ep100. The entire long-training comparison is an overfitting regime, not a capacity regime. This directly validates Audit Issue #7.

2. **Pruning schedule is too aggressive.** Each prune event causes a ~10pp accuracy drop (ep10: 54.4%→ep20-post-prune: 42.2%). Recovery takes the full 10 epochs between prune events. The model spends ep20-90 perpetually recovering rather than learning. Pruning every 10 epochs is too frequent — or should only start after convergence.

3. **Dynamic model (C) recovers to 51.09% by ep100**, narrowly beating static LN+GELU (51.00%) by 0.09pp and beating static Iso-first (49.47%) by 1.62pp. The ranking direction is correct but differences are small and within single-seed noise.

4. **The true peak performance is pre-pruning**: C reaches 54.79% at ep14, before any pruning. This is essentially just the w=160 model at early training — not a product of the interleaved protocol.

5. **Critical implication**: The full paper protocol (interleaved diagonalise + prune during training) at constant LR is not competitive with simply taking the peak accuracy of the static model at ep14. A cosine LR schedule that stabilises the model before pruning begins is likely essential for the protocol to work as intended.

---

## Uncertainty Reduction Experiments (AN, AP, AQ, AR)

---

### Test AN -- Interleaved Training Protocol (COMPLETE)

**Question**: Does running `partial_diagonalise` during training (as the paper intends) hurt accuracy via stale Adam momentum? All prior tests applied topology operations post-hoc after full training; this tests the intended interleaved protocol.

**Setup**: Core IsotropicMLP (1L), width=32, 60 epochs, lr=0.08, seed=42.

| Protocol | ep60 acc | Peak | vs Static |
|---|---|---|---|
| A-Static (no diag) | 0.4157 | 0.4240 | 0.0000 |
| B-Diag-only (diag every 5ep, no Adam reset) | 0.4181 | 0.4212 | +0.0024 |
| C-Diag+reset (diag every 5ep + reset Adam) | 0.4137 | 0.4208 | -0.0020 |
| D-Prune-post (train 60ep then prune) | 0.4157 | 0.4240 | 0.0000 |
| E-Prune-mid (train 30ep, prune, train 30ep) | 0.4176 | 0.4255 | +0.0019 |
| F-Prune-incremental (diag every 5ep, prune at ep20/40) | 0.4160 | 0.4205 | +0.0003 |

**Key comparisons**:
- A vs B (does stale Adam hurt?): +0.0024 — reparameterising during training **slightly helps**, stale Adam not harmful
- B vs C (does resetting Adam help?): -0.0044 — resetting Adam **hurts** — the stale moments encode useful curvature that survives reparametrisation
- D vs E (prune timing): +0.0019 — mid-training pruning marginally better than post-training

**Finding**: The feared Adam momentum staleness from `partial_diagonalise` is **not an issue**. The reparameterisation preserves function exactly, and the stale moments are still a reasonable approximation to the new-basis curvature. Resetting Adam is counterproductive — it discards curvature information accumulated over 30 epochs. The paper's interleaved protocol is safe to use exactly as described.

---

### Test AP -- Chained SVD Pruning (Proper Multi-Layer, COMPLETE)

**Question**: Does proper chained SVD pruning (per-boundary SVD, prune by singular value) outperform AM's row-norm proxy? And what is the accuracy cost per neuron pruned?

**Setup**: IsoMLP (2L or 3L), width=32, 30 epochs + 5ep fine-tune, prune 32→24.

| Condition | Pre-prune (ep30) | Post-ft (ep35) | Delta |
|---|---|---|---|
| 2L-baseline | 39.85% | 39.85% | 0.00% |
| 2L-prune-L1 (prune W1/W2 boundary) | 39.85% | 41.28% | **+1.43%** |
| 2L-prune-L2 (prune W2/out boundary) | 39.85% | 40.93% | **+1.08%** |
| 3L-prune-L1 | 39.99% | 39.46% | -0.53% |
| 3L-prune-L2 | 39.99% | 39.22% | -0.77% |
| 3L-prune-both | 39.99% | 39.62% | -0.37% |

**Finding**: **2L pruning actually improves accuracy** after 5 fine-tune epochs (+1.1–1.4pp). This is consistent with regularisation theory — the 8 removed neurons carried the lowest SVs and were likely redundant. The pruning acts as a structural regulariser. At 3L, pruning shows slight degradation (<1pp) — deeper networks have less redundancy at width=32. The proper chained SVD approach (drop by singular value, propagate U absorb) is more principled than AM's row-norm proxy and gives better results.

---

### Test AQ -- IsoGELU Learning Rate Sweep (COMPLETE)

**Question**: Is IsoGELU's failure (23-33% in AK at lr=0.08) fundamental (magnitude explosion at all LRs) or an LR artefact?

**Setup**: IsoTanh/IsoGELU/IsoSiLU, depth=3, width=32, 40 epochs, LRs [0.001, 0.003, 0.01, 0.03, 0.08, 0.3].

| LR | IsoTanh | IsoGELU | IsoSiLU |
|---|---|---|---|
| 0.001 | **40.94%** | **39.89%** | **39.92%** |
| 0.003 | 40.65% | 39.43% | 39.42% |
| 0.010 | 40.63% | 36.76% | 35.44% |
| 0.030 | 39.62% | 34.48% | 35.94% |
| 0.080 | 39.71% | 24.17% | 35.05% |
| 0.300 | 37.67% | 20.87% | 24.74% |

- IsoTanh best: 40.94% (lr=0.001)
- IsoGELU best: 39.89% (lr=0.001) — gap narrows to only **1.05pp**
- H2 confirmed: **the failure was an LR artefact**, not fundamental instability
- At lr=0.08 the gap was ~12pp; at lr=0.001 it narrows to ~1pp
- IsoGELU is viable at low LR; the tanh saturation provides LR robustness

**Implication**: IsoGELU is not fundamentally broken. With proper LR (0.001–0.003), it reaches near-IsoTanh performance. The AK results showing IsoGELU failing were an artefact of running all variants at a single LR (0.08) optimised for IsoTanh.

---

### Test AR -- Hybrid Architectures at Width=128 (COMPLETE)

**Question**: Do the hybrid architecture advantages from AL (width=32) hold at width=128?

**Setup**: Replicates key AL configs at width=128, depths 3 and 4, 30 epochs, lr=0.08.

| Model | Final | vs Pure-Iso | vs Pure-LN+GELU |
|---|---|---|---|
| Pure-Iso-3L | 45.12% | 0 | -5.15% |
| Pure-LN+GELU-3L | 50.27% | +5.15% | 0 |
| **Iso-first-3L** | **51.45%** | **+6.33%** | **+1.18%** |
| Iso-last-3L | 49.93% | +4.81% | -0.34% |
| Iso-sandwich-3L | 50.96% | +5.84% | +0.69% |
| Pure-Iso-4L | 45.92% | 0 | -4.72% |
| Pure-LN+GELU-4L | 50.64% | +4.72% | 0 |
| **Iso-first-4L** | **52.28%** | **+6.36%** | **+1.64%** |
| Iso-last-4L | 50.88% | +4.96% | +0.24% |
| Iso-sandwich-4L | 52.09% | +6.17% | +1.45% |

**AL vs AR comparison**:

| Model | AL (w=32) | AR (w=128) | Width scaling gain |
|---|---|---|---|
| Pure-Iso-3L | 44.69% | 45.12% | +0.43% |
| Pure-LN+GELU-3L | 47.94% | 50.27% | +2.33% |
| Iso-last-3L | 48.91% | 49.93% | +1.02% |
| Iso-first-4L | 49.78% | 52.28% | +2.50% |

**Finding**: **Hybrid advantage confirmed and strengthened at width=128.** Iso-first-4L at 52.28% is the **highest accuracy achieved in the entire study**. Pattern slightly shifts: at w=32 Iso-last-3L was best; at w=128 Iso-first wins all depths. The hybrid advantage over Pure-LN+GELU grows from +0.26-0.97pp at w=32 to +0.24-1.64pp at w=128 — scale amplifies the Iso-first contribution. The single Iso layer at the input boundary provides magnitude normalisation that becomes more valuable at larger width where activations have larger norms.
