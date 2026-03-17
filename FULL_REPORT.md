# Empirical Investigation of Isotropic Activation Functions and Dynamic Neural Network Topologies
### A Full Report on 34 Experiments
#### Based on: "On De-Individuated Neurons: Continuous Symmetries Enable Dynamic Topologies" — Bird, 2026 (arXiv:2602.23405v1)

---

## 1. What the Paper Claims

The paper argues that standard neural networks have a fundamental architectural problem: elementwise activation functions (tanh, ReLU, GELU etc.) individualise neurons. Each neuron has a fixed identity tied to its position in the basis — its weight row in W1 and its weight column in W2. This makes the network's function sensitive to which specific neuron does what. As a result, you cannot add or remove neurons without changing the function the network computes, and you cannot compare neuron importance across layers in a principled way.

The proposed solution is **isotropic activation functions**: functions that are equivariant under the full orthogonal group O(n). The key example is isotropic tanh:

```
f(x) = tanh(‖x‖) · x / ‖x‖
```

This function takes in a vector, computes its norm, applies tanh to the scalar norm to get a scaling factor, and then returns the original vector rescaled by that factor. The direction of x is always preserved exactly. Rotating the input rotates the output by the same amount. Because direction is always preserved, no individual dimension (neuron basis) has special status — the function is fully basis-independent.

This basis independence enables three things:
1. **Reparameterisation**: You can diagonalise weight matrices via SVD without changing the function the network computes. This ranks neurons by importance (singular value = importance).
2. **Pruning**: Remove low-SV neurons. Their contribution was small; loss is controlled.
3. **Growth (neurogenesis)**: Add scaffold neurons with zero weight rows/columns. They are exactly inert until training moves their weights.

Together these enable **dynamic topology** — changing the network architecture during or after training with minimal disruption.

---

## 2. Mathematical Foundations (Tests A, B, C)

### 2.1 Reparameterisation Invariance (Test A)

The paper claims that partial left-diagonalisation of a weight matrix W1 = UΣVᵀ preserves the network function exactly. The transformation is:

```
W1' = ΣVᵀ
W2' = W2 · U
b1' = Uᵀ · b1
```

This is confirmed: the maximum logit difference after reparameterisation is **3.06×10⁻⁵** (pure float32 rounding error). 100% of 10,000 test samples classify identically. This is the foundation of everything else — if the SVD didn't preserve function, none of the dynamic topology would work.

Why does it work? Because O(n)-equivariance means the activation function doesn't care which orthogonal basis it operates in. Absorbing the unitary matrix U into W2 and simultaneously rotating W1 is invisible to the isotropic activation layer between them.

### 2.2 Neurogenesis Invariance (Test B)

Adding scaffold neurons (zero row in W1, zero column in W2) leaves the network function unchanged. This is confirmed exactly: max diff = **3.10×10⁻⁵** when adding up to 10 scaffold neurons simultaneously.

Why are scaffold neurons inert? The zero W2 column means the new neuron contributes exactly 0 to the output, regardless of what the activation layer does with its pre-activation. The zero W1 row means the new neuron's pre-activation is always zero (plus bias — which is why biases should be initialised small). The network literally ignores the new neuron until gradient updates move those weights.

### 2.3 Intrinsic Length (Test C)

The paper introduces a trainable scalar `o = exp(λ) > 0` per layer to absorb the residual bias term that arises when pruning near-zero-SV neurons (the bias has a tiny contribution from the pruned neuron that gets lost). The correction is proportional to |b_i|²/‖h‖². In practice, **this is negligible** — the correction ratio is 1.000× when biases are small relative to the representation norm. The parameter is theoretically necessary for exact invariance near zero, practically irrelevant when scaffold biases are initialised near zero (recommended practice).

---

## 3. The Core Accuracy Advantage (Tests D, E, L, M, Q)

Before investigating dynamic topology, a more basic question: is isotropic tanh actually better than standard tanh as an activation function?

**Yes, substantially.** Across all experiments, isotropic MLP consistently beats standard MLP by 13–16 percentage points on CIFAR-10. This is not a tuning artefact — it holds across:
- Widths from 8 to 512 neurons
- Depths from 1 to 4 layers
- Datasets: MNIST (+7.3%), Fashion-MNIST (+11.9%), CIFAR-10 (+14–16%)
- Multiple random seeds

The most striking result is how the gap changes with depth:

| Depth | Iso (width=24) | Base (width=24) |
|---|---|---|
| 1L | 40.6% | 27.6% |
| 2L | 43.4% | 25.6% |
| 3L | 44.7% | 21.4% |

**Iso gains ~3% per added layer. Base loses ~3% per added layer.** This divergence, confirmed across multiple width settings and seeds, is the central empirical finding of the paper — and one of the strongest results in the entire experimental suite.

---

## 4. Why Base Networks Fail at Depth (Tests R, X, AC, AD)

This took four experiments to fully resolve. The answer is not obvious.

### 4.1 What it is NOT

**Not gradient vanishing (Test R)**: In classic vanishing gradient, the gradient norm falls to zero in early layers. The opposite is true here — Base-3L has *larger* W1 gradient norms (0.074) than Base-1L (0.026). Gradients are not vanishing; something else is failing.

**Not representational collapse (Test AC)**: You might expect that deep Base networks collapse their hidden representations to low rank — all neurons learn the same thing, diversity is lost. The Participation Ratio (PR = (Σσᵢ)²/Σ(σᵢ²), measuring representation diversity) shows Base-3L has PR=18.7, essentially identical to Iso-1L's PR=20.3. Base maintains a broad, diverse representation spectrum even as its accuracy degrades. This rules out representational collapse entirely.

**Not fixable by regularisation (Test X)**: Since AC showed diverse representations, perhaps forcing even more diversity would help? A spectral isotropy regulariser was applied (penalising deviation from uniform singular value distribution). Result: **zero effect at every depth and every regularisation strength** — to three decimal places. This conclusively shows the failure is not in the training objective or the loss landscape.

### 4.2 What it IS: Elementwise Jacobian Collapse (Test AD)

The mechanism was resolved by examining saturation and gradient anatomy at the per-layer level.

Both models are essentially 100% saturated — Base neurons hit tanh(x) ≈ ±1 at 99.94% rate; Iso vectors have ‖x‖ >> 1 at 100% rate. So saturation fraction is not the distinguishing feature.

The distinguishing feature is the **Jacobian structure**:

For standard tanh, the Jacobian of the activation at sample i is:
```
J_Base = diag(sech²(h_1), sech²(h_2), ..., sech²(h_d))
```

When |h_i| >> 1 (saturated), sech²(h_i) ≈ 0. The Jacobian is diagonal with near-zero entries. Backpropagating through this layer multiplies the gradient by ≈ 0 per neuron. The layer is essentially disconnected from the gradient flow.

For isotropic tanh, the Jacobian at a vector x with norm r is:
```
J_Iso = tanh'(r)/r · x̂x̂ᵀ + tanh(r)/r · (I - x̂x̂ᵀ)
```

The first term (radial component) does go to zero when r is large — tanh'(r) → 0 just as sech²(h) → 0. But the second term (tangential component) equals tanh(r)/r, which at large r is 1/r → a small but nonzero value times (I - x̂x̂ᵀ). The tangential component of the gradient — the component orthogonal to the direction of x — is always preserved, regardless of how saturated the norm is.

This is why all layers in an Iso network keep learning. Even when the norm is very large, the directions of gradients are transmitted faithfully through every layer. The absolute gradient magnitudes tell the whole story:

| Model | grad W1 | grad W_last | W1/W_last ratio |
|---|---|---|---|
| Base-1L | 0.026 | 0.594 | 0.044 |
| Base-3L | 0.074 | 0.585 | 0.126 |
| Iso-1L | 0.010 | 0.076 | 0.125 |
| Iso-3L | 0.008 | 0.080 | 0.098 |

Base's output layer gradient is **7–8× larger** than all intermediate layers combined. The output layer is doing all the learning; every hidden layer is frozen by its Jacobian. Iso's gradients are balanced within a factor of ~10× across all layers, and the ratio stays approximately constant as depth increases.

Adding more layers to a Base network doesn't add more learning capacity — it just adds more frozen random feature extractors between input and the single learning layer. This is why accuracy *falls* with depth: more fixed random transformations make the final linear problem harder.

---

## 5. Iso Is Not Uniquely Necessary: The LayerNorm Finding (Tests AE, AH, AI)

One of the most important discoveries from the extended experiments: **isotropic activation is not the only way to achieve depth stability.**

### 5.1 LayerNorm Also Solves the Problem (Test AE)

LayerNorm before the activation function normalises the pre-activation vector to zero mean and unit variance before feeding it to tanh. This keeps tanh inputs in the near-linear regime (tanh ≈ identity near zero), meaning sech²(x) ≈ 1. The elementwise Jacobian stays near the identity matrix — gradient flows freely.

Results at depth 3, width=32:

| Model | 3L accuracy | vs Base (23%) |
|---|---|---|
| Base | 23.02% | — |
| Iso | 43.54% | +20.52% |
| LN+tanh | **46.54%** | +23.52% |
| RMS+tanh | **47.51%** | +24.49% |
| LN+Iso | **47.27%** | +24.25% |

LN+tanh not only matches Iso — it beats it by 3 percentage points at 24 epochs. RMS+tanh (RMSNorm before tanh) achieves 47.51%. Both close more than 100% of the Iso-Base gap.

### 5.2 The Effect Generalises to All Modern Activations (Test AH)

LN+GELU (BERT/GPT), LN+SiLU (LLaMA/Mistral), LN+ReLU all beat Iso at 3 layers. Critically, these activations without LN are catastrophically worse than either Iso or Base:

| Model | 2L accuracy |
|---|---|
| Iso | 43.42% |
| LN+GELU | 47.83% |
| LN+SiLU | **49.23%** |
| GELU (no LN) | 10.00% (random!) |
| SiLU (no LN) | 10.00% (random!) |
| ReLU (no LN) | 10.00% (random!) |

Bare GELU/SiLU/ReLU without normalisation collapse to random-chance at depth 2. The normalisation isn't optional — it's the entire reason modern deep learning works. The transformer block (LN + attention + LN + FFN) is, in part, an empirically discovered solution to the same depth stability problem that isotropic activation solves from a theoretical direction.

### 5.3 LN Overfits; Iso Is More Stable Long-Term (Test AI)

The critical nuance: LN's advantage at 24 epochs reverses at 100 epochs.

| Model | ep24 | ep50 | ep100 | Peak epoch |
|---|---|---|---|---|
| Iso | 43.54% | 43.23% | **44.43%** | ep41 |
| LN+tanh | **46.54%** | 44.88% | 42.89% | ep25 |
| RMS+tanh | **47.51%** | 46.64% | 44.93% | ep22 |
| LN+Iso | **47.27%** | 46.56% | 46.25% | ep43 |

LN+tanh peaks at epoch 25 then degrades monotonically — it is overfitting. Iso peaks at epoch 41 and holds nearly flat. By epoch 100, plain Iso (44.43%) beats plain LN+tanh (42.89%). LN+Iso (46.25% at ep100) is the best long-term combination — it converges fast (via LN) and stays stable (via Iso's regularisation property).

**The revised picture**: LN is a convergence accelerator; Iso is a stability mechanism. They address slightly different problems. For short training runs, LN wins. For long training or when generalisation is paramount, Iso is more reliable.

---

## 6. Scaling: Width and Depth (Tests AF, AG, AG-B)

### 6.1 Width Scaling (Test AF)

Iso accuracy is **flat across widths** 32 to 512 at 24 epochs (~42-43%). Base degrades as width increases. LN variants improve with width (45% → 48%).

| Width | Base-2L | Iso-2L | Gap | LN+tanh-2L |
|---|---|---|---|---|
| 32 | 26.38% | 43.42% | +17.04% | 45.35% |
| 128 | 22.27% | 42.90% | +20.63% | 47.06% |
| 512 | 20.13% | 42.26% | +22.13% | 48.05% |

Why does Iso not benefit from more width? At 24 epochs, Iso has already converged near its representational ceiling for this task at this depth. More neurons don't help if you've already learned the relevant features. Base degrades with width because more neurons means more Jacobian collapse — a wider saturated layer is a worse random feature extractor. LN models improve because normalisation prevents saturation, so wider means genuinely more capacity.

### 6.2 Depth Scaling at Width=128 (Tests AG, AG-B)

This is the most comprehensive depth scaling result. All models at depths 1–6 with width=128:

| Depth | Base | Iso | LN+tanh | RMS+tanh | LN+Iso |
|---|---|---|---|---|---|
| 1 | 24.94% | 40.97% | 45.14% | 41.63% | 43.06% |
| 2 | 21.74% | 44.08% | 47.34% | 42.45% | 47.23% |
| 3 | 21.92% | 45.14% | 47.77% | 46.41% | 48.85% |
| 4 | 17.64% | **45.95%** | **49.17%** | **47.53%** | 45.50% |
| 5 | 18.76% | 40.49% | 49.01% | 47.53% | 40.59% |
| 6 | 20.31% | 29.98% | 46.57% | 43.76% | 30.55% |

**Iso's depth window is finite.** It gains +5% going from 1L to 4L, then collapses at 5-6L. At depth 6, Iso (29.98%) is barely above Base (20.31%). LN+tanh is far more depth-robust — it holds 46.57% even at 6L. RMS+tanh is the most robust of all (positive slope across all depths, 43.76% at 6L).

**Why does Iso fail at very deep networks?** The tangential gradient preservation mechanism works per-layer. At 5-6 layers, even a small per-layer signal reduction compounds multiplicatively. The tangential component is preserved but it's a projection onto a subspace that gets increasingly constrained as depth grows. LN avoids this by keeping activations in the linear tanh regime where the Jacobian is close to the identity rather than a projection.

The depth stability hierarchy: **RMS+tanh > LN+tanh > Iso ≈ LN+Iso >> Base**

Note that LN+Iso collapses even faster than plain Iso at depth 6 (30.55% vs 29.98%). The combination of LN's normalisation and Iso's projection seems to over-constrain the representation at very deep networks.

---

## 7. Dynamic Topology: Pruning (Tests G, H, U2, V, W, Y, Z)

### 7.1 The Core Claim: SV Predicts Pruning Impact (Test G)

After diagonalising, neurons are ranked by their singular value Σᵢᵢ. The paper claims small-SV neurons are less important and can be removed with small accuracy loss. This is confirmed: r(SV, accuracy_drop) = 0.77. The correlation is strong and consistent across widths.

### 7.2 The Best Pruning Criterion (Tests U2, Z)

The paper mentions in a footnote that gradient-based criteria might improve on the SV alone. Testing four criteria:

| Criterion | Pearson r |
|---|---|
| SV (Σᵢᵢ) alone | 0.815 |
| W2 column norm alone | 0.798 |
| **Composite: Σᵢᵢ × ‖W2 col‖** | **0.859** |
| Gradient \|dL/dΣᵢᵢ\| | **0.464** (fails badly) |

The composite criterion is best (+4.4% over SV alone). The gradient criterion, despite being the paper's own footnote suggestion, fails significantly — gradient magnitude at just a few batches is too noisy. The SV captures the structural importance of the neuron independent of the current data batch; the W2 column norm captures how much the neuron's output matters to the next layer. Together they form the most reliable criterion.

For LN+tanh networks, the SV criterion is essentially useless (r = -0.11). The W2 column norm alone (r = 0.91) is needed. The SV criterion is specific to the diagonalised Iso architecture.

### 7.3 When to Prune (Tests V, S)

Pruning is best done late — after training has converged (epoch 24 in 24-epoch runs). The SV spectrum needs time to develop meaningful differentiation:

| Epoch | Condition number (SV_max/SV_min) |
|---|---|
| 0 | 1.2 (nearly uniform) |
| 1 | 4.0 (fast initial differentiation) |
| 24 | 8.4 (mature spectrum) |

A condition number of 8.4 means the most important neuron's SV is 8.4× larger than the least important — giving reliable pruning signal. Pruning at epoch 0 also works surprisingly well because even the initial spectrum has some differentiation (condition number 1.2 is still nonzero). Middle epochs (4–16) are marginally worse.

### 7.4 Recovery After Pruning (Test Y)

Pruning 50% of neurons (24→12) and fine-tuning:

| Fine-tune epochs | Recovery |
|---|---|
| 0 | 0% (accuracy drops 4.5%) |
| **1** | **91%** |
| 3 | 115% (exceeds baseline) |

A single fine-tune epoch recovers 91% of the accuracy loss from removing half the neurons. After 3 epochs of fine-tuning, the pruned model actually outperforms the unpruned model — apparently the pruning acts as a regulariser, forcing the remaining neurons to be more efficient. This is the most striking practical result for dynamic topology.

### 7.5 Overabundance (Test W)

The paper suggests starting with 2× more neurons than needed, letting them compete during training, then pruning. The mean gain over training directly at the target width: **+0.14%** — essentially noise. Inconsistent across widths (positive at small widths, negative at larger ones). The biological analogy (neural pruning in development) is appealing but doesn't provide measurable benefit at these scales.

### 7.6 Sequential Pruning Stability (Test H)

Pruning neurons one at a time (always the smallest SV), Iso can be pruned to 50% width with near-zero accuracy loss. The cliff — where accuracy drops sharply — appears only when you've removed more than 70% of neurons. This demonstrates the robustness of the architecture to iterative compression.

---

## 8. Dynamic Topology: Growth (Tests B, K, P)

### 8.1 Scaffold Neurons are Inert (Test B)

Confirmed: adding up to 10 scaffold neurons simultaneously produces no measurable prediction change (max diff 3×10⁻⁵). The zero W2 column mathematically guarantees zero contribution to output. This is the key property that makes neurogenesis safe — you can add neurons to a deployed model without changing its current behaviour.

### 8.2 Gradient Flows Only if W2 ≠ 0 (Test K)

The paper implies that the non-diagonal Jacobian of the isotropic activation will route gradients to scaffold neurons via W2, even starting from zero. This is incorrect.

| Scaffold W2 init | W1 gradient after 1 step |
|---|---|
| Zero W2 | 0.000000 (exactly zero) |
| Random W2 (scale=0.01) | 0.000252 |

When W2 = 0, the gradient dL/dW1_new = (dL/dh) · Jᵀ · W2_newᵀ = (dL/dh) · Jᵀ · 0 = 0 exactly. The non-diagonal Jacobian helps only when W2 is nonzero — it distributes the gradient from the loss *through* W2 to reach W1. The chain rule requires nonzero W2. **Always initialise scaffold W2 columns with small random values.**

### 8.3 Semi-Orthogonal Init is Impractical (Test P)

The paper recommends semi-orthogonal W2 initialisation for new neurons. This requires the new column to be orthogonal to all existing W2 columns. When width > num_classes (always true in practice — 24 neurons, 10 classes), the existing columns already span all of ℝ¹⁰, making true orthogonality geometrically impossible. Random init at scale=0.01 gives identical performance and is always possible.

---

## 9. The Topology Compatibility Problem (Test AJ)

A critical question: if LN+tanh is better than Iso at short training, should we just use LN+tanh everywhere and forget about isotropic activation?

**No — because LN+tanh cannot support dynamic topology.**

When you add a scaffold neuron (zero weight) to an LN+tanh network:

| Model | Max output diff after scaffold insertion | Verdict |
|---|---|---|
| Iso | 0.000003 | Exactly inert |
| LN+tanh | **0.086** | NOT inert |

LayerNorm normalises by computing the mean and variance across all neurons in the layer. When you add a new neuron (even with zero weight), you change the denominator of the normalisation for every existing neuron. Their outputs shift — not by much, but not by zero either. The scaffold is not inert.

**This is a fundamental incompatibility.** Dynamic topology — adding neurons without changing the existing network function — is only possible with Iso. LN breaks the mathematical guarantee that makes neurogenesis safe.

The paper's topology claims survive as a *unique advantage of isotropic activation that no normalisation-based approach can replicate*. The finding reframes the contribution: Iso is not primarily about accuracy (LN is better there). It is primarily about enabling architectures that can structurally change at runtime.

---

## 10. Shell Collapse: A Genuine Proof Gap (Tests F, O, T, AB)

The paper's Appendix C argues that a specific variant (CollapsingIso — isotropic tanh followed by hyperspherical normalisation) should collapse to an affine map. If this were true, the expressivity of the collapsed model would be severely limited.

The experimental finding: **this collapse is far weaker than claimed.**

The affine residual (how far CollapsingIso's representation is from an affine transformation of the input) was measured across widths 32–256:

| Width | Affine residual (raw input) | Affine residual (L2-normalised) |
|---|---|---|
| 32 | 0.305 | 0.005 |
| 64 | 0.291 | 0.006 |
| 128 | 0.298 | 0.042 |
| 256 | 0.296 | 0.066 |

The raw-input residual is flat at ~0.22–0.30 across all widths — no decay. This is not a finite-size effect that will vanish at large scale. Even under the paper's specific theoretical conditions (L2-normalised inputs), the residual is small but *grows* with width — opposite to what Hypothesis A (finite-size effect) predicts.

**Appendix C's proof holds only asymptotically.** At any finite scale — including any real network — CollapsingIso retains genuine nonlinearity. The proof gap is a genuine unresolved issue in the paper's theory, not a measurement artefact.

---

## 11. The Complete Picture: What Each Activation Does

After 34 experiments, here is a precise characterisation of each model type:

### Base (standard tanh, no normalisation)
- **Mechanism**: Elementwise Jacobian collapses (sech² → 0 under saturation)
- **Depth behaviour**: Degrades at every depth level (-3 to -6% per layer)
- **Width behaviour**: Degrades as width increases (more neurons to saturate)
- **Long training**: Stays bad (inherent structural failure)
- **Topology support**: No (elementwise activations individuate neurons)
- **Use case**: Never recommended

### Iso (isotropic tanh, no normalisation)
- **Mechanism**: Tangential Jacobian preserved regardless of norm; all layers learn
- **Depth behaviour**: Gains at 1→4L, collapses at 5-6L
- **Width behaviour**: Flat (near convergence ceiling at 24 epochs)
- **Short training (24-30 ep)**: Beats Base by 15-20%; beaten by LN variants by 3-5%
- **Long training (100 ep)**: Overtakes LN+tanh; more stable against overfitting
- **Topology support**: Yes — exact scaffold inertness, SV pruning works well
- **Use case**: Dynamic topology systems; long training runs

### LN+tanh (LayerNorm + tanh)
- **Mechanism**: LN keeps tanh in near-linear regime (sech² ≈ 1)
- **Depth behaviour**: Gains to 4L, then slow decline; holds well to 6L (46.57%)
- **Width behaviour**: Improves with width (genuine capacity scaling)
- **Short training**: Best at 24-30 epochs (+3-5% over Iso)
- **Long training**: Overfits from ep25; falls below Iso by ep100
- **Topology support**: No — LN scaffold insertion is not inert
- **Use case**: Fixed architectures, short training schedules

### RMS+tanh (RMSNorm + tanh)
- **Mechanism**: RMSNorm normalises by magnitude only (no mean subtraction)
- **Depth behaviour**: Most robust — only model with positive slope across all depths (+0.008/layer to depth 6)
- **Topology support**: No
- **Use case**: Maximum depth robustness in fixed architectures

### LN+Iso (LayerNorm + isotropic tanh)
- **Mechanism**: Both normalisation mechanisms active simultaneously
- **Depth behaviour**: Peaks at 3L (48.85%), collapses fastest at 6L (30.55%) — over-constrained
- **Long training**: Best at 100 epochs (46.25%) — fast convergence + stability
- **Topology support**: No (LN breaks scaffold inertness)
- **Use case**: Long training, fixed architecture, maximum accuracy

### LN+GELU / LN+SiLU (modern transformer FFN)
- **Mechanism**: LN keeps GELU/SiLU in their non-saturating regimes
- **Depth behaviour**: Scales well to 3-4L; LN+SiLU best at depth 2 (49.23%)
- **Topology support**: No
- **Use case**: Already the industry standard — this is the transformer FFN block

---

## 12. Revised Narrative on the Paper's Contribution

### What the paper got right
1. The mathematical framework for dynamic topology is correct and implementable (Tests A, B)
2. Isotropic activation genuinely solves depth stability (Tests D, E, Q, AD)
3. SV-based pruning works well (Tests G, U2)
4. Dynamic topology enables fast recovery from pruning (Test Y)
5. The scaffold inertness property is unique to isotropic activation (Test AJ)

### What the paper overstates
1. **Isotropic activation as uniquely necessary for depth stability**: LN+tanh and all modern normalised activations achieve the same thing via a different mechanism. The paper presents Iso as the solution; the experiments show it is *one* solution to a more general problem (Jacobian preservation).
2. **Dynamic topology providing an accuracy advantage**: Fair comparison (equal training time) shows Dynamic ≈ Static accuracy. The value is flexibility, not performance (Test J).
3. **Shell collapse being exact**: Appendix C's proof has a genuine finite-scale gap (Tests O, T, AB). The collapse is much weaker than claimed.

### What the paper misses
1. **The depth ceiling**: Iso fails at depth 5-6. The paper only tests shallow networks (1-3 layers) where Iso's advantage is cleanest.
2. **The long-training reversal**: LN+tanh overfits; Iso is more stable. At the paper's training duration, this is invisible.
3. **W2 must be nonzero**: The paper implies isotropic Jacobian routes gradient to scaffold neurons; Test K shows W2 must be nonzero first.

### The unified framing
The paper's deepest contribution is identifying **why** standard networks fail at depth: elementwise activations make intermediate layers disappear under saturation. The paper's proposed fix (isotropic activation) is correct, elegant, and unique in enabling dynamic topology. It is not the only fix for accuracy — normalisation achieves similar depth stability. But it is the only fix that preserves the mathematical structure (basis independence) needed for architecture change at runtime.

**The principle is**: depth stability requires Jacobian preservation. This can be achieved by isotropy (preserved tangential component) or normalisation (prevented saturation). Isotropy additionally gives you dynamic topology; normalisation does not.

---

## 13. Practical Recommendations

| Decision | Recommendation | Why |
|---|---|---|
| Activation (fixed architecture) | LN+tanh or LN+SiLU | Best short-term accuracy, scales with width/depth |
| Activation (long training) | LN+Iso | Fast convergence + stable; best at 100 epochs |
| Activation (dynamic topology) | Iso (no LN) | Only architecture with exact scaffold inertness |
| Depth | ≤4 layers for Iso; ≤5 for LN+tanh; any for RMS+tanh | Iso collapses at 5-6L |
| Pruning criterion | Composite: Σᵢᵢ × ‖W2 col‖ | r=0.859 (vs r=0.815 for SV alone) |
| Pruning criterion for LN models | W2 column norm alone | SV criterion is useless for LN (r=-0.11) |
| Pruning timing | After convergence (late training) | Spectrum needs time to mature |
| Fine-tune after pruning | 1 epoch minimum; 3 epochs optimal | 91% recovery in 1 epoch; exceeds baseline in 3 |
| Scaffold W2 init | Random, scale=0.01 | Zero W2 gives zero gradient; semi-orthogonal often impossible |
| SVD frequency | Every 5–10 epochs | Overhead is negligible; no need for every step |
| Width | Start at target width | Overabundance adds only +0.14% on average |

---

## 14. Open Questions

1. **Why does Iso collapse at depth 5-6?** The tangential Jacobian preservation is multiplicative across layers. At 5+ layers, even small per-layer losses may compound. A formal analysis of how the tangential component degrades across depth is needed.

2. **Can Iso be made depth-robust?** RMS+tanh stays positive-slope to depth 6. Is there an "isotropic RMSNorm" that provides both exact scaffold inertness and depth robustness beyond 4 layers?

3. **Does LN+Iso at long training stay best?** At 100 epochs LN+Iso (46.25%) leads. Does this advantage grow or shrink with even longer training? The combination avoids both LN overfitting and Iso depth limits.

4. **Shell collapse theory**: Why does the affine residual *grow* with width under L2-normalised inputs? A theoretical explanation for this unexpected direction would clarify the gap in Appendix C.

5. **Scaling to deeper/wider architectures**: These experiments use widths up to 512 and depths up to 6, with CIFAR-10 as the test bed. The results may differ in larger architectures (ResNets, Transformers). Does Iso's unique topology support survive at scale, or does LN become the dominant design choice entirely?

---

*34 experiments. Full results at `dynamic_topology_net/results/ALL_RESULTS.md`. Code at `dynamic_topology_net/experiments/`. GitHub: github.com/alfiemowle/isotropic-mlp-research*
