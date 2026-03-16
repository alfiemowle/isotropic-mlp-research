# Isotropic MLP Research

An empirical investigation of **isotropic dynamic networks** based on:

> *"On De-Individuated Neurons: Continuous Symmetries Enable Dynamic Topologies"*
> George Bird, University of Manchester — [arXiv:2602.23405](https://arxiv.org/abs/2602.23405) (February 2026)

30 experiments (A–Z, AA–AD) testing, validating, and extending the paper's claims on CIFAR-10.

---

## Key Findings

**What the paper gets right**
- Isotropic tanh consistently outperforms standard tanh: **+14–16% accuracy** on CIFAR-10, scaling across widths, depths, and datasets
- Reparameterisation and neurogenesis invariance are exact (to float32)
- Singular value criterion predicts pruning impact well (r = 0.815)
- 50% pruning is stable with near-zero accuracy loss; 91% recovery after just 1 fine-tune epoch

**What we found beyond the paper**
- **Composite pruning criterion** (Σᵢᵢ × ‖W₂ col‖) beats SV alone: r = 0.859 vs 0.815
- **Dynamic topology = flexibility, not accuracy**: fair epoch-matched comparison shows Dynamic ≈ Static
- **Shell collapse (Appendix C) is a genuine proof gap**: affine residual flat at ~0.22 across widths 32–256 — not a finite-size effect
- **Base depth failure mechanism resolved**: elementwise tanh saturates 99.94% of neurons, collapsing the Jacobian and forcing all learning into the output layer (gradient 7–8× larger than Iso). Iso's tangential Jacobian component preserves gradient flow through all layers regardless of norm.
- Gradient-based pruning criterion (paper's footnote 5) fails badly: r = 0.464
- Semi-orthogonal scaffold initialisation is impossible when width > num_classes

---

## Structure

```
├── RESEARCH_NOTES.md                  # Full project briefing and empirical knowledge base
├── dynamic_topology_net/
│   ├── core/
│   │   ├── models.py                  # IsotropicMLP, BaselineMLP
│   │   ├── activations.py             # IsotropicTanh
│   │   └── train_utils.py             # train_model, evaluate, load_cifar10
│   ├── experiments/
│   │   ├── test_A_reparameterisation.py
│   │   ├── ...                        # 30 experiments total (A–Z, AA–AD)
│   │   └── test_AD_saturation_gradient_anatomy.py
│   ├── results/
│   │   ├── ALL_RESULTS.md             # Master summary of all 30 experiments
│   │   └── test_*/                    # Per-experiment results, plots, and notes
│   └── train.py                       # Original baseline training script
```

---

## Setup

```bash
# Python 3.11 + PyTorch with CUDA recommended
pip install torch torchvision numpy matplotlib

# CIFAR-10 downloads automatically on first run via torchvision
```

---

## Running Experiments

Each experiment is a self-contained script:

```bash
python dynamic_topology_net/experiments/test_A_reparameterisation.py
python dynamic_topology_net/experiments/test_AD_saturation_gradient_anatomy.py
# etc.
```

Results and plots are saved to `dynamic_topology_net/results/test_*/`.

See `dynamic_topology_net/results/ALL_RESULTS.md` for the full summary of all findings.

---

## Experiment Index

| Test | Topic | Key result |
|------|-------|-----------|
| A | Reparameterisation invariance | Exact (float32) |
| B | Neurogenesis invariance | Exact |
| C | Intrinsic length | Negligible in practice |
| D | Iso vs Base accuracy | +14.6% for Iso |
| E | Depth scaling | Iso +3.4%/layer; Base −6.1%/layer |
| F | Affine residual (small N) | Underdetermined — see Tests O, T, AB |
| G | SV pruning criterion | r = 0.815 |
| H | Sequential pruning | 50% prunable, <0.1% loss |
| I | Width transitions | Smooth for moderate changes |
| J | Dynamic vs static (fair) | Equal accuracy — topology adds flexibility |
| K | Scaffold gradient with W₂=0 | Gradient = 0; W₂ must be nonzero |
| L | Cross-dataset generalisation | Confirmed on MNIST, F-MNIST, CIFAR-10 |
| M | Multi-layer depth scaling | Consistent with Test E |
| N | SVD overhead | ~0.1% per epoch |
| O | Shell collapse (1L) | Residual 0.32–0.81, not zero |
| P | Semi-orth scaffold init | Impossible when width > num_classes |
| Q | Depth + dataset sweep | Iso advantage consistent |
| R | Effective rank at depth | Inconclusive (hook issue) |
| S | Pruning timing (spectrum) | Spectrum matures over training |
| T | Shell collapse (L2 inputs) | L2 inputs increase residual — opposite of theory |
| U2 | Composite pruning criterion | r = 0.859 (best found) |
| V | Pruning timing (accuracy) | Late pruning best |
| W | Overabundance protocol | +0.14% mean — marginal |
| X | Rank regularisation | Zero effect on depth collapse |
| Y | Fine-tune budget after pruning | 91% recovery in 1 epoch |
| Z | Gradient pruning criterion | r = 0.464 — fails |
| AA | Chi-normalisation | Accuracy-neutral; IL instability with Chi-norm |
| AB | Shell collapse width scaling | Flat at ~0.22 across widths 32–256 — genuine proof gap |
| AC | Representation spectrum | Base depth failure ≠ representational collapse |
| AD | Saturation & gradient anatomy | Mechanism resolved: elementwise Jacobian collapse |
