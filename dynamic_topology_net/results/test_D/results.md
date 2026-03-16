# Test D: Expressivity -- Isotropic vs Standard Tanh MLP
Seeds: [42, 123, 7], Epochs (synthetic): 100, Epochs (CIFAR-10): 24, LR: 0.08, Batch: 64, Width: 64

## Results Table

| Task | Isotropic (mean+/-std) | Baseline (mean+/-std) | Gap (Iso-Base) | Significance |
|------|------------------------|----------------------|-----------------|---------------|
| XOR (100-dim, 2 classes) | 0.6367 +/- 0.0145 | 0.4997 +/- 0.0111 | +0.1370 | |significant| |
| Selective Gate (200-dim, 4 classes) | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | +0.0000 | not significant |
| CIFAR-10 (3072-dim, 10 classes) | 0.4051 +/- 0.0045 | 0.2404 +/- 0.0061 | +0.1647 | |significant| |

## Per-seed Raw Accuracies

### XOR (100-dim, 2 classes)
- Seed 42: Iso=0.6290, Base=0.4840
- Seed 123: Iso=0.6240, Base=0.5080
- Seed 7: Iso=0.6570, Base=0.5070

### Selective Gate (200-dim, 4 classes)
- Seed 42: Iso=1.0000, Base=1.0000
- Seed 123: Iso=1.0000, Base=1.0000
- Seed 7: Iso=1.0000, Base=1.0000

### CIFAR-10 (3072-dim, 10 classes)
- Seed 42: Iso=0.4115, Base=0.2490
- Seed 123: Iso=0.4027, Base=0.2351
- Seed 7: Iso=0.4012, Base=0.2372

## Interpretation

- **XOR (100-dim, 2 classes)**: Gap=+0.1370 (|significant|). Isotropic MLP outperforms baseline -- suggests the isotropic activation benefits this task.
- **Selective Gate (200-dim, 4 classes)**: Gap=+0.0000 (not significant). Results are roughly equivalent -- no meaningful gap between the two architectures.
- **CIFAR-10 (3072-dim, 10 classes)**: Gap=+0.1647 (|significant|). Isotropic MLP outperforms baseline -- suggests the isotropic activation benefits this task.

## Notes

- The XOR task requires gating on specific input coordinates; standard tanh can individuate neurons per-dimension.
- Isotropic tanh treats the full pre-activation vector as a single entity; it is basis-independent.
- If isotropic matches or beats baseline on XOR/gate tasks, it demonstrates sufficient expressivity despite the O(n)-equivariance constraint.
