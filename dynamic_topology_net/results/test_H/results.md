# Test H: Sequential Pruning Stability

Seed: 42, Epochs: 24, LR: 0.08, Batch: 24, Width: 32

Isotropic: prune smallest singular value first (re-diagonalise each step).
Baseline: prune neuron with smallest max incoming weight norm.

## Base Accuracies

- IsotropicMLP (trained, full width): 0.4069
- BaselineMLP (trained, full width):  0.2352

## Pruning Trajectory

| Neurons Remaining | IsotropicMLP Acc | BaselineMLP Acc |
|-------------------|-----------------|----------------|
|                32 | 0.4069 | 0.2352 |
|                31 | 0.4073 | 0.2347 |
|                30 | 0.4066 | 0.2399 |
|                29 | 0.4069 | 0.2315 |
|                28 | 0.4079 | 0.2327 |
|                27 | 0.4084 | 0.2341 |
|                26 | 0.4062 | 0.2326 |
|                25 | 0.4064 | 0.2295 |
|                24 | 0.4049 | 0.2320 |
|                23 | 0.4060 | 0.2218 |
|                22 | 0.4058 | 0.2230 |
|                21 | 0.4055 | 0.2183 |
|                20 | 0.4036 | 0.2193 |
|                19 | 0.4033 | 0.2077 |
|                18 | 0.4028 | 0.2080 |
|                17 | 0.4034 | 0.2095 |
|                16 | 0.4034 | 0.2011 |
|                15 | 0.3966 | 0.1985 |
|                14 | 0.3853 | 0.2025 |
|                13 | 0.3785 | 0.2083 |
|                12 | 0.3424 | 0.1969 |
|                11 | 0.3142 | 0.1938 |
|                10 | 0.2666 | 0.1877 |
|                 9 | 0.2380 | 0.1866 |
|                 8 | 0.2132 | 0.1769 |
|                 7 | 0.1970 | 0.1702 |
|                 6 | 0.1972 | 0.1451 |
|                 5 | 0.1804 | 0.1345 |
|                 4 | 0.1443 | 0.1430 |
|                 3 | 0.1432 | 0.1437 |
|                 2 | 0.1184 | 0.1339 |
|                 1 | 0.1139 | 0.1206 |

## Cliff Analysis (where acc drops below 70% of initial)

- IsotropicMLP cliff: 10 neurons remaining (acc = 0.2666, initial = 0.4069)
- BaselineMLP cliff:  6 neurons remaining (acc = 0.1451, initial = 0.2352)

**Baseline magnitude pruning holds longer** in this run. Note that the comparison is
not perfectly controlled (different initial accuracies, different pruning criteria).

Average accuracy across all pruning steps (AUC proxy):
- Isotropic: 0.3289
- Baseline:  0.1985
- Advantage: +0.1304 (positive = isotropic wins)

See `sequential_pruning.png` for the pruning curve.
