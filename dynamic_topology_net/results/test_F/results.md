# Test F: Hyperspherical Shell Collapse

Seed: 42, Epochs: 24, LR: 0.08, Batch: 128, Width: 32

## Experiment 1: Mathematical walkthrough

We forward a test batch through CollapsingIsotropicMLP and fit the best possible
linear map (least-squares) to its outputs.

- Mean absolute residual (network vs best linear fit): 0.000000
- Max absolute residual: 0.000000
- Mean output magnitude: 0.136027
- Relative residual: 0.000000

**Result: The collapsing network output is almost perfectly explained by a single linear map.**
This numerically confirms Appendix C's claim that unit-norm normalisation after
isotropic activation collapses the network to an affine map.

## Experiment 2: CIFAR-10 Training Comparison

| Model | Final Test Accuracy |
|-------|--------------------|
| CollapsingIsotropicMLP | 0.4090 |
| IsotropicMLP (standard) | 0.4068 |
| nn.Linear (logistic reg) | 0.2959 |

### Epoch-by-epoch test accuracy

| Epoch | CollapsingIso | IsotropicMLP | nn.Linear |
|-------|--------------|--------------|----------|
|     1 | 0.3626 | 0.3635 | 0.3081 |
|     2 | 0.3966 | 0.3931 | 0.2534 |
|     3 | 0.3942 | 0.3932 | 0.2814 |
|     4 | 0.3938 | 0.3956 | 0.2741 |
|     5 | 0.4002 | 0.3990 | 0.2562 |
|     6 | 0.4068 | 0.4051 | 0.3016 |
|     7 | 0.4043 | 0.4060 | 0.2823 |
|     8 | 0.4016 | 0.4042 | 0.2931 |
|     9 | 0.3937 | 0.3932 | 0.2699 |
|    10 | 0.3886 | 0.3921 | 0.2752 |
|    11 | 0.4095 | 0.4069 | 0.2731 |
|    12 | 0.4027 | 0.4000 | 0.2624 |
|    13 | 0.4137 | 0.4148 | 0.2781 |
|    14 | 0.3949 | 0.3973 | 0.2841 |
|    15 | 0.3925 | 0.3907 | 0.2611 |
|    16 | 0.3955 | 0.3928 | 0.3037 |
|    17 | 0.4006 | 0.4011 | 0.2617 |
|    18 | 0.4102 | 0.4082 | 0.2648 |
|    19 | 0.4101 | 0.4097 | 0.2979 |
|    20 | 0.4091 | 0.4090 | 0.2637 |
|    21 | 0.4121 | 0.4062 | 0.2771 |
|    22 | 0.4091 | 0.4059 | 0.3005 |
|    23 | 0.4128 | 0.4096 | 0.2974 |
|    24 | 0.4090 | 0.4068 | 0.2959 |

## Interpretation

- IsotropicMLP vs CollapsingIsotropicMLP gap: -0.0022
- CollapsingIsotropicMLP vs nn.Linear gap: +0.1131

**Shell collapse PARTIAL**: The collapsing network (0.4090) is above the linear
baseline (0.2959) by more than 5%. There may be residual nonlinearity,
or the network learned a partially useful representation despite the collapse.

See `collapse_comparison.png` for training curves.
