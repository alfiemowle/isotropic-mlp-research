# Test AG-B -- LN Variant Depth Scaling (companion to AG)

## Setup
Identical to AG: Width=128, Epochs=30, lr=0.08, seed=42, depths=[1, 2, 3, 4, 5, 6]

## Results: LN variants

| Depth | LN+tanh | Delta | RMS+tanh | Delta | LN+Iso | Delta |
|---|---|---|---|---|---|---|
| 1 | 0.4514 | +0.0000 | 0.4163 | +0.0000 | 0.4306 | +0.0000 |
| 2 | 0.4734 | +0.0220 | 0.4245 | +0.0082 | 0.4723 | +0.0417 |
| 3 | 0.4777 | +0.0263 | 0.4641 | +0.0478 | 0.4885 | +0.0579 |
| 4 | 0.4917 | +0.0403 | 0.4753 | +0.0590 | 0.4550 | +0.0244 |
| 5 | 0.4901 | +0.0387 | 0.4753 | +0.0590 | 0.4059 | -0.0247 |
| 6 | 0.4657 | +0.0143 | 0.4376 | +0.0213 | 0.3055 | -0.1251 |

## Per-layer slope (linear fit)
- LN+tanh: +0.0039/layer
- RMS+tanh: +0.0077/layer
- LN+Iso: -0.0245/layer

(For reference from width=32, Test AE: Base=-0.020/layer, Iso=+0.014/layer)
