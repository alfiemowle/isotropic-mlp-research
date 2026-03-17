# Test AF-B -- LN Variant Width Scaling (companion to AF)

## Setup
Identical to AF: Epochs=24, lr=0.08, depth=2, seed=42, widths=[32, 64, 128, 256, 512]

## Results: LN variants

| Width | LN+tanh | RMS+tanh | LN+Iso |
|---|---|---|---|
| 32 | 0.4535 | 0.4323 | 0.4589 |
| 64 | 0.4641 | 0.4433 | 0.4786 |
| 128 | 0.4706 | 0.4367 | 0.4730 |
| 256 | 0.4746 | 0.4267 | 0.4758 |
| 512 | 0.4805 | 0.4102 | 0.4820 |

## Combined with AF (for full comparison, see plots below)
