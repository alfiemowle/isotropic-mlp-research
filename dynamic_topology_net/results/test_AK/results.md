# Test AK -- Isotropic Activation Variants

## Setup
- Width: 32, Epochs: 30, lr=0.08, seed=42
- Device: cpu

## Question
Does replacing tanh with GELU/SiLU in f(x)=sigma(||x||)*x/||x|| improve accuracy?
Can non-saturating sigma push Iso's depth ceiling beyond 4L?

## Results

| Model | 1L  |  2L  |  3L  |  4L  |  5L | Slope |
|---|---|---|---|---|---|---|
| IsoTanh | 0.4139 | 0.4396 | 0.4469 | 0.4513 | 0.4246 | +0.0033 |
| IsoGELU | 0.2366 | 0.2812 | 0.3334 | 0.2372 | 0.2540 | -0.0009 |
| IsoSiLU | 0.2249 | 0.2790 | 0.2510 | 0.2690 | 0.2256 | -0.0009 |
| IsoSoftplus | 0.2665 | 0.3395 | 0.2895 | 0.2763 | 0.2436 | -0.0109 |

## Interpretation
- IsoTanh: saturates sigma(r)->1 for large r; tangential gradient preserved but radial shrinks
- IsoGELU/SiLU: sigma(r)~r for large r; no saturation ceiling; radial gradient stays ~1
- IsoSoftplus: log(1+exp(r)), smooth and always positive, intermediate saturation

