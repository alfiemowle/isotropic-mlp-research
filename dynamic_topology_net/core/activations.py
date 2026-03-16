"""
Activation functions for isotropic and standard networks.

Key concept
-----------
Isotropic activation:  f(x) = tanh(||x||_2) * x / ||x||_2
    - Equivariant under the full orthogonal group O(n): f(Rx) = R f(x)
    - Basis-independent: no preferred neuron decomposition
    - Reduces to the identity direction scaled by tanh of the radius

Standard (anisotropic) tanh:  f(x)_i = tanh(x_i)
    - Only equivariant under the discrete permutation group S_n
    - Individuates neurons by applying a function to each separately

The difference is subtle but fundamental: isotropic activations treat the
input as a vector with a magnitude and direction, not as a list of scalars.
"""

import torch
import torch.nn as nn


class IsotropicTanh(nn.Module):
    """
    Isotropic activation function: f(x) = tanh(||x||) * x / ||x||

    Equivariant under O(n): for any orthogonal matrix R,
        f(R x) = R f(x)

    This holds because:
        ||Rx|| = ||x||  (orthogonal matrices preserve norms)
        f(Rx) = tanh(||Rx||) * Rx/||Rx|| = tanh(||x||) * Rx/||x|| = R f(x)

    At x = 0: the function is 0 by continuous extension
    (tanh(r)/r -> 1 as r -> 0, and tanh(0) = 0, so f(0) = 0).
    """

    def forward(self, x):
        # x: (..., n)
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.tanh(norm) * x / norm


class IsotropicTanhWithLength(nn.Module):
    """
    Isotropic activation with a trainable intrinsic length parameter o = exp(log_o).

    Forward pass:  f(x; o) = tanh(sqrt(||x||^2 + o^2)) * x / sqrt(||x||^2 + o^2)

    The intrinsic length o acts as a 'phantom' orthogonal dimension.
    Geometrically, it offsets the norm computation away from zero, so that
    when a neuron is pruned (its singular value -> 0), the norm of the
    remaining representation is not destabilised.

    This is required for correct neurodegeneration (Sec. 3.1, Eqns. 26-29).
    o is positive by construction: o = exp(log_o) > 0.
    """

    def __init__(self):
        super().__init__()
        self.log_o = nn.Parameter(torch.tensor(0.0))  # o = exp(0) = 1 initially

    def forward(self, x):
        o = self.log_o.exp()
        # Extended norm: sqrt(||x||^2 + o^2)
        norm = (x.pow(2).sum(dim=-1, keepdim=True) + o.pow(2)).sqrt().clamp(min=1e-8)
        return torch.tanh(norm) * x / norm

    @property
    def o(self):
        return self.log_o.exp().item()


class StandardTanh(nn.Module):
    """
    Standard elementwise tanh (anisotropic baseline).
    f(x)_i = tanh(x_i)

    This is the conventional activation function. It is equivariant only
    under the discrete permutation group S_n (neuron permutations), not
    under the full orthogonal group O(n).
    """

    def forward(self, x):
        return torch.tanh(x)


class HypersphericalNorm(nn.Module):
    """
    Projects x onto the unit hypersphere: f(x) = x / ||x||

    When this is composed with isotropic activations inside a network,
    Appendix C of the paper proves the entire network collapses to an
    affine map (Eqn. 48): x^(l) = W x^(0) + b

    This is used in Test F to empirically confirm that claim.
    The intuition: normalising to unit norm after an isotropic activation
    makes all non-linearities trivial, so the network is effectively linear.
    """

    def forward(self, x):
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
