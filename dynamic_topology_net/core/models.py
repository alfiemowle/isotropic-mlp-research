"""
Neural network models for dynamic topology experiments.

Models
------
IsotropicMLP       : [input -> IsotropicTanh -> output], supports diagonalise/prune/grow
BaselineMLP        : [input -> StandardTanh  -> output], static reference
DeepIsotropicMLP   : [input -> Iso -> Iso -> output],   two hidden layers
DeepBaselineMLP    : [input -> Tanh -> Tanh -> output], two hidden layers

The IsotropicMLP is the central model. Its key operations are:

  partial_diagonalise()
      Applies SVD reparameterisation (Eqn. 25) to W1:
          W1 = U S Vt  ->  W1' = diag(S) Vt,  W2' = W2 U,  b1' = U^T b1
      The network function is EXACTLY preserved. After this call,
      self.W1.data[i, :] has L2 norm equal to singular value sigma_i,
      and singular values are sorted descending.

  prune_neuron(idx)
      Removes the neuron at index idx (smallest singular value in diagonalised basis).
      Updates W2 by column deletion (approximate for small sigma; exact at sigma=0).
      Optionally updates the intrinsic length o to absorb residual bias.

  grow_neuron()
      Appends a scaffold neuron: zero row in W1, zero column in W2, zero bias.
      Forward pass is EXACTLY preserved (new neuron contributes zero output).
      The non-diagonal isotropic Jacobian allows gradient flow to reach it.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import IsotropicTanh, IsotropicTanhWithLength, StandardTanh, HypersphericalNorm


# ---------------------------------------------------------------------------
# Single-hidden-layer models
# ---------------------------------------------------------------------------

class IsotropicMLP(nn.Module):
    """
    MLP with one hidden layer using isotropic-tanh activation.

    Architecture:  x -> Linear(W1, b1) -> IsotropicTanh -> Linear(W2, b2) -> y

    Supports partial SVD diagonalisation and dynamic grow/prune operations.
    The use_intrinsic_length flag adds a trainable scalar o = exp(log_o) that
    modifies the norm in the activation: norm = sqrt(||h||^2 + o^2).
    This is required for theoretically exact neurodegeneration.
    """

    def __init__(self, input_dim=3072, width=24, num_classes=10,
                 use_intrinsic_length=False):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_intrinsic_length = use_intrinsic_length

        self.W1 = nn.Parameter(torch.empty(width, input_dim))
        self.b1 = nn.Parameter(torch.zeros(width))
        self.W2 = nn.Parameter(torch.empty(num_classes, width))
        self.b2 = nn.Parameter(torch.zeros(num_classes))

        if use_intrinsic_length:
            self.log_o = nn.Parameter(torch.tensor(0.0))

        # Kaiming uniform init (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        fan_in = input_dim
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b1, -bound, bound)
        fan_in2 = width
        bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
        nn.init.uniform_(self.b2, -bound2, bound2)

    @property
    def width(self):
        return self.W1.shape[0]

    def forward(self, x):
        h = F.linear(x, self.W1, self.b1)   # (batch, width)

        if self.use_intrinsic_length:
            o = self.log_o.exp()
            norm = (h.pow(2).sum(dim=-1, keepdim=True) + o.pow(2)).sqrt().clamp(min=1e-8)
        else:
            norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        a = torch.tanh(norm) * h / norm      # isotropic tanh
        return F.linear(a, self.W2, self.b2)

    # ------------------------------------------------------------------
    # Structural operations
    # ------------------------------------------------------------------

    def get_singular_values(self):
        """
        Returns singular values of W1 in descending order.
        Call partial_diagonalise() first for the values to equal row norms.
        """
        with torch.no_grad():
            return torch.linalg.svdvals(self.W1)

    def partial_diagonalise(self):
        """
        Apply right-sided SVD reparameterisation (Eqns. 20-25).

        W1 = U S Vt  (thin SVD, U is width x width orthogonal)
        After:
            W1' = diag(S) @ Vt        shape (width, input_dim)
            W2' = W2 @ U              shape (num_classes, width)
            b1' = U^T @ b1            shape (width,)

        This is an EXACT function-preserving reparameterisation.
        Proof: h' = W1' x + b1' = U^T(W1 x + b1) = U^T h
               iso_tanh(h') = U^T iso_tanh(h)   [because ||U^T h|| = ||h||]
               W2' iso_tanh(h') = W2 U U^T iso_tanh(h) = W2 iso_tanh(h)

        Returns the singular values S (descending order).
        """
        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(self.W1.data, full_matrices=False)
            self.W1.data = S.unsqueeze(1) * Vt   # diag(S) @ Vt
            self.W2.data = self.W2.data @ U       # W2 @ U
            self.b1.data = U.T @ self.b1.data     # U^T b1
        return S

    def prune_neuron(self, idx):
        """
        Remove the neuron at position idx (in the diagonalised basis).

        Steps (from Sec. 3.1):
          1. Remove row idx from W1
          2. Remove entry idx from b1
          3. Remove column idx from W2
          4. If use_intrinsic_length: update o to absorb residual bias b_idx
             (o'^2 = o^2 + b_idx^2, from Eqn. 29 step 3)

        NOTE: Call partial_diagonalise() first so that idx corresponds to
        a specific singular value. Smallest singular value is pruned last,
        so idx should be width-1 after sorting.

        Returns the singular value of the pruned neuron.
        """
        with torch.no_grad():
            # Record what's being removed
            pruned_sv = self.W1.data[idx].norm().item()
            pruned_b  = self.b1.data[idx].item()

            keep = [i for i in range(self.width) if i != idx]

            new_W1 = self.W1.data[keep].clone()
            new_b1 = self.b1.data[keep].clone()
            new_W2 = self.W2.data[:, keep].clone()

            # Update intrinsic length to absorb residual bias
            if self.use_intrinsic_length:
                o_sq     = self.log_o.exp().item() ** 2
                new_o_sq = max(o_sq + pruned_b ** 2, 1e-10)
                self.log_o.data = torch.tensor(0.5 * math.log(new_o_sq),
                                               dtype=torch.float32)

            # Replace parameters (new tensors, smaller shape)
            self.W1 = nn.Parameter(new_W1)
            self.b1 = nn.Parameter(new_b1)
            self.W2 = nn.Parameter(new_W2)

        return pruned_sv

    def grow_neuron(self, b_star=0.0, w2_init='zero'):
        """
        Add a scaffold neuron with zero singular value.

        The new neuron is initialised with:
          - W1 row: zeros  ->  zero singular value, does NOT affect forward pass
          - b1 entry: b_star (default 0)
          - W2 column: zeros (w2_init='zero') or small random (w2_init='random')

        With w2_init='zero': forward pass is EXACTLY preserved (Test B).
        With w2_init='random': forward pass changes slightly but gradient
          flows immediately (better for practical training).

        If use_intrinsic_length: update o so that o'^2 = o^2 - b_star^2
        (reverse of the pruning update, Eqn. 29 step 3 inverted).

        Returns the new width.
        """
        with torch.no_grad():
            dev = self.W1.device
            m, n = self.W1.shape

            new_row = torch.zeros(1, n, device=dev, dtype=self.W1.dtype)
            new_W1  = nn.Parameter(torch.cat([self.W1.data, new_row], dim=0))

            new_entry = torch.tensor([b_star], device=dev, dtype=self.b1.dtype)
            new_b1    = nn.Parameter(torch.cat([self.b1.data, new_entry]))

            if w2_init == 'zero':
                new_col = torch.zeros(self.num_classes, 1, device=dev, dtype=self.W2.dtype)
            else:
                new_col = torch.randn(self.num_classes, 1, device=dev, dtype=self.W2.dtype) * 0.01
            new_W2 = nn.Parameter(torch.cat([self.W2.data, new_col], dim=1))

            if self.use_intrinsic_length:
                o_sq     = self.log_o.exp().item() ** 2
                new_o_sq = max(o_sq - b_star ** 2, 1e-10)
                self.log_o.data = torch.tensor(0.5 * math.log(new_o_sq),
                                               dtype=torch.float32)

            self.W1 = new_W1
            self.b1 = new_b1
            self.W2 = new_W2

        return self.width

    def clone_weights_to(self, other):
        """Copy weights from this model into another IsotropicMLP (same shape)."""
        with torch.no_grad():
            other.W1.data.copy_(self.W1.data)
            other.b1.data.copy_(self.b1.data)
            other.W2.data.copy_(self.W2.data)
            other.b2.data.copy_(self.b2.data)
            if self.use_intrinsic_length and other.use_intrinsic_length:
                other.log_o.data.copy_(self.log_o.data)


class BaselineMLP(nn.Module):
    """
    Standard MLP with elementwise tanh activation (anisotropic baseline).
    Architecture:  x -> Linear -> StandardTanh -> Linear -> y
    """

    def __init__(self, input_dim=3072, width=24, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            StandardTanh(),
            nn.Linear(width, num_classes),
        )

    @property
    def width(self):
        return self.net[0].out_features

    def forward(self, x):
        return self.net(x)


class CollapsingIsotropicMLP(nn.Module):
    """
    Isotropic MLP with hyperspherical normalisation after each activation.

    Appendix C proves this collapses to an affine map: x^(l) = W x^(0) + b
    Used in Test F to empirically confirm the paper's claim.
    """

    def __init__(self, input_dim=3072, width=24, num_classes=10):
        super().__init__()
        self.W1 = nn.Linear(input_dim, width)
        self.W2 = nn.Linear(width, num_classes)
        self.iso  = IsotropicTanh()
        self.norm = HypersphericalNorm()

    def forward(self, x):
        h = self.W1(x)
        a = self.norm(self.iso(h))   # project to unit sphere after isotropic activation
        return self.W2(a)


# ---------------------------------------------------------------------------
# Two-hidden-layer models (Test M, Test E)
# ---------------------------------------------------------------------------

class DeepIsotropicMLP(nn.Module):
    """
    Two-hidden-layer isotropic MLP.
    Architecture:  x -> Linear -> Iso -> Linear -> Iso -> Linear -> y

    With three affine layers surrounding two isotropic non-linearities,
    full diagonalisation (Sec. 2.3) is possible on the middle affine layer.
    """

    def __init__(self, input_dim=3072, width=24, num_classes=10):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(width, input_dim))
        self.b1 = nn.Parameter(torch.zeros(width))
        self.W2 = nn.Parameter(torch.empty(width, width))
        self.b2 = nn.Parameter(torch.zeros(width))
        self.W3 = nn.Parameter(torch.empty(num_classes, width))
        self.b3 = nn.Parameter(torch.zeros(num_classes))

        for W in [self.W1, self.W2, self.W3]:
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))

        self.input_dim   = input_dim
        self.num_classes = num_classes

    @property
    def width(self):
        return self.W1.shape[0]

    def forward(self, x):
        h1   = F.linear(x, self.W1, self.b1)
        n1   = h1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a1   = torch.tanh(n1) * h1 / n1

        h2   = F.linear(a1, self.W2, self.b2)
        n2   = h2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a2   = torch.tanh(n2) * h2 / n2

        return F.linear(a2, self.W3, self.b3)


class DeepBaselineMLP(nn.Module):
    """
    Two-hidden-layer standard tanh MLP (anisotropic baseline for Test M/E).
    Architecture:  x -> Linear -> Tanh -> Linear -> Tanh -> Linear -> y
    """

    def __init__(self, input_dim=3072, width=24, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            StandardTanh(),
            nn.Linear(width, width),
            StandardTanh(),
            nn.Linear(width, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Test O: Two-layer collapsing model
# ---------------------------------------------------------------------------

class DeepCollapsingIsotropicMLP(nn.Module):
    """
    Two-hidden-layer isotropic MLP with unit-norm normalisation after each hidden layer.

    Appendix C applied recursively guarantees this is an affine function of the input.
    Used in Test O: if Iso-2L beats CollapsingIso-2L, nonlinearity contributes at depth
    even though it made no difference at 1 layer (Test F).
    """

    def __init__(self, input_dim=3072, width=24, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            IsotropicTanh(),
            HypersphericalNorm(),
            nn.Linear(width, width),
            IsotropicTanh(),
            HypersphericalNorm(),
            nn.Linear(width, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Test Q: Three-hidden-layer models
# ---------------------------------------------------------------------------

class IsotropicMLP3L(nn.Module):
    """
    Three-hidden-layer isotropic MLP.
    Architecture:  x -> Iso -> Iso -> Iso -> y
    Used in Test Q to extend the depth curve beyond 2 layers.
    """

    def __init__(self, input_dim=3072, width=24, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            IsotropicTanh(),
            nn.Linear(width, width),
            IsotropicTanh(),
            nn.Linear(width, width),
            IsotropicTanh(),
            nn.Linear(width, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class BaselineMLP3L(nn.Module):
    """
    Three-hidden-layer standard tanh MLP.
    Architecture:  x -> Tanh -> Tanh -> Tanh -> y
    Used in Test Q.
    """

    def __init__(self, input_dim=3072, width=24, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            StandardTanh(),
            nn.Linear(width, width),
            StandardTanh(),
            nn.Linear(width, width),
            StandardTanh(),
            nn.Linear(width, num_classes),
        )

    def forward(self, x):
        return self.net(x)
