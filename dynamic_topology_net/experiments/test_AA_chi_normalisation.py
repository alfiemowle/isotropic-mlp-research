"""
Test AA -- Chi-Normalisation
============================
The paper (Section 3.1, Appendix C) repeatedly recommends Chi-normalisation
to control representation magnitudes in isotropic networks. It is the paper's
recommended normalisation, distinct from:
  - No normalisation (what all previous tests use)
  - Hyperspherical normalisation (x/||x|| -- collapses to affine, Tests F/O/T)

Chi-normalisation: divide by running mean of ||h||, not the exact norm.
This preserves relative magnitude differences between samples (nonlinearity
is preserved) while controlling the scale of representations globally.

It is isotropic by construction (scalar division, equivariant under O(n)).

This test asks three questions:
  Q1: Does Chi-norm improve accuracy at 1L and 2L?
  Q2: Does the Intrinsic Length parameter become more effective when norms
      are controlled by Chi-norm? (Test C found IL negligible on plain Iso)
  Q3: What does Chi-norm do to representation magnitudes during training?
      (Do norms blow up without it? Does Chi-norm stabilise them?)

Models compared:
  - Iso:          IsotropicMLP, no normalisation [current baseline]
  - IsoChiNorm:   IsotropicMLP + Chi-norm after activation
  - IsoChiNorm+IL: IsoChiNorm + trainable intrinsic length
  - CollapsingIso: IsotropicMLP + HypersphericalNorm [known to be ~affine]

Depths: 1L, 2L. Width: 24. Seeds: [42, 123]. Device: CPU/GPU auto.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from dynamic_topology_net.core import load_cifar10
from dynamic_topology_net.core.train_utils import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AA')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS  = 24
LR      = 0.08
BATCH   = 128
WIDTH   = 24
SEEDS   = [42, 123]
CHI_MOM = 0.01   # Chi-norm running mean momentum


# =============================================================================
# Chi Normalizer
# =============================================================================

class ChiNormalizer(nn.Module):
    """
    Running-mean norm normaliser. Isotropic: divides h by scalar E[||h||].

    Tracks a running estimate of E[||h||] over training batches (like
    BatchNorm but for the vector norm, not per-feature statistics).
    At eval time, uses the frozen running estimate.

    init_val: initial estimate. sqrt(width) is a reasonable start
    (expected ||h|| for a random width-d vector with unit-variance components).
    """
    def __init__(self, init_val=1.0, momentum=0.01):
        super().__init__()
        self.register_buffer('running_norm', torch.tensor(float(init_val)))
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                batch_mean = x.detach().norm(dim=-1).mean()
                self.running_norm.mul_(1 - self.momentum).add_(
                    self.momentum * batch_mean)
        return x / self.running_norm.clamp(min=1e-8)


# =============================================================================
# Models
# =============================================================================

class IsoMLP1L(nn.Module):
    """Standard 1-layer isotropic MLP — no normalisation."""
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(width, input_dim))
        self.b1 = nn.Parameter(torch.zeros(width))
        self.W2 = nn.Parameter(torch.empty(num_classes, width))
        self.b2 = nn.Parameter(torch.zeros(num_classes))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        nn.init.uniform_(self.b1, -1/math.sqrt(input_dim), 1/math.sqrt(input_dim))
        nn.init.uniform_(self.b2, -1/math.sqrt(width), 1/math.sqrt(width))
        self._norm_history = []  # track ||a|| per epoch

    def forward(self, x):
        h = F.linear(x, self.W1, self.b1)
        norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a = torch.tanh(norm) * h / norm
        return F.linear(a, self.W2, self.b2)

    def get_repr_norm(self, x):
        with torch.no_grad():
            h = F.linear(x, self.W1, self.b1)
            norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            a = torch.tanh(norm) * h / norm
            return a.norm(dim=-1).mean().item()


class ChiNormMLP1L(nn.Module):
    """1-layer isotropic MLP with Chi-norm after activation."""
    def __init__(self, input_dim, width, num_classes, use_il=False):
        super().__init__()
        self.use_il = use_il
        self.W1 = nn.Parameter(torch.empty(width, input_dim))
        self.b1 = nn.Parameter(torch.zeros(width))
        self.W2 = nn.Parameter(torch.empty(num_classes, width))
        self.b2 = nn.Parameter(torch.zeros(num_classes))
        if use_il:
            self.log_o = nn.Parameter(torch.tensor(0.0))
        self.chi = ChiNormalizer(init_val=math.sqrt(width), momentum=CHI_MOM)
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        nn.init.uniform_(self.b1, -1/math.sqrt(input_dim), 1/math.sqrt(input_dim))
        nn.init.uniform_(self.b2, -1/math.sqrt(width), 1/math.sqrt(width))

    def forward(self, x):
        h = F.linear(x, self.W1, self.b1)
        if self.use_il:
            o    = self.log_o.exp()
            norm = (h.pow(2).sum(-1, keepdim=True) + o.pow(2)).sqrt().clamp(min=1e-8)
        else:
            norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a = torch.tanh(norm) * h / norm
        a = self.chi(a)
        return F.linear(a, self.W2, self.b2)

    def get_il_value(self):
        if self.use_il:
            return self.log_o.exp().item()
        return None


class CollapsingMLP1L(nn.Module):
    """1-layer isotropic MLP + hyperspherical normalisation (known ~affine)."""
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        self.W1 = nn.Linear(input_dim, width)
        self.W2 = nn.Linear(width, num_classes)

    def forward(self, x):
        h    = self.W1(x)
        norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a    = torch.tanh(norm) * h / norm     # isotropic activation
        a    = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # unit sphere
        return self.W2(a)


class IsoMLP2L(nn.Module):
    """Standard 2-layer isotropic MLP — no normalisation."""
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(width, input_dim))
        self.b1 = nn.Parameter(torch.zeros(width))
        self.W2 = nn.Parameter(torch.empty(width, width))
        self.b2 = nn.Parameter(torch.zeros(width))
        self.W3 = nn.Parameter(torch.empty(num_classes, width))
        self.b3 = nn.Parameter(torch.zeros(num_classes))
        for W in [self.W1, self.W2, self.W3]:
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))

    def forward(self, x):
        h1 = F.linear(x, self.W1, self.b1)
        n1 = h1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a1 = torch.tanh(n1) * h1 / n1
        h2 = F.linear(a1, self.W2, self.b2)
        n2 = h2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a2 = torch.tanh(n2) * h2 / n2
        return F.linear(a2, self.W3, self.b3)


class ChiNormMLP2L(nn.Module):
    """2-layer isotropic MLP with Chi-norm after each activation."""
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        self.W1   = nn.Parameter(torch.empty(width, input_dim))
        self.b1   = nn.Parameter(torch.zeros(width))
        self.W2   = nn.Parameter(torch.empty(width, width))
        self.b2   = nn.Parameter(torch.zeros(width))
        self.W3   = nn.Parameter(torch.empty(num_classes, width))
        self.b3   = nn.Parameter(torch.zeros(num_classes))
        self.chi1 = ChiNormalizer(init_val=math.sqrt(width), momentum=CHI_MOM)
        self.chi2 = ChiNormalizer(init_val=math.sqrt(width), momentum=CHI_MOM)
        for W in [self.W1, self.W2, self.W3]:
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))

    def forward(self, x):
        h1 = F.linear(x, self.W1, self.b1)
        n1 = h1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a1 = torch.tanh(n1) * h1 / n1
        a1 = self.chi1(a1)
        h2 = F.linear(a1, self.W2, self.b2)
        n2 = h2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        a2 = torch.tanh(n2) * h2 / n2
        a2 = self.chi2(a2)
        return F.linear(a2, self.W3, self.b3)


# =============================================================================
# Training
# =============================================================================

def train_and_track(model, train_loader, test_loader, epochs, lr, device, tag):
    """Train model, return per-epoch accuracy and representation norm."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    acc_history  = []
    norm_history = []   # running_norm if Chi model, else mean ||a||

    # Probe batch for norm tracking
    probe_x, _ = next(iter(test_loader))
    probe_x = probe_x[:256].to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        acc = evaluate(model, test_loader, device)
        acc_history.append(acc)

        # Track representation norm
        model.eval()
        with torch.no_grad():
            if isinstance(model, (ChiNormMLP1L, ChiNormMLP2L)):
                norm_val = model.chi.running_norm.item() if hasattr(model, 'chi') \
                           else model.chi1.running_norm.item()
            else:
                # Handle both nn.Parameter W1/b1 and nn.Linear W1
                if isinstance(model.W1, nn.Linear):
                    h = model.W1(probe_x)
                else:
                    h = F.linear(probe_x, model.W1, model.b1)
                n = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                a = torch.tanh(n) * h / n
                norm_val = a.norm(dim=-1).mean().item()
        norm_history.append(norm_val)

        if epoch in (1, 6, 12, 18, 24):
            il_str = ''
            if isinstance(model, ChiNormMLP1L) and model.use_il:
                il_str = f'  o={model.get_il_value():.4f}'
            print(f'  [{tag}] Epoch {epoch:2d}/{epochs}  '
                  f'acc={acc:.4f}  repr_norm={norm_val:.4f}{il_str}')

    return acc_history, norm_history


# =============================================================================
# Main
# =============================================================================

def main():
    print(f'Device: {DEVICE}, Width={WIDTH}, Epochs={EPOCHS}')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    configs_1L = [
        ('Iso-1L',          lambda: IsoMLP1L(input_dim, WIDTH, num_classes)),
        ('ChiNorm-1L',      lambda: ChiNormMLP1L(input_dim, WIDTH, num_classes, use_il=False)),
        ('ChiNorm+IL-1L',   lambda: ChiNormMLP1L(input_dim, WIDTH, num_classes, use_il=True)),
        ('Collapsing-1L',   lambda: CollapsingMLP1L(input_dim, WIDTH, num_classes)),
    ]
    configs_2L = [
        ('Iso-2L',          lambda: IsoMLP2L(input_dim, WIDTH, num_classes)),
        ('ChiNorm-2L',      lambda: ChiNormMLP2L(input_dim, WIDTH, num_classes)),
    ]

    results = {}
    total = (len(configs_1L) + len(configs_2L)) * len(SEEDS)
    run = 0

    for tag, factory in configs_1L + configs_2L:
        results[tag] = {'acc': [], 'norm': [], 'final_acc': []}
        for seed in SEEDS:
            run += 1
            print(f'\n[{run}/{total}] {tag}  seed={seed}')
            torch.manual_seed(seed)
            model = factory().to(DEVICE)
            acc_h, norm_h = train_and_track(
                model, train_loader, test_loader, EPOCHS, LR, DEVICE, tag)
            results[tag]['acc'].append(acc_h)
            results[tag]['norm'].append(norm_h)
            results[tag]['final_acc'].append(acc_h[-1])

    # =========================================================================
    # Summary
    # =========================================================================
    print(f'\n{"="*65}')
    print('SUMMARY: Final accuracy (mean over seeds)')
    print(f'{"="*65}')
    for tag in results:
        accs = results[tag]['final_acc']
        print(f'  {tag:20s}: {np.mean(accs):.4f} +/- {np.std(accs):.4f}')

    print(f'\n{"="*65}')
    print('REPR NORM at end of training (mean over seeds)')
    print(f'{"="*65}')
    for tag in results:
        norms = [h[-1] for h in results[tag]['norm']]
        print(f'  {tag:20s}: {np.mean(norms):.4f}')

    # Check IL effectiveness for Chi+IL vs Chi
    chi_acc    = np.mean(results['ChiNorm-1L']['final_acc'])
    chi_il_acc = np.mean(results['ChiNorm+IL-1L']['final_acc'])
    iso_acc    = np.mean(results['Iso-1L']['final_acc'])
    print(f'\n{"="*65}')
    print('INTRINSIC LENGTH with Chi-norm:')
    print(f'  ChiNorm-1L:    {chi_acc:.4f}')
    print(f'  ChiNorm+IL-1L: {chi_il_acc:.4f}  (delta={chi_il_acc-chi_acc:+.4f})')
    print(f'  Iso-1L:        {iso_acc:.4f}  (Chi-norm delta={chi_acc-iso_acc:+.4f})')

    # =========================================================================
    # Plot 1: Accuracy curves
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    depth_groups = [
        ('1L models', ['Iso-1L', 'ChiNorm-1L', 'ChiNorm+IL-1L', 'Collapsing-1L']),
        ('2L models', ['Iso-2L', 'ChiNorm-2L']),
    ]
    colors = ['steelblue', 'darkorange', 'green', 'crimson']
    epochs_range = range(1, EPOCHS + 1)

    for ax, (title, tags) in zip(axes, depth_groups):
        for tag, color in zip(tags, colors):
            if tag not in results:
                continue
            mean_acc = np.mean(results[tag]['acc'], axis=0)
            std_acc  = np.std(results[tag]['acc'], axis=0)
            ax.plot(epochs_range, mean_acc, label=tag, color=color)
            ax.fill_between(epochs_range,
                            mean_acc - std_acc, mean_acc + std_acc,
                            alpha=0.15, color=color)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test accuracy')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Test AA: Chi-Normalisation — Accuracy Curves', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_curves.png'), dpi=150)

    # =========================================================================
    # Plot 2: Representation norm over training
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    for (tag, color) in zip(['Iso-1L', 'ChiNorm-1L', 'ChiNorm+IL-1L'], colors):
        if tag not in results:
            continue
        mean_norm = np.mean(results[tag]['norm'], axis=0)
        ax.plot(epochs_range, mean_norm, label=tag, color=color)
    ax.set_title('Representation Norm During Training\n'
                 '(Iso: mean ||a||; ChiNorm: running_norm estimate)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('||representation||')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'repr_norms.png'), dpi=150)

    # =========================================================================
    # Save results.md
    # =========================================================================
    rows_acc = '\n'.join(
        f'| {tag} | {np.mean(results[tag]["final_acc"]):.4f} | '
        f'{np.std(results[tag]["final_acc"]):.4f} | '
        f'{np.mean([h[-1] for h in results[tag]["norm"]]):.4f} |'
        for tag in results
    )

    chi_delta   = chi_acc    - iso_acc
    chi_il_delta = chi_il_acc - chi_acc
    chi_vs_iso_2l = np.mean(results['ChiNorm-2L']['final_acc']) - \
                    np.mean(results['Iso-2L']['final_acc'])

    verdict_chi = (
        f'Chi-norm {"improves" if chi_delta > 0.005 else "marginally changes" if chi_delta > 0 else "does not improve"} '
        f'accuracy vs plain Iso ({chi_acc:.4f} vs {iso_acc:.4f}, delta={chi_delta:+.4f}).'
    )
    verdict_il = (
        f'Intrinsic length {"becomes effective" if abs(chi_il_delta) > 0.005 else "remains negligible"} '
        f'with Chi-norm (delta={chi_il_delta:+.4f}).'
    )

    md = f"""# Test AA -- Chi-Normalisation

## Setup
- Model: IsotropicMLP [3072->{WIDTH}->10], trained {EPOCHS} epochs
- Depths: 1L and 2L; Seeds: {SEEDS}; lr={LR}, batch={BATCH}
- Chi-norm momentum: {CHI_MOM}
- Device: {DEVICE}

## Question
Does Chi-normalisation (running mean of ||h||) improve accuracy?
Does Intrinsic Length become more effective with controlled norms?

## Results

| Model | Mean Acc | Std | Final repr norm |
|---|---|---|---|
{rows_acc}

## Intrinsic Length Analysis
- Iso-1L (no norm):      {iso_acc:.4f}
- ChiNorm-1L:            {chi_acc:.4f}  (delta vs Iso: {chi_delta:+.4f})
- ChiNorm+IL-1L:         {chi_il_acc:.4f}  (delta vs ChiNorm: {chi_il_delta:+.4f})
- ChiNorm-2L vs Iso-2L:  {chi_vs_iso_2l:+.4f}

## Verdict
{verdict_chi}
{verdict_il}

![Accuracy curves](accuracy_curves.png)
![Representation norms](repr_norms.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    print('\nResults saved to results/test_AA/results.md')
    print('Plots saved to results/test_AA/')


if __name__ == '__main__':
    main()
