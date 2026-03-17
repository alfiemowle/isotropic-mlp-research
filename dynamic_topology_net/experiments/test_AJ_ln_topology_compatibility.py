"""
Test AJ -- Does LN+tanh Support Dynamic Topology?
==================================================
AE showed LN+tanh beats Iso at depth. But the paper's core claim is not just
accuracy -- it is that isotropic activations enable DYNAMIC TOPOLOGY:
exact pruning and growing with minimal function loss.

The key properties that enable dynamic topology in Iso networks:
  1. Reparameterisation invariance: diagonalisation is exact (Test A)
  2. Singular value ranking: Sigma_ii predicts pruning impact (Test G)
  3. Neurogenesis: scaffold neurons are exactly inert (Test B)
  4. Pruning stability: 50% pruning with minimal accuracy loss (Test H)

Does LN+tanh preserve these properties? Specifically:
  (a) Can you still diagonalise the weight matrix with the same exactness?
      LN introduces an extra normalisation step that changes the effective
      weight space -- the SVD of W alone may no longer rank neurons correctly.
  (b) Does the singular value still predict pruning impact with LN present?
      Or does LN's per-sample normalisation make SV-based ranking unreliable?
  (c) Are scaffold neurons still inert when LN is present?
      LN normalises across ALL neurons including scaffolds -- adding a
      zero-weight neuron changes the normalisation for all others.

This test directly compares:
  - Iso: prune by SV, measure accuracy drop correlation
  - LN+tanh: prune by SV, measure accuracy drop correlation
  - LN+tanh: prune by neuron output norm (alternative criterion with LN)

If LN breaks (c) -- scaffolds aren't inert -- then LN+tanh cannot do
neurogenesis, and the paper's topology claim survives regardless of accuracy.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from dynamic_topology_net.core import load_cifar10
from dynamic_topology_net.core.train_utils import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_AJ')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS  = 24
LR      = 0.08
BATCH   = 128
WIDTH   = 32
SEED    = 42


class IsoAct(nn.Module):
    def forward(self, x):
        n = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.tanh(n) * x / n


class IsoMLP(nn.Module):
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        self.W1 = nn.Linear(input_dim, width)
        self.W2 = nn.Linear(width, num_classes)
        self.act = IsoAct()

    def forward(self, x):
        return self.W2(self.act(self.W1(x)))


class LNTanhMLP(nn.Module):
    def __init__(self, input_dim, width, num_classes):
        super().__init__()
        self.W1 = nn.Linear(input_dim, width)
        self.ln = nn.LayerNorm(width)
        self.W2 = nn.Linear(width, num_classes)

    def forward(self, x):
        return self.W2(torch.tanh(self.ln(self.W1(x))))


# =============================================================================
# Tests
# =============================================================================

def test_scaffold_inertness(model_class, input_dim, width, num_classes,
                             train_loader, test_loader, tag):
    """
    Add scaffold neurons (zero W1 row + zero W2 col) and check if predictions change.
    For Iso: should be exactly inert.
    For LN+tanh: LN normalises across all neurons, so adding zeros CHANGES
                 the normalisation denominator -> predictions WILL change.
    """
    torch.manual_seed(SEED)
    model = model_class(input_dim, width, num_classes).to(DEVICE)

    # Train briefly
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    for epoch in range(1, 6):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()

    model.eval()
    probe_x = next(iter(test_loader))[0][:256].to(DEVICE)

    with torch.no_grad():
        out_before = model(probe_x)

    # Add scaffold neuron
    if isinstance(model, IsoMLP):
        W1_old = model.W1.weight.data   # (width, input_dim)
        b1_old = model.W1.bias.data     # (width,)
        W2_old = model.W2.weight.data   # (num_classes, width)
        b2_old = model.W2.bias.data

        new_W1 = torch.cat([W1_old, torch.zeros(1, input_dim, device=DEVICE)], dim=0)
        new_b1 = torch.cat([b1_old, torch.zeros(1, device=DEVICE)])
        new_W2 = torch.cat([W2_old, torch.zeros(num_classes, 1, device=DEVICE)], dim=1)

        model.W1 = nn.Linear(input_dim, width + 1, bias=True)
        model.W1.weight.data = new_W1
        model.W1.bias.data   = new_b1
        model.W2 = nn.Linear(width + 1, num_classes, bias=True)
        model.W2.weight.data = new_W2
        model.W2.bias.data   = b2_old

    elif isinstance(model, LNTanhMLP):
        W1_old = model.W1.weight.data
        b1_old = model.W1.bias.data
        W2_old = model.W2.weight.data
        b2_old = model.W2.bias.data
        ln_w   = model.ln.weight.data
        ln_b   = model.ln.bias.data

        new_W1 = torch.cat([W1_old, torch.zeros(1, input_dim, device=DEVICE)], dim=0)
        new_b1 = torch.cat([b1_old, torch.zeros(1, device=DEVICE)])
        new_W2 = torch.cat([W2_old, torch.zeros(num_classes, 1, device=DEVICE)], dim=1)
        new_ln_w = torch.cat([ln_w, torch.ones(1, device=DEVICE)])
        new_ln_b = torch.cat([ln_b, torch.zeros(1, device=DEVICE)])

        model.W1 = nn.Linear(input_dim, width + 1, bias=True)
        model.W1.weight.data = new_W1
        model.W1.bias.data   = new_b1
        model.W2 = nn.Linear(width + 1, num_classes, bias=True)
        model.W2.weight.data = new_W2
        model.W2.bias.data   = b2_old
        model.ln = nn.LayerNorm(width + 1)
        model.ln.weight.data = new_ln_w
        model.ln.bias.data   = new_ln_b

    model.eval()
    with torch.no_grad():
        out_after = model(probe_x)

    max_diff = (out_after - out_before).abs().max().item()
    mean_diff = (out_after - out_before).abs().mean().item()
    pred_match = (out_before.argmax(1) == out_after.argmax(1)).float().mean().item()

    print(f'  [{tag}] Scaffold inertness: max_diff={max_diff:.6f}  '
          f'mean_diff={mean_diff:.6f}  pred_match={pred_match:.4f}')
    return max_diff, mean_diff, pred_match


def test_sv_pruning_correlation(model_class, input_dim, width, num_classes,
                                 train_loader, test_loader, tag):
    """
    Train model. Prune each neuron individually. Measure accuracy drop.
    Correlate with: SV (W1 row norm proxy) and output magnitude.
    Returns pearson r for each criterion.
    """
    torch.manual_seed(SEED)
    model = model_class(input_dim, width, num_classes).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()

    base_acc = evaluate(model, test_loader, DEVICE)
    print(f'  [{tag}] Trained acc: {base_acc:.4f}')

    acc_drops = []
    sv_values = []
    w2_norms  = []

    W1 = model.W1.weight.data  # (width, input_dim)
    W2 = model.W2.weight.data  # (num_classes, width)

    for i in range(width):
        # Prune neuron i: zero its W1 row and W2 column
        W1_pruned = W1.clone(); W1_pruned[i] = 0
        W2_pruned = W2.clone(); W2_pruned[:, i] = 0

        model.W1.weight.data = W1_pruned
        model.W2.weight.data = W2_pruned
        pruned_acc = evaluate(model, test_loader, DEVICE)
        acc_drops.append(base_acc - pruned_acc)

        # Restore
        model.W1.weight.data = W1
        model.W2.weight.data = W2

        sv_values.append(W1[i].norm().item())
        w2_norms.append(W2[:, i].norm().item())

    sv_arr   = np.array(sv_values)
    w2_arr   = np.array(w2_norms)
    drop_arr = np.array(acc_drops)

    r_sv, _    = pearsonr(sv_arr,        drop_arr)
    r_w2, _    = pearsonr(w2_arr,        drop_arr)
    r_comp, _  = pearsonr(sv_arr*w2_arr, drop_arr)

    print(f'  [{tag}] Pruning r: SV={r_sv:.4f}  W2-norm={r_w2:.4f}  '
          f'Composite={r_comp:.4f}')
    return r_sv, r_w2, r_comp, base_acc


def main():
    print(f'Device: {DEVICE}')
    print('Question: Does LN+tanh break dynamic topology properties?')
    print('Loading CIFAR-10...')
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    print('\n--- TEST 1: Scaffold neuron inertness ---')
    scaffold_results = {}
    for cls, tag in [(IsoMLP, 'Iso'), (LNTanhMLP, 'LN+tanh')]:
        max_d, mean_d, pred_m = test_scaffold_inertness(
            cls, input_dim, WIDTH, num_classes, train_loader, test_loader, tag)
        scaffold_results[tag] = (max_d, mean_d, pred_m)

    print('\n--- TEST 2: SV pruning criterion correlation ---')
    pruning_results = {}
    for cls, tag in [(IsoMLP, 'Iso'), (LNTanhMLP, 'LN+tanh')]:
        r_sv, r_w2, r_comp, acc = test_sv_pruning_correlation(
            cls, input_dim, WIDTH, num_classes, train_loader, test_loader, tag)
        pruning_results[tag] = (r_sv, r_w2, r_comp, acc)

    # Verdict
    print(f'\n{"="*65}')
    print('TOPOLOGY COMPATIBILITY VERDICT')
    print(f'{"="*65}')

    iso_inert  = scaffold_results['Iso'][0] < 1e-4
    ln_inert   = scaffold_results['LN+tanh'][0] < 1e-4
    print(f'\nScaffold inertness:')
    print(f'  Iso:     max_diff={scaffold_results["Iso"][0]:.6f}  '
          f'pred_match={scaffold_results["Iso"][2]:.4f}  '
          f'-> {"INERT" if iso_inert else "NOT INERT"}')
    print(f'  LN+tanh: max_diff={scaffold_results["LN+tanh"][0]:.6f}  '
          f'pred_match={scaffold_results["LN+tanh"][2]:.4f}  '
          f'-> {"INERT" if ln_inert else "NOT INERT (LN normalises across all neurons)"}')

    print(f'\nPruning criterion:')
    for tag in ('Iso', 'LN+tanh'):
        r_sv, r_w2, r_comp, acc = pruning_results[tag]
        print(f'  {tag}: r_SV={r_sv:.4f}  r_W2={r_w2:.4f}  r_composite={r_comp:.4f}  acc={acc:.4f}')

    scaffold_verdict = ('Iso: inert. LN+tanh: NOT inert -- LN normalises across all neurons, '
                        'so adding a zero scaffold changes existing neuron outputs.'
                        if not ln_inert else
                        'Both models support inert scaffolds.')

    pruning_verdict = ('SV criterion weaker for LN+tanh than Iso.'
                       if pruning_results['LN+tanh'][0] < pruning_results['Iso'][0] - 0.05
                       else 'SV criterion comparable for both models.')

    overall = ('LN+tanh CANNOT support exact dynamic topology: scaffold neurons are not inert '
               'due to LayerNorm normalising across all neurons. The paper\'s topology claims '
               'survive as a unique advantage of isotropic activation.'
               if not ln_inert else
               'LN+tanh supports comparable dynamic topology properties to Iso.')

    print(f'\nOverall: {overall}')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for tag, color in [('Iso', '#1f77b4'), ('LN+tanh', '#2ca02c')]:
        _, _, pred_m = scaffold_results[tag]
        max_d, mean_d, _ = scaffold_results[tag]
        ax.bar([tag], [1 - pred_m], color=color, alpha=0.8, label=tag)
    ax.set_ylabel('Fraction of predictions changed by scaffold')
    ax.set_title('Scaffold inertness\n(0 = perfectly inert, >0 = broken)')
    ax.set_ylim(0, 1)
    ax.axhline(0, color='black', ls='-', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    ax = axes[1]
    x = np.arange(3)
    width_bar = 0.35
    iso_bars  = [pruning_results['Iso'][i]    for i in range(3)]
    ln_bars   = [pruning_results['LN+tanh'][i] for i in range(3)]
    ax.bar(x - width_bar/2, iso_bars,  width_bar, label='Iso',     color='#1f77b4', alpha=0.8)
    ax.bar(x + width_bar/2, ln_bars,   width_bar, label='LN+tanh', color='#2ca02c', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(['r(SV)', 'r(W2-norm)', 'r(Composite)'])
    ax.set_ylabel('Pearson r with accuracy drop')
    ax.set_title('Pruning criterion correlation\n(higher = better criterion)')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    plt.suptitle('Test AJ: LN+tanh Topology Compatibility', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'topology_compatibility.png'), dpi=150)
    print('\nPlot saved to results/test_AJ/topology_compatibility.png')

    md = f"""# Test AJ -- LN+tanh Topology Compatibility

## Setup
- Width: {WIDTH}, Epochs: {EPOCHS}, seed={SEED}
- Device: {DEVICE}

## Question
Does LN+tanh support exact dynamic topology (pruning/growing) like Iso?
If not, the paper's contribution survives as topology-specific even if
LN+tanh beats Iso on accuracy.

## Test 1: Scaffold Neuron Inertness

| Model | Max output diff | Mean output diff | Pred match | Verdict |
|---|---|---|---|---|
| Iso     | {scaffold_results['Iso'][0]:.6f} | {scaffold_results['Iso'][1]:.6f} | {scaffold_results['Iso'][2]:.4f} | {"INERT" if iso_inert else "NOT INERT"} |
| LN+tanh | {scaffold_results['LN+tanh'][0]:.6f} | {scaffold_results['LN+tanh'][1]:.6f} | {scaffold_results['LN+tanh'][2]:.4f} | {"INERT" if ln_inert else "NOT INERT"} |

{scaffold_verdict}

## Test 2: Pruning Criterion Correlation

| Model | r(SV) | r(W2-norm) | r(Composite) | Acc |
|---|---|---|---|---|
| Iso     | {pruning_results['Iso'][0]:.4f} | {pruning_results['Iso'][1]:.4f} | {pruning_results['Iso'][2]:.4f} | {pruning_results['Iso'][3]:.4f} |
| LN+tanh | {pruning_results['LN+tanh'][0]:.4f} | {pruning_results['LN+tanh'][1]:.4f} | {pruning_results['LN+tanh'][2]:.4f} | {pruning_results['LN+tanh'][3]:.4f} |

{pruning_verdict}

## Overall Verdict
{overall}

![Topology compatibility](topology_compatibility.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w') as f:
        f.write(md)
    print('Results saved to results/test_AJ/results.md')


if __name__ == '__main__':
    main()
