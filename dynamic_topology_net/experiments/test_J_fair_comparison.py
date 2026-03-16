"""
Test J -- Fair vs Unfair Comparison: Dynamic vs Static
=======================================================
The paper's Fig 5 compares:
  (A) Static network trained for 24 epochs at a fixed width
  (B) Network trained 24 epochs wide, then adapted 48 epochs (72 total)

This is unfair: (B) gets 3x more training. This test runs BOTH comparisons:
  - UNFAIR: static 24 epochs vs dynamic 24+48=72 epochs
  - FAIR:   static 72 epochs vs dynamic 24+48=72 epochs

Also replicates Fig 5 by including the anisotropic (standard tanh) baseline.

Question: does the "start wide, prune down" advantage hold under fair conditions?
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

from dynamic_topology_net.core import IsotropicMLP, BaselineMLP, load_cifar10
from dynamic_topology_net.core.train_utils import train_epoch, evaluate, train_model, make_optimizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_J')
os.makedirs(RESULTS_DIR, exist_ok=True)

PRETRAIN_EPOCHS = 24
ADAPT_EPOCHS    = 48
TOTAL_EPOCHS    = PRETRAIN_EPOCHS + ADAPT_EPOCHS   # 72
LR              = 0.08
BATCH_SIZE      = 24
TARGET_WIDTHS   = [8, 16, 24, 32]
START_WIDTH     = 32     # "start wide" strategy
N_REPEATS       = 4
SEEDS           = list(range(N_REPEATS))
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def adapt_to_width(model, target_width, train_loader, test_loader, device):
    """Adapt model to target_width over ADAPT_EPOCHS (+/-1 per epoch)."""
    opt   = make_optimizer(model, LR)
    crit  = nn.CrossEntropyLoss()
    curve = []

    for epoch in range(1, ADAPT_EPOCHS + 1):
        if model.width < target_width:
            model.grow_neuron(b_star=0.0, w2_init='random')
            opt = make_optimizer(model, LR)
        elif model.width > target_width:
            svs = model.partial_diagonalise()
            model.prune_neuron(svs.argmin().item())
            opt = make_optimizer(model, LR)

        train_epoch(model, train_loader, opt, crit, device)
        acc = evaluate(model, test_loader, device)
        curve.append(acc)

    return curve


def main():
    torch.manual_seed(42)
    print(f"Device: {DEVICE}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH_SIZE)

    # Storage
    results = {
        'static_24':      {},   # {target_width: [acc per seed]}
        'static_72':      {},
        'static_72_tanh': {},
        'dynamic':        {},   # {target_width: [final_acc per seed]}
        'dynamic_tanh':   {},
    }
    curves = {
        'dynamic':      {},   # {target_width: [[curve per seed]]}
        'dynamic_tanh': {},
    }
    for tw in TARGET_WIDTHS:
        results['static_24'][tw]      = []
        results['static_72'][tw]      = []
        results['static_72_tanh'][tw] = []
        results['dynamic'][tw]        = []
        results['dynamic_tanh'][tw]   = []
        curves['dynamic'][tw]         = []
        curves['dynamic_tanh'][tw]    = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        print(f"\n--- Seed {seed} ---")

        # ------------------------------------------------------------------
        # 1. Pretrain wide isotropic (width=32, 24 epochs)
        # ------------------------------------------------------------------
        print(f"  Pretrain IsotropicMLP width={START_WIDTH} x{PRETRAIN_EPOCHS} epochs...", end=' ')
        model_iso_wide = IsotropicMLP(input_dim=input_dim, width=START_WIDTH,
                                       num_classes=num_classes).to(DEVICE)
        train_model(model_iso_wide, train_loader, test_loader,
                    PRETRAIN_EPOCHS, LR, DEVICE, verbose=False)
        acc_pretrain = evaluate(model_iso_wide, test_loader, DEVICE)
        print(f"acc={acc_pretrain:.3f}")

        # ------------------------------------------------------------------
        # 2. Pretrain wide tanh (width=32, 24 epochs)
        # ------------------------------------------------------------------
        print(f"  Pretrain BaselineMLP  width={START_WIDTH} x{PRETRAIN_EPOCHS} epochs...", end=' ')
        model_tanh_wide = BaselineMLP(input_dim=input_dim, width=START_WIDTH,
                                       num_classes=num_classes).to(DEVICE)
        train_model(model_tanh_wide, train_loader, test_loader,
                    PRETRAIN_EPOCHS, LR, DEVICE, verbose=False)
        print(f"acc={evaluate(model_tanh_wide, test_loader, DEVICE):.3f}")

        for target_width in TARGET_WIDTHS:
            # ----------------------------------------------------------------
            # 3. Static isotropic at target width, 24 epochs (UNFAIR baseline)
            # ----------------------------------------------------------------
            m = IsotropicMLP(input_dim=input_dim, width=target_width,
                              num_classes=num_classes).to(DEVICE)
            torch.manual_seed(seed + 500)
            train_model(m, train_loader, test_loader,
                        PRETRAIN_EPOCHS, LR, DEVICE, verbose=False)
            results['static_24'][target_width].append(evaluate(m, test_loader, DEVICE))

            # ----------------------------------------------------------------
            # 4. Static isotropic at target width, 72 epochs (FAIR baseline)
            # ----------------------------------------------------------------
            m2 = IsotropicMLP(input_dim=input_dim, width=target_width,
                               num_classes=num_classes).to(DEVICE)
            torch.manual_seed(seed + 500)
            train_model(m2, train_loader, test_loader,
                        TOTAL_EPOCHS, LR, DEVICE, verbose=False)
            results['static_72'][target_width].append(evaluate(m2, test_loader, DEVICE))

            # ----------------------------------------------------------------
            # 5. Static tanh at target width, 72 epochs
            # ----------------------------------------------------------------
            m3 = BaselineMLP(input_dim=input_dim, width=target_width,
                              num_classes=num_classes).to(DEVICE)
            torch.manual_seed(seed + 500)
            train_model(m3, train_loader, test_loader,
                        TOTAL_EPOCHS, LR, DEVICE, verbose=False)
            results['static_72_tanh'][target_width].append(evaluate(m3, test_loader, DEVICE))

            # ----------------------------------------------------------------
            # 6. Dynamic isotropic: start wide, adapt to target
            # ----------------------------------------------------------------
            m4 = copy.deepcopy(model_iso_wide)
            curve_iso = adapt_to_width(m4, target_width, train_loader, test_loader, DEVICE)
            results['dynamic'][target_width].append(curve_iso[-1])
            curves['dynamic'][target_width].append(curve_iso)
            print(f"  Iso  {START_WIDTH}->{target_width}: final={curve_iso[-1]:.3f}")

            # ----------------------------------------------------------------
            # 7. Dynamic tanh: start wide, adapt (Net2Net-style grow, magnitude prune)
            # ----------------------------------------------------------------
            m5 = copy.deepcopy(model_tanh_wide)
            opt5 = make_optimizer(m5, LR)
            crit5 = nn.CrossEntropyLoss()
            curve_tanh = []
            for ep in range(1, ADAPT_EPOCHS + 1):
                # Prune: remove neuron with smallest max |W1| row norm
                w = m5.net[0].weight.data  # (width, input_dim)
                if m5.width > target_width:
                    row_norms = w.norm(dim=1)
                    prune_idx = row_norms.argmin().item()
                    keep = [i for i in range(m5.width) if i != prune_idx]
                    old_b1 = m5.net[0].bias.data.clone()
                    m5.net[0] = nn.Linear(input_dim, len(keep)).to(DEVICE)
                    m5.net[0].weight.data = w[keep]
                    m5.net[0].bias.data   = old_b1[keep]
                    # Also prune W2
                    w2 = m5.net[2].weight.data
                    b2 = m5.net[2].bias.data
                    m5.net[2] = nn.Linear(len(keep), num_classes).to(DEVICE)
                    m5.net[2].weight.data = w2[:, keep]
                    m5.net[2].bias.data   = b2
                    opt5 = make_optimizer(m5, LR)
                elif m5.width < target_width:
                    # Grow: duplicate a random neuron
                    new_w  = w[np.random.randint(m5.width)].unsqueeze(0)
                    old_b1 = m5.net[0].bias.data.clone()
                    new_b  = old_b1[np.random.randint(m5.width)].unsqueeze(0)
                    new_w2 = torch.zeros(num_classes, 1, device=DEVICE)
                    m5.net[0] = nn.Linear(input_dim, m5.width + 1).to(DEVICE)
                    m5.net[0].weight.data = torch.cat([w, new_w], dim=0)
                    m5.net[0].bias.data   = torch.cat([old_b1, new_b])
                    old_w2 = m5.net[2].weight.data
                    b2     = m5.net[2].bias.data
                    m5.net[2] = nn.Linear(m5.width, num_classes).to(DEVICE)
                    m5.net[2].weight.data = torch.cat([old_w2, new_w2], dim=1)
                    m5.net[2].bias.data   = b2
                    opt5 = make_optimizer(m5, LR)

                train_epoch(m5, train_loader, opt5, crit5, DEVICE)
                acc5 = evaluate(m5, test_loader, DEVICE)
                curve_tanh.append(acc5)

            results['dynamic_tanh'][target_width].append(curve_tanh[-1])
            curves['dynamic_tanh'][target_width].append(curve_tanh)
            print(f"  Tanh {START_WIDTH}->{target_width}: final={curve_tanh[-1]:.3f}")

    # =========================================================================
    # Print summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Target':>8}  {'Stat24':>8}  {'Stat72':>8}  {'Stat72T':>8}  {'Dyn72':>8}  {'DynT72':>8}")
    for tw in TARGET_WIDTHS:
        s24  = np.mean(results['static_24'][tw])
        s72  = np.mean(results['static_72'][tw])
        s72t = np.mean(results['static_72_tanh'][tw])
        d72  = np.mean(results['dynamic'][tw])
        dt72 = np.mean(results['dynamic_tanh'][tw])
        print(f"{tw:>8}  {s24:>8.3f}  {s72:>8.3f}  {s72t:>8.3f}  {d72:>8.3f}  {dt72:>8.3f}")

    # =========================================================================
    # Plot
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(TARGET_WIDTHS))
    w = 0.18

    ax = axes[0]
    ax.set_title('Unfair comparison (paper protocol)\nDynamic = 72ep, Static = 24ep')
    for i, (label, key, offset, color) in enumerate([
        ('Static-Iso 24ep', 'static_24', -w, 'steelblue'),
        ('Dynamic-Iso 72ep', 'dynamic', w, 'darkorange'),
        ('Dynamic-Tanh 72ep', 'dynamic_tanh', 2*w, 'green'),
    ]):
        means = [np.mean(results[key][tw]) * 100 for tw in TARGET_WIDTHS]
        stds  = [np.std(results[key][tw])  * 100 for tw in TARGET_WIDTHS]
        ax.bar(x + offset, means, w, yerr=stds, label=label, color=color, alpha=0.8, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([f'w={tw}' for tw in TARGET_WIDTHS])
    ax.set_ylabel('Test Accuracy (%)'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.set_title('Fair comparison (equal epochs = 72)\nAll variants trained for 72 total epochs')
    for i, (label, key, offset, color) in enumerate([
        ('Static-Iso 72ep', 'static_72', -w, 'steelblue'),
        ('Static-Tanh 72ep', 'static_72_tanh', 0, 'royalblue'),
        ('Dynamic-Iso 72ep', 'dynamic', w, 'darkorange'),
        ('Dynamic-Tanh 72ep', 'dynamic_tanh', 2*w, 'green'),
    ]):
        means = [np.mean(results[key][tw]) * 100 for tw in TARGET_WIDTHS]
        stds  = [np.std(results[key][tw])  * 100 for tw in TARGET_WIDTHS]
        ax.bar(x + offset, means, w, yerr=stds, label=label, color=color, alpha=0.8, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([f'w={tw}' for tw in TARGET_WIDTHS])
    ax.set_ylabel('Test Accuracy (%)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fair_vs_unfair.png'), dpi=150)
    print("Plot saved to results/test_J/fair_vs_unfair.png")

    # =========================================================================
    # Save results.md
    # =========================================================================
    def fmt_row(key, label):
        vals = [f"{np.mean(results[key][tw])*100:.1f}+-{np.std(results[key][tw])*100:.1f}" for tw in TARGET_WIDTHS]
        return f"| {label} | " + " | ".join(vals) + " |"

    results_text = f"""# Test J -- Fair vs Unfair Comparison: Dynamic vs Static

## The Issue
The paper's Fig 5 compares dynamic networks (72 total epochs) against static
networks (24 epochs). This is potentially unfair -- more training time could
explain the improvement, not the dynamic topology itself.

## Setup
- Start width: {START_WIDTH} (wide initialisation)
- Target widths: {TARGET_WIDTHS}
- Dynamic schedule: +/-1 neuron/epoch during {ADAPT_EPOCHS} adaptation epochs
- Repeats: {N_REPEATS} seeds

## Results

### Unfair comparison (paper protocol)

| Model | w=8 | w=16 | w=24 | w=32 |
|---|---|---|---|---|
{fmt_row('static_24', 'Static-Iso 24ep')}
{fmt_row('dynamic', 'Dynamic-Iso 72ep')}
{fmt_row('dynamic_tanh', 'Dynamic-Tanh 72ep')}

### Fair comparison (all 72 epochs)

| Model | w=8 | w=16 | w=24 | w=32 |
|---|---|---|---|---|
{fmt_row('static_72', 'Static-Iso 72ep')}
{fmt_row('static_72_tanh', 'Static-Tanh 72ep')}
{fmt_row('dynamic', 'Dynamic-Iso 72ep')}
{fmt_row('dynamic_tanh', 'Dynamic-Tanh 72ep')}

## Key Questions Answered

### Q1: Does isotropic outperform standard tanh?
- Static-Iso 72ep vs Static-Tanh 72ep: isotropic {'wins' if all(np.mean(results['static_72'][tw]) > np.mean(results['static_72_tanh'][tw]) for tw in TARGET_WIDTHS) else 'has mixed results'}
- Dynamic-Iso 72ep vs Dynamic-Tanh 72ep: isotropic {'wins' if all(np.mean(results['dynamic'][tw]) > np.mean(results['dynamic_tanh'][tw]) for tw in TARGET_WIDTHS) else 'has mixed results'}

### Q2: Does dynamic topology help beyond just more training?
- Dynamic-Iso 72ep vs Static-Iso 72ep:
{'  - ' + chr(10) + '  - '.join(f"w={tw}: Dyn={np.mean(results['dynamic'][tw]):.3f} vs Stat={np.mean(results['static_72'][tw]):.3f} (diff={np.mean(results['dynamic'][tw])-np.mean(results['static_72'][tw]):+.3f})" for tw in TARGET_WIDTHS)}

### Q3: Is the paper's advantage real or an artifact of unequal training?
- Under **unfair** conditions (paper protocol): dynamic has extra training advantage
- Under **fair** conditions (equal epochs): {'dynamic still outperforms static' if np.mean([np.mean(results['dynamic'][tw]) for tw in TARGET_WIDTHS]) > np.mean([np.mean(results['static_72'][tw]) for tw in TARGET_WIDTHS]) else 'the advantage is reduced or eliminated'}

![Fair vs Unfair comparison](fair_vs_unfair.png)
"""

    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print("Results saved to results/test_J/results.md")


if __name__ == '__main__':
    main()
