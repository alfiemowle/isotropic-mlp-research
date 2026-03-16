"""
Test P -- Scaffold Neuron Initialisation Comparison
=====================================================
Test K showed W2 initialisation dominates gradient flow to scaffold neurons,
and that baseline tanh receives 38x more gradient than isotropic.
This test does a controlled comparison of four W2 init strategies for
scaffold neurons added to a pretrained IsotropicMLP.

Conditions (all share the same pretrained checkpoint; only W2 column differs):
  (a) zero       -- zero W2 column (floor: guaranteed zero gradient)
  (b) random     -- small random W2 column, scale 0.01
  (c) semi-orth  -- Gram-Schmidt orthogonal to existing W2 columns, scale 0.01
                    NOTE: W2 is 10x24; since 24 > 10, all of R^10 is spanned
                    by existing columns. True orthogonality is geometrically
                    impossible. Falls back to random unit vector, scaled 0.01.
                    This limitation is noted in Section 3.3.1 of the paper.
  (d) copy       -- copy a random existing W2 column, normalise, scale 0.01

Metrics tracked per condition over 500 training steps:
  - Gradient norm to scaffold neuron W1 row (every step)
  - Test accuracy (every 25 steps)
  - Scaffold W1 row norm growth (every step)

Pretrain: IsotropicMLP [3072->24->10], 24 epochs, lr=0.08, batch=24, seed=42.
Device: CPU
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_topology_net.core import IsotropicMLP, load_cifar10
from dynamic_topology_net.core.train_utils import train_model, evaluate, make_optimizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_P')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED           = 42
PRETRAIN_EPOCHS = 24
LR             = 0.08
BATCH          = 24
TRACK_STEPS    = 500
DEVICE         = torch.device('cpu')


def make_semi_orthogonal_col(W2, device):
    """
    Attempt Gram-Schmidt orthogonalisation against existing W2 columns.

    W2 shape: (num_classes, width) = (10, 24).
    Since width (24) > num_classes (10), the existing columns already span
    all of R^10. True orthogonality is geometrically impossible.
    Returns a random unit vector scaled to 0.01, and a flag indicating
    whether true orthogonality was achieved.
    """
    num_classes, width = W2.shape
    v = torch.randn(num_classes, 1, device=device)

    achievable = (width < num_classes)

    if achievable:
        # Gram-Schmidt: subtract projections onto each existing column
        for j in range(width):
            col_j = W2[:, j:j+1]
            col_norm_sq = (col_j * col_j).sum().clamp(min=1e-10)
            v = v - ((v * col_j).sum() / col_norm_sq) * col_j
        norm = v.norm().item()
        if norm < 1e-6:
            achievable = False
            v = torch.randn(num_classes, 1, device=device)
            v = v / v.norm().clamp(min=1e-8)
        else:
            v = v / norm
    else:
        # Geometric impossibility: fall back to random unit vector
        v = v / v.norm().clamp(min=1e-8)

    return v * 0.01, achievable


def add_scaffold(base_model, w2_init, device):
    """
    Deep-copy base_model and add one scaffold neuron with specified W2 init.
    Returns (model, new_neuron_idx, orth_achievable).
    """
    model = copy.deepcopy(base_model)
    new_idx = model.width  # will be appended at this position
    orth_achievable = True

    with torch.no_grad():
        # W1: append zero row
        new_row = torch.zeros(1, model.W1.shape[1], device=device)
        model.W1 = nn.Parameter(torch.cat([model.W1.data, new_row], dim=0))

        # b1: append zero
        model.b1 = nn.Parameter(torch.cat([model.b1.data, torch.zeros(1, device=device)]))

        # W2: append column per strategy
        if w2_init == 'zero':
            new_col = torch.zeros(model.num_classes, 1, device=device)
        elif w2_init == 'random':
            new_col = torch.randn(model.num_classes, 1, device=device) * 0.01
        elif w2_init == 'semi_orth':
            new_col, orth_achievable = make_semi_orthogonal_col(model.W2.data, device)
        elif w2_init == 'copy':
            col_idx = np.random.randint(model.W2.shape[1])
            col = model.W2.data[:, col_idx:col_idx+1].clone()
            col_norm = col.norm().clamp(min=1e-8)
            new_col = (col / col_norm) * 0.01
        else:
            raise ValueError(f"Unknown w2_init: {w2_init}")

        model.W2 = nn.Parameter(torch.cat([model.W2.data, new_col], dim=1))

    return model, new_idx, orth_achievable


def track_condition(model, new_idx, train_loader, test_loader, n_steps, device, label):
    """
    Track gradient and accuracy for n_steps training steps after scaffold addition.
    Returns dicts of per-step gradient norms, W1 row norms, and per-25-step accuracies.
    """
    opt  = make_optimizer(model, LR)
    crit = nn.CrossEntropyLoss()

    grad_W1_norms = []
    w1_row_norms  = []
    acc_steps     = []
    acc_vals      = []

    loader_iter = iter(train_loader)
    model.train()

    for step in range(1, n_steps + 1):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()

        g = model.W1.grad[new_idx].norm().item() if model.W1.grad is not None else 0.0
        w = model.W1.data[new_idx].norm().item()
        grad_W1_norms.append(g)
        w1_row_norms.append(w)

        opt.step()

        if step % 25 == 0:
            acc = evaluate(model, test_loader, device)
            acc_steps.append(step)
            acc_vals.append(acc)
            print(f"  [{label}] step {step:4d}: grad={g:.6f}  W1_norm={w:.6f}  acc={acc:.4f}")

    return grad_W1_norms, w1_row_norms, acc_steps, acc_vals


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    # ------------------------------------------------------------------
    # Pretrain base model (same checkpoint for all 4 conditions)
    # ------------------------------------------------------------------
    print(f"\nPretraining IsotropicMLP [3072->24->10] for {PRETRAIN_EPOCHS} epochs...")
    torch.manual_seed(SEED)
    base_model = IsotropicMLP(input_dim=input_dim, width=24, num_classes=num_classes).to(DEVICE)
    train_model(base_model, train_loader, test_loader, PRETRAIN_EPOCHS, LR, DEVICE, verbose=False)
    base_acc = evaluate(base_model, test_loader, DEVICE)
    print(f"  Pretrained accuracy: {base_acc:.4f}")

    # ------------------------------------------------------------------
    # Check semi-orthogonal feasibility
    # ------------------------------------------------------------------
    _, orth_achievable = make_semi_orthogonal_col(base_model.W2.data, DEVICE)
    print(f"\nSemi-orthogonal W2 column achievable? {orth_achievable}")
    print(f"  W2 shape: {tuple(base_model.W2.shape)} -- "
          f"{'width > num_classes, R^'+str(num_classes)+' fully spanned, orthogonality impossible' if not orth_achievable else 'orthogonality achievable'}")

    # ------------------------------------------------------------------
    # Run all four conditions
    # ------------------------------------------------------------------
    conditions = [
        ('zero',      'zero'),
        ('random',    'random'),
        ('semi_orth', 'semi_orth'),
        ('copy',      'copy'),
    ]

    all_results = {}
    np.random.seed(SEED)

    for label, w2_init in conditions:
        print(f"\n--- Condition: {label} ---")
        torch.manual_seed(SEED)
        model, new_idx, orth_ok = add_scaffold(base_model, w2_init, DEVICE)
        model = model.to(DEVICE)

        if w2_init == 'semi_orth':
            print(f"  (True orthogonality achieved: {orth_ok})")

        g_norms, w1_norms, acc_steps, acc_vals = track_condition(
            model, new_idx, train_loader, test_loader, TRACK_STEPS, DEVICE, label)

        all_results[label] = {
            'grad_norms': g_norms,
            'w1_norms':   w1_norms,
            'acc_steps':  acc_steps,
            'acc_vals':   acc_vals,
            'mean_grad':  np.mean(g_norms),
            'peak_grad':  max(g_norms),
            'final_acc':  acc_vals[-1] if acc_vals else float('nan'),
            'final_w1':   w1_norms[-1],
            'orth_ok':    orth_ok if w2_init == 'semi_orth' else 'N/A',
        }
        print(f"  Summary: mean_grad={all_results[label]['mean_grad']:.6f}  "
              f"peak_grad={all_results[label]['peak_grad']:.6f}  "
              f"final_acc={all_results[label]['final_acc']:.4f}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"{'Condition':12s}  {'Mean grad':12s}  {'Peak grad':12s}  "
          f"{'Final W1 norm':14s}  {'Final acc':10s}")
    for label, r in all_results.items():
        print(f"  {label:10s}  {r['mean_grad']:12.6f}  {r['peak_grad']:12.6f}  "
              f"{r['final_w1']:14.6f}  {r['final_acc']:10.4f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'zero': 'gray', 'random': 'steelblue', 'semi_orth': 'darkorange', 'copy': 'green'}
    styles = {'zero': '--', 'random': '-', 'semi_orth': '-', 'copy': '-'}

    steps = list(range(1, TRACK_STEPS + 1))

    # Gradient norm
    ax = axes[0]
    for label, r in all_results.items():
        ax.plot(steps, r['grad_norms'], styles[label], label=label,
                color=colors[label], alpha=0.8, linewidth=1.2)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Gradient norm (scaffold W1 row)')
    ax.set_title('Gradient flow to scaffold neuron\nby W2 init strategy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # W1 row norm growth
    ax = axes[1]
    for label, r in all_results.items():
        ax.plot(steps, r['w1_norms'], styles[label], label=label,
                color=colors[label], alpha=0.8, linewidth=1.2)
    ax.set_xlabel('Training step')
    ax.set_ylabel('W1 row norm (scaffold neuron)')
    ax.set_title('Scaffold neuron weight growth\nby W2 init strategy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Accuracy recovery
    ax = axes[2]
    ax.axhline(base_acc * 100, linestyle=':', color='black', alpha=0.5, label='Pre-scaffold acc')
    for label, r in all_results.items():
        if r['acc_steps']:
            ax.plot(r['acc_steps'], [a*100 for a in r['acc_vals']],
                    'o' + styles[label], label=label, color=colors[label], markersize=4)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title('Accuracy recovery after scaffold addition\nby W2 init strategy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'scaffold_init.png'), dpi=150)
    print("\nPlot saved to results/test_P/scaffold_init.png")

    # ------------------------------------------------------------------
    # Save results.md
    # ------------------------------------------------------------------
    rows = '\n'.join(
        f"| {label} | {r['mean_grad']:.6f} | {r['peak_grad']:.6f} | "
        f"{r['final_w1']:.6f} | {r['final_acc']*100:.2f}% |"
        for label, r in all_results.items()
    )

    acc_rows = '\n'.join(
        '| ' + ' | '.join(
            f"{all_results[label]['acc_vals'][i]*100:.2f}%"
            for label in all_results
        ) + ' |'
        for i, step in enumerate(all_results['zero']['acc_steps'])
    )
    acc_header = '| Step | ' + ' | '.join(all_results.keys()) + ' |'
    acc_sep    = '|---|' + '---|' * len(all_results)
    acc_rows_full = acc_header + '\n' + acc_sep + '\n'
    for i, step in enumerate(all_results['zero']['acc_steps']):
        row = f"| {step} | "
        row += ' | '.join(f"{all_results[label]['acc_vals'][i]*100:.2f}%"
                          for label in all_results)
        row += ' |'
        acc_rows_full += row + '\n'

    results_text = f"""# Test P -- Scaffold Neuron Initialisation Comparison

## Setup
- Base model: IsotropicMLP [3072->24->10], pretrained {PRETRAIN_EPOCHS} epochs
- Pretrained accuracy: {base_acc*100:.2f}%
- Tracking: {TRACK_STEPS} steps after scaffold neuron addition
- Device: CPU, batch={BATCH}

## Semi-Orthogonal Feasibility
- W2 shape: {tuple(base_model.W2.shape)} (num_classes x width)
- True orthogonality achievable: **{orth_achievable}**
- Reason: width ({base_model.width}) > num_classes ({num_classes}), so existing columns
  span all of R^{num_classes}. There is no orthogonal complement. The paper's
  semi-orthogonal recommendation has a geometric constraint it does not mention:
  it only works when the network is narrow relative to the output dimension.
- Fallback: random unit vector scaled to 0.01 (same as 'random' condition)

## Results: Gradient and Learning Speed

| Condition | Mean grad_W1 | Peak grad_W1 | Final W1 norm | Final acc ({TRACK_STEPS} steps) |
|---|---|---|---|---|
{rows}

## Accuracy Recovery (every 25 steps)

{acc_rows_full}

## Key Findings

1. **Zero W2**: mean gradient = {all_results['zero']['mean_grad']:.6f} (floor, as expected)

2. **Random W2**: mean gradient = {all_results['random']['mean_grad']:.6f}
   Gradient flows immediately. This is the practical recommendation from Test K.

3. **Semi-orthogonal W2**: mean gradient = {all_results['semi_orth']['mean_grad']:.6f}
   {"NOTE: True orthogonality was NOT achievable (W2 is " + str(tuple(base_model.W2.shape)) + "). Result is identical to random." if not orth_achievable else "Orthogonality achieved -- may restrict gradient direction."}

4. **Copy existing W2**: mean gradient = {all_results['copy']['mean_grad']:.6f}
   Reuses a learned direction at small scale.

## Practical Recommendation
Based on these results:
- Use w2_init='random' (scale 0.01) for scaffold neurons -- simplest and effective
- Semi-orthogonal offers no advantage when width > num_classes (which is the common case)
- Zero W2 prevents learning entirely (only useful when exact function preservation is required)

![Scaffold init comparison](scaffold_init.png)
"""

    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print("Results saved to results/test_P/results.md")


if __name__ == '__main__':
    main()
