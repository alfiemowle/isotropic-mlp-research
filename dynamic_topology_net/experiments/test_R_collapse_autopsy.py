"""
Test R -- Representational Collapse Autopsy
============================================
Test M/Q showed Base-2L and Base-3L accuracy collapses while Iso keeps improving.
Two competing hypotheses:
  (a) GRADIENT VANISHING: gradients shrink through baseline layers, parameters stop updating
  (b) REPRESENTATIONAL COLLAPSE: all neurons converge to the same direction,
      the hidden representation loses rank, the network becomes effectively 1-neuron wide

These have different implications. If (a): a training trick (residual connection, normalisation)
would fix it. If (b): the architecture is fundamentally broken at depth — isotropy is providing
something structurally necessary, not just a training convenience.

We track three diagnostics at every epoch for Iso/Base x 1L/2L/3L:
  1. Per-layer mean gradient norm (vanishing gradient test)
  2. Effective rank of each hidden layer's representation on a fixed eval batch
     effective_rank = exp(-sum(p_i * log(p_i)))  where p_i = s_i^2 / sum(s_j^2)
     Range: 1 (all variance on one direction) to width (perfectly isotropic)
  3. Mean pairwise cosine similarity of the FIRST weight matrix rows
     (neuron alignment: 0=orthogonal/diverse, 1=all identical)

Width=24, 24 epochs, batch=128, seed=42, CPU.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from dynamic_topology_net.core import (
    IsotropicMLP, BaselineMLP,
    DeepIsotropicMLP, DeepBaselineMLP,
    IsotropicMLP3L, BaselineMLP3L,
    load_cifar10
)
from dynamic_topology_net.core.train_utils import evaluate, make_optimizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_R')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED   = 42
EPOCHS = 24
LR     = 0.08
BATCH  = 128
WIDTH  = 24
DEVICE = torch.device('cpu')
EVAL_N = 512   # samples for representation analysis per epoch


def effective_rank(H):
    """
    Effective rank of representation matrix H (N x d).
    Uses squared singular values as a probability distribution.
    effective_rank = exp(entropy of eigenspectrum)
    """
    with torch.no_grad():
        svs = torch.linalg.svdvals(H)
        p = svs.pow(2)
        p = p / p.sum().clamp(min=1e-10)
        # Clamp to avoid log(0)
        p = p.clamp(min=1e-10)
        entropy = -(p * p.log()).sum().item()
        return float(np.exp(entropy))


def mean_pairwise_cosine(W):
    """Mean cosine similarity of all pairs of rows of W (excluding self-similarity)."""
    with torch.no_grad():
        norms = W.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W / norms
        gram = W_norm @ W_norm.T  # (width, width)
        n = gram.shape[0]
        if n <= 1:
            return 0.0
        # Off-diagonal mean
        mask = ~torch.eye(n, dtype=torch.bool)
        return gram[mask].mean().item()


def get_first_weight(model):
    """Get the first weight matrix regardless of model type."""
    if hasattr(model, 'W1'):
        return model.W1.data
    elif hasattr(model, 'net'):
        return model.net[0].weight.data
    return None


def train_with_diagnostics(model, model_label, train_loader, test_loader,
                            epochs, lr, device, eval_batch):
    """
    Custom training loop that records diagnostic metrics every epoch.
    Returns history dict.
    """
    opt  = make_optimizer(model, lr)
    crit = nn.CrossEntropyLoss()
    eval_x, _ = eval_batch

    history = {
        'test_acc':     [],
        'grad_norms':   defaultdict(list),  # layer_name -> [epoch values]
        'eff_ranks':    defaultdict(list),  # layer_name -> [epoch values]
        'neuron_sim':   [],
    }

    # Register hooks to capture hidden representations
    layer_outputs = {}

    def make_hook(name):
        def hook(module, inp, out):
            layer_outputs[name] = out.detach()
        return hook

    handles = []
    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(make_hook(name)))
            linear_names.append(name)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_grad_norms = defaultdict(list)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None and ('W' in name or 'weight' in name):
                    epoch_grad_norms[name].append(param.grad.norm().item())

            opt.step()

        # --- End of epoch diagnostics ---
        model.eval()
        with torch.no_grad():
            acc = evaluate(model, test_loader, device)
            history['test_acc'].append(acc)

            # Representation analysis via hooks
            _ = model(eval_x)
            # Exclude the final output layer (last linear)
            hidden_names = linear_names[:-1]
            for name in hidden_names:
                if name in layer_outputs:
                    H = layer_outputs[name]  # (EVAL_N, width)
                    history['eff_ranks'][name].append(effective_rank(H))

            # Gradient norms
            for name, norms in epoch_grad_norms.items():
                history['grad_norms'][name].append(np.mean(norms))

            # Neuron cosine similarity (first weight matrix)
            W = get_first_weight(model)
            if W is not None:
                history['neuron_sim'].append(mean_pairwise_cosine(W))

        print(f"  [{model_label}] Epoch {epoch:2d}/{epochs}  "
              f"acc={acc:.3f}  "
              f"neuron_sim={history['neuron_sim'][-1]:.4f}  "
              + (f"eff_rank={list(history['eff_ranks'].values())[0][-1]:.2f}" if history['eff_ranks'] else ''))

    for h in handles:
        h.remove()

    return history


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    # Fixed eval batch for representation analysis
    eval_x_full, eval_y_full = next(iter(test_loader))
    eval_batch = (eval_x_full[:EVAL_N].to(DEVICE), eval_y_full[:EVAL_N].to(DEVICE))

    configs = [
        ('Iso-1L',  lambda: IsotropicMLP(input_dim, WIDTH, num_classes)),
        ('Iso-2L',  lambda: DeepIsotropicMLP(input_dim, WIDTH, num_classes)),
        ('Iso-3L',  lambda: IsotropicMLP3L(input_dim, WIDTH, num_classes)),
        ('Base-1L', lambda: BaselineMLP(input_dim, WIDTH, num_classes)),
        ('Base-2L', lambda: DeepBaselineMLP(input_dim, WIDTH, num_classes)),
        ('Base-3L', lambda: BaselineMLP3L(input_dim, WIDTH, num_classes)),
    ]

    all_history = {}

    for label, make_model in configs:
        print(f"\n{'='*55}")
        print(f"Model: {label}")
        print(f"{'='*55}")
        torch.manual_seed(SEED)
        model = make_model().to(DEVICE)
        history = train_with_diagnostics(
            model, label, train_loader, test_loader,
            EPOCHS, LR, DEVICE, eval_batch)
        all_history[label] = history

    # =========================================================================
    # Analysis: final-epoch summary
    # =========================================================================
    print(f"\n{'='*65}")
    print("FINAL EPOCH DIAGNOSTICS")
    print(f"{'='*65}")
    print(f"{'Model':>10}  {'Acc':>6}  {'NeuronSim':>10}  {'Min eff_rank':>14}  {'Max grad':>10}")
    for label, h in all_history.items():
        acc = h['test_acc'][-1]
        sim = h['neuron_sim'][-1] if h['neuron_sim'] else float('nan')
        # Min effective rank across all hidden layers (worst case)
        min_er = min(min(v) for v in h['eff_ranks'].values()) if h['eff_ranks'] else float('nan')
        max_g  = max(max(v) for v in h['grad_norms'].values()) if h['grad_norms'] else float('nan')
        print(f"  {label:>8}  {acc:>6.3f}  {sim:>10.4f}  {min_er:>14.2f}  {max_g:>10.6f}")

    # =========================================================================
    # Plots
    # =========================================================================
    iso_labels  = ['Iso-1L',  'Iso-2L',  'Iso-3L']
    base_labels = ['Base-1L', 'Base-2L', 'Base-3L']
    iso_colors  = ['#f97a0a', '#d94f00', '#7f2a00']
    base_colors = ['#3a8fd1', '#1a5fa0', '#0a2a5e']
    epochs_x = list(range(1, EPOCHS + 1))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: Iso models
    # Row 2: Base models
    for row_idx, (labels, colors) in enumerate([(iso_labels, iso_colors),
                                                 (base_labels, base_colors)]):
        # --- Test accuracy ---
        ax = axes[row_idx][0]
        for label, color in zip(labels, colors):
            h = all_history[label]
            ax.plot(epochs_x, h['test_acc'], '-', label=label, color=color, linewidth=1.5)
        ax.set_title('Test Accuracy')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # --- Neuron cosine similarity ---
        ax = axes[row_idx][1]
        for label, color in zip(labels, colors):
            h = all_history[label]
            if h['neuron_sim']:
                ax.plot(epochs_x, h['neuron_sim'], '-', label=label, color=color, linewidth=1.5)
        ax.set_title('Neuron Alignment\n(mean pairwise cos-sim of W1 rows)')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Cosine similarity')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # --- Effective rank of first hidden layer ---
        ax = axes[row_idx][2]
        for label, color in zip(labels, colors):
            h = all_history[label]
            if h['eff_ranks']:
                first_key = list(h['eff_ranks'].keys())[0]
                ax.plot(epochs_x, h['eff_ranks'][first_key], '-',
                        label=f'{label} (L1)', color=color, linewidth=1.5)
                if len(h['eff_ranks']) > 1:
                    second_key = list(h['eff_ranks'].keys())[1]
                    ax.plot(epochs_x, h['eff_ranks'][second_key], '--',
                            label=f'{label} (L2)', color=color, linewidth=1.2, alpha=0.7)
        ax.axhline(WIDTH, linestyle=':', color='gray', alpha=0.5, label=f'Max={WIDTH}')
        ax.set_title('Effective Rank of Hidden Representation')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Effective rank')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle('Representational Collapse Autopsy: Iso vs Baseline at 1L/2L/3L', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'collapse_autopsy.png'), dpi=150)
    print("\nPlot saved to results/test_R/collapse_autopsy.png")

    # =========================================================================
    # Save results.md
    # =========================================================================
    def gradient_summary(h):
        if not h['grad_norms']:
            return 'N/A'
        lines = []
        for name, norms in h['grad_norms'].items():
            short = name.split('.')[-2] if '.' in name else name
            lines.append(f"{short}: final={norms[-1]:.6f}, min={min(norms):.6f}")
        return ' | '.join(lines)

    def rank_summary(h):
        if not h['eff_ranks']:
            return 'N/A'
        lines = []
        for name, ranks in h['eff_ranks'].items():
            short = name.split('.')[-2] if '.' in name else name
            lines.append(f"{short}: final={ranks[-1]:.2f}, min={min(ranks):.2f}")
        return ' | '.join(lines)

    model_rows = '\n'.join(
        f"| {label} | {h['test_acc'][-1]*100:.2f}% | "
        f"{h['neuron_sim'][-1]:.4f} | "
        f"{(min(min(v) for v in h['eff_ranks'].values()) if h['eff_ranks'] else float('nan')):.2f} |"
        for label, h in all_history.items()
    )

    # Determine which hypothesis the data supports
    iso_3l_sim  = all_history['Iso-3L']['neuron_sim'][-1] if all_history['Iso-3L']['neuron_sim'] else 0
    base_3l_sim = all_history['Base-3L']['neuron_sim'][-1] if all_history['Base-3L']['neuron_sim'] else 0
    iso_3l_rank  = min(min(v) for v in all_history['Iso-3L']['eff_ranks'].values()) if all_history['Iso-3L']['eff_ranks'] else WIDTH
    base_3l_rank = min(min(v) for v in all_history['Base-3L']['eff_ranks'].values()) if all_history['Base-3L']['eff_ranks'] else WIDTH

    if base_3l_rank < iso_3l_rank * 0.5 and base_3l_sim > iso_3l_sim * 1.5:
        hypothesis = "REPRESENTATIONAL COLLAPSE: Base-3L loses effective rank and neurons align. Isotropy prevents this structurally."
    elif base_3l_rank < iso_3l_rank * 0.5:
        hypothesis = "REPRESENTATIONAL COLLAPSE (rank): Base-3L representation collapses to low-rank subspace."
    elif base_3l_sim > iso_3l_sim * 1.5:
        hypothesis = "REPRESENTATIONAL COLLAPSE (alignment): Base-3L neurons become redundant (high cosine similarity)."
    else:
        hypothesis = "Results inconclusive or suggest gradient vanishing — check gradient norm trajectories."

    results_text = f"""# Test R -- Representational Collapse Autopsy

## Setup
- Width: {WIDTH}, Epochs: {EPOCHS}, lr={LR}, batch={BATCH}, seed={SEED}
- Device: CPU
- Eval representations on {EVAL_N} fixed samples per epoch

## Question
Why does Base-2L/3L accuracy collapse while Iso-2L/3L keeps improving?
Hypothesis A: Gradient vanishing (signal shrinks through layers)
Hypothesis B: Representational collapse (neurons align, rank drops)

## Final-Epoch Summary

| Model | Final Acc | Neuron Sim (W1) | Min Eff Rank |
|---|---|---|---|
{model_rows}

## Gradient Norms (per layer, final epoch)

{chr(10).join(f"**{label}**: {gradient_summary(h)}" for label, h in all_history.items())}

## Effective Rank (per hidden layer, final epoch)

{chr(10).join(f"**{label}**: {rank_summary(h)}" for label, h in all_history.items())}

## Verdict

{hypothesis}

- Iso-3L neuron similarity (final): {iso_3l_sim:.4f}
- Base-3L neuron similarity (final): {base_3l_sim:.4f}
- Iso-3L min effective rank: {iso_3l_rank:.2f} / {WIDTH}
- Base-3L min effective rank: {base_3l_rank:.2f} / {WIDTH}

![Collapse autopsy](collapse_autopsy.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print("Results saved to results/test_R/results.md")


if __name__ == '__main__':
    main()
