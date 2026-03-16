"""
Test S -- Singular Value Spectrum Phase Transitions
====================================================
We treat SVD as a post-training diagnostic. But the SV spectrum evolves during
training — it starts flat (Kaiming random init) and concentrates as neurons
specialise. This test tracks that evolution at every epoch.

Key questions:
  1. Is there a phase transition where the spectrum suddenly concentrates?
     If so, does it coincide with the accuracy inflection point?
  2. When is the SV distribution "ready" for pruning? Early pruning (before
     specialisation) vs late pruning (after hardening) may give different results.
  3. Does Iso-2L develop a richer spectrum than Iso-1L? Does the second layer
     learn complementary features or redundant ones?
  4. How does condition number (max_sv/min_sv) relate to accuracy?
     A high condition number = one dominant direction = prunable neurons exist.

Models: Iso-1L and Iso-2L (width=24)
Tracked at every epoch: full SV vector, condition number, spectral entropy,
  proportion of variance in top-k SVs.
Width=24, 24 epochs, batch=128, seed=42, CPU.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamic_topology_net.core import IsotropicMLP, DeepIsotropicMLP, load_cifar10
from dynamic_topology_net.core.train_utils import train_epoch, evaluate, make_optimizer
import torch.nn as nn

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_S')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED   = 42
EPOCHS = 24
LR     = 0.08
BATCH  = 128
WIDTH  = 24
DEVICE = torch.device('cpu')


def spectral_entropy(svs):
    """Normalised spectral entropy. 0=one dominant direction, 1=perfectly uniform."""
    p = svs / svs.sum().clamp(min=1e-10)
    p = p.clamp(min=1e-10)
    H = -(p * p.log()).sum().item()
    H_max = np.log(len(svs))
    return H / H_max if H_max > 0 else 0.0


def sv_metrics(svs):
    """Compute all SV metrics from a 1D tensor of singular values."""
    svs_np = svs.cpu().numpy()
    cond   = float(svs_np[0] / max(svs_np[-1], 1e-10))
    entropy = spectral_entropy(svs)
    top1_frac = float(svs_np[0]  / svs_np.sum())
    top3_frac = float(svs_np[:3].sum() / svs_np.sum())
    top8_frac = float(svs_np[:8].sum() / svs_np.sum())
    gini = float(np.sum(np.abs(np.subtract.outer(svs_np, svs_np))) / (2 * len(svs_np) * svs_np.sum() + 1e-10))
    return {
        'svs':       svs_np,
        'cond':      cond,
        'entropy':   entropy,
        'top1_frac': top1_frac,
        'top3_frac': top3_frac,
        'top8_frac': top8_frac,
        'gini':      gini,
        'min_sv':    float(svs_np[-1]),
        'max_sv':    float(svs_np[0]),
        'mean_sv':   float(svs_np.mean()),
    }


def get_W1(model):
    if hasattr(model, 'W1'):
        return model.W1.data
    return None


def get_W2(model):
    """Middle weight matrix for 2L model."""
    if hasattr(model, 'W2') and model.W2.shape == (WIDTH, WIDTH):
        return model.W2.data
    return None


def train_and_track_svs(model, model_label, train_loader, test_loader, epochs, lr, device):
    opt  = make_optimizer(model, lr)
    crit = nn.CrossEntropyLoss()

    sv_history_W1 = []
    sv_history_W2 = []
    acc_history   = []
    loss_history  = []

    # Record at init (epoch 0)
    with torch.no_grad():
        W1 = get_W1(model)
        if W1 is not None:
            svs = torch.linalg.svdvals(W1)
            sv_history_W1.append(sv_metrics(svs))
        W2 = get_W2(model)
        if W2 is not None:
            svs2 = torch.linalg.svdvals(W2)
            sv_history_W2.append(sv_metrics(svs2))
        acc_history.append(evaluate(model, test_loader, device))
        loss_history.append(float('nan'))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item(); steps += 1

        model.eval()
        with torch.no_grad():
            acc = evaluate(model, test_loader, device)
            acc_history.append(acc)
            loss_history.append(total_loss / steps)

            W1 = get_W1(model)
            if W1 is not None:
                svs = torch.linalg.svdvals(W1)
                sv_history_W1.append(sv_metrics(svs))
            W2 = get_W2(model)
            if W2 is not None:
                svs2 = torch.linalg.svdvals(W2)
                sv_history_W2.append(sv_metrics(svs2))

        m = sv_history_W1[-1]
        print(f"  [{model_label}] Epoch {epoch:2d}/{epochs}  "
              f"acc={acc:.3f}  cond={m['cond']:.1f}  "
              f"entropy={m['entropy']:.3f}  top3={m['top3_frac']:.3f}")

    return sv_history_W1, sv_history_W2, acc_history, loss_history


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print("Loading CIFAR-10...")
    train_loader, test_loader, input_dim, num_classes = load_cifar10(batch_size=BATCH)

    configs = [
        ('Iso-1L', lambda: IsotropicMLP(input_dim, WIDTH, num_classes)),
        ('Iso-2L', lambda: DeepIsotropicMLP(input_dim, WIDTH, num_classes)),
    ]

    all_results = {}
    for label, make_model in configs:
        print(f"\n{'='*55}\nModel: {label}\n{'='*55}")
        torch.manual_seed(SEED)
        model = make_model().to(DEVICE)
        sv_W1, sv_W2, accs, losses = train_and_track_svs(
            model, label, train_loader, test_loader, EPOCHS, LR, DEVICE)
        all_results[label] = {
            'sv_W1': sv_W1, 'sv_W2': sv_W2,
            'accs': accs, 'losses': losses
        }

    # =========================================================================
    # Detect phase transitions (max delta in entropy or cond)
    # =========================================================================
    print(f"\n{'='*55}")
    print("PHASE TRANSITION ANALYSIS")
    print(f"{'='*55}")
    epochs_x = list(range(0, EPOCHS + 1))

    for label, r in all_results.items():
        entropies = [m['entropy'] for m in r['sv_W1']]
        conds     = [m['cond']    for m in r['sv_W1']]
        accs      = r['accs']

        # Largest single-epoch drop in entropy (concentration event)
        d_entropy = [entropies[i] - entropies[i-1] for i in range(1, len(entropies))]
        transition_epoch = int(np.argmin(d_entropy)) + 1
        acc_at_transition = accs[transition_epoch] if transition_epoch < len(accs) else float('nan')

        print(f"\n{label}:")
        print(f"  Init entropy:  {entropies[0]:.4f}  (1=uniform, 0=one-direction)")
        print(f"  Final entropy: {entropies[-1]:.4f}")
        print(f"  Init cond:     {conds[0]:.1f}")
        print(f"  Final cond:    {conds[-1]:.1f}")
        print(f"  Largest entropy drop: epoch {transition_epoch}  "
              f"(delta={d_entropy[transition_epoch-1]:.4f}, acc={acc_at_transition:.3f})")
        print(f"  Top-3 SVs account for {r['sv_W1'][-1]['top3_frac']*100:.1f}% of total at end")

    # =========================================================================
    # Plots
    # =========================================================================
    fig = plt.figure(figsize=(18, 12))

    colors = {'Iso-1L': 'darkorange', 'Iso-2L': 'red'}
    n_cols = 3
    n_rows = 3

    # 1. Accuracy + entropy on same axis
    ax = fig.add_subplot(n_rows, n_cols, 1)
    for label, r in all_results.items():
        ax.plot(epochs_x, r['accs'], '-', color=colors[label], label=f'{label} acc')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    for label, r in all_results.items():
        entropies = [m['entropy'] for m in r['sv_W1']]
        ax2.plot(epochs_x, entropies, '--', color=colors[label], alpha=0.6, label=f'{label} entropy')
    ax2.set_ylabel('Spectral entropy (W1)', color='gray')
    ax2.legend(fontsize=8, loc='lower right')

    # 2. Condition number
    ax = fig.add_subplot(n_rows, n_cols, 2)
    for label, r in all_results.items():
        conds = [m['cond'] for m in r['sv_W1']]
        ax.semilogy(epochs_x, conds, '-', color=colors[label], label=label)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Condition number (log scale)')
    ax.set_title('Condition Number of W1'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3. Top-k fraction
    ax = fig.add_subplot(n_rows, n_cols, 3)
    for label, r in all_results.items():
        top1 = [m['top1_frac'] for m in r['sv_W1']]
        top3 = [m['top3_frac'] for m in r['sv_W1']]
        top8 = [m['top8_frac'] for m in r['sv_W1']]
        ax.plot(epochs_x, top1, '-',  color=colors[label], label=f'{label} top-1', linewidth=1.5)
        ax.plot(epochs_x, top3, '--', color=colors[label], label=f'{label} top-3', linewidth=1.2)
        ax.plot(epochs_x, top8, ':',  color=colors[label], label=f'{label} top-8', linewidth=1.0)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Fraction of total SV mass')
    ax.set_title('Top-k SV Concentration'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 4-5. SV spectrum heatmaps
    for plot_idx, label in enumerate(['Iso-1L', 'Iso-2L']):
        ax = fig.add_subplot(n_rows, n_cols, 4 + plot_idx)
        r = all_results[label]
        sv_matrix = np.array([m['svs'] for m in r['sv_W1']])  # (epochs+1, width)
        im = ax.imshow(sv_matrix.T, aspect='auto', origin='lower',
                       extent=[0, EPOCHS, 0, WIDTH], cmap='viridis')
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Epoch'); ax.set_ylabel('SV index (0=largest)')
        ax.set_title(f'{label}: W1 SV spectrum over training')

    # 6. Min SV evolution (key: when does smallest SV diverge from largest?)
    ax = fig.add_subplot(n_rows, n_cols, 6)
    for label, r in all_results.items():
        min_svs = [m['min_sv'] for m in r['sv_W1']]
        max_svs = [m['max_sv'] for m in r['sv_W1']]
        ax.plot(epochs_x, max_svs, '-',  color=colors[label], label=f'{label} max', linewidth=1.5)
        ax.plot(epochs_x, min_svs, '--', color=colors[label], label=f'{label} min', linewidth=1.2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Singular value')
    ax.set_title('Min and Max SV over Training'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 7-9. W2 (middle layer) metrics for Iso-2L
    label = 'Iso-2L'
    r = all_results[label]
    if r['sv_W2']:
        ax = fig.add_subplot(n_rows, n_cols, 7)
        conds_W2 = [m['cond'] for m in r['sv_W2']]
        ax.semilogy(epochs_x, conds_W2, '-', color='red', label='W2 cond')
        conds_W1 = [m['cond'] for m in r['sv_W1']]
        ax.semilogy(epochs_x, conds_W1, '--', color='red', alpha=0.6, label='W1 cond')
        ax.set_title('Iso-2L: W1 vs W2 condition number')
        ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(n_rows, n_cols, 8)
        ent_W2 = [m['entropy'] for m in r['sv_W2']]
        ent_W1 = [m['entropy'] for m in r['sv_W1']]
        ax.plot(epochs_x, ent_W2, '-', color='red', label='W2 entropy')
        ax.plot(epochs_x, ent_W1, '--', color='red', alpha=0.6, label='W1 entropy')
        ax.set_title('Iso-2L: W1 vs W2 spectral entropy')
        ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(n_rows, n_cols, 9)
        sv_mat_W2 = np.array([m['svs'] for m in r['sv_W2']])
        im2 = ax.imshow(sv_mat_W2.T, aspect='auto', origin='lower',
                        extent=[0, EPOCHS, 0, WIDTH], cmap='plasma')
        plt.colorbar(im2, ax=ax)
        ax.set_xlabel('Epoch'); ax.set_ylabel('SV index')
        ax.set_title('Iso-2L: W2 (middle layer) SV spectrum')

    plt.suptitle('SV Spectrum Phase Transitions During Training', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sv_evolution.png'), dpi=150)
    print("\nPlot saved to results/test_S/sv_evolution.png")

    # =========================================================================
    # Save results.md
    # =========================================================================
    def sv_table(label):
        r = all_results[label]
        rows = []
        for ep in [0, 1, 2, 4, 8, 12, 16, 20, 24]:
            if ep < len(r['sv_W1']):
                m   = r['sv_W1'][ep]
                acc = r['accs'][ep]
                rows.append(
                    f"| {ep} | {acc:.3f} | {m['cond']:.1f} | {m['entropy']:.4f} | "
                    f"{m['top3_frac']:.3f} | {m['min_sv']:.2f} | {m['max_sv']:.2f} |")
        return '\n'.join(rows)

    results_text = f"""# Test S -- SV Spectrum Phase Transitions During Training

## Setup
- Width: {WIDTH}, Epochs: {EPOCHS}, lr={LR}, batch={BATCH}, seed={SEED}
- Device: CPU

## Iso-1L: W1 SV Metrics at Key Epochs

| Epoch | Acc | Cond | Entropy | Top-3 frac | Min SV | Max SV |
|---|---|---|---|---|---|---|
{sv_table('Iso-1L')}

## Iso-2L: W1 SV Metrics at Key Epochs

| Epoch | Acc | Cond | Entropy | Top-3 frac | Min SV | Max SV |
|---|---|---|---|---|---|---|
{sv_table('Iso-2L')}

## Key Findings

### Iso-1L
- Init condition number: {all_results['Iso-1L']['sv_W1'][0]['cond']:.1f}
- Final condition number: {all_results['Iso-1L']['sv_W1'][-1]['cond']:.1f}
- Init spectral entropy: {all_results['Iso-1L']['sv_W1'][0]['entropy']:.4f}
- Final spectral entropy: {all_results['Iso-1L']['sv_W1'][-1]['entropy']:.4f}
- Top-3 SVs at end: {all_results['Iso-1L']['sv_W1'][-1]['top3_frac']*100:.1f}% of total mass

### Iso-2L (W1)
- Init condition number: {all_results['Iso-2L']['sv_W1'][0]['cond']:.1f}
- Final condition number: {all_results['Iso-2L']['sv_W1'][-1]['cond']:.1f}
- Final spectral entropy: {all_results['Iso-2L']['sv_W1'][-1]['entropy']:.4f}

### Implications for Pruning
A high condition number means a large spread in SVs — many neurons with small SVs
are prunable with low error. The epoch at which this spread emerges is the earliest
point at which principled pruning can work effectively. This informs Test V's
prune-timing experiment.

![SV spectrum evolution](sv_evolution.png)
"""
    with open(os.path.join(RESULTS_DIR, 'results.md'), 'w', encoding='utf-8') as f:
        f.write(results_text)
    print("Results saved to results/test_S/results.md")


if __name__ == '__main__':
    main()
