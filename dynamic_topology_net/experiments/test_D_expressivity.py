"""
Test D: Expressivity -- Isotropic vs Standard Tanh MLP
Compare on dimension-gating tasks (XOR, Selective Gate) and CIFAR-10.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

matplotlib_import = True
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib_import = False

from dynamic_topology_net.core import IsotropicMLP, BaselineMLP, load_cifar10
from dynamic_topology_net.core.train_utils import train_model, evaluate

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

SEEDS = [42, 123, 7]
EPOCHS_SYNTHETIC = 100   # XOR and gate tasks (small inputs, fast)
EPOCHS_CIFAR     = 24    # CIFAR-10 (large input, use same as other tests)
LR = 0.08
BATCH = 64
WIDTH = 64


# ---------------------------------------------------------------------------
# Synthetic task generators
# ---------------------------------------------------------------------------

def make_xor_task(n_train=5000, n_test=1000, input_dim=100, seed=0):
    """Binary classification: label = sign(x[0]) XOR sign(x[1])."""
    rng = np.random.default_rng(seed)
    def _gen(n):
        X = rng.standard_normal((n, input_dim)).astype(np.float32)
        s0 = (X[:, 0] > 0).astype(int)
        s1 = (X[:, 1] > 0).astype(int)
        y = (s0 ^ s1).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(y)
    X_tr, y_tr = _gen(n_train)
    X_te, y_te = _gen(n_test)
    tr_ds = TensorDataset(X_tr, y_tr)
    te_ds = TensorDataset(X_te, y_te)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=1000)
    return tr_loader, te_loader, input_dim, 2


def make_selective_gate_task(n_train=5000, n_test=1000, input_dim=200, n_groups=4, seed=0):
    """
    4-class task: label = argmax k where group k is active.
    Group k active if x[k*50:(k+1)*50].mean() > 0.
    Exactly one group is active per sample by construction.
    """
    group_size = input_dim // n_groups
    rng = np.random.default_rng(seed)
    def _gen(n):
        X = rng.standard_normal((n, input_dim)).astype(np.float32) * 0.3
        y = rng.integers(0, n_groups, size=n).astype(np.int64)
        for i in range(n):
            k = y[i]
            # Make group k mean strongly positive, others strongly negative
            X[i, k*group_size:(k+1)*group_size] += 1.5
            for j in range(n_groups):
                if j != k:
                    X[i, j*group_size:(j+1)*group_size] -= 1.5
        return torch.from_numpy(X), torch.from_numpy(y)
    X_tr, y_tr = _gen(n_train)
    X_te, y_te = _gen(n_test)
    tr_ds = TensorDataset(X_tr, y_tr)
    te_ds = TensorDataset(X_te, y_te)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=1000)
    return tr_loader, te_loader, input_dim, n_groups


# ---------------------------------------------------------------------------
# Run one task across seeds
# ---------------------------------------------------------------------------

def run_task(task_name, make_loader_fn, loader_kwargs):
    iso_accs = []
    base_accs = []

    epochs = EPOCHS_CIFAR if task_name == 'cifar10' else EPOCHS_SYNTHETIC

    # Load CIFAR-10 once if needed (avoid repeated reloads)
    cifar_loaders = None
    if task_name == 'cifar10':
        print("  Loading CIFAR-10...")
        cifar_loaders = load_cifar10(batch_size=BATCH)
        print("  CIFAR-10 loaded.")

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        if task_name == 'cifar10':
            tr_loader, te_loader, input_dim, num_classes = cifar_loaders
        else:
            tr_loader, te_loader, input_dim, num_classes = make_loader_fn(seed=seed, **loader_kwargs)

        print(f"\n  [{task_name}] seed={seed}, input_dim={input_dim}, num_classes={num_classes}, epochs={epochs}")

        # Isotropic
        model_iso = IsotropicMLP(input_dim=input_dim, width=WIDTH, num_classes=num_classes).to(DEVICE)
        history_iso = train_model(model_iso, tr_loader, te_loader, epochs, LR, DEVICE,
                                  verbose=False, prefix=f'    Iso  ')
        acc_iso = history_iso[-1][1]
        iso_accs.append(acc_iso)
        print(f"    Iso  final acc: {acc_iso:.4f}")

        # Baseline
        model_base = BaselineMLP(input_dim=input_dim, width=WIDTH, num_classes=num_classes).to(DEVICE)
        history_base = train_model(model_base, tr_loader, te_loader, epochs, LR, DEVICE,
                                   verbose=False, prefix=f'    Base ')
        acc_base = history_base[-1][1]
        base_accs.append(acc_base)
        print(f"    Base final acc: {acc_base:.4f}")

    iso_mean  = np.mean(iso_accs)
    iso_std   = np.std(iso_accs)
    base_mean = np.mean(base_accs)
    base_std  = np.std(base_accs)
    gap = iso_mean - base_mean

    print(f"\n  [{task_name}] SUMMARY:")
    print(f"    Isotropic : {iso_mean:.4f} +/- {iso_std:.4f}")
    print(f"    Baseline  : {base_mean:.4f} +/- {base_std:.4f}")
    print(f"    Gap (Iso-Base): {gap:+.4f}")

    return {
        'iso_accs': iso_accs,
        'base_accs': base_accs,
        'iso_mean': iso_mean,
        'iso_std': iso_std,
        'base_mean': base_mean,
        'base_std': base_std,
        'gap': gap,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("=" * 60)
print("TEST D: Expressivity Comparison")
print("=" * 60)

results = {}

print("\n[1/3] XOR Task (100-dim input, 2 classes)")
results['xor'] = run_task('xor', make_xor_task, {'n_train': 5000, 'n_test': 1000, 'input_dim': 100})

print("\n[2/3] Selective Gate Task (200-dim input, 4 classes)")
results['gate'] = run_task('selective_gate', make_selective_gate_task,
                           {'n_train': 5000, 'n_test': 1000, 'input_dim': 200, 'n_groups': 4})

print("\n[3/3] CIFAR-10")
results['cifar10'] = run_task('cifar10', None, {})


# ---------------------------------------------------------------------------
# Save results.md
# ---------------------------------------------------------------------------

def significance(gap, std_iso, std_base, n=3):
    """Very rough significance check: gap > 2 * pooled SE."""
    se = np.sqrt((std_iso**2 + std_base**2) / n)
    return "|significant|" if abs(gap) > 2 * se else "not significant"


out_lines = []
out_lines.append("# Test D: Expressivity -- Isotropic vs Standard Tanh MLP\n")
out_lines.append(f"Seeds: {SEEDS}, Epochs (synthetic): {EPOCHS_SYNTHETIC}, Epochs (CIFAR-10): {EPOCHS_CIFAR}, LR: {LR}, Batch: {BATCH}, Width: {WIDTH}\n\n")

task_display = [
    ('xor',    'XOR (100-dim, 2 classes)'),
    ('gate',   'Selective Gate (200-dim, 4 classes)'),
    ('cifar10','CIFAR-10 (3072-dim, 10 classes)'),
]

out_lines.append("## Results Table\n\n")
out_lines.append("| Task | Isotropic (mean+/-std) | Baseline (mean+/-std) | Gap (Iso-Base) | Significance |\n")
out_lines.append("|------|------------------------|----------------------|-----------------|---------------|\n")
for key, label in task_display:
    r = results[key]
    sig = significance(r['gap'], r['iso_std'], r['base_std'])
    out_lines.append(f"| {label} | {r['iso_mean']:.4f} +/- {r['iso_std']:.4f} "
                     f"| {r['base_mean']:.4f} +/- {r['base_std']:.4f} "
                     f"| {r['gap']:+.4f} | {sig} |\n")

out_lines.append("\n## Per-seed Raw Accuracies\n\n")
for key, label in task_display:
    r = results[key]
    out_lines.append(f"### {label}\n")
    for i, s in enumerate(SEEDS):
        out_lines.append(f"- Seed {s}: Iso={r['iso_accs'][i]:.4f}, Base={r['base_accs'][i]:.4f}\n")
    out_lines.append("\n")

out_lines.append("## Interpretation\n\n")
for key, label in task_display:
    r = results[key]
    sig = significance(r['gap'], r['iso_std'], r['base_std'])
    if r['gap'] > 0.01:
        interp = "Isotropic MLP outperforms baseline -- suggests the isotropic activation benefits this task."
    elif r['gap'] < -0.01:
        interp = "Baseline MLP outperforms isotropic -- the anisotropic activation has an edge here."
    else:
        interp = "Results are roughly equivalent -- no meaningful gap between the two architectures."
    out_lines.append(f"- **{label}**: Gap={r['gap']:+.4f} ({sig}). {interp}\n")

out_lines.append("\n## Notes\n\n")
out_lines.append("- The XOR task requires gating on specific input coordinates; standard tanh can individuate neurons per-dimension.\n")
out_lines.append("- Isotropic tanh treats the full pre-activation vector as a single entity; it is basis-independent.\n")
out_lines.append("- If isotropic matches or beats baseline on XOR/gate tasks, it demonstrates sufficient expressivity despite the O(n)-equivariance constraint.\n")

results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_D', 'results.md')
with open(results_path, 'w', encoding='utf-8') as f:
    f.writelines(out_lines)

print(f"\nResults saved to {results_path}")
print("\nDone -- Test D complete.")
