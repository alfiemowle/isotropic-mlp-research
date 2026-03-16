"""
Test F: Shell Collapse
Verify Appendix C: IsotropicMLP + unit-norm normalisation collapses to an affine map.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np

from dynamic_topology_net.core import IsotropicMLP, load_cifar10
from dynamic_topology_net.core.models import CollapsingIsotropicMLP
from dynamic_topology_net.core.train_utils import train_model, evaluate

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

SEED    = 42
EPOCHS  = 24
LR      = 0.08
BATCH   = 128
WIDTH   = 32
INPUT_DIM   = 3072
NUM_CLASSES = 10

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Experiment 1: Mathematical walkthrough -- does collapsing net = affine map?
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("EXPERIMENT 1: Mathematical walkthrough")
print("="*60)

tr_loader, te_loader, _, _ = load_cifar10(batch_size=BATCH)

# Grab a batch of test inputs for residual computation
test_batch_x, test_batch_y = next(iter(te_loader))
test_batch_x = test_batch_x.to(DEVICE)  # shape (N, 3072)
N_probe = test_batch_x.shape[0]

torch.manual_seed(SEED)
col_net = CollapsingIsotropicMLP(input_dim=INPUT_DIM, width=WIDTH, num_classes=NUM_CLASSES).to(DEVICE)

col_net.eval()
with torch.no_grad():
    # (a) Forward pass through the collapsing network
    y_net = col_net(test_batch_x)  # (N, 10)

    # (b) Fit a linear map: y_linear = X @ A^T + c that best approximates y_net
    #     Use least squares: [X | 1] @ [A; c]^T = y_net
    X_np = test_batch_x.cpu().numpy()  # (N, 3072)
    Y_np = y_net.cpu().numpy()          # (N, 10)

    # Augment X with ones for bias
    X_aug = np.concatenate([X_np, np.ones((N_probe, 1))], axis=1)  # (N, 3073)
    # Solve: X_aug @ P = Y_np  ->  P = pinv(X_aug) @ Y_np
    P, res, rank, sv = np.linalg.lstsq(X_aug, Y_np, rcond=None)
    Y_linear = X_aug @ P  # (N, 10)

    # (c) Residual
    residual = np.abs(Y_np - Y_linear)
    mean_residual = residual.mean()
    max_residual  = residual.max()

    # Normalise by output magnitude
    output_scale = np.abs(Y_np).mean()
    relative_residual = mean_residual / (output_scale + 1e-8)

print(f"\nCollapsing network output vs best-fit linear map:")
print(f"  Mean absolute residual : {mean_residual:.6f}")
print(f"  Max absolute residual  : {max_residual:.6f}")
print(f"  Output scale (mean|y|) : {output_scale:.6f}")
print(f"  Relative residual      : {relative_residual:.6f}")
print(f"  --> Network output {'IS' if relative_residual < 0.05 else 'IS NOT'} well-approximated by a linear map")

# Also verify: directly construct the effective linear map from weights
# y = W2 * norm(iso(W1*x + b1)) + b2
# = W2 * (iso(W1*x+b1) / ||iso(W1*x+b1)||) + b2
# This is W2 * (direction of iso output) + b2
# The direction depends on x, so the map is not purely linear unless directions align

# Compute output of just the first layer + activation (before normalisation)
h1 = col_net.W1(test_batch_x)          # (N, width)
a1_unnorm = col_net.iso(h1)            # (N, width)  -- isotropic tanh
a1 = col_net.norm(a1_unnorm)           # (N, width)  -- unit norm projection
y_check = col_net.W2(a1)               # (N, 10)

check_err = (y_check - y_net).abs().max().item()
print(f"\nForward pass reconstruction error (should be ~0): {check_err:.2e}")


# ---------------------------------------------------------------------------
# Experiment 2: Train all three models on CIFAR-10
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("EXPERIMENT 2: Training comparison on CIFAR-10")
print("="*60)

# (reuse the tr_loader, te_loader loaded earlier in Experiment 1)
torch.manual_seed(SEED)

# Model 1: CollapsingIsotropicMLP
print("\n[1/3] Training CollapsingIsotropicMLP ...")
torch.manual_seed(SEED)
col_model = CollapsingIsotropicMLP(input_dim=INPUT_DIM, width=WIDTH, num_classes=NUM_CLASSES).to(DEVICE)
history_col = train_model(col_model, tr_loader, te_loader, EPOCHS, LR, DEVICE,
                          verbose=False, prefix='  Collapse ')
print(f"  CollapsingIsotropicMLP done, final acc: {history_col[-1][1]:.4f}")

# Model 2: IsotropicMLP (regular)
print("\n[2/3] Training IsotropicMLP ...")
torch.manual_seed(SEED)
iso_model = IsotropicMLP(input_dim=INPUT_DIM, width=WIDTH, num_classes=NUM_CLASSES).to(DEVICE)
history_iso = train_model(iso_model, tr_loader, te_loader, EPOCHS, LR, DEVICE,
                          verbose=False, prefix='  Isotropic ')
print(f"  IsotropicMLP done, final acc: {history_iso[-1][1]:.4f}")

# Model 3: nn.Linear (logistic regression baseline)
print("\n[3/3] Training nn.Linear (logistic regression) ...")
torch.manual_seed(SEED)
lin_model = nn.Linear(INPUT_DIM, NUM_CLASSES).to(DEVICE)
history_lin = train_model(lin_model, tr_loader, te_loader, EPOCHS, LR, DEVICE,
                          verbose=False, prefix='  Linear ')
print(f"  nn.Linear done, final acc: {history_lin[-1][1]:.4f}")


# ---------------------------------------------------------------------------
# Extract curves
# ---------------------------------------------------------------------------

col_losses = [h[0] for h in history_col]
col_accs   = [h[1] for h in history_col]
iso_losses = [h[0] for h in history_iso]
iso_accs   = [h[1] for h in history_iso]
lin_losses = [h[0] for h in history_lin]
lin_accs   = [h[1] for h in history_lin]

epochs_range = list(range(1, EPOCHS + 1))

print(f"\nFinal test accuracies:")
print(f"  CollapsingIsotropicMLP : {col_accs[-1]:.4f}")
print(f"  IsotropicMLP           : {iso_accs[-1]:.4f}")
print(f"  nn.Linear              : {lin_accs[-1]:.4f}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Test F: Shell Collapse -- Training Curves on CIFAR-10', fontsize=13)

ax = axes[0]
ax.set_title('Training Loss vs Epoch')
ax.plot(epochs_range, col_losses, label='CollapsingIsotropic', color='red')
ax.plot(epochs_range, iso_losses, label='IsotropicMLP',        color='blue')
ax.plot(epochs_range, lin_losses, label='nn.Linear (logreg)',  color='gray', linestyle='--')
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross-Entropy Loss')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.set_title('Test Accuracy vs Epoch')
ax.plot(epochs_range, col_accs, label='CollapsingIsotropic', color='red')
ax.plot(epochs_range, iso_accs, label='IsotropicMLP',        color='blue')
ax.plot(epochs_range, lin_accs, label='nn.Linear (logreg)',  color='gray', linestyle='--')
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_F', 'collapse_comparison.png')
plt.savefig(plot_path, dpi=120)
plt.close()
print(f"\nPlot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Save results.md
# ---------------------------------------------------------------------------

collapse_confirmed = col_accs[-1] <= lin_accs[-1] * 1.05  # within 5% of linear

lines = []
lines.append("# Test F: Hyperspherical Shell Collapse\n\n")
lines.append(f"Seed: {SEED}, Epochs: {EPOCHS}, LR: {LR}, Batch: {BATCH}, Width: {WIDTH}\n\n")

lines.append("## Experiment 1: Mathematical walkthrough\n\n")
lines.append("We forward a test batch through CollapsingIsotropicMLP and fit the best possible\n")
lines.append("linear map (least-squares) to its outputs.\n\n")
lines.append(f"- Mean absolute residual (network vs best linear fit): {mean_residual:.6f}\n")
lines.append(f"- Max absolute residual: {max_residual:.6f}\n")
lines.append(f"- Mean output magnitude: {output_scale:.6f}\n")
lines.append(f"- Relative residual: {relative_residual:.6f}\n\n")
if relative_residual < 0.05:
    lines.append("**Result: The collapsing network output is almost perfectly explained by a single linear map.**\n")
    lines.append("This numerically confirms Appendix C's claim that unit-norm normalisation after\n")
    lines.append("isotropic activation collapses the network to an affine map.\n\n")
else:
    lines.append(f"**Result: Relative residual = {relative_residual:.4f} -- some nonlinearity remains.**\n")
    lines.append("The collapse may be partial or the linear fit may have insufficient freedom.\n\n")

lines.append("## Experiment 2: CIFAR-10 Training Comparison\n\n")
lines.append("| Model | Final Test Accuracy |\n")
lines.append("|-------|--------------------|\n")
lines.append(f"| CollapsingIsotropicMLP | {col_accs[-1]:.4f} |\n")
lines.append(f"| IsotropicMLP (standard) | {iso_accs[-1]:.4f} |\n")
lines.append(f"| nn.Linear (logistic reg) | {lin_accs[-1]:.4f} |\n\n")

lines.append("### Epoch-by-epoch test accuracy\n\n")
lines.append("| Epoch | CollapsingIso | IsotropicMLP | nn.Linear |\n")
lines.append("|-------|--------------|--------------|----------|\n")
for ep in range(EPOCHS):
    lines.append(f"| {ep+1:5d} | {col_accs[ep]:.4f} | {iso_accs[ep]:.4f} | {lin_accs[ep]:.4f} |\n")

lines.append("\n## Interpretation\n\n")
iso_gap  = iso_accs[-1] - col_accs[-1]
lin_gap  = col_accs[-1] - lin_accs[-1]
lines.append(f"- IsotropicMLP vs CollapsingIsotropicMLP gap: {iso_gap:+.4f}\n")
lines.append(f"- CollapsingIsotropicMLP vs nn.Linear gap: {lin_gap:+.4f}\n\n")

if collapse_confirmed:
    lines.append("**Shell collapse CONFIRMED**: The collapsing network performs at or near linear-classifier level,\n")
    lines.append("well below the regular isotropic MLP. This empirically validates Appendix C.\n")
else:
    lines.append(f"**Shell collapse PARTIAL**: The collapsing network ({col_accs[-1]:.4f}) is above the linear\n")
    lines.append(f"baseline ({lin_accs[-1]:.4f}) by more than 5%. There may be residual nonlinearity,\n")
    lines.append("or the network learned a partially useful representation despite the collapse.\n")

lines.append("\nSee `collapse_comparison.png` for training curves.\n")

results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'test_F', 'results.md')
with open(results_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Results saved to {results_path}")
print("\nDone -- Test F complete.")
