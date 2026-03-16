import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
M          = 24          # hidden layer width
LR         = 0.08
BATCH_SIZE = 24
EPOCHS     = 24
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'baseline_mlp.pt')
PLOT_PATH  = os.path.join(os.path.dirname(__file__), 'results', 'accuracy.png')

# ── Model ────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim=3072, hidden=M, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# ── Data ─────────────────────────────────────────────────────────────────────
def load_data():
    # Download raw datasets (no normalisation yet — we compute stats ourselves)
    raw_transform = transforms.ToTensor()

    train_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True,  download=True, transform=raw_transform)
    test_set  = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=raw_transform)

    # Stack all training images to compute elementwise mean and std
    # Shape: (50000, 3, 32, 32) → (50000, 3072)
    train_loader_full = DataLoader(train_set, batch_size=1000, shuffle=False)
    all_x = torch.cat([x.view(x.size(0), -1) for x, _ in train_loader_full], dim=0)

    mu    = all_x.mean(dim=0)   # (3072,)
    sigma = all_x.std(dim=0)    # (3072,)
    # Avoid division by zero for constant pixels
    sigma = sigma.clamp(min=1e-8)

    print(f"Input stats  — mean: {mu.mean():.4f}, std: {sigma.mean():.4f}")

    def to_tensor_dataset(dataset):
        loader = DataLoader(dataset, batch_size=1000, shuffle=False)
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.view(x.size(0), -1))
            ys.append(y)
        X = torch.cat(xs, dim=0)
        Y = torch.cat(ys, dim=0)
        X = (X - mu) / sigma   # elementwise standardisation
        return TensorDataset(X, Y)

    train_ds = to_tensor_dataset(train_set)
    test_ds  = to_tensor_dataset(test_set)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=1000,       shuffle=False)

    return train_loader, test_loader

# ── Evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total

# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, test_loader = load_data()

    model     = MLP(hidden=M).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"\nArchitecture: [3072, {M}, 10]  |  lr={LR}  |  batch={BATCH_SIZE}  |  epochs={EPOCHS}\n")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Test Acc':>10}")
    print("-" * 34)

    test_accs = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps += 1

        avg_loss = running_loss / steps
        acc      = evaluate(model, test_loader, device)
        test_accs.append(acc)

        print(f"{epoch:>6}  {avg_loss:>12.4f}  {acc:>9.2%}")

    # ── Save model ────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved  → {MODEL_PATH}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, EPOCHS + 1), [a * 100 for a in test_accs], marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'CIFAR-10 MLP [3072, {M}, 10] — Test Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Plot saved   → {PLOT_PATH}")

    print(f"\nFinal test accuracy: {test_accs[-1]:.2%}")

if __name__ == '__main__':
    train()
