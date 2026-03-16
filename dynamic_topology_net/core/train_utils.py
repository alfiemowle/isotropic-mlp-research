"""
Training and evaluation utilities shared across all experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim


def train_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss, steps = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / steps


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns accuracy (0-1) over the full loader."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total


def train_model(model, train_loader, test_loader, epochs, lr, device,
                verbose=True, prefix=''):
    """
    Standard training loop. Returns list of (train_loss, test_acc) per epoch.
    Rebuilds optimizer each call so it works after structural changes.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history   = []

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        acc  = evaluate(model, test_loader, device)
        history.append((loss, acc))
        if verbose:
            print(f"{prefix}Epoch {epoch:3d}/{epochs}  loss={loss:.4f}  acc={acc:.2%}")

    return history


def make_optimizer(model, lr):
    """Create a fresh Adam optimizer for a model (needed after grow/prune)."""
    return optim.Adam(model.parameters(), lr=lr)
