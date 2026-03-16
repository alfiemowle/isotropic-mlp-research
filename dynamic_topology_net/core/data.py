"""
Data loading utilities — all datasets return normalised TensorDatasets.

All loaders return: (train_loader, test_loader, input_dim, num_classes)

Normalisation is computed from the training set (per-pixel mean/std for
CIFAR-10/SVHN, per-channel for MNIST-style datasets). This matches the
paper's protocol exactly.
"""

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _to_flat_tensor_dataset(dataset, mu, sigma, batch_size=1000):
    """Flatten images, normalise, return TensorDataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.view(x.size(0), -1))
        ys.append(y)
    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    X = (X - mu) / sigma
    return TensorDataset(X, Y)


def load_cifar10(batch_size=24, test_batch_size=1000):
    """
    CIFAR-10: 50k train / 10k test, 32x32 RGB -> 3072-dim flat vectors.
    Normalised per-pixel (elementwise mean/std across training set).
    """
    raw = transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=raw)
    test_set  = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=raw)

    # Compute training set statistics
    full_loader = DataLoader(train_set, batch_size=1000, shuffle=False)
    all_x = torch.cat([x.view(x.size(0), -1) for x, _ in full_loader], dim=0)
    mu    = all_x.mean(dim=0)
    sigma = all_x.std(dim=0).clamp(min=1e-8)

    train_ds = _to_flat_tensor_dataset(train_set, mu, sigma)
    test_ds  = _to_flat_tensor_dataset(test_set,  mu, sigma)

    train_loader = DataLoader(train_ds, batch_size=batch_size,      shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, 3072, 10


def load_mnist(batch_size=24, test_batch_size=1000):
    """
    MNIST: 60k train / 10k test, 28x28 greyscale -> 784-dim flat vectors.
    """
    raw = transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(DATA_DIR, train=True,  download=True, transform=raw)
    test_set  = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=raw)

    full_loader = DataLoader(train_set, batch_size=1000, shuffle=False)
    all_x = torch.cat([x.view(x.size(0), -1) for x, _ in full_loader], dim=0)
    mu    = all_x.mean(dim=0)
    sigma = all_x.std(dim=0).clamp(min=1e-8)

    train_ds = _to_flat_tensor_dataset(train_set, mu, sigma)
    test_ds  = _to_flat_tensor_dataset(test_set,  mu, sigma)

    train_loader = DataLoader(train_ds, batch_size=batch_size,      shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, 784, 10


def load_fashion_mnist(batch_size=24, test_batch_size=1000):
    """
    Fashion-MNIST: 60k train / 10k test, 28x28 greyscale -> 784-dim flat vectors.
    """
    raw = transforms.ToTensor()
    train_set = torchvision.datasets.FashionMNIST(DATA_DIR, train=True,  download=True, transform=raw)
    test_set  = torchvision.datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=raw)

    full_loader = DataLoader(train_set, batch_size=1000, shuffle=False)
    all_x = torch.cat([x.view(x.size(0), -1) for x, _ in full_loader], dim=0)
    mu    = all_x.mean(dim=0)
    sigma = all_x.std(dim=0).clamp(min=1e-8)

    train_ds = _to_flat_tensor_dataset(train_set, mu, sigma)
    test_ds  = _to_flat_tensor_dataset(test_set,  mu, sigma)

    train_loader = DataLoader(train_ds, batch_size=batch_size,      shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, 784, 10


def load_svhn(batch_size=24, test_batch_size=1000):
    """
    SVHN: ~73k train / 26k test, 32x32 RGB -> 3072-dim flat vectors.
    """
    raw = transforms.ToTensor()
    train_set = torchvision.datasets.SVHN(DATA_DIR, split='train', download=True, transform=raw)
    test_set  = torchvision.datasets.SVHN(DATA_DIR, split='test',  download=True, transform=raw)

    full_loader = DataLoader(train_set, batch_size=1000, shuffle=False)
    all_x = torch.cat([x.view(x.size(0), -1) for x, _ in full_loader], dim=0)
    mu    = all_x.mean(dim=0)
    sigma = all_x.std(dim=0).clamp(min=1e-8)

    train_ds = _to_flat_tensor_dataset(train_set, mu, sigma)
    test_ds  = _to_flat_tensor_dataset(test_set,  mu, sigma)

    train_loader = DataLoader(train_ds, batch_size=batch_size,      shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, 3072, 10
