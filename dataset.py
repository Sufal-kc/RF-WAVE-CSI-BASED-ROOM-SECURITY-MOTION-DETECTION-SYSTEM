"""Torch Dataset wrappers around the .npz files produced by preprocess.py."""

import numpy as np
import torch
from torch.utils.data import Dataset


def _per_channel_normalize(X, stats=None):
    """Normalize each of the 4 feature channels to zero mean / unit std,
    computed over (N, T, F). Returns (X_norm, stats) so the same stats can
    be reused on val/test/inference data."""
    if stats is None:
        mean = X.mean(axis=(0, 1, 2), keepdims=True)
        std = X.std(axis=(0, 1, 2), keepdims=True) + 1e-6
        stats = (mean, std)
    mean, std = stats
    return (X - mean) / std, stats


class CSIWindowDataset(Dataset):
    """Unlabeled CSI windows, for self-supervised (autoencoder) pretraining.
    Returns a single tensor shaped (C=4, T, F) -- channel-first for conv2d."""

    def __init__(self, npz_path, stats=None):
        data = np.load(npz_path)
        X = data["X"]  # (N, T, F, 4)
        X, self.stats = _per_channel_normalize(X, stats)
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2).float()  # (N, 4, T, F)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class CSIPoseDataset(Dataset):
    """Labeled (CSI window, pose keypoints, per-joint confidence) triples."""

    def __init__(self, npz_path, stats=None):
        data = np.load(npz_path)
        X, self.stats = _per_channel_normalize(data["X"], stats)
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2).float()  # (N, 4, T, F)
        self.Y = torch.from_numpy(data["Y"]).float()               # (N, NUM_JOINTS*2)
        self.conf = torch.from_numpy(data["conf"]).float()         # (N, NUM_JOINTS)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.conf[idx]
