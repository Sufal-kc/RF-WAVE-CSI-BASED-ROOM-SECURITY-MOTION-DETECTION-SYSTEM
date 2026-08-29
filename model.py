"""
CNN architecture for CSI -> pose.

Input tensor per example: (4, T, F)
    4 channels = amp_mean, amp_std, phase_mean, phase_circvar
    T          = history window count (time axis)
    F          = subcarrier count (frequency axis)

Treated as a small time-frequency "image", same idea used in RF-based
pose-estimation literature (RF-Pose / Person-in-WiFi-style encoders): a
2D CNN over (time, subcarrier) with the calibration channels stacked as
input depth.
"""

import torch
import torch.nn as nn

from common import NUM_JOINTS


def conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CNNEncoder(nn.Module):
    """(N, 4, T, F) -> (N, embed_dim)"""

    def __init__(self, in_channels=4, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2),          # T/2, F/2
            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),          # T/4, F/4
            conv_block(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.proj(h)


class ConvDecoder(nn.Module):
    """Mirror of CNNEncoder, used only during self-supervised pretraining
    to reconstruct the input window from its embedding. Output spatial
    size is fixed via the target (T, F) passed at construction time."""

    def __init__(self, embed_dim=128, out_channels=4, out_size=(8, 52)):
        super().__init__()
        self.out_size = out_size
        self.fc = nn.Linear(embed_dim, 128)
        self.net = nn.Sequential(
            nn.Upsample(size=(max(1, out_size[0] // 4), max(1, out_size[1] // 4)), mode="nearest"),
            conv_block(128, 64),
            nn.Upsample(size=(max(1, out_size[0] // 2), max(1, out_size[1] // 2)), mode="nearest"),
            conv_block(64, 32),
            nn.Upsample(size=out_size, mode="nearest"),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z):
        h = self.fc(z).unsqueeze(-1).unsqueeze(-1)  # (N, 128, 1, 1)
        return self.net(h)


class CSIAutoencoder(nn.Module):
    """Used only for stage-1 self-supervised pretraining on the large
    unlabeled CSI pool: learn a representation by reconstructing CSI
    windows, no pose labels involved at all."""

    def __init__(self, in_channels=4, embed_dim=128, window_shape=(8, 52)):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, embed_dim)
        self.decoder = ConvDecoder(embed_dim, in_channels, out_size=window_shape)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class PoseHead(nn.Module):
    def __init__(self, embed_dim=128, num_joints=NUM_JOINTS, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_joints * 2),
        )

    def forward(self, z):
        return self.net(z)


class CSIPoseNet(nn.Module):
    """Full model: CNN encoder (optionally loaded from autoencoder
    pretraining) + pose regression head."""

    def __init__(self, in_channels=4, embed_dim=128, num_joints=NUM_JOINTS):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, embed_dim)
        self.head = PoseHead(embed_dim, num_joints)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z).view(x.shape[0], -1)  # (N, num_joints*2)

    def load_pretrained_encoder(self, autoencoder: CSIAutoencoder):
        self.encoder.load_state_dict(autoencoder.encoder.state_dict())
