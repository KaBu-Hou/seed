from __future__ import annotations

import torch
from torch import nn


class EEGSpatialAttention(nn.Module):
    """Spatial attention over the channel-band EEG map."""

    def __init__(self, in_channels: int = 5, reduction: int = 2):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.value = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.query = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.proj = nn.Conv2d(hidden, in_channels, kernel_size=1)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch x bands x channels x time
        v = self.value(x)
        q = self.query(x).mean(dim=(2, 3), keepdim=True)
        weights = torch.softmax(q, dim=1)
        attn = torch.sigmoid(self.proj(self.act(v * weights)))
        return x * attn


class MicroTemporalAttention(nn.Module):
    """Attention for five-frame 56x56 face tensors."""

    def __init__(self, frames: int = 5, reduction: int = 2):
        super().__init__()
        hidden = max(1, frames // reduction)
        self.value = nn.Conv2d(frames, hidden, kernel_size=1)
        self.query = nn.Conv2d(frames, 1, kernel_size=1)
        self.proj = nn.Conv2d(hidden, frames, kernel_size=1)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch x frames x height x width
        v = self.value(x)
        q = torch.softmax(self.query(x).flatten(2), dim=-1).view(x.size(0), 1, x.size(2), x.size(3))
        attn = torch.sigmoid(self.proj(self.act(v * q)))
        return x * attn


class EEGEncoder(nn.Module):
    def __init__(self, bands: int = 5, hidden: int = 64):
        super().__init__()
        self.attn = EEGSpatialAttention(bands)
        self.net = nn.Sequential(
            nn.Conv2d(bands, 32, kernel_size=(7, 5), padding=(3, 2)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(2)
        # Dataset stores batch x channels x time x bands.
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.net(self.attn(x))


class MicroExpressionEncoder(nn.Module):
    def __init__(self, frames: int = 5, hidden: int = 64):
        super().__init__()
        self.attn = MicroTemporalAttention(frames)
        self.net = nn.Sequential(
            nn.Conv2d(frames, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.attn(x))


class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.PReLU())

    def forward(self, eeg: torch.Tensor, me: torch.Tensor | None = None) -> torch.Tensor:
        if me is None:
            return eeg
        joint = torch.cat([eeg, me], dim=-1)
        gate = self.gate(joint)
        mixed = self.proj(joint)
        return gate * mixed + (1.0 - gate) * eeg


class STSTEmotionModel(nn.Module):
    """Compact STAE/STST-style model for SEED reproduction.

    The original paper uses STAE followed by Swin Transformer blocks. To keep
    this reproduction runnable without timm, this implementation keeps the
    modality-respecting attention and gated fusion, then uses a lightweight MLP
    head. It accepts EEG-only input when participant face videos are missing.
    """

    def __init__(self, num_classes: int = 3, hidden: int = 64, use_me: bool = False):
        super().__init__()
        self.use_me = use_me
        self.eeg_encoder = EEGEncoder(hidden=hidden)
        self.me_encoder = MicroExpressionEncoder(hidden=hidden) if use_me else None
        self.fusion = GatedFusion(hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, eeg: torch.Tensor, me: torch.Tensor | None = None) -> torch.Tensor:
        eeg_feat = self.eeg_encoder(eeg)
        me_feat = self.me_encoder(me) if self.use_me and me is not None else None
        return self.head(self.fusion(eeg_feat, me_feat))
