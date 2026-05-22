from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class EEGSpatialAttention(nn.Module):
    """Spatial attention over EEG band/channel-time maps."""

    def __init__(self, in_channels: int = 5, reduction: int = 2):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.value = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.query = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.proj = nn.Conv2d(hidden, in_channels, kernel_size=1)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.value(x)
        q = self.query(x).mean(dim=(2, 3), keepdim=True)
        weights = torch.softmax(q, dim=1)
        attn = torch.sigmoid(self.proj(self.act(v * weights)))
        return x * attn


class EEGSTAEBlock(nn.Module):
    """EEG-only STAE block used before the Swin backbone.

    The paper specifies modality-respecting STAE and reports a 4-layer tradeoff,
    but does not publish all implementation details. This block keeps the
    paper's spatial-attention intent and figure-level 7x7 local mixing.
    """

    def __init__(self, bands: int = 5, dropout: float = 0.1):
        super().__init__()
        self.local = nn.Conv2d(bands, bands, kernel_size=7, padding=3, groups=bands)
        self.attn = EEGSpatialAttention(bands)
        self.mix = nn.Sequential(
            nn.Conv2d(bands, bands * 2, kernel_size=1),
            nn.PReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(bands * 2, bands, kernel_size=1),
        )
        self.norm = nn.BatchNorm2d(bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.local(x)
        z = self.attn(z)
        z = self.mix(z)
        return self.norm(x + z)


class MicroTemporalAttention(nn.Module):
    """Attention for five-frame 56x56 face tensors; retained for future multimodal runs."""

    def __init__(self, frames: int = 5, reduction: int = 2):
        super().__init__()
        hidden = max(1, frames // reduction)
        self.value = nn.Conv2d(frames, hidden, kernel_size=1)
        self.query = nn.Conv2d(frames, 1, kernel_size=1)
        self.proj = nn.Conv2d(hidden, frames, kernel_size=1)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.value(x)
        q = torch.softmax(self.query(x).flatten(2), dim=-1).view(x.size(0), 1, x.size(2), x.size(3))
        attn = torch.sigmoid(self.proj(self.act(v * q)))
        return x * attn


class CompactEEGEncoder(nn.Module):
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


class CompactSTSTEmotionModel(nn.Module):
    """Original lightweight baseline retained as --model compact."""

    def __init__(self, num_classes: int = 3, hidden: int = 64, use_me: bool = False):
        super().__init__()
        self.use_me = use_me
        self.eeg_encoder = CompactEEGEncoder(hidden=hidden)
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


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        num_relative = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative, num_heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_windows, n, c = x.shape
        qkv = self.qkv(x).reshape(b_windows, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(n, n, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(b_windows // num_windows, num_windows, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_windows, n, c)
        return self.proj_drop(self.proj(x))


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = min(window_size, input_resolution[0], input_resolution[1])
        self.shift_size = 0 if min(input_resolution) <= window_size else shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.window_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout)

        if self.shift_size > 0:
            self.register_buffer("attn_mask", self._make_mask(input_resolution), persistent=False)
        else:
            self.attn_mask = None

    def _make_mask(self, resolution: tuple[int, int]) -> torch.Tensor:
        h, w = resolution
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(x_windows, self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 5, embed_dim: int = 64, patch_size: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return self.norm(x)


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        if h % 2 == 1 or w % 2 == 1:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            h, w = x.shape[1], x.shape[2]
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        return self.reduction(self.norm(x))


class SwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        blocks = []
        for i in range(depth):
            blocks.append(
                SwinBlock(
                    dim=dim,
                    input_resolution=resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


@dataclass(frozen=True)
class SwinConfig:
    depths: tuple[int, int, int] = (2, 2, 2)
    heads: tuple[int, int, int] = (2, 4, 8)
    window_size: int = 7
    patch_size: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    stae_layers: int = 4


class SwinSTSTEmotionModel(nn.Module):
    """EEG-only STAE + Swin/STST implementation for 56x56x5 DE maps."""

    def __init__(self, num_classes: int = 3, hidden: int = 64, config: SwinConfig | None = None):
        super().__init__()
        self.config = config or SwinConfig()
        self.stae = nn.Sequential(*[EEGSTAEBlock(5, self.config.dropout) for _ in range(self.config.stae_layers)])
        self.patch_embed = PatchEmbed(5, hidden, self.config.patch_size)

        resolution = 56 // self.config.patch_size
        dims = [hidden, hidden * 2, hidden * 4]
        resolutions = [(resolution, resolution), (resolution // 2, resolution // 2), (resolution // 4, resolution // 4)]
        self.stage1 = SwinStage(dims[0], resolutions[0], self.config.depths[0], self.config.heads[0], self.config.window_size, self.config.mlp_ratio, self.config.dropout)
        self.merge1 = PatchMerging(dims[0])
        self.stage2 = SwinStage(dims[1], resolutions[1], self.config.depths[1], self.config.heads[1], self.config.window_size, self.config.mlp_ratio, self.config.dropout)
        self.merge2 = PatchMerging(dims[1])
        self.stage3 = SwinStage(dims[2], resolutions[2], self.config.depths[2], self.config.heads[2], self.config.window_size, self.config.mlp_ratio, self.config.dropout)
        self.norm = nn.LayerNorm(dims[2])
        self.head = nn.Sequential(
            nn.Linear(dims[2], dims[2]),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(dims[2], num_classes),
        )

    def forward(self, eeg: torch.Tensor, me: torch.Tensor | None = None) -> torch.Tensor:
        if eeg.ndim != 4:
            raise ValueError(f"SwinSTSTEmotionModel expects Bx5x56x56 EEG maps, got {tuple(eeg.shape)}")
        x = self.stae(eeg)
        x = self.patch_embed(x)
        x = self.stage1(x)
        x = self.merge1(x)
        x = self.stage2(x)
        x = self.merge2(x)
        x = self.stage3(x)
        x = self.norm(x).mean(dim=(1, 2))
        return self.head(x)


def build_model(
    model_name: str,
    num_classes: int = 3,
    hidden: int = 64,
    use_me: bool = False,
    stae_layers: int = 4,
    swin_window_size: int = 7,
    swin_depths: tuple[int, int, int] = (2, 2, 2),
    swin_heads: tuple[int, int, int] = (2, 4, 8),
    swin_mlp_ratio: float = 4.0,
    dropout: float = 0.1,
) -> nn.Module:
    if model_name == "compact":
        return CompactSTSTEmotionModel(num_classes=num_classes, hidden=hidden, use_me=use_me)
    if model_name == "stst_swin":
        config = SwinConfig(
            depths=swin_depths,
            heads=swin_heads,
            window_size=swin_window_size,
            mlp_ratio=swin_mlp_ratio,
            dropout=dropout,
            stae_layers=stae_layers,
        )
        return SwinSTSTEmotionModel(num_classes=num_classes, hidden=hidden, config=config)
    raise ValueError(f"Unknown model: {model_name}")


# Backwards-compatible name used by older scripts.
STSTEmotionModel = CompactSTSTEmotionModel
