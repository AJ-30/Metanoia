"""Conditional U-Net denoiser.

Input channels = [noisy target x_t (C), incomplete plan I (C), optional x/y
coordinate maps (2)]. The coordinates give convolutions absolute position,
which helps decide *where* something is missing.

Output: v-prediction of the missing part (C channels) and logits of the
missing component type from an auxiliary head on the bottleneck.

Building blocks: residual blocks with timestep-conditioned GroupNorm
(scale/shift), self-attention at selected resolutions, strided-conv
downsampling and nearest-neighbour upsampling with skip connections.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps (Transformer positional encoding).

    Args:
        t (torch.Tensor): (B,) timesteps.
        dim (int): Embedding size (even).
        max_period (float): Longest sinusoid period.

    Returns:
        torch.Tensor: (B, dim) embedding.
    """
    half: int = dim // 2
    freqs: torch.Tensor = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args: torch.Tensor = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


def _zero(module: nn.Module) -> nn.Module:
    """Zero-initialize a module's parameters (residual branches start as identity).

    Args:
        module (nn.Module): Module.

    Returns:
        nn.Module: The same module.
    """
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv residual block with timestep scale/shift."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int, dropout: float) -> None:
        """Build the block.

        Args:
            in_ch (int): Input channels.
            out_ch (int): Output channels.
            emb_dim (int): Timestep embedding size.
            groups (int): GroupNorm groups.
            dropout (float): Dropout before the second convolution.
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = _zero(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x (torch.Tensor): (B, in_ch, H, W) features.
            emb (torch.Tensor): (B, emb_dim) timestep embedding.

        Returns:
            torch.Tensor: (B, out_ch, H, W) features.
        """
        h: torch.Tensor = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    """Multi-head self-attention over all spatial positions."""

    def __init__(self, channels: int, heads: int, groups: int) -> None:
        """Build the block.

        Args:
            channels (int): Feature channels (divisible by ``heads``).
            heads (int): Attention heads.
            groups (int): GroupNorm groups.
        """
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"
        self.heads: int = heads
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = _zero(nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention with a residual connection.

        Args:
            x (torch.Tensor): (B, C, H, W) features.

        Returns:
            torch.Tensor: (B, C, H, W) features.
        """
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(dim=1)
        # (B, heads, tokens, head_dim) layout for scaled_dot_product_attention.
        out: torch.Tensor = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class ConditionalUNet(nn.Module):
    """U-Net predicting v (and the missing type) from (x_t, incomplete plan, t)."""

    def __init__(self, num_channels: int, image_size: int, cfg: SimpleNamespace) -> None:
        """Build the network.

        Args:
            num_channels (int): Semantic channels C of target and condition.
            image_size (int): Canvas size (square).
            cfg (SimpleNamespace): ``[model]`` hyperparameters.
        """
        super().__init__()
        base: int = cfg.BASE_CHANNELS
        emb_dim: int = 4 * base
        groups: int = cfg.GROUP_NORM_GROUPS
        self.base: int = base
        self.use_coordinates: bool = cfg.USE_COORDINATES
        if self.use_coordinates:
            axis: torch.Tensor = torch.linspace(-1.0, 1.0, image_size)
            self.register_buffer("coords", torch.stack(torch.meshgrid(axis, axis, indexing="ij"))[None], persistent=False)
        in_ch: int = 2 * num_channels + (2 if self.use_coordinates else 0)

        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        block = lambda i, o: ResBlock(i, o, emb_dim, groups, cfg.DROPOUT)

        # Encoder: remember the channel count of every skip connection.
        self.down = nn.ModuleList()
        skips: list[int] = [base]
        ch: int = base
        resolution: int = image_size
        for level, mult in enumerate(cfg.CHANNEL_MULTS):
            for _ in range(cfg.NUM_RES_BLOCKS):
                layers: list[nn.Module] = [block(ch, base * mult)]
                ch = base * mult
                if resolution in cfg.ATTENTION_RESOLUTIONS:
                    layers.append(AttentionBlock(ch, cfg.NUM_HEADS, groups))
                self.down.append(nn.ModuleList(layers))
                skips.append(ch)
            if level < len(cfg.CHANNEL_MULTS) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                skips.append(ch)
                resolution //= 2

        self.mid = nn.ModuleList([block(ch, ch), AttentionBlock(ch, cfg.NUM_HEADS, groups), block(ch, ch)])
        self.type_head = nn.Sequential(nn.GroupNorm(groups, ch), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                       nn.Linear(ch, len(cfg.TYPE_CLASSES)))

        # Decoder mirrors the encoder, consuming one skip per block.
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(cfg.CHANNEL_MULTS))):
            for index in range(cfg.NUM_RES_BLOCKS + 1):
                layers = [block(ch + skips.pop(), base * mult)]
                ch = base * mult
                if resolution in cfg.ATTENTION_RESOLUTIONS:
                    layers.append(AttentionBlock(ch, cfg.NUM_HEADS, groups))
                if level > 0 and index == cfg.NUM_RES_BLOCKS:
                    layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    layers.append(nn.Conv2d(ch, ch, 3, padding=1))
                    resolution *= 2
                self.up.append(nn.ModuleList(layers))
        assert not skips, "encoder/decoder skip bookkeeping is inconsistent"
        self.out = nn.Sequential(nn.GroupNorm(groups, ch), nn.SiLU(), _zero(nn.Conv2d(ch, num_channels, 3, padding=1)))

    @staticmethod
    def _run(layers: nn.ModuleList, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Apply a list of layers, passing the embedding to residual blocks.

        Args:
            layers (nn.ModuleList): Layers of one encoder/decoder step.
            h (torch.Tensor): Features.
            emb (torch.Tensor): Timestep embedding.

        Returns:
            torch.Tensor: Features.
        """
        for layer in layers:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        return h

    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict v and the missing-type logits.

        Args:
            x_t (torch.Tensor): (B, C, H, W) noisy target in [-1, 1] scale.
            condition (torch.Tensor): (B, C, H, W) incomplete plan in [-1, 1].
            t (torch.Tensor): (B,) integer timesteps.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: v (B, C, H, W) and type logits (B, num_types).
        """
        inputs: list[torch.Tensor] = [x_t, condition]
        if self.use_coordinates:
            inputs.append(self.coords.expand(x_t.shape[0], -1, -1, -1).to(x_t.dtype))
        emb: torch.Tensor = self.time_mlp(timestep_embedding(t, self.base))
        h: torch.Tensor = self.stem(torch.cat(inputs, dim=1))
        skips: list[torch.Tensor] = [h]
        for layers in self.down:
            h = self._run(layers, h, emb)
            skips.append(h)
        h = self._run(self.mid, h, emb)
        type_logits: torch.Tensor = self.type_head(h)
        for layers in self.up:
            h = self._run(layers, torch.cat([h, skips.pop()], dim=1), emb)
        return self.out(h), type_logits
