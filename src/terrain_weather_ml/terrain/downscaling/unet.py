"""U-Net encoder-decoder architecture for terrain downscaling (task 5.1).

Encoder with 4 downsampling stages, decoder with matching upsampling stages,
and skip connections concatenating encoder features at each resolution level.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # GroupNorm instead of BatchNorm -- works with batch_size=1 and
        # small spatial dims at the bottleneck.
        num_groups = min(32, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Downsampling encoder block: MaxPool -> ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class DecoderBlock(nn.Module):
    """Upsampling decoder block: Upsample -> Concat skip -> ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle potential size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net with 4 encoder/decoder stages and skip connections.

    Args:
        in_channels: Number of input channels (e.g. 23 for terrain+weather).
        out_channels: Number of output channels (e.g. 4 for u, v, T, precip).
        base_features: Number of features in the first encoder stage.
    """

    def __init__(
        self,
        in_channels: int = 23,
        out_channels: int = 4,
        base_features: int = 64,
    ):
        super().__init__()
        f = base_features

        # Initial convolution (no downsampling)
        self.init_conv = ConvBlock(in_channels, f)

        # Encoder: 4 downsampling stages
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(f, f * 2),
            EncoderBlock(f * 2, f * 4),
            EncoderBlock(f * 4, f * 8),
            EncoderBlock(f * 8, f * 16),
        ])

        # Decoder: 4 upsampling stages (reversed channel progression)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(f * 16, f * 8, f * 8),
            DecoderBlock(f * 8, f * 4, f * 4),
            DecoderBlock(f * 4, f * 2, f * 2),
            DecoderBlock(f * 2, f, f),
        ])

        # Final 1x1 convolution to output channels
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H, W).
        """
        skips, x = self._encode(x)
        return self._decode(x, skips)

    def forward_ablated(
        self, x: torch.Tensor, zero_skip_idx: int
    ) -> torch.Tensor:
        """Forward pass with one skip connection zeroed out (for ablation tests).

        Args:
            x: Input tensor.
            zero_skip_idx: Index of the skip connection to zero (0 = deepest).
        """
        skips, x = self._encode(x)
        # Zero out the specified skip connection
        # skips are stored from shallow to deep: [s0, s1, s2, s3]
        # decoder expects them reversed: [s3, s2, s1, s0]
        # zero_skip_idx 0 = deepest skip (s3)
        target = len(skips) - 1 - zero_skip_idx
        skips[target] = torch.zeros_like(skips[target])
        return self._decode(x, skips)

    def _encode(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run encoder, returning skip connections and bottleneck output."""
        skips = []
        x = self.init_conv(x)
        skips.append(x)

        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if i < len(self.encoder_blocks) - 1:
                skips.append(x)

        return skips, x

    def _decode(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        """Run decoder with skip connections."""
        for block, skip in zip(self.decoder_blocks, reversed(skips)):
            x = block(x, skip)
        return self.final_conv(x)
