"""U-Net encoder-decoder architecture for terrain downscaling (task 5.1).

Encoder with 4 downsampling stages, decoder with matching upsampling stages,
and skip connections concatenating encoder features at each resolution level.

Supports optional FiLM (Feature-wise Linear Modulation) conditioning in the
decoder path: when film_params are provided, affine modulation is applied
after the first GroupNorm in each decoder ConvBlock.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Double convolution block: Conv -> GroupNorm -> ReLU -> Conv -> GroupNorm -> ReLU.

    Supports optional FiLM conditioning after the first GroupNorm:
    Conv -> FiLM(GroupNorm) -> ReLU -> Conv -> GroupNorm -> ReLU
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        num_groups = min(32, out_ch)
        # Store layers individually for FiLM insertion, but keep
        # self.block as a Sequential for backward compatibility
        # (existing code accesses block[0].in_channels).
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional FiLM modulation.

        Args:
            x: Input tensor (B, C_in, H, W).
            gamma: FiLM scale parameter (B, C_out, 1, 1). None = no FiLM.
            beta: FiLM shift parameter (B, C_out, 1, 1). None = no FiLM.

        Returns:
            Output tensor (B, C_out, H, W).
        """
        if gamma is None:
            return self.block(x)

        # FiLM path: insert modulation after first GroupNorm
        # block[0] = Conv2d, block[1] = GroupNorm, block[2] = ReLU
        # block[3] = Conv2d, block[4] = GroupNorm, block[5] = ReLU
        x = self.block[0](x)   # Conv2d
        x = self.block[1](x)   # GroupNorm
        x = gamma * x + beta   # FiLM modulation
        x = self.block[2](x)   # ReLU
        x = self.block[3](x)   # Conv2d
        x = self.block[4](x)   # GroupNorm
        x = self.block[5](x)   # ReLU
        return x


class EncoderBlock(nn.Module):
    """Downsampling encoder block: MaxPool -> ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class DecoderBlock(nn.Module):
    """Upsampling decoder block: Upsample -> Concat skip -> ConvBlock.

    Supports optional FiLM conditioning passed through to its ConvBlock.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        gamma: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional FiLM conditioning.

        Args:
            x: Input tensor from previous stage.
            skip: Skip connection tensor from encoder.
            gamma: FiLM scale parameter. None = no FiLM.
            beta: FiLM shift parameter. None = no FiLM.

        Returns:
            Decoded tensor.
        """
        x = self.up(x)
        # Handle potential size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, gamma=gamma, beta=beta)


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

    def forward(
        self,
        x: torch.Tensor,
        film_params: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Forward pass through U-Net.

        Args:
            x: Input tensor of shape (B, C_in, H, W).
            film_params: Optional list of (gamma, beta) tuples, one per
                decoder block. When None, decoder runs unconditioned.

        Returns:
            Output tensor of shape (B, C_out, H, W).
        """
        skips, x = self._encode(x)
        return self._decode(x, skips, film_params=film_params)

    def forward_ablated(
        self,
        x: torch.Tensor,
        zero_skip_idx: int,
        film_params: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Forward pass with one skip connection zeroed out (for ablation tests).

        Args:
            x: Input tensor.
            zero_skip_idx: Index of the skip connection to zero (0 = deepest).
            film_params: Optional FiLM conditioning parameters.
        """
        skips, x = self._encode(x)
        # Zero out the specified skip connection
        # skips are stored from shallow to deep: [s0, s1, s2, s3]
        # decoder expects them reversed: [s3, s2, s1, s0]
        # zero_skip_idx 0 = deepest skip (s3)
        target = len(skips) - 1 - zero_skip_idx
        skips[target] = torch.zeros_like(skips[target])
        return self._decode(x, skips, film_params=film_params)

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
        self,
        x: torch.Tensor,
        skips: list[torch.Tensor],
        film_params: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Run decoder with skip connections and optional FiLM conditioning.

        Args:
            x: Bottleneck tensor.
            skips: Skip connection tensors from encoder.
            film_params: Optional list of (gamma, beta) tuples, one per
                decoder block. When None, decoder runs unconditioned.
        """
        for i, (block, skip) in enumerate(
            zip(self.decoder_blocks, reversed(skips))
        ):
            if film_params is not None:
                gamma, beta = film_params[i]
                x = block(x, skip, gamma=gamma, beta=beta)
            else:
                x = block(x, skip)
        return self.final_conv(x)
