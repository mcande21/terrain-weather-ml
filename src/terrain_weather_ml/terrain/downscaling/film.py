"""FiLM (Feature-wise Linear Modulation) conditioning for terrain downscaling.

Implements FiLM conditioning as described in Perez et al. (2018), adapted for
terrain-conditioned weather downscaling. A lightweight CNN encodes terrain
features into per-decoder-block affine modulation parameters (gamma, beta),
applied after GroupNorm in each decoder ConvBlock.

Architecture reference: Karpos (github.com/maurinl26/karpos-downscaling).
"""

from __future__ import annotations

import torch
from torch import nn


class FiLMBlock(nn.Module):
    """Applies affine modulation: output = gamma * GroupNorm(x) + beta.

    Contains its own GroupNorm layer. Applied in place of a standalone
    GroupNorm within a ConvBlock when FiLM conditioning is active.

    Args:
        channels: Number of feature channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        num_groups = min(32, channels)
        self.norm = nn.GroupNorm(num_groups, channels)

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            x: Input tensor (B, C, H, W).
            gamma: Scale parameter (B, C, 1, 1).
            beta: Shift parameter (B, C, 1, 1).

        Returns:
            Modulated tensor (B, C, H, W).
        """
        return gamma * self.norm(x) + beta


class FiLMGenerator(nn.Module):
    """Maps terrain features to per-decoder-block (gamma, beta) pairs.

    Lightweight CNN that processes the terrain tensor through 3 conv layers,
    global average pools, and projects to per-block affine parameters via
    independent linear heads. Gamma initialized to 1.0, beta to 0.0 so
    the initial model behaves as an unconditioned U-Net.

    Args:
        c_terrain: Number of terrain input channels (default: 17).
        base_features: Base feature count of the target U-Net (default: 64).
            Decoder block channels are derived as [8f, 4f, 2f, f].
    """

    def __init__(
        self,
        c_terrain: int = 17,
        base_features: int = 64,
    ):
        super().__init__()
        self.base_features = base_features

        # Decoder block channel counts (from deepest to shallowest)
        self.decoder_channels = [
            base_features * 8,   # 512 for f=64
            base_features * 4,   # 256
            base_features * 2,   # 128
            base_features,       # 64
        ]

        # Lightweight terrain encoder: 3 conv layers with downsampling
        hidden = 128
        self.encoder = nn.Sequential(
            nn.Conv2d(c_terrain, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, 64), 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, 256), 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Per-block linear heads: each projects to 2 * C_block (gamma + beta)
        self.heads = nn.ModuleList()
        for ch in self.decoder_channels:
            head = nn.Linear(256, 2 * ch)
            # Identity init: very small weights + bias = [1,...,0,...]
            # Small but non-zero weights ensure gradients flow while output
            # stays near identity (dominated by bias term).
            nn.init.uniform_(head.weight, -1e-3, 1e-3)
            with torch.no_grad():
                head.bias[:ch] = 1.0
                head.bias[ch:] = 0.0
            self.heads.append(head)

    def forward(
        self, terrain: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Generate FiLM parameters from terrain features.

        Args:
            terrain: Terrain tensor (B, C_terrain, H, W).

        Returns:
            List of (gamma, beta) tuples, one per decoder block.
            Each gamma/beta has shape (B, C_block, 1, 1).
        """
        # Encode terrain to feature vector
        z = self.encoder(terrain)  # (B, 256)

        params = []
        for i, head in enumerate(self.heads):
            ch = self.decoder_channels[i]
            out = head(z)  # (B, 2 * ch)
            gamma = out[:, :ch].unsqueeze(-1).unsqueeze(-1)  # (B, ch, 1, 1)
            beta = out[:, ch:].unsqueeze(-1).unsqueeze(-1)   # (B, ch, 1, 1)
            params.append((gamma, beta))

        return params
