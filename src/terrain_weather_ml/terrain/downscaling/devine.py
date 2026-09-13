"""DEVINE weight initialization for wind channels (task 5.5).

Loads pre-trained wind downscaling weights from DEVINE
(louisletoumelin/wind_downscaling_cnn) into the terrain downscaling
head's U-Net encoder and decoder. Temperature and precipitation
channels are left randomly initialized.

Supports freezing wind channels during warmup training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


def create_mock_devine_checkpoint(
    save_dir: str | Path,
    in_channels: int = 23,
    base_features: int = 64,
) -> Path:
    """Create a mock DEVINE checkpoint for testing.

    Generates a small state dict mimicking the DEVINE U-Net structure.
    The weights are random but follow the correct naming and shapes.

    Args:
        save_dir: Directory to save the checkpoint.
        in_channels: Number of input channels.
        base_features: Base feature count matching the target U-Net.

    Returns:
        Path to the saved checkpoint file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build a minimal UNet to get correctly-shaped weights
    from terrain_weather_ml.terrain.downscaling.unet import UNet
    mock_unet = UNet(
        in_channels=in_channels,
        out_channels=2,  # DEVINE only has u, v wind output
        base_features=base_features,
    )

    # Save its state dict as a DEVINE checkpoint
    ckpt_path = save_dir / "devine_checkpoint.pt"
    torch.save(
        {"model_state_dict": mock_unet.state_dict()},
        ckpt_path,
    )

    return ckpt_path


def load_devine_weights(
    head: nn.Module,
    checkpoint_path: str | Path,
    freeze_wind: bool = False,
) -> None:
    """Load DEVINE pre-trained weights into the downscaling head.

    Loads encoder and decoder weights from DEVINE, which was trained
    for wind-only (u, v) downscaling. Temperature and precipitation
    output channels are left with their random initialization.

    The final output convolution is handled specially:
    - DEVINE's 2-channel output (u, v) maps to the first 2 channels
      of the head's 4-channel output.
    - Temperature and precipitation channels (2, 3) keep random init.

    Args:
        head: TerrainDownscalingHead instance.
        checkpoint_path: Path to DEVINE checkpoint file.
        freeze_wind: If True, freeze all encoder/decoder parameters
            loaded from DEVINE (requires_grad=False).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"DEVINE checkpoint not found: {checkpoint_path}"
        )

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # Support both raw state dict and wrapped checkpoint
    if "model_state_dict" in ckpt:
        devine_state = ckpt["model_state_dict"]
    else:
        devine_state = ckpt

    unet = head.unet

    # Load shared encoder/decoder weights (these are shape-compatible
    # since DEVINE uses the same U-Net architecture for spatial processing)
    unet_state = unet.state_dict()
    loaded_keys = []
    skipped_keys = []

    for key, value in devine_state.items():
        if key in unet_state:
            if key.startswith("final_conv"):
                # Handle final conv specially: DEVINE has 2 output channels,
                # our head has 4. Load DEVINE's weights into first 2 channels.
                if "weight" in key:
                    # Shape: (C_out, C_in, kH, kW)
                    # DEVINE: (2, C_in, 1, 1), Ours: (4, C_in, 1, 1)
                    devine_out = value.shape[0]
                    unet_state[key][:devine_out] = value
                    loaded_keys.append(key)
                elif "bias" in key:
                    devine_out = value.shape[0]
                    unet_state[key][:devine_out] = value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(key)
            elif unet_state[key].shape == value.shape:
                unet_state[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
        else:
            skipped_keys.append(key)

    unet.load_state_dict(unet_state)

    logger.info(
        "Loaded %d DEVINE weights, skipped %d (shape mismatch or missing)",
        len(loaded_keys),
        len(skipped_keys),
    )

    if freeze_wind:
        _freeze_wind_parameters(head)


def _freeze_wind_parameters(head: nn.Module) -> None:
    """Freeze DEVINE-loaded parameters, keep output head trainable.

    Freezes encoder/decoder blocks (loaded from DEVINE for wind).
    Keeps the final output convolution trainable so temperature and
    precipitation channels can learn during warmup.

    After warmup, the user unfreezes all parameters for end-to-end
    joint optimization.
    """
    unet = head.unet

    # Freeze encoder blocks (DEVINE-loaded)
    for param in unet.init_conv.parameters():
        param.requires_grad = False
    for block in unet.encoder_blocks:
        for param in block.parameters():
            param.requires_grad = False

    # Freeze decoder blocks (DEVINE-loaded)
    for block in unet.decoder_blocks:
        for param in block.parameters():
            param.requires_grad = False

    # Keep final_conv trainable (has wind + temp + precip output)
    # The user can additionally freeze wind channels 0-1 if needed

    # Freeze divergence-free projection params (if any registered)
    if hasattr(head, "div_free_proj"):
        for param in head.div_free_proj.parameters():
            param.requires_grad = False

    frozen_count = sum(
        1 for p in head.parameters() if not p.requires_grad
    )
    total_count = sum(1 for _ in head.parameters())
    logger.info(
        "Froze %d/%d parameters for warmup (final_conv remains trainable)",
        frozen_count,
        total_count,
    )
