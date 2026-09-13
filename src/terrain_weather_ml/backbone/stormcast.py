"""StormCast backbone adapter -- EXPERIMENTAL.

This adapter wraps the frozen NVIDIA StormCast model for use as an
experimental backbone. The default backbone is NWP passthrough; this
adapter is selected via `backbone: "stormcast"` configuration.

StormCast provides 4 surface variables (T2M, U10, V10, MSLP) from its
99-channel output. The remaining 2 variables (PRATE, BLH) are supplied
by HRRR passthrough to satisfy the C_WEATHER=6 interface contract.

EXPERIMENTAL STATUS:
    StormCast is preserved for A/B comparison against raw NWP passthrough.
    It is NOT the default and should not be used without explicit opt-in.
    See Design Decision D3 in the raw-nwp-backbone-refactor change.

Key corrections from real-weight analysis:
    - .mdlus loading: repack zip to strip model/ prefix
    - Surface indices: [2, 0, 1, 3] (not [0..5])
    - LoRA targets: 'affine' nn.Linear layers (not attn QKV Conv2d)
    - Only 4 of 6 weather vars from StormCast; 2 via HRRR passthrough

Implements the WeatherAdapter protocol (adapter_protocol.py).
"""

from __future__ import annotations

import importlib
import io
import logging
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Surface variable extraction constants
#
# StormCast's 99 output variables include 4 surface fields:
#   idx 0: u10m (10m u-wind)
#   idx 1: v10m (10m v-wind)
#   idx 2: t2m  (2m temperature)
#   idx 3: mslp (mean sea level pressure)
#
# The remaining C_WEATHER channels (PRATE, BLH) come from HRRR passthrough.
# ---------------------------------------------------------------------------

C_WEATHER = 6

# StormCast provides these 4 surface variables
STORMCAST_SURFACE_NAMES = ["T2M", "U10", "V10", "MSLP"]
STORMCAST_SURFACE_INDICES = [2, 0, 1, 3]  # T2M=idx2, U10=idx0, V10=idx1, MSLP=idx3

# HRRR passthrough provides these 2 variables StormCast lacks
HRRR_PASSTHROUGH_NAMES = ["PRATE", "BLH"]

# Combined output order: 4 from StormCast + 2 from HRRR passthrough
SURFACE_VARIABLE_NAMES = STORMCAST_SURFACE_NAMES + HRRR_PASSTHROUGH_NAMES

# Legacy alias kept for backward compatibility during migration
DEFAULT_SURFACE_INDICES = list(STORMCAST_SURFACE_INDICES)


def _get_device() -> str:
    """Auto-detect best available device (MPS for Apple Silicon)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StormCastConfig:
    """Configuration for the StormCast backbone adapter.

    EXPERIMENTAL: StormCast is not the default backbone. Use
    `backbone: "stormcast"` configuration to enable it.

    Args:
        checkpoint_path: Local path to StormCast weights (.pt or .mdlus).
        hf_repo_id: HuggingFace repo for downloading weights.
        model_class: Fully qualified class name for the model architecture.
            Use for testing with mock models.
        surface_indices: Indices into StormCast's 99-var output for the
            4 surface variables it provides. Default [2, 0, 1, 3].
        device: PyTorch device string. Auto-detected if None.
        lora_rank: LoRA adapter rank (r). Default 8 per WeatherPEFT.
        lora_alpha: LoRA scaling factor. Default 16.
        lora_dropout: LoRA dropout rate. Default 0.0.
        lora_target_modules: Layer name patterns to target for LoRA.
            Default ['affine'] -- targets the 55 nn.Linear affine
            projections in StormCast UNet blocks. Skips fused QKV Conv2d.
    """

    checkpoint_path: str | None = None
    hf_repo_id: str = "nvidia/stormcast-v1-era5-hrrr"
    model_class: str | None = None
    surface_indices: list[int] = field(
        default_factory=lambda: list(STORMCAST_SURFACE_INDICES)
    )
    device: str | None = None
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["affine"]
    )


# ---------------------------------------------------------------------------
# .mdlus format handling
# ---------------------------------------------------------------------------

def _load_mdlus_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load state dict from NVIDIA Modulus .mdlus format.

    The .mdlus format is a zip archive where weight files are stored
    under a ``model/`` prefix. This function repacks the archive to
    strip that prefix before loading via ``torch.load``.

    Args:
        path: Path to the .mdlus file.

    Returns:
        The model state dict.
    """
    with zipfile.ZipFile(path, "r") as zf:
        # Find the weights file (model/weights.pt or similar)
        weight_entries = [
            n for n in zf.namelist()
            if n.startswith("model/") and n.endswith(".pt")
        ]
        if not weight_entries:
            raise FileNotFoundError(
                f"No model/*.pt found in .mdlus archive: {path}\n"
                f"Archive contents: {zf.namelist()}"
            )

        # Read the weights file, stripping the model/ prefix
        weights_data = zf.read(weight_entries[0])

    # Load via BytesIO to avoid temp files
    buf = io.BytesIO(weights_data)
    state_dict = torch.load(buf, map_location="cpu", weights_only=True)
    logger.info(
        "Loaded .mdlus checkpoint from %s (entry: %s)",
        path, weight_entries[0],
    )
    return state_dict


# ---------------------------------------------------------------------------
# LoRA layer (lightweight, no PEFT dependency)
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """LoRA adapter wrapping a frozen nn.Linear layer.

    Implements: output = frozen_linear(x) + (x @ A^T @ B^T) * (alpha / r)
    Only A and B are trainable; the original linear's parameters stay frozen.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming, B with zeros (standard LoRA init)
        nn.init.kaiming_uniform_(self.lora_A)
        # B starts at zero so initial output = original output

        # Freeze original
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling


# ---------------------------------------------------------------------------
# StormCast Adapter -- EXPERIMENTAL
# ---------------------------------------------------------------------------

class StormCastAdapter(nn.Module):
    """Frozen StormCast backbone with LoRA adapters -- EXPERIMENTAL.

    Loads a StormCast model, freezes all parameters, optionally injects
    LoRA adapters into ``affine`` layers, and extracts 4 surface weather
    variables. Combined with 2 HRRR passthrough variables via the
    ``extract()`` method to produce C_WEATHER=6 output.

    This adapter implements the ``WeatherAdapter`` protocol.

    EXPERIMENTAL: Not the default backbone. Enable via
    ``backbone: "stormcast"`` configuration. See Design Decision D3.

    Output shapes:
        - extract_surface_variables(): (B, 4, H, W) -- StormCast only
        - extract(): (B, 6, H, W) -- StormCast + HRRR passthrough
        - forward(extract_surface=True): (B, 4, H, W)
        - forward(extract_surface=False): (B, 99, H, W)
    """

    # Mark as experimental for runtime checks
    experimental = True

    def __init__(self, config: StormCastConfig):
        super().__init__()
        self.config = config
        self.device_str = config.device or _get_device()

        warnings.warn(
            "StormCastAdapter is EXPERIMENTAL. The default backbone is NWP "
            "passthrough. Use StormCast only for A/B comparison experiments. "
            "See Design Decision D3 in raw-nwp-backbone-refactor.",
            stacklevel=2,
        )

        # Load backbone
        self.backbone = self._load_backbone(config)
        self._freeze_backbone()

        self.surface_indices = config.surface_indices
        self._lora_injected = False

    def _load_backbone(self, config: StormCastConfig) -> nn.Module:
        """Load the StormCast model from checkpoint."""
        if config.checkpoint_path is None and config.model_class is None:
            raise FileNotFoundError(
                f"StormCast checkpoint not found. Provide a local "
                f"checkpoint_path or download from HuggingFace: "
                f"{config.hf_repo_id}\n"
                f"  pip install huggingface-hub\n"
                f"  huggingface-cli download {config.hf_repo_id}"
            )

        # Resolve model class
        model = self._instantiate_model(config)

        # Load checkpoint weights
        if config.checkpoint_path is not None:
            path = Path(config.checkpoint_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"StormCast checkpoint not found at: {path}\n"
                    f"Download from HuggingFace: {config.hf_repo_id}\n"
                    f"  huggingface-cli download {config.hf_repo_id}"
                )

            # Handle .mdlus (NVIDIA Modulus) format
            if path.suffix == ".mdlus":
                state_dict = _load_mdlus_state_dict(path)
            else:
                state_dict = torch.load(
                    path, map_location="cpu", weights_only=True
                )

            model.load_state_dict(state_dict)
            logger.info("Loaded StormCast checkpoint from %s", path)

        model.eval()
        return model

    @staticmethod
    def _instantiate_model(config: StormCastConfig) -> nn.Module:
        """Instantiate the model class from its fully-qualified name."""
        if config.model_class is None:
            raise ValueError("model_class must be provided")

        module_path, class_name = config.model_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_cls = getattr(module, class_name)
        return model_cls()

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def model_info(self) -> dict:
        """Report parameter count and memory footprint."""
        total_params = sum(p.numel() for p in self.backbone.parameters())
        trainable_params = sum(
            p.numel()
            for p in self.backbone.parameters()
            if p.requires_grad
        )
        # Include LoRA params if injected
        if self._lora_injected:
            total_params += sum(
                p.numel() for p in self.parameters()
            ) - sum(p.numel() for p in self.backbone.parameters())
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )

        memory_bytes = sum(
            p.numel() * p.element_size()
            for p in self.backbone.parameters()
        )
        memory_mb = memory_bytes / (1024 * 1024)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "memory_mb": round(memory_mb, 2),
            "frozen": all(
                not p.requires_grad
                for p in self.backbone.parameters()
            ),
        }

    def extract_surface_variables(
        self, full_output: torch.Tensor
    ) -> torch.Tensor:
        """Extract the 4 StormCast surface variables from full output.

        StormCast provides 4 surface variables at indices [2, 0, 1, 3]:
        T2M (idx 2), U10 (idx 0), V10 (idx 1), MSLP (idx 3).

        Args:
            full_output: StormCast output tensor of shape (B, 99, H, W).

        Returns:
            Surface variables tensor of shape (B, 4, H, W).
        """
        if full_output.shape[1] < max(self.surface_indices) + 1:
            raise ValueError(
                f"StormCast output has {full_output.shape[1]} channels, "
                f"but surface indices require at least "
                f"{max(self.surface_indices) + 1}"
            )
        return full_output[:, self.surface_indices, :, :]

    def extract(
        self, nwp_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Extract 6 weather variables (WeatherAdapter protocol).

        Combines 4 StormCast surface variables with 2 HRRR passthrough
        variables (PRATE, BLH) to produce the C_WEATHER=6 output.

        Args:
            nwp_data: Dictionary containing:
                - 'stormcast_input': (B, 99, H, W) tensor for StormCast
                - 'PRATE': (B, 1, H, W) precipitation rate from HRRR
                - 'BLH': (B, 1, H, W) boundary layer height from HRRR

        Returns:
            Weather tensor of shape (B, C_WEATHER, H, W) = (B, 6, H, W).
        """
        # Run StormCast and extract 4 surface variables
        stormcast_input = nwp_data["stormcast_input"]
        stormcast_output = self.forward(
            stormcast_input, extract_surface=True
        )

        # Get HRRR passthrough variables
        prate = nwp_data["PRATE"]
        blh = nwp_data["BLH"]

        # Concatenate: 4 StormCast + 2 HRRR = 6 total
        return torch.cat([stormcast_output, prate, blh], dim=1)

    def inject_lora(self) -> int:
        """Inject LoRA adapters into backbone layers.

        Targets ``affine`` nn.Linear layers (55 in real StormCast).
        Skips Conv2d layers (including fused QKV) since LoRA only
        wraps nn.Linear modules.

        Returns:
            Number of LoRA adapters injected.
        """
        config = self.config
        injected = 0

        for name, module in list(self.backbone.named_modules()):
            if not isinstance(module, nn.Linear):
                continue

            # Check if this module matches any target pattern
            module_name = name.split(".")[-1]
            if module_name not in config.lora_target_modules:
                continue

            # Replace with LoRA-wrapped version
            lora_layer = LoRALinear(
                module,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
            )

            # Navigate to parent and replace
            parts = name.split(".")
            parent = self.backbone
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], lora_layer)
            injected += 1

        self._lora_injected = True
        logger.info(
            "Injected %d LoRA adapters (rank=%d, alpha=%d, "
            "targets=%s)",
            injected, config.lora_rank, config.lora_alpha,
            config.lora_target_modules,
        )
        return injected

    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract only LoRA parameters for saving."""
        lora_state = {}
        for name, param in self.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                lora_state[name] = param.data.clone()
        return lora_state

    def load_lora_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> None:
        """Load LoRA-only parameters onto the model."""
        current_state = self.state_dict()
        for key, value in state_dict.items():
            if key in current_state:
                current_state[key] = value
            else:
                logger.warning(
                    "LoRA key %s not found in model, skipping", key
                )
        self.load_state_dict(current_state, strict=False)

    def forward(
        self,
        x: torch.Tensor,
        extract_surface: bool = True,
    ) -> torch.Tensor:
        """Run StormCast inference and optionally extract surface variables.

        Args:
            x: Input tensor of shape (B, 99, H, W).
            extract_surface: If True, return 4 StormCast surface variables.

        Returns:
            If extract_surface: tensor of shape (B, 4, H, W).
            Otherwise: full output of shape (B, 99, H, W).
        """
        ctx = (
            torch.enable_grad()
            if self._lora_injected
            else torch.no_grad()
        )
        with ctx:
            output = self.backbone(x)

        if extract_surface:
            return self.extract_surface_variables(output)
        return output
