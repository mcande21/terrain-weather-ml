"""StormCast backbone adapter.

Loads frozen NVIDIA StormCast, injects LoRA adapters for regional
specialization, and extracts surface weather variables for the
downscaling head.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Surface variable extraction constants (Design Decision D2)
# ---------------------------------------------------------------------------

C_WEATHER = 6

SURFACE_VARIABLE_NAMES = [
    "T2M",    # 2m temperature
    "U10",    # 10m u-wind component
    "V10",    # 10m v-wind component
    "PRATE",  # Precipitation rate
    "SP",     # Surface pressure
    "BLH",    # Boundary layer height
]

# Default StormCast output indices for surface variables.
# These are validated against model metadata at load time when available.
DEFAULT_SURFACE_INDICES = [0, 1, 2, 3, 4, 5]


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

    Args:
        checkpoint_path: Local path to StormCast weights.
        hf_repo_id: HuggingFace repo for downloading weights.
        model_class: Fully qualified class name for the model architecture.
            Use for testing with mock models.
        surface_indices: Indices into StormCast's 99-var output for the
            6 surface variables.
        device: PyTorch device string. Auto-detected if None.
        lora_rank: LoRA adapter rank (r). Default 8 per WeatherPEFT.
        lora_alpha: LoRA scaling factor. Default 16.
        lora_dropout: LoRA dropout rate. Default 0.0.
        lora_target_modules: Attention layer name patterns to target.
    """

    checkpoint_path: str | None = None
    hf_repo_id: str = "nvidia/stormcast-v1-era5-hrrr"
    model_class: str | None = None
    surface_indices: list[int] = field(
        default_factory=lambda: list(DEFAULT_SURFACE_INDICES)
    )
    device: str | None = None
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["attn_q", "attn_k", "attn_v"]
    )


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
# StormCast Adapter
# ---------------------------------------------------------------------------

class StormCastAdapter(nn.Module):
    """Frozen StormCast backbone with LoRA adapters.

    Loads a StormCast model, freezes all parameters, optionally injects
    LoRA adapters into attention layers, and extracts the 6 surface
    weather variables consumed by the downscaling head.

    Output shape: (B, C_WEATHER, H, W) where C_WEATHER=6.
    """

    def __init__(self, config: StormCastConfig):
        super().__init__()
        self.config = config
        self.device_str = config.device or _get_device()

        # Load backbone
        self.backbone = self._load_backbone(config)
        self._freeze_backbone()

        self.surface_indices = config.surface_indices
        self._lora_injected = False

    def _load_backbone(self, config: StormCastConfig) -> nn.Module:
        """Load the StormCast model from checkpoint."""
        if config.checkpoint_path is None and config.model_class is None:
            raise FileNotFoundError(
                f"StormCast checkpoint not found. Provide a local checkpoint_path "
                f"or download from HuggingFace: {config.hf_repo_id}\n"
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
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
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
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        # Include LoRA params if injected
        if self._lora_injected:
            total_params += sum(p.numel() for p in self.parameters()) - sum(
                p.numel() for p in self.backbone.parameters()
            )
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )

        memory_bytes = sum(
            p.numel() * p.element_size() for p in self.backbone.parameters()
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

    def extract_surface_variables(self, full_output: torch.Tensor) -> torch.Tensor:
        """Extract the 6 surface variables from StormCast's full output.

        Args:
            full_output: StormCast output tensor of shape (B, 99, H, W).

        Returns:
            Surface variables tensor of shape (B, C_WEATHER, H, W).
        """
        if full_output.shape[1] < max(self.surface_indices) + 1:
            raise ValueError(
                f"StormCast output has {full_output.shape[1]} channels, "
                f"but surface indices require at least "
                f"{max(self.surface_indices) + 1}"
            )
        return full_output[:, self.surface_indices, :, :]

    def inject_lora(self) -> int:
        """Inject LoRA adapters into backbone attention layers.

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
            "Injected %d LoRA adapters (rank=%d, alpha=%d)",
            injected, config.lora_rank, config.lora_alpha,
        )
        return injected

    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Extract only LoRA parameters for saving."""
        lora_state = {}
        for name, param in self.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                lora_state[name] = param.data.clone()
        return lora_state

    def load_lora_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load LoRA-only parameters onto the model."""
        current_state = self.state_dict()
        for key, value in state_dict.items():
            if key in current_state:
                current_state[key] = value
            else:
                logger.warning("LoRA key %s not found in model, skipping", key)
        self.load_state_dict(current_state, strict=False)

    def forward(
        self,
        x: torch.Tensor,
        extract_surface: bool = True,
    ) -> torch.Tensor:
        """Run StormCast inference and optionally extract surface variables.

        Args:
            x: Input tensor of shape (B, 99, H, W).
            extract_surface: If True, return only 6 surface variables.

        Returns:
            If extract_surface: tensor of shape (B, C_WEATHER, H, W).
            Otherwise: full output of shape (B, 99, H, W).
        """
        with torch.no_grad() if not self._lora_injected else torch.enable_grad():
            output = self.backbone(x)

        if extract_surface:
            return self.extract_surface_variables(output)
        return output
