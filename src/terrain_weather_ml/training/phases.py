"""Training phase implementations.

Phase 1: CFD pre-training of the downscaling head (wind-only).
Phase 2: Alpine fine-tuning with ERA5 passthrough + DEVINE init (head only).
Phase 3: Colorado adaptation with HRRR passthrough + quantile mapping.

Default backbone is NWP passthrough (raw ERA5/HRRR). StormCast is
experimental and only used when explicitly configured.

Each phase is a self-contained trainer that produces a phase-gated checkpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from terrain_weather_ml.terrain.downscaling.divergence import compute_divergence
from terrain_weather_ml.terrain.downscaling.head import (
    TerrainDownscalingHead,
)
from terrain_weather_ml.terrain.downscaling.loss import DownscalingLoss
from terrain_weather_ml.training.checkpoint import (
    CheckpointMetadata,
    PhaseCheckpoint,
    save_phase_checkpoint,
    validate_phase_prerequisites,
)

logger = logging.getLogger(__name__)


def _get_device(device: str | None = None) -> torch.device:
    """Resolve device string to torch.device with MPS auto-detection."""
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class PhaseConfig:
    """Configuration shared across all training phases.

    Attributes:
        phase: Training phase (1, 2, or 3).
        epochs: Number of training epochs.
        batch_size: Batch size for DataLoader.
        learning_rate: Optimizer learning rate.
        lambda_div: Divergence penalty weight.
        lambda_oro: Orographic penalty weight.
        checkpoint_dir: Directory to save checkpoints.
        device: PyTorch device string (auto-detected if None).
        base_features: Base feature count for U-Net (small for tests).
        c_terrain: Number of terrain feature channels.
        c_weather: Number of weather input channels.
        c_out: Number of output channels.
        warmup_epochs: Epochs with wind channels frozen (Phase 2).
        prerequisite_checkpoint: Path to prerequisite checkpoint.
        devine_checkpoint: Path to DEVINE weights (Phase 2).
        freeze_wind_during_warmup: Freeze DEVINE wind channels (Phase 2).
        lora_rank: LoRA adapter rank (Phase 2/3).
        lora_alpha: LoRA scaling factor.
        stormcast_model_class: Fully qualified StormCast model class.
        stormcast_checkpoint_path: Path to StormCast weights.
    """

    phase: int = 1
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    lambda_div: float = 0.1
    lambda_oro: float = 0.0
    checkpoint_dir: str = "./checkpoints"
    device: str | None = None
    base_features: int = 64
    c_terrain: int = 17
    c_weather: int = 6
    c_out: int = 4
    warmup_epochs: int = 0
    prerequisite_checkpoint: str | None = None
    devine_checkpoint: str | None = None
    freeze_wind_during_warmup: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    stormcast_model_class: str | None = None
    stormcast_checkpoint_path: str | None = None


class Phase1Trainer:
    """Phase 1: CFD pre-training of the downscaling head.

    Trains only the downscaling head on synthetic CFD data. No StormCast
    backbone is involved. Loss is MSE + mass-conservation penalty.

    The downscaling head receives terrain features directly (no weather
    input from StormCast) and predicts wind fields (u, v).
    """

    def __init__(self, config: PhaseConfig):
        self.config = config
        self.device = _get_device(config.device)

        # Phase 1 uses a wind-only downscaling head (2 output channels)
        # with terrain input only (no weather channels from StormCast)
        self.head = TerrainDownscalingHead(
            c_terrain=config.c_terrain,
            c_weather=0,  # No weather input in Phase 1
            c_out=2,      # u, v wind only
            base_features=config.base_features,
            apply_divergence_free=False,  # Soft constraint via loss only
        )
        self.head.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.head.parameters(), lr=config.learning_rate
        )
        self.lambda_div = config.lambda_div

        # Tracking
        self.epoch_losses: list[float] = []
        self.best_val_loss: float | None = None

    def train(
        self,
        dataset: Dataset,
        val_dataset: Dataset | None = None,
    ) -> Path:
        """Run Phase 1 training loop.

        Args:
            dataset: CFDPhaseDataset with (terrain, wind_target) samples.
            val_dataset: Optional validation dataset.

        Returns:
            Path to the saved Phase 1 checkpoint.
        """
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

        best_val_loss = float("inf")

        for epoch in range(self.config.epochs):
            self.head.train()
            epoch_loss = 0.0
            n_batches = 0

            for terrain, wind_target in loader:
                terrain = terrain.to(self.device)
                wind_target = wind_target.to(self.device)

                # Phase 1: terrain-only input, create zero-sized weather
                # The head expects weather input; with c_weather=0 we pass
                # a dummy tensor matching terrain spatial dims
                dummy_weather = torch.zeros(
                    terrain.shape[0], 0, terrain.shape[2], terrain.shape[3],
                    device=self.device,
                )

                self.optimizer.zero_grad()
                pred = self.head(terrain, dummy_weather)

                # MSE loss on wind prediction
                mse_loss = nn.functional.mse_loss(pred, wind_target)

                # Divergence penalty on predicted wind field
                u = pred[:, 0:1]
                v = pred[:, 1:2]
                div = compute_divergence(u, v)
                div_penalty = (div ** 2).mean()

                loss = mse_loss + self.lambda_div * div_penalty

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.epoch_losses.append(avg_loss)

            # Validation
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                best_val_loss = min(best_val_loss, val_loss)
                self.best_val_loss = best_val_loss
                logger.info(
                    "Phase 1 epoch %d/%d: train_loss=%.4f val_loss=%.4f",
                    epoch + 1, self.config.epochs, avg_loss, val_loss,
                )
            else:
                # Use training loss as proxy
                self.best_val_loss = avg_loss
                logger.info(
                    "Phase 1 epoch %d/%d: train_loss=%.4f",
                    epoch + 1, self.config.epochs, avg_loss,
                )

        # Save checkpoint
        return self._save_checkpoint(dataset)

    def _validate(self, val_loader: DataLoader) -> float:
        """Compute validation loss."""
        self.head.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for terrain, wind_target in val_loader:
                terrain = terrain.to(self.device)
                wind_target = wind_target.to(self.device)
                dummy_weather = torch.zeros(
                    terrain.shape[0], 0, terrain.shape[2], terrain.shape[3],
                    device=self.device,
                )
                pred = self.head(terrain, dummy_weather)
                loss = nn.functional.mse_loss(pred, wind_target)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, dataset: Dataset) -> Path:
        """Save Phase 1 checkpoint with metadata."""
        data_hash = "unknown"
        if hasattr(dataset, "compute_data_hash"):
            data_hash = dataset.compute_data_hash()

        metadata = CheckpointMetadata(
            phase=1,
            epoch=self.config.epochs,
            best_val_loss=self.best_val_loss or 0.0,
            training_data_hash=data_hash,
            parent_checkpoint_hash=None,
        )

        ckpt = PhaseCheckpoint(
            metadata=metadata,
            model_state_dict=self.head.state_dict(),
        )

        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / "phase1_checkpoint.pt"
        save_phase_checkpoint(ckpt, path)

        return path


class Phase2Trainer:
    """Phase 2: Alpine fine-tuning with ERA5 passthrough + DEVINE init.

    Uses raw ERA5 data via NWP passthrough adapter (no neural backbone).
    No LoRA injection -- trains the downscaling head only. Optionally
    initializes wind channels from DEVINE weights with warmup freeze.
    """

    def __init__(self, config: PhaseConfig):
        self.config = config
        self.device = _get_device(config.device)

        # Validate prerequisite
        self.phase1_ckpt = validate_phase_prerequisites(
            target_phase=2,
            prerequisite_checkpoint_path=config.prerequisite_checkpoint,
        )

        # Build downscaling head (no StormCast adapter -- ERA5 passthrough)
        self.head = TerrainDownscalingHead(
            c_terrain=config.c_terrain,
            c_weather=config.c_weather,
            c_out=config.c_out,
            base_features=config.base_features,
            apply_divergence_free=False,
        )

        # Initialize from Phase 1 checkpoint
        self._init_from_phase1()

        # Initialize from DEVINE if provided
        if config.devine_checkpoint:
            from terrain_weather_ml.terrain.downscaling.devine import (
                load_devine_weights,
            )
            load_devine_weights(
                self.head,
                config.devine_checkpoint,
                freeze_wind=config.freeze_wind_during_warmup,
            )
            self._devine_loaded = True
        else:
            self._devine_loaded = False

        # Move to device
        self.head.to(self.device)

        # Head-only optimizer (no LoRA, no backbone params)
        self.optimizer = torch.optim.Adam(
            self.head.parameters(), lr=config.learning_rate
        )

        self.loss_fn = DownscalingLoss(
            lambda_div=config.lambda_div,
            lambda_oro=config.lambda_oro,
        )
        self.loss_fn.to(self.device)

        # Tracking
        self.epoch_losses: list[float] = []
        self.best_val_loss: float | None = None

    def _init_from_phase1(self) -> None:
        """Initialize downscaling head wind channels from Phase 1 checkpoint.

        Phase 1 head has c_weather=0, c_out=2 (wind only).
        Phase 2 head has c_weather=6, c_out=4 (wind + temp + precip).

        We transfer the encoder/decoder weights where shapes match,
        and the first 2 output channels from the final conv.
        """
        if self.phase1_ckpt is None:
            return

        p1_state = self.phase1_ckpt.model_state_dict
        p2_state = self.head.state_dict()

        loaded = 0
        skipped = 0

        for key, value in p1_state.items():
            if key not in p2_state:
                skipped += 1
                continue

            target_shape = p2_state[key].shape

            if key.startswith("unet.final_conv"):
                # Transfer first 2 channels (wind) from Phase 1
                if "weight" in key:
                    # Phase 1: (2, C_in, 1, 1), Phase 2: (4, C_in, 1, 1)
                    min_out = min(value.shape[0], target_shape[0])
                    min_in = min(value.shape[1], target_shape[1])
                    p2_state[key][:min_out, :min_in] = value[:min_out, :min_in]
                    loaded += 1
                elif "bias" in key:
                    min_out = min(value.shape[0], target_shape[0])
                    p2_state[key][:min_out] = value[:min_out]
                    loaded += 1
            elif key.startswith("unet.init_conv"):
                # Input conv has different in_channels (17 vs 23)
                if "weight" in key and value.shape != target_shape:
                    # Partial load: terrain channels are the same
                    min_in = min(value.shape[1], target_shape[1])
                    p2_state[key][:, :min_in] = value[:, :min_in]
                    loaded += 1
                elif value.shape == target_shape:
                    p2_state[key] = value
                    loaded += 1
                else:
                    skipped += 1
            elif value.shape == target_shape:
                p2_state[key] = value
                loaded += 1
            else:
                skipped += 1

        self.head.load_state_dict(p2_state)
        logger.info(
            "Phase 2: loaded %d params from Phase 1 checkpoint, skipped %d",
            loaded, skipped,
        )

    def _unfreeze_wind_channels(self) -> None:
        """Unfreeze wind channels after warmup period."""
        for param in self.head.parameters():
            param.requires_grad = True
        # Re-create optimizer with all head params
        self.optimizer = torch.optim.Adam(
            self.head.parameters(), lr=self.config.learning_rate,
        )
        logger.info("Phase 2: unfroze wind channels after warmup")

    def train(
        self,
        dataset: Dataset,
        val_dataset: Dataset | None = None,
    ) -> Path:
        """Run Phase 2 training loop.

        Args:
            dataset: ERA5StationDataset with (terrain, weather,
                target, mask) samples.
            val_dataset: Optional validation dataset.

        Returns:
            Path to the saved Phase 2 checkpoint.
        """
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

        best_val_loss = float("inf")

        for epoch in range(self.config.epochs):
            # Unfreeze wind channels after warmup
            if (
                epoch == self.config.warmup_epochs
                and self.config.warmup_epochs > 0
                and self._devine_loaded
            ):
                self._unfreeze_wind_channels()

            self.head.train()
            epoch_loss = 0.0
            n_batches = 0

            for terrain, weather, weather_target, mask in loader:
                terrain = terrain.to(self.device)
                weather = weather.to(self.device)
                weather_target = weather_target.to(self.device)
                mask = mask.to(self.device)

                self.optimizer.zero_grad()

                # ERA5 weather is already extracted via NWP passthrough
                # -- feed directly to downscaling head
                pred = self.head(terrain, weather)

                # Compute loss
                loss_dict = self.loss_fn(pred, weather_target, terrain)
                loss = loss_dict["total"]

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.epoch_losses.append(avg_loss)

            if val_loader is not None:
                val_loss = self._validate(val_loader)
                best_val_loss = min(best_val_loss, val_loss)
                self.best_val_loss = best_val_loss
            else:
                self.best_val_loss = avg_loss

            logger.info(
                "Phase 2 epoch %d/%d: train_loss=%.4f",
                epoch + 1, self.config.epochs, avg_loss,
            )

        return self._save_checkpoint(dataset)

    def _validate(self, val_loader: DataLoader) -> float:
        """Compute validation loss."""
        self.head.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for terrain, weather, weather_target, mask in val_loader:
                terrain = terrain.to(self.device)
                weather = weather.to(self.device)
                weather_target = weather_target.to(self.device)

                pred = self.head(terrain, weather)
                loss_dict = self.loss_fn(pred, weather_target, terrain)
                total_loss += loss_dict["total"].item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, dataset: Dataset) -> Path:
        """Save Phase 2 checkpoint with head weights only."""
        data_hash = "unknown"
        if hasattr(dataset, "compute_data_hash"):
            data_hash = dataset.compute_data_hash()

        parent_hash = None
        if self.phase1_ckpt is not None:
            parent_hash = self.phase1_ckpt.compute_hash()

        metadata = CheckpointMetadata(
            phase=2,
            epoch=self.config.epochs,
            best_val_loss=self.best_val_loss or 0.0,
            training_data_hash=data_hash,
            parent_checkpoint_hash=parent_hash,
        )

        # Head-only state (no LoRA in passthrough mode)
        combined_state = {}
        for key, value in self.head.state_dict().items():
            combined_state[f"head.{key}"] = value

        ckpt = PhaseCheckpoint(
            metadata=metadata,
            model_state_dict=combined_state,
        )

        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / "phase2_checkpoint.pt"
        save_phase_checkpoint(ckpt, path)

        return path


class Phase3Trainer:
    """Phase 3: Colorado adaptation with HRRR passthrough + quantile mapping.

    Default mode: Uses raw HRRR via NWP passthrough, fine-tunes the
    downscaling head with reduced LR. No LoRA, no StormCast.

    Experimental StormCast mode (stormcast_model_class set): Uses StormCast
    backbone with LoRA adapters, freezes head, trains LoRA only.
    """

    def __init__(
        self,
        config: PhaseConfig,
        quantile_mapper: Any | None = None,
    ):
        self.config = config
        self.device = _get_device(config.device)
        self.quantile_mapper = quantile_mapper

        # Validate prerequisite
        self.phase2_ckpt = validate_phase_prerequisites(
            target_phase=3,
            prerequisite_checkpoint_path=config.prerequisite_checkpoint,
        )

        # Determine mode: passthrough (default) vs experimental StormCast
        self._use_stormcast = config.stormcast_model_class is not None

        # Build downscaling head
        self.head = TerrainDownscalingHead(
            c_terrain=config.c_terrain,
            c_weather=config.c_weather,
            c_out=config.c_out,
            base_features=config.base_features,
            apply_divergence_free=False,
        )

        # Load Phase 2 head weights
        self._init_from_phase2()

        if self._use_stormcast:
            # Experimental: StormCast + LoRA mode
            from terrain_weather_ml.backbone.stormcast import (
                StormCastAdapter,
                StormCastConfig,
            )

            sc_config = StormCastConfig(
                checkpoint_path=config.stormcast_checkpoint_path,
                model_class=config.stormcast_model_class,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
            )
            self.adapter = StormCastAdapter(sc_config)
            self.adapter.inject_lora()

            # Freeze head, train only LoRA
            for param in self.head.parameters():
                param.requires_grad = False

            self.head.to(self.device)
            self.adapter.to(self.device)

            trainable_params = [
                p for p in self.adapter.parameters() if p.requires_grad
            ]
            self.optimizer = torch.optim.Adam(
                trainable_params, lr=config.learning_rate
            )
        else:
            # Default: passthrough mode, fine-tune head
            self.head.to(self.device)
            self.optimizer = torch.optim.Adam(
                self.head.parameters(), lr=config.learning_rate
            )

        self.loss_fn = DownscalingLoss(
            lambda_div=config.lambda_div,
            lambda_oro=config.lambda_oro,
        )
        self.loss_fn.to(self.device)

        # Tracking
        self.epoch_losses: list[float] = []
        self.best_val_loss: float | None = None

    def _init_from_phase2(self) -> None:
        """Load downscaling head weights from Phase 2 checkpoint.

        In default passthrough mode, only head weights transfer.
        No backbone/LoRA weights carry over.
        """
        if self.phase2_ckpt is None:
            return

        p2_state = self.phase2_ckpt.model_state_dict

        # Extract head weights
        head_state = {}
        for key, value in p2_state.items():
            if key.startswith("head."):
                head_state[key[5:]] = value  # Strip "head." prefix

        if head_state:
            self.head.load_state_dict(head_state)
            logger.info(
                "Phase 3: loaded %d head params from Phase 2",
                len(head_state),
            )

    def train(
        self,
        dataset: Dataset,
        val_dataset: Dataset | None = None,
    ) -> Path:
        """Run Phase 3 training loop.

        Args:
            dataset: ColoradoPhaseDataset with quantile-mapped inputs.
            val_dataset: Optional validation dataset.

        Returns:
            Path to the saved Phase 3 checkpoint.
        """
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

        best_val_loss = float("inf")

        for epoch in range(self.config.epochs):
            if self._use_stormcast:
                self.head.eval()  # Head frozen in StormCast mode
                self.adapter.train()
            else:
                self.head.train()  # Head trains in passthrough mode

            epoch_loss = 0.0
            n_batches = 0

            for terrain, weather_in, weather_target, mask in loader:
                terrain = terrain.to(self.device)
                weather_in = weather_in.to(self.device)
                weather_target = weather_target.to(self.device)
                mask = mask.to(self.device)

                self.optimizer.zero_grad()

                if self._use_stormcast:
                    # StormCast mode: pass through adapter first
                    weather_features = self.adapter(
                        weather_in, extract_surface=True
                    )
                    pred = self.head(terrain, weather_features)
                else:
                    # Passthrough mode: weather_in already has 6 channels
                    pred = self.head(terrain, weather_in)

                loss_dict = self.loss_fn(pred, weather_target, terrain)
                loss = loss_dict["total"]

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.epoch_losses.append(avg_loss)

            if val_loader is not None:
                val_loss = self._validate(val_loader)
                best_val_loss = min(best_val_loss, val_loss)
                self.best_val_loss = best_val_loss
            else:
                self.best_val_loss = avg_loss

            logger.info(
                "Phase 3 epoch %d/%d: train_loss=%.4f",
                epoch + 1, self.config.epochs, avg_loss,
            )

        return self._save_checkpoint(dataset)

    def _validate(self, val_loader: DataLoader) -> float:
        """Compute validation loss."""
        self.head.eval()
        if self._use_stormcast:
            self.adapter.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for terrain, weather_in, weather_target, mask in val_loader:
                terrain = terrain.to(self.device)
                weather_in = weather_in.to(self.device)
                weather_target = weather_target.to(self.device)

                if self._use_stormcast:
                    weather_features = self.adapter(
                        weather_in, extract_surface=True
                    )
                    pred = self.head(terrain, weather_features)
                else:
                    pred = self.head(terrain, weather_in)

                loss_dict = self.loss_fn(pred, weather_target, terrain)
                total_loss += loss_dict["total"].item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, dataset: Dataset) -> Path:
        """Save Phase 3 checkpoint with head + optional LoRA + quantile."""
        data_hash = "unknown"
        if hasattr(dataset, "compute_data_hash"):
            data_hash = dataset.compute_data_hash()

        parent_hash = None
        if self.phase2_ckpt is not None:
            parent_hash = self.phase2_ckpt.compute_hash()

        metadata = CheckpointMetadata(
            phase=3,
            epoch=self.config.epochs,
            best_val_loss=self.best_val_loss or 0.0,
            training_data_hash=data_hash,
            parent_checkpoint_hash=parent_hash,
        )

        # Save head weights
        combined_state = {}
        for key, value in self.head.state_dict().items():
            combined_state[f"head.{key}"] = value

        # Save LoRA weights only if using StormCast
        if self._use_stormcast:
            lora_state = self.adapter.get_lora_state_dict()
            for key, value in lora_state.items():
                combined_state[f"adapter.{key}"] = value

        # Include quantile mapping parameters
        extra_data = {}
        if self.quantile_mapper is not None:
            extra_data["quantile_params"] = self.quantile_mapper.get_params()

        ckpt = PhaseCheckpoint(
            metadata=metadata,
            model_state_dict=combined_state,
            extra_data=extra_data,
        )

        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / "phase3_checkpoint.pt"
        save_phase_checkpoint(ckpt, path)

        return path
