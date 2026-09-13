"""Task 6.4: Phase 2 Alpine fine-tuning tests (NWP passthrough refactor).

Verify: Phase 2 uses ERA5 passthrough adapter (no StormCast); no LoRA
injection; DEVINE weight initialization works; warmup freezing disables
wind channel gradients; head-only training produces valid checkpoints.
"""

from __future__ import annotations

import pytest
import torch

from terrain_weather_ml.terrain.downscaling.devine import (
    create_mock_devine_checkpoint,
)
from terrain_weather_ml.training.checkpoint import load_phase_checkpoint
from terrain_weather_ml.training.datasets import ERA5StationDataset
from terrain_weather_ml.training.phases import Phase1Trainer, Phase2Trainer, PhaseConfig


@pytest.fixture
def phase1_checkpoint_path(tmp_path):
    """Run Phase 1 and return checkpoint path."""
    config = PhaseConfig(
        phase=1, epochs=1, batch_size=4, learning_rate=1e-3,
        checkpoint_dir=str(tmp_path / "ckpts"),
        device="cpu", base_features=16,
    )
    trainer = Phase1Trainer(config)

    samples = [
        {"terrain": torch.randn(17, 32, 32), "wind_target": torch.randn(2, 32, 32)}
        for _ in range(8)
    ]
    from terrain_weather_ml.training.datasets import CFDPhaseDataset
    dataset = CFDPhaseDataset(samples)

    return trainer.train(dataset)


@pytest.fixture
def devine_checkpoint_path(tmp_path):
    """Create a mock DEVINE checkpoint."""
    return create_mock_devine_checkpoint(
        tmp_path / "devine", in_channels=23, base_features=16
    )


@pytest.fixture
def era5_dataset():
    """Synthetic ERA5+station dataset for Phase 2."""
    n_samples = 8
    h_fine, w_fine = 32, 32
    c_terrain = 17
    c_weather = 6
    c_out = 4

    samples = []
    for _ in range(n_samples):
        samples.append({
            "terrain": torch.randn(c_terrain, h_fine, w_fine),
            "weather": torch.randn(c_weather, h_fine, w_fine),
            "target": torch.randn(c_out, h_fine, w_fine),
            "mask": torch.ones(h_fine, w_fine),
        })
    return ERA5StationDataset(samples)


class TestPhase2Initialization:
    """Phase 2 init with NWP passthrough (no StormCast)."""

    def test_init_from_phase1_succeeds(self, phase1_checkpoint_path):
        """Phase 2 initialization from Phase 1 checkpoint should succeed."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p2",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
        )
        trainer = Phase2Trainer(config)
        assert trainer.head is not None

    def test_no_stormcast_adapter(self, phase1_checkpoint_path):
        """Phase 2 should NOT have a StormCast adapter."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p2",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
        )
        trainer = Phase2Trainer(config)
        assert not hasattr(trainer, "adapter")

    def test_no_lora_injection(self, phase1_checkpoint_path):
        """Phase 2 should NOT inject LoRA (raw NWP, no neural backbone)."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p2",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
        )
        trainer = Phase2Trainer(config)
        # Only head params should be trainable, no LoRA
        trainable = [
            name for name, p in trainer.head.named_parameters()
            if p.requires_grad
        ]
        assert len(trainable) > 0
        lora_params = [n for n in trainable if "lora" in n.lower()]
        assert len(lora_params) == 0

    def test_init_without_phase1_raises(self):
        """Phase 2 without Phase 1 checkpoint should raise ValueError."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p2",
            device="cpu", base_features=16,
            prerequisite_checkpoint=None,
        )
        with pytest.raises(ValueError, match="Phase 1"):
            Phase2Trainer(config)


class TestDEVINEInit:
    """DEVINE weight initialization for Phase 2."""

    def test_devine_weights_loaded(
        self, phase1_checkpoint_path, devine_checkpoint_path,
    ):
        """DEVINE weights should be loaded into wind channels."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p2_devine",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
            devine_checkpoint=str(devine_checkpoint_path),
            freeze_wind_during_warmup=True,
        )
        trainer = Phase2Trainer(config)
        assert trainer._devine_loaded is True

    def test_warmup_freezing(
        self, phase1_checkpoint_path, devine_checkpoint_path,
    ):
        """During warmup, DEVINE-loaded encoder/decoder params should be frozen."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p2_warmup",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
            devine_checkpoint=str(devine_checkpoint_path),
            freeze_wind_during_warmup=True,
            warmup_epochs=2,
        )
        trainer = Phase2Trainer(config)

        # Check that encoder blocks are frozen
        frozen_count = sum(
            1 for p in trainer.head.unet.encoder_blocks.parameters()
            if not p.requires_grad
        )
        total_count = sum(
            1 for _ in trainer.head.unet.encoder_blocks.parameters()
        )
        assert frozen_count == total_count, (
            f"Expected all encoder params frozen, but {frozen_count}/{total_count}"
        )


class TestPhase2Training:
    """Phase 2 training loop with ERA5 passthrough."""

    def test_one_epoch_completes(
        self, phase1_checkpoint_path, era5_dataset, tmp_path,
    ):
        """One epoch of Phase 2 training should complete without error."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
        )
        trainer = Phase2Trainer(config)
        ckpt_path = trainer.train(era5_dataset)
        assert ckpt_path.exists()

    def test_checkpoint_metadata(
        self, phase1_checkpoint_path, era5_dataset, tmp_path,
    ):
        """Phase 2 checkpoint metadata should be complete."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
        )
        trainer = Phase2Trainer(config)
        ckpt_path = trainer.train(era5_dataset)

        ckpt = load_phase_checkpoint(ckpt_path)
        assert ckpt.metadata.phase == 2
        assert ckpt.metadata.epoch == 1
        assert ckpt.metadata.parent_checkpoint_hash is not None

    def test_checkpoint_contains_head_only(
        self, phase1_checkpoint_path, era5_dataset, tmp_path,
    ):
        """Phase 2 checkpoint should have head params only (no LoRA)."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
        )
        trainer = Phase2Trainer(config)
        ckpt_path = trainer.train(era5_dataset)

        ckpt = load_phase_checkpoint(ckpt_path)

        head_keys = [k for k in ckpt.model_state_dict if k.startswith("head.")]
        lora_keys = [k for k in ckpt.model_state_dict if k.startswith("adapter.")]

        assert len(head_keys) > 0, "No head keys in checkpoint"
        assert len(lora_keys) == 0, "LoRA keys found in passthrough checkpoint"

    def test_head_params_update(
        self, phase1_checkpoint_path, era5_dataset, tmp_path,
    ):
        """Head parameters should update during Phase 2 training."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
        )
        trainer = Phase2Trainer(config)

        # Snapshot before
        before = {
            name: p.data.clone()
            for name, p in trainer.head.named_parameters()
            if p.requires_grad
        }

        trainer.train(era5_dataset)

        # At least some params should have changed
        changed = False
        for name, p in trainer.head.named_parameters():
            if name in before and not torch.equal(p.data, before[name]):
                changed = True
                break
        assert changed, "Head parameters did not update during training"


class TestERA5StationDataset:
    """Task 3.3: ERA5StationDataset."""

    def test_dataset_length(self):
        """Dataset should report correct length."""
        samples = [
            {
                "terrain": torch.randn(17, 32, 32),
                "weather": torch.randn(6, 32, 32),
                "target": torch.randn(4, 32, 32),
                "mask": torch.ones(32, 32),
            }
            for _ in range(5)
        ]
        ds = ERA5StationDataset(samples)
        assert len(ds) == 5

    def test_dataset_item_shape(self):
        """Dataset items should be (terrain, weather, target, mask) tuples."""
        samples = [
            {
                "terrain": torch.randn(17, 32, 32),
                "weather": torch.randn(6, 32, 32),
                "target": torch.randn(4, 32, 32),
                "mask": torch.ones(32, 32),
            }
        ]
        ds = ERA5StationDataset(samples)
        terrain, weather, target, mask = ds[0]
        assert terrain.shape == (17, 32, 32)
        assert weather.shape == (6, 32, 32)
        assert target.shape == (4, 32, 32)
        assert mask.shape == (32, 32)

    def test_dataset_hash(self):
        """Dataset should compute a data hash."""
        samples = [
            {
                "terrain": torch.randn(17, 32, 32),
                "weather": torch.randn(6, 32, 32),
                "target": torch.randn(4, 32, 32),
                "mask": torch.ones(32, 32),
            }
        ]
        ds = ERA5StationDataset(samples)
        h = ds.compute_data_hash()
        assert isinstance(h, str)
        assert len(h) == 16
