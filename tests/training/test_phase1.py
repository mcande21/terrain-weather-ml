"""Task 6.1: Phase 1 CFD pre-training tests.

Verify: one epoch completes without error on synthetic CFD data; checkpoint
contains only downscaling head parameters (no StormCast weights); checkpoint
metadata is complete.
"""

from __future__ import annotations

import pytest
import torch

from terrain_weather_ml.training.checkpoint import load_phase_checkpoint
from terrain_weather_ml.training.datasets import CFDPhaseDataset
from terrain_weather_ml.training.phases import Phase1Trainer, PhaseConfig


@pytest.fixture
def synthetic_cfd_samples():
    """Create synthetic CFD training data (small tensors for testing)."""
    n_samples = 8
    h, w = 32, 32
    c_terrain = 17
    samples = []
    for i in range(n_samples):
        samples.append({
            "terrain": torch.randn(c_terrain, h, w),
            "wind_target": torch.randn(2, h, w),  # u, v only for Phase 1
        })
    return samples


@pytest.fixture
def cfd_dataset(synthetic_cfd_samples):
    """Wrap synthetic data in a PyTorch Dataset."""
    return CFDPhaseDataset(synthetic_cfd_samples)


@pytest.fixture
def phase1_config(tmp_path):
    """Phase 1 training configuration."""
    return PhaseConfig(
        phase=1,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        lambda_div=0.1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
        base_features=16,  # Small U-Net for testing
    )


class TestPhase1Training:
    """Phase 1 CFD pre-training."""

    def test_one_epoch_completes(self, cfd_dataset, phase1_config):
        """One epoch of Phase 1 training should complete without error."""
        trainer = Phase1Trainer(phase1_config)
        trainer.train(cfd_dataset, val_dataset=None)

    def test_checkpoint_saved(self, cfd_dataset, phase1_config, tmp_path):
        """Phase 1 should save a checkpoint after training."""
        trainer = Phase1Trainer(phase1_config)
        ckpt_path = trainer.train(cfd_dataset, val_dataset=None)

        assert ckpt_path is not None
        assert ckpt_path.exists()

    def test_checkpoint_no_stormcast_weights(
        self, cfd_dataset, phase1_config, tmp_path
    ):
        """Phase 1 checkpoint should contain only downscaling head params."""
        trainer = Phase1Trainer(phase1_config)
        ckpt_path = trainer.train(cfd_dataset, val_dataset=None)

        ckpt = load_phase_checkpoint(ckpt_path)

        # No StormCast/LoRA keys should be present
        for key in ckpt.model_state_dict:
            assert "backbone" not in key, f"StormCast key found: {key}"
            assert "lora" not in key.lower(), f"LoRA key found: {key}"

        # Should have U-Net keys
        unet_keys = [k for k in ckpt.model_state_dict if "unet" in k]
        assert len(unet_keys) > 0, "No U-Net keys in checkpoint"

    def test_checkpoint_metadata_complete(
        self, cfd_dataset, phase1_config, tmp_path
    ):
        """Phase 1 checkpoint metadata should be fully populated."""
        trainer = Phase1Trainer(phase1_config)
        ckpt_path = trainer.train(cfd_dataset, val_dataset=None)

        ckpt = load_phase_checkpoint(ckpt_path)
        meta = ckpt.metadata

        assert meta.phase == 1
        assert meta.epoch == phase1_config.epochs
        assert isinstance(meta.best_val_loss, float)
        assert meta.training_data_hash is not None
        assert meta.parent_checkpoint_hash is None

    def test_loss_decreases(self, cfd_dataset, phase1_config):
        """Training loss should decrease over epochs (on this small data)."""
        phase1_config.epochs = 5
        trainer = Phase1Trainer(phase1_config)
        trainer.train(cfd_dataset, val_dataset=None)

        losses = trainer.epoch_losses
        assert len(losses) == 5
        # Loss at end should be less than at start
        assert losses[-1] < losses[0]

    def test_validation_loss_computed(self, cfd_dataset, phase1_config):
        """Validation loss should be computed when val_dataset is provided."""
        trainer = Phase1Trainer(phase1_config)
        trainer.train(cfd_dataset, val_dataset=cfd_dataset)

        assert trainer.best_val_loss is not None
        assert trainer.best_val_loss >= 0.0


class TestCFDPhaseDataset:
    """CFD dataset for Phase 1."""

    def test_dataset_length(self, synthetic_cfd_samples):
        """Dataset length should match sample count."""
        dataset = CFDPhaseDataset(synthetic_cfd_samples)
        assert len(dataset) == 8

    def test_dataset_item_shapes(self, synthetic_cfd_samples):
        """Dataset items should have correct tensor shapes."""
        dataset = CFDPhaseDataset(synthetic_cfd_samples)
        terrain, wind_target = dataset[0]
        assert terrain.shape == (17, 32, 32)
        assert wind_target.shape == (2, 32, 32)
