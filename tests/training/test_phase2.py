"""Task 6.2: Phase 2 Alpine fine-tuning tests.

Verify: Phase 2 initialization from Phase 1 checkpoint succeeds; DEVINE
weights are loaded into wind channels; warmup freezing disables wind channel
gradients; station-HRRR pairing produces valid training samples.
"""

from __future__ import annotations

import pytest
import torch

from terrain_weather_ml.backbone.stormcast import LoRALinear
from terrain_weather_ml.terrain.downscaling.devine import (
    create_mock_devine_checkpoint,
)
from terrain_weather_ml.training.checkpoint import load_phase_checkpoint
from terrain_weather_ml.training.datasets import AlpinePhaseDataset
from terrain_weather_ml.training.phases import Phase1Trainer, Phase2Trainer, PhaseConfig
from tests.backbone.test_checkpoint import MockStormCast


@pytest.fixture
def mock_stormcast_path(tmp_path):
    """Save mock StormCast weights for Phase 2 testing."""
    model = MockStormCast(dim=64, n_blocks=2)
    path = tmp_path / "stormcast.pt"
    torch.save(model.state_dict(), path)
    return path


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
def alpine_dataset():
    """Synthetic Alpine dataset for Phase 2."""
    n_samples = 8
    h_fine, w_fine = 32, 32
    h_coarse, w_coarse = 8, 8
    c_terrain = 17
    c_out = 4

    samples = []
    for _ in range(n_samples):
        samples.append({
            "terrain": torch.randn(c_terrain, h_fine, w_fine),
            "weather_in": torch.randn(99, h_coarse, w_coarse),
            "weather_target": torch.randn(c_out, h_fine, w_fine),
            "mask": torch.ones(h_fine, w_fine),
        })
    return AlpinePhaseDataset(samples)


class TestPhase2Initialization:
    """Phase 2 init from Phase 1 checkpoint."""

    def test_init_from_phase1_succeeds(
        self, phase1_checkpoint_path, mock_stormcast_path
    ):
        """Phase 2 initialization from Phase 1 checkpoint should succeed."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p2",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
        )
        trainer = Phase2Trainer(config)
        assert trainer.head is not None
        assert trainer.adapter is not None

    def test_init_without_phase1_raises(self, mock_stormcast_path):
        """Phase 2 without Phase 1 checkpoint should raise ValueError."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p2",
            device="cpu", base_features=16,
            prerequisite_checkpoint=None,
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
        )
        with pytest.raises(ValueError, match="Phase 1"):
            Phase2Trainer(config)


class TestDEVINEInit:
    """DEVINE weight initialization for Phase 2."""

    def test_devine_weights_loaded(
        self,
        phase1_checkpoint_path,
        mock_stormcast_path,
        devine_checkpoint_path,
    ):
        """DEVINE weights should be loaded into wind channels."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p2_devine",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
            devine_checkpoint=str(devine_checkpoint_path),
            freeze_wind_during_warmup=True,
        )
        trainer = Phase2Trainer(config)
        assert trainer._devine_loaded is True

    def test_warmup_freezing(
        self,
        phase1_checkpoint_path,
        mock_stormcast_path,
        devine_checkpoint_path,
    ):
        """During warmup, DEVINE-loaded encoder/decoder params should be frozen."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p2_warmup",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
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
    """Phase 2 training loop."""

    def test_one_epoch_completes(
        self,
        phase1_checkpoint_path,
        mock_stormcast_path,
        alpine_dataset,
        tmp_path,
    ):
        """One epoch of Phase 2 training should complete without error."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
        )
        trainer = Phase2Trainer(config)
        ckpt_path = trainer.train(alpine_dataset)
        assert ckpt_path.exists()

    def test_checkpoint_metadata(
        self,
        phase1_checkpoint_path,
        mock_stormcast_path,
        alpine_dataset,
        tmp_path,
    ):
        """Phase 2 checkpoint metadata should be complete."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
        )
        trainer = Phase2Trainer(config)
        ckpt_path = trainer.train(alpine_dataset)

        ckpt = load_phase_checkpoint(ckpt_path)
        assert ckpt.metadata.phase == 2
        assert ckpt.metadata.epoch == 1
        assert ckpt.metadata.parent_checkpoint_hash is not None

    def test_checkpoint_contains_head_and_lora(
        self,
        phase1_checkpoint_path,
        mock_stormcast_path,
        alpine_dataset,
        tmp_path,
    ):
        """Phase 2 checkpoint should have both head and LoRA params."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
        )
        trainer = Phase2Trainer(config)
        ckpt_path = trainer.train(alpine_dataset)

        ckpt = load_phase_checkpoint(ckpt_path)

        head_keys = [k for k in ckpt.model_state_dict if k.startswith("head.")]
        lora_keys = [k for k in ckpt.model_state_dict if k.startswith("adapter.")]

        assert len(head_keys) > 0, "No head keys in checkpoint"
        assert len(lora_keys) > 0, "No LoRA keys in checkpoint"

    def test_lora_injected(
        self,
        phase1_checkpoint_path,
        mock_stormcast_path,
    ):
        """LoRA adapters should be injected into StormCast."""
        config = PhaseConfig(
            phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_lora",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
        )
        trainer = Phase2Trainer(config)

        lora_count = sum(
            1 for _, m in trainer.adapter.backbone.named_modules()
            if isinstance(m, LoRALinear)
        )
        assert lora_count > 0
