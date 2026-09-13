"""Task 6.3: Phase 3 Colorado LoRA adaptation tests.

Verify: Phase 3 initialization from Phase 2 checkpoint succeeds; downscaling
head is frozen (no gradients); only LoRA parameters update during training
step; checkpoint includes quantile mapping parameters.
"""

from __future__ import annotations

import pytest
import torch

from terrain_weather_ml.training.checkpoint import load_phase_checkpoint
from terrain_weather_ml.training.datasets import (
    AlpinePhaseDataset,
    CFDPhaseDataset,
    ColoradoPhaseDataset,
)
from terrain_weather_ml.training.phases import (
    Phase1Trainer,
    Phase2Trainer,
    Phase3Trainer,
    PhaseConfig,
)
from terrain_weather_ml.training.quantile_mapping import QuantileMapper
from tests.backbone.test_checkpoint import MockStormCast


@pytest.fixture
def mock_stormcast_path(tmp_path):
    """Save mock StormCast weights."""
    model = MockStormCast(dim=64, n_blocks=2)
    path = tmp_path / "stormcast.pt"
    torch.save(model.state_dict(), path)
    return path


@pytest.fixture
def phase1_checkpoint_path(tmp_path):
    """Run Phase 1 and return checkpoint path."""
    config = PhaseConfig(
        phase=1, epochs=1, batch_size=4, learning_rate=1e-3,
        checkpoint_dir=str(tmp_path / "p1_ckpts"),
        device="cpu", base_features=16,
    )
    trainer = Phase1Trainer(config)
    samples = [
        {"terrain": torch.randn(17, 32, 32), "wind_target": torch.randn(2, 32, 32)}
        for _ in range(8)
    ]
    return trainer.train(CFDPhaseDataset(samples))


@pytest.fixture
def phase2_checkpoint_path(tmp_path, phase1_checkpoint_path, mock_stormcast_path):
    """Run Phase 2 and return checkpoint path."""
    config = PhaseConfig(
        phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
        checkpoint_dir=str(tmp_path / "p2_ckpts"),
        device="cpu", base_features=16,
        prerequisite_checkpoint=str(phase1_checkpoint_path),
        stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
        stormcast_checkpoint_path=str(mock_stormcast_path),
    )
    trainer = Phase2Trainer(config)

    samples = []
    for _ in range(8):
        samples.append({
            "terrain": torch.randn(17, 32, 32),
            "weather_in": torch.randn(99, 8, 8),
            "weather_target": torch.randn(4, 32, 32),
            "mask": torch.ones(32, 32),
        })
    return trainer.train(AlpinePhaseDataset(samples))


@pytest.fixture
def colorado_dataset():
    """Synthetic Colorado SNOTEL dataset for Phase 3."""
    samples = []
    for _ in range(8):
        samples.append({
            "terrain": torch.randn(17, 32, 32),
            "weather_in": torch.randn(99, 8, 8),
            "weather_target": torch.randn(4, 32, 32),
            "mask": torch.ones(32, 32),
        })
    return ColoradoPhaseDataset(samples)


@pytest.fixture
def quantile_mapper():
    """Fitted quantile mapper for testing."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 500
    mapper = QuantileMapper()
    mapper.fit(
        source_data={
            "temperature": torch.tensor(
                rng.normal(275, 10, n), dtype=torch.float32
            ),
            "wind_speed": torch.tensor(
                rng.exponential(5, n), dtype=torch.float32
            ),
        },
        target_data={
            "temperature": torch.tensor(
                rng.normal(280, 12, n), dtype=torch.float32
            ),
            "wind_speed": torch.tensor(
                rng.exponential(7, n), dtype=torch.float32
            ),
        },
    )
    return mapper


class TestPhase3Initialization:
    """Phase 3 init from Phase 2 checkpoint."""

    def test_init_from_phase2_succeeds(
        self, phase2_checkpoint_path, mock_stormcast_path
    ):
        """Phase 3 initialization from Phase 2 checkpoint should succeed."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p3",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
        )
        trainer = Phase3Trainer(config)
        assert trainer.head is not None

    def test_init_without_phase2_raises(self, mock_stormcast_path):
        """Phase 3 without Phase 2 checkpoint should raise ValueError."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p3",
            device="cpu", base_features=16,
            prerequisite_checkpoint=None,
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
        )
        with pytest.raises(ValueError, match="Phase 2"):
            Phase3Trainer(config)

    def test_init_with_phase1_raises(
        self, phase1_checkpoint_path, mock_stormcast_path
    ):
        """Phase 3 given Phase 1 (not Phase 2) checkpoint should raise."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p3",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
        )
        with pytest.raises(ValueError, match="Phase 2"):
            Phase3Trainer(config)


class TestPhase3Freezing:
    """Downscaling head freezing in Phase 3."""

    def test_head_frozen(self, phase2_checkpoint_path, mock_stormcast_path):
        """Downscaling head should be completely frozen."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p3_freeze",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
        )
        trainer = Phase3Trainer(config)

        for name, param in trainer.head.named_parameters():
            assert not param.requires_grad, (
                f"Head param {name} should be frozen"
            )

    def test_only_lora_updates(
        self,
        phase2_checkpoint_path,
        mock_stormcast_path,
        colorado_dataset,
        tmp_path,
    ):
        """Only LoRA parameters should update during a training step."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
        )
        trainer = Phase3Trainer(config)

        # Snapshot LoRA params before training
        lora_before = {}
        for name, param in trainer.adapter.named_parameters():
            if param.requires_grad:
                lora_before[name] = param.data.clone()

        # Snapshot head params before training
        head_before = {}
        for name, param in trainer.head.named_parameters():
            head_before[name] = param.data.clone()

        # Run one training step
        trainer.train(colorado_dataset)

        # LoRA params should have changed
        lora_changed = False
        for name, param in trainer.adapter.named_parameters():
            if name in lora_before and not torch.equal(param.data, lora_before[name]):
                lora_changed = True
                break
        assert lora_changed, "LoRA parameters did not update"

        # Head params should be identical
        for name, param in trainer.head.named_parameters():
            if name in head_before:
                torch.testing.assert_close(
                    param.data, head_before[name],
                    msg=f"Head param {name} changed during Phase 3"
                )


class TestPhase3Training:
    """Phase 3 training loop."""

    def test_one_epoch_completes(
        self,
        phase2_checkpoint_path,
        mock_stormcast_path,
        colorado_dataset,
        tmp_path,
    ):
        """One epoch of Phase 3 training should complete without error."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
        )
        trainer = Phase3Trainer(config)
        ckpt_path = trainer.train(colorado_dataset)
        assert ckpt_path.exists()

    def test_checkpoint_metadata(
        self,
        phase2_checkpoint_path,
        mock_stormcast_path,
        colorado_dataset,
        tmp_path,
    ):
        """Phase 3 checkpoint metadata should be complete."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
        )
        trainer = Phase3Trainer(config)
        ckpt_path = trainer.train(colorado_dataset)

        ckpt = load_phase_checkpoint(ckpt_path)
        assert ckpt.metadata.phase == 3
        assert ckpt.metadata.parent_checkpoint_hash is not None

    def test_checkpoint_includes_quantile_params(
        self,
        phase2_checkpoint_path,
        mock_stormcast_path,
        colorado_dataset,
        quantile_mapper,
        tmp_path,
    ):
        """Phase 3 checkpoint should include quantile mapping parameters."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
            stormcast_model_class="tests.backbone.test_checkpoint.MockStormCast",
            stormcast_checkpoint_path=str(mock_stormcast_path),
        )
        trainer = Phase3Trainer(config, quantile_mapper=quantile_mapper)
        ckpt_path = trainer.train(colorado_dataset)

        ckpt = load_phase_checkpoint(ckpt_path)
        assert "quantile_params" in ckpt.extra_data
        assert "temperature" in ckpt.extra_data["quantile_params"]
