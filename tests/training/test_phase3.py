"""Task 6.5: Phase 3 Colorado adaptation tests (NWP passthrough refactor).

Verify: Phase 3 uses HRRR passthrough adapter (not StormCast) by default;
head fine-tunes with reduced LR; LoRA is optional (only with StormCast);
checkpoint includes quantile mapping parameters; Phase 2->3 weight transfer
carries head weights only (no backbone weights in default mode).
"""

from __future__ import annotations

import pytest
import torch

from terrain_weather_ml.training.checkpoint import load_phase_checkpoint
from terrain_weather_ml.training.datasets import (
    CFDPhaseDataset,
    ColoradoPhaseDataset,
    ERA5StationDataset,
)
from terrain_weather_ml.training.phases import (
    Phase1Trainer,
    Phase2Trainer,
    Phase3Trainer,
    PhaseConfig,
)
from terrain_weather_ml.training.quantile_mapping import QuantileMapper


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
def phase2_checkpoint_path(tmp_path, phase1_checkpoint_path):
    """Run Phase 2 (with ERA5 passthrough) and return checkpoint path."""
    config = PhaseConfig(
        phase=2, epochs=1, batch_size=4, learning_rate=1e-3,
        checkpoint_dir=str(tmp_path / "p2_ckpts"),
        device="cpu", base_features=16,
        prerequisite_checkpoint=str(phase1_checkpoint_path),
    )
    trainer = Phase2Trainer(config)

    samples = []
    for _ in range(8):
        samples.append({
            "terrain": torch.randn(17, 32, 32),
            "weather": torch.randn(6, 32, 32),
            "target": torch.randn(4, 32, 32),
            "mask": torch.ones(32, 32),
        })
    return trainer.train(ERA5StationDataset(samples))


@pytest.fixture
def colorado_dataset():
    """Synthetic Colorado SNOTEL dataset for Phase 3."""
    samples = []
    for _ in range(8):
        samples.append({
            "terrain": torch.randn(17, 32, 32),
            "weather_in": torch.randn(6, 32, 32),
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
    """Phase 3 init from Phase 2 checkpoint (default: passthrough mode)."""

    def test_init_from_phase2_succeeds(self, phase2_checkpoint_path):
        """Phase 3 initialization from Phase 2 checkpoint should succeed."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p3",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
        )
        trainer = Phase3Trainer(config)
        assert trainer.head is not None

    def test_init_without_phase2_raises(self):
        """Phase 3 without Phase 2 checkpoint should raise ValueError."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p3",
            device="cpu", base_features=16,
            prerequisite_checkpoint=None,
        )
        with pytest.raises(ValueError, match="Phase 2"):
            Phase3Trainer(config)

    def test_init_with_phase1_raises(self, phase1_checkpoint_path):
        """Phase 3 given Phase 1 (not Phase 2) checkpoint should raise."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p3",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase1_checkpoint_path),
        )
        with pytest.raises(ValueError, match="Phase 2"):
            Phase3Trainer(config)

    def test_no_stormcast_in_default_mode(self, phase2_checkpoint_path):
        """Default Phase 3 should NOT have a StormCast adapter."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p3_default",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
        )
        trainer = Phase3Trainer(config)
        assert not hasattr(trainer, "adapter")


class TestPhase3HeadFineTuning:
    """Default Phase 3: head fine-tunes with reduced LR (no LoRA)."""

    def test_head_trainable(self, phase2_checkpoint_path):
        """In default mode, head should have trainable parameters."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-4,
            checkpoint_dir="/tmp/test_p3_trainable",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
        )
        trainer = Phase3Trainer(config)

        trainable = [
            name for name, p in trainer.head.named_parameters()
            if p.requires_grad
        ]
        assert len(trainable) > 0, "Head should be trainable in default mode"

    def test_head_params_update(
        self, phase2_checkpoint_path, colorado_dataset, tmp_path,
    ):
        """Head parameters should update during Phase 3 default training."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
        )
        trainer = Phase3Trainer(config)

        # Snapshot head params before training
        head_before = {}
        for name, param in trainer.head.named_parameters():
            if param.requires_grad:
                head_before[name] = param.data.clone()

        trainer.train(colorado_dataset)

        # Head params should have changed
        changed = False
        for name, param in trainer.head.named_parameters():
            if name in head_before and not torch.equal(
                param.data, head_before[name]
            ):
                changed = True
                break
        assert changed, "Head parameters did not update"

    def test_no_lora_params_in_default(self, phase2_checkpoint_path):
        """Default mode should have no LoRA parameters."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p3_no_lora",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
        )
        trainer = Phase3Trainer(config)

        lora_params = [
            name for name, p in trainer.head.named_parameters()
            if "lora" in name.lower()
        ]
        assert len(lora_params) == 0, "LoRA params found in default mode"


class TestPhase3Training:
    """Phase 3 training loop (default passthrough mode)."""

    def test_one_epoch_completes(
        self, phase2_checkpoint_path, colorado_dataset, tmp_path,
    ):
        """One epoch of Phase 3 training should complete without error."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
        )
        trainer = Phase3Trainer(config)
        ckpt_path = trainer.train(colorado_dataset)
        assert ckpt_path.exists()

    def test_checkpoint_metadata(
        self, phase2_checkpoint_path, colorado_dataset, tmp_path,
    ):
        """Phase 3 checkpoint metadata should be complete."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
        )
        trainer = Phase3Trainer(config)
        ckpt_path = trainer.train(colorado_dataset)

        ckpt = load_phase_checkpoint(ckpt_path)
        assert ckpt.metadata.phase == 3
        assert ckpt.metadata.parent_checkpoint_hash is not None

    def test_checkpoint_head_only(
        self, phase2_checkpoint_path, colorado_dataset, tmp_path,
    ):
        """Phase 3 default checkpoint should have head params only."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
        )
        trainer = Phase3Trainer(config)
        ckpt_path = trainer.train(colorado_dataset)

        ckpt = load_phase_checkpoint(ckpt_path)
        head_keys = [k for k in ckpt.model_state_dict if k.startswith("head.")]
        lora_keys = [k for k in ckpt.model_state_dict if k.startswith("adapter.")]

        assert len(head_keys) > 0, "No head keys in checkpoint"
        assert len(lora_keys) == 0, "LoRA keys in default passthrough checkpoint"

    def test_checkpoint_includes_quantile_params(
        self, phase2_checkpoint_path, colorado_dataset, quantile_mapper,
        tmp_path,
    ):
        """Phase 3 checkpoint should include quantile mapping parameters."""
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
        )
        trainer = Phase3Trainer(config, quantile_mapper=quantile_mapper)
        ckpt_path = trainer.train(colorado_dataset)

        ckpt = load_phase_checkpoint(ckpt_path)
        assert "quantile_params" in ckpt.extra_data
        assert "temperature" in ckpt.extra_data["quantile_params"]


class TestPhase2To3WeightTransfer:
    """Task 3.4: Phase 2->3 weight transfer."""

    def test_head_weights_transfer(
        self, phase2_checkpoint_path,
    ):
        """Head weights from Phase 2 should transfer to Phase 3."""
        # Load Phase 2 checkpoint
        p2_ckpt = load_phase_checkpoint(phase2_checkpoint_path)
        p2_head_keys = [
            k for k in p2_ckpt.model_state_dict if k.startswith("head.")
        ]

        # Init Phase 3
        config = PhaseConfig(
            phase=3, epochs=1, batch_size=4, learning_rate=1e-3,
            checkpoint_dir="/tmp/test_p3_transfer",
            device="cpu", base_features=16,
            prerequisite_checkpoint=str(phase2_checkpoint_path),
        )
        trainer = Phase3Trainer(config)

        # Phase 3 head should have matching param values
        p3_state = trainer.head.state_dict()
        for key in p2_head_keys:
            stripped = key[5:]  # Remove "head." prefix
            if stripped in p3_state:
                p2_val = p2_ckpt.model_state_dict[key]
                torch.testing.assert_close(
                    p3_state[stripped], p2_val,
                    msg=f"Weight mismatch in {stripped}"
                )

    def test_no_backbone_weights_carry_over(self, phase2_checkpoint_path):
        """No backbone/adapter weights should carry over in default mode."""
        p2_ckpt = load_phase_checkpoint(phase2_checkpoint_path)
        adapter_keys = [
            k for k in p2_ckpt.model_state_dict if k.startswith("adapter.")
        ]
        # Phase 2 in default passthrough mode should have no adapter keys
        assert len(adapter_keys) == 0, (
            "Phase 2 passthrough checkpoint should not have adapter keys"
        )
