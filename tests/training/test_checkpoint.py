"""Task 6.5: Phase-gated checkpointing tests.

Verify: launching Phase 3 without Phase 2 checkpoint raises descriptive error;
launching Phase 2 without Phase 1 checkpoint raises descriptive error;
checkpoint metadata fields are all populated and parent hash matches input.
"""

from __future__ import annotations

import pytest
import torch

from terrain_weather_ml.training.checkpoint import (
    CheckpointMetadata,
    PhaseCheckpoint,
    load_phase_checkpoint,
    save_phase_checkpoint,
    validate_phase_prerequisites,
)


@pytest.fixture
def phase1_state_dict():
    """Minimal state dict simulating Phase 1 downscaling head weights."""
    return {"unet.init_conv.block.0.weight": torch.randn(64, 23, 3, 3)}


@pytest.fixture
def phase1_checkpoint(tmp_path, phase1_state_dict):
    """Save a Phase 1 checkpoint and return its path."""
    metadata = CheckpointMetadata(
        phase=1,
        epoch=10,
        best_val_loss=0.05,
        training_data_hash="abc123",
        parent_checkpoint_hash=None,
    )
    ckpt = PhaseCheckpoint(
        metadata=metadata,
        model_state_dict=phase1_state_dict,
    )
    path = tmp_path / "phase1.pt"
    save_phase_checkpoint(ckpt, path)
    return path


@pytest.fixture
def phase2_checkpoint(tmp_path, phase1_checkpoint):
    """Save a Phase 2 checkpoint (depends on Phase 1) and return its path."""
    # Load Phase 1 to get its hash
    p1 = load_phase_checkpoint(phase1_checkpoint)
    p1_hash = p1.compute_hash()

    metadata = CheckpointMetadata(
        phase=2,
        epoch=20,
        best_val_loss=0.03,
        training_data_hash="def456",
        parent_checkpoint_hash=p1_hash,
    )
    state = {
        "unet.init_conv.block.0.weight": torch.randn(64, 23, 3, 3),
        "lora.backbone.blocks.0.attn_q.lora_A": torch.randn(8, 64),
    }
    ckpt = PhaseCheckpoint(
        metadata=metadata,
        model_state_dict=state,
    )
    path = tmp_path / "phase2.pt"
    save_phase_checkpoint(ckpt, path)
    return path


class TestPhaseGating:
    """Phase ordering enforcement."""

    def test_phase2_without_phase1_raises(self, tmp_path):
        """Launching Phase 2 without Phase 1 checkpoint raises error."""
        with pytest.raises(ValueError, match="Phase 1"):
            validate_phase_prerequisites(
                target_phase=2,
                prerequisite_checkpoint_path=None,
            )

    def test_phase3_without_phase2_raises(self, tmp_path):
        """Launching Phase 3 without Phase 2 checkpoint raises error."""
        with pytest.raises(ValueError, match="Phase 2"):
            validate_phase_prerequisites(
                target_phase=3,
                prerequisite_checkpoint_path=None,
            )

    def test_phase2_with_wrong_phase_raises(self, tmp_path, phase1_checkpoint):
        """Phase 2 given a non-Phase-1 checkpoint raises error."""
        # Create a Phase 2 checkpoint
        metadata = CheckpointMetadata(
            phase=2, epoch=5, best_val_loss=0.04,
            training_data_hash="x", parent_checkpoint_hash="y",
        )
        wrong = PhaseCheckpoint(
            metadata=metadata,
            model_state_dict={"k": torch.tensor(1.0)},
        )
        wrong_path = tmp_path / "wrong.pt"
        save_phase_checkpoint(wrong, wrong_path)

        with pytest.raises(ValueError, match="Phase 1"):
            validate_phase_prerequisites(
                target_phase=2,
                prerequisite_checkpoint_path=wrong_path,
            )

    def test_phase3_with_phase1_raises(self, phase1_checkpoint):
        """Phase 3 given a Phase 1 (not Phase 2) checkpoint raises error."""
        with pytest.raises(ValueError, match="Phase 2"):
            validate_phase_prerequisites(
                target_phase=3,
                prerequisite_checkpoint_path=phase1_checkpoint,
            )

    def test_phase1_needs_no_prerequisite(self):
        """Phase 1 should pass without any prerequisite."""
        validate_phase_prerequisites(
            target_phase=1,
            prerequisite_checkpoint_path=None,
        )

    def test_phase2_with_phase1_passes(self, phase1_checkpoint):
        """Phase 2 with valid Phase 1 checkpoint passes validation."""
        validate_phase_prerequisites(
            target_phase=2,
            prerequisite_checkpoint_path=phase1_checkpoint,
        )

    def test_phase3_with_phase2_passes(self, phase2_checkpoint):
        """Phase 3 with valid Phase 2 checkpoint passes validation."""
        validate_phase_prerequisites(
            target_phase=3,
            prerequisite_checkpoint_path=phase2_checkpoint,
        )


class TestCheckpointMetadata:
    """Checkpoint metadata completeness."""

    def test_all_metadata_fields_populated(self, phase1_checkpoint):
        """Checkpoint metadata fields must all be populated."""
        ckpt = load_phase_checkpoint(phase1_checkpoint)
        assert ckpt.metadata.phase == 1
        assert ckpt.metadata.epoch == 10
        assert ckpt.metadata.best_val_loss == 0.05
        assert ckpt.metadata.training_data_hash == "abc123"
        # Phase 1 has no parent
        assert ckpt.metadata.parent_checkpoint_hash is None

    def test_parent_hash_matches_input(self, phase1_checkpoint, phase2_checkpoint):
        """Phase 2's parent_checkpoint_hash should match Phase 1's hash."""
        p1 = load_phase_checkpoint(phase1_checkpoint)
        p2 = load_phase_checkpoint(phase2_checkpoint)
        assert p2.metadata.parent_checkpoint_hash == p1.compute_hash()

    def test_save_load_roundtrip(self, tmp_path, phase1_state_dict):
        """Save and load should produce identical checkpoint."""
        metadata = CheckpointMetadata(
            phase=1, epoch=5, best_val_loss=0.1,
            training_data_hash="test", parent_checkpoint_hash=None,
        )
        original = PhaseCheckpoint(
            metadata=metadata,
            model_state_dict=phase1_state_dict,
        )
        path = tmp_path / "roundtrip.pt"
        save_phase_checkpoint(original, path)
        loaded = load_phase_checkpoint(path)

        assert loaded.metadata.phase == original.metadata.phase
        assert loaded.metadata.epoch == original.metadata.epoch
        assert loaded.metadata.best_val_loss == original.metadata.best_val_loss
        assert loaded.metadata.training_data_hash == original.metadata.training_data_hash
        for key in original.model_state_dict:
            torch.testing.assert_close(
                loaded.model_state_dict[key],
                original.model_state_dict[key],
            )

    def test_checkpoint_hash_deterministic(self, phase1_checkpoint):
        """Same checkpoint loaded twice should produce same hash."""
        ckpt1 = load_phase_checkpoint(phase1_checkpoint)
        ckpt2 = load_phase_checkpoint(phase1_checkpoint)
        assert ckpt1.compute_hash() == ckpt2.compute_hash()

    def test_different_checkpoints_different_hash(self, tmp_path):
        """Different model weights should produce different hashes."""
        meta = CheckpointMetadata(
            phase=1, epoch=1, best_val_loss=0.5,
            training_data_hash="a", parent_checkpoint_hash=None,
        )
        ckpt_a = PhaseCheckpoint(
            metadata=meta,
            model_state_dict={"w": torch.ones(10)},
        )
        ckpt_b = PhaseCheckpoint(
            metadata=meta,
            model_state_dict={"w": torch.zeros(10)},
        )
        path_a = tmp_path / "a.pt"
        path_b = tmp_path / "b.pt"
        save_phase_checkpoint(ckpt_a, path_a)
        save_phase_checkpoint(ckpt_b, path_b)

        loaded_a = load_phase_checkpoint(path_a)
        loaded_b = load_phase_checkpoint(path_b)
        assert loaded_a.compute_hash() != loaded_b.compute_hash()

    def test_checkpoint_extra_data(self, tmp_path):
        """Checkpoint should support extra_data for quantile mapping etc."""
        meta = CheckpointMetadata(
            phase=3, epoch=15, best_val_loss=0.02,
            training_data_hash="qm", parent_checkpoint_hash="parent",
        )
        extra = {"quantile_params": {"temperature": [1.0, 2.0, 3.0]}}
        ckpt = PhaseCheckpoint(
            metadata=meta,
            model_state_dict={"w": torch.tensor(1.0)},
            extra_data=extra,
        )
        path = tmp_path / "extra.pt"
        save_phase_checkpoint(ckpt, path)
        loaded = load_phase_checkpoint(path)
        assert loaded.extra_data["quantile_params"]["temperature"] == [1.0, 2.0, 3.0]
