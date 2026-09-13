"""Phase-gated checkpointing for the 3-phase training pipeline (task 6.5).

Enforces phase ordering: Phase 2 requires Phase 1 checkpoint, Phase 3
requires Phase 2 checkpoint. Each checkpoint carries metadata: phase
number, epochs trained, validation loss, training data hash, and parent
checkpoint hash.
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Phase prerequisite mapping: target_phase -> required_predecessor_phase
PHASE_PREREQUISITES = {
    1: None,
    2: 1,
    3: 2,
}


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside model weights in each phase checkpoint.

    Attributes:
        phase: Training phase number (1, 2, or 3).
        epoch: Total epochs trained in this phase.
        best_val_loss: Best validation loss achieved.
        training_data_hash: Hash of training data configuration.
        parent_checkpoint_hash: Hash of the input checkpoint from the
            previous phase. None for Phase 1.
    """

    phase: int
    epoch: int
    best_val_loss: float
    training_data_hash: str
    parent_checkpoint_hash: str | None


@dataclass
class PhaseCheckpoint:
    """Complete checkpoint including metadata and model weights.

    Attributes:
        metadata: Phase metadata.
        model_state_dict: PyTorch model state dict.
        extra_data: Optional additional data (e.g. quantile mapping params).
    """

    metadata: CheckpointMetadata
    model_state_dict: dict[str, torch.Tensor]
    extra_data: dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute a deterministic hash of this checkpoint's contents.

        Uses SHA-256 over the metadata and model weights (serialized to
        bytes). The hash is stable across loads as long as the data is
        identical.
        """
        h = hashlib.sha256()

        # Hash metadata fields
        h.update(str(self.metadata.phase).encode())
        h.update(str(self.metadata.epoch).encode())
        h.update(str(self.metadata.best_val_loss).encode())
        h.update(self.metadata.training_data_hash.encode())
        parent = self.metadata.parent_checkpoint_hash or ""
        h.update(parent.encode())

        # Hash model weights deterministically
        for key in sorted(self.model_state_dict.keys()):
            h.update(key.encode())
            buf = io.BytesIO()
            torch.save(
                self.model_state_dict[key],
                buf,
                _use_new_zipfile_serialization=True,
            )
            h.update(buf.getvalue())

        return h.hexdigest()


def save_phase_checkpoint(
    checkpoint: PhaseCheckpoint,
    path: str | Path,
) -> None:
    """Save a phase checkpoint to disk.

    Args:
        checkpoint: The checkpoint to save.
        path: File path to write to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "phase": checkpoint.metadata.phase,
            "epoch": checkpoint.metadata.epoch,
            "best_val_loss": checkpoint.metadata.best_val_loss,
            "training_data_hash": checkpoint.metadata.training_data_hash,
            "parent_checkpoint_hash": checkpoint.metadata.parent_checkpoint_hash,
        },
        "model_state_dict": checkpoint.model_state_dict,
        "extra_data": checkpoint.extra_data,
    }
    torch.save(data, path)
    logger.info("Saved Phase %d checkpoint to %s", checkpoint.metadata.phase, path)


def load_phase_checkpoint(path: str | Path) -> PhaseCheckpoint:
    """Load a phase checkpoint from disk.

    Args:
        path: File path to read from.

    Returns:
        The loaded PhaseCheckpoint.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)

    meta_dict = data["metadata"]
    metadata = CheckpointMetadata(
        phase=meta_dict["phase"],
        epoch=meta_dict["epoch"],
        best_val_loss=meta_dict["best_val_loss"],
        training_data_hash=meta_dict["training_data_hash"],
        parent_checkpoint_hash=meta_dict["parent_checkpoint_hash"],
    )

    return PhaseCheckpoint(
        metadata=metadata,
        model_state_dict=data["model_state_dict"],
        extra_data=data.get("extra_data", {}),
    )


def validate_phase_prerequisites(
    target_phase: int,
    prerequisite_checkpoint_path: str | Path | None,
) -> PhaseCheckpoint | None:
    """Validate that a training phase has its required prerequisite checkpoint.

    Args:
        target_phase: The phase about to start (1, 2, or 3).
        prerequisite_checkpoint_path: Path to the checkpoint from the
            previous phase. None if no checkpoint is provided.

    Returns:
        The loaded prerequisite checkpoint if one was required and valid,
        or None for Phase 1.

    Raises:
        ValueError: If the prerequisite is missing or has the wrong phase.
    """
    required_phase = PHASE_PREREQUISITES.get(target_phase)

    if required_phase is None:
        # Phase 1 needs no prerequisite
        return None

    if prerequisite_checkpoint_path is None:
        raise ValueError(
            f"Phase {target_phase} requires a Phase {required_phase} checkpoint, "
            f"but no checkpoint path was provided. "
            f"Run Phase {required_phase} first and provide its checkpoint."
        )

    prerequisite_checkpoint_path = Path(prerequisite_checkpoint_path)
    if not prerequisite_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Phase {required_phase} checkpoint not found at: "
            f"{prerequisite_checkpoint_path}"
        )

    ckpt = load_phase_checkpoint(prerequisite_checkpoint_path)

    if ckpt.metadata.phase != required_phase:
        raise ValueError(
            f"Phase {target_phase} requires a Phase {required_phase} checkpoint, "
            f"but the provided checkpoint is from Phase {ckpt.metadata.phase}. "
            f"Provide a valid Phase {required_phase} checkpoint."
        )

    logger.info(
        "Phase %d prerequisite validated: Phase %d checkpoint (epoch %d, "
        "val_loss=%.4f)",
        target_phase,
        ckpt.metadata.phase,
        ckpt.metadata.epoch,
        ckpt.metadata.best_val_loss,
    )

    return ckpt
