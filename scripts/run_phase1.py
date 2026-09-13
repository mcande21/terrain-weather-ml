#!/usr/bin/env python3
"""Phase 1 CFD pre-training: train the downscaling head on real wind field data.

Loads zarr cases from data/cfd/data/, applies domain randomization to expand
the effective dataset, and trains a terrain-only downscaling head to predict
near-surface (u, v) wind from terrain features.
"""

import logging
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from terrain_weather_ml.data.cfd_pipeline import (
    augment_sample,
    format_cfd_tensors,
    load_zarr_case,
)
from terrain_weather_ml.training.checkpoint import load_phase_checkpoint
from terrain_weather_ml.training.datasets import CFDPhaseDataset
from terrain_weather_ml.training.phases import Phase1Trainer, PhaseConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

ZARR_DIR = PROJECT_ROOT / "data" / "cfd" / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "phase1"
AUGMENTED_SAMPLES_PER_CASE = 50
CROP_SIZE = 128
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
LAMBDA_DIV = 0.1
BASE_FEATURES = 32


def main() -> None:
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Device: %s", device_name)

    zarr_paths = sorted(ZARR_DIR.glob("*.zarr"))
    if not zarr_paths:
        logger.error("No zarr cases found in %s", ZARR_DIR)
        sys.exit(1)

    logger.info("Found %d zarr cases", len(zarr_paths))

    base_samples = []
    for zp in zarr_paths:
        logger.info("Loading %s", zp.name)
        raw = load_zarr_case(zp)
        sample = format_cfd_tensors(raw)
        logger.info(
            "  %s: terrain %s, wind %s, bc %s",
            sample.case_id,
            sample.terrain_features.shape,
            sample.wind_field.shape,
            sample.boundary_conditions.tolist(),
        )
        base_samples.append(sample)

    logger.info(
        "Generating %d augmented samples per case (%d total)",
        AUGMENTED_SAMPLES_PER_CASE,
        AUGMENTED_SAMPLES_PER_CASE * len(base_samples),
    )

    dataset_samples = []
    for base in base_samples:
        for _ in range(AUGMENTED_SAMPLES_PER_CASE):
            aug = augment_sample(base, min_crop_size=CROP_SIZE)
            _, h, w = aug.terrain_features.shape
            top = torch.randint(0, max(1, h - CROP_SIZE + 1), (1,)).item()
            left = torch.randint(0, max(1, w - CROP_SIZE + 1), (1,)).item()
            terrain = aug.terrain_features[
                :, top:top + CROP_SIZE, left:left + CROP_SIZE
            ].clone()
            wind_uv = aug.wind_field[
                :2, top:top + CROP_SIZE, left:left + CROP_SIZE
            ].clone()
            dataset_samples.append({
                "terrain": terrain,
                "wind_target": wind_uv,
            })

    logger.info(
        "Dataset: %d samples, terrain %s, wind_target %s",
        len(dataset_samples),
        dataset_samples[0]["terrain"].shape,
        dataset_samples[0]["wind_target"].shape,
    )

    dataset = CFDPhaseDataset(dataset_samples)

    config = PhaseConfig(
        phase=1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lambda_div=LAMBDA_DIV,
        checkpoint_dir=str(CHECKPOINT_DIR),
        device=device_name,
        base_features=BASE_FEATURES,
        c_terrain=1,
        c_weather=0,
        c_out=2,
    )

    trainer = Phase1Trainer(config)
    param_count = sum(p.numel() for p in trainer.head.parameters())
    logger.info("Model: %d parameters (%.2f MB)", param_count, param_count * 4 / 1e6)

    logger.info("Training: %d epochs, batch_size=%d, lr=%s", EPOCHS, BATCH_SIZE, LEARNING_RATE)
    t0 = time.time()
    checkpoint_path = trainer.train(dataset)
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print("Phase 1 Training Complete")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Time: {elapsed:.1f}s ({elapsed / EPOCHS:.1f}s/epoch)")
    print()
    print("Loss curve:")
    for i, loss in enumerate(trainer.epoch_losses):
        marker = " <-- best" if loss == min(trainer.epoch_losses) else ""
        print(f"  Epoch {i + 1:2d}: {loss:.6f}{marker}")

    initial = trainer.epoch_losses[0]
    final = trainer.epoch_losses[-1]
    print()
    print(f"Initial loss: {initial:.6f}")
    print(f"Final loss:   {final:.6f}")
    if initial > 0:
        print(f"Reduction:    {(initial - final) / initial * 100:.1f}%")

    ckpt = load_phase_checkpoint(checkpoint_path)
    print()
    print(f"Checkpoint verified: phase={ckpt.metadata.phase}, "
          f"epoch={ckpt.metadata.epoch}, "
          f"val_loss={ckpt.metadata.best_val_loss:.6f}")
    print(f"Checkpoint hash: {ckpt.compute_hash()[:16]}")


if __name__ == "__main__":
    main()
