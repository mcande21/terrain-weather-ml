"""Task 1.7: Normalization tests.

Verify: round-trip test -- compute stats, save, load from checkpoint,
apply to same data, verify output matches original normalized values exactly.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from terrain_weather_ml.terrain.normalization import TerrainNormalizer
from terrain_weather_ml.terrain.encoder import C_TERRAIN


@pytest.fixture
def sample_tensor():
    """Create a sample terrain tensor with known statistics."""
    torch.manual_seed(42)
    return torch.randn(C_TERRAIN, 50, 60) * 100 + 500


@pytest.fixture
def multi_region_tensors():
    """Create multiple terrain tensors to compute stats from."""
    torch.manual_seed(42)
    return [
        torch.randn(C_TERRAIN, 30, 40) * 100 + 500,
        torch.randn(C_TERRAIN, 30, 40) * 120 + 480,
        torch.randn(C_TERRAIN, 30, 40) * 80 + 520,
    ]


class TestTerrainNormalizer:
    def test_fit_computes_stats(self, multi_region_tensors):
        """fit() should compute per-channel mean and std."""
        normalizer = TerrainNormalizer()
        normalizer.fit(multi_region_tensors)
        assert normalizer.mean.shape == (C_TERRAIN,)
        assert normalizer.std.shape == (C_TERRAIN,)
        assert torch.all(normalizer.std > 0)

    def test_normalize_produces_zero_mean_unit_var(self, multi_region_tensors):
        """After normalization, stats should be approximately zero-mean unit-var."""
        normalizer = TerrainNormalizer()
        normalizer.fit(multi_region_tensors)

        # Normalize all tensors and check aggregate stats
        all_normalized = torch.cat(
            [normalizer.normalize(t) for t in multi_region_tensors], dim=-1
        )
        for c in range(C_TERRAIN):
            channel = all_normalized[c]
            assert abs(channel.mean().item()) < 0.2
            assert abs(channel.std().item() - 1.0) < 0.2

    def test_normalize_output_shape(self, sample_tensor):
        """Normalized tensor should have same shape as input."""
        normalizer = TerrainNormalizer()
        normalizer.fit([sample_tensor])
        result = normalizer.normalize(sample_tensor)
        assert result.shape == sample_tensor.shape

    def test_save_load_roundtrip(self, multi_region_tensors, tmp_path):
        """Save and load should preserve normalization parameters exactly."""
        checkpoint_path = tmp_path / "norm_checkpoint.pt"

        normalizer1 = TerrainNormalizer()
        normalizer1.fit(multi_region_tensors)
        normalizer1.save(checkpoint_path)

        normalizer2 = TerrainNormalizer()
        normalizer2.load(checkpoint_path)

        torch.testing.assert_close(normalizer1.mean, normalizer2.mean)
        torch.testing.assert_close(normalizer1.std, normalizer2.std)

    def test_loaded_normalizer_matches_original(self, multi_region_tensors, tmp_path):
        """Normalization with loaded params should match original exactly."""
        checkpoint_path = tmp_path / "norm_checkpoint.pt"
        test_tensor = multi_region_tensors[0]

        normalizer1 = TerrainNormalizer()
        normalizer1.fit(multi_region_tensors)
        result1 = normalizer1.normalize(test_tensor)
        normalizer1.save(checkpoint_path)

        normalizer2 = TerrainNormalizer()
        normalizer2.load(checkpoint_path)
        result2 = normalizer2.normalize(test_tensor)

        torch.testing.assert_close(result1, result2)

    def test_denormalize_inverts_normalize(self, multi_region_tensors):
        """denormalize(normalize(x)) should recover x."""
        normalizer = TerrainNormalizer()
        normalizer.fit(multi_region_tensors)

        original = multi_region_tensors[0]
        normalized = normalizer.normalize(original)
        recovered = normalizer.denormalize(normalized)

        torch.testing.assert_close(recovered, original, atol=1e-5, rtol=1e-5)

    def test_state_dict_integration(self, multi_region_tensors, tmp_path):
        """state_dict should be embeddable in a model checkpoint."""
        normalizer = TerrainNormalizer()
        normalizer.fit(multi_region_tensors)

        # Simulate embedding in a model checkpoint
        model_checkpoint = {
            "model_state": {"dummy": torch.zeros(1)},
            "terrain_normalizer": normalizer.state_dict(),
        }
        checkpoint_path = tmp_path / "model.pt"
        torch.save(model_checkpoint, checkpoint_path)

        # Restore from checkpoint
        loaded = torch.load(checkpoint_path, weights_only=True)
        normalizer2 = TerrainNormalizer()
        normalizer2.load_state_dict(loaded["terrain_normalizer"])

        torch.testing.assert_close(normalizer.mean, normalizer2.mean)
        torch.testing.assert_close(normalizer.std, normalizer2.std)

    def test_normalize_before_fit_raises(self, sample_tensor):
        """Attempting to normalize without fitting should raise."""
        normalizer = TerrainNormalizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            normalizer.normalize(sample_tensor)
