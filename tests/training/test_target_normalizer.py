"""Tests for TargetNormalizer: z-score normalization for training targets."""

import torch
import pytest

from terrain_weather_ml.training.normalizer import TargetNormalizer


class TestTargetNormalizerFit:
    def test_fit_computes_mean_and_std(self):
        normalizer = TargetNormalizer()
        targets = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
            [[100.0, 200.0], [300.0, 400.0]],
        ]).unsqueeze(0).expand(5, -1, -1, -1)  # (5, 3, 2, 2)
        normalizer.fit(targets)
        assert normalizer.mean.shape == (3,)
        assert normalizer.std.shape == (3,)
        assert torch.isclose(normalizer.mean[0], torch.tensor(2.5))
        assert torch.isclose(normalizer.mean[1], torch.tensor(25.0))

    def test_fit_clamps_std_to_avoid_division_by_zero(self):
        normalizer = TargetNormalizer()
        targets = torch.ones(4, 3, 8, 8) * 273.15
        normalizer.fit(targets)
        assert (normalizer.std >= 1e-6).all()

    def test_fit_from_list_of_tensors(self):
        normalizer = TargetNormalizer()
        t1 = torch.zeros(1, 4, 8, 8)
        t2 = torch.ones(1, 4, 8, 8) * 10.0
        normalizer.fit(torch.cat([t1, t2], dim=0))
        assert normalizer.mean.shape == (4,)


class TestNormalizeDenormalize:
    def test_normalize_produces_zero_mean_unit_variance(self):
        normalizer = TargetNormalizer()
        torch.manual_seed(42)
        targets = torch.randn(100, 4, 8, 8) * 50 + 273.0
        normalizer.fit(targets)
        normed = normalizer.normalize(targets)
        per_channel_mean = normed.mean(dim=(0, 2, 3))
        per_channel_std = normed.std(dim=(0, 2, 3))
        assert torch.allclose(per_channel_mean, torch.zeros(4), atol=0.1)
        assert torch.allclose(per_channel_std, torch.ones(4), atol=0.1)

    def test_denormalize_inverts_normalize(self):
        normalizer = TargetNormalizer()
        torch.manual_seed(42)
        targets = torch.randn(10, 4, 16, 16) * 100 + 270.0
        normalizer.fit(targets)
        roundtrip = normalizer.denormalize(normalizer.normalize(targets))
        assert torch.allclose(targets, roundtrip, atol=1e-4)

    def test_normalize_denormalize_single_sample(self):
        normalizer = TargetNormalizer()
        targets = torch.randn(20, 4, 8, 8) * 10 + 5.0
        normalizer.fit(targets)
        single = targets[0:1]
        roundtrip = normalizer.denormalize(normalizer.normalize(single))
        assert torch.allclose(single, roundtrip, atol=1e-5)


class TestStateDict:
    def test_state_dict_roundtrip(self):
        normalizer = TargetNormalizer()
        torch.manual_seed(0)
        targets = torch.randn(50, 4, 8, 8) * 30 + 260.0
        normalizer.fit(targets)

        state = normalizer.state_dict()
        assert "mean" in state
        assert "std" in state

        normalizer2 = TargetNormalizer()
        normalizer2.load_state_dict(state)
        assert torch.equal(normalizer.mean, normalizer2.mean)
        assert torch.equal(normalizer.std, normalizer2.std)

    def test_loaded_normalizer_produces_same_results(self):
        normalizer = TargetNormalizer()
        torch.manual_seed(1)
        targets = torch.randn(20, 4, 8, 8) * 50 + 273.0
        normalizer.fit(targets)

        state = normalizer.state_dict()
        normalizer2 = TargetNormalizer()
        normalizer2.load_state_dict(state)

        sample = targets[0:1]
        assert torch.equal(
            normalizer.normalize(sample),
            normalizer2.normalize(sample),
        )
        assert torch.equal(
            normalizer.denormalize(sample),
            normalizer2.denormalize(sample),
        )

    def test_state_dict_values_are_plain_tensors(self):
        normalizer = TargetNormalizer()
        normalizer.fit(torch.randn(10, 2, 4, 4))
        state = normalizer.state_dict()
        assert isinstance(state["mean"], torch.Tensor)
        assert isinstance(state["std"], torch.Tensor)


class TestEdgeCases:
    def test_raises_before_fit(self):
        normalizer = TargetNormalizer()
        with pytest.raises(RuntimeError):
            normalizer.normalize(torch.randn(1, 4, 8, 8))
        with pytest.raises(RuntimeError):
            normalizer.denormalize(torch.randn(1, 4, 8, 8))

    def test_device_transfer(self):
        normalizer = TargetNormalizer()
        targets = torch.randn(10, 4, 8, 8)
        normalizer.fit(targets)
        normed = normalizer.normalize(targets)
        assert normed.device == targets.device
