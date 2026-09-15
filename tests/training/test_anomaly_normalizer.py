"""Tests for AnomalyNormalizer: anomaly-based target normalization.

Removes monthly station climatology before z-scoring residuals,
per WeatherBench standard. Prevents seasonal-cycle bias.
"""

import pytest
import torch

from terrain_weather_ml.training.normalizer import AnomalyNormalizer


class TestAnomalyNormalizerFit:
    def test_fit_computes_climatology_and_std(self):
        """After fit, climatology and residual_std should be populated."""
        norm = AnomalyNormalizer()
        targets = torch.randn(20, 3)
        station_ids = ["S1"] * 10 + ["S2"] * 10
        months = [1] * 5 + [7] * 5 + [1] * 5 + [7] * 5
        norm.fit(targets, station_ids, months)
        assert norm.climatology is not None
        assert norm.residual_std is not None

    def test_fit_climatology_keys(self):
        """Climatology should have an entry per unique (station, month) pair."""
        norm = AnomalyNormalizer()
        targets = torch.randn(12, 2)
        station_ids = ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"]
        months = [1, 1, 7, 7, 1, 1, 7, 7, 1, 1, 7, 7]
        norm.fit(targets, station_ids, months)
        # 3 stations x 2 months = 6 keys
        assert len(norm.climatology) == 6

    def test_fit_climatology_values_are_per_channel_means(self):
        """Each climatology entry should be the mean of samples in that group."""
        norm = AnomalyNormalizer()
        # 4 samples, 2 channels. Station "X", month 3, all same values.
        targets = torch.tensor([
            [10.0, 20.0],
            [12.0, 22.0],
            [14.0, 24.0],
            [16.0, 26.0],
        ])
        station_ids = ["X"] * 4
        months = [3] * 4
        norm.fit(targets, station_ids, months)
        clim = norm.climatology[("X", 3)]
        assert torch.isclose(clim[0], torch.tensor(13.0))
        assert torch.isclose(clim[1], torch.tensor(23.0))

    def test_fit_residual_std_shape(self):
        """Residual std should have one value per channel."""
        norm = AnomalyNormalizer()
        n_channels = 5
        targets = torch.randn(100, n_channels)
        station_ids = [f"S{i % 3}" for i in range(100)]
        months = [(i % 12) + 1 for i in range(100)]
        norm.fit(targets, station_ids, months)
        assert norm.residual_std.shape == (n_channels,)

    def test_fit_residual_std_clamps_minimum(self):
        """Residual std should be clamped to avoid division by zero."""
        norm = AnomalyNormalizer()
        # All samples identical per station-month -> zero residual
        targets = torch.ones(10, 3) * 5.0
        station_ids = ["A"] * 10
        months = [1] * 10
        norm.fit(targets, station_ids, months)
        assert (norm.residual_std >= 1e-6).all()

    def test_fit_validates_input_lengths(self):
        """Mismatched lengths should raise ValueError."""
        norm = AnomalyNormalizer()
        with pytest.raises(ValueError, match="length"):
            norm.fit(torch.randn(10, 3), ["A"] * 9, [1] * 10)
        with pytest.raises(ValueError, match="length"):
            norm.fit(torch.randn(10, 3), ["A"] * 10, [1] * 9)

    def test_fit_handles_4d_input(self):
        """fit() should accept (N, C, 1, 1) shaped targets too."""
        norm = AnomalyNormalizer()
        targets = torch.randn(20, 3, 1, 1)
        station_ids = ["S1"] * 20
        months = [6] * 20
        norm.fit(targets, station_ids, months)
        assert norm.residual_std.shape == (3,)


class TestAnomalyNormalizeDenormalize:
    def test_normalize_removes_climatology(self):
        """Normalized anomalies should center around zero for each station-month."""
        norm = AnomalyNormalizer()
        torch.manual_seed(42)
        n = 200
        # Station A month 1: base temp ~280K
        # Station A month 7: base temp ~300K
        targets_a1 = torch.randn(n, 2) * 3 + torch.tensor([280.0, 5.0])
        targets_a7 = torch.randn(n, 2) * 3 + torch.tensor([300.0, 2.0])
        targets = torch.cat([targets_a1, targets_a7])
        station_ids = ["A"] * (2 * n)
        months = [1] * n + [7] * n

        norm.fit(targets, station_ids, months)
        normed = norm.normalize(targets, station_ids, months)

        # Overall normalized mean should be near zero
        assert torch.allclose(normed.mean(dim=0), torch.zeros(2), atol=0.15)

    def test_denormalize_inverts_normalize(self):
        """Round-trip: denormalize(normalize(x)) == x."""
        norm = AnomalyNormalizer()
        torch.manual_seed(42)
        targets = torch.randn(50, 4) * 20 + 270.0
        station_ids = [f"S{i % 5}" for i in range(50)]
        months = [(i % 12) + 1 for i in range(50)]

        norm.fit(targets, station_ids, months)
        roundtrip = norm.denormalize(
            norm.normalize(targets, station_ids, months),
            station_ids, months,
        )
        assert torch.allclose(targets, roundtrip, atol=1e-4)

    def test_denormalize_single_sample(self):
        """Round-trip works for a single sample."""
        norm = AnomalyNormalizer()
        torch.manual_seed(7)
        targets = torch.randn(30, 3) * 10 + 5.0
        station_ids = [f"S{i % 3}" for i in range(30)]
        months = [(i % 6) + 1 for i in range(30)]

        norm.fit(targets, station_ids, months)
        single_t = targets[0:1]
        single_sid = station_ids[0:1]
        single_m = months[0:1]
        roundtrip = norm.denormalize(
            norm.normalize(single_t, single_sid, single_m),
            single_sid, single_m,
        )
        assert torch.allclose(single_t, roundtrip, atol=1e-5)

    def test_normalize_handles_4d_input(self):
        """normalize/denormalize should work with (N, C, 1, 1) targets."""
        norm = AnomalyNormalizer()
        torch.manual_seed(0)
        targets_2d = torch.randn(20, 3) * 10 + 273.0
        station_ids = ["A"] * 20
        months = [6] * 20

        norm.fit(targets_2d, station_ids, months)

        targets_4d = targets_2d.unsqueeze(-1).unsqueeze(-1)
        normed = norm.normalize(targets_4d, station_ids, months)
        assert normed.shape == (20, 3, 1, 1)

        roundtrip = norm.denormalize(normed, station_ids, months)
        assert torch.allclose(targets_4d, roundtrip, atol=1e-4)

    def test_anomalies_smaller_than_raw_values(self):
        """Anomalies should be substantially smaller than raw physical values."""
        norm = AnomalyNormalizer()
        torch.manual_seed(42)
        n = 100
        # Simulate seasonal temperature: winter ~260K, summer ~300K
        winter = torch.randn(n, 1) * 5 + 260.0
        summer = torch.randn(n, 1) * 5 + 300.0
        targets = torch.cat([winter, summer])
        station_ids = ["A"] * (2 * n)
        months = [1] * n + [7] * n

        norm.fit(targets, station_ids, months)
        normed = norm.normalize(targets, station_ids, months)

        # Raw values ~260-300, normalized should be ~O(1)
        assert normed.abs().mean() < 2.0


class TestAnomalyStateDict:
    def test_state_dict_roundtrip(self):
        """Save/load state_dict preserves all normalization parameters."""
        norm = AnomalyNormalizer()
        torch.manual_seed(0)
        targets = torch.randn(40, 3) * 30 + 260.0
        station_ids = [f"S{i % 4}" for i in range(40)]
        months = [(i % 12) + 1 for i in range(40)]
        norm.fit(targets, station_ids, months)

        state = norm.state_dict()
        assert "climatology" in state
        assert "residual_std" in state

        norm2 = AnomalyNormalizer()
        norm2.load_state_dict(state)
        assert len(norm2.climatology) == len(norm.climatology)
        assert torch.equal(norm.residual_std, norm2.residual_std)

    def test_loaded_normalizer_produces_same_results(self):
        """A loaded normalizer should produce identical normalize/denormalize."""
        norm = AnomalyNormalizer()
        torch.manual_seed(1)
        targets = torch.randn(30, 4) * 50 + 273.0
        station_ids = [f"S{i % 3}" for i in range(30)]
        months = [(i % 6) + 1 for i in range(30)]
        norm.fit(targets, station_ids, months)

        state = norm.state_dict()
        norm2 = AnomalyNormalizer()
        norm2.load_state_dict(state)

        sample = targets[0:5]
        sids = station_ids[0:5]
        ms = months[0:5]

        assert torch.equal(
            norm.normalize(sample, sids, ms),
            norm2.normalize(sample, sids, ms),
        )
        assert torch.equal(
            norm.denormalize(sample, sids, ms),
            norm2.denormalize(sample, sids, ms),
        )

    def test_state_dict_serializable_with_torch_save(self):
        """State dict should be serializable via torch.save/load."""
        import os
        import tempfile

        norm = AnomalyNormalizer()
        targets = torch.randn(20, 2) * 10 + 5.0
        station_ids = ["A"] * 10 + ["B"] * 10
        months = [1] * 10 + [7] * 10
        norm.fit(targets, station_ids, months)

        state = norm.state_dict()
        path = os.path.join(tempfile.mkdtemp(), "norm_state.pt")
        torch.save(state, path)
        loaded = torch.load(path, weights_only=False)

        norm2 = AnomalyNormalizer()
        norm2.load_state_dict(loaded)
        assert torch.equal(norm.residual_std, norm2.residual_std)


class TestAnomalyEdgeCases:
    def test_raises_before_fit(self):
        """Operations should raise RuntimeError before fit()."""
        norm = AnomalyNormalizer()
        with pytest.raises(RuntimeError):
            norm.normalize(torch.randn(5, 3), ["A"] * 5, [1] * 5)
        with pytest.raises(RuntimeError):
            norm.denormalize(torch.randn(5, 3), ["A"] * 5, [1] * 5)
        with pytest.raises(RuntimeError):
            norm.state_dict()

    def test_unknown_station_month_at_normalize_raises(self):
        """normalize/denormalize with unseen (station, month) should raise."""
        norm = AnomalyNormalizer()
        targets = torch.randn(10, 2)
        norm.fit(targets, ["A"] * 10, [1] * 10)

        with pytest.raises(KeyError):
            norm.normalize(torch.randn(1, 2), ["UNKNOWN"], [1])
        with pytest.raises(KeyError):
            norm.denormalize(torch.randn(1, 2), ["A"], [7])

    def test_single_sample_per_group(self):
        """Should handle groups with a single sample (std -> clamp)."""
        norm = AnomalyNormalizer()
        targets = torch.tensor([[10.0, 20.0]])
        norm.fit(targets, ["A"], [1])
        normed = norm.normalize(targets, ["A"], [1])
        roundtrip = norm.denormalize(normed, ["A"], [1])
        assert torch.allclose(targets, roundtrip, atol=1e-5)
