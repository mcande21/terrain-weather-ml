"""Tests for AnomalyNormalizer integration with training scripts.

Verifies the full training/checkpoint/validation flow and the
Phase 3 temporal split + month-day HRRR matching.
"""

import numpy as np
import pytest
import torch

from terrain_weather_ml.training.normalizer import AnomalyNormalizer


class TestAnomalyTrainingFlow:
    """Test the training -> checkpoint -> validation normalization flow."""

    def test_training_checkpoint_validation_roundtrip(self):
        """Full flow: fit, normalize, save, load, denormalize recovers values."""
        n_samples = 100
        c_out = 4
        target_scalars = torch.randn(n_samples, c_out) * 20 + 273.0
        station_ids = [f"stn_{i % 10}" for i in range(n_samples)]
        months = [(i % 12) + 1 for i in range(n_samples)]

        normalizer = AnomalyNormalizer()
        normalizer.fit(target_scalars, station_ids, months)

        normalized = normalizer.normalize(target_scalars, station_ids, months)
        assert normalized.abs().mean() < target_scalars.abs().mean()

        state = normalizer.state_dict()
        checkpoint = {"extra_data": {"normalizer": state}}

        loaded = AnomalyNormalizer()
        loaded.load_state_dict(checkpoint["extra_data"]["normalizer"])

        pred_norm = normalized[:5]
        pred_sids = station_ids[:5]
        pred_months = months[:5]
        denormalized = loaded.denormalize(pred_norm, pred_sids, pred_months)

        assert torch.allclose(denormalized, target_scalars[:5], atol=1e-4)

    def test_subsampled_metadata_stays_aligned(self):
        """After subsampling, station_ids/months must remain aligned with targets."""
        n = 200
        targets = torch.randn(n, 4)
        station_ids = [f"stn_{i % 20}" for i in range(n)]
        months = [(i % 12) + 1 for i in range(n)]

        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=50, replace=False)
        idx.sort()

        sub_targets = targets[idx]
        sub_sids = [station_ids[i] for i in idx]
        sub_months = [months[i] for i in idx]

        normalizer = AnomalyNormalizer()
        normalizer.fit(sub_targets, sub_sids, sub_months)
        normalized = normalizer.normalize(sub_targets, sub_sids, sub_months)
        roundtrip = normalizer.denormalize(normalized, sub_sids, sub_months)
        assert torch.allclose(sub_targets, roundtrip, atol=1e-4)


class TestPhase3TemporalSplit:
    """Test WY-based temporal split and month-day HRRR matching."""

    def test_snotel_date_to_hrrr_key(self):
        """Map SNOTEL dates to HRRR date keys by month-day in WY2024."""
        from terrain_weather_ml.training.temporal_utils import snotel_date_to_hrrr_key

        assert snotel_date_to_hrrr_key("2021-01-15") == "20240115"
        assert snotel_date_to_hrrr_key("2020-07-04") == "20240704"
        assert snotel_date_to_hrrr_key("2022-11-05") == "20231105"
        assert snotel_date_to_hrrr_key("2019-10-01") == "20231001"
        assert snotel_date_to_hrrr_key("2023-09-30") == "20240930"

    def test_snotel_date_to_hrrr_key_feb29(self):
        """Feb 29 from non-leap years doesn't exist, but leap year dates map."""
        from terrain_weather_ml.training.temporal_utils import snotel_date_to_hrrr_key

        # 2020 is a leap year, maps to 2024 (also leap)
        assert snotel_date_to_hrrr_key("2020-02-29") == "20240229"

    def test_filter_wy2024_dates(self):
        """Filter SNOTEL observations to exclude WY2024."""
        from terrain_weather_ml.training.temporal_utils import is_wy2024_date

        import pandas as pd
        assert not is_wy2024_date(pd.Timestamp("2023-09-30"))
        assert is_wy2024_date(pd.Timestamp("2023-10-01"))
        assert is_wy2024_date(pd.Timestamp("2024-01-15"))
        assert is_wy2024_date(pd.Timestamp("2024-09-30"))
        assert not is_wy2024_date(pd.Timestamp("2024-10-01"))
