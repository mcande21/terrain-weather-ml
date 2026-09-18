"""Tests for terrain feature extraction from station coordinates."""

from pathlib import Path

import pandas as pd
import pytest

from terrain_weather_ml.terrain import C_TERRAIN, CHANNEL_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEM_PATH = PROJECT_ROOT / "data" / "dem" / "colorado_snotel_utm.tif"

SAMPLE_STATIONS = pd.DataFrame({
    "name": ["Berthoud Summit", "Red Mountain Pass", "Joe Wright"],
    "station_id": [335, 713, 551],
    "latitude": [39.7967, 37.8994, 40.5350],
    "longitude": [-105.7781, -107.7131, -105.8881],
    "elevation_ft": [11300, 11060, 10140],
})


@pytest.fixture
def feature_extractor():
    from terrain_weather_ml.terrain.feature_extraction import TerrainFeatureExtractor
    return TerrainFeatureExtractor(str(DEM_PATH))


@pytest.mark.skipif(not DEM_PATH.exists(), reason="DEM not available")
class TestTerrainFeatureExtraction:

    def test_extract_returns_correct_shape(self, feature_extractor):
        result = feature_extractor.extract_features(SAMPLE_STATIONS)
        assert len(result) == 3
        expected_cols = 1 + 1 + C_TERRAIN + C_TERRAIN * 3
        assert len(result.columns) >= expected_cols

    def test_point_features_present(self, feature_extractor):
        result = feature_extractor.extract_features(SAMPLE_STATIONS)
        for name in CHANNEL_NAMES:
            assert name in result.columns, f"Missing point feature: {name}"

    def test_area_features_present(self, feature_extractor):
        result = feature_extractor.extract_features(SAMPLE_STATIONS)
        for name in CHANNEL_NAMES:
            for stat in ["mean", "max", "std"]:
                col = f"{name}_{stat}"
                assert col in result.columns, f"Missing area feature: {col}"

    def test_elevation_reasonable(self, feature_extractor):
        result = feature_extractor.extract_features(SAMPLE_STATIONS)
        for _, row in result.iterrows():
            assert 2000 < row["elevation"] < 5000, (
                f"Elevation {row['elevation']:.0f}m outside expected range"
            )

    def test_svf_range(self, feature_extractor):
        result = feature_extractor.extract_features(SAMPLE_STATIONS)
        assert (result["svf"] >= 0).all() and (result["svf"] <= 1).all()

    def test_slope_nonnegative(self, feature_extractor):
        result = feature_extractor.extract_features(SAMPLE_STATIONS)
        assert (result["slope"] >= 0).all()

    def test_station_metadata_preserved(self, feature_extractor):
        result = feature_extractor.extract_features(SAMPLE_STATIONS)
        assert list(result["station_id"]) == [335, 713, 551]
        assert list(result["name"]) == [
            "Berthoud Summit", "Red Mountain Pass", "Joe Wright"
        ]
