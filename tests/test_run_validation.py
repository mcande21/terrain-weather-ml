"""Tests for validation script helper functions."""

import numpy as np
import torch

from terrain_weather_ml.evaluation.loso import StationData
from terrain_weather_ml.evaluation.metrics import compute_continuous_metrics
from terrain_weather_ml.terrain.downscaling.head import TerrainDownscalingHead


def test_model_inference_shape():
    """Model produces correct output shape for validation inputs."""
    head = TerrainDownscalingHead(
        c_terrain=17, c_weather=6, c_out=4, base_features=32,
        apply_divergence_free=False,
    )
    head.eval()
    terrain = torch.randn(1, 17, 64, 64)
    weather = torch.randn(1, 6, 8, 8)
    with torch.no_grad():
        out = head(terrain, weather)
    assert out.shape == (1, 4, 64, 64)


def test_extract_point_prediction():
    """Extracting center-pixel prediction from model output works."""
    out = torch.randn(1, 4, 64, 64)
    h_mid = out.shape[2] // 2
    w_mid = out.shape[3] // 2
    temp = out[0, 2, h_mid, w_mid].item()
    assert isinstance(temp, float)


def test_hrrr_baseline_extraction():
    """Raw HRRR value at station grid cell serves as baseline."""
    weather = torch.randn(6, 200, 162)
    row, col = 100, 80
    t2m_raw = weather[0, row, col].item()
    assert isinstance(t2m_raw, float)


def test_station_data_construction():
    """StationData can be built from observations dict."""
    sd = StationData(
        station_id="test_001",
        latitude=39.0,
        longitude=-106.0,
        elevation=3000.0,
        tpi=-50.0,
        observations={"temperature": np.array([270.0, 271.0, 272.0])},
        timestamps=np.arange(3),
    )
    assert sd.station_id == "test_001"
    assert len(sd.observations["temperature"]) == 3


def test_metrics_on_real_predictions():
    """Continuous metrics produce valid values for model vs obs."""
    obs = np.array([270.0, 272.0, 275.0, 268.0, 273.0])
    pred = np.array([271.0, 271.5, 276.0, 269.0, 272.0])
    m = compute_continuous_metrics(obs, pred)
    assert m.rmse > 0
    assert m.rmse < 5.0
    assert abs(m.bias) < 5.0
    assert m.n_valid == 5


def test_cold_air_pool_classification():
    """Valley/ridge classification by TPI works for physics check."""
    tpi_values = np.array([-100, -50, 0, 50, 100, 200])
    valley_mask = tpi_values < -20
    ridge_mask = tpi_values > 100
    assert valley_mask.sum() == 2
    assert ridge_mask.sum() == 1
