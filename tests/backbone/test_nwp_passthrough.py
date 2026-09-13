"""Tasks 1.1-1.4: NWP passthrough adapter, protocol, interface, and wiring.

Verify:
- 1.1: NWPPassthroughAdapter extracts 6 surface variables from HRRR/ERA5 dicts
- 1.2: WeatherAdapter Protocol defined and satisfied by NWPPassthroughAdapter
- 1.3: Interface contract validates against WeatherAdapter protocol
- 1.4: NWPPassthroughAdapter is primary export from backbone
"""

from __future__ import annotations

import pytest
import torch

from terrain_weather_ml.backbone.stormcast import C_WEATHER


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_hrrr_data(h: int = 32, w: int = 64) -> dict[str, torch.Tensor]:
    """Create mock HRRR data dict with cfgrib-style variable names."""
    return {
        "t2m": torch.randn(h, w),
        "u10": torch.randn(h, w),
        "v10": torch.randn(h, w),
        "prate": torch.randn(h, w).abs(),
        "sp": torch.full((h, w), 101325.0),
        "blh": torch.randn(h, w).abs() * 1000,
    }


def _make_era5_data(h: int = 32, w: int = 64) -> dict[str, torch.Tensor]:
    """Create mock ERA5 data dict with ERA5 short names."""
    return {
        "2t": torch.randn(h, w) + 273.15,
        "10u": torch.randn(h, w),
        "10v": torch.randn(h, w),
        "tp": torch.randn(h, w).abs() * 0.01,  # accumulated meters
        "sp": torch.full((h, w), 101325.0),
        "blh": torch.randn(h, w).abs() * 1000,
    }


# ---------------------------------------------------------------------------
# Task 1.1: NWPPassthroughAdapter
# ---------------------------------------------------------------------------

class TestNWPPassthroughHRRR:
    """Task 1.1: HRRR extraction via NWP passthrough adapter."""

    def test_hrrr_extraction_shape(self):
        """HRRR data dict produces (1, 6, H, W) tensor."""
        from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter

        adapter = NWPPassthroughAdapter()
        data = _make_hrrr_data(32, 64)
        result = adapter.extract(data)
        assert result.shape == (1, C_WEATHER, 32, 64)

    def test_hrrr_variable_ordering(self):
        """Extracted channels match canonical order: T2M, U10, V10, PRATE, SP, BLH."""
        from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter

        adapter = NWPPassthroughAdapter()
        h, w = 8, 8
        data = {
            "t2m": torch.ones(h, w) * 1.0,
            "u10": torch.ones(h, w) * 2.0,
            "v10": torch.ones(h, w) * 3.0,
            "prate": torch.ones(h, w) * 4.0,
            "sp": torch.ones(h, w) * 5.0,
            "blh": torch.ones(h, w) * 6.0,
        }
        result = adapter.extract(data)
        # T2M=1, U10=2, V10=3, PRATE=4, SP=5, BLH=6
        assert result[0, 0, 0, 0].item() == pytest.approx(1.0)
        assert result[0, 1, 0, 0].item() == pytest.approx(2.0)
        assert result[0, 2, 0, 0].item() == pytest.approx(3.0)
        assert result[0, 3, 0, 0].item() == pytest.approx(4.0)
        assert result[0, 4, 0, 0].item() == pytest.approx(5.0)
        assert result[0, 5, 0, 0].item() == pytest.approx(6.0)

    def test_hrrr_output_dtype_float32(self):
        """Output tensor should be float32."""
        from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter

        adapter = NWPPassthroughAdapter()
        result = adapter.extract(_make_hrrr_data())
        assert result.dtype == torch.float32

    def test_hrrr_batch_dimension(self):
        """Single sample gets batch dimension of 1."""
        from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter

        adapter = NWPPassthroughAdapter()
        result = adapter.extract(_make_hrrr_data(16, 16))
        assert result.shape[0] == 1


class TestNWPPassthroughERA5:
    """Task 1.1: ERA5 extraction via NWP passthrough adapter."""

    def test_era5_extraction_shape(self):
        """ERA5 data dict produces (1, 6, H, W) tensor."""
        from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter

        adapter = NWPPassthroughAdapter()
        data = _make_era5_data(24, 48)
        result = adapter.extract(data)
        assert result.shape == (1, C_WEATHER, 24, 48)

    def test_era5_variable_ordering(self):
        """ERA5 variables mapped to canonical order: T2M, U10, V10, PRATE, SP, BLH."""
        from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter

        adapter = NWPPassthroughAdapter()
        h, w = 8, 8
        data = {
            "2t": torch.ones(h, w) * 10.0,
            "10u": torch.ones(h, w) * 20.0,
            "10v": torch.ones(h, w) * 30.0,
            "tp": torch.ones(h, w) * 40.0,
            "sp": torch.ones(h, w) * 50.0,
            "blh": torch.ones(h, w) * 60.0,
        }
        result = adapter.extract(data)
        # T2M=10, U10=20, V10=30, PRATE=converted from tp, SP=50, BLH=60
        assert result[0, 0, 0, 0].item() == pytest.approx(10.0)  # 2t -> T2M
        assert result[0, 1, 0, 0].item() == pytest.approx(20.0)  # 10u -> U10
        assert result[0, 2, 0, 0].item() == pytest.approx(30.0)  # 10v -> V10
        # channel 3 (PRATE) is converted; tested separately
        assert result[0, 4, 0, 0].item() == pytest.approx(50.0)  # sp -> SP
        assert result[0, 5, 0, 0].item() == pytest.approx(60.0)  # blh -> BLH

    def test_era5_precip_accumulation_to_rate(self):
        """ERA5 tp (accumulated meters) converted to rate (kg/m^2/s).

        Conversion: rate = tp * 1000 / dt_seconds
        where 1000 converts meters of water to kg/m^2, dt_seconds is the
        accumulation interval (default 3600s for hourly ERA5).
        """
        from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter

        adapter = NWPPassthroughAdapter()
        h, w = 4, 4
        tp_meters = 0.001  # 1mm accumulated in 1 hour
        data = {
            "2t": torch.zeros(h, w),
            "10u": torch.zeros(h, w),
            "10v": torch.zeros(h, w),
            "tp": torch.full((h, w), tp_meters),
            "sp": torch.zeros(h, w),
            "blh": torch.zeros(h, w),
        }
        result = adapter.extract(data)
        # rate = 0.001m * 1000 kg/m^3 / 3600s = 1.0 / 3600 ~ 2.778e-4 kg/m^2/s
        expected_rate = tp_meters * 1000.0 / 3600.0
        assert result[0, 3, 0, 0].item() == pytest.approx(expected_rate, rel=1e-5)

    def test_era5_precip_custom_interval(self):
        """ERA5 precip conversion with custom accumulation interval."""
        from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter

        adapter = NWPPassthroughAdapter(era5_dt_seconds=10800)  # 3-hour
        h, w = 4, 4
        tp_meters = 0.003
        data = {
            "2t": torch.zeros(h, w),
            "10u": torch.zeros(h, w),
            "10v": torch.zeros(h, w),
            "tp": torch.full((h, w), tp_meters),
            "sp": torch.zeros(h, w),
            "blh": torch.zeros(h, w),
        }
        result = adapter.extract(data)
        expected_rate = tp_meters * 1000.0 / 10800.0
        assert result[0, 3, 0, 0].item() == pytest.approx(expected_rate, rel=1e-5)


class TestNWPPassthroughErrors:
    """Task 1.1: Error handling for unknown/invalid sources."""

    def test_unknown_source_raises(self):
        """Unknown variable names raise ValueError."""
        from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter

        adapter = NWPPassthroughAdapter()
        data = {
            "temperature_2m": torch.randn(8, 8),
            "wind_u_10m": torch.randn(8, 8),
        }
        with pytest.raises(ValueError, match="(?i)recognize|unknown|expected"):
            adapter.extract(data)

    def test_missing_variable_raises(self):
        """Missing required variable raises ValueError."""
        from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter

        adapter = NWPPassthroughAdapter()
        data = {
            "t2m": torch.randn(8, 8),
            "u10": torch.randn(8, 8),
            # missing v10, prate, sp, blh
        }
        with pytest.raises(ValueError, match="(?i)missing"):
            adapter.extract(data)

    def test_mismatched_shapes_raises(self):
        """Variables with different spatial shapes raise ValueError."""
        from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter

        adapter = NWPPassthroughAdapter()
        data = {
            "t2m": torch.randn(8, 8),
            "u10": torch.randn(16, 16),  # wrong shape
            "v10": torch.randn(8, 8),
            "prate": torch.randn(8, 8),
            "sp": torch.randn(8, 8),
            "blh": torch.randn(8, 8),
        }
        with pytest.raises(ValueError, match="(?i)shape"):
            adapter.extract(data)


# ---------------------------------------------------------------------------
# Task 1.2: WeatherAdapter Protocol
# ---------------------------------------------------------------------------

class TestWeatherAdapterProtocol:
    """Task 1.2: WeatherAdapter Protocol definition and compliance."""

    def test_protocol_exists(self):
        """WeatherAdapter Protocol is importable."""
        from terrain_weather_ml.backbone.adapter_protocol import WeatherAdapter
        assert WeatherAdapter is not None

    def test_nwp_passthrough_satisfies_protocol(self):
        """NWPPassthroughAdapter satisfies WeatherAdapter protocol."""
        from terrain_weather_ml.backbone.adapter_protocol import WeatherAdapter
        from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter

        adapter = NWPPassthroughAdapter()
        assert isinstance(adapter, WeatherAdapter)

    def test_protocol_requires_extract(self):
        """WeatherAdapter requires an extract method."""
        from terrain_weather_ml.backbone.adapter_protocol import WeatherAdapter

        class NoExtract:
            pass

        obj = NoExtract()
        assert not isinstance(obj, WeatherAdapter)

    def test_protocol_extract_callable(self):
        """WeatherAdapter.extract is callable on conforming implementations."""
        from terrain_weather_ml.backbone.adapter_protocol import WeatherAdapter
        from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter

        adapter: WeatherAdapter = NWPPassthroughAdapter()
        assert callable(adapter.extract)


# ---------------------------------------------------------------------------
# Task 1.3: Interface contract update
# ---------------------------------------------------------------------------

class TestInterfaceContractUpdate:
    """Task 1.3: Interface validation against WeatherAdapter protocol."""

    def test_nwp_passthrough_passes_contract(self):
        """NWP passthrough output passes validate_interface_contract."""
        from terrain_weather_ml.backbone.interface import validate_interface_contract
        from terrain_weather_ml.backbone.nwp_passthrough import NWPPassthroughAdapter

        adapter = NWPPassthroughAdapter()
        result = adapter.extract(_make_hrrr_data())
        validate_interface_contract(
            weather_channels=result.shape[1],
            terrain_channels=17,
            expected_weather=6,
            expected_terrain=17,
        )

    def test_interface_docstring_not_stormcast_specific(self):
        """Interface module docstring should reference adapters generically."""
        from terrain_weather_ml.backbone import interface
        doc = interface.__doc__ or ""
        # Should not be StormCast-specific anymore
        assert "StormCast" not in doc or "adapter" in doc.lower()


# ---------------------------------------------------------------------------
# Task 1.4: Backbone exports
# ---------------------------------------------------------------------------

class TestBackboneExports:
    """Task 1.4: NWPPassthroughAdapter as primary backbone export."""

    def test_nwp_passthrough_importable_from_backbone(self):
        """NWPPassthroughAdapter importable from terrain_weather_ml.backbone."""
        from terrain_weather_ml.backbone import NWPPassthroughAdapter
        assert NWPPassthroughAdapter is not None

    def test_weather_adapter_importable_from_backbone(self):
        """WeatherAdapter importable from terrain_weather_ml.backbone."""
        from terrain_weather_ml.backbone import WeatherAdapter
        assert WeatherAdapter is not None

    def test_nwp_passthrough_in_all(self):
        """NWPPassthroughAdapter is listed in __all__."""
        import terrain_weather_ml.backbone as backbone
        assert "NWPPassthroughAdapter" in backbone.__all__

    def test_weather_adapter_in_all(self):
        """WeatherAdapter is listed in __all__."""
        import terrain_weather_ml.backbone as backbone
        assert "WeatherAdapter" in backbone.__all__

    def test_stormcast_still_available(self):
        """StormCastAdapter still importable (not removed)."""
        from terrain_weather_ml.backbone import StormCastAdapter
        assert StormCastAdapter is not None
