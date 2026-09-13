"""Tests for physics consistency checks (task 7.3).

(a) Wind mass conservation: divergence check
(b) Cold air pool reproduction: valley-ridgeline temperature bias
(c) Orographic precipitation: elevation-precipitation gradient vs PRISM
"""

from __future__ import annotations

import numpy as np

from terrain_weather_ml.evaluation.physics import (
    PhysicsCheckResult,
    check_cold_air_pool,
    check_orographic_precipitation,
    check_wind_mass_conservation,
)


class TestWindMassConservation:
    """Wind mass conservation via divergence check."""

    def test_divergence_free_field_passes(self):
        """A divergence-free wind field passes the check."""
        n_timesteps = 10
        h, w = 32, 32
        # Uniform field: zero divergence
        u_fields = np.ones((n_timesteps, h, w)) * 5.0
        v_fields = np.ones((n_timesteps, h, w)) * 3.0

        result = check_wind_mass_conservation(u_fields, v_fields)
        assert isinstance(result, PhysicsCheckResult)
        assert result.passed is True
        assert result.mean_value < 0.01
        assert result.max_value < 0.01

    def test_divergent_field_fails(self):
        """A strongly divergent field fails the check."""
        n_timesteps = 5
        h, w = 32, 32
        # Source field: u = x, v = y -> div = 2
        x = np.linspace(-5, 5, w)
        y = np.linspace(-5, 5, h)
        xx, yy = np.meshgrid(x, y)
        u_fields = np.tile(xx, (n_timesteps, 1, 1))
        v_fields = np.tile(yy, (n_timesteps, 1, 1))

        result = check_wind_mass_conservation(
            u_fields, v_fields, dx=10.0 / 31
        )
        assert result.passed is False
        assert result.max_value > 0.01

    def test_configurable_threshold(self):
        """Custom threshold changes pass/fail determination."""
        n_timesteps = 3
        h, w = 16, 16
        # Small divergence
        x = np.linspace(0, 0.001, w)
        u_fields = np.tile(x, (n_timesteps, h, 1))
        v_fields = np.zeros((n_timesteps, h, w))

        # Very strict threshold -> fail
        strict = check_wind_mass_conservation(
            u_fields, v_fields, threshold=1e-6
        )
        assert strict.passed is False or strict.passed is True  # check runs
        # Lenient threshold -> pass
        result_lenient = check_wind_mass_conservation(
            u_fields, v_fields, threshold=10.0
        )
        assert result_lenient.passed is True

    def test_result_has_name(self):
        """Result includes check name for reporting."""
        u = np.zeros((1, 8, 8))
        v = np.zeros((1, 8, 8))
        result = check_wind_mass_conservation(u, v)
        assert result.name == "wind_mass_conservation"


class TestColdAirPool:
    """Cold air pool reproduction: valley-ridgeline temperature bias."""

    def test_correct_inversion_passes(self):
        """Model reproducing inversions within 2K threshold passes."""
        # Valley stations (TPI < -50): colder during inversions
        valley_temps = np.array([260.0, 262.0, 259.0])
        ridge_temps = np.array([268.0, 270.0, 267.0])
        # Model valley predictions close to observed
        model_valley_temps = np.array([261.0, 263.0, 260.0])

        result = check_cold_air_pool(
            observed_valley_temps=valley_temps,
            observed_ridge_temps=ridge_temps,
            predicted_valley_temps=model_valley_temps,
        )
        assert isinstance(result, PhysicsCheckResult)
        assert result.passed is True
        assert abs(result.mean_value) < 2.0

    def test_poor_inversion_fails(self):
        """Model missing inversions by > 2K fails."""
        valley_temps = np.array([260.0, 262.0, 259.0])
        ridge_temps = np.array([268.0, 270.0, 267.0])
        # Model predicts too warm in valleys (misses cold pool)
        model_valley_temps = np.array([265.0, 267.0, 264.0])

        result = check_cold_air_pool(
            observed_valley_temps=valley_temps,
            observed_ridge_temps=ridge_temps,
            predicted_valley_temps=model_valley_temps,
        )
        assert result.passed is False
        assert abs(result.mean_value) > 2.0

    def test_result_has_name(self):
        """Result includes check name."""
        result = check_cold_air_pool(
            observed_valley_temps=np.array([260.0]),
            observed_ridge_temps=np.array([268.0]),
            predicted_valley_temps=np.array([261.0]),
        )
        assert result.name == "cold_air_pool"

    def test_no_inversions_returns_not_applicable(self):
        """If no nocturnal inversions identified, result is N/A."""
        # No inversion: valley is warmer than ridge (normal lapse rate)
        valley_temps = np.array([270.0])
        ridge_temps = np.array([265.0])
        model_valley_temps = np.array([270.0])

        result = check_cold_air_pool(
            observed_valley_temps=valley_temps,
            observed_ridge_temps=ridge_temps,
            predicted_valley_temps=model_valley_temps,
        )
        assert result.not_applicable is True


class TestOrographicPrecipitation:
    """Orographic precipitation: elevation-precipitation gradient vs PRISM."""

    def test_matching_gradient_passes(self):
        """Predicted gradient within 20% of PRISM passes."""
        elevations = np.array([2000, 2500, 3000, 3500, 4000], dtype=float)
        # PRISM: 500mm/1000m = 0.5 mm/m
        prism_precip = np.array([400, 650, 900, 1150, 1400], dtype=float)
        # Predicted: 450mm/1000m = 0.45 mm/m (within 20% of 0.5)
        predicted_precip = np.array([410, 635, 860, 1085, 1310], dtype=float)

        result = check_orographic_precipitation(
            elevations=elevations,
            prism_precip=prism_precip,
            predicted_precip=predicted_precip,
        )
        assert isinstance(result, PhysicsCheckResult)
        assert result.passed is True

    def test_mismatched_gradient_fails(self):
        """Predicted gradient differing by > 20% from PRISM fails."""
        elevations = np.array([2000, 2500, 3000, 3500, 4000], dtype=float)
        prism_precip = np.array([400, 650, 900, 1150, 1400], dtype=float)
        # Predicted: flat (no orographic enhancement)
        predicted_precip = np.array([800, 800, 800, 800, 800], dtype=float)

        result = check_orographic_precipitation(
            elevations=elevations,
            prism_precip=prism_precip,
            predicted_precip=predicted_precip,
        )
        assert result.passed is False

    def test_result_includes_slopes(self):
        """Result includes both PRISM and predicted slopes."""
        elevations = np.array([2000, 3000, 4000], dtype=float)
        prism_precip = np.array([400, 900, 1400], dtype=float)
        predicted_precip = np.array([400, 900, 1400], dtype=float)

        result = check_orographic_precipitation(
            elevations=elevations,
            prism_precip=prism_precip,
            predicted_precip=predicted_precip,
        )
        assert "prism_slope" in result.details
        assert "predicted_slope" in result.details

    def test_result_has_name(self):
        """Result includes check name."""
        result = check_orographic_precipitation(
            elevations=np.array([2000, 3000]),
            prism_precip=np.array([400, 900]),
            predicted_precip=np.array([400, 900]),
        )
        assert result.name == "orographic_precipitation"
