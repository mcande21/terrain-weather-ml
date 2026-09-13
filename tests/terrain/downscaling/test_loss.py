"""Tests for loss function (task 5.6)."""

import torch

from terrain_weather_ml.terrain.downscaling.loss import (
    DownscalingLoss,
    orographic_precipitation_penalty,
)


class TestDownscalingLoss:
    """MSE + divergence penalty + optional orographic penalty."""

    def test_basic_loss_computation(self):
        """Loss includes MSE and divergence terms."""
        loss_fn = DownscalingLoss(lambda_div=0.1, lambda_oro=0.0)
        torch.manual_seed(42)
        pred = torch.randn(2, 4, 16, 16, requires_grad=True)
        target = torch.randn(2, 4, 16, 16)
        terrain = torch.randn(2, 17, 16, 16)

        result = loss_fn(pred, target, terrain)
        assert "total" in result
        assert "mse" in result
        assert "divergence" in result
        assert result["total"].item() > 0

    def test_mse_only_when_lambda_zero(self):
        """lambda_div=0 and lambda_oro=0 produces MSE-only loss."""
        loss_fn = DownscalingLoss(lambda_div=0.0, lambda_oro=0.0)
        torch.manual_seed(42)
        pred = torch.randn(2, 4, 16, 16, requires_grad=True)
        target = torch.randn(2, 4, 16, 16)
        terrain = torch.randn(2, 17, 16, 16)

        result = loss_fn(pred, target, terrain)

        # Total should equal MSE
        assert torch.allclose(result["total"], result["mse"])

    def test_divergence_penalty_nonzero(self):
        """Divergence penalty is nonzero for random predictions."""
        loss_fn = DownscalingLoss(lambda_div=0.1, lambda_oro=0.0)
        torch.manual_seed(42)
        pred = torch.randn(2, 4, 16, 16, requires_grad=True)
        target = torch.randn(2, 4, 16, 16)
        terrain = torch.randn(2, 17, 16, 16)

        result = loss_fn(pred, target, terrain)
        assert result["divergence"].item() > 0

    def test_lambda_div_scales_divergence(self):
        """Higher lambda_div increases total loss."""
        torch.manual_seed(42)
        pred = torch.randn(2, 4, 16, 16, requires_grad=True)
        target = torch.randn(2, 4, 16, 16)
        terrain = torch.randn(2, 17, 16, 16)

        loss_low = DownscalingLoss(lambda_div=0.01)
        loss_high = DownscalingLoss(lambda_div=1.0)

        r_low = loss_low(pred, target, terrain)
        r_high = loss_high(pred, target, terrain)

        # Same MSE, but higher lambda_div increases total
        assert r_high["total"].item() > r_low["total"].item()

    def test_orographic_penalty_when_enabled(self):
        """lambda_oro > 0 adds orographic penalty to total loss."""
        loss_fn = DownscalingLoss(lambda_div=0.0, lambda_oro=0.5)
        torch.manual_seed(42)
        pred = torch.randn(2, 4, 16, 16, requires_grad=True)
        target = torch.randn(2, 4, 16, 16)
        terrain = torch.randn(2, 17, 16, 16)

        result = loss_fn(pred, target, terrain)
        assert "orographic" in result
        assert result["orographic"].item() >= 0

        # Total = MSE + oro penalty
        expected_total = result["mse"] + 0.5 * result["orographic"]
        assert torch.allclose(result["total"], expected_total, atol=1e-5)

    def test_per_variable_weights(self):
        """Custom per-variable weights change MSE computation."""
        # Weight wind heavily, ignore temp/precip
        weights = [10.0, 10.0, 0.0, 0.0]
        loss_fn = DownscalingLoss(variable_weights=weights, lambda_div=0.0)
        torch.manual_seed(42)
        pred = torch.randn(2, 4, 16, 16, requires_grad=True)
        target = torch.randn(2, 4, 16, 16)
        terrain = torch.randn(2, 17, 16, 16)

        # Compare with equal weights
        loss_equal = DownscalingLoss(lambda_div=0.0)
        r_weighted = loss_fn(pred, target, terrain)
        r_equal = loss_equal(pred, target, terrain)

        assert not torch.allclose(r_weighted["mse"], r_equal["mse"])

    def test_gradient_flows_through_all_terms(self):
        """Gradients flow through MSE, divergence, and orographic terms."""
        loss_fn = DownscalingLoss(lambda_div=0.1, lambda_oro=0.1)
        torch.manual_seed(42)
        pred = torch.randn(2, 4, 16, 16, requires_grad=True)
        target = torch.randn(2, 4, 16, 16)
        terrain = torch.randn(2, 17, 16, 16)

        result = loss_fn(pred, target, terrain)
        result["total"].backward()

        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_default_lambda_values(self):
        """Default lambda_div=0.1, lambda_oro=0.0."""
        loss_fn = DownscalingLoss()
        assert loss_fn.lambda_div == 0.1
        assert loss_fn.lambda_oro == 0.0


class TestOrographicPenalty:
    """Orographic precipitation penalty computation."""

    def test_penalty_shape(self):
        """Penalty is a scalar."""
        torch.manual_seed(42)
        pred = torch.randn(2, 4, 16, 16)
        terrain = torch.randn(2, 17, 16, 16)

        penalty = orographic_precipitation_penalty(pred, terrain)
        assert penalty.dim() == 0  # scalar

    def test_penalty_nonnegative(self):
        """Penalty is non-negative."""
        torch.manual_seed(42)
        pred = torch.randn(2, 4, 16, 16)
        terrain = torch.randn(2, 17, 16, 16)

        penalty = orographic_precipitation_penalty(pred, terrain)
        assert penalty.item() >= 0
