"""Tests for divergence-free projection layer (task 5.4)."""

import math

import torch

from terrain_weather_ml.terrain.downscaling.divergence import (
    DivergenceFreeProjection,
    compute_divergence,
    divergence_free_projection,
)
from terrain_weather_ml.terrain.downscaling.head import TerrainDownscalingHead


def _spectral_divergence_max(u: torch.Tensor, v: torch.Tensor, dx: float = 1.0) -> float:
    """Compute max spectral divergence excluding Nyquist modes.

    Nyquist modes are excluded because the spectral projection deliberately
    does not correct them (they represent sub-grid aliased structure that
    violates rfft2 conjugate symmetry when projected).
    """
    h, w = u.shape[-2], u.shape[-1]
    u_hat = torch.fft.rfft2(u)
    v_hat = torch.fft.rfft2(v)
    kx = torch.fft.rfftfreq(w, d=dx)
    ky = torch.fft.fftfreq(h, d=dx)
    ky_g, kx_g = torch.meshgrid(ky, kx, indexing="ij")
    div_hat = 2j * math.pi * kx_g * u_hat + 2j * math.pi * ky_g * v_hat

    # Exclude Nyquist kx (last column) and Nyquist ky (row h//2)
    mask = torch.ones_like(div_hat.real, dtype=torch.bool)
    mask[..., -1] = False  # Nyquist kx
    if h % 2 == 0:
        mask[..., h // 2, :] = False  # Nyquist ky

    return div_hat[mask].abs().max().item()


class TestDivergenceComputation:
    """2D divergence via finite differences."""

    def test_uniform_field_zero_divergence(self):
        """Uniform wind field has zero divergence."""
        u = torch.ones(1, 1, 16, 16) * 5.0
        v = torch.ones(1, 1, 16, 16) * 3.0
        div = compute_divergence(u, v)
        assert torch.allclose(div, torch.zeros_like(div), atol=1e-6)

    def test_divergent_field_nonzero(self):
        """Divergent field (source) has positive divergence."""
        # u = x, v = y -> div = du/dx + dv/dy = 1 + 1 = 2
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing="ij",
        )
        u = x.unsqueeze(0).unsqueeze(0)
        v = y.unsqueeze(0).unsqueeze(0)
        div = compute_divergence(u, v, dx=2.0 / 15)
        # Interior points should have divergence ~ 2.0
        interior = div[0, 0, 2:-2, 2:-2]
        assert interior.mean().item() > 1.5

    def test_rotational_field_zero_divergence(self):
        """Pure rotation has zero divergence: u = -y, v = x."""
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, 32),
            torch.linspace(-1, 1, 32),
            indexing="ij",
        )
        u = -y.unsqueeze(0).unsqueeze(0)
        v = x.unsqueeze(0).unsqueeze(0)
        div = compute_divergence(u, v, dx=2.0 / 31)
        assert div.abs().max().item() < 1e-4


def _make_periodic_divergence_free(n: int = 32):
    """Create a periodic divergence-free field from a stream function.

    psi = sin(2*pi*x) * sin(2*pi*y)
    u = dpsi/dy, v = -dpsi/dx  (guaranteed div-free)
    """
    x = torch.linspace(0, 1, n + 1)[:-1]  # periodic: exclude endpoint
    y = torch.linspace(0, 1, n + 1)[:-1]
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    u = (2 * math.pi * torch.sin(2 * math.pi * xx)
         * torch.cos(2 * math.pi * yy))
    v = -(2 * math.pi * torch.cos(2 * math.pi * xx)
          * torch.sin(2 * math.pi * yy))

    return u.unsqueeze(0).unsqueeze(0), v.unsqueeze(0).unsqueeze(0)


def _make_periodic_divergent(n: int = 32):
    """Create a periodic divergent field.

    u = sin(2*pi*x), v = sin(2*pi*y)
    div = 2*pi*cos(2*pi*x) + 2*pi*cos(2*pi*y)
    """
    x = torch.linspace(0, 1, n + 1)[:-1]
    y = torch.linspace(0, 1, n + 1)[:-1]
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    u = torch.sin(2 * math.pi * xx)
    v = torch.sin(2 * math.pi * yy)

    return u.unsqueeze(0).unsqueeze(0), v.unsqueeze(0).unsqueeze(0)


class TestDivergenceFreeProjection:
    """Spectral Helmholtz decomposition for mass conservation."""

    def test_preserves_divergence_free_input(self):
        """Input that is already divergence-free passes through unchanged."""
        u, v = _make_periodic_divergence_free(32)
        dx = 1.0 / 32

        u_proj, v_proj = divergence_free_projection(u, v, dx=dx)

        # Should be very close to original
        assert torch.allclose(u_proj, u, atol=1e-4)
        assert torch.allclose(v_proj, v, atol=1e-4)

    def test_corrects_divergent_input(self):
        """Divergent input is corrected to have near-zero spectral divergence."""
        u, v = _make_periodic_divergent(32)
        dx = 1.0 / 32

        u_proj, v_proj = divergence_free_projection(u, v, dx=dx)

        # Non-Nyquist spectral divergence should be near-zero
        assert _spectral_divergence_max(u_proj, v_proj, dx=dx) < 1e-5

    def test_spectral_divergence_is_zero(self):
        """Spectral divergence of projected field is near-zero (non-Nyquist)."""
        u, v = _make_periodic_divergent(32)
        dx = 1.0 / 32

        u_proj, v_proj = divergence_free_projection(u, v, dx=dx)

        assert _spectral_divergence_max(u_proj, v_proj, dx=dx) < 1e-5

    def test_output_divergence_below_threshold(self):
        """Random field projection produces near-zero spectral divergence."""
        torch.manual_seed(42)
        u = torch.randn(1, 1, 32, 32)
        v = torch.randn(1, 1, 32, 32)

        u_proj, v_proj = divergence_free_projection(u, v)

        # float32 precision on random data gives O(1e-5) residuals;
        # threshold 1e-4 allows margin while still being tight
        assert _spectral_divergence_max(u_proj, v_proj) < 1e-4

    def test_batch_processing(self):
        """Projection works on batched tensors."""
        torch.manual_seed(42)
        u = torch.randn(4, 1, 16, 16)
        v = torch.randn(4, 1, 16, 16)

        u_proj, v_proj = divergence_free_projection(u, v)

        assert u_proj.shape == (4, 1, 16, 16)
        assert v_proj.shape == (4, 1, 16, 16)

        for b in range(4):
            assert _spectral_divergence_max(
                u_proj[b:b + 1], v_proj[b:b + 1]
            ) < 1e-4


class TestDivergenceFreeProjectionModule:
    """nn.Module wrapper for divergence-free projection."""

    def test_module_forward(self):
        """Module applies projection to wind channels only."""
        proj = DivergenceFreeProjection()
        torch.manual_seed(42)
        out = torch.randn(1, 4, 16, 16)
        temp_before = out[:, 2:3].clone()
        precip_before = out[:, 3:4].clone()

        projected = proj(out)

        assert projected.shape == out.shape

        # Temperature and precipitation should be bitwise identical
        assert torch.equal(projected[:, 2:3], temp_before)
        assert torch.equal(projected[:, 3:4], precip_before)

        # Wind output should be spectrally divergence-free (non-Nyquist)
        assert _spectral_divergence_max(
            projected[:, 0:1], projected[:, 1:2]
        ) < 1e-4

    def test_no_grad_passthrough(self):
        """Gradients flow through the projection."""
        proj = DivergenceFreeProjection()
        out = torch.randn(1, 4, 16, 16, requires_grad=True)
        projected = proj(out)
        loss = projected.sum()
        loss.backward()
        assert out.grad is not None


class TestHeadWithDivergenceFree:
    """Integration: TerrainDownscalingHead with divergence-free projection."""

    def test_head_output_divergence_free(self):
        """Full head output has spectrally divergence-free wind."""
        head = TerrainDownscalingHead(apply_divergence_free=True)
        head.eval()
        torch.manual_seed(42)
        terrain = torch.randn(1, 17, 16, 16)
        weather = torch.randn(1, 6, 4, 4)

        with torch.no_grad():
            out = head(terrain, weather)

        assert _spectral_divergence_max(out[:, 0:1], out[:, 1:2]) < 1e-4

    def test_head_without_projection(self):
        """Head without projection does not enforce divergence-free."""
        head = TerrainDownscalingHead(apply_divergence_free=False)
        head.eval()
        torch.manual_seed(42)
        terrain = torch.randn(1, 17, 16, 16)
        weather = torch.randn(1, 6, 4, 4)

        with torch.no_grad():
            out = head(terrain, weather)

        assert out.shape == (1, 4, 16, 16)
