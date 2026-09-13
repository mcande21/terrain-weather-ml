"""Divergence-free projection for wind fields (task 5.4).

Implements mass-conservation constraint via spectral Helmholtz decomposition:
1. FFT the (u, v) wind field
2. Compute spectral divergence
3. Solve Poisson equation for pressure correction
4. Subtract spectral gradient of correction
5. IFFT to get divergence-free field

The fully spectral approach guarantees exact divergence-free output
(up to machine precision) under periodic boundary conditions.

A finite-difference `compute_divergence` is also provided for diagnostics
and for the soft divergence penalty in the training loss.
"""

from __future__ import annotations

import torch
from torch import nn


def compute_divergence(
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float = 1.0,
) -> torch.Tensor:
    """Compute 2D divergence of a vector field via central finite differences.

    div(u, v) = du/dx + dv/dy

    Used for diagnostics and the training loss divergence penalty.
    For the hard projection, use `divergence_free_projection` which is
    fully spectral and self-consistent.

    Args:
        u: Eastward wind component (..., H, W).
        v: Northward wind component (..., H, W).
        dx: Grid spacing (uniform in both directions).

    Returns:
        Divergence field, same shape as input.
    """
    # Central differences for interior, forward/backward at boundaries
    # du/dx along W dimension (axis=-1)
    dudx = torch.zeros_like(u)
    dudx[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2 * dx)
    dudx[..., 0] = (u[..., 1] - u[..., 0]) / dx
    dudx[..., -1] = (u[..., -1] - u[..., -2]) / dx

    # dv/dy along H dimension (axis=-2)
    dvdy = torch.zeros_like(v)
    dvdy[..., 1:-1, :] = (v[..., 2:, :] - v[..., :-2, :]) / (2 * dx)
    dvdy[..., 0, :] = (v[..., 1, :] - v[..., 0, :]) / dx
    dvdy[..., -1, :] = (v[..., -1, :] - v[..., -2, :]) / dx

    return dudx + dvdy


def divergence_free_projection(
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a 2D vector field onto its divergence-free component.

    Fully spectral Helmholtz decomposition:
      (u, v) = (u_solenoidal) + grad(phi)
    where the solenoidal (divergence-free) part is extracted by
    subtracting the irrotational component in Fourier space.

    The spectral projection is exact: the output has zero spectral
    divergence, so `compute_divergence` on the result will show only
    the small discretization error of finite differences.

    Args:
        u: Eastward wind component (B, 1, H, W).
        v: Northward wind component (B, 1, H, W).
        dx: Grid spacing.

    Returns:
        Tuple of divergence-free (u, v) components.
    """
    h, w = u.shape[-2], u.shape[-1]

    # Forward FFT
    u_hat = torch.fft.rfft2(u)
    v_hat = torch.fft.rfft2(v)

    # Wavenumbers (cycles per unit length)
    # rfftfreq for kx (gives 0..N/2 with correct +0.5 Nyquist sign)
    # fftfreq for ky (full spectrum)
    kx = torch.fft.rfftfreq(w, d=dx, device=u.device, dtype=u.dtype)
    ky = torch.fft.fftfreq(h, d=dx, device=u.device, dtype=u.dtype)

    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")

    # Angular wavenumbers
    kx_ang = 2 * torch.pi * kx_grid
    ky_ang = 2 * torch.pi * ky_grid

    # k^2 = kx^2 + ky^2 (angular)
    k_sq = kx_ang**2 + ky_ang**2

    # Spectral divergence: div_hat = i*kx*u_hat + i*ky*v_hat
    div_hat = 1j * kx_ang * u_hat + 1j * ky_ang * v_hat

    # Avoid division by zero at k=0
    k_sq_safe = k_sq.clone()
    k_sq_safe[0, 0] = 1.0

    # Poisson solve: nabla^2(phi) = div
    # In spectral: (-k^2) * phi_hat = div_hat => phi_hat = -div_hat / k^2
    phi_hat = -div_hat / k_sq_safe
    phi_hat[..., 0, 0] = 0.0  # zero mean

    # Zero out Nyquist modes: the projection at these modes creates
    # coefficients that violate rfft2's conjugate symmetry, so irfft2
    # destroys them. These sub-grid modes are not physically meaningful
    # for weather data. Zero phi_hat at Nyquist kx and ky.
    phi_hat[..., -1] = 0.0  # Nyquist kx column
    if h % 2 == 0:
        phi_hat[..., h // 2, :] = 0.0  # Nyquist ky row

    # Subtract irrotational component: grad(phi)
    u_proj_hat = u_hat - 1j * kx_ang * phi_hat
    v_proj_hat = v_hat - 1j * ky_ang * phi_hat

    # Inverse FFT
    u_proj = torch.fft.irfft2(u_proj_hat, s=(h, w))
    v_proj = torch.fft.irfft2(v_proj_hat, s=(h, w))

    return u_proj, v_proj


class DivergenceFreeProjection(nn.Module):
    """nn.Module wrapper for divergence-free projection on wind channels.

    Applies the spectral Helmholtz projection to the u-wind and v-wind
    channels of the output tensor. Temperature and precipitation channels
    pass through unchanged.

    Args:
        dx: Grid spacing for the projection.
        u_idx: Index of u-wind channel in output tensor.
        v_idx: Index of v-wind channel in output tensor.
    """

    def __init__(self, dx: float = 1.0, u_idx: int = 0, v_idx: int = 1):
        super().__init__()
        self.dx = dx
        self.u_idx = u_idx
        self.v_idx = v_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply divergence-free projection to wind channels.

        Args:
            x: Output tensor (B, C, H, W) with wind at channels u_idx, v_idx.

        Returns:
            Tensor with projected wind and unchanged non-wind channels.
        """
        u = x[:, self.u_idx : self.u_idx + 1]
        v = x[:, self.v_idx : self.v_idx + 1]

        u_proj, v_proj = divergence_free_projection(u, v, dx=self.dx)

        # Build output preserving non-wind channels exactly
        out = x.clone()
        out[:, self.u_idx : self.u_idx + 1] = u_proj
        out[:, self.v_idx : self.v_idx + 1] = v_proj

        return out
