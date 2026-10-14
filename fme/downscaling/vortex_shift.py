"""Translation-aware temporal baseline for coarse-endpoint trajectories.

Linear temporal interpolation of two coarse endpoint fields cross-fades a
moving vortex instead of translating it: a tropical cyclone that travels a few
degrees in 24 h becomes two weak half-vortices at the endpoint positions, so
the baseline is too shallow and ``r = x_c - I`` has to carry the whole
displacement. ``vortex_shift_interp`` replaces ``I(tau)`` near a detected
vortex by a blend of the two endpoint fields, each translated to the storm
centre linearly interpolated between the two endpoint centres, and keeps the
plain linear interpolation elsewhere.

Everything is computed from the two coarse endpoint frames only, so it is
available at inference. Properties the two-block construction relies on:

- the result equals the endpoint frames exactly at ``tau = 0`` and ``tau = 1``
  (so ``r`` stays pinned to zero there, Prop 1 of the theory doc);
- samples with no detected vortex get exactly the plain linear interpolation;
- it is a pure function of the endpoints (no learned parameters), so exact
  coarse recovery ``D(x_hat) = I + r_hat`` (Prop 4) is unaffected.
"""

import math

import torch

_REFINE_EPS = 1e-8


def _refine_1d(v: torch.Tensor, i: int) -> float:
    """Sub-cell parabolic refinement of a 1D minimum at index ``i``."""
    if 0 < i < v.shape[0] - 1:
        curvature = v[i - 1] - 2 * v[i] + v[i + 1]
        if curvature > _REFINE_EPS:
            return i + float(0.5 * (v[i - 1] - v[i + 1]) / curvature)
    return float(i)


def _radius_grid(
    shape: tuple[int, int], center: tuple[float, float], device: torch.device
) -> torch.Tensor:
    rows = torch.arange(shape[0], device=device, dtype=torch.float32)[:, None]
    cols = torch.arange(shape[1], device=device, dtype=torch.float32)[None, :]
    return torch.sqrt((rows - center[0]) ** 2 + (cols - center[1]) ** 2)


def _find_center(
    slp: torch.Tensor,
    min_depth: float,
    search_center: tuple[float, float] | None,
    search_cells: float,
    margin: int,
    ring: tuple[float, float],
) -> tuple[float, float] | None:
    """Deepest low of ``slp`` (h, w) that is at least ``min_depth`` below the
    mean of the surrounding ring; searched within ``search_cells`` of
    ``search_center`` when given. Returns sub-cell (row, col) or ``None``.
    """
    height, width = slp.shape
    valid = torch.zeros_like(slp, dtype=torch.bool)
    valid[margin : height - margin, margin : width - margin] = True
    if search_center is not None:
        valid &= _radius_grid(slp.shape, search_center, slp.device) <= search_cells
    if not valid.any():
        return None
    flat = torch.where(valid, slp, torch.full_like(slp, float("inf"))).reshape(-1)
    index = int(flat.argmin())
    row, col = index // width, index % width
    radius = _radius_grid(slp.shape, (float(row), float(col)), slp.device)
    annulus = (radius >= ring[0]) & (radius <= ring[1])
    if not annulus.any() or float(slp[annulus].mean() - slp[row, col]) < min_depth:
        return None
    return _refine_1d(slp[:, col], row), _refine_1d(slp[row, :], col)


def _translate(x: torch.Tensor, d_row: float, d_col: float) -> torch.Tensor:
    """Translate ``x`` (C, h, w) by (d_row, d_col) cells (bicubic, edge-clamped)."""
    if d_row == 0.0 and d_col == 0.0:
        return x
    _, height, width = x.shape
    rows = (torch.arange(height, device=x.device, dtype=x.dtype) - d_row) / max(
        height - 1, 1
    ) * 2 - 1
    cols = (torch.arange(width, device=x.device, dtype=x.dtype) - d_col) / max(
        width - 1, 1
    ) * 2 - 1
    grid = torch.stack(torch.meshgrid(rows, cols, indexing="ij")[::-1], dim=-1)
    out = torch.nn.functional.grid_sample(
        x[None],
        grid[None],
        mode="bicubic",
        padding_mode="border",
        align_corners=True,
    )
    return out[0]


def _anomaly(
    x: torch.Tensor, center: tuple[float, float], taper_cells: float
) -> torch.Tensor:
    """Vortex anomaly of ``x`` (C, h, w): the part of ``x`` inside a cosine
    taper around ``center``, relative to its mean over the annulus just outside
    the taper. Subtracting it leaves a vortex-free background.
    """
    radius = _radius_grid(x.shape[-2:], center, x.device)
    dist = radius / taper_cells
    taper = torch.where(
        dist < 1, 0.5 * (1 + torch.cos(math.pi * dist)), torch.zeros_like(dist)
    )
    outside = (radius >= taper_cells) & (radius <= 1.5 * taper_cells)
    fill = x[:, outside].mean(dim=-1)[:, None, None] if outside.any() else x.mean()
    return taper * (x - fill)


@torch.no_grad()
def vortex_shift_interp(
    coarse_clip: torch.Tensor,
    slp_index: int,
    min_depth: float,
    taper_cells: float = 6.0,
    search_cells: float = 10.0,
    ring: tuple[float, float] = (3.0, 6.0),
) -> torch.Tensor:
    """Translation-aware replacement for ``twoblock.coarse_temporal_interp``.

    Each endpoint is split into a vortex anomaly (inside a taper around its
    detected centre) and a vortex-free background. The backgrounds are
    interpolated linearly; the two anomalies are translated to the storm centre
    linearly interpolated between the endpoint centres and blended with the
    usual linear weights. This keeps one vortex at the right place instead of
    two half-vortices at the endpoint positions, and leaves everything outside
    the storm exactly as linear interpolation.

    Args:
        coarse_clip: ``(B, C, T, h, w)`` coarse clip; only frames 0 and -1 are
            used. Frame times are uniform.
        slp_index: Channel holding sea-level pressure, used to find vortices.
        min_depth: Minimum depth of a low below its surrounding ring (same
            units as ``coarse_clip``) for it to count as a vortex.
        taper_cells: Radius (coarse cells) of the cosine taper that defines
            the vortex anomaly around each endpoint centre.
        search_cells: Max distance (coarse cells) between the endpoint centres.
        ring: Inner/outer radius (coarse cells) of the depth-check ring.

    Returns:
        ``(B, C, T, h, w)`` baseline, equal to the endpoint frames at both
        ends, and to plain linear interpolation for samples with no vortex.
    """
    batch, _, n_times, _, _ = coarse_clip.shape
    weights = torch.linspace(0.0, 1.0, n_times, device=coarse_clip.device).reshape(
        1, 1, n_times, 1, 1
    )
    first, last = coarse_clip[:, :, :1], coarse_clip[:, :, -1:]
    out = ((1 - weights) * first + weights * last).clone()
    margin = max(int(math.ceil(ring[1] / 2)), 2)
    for b in range(batch):
        c0 = _find_center(
            coarse_clip[b, slp_index, 0], min_depth, None, search_cells, margin, ring
        )
        if c0 is None:
            continue
        c1 = _find_center(
            coarse_clip[b, slp_index, -1], min_depth, c0, search_cells, margin, ring
        )
        if c1 is None:
            continue
        anomaly_first = _anomaly(coarse_clip[b, :, 0], c0, taper_cells)
        anomaly_last = _anomaly(coarse_clip[b, :, -1], c1, taper_cells)
        for k in range(1, n_times - 1):
            tau = k / (n_times - 1)
            c_row = (1 - tau) * c0[0] + tau * c1[0]
            c_col = (1 - tau) * c0[1] + tau * c1[1]
            out[b, :, k] += (1 - tau) * (
                _translate(anomaly_first, c_row - c0[0], c_col - c0[1]) - anomaly_first
            ) + tau * (
                _translate(anomaly_last, c_row - c1[0], c_col - c1[1]) - anomaly_last
            )
    out[:, :, 0] = coarse_clip[:, :, 0]
    out[:, :, -1] = coarse_clip[:, :, -1]
    return out
