import pytest
import torch

from fme.downscaling.twoblock import coarse_temporal_interp
from fme.downscaling.vortex_shift import vortex_shift_blend, vortex_shift_interp

N_TIMES = 9
SHAPE = (40, 60)


def _vortex_clip(
    start: tuple[float, float],
    end: tuple[float, float],
    depth: float = 10.0,
    width: float = 2.5,
) -> torch.Tensor:
    """(1, 2, T, h, w): channel 0 is a Gaussian low translating linearly from
    ``start`` to ``end`` (the ground truth path), channel 1 a passive copy."""
    rows = torch.arange(SHAPE[0], dtype=torch.float32)[:, None]
    cols = torch.arange(SHAPE[1], dtype=torch.float32)[None, :]
    frames = []
    for k in range(N_TIMES):
        tau = k / (N_TIMES - 1)
        c_row = (1 - tau) * start[0] + tau * end[0]
        c_col = (1 - tau) * start[1] + tau * end[1]
        r2 = (rows - c_row) ** 2 + (cols - c_col) ** 2
        frames.append(-depth * torch.exp(-r2 / (2 * width**2)))
    slp = torch.stack(frames)
    return torch.stack([slp, 0.5 * slp])[None]


def _shift(clip: torch.Tensor) -> torch.Tensor:
    return vortex_shift_interp(clip, slp_index=0, min_depth=2.0, search_cells=25.0)


def test_endpoints_are_exact():
    clip = _vortex_clip((20.0, 15.0), (20.0, 35.0))
    out = _shift(clip)
    assert torch.equal(out[:, :, 0], clip[:, :, 0])
    assert torch.equal(out[:, :, -1], clip[:, :, -1])


def test_translating_vortex_is_tracked_better_than_linear():
    truth = _vortex_clip((20.0, 15.0), (20.0, 35.0))
    linear_mse = ((coarse_temporal_interp(truth) - truth) ** 2).mean()
    shifted_mse = ((_shift(truth) - truth) ** 2).mean()
    assert shifted_mse < 0.25 * linear_mse
    # Mid-trajectory the storm sits at its true centre and keeps its depth
    # (linear interpolation leaves two half-depth lows at the endpoints).
    mid = _shift(truth)[0, 0, N_TIMES // 2]
    mid_truth = truth[0, 0, N_TIMES // 2]
    assert mid.argmin() == mid_truth.argmin()
    assert float(mid.min()) < 0.9 * float(mid_truth.min())
    assert float(coarse_temporal_interp(truth)[0, 0, N_TIMES // 2].min()) > 0.7 * float(
        mid_truth.min()
    )


def test_passive_channels_are_translated_with_the_storm():
    truth = _vortex_clip((20.0, 15.0), (20.0, 35.0))
    out = _shift(truth)
    assert torch.allclose(out[:, 1], 0.5 * out[:, 0], atol=1e-4)


def test_no_vortex_falls_back_to_exact_linear_interpolation():
    flat = 0.01 * torch.randn(2, 2, N_TIMES, *SHAPE, generator=torch.manual_seed(0))
    assert torch.equal(_shift(flat), coarse_temporal_interp(flat))


def test_mixed_batch_only_shifts_samples_with_a_vortex():
    storm = _vortex_clip((20.0, 15.0), (20.0, 35.0))
    flat = torch.zeros_like(storm)
    batch = torch.cat([storm, flat])
    out = _shift(batch)
    assert torch.equal(out[1], coarse_temporal_interp(flat)[0])
    assert not torch.allclose(out[0], coarse_temporal_interp(storm)[0])


def test_unmatched_second_endpoint_falls_back_to_linear():
    """A low at frame 0 with no low within the search radius at frame -1 is
    not treated as a travelling vortex."""
    clip = _vortex_clip((20.0, 10.0), (20.0, 50.0))  # 40 cells apart > search
    assert torch.equal(_shift(clip), coarse_temporal_interp(clip))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_preserves_shape_and_dtype(dtype):
    clip = _vortex_clip((20.0, 15.0), (22.0, 30.0)).to(dtype)
    out = _shift(clip)
    assert out.shape == clip.shape and out.dtype == dtype


def _blend(clip, base, **kwargs):
    return vortex_shift_blend(
        clip, base, slp_index=0, min_depth=2.0, search_cells=25.0, **kwargs
    )


def _shallow_base(truth, depth_fraction=0.6):
    """A baseline that tracks the storm but smooths its core (like V3): the true
    trajectory with the storm scaled to ``depth_fraction`` of its depth."""
    base = truth.clone()
    base[:, :, 1:-1] = depth_fraction * truth[:, :, 1:-1]
    return base


def test_blend_pins_endpoints_and_keeps_the_base_far_from_the_storm():
    truth = _vortex_clip((20.0, 15.0), (20.0, 35.0))
    base = _shallow_base(truth)
    out = _blend(truth, base)
    assert torch.equal(out[:, :, 0], truth[:, :, 0])
    assert torch.equal(out[:, :, -1], truth[:, :, -1])
    # rows 0-4 are >= 15 cells from the storm track, beyond the 9-cell blend disc
    assert torch.equal(out[:, :, 1:-1, :5], base[:, :, 1:-1, :5])
    assert not torch.allclose(out[:, :, 1:-1], base[:, :, 1:-1])


def test_blend_restores_the_core_depth_of_a_smoothed_baseline():
    truth = _vortex_clip((20.0, 15.0), (20.0, 35.0))
    base = _shallow_base(truth)
    out = _blend(truth, base)
    mid = N_TIMES // 2
    true_min = float(truth[0, 0, mid].min())
    assert abs(float(out[0, 0, mid].min()) - true_min) < 0.5 * abs(
        float(base[0, 0, mid].min()) - true_min
    )
    assert out[0, 0, mid].argmin() == truth[0, 0, mid].argmin()
    # the passive channel is blended consistently with the storm
    assert torch.allclose(out[:, 1], 0.5 * out[:, 0], atol=1e-4)


def test_blend_without_a_detected_storm_returns_the_base():
    flat = 0.01 * torch.randn(2, 2, N_TIMES, *SHAPE, generator=torch.manual_seed(0))
    base = coarse_temporal_interp(flat)
    base[:, :, 1:-1] += 0.3  # any baseline that keeps the endpoints
    assert torch.equal(_blend(flat, base), base)


def test_blend_only_changes_samples_with_a_storm():
    storm = _vortex_clip((20.0, 15.0), (20.0, 35.0))
    batch = torch.cat([storm, torch.zeros_like(storm)])
    base = _shallow_base(batch)
    out = _blend(batch, base)
    assert torch.equal(out[1], base[1])
    assert not torch.allclose(out[0], base[0])


def test_blend_rejects_a_base_with_the_wrong_shape():
    truth = _vortex_clip((20.0, 15.0), (20.0, 35.0))
    with pytest.raises(ValueError, match="must match"):
        _blend(truth, truth[:, :, :-1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_blend_preserves_shape_and_dtype(dtype):
    truth = _vortex_clip((20.0, 15.0), (22.0, 30.0)).to(dtype)
    out = _blend(truth, _shallow_base(truth))
    assert out.shape == truth.shape and out.dtype == dtype
