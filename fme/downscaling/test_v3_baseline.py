import os

import numpy as np
import pytest
import torch

from fme.downscaling.v3_baseline import (
    N_STEPS,
    V3Baseline,
    V3Config,
    load_v3_baseline,
    time_assignment,
)

N_CHANNELS = 3


def _tiny_model(zero_field=False) -> V3Baseline:
    config = V3Config(
        n_channels=N_CHANNELS,
        embed_dim=16,
        num_blocks=(1, 1, 1),
        depths=(1, 1, 1),
        num_heads=2,
        output_dims=8,
        num_basis=16,
    )
    torch.manual_seed(0)
    model = V3Baseline(config).eval()
    with torch.no_grad():
        model.freqs.copy_(torch.randn_like(model.freqs))
        model.k.fill_(0.05)
        if zero_field:
            model.out_conv.weight.zero_()
            model.out_conv.bias.zero_()
    return model


def _source(batch=2, height=12, width=20):
    return torch.randn(batch, 2, height, width, N_CHANNELS)


def test_time_assignment_gives_each_step_to_its_nearest_endpoint():
    idx, rel = time_assignment()
    assert idx.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1]  # ties go to the earlier frame
    assert rel[0] == pytest.approx(0.0) and rel[-1] == pytest.approx(0.0)
    assert np.abs(rel).max() <= 0.5 + 1e-6


def test_output_shape_and_step_subsetting():
    model = _tiny_model()
    source = _source()
    full = model(source)
    assert full.shape == (2, N_STEPS, 48, 80, N_CHANNELS)
    assert torch.isfinite(full).all()
    subset = model(source, steps=[1, 6])
    assert torch.allclose(subset, full[:, [1, 6]], atol=1e-5)


def test_zero_field_returns_the_nearest_endpoint_upsampled():
    model = _tiny_model(zero_field=True)
    source = _source()
    out = model(source)
    idx, _ = time_assignment()
    expected = source[:, idx].repeat_interleave(4, 2).repeat_interleave(4, 3)
    assert torch.allclose(out, expected, atol=1e-6)


def test_is_deterministic_and_batch_independent():
    model = _tiny_model()
    source = _source()
    assert torch.equal(model(source), model(source))
    assert torch.allclose(model(source)[:1], model(source[:1]), atol=1e-4)


def test_rejects_patches_smaller_than_the_attention_window():
    with pytest.raises(ValueError, match="window"):
        _tiny_model()(_source(height=6, width=20))


_REF = "/data/pmd/v3_baseline/ref.npz"
_CKPT = "/data/pmd/v3_baseline/v3_baseline.pt"


@pytest.mark.skipif(
    not (os.path.exists(_REF) and os.path.exists(_CKPT)),
    reason="needs the converted V3 checkpoint and the JAX reference outputs "
    "(scripts/v3_baseline/make_jax_reference.py)",
)
def test_matches_the_jax_model_on_real_weights():
    model, _ = load_v3_baseline(_CKPT)
    ref = np.load(_REF)
    source = torch.from_numpy(ref["source"]).float()
    assert np.abs(model.encode(source).numpy() - ref["encoding"]).max() < 1e-4
    assert np.abs(model(source).numpy() - ref["fine"]).mean() < 1e-4
