import pytest
import torch

from fme.downscaling.pseudo_temporal import (
    pseudo_temporal_sample,
    random_block_mask,
    visibility_weight,
)


def test_random_block_mask_masks_exact_fraction_per_frame():
    generator = torch.Generator().manual_seed(0)
    mask = random_block_mask(3, 2, 44, 72, 4, 0.5, generator=generator)
    assert mask.shape == (3, 1, 2, 44, 72)
    blocks = mask[..., ::4, ::4]  # one value per 4x4 block
    n_blocks = 11 * 18
    n_masked = (blocks == 0).flatten(start_dim=3).sum(dim=-1)
    assert (n_masked == round(0.5 * n_blocks)).all()
    # constant within each block
    upsampled = blocks.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)
    assert torch.equal(upsampled, mask)


def test_random_block_mask_frames_are_independent():
    generator = torch.Generator().manual_seed(0)
    mask = random_block_mask(4, 2, 16, 16, 4, 0.5, generator=generator)
    assert not torch.equal(mask[:, :, 0], mask[:, :, 1])


def test_random_block_mask_partial_edge_blocks():
    mask = random_block_mask(1, 1, 10, 7, 4, 0.0)
    assert mask.shape == (1, 1, 1, 10, 7)
    assert (mask == 1).all()


def test_random_block_mask_rejects_bad_ratio():
    with pytest.raises(ValueError, match="mask_ratio"):
        random_block_mask(1, 1, 4, 4, 2, 1.5)


def test_visibility_weight_is_reciprocal_of_one_plus_visible_count():
    visible = torch.tensor(
        [[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]  # frames x pixels
    ).reshape(1, 1, 2, 1, 3)
    weight = visibility_weight(visible, scale=2)
    assert weight.shape == (1, 1, 1, 2, 6)
    expected = torch.tensor([1.0, 1.0, 1 / 2, 1 / 2, 1 / 3, 1 / 3])
    torch.testing.assert_close(weight[0, 0, 0, 0], expected)
    torch.testing.assert_close(weight[0, 0, 0, 1], expected)


def test_pseudo_temporal_sample_uses_one_snapshot_per_clip():
    batch, channels, n_times, h, w, scale = 3, 2, 5, 8, 8, 2
    # Every pixel of frame k holds k + 1 (nonzero, so masked zeros are
    # distinguishable), offset per sample so frames can't be mixed up.
    frame_values = torch.arange(1, n_times + 1, dtype=torch.float32)
    offsets = 100.0 * torch.arange(batch, dtype=torch.float32)
    values = (offsets[:, None] + frame_values[None, :]).reshape(batch, 1, n_times, 1, 1)
    coarse = values.expand(batch, channels, n_times, h, w)
    fine = values.expand(batch, channels, n_times, h * scale, w * scale)

    generator = torch.Generator().manual_seed(0)
    inputs, target, weight = pseudo_temporal_sample(
        coarse, fine, 2, 2, 0.5, generator=generator
    )
    assert inputs.shape == (batch, channels, 2, h, w)
    assert target.shape == (batch, channels, n_times, h * scale, w * scale)
    assert weight.shape == (batch, 1, 1, h * scale, w * scale)
    for b in range(batch):
        snapshot = target[b, 0, 0, 0, 0]
        # target is the same fine snapshot at every output frame
        assert (target[b] == snapshot).all()
        # inputs are that same time step's coarse frame, zero where masked
        visible = inputs[b] != 0
        assert (inputs[b][visible] == snapshot).all()
        assert 0 < visible.float().mean() < 1
