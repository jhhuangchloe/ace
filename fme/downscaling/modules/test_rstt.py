import pytest
import torch

from fme.downscaling.modules.rstt import RSTT, input_frame_positions


def _small_rstt(n_in=2, n_out=9, upscale=4, channels=3):
    return RSTT(
        in_channels=channels,
        out_channels=channels,
        n_in=n_in,
        n_out=n_out,
        upscale=upscale,
        embed_dim=16,
        depths=(1, 2, 1, 2),
        num_heads=(2, 2, 2, 2),
        window_size=(4, 4),
    )


@pytest.mark.parametrize("hw", [(8, 8), (11, 18)])
def test_rstt_endpoints_to_clip_shape(hw):
    # (11, 18) is neither a multiple of the window nor of the encoder's 2x
    # downsampling, exercising both padding paths.
    net = _small_rstt()
    x = torch.randn(2, 3, 2, *hw)
    out = net(x)
    assert out.shape == (2, 3, 9, hw[0] * 4, hw[1] * 4)
    assert torch.isfinite(out).all()


def test_rstt_reference_frame_counts():
    # The reference's own setting: 4 input frames -> 7 output frames.
    net = _small_rstt(n_in=4, n_out=7, upscale=2)
    out = net(torch.randn(1, 3, 4, 8, 8))
    assert out.shape == (1, 3, 7, 16, 16)


def test_rstt_rejects_wrong_input_frame_count():
    net = _small_rstt()
    with pytest.raises(ValueError, match="Expected 2 input frames"):
        net(torch.randn(1, 3, 3, 8, 8))


def test_input_frame_positions():
    assert input_frame_positions(2, 9) == [0, 8]
    assert input_frame_positions(4, 7) == [0, 2, 4, 6]
    with pytest.raises(ValueError, match="output grid"):
        input_frame_positions(3, 8)


def test_query_interpolation_reduces_to_reference_for_4_to_7():
    # Reference: copy the input frame at even output frames, average the two
    # neighbours at odd ones.
    net = _small_rstt(n_in=4, n_out=7, upscale=2)
    expected = torch.zeros(7, 4)
    for k in range(7):
        if k % 2 == 0:
            expected[k, k // 2] = 1.0
        else:
            expected[k, k // 2] = 0.5
            expected[k, k // 2 + 1] = 0.5
    torch.testing.assert_close(net.query_interp, expected)


def test_query_interpolation_is_linear_for_2_to_9():
    net = _small_rstt()
    tau = torch.linspace(0, 1, 9)
    torch.testing.assert_close(net.query_interp, torch.stack([1 - tau, tau], dim=1))


def test_rstt_all_parameters_receive_gradients():
    net = _small_rstt()
    net(torch.randn(1, 3, 2, 8, 8)).square().mean().backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing
