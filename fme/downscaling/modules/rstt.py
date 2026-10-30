"""RSTT: Real-time Spatial Temporal Transformer for space-time video
super-resolution (Geng et al., CVPR 2022), reimplemented for the
endpoint-conditioned downscaling setting.

Reference: https://github.com/llmpass/RSTT (``models/RSTT.py``,
``models/layers.py``; CC BY-NC-SA 4.0). The architecture follows that code
block for block -- input projection, a Swin-style encoder of windowed
spatio-temporal self-attention with stride-2 conv downsampling, a decoder whose
per-output-frame queries self-attend and cross-attend to the encoder features of
the same scale, and a pixel-shuffle reconstruction head that predicts a residual
over an interpolated baseline -- with these generalizations:

- Frame counts: the reference is fixed to ``n_out = 2 * n_in - 1`` (4 -> 7).
  Here ``n_in`` and ``n_out`` are free as long as the input frames fall on the
  output grid (``(n_out - 1) % (n_in - 1) == 0``), e.g. 2 endpoints -> 9 frames.
- Decoder queries: the reference initializes them by copying the bottleneck
  feature at input frames and averaging the two neighbours in between; here they
  are the linear interpolation of the bottleneck features at each output time,
  which reduces to exactly that for ``n_out = 2 * n_in - 1``.
- Cross-attention relative time bias: the reference places input frame ``i`` at
  output position ``i * (n_out + 1) // n_in``, which is ``2 * i`` for 4 -> 7;
  here it is ``i * (n_out - 1) // (n_in - 1)`` (same value there).
- Channels are configurable (``in_channels``/``out_channels``), and the
  reconstruction head generalizes the two pixel-shuffle stages to any
  power-of-two upscale factor (identical layers for the reference's 4x).
- The baseline is not added here: ``forward`` returns the residual, and the
  caller adds its own baseline (the reference adds a trilinear upsample of the
  input clip; ``fme.downscaling.stvsr`` uses the linear-in-time, bicubic-in-space
  baseline of the other video models).

Attention uses ``F.scaled_dot_product_attention`` with the relative position
bias and the shifted-window mask folded into one additive mask, which is the
same computation as the reference's explicit softmax.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fme.downscaling.modules.swinir import DropPath, Mlp, trunc_normal_


def input_frame_positions(n_in: int, n_out: int) -> list[int]:
    """Position of each input frame on the output frame grid (first and last
    input frames coincide with the first and last output frames).
    """
    if n_in < 2 or n_out < n_in:
        raise ValueError(f"Need 2 <= n_in <= n_out, got n_in={n_in}, n_out={n_out}.")
    if (n_out - 1) % (n_in - 1) != 0:
        raise ValueError(
            f"Input frames must fall on the output grid: (n_out - 1) = {n_out - 1} "
            f"must be divisible by (n_in - 1) = {n_in - 1}."
        )
    stride = (n_out - 1) // (n_in - 1)
    return [i * stride for i in range(n_in)]


def _window_partition(x: torch.Tensor, window_size: tuple[int, int]) -> torch.Tensor:
    """``(B, D, H, W, C) -> (B * nW, D, Wh, Ww, C)``."""
    b, d, h, w, c = x.shape
    x = x.view(
        b,
        d,
        h // window_size[0],
        window_size[0],
        w // window_size[1],
        window_size[1],
        c,
    )
    return x.permute(0, 2, 4, 1, 3, 5, 6).reshape(
        -1, d, window_size[0], window_size[1], c
    )


def _window_reverse(
    windows: torch.Tensor, window_size: tuple[int, int], b: int, d: int, h: int, w: int
) -> torch.Tensor:
    """Inverse of ``_window_partition``:
    ``(B * nW, D, Wh, Ww, C) -> (B, D, H, W, C)``.
    """
    x = windows.view(
        b,
        h // window_size[0],
        w // window_size[1],
        d,
        window_size[0],
        window_size[1],
        -1,
    )
    return x.permute(0, 3, 1, 4, 2, 5, 6).reshape(b, d, h, w, -1)


def _effective_window(
    hw: tuple[int, int], window_size: tuple[int, int], shift_size: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Shrink the window to the feature map (and drop the shift) along any axis
    no larger than the window, as the reference's ``get_window_size`` does.
    """
    window = list(window_size)
    shift = list(shift_size)
    for i in range(2):
        if hw[i] <= window_size[i]:
            window[i] = hw[i]
            shift[i] = 0
    return (window[0], window[1]), (shift[0], shift[1])


def _shift_region_ids(
    hp: int, wp: int, window: tuple[int, int], shift: tuple[int, int], device
) -> torch.Tensor:
    """``(nW, Wh * Ww)`` id of the pre-shift region each token of each window
    came from; tokens from different regions must not attend to each other.
    """
    img = torch.zeros((1, 1, hp, wp, 1), device=device)
    h_slices = (
        slice(0, -window[0]),
        slice(-window[0], -shift[0]),
        slice(-shift[0], None),
    )
    w_slices = (
        slice(0, -window[1]),
        slice(-window[1], -shift[1]),
        slice(-shift[1], None),
    )
    count = 0
    for hs in h_slices:
        for ws in w_slices:
            img[:, :, hs, ws, :] = count
            count += 1
    return _window_partition(img, window).reshape(-1, window[0] * window[1])


def _shift_attention_mask(
    region_ids: torch.Tensor, n_frames_q: int, n_frames_kv: int
) -> torch.Tensor:
    """``(nW, Dq * Wh * Ww, Dkv * Wh * Ww)`` additive mask (0 or -100, as in the
    reference) for shifted-window attention, repeating the spatial region ids
    over frames in the token order ``(frame, row, col)``.
    """
    q_ids = region_ids.repeat(1, n_frames_q)
    kv_ids = region_ids.repeat(1, n_frames_kv)
    diff = kv_ids.unsqueeze(1) - q_ids.unsqueeze(2)
    return torch.zeros_like(diff).masked_fill(diff != 0, -100.0)


class WindowAttention3D(nn.Module):
    """Windowed multi-head self/cross attention over ``(frame, row, col)``
    tokens with a learned relative position bias in time and space.

    ``q_frames``/``kv_frames`` are the frame positions of the query and key
    tokens; the time component of the bias is indexed by their difference.
    """

    def __init__(
        self,
        dim: int,
        q_frames: list[int],
        kv_frames: list[int],
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn_drop = attn_drop
        n_q = len(q_frames)
        time_span = max(q_frames + kv_frames) - min(q_frames + kv_frames)
        n_rel_t = 2 * time_span + 1
        n_rel_h = 2 * window_size[0] - 1
        n_rel_w = 2 * window_size[1] - 1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(n_rel_t * n_rel_h * n_rel_w, num_heads)
        )
        hh = torch.arange(window_size[0])
        ww = torch.arange(window_size[1])
        coords_q = torch.stack(
            torch.meshgrid(torch.tensor(q_frames), hh, ww, indexing="ij")
        ).flatten(1)
        coords_kv = torch.stack(
            torch.meshgrid(torch.tensor(kv_frames), hh, ww, indexing="ij")
        ).flatten(1)
        rel = (coords_q[:, :, None] - coords_kv[:, None, :]).permute(1, 2, 0)
        rel[:, :, 0] += time_span
        rel[:, :, 1] += window_size[0] - 1
        rel[:, :, 2] += window_size[1] - 1
        index = (rel[:, :, 0] * n_rel_h + rel[:, :, 1]) * n_rel_w + rel[:, :, 2]
        self.register_buffer("relative_position_index", index, persistent=False)
        self._n_q_tokens = n_q * window_size[0] * window_size[1]

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            q: ``(B * nW, Nq, C)`` query tokens.
            kv: ``(B * nW, Nkv, C)`` key/value tokens; ``q`` if omitted.
            mask: ``(nW, Nq, Nkv)`` additive shifted-window mask, or None.
        """
        kv = q if kv is None else kv
        b_, n_q, c = q.shape
        n_kv = kv.shape[1]
        head_dim = c // self.num_heads
        q_ = self.q(q).reshape(b_, n_q, self.num_heads, head_dim).transpose(1, 2)
        kv_ = self.kv(kv).reshape(b_, n_kv, 2, self.num_heads, head_dim)
        kv_ = kv_.permute(2, 0, 3, 1, 4)
        k_, v_ = kv_[0], kv_[1]
        bias = (
            self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
            .reshape(n_q, n_kv, self.num_heads)
            .permute(2, 0, 1)
        )
        attn_mask = bias.unsqueeze(0)  # (1, nH, Nq, Nkv)
        if mask is not None:
            n_w = mask.shape[0]
            attn_mask = (
                (attn_mask.unsqueeze(0) + mask.unsqueeze(1).unsqueeze(0))
                .expand(b_ // n_w, -1, -1, -1, -1)
                .reshape(b_, self.num_heads, n_q, n_kv)
            )
        out = F.scaled_dot_product_attention(
            q_,
            k_,
            v_,
            attn_mask=attn_mask.to(q_.dtype),
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b_, n_q, c)
        return self.proj_drop(self.proj(out))


def _to_windows(
    x: torch.Tensor, window: tuple[int, int], shift: tuple[int, int]
) -> torch.Tensor:
    """Pad ``(B, D, H, W, C)`` to whole windows, cyclic-shift, and flatten to
    ``(B * nW, D * Wh * Ww, C)`` tokens.
    """
    _, d, h, w, c = x.shape
    pad_b = (window[0] - h % window[0]) % window[0]
    pad_r = (window[1] - w % window[1]) % window[1]
    x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
    if any(s > 0 for s in shift):
        x = torch.roll(x, shifts=(-shift[0], -shift[1]), dims=(2, 3))
    return _window_partition(x, window).reshape(-1, d * window[0] * window[1], c)


def _from_windows(
    tokens: torch.Tensor,
    window: tuple[int, int],
    shift: tuple[int, int],
    b: int,
    d: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """Inverse of ``_to_windows``: back to ``(B, D, H, W, C)``, unshifted and
    cropped to ``(h, w)``.
    """
    hp = h + (window[0] - h % window[0]) % window[0]
    wp = w + (window[1] - w % window[1]) % window[1]
    x = _window_reverse(
        tokens.reshape(-1, d, window[0], window[1], tokens.shape[-1]),
        window,
        b,
        d,
        hp,
        wp,
    )
    if any(s > 0 for s in shift):
        x = torch.roll(x, shifts=(shift[0], shift[1]), dims=(2, 3))
    return x[:, :, :h, :w, :]


class EncoderBlock(nn.Module):
    """(Shifted-)window spatio-temporal self-attention + MLP, pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        frames: list[int],
        window_size: tuple[int, int],
        shift_size: tuple[int, int],
        mlp_ratio: float,
        drop_path: float,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, frames, frames, window_size, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """``x``: ``(B, D, H, W, C)``; ``mask``: shifted-window mask or None."""
        b, d, h, w, _ = x.shape
        window, shift = _effective_window((h, w), self.window_size, self.shift_size)
        tokens = _to_windows(self.norm1(x), window, shift)
        tokens = self.attn(tokens, mask=mask if any(shift) else None)
        x = x + self.drop_path(_from_windows(tokens, window, shift, b, d, h, w))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class DecoderBlock(nn.Module):
    """Query self-attention, then cross-attention to the encoder features of
    the same scale, then MLP; all windowed and pre-norm.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        q_frames: list[int],
        kv_frames: list[int],
        window_size: tuple[int, int],
        shift_size: tuple[int, int],
        mlp_ratio: float,
        drop_path: float,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = WindowAttention3D(dim, q_frames, q_frames, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn2 = WindowAttention3D(dim, q_frames, kv_frames, window_size, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        mask_q: torch.Tensor | None,
        mask_qkv: torch.Tensor | None,
    ) -> torch.Tensor:
        """``x``: ``(B, Dq, H, W, C)`` queries; ``enc``: ``(B, Dkv, H, W, C)``."""
        b, d_q, h, w, _ = x.shape
        window, shift = _effective_window((h, w), self.window_size, self.shift_size)
        shifted = any(shift)
        tokens = _to_windows(self.norm1(x), window, shift)
        tokens = self.attn1(tokens, mask=mask_q if shifted else None)
        x = x + self.drop_path(_from_windows(tokens, window, shift, b, d_q, h, w))
        q_tokens = _to_windows(self.norm2(x), window, shift)
        kv_tokens = _to_windows(self.norm_kv(enc), window, shift)
        tokens = self.attn2(q_tokens, kv_tokens, mask=mask_qkv if shifted else None)
        x = x + self.drop_path(_from_windows(tokens, window, shift, b, d_q, h, w))
        return x + self.drop_path(self.mlp(self.norm3(x)))


def _layer_masks(
    hw: tuple[int, int],
    window_size: tuple[int, int],
    n_frames_q: int,
    n_frames_kv: int,
    device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Shifted-window masks for one layer: (self-attention over the query
    frames, cross-attention from query to key frames); None when no shift.
    """
    shift_size = (window_size[0] // 2, window_size[1] // 2)
    window, shift = _effective_window(hw, window_size, shift_size)
    if not any(shift):
        return None, None
    hp = math.ceil(hw[0] / window[0]) * window[0]
    wp = math.ceil(hw[1] / window[1]) * window[1]
    ids = _shift_region_ids(hp, wp, window, shift, device)
    return (
        _shift_attention_mask(ids, n_frames_q, n_frames_q),
        _shift_attention_mask(ids, n_frames_q, n_frames_kv),
    )


class EncoderLayer(nn.Module):
    def __init__(
        self, dim, depth, num_heads, frames, window_size, mlp_ratio, drop_path
    ):
        super().__init__()
        self.window_size = window_size
        self.n_frames = len(frames)
        shift = (window_size[0] // 2, window_size[1] // 2)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim,
                    num_heads,
                    frames,
                    window_size,
                    (0, 0) if i % 2 == 0 else shift,
                    mlp_ratio,
                    drop_path[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, D, H, W, C) -> (B, D, H, W, C)``."""
        mask, _ = _layer_masks(
            x.shape[2:4], self.window_size, self.n_frames, self.n_frames, x.device
        )
        for block in self.blocks:
            x = block(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        q_frames,
        kv_frames,
        window_size,
        mlp_ratio,
        drop_path,
    ):
        super().__init__()
        self.window_size = window_size
        self.n_q = len(q_frames)
        self.n_kv = len(kv_frames)
        shift = (window_size[0] // 2, window_size[1] // 2)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim,
                    num_heads,
                    q_frames,
                    kv_frames,
                    window_size,
                    (0, 0) if i % 2 == 0 else shift,
                    mlp_ratio,
                    drop_path[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        mask_q, mask_qkv = _layer_masks(
            x.shape[2:4], self.window_size, self.n_q, self.n_kv, x.device
        )
        for block in self.blocks:
            x = block(x, enc, mask_q, mask_qkv)
        return x


class _ResidualBlockNoBN(nn.Module):
    """Conv-ReLU-Conv with identity skip, conv weights scaled by 0.1 at init
    (the reference's ``ResidualBlock_noBN``).
    """

    def __init__(self, nf: int):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        for conv in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(conv.weight, a=0, mode="fan_in")
            conv.weight.data *= 0.1
            nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.relu(self.conv1(x)))


class RSTT(nn.Module):
    """Space-time super-resolution of a ``(B, C_in, n_in, H, W)`` clip to a
    ``(B, C_out, n_out, H * upscale, W * upscale)`` residual.

    Args:
        in_channels: Input channels per frame.
        out_channels: Output channels per frame.
        n_in: Number of input frames.
        n_out: Number of output frames; input frame ``i`` sits at output frame
            ``input_frame_positions(n_in, n_out)[i]``.
        upscale: Spatial upscale factor, a power of two.
        embed_dim: Feature width, constant across stages (as in the reference).
        depths: Blocks per stage, encoder stages then decoder stages (even
            length; the encoder downsamples 2x between its stages).
        num_heads: Attention heads per stage, same length as ``depths``.
        window_size: Spatial attention window ``(Wh, Ww)`` for every stage.
        mlp_ratio: MLP hidden width as a multiple of ``embed_dim``.
        drop_path_rate: Maximum stochastic-depth rate (linear ramp over the
            encoder blocks, reversed over the decoder blocks).
        back_rbs: Residual conv blocks before the upsampling head.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_in: int,
        n_out: int,
        upscale: int = 4,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (4, 4, 4, 4, 4, 4, 4, 4),
        num_heads: tuple[int, ...] = (2, 4, 8, 16, 16, 8, 4, 2),
        window_size: tuple[int, int] = (4, 4),
        mlp_ratio: float = 2.0,
        drop_path_rate: float = 0.1,
        back_rbs: int = 0,
    ):
        super().__init__()
        if len(depths) % 2 != 0 or len(num_heads) != len(depths):
            raise ValueError(
                "depths must have even length and num_heads the same length, got "
                f"{depths} and {num_heads}."
            )
        if upscale < 2 or upscale & (upscale - 1) != 0:
            raise ValueError(f"upscale must be a power of two >= 2, got {upscale}.")
        self.n_in = n_in
        self.n_out = n_out
        self.n_stages = len(depths) // 2
        self.pad_multiple = 2 ** (self.n_stages - 1)
        in_frames = input_frame_positions(n_in, n_out)
        out_frames = list(range(n_out))
        # (n_out, n_in) weights that linearly interpolate the input frames to
        # each output frame, for the decoder query initialization.
        interp = torch.zeros(n_out, n_in)
        for k in range(n_out):
            pos = k * (n_in - 1) / (n_out - 1)
            i = min(int(pos), n_in - 2)
            frac = pos - i
            interp[k, i] = 1.0 - frac
            interp[k, i + 1] = frac
        self.register_buffer("query_interp", interp, persistent=False)

        enc_depths = depths[: self.n_stages]
        dec_depths = depths[self.n_stages :]
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        dec_dpr = enc_dpr[::-1]

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        self.encoder_layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        # Encoder self-attention indexes time by input frame (0..n_in-1), as in
        # the reference; only decoder cross-attention uses output-grid positions.
        for i in range(self.n_stages):
            start = sum(enc_depths[:i])
            self.encoder_layers.append(
                EncoderLayer(
                    embed_dim,
                    enc_depths[i],
                    num_heads[i],
                    list(range(n_in)),
                    window_size,
                    mlp_ratio,
                    enc_dpr[start : start + enc_depths[i]],
                )
            )
            if i != self.n_stages - 1:
                self.downsample.append(nn.Conv2d(embed_dim, embed_dim, 4, 2, 1))
        self.decoder_layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(self.n_stages):
            start = sum(dec_depths[:i])
            self.decoder_layers.append(
                DecoderLayer(
                    embed_dim,
                    dec_depths[i],
                    num_heads[self.n_stages + i],
                    out_frames,
                    in_frames,
                    window_size,
                    mlp_ratio,
                    dec_dpr[start : start + dec_depths[i]],
                )
            )
            if i != self.n_stages - 1:
                self.upsample.append(nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2))

        self.recon_trunk = nn.Sequential(
            *[_ResidualBlockNoBN(embed_dim) for _ in range(back_rbs)]
        )
        n_up = int(math.log2(upscale))
        self.upconvs = nn.ModuleList(
            [nn.Conv2d(embed_dim, embed_dim * 4, 3, 1, 1) for _ in range(n_up - 1)]
            + [nn.Conv2d(embed_dim, 64 * 4, 3, 1, 1)]
        )
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.hr_conv = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _per_frame(self, fn, x: torch.Tensor) -> torch.Tensor:
        """Apply a 2D ``fn`` to every frame of a ``(B, D, C, H, W)`` tensor."""
        b, d = x.shape[:2]
        y = fn(x.reshape(b * d, *x.shape[2:]))
        return y.reshape(b, d, *y.shape[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, n_in, H, W) -> (B, C_out, n_out, H * upscale, W * upscale)``."""
        b, _, n_in, h, w = x.shape
        if n_in != self.n_in:
            raise ValueError(f"Expected {self.n_in} input frames, got {n_in}.")
        x = x.permute(0, 2, 1, 3, 4)  # (B, D, C, H, W)
        x = self._per_frame(self.input_proj, x)
        hp = math.ceil(h / self.pad_multiple) * self.pad_multiple
        wp = math.ceil(w / self.pad_multiple) * self.pad_multiple
        x = F.pad(x, (0, wp - w, 0, hp - h))

        x = x.permute(0, 1, 3, 4, 2)  # (B, D, H, W, C)
        encoder_features = []
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            encoder_features.append(x)
            if i != self.n_stages - 1:
                x = self._per_frame(
                    self.downsample[i], x.permute(0, 1, 4, 2, 3)
                ).permute(0, 1, 3, 4, 2)

        y = torch.einsum("ki,bihwc->bkhwc", self.query_interp.to(x.dtype), x)
        for i, layer in enumerate(self.decoder_layers):
            y = layer(y, encoder_features[-i - 1])
            if i != self.n_stages - 1:
                y = self._per_frame(self.upsample[i], y.permute(0, 1, 4, 2, 3)).permute(
                    0, 1, 3, 4, 2
                )

        y = y.permute(0, 1, 4, 2, 3)[:, :, :, :h, :w]  # (B, n_out, C, h, w)
        out = y.reshape(b * self.n_out, *y.shape[2:])
        out = self.recon_trunk(out)
        for upconv in self.upconvs:
            out = self.lrelu(self.pixel_shuffle(upconv(out)))
        out = self.conv_last(self.lrelu(self.hr_conv(out)))
        out = out.reshape(b, self.n_out, *out.shape[1:])
        return out.permute(0, 2, 1, 3, 4)
