"""PyTorch port of the two-frame V3 model, used as a learned temporal baseline.

V3 (Becker et al., "Continuous Space-Time Video Super-Resolution with 3D Fourier
Fields", ICLR 2026, https://github.com/prs-eth/v3) was adapted to paired 100 km ->
25 km climate data and trained in JAX with two coarse endpoint frames as input
(``~/Projects/v3``, branch ``climate-adapt``). This module re-implements exactly that
inference path in PyTorch, so the baseline can be evaluated inside the diffusion
model without a cross-framework call. Weights are converted from the JAX checkpoint
with ``scripts/v3_baseline/convert_checkpoint.py`` and the port is checked against
the JAX outputs in ``scripts/v3_baseline/check_against_jax.py``.

Only the two-frame path exists in the port: with a single clip the recurrent model
never uses optical flow or deformable attention, so neither is implemented.

Conventions (as in the JAX code): features are channels-last ``(B, T, H, W, C)``;
coordinates are normalized per axis to ``[-0.5, 0.5]``; the decoded value is the
Fourier-field residual plus the nearest-endpoint, nearest-neighbour-upsampled source.
"""

import dataclasses
import math
from collections.abc import Sequence
from functools import reduce
from operator import mul
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_STEPS = 9  # fine time steps spanned by the two endpoint frames
SEQ_LEN = 2


@dataclasses.dataclass(frozen=True)
class V3Config:
    n_channels: int = 5
    embed_dim: int = 90
    num_blocks: tuple[int, int, int] = (2, 4, 2)
    depths: tuple[int, int, int] = (2, 2, 2)
    num_heads: int = 6
    output_dims: int = 256
    num_basis: int = 512
    scale: float = 4.0
    window_size: tuple[int, int, int] = (2, 8, 8)
    mlp_ratio: float = 2.0


def time_assignment(
    n_steps: int = N_STEPS, seq_len: int = SEQ_LEN
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest input frame of every fine step and the step time relative to it (in units
    of
    the input-frame spacing); ties go to the earlier frame (as in the released V3 code).
    """
    hr = np.linspace(-0.5, 0.5, n_steps, dtype=np.float64)
    lr = np.linspace(-0.5, 0.5, seq_len, dtype=np.float64)
    hr_int = (hr + 0.5) * (seq_len - 1) - 1e-5
    idx = np.clip(np.round(hr_int).astype(np.int64), 0, seq_len - 1)
    rel = ((hr - lr[idx]) * (seq_len - 1)).astype(np.float32)
    return idx, rel


def _window_partition(x: torch.Tensor, ws: Sequence[int]) -> torch.Tensor:
    b, d, h, w, c = x.shape
    x = x.reshape(b, d // ws[0], ws[0], h // ws[1], ws[1], w // ws[2], ws[2], c)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, reduce(mul, ws), c)


def _window_reverse(windows, ws, b, d, h, w) -> torch.Tensor:
    x = windows.reshape(b, d // ws[0], h // ws[1], w // ws[2], ws[0], ws[1], ws[2], -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, d, h, w, -1)


def _get_window_size(x_size, window_size, shift_size):
    use_window, use_shift = list(window_size), list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window[i] = x_size[i]
            use_shift[i] = 0
    return tuple(use_window), tuple(use_shift)


def _make_att_mask(d, h, w, window_size, shift_size) -> torch.Tensor:
    """Literal port of ``STG.make_att_mask`` (incl. its behaviour for zero shifts)."""
    img_mask = np.zeros((1, d, h, w, 1), dtype=np.float32)
    cnt = 0
    for sd in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for sh in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for sw in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                img_mask[:, sd, sh, sw, :] = cnt
                cnt += 1
    windows = _window_partition(torch.from_numpy(img_mask), window_size)[..., 0]
    diff = windows[:, None] - windows[:, :, None]
    return torch.where(diff != 0.0, -100.0, 0.0)


class _WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: Sequence[int], num_heads: int):
        super().__init__()
        self.dim, self.num_heads, self.window_size = dim, num_heads, tuple(window_size)
        n_bias = math.prod(2 * s - 1 for s in window_size)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(n_bias, num_heads))
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        ws = window_size
        idx = np.stack(np.meshgrid(*[np.arange(s) for s in ws], indexing="ij")).reshape(
            3, -1
        )
        rel = (idx[:, :, None] - idx[:, None, :]).transpose(1, 2, 0).copy()
        rel[:, :, 0] += ws[0] - 1
        rel[:, :, 1] += ws[1] - 1
        rel[:, :, 2] += ws[2] - 1
        rel[:, :, 0] *= (2 * ws[1] - 1) * (2 * ws[2] - 1)
        rel[:, :, 1] *= 2 * ws[2] - 1
        self.register_buffer(
            "rel_index", torch.from_numpy(rel.sum(-1)).long(), persistent=False
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q = q * (self.dim // self.num_heads) ** -0.5
        att = q.float() @ k.float().transpose(-2, -1)
        # The released JAX code reads the bias with ``table.take(index)`` and no axis,
        # which
        # flattens the (n_bias, heads) table first, so one (n, n) bias is shared by all
        # heads.
        # The weights were trained with that behaviour, so it is reproduced here on
        # purpose.
        bias = self.relative_position_bias_table.reshape(-1)[
            self.rel_index[:n, :n].reshape(-1)
        ]
        att = att + bias.reshape(1, 1, n, n)
        if mask is not None:
            nw = mask.shape[0]
            att = att.reshape(b // nw, nw, self.num_heads, n, n)
            att = att + mask[None, :, None, :n, :n]
            att = att.reshape(-1, self.num_heads, n, n)
        att = att.softmax(-1).to(v.dtype)
        out = (att @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class _STL(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio):
        super().__init__()
        self.window_size, self.shift_size = tuple(window_size), tuple(shift_size)
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = _WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp_fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.mlp_fc2 = nn.Linear(int(dim * mlp_ratio), dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, d, h, w, c = x.shape
        ws, shift = _get_window_size((d, h, w), self.window_size, self.shift_size)
        if tuple(ws) != self.window_size:
            raise ValueError(
                f"V3 baseline needs T, H, W >= window {self.window_size}; "
                f"got {(d, h, w)}"
            )
        res1 = x
        x = self.norm1(x)
        pad_d = (ws[0] - d % ws[0]) % ws[0]
        pad_h = (ws[1] - h % ws[1]) % ws[1]
        pad_w = (ws[2] - w % ws[2]) % ws[2]
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        _, dp, hp, wp, _ = x.shape
        shifted = any(s > 0 for s in shift)
        if shifted:
            x = torch.roll(x, (-shift[0], -shift[1], -shift[2]), dims=(1, 2, 3))
        windows = _window_partition(x, self.window_size)
        windows = self.attn(windows, mask if shifted else None)
        x = _window_reverse(windows.reshape(-1, *ws, c), ws, b, dp, hp, wp)
        if shifted:
            x = torch.roll(x, tuple(shift), dims=(1, 2, 3))
        if pad_d or pad_h or pad_w:
            x = x[:, :d, :h, :w, :]
        x = res1 + x
        res2 = x
        y = self.norm2(x)
        y = self.mlp_fc2(F.gelu(self.mlp_fc1(y), approximate="tanh"))
        return res2 + y


class _RSTB(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio):
        super().__init__()
        self.window_size = tuple(window_size)
        self.shift = tuple(s // 2 for s in window_size)
        self.layers = nn.ModuleList(
            [
                _STL(
                    dim,
                    num_heads,
                    window_size,
                    (0, 0, 0) if i % 2 == 0 else self.shift,
                    mlp_ratio,
                )
                for i in range(depth)
            ]
        )
        self.dense = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, d, h, w, _ = x.shape
        ws, shift = _get_window_size((d, h, w), self.window_size, self.shift)
        dp = int(np.ceil(d / ws[0])) * ws[0]
        hp = int(np.ceil(h / ws[1])) * ws[1]
        wp = int(np.ceil(w / ws[2])) * ws[2]
        mask = _make_att_mask(dp, hp, wp, ws, shift).to(x.device, x.dtype)
        y = x
        for i, layer in enumerate(self.layers):
            # odd layers use the (clamped) shift, as in the JAX code
            if i % 2 == 1:
                layer.shift_size = shift
            y = layer(y, mask)
        return x + self.dense(y)


class _RSTBWithInputConv(nn.Module):
    def __init__(
        self, in_dim, dim, num_blocks, depth, num_heads, window_size, mlp_ratio
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_dim, dim, (1, 3, 3), padding=(0, 1, 1))
        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.blocks = nn.ModuleList(
            [
                _RSTB(dim, depth, num_heads, window_size, mlp_ratio)
                for _ in range(num_blocks)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = self.norm_in(x)
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)


class _ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 3, padding=1, bias=False)
        self.norm = nn.LayerNorm(
            dim, eps=1e-6
        )  # the hypernetwork's LayerNorm uses flax's default
        self.conv2 = nn.Conv3d(dim, 4 * dim, 1)
        self.conv4 = nn.Conv3d(4 * dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv0(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        y = self.norm(y)
        y = self.conv2(y.permute(0, 4, 1, 2, 3))
        y = F.gelu(y, approximate="tanh")
        y = self.conv4(y).permute(0, 2, 3, 4, 1)
        return x + y


class V3Baseline(nn.Module):
    """Two-frame V3: RVRT encoder (no flow), hypernetwork, Fourier-field decoder."""

    def __init__(self, config: V3Config = V3Config()):
        super().__init__()
        self.config = config
        c, dim, wsz = config, config.embed_dim, config.window_size
        w1 = (1, wsz[1], wsz[2])
        kw = dict(num_heads=c.num_heads, mlp_ratio=c.mlp_ratio)
        self.feat_extract = _RSTBWithInputConv(
            c.n_channels, dim, c.num_blocks[0], c.depths[0], window_size=w1, **kw
        )
        self.module_names = ["backward_1", "forward_1", "backward_2", "forward_2"]
        self.backbones = nn.ModuleDict(
            {
                name: _RSTBWithInputConv(
                    dim * (2 + i),
                    dim,
                    c.num_blocks[1],
                    c.depths[1],
                    window_size=wsz,
                    **kw,
                )
                for i, name in enumerate(self.module_names)
            }
        )
        self.reconstruction = _RSTBWithInputConv(
            dim * 5, dim, c.num_blocks[2], c.depths[2], window_size=w1, **kw
        )
        self.conv_before_upsampler = nn.Conv3d(dim, c.output_dims, 1)
        self.refine = _ConvNeXtBlock(c.output_dims)
        self.out_conv = nn.Linear(
            c.output_dims, c.num_basis + c.num_basis * c.n_channels
        )
        self.register_parameter("freqs", nn.Parameter(torch.zeros(3, c.num_basis)))
        self.register_parameter("k", nn.Parameter(torch.zeros(())))

    @torch.no_grad()
    def encode(self, source: torch.Tensor) -> torch.Tensor:
        """source: (B, 2, h, w, C) normalized endpoint frames -> (B, 2, h, w, E)."""
        shallow = self.feat_extract(source)
        feats = {"shallow": shallow}
        for name in self.module_names:
            other = [v for k, v in feats.items()]
            backward = name.startswith("backward")
            if backward:
                other = [torch.flip(f, dims=(1,)) for f in other]
            feat_prop = torch.zeros_like(shallow)
            x = torch.cat(other + [feat_prop], dim=-1)
            out = feat_prop + self.backbones[name](x)
            feats[name] = torch.flip(out, dims=(1,)) if backward else out
        hr = torch.cat(list(feats.values()), dim=-1)
        hr = self.reconstruction(hr)
        hr = self.conv_before_upsampler(hr.permute(0, 4, 1, 2, 3))
        hr = F.leaky_relu(hr, 0.1).permute(0, 2, 3, 4, 1)
        return self.refine(hr)

    @torch.no_grad()
    def decode_frames(
        self,
        encoding: torch.Tensor,
        source: torch.Tensor,
        steps: Sequence[int],
        n_steps: int = N_STEPS,
    ) -> torch.Tensor:
        """Fine-grid values at the given time steps: (B, len(steps), 4h, 4w, C)."""
        cfg = self.config
        b, _, h, w, _ = encoding.shape
        factor = int(round(cfg.scale))
        idx_all, rel_all = time_assignment(n_steps)
        params = self.out_conv(encoding)  # (B, 2, h, w, num_basis * (1 + C))
        # the hypernetwork emits the flattened field parameters in jax's sorted-key
        # order:
        # the (num_basis, C) output kernel first, then the (num_basis,) phase
        n_kernel = cfg.num_basis * cfg.n_channels
        kernel = params[..., :n_kernel].reshape(
            b, SEQ_LEN, h, w, cfg.num_basis, cfg.n_channels
        )
        phase = params[..., n_kernel:]
        # sub-pixel offsets of the fine pixel centres from their coarse pixel centre, in
        # coarse
        # pixel units (pixel centres of a 4x finer grid)
        offs = (torch.arange(factor, device=encoding.device) + 0.5) / factor - 0.5
        dy, dx = torch.meshgrid(offs, offs, indexing="ij")
        dy, dx = dy.reshape(-1), dx.reshape(-1)  # (factor^2,)
        spatial = (cfg.scale**-2) * self.k * (self.freqs[:2] ** 2).sum(0)
        damp = torch.exp(-spatial)  # (num_basis,)
        outs = []
        for step in steps:
            f_idx, z = int(idx_all[step]), float(rel_all[step])
            arg = (
                dy[:, None] * self.freqs[0][None]
                + dx[:, None] * self.freqs[1][None]
                + z * self.freqs[2][None]
            )  # (f^2, num_basis)
            ph = phase[:, f_idx]  # (B, h, w, num_basis)
            wave = torch.sin(arg[None, None, None] + ph[:, :, :, None]) * damp
            res = torch.einsum("bhwsn,bhwnc->bhwsc", wave, kernel[:, f_idx])
            res = res.reshape(b, h, w, factor, factor, cfg.n_channels)
            res = res.permute(0, 1, 3, 2, 4, 5).reshape(
                b, h * factor, w * factor, cfg.n_channels
            )
            base = (
                source[:, f_idx]
                .repeat_interleave(factor, 1)
                .repeat_interleave(factor, 2)
            )
            outs.append(res + base)
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def forward(
        self, source: torch.Tensor, steps: Sequence[int] | None = None
    ) -> torch.Tensor:
        steps = list(range(N_STEPS)) if steps is None else list(steps)
        return self.decode_frames(self.encode(source), source, steps)

    @classmethod
    def from_jax_params(cls, params: dict, config: V3Config) -> "V3Baseline":
        """Build from the nested dict of numpy arrays of a JAX V3 checkpoint
        (``pickle.load(...)["params"]``).
        """
        model = cls(config)
        enc = params["encoder"]

        def lin(p):  # flax Dense -> torch Linear
            return {
                "weight": torch.as_tensor(np.asarray(p["kernel"])).T,
                "bias": torch.as_tensor(np.asarray(p["bias"])),
            }

        def conv(p):  # flax Conv (k..., in, out) -> torch Conv3d (out, in, k...)
            k = torch.as_tensor(np.asarray(p["kernel"])).permute(4, 3, 0, 1, 2)
            out = {"weight": k}
            if "bias" in p:
                out["bias"] = torch.as_tensor(np.asarray(p["bias"]))
            return out

        def ln(p):
            return {
                "weight": torch.as_tensor(np.asarray(p["LayerNorm_0"]["scale"])),
                "bias": torch.as_tensor(np.asarray(p["LayerNorm_0"]["bias"])),
            }

        def load_rstb_conv(mod: _RSTBWithInputConv, p: dict) -> None:
            mod.conv.load_state_dict(conv(p["Conv_0"]))
            mod.norm_in.load_state_dict(ln(p["LayerNorm_0"]))
            mod.norm_out.load_state_dict(ln(p["LayerNorm_1"]))
            for i, block in enumerate(mod.blocks):
                lp = p["Layers_0"][f"layers_{i}"]
                block.dense.load_state_dict(lin(lp["Dense_0"]))
                for j, layer in enumerate(block.layers):
                    sp = lp["STG_0"][f"STL_{j}"]
                    layer.norm1.load_state_dict(ln(sp["LayerNorm_0"]))
                    layer.norm2.load_state_dict(ln(sp["LayerNorm_1"]))
                    layer.mlp_fc1.load_state_dict(lin(sp["Mlp_0"]["Dense_0"]))
                    layer.mlp_fc2.load_state_dict(lin(sp["Mlp_0"]["Dense_1"]))
                    ap = sp["WindowAttention_0"]
                    layer.attn.qkv.load_state_dict(lin(ap["qkv"]))
                    layer.attn.proj.load_state_dict(lin(ap["proj"]))
                    layer.attn.relative_position_bias_table.data.copy_(
                        torch.as_tensor(np.asarray(ap["relative_position_bias_table"]))
                    )

        load_rstb_conv(model.feat_extract, enc["feat_extract"])
        for name in model.module_names:
            load_rstb_conv(model.backbones[name], enc[f"backbone_{name}"])
        load_rstb_conv(model.reconstruction, enc["reconstruction"])
        model.conv_before_upsampler.load_state_dict(
            conv(enc["conv_before_upsampler"]["layers_0"])
        )
        res = params["refine"]["layers_0"]["residual"]
        model.refine.conv0.load_state_dict(conv(res["layers_0"]))
        model.refine.norm.load_state_dict(ln(res["layers_1"]))
        model.refine.conv2.load_state_dict(conv(res["layers_2"]))
        model.refine.conv4.load_state_dict(conv(res["layers_4"]))
        oc = params["out_conv"]
        model.out_conv.load_state_dict(
            {
                "weight": torch.as_tensor(np.asarray(oc["kernel"]))[0, 0, 0].T,
                "bias": torch.as_tensor(np.asarray(oc["bias"])),
            }
        )
        model.freqs.data.copy_(torch.as_tensor(np.asarray(params["freqs"]))[0, 0])
        model.k.data.copy_(torch.as_tensor(np.asarray(params["k"])))
        return model


@dataclasses.dataclass(frozen=True)
class V3Normalization:
    """Per-channel normalization the V3 weights were trained with (applied to values
    that are
    already in log1p space for the channels listed in ``log_scales``).
    """

    names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    log_scales: dict[str, float]


def load_v3_baseline(
    path: str, device: torch.device | str = "cpu"
) -> tuple[V3Baseline, V3Normalization]:
    """Load a checkpoint written by ``scripts/v3_baseline/convert_checkpoint.py``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    kwargs: dict[str, Any] = {
        k: tuple(v) if isinstance(v, list) else v for k, v in ckpt["config"].items()
    }
    config = V3Config(**kwargs)
    model = V3Baseline(config)
    model.load_state_dict(ckpt["state_dict"])
    norm = ckpt["normalization"]
    return model.to(device).eval(), V3Normalization(
        names=tuple(norm["names"]),
        means=tuple(norm["means"]),
        stds=tuple(norm["stds"]),
        log_scales=dict(norm["log_scales"]),
    )


@torch.no_grad()
def v3_coarse_interior(
    model: V3Baseline,
    norm: V3Normalization,
    endpoints: dict[str, torch.Tensor],
    n_steps: int = N_STEPS,
) -> dict[str, torch.Tensor]:
    """Coarse-grid interior frames implied by V3's fine-grid output.

    Args:
        model: The V3 model.
        norm: Its normalization.
        endpoints: Per channel ``(B, 2, h, w)`` first and last coarse frames, in log1p
            space for the channels in ``norm.log_scales`` (physical otherwise).
        n_steps: Total time steps including the two endpoints.

    Returns:
        Per channel ``(B, n_steps - 2, h, w)`` interior frames in the same space as
        the input: the fine output is converted to physical values (inverting the log
        transform), averaged over each coarse cell, and transformed back.
    """
    ends = torch.stack(
        [
            (endpoints[name] - norm.means[i]) / norm.stds[i]
            for i, name in enumerate(norm.names)
        ],
        dim=-1,
    )
    fine = model(ends.float(), steps=range(1, n_steps - 1))
    coarse_h, coarse_w = ends.shape[2:4]
    factor = fine.shape[-3] // coarse_h
    interior = {}
    for i, name in enumerate(norm.names):
        value = fine[..., i] * norm.stds[i] + norm.means[i]
        scale = norm.log_scales.get(name)
        if scale is not None:
            value = torch.expm1(value).clamp(min=0.0) / scale
        batch, steps = value.shape[:2]
        value = value.reshape(batch, steps, coarse_h, factor, coarse_w, factor)
        value = value.mean(dim=(3, 5))
        interior[name] = torch.log1p(value * scale) if scale is not None else value
    return interior
