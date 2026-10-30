"""Pseudo-temporal space-time reconstruction pretraining (Choi et al., "Time
Without Time: Pseudo-Temporal Representation for Space-Time Super-Resolution",
CVPR 2026), adapted to endpoint-conditioned downscaling.

The pretext task duplicates a single snapshot into every frame of a clip and
zero-fills a random subset of non-overlapping blocks independently in each input
frame; the network reconstructs the clean, full-frame-rate, high-resolution clip.
The loss is a squared error weighted per output region by ``1 / (1 + v)``, where
``v`` is the number of input frames in which that region's block is visible, so
regions that are hidden in every input frame (and must be inpainted from context)
weigh most.

Here the "snapshot" is one time step of a training clip: its coarse frame is
copied into both endpoint inputs and its fine frame is the target at every output
frame. Masking zero-fills in normalized units, i.e. sets the block to the
climatological mean.
"""

import torch


def random_block_mask(
    batch_size: int,
    n_frames: int,
    height: int,
    width: int,
    block_size: int,
    mask_ratio: float,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """``(B, 1, n_frames, height, width)`` visibility mask: 1 where the input is
    kept, 0 where it is zero-filled.

    Each frame of each sample masks exactly ``round(mask_ratio * n_blocks)`` of
    its ``block_size x block_size`` blocks, drawn without replacement and
    independently per frame (the paper's spatio-temporal-agnostic random
    masking). Blocks at the bottom/right edge are partial when ``height`` or
    ``width`` is not a multiple of ``block_size``.
    """
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}.")
    n_bh = -(-height // block_size)
    n_bw = -(-width // block_size)
    n_blocks = n_bh * n_bw
    n_masked = round(mask_ratio * n_blocks)
    scores = torch.rand(
        batch_size * n_frames, n_blocks, generator=generator, device=device
    )
    masked_idx = scores.argsort(dim=1)[:, :n_masked]
    visible = torch.ones(batch_size * n_frames, n_blocks, device=device)
    visible.scatter_(1, masked_idx, 0.0)
    visible = visible.reshape(batch_size, 1, n_frames, n_bh, n_bw)
    visible = visible.repeat_interleave(block_size, dim=-2).repeat_interleave(
        block_size, dim=-1
    )
    return visible[..., :height, :width]


def visibility_weight(visible: torch.Tensor, scale: int) -> torch.Tensor:
    """Per-output-pixel loss weight ``1 / (1 + v)`` from a ``(B, 1, T_in, h, w)``
    input visibility mask, where ``v`` counts the input frames in which the pixel
    is visible. Returned as ``(B, 1, 1, h * scale, w * scale)``, broadcasting over
    channels and output frames (all output frames show the same snapshot, so the
    weight is the same at every output frame).
    """
    count = visible.sum(dim=2, keepdim=True)
    weight = 1.0 / (1.0 + count)
    return weight.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)


def pseudo_temporal_sample(
    coarse_clip: torch.Tensor,
    fine_clip: torch.Tensor,
    n_inputs: int,
    block_size: int,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one pseudo-temporal sample per clip from a random time step.

    Args:
        coarse_clip: ``(B, C, T, h, w)`` normalized coarse clip.
        fine_clip: ``(B, C, T, H, W)`` normalized fine clip of the same frames.
        n_inputs: Number of input frames to build (the model's input frames).
        block_size: Mask block size, in coarse pixels.
        mask_ratio: Fraction of blocks zero-filled in each input frame.
        generator: Optional RNG for the frame choice and the masks.

    Returns:
        ``(inputs, target, weight)``: masked inputs ``(B, C, n_inputs, h, w)``,
        the fine snapshot repeated over all ``T`` frames ``(B, C, T, H, W)``, and
        the ``visibility_weight`` ``(B, 1, 1, H, W)``.
    """
    batch_size, _, n_times, height, width = coarse_clip.shape
    scale = fine_clip.shape[-2] // height
    device = coarse_clip.device
    frame = torch.randint(n_times, (batch_size,), generator=generator, device=device)
    rows = torch.arange(batch_size, device=device)
    coarse = coarse_clip.permute(0, 2, 1, 3, 4)[rows, frame]  # (B, C, h, w)
    fine = fine_clip.permute(0, 2, 1, 3, 4)[rows, frame]  # (B, C, H, W)
    visible = random_block_mask(
        batch_size,
        n_inputs,
        height,
        width,
        block_size,
        mask_ratio,
        generator=generator,
        device=device,
    ).to(coarse.dtype)
    inputs = coarse.unsqueeze(2) * visible
    target = fine.unsqueeze(2).expand(-1, -1, n_times, -1, -1)
    return inputs, target, visibility_weight(visible, scale)
