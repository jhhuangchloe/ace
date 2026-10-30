"""Deterministic space-time super-resolution baseline: RSTT (Geng et al., CVPR
2022) with optional pseudo-temporal pretraining (Choi et al., CVPR 2026).

Same task and I/O as the diffusion models' ``coarse_endpoints_only`` mode: the
two coarse endpoint frames of a clip in, every fine frame of the clip out
(endpoints included, nothing pinned). As in the reference RSTT, the network
predicts a residual over an interpolated baseline; here that baseline is the
linear-in-time interpolation of the bicubic-upsampled endpoints, the same
baseline the single-stage video diffusion model uses. RSTT sees no calendar
inputs, as in the reference.

``STVSRModel`` exposes the subset of ``VideoDiffusionModel``'s interface that
``VideoTrainer`` and ``video_inference`` use, so it trains and runs inference
through the same entry points. Generation is deterministic: every ensemble
member is the same prediction.

Two-stage recipe from the pretraining paper:
1. Pretrain with ``pseudo_temporal_pretraining`` set (see ``pseudo_temporal.py``).
2. Fine-tune on real clips with it unset, starting from the pretrained weights
   via ``VideoTrainerConfig.init_weights_from``.
Training only step 2 from scratch is the no-pretraining control.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

import dacite
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.packer import Packer
from fme.core.typing_ import TensorDict, TensorMapping
from fme.downscaling.data import PairedVideoBatchData, VideoBatchData
from fme.downscaling.models import ModelOutputs
from fme.downscaling.modules.rstt import RSTT
from fme.downscaling.pseudo_temporal import pseudo_temporal_sample
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.video_models import (
    VideoDiffusionModelConfig,
    _upsample_coarse_clip,
)

CHANNEL_AXIS = 1
N_INPUT_FRAMES = 2  # the two coarse endpoint frames


@dataclasses.dataclass
class RSTTConfig:
    """RSTT architecture; defaults are RSTT-S (``configs/RSTT-S.yml`` of the
    reference implementation). See ``fme.downscaling.modules.rstt.RSTT``.
    """

    embed_dim: int = 96
    depths: list[int] = dataclasses.field(default_factory=lambda: [4] * 8)
    num_heads: list[int] = dataclasses.field(
        default_factory=lambda: [2, 4, 8, 16, 16, 8, 4, 2]
    )
    window_size: list[int] = dataclasses.field(default_factory=lambda: [4, 4])
    mlp_ratio: float = 2.0
    drop_path_rate: float = 0.1
    back_rbs: int = 0

    def __post_init__(self):
        if len(self.window_size) != 2:
            raise ValueError(
                f"window_size must be [height, width], got {self.window_size}."
            )

    def build(self, n_channels: int, n_out: int, upscale: int) -> torch.nn.Module:
        return RSTT(
            in_channels=n_channels,
            out_channels=n_channels,
            n_in=N_INPUT_FRAMES,
            n_out=n_out,
            upscale=upscale,
            embed_dim=self.embed_dim,
            depths=tuple(self.depths),
            num_heads=tuple(self.num_heads),
            window_size=(self.window_size[0], self.window_size[1]),
            mlp_ratio=self.mlp_ratio,
            drop_path_rate=self.drop_path_rate,
            back_rbs=self.back_rbs,
        )


@dataclasses.dataclass
class PseudoTemporalPretrainingConfig:
    """Pretext-task settings; defaults are the paper's (4x4 blocks, 50% masked
    per input frame).

    Parameters:
        block_size: Mask block size, in coarse pixels.
        mask_ratio: Fraction of blocks zero-filled in each input frame.
    """

    block_size: int = 4
    mask_ratio: float = 0.5

    def __post_init__(self):
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}.")
        if not 0.0 <= self.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1), got {self.mask_ratio}.")


@dataclasses.dataclass
class STVSRModelConfig:
    """Configuration for ``STVSRModel``.

    Parameters:
        rstt: Network architecture.
        n_timesteps: Frames per clip; output frames, endpoints included.
        out_names: Variables, used for both input (coarse) and output (fine).
        normalization: Fine-resolution normalization. Also applied to the
            coarse inputs, so the inputs, baseline and output share units.
        upscale: Coarse-to-fine spatial factor.
        log_transform_channels: Channels modeled as ``log1p(x * scale)``, as in
            ``VideoDiffusionModelConfig``.
        pseudo_temporal_pretraining: If set, train on the pseudo-temporal
            pretext task instead of real clips. Validation and generation
            always use real clips.
    """

    rstt: RSTTConfig
    n_timesteps: int
    out_names: list[str]
    normalization: NormalizationConfig
    upscale: int = 4
    log_transform_channels: dict[str, float] | None = None
    pseudo_temporal_pretraining: PseudoTemporalPretrainingConfig | None = None

    def __post_init__(self):
        if self.n_timesteps < N_INPUT_FRAMES:
            raise ValueError(
                f"n_timesteps must be >= {N_INPUT_FRAMES}, got {self.n_timesteps}."
            )
        unknown = set(self.log_transform_channels or {}) - set(self.out_names)
        if unknown:
            raise ValueError(
                f"log_transform_channels {sorted(unknown)} not in out_names."
            )

    @property
    def endpoints_observed(self) -> bool:
        """No frame is pinned to fine truth (read by ``video_inference``)."""
        return False

    @property
    def data_requirements(self) -> DataRequirements:
        return DataRequirements(
            fine_names=self.out_names,
            coarse_names=self.out_names,
            n_timesteps=1,
            use_fine_topography=False,
        )

    def build(
        self,
        full_fine_coords: LatLonCoordinates | None = None,
        downscale_factor: int | None = None,
    ) -> "STVSRModel":
        """Build the model. ``full_fine_coords`` is accepted for signature
        parity with ``VideoDiffusionModelConfig.build`` and unused;
        ``downscale_factor``, when given (by the trainer, from the data), must
        match ``upscale``.
        """
        if downscale_factor is not None and downscale_factor != self.upscale:
            raise ValueError(
                f"Data downscale factor {downscale_factor} does not match "
                f"upscale={self.upscale}."
            )
        module = self.rstt.build(len(self.out_names), self.n_timesteps, self.upscale)
        return STVSRModel(self, module, self.normalization.build(self.out_names))


def parse_video_model_config(
    data: Mapping[str, Any],
) -> VideoDiffusionModelConfig | STVSRModelConfig:
    """Parse a video ``model:`` config block as the class it describes: an
    ``STVSRModelConfig`` if it has an ``rstt`` key, else a
    ``VideoDiffusionModelConfig``.

    Entry points call this on the model block before parsing the whole config,
    because inside the trainer/inference configs' model union dacite reports any
    model validation error only as "can not match ... union".
    """
    config_class: type[VideoDiffusionModelConfig] | type[STVSRModelConfig] = (
        STVSRModelConfig if "rstt" in data else VideoDiffusionModelConfig
    )
    return dacite.from_dict(config_class, data, config=dacite.Config(strict=True))


class STVSRModel:
    # VideoDiffusionModel's optional sub-modules, absent here; defined so
    # VideoTrainer's checkpointing and video_inference.load_video_model, which
    # check them, handle both model types.
    endpoint_sr_module: torch.nn.Module | None = None
    _r_encoder: torch.nn.Module | None = None
    _d_encoder: torch.nn.Module | None = None

    def __init__(
        self,
        config: STVSRModelConfig,
        module: torch.nn.Module,
        normalizer: StandardNormalizer,
    ):
        self.config = config
        self.module = Distributed.get_instance().wrap_module(module.to(get_device()))
        self.normalizer = normalizer
        self.out_names = config.out_names
        self.packer = Packer(config.out_names)
        self.n_timesteps = config.n_timesteps
        self.log_transform_channels = dict(config.log_transform_channels or {})
        self.pretraining = config.pseudo_temporal_pretraining

    @property
    def modules(self) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([self.module])

    def _pack_normalized(self, data: TensorMapping) -> torch.Tensor:
        """``{name: (B, T, H, W)}`` -> normalized ``(B, C, T, H, W)``."""
        selected = {
            k: torch.log1p(data[k].clamp(min=0.0) * scale)
            if (scale := self.log_transform_channels.get(k))
            else data[k]
            for k in self.out_names
        }
        normalized = self.normalizer.normalize(selected)
        return self.packer.pack(normalized, axis=CHANNEL_AXIS)

    def _denormalize_invert(self, packed: torch.Tensor) -> TensorDict:
        """Inverse of ``_pack_normalized``, back to physical units."""
        data = self.normalizer.denormalize(
            self.packer.unpack(packed, axis=CHANNEL_AXIS)
        )
        return {
            k: (torch.expm1(v) / scale).clamp(min=0.0)
            if (scale := self.log_transform_channels.get(k))
            else v
            for k, v in data.items()
        }

    def _baseline(self, inputs: torch.Tensor) -> torch.Tensor:
        """Linear-in-time interpolation of the bicubic-upsampled endpoint frames
        ``(B, C, 2, h, w)`` to ``(B, C, n_timesteps, H, W)``.
        """
        h, w = inputs.shape[-2:]
        upsampled = _upsample_coarse_clip(
            inputs, (h * self.config.upscale, w * self.config.upscale)
        )
        weight = torch.linspace(
            0.0, 1.0, self.n_timesteps, device=inputs.device, dtype=inputs.dtype
        ).reshape(1, 1, -1, 1, 1)
        return (1 - weight) * upsampled[:, :, :1] + weight * upsampled[:, :, 1:]

    def _predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalized ``(B, C, n_timesteps, H, W)`` prediction from normalized
        ``(B, C, 2, h, w)`` endpoint inputs.
        """
        return self._baseline(inputs) + self.module(inputs)

    def train_on_batch(self, batch: PairedVideoBatchData, optimizer) -> ModelOutputs:
        coarse = self._pack_normalized(batch.coarse.data)
        fine = self._pack_normalized(batch.fine.data)
        if self.pretraining is not None:
            inputs, target, weight = pseudo_temporal_sample(
                coarse,
                fine,
                N_INPUT_FRAMES,
                self.pretraining.block_size,
                self.pretraining.mask_ratio,
            )
        else:
            inputs = coarse[:, :, [0, -1]]
            target = fine
            weight = None
        prediction = self._predict(inputs)
        sq_err = (prediction - target) ** 2
        loss = (sq_err if weight is None else sq_err * weight).mean()
        optimizer.accumulate_loss(loss)
        optimizer.step_weights()
        return ModelOutputs(
            prediction=self.packer.unpack(prediction.detach(), axis=CHANNEL_AXIS),
            target=self.packer.unpack(target, axis=CHANNEL_AXIS),
            loss=loss.detach(),
        )

    @torch.no_grad()
    def _generate_from_coarse(
        self,
        coarse_data: TensorMapping,
        n_samples: int,
        frames: list[int] | None,
    ) -> TensorDict:
        coarse = self._pack_normalized(coarse_data)
        prediction = self._predict(coarse[:, :, [0, -1]])
        if frames is not None:
            prediction = prediction[:, :, frames]
        generated = self._denormalize_invert(prediction)
        # (B, T, H, W) -> (B, n_samples, T, H, W); deterministic, so every
        # member is the same prediction.
        return {
            k: v.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
            for k, v in generated.items()
        }

    def generate(
        self,
        batch: PairedVideoBatchData,
        n_samples: int = 1,
        frames: list[int] | None = None,
    ) -> TensorDict:
        """Predict every frame of the clip from the coarse endpoints; with
        ``frames``, return only those frame indices. ``batch.fine`` is not read.
        """
        return self._generate_from_coarse(batch.coarse.data, n_samples, frames)

    def generate_on_batch_no_target(
        self,
        coarse: VideoBatchData,
        n_samples: int = 1,
        frames: list[int] | None = None,
    ) -> TensorDict:
        return self._generate_from_coarse(coarse.data, n_samples, frames)

    def get_state(self) -> Mapping[str, Any]:
        return {
            "config": dataclasses.asdict(self.config),
            "module": self.module.state_dict(),
        }
