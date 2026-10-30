import os

import dacite
import pytest
import torch

from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.ema import EMAConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.optimization import NullOptimization, OptimizationConfig
from fme.downscaling.data.config import PairedDataLoaderConfig
from fme.downscaling.data.utils import ClosedInterval
from fme.downscaling.stvsr import (
    PseudoTemporalPretrainingConfig,
    RSTTConfig,
    STVSRModel,
    STVSRModelConfig,
    parse_video_model_config,
)
from fme.downscaling.test_utils import data_paths_helper
from fme.downscaling.video_inference import load_video_model
from fme.downscaling.video_models import (
    VideoDiffusionModelConfig,
    _upsample_coarse_clip,
)
from fme.downscaling.video_train import (
    VideoTrainerConfig,
    init_weights_from_checkpoint,
    restore_checkpoint,
)

OUT_NAMES = ["var0", "var1"]
N_TIMESTEPS = 5
UPSCALE = 2  # data_paths_helper: 8x8 coarse, 16x16 fine


def _model_config(pretraining: bool = False) -> STVSRModelConfig:
    return STVSRModelConfig(
        rstt=RSTTConfig(
            embed_dim=16,
            depths=[1, 1, 1, 1],
            num_heads=[2, 2, 2, 2],
            window_size=[2, 2],
        ),
        n_timesteps=N_TIMESTEPS,
        out_names=OUT_NAMES,
        normalization=NormalizationConfig(
            means={"var0": 0.0, "var1": 0.0}, stds={"var0": 1.0, "var1": 1.0}
        ),
        upscale=UPSCALE,
        pseudo_temporal_pretraining=(
            PseudoTemporalPretrainingConfig(block_size=2) if pretraining else None
        ),
    )


def _data_config(paths) -> PairedDataLoaderConfig:
    return PairedDataLoaderConfig(
        fine=[XarrayDataConfig(paths.fine)],
        coarse=[XarrayDataConfig(paths.coarse)],
        batch_size=2,
        num_data_workers=0,
        strict_ensemble=False,
        lat_extent=ClosedInterval(0, 8),
        lon_extent=ClosedInterval(0, 8),
        n_timesteps=N_TIMESTEPS,
    )


def _trainer_config(tmp_path, name="exp", pretraining=False, **kwargs):
    data_dir = tmp_path / f"data_{name}"
    data_dir.mkdir(exist_ok=True)
    paths = data_paths_helper(data_dir, num_timesteps=18)
    return VideoTrainerConfig(
        model=_model_config(pretraining),
        optimization=OptimizationConfig(lr=1e-3),
        train_data=_data_config(paths),
        validation_data=_data_config(paths),
        test_data=_data_config(paths),
        test_interval=1,
        max_epochs=1,
        experiment_dir=str(tmp_path / name),
        logging=LoggingConfig(
            log_to_screen=False, log_to_wandb=False, log_to_file=False
        ),
        save_checkpoints=True,
        validate_using_ema=True,
        ema=EMAConfig(decay=0.9),
        **kwargs,
    )


def _first_batch(trainer):
    return next(iter(trainer.train_data.get_generator()))


def _eval_model(trainer) -> STVSRModel:
    """The model in eval mode (no stochastic depth), as the trainer and
    inference use it for generation.
    """
    trainer.model.modules.eval()
    return trainer.model


def test_prediction_is_baseline_when_network_outputs_zero(tmp_path):
    trainer = _trainer_config(tmp_path).build()
    model: STVSRModel = trainer.model
    for p in model.module.parameters():
        torch.nn.init.zeros_(p)
    batch = _first_batch(trainer)
    out = model.generate(batch)
    for name in OUT_NAMES:
        pred = out[name][:, 0]  # (B, T, H, W)
        # endpoints are the bicubic-upsampled coarse endpoints (identity
        # normalization, so normalized and physical units agree)
        coarse_endpoints = batch.coarse.data[name][:, [0, -1]].unsqueeze(1)
        upsampled = _upsample_coarse_clip(coarse_endpoints, (16, 16))[:, 0]
        torch.testing.assert_close(pred[:, [0, -1]], upsampled)
        # interior frames are the linear interpolation of the endpoints
        tau = torch.linspace(0, 1, N_TIMESTEPS, device=pred.device)
        expected = (1 - tau)[:, None, None] * pred[:, :1] + tau[:, None, None] * pred[
            :, -1:
        ]
        torch.testing.assert_close(pred, expected)


def test_generate_shapes_frames_and_determinism(tmp_path):
    trainer = _trainer_config(tmp_path).build()
    model = _eval_model(trainer)
    batch = _first_batch(trainer)
    out = model.generate(batch, n_samples=3)
    for name in OUT_NAMES:
        assert out[name].shape == (2, 3, N_TIMESTEPS, 16, 16)
        assert torch.equal(out[name][:, 0], out[name][:, 2])
    subset = model.generate(batch, n_samples=1, frames=[0, 2, 4])
    torch.testing.assert_close(subset["var0"], out["var0"][:, :1, [0, 2, 4]])
    no_target = model.generate_on_batch_no_target(batch.coarse)
    torch.testing.assert_close(no_target["var0"], out["var0"][:, :1])


def test_generate_does_not_read_fine_values(tmp_path):
    trainer = _trainer_config(tmp_path).build()
    model = _eval_model(trainer)
    batch = _first_batch(trainer)
    before = model.generate(batch)["var0"]
    for v in batch.fine.data.values():
        v.zero_()
    torch.testing.assert_close(model.generate(batch)["var0"], before)


@pytest.mark.parametrize("pretraining", [False, True])
def test_training_reduces_loss_on_fixed_batch(tmp_path, pretraining):
    torch.manual_seed(0)
    trainer = _trainer_config(tmp_path, pretraining=pretraining).build()
    batch = _first_batch(trainer)
    model = trainer.model

    def step(optimization):
        # fixed pseudo-temporal frame choice and masks, so the model can fit
        # this one sample and losses are comparable
        torch.manual_seed(1)
        return model.train_on_batch(batch, optimization).loss.item()

    initial = step(NullOptimization())
    for _ in range(30):
        step(trainer.optimization)
    assert step(NullOptimization()) < 0.8 * initial


def test_config_rejects_mismatched_downscale_factor():
    with pytest.raises(ValueError, match="does not match upscale"):
        _model_config().build(downscale_factor=4)


def _model_block(**overrides):
    block = {
        "n_timesteps": N_TIMESTEPS,
        "out_names": ["var0"],
        "normalization": {"means": {"var0": 0.0}, "stds": {"var0": 1.0}},
    }
    return {**block, **overrides}


def test_trainer_config_parses_either_model_type(tmp_path):
    data = {
        "fine": [],
        "coarse": [],
        "batch_size": 1,
        "num_data_workers": 0,
        "strict_ensemble": False,
        "n_timesteps": N_TIMESTEPS,
    }
    base = {
        "optimization": {"lr": 1e-3},
        "train_data": data,
        "validation_data": data,
        "max_epochs": 1,
        "experiment_dir": str(tmp_path),
        "logging": {},
    }
    for model, expected in [
        (_model_block(rstt={"embed_dim": 16}), STVSRModelConfig),
        (_model_block(), VideoDiffusionModelConfig),
    ]:
        assert isinstance(parse_video_model_config(model), expected)
        config = dacite.from_dict(
            VideoTrainerConfig,
            {**base, "model": model},
            config=dacite.Config(strict=True),
        )
        assert isinstance(config.model, expected)


@pytest.mark.parametrize(
    "block, match",
    [
        (_model_block(n_timesteps=2), "n_timesteps"),  # diffusion needs >= 3
        (_model_block(rstt={}, n_timesteps=1), "n_timesteps"),
        (_model_block(rstt={}, upscale=4, model_channels=8), "model_channels"),
    ],
)
def test_parse_video_model_config_surfaces_model_errors(block, match):
    with pytest.raises(Exception, match=match):
        parse_video_model_config(block)


def test_pretrain_then_finetune_then_inference(tmp_path):
    """The two-stage recipe end to end: pretrain on the pseudo-temporal task,
    fine-tune from the pretrained EMA weights, load the result for inference.
    """
    pretrain = _trainer_config(tmp_path, name="pretrain", pretraining=True)
    pretrainer = pretrain.build()
    pretrainer.train()
    pretrained_path = os.path.join(pretrain.checkpoint_dir, "latest.ckpt")
    assert os.path.isfile(pretrained_path)

    finetune = _trainer_config(
        tmp_path, name="finetune", init_weights_from=pretrained_path
    )
    finetuner = finetune.build()
    init_weights_from_checkpoint(finetuner, pretrained_path)
    # starts from the pretrained run's EMA weights, with a fresh EMA
    with pretrainer._ema_context():
        expected = [p.detach().clone() for p in pretrainer.model.module.parameters()]
    for p, e in zip(finetuner.model.module.parameters(), expected):
        torch.testing.assert_close(p, e)
    with finetuner._ema_context():
        for p, e in zip(finetuner.model.module.parameters(), expected):
            torch.testing.assert_close(p, e)
    assert finetuner.num_batches_seen == 0
    finetuner.train()

    # resuming the fine-tune run restores its own state
    resumed = finetune.build()
    assert resumed.resuming
    restore_checkpoint(resumed)
    assert resumed.num_batches_seen == finetuner.num_batches_seen

    model = load_video_model(
        finetune.model,
        os.path.join(finetune.checkpoint_dir, "best.ckpt"),
        torch.device("cpu"),
    )
    assert isinstance(model, STVSRModel)
    out = model.generate(_first_batch(finetuner), n_samples=2)
    assert out["var0"].shape == (2, 2, N_TIMESTEPS, 16, 16)
