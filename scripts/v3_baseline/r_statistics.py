"""Statistics of the two-block residual ``r = x_c - I(tau)`` (real training pipeline).

Builds the trainer from a config (so the baseline in the config, V3, vortex-shift or
linear, is applied by the model's own code on the patched training / validation
batches) and measures ``r`` on the interior steps in coarse-normalized units:

- per-channel RMS (-> ``r_sigma_data_by_channel``) for the configured baseline and for
  linear interpolation, on train and held-out windows;
- the 7x7 temporal covariance per channel, saved for ``fit_r_kernel.py``;
- with ``v3_detail_baseline`` in the config, the RMS of the fine-detail block
  ``d = Pi(x_f)`` with and without V3's fine-scale baseline (the new
  ``d_sigma_data_by_channel``).

    python r_statistics.py <video_train.yaml> <out_cov.npz> [n_batches]
"""

import sys
import tempfile

import dacite
import numpy as np
import torch
import yaml

from fme.downscaling.twoblock import coarse_temporal_interp, null_space_projector
from fme.downscaling.video_train import VideoTrainerConfig


def main(config_path: str, out_path: str, n_batches: int = 250) -> None:
    cfg = yaml.safe_load(open(config_path))
    for section in ("train_data", "validation_data"):
        cfg[section]["num_data_workers"] = 0
    cfg["experiment_dir"] = tempfile.mkdtemp(prefix="r_statistics_")
    trainer = dacite.from_dict(
        VideoTrainerConfig, cfg, config=dacite.Config(strict=True)
    ).build()
    model = trainer.model
    names = model.out_names
    generators = {
        "train": trainer._get_batch_generator(
            trainer.train_data, random_offset=True, shuffle=True
        ),
        "val": trainer._get_batch_generator(
            trainer.validation_data, random_offset=True, shuffle=True
        ),
    }
    covariances: dict[str, np.ndarray] = {}
    for split, generator in generators.items():
        residuals: dict[str, list[torch.Tensor]] = {"base": [], "lin": []}
        d_sq: dict[str, list[torch.Tensor]] = {"d": [], "d_res": []}
        for j, batch in enumerate(generator):
            if j >= n_batches:
                break
            coarse = model._pack_normalized(batch.coarse.data, model.coarse_normalizer)
            factor = batch.fine.data[names[0]].shape[-1] // coarse.shape[-1]
            baseline, detail = model._two_block_baselines(coarse, factor)
            if detail is not None:
                fine = model._pack_normalized(batch.fine.data)
                d = null_space_projector(fine, factor)
                for key, x in (("d", d), ("d_res", d - detail)):
                    d_sq[key].append(x.pow(2).mean(dim=(0, 2, 3, 4)).cpu())
            for key, interp in (
                ("base", baseline),
                ("lin", coarse_temporal_interp(coarse)),
            ):
                residuals[key].append((coarse - interp).float().cpu())
        print(f"\n== {split}: {sum(len(r) for r in residuals['base'])} patches")
        print(f"{'channel':34}{'linear rms':>12}{'baseline rms':>14}")
        for i, name in enumerate(names):
            rms = {}
            for key, parts in residuals.items():
                x = torch.cat(parts)[:, i, 1:-1].permute(1, 0, 2, 3)  # (T-2, P, h, w)
                rms[key] = float(x.pow(2).mean().sqrt())
                flat = x.reshape(x.shape[0], -1).double().numpy()
                covariances[f"{split}|{key}|{name}"] = flat @ flat.T / flat.shape[1]
            print(f"{name:34}{rms['lin']:12.4f}{rms['base']:14.4f}")
        if d_sq["d"]:
            print(
                "detail block d (fine-normalized RMS): plain vs. after V3 fine baseline"
            )
            for i, name in enumerate(names):
                plain = float(torch.stack(d_sq["d"])[:, i].mean().sqrt())
                res = float(torch.stack(d_sq["d_res"])[:, i].mean().sqrt())
                print(f"    {name:34}{plain:9.4f}{res:9.4f}   ratio {res / plain:.2f}")
    np.savez(out_path, **covariances)
    print(f"\nsaved covariances to {out_path}")
    print("r_sigma_data_by_channel (mean of train and held-out baseline RMS):")
    for name in names:
        values = [
            np.sqrt(np.trace(covariances[f"{s}|base|{name}"]) / 7) for s in generators
        ]
        print(f"    {name}: {np.mean(values):.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 250)
