"""Per-channel std of the two-block residual ``r = x_c - I(tau)`` for a V3 baseline.

``r_sigma_data_by_channel`` of the two-block configs was calibrated for the linear
baseline; a better baseline shrinks ``r``, so the EDM preconditioning constant must be
re-estimated. Computed in the model's coarse-normalized units (log1p for precipitation),
over random 44x72 coarse patches and 24 h windows of the training years, interior steps
only (r is exactly zero at the endpoints). The linear-baseline numbers are printed as a
check against the values already in the config.

    python calibrate_r_sigma.py <video_train.yaml> <v3_baseline.pt> [n_windows]
"""

import sys

import numpy as np
import torch
import yaml
import zarr

from fme.downscaling.v3_baseline import load_v3_baseline, v3_coarse_interior

PATCH = (44, 72)
N_STEPS = 9
DATA_DIR = "/data/pmd/local_data"
COARSE_STORE = "2026-07-14-X-SHiELD-AMIP-FME-3h-100km.zarr"
TRAIN_STEPS = 8744  # 2013-01-02 .. 2015-12-31 at 3 h


def main(config_path: str, v3_path: str, n_windows: int = 400, seed: int = 0) -> None:
    model_cfg = yaml.safe_load(open(config_path))["model"]
    names = model_cfg["out_names"]
    norm_cfg = model_cfg["coarse_normalization"]
    log_scales = model_cfg.get("log_transform_channels") or {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v3, v3_norm = load_v3_baseline(v3_path, device)
    store = zarr.open_group(f"{DATA_DIR}/{COARSE_STORE}", mode="r", zarr_format=3)
    rng = np.random.default_rng(seed)
    h, w = PATCH
    sq_v3: dict[str, list[np.ndarray]] = {n: [] for n in names}
    sq_lin: dict[str, list[np.ndarray]] = {n: [] for n in names}
    weights = torch.linspace(0.0, 1.0, N_STEPS)[None, :, None, None]
    for i in range(n_windows):
        t0 = int(rng.integers(0, TRAIN_STEPS - N_STEPS))
        r0, c0 = int(rng.integers(0, 180 - h + 1)), int(rng.integers(0, 360 - w + 1))
        clip = {}
        for n in names:
            x = np.asarray(store[n][t0 : t0 + N_STEPS, r0 : r0 + h, c0 : c0 + w])
            x = torch.from_numpy(x).float()
            if n in log_scales:
                x = torch.log1p(x.clamp(min=0.0) * log_scales[n])
            clip[n] = x[None]  # (1, T, h, w), log1p space for log channels
        ends = {n: clip[n][:, [0, -1]].to(device) for n in names}
        interior = v3_coarse_interior(v3, v3_norm, ends, n_steps=N_STEPS)
        for n in names:
            mean, std = norm_cfg["means"][n], norm_cfg["stds"][n]
            truth = (clip[n][:, 1:-1] - mean) / std
            v3_i = (interior[n].cpu() - mean) / std
            lin = ((1 - weights) * clip[n][:, :1] + weights * clip[n][:, -1:])[:, 1:-1]
            lin = (lin - mean) / std
            sq_v3[n].append(((truth - v3_i) ** 2).numpy().ravel())
            sq_lin[n].append(((truth - lin) ** 2).numpy().ravel())
    print(f"r std over {n_windows} windows (coarse-normalized units, interior steps)")
    print(f"{'channel':34}{'linear':>9}{'config':>9}{'V3':>9}{'V3/linear':>11}")
    out = {}
    for n in names:
        lin = float(np.sqrt(np.mean(np.concatenate(sq_lin[n]))))
        v3s = float(np.sqrt(np.mean(np.concatenate(sq_v3[n]))))
        cfg = (model_cfg.get("r_sigma_data_by_channel") or {}).get(n, float("nan"))
        print(f"{n:34}{lin:9.4f}{cfg:9.4f}{v3s:9.4f}{v3s / lin:11.2f}")
        out[n] = round(v3s, 4)
    print("\nr_sigma_data_by_channel for the V3 baseline:")
    for n, v in out.items():
        print(f"    {n}: {v}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 400)
