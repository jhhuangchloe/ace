"""Configs for the cascade (temporal infill -> spatial SR) on one storm window, run locally.

Usage: make_cascade_configs.py OUT_DIR START STOP N_ENS LAT0 LAT1 LON0 LON1

Stage 1 (video_inference): the 5-channel per-channel-OU 1-degree 24h->3h infill model (checkpoint
/data/pmd/checkpoints/5ch-kernel), run on the local 100 km store (the same 1-degree dataset it was trained on),
one member. Its model:/data: blocks are copied from
../2026-07-22-video-pmd-5ch-per-channel-kernel-test-inference/video_inference.yaml.
Stage 2 (fme.downscaling.inference): the HiRO SR model (hiro-v6 ema_ckpt.tar) conditioned on member 0 of the stage-1
output (as ../2026-08-24-hiro-downscaling-cascade-infill-then-sr/inference.yaml), n_ens members, regional output.
Writes OUT_DIR/stage1.yaml and OUT_DIR/stage2.yaml.
"""
import copy
import sys
from pathlib import Path

import torch
import yaml

out_dir, start, stop, n_ens, lat0, lat1, lon0, lon1 = sys.argv[1:]
out = Path(out_dir)
REPO = Path(__file__).resolve().parents[3]
LOCAL = "/data/pmd/local_data"
COARSE = "2026-07-14-X-SHiELD-AMIP-FME-3h-100km.zarr"
stage1_zarr = f"{out}/stage1-infill.zarr"

base = yaml.safe_load(open(REPO / "configs/experiments/2026-07-22-video-pmd-5ch-per-channel-kernel-test-inference/video_inference.yaml"))
s1 = copy.deepcopy(base)
s1["checkpoint_path"] = "/data/pmd/checkpoints/5ch-kernel/checkpoints/latest.ckpt"
s1["experiment_dir"] = f"{out}/stage1-logs"
s1["output_path"] = stage1_zarr
s1["n_ensemble"] = 1
s1["resume"] = True
s1["logging"]["name"] = "cascade-stage1-infill-local"
for side in ("fine", "coarse"):
    s1["data"][side] = [{"data_path": LOCAL, "file_pattern": COARSE, "engine": "zarr",
                         "subset": {"start_time": start, "stop_time": stop}}]
s1["data"]["batch_size"] = 1  # the loader drops a partial last batch
s1["data"]["num_data_workers"] = 1
yaml.safe_dump(s1, open(out / "stage1.yaml", "w"), sort_keys=False)

casc = yaml.safe_load(open(REPO / "configs/experiments/2026-08-24-hiro-downscaling-cascade-infill-then-sr/inference.yaml"))
# The HiRO checkpoint was trained with apex GroupNorm (use_apex_gn: true), which is not installed here. The weights
# are the same either way, so write a copy of the checkpoint with use_apex_gn switched off.
hiro_src = "/data/pmd/checkpoints/hiro-v6/checkpoints/ema_ckpt.tar"
hiro = out / "hiro-v6-noapex.ckpt"
if not hiro.exists():
    ck = torch.load(hiro_src, map_location="cpu", weights_only=False)
    ck["model"]["config"]["module"]["config"]["use_apex_gn"] = False
    torch.save(ck, hiro)
    print(f"wrote {hiro} (use_apex_gn=False)")
s2 = {
    "model": {"checkpoint_path": str(hiro)},
    "experiment_dir": f"{out}/stage2",
    "logging": {**casc["logging"], "log_to_wandb": False, "name": "cascade-stage2-sr-local"},
    "data": {
        "coarse": [{"data_path": str(Path(stage1_zarr).parent), "file_pattern": Path(stage1_zarr).name, "engine": "zarr",
                    "isel": {"ensemble": 0}, "subset": {"start_time": start, "stop_time": stop}}],
        "batch_size": 4, "num_data_workers": 1, "strict_ensemble": False,
    },
    "patch": casc["patch"],
    "outputs": [{
        "name": "S2022", "n_ens": int(n_ens), "max_samples_per_gpu": 4, "resume": True,
        "save_vars": casc["outputs"][0]["save_vars"],
        "time_range": {"start_time": start, "stop_time": stop},
        "lat_extent": {"start": float(lat0), "stop": float(lat1)},
        "lon_extent": {"start": float(lon0), "stop": float(lon1)},
    }],
}
yaml.safe_dump(s2, open(out / "stage2.yaml", "w"), sort_keys=False)
print(f"wrote {out}/stage1.yaml and stage2.yaml")
