"""Build a video_inference config from a training config's model/test_data blocks.

Usage: make_inference_config.py TRAIN_YAML CKPT OUT_ZARR EXP_DIR START_DATE DAYS N_ENS OUT_YAML [LAT0 LON0 [BATCH]]

Without LAT0/LON0: global domain via divide_generation (tiled with the training patch).
With LAT0 LON0 (integer degrees): a single regional patch [LAT0, LAT0+lat_cells] x
[LON0, LON0+lon_cells] (exactly the model's training patch size at 1 deg coarse cells),
generated in one forward pass -- used for storm-centred runs. BATCH = clips per batch.
"""
import copy
import datetime
import sys

import yaml

train_yaml, ckpt, out_zarr, exp_dir, start, days, n_ens, out_yaml, *region = sys.argv[1:]
cfg = yaml.safe_load(open(train_yaml))
stop = (datetime.date.fromisoformat(start) + datetime.timedelta(days=int(days))).isoformat()

data = copy.deepcopy(cfg["test_data"])
for side in ("fine", "coarse"):
    for entry in data[side]:
        entry["subset"] = {"start_time": start, "stop_time": stop}
data["batch_size"] = max(int(days), 1)  # one raw batch = all daily clips of the window
regional = len(region) >= 2
if regional:
    lat0, lon0 = float(region[0]), float(region[1])
    data["batch_size"] = int(region[2]) if len(region) > 2 else 2
    data["lat_extent"] = {"start": lat0, "stop": lat0 + cfg["coarse_patch_extent_lat"]}
    data["lon_extent"] = {"start": lon0, "stop": lon0 + cfg["coarse_patch_extent_lon"]}
data["num_data_workers"] = 2

inf = {
    "checkpoint_path": ckpt,
    "experiment_dir": exp_dir,
    "output_path": out_zarr,
    "n_ensemble": int(n_ens),
    "ensemble_chunk_size": 1,
    "use_ema": True,
    "resume": True,  # safe to rerun the script after an interruption
    "divide_generation": not regional,
    "logging": {**cfg["logging"], "log_to_wandb": False},
    "model": cfg["model"],
    "data": data,
}
if not regional:
    inf["coarse_patch_extent"] = [cfg["coarse_patch_extent_lat"], cfg["coarse_patch_extent_lon"]]
    inf["coarse_horizontal_overlap"] = 1
yaml.safe_dump(inf, open(out_yaml, "w"), sort_keys=False)
