"""Write the deterministic TC baselines as one-member "ensemble" zarrs, so tc_score.py / tc_mse_crps.py
score them exactly like the diffusion models.

Usage: baseline_zarr.py ROOT STORM INFERENCE_YAML TRUTH_ZARR NAME [NAME ...]
  NAME: linear-bicubic, linear-nearest (linear_baseline.py), v3-baseline, v3-raw (v3_curve.py)
Output: ROOT/NAME/STORM.zarr on the grid and time axis of INFERENCE_YAML's output zarr (a model run of
the same storm patch); frames that run never generated are left NaN so all methods are scored on the
same frames. Set V3_CACHE_DIR to reuse the V3 CPU inference across scripts.
"""

import os
import sys

import numpy as np
import xarray as xr
import yaml
from linear_baseline import LinearBaseline
from v3_curve import V3Curves

root, storm, inf_yaml, truth_path, *names = sys.argv[1:]
VARS = [
    "eastward_wind_at_ten_meters",
    "northward_wind_at_ten_meters",
    "PRMSL",
    "PRATEsfc",
    "air_temperature_at_two_meters",
]
inf = yaml.safe_load(open(inf_yaml))
ref = xr.open_zarr(inf["output_path"])
generated = np.isfinite(ref["PRMSL"].isel(ensemble=0, latitude=0, longitude=0).values)
coarse_path = truth_path.replace("25km", "100km")
truth_full = xr.open_zarr(truth_path)
v3 = None
if any(n.startswith("v3") for n in names):
    v3 = V3Curves(
        coarse_path,
        truth_full,
        os.environ.get("V3_WEIGHTS", "/data/pmd/v3_baseline/v3_baseline_9yr.pt"),
        inf["data"]["lat_extent"],
        inf["data"]["lon_extent"],
    )

for name in names:
    if name.startswith("linear-"):
        base = LinearBaseline(coarse_path, truth_full, name.split("-", 1)[1])
        get, have = base.field, VARS
    else:
        get, have = (v3.baseline if name == "v3-baseline" else v3.raw), v3.norm.names
    out = {
        v: np.full(
            (ref.sizes["time"], 1, ref.sizes["latitude"], ref.sizes["longitude"]),
            np.nan,
            np.float32,
        )
        for v in VARS
    }
    for i, t in enumerate(ref.time.values):
        if not generated[i]:
            continue
        for v in have:
            f = get(v, t).sel(
                latitude=ref.latitude, longitude=ref.longitude, method="nearest"
            )
            out[v][i, 0] = f.values
    ds = xr.Dataset(
        {v: (("time", "ensemble", "latitude", "longitude"), a) for v, a in out.items()},
        coords={
            "time": ref.time,
            "ensemble": [0.0],
            "latitude": ref.latitude,
            "longitude": ref.longitude,
        },
        attrs={"description": f"deterministic baseline {name}"},
    )
    path = f"{root}/{name}/{storm}.zarr"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.to_zarr(path, mode="w")
    print(f"wrote {path} ({int(generated.sum())} frames)")
