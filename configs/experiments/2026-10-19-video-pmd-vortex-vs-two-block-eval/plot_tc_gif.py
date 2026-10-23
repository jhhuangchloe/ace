"""Storm-following GIF: top row 25km truth | two-block | vortex | linear interp. of the 00Z 100 km
endpoints, bottom row the min-SLP / max-wind time series (moving marker) | each of the three minus truth
(ensemble member 0).

Usage: plot_tc_gif.py ROOT STORM TRACKS_JSON TRUTH_ZARR OUT_GIF FIELD [MEMBER]
  FIELD: prmsl (hPa) or wind (10 m wind speed, m/s)
Each frame is a 12x12 deg window centred on the known track position; shared colour scales.
"""
import io
import json
import re
import sys

import cftime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

import xarray as xr
import yaml
from linear_baseline import LinearBaseline
from v3_curve import V3Curves
from PIL import Image

root, storm, tracks_json, truth_path, out_gif, field = sys.argv[1:7]
member = int(sys.argv[7]) if len(sys.argv) > 7 else 0
HALF = 6.0
MODELS = ["two-block", "vortex"]
LINEAR = "linear interp. (bicubic)"
NEAREST = "linear interp. (nearest)"
V3_BASE = "V3 baseline"
V3_RAW = "V3 raw 25 km"
COLORS = {"25 km truth": "black", "two-block": "#2ca02c", "vortex": "#d62728", LINEAR: "#1f77b4", V3_BASE: "#9467bd"}
SERIES_COLORS = {**COLORS, NEAREST: "#9ecae1", V3_RAW: "#8c564b"}
CFG = {
    "prmsl": dict(cmap="turbo_r", vmin=920, vmax=1010, dmax=30, label="PRMSL (hPa)", ts="min SLP in window (hPa)"),
    "wind": dict(cmap="viridis", vmin=0, vmax=70, dmax=25, label="10 m wind speed (m/s)", ts="max wind in window (m/s)"),
}[field]

track = json.load(open(tracks_json))[storm]
preds = {m: xr.open_zarr(f"{root}/{m}/{storm}.zarr") for m in MODELS}
ref = preds[MODELS[0]]
truth_full = xr.open_zarr(truth_path)
truth = truth_full.sel(latitude=ref.latitude, longitude=ref.longitude, method="nearest")
bases = {u: LinearBaseline(truth_path.replace("25km", "100km"), truth_full, u) for u in ("nearest", "bicubic")}
region = yaml.safe_load(open(f"{root}/{MODELS[0]}/{storm}.yaml"))["data"]
v3 = V3Curves(
    truth_path.replace("25km", "100km"), truth_full,
    os.environ.get("V3_WEIGHTS", "/data/pmd/v3_baseline/v3_baseline_9yr.pt"),
    region["lat_extent"], region["lon_extent"],
)


def get(ds, t, la, lo):
    d = ds.sel(time=t)
    if "ensemble" in d.dims:
        d = d.isel(ensemble=member)
    d = d.sel(latitude=slice(la - HALF, la + HALF), longitude=slice(lo - HALF, lo + HALF)).load()
    if field == "prmsl":
        v = d["PRMSL"].values
        v = v / 100.0 if np.nanmean(v) > 2000 else v
    else:
        v = np.hypot(d["eastward_wind_at_ten_meters"].values, d["northward_wind_at_ten_meters"].values)
    return v, d.latitude.values, d.longitude.values


def get_linear(t, la, lo, upsample="bicubic"):
    base = bases[upsample]
    names = ["PRMSL"] if field == "prmsl" else ["eastward_wind_at_ten_meters", "northward_wind_at_ten_meters"]
    fields = [base.field(n, t).sel(latitude=slice(la - HALF, la + HALF), longitude=slice(lo - HALF, lo + HALF)) for n in names]
    if field == "prmsl":
        v = fields[0].values
        v = v / 100.0 if np.nanmean(v) > 2000 else v
    else:
        v = np.hypot(fields[0].values, fields[1].values)
    return v, fields[0].latitude.values, fields[0].longitude.values


def get_v3(t, la, lo, which):
    names = ["PRMSL"] if field == "prmsl" else ["eastward_wind_at_ten_meters", "northward_wind_at_ten_meters"]
    pick = v3.baseline if which == "baseline" else v3.raw
    fields = [pick(n, t).sel(latitude=slice(la - HALF, la + HALF), longitude=slice(lo - HALF, lo + HALF)) for n in names]
    if field == "prmsl":
        v = fields[0].values
        v = v / 100.0 if np.nanmean(v) > 2000 else v
    else:
        v = np.hypot(fields[0].values, fields[1].values)
    return v, fields[0].latitude.values, fields[0].longitude.values


stat = np.nanmin if field == "prmsl" else np.nanmax
data = []  # (t_str, {name: (v, lat, lon)})
for t_str, la, lo in track:
    t = cftime.DatetimeJulian(*(int(g) for g in re.match(r"(\d+)-(\d+)-(\d+)[ T](\d+):(\d+):(\d+)", t_str).groups()))
    if t not in ref.time.values:
        continue
    fr = {"25 km truth": get(truth, t, la, lo)}
    fr.update({m: get(preds[m], t, la, lo) for m in MODELS})
    fr[LINEAR] = get_linear(t, la, lo)
    fr[NEAREST] = get_linear(t, la, lo, "nearest")
    fr[V3_BASE] = get_v3(t, la, lo, "baseline")
    fr[V3_RAW] = get_v3(t, la, lo, "raw")
    if not all(np.isfinite(fr[m][0]).all() for m in MODELS):
        continue  # frame never generated
    data.append((t_str, fr))
series = {n: np.array([stat(fr[n][0]) for _, fr in data]) for n in SERIES_COLORS}
x = np.arange(len(data))

frames = []
for i, (t_str, fr) in enumerate(data):
    fig, axes = plt.subplots(2, 5, figsize=(25, 9), constrained_layout=True)
    for ax, name in zip(axes[0], COLORS):
        v, lat, lon = fr[name]
        im = ax.imshow(v, origin="lower", cmap=CFG["cmap"], vmin=CFG["vmin"], vmax=CFG["vmax"],
                       extent=[lon[0], lon[-1], lat[0], lat[-1]], aspect="equal")
        ax.set_title(f"{name}  ({'min' if field == 'prmsl' else 'max'} {stat(v):.0f})", fontsize=13)
    fig.colorbar(im, ax=axes[0], shrink=0.85, label=CFG["label"])
    ax = axes[1][0]
    for n, c in SERIES_COLORS.items():
        ax.plot(x, series[n], color=c, lw=3 if n.startswith("25") else 1.8, ls="--" if n.startswith("linear") else (":" if n == V3_RAW else "-"), label=n)
        ax.plot(x[i], series[n][i], "o", color=c, ms=8)
    ax.axvline(i, color="0.6", lw=0.8)
    ax.set_ylabel(CFG["ts"])
    ax.set_xticks(x[::8])
    ax.set_xticklabels([d[0][5:13] for d in data][::8], rotation=45, ha="right", fontsize=8)
    if field == "prmsl":
        ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for ax, name in zip(axes[1][1:], MODELS + [LINEAR, V3_BASE]):
        diff = fr[name][0] - fr["25 km truth"][0]
        im2 = ax.imshow(diff, origin="lower", cmap="RdBu_r" if field == "wind" else "RdBu",
                        vmin=-CFG["dmax"], vmax=CFG["dmax"], extent=[lon[0], lon[-1], lat[0], lat[-1]], aspect="equal")
        ax.set_title(f"{name} − truth  (MAE {np.abs(diff).mean():.1f}, RMSE {np.sqrt((diff ** 2).mean()):.1f})", fontsize=12)
    fig.colorbar(im2, ax=axes[1][1:], shrink=0.85, label="model − truth" + (" (red = model stronger)" if field == "wind" else " (red = model lower pressure = stronger)"))
    fig.suptitle(f"{storm}  {t_str[:16]} UTC  (ensemble member {member})", fontsize=14)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=60)
    plt.close(fig)
    frames.append(Image.open(io.BytesIO(buf.getvalue())).convert("P", palette=Image.ADAPTIVE))

frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=300, loop=0)
print(f"wrote {out_gif} ({len(frames)} frames)")
