"""Min-SLP lifecycle plot: 25km truth vs each model's ensemble (members thin, mean thick).

Min PRMSL in a 5x5 degree window around the known track position, per 3-hourly frame.
Usage: plot_tc_minslp.py ROOT STORM TRACKS_JSON TRUTH_ZARR OUT_PNG MODEL [MODEL ...]
"""
import json
import re
import sys

import cftime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

root, storm, tracks_json, truth_path, out_png, *models = sys.argv[1:]
HALF = 2.5
COLORS = {"two-block": "#2ca02c", "vortex": "#d62728"}
track = json.load(open(tracks_json))[storm]
truth = xr.open_zarr(truth_path)["PRMSL"]
preds = {m: xr.open_zarr(f"{root}/{m}/{storm}.zarr")["PRMSL"] for m in models}
ref = preds[models[0]]
truth = truth.sel(latitude=ref.latitude, longitude=ref.longitude, method="nearest")


def hpa(a):
    return a / 100.0 if np.nanmean(a) > 2000 else a


def win_min(f, la, lo):
    sub = f.sel(latitude=slice(la - HALF, la + HALF), longitude=slice(lo - HALF, lo + HALF))
    return np.nanmin(hpa(sub.values), axis=(-2, -1))


times, tr, mem = [], [], {m: [] for m in models}
for t_str, la, lo in track:
    t = cftime.DatetimeJulian(*(int(g) for g in re.match(r"(\d+)-(\d+)-(\d+)[ T](\d+):(\d+):(\d+)", t_str).groups()))
    if t not in ref.time.values:
        continue
    per_model = {m: win_min(preds[m].sel(time=t), la, lo) for m in models}  # (ensemble,)
    if not all(np.isfinite(v).all() for v in per_model.values()):
        continue  # frame inside the store's time axis but never generated
    times.append(t_str)
    tr.append(float(win_min(truth.sel(time=t), la, lo)))
    for m in models:
        mem[m].append(per_model[m])

x = np.arange(len(times))
is00 = np.array([s.endswith("00:00:00") for s in times])
fig, ax = plt.subplots(figsize=(11, 5))
for i in np.where(is00)[0]:
    ax.axvline(i, color="0.88", lw=0.8, zorder=0)
for m in models:
    a = np.array(mem[m])  # (T, E)
    for e in range(a.shape[1]):
        ax.plot(x, a[:, e], color=COLORS.get(m), alpha=0.25, lw=0.8, zorder=2)
    mae = np.mean(np.abs(a - np.array(tr)[:, None]))
    ax.plot(x, a.mean(1), color=COLORS.get(m), lw=2, marker="o", ms=3, zorder=4,
            label=f"{m} (ens. mean; per-member MAE {mae:.1f} hPa)")
ax.plot(x, tr, color="black", lw=3, zorder=6, label="25 km truth")
ax.set_xticks(x[::8])
ax.set_xticklabels([times[i][5:13] for i in x[::8]], rotation=45, ha="right", fontsize=8)
ax.set_ylabel(f"min PRMSL in {2 * HALF:.0f}°×{2 * HALF:.0f}° window (hPa)")
ax.set_title(f"{storm}: tropical-cyclone min-pressure lifecycle (thin lines = ensemble members; grey = 00Z)")
ax.invert_yaxis()
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(out_png, dpi=140)
print("wrote", out_png)
