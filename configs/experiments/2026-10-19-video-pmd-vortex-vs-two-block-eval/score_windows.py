"""Patch-wide MSE and fair CRPS per channel over many regional windows, with day-block bootstrap intervals.

Usage: score_windows.py ROOT TRUTH_ZARR --ens MODEL [MODEL ...] [--det MODEL ...] [--ref MODEL] [--windows W ...] [--out PREFIX]
  ROOT/<model>/<window>.zarr   dims (time, ensemble, latitude, longitude), all on the same grid/time axis per window
  --ens  ensemble models (table 1)   --det  single-member baselines (table 2; CRPS reduces to MAE)
  --ref  model the others are compared against (default: first --ens); windows default: those present for every model
Scores are cos(lat)-weighted means over the whole patch and over the frames finite for EVERY model of that window.
Channels: u10, v10 (m/s), PRMSL (hPa), precip in log1p(mm/day), T2m (K). MSE is of the ensemble mean.
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("truth")
p.add_argument("--ens", nargs="+", required=True)
p.add_argument("--det", nargs="*", default=[])
p.add_argument("--ref")
p.add_argument("--windows", nargs="*")
p.add_argument("--out", default=None)
a = p.parse_args()
models = a.ens + a.det
ref_model = a.ref or a.ens[0]
out_prefix = a.out or os.path.join(a.root, "large_eval")

CHANNELS = {
    "u10": lambda d: d["eastward_wind_at_ten_meters"].values,
    "v10": lambda d: d["northward_wind_at_ten_meters"].values,
    "PRMSL": lambda d: (lambda x: x / 100.0 if np.nanmean(x) > 2000 else x)(d["PRMSL"].values),
    "precip": lambda d: np.log1p(np.maximum(d["PRATEsfc"].values * 86400.0, 0.0)),
    "T2m": lambda d: d["air_temperature_at_two_meters"].values,
}


def fair_crps(ens, y):
    """ens (T, E, H, W), y (T, H, W) -> (T, H, W); fair CRPS (Ferro 2008); one member -> |error|."""
    m = ens.shape[1]
    if m == 1:
        return np.abs(ens[:, 0] - y)
    s = np.sort(ens, axis=1)
    w = (2 * np.arange(1, m + 1) - m - 1).reshape(1, m, 1, 1)
    return np.abs(ens - y[:, None]).mean(1) - (w * s).sum(1) / (m * (m - 1))


def exists(m, w):
    return os.path.isdir(f"{a.root}/{m}/{w}.zarr")


windows = a.windows or sorted(w for w in {f[:-5] for f in os.listdir(f"{a.root}/{models[0]}") if f.endswith(".zarr")} if all(exists(m, w) for m in models))
print("windows:", windows)
truth_all = xr.open_zarr(a.truth)
rows = []
for w in windows:
    ds = {m: xr.open_zarr(f"{a.root}/{m}/{w}.zarr") for m in models}
    ref = ds[ref_model]
    ok = np.ones(ref.sizes["time"], bool)
    for m in models:
        ok &= np.isfinite(ds[m]["PRMSL"].isel(latitude=slice(None, None, 20), longitude=slice(None, None, 20)).values).all(axis=(1, 2, 3))
    idx = np.where(ok)[0]
    print(f"{w}: {len(idx)}/{ref.sizes['time']} frames common to all models")
    if len(idx) == 0:
        continue
    times = ref.time.values[idx]
    tr = truth_all.sel(latitude=ref.latitude, longitude=ref.longitude, method="nearest").sel(time=times).load()
    wlat = np.cos(np.deg2rad(ref.latitude.values))[None, :, None]
    wlat = wlat / wlat.mean()
    y = {c: f(tr) for c, f in CHANNELS.items()}
    day = pd.to_datetime([str(t) for t in times]).floor("D")
    for m in models:
        pm = ds[m].isel(time=idx).load()
        for c, f in CHANNELS.items():
            x = np.stack([f(pm.isel(ensemble=e)) for e in range(pm.sizes["ensemble"])], axis=1)  # (T, E, H, W)
            mse = (wlat * (x.mean(1) - y[c]) ** 2).mean(axis=(1, 2))
            crps = (wlat * fair_crps(x, y[c])).mean(axis=(1, 2))
            rows += [dict(window=w, model=m, channel=c, time=str(t), day=d, mse=a_, crps=b_) for t, d, a_, b_ in zip(times, day, mse, crps)]
df = pd.DataFrame(rows)
df.to_csv(f"{out_prefix}_per_frame.csv", index=False)

# windows are equally weighted through equal frame counts per window; report the frame-mean
agg = df.groupby(["model", "channel"])[["mse", "crps"]].mean()
UNITS = {"u10": "u10 (m/s)", "v10": "v10 (m/s)", "PRMSL": "PRMSL (hPa)", "precip": "Precip (log1p mm/day)", "T2m": "T2m (K)"}


def table(ms):
    head = "| Channel | Metric | " + " | ".join(ms) + " |\n|---|---|" + "---|" * len(ms)
    lines = [head]
    for c in CHANNELS:
        for met, lab in (("mse", "MSE"), ("crps", "CRPS")):
            lines.append(f"| {UNITS[c]} | {lab} | " + " | ".join(f"{agg.loc[(m, c), met]:.3f}" for m in ms) + " |")
    return "\n".join(lines)


n_frames = df[(df.model == ref_model) & (df.channel == "u10")].shape[0]
txt = [f"windows: {len(windows)}  frames scored: {n_frames}  ({df.day.nunique()} distinct day-blocks)\n",
       "TABLE 1 (ensemble models)\n" + table(a.ens)]
if a.det:
    txt.append("TABLE 2 (single-member baselines)\n" + table(a.det))

# day-block bootstrap of (ref - other): blocks = (window, day); negative = ref better
rng = np.random.default_rng(0)
blk = df.assign(block=df.window + "|" + df.day.astype(str)).groupby(["block", "model", "channel"])[["mse", "crps"]].mean()
rows = []
for other in [m for m in models if m != ref_model]:
    for c in CHANNELS:
        for met in ("mse", "crps"):
            d = (blk.xs((ref_model, c), level=("model", "channel"))[met] - blk.xs((other, c), level=("model", "channel"))[met]).dropna().values
            bs = np.array([rng.choice(d, len(d)).mean() for _ in range(5000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(f"{ref_model} - {other:10s} {c:7s} {met.upper():4s} {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {'SIGNIFICANT' if lo > 0 or hi < 0 else 'within noise'}  (better on {int((d < 0).sum())}/{len(d)} blocks)")
txt.append(f"DIFFERENCES vs {ref_model} (negative = {ref_model} better), day-block bootstrap\n" + "\n".join(rows))
open(f"{out_prefix}_tables.txt", "w").write("\n\n".join(txt) + "\n")
print("\n\n".join(txt))
print(f"\nwrote {out_prefix}_tables.txt and {out_prefix}_per_frame.csv")
