"""Score ensemble inference zarrs of several models against 25km truth.

Reads <root>/<model>/<window>.zarr (written by run_eval.sh) and prints/saves, per
model and channel: area-weighted RMSE and MAE of the ensemble mean, fair CRPS, and
spread/skill ratio -- on generated (interior) frames only. Also prints the
relative change of the first model vs each other model (negative = first is better).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--models", nargs="+", required=True, help="first = model under test")
p.add_argument("--truth", required=True, help="25km truth zarr")
args = p.parse_args()
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

truth_ds = xr.open_zarr(args.truth)


def fair_crps(ens, y):
    """ens: (T,E,H,W), y: (T,H,W) -> (T,H,W); Ferro et al. 2008 via order statistics."""
    m = ens.shape[1]
    term1 = np.abs(ens - y[:, None]).mean(1)
    s = np.sort(ens, axis=1)
    w = (2 * np.arange(1, m + 1) - m - 1).reshape(1, m, 1, 1)
    return term1 - (w * s).sum(1) / (m * (m - 1))


rows = []
for model in args.models:
    for path in sorted(glob.glob(os.path.join(args.root, model, "*.zarr"))):
        window = os.path.basename(path)[: -len(".zarr")]
        pred = xr.open_zarr(path)
        t0, t1 = pred.time.values[0], pred.time.values[-1]
        keep = pred["frame_source"].values == 1 if "frame_source" in pred else slice(None)
        truth = (
            truth_ds.sel(time=slice(t0, t1))
            .sel(latitude=pred.latitude, longitude=pred.longitude, method="nearest")
        )
        w = np.cos(np.deg2rad(pred.latitude.values))[None, :, None]
        w = w / w.mean()
        for ch in pred.data_vars:
            if ch == "frame_source" or ch not in truth:
                continue
            ens = pred[ch].values[keep].astype(np.float64)  # (T,E,H,W)
            y = truth[ch].values[keep].astype(np.float64)
            ok = np.isfinite(ens).all(axis=(1, 2, 3)) & np.isfinite(y).all(axis=(1, 2))  # drop unwritten frames
            ens, y = ens[ok], y[ok]
            if len(y) == 0 or ens.shape[1] < 2:
                continue
            mean = ens.mean(1)
            mse = (w * (mean - y) ** 2).mean()
            spread2 = (w * ens.var(1, ddof=1)).mean()
            rows.append(
                dict(
                    model=model, window=window, channel=ch, n_frames=len(y),
                    rmse=np.sqrt(mse), mae=(w * np.abs(mean - y)).mean(),
                    crps=(w * fair_crps(ens, y)).mean(),
                    spread_skill=np.sqrt(spread2 / mse),
                )
            )

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit(f"No scoreable zarrs found under {args.root}")
df.to_csv(os.path.join(args.root, "scores_per_window.csv"), index=False)
metrics = ["rmse", "mae", "crps", "spread_skill"]
summary = df.groupby(["model", "channel"])[metrics].mean()
summary.to_csv(os.path.join(args.root, "scores_summary.csv"))
print("\n== Mean over windows (generated frames, area-weighted, physical units) ==")
print(summary.round(4).to_string())

first = args.models[0]
for other in args.models[1:]:
    a, b = summary.loc[first], summary.loc[other]
    rel = 100 * (a[["rmse", "mae", "crps"]] - b[["rmse", "mae", "crps"]]) / b[["rmse", "mae", "crps"]]
    print(f"\n== % change, {first} vs {other} (negative = {first} better) ==")
    print(rel.round(2).to_string())
    pw = df.pivot_table(index=["window", "channel"], columns="model", values="crps")
    wins = (pw[first] < pw[other]).groupby("channel").agg(["sum", "count"])
    print(f"\nWindows where {first} has lower CRPS (wins/total) per channel:")
    print(wins.to_string())
