"""MSE and fair CRPS around a tropical cyclone, from tc_run_eval.sh's inference zarrs.

(1) scalar: min SLP in the 5x5 deg window (one value per frame/member; truth = 25km min SLP)
(2) field: every grid cell in a 10x10 deg box centred on the known track position
    (PRMSL hPa; 10 m wind speed/u/v m/s; precip mm/day; T2m K); MSE of the ensemble mean + fair CRPS, area-weighted.
Reported on intense frames (truth min SLP < 990 hPa) and on all frames.

Usage: tc_mse_crps.py ROOT STORM TRACKS_JSON TRUTH_ZARR MODEL [MODEL ...]   (first = model under test)
"""
import json
import re
import sys

import cftime
import numpy as np
import pandas as pd
import xarray as xr

root, storm, tracks_json, truth_path, *models = sys.argv[1:]
HALF_MIN, HALF_BOX = 2.5, 5.0
track = json.load(open(tracks_json))[storm]
preds = {m: xr.open_zarr(f"{root}/{m}/{storm}.zarr") for m in models}
ref = preds[models[0]]
truth = xr.open_zarr(truth_path).sel(latitude=ref.latitude, longitude=ref.longitude, method="nearest")
lat, lon = ref.latitude.values, ref.longitude.values


def hpa(a):
    return a / 100.0 if np.nanmean(a) > 2000 else a


def fair_crps(ens, y):
    """ens (E, ...), y (...) -> CRPS per element (Ferro 2008, sorted-order-statistics identity)."""
    m = ens.shape[0]
    s = np.sort(ens, axis=0)
    w = (2 * np.arange(1, m + 1) - m - 1).reshape((m,) + (1,) * y.ndim)
    return np.abs(ens - y).mean(0) - (w * s).sum(0) / (m * (m - 1))


def box(f, la, lo, half):
    return f.sel(latitude=slice(la - half, la + half), longitude=slice(lo - half, lo + half))


rows = []
for t_str, la, lo in track:
    t = cftime.DatetimeJulian(*(int(g) for g in re.match(r"(\d+)-(\d+)-(\d+)[ T](\d+):(\d+):(\d+)", t_str).groups()))
    if t not in ref.time.values:
        continue
    pm = {m: preds[m].sel(time=t).load() for m in models}
    if not all(np.isfinite(p["PRMSL"].values).all() for p in pm.values()):
        continue
    tt = truth.sel(time=t).load()
    y_slp_min = float(np.nanmin(hpa(box(tt["PRMSL"], la, lo, HALF_MIN).values)))
    w = np.cos(np.deg2rad(box(tt["PRMSL"], la, lo, HALF_BOX).latitude.values))[:, None]
    w = w / w.mean()
    for m, p in pm.items():
        r = dict(model=m, time=t_str, truth_slp=y_slp_min)
        x_min = np.array([np.nanmin(hpa(box(p["PRMSL"].isel(ensemble=e), la, lo, HALF_MIN).values)) for e in range(p.sizes["ensemble"])])
        r["minSLP_MSE_ensmean"] = (x_min.mean() - y_slp_min) ** 2
        r["minSLP_MSE_member"] = ((x_min - y_slp_min) ** 2).mean()
        r["minSLP_CRPS"] = float(fair_crps(x_min[:, None], np.array([y_slp_min]))[0])
        for name, get in (
            ("PRMSL", lambda d: hpa(d["PRMSL"].values)),
            ("wind10m", lambda d: np.hypot(d["eastward_wind_at_ten_meters"].values, d["northward_wind_at_ten_meters"].values)),
            ("u10", lambda d: d["eastward_wind_at_ten_meters"].values),
            ("v10", lambda d: d["northward_wind_at_ten_meters"].values),
            ("precip_mmday", lambda d: d["PRATEsfc"].values * 86400.0),
            ("T2m", lambda d: d["air_temperature_at_two_meters"].values),
        ):
            ens = np.stack([get(box(p.isel(ensemble=e), la, lo, HALF_BOX)) for e in range(p.sizes["ensemble"])])
            y = get(box(tt, la, lo, HALF_BOX))
            r[f"{name}_MSE"] = float((w * (ens.mean(0) - y) ** 2).mean())
            r[f"{name}_CRPS"] = float((w * fair_crps(ens, y)).mean())
        rows.append(r)

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
cols = ["minSLP_MSE_ensmean", "minSLP_MSE_member", "minSLP_CRPS", "PRMSL_MSE", "PRMSL_CRPS", "wind10m_MSE", "wind10m_CRPS",
        "u10_MSE", "u10_CRPS", "v10_MSE", "v10_CRPS", "precip_mmday_MSE", "precip_mmday_CRPS", "T2m_MSE", "T2m_CRPS"]
for label, sub in (("INTENSE (truth min SLP < 990 hPa)", df[df.truth_slp < 990]), ("ALL FRAMES", df)):
    tab = sub.groupby("model")[cols].mean().loc[models]
    tab["n_frames"] = sub.groupby("model").size()
    print(f"\n===== {label} =====\n{tab.round(3).T.to_string()}")
    if len(models) > 1:
        d = 100 * (tab.loc[models[0], cols] - tab.loc[models[1], cols]) / tab.loc[models[1], cols]
        print(f"\n% change {models[0]} vs {models[1]} (negative = {models[0]} better):\n{d.round(1).to_string()}")
df.to_csv(f"{root}/tc_mse_crps_{storm}.csv", index=False)
