"""Tropical-cyclone verification of ensemble inference zarrs against 25km truth.

Reads <root>/<model>/<storm>.zarr (storm-centred regional runs from run_tc_eval.sh).
Storm track points are the known tracks hard-coded in
scripts/tropical_cyclones/two_block_family_analysis.py (T789 / T795 / T829).

Per track point (3-hourly) and per ensemble member, in a 5x5 degree window around the
known position, vs the 25km truth:
  slp_err      model min SLP - truth min SLP (hPa; signed bias + abs error)
  vmax_err     model max 10 m wind speed - truth (m/s)
  center_km    great-circle distance between model and truth min-SLP locations
  + the radial-structure scorecard of scripts/tropical_cyclones/tc_radial_metrics.py
    (radius of max wind, wind radii, radial wind MAE, wind-pressure imbalance, ...),
    computed about the truth's min-SLP centre for both, on frames where the truth
    storm has >= hurricane-ish winds (vmax >= 17.5 m/s = 34 kt).
Errors are averaged over members (so spread is penalised, not hidden by averaging).
"""
import argparse
import ast
import os
import re
import sys

import cftime
import numpy as np
import pandas as pd
import xarray as xr

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(REPO, "scripts/tropical_cyclones"))
import tc_radial_metrics as trm  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
HALF = 2.5  # deg half-width of the min-SLP / max-wind window
MIN_VMAX = 17.5  # m/s; frames with a weaker truth storm skip the radial scorecard
EDGES = trm.radial_bin_edges(dr_km=25.0, r_max_km=600.0)

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--models", nargs="+", required=True, help="first = model under test")
p.add_argument("--truth", required=True)
p.add_argument("--storms", nargs="+", default=["T789", "T795", "T829"])
p.add_argument("--tracks-json", help="{storm: [[time, lat, lon], ...]} tracks that add to / override the known ones")
args = p.parse_args()

src = open(os.path.join(REPO, "scripts/tropical_cyclones/two_block_family_analysis.py")).read()
import json  # noqa: E402

extra = json.load(open(args.tracks_json)) if args.tracks_json else {}
TRACKS = {
    n: [tuple(x) for x in extra[n]]
    if n in extra
    else ast.literal_eval(re.search(n + r" = (\[.*?\n\])", src, re.S).group(1))
    for n in args.storms
}
truth_ds = xr.open_zarr(args.truth)


def to_hpa(a):
    a = np.asarray(a, dtype="float64")
    return a / 100.0 if np.nanmean(a) > 2000 else a


def window_mask(lat, lon, lat0, lon0):
    return ((np.abs(lat - lat0) <= HALF)[:, None]) & ((np.abs(lon - lon0) <= HALF)[None, :])


def min_loc(slp_hpa, mask, lat, lon):
    v = np.where(mask, slp_hpa, np.inf)
    j, i = np.unravel_index(np.argmin(v), v.shape)
    return float(v[j, i]), float(lat[j]), float(lon[i])


rows = []
for storm, track in TRACKS.items():
    for model in args.models:
        path = os.path.join(args.root, model, f"{storm}.zarr")
        if not os.path.exists(path):
            print(f"missing {path}", file=sys.stderr)
            continue
        pred = xr.open_zarr(path)
        lat, lon = pred.latitude.values, pred.longitude.values
        truth = truth_ds.sel(latitude=pred.latitude, longitude=pred.longitude, method="nearest")
        for t_str, la, lo in track:
            t = cftime.DatetimeJulian(*(int(g) for g in re.match(r"(\d+)-(\d+)-(\d+)[ T](\d+):(\d+):(\d+)", t_str).groups()))
            if t not in pred.time.values:
                continue
            tt = truth.sel(time=t)
            slp_t = to_hpa(tt["PRMSL"].values)
            ws_t = np.hypot(tt["eastward_wind_at_ten_meters"].values, tt["northward_wind_at_ten_meters"].values)
            mask = window_mask(lat, lon, la, lo)
            ts, tlat, tlon = min_loc(slp_t, mask, lat, lon)
            tv = float(np.nanmax(np.where(mask, ws_t, -np.inf)))
            pm = pred.sel(time=t).load()
            if not np.isfinite(pm["PRMSL"].values).all():
                continue  # frame not written
            tgt_prof = None
            if tv >= MIN_VMAX:
                tgt_prof = trm.StormProfiles(
                    r_km=trm.bin_centers(EDGES),
                    wind_mean=trm.compute_radial_profile(ws_t, tlat, tlon, lat, lon, EDGES).mean,
                    pressure_mean=trm.compute_radial_profile(slp_t * 100, tlat, tlon, lat, lon, EDGES).mean,
                    wind_var=trm.compute_radial_profile(ws_t, tlat, tlon, lat, lon, EDGES).var,
                )
            is00 = t_str.endswith("00:00:00")
            for m in range(pm.sizes["ensemble"]):
                e = pm.isel(ensemble=m)
                slp_m = to_hpa(e["PRMSL"].values)
                ws_m = np.hypot(e["eastward_wind_at_ten_meters"].values, e["northward_wind_at_ten_meters"].values)
                ms, mlat, mlon = min_loc(slp_m, mask, lat, lon)
                row = dict(
                    storm=storm, model=model, time=t_str, member=m, is00z=is00, truth_slp=ts, truth_vmax=tv,
                    slp_err=ms - ts, vmax_err=float(np.nanmax(np.where(mask, ws_m, -np.inf))) - tv,
                    center_km=float(trm.great_circle_distance_km(tlat, tlon, np.array([mlat]), np.array([mlon]))[0, 0]),
                )
                if tgt_prof is not None:
                    sim = trm.StormProfiles(
                        r_km=trm.bin_centers(EDGES),
                        wind_mean=trm.compute_radial_profile(ws_m, tlat, tlon, lat, lon, EDGES).mean,
                        pressure_mean=trm.compute_radial_profile(slp_m * 100, tlat, tlon, lat, lon, EDGES).mean,
                        wind_var=trm.compute_radial_profile(ws_m, tlat, tlon, lat, lon, EDGES).var,
                    )
                    try:
                        sc = trm.scorecard(sim, tgt_prof)
                        row.update({k: v for k, v in sc.items() if k in (
                            "dR_max_km", "MAE_v_radial", "dR_34kt_km", "dR_50kt_km", "dR_64kt_km", "dImbalance")})
                    except Exception as exc:  # noqa: BLE001
                        print(f"scorecard failed {storm} {model} {t_str}: {exc}", file=sys.stderr)
                rows.append(row)

df = pd.DataFrame(rows)
if df.empty:
    sys.exit(f"No scoreable zarrs under {args.root}")
df.to_csv(os.path.join(args.root, "tc_scores_per_member_frame.csv"), index=False)
df["abs_slp_err"] = df.slp_err.abs()
df["abs_vmax_err"] = df.vmax_err.abs()
for c in ("dR_max_km", "dR_34kt_km", "dR_50kt_km", "dR_64kt_km"):
    if c in df:
        df["abs_" + c] = df[c].abs()
cols = ["abs_slp_err", "slp_err", "abs_vmax_err", "vmax_err", "center_km", "dR_max_km", "abs_dR_34kt_km",
        "abs_dR_50kt_km", "abs_dR_64kt_km", "MAE_v_radial", "dImbalance"]
cols = [c for c in cols if c in df]
rename = {"abs_slp_err": "SLP_MAE_hPa", "slp_err": "SLP_bias_hPa", "abs_vmax_err": "Vmax_MAE_ms",
          "vmax_err": "Vmax_bias_ms", "center_km": "center_err_km"}

fd = df.groupby(["storm", "model", "time"])[cols].mean().reset_index()  # mean over members first
for label, sub in (("ALL FRAMES", fd), ("INTENSE (truth min SLP < 990 hPa)",
                   fd.merge(df[["storm", "time", "truth_slp"]].drop_duplicates(), on=["storm", "time"]).query("truth_slp < 990"))):
    out = sub.groupby(["storm", "model"])[cols].mean().rename(columns=rename)
    out["n_frames"] = sub.groupby(["storm", "model"]).size()
    pooled = sub.groupby("model")[cols].mean().rename(columns=rename)
    pooled["n_frames"] = sub.groupby("model").size()
    print(f"\n===== {label}: per storm =====\n{out.round(3).to_string()}")
    print(f"\n===== {label}: pooled over storms =====\n{pooled.round(3).to_string()}")
    first = args.models[0]
    for other in args.models[1:]:
        if first in pooled.index and other in pooled.index:
            d = 100 * (pooled.loc[first] - pooled.loc[other]) / pooled.loc[other].abs()
            print(f"\n% change {first} vs {other} (negative = {first} better; bias columns omitted, compare their signed values above):")
            print(d.drop(["n_frames", "SLP_bias_hPa", "Vmax_bias_ms"], errors="ignore").round(1).to_string())
            pa = sub.pivot_table(index=["storm", "time"], columns="model", values="abs_slp_err")
            diff = (pa[first] - pa[other]).dropna()
            print(f"\nPaired per-frame |SLP err| {first} - {other}: mean {diff.mean():+.2f} hPa "
                  f"(naive s.e. {diff.std(ddof=1) / np.sqrt(len(diff)):.2f}; frames are autocorrelated, so "
                  f"treat as optimistic), {first} better in {(diff < 0).mean() * 100:.0f}% of {len(diff)} frames")
