"""Do V3 and the vortex shift complement each other? Coarse-baseline error on real storm windows.

For each daily clip (00Z..00Z, 9 coarse frames) around a known storm, builds the interior frames I(tau)
from the two coarse endpoints with: linear, vortex_shift, V3, and two prototype hybrids
  v3+dvs : V3 + (vortex_shift - linear)        (adds the translation correction on top of V3)
  v3|vs  : V3 outside the storm, vortex_shift inside a cosine disc (1.5 x taper) at the interpolated centre
and compares them with the REAL coarse interior frames (what r = x_c - I must carry). CPU only.

Usage: baseline_compare_tc.py OUT_CSV STORM [STORM ...]   (STORM in S2022 T789 T795 T829)
"""
import ast
import datetime
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from fme.downscaling import vortex_shift as vs  # noqa: E402
from fme.downscaling.v3_baseline import load_v3_baseline, v3_coarse_interior  # noqa: E402

torch.set_num_threads(8)
V3_PATH = "/data/pmd/v3_baseline/v3_baseline_9yr.pt"
COARSE = "/data/pmd/local_data/2026-07-14-X-SHiELD-AMIP-FME-3h-100km.zarr"
NAMES = ["PRMSL", "eastward_wind_at_ten_meters", "northward_wind_at_ten_meters"]
MIN_DEPTH_HPA, TAPER, SEARCH = 5.0, 6.0, 10.0  # the vortex-shift run's settings
BOX = {"S2022": (3, 106), "T789": (0, 111), "T795": (0, 97), "T829": (0, 96)}  # lat0, lon0 of the 44x72 patch
WIN = 5  # coarse cells (deg) half-width of the storm windows

out_csv, *storms = sys.argv[1:]
src = open(REPO / "scripts/tropical_cyclones/two_block_family_analysis.py").read()
TRACKS = {n: ast.literal_eval(re.search(n + r" = (\[.*?\n\])", src, re.S).group(1)) for n in ("T789", "T795", "T829")}
TRACKS["S2022"] = [tuple(x) for x in json.load(open(Path(__file__).parent / "tracks_extra.json"))["S2022"]]

ds = xr.open_zarr(COARSE)
lat_all, lon_all = ds.latitude.values, ds.longitude.values
model, norm = load_v3_baseline(V3_PATH, "cpu")
slp_scale = 100.0 if float(ds["PRMSL"].isel(time=0, latitude=90, longitude=180)) > 2000 else 1.0  # Pa -> hPa
print("PRMSL stored in", "Pa" if slp_scale == 100.0 else "hPa", flush=True)


def clip_arrays(t0, ilat, ilon):
    """(3, 9, h, w) real coarse clip starting at t0, store units."""
    times = [t0 + datetime.timedelta(hours=3 * k) for k in range(9)]
    sub = ds[NAMES].sel(time=times).isel(latitude=ilat, longitude=ilon)
    return np.stack([sub[n].values.astype(np.float32) for n in NAMES])


def v3_interior(t0, ilat, ilon):
    """V3's interior frames (3, 7, h, w) for PRMSL/u/v from the clip's two coarse endpoints (needs all V3 channels)."""
    ends = {}
    for name in norm.names:
        sel = ds[name].sel(time=[t0, t0 + datetime.timedelta(days=1)]).isel(latitude=ilat, longitude=ilon)
        x = torch.from_numpy(sel.values.astype(np.float32))[None]  # (1, 2, h, w), physical
        scale = norm.log_scales.get(name)
        ends[name] = torch.log1p(x.clamp(min=0.0) * scale) if scale else x
    interior = v3_coarse_interior(model, norm, ends, n_steps=9)
    return np.stack([interior[n][0].numpy() for n in NAMES])  # PRMSL/u/v are not log channels -> physical


def hybrids(x, lin, vsh, v3):
    """lin/vsh/v3: (3, 7, h, w) interior arrays. Returns (v3+dvs, v3|vs)."""
    h1 = v3 + (vsh - lin)
    clip = torch.from_numpy(x)
    c0 = vs._find_center(clip[0, 0] / slp_scale, MIN_DEPTH_HPA, None, SEARCH, 3, (3.0, 6.0))
    c1 = None if c0 is None else vs._find_center(clip[0, -1] / slp_scale, MIN_DEPTH_HPA, c0, SEARCH, 3, (3.0, 6.0))
    if c0 is None or c1 is None:
        return h1, v3.copy()
    h2 = v3.copy()
    for k in range(7):
        tau = (k + 1) / 8
        c = ((1 - tau) * c0[0] + tau * c1[0], (1 - tau) * c0[1] + tau * c1[1])
        dist = vs._radius_grid(v3.shape[-2:], c, torch.device("cpu")).numpy() / (1.5 * TAPER)
        w = np.where(dist < 1, 0.5 * (1 + np.cos(np.pi * dist)), 0.0)
        h2[:, k] = v3[:, k] + w * (vsh[:, k] - v3[:, k])
    return h1, h2


def centre(slp, mask_rows, mask_cols):
    """Sub-cell (row, col) of the min of slp (h, w) inside the window rows/cols (index arrays)."""
    sub = torch.from_numpy(slp[np.ix_(mask_rows, mask_cols)])
    j, i = np.unravel_index(int(sub.argmin()), sub.shape)
    return (mask_rows[0] + vs._refine_1d(sub[:, i], j), mask_cols[0] + vs._refine_1d(sub[j, :], i))


rows = []
for storm in storms:
    lat0, lon0 = BOX[storm]
    ilat = np.where((lat_all >= lat0) & (lat_all <= lat0 + 44))[0]
    ilon = np.where((lon_all >= lon0) & (lon_all <= lon0 + 72))[0]
    track = TRACKS[storm]
    days = sorted({t[0][:10] for t in track})
    # clips fully inside the track's days, for the intense part (S2022: peak days; others: middle)
    sel = {"S2022": days[3:8], "T789": days[2:7], "T795": days[3:8], "T829": days[2:7]}[storm]
    for d in sel:
        t0 = datetime.datetime.fromisoformat(d)
        t0 = type(ds.time.values[0])(t0.year, t0.month, t0.day, 0, 0, 0)
        tic = time.time()
        x = clip_arrays(t0, ilat, ilon)  # (3, 9, h, w)
        truth = x[:, 1:-1]
        lin = (1 - np.linspace(0, 1, 9)[1:-1])[None, :, None, None] * x[:, :1] + np.linspace(0, 1, 9)[1:-1][None, :, None, None] * x[:, -1:]
        vsh = vs.vortex_shift_interp(torch.from_numpy(x)[None], 0, MIN_DEPTH_HPA * slp_scale, TAPER, SEARCH)[0, :, 1:-1].numpy()
        v3 = v3_interior(t0, ilat, ilon)
        h1, h2 = hybrids(x, lin, vsh, v3)
        cands = {"linear": lin, "vortex_shift": vsh, "v3": v3, "v3+dvs": h1, "v3|vs": h2}
        for k in range(7):
            t = t0 + datetime.timedelta(hours=3 * (k + 1))
            tk = t.strftime("%Y-%m-%d %H:%M:%S")
            pts = [p for p in track if p[0] == tk]
            if not pts:
                continue
            la, lo = pts[0][1], pts[0][2]
            r_c = int(np.argmin(np.abs(lat_all[ilat] - la))); c_c = int(np.argmin(np.abs(lon_all[ilon] - lo)))
            rr = np.arange(max(r_c - WIN, 0), min(r_c + WIN + 1, len(ilat)))
            cc = np.arange(max(c_c - WIN, 0), min(c_c + WIN + 1, len(ilon)))
            slp_t = truth[0, k] / slp_scale
            tc = centre(slp_t, rr, cc)
            tmin = float(slp_t[np.ix_(rr, cc)].min())
            ws_t = np.hypot(truth[1, k], truth[2, k])
            box_r = np.arange(max(int(round(tc[0])) - 6, 0), min(int(round(tc[0])) + 7, len(ilat)))
            box_c = np.arange(max(int(round(tc[1])) - 6, 0), min(int(round(tc[1])) + 7, len(ilon)))
            for name, arr in cands.items():
                slp = arr[0, k] / slp_scale
                ws = np.hypot(arr[1, k], arr[2, k])
                cm = centre(slp, rr, cc)
                dlat, dlon = (cm[0] - tc[0]), (cm[1] - tc[1])
                km = 111.0 * math.hypot(dlat, dlon * math.cos(math.radians(la)))
                rows.append(dict(
                    storm=storm, time=tk, cand=name, truth_slp=tmin, slp_err=float(slp[np.ix_(rr, cc)].min()) - tmin,
                    centre_km=km,
                    patch_rmse_slp=float(np.sqrt(((slp - slp_t) ** 2).mean())),
                    patch_rmse_ws=float(np.sqrt(((ws - ws_t) ** 2).mean())),
                    box_rmse_slp=float(np.sqrt(((slp - slp_t)[np.ix_(box_r, box_c)] ** 2).mean())),
                    box_rmse_ws=float(np.sqrt(((ws - ws_t)[np.ix_(box_r, box_c)] ** 2).mean())),
                ))
        print(f"{storm} {d}: {time.time() - tic:.0f}s", flush=True)

import pandas as pd  # noqa: E402

df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)
df["abs_slp_err"] = df.slp_err.abs()
cols = ["abs_slp_err", "centre_km", "box_rmse_slp", "box_rmse_ws", "patch_rmse_slp", "patch_rmse_ws"]
pd.set_option("display.width", 200)
order = ["linear", "vortex_shift", "v3", "v3+dvs", "v3|vs"]
for label, sub in (("ALL interior frames", df), ("INTENSE (truth min SLP < 990 hPa)", df[df.truth_slp < 990])):
    t = sub.groupby("cand")[cols].mean().loc[order]
    t["n_frames"] = sub.groupby("cand").size()
    print(f"\n===== {label} =====\n{t.round(3).to_string()}")
    print("\n% change vs plain V3 (negative = better than V3):")
    print((100 * (t[cols] - t.loc["v3", cols]) / t.loc["v3", cols]).loc[order].round(1).to_string())
