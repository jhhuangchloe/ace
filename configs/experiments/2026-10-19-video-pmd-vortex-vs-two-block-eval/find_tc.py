"""Find the deepest tropical cyclone in a period of the 25km truth and track it.

Seeds at the lowest PRMSL in |lat|<=LATMAX (scanned every 12 h), then follows the
SLP minimum forward/backward in 3-hourly steps (within +-6 deg, <=4 deg/step) while
SLP < FILL_HPA. Writes {name: [[time, lat, lon], ...]} for tc_score.py --tracks-json
and prints the track extent plus a suggested 44x72 box.

Usage: find_tc.py TRUTH_ZARR START STOP NAME OUT_JSON
"""
import json
import sys

import numpy as np
import xarray as xr

truth_path, start, stop, name, out_json = sys.argv[1:]
LATMAX, FILL_HPA, WIN, MAX_STEP = 35, 1000.0, 6.0, 4.0
da = xr.open_zarr(truth_path)["PRMSL"].sel(time=slice(start, stop))
hpa = lambda a: a / 100.0 if np.nanmean(a) > 2000 else a  # noqa: E731
trop = da.sel(latitude=slice(-LATMAX, LATMAX))
best = None
for i in range(0, da.sizes["time"], 4):
    f = trop.isel(time=i).load()
    v = hpa(f.values)
    j, k = np.unravel_index(np.nanargmin(v), v.shape)
    if best is None or v[j, k] < best[3]:
        best = (i, float(f.latitude[j]), float(f.longitude[k]), float(v[j, k]))
si, lat0, lon0, p0 = best
print(f"seed: {str(da.time.values[si])} lat {lat0:.1f} lon {lon0:.1f} SLP {p0:.1f} hPa")


def step(i, lat, lon):
    f = da.isel(time=i).sel(
        latitude=slice(max(lat - WIN, -88), min(lat + WIN, 88)),
        longitude=slice(lon - WIN, lon + WIN),
    ).load()
    v = hpa(f.values)
    j, k = np.unravel_index(np.nanargmin(v), v.shape)
    return float(f.latitude[j]), float(f.longitude[k]), float(v[j, k])


def walk(d):
    out, lat, lon = [], lat0, lon0
    i = si
    while 0 <= i + d < da.sizes["time"]:
        i += d
        nlat, nlon, p = step(i, lat, lon)
        if abs(nlat - lat) > MAX_STEP or abs(nlon - lon) > MAX_STEP or p > FILL_HPA:
            break
        lat, lon = nlat, nlon
        out.append((i, lat, lon, p))
    return out


fwd, bwd = walk(1), walk(-1)[::-1]
pts = bwd + [(si, lat0, lon0, p0)] + fwd
track = [[str(da.time.values[i])[:19].replace("T", " "), la, lo] for i, la, lo, _ in pts]
json.dump({name: track}, open(out_json, "w"))
lats = [p[1] for p in track]
lons = [p[2] for p in track]
print(f"{len(track)} points {track[0][0]} -> {track[-1][0]}; lat {min(lats):.1f}..{max(lats):.1f} "
      f"lon {min(lons):.1f}..{max(lons):.1f}; min SLP {min(p[3] for p in pts):.1f} hPa")
clat, clon = (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2
print(f"suggested box: lat0={int(round(clat - 22))} lon0={int(round(clon - 36))} (44x72 deg)")
