"""Put the cascade's SR output on the common S2022 grid/time axis of the other models' zarrs.

Usage: cascade_to_zarr.py CASCADE_SR_ZARR REF_ZARR OUT_ZARR
Takes the 5 channels, selects REF_ZARR's latitude/longitude (nearest cell) and time axis (frames the cascade
did not produce stay NaN, so scoring uses only the frames every model has), and writes dims (time, ensemble, latitude, longitude).
"""
import sys

import numpy as np
import xarray as xr

src_path, ref_path, out_path = sys.argv[1:]
VARS = ["eastward_wind_at_ten_meters", "northward_wind_at_ten_meters", "PRMSL", "PRATEsfc", "air_temperature_at_two_meters"]
src = xr.open_zarr(src_path)
ref = xr.open_zarr(ref_path)
print("cascade:", dict(src.sizes), "ref:", dict(ref.sizes))
ds = src[VARS].sel(latitude=ref.latitude, longitude=ref.longitude, method="nearest")
ds = ds.assign_coords(latitude=ref.latitude, longitude=ref.longitude)
if "ensemble" not in ds.dims:
    ds = ds.expand_dims("ensemble")
common = ref.time.values[np.isin(ref.time.values, ds.time.values)]
print(f"{len(common)}/{ref.sizes['time']} reference frames available in the cascade output")
if len(common) == 0:
    sys.exit("no common frames: time axes do not overlap")
ds = ds.reindex(time=ref.time).transpose("time", "ensemble", "latitude", "longitude")
ds = ds.load().drop_encoding()  # small regional field; the env has no dask
ds.to_zarr(out_path, mode="w")
print("wrote", out_path)
