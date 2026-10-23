"""V3 baseline and V3 raw 25 km output for the TC plots (storm-centred patch, daily clips).

For each daily clip (00Z of day d to 00Z of day d + 1) the two coarse 100 km endpoint frames of the
same 44 x 72 coarse patch the models are run on (taken from the ensemble run's own config) go through
the PyTorch port of the trained V3 (fme.downscaling.v3_baseline). Two things are exposed, both on the
patch's 25 km grid and at the 3-hourly steps of the clip:

- ``baseline``: the temporal baseline the diffusion model would use, I(tau): V3's 25 km output
  averaged back to the 100 km grid in physical units (precipitation averaged as physical values),
  pinned to the endpoint frames, then block-replicated to 25 km (like the linear baseline);
- ``raw``: V3's own 25 km output (it sub-grid detail included), which is a stand-alone deterministic
  predictor of the trajectory.
"""
import datetime
import os
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fme.downscaling.v3_baseline import load_v3_baseline, v3_coarse_interior  # noqa: E402

FACTOR = 4
N_STEPS = 9
STEP = datetime.timedelta(hours=3)


class V3Curves:
    def __init__(self, coarse_zarr, truth_ds, v3_path, lat_extent, lon_extent):
        self.coarse = xr.open_zarr(coarse_zarr)
        lat, lon = self.coarse.latitude.values, self.coarse.longitude.values
        self.ilat = np.where((lat >= lat_extent["start"]) & (lat <= lat_extent["stop"]))[0]
        self.ilon = np.where((lon >= lon_extent["start"]) & (lon <= lon_extent["stop"]))[0]
        n_lat, n_lon = len(self.ilat), len(self.ilon)
        fine_lat = truth_ds.latitude.values[FACTOR * self.ilat[0] : FACTOR * (self.ilat[-1] + 1)]
        fine_lon = truth_ds.longitude.values[FACTOR * self.ilon[0] : FACTOR * (self.ilon[-1] + 1)]
        assert (len(fine_lat), len(fine_lon)) == (FACTOR * n_lat, FACTOR * n_lon)
        self.fine_coords = {"latitude": fine_lat, "longitude": fine_lon}
        self.model, self.norm = load_v3_baseline(v3_path, "cpu")
        self._clips = {}

    def _clip(self, d0):
        """(baseline, raw) dicts: variable -> (9, ...) physical arrays for the clip starting d0."""
        if d0 in self._clips:
            return self._clips[d0]
        cache = os.environ.get("V3_CACHE_DIR")  # optional: reuse the CPU inference across plot scripts
        cache_file = (
            Path(cache) / f"v3_{d0.strftime('%Y%m%d')}_{self.ilat[0]}_{self.ilon[0]}.npz" if cache else None
        )
        if cache_file is not None and cache_file.exists():
            z = np.load(cache_file)
            self._clips[d0] = (
                {n: z[f"b_{n}"] for n in self.norm.names},
                {n: z[f"r_{n}"] for n in self.norm.names},
            )
            return self._clips[d0]
        norm = self.norm
        raw_ends, log_ends = {}, {}
        for name in norm.names:
            sel = self.coarse[name].sel(time=[d0, d0 + datetime.timedelta(days=1)])
            x = torch.from_numpy(sel.isel(latitude=self.ilat, longitude=self.ilon).values.astype(np.float32))
            raw_ends[name] = x[None]  # (1, 2, h, w) physical
            scale = norm.log_scales.get(name)
            log_ends[name] = torch.log1p(x.clamp(min=0.0) * scale)[None] if scale else x[None]
        interior = v3_coarse_interior(self.model, norm, log_ends, n_steps=N_STEPS)
        baseline = {}
        for name in norm.names:
            v = interior[name]
            if (scale := norm.log_scales.get(name)) is not None:
                v = torch.expm1(v).clamp(min=0.0) / scale
            frames = torch.cat([raw_ends[name][:, :1], v, raw_ends[name][:, -1:]], dim=1)[0]
            baseline[name] = np.repeat(np.repeat(frames.numpy(), FACTOR, axis=1), FACTOR, axis=2)
        ends = torch.stack(
            [(log_ends[n] - norm.means[i]) / norm.stds[i] for i, n in enumerate(norm.names)], dim=-1
        )
        out = self.model(ends.float())[0]  # (9, 4h, 4w, C), normalized
        raw = {}
        for i, name in enumerate(norm.names):
            v = out[..., i] * norm.stds[i] + norm.means[i]
            if (scale := norm.log_scales.get(name)) is not None:
                v = torch.expm1(v).clamp(min=0.0) / scale
            raw[name] = v.numpy()
        self._clips[d0] = (baseline, raw)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_file, **{f"b_{n}": a for n, a in baseline.items()}, **{f"r_{n}": a for n, a in raw.items()})
        return self._clips[d0]

    def _field(self, which, var, t):
        d0 = type(t)(t.year, t.month, t.day, 0, 0, 0)
        k = int(round((t - d0) / STEP))
        data = self._clip(d0)[0 if which == "baseline" else 1][var][k]
        return xr.DataArray(data, dims=("latitude", "longitude"), coords=self.fine_coords)

    def baseline(self, var, t):
        return self._field("baseline", var, t)

    def raw(self, var, t):
        return self._field("raw", var, t)
