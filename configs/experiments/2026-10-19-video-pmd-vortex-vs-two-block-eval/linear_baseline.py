"""Linear-in-time baseline from the daily 00Z 100 km endpoints, for the TC plots.

Each 3-hourly frame at time t is ``(1 - tau) * U(C(00Z of that day)) + tau * U(C(00Z of the next
day))`` with ``tau = (t - 00Z) / 24 h``, ``C`` the 100 km store's field and ``U`` the spatial
upsampling to the 25 km grid:

- ``nearest``: block replication (each 100 km cell repeated 4 x 4), which is what the two-block
  method starts from;
- ``bicubic``: bicubic interpolation (cell-centre aligned, periodic in longitude, edge-replicated in
  latitude), a smoother 25 km version of the same endpoints.

Either way there is no storm motion and no sub-grid detail beyond what ``U`` interpolates: only a
cross-fade between the two endpoint fields. (Upsampling and the time blend commute, so ``U`` is
applied once per endpoint.)
"""
import datetime

import numpy as np
import torch
import xarray as xr

FACTOR = 4
PAD = 2  # coarse cells of padding for bicubic (its stencil is 4 wide)


def _bicubic(coarse: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(coarse)[None, None]
    x = torch.cat([x[..., -PAD:], x, x[..., :PAD]], dim=-1)  # periodic in longitude
    x = torch.nn.functional.pad(x, (0, 0, PAD, PAD), mode="replicate")  # latitude edges
    y = torch.nn.functional.interpolate(x, scale_factor=FACTOR, mode="bicubic", align_corners=False)
    m = PAD * FACTOR
    return y[0, 0, m:-m, m:-m].numpy()


class LinearBaseline:
    def __init__(self, coarse_zarr: str, truth_ds: xr.Dataset, upsample: str = "nearest"):
        assert upsample in ("nearest", "bicubic")
        self.upsample = upsample
        self.coarse = xr.open_zarr(coarse_zarr)
        self.lat, self.lon = truth_ds.latitude, truth_ds.longitude
        n_lat, n_lon = self.coarse.sizes["latitude"], self.coarse.sizes["longitude"]
        assert (self.lat.size, self.lon.size) == (FACTOR * n_lat, FACTOR * n_lon), "grids not 4x apart"
        self._cache = {}

    def _endpoint(self, var, t):
        """The 00Z field at time ``t`` upsampled to the 25 km grid."""
        if (var, t) not in self._cache:
            coarse = self.coarse[var].sel(time=t).values.astype(np.float64)
            if self.upsample == "bicubic":
                fine = _bicubic(coarse)
            else:
                fine = np.repeat(np.repeat(coarse, FACTOR, axis=0), FACTOR, axis=1)
            self._cache[(var, t)] = fine
        return self._cache[(var, t)]

    def field(self, var: str, t) -> xr.DataArray:
        """The baseline ``var`` at time ``t`` on the 25 km grid, dims (latitude, longitude)."""
        t0 = type(t)(t.year, t.month, t.day, 0, 0, 0)
        tau = (t - t0).total_seconds() / 86400.0
        first = self._endpoint(var, t0)
        last = self._endpoint(var, t0 + datetime.timedelta(days=1)) if tau > 0 else first
        return xr.DataArray(
            (1.0 - tau) * first + tau * last,
            dims=("latitude", "longitude"),
            coords={"latitude": self.lat, "longitude": self.lon},
        )
