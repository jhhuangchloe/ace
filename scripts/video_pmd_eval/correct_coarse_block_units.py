"""Post-hoc fix for the coarse-block units offset in two-block video PMD output.

Two-block models build the coarse block I + r_hat in coarse_normalization units,
but the output is denormalized with normalization, so each coarse cell is off by
(c - mean_f) * (std_c / std_f - 1) + (mean_c - mean_f), where c is the cell's
value in (log1p-transformed) physical units as written (e.g. +0.58 K in T2m at
301 K). The fine detail averages to zero over each coarse cell, so c is the block
mean of the written output and the offset can be added back exactly, chunk by
chunk. Runs generated with the model flag coarse_block_in_fine_units: true need
no correction. See COARSE_BLOCK_UNITS.md.

Exact for every channel except log-transformed ones (PRATEsfc): inference clamps
expm1(.) at 0, which loses the log1p value of near-dry pixels, so their block
mean (and hence the correction) is approximate there.

Usage:
  python correct_coarse_block_units.py INPUT_ZARR INFERENCE_YAML
      [--out OUT_ZARR | --replace]

  default    write INPUT_ZARR with "-unitsfix" before .zarr; input untouched
  --replace  write a corrected copy, verify it, then rename INPUT_ZARR to
             <name>-uncorrected.zarr and the copy to INPUT_ZARR (delete the
             backup once you are happy)
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr
import yaml

MARK = "coarse_block_units_corrected"


def _correction_params(
    model: dict,
) -> dict[str, tuple[float, float, float, float, float | None]]:
    if not model.get("two_block"):
        sys.exit("Model is not two_block; the coarse block offset does not apply.")
    if model.get("coarse_block_in_fine_units"):
        sys.exit(
            "Model already ran with coarse_block_in_fine_units: true; "
            "nothing to correct."
        )
    fn, cn = model.get("normalization"), model.get("coarse_normalization")
    if fn is None or cn is None:
        sys.exit("Model needs both normalization and coarse_normalization.")
    logs = model.get("log_transform_channels") or {}
    return {
        v: (fn["means"][v], fn["stds"][v], cn["means"][v], cn["stds"][v], logs.get(v))
        for v in model["out_names"]
    }


def _fix_block(x: np.ndarray, factor: int, mf, sf, mc, sc, log_scale) -> np.ndarray:
    """x (..., H, W) physical output -> corrected; coarse cells are the last two
    axes' factor x factor blocks."""
    y = x.astype(np.float64)
    if log_scale is not None:
        y = np.log1p(y * log_scale)
    *lead, h, w = y.shape
    blocks = y.reshape(*lead, h // factor, factor, w // factor, factor).mean(
        axis=(-3, -1)
    )
    offset = (blocks - mf) * (sc / sf - 1.0) + (mc - mf)
    y = y + np.repeat(np.repeat(offset, factor, axis=-2), factor, axis=-1)
    if log_scale is not None:
        y = np.maximum(np.expm1(y) / log_scale, 0.0)
    return y.astype(x.dtype)


def _coarse_store(data: dict) -> str:
    entry = data["coarse"][0]
    return os.path.join(entry["data_path"], entry["file_pattern"])


def _verify(
    path: str, coarse_path: str, params: dict, factor: int, n_timesteps: int
) -> None:
    """At the first fully written clip endpoint, block means must match the coarse
    input."""
    ds = xr.open_zarr(path)
    probe = next(iter(params))
    for i in range(0, ds.sizes["time"], n_timesteps - 1):
        frame = ds[probe].isel(time=i).values
        if np.isfinite(frame).all():
            break
    else:
        sys.exit(f"Verification: no fully written endpoint frame in {path}.")
    t = ds.time.values[i]
    coarse = xr.open_zarr(coarse_path)
    lat_c = ds.latitude.values.reshape(-1, factor).mean(1)
    lon_c = ds.longitude.values.reshape(-1, factor).mean(1)
    for v, (*_, log_scale) in params.items():
        y = ds[v].sel(time=t).values.astype(np.float64)
        x = (
            coarse[v]
            .sel(time=t)
            .sel(latitude=lat_c, longitude=lon_c, method="nearest")
            .values
        )
        x = x.astype(np.float64)
        if log_scale is not None:
            y, x = np.log1p(y * log_scale), np.log1p(np.maximum(x, 0.0) * log_scale)
        h, w = y.shape[-2:]
        blocks = y.reshape(-1, h // factor, factor, w // factor, factor).mean(
            axis=(-3, -1)
        )
        err = float(np.abs(blocks - x).max())
        exact = log_scale is None
        print(
            f"  verify {v:30s} max |block mean - 100 km input| = {err:.5f}"
            f"{'' if exact else ' (approximate channel)'}"
        )
        if exact and err > 1e-2:
            sys.exit(f"Verification failed for {v}; leaving {path} for inspection.")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input_zarr")
    p.add_argument("inference_yaml")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--out")
    g.add_argument("--replace", action="store_true")
    p.add_argument(
        "--factor", type=int, default=4, help="fine cells per coarse cell (default 4)"
    )
    args = p.parse_args()

    src = args.input_zarr.rstrip("/")
    cfg = yaml.safe_load(open(args.inference_yaml))
    params = _correction_params(cfg["model"])
    ds = xr.open_zarr(src)
    if ds.attrs.get(MARK):
        sys.exit(f"{src} is already corrected ({MARK}={ds.attrs[MARK]}).")
    for dim in ("latitude", "longitude"):
        if ds.sizes[dim] % args.factor:
            sys.exit(
                f"{dim} size {ds.sizes[dim]} is not a multiple of factor {args.factor}."
            )

    stem = src[: -len(".zarr")] if src.endswith(".zarr") else src
    dst = (
        f"{stem}.unitsfix-tmp.zarr"
        if args.replace
        else (args.out or f"{stem}-unitsfix.zarr")
    )
    if os.path.exists(dst):
        sys.exit(f"{dst} already exists; remove it first.")

    out = ds.copy()
    for v, (mf, sf, mc, sc, log_scale) in params.items():
        da = ds[v]
        out[v] = da.copy(
            data=da.data.map_blocks(
                _fix_block, args.factor, mf, sf, mc, sc, log_scale, dtype=da.dtype
            )
        )
        out[v].encoding = {k: da.encoding[k] for k in ("chunks",) if k in da.encoding}
        out[v].encoding["fill_value"] = np.float32(np.nan)
    out.attrs[MARK] = "post-hoc, scripts/video_pmd_eval/correct_coarse_block_units.py"
    print(f"writing {dst} ...")
    out.to_zarr(dst, mode="w-", consolidated=True)

    _verify(
        dst, _coarse_store(cfg["data"]), params, args.factor, cfg["data"]["n_timesteps"]
    )
    if args.replace:
        backup = f"{stem}-uncorrected.zarr"
        if os.path.exists(backup):
            sys.exit(f"{backup} exists; corrected copy left at {dst}.")
        os.rename(src, backup)
        os.rename(dst, src)
        print(f"replaced {src} (original kept at {backup})")
    else:
        print(f"done: {dst}")


if __name__ == "__main__":
    main()
