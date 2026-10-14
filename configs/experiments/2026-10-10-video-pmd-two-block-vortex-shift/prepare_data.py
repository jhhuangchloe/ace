"""Copy a time-subset of the paired 25km/100km X-SHiELD stores from the public
Hugging Face bucket to a local directory, preserving the original zarr layout
(so the downscaling data loader sees the same store it sees on weka).

Only the 5 variables the two-block model uses are copied, plus coordinates.
Chunks outside the requested time ranges are simply absent; the training
config's ``subset.start_time/stop_time`` must stay inside the copied ranges.

Source: https://huggingface.co/buckets/allenai/ai2cm-hosted-test
Size: ~55 GB per year of the 25km store + ~3 GB per year of the 100km store.

Usage (resumable -- re-running skips complete files):
    python prepare_data.py --dry-run
    python prepare_data.py
    python prepare_data.py --train-stop 2017-12-31     # more training years
"""

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BUCKET = "allenai/ai2cm-hosted-test"
API = f"https://huggingface.co/api/buckets/{BUCKET}/tree"
RESOLVE = f"https://huggingface.co/buckets/{BUCKET}/resolve"
STORES = {  # name -> time steps per chunk along axis 0
    "2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr": 128,
    "2026-07-14-X-SHiELD-AMIP-FME-3h-100km.zarr": 420,
}
VARIABLES = [
    "eastward_wind_at_ten_meters",
    "northward_wind_at_ten_meters",
    "PRMSL",
    "PRATEsfc",
    "air_temperature_at_two_meters",
]
COORDS = ["time", "latitude", "longitude", "grid_xt_bnds", "grid_yt_bnds"]
T0 = dt.datetime(2013, 1, 2, 0)  # time[0]; 3-hourly


def step(date: str) -> int:
    return int((dt.datetime.fromisoformat(date) - T0).total_seconds() // 10800)


def list_bucket() -> list[dict]:
    url: str | None = f"{API}?recursive=true&limit=10000&sort=path"
    items: list[dict] = []
    while url:
        with urllib.request.urlopen(url) as resp:
            items += json.load(resp)
            link = resp.headers.get("link") or ""
        url = link.split("<")[1].split(">")[0] if 'rel="next"' in link else None
    return items


def wanted_files(items, ranges) -> list[dict]:
    out = []
    for e in items:
        if e["type"] != "file":
            continue
        store, _, rest = e["path"].partition("/")
        if store not in STORES:
            continue
        parts = rest.split("/")
        if rest == "zarr.json":
            out.append(e)
        elif parts[0] in COORDS:
            out.append(e)
        elif parts[0] in VARIABLES:
            if parts[1] == "zarr.json":
                out.append(e)
            elif parts[1] == "c":
                chunk = int(parts[2])
                lo = chunk * STORES[store]
                hi = lo + STORES[store] - 1
                if any(lo <= b and a <= hi for a, b in ranges):
                    out.append(e)
    return out


def fetch(e, out_dir) -> int:
    dest = os.path.join(out_dir, e["path"])
    if os.path.exists(dest) and os.path.getsize(dest) == e["size"]:
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(4):
        try:
            tmp = dest + ".part"
            with (
                urllib.request.urlopen(f"{RESOLVE}/{e['path']}") as r,
                open(tmp, "wb") as f,
            ):
                shutil.copyfileobj(r, f, 1 << 20)
            if os.path.getsize(tmp) != e["size"]:
                raise OSError("size mismatch")
            os.replace(tmp, dest)
            return e["size"]
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                raise
            print(f"retry {e['path']}: {exc}", file=sys.stderr)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="/data/pmd/local_data")
    p.add_argument("--train-start", default="2013-01-02")
    p.add_argument("--train-stop", default="2015-12-31")
    p.add_argument("--val-start", default="2022-06-01")
    p.add_argument("--val-stop", default="2022-09-30")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ranges = [
        (step(args.train_start), step(args.train_stop) + 7),
        (step(args.val_start), step(args.val_stop) + 7),
    ]
    files = wanted_files(list_bucket(), ranges)
    total = sum(e["size"] for e in files)
    os.makedirs(args.out_dir, exist_ok=True)
    free = shutil.disk_usage(args.out_dir).free
    print(
        f"{len(files)} files, {total / 1e9:.1f} GB to copy to {args.out_dir} "
        f"({free / 1e9:.0f} GB free)"
    )
    if args.dry_run:
        return
    if total > 0.5 * free:
        sys.exit("refusing: copy would use more than half of the free space")
    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for i, n in enumerate(pool.map(lambda e: fetch(e, args.out_dir), files)):
            done += n
            if i % 50 == 0:
                print(
                    f"{i}/{len(files)} files, {done / 1e9:.1f} GB downloaded",
                    flush=True,
                )
    print("done")


if __name__ == "__main__":
    main()
