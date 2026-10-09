#!/usr/bin/env python3
"""Build monthly composite GeoTIFFs of PlanetScope 4-band and 8-band imagery.

For every calendar month present in the raw archive, this builds up to three
composites:
  - "4b": every band of the 4-band product (blue, green, red, nir)
  - "8b": every band of the 8-band product (coastal_blue, blue, green_i,
    green, yellow, red, rededge, nir)
  - "combined": only the 4 bands the two products share by logical name
    (blue, green, red, nir), pooling scenes from both products together

Each composite carries, per spectral band, the per-pixel min, 25th
percentile, median, 75th percentile, and max reflectance across every scene
acquired that month that covers that pixel and passes its UDM2 `clear` mask
-- plus two extra bands shared across the whole image: `n_scenes` (how many
scenes had real (non-nodata) data at that pixel, regardless of cloud status)
and `n_unmasked` (the subset of those that were also UDM2-clear, i.e. how
many observations the reflectance statistics above were actually drawn
from). A pixel with n_unmasked == 0 gets NODATA in every statistic band even
if n_scenes > 0 (covered, but every covering scene was masked there).

Scenes are grouped into months by their own acquisition timestamp (the
`YYYYMMDD_HHMMSS...` prefix PlanetScope encodes in every filename -- see
io.parse_acquisition_time), not by the enclosing raw folder's day-of-year
range, though in this archive the two coincide.

The actual compositing (grid union, open-once scene handles, the 2D-tile
percentile loop, output writing) lives in psnorm/compositing.py, shared
with scripts/build_solstice_composite_reference.py -- see that module's
docstring for the engine's design and the profiling behind it.

--workers parallelizes across months (fully independent units of work,
each with modest peak memory), using a process pool so each worker gets
its own GDAL/numpy state.

Usage:
    .venv/bin/python scripts/build_monthly_composites.py \\
        --raw-root /mnt/e/E_Coast_PlanetScope/philadelphia/raw \\
        --output-root /mnt/e/E_Coast_PlanetScope/philadelphia/monthly_composites \\
        --workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from collections import defaultdict

from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psnorm import compositing, io, sensors  # noqa: E402

gdal.UseExceptions()

COMMON_BANDS = ["blue", "green", "red", "nir"]  # shared by the 4b and 8b products

FOUR_BAND_SUFFIX = "_3B_AnalyticMS_SR_harmonized_clip.tif"
EIGHT_BAND_SUFFIX = "_3B_AnalyticMS_SR_8b_harmonized_clip.tif"
UDM2_SUFFIX = "_3B_udm2_clip.tif"


# --------------------------------------------------------------------------
# Scene discovery, grouped by calendar month
# --------------------------------------------------------------------------

def discover_month_scenes(
    raw_root: str, sensor_subdir: str, analytic_suffix: str, *, log=print,
) -> dict[tuple[int, int], list[io.Scene]]:
    """{(year, month): [Scene, ...]}, keyed by each scene's own acquisition
    date rather than the enclosing folder name."""
    scenes = compositing.discover_product_scenes(raw_root, sensor_subdir, analytic_suffix, UDM2_SUFFIX, log=log)
    by_month: dict[tuple[int, int], list[io.Scene]] = defaultdict(list)
    undated = 0
    for s in scenes:
        if s.acquired is None:
            undated += 1
            continue
        by_month[(s.acquired.year, s.acquired.month)].append(s)
    total = sum(len(v) for v in by_month.values())
    log(f"    {total} dated scene(s) across {len(by_month)} month(s)"
        + (f" ({undated} undated scene(s) skipped)" if undated else ""))
    return dict(by_month)


# --------------------------------------------------------------------------
# Orchestration across every (year, month) x {4b, 8b, combined}
# --------------------------------------------------------------------------

def _process_one_month(
    year: int, month: int,
    scenes_4b_month: list[io.Scene], scenes_8b_month: list[io.Scene],
    output_root: str, block_rows: int, block_cols: int, gdal_cache_mb: int,
) -> str:
    """All three composites ({4b, 8b, combined}) for one (year, month).
    Self-contained (takes only picklable Scene lists, not the full
    discovery dicts) so it can run standalone in a worker process under
    --workers > 1, as well as in-process for the sequential path."""
    gdal.UseExceptions()
    gdal.SetCacheMax(gdal_cache_mb * 1024 * 1024)

    label = f"{year}-{month:02d}"
    log_prefix = f"[pid {os.getpid()}] " if multiprocessing.current_process().name != "MainProcess" else ""

    def log(msg: str) -> None:
        print(f"{log_prefix}{msg}", flush=True)

    if scenes_4b_month:
        log(f"[4b {label}] {len(scenes_4b_month)} scene(s)")
        resolved = compositing.resolve_scenes(
            [(s, None) for s in scenes_4b_month], sensors.BAND_PROFILES[4], log=log,
        )
        if resolved:
            out_path = os.path.join(output_root, "4b", f"philadelphia_{label}.tif")
            compositing.build_composite(
                resolved, sensors.BAND_PROFILES[4], list(compositing.RANK_QUANTILES), out_path,
                block_rows=block_rows, block_cols=block_cols, log=log,
            )
        else:
            log(f"  No usable 4b scenes for {label}; skipping.")

    if scenes_8b_month:
        log(f"[8b {label}] {len(scenes_8b_month)} scene(s)")
        resolved = compositing.resolve_scenes(
            [(s, None) for s in scenes_8b_month], sensors.BAND_PROFILES[8], log=log,
        )
        if resolved:
            out_path = os.path.join(output_root, "8b", f"philadelphia_{label}.tif")
            compositing.build_composite(
                resolved, sensors.BAND_PROFILES[8], list(compositing.RANK_QUANTILES), out_path,
                block_rows=block_rows, block_cols=block_cols, log=log,
            )
        else:
            log(f"  No usable 8b scenes for {label}; skipping.")

    combined_scenes = scenes_4b_month + scenes_8b_month
    if combined_scenes:
        log(f"[combined {label}] {len(combined_scenes)} scene(s)")
        resolved = compositing.resolve_scenes(
            [(s, None) for s in combined_scenes], COMMON_BANDS, log=log,
        )
        if resolved:
            out_path = os.path.join(output_root, "combined", f"philadelphia_{label}.tif")
            compositing.build_composite(
                resolved, COMMON_BANDS, list(compositing.RANK_QUANTILES), out_path,
                block_rows=block_rows, block_cols=block_cols, log=log,
            )
        else:
            log(f"  No usable scenes for combined {label}; skipping.")

    return label


def _process_one_month_star(args: tuple) -> str:
    """multiprocessing.Pool.imap_unordered needs a single-argument callable;
    this just unpacks the tuple _process_one_month otherwise takes as
    separate arguments."""
    return _process_one_month(*args)


def run(
    raw_root: str, output_root: str, *, months: set[tuple[int, int]] | None = None,
    block_rows: int = compositing.DEFAULT_BLOCK_ROWS, block_cols: int = compositing.DEFAULT_BLOCK_COLS,
    gdal_cache_mb: int = compositing.DEFAULT_GDAL_CACHE_MB, workers: int = 1, log=print,
) -> None:
    gdal.SetCacheMax(gdal_cache_mb * 1024 * 1024)

    log("Discovering 4-band scenes...")
    scenes_4b = discover_month_scenes(raw_root, "philadelphia4b", FOUR_BAND_SUFFIX, log=log)
    log("Discovering 8-band scenes...")
    scenes_8b = discover_month_scenes(raw_root, "philadelphia8b", EIGHT_BAND_SUFFIX, log=log)

    all_months = sorted(set(scenes_4b) | set(scenes_8b))
    if months is not None:
        all_months = [m for m in all_months if m in months]
    log(f"{len(all_months)} month(s) to process with {workers} worker(s).")

    jobs = [
        (year, month, scenes_4b.get((year, month), []), scenes_8b.get((year, month), []),
         output_root, block_rows, block_cols, gdal_cache_mb)
        for year, month in all_months
    ]

    if workers <= 1:
        for job in jobs:
            _process_one_month_star(job)
        return

    # "fork" (the Linux default) is used explicitly rather than relying on
    # the platform default: discovery above only reads headers/paths (no
    # open GDAL datasets to duplicate across the fork), and every job's
    # arguments are plain picklable dataclasses/strings/ints either way, so
    # this is safe and avoids re-importing/re-running module-level code the
    # "spawn" start method would require.
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        for label in pool.imap_unordered(_process_one_month_star, jobs):
            log(f"=== finished month {label} ===")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", default="/mnt/e/E_Coast_PlanetScope/philadelphia/raw")
    parser.add_argument("--output-root", default="/mnt/e/E_Coast_PlanetScope/philadelphia/monthly_composites")
    parser.add_argument("--months", nargs="+", default=None,
                         help="restrict to specific YYYY-MM month(s) (default: every month found)")
    parser.add_argument("--block-rows", type=int, default=compositing.DEFAULT_BLOCK_ROWS)
    parser.add_argument("--block-cols", type=int, default=compositing.DEFAULT_BLOCK_COLS)
    parser.add_argument("--gdal-cache-mb", type=int, default=compositing.DEFAULT_GDAL_CACHE_MB,
                         help="GDAL block cache size per process, in MB (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=1,
                         help="process this many months in parallel (default: 1, sequential)")
    args = parser.parse_args()

    months = None
    if args.months:
        months = set()
        for m in args.months:
            y, mo = m.split("-")
            months.add((int(y), int(mo)))

    run(
        args.raw_root, args.output_root, months=months,
        block_rows=args.block_rows, block_cols=args.block_cols,
        gdal_cache_mb=args.gdal_cache_mb, workers=args.workers,
    )


if __name__ == "__main__":
    main()
