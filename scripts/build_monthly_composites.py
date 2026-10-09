#!/usr/bin/env python3
"""Build monthly composite GeoTIFFs of PlanetScope 4-band and 8-band imagery.

For every calendar month present in the raw archive, this builds up to three
composites:
  - "4b": every band of the 4-band product (blue, green, red, nir)
  - "8b": every band of the 8-band product (coastal_blue, blue, green_i,
    green, yellow, red, rededge, nir)
  - "combined": only the 4 bands the two products share by logical name
    (blue, green, red, nir), pooling scenes from both products together

Each composite band set carries, per spectral band, the per-pixel min, 25th
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

This reuses the same grid-union + 2D-tile + sort-based rank-statistic
machinery as scripts/build_solstice_composite_reference.py (percentiles
aren't separable into a running-sum update rule the way mean/stddev are, so
each tile briefly holds every contributing scene's values stacked together
before collapsing them -- see that script's module docstring and
_fast_nan_rank_stat for why). The same trick also produces min/max cheaply
(q=0 and q=100 on the same sorted stack), so this script doesn't need a
separate nanmin/nanmax pass.

Performance notes (see _open_scenes/OpenScene): profiling the first full
run of this script (psnorm.io.read_window_bands/masking.read_udm2_valid_mask
each call gdal.Open() internally) showed essentially 100% of wall time spent
in per-(tile, scene) reads, dominated by repeatedly reopening and
re-decompressing the *same* scene file once per output tile it overlaps
(up to ~400x for a wide scene). This version opens each scene's analytic +
UDM2 datasets exactly once per month and reuses the open GDAL Band objects
across every tile -- a ~27x speedup on a warm-cache microbenchmark against
this same data, and the measured production bottleneck. A larger GDAL
block cache (--gdal-cache-mb) and larger tiles (--block-rows/--block-cols)
compound with this by letting more of a scene's decompressed blocks stay
resident across the tiles it touches. --workers parallelizes across months
(fully independent units of work, each with modest peak memory), using a
process pool so each worker gets its own GDAL/numpy state.

Usage:
    .venv/bin/python scripts/build_monthly_composites.py \\
        --raw-root /mnt/e/E_Coast_PlanetScope/philadelphia/raw \\
        --output-root /mnt/e/E_Coast_PlanetScope/philadelphia/monthly_composites \\
        --workers 4
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psnorm import io, masking, sensors  # noqa: E402

gdal.UseExceptions()

NODATA = -9999.0
STATS = ["min", "p25", "median", "p75", "max"]
_STAT_QUANTILES = {"min": 0, "p25": 25, "median": 50, "p75": 75, "max": 100}
COMMON_BANDS = ["blue", "green", "red", "nir"]  # shared by the 4b and 8b products

FOUR_BAND_SUFFIX = "_3B_AnalyticMS_SR_harmonized_clip.tif"
EIGHT_BAND_SUFFIX = "_3B_AnalyticMS_SR_8b_harmonized_clip.tif"
UDM2_SUFFIX = "_3B_udm2_clip.tif"

DEFAULT_BLOCK_ROWS = 512
DEFAULT_BLOCK_COLS = 2048
DEFAULT_GDAL_CACHE_MB = 512


# --------------------------------------------------------------------------
# Scene discovery, grouped by calendar month
# --------------------------------------------------------------------------

def discover_month_scenes(
    raw_root: str, sensor_subdir: str, analytic_suffix: str, *, log=print,
) -> dict[tuple[int, int], list[io.Scene]]:
    """{(year, month): [Scene, ...]} for every scene under
    `raw_root`/`sensor_subdir`/`sensor_subdir`_*/, keyed by each scene's own
    acquisition date rather than the enclosing folder name."""
    sensor_dir = os.path.join(raw_root, sensor_subdir)
    period_dirs = sorted(
        d for d in glob.glob(os.path.join(sensor_dir, f"{sensor_subdir}_*")) if os.path.isdir(d)
    )
    by_month: dict[tuple[int, int], list[io.Scene]] = defaultdict(list)
    undated = 0
    for d in period_dirs:
        found = io.discover_scenes(d, analytic_suffix=analytic_suffix, udm2_suffix=UDM2_SUFFIX)
        for s in found:
            if s.acquired is None:
                undated += 1
                continue
            by_month[(s.acquired.year, s.acquired.month)].append(s)
    total = sum(len(v) for v in by_month.values())
    log(f"  '{sensor_dir}': {total} dated scene(s) across {len(by_month)} month(s)"
        + (f" ({undated} undated scene(s) skipped)" if undated else ""))
    return dict(by_month)


def _resolve_scenes(
    scenes: list[io.Scene], band_names: list[str], *, log=print,
) -> list[tuple[io.Scene, io.RasterInfo, list[int]]]:
    """(Scene, RasterInfo, band_indices) for each scene that has every band
    in `band_names` and shares a pixel grid with the first resolved scene.
    Scenes that fail either check are logged and skipped rather than
    aborting the whole month -- a single bad/mismatched file in a batch
    spanning years of imagery shouldn't kill the run. Only reads headers
    (io.get_raster_info/sensors.detect_band_names never call
    ReadAsArray) -- actual pixel-data datasets are opened once later, by
    _open_scenes, and reused across every tile."""
    resolved: list[tuple[io.Scene, io.RasterInfo, list[int]]] = []
    ref_info: io.RasterInfo | None = None
    for s in scenes:
        try:
            info = io.get_raster_info(s.analytic_path)
        except Exception as exc:
            log(f"    WARNING: could not read '{s.analytic_path}': {exc}; skipping.")
            continue
        try:
            detected = sensors.detect_band_names(s.analytic_path)
        except Exception as exc:
            log(f"    WARNING: could not detect bands for '{s.analytic_path}': {exc}; skipping.")
            continue
        indices = [sensors.band_index(detected, name) for name in band_names]
        if any(i is None for i in indices):
            log(f"    WARNING: '{s.analytic_path}' (bands {detected}) is missing one of "
                f"{band_names}; skipping.")
            continue
        if ref_info is None:
            ref_info = info
        elif not io.grids_aligned(ref_info, info):
            log(f"    WARNING: '{s.analytic_path}' is not on the same pixel grid as "
                f"'{resolved[0][0].analytic_path}'; skipping.")
            continue
        resolved.append((s, info, indices))
    return resolved


# --------------------------------------------------------------------------
# Open-once-per-month scene handles (the main perf fix -- see module
# docstring). Every tile in build_month_composite reuses these same Band
# objects instead of each tile reopening every scene's files from scratch.
# --------------------------------------------------------------------------

@dataclass
class OpenScene:
    scene: io.Scene
    info: io.RasterInfo
    band_indices: list[int]
    sr_dataset: "gdal.Dataset"
    sr_bands: list  # gdal.Band, one per band_indices entry
    udm2_dataset: "gdal.Dataset | None"
    udm2_band: "gdal.Band | None"


def _open_scenes(
    resolved_scenes: list[tuple[io.Scene, io.RasterInfo, list[int]]], *, log=print,
) -> list[OpenScene]:
    opened = []
    for scene, info, band_indices in resolved_scenes:
        try:
            sr_dataset = gdal.Open(scene.analytic_path, gdal.GA_ReadOnly)
            sr_bands = [sr_dataset.GetRasterBand(i) for i in band_indices]
            udm2_dataset = None
            udm2_band = None
            if scene.udm2_path is not None:
                udm2_dataset = gdal.Open(scene.udm2_path, gdal.GA_ReadOnly)
                udm2_band = udm2_dataset.GetRasterBand(1)
        except Exception as exc:
            log(f"    WARNING: could not open '{scene.analytic_path}': {exc}; skipping.")
            continue
        opened.append(OpenScene(scene, info, band_indices, sr_dataset, sr_bands, udm2_dataset, udm2_band))
    return opened


def _close_scenes(opened_scenes: list[OpenScene]) -> None:
    for s in opened_scenes:
        s.sr_bands = []
        s.sr_dataset = None
        s.udm2_band = None
        s.udm2_dataset = None


def _masked_read_open(
    opened: OpenScene, window: io.Window,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(values, footprint_valid, fully_valid) for `window`, read from
    `opened`'s already-open Band objects -- no gdal.Open() call here, which
    is the entire point (see OpenScene)."""
    xoff, yoff, xsize, ysize = window
    arr = np.stack(
        [b.ReadAsArray(xoff, yoff, xsize, ysize) for b in opened.sr_bands], axis=0,
    ).astype(np.float32)
    if opened.info.nodata is None:
        footprint_valid = np.ones((ysize, xsize), dtype=bool)
    else:
        footprint_valid = np.all(arr != opened.info.nodata, axis=0)
    if opened.udm2_band is not None:
        clear = opened.udm2_band.ReadAsArray(xoff, yoff, xsize, ysize) == 1
        fully_valid = footprint_valid & clear
    else:
        fully_valid = footprint_valid
    return arr, footprint_valid, fully_valid


# --------------------------------------------------------------------------
# Compositing: per-pixel min/p25/median/p75/max + scene counts
# --------------------------------------------------------------------------

def _union_output_info(infos: list[io.RasterInfo]) -> io.RasterInfo:
    """A RasterInfo covering the union of every scene's bounds, on the same
    grid (same origin phase, same pixel size/CRS) as the first scene --
    valid only when every scene shares one grid (see io.grids_aligned)."""
    first = infos[0]
    px, py = first.geotransform[1], first.geotransform[5]
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for info in infos:
        gt = info.geotransform
        x0, y1 = gt[0], gt[3]
        x1 = x0 + info.width * gt[1]
        y0 = y1 + info.height * gt[5]
        minx, maxx = min(minx, x0, x1), max(maxx, x0, x1)
        miny, maxy = min(miny, y0, y1), max(maxy, y0, y1)

    ox, oy = first.geotransform[0], first.geotransform[3]
    x0 = ox + np.floor((minx - ox) / px) * px
    y1 = oy + np.floor((maxy - oy) / py) * py
    width = int(np.ceil((maxx - x0) / px))
    height = int(np.ceil((y1 - miny) / abs(py)))
    gt = (float(x0), px, 0.0, float(y1), 0.0, py)
    return replace(first, path="", width=width, height=height, geotransform=gt, nodata=None)


def _fast_nan_rank_stat(sorted_stack: np.ndarray, counts: np.ndarray, q: float) -> np.ndarray:
    """The `q`-th percentile (0-100) along axis 0 of `sorted_stack`
    ((n_scenes, bands, rows, cols), already np.sort()-ed along axis 0 so
    every pixel's NaNs trail its real values), using `counts` (valid values
    per pixel, shared across bands) to pick each pixel's pair of order
    statistics to interpolate between. q=0/100 fall out as min/max for
    free. See build_solstice_composite_reference.py's copy of this
    function for the profiling rationale (np.nanpercentile is ~1000x
    slower on arrays shaped like these)."""
    idx = (np.maximum(counts, 1) - 1) * (q / 100.0)
    lower_idx = np.floor(idx).astype(np.int64)
    upper_idx = np.ceil(idx).astype(np.int64)
    frac = (idx - lower_idx).astype(np.float32)
    bshape = (1,) + sorted_stack.shape[1:]
    lower_vals = np.take_along_axis(sorted_stack, np.broadcast_to(lower_idx[None, None], bshape), axis=0)[0]
    upper_vals = np.take_along_axis(sorted_stack, np.broadcast_to(upper_idx[None, None], bshape), axis=0)[0]
    return lower_vals + (upper_vals - lower_vals) * frac[None, :, :]


def _output_band_names(band_names: list[str]) -> list[str]:
    names = [f"{band}_{stat}" for band in band_names for stat in STATS]
    names += ["n_scenes", "n_unmasked"]
    return names


def _create_output(path: str, out_info: io.RasterInfo, out_band_names: list[str]) -> "gdal.Dataset":
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        path, out_info.width, out_info.height, len(out_band_names), gdal.GDT_Float32,
        options=["COMPRESS=LZW", "BIGTIFF=IF_SAFER", "NUM_THREADS=ALL_CPUS"],
    )
    ds.SetGeoTransform(out_info.geotransform)
    ds.SetProjection(out_info.crs)
    for b, name in enumerate(out_band_names):
        ds.GetRasterBand(b + 1).SetNoDataValue(NODATA)
        ds.GetRasterBand(b + 1).SetDescription(name)
    return ds


def build_month_composite(
    resolved_scenes: list[tuple[io.Scene, io.RasterInfo, list[int]]],
    band_names: list[str],
    output_path: str,
    *, block_rows: int = DEFAULT_BLOCK_ROWS, block_cols: int = DEFAULT_BLOCK_COLS, log=print,
) -> str:
    """Composite `resolved_scenes` (see _resolve_scenes) into one multi-band
    GeoTIFF at `output_path`: per spectral band in `band_names`, the
    min/p25/median/p75/max reflectance across every scene's UDM2-clear,
    non-nodata pixels at each location, plus two shared n_scenes/n_unmasked
    count bands. Memory-bounded by processing the output grid in 2D tiles
    (contributing scenes x bands x tile_rows x tile_cols), since
    percentiles need every contributing scene's value stacked together per
    tile -- unlike a running-sum composite, there's no incremental update
    rule for a percentile. Every scene is opened exactly once (see
    _open_scenes) and its Band objects are reused across every tile."""
    if not resolved_scenes:
        raise ValueError("No resolved scenes to composite.")

    out_info = _union_output_info([info for _s, info, _idx in resolved_scenes])
    n_bands = len(band_names)
    out_band_names = _output_band_names(band_names)
    log(f"  Output grid: {out_info.width}x{out_info.height} px "
        f"({out_info.width * out_info.height / 1e6:.1f} Mpx), {len(out_band_names)} bands, "
        f"{len(resolved_scenes)} scene(s)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds = _create_output(output_path, out_info, out_band_names)

    opened_scenes = _open_scenes(resolved_scenes, log=log)
    if not opened_scenes:
        raise ValueError(f"Could not open any scene for '{output_path}'.")

    row_blocks = list(io.iter_row_blocks(out_info.height, block_rows))
    col_blocks = list(io.iter_row_blocks(out_info.width, block_cols))
    n_tiles = len(row_blocks) * len(col_blocks)

    run_start = time.time()
    tile_i = 0
    try:
        for row_start, n_rows in row_blocks:
            for col_start, n_cols in col_blocks:
                tile_i += 1
                tile_info = io.windowed_raster_info(out_info, (col_start, row_start, n_cols, n_rows))

                footprint_count = np.zeros((n_rows, n_cols), dtype=np.int32)
                unmasked_count = np.zeros((n_rows, n_cols), dtype=np.int32)
                value_contributions = []

                for opened in opened_scenes:
                    overlap = io.overlap_window(tile_info, opened.info)
                    if overlap is None:
                        continue
                    tile_window, scene_window = overlap
                    values, footprint_valid, fully_valid = _masked_read_open(opened, scene_window)
                    if not footprint_valid.any():
                        continue
                    tx, ty, tw, th = tile_window
                    footprint_count[ty : ty + th, tx : tx + tw] += footprint_valid
                    unmasked_count[ty : ty + th, tx : tx + tw] += fully_valid
                    if fully_valid.any():
                        buf = np.full((n_bands, n_rows, n_cols), np.nan, dtype=np.float32)
                        buf[:, ty : ty + th, tx : tx + tw] = np.where(fully_valid, values, np.nan)
                        value_contributions.append(buf)

                no_footprint = footprint_count == 0
                no_unmasked = unmasked_count == 0

                if value_contributions:
                    sorted_stack = np.sort(np.stack(value_contributions, axis=0), axis=0)
                    with np.errstate(invalid="ignore"):
                        stat_arrays = {
                            stat: _fast_nan_rank_stat(sorted_stack, unmasked_count, q)
                            for stat, q in _STAT_QUANTILES.items()
                        }
                else:
                    stat_arrays = {
                        stat: np.full((n_bands, n_rows, n_cols), NODATA, dtype=np.float32) for stat in STATS
                    }

                band_cursor = 1
                for bi in range(n_bands):
                    for stat in STATS:
                        arr = stat_arrays[stat][bi].astype(np.float32, copy=True)
                        arr[no_unmasked] = NODATA
                        ds.GetRasterBand(band_cursor).WriteArray(arr, xoff=col_start, yoff=row_start)
                        band_cursor += 1

                n_scenes_arr = footprint_count.astype(np.float32)
                n_scenes_arr[no_footprint] = NODATA
                ds.GetRasterBand(band_cursor).WriteArray(n_scenes_arr, xoff=col_start, yoff=row_start)
                band_cursor += 1

                n_unmasked_arr = unmasked_count.astype(np.float32)
                n_unmasked_arr[no_footprint] = NODATA  # a real 0 (covered but all masked) is kept
                ds.GetRasterBand(band_cursor).WriteArray(n_unmasked_arr, xoff=col_start, yoff=row_start)

                elapsed = time.time() - run_start
                eta = elapsed / tile_i * (n_tiles - tile_i)
                log(f"    tile {tile_i}/{n_tiles}: {len(value_contributions)} scene(s) contributed "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    finally:
        _close_scenes(opened_scenes)

    ds.FlushCache()
    log(f"  Wrote {output_path} in {time.time() - run_start:.0f}s")
    return output_path


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
        resolved = _resolve_scenes(scenes_4b_month, sensors.BAND_PROFILES[4], log=log)
        if resolved:
            out_path = os.path.join(output_root, "4b", f"philadelphia_{label}.tif")
            build_month_composite(
                resolved, sensors.BAND_PROFILES[4], out_path,
                block_rows=block_rows, block_cols=block_cols, log=log,
            )
        else:
            log(f"  No usable 4b scenes for {label}; skipping.")

    if scenes_8b_month:
        log(f"[8b {label}] {len(scenes_8b_month)} scene(s)")
        resolved = _resolve_scenes(scenes_8b_month, sensors.BAND_PROFILES[8], log=log)
        if resolved:
            out_path = os.path.join(output_root, "8b", f"philadelphia_{label}.tif")
            build_month_composite(
                resolved, sensors.BAND_PROFILES[8], out_path,
                block_rows=block_rows, block_cols=block_cols, log=log,
            )
        else:
            log(f"  No usable 8b scenes for {label}; skipping.")

    combined_scenes = scenes_4b_month + scenes_8b_month
    if combined_scenes:
        log(f"[combined {label}] {len(combined_scenes)} scene(s)")
        resolved = _resolve_scenes(combined_scenes, COMMON_BANDS, log=log)
        if resolved:
            out_path = os.path.join(output_root, "combined", f"philadelphia_{label}.tif")
            build_month_composite(
                resolved, COMMON_BANDS, out_path,
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
    block_rows: int = DEFAULT_BLOCK_ROWS, block_cols: int = DEFAULT_BLOCK_COLS,
    gdal_cache_mb: int = DEFAULT_GDAL_CACHE_MB, workers: int = 1, log=print,
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
    parser.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_ROWS)
    parser.add_argument("--block-cols", type=int, default=DEFAULT_BLOCK_COLS)
    parser.add_argument("--gdal-cache-mb", type=int, default=DEFAULT_GDAL_CACHE_MB,
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
