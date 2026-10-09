"""Shared multi-scene compositing engine: scene resolution, open-once
dataset handles, the memory-bounded 2D-tile statistics loop, and output
writing.

Used by both scripts/build_monthly_composites.py (one composite per
calendar month per sensor-product group) and
scripts/build_solstice_composite_reference.py (one composite pooling
scenes from a solstice-date window, across sensor products with different
band counts/orders) -- both need the exact same machinery, just fed a
different scene selection, so it lives here once rather than duplicated.

Key design points:

- `resolve_scenes` reads each scene's own header and band descriptions
  (never pixel data) and resolves, per scene, a `band_indices` list
  aligned to the caller's `canonical_bands` -- one entry per canonical
  band, `None` where that particular scene's product doesn't have it.
  With `require_all_bands=True` (the monthly script's single-product and
  common-band "combined" groups) a scene missing any requested band is
  dropped entirely. With `require_all_bands=False` (the solstice script's
  full 8-band pooling across a 4-band and an 8-band product) a scene
  simply contributes to whichever canonical bands it actually has.

- `open_scenes`/`close_scenes`/`masked_read_open` open each scene's
  analytic + UDM2 datasets exactly once and reuse the Band objects across
  every output tile that scene overlaps, rather than every tile
  reopening every scene from scratch.

- `build_composite`'s per-tile loop tracks two different kinds of counts:
  a per-canonical-band `band_unmasked_count` (since different scenes may
  contribute to different subsets of the canonical bands, each band's
  percentile/mean/stddev needs its own count of how many real values
  back it at each pixel) used only to pick NODATA/min_observations
  thresholds and percentile order-statistic positions; and a per-`group`
  (e.g. "4b"/"8b", or a single implicit group) `footprint_count`/
  `unmasked_count`, written out as the `n_scenes`/`n_unmasked` band pair
  -- unsuffixed if every resolved scene shares one group, or suffixed
  `n_scenes_<group>`/`n_unmasked_<group>` per distinct group otherwise.

- Percentiles/min/max all derive from one `np.sort` per tile plus
  `fast_nan_rank_stat` (q=0/100 fall out as min/max for free), since
  np.nanpercentile/np.nanmedian are ~1000x slower on arrays shaped like
  these (few contributing scenes, many pixels) -- see that function's
  docstring for the profiling rationale.
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, replace

import numpy as np
from osgeo import gdal, gdal_array

from . import io, masking, sensors

gdal.UseExceptions()

NODATA = -9999.0
RANK_QUANTILES = {"min": 0, "p25": 25, "median": 50, "p75": 75, "max": 100}
EXTRA_STATS = ["mean", "stddev"]
ALL_STATS = list(RANK_QUANTILES) + EXTRA_STATS

DEFAULT_BLOCK_ROWS = 512
DEFAULT_BLOCK_COLS = 2048
DEFAULT_GDAL_CACHE_MB = 512


# --------------------------------------------------------------------------
# Raw-archive scene discovery (one call per sensor product; callers group
# or filter the flat result by whatever date logic they need -- calendar
# month, a solstice-date window, ...)
# --------------------------------------------------------------------------

def discover_product_scenes(
    raw_root: str, sensor_subdir: str, analytic_suffix: str, udm2_suffix: str, *, log=print,
) -> list[io.Scene]:
    """Every scene under `raw_root`/`sensor_subdir`/`sensor_subdir`_*/ --
    this project's raw-archive layout, one folder per band-count product,
    each containing one subfolder per roughly-monthly day-of-year period
    -- flattened into a single list."""
    sensor_dir = os.path.join(raw_root, sensor_subdir)
    period_dirs = sorted(
        d for d in glob.glob(os.path.join(sensor_dir, f"{sensor_subdir}_*")) if os.path.isdir(d)
    )
    scenes: list[io.Scene] = []
    for d in period_dirs:
        scenes.extend(io.discover_scenes(d, analytic_suffix=analytic_suffix, udm2_suffix=udm2_suffix))
    log(f"  '{sensor_dir}': {len(scenes)} scene(s) across {len(period_dirs)} period folder(s)")
    return scenes


# --------------------------------------------------------------------------
# Scene resolution: header-only metadata + per-scene band-index mapping
# --------------------------------------------------------------------------

@dataclass
class ResolvedScene:
    scene: io.Scene
    info: io.RasterInfo
    band_indices: list[int | None]  # one per canonical_bands entry; None = this scene lacks it
    group: str | None  # e.g. "4b"/"8b" for per-group n_scenes/n_unmasked naming


def resolve_scenes(
    scenes_with_groups: list[tuple[io.Scene, str | None]],
    canonical_bands: list[str],
    *, require_all_bands: bool = True, log=print,
) -> list[ResolvedScene]:
    """(ResolvedScene) for each (scene, group) whose header can be read and
    whose bands satisfy `require_all_bands`, and which shares a pixel grid
    with the first resolved scene. Scenes that fail any check are logged
    and skipped rather than aborting the whole batch -- a single bad or
    off-grid file in a run spanning years of imagery shouldn't kill it."""
    resolved: list[ResolvedScene] = []
    ref_info: io.RasterInfo | None = None
    for s, group in scenes_with_groups:
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
        indices = [sensors.band_index(detected, name) for name in canonical_bands]
        if require_all_bands and any(i is None for i in indices):
            log(f"    WARNING: '{s.analytic_path}' (bands {detected}) is missing one of "
                f"{canonical_bands}; skipping.")
            continue
        if not any(i is not None for i in indices):
            log(f"    WARNING: '{s.analytic_path}' (bands {detected}) has none of "
                f"{canonical_bands}; skipping.")
            continue
        if ref_info is None:
            ref_info = info
        elif not io.grids_aligned(ref_info, info):
            log(f"    WARNING: '{s.analytic_path}' is not on the same pixel grid as "
                f"'{resolved[0].scene.analytic_path}'; skipping.")
            continue
        resolved.append(ResolvedScene(s, info, indices, group))
    return resolved


# --------------------------------------------------------------------------
# Open-once-per-scene dataset handles, reused across every tile
# --------------------------------------------------------------------------

@dataclass
class OpenScene:
    resolved: ResolvedScene
    sr_dataset: "gdal.Dataset"
    sr_bands: list  # gdal.Band or None, aligned with resolved.band_indices
    udm2_dataset: "gdal.Dataset | None"
    udm2_band: "gdal.Band | None"


def open_scenes(resolved_scenes: list[ResolvedScene], *, log=print) -> list[OpenScene]:
    opened = []
    for r in resolved_scenes:
        try:
            sr_dataset = gdal.Open(r.scene.analytic_path, gdal.GA_ReadOnly)
            sr_bands = [sr_dataset.GetRasterBand(i) if i is not None else None for i in r.band_indices]
            udm2_dataset = None
            udm2_band = None
            if r.scene.udm2_path is not None:
                udm2_dataset = gdal.Open(r.scene.udm2_path, gdal.GA_ReadOnly)
                udm2_band = udm2_dataset.GetRasterBand(1)
        except Exception as exc:
            log(f"    WARNING: could not open '{r.scene.analytic_path}': {exc}; skipping.")
            continue
        opened.append(OpenScene(r, sr_dataset, sr_bands, udm2_dataset, udm2_band))
    return opened


def close_scenes(opened_scenes: list[OpenScene]) -> None:
    for o in opened_scenes:
        o.sr_bands = []
        o.sr_dataset = None
        o.udm2_band = None
        o.udm2_dataset = None


def masked_read_open(opened: OpenScene, window: io.Window) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(values, footprint_valid, fully_valid) for `window`, read from
    `opened`'s already-open Band objects. values is (n_canonical_bands,
    rows, cols) float32, with NaN in any band position this scene's
    product doesn't provide at all (resolved.band_indices[i] is None).
    footprint_valid/fully_valid are (rows, cols) bool, computed only from
    whichever bands the scene does provide -- a scene's nodata footprint
    is uniform across its own bands, so this is well-defined even when
    only a subset of the canonical bands are present."""
    xoff, yoff, xsize, ysize = window
    n_bands = len(opened.sr_bands)
    values = np.full((n_bands, ysize, xsize), np.nan, dtype=np.float32)
    present_arrays = []
    for i, band in enumerate(opened.sr_bands):
        if band is None:
            continue
        arr = band.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float32)
        values[i] = arr
        present_arrays.append(arr)
    if not present_arrays:
        footprint_valid = np.zeros((ysize, xsize), dtype=bool)
    elif opened.resolved.info.nodata is None:
        footprint_valid = np.ones((ysize, xsize), dtype=bool)
    else:
        footprint_valid = np.all(np.stack(present_arrays, axis=0) != opened.resolved.info.nodata, axis=0)
    if opened.udm2_band is not None:
        clear = opened.udm2_band.ReadAsArray(xoff, yoff, xsize, ysize) == 1
        fully_valid = footprint_valid & clear
    else:
        fully_valid = footprint_valid
    return values, footprint_valid, fully_valid


# --------------------------------------------------------------------------
# Union output grid + fast rank statistic
# --------------------------------------------------------------------------

def union_output_info(infos: list[io.RasterInfo]) -> io.RasterInfo:
    """A RasterInfo covering the union of every scene's bounds, on the same
    grid (same origin phase, same pixel size/CRS) as the first scene --
    valid only when every scene shares one grid (see io.grids_aligned,
    already enforced by resolve_scenes)."""
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


def fast_nan_rank_stat(sorted_stack: np.ndarray, counts: np.ndarray, q: float) -> np.ndarray:
    """The `q`-th percentile (0-100; 50 = median, 0/100 = min/max) along
    axis 0 of `sorted_stack` ((n_scenes, bands, rows, cols), already
    np.sort()-ed along axis 0 so every pixel's NaNs trail its real
    values), using `counts` ((bands, rows, cols) -- valid values per band
    per pixel, which can differ across bands when different scenes/
    sensors contribute to different subsets of the canonical bands -- see
    masked_read_open) to pick each (band, pixel)'s own pair of order
    statistics to interpolate between.

    np.nanpercentile/np.nanmedian are not used here because they take
    ~1000x longer than this on arrays this shape: profiling this engine
    against real imagery showed a single np.nanpercentile call costing
    ~20s per compositing tile (dominating total runtime over everything
    else combined), against ~0.02s this way -- nanpercentile's generic
    implementation isn't vectorized well for "most of the axis is one
    size, reduce over a small axis" the way these (n_scenes small, bands x
    rows x cols large) stacks are. Sorting once and reusing it for every
    requested rank statistic is both correct (checked against
    np.nanpercentile/np.nanmedian to float32 precision on synthetic data
    with a realistic per-band NaN-count pattern) and the same cost as a
    single nanpercentile call.
    """
    idx = (np.maximum(counts, 1) - 1) * (q / 100.0)  # (bands, rows, cols)
    lower_idx = np.floor(idx).astype(np.int64)
    upper_idx = np.ceil(idx).astype(np.int64)
    frac = (idx - lower_idx).astype(np.float32)
    lower_vals = np.take_along_axis(sorted_stack, lower_idx[None], axis=0)[0]
    upper_vals = np.take_along_axis(sorted_stack, upper_idx[None], axis=0)[0]
    return lower_vals + (upper_vals - lower_vals) * frac


def create_output(path: str, out_info: io.RasterInfo, out_band_names: list[str]) -> "gdal.Dataset":
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


# --------------------------------------------------------------------------
# The composite itself: memory-bounded 2D-tile loop
# --------------------------------------------------------------------------

def build_composite(
    resolved_scenes: list[ResolvedScene],
    canonical_bands: list[str],
    stats: list[str],
    output_path: str,
    *, all_groups: list[str | None] | None = None,
    block_rows: int = DEFAULT_BLOCK_ROWS, block_cols: int = DEFAULT_BLOCK_COLS,
    min_observations: int = 1, log=print,
) -> str:
    """Composite `resolved_scenes` into one multi-band GeoTIFF at
    `output_path`: per canonical band, the requested `stats`
    (min/p25/median/p75/max/mean/stddev, whichever are present in
    `stats`) across every scene's UDM2-clear, non-nodata pixels at each
    location -- plus n_scenes/n_unmasked count band(s), one pair per
    group in `all_groups` (unsuffixed if there's only one group overall).

    `all_groups` fixes the *set* of groups (and therefore the output band
    layout) independently of which groups actually show up in
    `resolved_scenes` for this particular call -- e.g. the solstice
    script always passes `["4b", "8b"]` so every year's composite has the
    same 60-band layout even in a year whose solstice window only
    happened to select scenes from one product (the other product's
    count bands are then all-NODATA, consistent with how an absent
    product's exclusive canonical bands are also all-NODATA). Defaults to
    the distinct groups actually present among `resolved_scenes` --
    build_monthly_composites.py relies on this default (it always passes
    `group=None` uniformly, so there's only ever one implicit group).

    Memory-bounded by processing the output grid in 2D tiles, since
    percentiles need every contributing scene's value stacked together
    per tile -- unlike a running-sum composite, there's no incremental
    update rule for a percentile. Every scene is opened exactly once (see
    open_scenes) and its Band objects are reused across every tile."""
    if not resolved_scenes:
        raise ValueError("No resolved scenes to composite.")
    unknown = set(stats) - set(ALL_STATS)
    if unknown:
        raise ValueError(f"Unknown statistic(s) {unknown!r}; expected a subset of {ALL_STATS}")

    out_info = union_output_info([r.info for r in resolved_scenes])
    n_bands = len(canonical_bands)

    if all_groups is not None:
        groups = all_groups
        unknown_groups = {r.group for r in resolved_scenes} - set(groups)
        if unknown_groups:
            raise ValueError(f"resolved_scenes contain group(s) {unknown_groups!r} not in all_groups={groups!r}")
    else:
        groups = sorted({r.group for r in resolved_scenes}, key=lambda g: (g is None, g))
    single_group = len(groups) <= 1

    stat_order = [s for s in ALL_STATS if s in stats]
    out_band_names = [f"{b}_{s}" for b in canonical_bands for s in stat_order]
    if single_group:
        out_band_names += ["n_scenes", "n_unmasked"]
    else:
        for g in groups:
            out_band_names += [f"n_scenes_{g}", f"n_unmasked_{g}"]

    log(f"  Output grid: {out_info.width}x{out_info.height} px "
        f"({out_info.width * out_info.height / 1e6:.1f} Mpx), {len(out_band_names)} bands, "
        f"{len(resolved_scenes)} scene(s)" + ("" if single_group else f", groups={groups}"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds = create_output(output_path, out_info, out_band_names)

    opened_scenes = open_scenes(resolved_scenes, log=log)
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

                footprint_count = {g: np.zeros((n_rows, n_cols), dtype=np.int32) for g in groups}
                unmasked_count = {g: np.zeros((n_rows, n_cols), dtype=np.int32) for g in groups}
                band_unmasked_count = np.zeros((n_bands, n_rows, n_cols), dtype=np.int32)
                value_contributions = []

                for opened in opened_scenes:
                    overlap = io.overlap_window(tile_info, opened.resolved.info)
                    if overlap is None:
                        continue
                    tile_window, scene_window = overlap
                    values, footprint_valid, fully_valid = masked_read_open(opened, scene_window)
                    if not footprint_valid.any():
                        continue
                    tx, ty, tw, th = tile_window
                    g = opened.resolved.group
                    footprint_count[g][ty : ty + th, tx : tx + tw] += footprint_valid
                    unmasked_count[g][ty : ty + th, tx : tx + tw] += fully_valid
                    if fully_valid.any():
                        present = [i is not None for i in opened.resolved.band_indices]
                        for b in range(n_bands):
                            if present[b]:
                                band_unmasked_count[b, ty : ty + th, tx : tx + tw] += fully_valid
                        buf = np.full((n_bands, n_rows, n_cols), np.nan, dtype=np.float32)
                        buf[:, ty : ty + th, tx : tx + tw] = np.where(fully_valid[None, :, :], values, np.nan)
                        value_contributions.append(buf)

                sparse = band_unmasked_count < min_observations

                if value_contributions:
                    stack = np.stack(value_contributions, axis=0)  # (n_scenes, bands, rows, cols)
                    needs_sort = set(RANK_QUANTILES) & set(stats)
                    sorted_stack = np.sort(stack, axis=0) if needs_sort else None
                    with np.errstate(invalid="ignore"):
                        results = {
                            stat: fast_nan_rank_stat(sorted_stack, band_unmasked_count, q)
                            for stat, q in RANK_QUANTILES.items() if stat in stats
                        }
                        if "mean" in stats:
                            results["mean"] = np.nanmean(stack, axis=0)
                        if "stddev" in stats:
                            results["stddev"] = np.nanstd(stack, axis=0)
                else:
                    results = {stat: np.full((n_bands, n_rows, n_cols), NODATA, dtype=np.float32) for stat in stats}

                # Every band's 2D array is assembled into one (n_out_bands,
                # rows, cols) buffer and written in a single
                # DatasetWriteArray call below, rather than one
                # WriteArray-per-band call -- writing band-by-band to a
                # PIXEL-interleaved GeoTIFF (GDAL's default, used here)
                # makes GDAL read back and rewrite the whole interleaved
                # block on every single-band write, since the block isn't
                # complete until every band has been written; that cost
                # compounds with band count (measured ~3s/tile at 22
                # bands, 33-89s/tile at 42, 117s+/tile at 60 -- clearly
                # superlinear). One multi-band write gives GDAL the whole
                # block at once, avoiding the readback entirely (measured
                # ~3s/tile regardless of band count once done this way).
                out_tile = np.empty((len(out_band_names), n_rows, n_cols), dtype=np.float32)
                band_cursor = 0
                for bi in range(n_bands):
                    for stat in stat_order:
                        arr2d = results[stat][bi].astype(np.float32, copy=True)
                        arr2d[sparse[bi]] = NODATA
                        out_tile[band_cursor] = arr2d
                        band_cursor += 1

                for g in groups:
                    no_footprint = footprint_count[g] == 0
                    arr_ns = footprint_count[g].astype(np.float32)
                    arr_ns[no_footprint] = NODATA
                    out_tile[band_cursor] = arr_ns
                    band_cursor += 1

                    arr_nu = unmasked_count[g].astype(np.float32)
                    arr_nu[no_footprint] = NODATA  # a real 0 (covered but all masked) is kept
                    out_tile[band_cursor] = arr_nu
                    band_cursor += 1

                gdal_array.DatasetWriteArray(ds, out_tile, xoff=col_start, yoff=row_start)

                elapsed = time.time() - run_start
                eta = elapsed / tile_i * (n_tiles - tile_i)
                log(f"    tile {tile_i}/{n_tiles}: {len(value_contributions)} scene(s) contributed "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
    finally:
        close_scenes(opened_scenes)

    ds.FlushCache()
    log(f"  Wrote {output_path} in {time.time() - run_start:.0f}s")
    return output_path
