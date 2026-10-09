#!/usr/bin/env python3
"""Build a single-image composite reference (min/p25/median/p75/max/mean/
stddev, per band, plus two whole-image n_scenes/n_unmasked count rasters)
from every scene in a dataset acquired within a window around that
dataset's own summer solstice.

This exists as an *alternative* to psnorm's per-reference-scene design
(scripts/run_multi_reference_pipeline.py, which still works unchanged and
is the right choice when different source images should remain the
reference across different spatial subsets of a wide AOI). A solstice
composite instead collapses many same-season images into one synthetic
reference covering the whole domain -- useful to try as a single, less
noisy, cloud-gap-filled reference image in its own right.

Pipeline:
  1. Discover every scene (see discover_all_scenes) -- from plain
     directories of already-extracted imagery (--input, as produced by
     `files/` folders elsewhere in this project) and/or directly from zip
     archives (--input-zip / --input-zip-glob) via GDAL's `/vsizip/`
     virtual filesystem, which reads a member's header or a pixel window
     straight out of the archive with no extraction to disk at all --
     only safe to rely on when the zip stores members uncompressed
     (`ZIP_STORED`), which is true of this project's own exports; a
     deflate-compressed zip would still open, just slower, since GDAL has
     to decompress from the start of the member up to the needed offset.
  2. Compute every scene's centroid from its raster header alone (CRS +
     geotransform + size -- `io.get_raster_info`, which never calls
     `ReadAsArray`) and reproject it to lon/lat, then take the median
     lon and the median lat across every scene as the dataset's overall
     domain centroid. This is metadata-only and touches no pixel data,
     so it scales to a full dataset's worth of scenes even when, as with
     the 2024 zips, that means thousands of header reads instead of
     thousands of full image decodes.
  3. The centroid's latitude sign picks the hemisphere, which picks which
     solstice (June for the Northern hemisphere, December for the
     Southern) counts as "summer" for this domain -- see
     summer_solstice(), a low-precision (Meeus, Astronomical Algorithms
     ch.27) but perfectly adequate closed-form estimate of the solstice
     instant for any year, accurate to well under a day with no external
     ephemeris data or network access needed.
  4. Every scene acquired within --window-days (default 14, i.e. "2
     weeks") of *any* nearby year's summer solstice is selected --
     checking each scene's own year plus the adjacent two years makes the
     Southern-hemisphere December solstice's turn-of-year wraparound just
     fall out of the same logic, with no special-casing.
  5. The selected scenes are composited band-by-band onto their union
     grid (same-grid assumption as elsewhere in this project -- see
     io.grids_aligned; no resampling), in small 2D row/column tiles whose
     size is independent of how many scenes happen to be selected: unlike
     a running-sum composite (e.g. scripts/temporal_stddev_composite.py),
     percentiles and the median aren't separable into an online update
     rule, so each tile must briefly hold every contributing scene's
     values stacked together before collapsing them with the relevant
     numpy nan-aware reduction. Tiling keeps that stack's footprint
     bounded by (contributing scenes x bands x tile_rows x tile_cols)
     instead of the whole output grid at once.

Usage:
    .venv/bin/python scripts/build_solstice_composite_reference.py \\
        --input /mnt/e/claude_psnorm_testbed/2018/files \\
        --output-dir /mnt/e/claude_psnorm_testbed/2018/solstice_composite \\
        --prefix solstice_2018

    .venv/bin/python scripts/build_solstice_composite_reference.py \\
        --input-zip-glob '/mnt/e/claude_psnorm_testbed/2024/output_2024*_4b.zip' \\
        --output-dir /mnt/e/claude_psnorm_testbed/2024/solstice_composite \\
        --prefix solstice_2024
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
from osgeo import gdal, osr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psnorm import io, masking, sensors  # noqa: E402

gdal.UseExceptions()
osr.UseExceptions()

NODATA = -9999.0
STATS = ["min", "p25", "median", "p75", "max", "mean", "stddev"]
_RANK_QUANTILES = {"min": 0, "p25": 25, "median": 50, "p75": 75, "max": 100}
# Shared, single-band whole-image rasters (not one band per spectral band
# like STATS) -- see build_monthly_composites.py's identical n_scenes/
# n_unmasked convention, which this mirrors.
COUNT_STATS = ["n_scenes", "n_unmasked"]


# --------------------------------------------------------------------------
# Scene discovery, including directly from zip archives (no extraction)
# --------------------------------------------------------------------------

def _list_zip_scenes(zip_path: str, analytic_suffix: str, udm2_suffix: str) -> list[io.Scene]:
    """Scenes found inside `zip_path`, with `analytic_path`/`udm2_path` set
    to `/vsizip/...` paths GDAL can open directly -- `zipfile.namelist()`
    only reads the archive's central directory, so this never touches a
    member's actual (uncompressed) bytes."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    name_set = set(names)
    analytic_names = sorted(
        n for n in names if n.endswith(analytic_suffix) and not os.path.basename(n).startswith("._")
    )
    scenes = []
    for name in analytic_names:
        dirpart = os.path.dirname(name)
        scene_id = os.path.basename(name)[: -len(analytic_suffix)]
        udm2_name = f"{dirpart}/{scene_id}{udm2_suffix}" if dirpart else f"{scene_id}{udm2_suffix}"
        scenes.append(
            io.Scene(
                scene_id=scene_id,
                analytic_path=f"/vsizip/{zip_path}/{name}",
                udm2_path=f"/vsizip/{zip_path}/{udm2_name}" if udm2_name in name_set else None,
                metadata_path=None,
                acquired=io.parse_acquisition_time(scene_id),
            )
        )
    return scenes


def discover_all_scenes(
    input_dirs: list[str], zip_paths: list[str], *,
    analytic_suffix: str, udm2_suffix: str, log=print,
) -> list[io.Scene]:
    scenes: list[io.Scene] = []
    for d in input_dirs:
        found = io.discover_scenes(d, analytic_suffix=analytic_suffix, udm2_suffix=udm2_suffix)
        log(f"  '{d}': {len(found)} scenes")
        scenes.extend(found)
    for z in zip_paths:
        found = _list_zip_scenes(z, analytic_suffix, udm2_suffix)
        log(f"  '{z}': {len(found)} scenes (read via /vsizip/, nothing extracted)")
        scenes.extend(found)
    return scenes


# --------------------------------------------------------------------------
# Metadata-only domain centroid
# --------------------------------------------------------------------------

def _axis_mapping_srs(wkt_or_epsg) -> osr.SpatialReference:
    """SpatialReference with traditional (x=lon/easting, y=lat/northing)
    axis order forced, so GDAL 3's authority-compliant default (lat/lon
    for EPSG:4326) can't silently swap our coordinates."""
    srs = osr.SpatialReference()
    if isinstance(wkt_or_epsg, int):
        srs.ImportFromEPSG(wkt_or_epsg)
    else:
        srs.ImportFromWkt(wkt_or_epsg)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _scene_centroid_lonlat(path: str, wgs84_srs: osr.SpatialReference) -> tuple[float, float] | None:
    """(lon, lat) of `path`'s own footprint centroid, read from its raster
    header alone -- `io.get_raster_info` opens the dataset only to read
    its geotransform/size/CRS, never calling ReadAsArray, so no pixel data
    is loaded for this. Returns None if the header can't be read."""
    try:
        info = io.get_raster_info(path)
    except Exception:
        return None
    gt = info.geotransform
    cx = gt[0] + info.width * gt[1] / 2.0
    cy = gt[3] + info.height * gt[5] / 2.0
    ct = osr.CoordinateTransformation(_axis_mapping_srs(info.crs), wgs84_srs)
    lon, lat, _ = ct.TransformPoint(cx, cy)
    return lon, lat


def compute_domain_centroid(scenes: list[io.Scene], *, log=print) -> tuple[float, float]:
    """(median longitude, median latitude) across every scene's own
    footprint centroid -- the dataset's overall study-domain centroid,
    used only to pick a hemisphere (see summer_solstice)."""
    wgs84 = _axis_mapping_srs(4326)
    t0 = time.time()
    lons, lats = [], []
    for i, scene in enumerate(scenes, start=1):
        centroid = _scene_centroid_lonlat(scene.analytic_path, wgs84)
        if centroid is not None:
            lons.append(centroid[0])
            lats.append(centroid[1])
        if i % 500 == 0:
            log(f"  read {i}/{len(scenes)} scene headers...")
    if not lons:
        raise ValueError("Could not read a header/centroid for any scene.")
    log(f"Read {len(lons)}/{len(scenes)} scene headers in {time.time() - t0:.1f}s "
        f"(metadata only -- no imagery loaded).")
    return float(np.median(lons)), float(np.median(lats))


# --------------------------------------------------------------------------
# Summer solstice date (Meeus low-precision mean solstice/equinox formula)
# --------------------------------------------------------------------------

def _jde_to_datetime(jde: float) -> datetime:
    """Julian Ephemeris Day -> UTC calendar datetime (Meeus, Astronomical
    Algorithms ch.7's standard JD-to-Gregorian-calendar algorithm)."""
    jde = jde + 0.5
    z = math.floor(jde)
    f = jde - z
    if z < 2299161:
        a = z
    else:
        alpha = math.floor((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - math.floor(alpha / 4)
    b = a + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)
    day = b - d - math.floor(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    day_int = int(math.floor(day))
    seconds = round((day - day_int) * 86400)
    return datetime(int(year), int(month), day_int) + timedelta(seconds=seconds)


def summer_solstice(year: int, hemisphere: str) -> datetime:
    """UTC instant of the summer solstice for `hemisphere` ('N' -> the
    June solstice, 'S' -> the December solstice) in `year`, via Meeus's
    low-precision mean-solstice polynomial (Astronomical Algorithms
    ch.27, valid 1000-3000 CE). This omits the ~24-term periodic
    correction Meeus adds for sub-hour precision -- the mean value alone
    is already accurate to well under a day (checked here against the
    known June 2018/2024 solstices), which is all a two-week selection
    window needs."""
    if hemisphere not in ("N", "S"):
        raise ValueError(f"hemisphere must be 'N' or 'S', got {hemisphere!r}")
    yp = (year - 2000) / 1000.0
    if hemisphere == "N":
        jde = 2451716.56767 + 365241.62603 * yp + 0.00325 * yp**2 + 0.00888 * yp**3 - 0.00030 * yp**4
    else:
        jde = 2451900.05952 + 365242.74049 * yp - 0.06223 * yp**2 - 0.00823 * yp**3 + 0.00032 * yp**4
    return _jde_to_datetime(jde)


def select_solstice_window_scenes(
    scenes: list[io.Scene], hemisphere: str, window_days: int, *, log=print,
) -> list[io.Scene]:
    """Every scene in `scenes` acquired within `window_days` of a nearby
    summer solstice -- checking each scene's own acquisition year plus
    the year before and after covers the Southern-hemisphere December
    solstice's turn-of-year wraparound without special-casing it."""
    dated = [s for s in scenes if s.acquired is not None]
    years = sorted({s.acquired.year for s in dated})
    candidate_years = sorted({y + delta for y in years for delta in (-1, 0, 1)})
    solstices = {y: summer_solstice(y, hemisphere) for y in candidate_years}

    label = "June (Northern hemisphere)" if hemisphere == "N" else "December (Southern hemisphere)"
    log(f"Summer solstice ({label}) by year:")
    for y, dt in solstices.items():
        log(f"  {y}: {dt.isoformat()} UTC")

    window = timedelta(days=window_days)
    selected = [s for s in dated if any(abs(s.acquired - dt) <= window for dt in solstices.values())]
    log(f"Selected {len(selected)}/{len(dated)} scenes within {window_days} days of a summer solstice.")
    return selected


# --------------------------------------------------------------------------
# Compositing: per-pixel median/mean/p25/p75/stddev across selected scenes
# --------------------------------------------------------------------------

def _union_output_info(infos: list[io.RasterInfo]) -> io.RasterInfo:
    """A RasterInfo covering the union of every scene's bounds, on the
    same grid (same origin phase, same pixel size/CRS) as the first scene
    -- valid only when every scene really does share one grid, which the
    caller must have already confirmed (see io.grids_aligned)."""
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


def _masked_read(
    scene: io.Scene, info: io.RasterInfo, band_indices: list[int], window: io.Window,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(values, footprint_valid, fully_valid) for `band_indices` over
    `window` of `scene`'s own raster: values is (bands, rows, cols)
    float32; footprint_valid is (rows, cols) bool, True where every
    requested band is not nodata (the scene has real data there, cloud
    status aside); fully_valid additionally requires the UDM2 `clear` band
    (if present). See build_monthly_composites.py's identical split --
    footprint_valid feeds the n_scenes count, fully_valid feeds both the
    n_unmasked count and the reflectance statistics themselves."""
    xoff, yoff, xsize, ysize = window
    arr = io.read_window_bands(scene.analytic_path, window, band_indices=band_indices).astype(np.float32)
    footprint_valid = np.ones((ysize, xsize), dtype=bool) if info.nodata is None else np.all(arr != info.nodata, axis=0)
    if scene.udm2_path is not None:
        fully_valid = footprint_valid & masking.read_udm2_valid_mask(scene.udm2_path, window)
    else:
        fully_valid = footprint_valid
    return arr, footprint_valid, fully_valid


def _fast_nan_rank_stat(sorted_stack: np.ndarray, counts: np.ndarray, q: float) -> np.ndarray:
    """The `q`-th percentile (0-100; 50 = median) along axis 0 of
    `sorted_stack` ((n_scenes, bands, rows, cols), already np.sort()-ed
    along axis 0 so every pixel's NaNs trail its real values), using
    `counts` (NaN-free values per pixel, shared across bands since every
    band shares one valid mask per scene -- see _masked_read) to pick
    each pixel's own pair of order statistics to interpolate between.

    np.nanpercentile/np.nanmedian are not used here because they take
    ~100x longer than this on arrays this shape: profiling this script
    against real imagery showed a single np.nanpercentile call costing
    ~20s per compositing tile (dominating total runtime over everything
    else combined), against ~0.02s this way -- nanpercentile's generic
    implementation isn't vectorized well for "most of the axis is one
    size, reduce over a small axis" the way this script's (n_scenes
    small, rows x cols large) stacks are. Sorting once and reusing it for
    median/p25/p75 together (see build_solstice_composite) is both
    correct (checked against np.nanpercentile/np.nanmedian to float32
    precision on synthetic data with a realistic shared-NaN-mask-across-
    bands pattern) and the same cost as a single nanpercentile call.
    """
    idx = (np.maximum(counts, 1) - 1) * (q / 100.0)
    lower_idx = np.floor(idx).astype(np.int64)
    upper_idx = np.ceil(idx).astype(np.int64)
    frac = (idx - lower_idx).astype(np.float32)
    bshape = (1,) + sorted_stack.shape[1:]
    lower_vals = np.take_along_axis(sorted_stack, np.broadcast_to(lower_idx[None, None], bshape), axis=0)[0]
    upper_vals = np.take_along_axis(sorted_stack, np.broadcast_to(upper_idx[None, None], bshape), axis=0)[0]
    return lower_vals + (upper_vals - lower_vals) * frac[None, :, :]


def _create_output(path: str, out_info: io.RasterInfo, out_band_names: list[str]) -> "gdal.Dataset":
    """`out_band_names` is one name per output band -- `band_names` (the
    source spectral bands) for a per-band statistic file, or a single
    entry (its own stat name) for a COUNT_STATS file."""
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        path, out_info.width, out_info.height, len(out_band_names), gdal.GDT_Float32,
        options=["COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
    )
    ds.SetGeoTransform(out_info.geotransform)
    ds.SetProjection(out_info.crs)
    for b, name in enumerate(out_band_names):
        ds.GetRasterBand(b + 1).SetNoDataValue(NODATA)
        ds.GetRasterBand(b + 1).SetDescription(name)
    return ds


def build_solstice_composite(
    scenes: list[io.Scene], output_paths: dict[str, str], *,
    band_names: list[str] | None = None, block_rows: int = 256, block_cols: int = 1024,
    min_observations: int = 1, log=print,
) -> dict[str, str]:
    """Composite `scenes` into one or more of STATS/COUNT_STATS, whichever
    keys are present in `output_paths`: a STATS file has one band per
    source spectral band; a COUNT_STATS file (n_scenes/n_unmasked) is a
    single shared band, since scene/cloud coverage doesn't vary by band
    (see _masked_read)."""
    unknown = set(output_paths) - set(STATS) - set(COUNT_STATS)
    if unknown:
        raise ValueError(
            f"Unknown statistic(s) {unknown!r}; expected a subset of {STATS + COUNT_STATS}"
        )
    if not output_paths:
        raise ValueError("output_paths is empty -- nothing to compute")
    if not scenes:
        raise ValueError("No scenes to composite.")

    t0 = time.time()
    scene_infos = []
    for s in scenes:
        try:
            scene_infos.append((s, io.get_raster_info(s.analytic_path)))
        except Exception as exc:
            log(f"  WARNING: could not read '{s.analytic_path}': {exc}")
    if not scene_infos:
        raise ValueError("Could not read headers for any selected scene.")
    log(f"Read {len(scene_infos)} scene headers in {time.time() - t0:.1f}s")

    if band_names is None:
        band_names = sensors.detect_band_names(scene_infos[0][0].analytic_path)
    log(f"Bands: {band_names}")
    log(f"Computing: {sorted(output_paths)}")

    ref_info = scene_infos[0][1]
    for s, info in scene_infos[1:]:
        if not io.grids_aligned(ref_info, info):
            raise ValueError(
                f"'{s.analytic_path}' is not on the same pixel grid as the reference scene -- "
                f"this script assumes every input scene shares one grid (see io.grids_aligned); "
                f"resampling/registration is not implemented here."
            )

    out_info = _union_output_info([info for _s, info in scene_infos])
    log(f"Output grid: {out_info.width}x{out_info.height} pixels "
        f"({out_info.width * out_info.height / 1e6:.1f} megapixels)")

    out_datasets = {
        key: _create_output(path, out_info, band_names if key in STATS else [key])
        for key, path in output_paths.items()
    }
    band_indices = [sensors.band_index(band_names, b) for b in band_names]
    n_bands = len(band_names)

    row_blocks = list(io.iter_row_blocks(out_info.height, block_rows))
    col_blocks = list(io.iter_row_blocks(out_info.width, block_cols))
    n_tiles = len(row_blocks) * len(col_blocks)

    run_start = time.time()
    tile_i = 0
    for row_start, n_rows in row_blocks:
        for col_start, n_cols in col_blocks:
            tile_i += 1
            tile_info = io.windowed_raster_info(out_info, (col_start, row_start, n_cols, n_rows))

            footprint_count = np.zeros((n_rows, n_cols), dtype=np.int32)
            unmasked_count = np.zeros((n_rows, n_cols), dtype=np.int32)
            contributions = []
            for scene, info in scene_infos:
                overlap = io.overlap_window(tile_info, info)
                if overlap is None:
                    continue
                tile_window, scene_window = overlap
                values, footprint_valid, fully_valid = _masked_read(scene, info, band_indices, scene_window)
                if not footprint_valid.any():
                    continue
                tx, ty, tw, th = tile_window
                footprint_count[ty : ty + th, tx : tx + tw] += footprint_valid
                unmasked_count[ty : ty + th, tx : tx + tw] += fully_valid
                if fully_valid.any():
                    buf = np.full((n_bands, n_rows, n_cols), np.nan, dtype=np.float32)
                    buf[:, ty : ty + th, tx : tx + tw] = np.where(fully_valid, values, np.nan)
                    contributions.append(buf)

            no_footprint = footprint_count == 0
            sparse = unmasked_count < min_observations

            stat_keys = set(out_datasets) & set(STATS)
            if stat_keys:
                if contributions:
                    stack = np.stack(contributions, axis=0)  # (n_scenes, bands, rows, cols)

                    # Sorted once (NaNs trail each pixel's real values) and shared by
                    # min/p25/median/p75/max via _fast_nan_rank_stat -- see its docstring
                    # for why this replaces np.nanmedian/np.nanpercentile, which are each
                    # ~1000x slower on arrays shaped like these (few scenes, many pixels).
                    needs_sort = set(_RANK_QUANTILES) & stat_keys
                    sorted_stack = np.sort(stack, axis=0) if needs_sort else None

                    with np.errstate(invalid="ignore"):
                        results = {
                            stat: _fast_nan_rank_stat(sorted_stack, unmasked_count, q)
                            for stat, q in _RANK_QUANTILES.items() if stat in stat_keys
                        }
                        if "mean" in stat_keys:
                            results["mean"] = np.nanmean(stack, axis=0)
                        if "stddev" in stat_keys:
                            results["stddev"] = np.nanstd(stack, axis=0)
                else:
                    results = {stat: np.full((n_bands, n_rows, n_cols), NODATA, dtype=np.float32) for stat in stat_keys}

                for key, arr in results.items():
                    arr = arr.astype(np.float32, copy=True)
                    arr[:, sparse] = NODATA
                    ds = out_datasets[key]
                    for b in range(n_bands):
                        ds.GetRasterBand(b + 1).WriteArray(arr[b], xoff=col_start, yoff=row_start)

            if "n_scenes" in out_datasets:
                arr = footprint_count.astype(np.float32)
                arr[no_footprint] = NODATA
                out_datasets["n_scenes"].GetRasterBand(1).WriteArray(arr, xoff=col_start, yoff=row_start)

            if "n_unmasked" in out_datasets:
                arr = unmasked_count.astype(np.float32)
                arr[no_footprint] = NODATA  # a real 0 (covered but all masked) is kept
                out_datasets["n_unmasked"].GetRasterBand(1).WriteArray(arr, xoff=col_start, yoff=row_start)

            elapsed = time.time() - run_start
            eta = elapsed / tile_i * (n_tiles - tile_i)
            log(f"  tile {tile_i}/{n_tiles}: {len(contributions)} scene(s) contributed "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    for key, ds in out_datasets.items():
        ds.FlushCache()
        out_datasets[key] = None
        log(f"Wrote {key}: {output_paths[key]}")

    log(f"Done in {time.time() - run_start:.1f}s.")
    return output_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", default=[],
                         help="directory of already-extracted scenes (repeatable)")
    parser.add_argument("--input-zip", action="append", default=[],
                         help="zip archive of scenes, read via GDAL's /vsizip/ with no "
                              "extraction to disk (repeatable)")
    parser.add_argument("--input-zip-glob", action="append", default=[],
                         help="glob pattern matching multiple zip archives (repeatable)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="solstice_composite")
    parser.add_argument("--window-days", type=int, default=14,
                         help="select scenes within this many days of the computed summer "
                              "solstice (default: 14, i.e. 2 weeks)")
    parser.add_argument("--analytic-suffix", default="_3B_AnalyticMS_SR_harmonized_clip.tif")
    parser.add_argument("--udm2-suffix", default="_3B_udm2_clip.tif")
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument("--block-cols", type=int, default=1024)
    parser.add_argument("--min-observations", type=int, default=1,
                         help="pixels seen by fewer than this many selected scenes are left nodata")
    parser.add_argument("--stats", nargs="+", default=STATS + COUNT_STATS, choices=STATS + COUNT_STATS)
    args = parser.parse_args()

    zip_paths = list(args.input_zip)
    for pattern in args.input_zip_glob:
        matched = sorted(glob.glob(pattern))
        if not matched:
            print(f"WARNING: --input-zip-glob '{pattern}' matched no files.")
        zip_paths.extend(matched)
    if not args.input and not zip_paths:
        parser.error("at least one of --input / --input-zip / --input-zip-glob is required")

    print("Discovering scenes...")
    scenes = discover_all_scenes(
        args.input, zip_paths, analytic_suffix=args.analytic_suffix, udm2_suffix=args.udm2_suffix,
    )
    print(f"Discovered {len(scenes)} scenes total.")
    if not scenes:
        raise ValueError("No scenes discovered -- nothing to do.")

    print("Computing domain centroid from scene header metadata (no imagery loaded)...")
    centroid_lon, centroid_lat = compute_domain_centroid(scenes)
    hemisphere = "N" if centroid_lat >= 0 else "S"
    print(f"Domain centroid: lon={centroid_lon:.5f}, lat={centroid_lat:.5f} "
          f"({'Northern' if hemisphere == 'N' else 'Southern'} hemisphere)")

    selected = select_solstice_window_scenes(scenes, hemisphere, args.window_days)
    if not selected:
        raise ValueError("No scenes fall within the solstice window -- nothing to composite.")

    os.makedirs(args.output_dir, exist_ok=True)
    output_paths = {stat: os.path.join(args.output_dir, f"{args.prefix}_{stat}.tif") for stat in args.stats}

    build_solstice_composite(
        selected, output_paths,
        block_rows=args.block_rows, block_cols=args.block_cols, min_observations=args.min_observations,
    )


if __name__ == "__main__":
    main()
