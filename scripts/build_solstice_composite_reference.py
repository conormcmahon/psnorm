#!/usr/bin/env python3
"""Build one composite reference (min/p25/median/p75/max/mean/stddev, per
band, plus per-sensor-product n_scenes/n_unmasked count bands) per solstice
year, from every scene in the raw archive acquired within a window around
that year's summer solstice -- pooling scenes from *both* the 4-band and
8-band products within each year's own window.

This exists as an *alternative* to psnorm's per-reference-scene design
(scripts/run_multi_reference_pipeline.py, which still works unchanged and
is the right choice when different source images should remain the
reference across different spatial subsets of a wide AOI). A solstice
composite instead collapses one season's worth of same-season images into
one synthetic reference covering the whole domain -- useful to try as a
single, less noisy, cloud-gap-filled reference image in its own right.

Pipeline:
  1. Discover every scene under --raw-root's philadelphia4b/philadelphia8b
     subfolders (see psnorm.compositing.discover_product_scenes) -- the
     same raw-archive layout scripts/build_monthly_composites.py reads,
     one folder per band-count product, each containing one subfolder per
     roughly-monthly day-of-year period.
  2. Compute every scene's centroid from its raster header alone (CRS +
     geotransform + size -- io.get_raster_info, which never calls
     ReadAsArray) and reproject it to lon/lat, then take the median lon
     and median lat across every scene as the dataset's overall domain
     centroid. This is metadata-only, so it scales to the full archive.
  3. The centroid's latitude sign picks the hemisphere, which picks which
     solstice (June for the Northern hemisphere, December for the
     Southern) counts as "summer" for this domain -- see
     summer_solstice(), a low-precision (Meeus, Astronomical Algorithms
     ch.27) but perfectly adequate closed-form estimate of the solstice
     instant for any year, accurate to well under a day with no external
     ephemeris data or network access needed.
  4. Every scene (4-band or 8-band) is assigned to whichever nearby
     year's summer solstice it falls within --window-days of, tagged with
     its product ("4b"/"8b") -- see group_by_solstice_year. Each solstice
     year becomes its own independent composite (one output file), NOT
     pooled together across years: checking each scene's own year plus
     the adjacent two years covers the Southern-hemisphere December
     solstice's turn-of-year wraparound (scenes acquired in early January
     belonging to the *previous* year's December solstice) without
     special-casing it, while still keeping each year's composite
     separate. In this archive the two products don't overlap in time,
     so any given year's composite will in practice draw from only one
     of them, but the pooling logic is general.
  5. Each year's output always estimates all 8 canonical bands
     (coastal_blue, blue, green_i, green, yellow, red, rededge, nir), even
     though only the 4-band product's 4 common bands (blue, green, red,
     nir) ever get contributions from both products -- the other 4 bands
     get contributions only from 8-band scenes (see
     psnorm.compositing.resolve_scenes' require_all_bands=False and
     masked_read_open). Because scene counts can genuinely differ by
     product, n_scenes/n_unmasked are written per product
     (n_scenes_4b/n_scenes_8b/n_unmasked_4b/n_unmasked_8b) instead of one
     pooled pair.

The actual compositing (grid union, open-once scene handles, the 2D-tile
statistics loop, output writing) lives in psnorm/compositing.py, shared
with scripts/build_monthly_composites.py -- see that module's docstring
for the engine's design and the profiling behind the rank-statistic trick.

--workers parallelizes across solstice years (each year's composite is a
fully independent unit of work, exactly like build_monthly_composites.py's
--workers parallelizing across months).

Usage:
    .venv/bin/python scripts/build_solstice_composite_reference.py \\
        --raw-root /mnt/e/E_Coast_PlanetScope/philadelphia/raw \\
        --output-dir /mnt/e/E_Coast_PlanetScope/philadelphia/solstice_composite \\
        --prefix solstice --window-days 14 --workers 4
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
from osgeo import gdal, osr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psnorm import compositing, io, sensors  # noqa: E402

gdal.UseExceptions()
osr.UseExceptions()

FOUR_BAND_SUFFIX = "_3B_AnalyticMS_SR_harmonized_clip.tif"
EIGHT_BAND_SUFFIX = "_3B_AnalyticMS_SR_8b_harmonized_clip.tif"
UDM2_SUFFIX = "_3B_udm2_clip.tif"
CANONICAL_BANDS = sensors.BAND_PROFILES[8]

# A quarter of compositing.py's own default tile area (half block_rows,
# half block_cols), not that default -- a solstice composite pools scenes
# across a whole --window-days span (and, under --workers, several of
# those run concurrently), so it routinely holds far more contributing
# scenes per tile than build_monthly_composites.py's one-calendar-month
# scope ever does. The production run across this archive's 9 solstice
# years (--workers 3 at the full-size default) pushed memory to
# 9.3GB/11GB with only three concurrent composites; this quarter-size
# tile kept the same run at a comfortable ~5GB throughout.
DEFAULT_BLOCK_ROWS = 256
DEFAULT_BLOCK_COLS = 1024


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


def group_by_solstice_year(
    tagged_scenes: list[tuple[io.Scene, str]], hemisphere: str, window_days: int, *, log=print,
) -> dict[int, list[tuple[io.Scene, str]]]:
    """{solstice_year: [(scene, product_tag), ...]} -- one independent
    composite's worth of scenes per nearby solstice year, NOT pooled
    across years (unlike an early version of this script): each
    `solstice_year` gets its own composite, exactly as
    build_monthly_composites.py builds one composite per calendar month
    rather than one pooled across every month. `solstice_year`'s own
    window is `summer_solstice(solstice_year, hemisphere) +-
    window_days`; candidate years run one before/after the years actually
    present in `tagged_scenes` so a Southern-hemisphere December
    solstice's turn-of-year wraparound (e.g. scenes acquired in early
    January belonging to the *previous* year's December solstice) is
    assigned to the right year without special-casing -- a scene can only
    ever fall within one year's narrow window (windows are a few weeks
    wide, consecutive solstices are ~365 days apart), so there's no
    double-counting between adjacent candidate years."""
    dated = [(s, tag) for s, tag in tagged_scenes if s.acquired is not None]
    years = sorted({s.acquired.year for s, _ in dated})
    candidate_years = sorted({y + delta for y in years for delta in (-1, 0, 1)})
    solstices = {y: summer_solstice(y, hemisphere) for y in candidate_years}

    label = "June (Northern hemisphere)" if hemisphere == "N" else "December (Southern hemisphere)"
    log(f"Summer solstice ({label}) by year:")
    for y, dt in solstices.items():
        log(f"  {y}: {dt.isoformat()} UTC")

    window = timedelta(days=window_days)
    by_year: dict[int, list[tuple[io.Scene, str]]] = {}
    for y, dt in solstices.items():
        matches = [(s, tag) for s, tag in dated if abs(s.acquired - dt) <= window]
        if matches:
            by_year[y] = matches

    total_selected = sum(len(v) for v in by_year.values())
    log(f"Selected {total_selected}/{len(dated)} scenes across {len(by_year)} solstice year(s) "
        f"(within {window_days} days of that year's solstice).")
    return by_year


def _process_one_year(
    year: int, selected: list[tuple[io.Scene, str]],
    output_dir: str, prefix: str, stats: list[str],
    block_rows: int, block_cols: int, gdal_cache_mb: int, min_observations: int,
) -> str:
    """One composite for one solstice year. Self-contained (takes only
    picklable Scene/tag tuples, not any shared discovery state) so it can
    run standalone in a worker process under --workers > 1, as well as
    in-process for the sequential path -- mirrors
    build_monthly_composites.py's _process_one_month."""
    gdal.UseExceptions()
    gdal.SetCacheMax(gdal_cache_mb * 1024 * 1024)

    log_prefix = f"[pid {os.getpid()}] " if multiprocessing.current_process().name != "MainProcess" else ""

    def log(msg: str) -> None:
        print(f"{log_prefix}{msg}", flush=True)

    n_4b = sum(1 for _s, tag in selected if tag == "4b")
    n_8b = sum(1 for _s, tag in selected if tag == "8b")
    log(f"[{year}] {len(selected)} scene(s) ({n_4b} 4-band, {n_8b} 8-band)")

    resolved = compositing.resolve_scenes(selected, CANONICAL_BANDS, require_all_bands=False, log=log)
    if not resolved:
        log(f"  No usable scenes for {year}; skipping.")
        return str(year)

    output_path = os.path.join(output_dir, f"{prefix}_{year}.tif")
    compositing.build_composite(
        resolved, CANONICAL_BANDS, stats, output_path,
        all_groups=["4b", "8b"],  # fixed band layout even in a year with only one product
        block_rows=block_rows, block_cols=block_cols, min_observations=min_observations, log=log,
    )
    return str(year)


def _process_one_year_star(args: tuple) -> str:
    """multiprocessing.Pool.imap_unordered needs a single-argument callable;
    this just unpacks the tuple _process_one_year otherwise takes as
    separate arguments."""
    return _process_one_year(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", default="/mnt/e/E_Coast_PlanetScope/philadelphia/raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="solstice_composite")
    parser.add_argument("--window-days", type=int, default=14,
                         help="select scenes within this many days of the computed summer "
                              "solstice (default: 14, i.e. 2 weeks)")
    parser.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_ROWS)
    parser.add_argument("--block-cols", type=int, default=DEFAULT_BLOCK_COLS)
    parser.add_argument("--gdal-cache-mb", type=int, default=compositing.DEFAULT_GDAL_CACHE_MB,
                         help="GDAL block cache size per process, in MB (default: %(default)s)")
    parser.add_argument("--min-observations", type=int, default=1,
                         help="(band, pixel) combinations seen by fewer than this many "
                              "selected scenes are left nodata for that band")
    parser.add_argument("--stats", nargs="+", default=compositing.ALL_STATS, choices=compositing.ALL_STATS)
    parser.add_argument("--years", nargs="+", type=int, default=None,
                         help="restrict to specific solstice year(s) (default: every year found)")
    parser.add_argument("--workers", type=int, default=1,
                         help="process this many solstice years in parallel (default: 1, sequential)")
    args = parser.parse_args()

    gdal.SetCacheMax(args.gdal_cache_mb * 1024 * 1024)

    print("Discovering 4-band scenes...")
    scenes_4b = compositing.discover_product_scenes(args.raw_root, "philadelphia4b", FOUR_BAND_SUFFIX, UDM2_SUFFIX)
    print("Discovering 8-band scenes...")
    scenes_8b = compositing.discover_product_scenes(args.raw_root, "philadelphia8b", EIGHT_BAND_SUFFIX, UDM2_SUFFIX)

    tagged = [(s, "4b") for s in scenes_4b] + [(s, "8b") for s in scenes_8b]
    print(f"Discovered {len(tagged)} scenes total.")
    if not tagged:
        raise ValueError("No scenes discovered -- nothing to do.")

    print("Computing domain centroid from scene header metadata (no imagery loaded)...")
    centroid_lon, centroid_lat = compute_domain_centroid([s for s, _tag in tagged])
    hemisphere = "N" if centroid_lat >= 0 else "S"
    print(f"Domain centroid: lon={centroid_lon:.5f}, lat={centroid_lat:.5f} "
          f"({'Northern' if hemisphere == 'N' else 'Southern'} hemisphere)")

    by_year = group_by_solstice_year(tagged, hemisphere, args.window_days)
    if not by_year:
        raise ValueError("No scenes fall within any solstice window -- nothing to composite.")

    years = sorted(by_year)
    if args.years is not None:
        years = [y for y in years if y in set(args.years)]
    print(f"{len(years)} solstice year(s) to process with {args.workers} worker(s).")

    os.makedirs(args.output_dir, exist_ok=True)
    jobs = [
        (year, by_year[year], args.output_dir, args.prefix, args.stats,
         args.block_rows, args.block_cols, args.gdal_cache_mb, args.min_observations)
        for year in years
    ]

    if args.workers <= 1:
        for job in jobs:
            _process_one_year_star(job)
        return

    # "fork" (the Linux default) is used explicitly rather than relying on
    # the platform default: discovery above only reads headers/paths (no
    # open GDAL datasets to duplicate across the fork), and every job's
    # arguments are plain picklable dataclasses/strings/ints either way, so
    # this is safe and avoids re-importing/re-running module-level code the
    # "spawn" start method would require -- see
    # build_monthly_composites.py's identical choice.
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=args.workers) as pool:
        for label in pool.imap_unordered(_process_one_year_star, jobs):
            print(f"=== finished year {label} ===", flush=True)


if __name__ == "__main__":
    main()
