#!/usr/bin/env python3
"""Standalone comparison tool, entirely separate from the IR-MAD/consensus
pipeline in psnorm.pipeline: composite every scene in a folder (typically a
full year) into multi-band GeoTIFFs of per-pixel temporal statistics --
mean, standard deviation, and coefficient of variation (stddev/mean) -- of
each band's DN across every scene that covers that pixel, after masking
each scene's clouds/shadow/snow/haze (Planet's UDM2 `clear` band) and
nodata.

Low standard deviation (or coefficient of variation, which additionally
normalizes out a pixel's own brightness -- useful since darker surfaces
mechanically have smaller absolute DN swings than bright ones even at
similar relative stability) is the same underlying idea IR-MAD/consensus is
built on -- a pixel that barely changes across a year of imagery is a
plausible invariant target -- but computed directly from the raw temporal
distribution instead of pairwise change detection + cross-scene voting. This
script exists to let the two approaches be compared against each other
directly (e.g. does psnorm's IR-MAD consensus mask land on the same areas
this flags as low-variance?).

All three statistics derive from the same per-pixel sum / sum-of-squares /
count accumulators, so they're computed together in one pass over the input
scenes -- getting all three costs barely more than getting just one, whereas
running this script twice would mean scanning every scene in the input
folder twice.

Usage:
    .venv/bin/python scripts/temporal_stddev_composite.py \\
        --input /path/to/2018/files --output-dir /path/to/2018 \\
        --mean-output mean_2018.tif --stddev-output stddev_2018.tif --cv-output cv_2018.tif

Memory-safe by construction: the output grid (union extent of every input
scene) is processed in row-blocks, matching the pattern already used in
psnorm.lidar_masks for the same reason -- holding sum/sum-of-squares/count
accumulators for the whole domain across 4 bands here would need ~18GB for
a full year of PlanetScope imagery over this test AOI, well beyond a memory
budget that's already proven tight in this sandbox.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psnorm import io, masking, sensors

gdal.UseExceptions()

NODATA = -9999.0


def _union_output_info(scene_infos: list[io.RasterInfo]) -> io.RasterInfo:
    """A RasterInfo covering the union of every scene's bounds, on the same
    grid (same origin phase, same pixel size/CRS) as the first scene --
    valid only when every scene really does share one grid, which the
    caller must have already confirmed (see grids_aligned)."""
    first = scene_infos[0]
    px, py = first.geotransform[1], first.geotransform[5]
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for info in scene_infos:
        gt = info.geotransform
        x0, y1 = gt[0], gt[3]
        x1 = x0 + info.width * gt[1]
        y0 = y1 + info.height * gt[5]
        minx, maxx = min(minx, x0, x1), max(maxx, x0, x1)
        miny, maxy = min(miny, y0, y1), max(maxy, y0, y1)

    # Snap the union bounds onto the first scene's own pixel grid (integer
    # pixel offsets from its origin) so every scene composites via plain
    # pixel-offset arithmetic -- no resampling, matching io.overlap_window's
    # own assumption of already-aligned grids.
    ox, oy = first.geotransform[0], first.geotransform[3]
    x0 = ox + np.floor((minx - ox) / px) * px
    y1 = oy + np.floor((maxy - oy) / py) * py
    width = int(np.ceil((maxx - x0) / px))
    height = int(np.ceil((y1 - miny) / abs(py)))
    gt = (float(x0), px, 0.0, float(y1), 0.0, py)
    return replace(first, path="", width=width, height=height, geotransform=gt, nodata=None)


def _masked_read(
    scene: io.Scene, info: io.RasterInfo, band_indices: list[int], window: io.Window,
) -> tuple[np.ndarray, np.ndarray]:
    """(values, valid) for `band_indices` over `window` of `scene`'s own
    raster: values is (bands, rows, cols) float32, valid is (rows, cols)
    bool -- UDM2 clear (if present) AND not-nodata. `info` is the scene's
    already-read RasterInfo (see build_composite's scene_infos) -- reused
    rather than re-opened here, since this runs once per (block, scene)
    pair and re-reading a header thousands of times over adds up."""
    xoff, yoff, xsize, ysize = window
    arr = io.read_window_bands(scene.analytic_path, window, band_indices=band_indices).astype(np.float32)
    valid = np.ones((ysize, xsize), dtype=bool) if info.nodata is None else np.all(arr != info.nodata, axis=0)
    if scene.udm2_path is not None:
        valid &= masking.read_udm2_valid_mask(scene.udm2_path, window)
    return arr, valid


def _create_output(path: str, out_info: io.RasterInfo, band_names: list[str]) -> "gdal.Dataset":
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        path, out_info.width, out_info.height, len(band_names), gdal.GDT_Float32,
        options=["COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
    )
    ds.SetGeoTransform(out_info.geotransform)
    ds.SetProjection(out_info.crs)
    for b, name in enumerate(band_names):
        ds.GetRasterBand(b + 1).SetNoDataValue(NODATA)
        ds.GetRasterBand(b + 1).SetDescription(name)
    return ds


def build_composites(
    input_dir: str, output_paths: dict[str, str], *, block_rows: int = 1024, min_observations: int = 3,
    band_names: list[str] | None = None, log=print,
) -> dict[str, str]:
    """Composite every scene in `input_dir` into one or more of {"mean",
    "stddev", "cv"} (coefficient of variation = stddev/mean), whichever
    keys are present in `output_paths`. All requested statistics are
    computed together in a single pass over the input scenes.
    """
    valid_keys = {"mean", "stddev", "cv"}
    unknown = set(output_paths) - valid_keys
    if unknown:
        raise ValueError(f"Unknown statistic(s) {unknown!r}; expected a subset of {valid_keys}")
    if not output_paths:
        raise ValueError("output_paths is empty -- nothing to compute")

    scenes = io.discover_scenes(input_dir)
    if not scenes:
        raise ValueError(f"No scenes found in '{input_dir}'.")
    log(f"Discovered {len(scenes)} scenes.")

    if band_names is None:
        band_names = sensors.detect_band_names(scenes[0].analytic_path)
    log(f"Bands: {band_names}")
    log(f"Computing: {sorted(output_paths)}")

    t0 = time.time()
    scene_infos = []
    for s in scenes:
        try:
            scene_infos.append((s, io.get_raster_info(s.analytic_path)))
        except Exception as exc:
            log(f"  WARNING: could not read '{s.analytic_path}': {exc}")
    log(f"Read {len(scene_infos)} scene headers in {time.time()-t0:.1f}s")

    ref_info = scene_infos[0][1]
    for _s, info in scene_infos[1:]:
        if not io.grids_aligned(ref_info, info):
            raise ValueError(
                f"'{_s.analytic_path}' is not on the same pixel grid as the reference scene -- "
                f"this script assumes every input scene shares one grid (see io.grids_aligned); "
                f"resampling/registration is not implemented here."
            )

    out_info = _union_output_info([info for _s, info in scene_infos])
    log(f"Output grid: {out_info.width}x{out_info.height} pixels "
        f"({out_info.width * out_info.height / 1e6:.1f} megapixels)")

    out_datasets = {key: _create_output(path, out_info, band_names) for key, path in output_paths.items()}
    band_indices = [sensors.band_index(band_names, b) for b in band_names]

    blocks = list(io.iter_row_blocks(out_info.height, block_rows))
    run_start = time.time()
    for block_i, (block_start, n_rows) in enumerate(blocks):
        block_info = io.windowed_raster_info(out_info, (0, block_start, out_info.width, n_rows))
        n_bands = len(band_names)
        sums = np.zeros((n_bands, n_rows, out_info.width), dtype=np.float64)
        sumsqs = np.zeros((n_bands, n_rows, out_info.width), dtype=np.float64)
        counts = np.zeros((n_rows, out_info.width), dtype=np.int32)

        n_scenes_touched = 0
        for scene, info in scene_infos:
            overlap = io.overlap_window(block_info, info)
            if overlap is None:
                continue
            block_window, scene_window = overlap
            values, valid = _masked_read(scene, info, band_indices, scene_window)
            if not valid.any():
                continue
            n_scenes_touched += 1
            bx, by, bw, bh = block_window
            dst = (slice(None), slice(by, by + bh), slice(bx, bx + bw))
            values = np.where(valid, values, 0.0)
            sums[dst] += values
            sumsqs[dst] += values * values
            counts[by : by + bh, bx : bx + bw] += valid

        sparse = counts < min_observations
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sums / counts
            variance = np.maximum(sumsqs / counts - mean * mean, 0.0)
            stddev = np.sqrt(variance)
            cv = stddev / mean

        results = {}
        if "mean" in out_datasets:
            m = mean.astype(np.float32).copy()
            m[:, sparse] = NODATA
            results["mean"] = m
        if "stddev" in out_datasets:
            s = stddev.astype(np.float32).copy()
            s[:, sparse] = NODATA
            results["stddev"] = s
        if "cv" in out_datasets:
            c = cv.astype(np.float32).copy()
            c[:, sparse] = NODATA
            c[:, mean.min(axis=0) <= 0] = NODATA  # mean<=0 (e.g. dark/nodata-adjacent DN) makes CV meaningless
            results["cv"] = c

        for key, arr in results.items():
            ds = out_datasets[key]
            for b in range(n_bands):
                ds.GetRasterBand(b + 1).WriteArray(arr[b], xoff=0, yoff=block_start)

        elapsed = time.time() - run_start
        done = block_i + 1
        eta = elapsed / done * (len(blocks) - done)
        log(f"  block {done}/{len(blocks)}: {n_scenes_touched} scenes contributed "
            f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    for key, ds in out_datasets.items():
        ds.FlushCache()
        out_datasets[key] = None
        log(f"Wrote {key}: {output_paths[key]}")

    log(f"Done in {time.time()-run_start:.1f}s.")
    return output_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--mean-output", default=None)
    parser.add_argument("--stddev-output", default=None)
    parser.add_argument("--cv-output", default=None, help="coefficient of variation = stddev / mean")
    parser.add_argument("--block-rows", type=int, default=1024)
    parser.add_argument("--min-observations", type=int, default=3,
                         help="pixels with fewer than this many clear observations across the year are left nodata")
    args = parser.parse_args()

    output_paths = {}
    if args.mean_output:
        output_paths["mean"] = args.mean_output
    if args.stddev_output:
        output_paths["stddev"] = args.stddev_output
    if args.cv_output:
        output_paths["cv"] = args.cv_output
    if not output_paths:
        parser.error("at least one of --mean-output / --stddev-output / --cv-output is required")

    build_composites(
        args.input, output_paths, block_rows=args.block_rows, min_observations=args.min_observations,
    )


if __name__ == "__main__":
    main()
