#!/usr/bin/env python3
"""Build one AOI-wide composite consensus mask from a multi-reference
psnorm run's per-reference consensus masks.

A wide-AOI multi-reference run (see scripts/run_multi_reference_pipeline.py)
writes one `ref_<id>/consensus/consensus_mask.tif` per reference group, each
on its own window of the shared pixel grid -- neighboring reference
footprints routinely overlap. This script OR-combines all of them onto one
grid covering their union extent: a pixel is TRUE in the composite if *any*
overlapping reference's consensus mask called it a consensus invariant
target there, FALSE if at least one reference evaluated it and found it was
not, and NODATA only where no reference ever evaluated it at all.

A reference's own consensus_mask.tif conflates "evaluated, not invariant"
and "never evaluated" as the same 0 value -- aggregate_consensus in
psnorm/consensus.py thresholds frequency only where times_evaluated > 0 and
leaves every other pixel (including ones entirely outside that reference's
useful footprint) at 0 too. That distinction matters here: one reference's
TRUE must never be cancelled out by a second, overlapping reference that
simply never evaluated that pixel. So each reference's frequency.tif (NaN
exactly where times_evaluated == 0, written alongside consensus_mask.tif by
consensus.save_consensus) is read too, to tell a real FALSE vote apart from
missing data before combining.

Usage:
    .venv/bin/python scripts/build_composite_consensus_mask.py \\
        --run-dir /path/to/psnorm_output_multiref \\
        --output /path/to/psnorm_output_multiref/composite_consensus_mask.tif
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from dataclasses import replace

import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psnorm import io

gdal.UseExceptions()

NODATA = 255  # consensus values are only ever 0 or 1, so this can't collide


def _union_output_info(infos: list[io.RasterInfo]) -> io.RasterInfo:
    """A RasterInfo covering the union of every raster's bounds, on the
    same grid (same origin phase, same pixel size/CRS) as the first --
    valid only once the caller has confirmed every raster really does
    share one grid (see io.grids_aligned)."""
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


def build_composite_consensus_mask(run_dir: str, output_path: str, *, log=print) -> str:
    """OR-combine every `ref_*/consensus/consensus_mask.tif` under
    `run_dir` into one composite GeoTIFF at `output_path` (0=FALSE,
    1=TRUE, 255=NODATA where no reference ever evaluated that pixel)."""
    mask_paths = sorted(glob.glob(os.path.join(run_dir, "ref_*", "consensus", "consensus_mask.tif")))
    if not mask_paths:
        raise ValueError(f"No 'ref_*/consensus/consensus_mask.tif' files found under '{run_dir}'.")
    log(f"Found {len(mask_paths)} reference consensus masks under '{run_dir}'.")

    infos = [io.get_raster_info(p) for p in mask_paths]
    for info in infos[1:]:
        if not io.grids_aligned(infos[0], info):
            raise ValueError(
                f"'{info.path}' is not on the same pixel grid as '{infos[0].path}' -- every "
                f"reference's consensus mask is expected to share one grid (see io.grids_aligned); "
                f"resampling is not implemented here."
            )

    out_info = _union_output_info(infos)
    log(f"Composite grid: {out_info.width}x{out_info.height} pixels "
        f"({out_info.width * out_info.height / 1e6:.1f} megapixels)")

    any_true = np.zeros((out_info.height, out_info.width), dtype=bool)
    any_valid = np.zeros((out_info.height, out_info.width), dtype=bool)

    t0 = time.time()
    for i, (path, info) in enumerate(zip(mask_paths, infos), start=1):
        ref_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
        freq_path = os.path.join(os.path.dirname(path), "frequency.tif")
        if not os.path.exists(freq_path):
            raise ValueError(f"Expected '{freq_path}' alongside '{path}' but it is missing.")

        overlap = io.overlap_window(out_info, info)
        if overlap is None:
            log(f"  [{i}/{len(mask_paths)}] WARNING: {ref_id} does not overlap the composite grid; skipping.")
            continue
        out_window, ref_window = overlap
        ox, oy, ow, oh = out_window
        rx, ry, rw, rh = ref_window

        mask_dataset, mask_bands = io.open_bands(path, band_indices=[1])
        mask_arr = mask_bands[0].ReadAsArray(rx, ry, rw, rh)
        mask_dataset = None

        freq_dataset, freq_bands = io.open_bands(freq_path, band_indices=[1])
        freq_arr = freq_bands[0].ReadAsArray(rx, ry, rw, rh)
        freq_dataset = None

        valid = ~np.isnan(freq_arr)
        true_ = valid & (mask_arr != 0)

        dst = (slice(oy, oy + oh), slice(ox, ox + ow))
        any_true[dst] |= true_
        any_valid[dst] |= valid
        log(f"  [{i}/{len(mask_paths)}] {ref_id}: {int(true_.sum())} consensus pixels, "
            f"{int(valid.sum())} evaluated pixels")

    composite = np.where(any_true, 1, np.where(any_valid, 0, NODATA)).astype(np.uint8)
    n_true = int(any_true.sum())
    n_valid = int(any_valid.sum())
    n_total = composite.size
    log(f"Composite: {n_true} TRUE, {n_valid - n_true} FALSE, {n_total - n_valid} NODATA, "
        f"out of {n_total} pixels total ({time.time() - t0:.1f}s).")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        output_path, out_info.width, out_info.height, 1, gdal.GDT_Byte,
        options=["COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
    )
    dataset.SetGeoTransform(out_info.geotransform)
    dataset.SetProjection(out_info.crs)
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(NODATA)
    band.WriteArray(composite)
    band.FlushCache()
    dataset = None
    log(f"Wrote composite consensus mask: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True,
                         help="multi-reference psnorm output dir (contains ref_* subfolders)")
    parser.add_argument("--output", required=True, help="output path for the composite GeoTIFF")
    args = parser.parse_args()
    build_composite_consensus_mask(args.run_dir, args.output)


if __name__ == "__main__":
    main()
