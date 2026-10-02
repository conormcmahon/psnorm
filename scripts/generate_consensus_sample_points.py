#!/usr/bin/env python3
"""Generate a stratified-random set of test points drawn from a psnorm run's
Phase B consensus candidates (`consensus/consensus_mask.tif`), stratified by
each pixel's own consensus `frequency` (`consensus/frequency.tif` -- the
fraction of cross-scene comparisons where that pixel was flagged invariant).

Pixels with consensus_mask == 1 are split into `--n-bins` equal-width
quantile bins of frequency (default 20 bins == 5% steps), and `--per-bin`
points (default 100) are drawn uniformly at random from each bin -- so the
resulting point set has roughly equal representation from low-consensus-
frequency pixels (right at the acceptance threshold) and high-consensus-
frequency pixels (flagged invariant almost every time), rather than being
dominated by whichever frequency level happens to be most common.

Output is a GeoPackage with the same schema `sample_test_points.py` expects
(WGS84 point geometry + a text 'class' field), so it drops straight into the
existing sample_test_points.py / plot_test_points.R pipeline. 'class' is set
to the bin's frequency range (e.g. "freq_75-80pct"), which is what
plot_test_points.R facets on. The consensus frequency value itself is also
stored as an extra field (informational only -- ignored by
sample_test_points.py) for anyone inspecting the GeoPackage directly.

Accepts *multiple* `--run-dir`s (one per reference group, e.g. a
multi-reference run's `ref_<scene_id>/` subfolders, each with its own
`consensus/consensus_mask.tif` + `consensus/frequency.tif` on that group's
own reference grid) -- consensus-candidate pixels from every given run_dir
are pooled into one distribution *before* the 20 quantile bins are computed,
so the bins reflect the frequency spread across the whole multi-reference
run rather than being computed separately (and inconsistently) per group.
Points are then drawn from whichever group(s) actually populate each bin --
a group whose candidates cluster at low frequency will naturally contribute
more of that bin's points, rather than every group contributing an equal
share regardless of its own distribution. An extra 'run_dir' field on each
point records which group it came from.

Usage:
    .venv/bin/python scripts/generate_consensus_sample_points.py \\
        --run-dir /path/to/psnorm_output \\
        --output /path/to/consensus_sample_points.gpkg \\
        --n-bins 20 --per-bin 100 --seed 0

    # multi-reference: pool candidates across every reference group
    .venv/bin/python scripts/generate_consensus_sample_points.py \\
        --run-dir /path/to/psnorm_output/ref_AAAA /path/to/psnorm_output/ref_BBBB \\
        --output /path/to/consensus_sample_points.gpkg
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from osgeo import gdal, ogr, osr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psnorm import io

gdal.UseExceptions()
ogr.UseExceptions()


def generate_points(
    run_dir: str | list[str], output_path: str, *, n_bins: int = 20, per_bin: int = 100, seed: int = 0, log=print,
) -> str:
    run_dirs = [run_dir] if isinstance(run_dir, str) else list(run_dir)

    # Pool every run_dir's consensus-candidate pixels into one flat table
    # (group label, row, col, frequency) before computing quantile bins --
    # each group keeps its own geotransform (read once, cached by group
    # label) since different reference groups sit on different grids.
    group_labels, group_rows, group_cols, group_freqs = [], [], [], []
    infos: dict[str, io.RasterInfo] = {}
    for rd in run_dirs:
        label = os.path.basename(os.path.normpath(rd))
        mask_path = os.path.join(rd, "consensus", "consensus_mask.tif")
        freq_path = os.path.join(rd, "consensus", "frequency.tif")

        info = io.get_raster_info(mask_path)
        infos[label] = info
        mask_ds = gdal.Open(mask_path)
        mask = mask_ds.GetRasterBand(1).ReadAsArray()
        freq_ds = gdal.Open(freq_path)
        freq = freq_ds.GetRasterBand(1).ReadAsArray()
        mask_ds = None
        freq_ds = None

        rows, cols = np.nonzero(mask != 0)
        log(f"{label}: {len(rows)} consensus-candidate pixels "
            f"(frequency range {freq[rows, cols].min():.4f}-{freq[rows, cols].max():.4f}).")
        group_labels.append(np.full(len(rows), label))
        group_rows.append(rows)
        group_cols.append(cols)
        group_freqs.append(freq[rows, cols])

    labels = np.concatenate(group_labels)
    rows = np.concatenate(group_rows)
    cols = np.concatenate(group_cols)
    freq_values = np.concatenate(group_freqs)
    log(f"Pooled across {len(run_dirs)} run_dir(s): {len(labels)} consensus-candidate pixels total "
        f"(frequency range {freq_values.min():.4f}-{freq_values.max():.4f}).")

    edges = np.quantile(freq_values, np.linspace(0.0, 1.0, n_bins + 1))
    # right=True with these edges would drop the minimum value's own bin
    # membership at the lower boundary of bin 0, so clamp it explicitly.
    bin_indices = np.digitize(freq_values, edges[1:-1], right=False)

    rng = np.random.default_rng(seed)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transforms: dict[str, osr.CoordinateTransformation] = {}
    for label, info in infos.items():
        src = osr.SpatialReference()
        src.ImportFromWkt(info.crs)
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transforms[label] = osr.CoordinateTransformation(src, dst)

    selected = []
    for b in range(n_bins):
        candidate_idx = np.nonzero(bin_indices == b)[0]
        lo_pct, hi_pct = round(100 * b / n_bins), round(100 * (b + 1) / n_bins)
        label_name = f"freq_{lo_pct:02d}-{hi_pct:02d}pct"
        n_take = min(per_bin, len(candidate_idx))
        if n_take < per_bin:
            log(f"  WARNING: bin '{label_name}' only has {len(candidate_idx)} candidate pixels "
                f"(< requested {per_bin}); taking all of them.")
        chosen = rng.choice(candidate_idx, size=n_take, replace=False)
        by_group = {}
        for idx in chosen:
            group = str(labels[idx])
            row, col = rows[idx], cols[idx]
            gt = infos[group].geotransform
            # Pixel-center coordinates, matching the convention _pixel_offset
            # in sample_test_points.py inverts (col/row from an x/y that was
            # itself derived this way lands back on the same pixel).
            x = gt[0] + (col + 0.5) * gt[1]
            y = gt[3] + (row + 0.5) * gt[5]
            lon, lat, _ = transforms[group].TransformPoint(x, y)
            selected.append({
                "class": label_name,
                "frequency": float(freq_values[idx]),
                "run_dir": group,
                "lon": lon,
                "lat": lat,
            })
            by_group[group] = by_group.get(group, 0) + 1
        log(f"  bin '{label_name}' (frequency [{edges[b]:.4f}, {edges[b+1]:.4f}]): "
            f"selected {n_take} of {len(candidate_idx)} candidates {by_group}.")

    if os.path.exists(output_path):
        os.remove(output_path)
    driver = ogr.GetDriverByName("GPKG")
    ds = driver.CreateDataSource(output_path)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer("points", srs=srs, geom_type=ogr.wkbPoint)
    layer.CreateField(ogr.FieldDefn("class", ogr.OFTString))
    layer.CreateField(ogr.FieldDefn("frequency", ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn("run_dir", ogr.OFTString))

    for p in selected:
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("class", p["class"])
        feat.SetField("frequency", p["frequency"])
        feat.SetField("run_dir", p["run_dir"])
        geom = ogr.Geometry(ogr.wkbPoint)
        geom.AddPoint(p["lon"], p["lat"])
        feat.SetGeometry(geom)
        layer.CreateFeature(feat)
        feat = None
    ds = None

    log(f"Wrote {len(selected)} points ({n_bins} bins x up to {per_bin} each) across "
        f"{len(run_dirs)} run_dir(s) to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="a psnorm_output directory (must contain consensus/)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--per-bin", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate_points(args.run_dir, args.output, n_bins=args.n_bins, per_bin=args.per_bin, seed=args.seed)


if __name__ == "__main__":
    main()
