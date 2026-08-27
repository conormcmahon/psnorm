#!/usr/bin/env python3
"""Sample point locations against every spatially-overlapping PlanetScope
scene and the corresponding psnorm pipeline outputs (exclusion flags,
normalized imagery, fitted model), producing one row per (point, image,
band) in a long-format table.

Usage:
    .venv/bin/python scripts/sample_points.py \
        --points ../test_images/example_points.gpkg \
        --input ../test_images/output_20241001_4b/files \
        --psnorm-output ../test_images/output_20241001_4b/psnorm_output_v5 \
        --output ../test_images/test_points_sampled.csv

Designed to scale to millions of points and hundreds of images:
  - Points are read once and reprojected once per distinct raster CRS
    (grouped, not once per image, and once up front for the WGS84 lon/lat
    output columns), using GDAL's batch CoordinateTransformation.
    TransformPoints rather than per-point calls.
  - Per image, pixel row/col for every point is computed with vectorized
    numpy array arithmetic (no per-point Python loop), and points outside
    that image's pixel grid are dropped by a single boolean mask.
  - Each raster (analytic/normalized/flags) is read exactly once per image
    as a single windowed ReadAsArray covering the bounding box of the
    points that actually fall inside it -- not one read per point, and not
    the whole scene when only a small area is actually needed.
  - Results are streamed to the output CSV image-by-image (append mode),
    so peak memory is bounded by one image's worth of sampled points, not
    the full points x images x bands table -- the whole point of this
    script is that that full table will not fit in memory once points and
    images both scale up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psnorm import io, masking, sensors  # noqa: E402

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

FLAG_BIT_COLUMNS = {
    "excl_nodata": masking.EXCLUDE_NODATA,
    "excl_water": masking.EXCLUDE_WATER,
    "excl_udm2": masking.EXCLUDE_UDM2,
    "excl_omnicloud": masking.EXCLUDE_OMNICLOUD,
    "excl_vegetation": masking.EXCLUDE_VEGETATION,
}

OUTPUT_COLUMNS = [
    "point_index", "longitude", "latitude", "point_class",
    "image_id", "image_date", "is_reference_scene", "band",
    "value_before", "value_after",
    "cloud_flags_raw", *FLAG_BIT_COLUMNS.keys(),
    "slope", "intercept", "r2", "rmse_before", "rmse_after",
    "identity_fallback", "fallback_reason",
]


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


def _batch_transform(x: np.ndarray, y: np.ndarray, src_srs, dst_srs) -> tuple[np.ndarray, np.ndarray]:
    ct = osr.CoordinateTransformation(src_srs, dst_srs)
    transformed = ct.TransformPoints(list(zip(x.tolist(), y.tolist())))
    tx = np.fromiter((p[0] for p in transformed), dtype=np.float64, count=len(transformed))
    ty = np.fromiter((p[1] for p in transformed), dtype=np.float64, count=len(transformed))
    return tx, ty


def load_points(points_path: str, layer_name: str | None = None):
    """Returns (fid, x, y, class_values, point_srs) straight from the
    source CRS -- no reprojection yet; that happens once for the WGS84
    output columns and once per distinct raster CRS group, not per image.
    """
    ds = ogr.Open(points_path)
    if ds is None:
        raise ValueError(f"Could not open points file: {points_path}")
    layer = ds.GetLayerByName(layer_name) if layer_name else ds.GetLayer(0)
    if layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in {points_path}")

    srs = _axis_mapping_srs(layer.GetSpatialRef().ExportToWkt())

    defn = layer.GetLayerDefn()
    class_field = next(
        (defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())
         if defn.GetFieldDefn(i).GetName().lower() == "class"),
        None,
    )

    fids, xs, ys, classes = [], [], [], []
    layer.ResetReading()
    for feat in layer:
        geom = feat.GetGeometryRef()
        fids.append(feat.GetFID())
        xs.append(geom.GetX())
        ys.append(geom.GetY())
        classes.append(feat.GetField(class_field) if class_field else None)
    ds = None

    return (
        np.asarray(fids, dtype=np.int64),
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(classes, dtype=object),
        srs,
    )


def _load_models(psnorm_output_dir: str) -> dict[str, dict]:
    """scene_id -> parsed model.json dict, for every scene that has one."""
    models_dir = os.path.join(psnorm_output_dir, "models")
    models = {}
    if not os.path.isdir(models_dir):
        return models
    for name in os.listdir(models_dir):
        if name.endswith("_model.json"):
            scene_id = name[: -len("_model.json")]
            with open(os.path.join(models_dir, name)) as f:
                models[scene_id] = json.load(f)
    return models


def _read_window(band, xoff, yoff, xsize, ysize):
    return band.ReadAsArray(int(xoff), int(yoff), int(xsize), int(ysize))


def _decode_flag_columns(flags: np.ndarray) -> dict[str, np.ndarray]:
    """`flags` may contain NaN (no flags file for that scene) -- decoded
    bit columns preserve NaN in that case rather than a bogus cast."""
    valid = ~np.isnan(flags)
    flags_int = np.where(valid, flags, 0).astype(np.int64)
    return {
        name: np.where(valid, (flags_int & bit) != 0, np.nan)
        for name, bit in FLAG_BIT_COLUMNS.items()
    }


def sample_image(
    scene: io.Scene,
    px: np.ndarray, py: np.ndarray, point_idx: np.ndarray,
    psnorm_output_dir: str,
    models: dict[str, dict],
) -> pd.DataFrame | None:
    """Sample one image for every point in (px, py) that falls inside its
    pixel grid. `point_idx` maps each (px[i], py[i]) back to a row in the
    caller's original point arrays. Returns a long-format DataFrame (one
    row per surviving point per band, `point_index` still 0-based into the
    caller's arrays) or None if nothing overlaps.
    """
    info = io.get_raster_info(scene.analytic_path)
    gt = info.geotransform

    col = np.floor((px - gt[0]) / gt[1]).astype(np.int64)
    row = np.floor((py - gt[3]) / gt[5]).astype(np.int64)
    inside = (col >= 0) & (col < info.width) & (row >= 0) & (row < info.height)
    if not inside.any():
        return None

    col, row, point_idx = col[inside], row[inside], point_idx[inside]
    col_min, col_max = int(col.min()), int(col.max())
    row_min, row_max = int(row.min()), int(row.max())
    win_w, win_h = col_max - col_min + 1, row_max - row_min + 1
    local_col, local_row = col - col_min, row - row_min

    band_names = sensors.detect_band_names(scene.analytic_path)
    n_bands = len(band_names)

    ds, bands = io.open_bands(scene.analytic_path, list(range(1, n_bands + 1)))
    before = np.stack([_read_window(b, col_min, row_min, win_w, win_h)[local_row, local_col] for b in bands], axis=1)
    ds = None

    flags_path = os.path.join(psnorm_output_dir, "masks", f"{scene.scene_id}_flags.tif")
    if os.path.exists(flags_path):
        fds, fbands = io.open_bands(flags_path, [1])
        flags = _read_window(fbands[0], col_min, row_min, win_w, win_h)[local_row, local_col].astype(np.float64)
        fds = None
    else:
        flags = np.full(local_row.shape, np.nan)

    normalized_path = os.path.join(psnorm_output_dir, "normalized", f"{scene.scene_id}_normalized.tif")
    if os.path.exists(normalized_path):
        nds, nbands = io.open_bands(normalized_path, list(range(1, n_bands + 1)))
        after = np.stack([_read_window(b, col_min, row_min, win_w, win_h)[local_row, local_col] for b in nbands], axis=1)
        nds = None
    else:
        after = np.full(before.shape, np.nan)

    model = models.get(scene.scene_id)
    is_reference = model is not None and model["reference_id"] == model["target_id"]
    band_models = {b["band_name"]: b for b in model["bands"]} if model else {}

    n_points = point_idx.size
    flags_repeated = np.repeat(flags, n_bands)
    tiled_bands = np.tile(band_names, n_points)

    rows = {
        "point_index": np.repeat(point_idx, n_bands),
        "image_id": scene.scene_id,
        "image_date": scene.acquired.isoformat() if scene.acquired else None,
        "is_reference_scene": is_reference,
        "band": tiled_bands,
        "value_before": before.ravel(),
        "value_after": after.ravel(),
        "cloud_flags_raw": flags_repeated,
        **_decode_flag_columns(flags_repeated),
    }
    for field, default in [("slope", np.nan), ("intercept", np.nan), ("r2", np.nan),
                            ("rmse_before", np.nan), ("rmse_after", np.nan),
                            ("identity_fallback", None), ("fallback_reason", None)]:
        rows[field] = [band_models.get(b, {}).get(field, default) for b in tiled_bands]

    return pd.DataFrame(rows)


def sample_all_images(
    points_path: str,
    input_folder: str,
    psnorm_output_dir: str,
    output_csv: str,
    layer_name: str | None = None,
    log=print,
) -> int:
    fid, x, y, classes, points_srs = load_points(points_path, layer_name)
    log(f"Loaded {fid.size} points from {points_path}")

    wgs84 = _axis_mapping_srs(4326)
    lon_all, lat_all = _batch_transform(x, y, points_srs, wgs84)

    scenes = io.discover_scenes(input_folder)
    if not scenes:
        raise ValueError(f"No scenes found in '{input_folder}'.")
    log(f"Discovered {len(scenes)} candidate scenes")

    models = _load_models(psnorm_output_dir)
    log(f"Loaded {len(models)} fitted models from {psnorm_output_dir}")

    # Reproject points once per distinct raster CRS (not once per image).
    transformed_by_crs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if os.path.exists(output_csv):
        os.remove(output_csv)
    header_written = False
    total_rows = 0
    point_idx_all = np.arange(fid.size)

    for i, scene in enumerate(scenes):
        info = io.get_raster_info(scene.analytic_path)
        if info.crs not in transformed_by_crs:
            target_srs = _axis_mapping_srs(info.crs)
            transformed_by_crs[info.crs] = _batch_transform(x, y, points_srs, target_srs)
        px, py = transformed_by_crs[info.crs]

        result = sample_image(scene, px, py, point_idx_all, psnorm_output_dir, models)
        if result is None:
            continue

        idx = result["point_index"].to_numpy()
        result["longitude"] = lon_all[idx]
        result["latitude"] = lat_all[idx]
        result["point_class"] = classes[idx]
        result["point_index"] = fid[idx]
        result = result[OUTPUT_COLUMNS]

        result.to_csv(output_csv, mode="a", header=not header_written, index=False)
        header_written = True
        total_rows += len(result)
        log(f"  [{i + 1}/{len(scenes)}] {scene.scene_id}: {len(result)} rows "
            f"({result['point_index'].nunique()} points)")

    if not header_written:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
    log(f"Wrote {total_rows} rows to {output_csv}")
    return total_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", required=True, help="Path to a point vector file (e.g. .gpkg)")
    parser.add_argument("--layer", default=None, help="Layer name within --points (default: first layer)")
    parser.add_argument("--input", required=True, help="Folder of source PlanetScope scenes")
    parser.add_argument("--psnorm-output", required=True, help="psnorm run output folder (masks/models/normalized)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    sample_all_images(args.points, args.input, args.psnorm_output, args.output, layer_name=args.layer)


if __name__ == "__main__":
    main()
