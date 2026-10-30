#!/usr/bin/env python3
"""Sample per-tree reflectance from the raw and registered PlanetScope scenes.

For every tree point in --points (a table with treeid, lon, lat in WGS84)
and every raw scene acquired in the chosen years and months, writes one CSV
row per (tree, scene, band) where the tree falls on the scene's data (not on
its nodata margin), with:

  - the reflectance at the tree's pixel in the raw scene and in the affine-
    and TPS-registered scenes (surface reflectance = stored value / 10000;
    the nearest pixel, no interpolation; empty where the pixel is nodata or
    the registered scene does not exist);
  - selected_reflectance: which of those versions later steps (including
    the radiometric normalization) use for the scene -- "affine" where its
    affine registration was accepted, else "original" -- from
    <affine-dir>/preferred_imagery.csv (or --preferred-imagery);
  - the UDM2 `clear` flag of the raw scene at that pixel (1 = clear);
  - the image-wide agreement of that band with the reference composite
    before and after each registration, as planet_georeg_opencv recorded it
    (alignment_metrics CSVs): R^2 and RMSE over the RANSAC-inlier pixels and
    over all sampled pixels. Only the bands used for matching have these;
  - the guided RANSAC inlier count of each registration, its status, and the
    fitted affine transform (scale, rotation, shear and the shift at the
    scene centre) from the affine manifest.

With --normalization-dir (the --output of
scripts/run_composite_reference_pipeline.py), each row also gets that
scene's radiometric normalization fit for the band, from
<composite>/models/<scene_id>_model.json (columns starting norm_; empty where
the scene has no model yet, e.g. its composite's group has not run):

  - norm_composite: the composite the scene was normalized against;
  - norm_consensus_pixels_unmasked: consensus invariant-target pixels inside
    the scene's data footprint, before the scene's own cloud, water and
    vegetation masks;
  - norm_consensus_pixels_fit: consensus pixels left after those masks, i.e.
    what the regression started from (the model's n_consensus_pixels);
  - norm_ransac_inliers: the band's RANSAC inliers (pixels the final fit
    used);
  - norm_slope, norm_intercept: the correction applied (intercept in DN, i.e.
    reflectance x 10000); identity (1, 0) where the fit was rejected;
  - norm_r2, norm_rmse_before, norm_rmse_after: the final fit's R^2, and the
    target-vs-reference RMSE (DN) at its inliers before and after the fitted
    correction;
  - norm_identity_fallback, norm_fallback_reason: whether the fit was
    rejected for the identity transform, and why.

Point reads are one pixel at a time with a small GDAL block cache, so memory
stays well under 1 GB however many scenes are sampled. --shard K/N samples
every N-th scene starting at the K-th (1-based), so N processes can split a
run; concatenate their outputs (one header) afterwards.

Usage:
    python scripts/sample_tree_points.py \\
        --points /mnt/e/E_Coast_PlanetScope/philadelphia/tree_point_samples.gpkg \\
        --years 2018,2024 --months 6,7 \\
        [--normalization-dir /mnt/e/E_Coast_PlanetScope/philadelphia/invariant_targets_composite_reference] \\
        --output /mnt/e/E_Coast_PlanetScope/philadelphia/tree_point_samples.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("GDAL_CACHEMAX", "32")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from osgeo import gdal, ogr  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.warp import transform  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from psnorm import masking, model_io, sensors  # noqa: E402

gdal.UseExceptions()
ogr.UseExceptions()

REFLECTANCE_SCALE = 10000.0  # PlanetScope surface reflectance is stored x 10000
MODELS = ("affine", "tps")

COLUMNS = [
    "treeid", "date", "acquired", "scene_id", "product", "band", "band_index",
    "reflectance_original", "reflectance_affine", "reflectance_tps", "selected_reflectance", "udm2_clear",
    "r2_original_inliers", "rmse_original_inliers", "r2_original_all", "rmse_original_all",
    "r2_affine_inliers", "rmse_affine_inliers", "r2_affine_all", "rmse_affine_all",
    "r2_tps_inliers", "rmse_tps_inliers", "r2_tps_all", "rmse_tps_all",
    "ransac_inliers_affine", "ransac_inliers_tps", "status_affine", "status_tps",
    "affine_scale_x", "affine_scale_y", "affine_rotation_deg", "affine_shear_deg",
    "affine_offset_x_px", "affine_offset_y_px", "affine_offset_x_m", "affine_offset_y_m",
    "pixel_col", "pixel_row",
]
NORM_COLUMNS = [
    "norm_composite", "norm_consensus_pixels_unmasked", "norm_consensus_pixels_fit", "norm_ransac_inliers",
    "norm_slope", "norm_intercept", "norm_r2", "norm_rmse_before", "norm_rmse_after",
    "norm_identity_fallback", "norm_fallback_reason",
]


def read_assignment(normalization_dir: str) -> dict:
    """{scene_id: composite name} from a composite-reference run."""
    with open(os.path.join(normalization_dir, "composite_assignment.csv"), newline="") as f:
        return {row["scene_id"]: row["composite"] for row in csv.DictReader(f)}


def normalization_fields(normalization_dir: str, composite: str | None, scene_id: str) -> dict:
    """{band name: {norm_ column: value}} for one scene's normalization fit,
    or {} where the scene has no model (yet)."""
    if composite is None:
        return {}
    group = os.path.join(normalization_dir, composite)
    model_path = os.path.join(group, "models", f"{scene_id}_model.json")
    if not model_io.model_is_valid(model_path):
        return {}
    model = model_io.load_model(model_path)

    unmasked = ""
    stats_path = os.path.join(group, "candidates", f"{scene_id}_stats.json")
    consensus_path = os.path.join(group, "consensus", "consensus_mask.tif")
    flags_path = os.path.join(group, "masks", f"{scene_id}_flags.tif")
    if all(os.path.exists(p) for p in (stats_path, consensus_path, flags_path)):
        stats = model_io.load_json(stats_path)
        consensus_window = masking.load_flags_window(consensus_path, tuple(stats["ref_window"])) != 0
        has_data = (masking.load_flags_window(flags_path, tuple(stats["tgt_window"])) & masking.EXCLUDE_NODATA) == 0
        unmasked = int((consensus_window & has_data).sum())

    def number(value):
        return "" if value is None or (isinstance(value, float) and math.isnan(value)) else value

    fields = {}
    for band in model.bands:
        fields[band.band_name] = {
            "norm_composite": composite,
            "norm_consensus_pixels_unmasked": unmasked,
            "norm_consensus_pixels_fit": model.n_consensus_pixels,
            "norm_ransac_inliers": band.n_invariant_pixels,
            "norm_slope": number(band.slope), "norm_intercept": number(band.intercept),
            "norm_r2": number(band.r2),
            "norm_rmse_before": number(band.rmse_before), "norm_rmse_after": number(band.rmse_after),
            "norm_identity_fallback": int(bool(band.identity_fallback)),
            "norm_fallback_reason": band.fallback_reason or "",
        }
    return fields


def read_points(path: str, id_field: str = "treeid", extra_fields=()) -> list[tuple]:
    """(id, lon, lat, {extra field: value}) for every feature of the first
    layer: lon/lat from the layer's lon and lat fields where it has them,
    else from the (WGS84) point geometry."""
    dataset = ogr.Open(path)
    layer = dataset.GetLayer(0)
    defn = layer.GetLayerDefn()
    names = {defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())}
    points = []
    for f in layer:
        if {"lon", "lat"} <= names:
            lon, lat = float(f["lon"]), float(f["lat"])
        else:
            geom = f.GetGeometryRef()
            lon, lat = geom.GetX(), geom.GetY()
        fid = f[id_field] if id_field in names else f.GetFID()
        points.append((str(fid), lon, lat, {name: f[name] for name in extra_fields}))
    dataset = None
    return points


def read_metrics(output_dir: str) -> dict:
    """{moving file name: metrics row} from a run's alignment_metrics CSVs."""
    metrics = {}
    metrics_dir = os.path.join(output_dir, "alignment_metrics")
    for name in sorted(os.listdir(metrics_dir)) if os.path.isdir(metrics_dir) else []:
        if name.startswith("registration_metrics_") and name.endswith("band.csv"):
            with open(os.path.join(metrics_dir, name), newline="") as f:
                for row in csv.DictReader(f):
                    metrics[row["moving_filename"]] = row
    return metrics


def read_selected_reflectance(path: str) -> dict:
    """{scene_id: "original" or "affine"}: which version of each scene later
    steps use, from a preferred_imagery.csv (see psnorm.preferred_imagery;
    "registered" there is the affine-registered scene)."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {row["scene_id"]: "affine" if row["preferred_source"] == "registered" else "original"
                for row in csv.DictReader(f)}


def read_manifest(output_dir: str) -> dict:
    path = os.path.join(output_dir, "registration_manifest.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {row["scene_id"]: row for row in csv.DictReader(f)}


def pixel_values(dataset, col: int, row: int) -> list:
    """Every band's value at one pixel, None where it equals the nodata value."""
    values = dataset.ReadAsArray(col, row, 1, 1)
    values = values.reshape(-1) if values.ndim > 1 else values.reshape(-1)
    out = []
    for band_index, value in enumerate(values, start=1):
        nodata = dataset.GetRasterBand(band_index).GetNoDataValue()
        out.append(None if nodata is not None and value == nodata else value.item())
    return out


# Above this many points in a scene, each band is read once as the window
# spanning them all instead of one pixel at a time: with many points spread
# over the scene, nearly every block is read anyway, and one read per band is
# far fewer round trips on the slow mounts.
WINDOW_READ_MIN_POINTS = 100


def points_values(dataset, cols, rows) -> list[list]:
    """[pixel_values(dataset, col, row) for each point], read pixel by pixel
    or, for many points, one window per band (see WINDOW_READ_MIN_POINTS)."""
    if len(cols) < WINDOW_READ_MIN_POINTS:
        return [pixel_values(dataset, c, r) for c, r in zip(cols, rows)]
    cols, rows = np.asarray(cols), np.asarray(rows)
    c0, r0 = int(cols.min()), int(rows.min())
    width, height = int(cols.max()) - c0 + 1, int(rows.max()) - r0 + 1
    per_band = []
    for b in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(b)
        values = band.ReadAsArray(c0, r0, width, height)[rows - r0, cols - c0]
        nodata = band.GetNoDataValue()
        per_band.append([None if nodata is not None and v == nodata else v.item() for v in values])
    return [list(v) for v in zip(*per_band)]


# The raster mounts intermittently fail opens and reads with "Bad file
# descriptor" (stale cached entries); clearing the kernel caches
# (PSNORM_IO_RECOVERY_CMD, as in psnorm's pipeline) and retrying clears it.
IO_RETRIES = 4
IO_RETRY_DELAY_S = 30
TRANSIENT_IO_ERRORS = ("Bad file descriptor", "No such file or directory", "Input/output error")


def with_io_retry(fn, *args):
    """fn(*args), retried (after the recovery command and a pause) when it
    fails with a transient I/O error."""
    import subprocess
    for attempt in range(IO_RETRIES + 1):
        try:
            return fn(*args)
        except (RuntimeError, OSError) as exc:
            if attempt == IO_RETRIES or not any(t in str(exc) for t in TRANSIENT_IO_ERRORS):
                raise
            print(f"I/O error ({exc}); retrying in {IO_RETRY_DELAY_S} s", flush=True)
            cmd = os.environ.get("PSNORM_IO_RECOVERY_CMD")
            if cmd:
                subprocess.run(cmd, shell=True)
            time.sleep(IO_RETRY_DELAY_S)


def points_values_at(path: str, cols, rows) -> list[list]:
    """points_values on the raster at `path`, opened afresh on every retry."""
    return with_io_retry(lambda: points_values(gdal.Open(path), cols, rows))


def scale(value):
    return "" if value is None else round(value / REFLECTANCE_SCALE, 6)


def output_columns(id_field: str = "treeid", extra_fields=(), normalization: bool = False) -> list[str]:
    """COLUMNS with the point id column named id_field and extra_fields after
    it, plus NORM_COLUMNS when normalization fields are sampled."""
    return [id_field, *extra_fields, *COLUMNS[1:], *(NORM_COLUMNS if normalization else [])]


def run_in_parallel(argv: list[str], workers: int, output: str, tmp_dir: str | None) -> None:
    """Run this script as `workers` shard processes (--shard k/workers) and
    concatenate their outputs, in shard order, into `output`."""
    import subprocess

    parts_dir = tempfile.mkdtemp(dir=tmp_dir)
    parts = [os.path.join(parts_dir, f"part{k}.csv") for k in range(1, workers + 1)]
    procs = [subprocess.Popen([sys.executable, "-W", "ignore", os.path.abspath(__file__), *argv,
                               "--shard", f"{k}/{workers}", "--output", part]
                              + (["--tmp-dir", tmp_dir] if tmp_dir else []))
             for k, part in enumerate(parts, start=1)]
    failed = [k for k, proc in enumerate(procs, start=1) if proc.wait() != 0]
    if failed:
        raise SystemExit(f"shard(s) {failed} failed; partial outputs left in {parts_dir}")
    n_rows = 0
    with open(output, "w", newline="") as out:
        for k, part in enumerate(parts):
            with open(part, newline="") as f:
                header = f.readline()
                if k == 0:
                    out.write(header)
                for line in f:
                    out.write(line)
                    n_rows += 1
            os.remove(part)
    os.rmdir(parts_dir)
    print(f"DONE: {n_rows} rows written to {output} ({workers} shards)", flush=True)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", required=True)
    parser.add_argument("--id-field", default="treeid",
                        help="point id field, written as the first column (default treeid; the feature id where "
                             "the layer has no such field)")
    parser.add_argument("--extra-fields", default="",
                        help="comma-separated point fields copied into columns after the id (e.g. class,frequency)")
    parser.add_argument("--task-cache", default="/home/conor/run_logs/register_tasks.json",
                        help="scene list written by register_to_solstice_composites.py")
    parser.add_argument("--affine-dir", default="/mnt/e/E_Coast_PlanetScope/philadelphia/affine_transform")
    parser.add_argument("--tps-dir", default="/mnt/e/E_Coast_PlanetScope/philadelphia/tps_transform")
    parser.add_argument("--years", default=None,
                        help="comma-separated acquisition years (default 2018,2024; every year with --composite)")
    parser.add_argument("--months", default=None,
                        help="comma-separated acquisition months (default 6,7; every month with --composite)")
    parser.add_argument("--normalization-dir", default=None,
                        help="output of run_composite_reference_pipeline.py: add each scene's normalization fit "
                             "(norm_ columns)")
    parser.add_argument("--composite", default=None,
                        help="only the scenes assigned to this composite (e.g. solstice_2019) in "
                             "--normalization-dir's composite_assignment.csv")
    parser.add_argument("--group-dir", action="append", default=[], metavar="COMPOSITE=FOLDER",
                        help="read the fits of the scenes assigned to COMPOSITE from --normalization-dir/FOLDER "
                             "instead of /COMPOSITE (e.g. solstice_2023=solstice_2023_vs_2024, the 2023 scenes "
                             "normalized against the 2024 composite); norm_composite then gives FOLDER. Repeatable.")
    parser.add_argument("--preferred-imagery", default=None,
                        help="preferred_imagery.csv giving the selected_reflectance column (default: "
                             "<affine-dir>/preferred_imagery.csv)")
    parser.add_argument("--shard", default=None, help="K/N: only every N-th scene, starting at the K-th (1-based)")
    parser.add_argument("--workers", type=int, default=1,
                        help="split the scenes over this many processes (shards) and concatenate their outputs")
    parser.add_argument("--tmp-dir", default=None,
                        help="where the output is written before being copied to --output (default: the system "
                             "temporary directory)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.composite and not args.normalization_dir:
        parser.error("--composite needs --normalization-dir")
    if any("=" not in g for g in args.group_dir):
        parser.error("--group-dir takes COMPOSITE=FOLDER")
    group_dirs = dict(g.split("=", 1) for g in args.group_dir)

    if args.workers > 1 and not args.shard:
        raw = list(sys.argv[1:] if argv is None else argv)
        passthrough, skip = [], False
        for i, a in enumerate(raw):
            if skip:
                skip = False
                continue
            if a in ("--workers", "--output", "--tmp-dir"):
                skip = True
                continue
            if a.startswith(("--workers=", "--output=", "--tmp-dir=")):
                continue
            passthrough.append(a)
        run_in_parallel(passthrough, args.workers, args.output, args.tmp_dir)
        return

    extra_fields = [f.strip() for f in args.extra_fields.split(",") if f.strip()]
    years = args.years or ("" if args.composite else "2018,2024")
    months = args.months or ("" if args.composite else "6,7")
    years = {y.strip() for y in years.split(",") if y.strip()}
    months = {int(m) for m in months.split(",") if m.strip()}
    points = read_points(args.points, args.id_field, extra_fields)
    assignment = read_assignment(args.normalization_dir) if args.normalization_dir else {}
    with open(args.task_cache) as f:
        tasks = [t for t in json.load(f)
                 if (not years or t["acquired"][:4] in years) and (not months or int(t["acquired"][5:7]) in months)
                 and (not args.composite or assignment.get(t["scene_id"]) == args.composite)]
    tasks.sort(key=lambda t: t["acquired"])
    if args.shard:
        k, n = (int(v) for v in args.shard.split("/"))
        tasks = tasks[k - 1::n]
    columns = output_columns(args.id_field, extra_fields, bool(args.normalization_dir))
    dirs = {"affine": args.affine_dir, "tps": args.tps_dir}
    manifests = {m: read_manifest(d) for m, d in dirs.items()}
    selected = read_selected_reflectance(args.preferred_imagery or os.path.join(args.affine_dir, "preferred_imagery.csv"))
    metrics = {m: read_metrics(d) for m, d in dirs.items()}
    print(f"{len(points)} points, {len(tasks)} scenes", flush=True)

    projected = {}  # scene CRS WKT -> [(point index, x, y)]

    # Written locally, then copied to the output path (renames are unreliable
    # on the output mount).
    tmp_dir = tempfile.mkdtemp(dir=args.tmp_dir)
    tmp_path = os.path.join(tmp_dir, os.path.basename(args.output))
    n_rows = 0
    start = time.time()
    with open(tmp_path, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for i, task in enumerate(tasks, start=1):
            original = with_io_retry(gdal.Open, task["analytic_path"])
            wkt = original.GetProjection()
            if wkt not in projected:
                # rasterio's own PROJ data: this environment's GDAL lacks it.
                xs, ys = transform(CRS.from_epsg(4326), CRS.from_wkt(wkt),
                                   [p[1] for p in points], [p[2] for p in points])
                projected[wkt] = list(zip(range(len(points)), xs, ys))
            gt = original.GetGeoTransform()
            inside = []
            for k, x, y in projected[wkt]:
                col, row = int((x - gt[0]) // gt[1]), int((y - gt[3]) // gt[5])
                if 0 <= col < original.RasterXSize and 0 <= row < original.RasterYSize:
                    inside.append((k, col, row))
            if not inside:
                original = None
                continue

            band_names = sensors.detect_band_names(task["analytic_path"])
            registered = {}
            for model in MODELS:
                path = os.path.join(dirs[model], task["output_relpath"])
                registered[model] = path if os.path.exists(path) else None
            udm2 = task["udm2_path"] if task.get("udm2_path") and os.path.exists(task["udm2_path"]) else None
            filename = os.path.basename(task["analytic_path"])
            rows = {m: manifests[m].get(task["scene_id"], {}) for m in MODELS}
            mets = {m: metrics[m].get(filename, {}) for m in MODELS}
            affine = rows["affine"]
            pixel_m = abs(gt[1])

            def offset_m(key):
                value = affine.get(key, "")
                return round(float(value) * pixel_m, 4) if value not in ("", None) else ""

            scene_fields = {
                "date": task["acquired"][:10], "acquired": task["acquired"], "scene_id": task["scene_id"],
                "product": task["product"],
                "selected_reflectance": selected.get(task["scene_id"], ""),
                "ransac_inliers_affine": rows["affine"].get("guided_ransac_inliers", ""),
                "ransac_inliers_tps": rows["tps"].get("guided_ransac_inliers", ""),
                "status_affine": rows["affine"].get("status", "not yet run"),
                "status_tps": rows["tps"].get("status", "not yet run"),
                "affine_scale_x": affine.get("affine_scale_x", ""), "affine_scale_y": affine.get("affine_scale_y", ""),
                "affine_rotation_deg": affine.get("affine_rotation_deg", ""),
                "affine_shear_deg": affine.get("affine_shear_deg", ""),
                "affine_offset_x_px": affine.get("affine_shift_x_px", ""),
                "affine_offset_y_px": affine.get("affine_shift_y_px", ""),
                "affine_offset_x_m": offset_m("affine_shift_x_px"),
                "affine_offset_y_m": offset_m("affine_shift_y_px"),
            }
            composite = assignment.get(task["scene_id"])
            norm = (normalization_fields(args.normalization_dir, group_dirs.get(composite, composite),
                                         task["scene_id"])
                    if args.normalization_dir else {})
            cols = [c for _, c, _ in inside]
            prows = [r for _, _, r in inside]
            values = {"original": points_values_at(task["analytic_path"], cols, prows)}
            for model in MODELS:
                values[model] = points_values_at(registered[model], cols, prows) if registered[model] else None
            clear = [v[0] for v in points_values_at(udm2, cols, prows)] if udm2 else [""] * len(inside)
            # Metrics are keyed by the 0-based moving band index; the
            # "before" values are the same for both models.
            before = mets["affine"] or mets["tps"]
            band_metrics = []
            for b in range(len(band_names)):
                m = {}
                for subset in ("inliers", "all"):
                    m[f"r2_original_{subset}"] = before.get(f"r2_band{b}_before_{subset}", "")
                    m[f"rmse_original_{subset}"] = before.get(f"rmse_band{b}_before_{subset}", "")
                    for model in MODELS:
                        m[f"r2_{model}_{subset}"] = mets[model].get(f"r2_band{b}_after_{subset}", "")
                        m[f"rmse_{model}_{subset}"] = mets[model].get(f"rmse_band{b}_after_{subset}", "")
                band_metrics.append(m)
            for j, (k, col, row) in enumerate(inside):
                if all(v is None for v in values["original"][j]):
                    # Inside the raster's rectangle but outside the scene's
                    # data (its clipped footprint): no observation of this point.
                    continue
                pid, _, _, extra = points[k]
                for b, name in enumerate(band_names):
                    record = {
                        args.id_field: pid, **extra, **scene_fields, "band": name, "band_index": b + 1,
                        "reflectance_original": scale(values["original"][j][b]),
                        "reflectance_affine": scale(values["affine"][j][b]) if values["affine"] else "",
                        "reflectance_tps": scale(values["tps"][j][b]) if values["tps"] else "",
                        "udm2_clear": clear[j] if clear[j] is not None else "", "pixel_col": col, "pixel_row": row,
                        **band_metrics[b], **norm.get(name, {}),
                    }
                    writer.writerow(record)
                    n_rows += 1
            original = udm2 = None
            registered = {}
            if i % 20 == 0 or i == len(tasks):
                print(f"[progress] {i}/{len(tasks)} scenes, {n_rows} rows, {time.time() - start:.0f} s", flush=True)
    shutil.copy(tmp_path, args.output)
    os.remove(tmp_path)
    os.rmdir(tmp_dir)
    print(f"DONE: {n_rows} rows written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
