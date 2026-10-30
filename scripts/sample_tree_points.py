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
  - norm_r2_all, norm_rmse_before_all, norm_rmse_after_all: the same three
    over ALL the fit's consensus pixels (norm_consensus_pixels_fit, i.e.
    after the masks but before RANSAC), for the same fitted line. RANSAC keeps
    only points near a line, so these show when most consensus pixels are far
    from linear despite a good inlier fit. Taken from the model where it
    records them, else recomputed from the scene's preferred imagery and the
    group's saved masks (one extra read of the scene; --norm-quality-cache
    keeps the results so a later run reuses them);
  - norm_identity_fallback, norm_fallback_reason: whether the fit was
    rejected for the identity transform, and why.

--format parquet writes a parquet dataset instead of one CSV: a folder per
--partition-by value (e.g. genus=Acer/treeid=5468701/part-0.parquet), each
file one point's rows sorted by acquisition time, with typed columns and
year and month columns added. The samples are staged scene by scene and then
arranged into those folders (see point_samples_parquet.py); both steps
resume where they stopped when rerun.

Points are read one pixel at a time, or for many points one window per
band; memory stays at a few hundred MB per process (about 2 GB with the
full Philadelphia tree inventory and --format parquet). --shard K/N samples
every N-th scene starting at the K-th (1-based), so N processes can split a
run; concatenate their outputs (one header) afterwards.

Usage (CSV; for parquet add e.g. --format parquet --partition-by genus,treeid
--extra-fields genus --staging-dir /local/disk/staging, with --output a folder):
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
from psnorm import masking, model_io, normalize, pipeline, sensors  # noqa: E402

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
    "norm_r2_all", "norm_rmse_before_all", "norm_rmse_after_all",
    "norm_identity_fallback", "norm_fallback_reason",
]


def read_assignment(normalization_dir: str) -> dict:
    """{scene_id: composite name} from a composite-reference run."""
    with open(os.path.join(normalization_dir, "composite_assignment.csv"), newline="") as f:
        return {row["scene_id"]: row["composite"] for row in csv.DictReader(f)}


def fit_quality_all(group: str, scene_id: str, model, target_path: str | None, cache_dir: str | None) -> dict:
    """{band name: (r2_all, rmse_before_all, rmse_after_all)}: from the model
    where it records them, else from --norm-quality-cache, else recomputed
    (pipeline.consensus_fit_quality_all; cached when a cache is given). {}
    where they cannot be had (no target path, or the recomputation failed)."""
    fitted = [b for b in model.bands if b.n_invariant_pixels > 0]
    if fitted and all(not math.isnan(b.r2_all) for b in fitted):
        return {b.band_name: (b.r2_all, b.rmse_before_all, b.rmse_after_all) for b in fitted}
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, os.path.basename(os.path.normpath(group)), f"{scene_id}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    return {k: tuple(v) for k, v in json.load(f).items()}
            except (OSError, ValueError):
                pass
    if not target_path:
        return {}
    try:
        quality = with_io_retry(pipeline.consensus_fit_quality_all, group, scene_id, target_path)
    except Exception as exc:  # a scene this cannot be done for keeps blank _all columns
        print(f"{scene_id}: no all-consensus-pixel fit quality ({exc})", flush=True)
        return {}
    save_fit_quality(cache_dir, group, scene_id, quality)
    return quality


def _quality_cache_path(cache_dir: str, group: str, scene_id: str) -> str:
    return os.path.join(cache_dir, os.path.basename(os.path.normpath(group)), f"{scene_id}.json")


def cached_fit_quality(cache_dir: str | None, group: str, scene_id: str) -> dict | None:
    if not cache_dir:
        return None
    path = _quality_cache_path(cache_dir, group, scene_id)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return {k: tuple(v) for k, v in json.load(f).items()}
        except (OSError, ValueError):
            return None
    return None


def save_fit_quality(cache_dir: str | None, group: str, scene_id: str, quality: dict) -> None:
    if not cache_dir:
        return
    path = _quality_cache_path(cache_dir, group, scene_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(quality, f)
    shutil.copy(tmp, path)   # not a rename: unreliable on the virtiofs mounts
    os.remove(tmp)


class ConsensusReference:
    """A normalization group's fit pixels on the reference side, read once:
    the consensus pixels that are clear in the reference's search mask (flat
    row-major indices on the reference grid, sorted) and every reference
    band's value there. With a scene's own search mask this gives the pixels
    its fit used (see pipeline.consensus_fit_quality_all, which reads the
    same pixels scene by scene)."""

    def __init__(self, group: str, reference_id: str):
        self.reference_path = pipeline.reference_raster_path(group)
        self.band_names = sensors.detect_band_names(self.reference_path)
        consensus = gdal.Open(os.path.join(group, "consensus", "consensus_mask.tif")).ReadAsArray() != 0
        search = gdal.Open(os.path.join(group, "masks", f"{reference_id}_search_mask.tif")).ReadAsArray()
        self.width = consensus.shape[1]
        self.index = np.flatnonzero(consensus & (search == 0))
        del consensus, search
        dataset = gdal.Open(self.reference_path)
        self.values = np.empty((self.index.size, len(self.band_names)), dtype=np.float32)
        rows, cols = self.index // self.width, self.index % self.width
        for b in range(len(self.band_names)):
            band = dataset.GetRasterBand(b + 1)
            # in row blocks, to keep memory to a block
            step = 1024
            for r0 in range(0, dataset.RasterYSize, step):
                n = min(step, dataset.RasterYSize - r0)
                lo, hi = np.searchsorted(rows, [r0, r0 + n])
                if hi > lo:
                    block = band.ReadAsArray(0, r0, dataset.RasterXSize, n)
                    self.values[lo:hi, b] = block[rows[lo:hi] - r0, cols[lo:hi]]

    def scene_pixels(self, ref_window, tgt_window, target_search_path):
        """(positions into self.index/values, target cols, target rows) of
        the scene's fit pixels."""
        rx, ry, w, h = ref_window
        tx, ty = tgt_window[0], tgt_window[1]
        lo, hi = np.searchsorted(self.index, [ry * self.width, (ry + h) * self.width])
        rows, cols = self.index[lo:hi] // self.width, self.index[lo:hi] % self.width
        inside = (cols >= rx) & (cols < rx + w)
        positions = np.arange(lo, hi)[inside]
        rr, cc = rows[inside] - ry, cols[inside] - rx
        clear = masking.load_flags_window(target_search_path, tuple(tgt_window))[rr, cc] == 0
        return positions[clear], cc[clear] + tx, rr[clear] + ty


def normalization_fields(normalization_dir: str, composite: str | None, scene_id: str,
                         target_path: str | None = None, cache_dir: str | None = None,
                         quality: dict | None = None) -> dict:
    """{band name: {norm_ column: value}} for one scene's normalization fit,
    or {} where the scene has no model (yet). `target_path` (the scene's
    preferred imagery) is needed to recompute the _all columns for models
    that do not record them (see fit_quality_all), unless `quality` gives
    them already."""
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

    if quality is None:
        quality = fit_quality_all(group, scene_id, model, target_path, cache_dir)
    fields = {}
    for band in model.bands:
        r2_all, rmse_before_all, rmse_after_all = quality.get(band.band_name, (None, None, None))
        fields[band.band_name] = {
            "norm_composite": composite,
            "norm_consensus_pixels_unmasked": unmasked,
            "norm_consensus_pixels_fit": model.n_consensus_pixels,
            "norm_ransac_inliers": band.n_invariant_pixels,
            "norm_slope": number(band.slope), "norm_intercept": number(band.intercept),
            "norm_r2": number(band.r2),
            "norm_rmse_before": number(band.rmse_before), "norm_rmse_after": number(band.rmse_after),
            "norm_r2_all": number(r2_all), "norm_rmse_before_all": number(rmse_before_all),
            "norm_rmse_after_all": number(rmse_after_all),
            "norm_identity_fallback": int(bool(band.identity_fallback)),
            "norm_fallback_reason": band.fallback_reason or "",
        }
    return fields


def read_points(path: str, id_field: str = "treeid", extra_fields=()) -> list[tuple]:
    """(id, lon, lat, {extra field: value}) for every feature of the first
    layer (a GPKG, or a CSV with lon/lat columns): lon/lat from the layer's
    lon and lat fields where it has them, else from the (WGS84) point
    geometry."""
    dataset = ogr.Open(path)
    layer = dataset.GetLayer(0)
    defn = layer.GetLayerDefn()
    # Field names are matched case-insensitively (e.g. TREEID, LON, LAT in a
    # CSV), so the output columns keep the names given here.
    names = {defn.GetFieldDefn(i).GetName().lower(): defn.GetFieldDefn(i).GetName()
             for i in range(defn.GetFieldCount())}
    lon_field, lat_field = names.get("lon"), names.get("lat")
    id_name = names.get(id_field.lower())
    extra_names = {name: names.get(name.lower(), name) for name in extra_fields}
    points = []
    for f in layer:
        if lon_field and lat_field:
            lon, lat = float(f[lon_field]), float(f[lat_field])
        else:
            geom = f.GetGeometryRef()
            lon, lat = geom.GetX(), geom.GetY()
        fid = f[id_name] if id_name else f.GetFID()
        points.append((str(fid), lon, lat, {name: f[field] for name, field in extra_names.items()}))
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


def read_preferred_paths(path: str) -> dict:
    """{scene_id: the file the normalization read for it} from a
    preferred_imagery.csv."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {row["scene_id"]: row["preferred_analytic_path"] for row in csv.DictReader(f)}


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


def points_array(path: str, cols, rows, n_masked: int | None = None) -> np.ndarray:
    """Every band's value at each (col, row) as a float (n points, n bands)
    array, NaN where it equals the band's nodata value (only in the first
    `n_masked` rows, if given: the rest keep their raw values); read as in
    points_values (one window per band for many points), retrying transient
    I/O errors."""
    def read():
        dataset = gdal.Open(path)
        out = np.empty((len(cols), dataset.RasterCount), dtype=np.float64)
        if len(cols) == 0:
            return out
        c, r = np.asarray(cols), np.asarray(rows)
        if len(c) < WINDOW_READ_MIN_POINTS:
            for j, (cj, rj) in enumerate(zip(c, r)):
                out[j] = dataset.ReadAsArray(int(cj), int(rj), 1, 1).reshape(-1)
        else:
            c0, r0 = int(c.min()), int(r.min())
            width, height = int(c.max()) - c0 + 1, int(r.max()) - r0 + 1
            for b in range(dataset.RasterCount):
                out[:, b] = dataset.GetRasterBand(b + 1).ReadAsArray(c0, r0, width, height)[r - r0, c - c0]
        masked = out if n_masked is None else out[:n_masked]
        for b in range(dataset.RasterCount):
            nodata = dataset.GetRasterBand(b + 1).GetNoDataValue()
            if nodata is not None:
                masked[masked[:, b] == nodata, b] = np.nan
        return out
    return with_io_retry(read)


def _csv_value(value):
    """A float array value as the CSV writes it: "" for NaN, ints as ints."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return int(value) if float(value).is_integer() else value


class SceneSample:
    """One scene's samples: the points on its data, their pixel values in
    each version of the scene, and the scene's (and each band's) fields."""

    def __init__(self, points, point_index, cols, rows, band_names, values, clear, scene_fields, band_metrics, norm):
        self.points = points              # every point (id, lon, lat, extra)
        self.point_index = point_index    # index into points of each sampled point
        self.cols, self.rows = cols, rows
        self.band_names = band_names
        self.values = values              # {"original"|"affine"|"tps": (n, bands) array, or None}
        self.clear = clear                # (n,) UDM2 clear flag (NaN = nodata), or None
        self.scene_fields = scene_fields  # COLUMNS fields that are the same for the whole scene
        self.band_metrics = band_metrics  # per band: registration fit metrics
        self.norm = norm                  # {band name: NORM_COLUMNS fields}

    def __len__(self):
        return len(self.point_index)

    def _reflectance(self, version: str, j: int, b: int):
        array = self.values[version]
        if array is None or math.isnan(array[j, b]):
            return ""
        return round(float(array[j, b]) / REFLECTANCE_SCALE, 6)

    def csv_records(self, id_field: str):
        """The CSV rows, one per (point, band) in point then band order."""
        for j, k in enumerate(self.point_index):
            pid, _, _, extra = self.points[k]
            clear = "" if self.clear is None else _csv_value(self.clear[j])
            for b, name in enumerate(self.band_names):
                yield {
                    id_field: pid, **extra, **self.scene_fields, "band": name, "band_index": b + 1,
                    **{f"reflectance_{v}": self._reflectance(v, j, b) for v in ("original", "affine", "tps")},
                    "udm2_clear": clear, "pixel_col": int(self.cols[j]), "pixel_row": int(self.rows[j]),
                    **self.band_metrics[b], **self.norm.get(name, {}),
                }


class SceneContext:
    """What sampling a scene needs besides the scene: the points, the
    registration manifests and metrics, the preferred-imagery choice and
    the normalization run."""

    def __init__(self, args, points, assignment, group_dirs):
        self.args = args
        self.points = points
        self.lon = np.array([p[1] for p in points], dtype=np.float64)
        self.lat = np.array([p[2] for p in points], dtype=np.float64)
        self.assignment = assignment
        self.group_dirs = group_dirs
        self.dirs = {"affine": args.affine_dir, "tps": args.tps_dir}
        self.manifests = {m: read_manifest(d) for m, d in self.dirs.items()}
        preferred_csv = args.preferred_imagery or os.path.join(args.affine_dir, "preferred_imagery.csv")
        self.selected = read_selected_reflectance(preferred_csv)
        self.preferred_paths = read_preferred_paths(preferred_csv)
        self.metrics = {m: read_metrics(d) for m, d in self.dirs.items()}
        self.projected = {}  # scene CRS WKT -> (x, y) arrays of every point
        self.references = {}  # group -> ConsensusReference (the latest one only)

    def _quality_plan(self, task):
        """Where the scene's norm_*_all values are not yet known: which of the
        versions read for the points is the file its fit read, and the fit's
        pixels there, so they are read in the same pass. None where they are
        known already, or the fit read some other file (the values are then
        recomputed separately, see fit_quality_all)."""
        args = self.args
        if not args.normalization_dir:
            return None
        composite = self.assignment.get(task["scene_id"])
        if composite is None:
            return None
        group = os.path.join(args.normalization_dir, self.group_dirs.get(composite, composite))
        sid = task["scene_id"]
        model_path = os.path.join(group, "models", f"{sid}_model.json")
        if not model_io.model_is_valid(model_path):
            return None
        model = model_io.load_model(model_path)
        fitted = [b for b in model.bands if b.n_invariant_pixels > 0]
        if not fitted or all(not math.isnan(b.r2_all) for b in fitted):
            return None
        if cached_fit_quality(args.norm_quality_cache, group, sid) is not None:
            return None
        preferred = self.preferred_paths.get(sid)
        affine = os.path.join(self.dirs["affine"], task["output_relpath"])
        version = ("original" if preferred == task["analytic_path"] else
                   "affine" if preferred and os.path.normpath(preferred) == os.path.normpath(affine) else None)
        if version is None:
            return None
        try:
            if group not in self.references:
                self.references = {group: with_io_retry(ConsensusReference, group, model.reference_id)}
            reference = self.references[group]
            stats = model_io.load_json(os.path.join(group, "candidates", f"{sid}_stats.json"))
            positions, ccols, crows = with_io_retry(
                reference.scene_pixels, stats["ref_window"], stats["tgt_window"],
                os.path.join(group, "masks", f"{sid}_search_mask.tif"))
        except Exception as exc:
            print(f"{sid}: fit pixels not read with the points ({exc}); recomputing them separately", flush=True)
            return None
        if positions.size != model.n_consensus_pixels:
            print(f"{sid}: {positions.size} fit pixels found, the model has {model.n_consensus_pixels}; "
                  "recomputing them separately", flush=True)
            return None
        return {"group": group, "model": model, "reference": reference, "version": version,
                "positions": positions, "cols": ccols, "rows": crows}

    def _quality_from_plan(self, plan, target_values, target_band_names) -> dict:
        """{band: (r2_all, rmse_before_all, rmse_after_all)} from the fit
        pixels' target values (read with the points) and the reference's."""
        reference = plan["reference"]
        out = {}
        for band in plan["model"].bands:
            if band.n_invariant_pixels <= 0 or band.band_name not in reference.band_names:
                continue
            x = target_values[:, sensors.band_index(target_band_names, band.band_name) - 1]
            y = reference.values[plan["positions"], sensors.band_index(reference.band_names, band.band_name) - 1]
            slope, intercept = ((band.raw_slope, band.raw_intercept)
                                if band.identity_fallback and band.raw_slope is not None else (band.slope, band.intercept))
            out[band.band_name] = normalize.fit_quality(x.astype(np.float64), y.astype(np.float64), slope, intercept)
        return out

    def sample_scene(self, task) -> SceneSample | None:
        """The scene's samples, or None where no point is on its data."""
        args = self.args
        original = with_io_retry(gdal.Open, task["analytic_path"])
        wkt = original.GetProjection()
        if wkt not in self.projected:
            # rasterio's own PROJ data: this environment's GDAL lacks it.
            xs, ys = transform(CRS.from_epsg(4326), CRS.from_wkt(wkt), list(self.lon), list(self.lat))
            self.projected[wkt] = (np.asarray(xs), np.asarray(ys))
        xs, ys = self.projected[wkt]
        gt = original.GetGeoTransform()
        all_cols = np.floor_divide(xs - gt[0], gt[1]).astype(np.int64)
        all_rows = np.floor_divide(ys - gt[3], gt[5]).astype(np.int64)
        on = (all_cols >= 0) & (all_cols < original.RasterXSize) & (all_rows >= 0) & (all_rows < original.RasterYSize)
        original = None
        if not on.any():
            return None
        point_index = np.flatnonzero(on)
        cols, rows = all_cols[on], all_rows[on]
        plan = self._quality_plan(task)
        fit_values = None

        def read(version, path, cols, rows):
            """points_array at the points, plus the fit pixels where this is
            the version the fit read."""
            nonlocal fit_values
            if plan is None or plan["version"] != version:
                return points_array(path, cols, rows)
            # The fit pixels keep their raw values (nodata included), as the
            # fit itself read them.
            both = points_array(path, np.concatenate([cols, plan["cols"]]), np.concatenate([rows, plan["rows"]]),
                                n_masked=len(cols))
            fit_values = both[len(cols):]
            return both[:len(cols)]

        values = {"original": read("original", task["analytic_path"], cols, rows)}
        # Inside the raster's rectangle but outside the scene's data (its
        # clipped footprint): no observation of the point.
        observed = ~np.isnan(values["original"]).all(axis=1)
        if not observed.any():
            return None
        point_index, cols, rows = point_index[observed], cols[observed], rows[observed]
        values["original"] = values["original"][observed]

        band_names = sensors.detect_band_names(task["analytic_path"])
        for model in MODELS:
            path = os.path.join(self.dirs[model], task["output_relpath"])
            values[model] = read(model, path, cols, rows) if os.path.exists(path) else None
        udm2 = task.get("udm2_path")
        clear = points_array(udm2, cols, rows)[:, 0] if udm2 and os.path.exists(udm2) else None

        filename = os.path.basename(task["analytic_path"])
        manifest_rows = {m: self.manifests[m].get(task["scene_id"], {}) for m in MODELS}
        mets = {m: self.metrics[m].get(filename, {}) for m in MODELS}
        affine = manifest_rows["affine"]
        pixel_m = abs(gt[1])

        def offset_m(key):
            value = affine.get(key, "")
            return round(float(value) * pixel_m, 4) if value not in ("", None) else ""

        scene_fields = {
            "date": task["acquired"][:10], "acquired": task["acquired"], "scene_id": task["scene_id"],
            "product": task["product"],
            "selected_reflectance": self.selected.get(task["scene_id"], ""),
            "ransac_inliers_affine": manifest_rows["affine"].get("guided_ransac_inliers", ""),
            "ransac_inliers_tps": manifest_rows["tps"].get("guided_ransac_inliers", ""),
            "status_affine": manifest_rows["affine"].get("status", "not yet run"),
            "status_tps": manifest_rows["tps"].get("status", "not yet run"),
            "affine_scale_x": affine.get("affine_scale_x", ""), "affine_scale_y": affine.get("affine_scale_y", ""),
            "affine_rotation_deg": affine.get("affine_rotation_deg", ""),
            "affine_shear_deg": affine.get("affine_shear_deg", ""),
            "affine_offset_x_px": affine.get("affine_shift_x_px", ""),
            "affine_offset_y_px": affine.get("affine_shift_y_px", ""),
            "affine_offset_x_m": offset_m("affine_shift_x_px"),
            "affine_offset_y_m": offset_m("affine_shift_y_px"),
        }
        norm = {}
        if args.normalization_dir:
            composite = self.assignment.get(task["scene_id"])
            quality = None
            if fit_values is not None:
                quality = self._quality_from_plan(plan, fit_values, band_names)
                save_fit_quality(args.norm_quality_cache, plan["group"], task["scene_id"], quality)
            norm = normalization_fields(args.normalization_dir, self.group_dirs.get(composite, composite),
                                        task["scene_id"], self.preferred_paths.get(task["scene_id"]),
                                        args.norm_quality_cache, quality)
        # Metrics are keyed by the 0-based moving band index; the "before"
        # values are the same for both models.
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
        return SceneSample(self.points, point_index, cols, rows, band_names, values, clear,
                           scene_fields, band_metrics, norm)


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
    parser.add_argument("--norm-quality-cache", default=None,
                        help="directory keeping each scene's recomputed norm_*_all values (see the module "
                             "docstring), so later runs reuse them")
    parser.add_argument("--preferred-imagery", default=None,
                        help="preferred_imagery.csv giving the selected_reflectance column (default: "
                             "<affine-dir>/preferred_imagery.csv)")
    parser.add_argument("--shard", default=None, help="K/N: only every N-th scene, starting at the K-th (1-based)")
    parser.add_argument("--workers", type=int, default=1,
                        help="split the scenes over this many processes (shards) and concatenate their outputs")
    parser.add_argument("--tmp-dir", default=None,
                        help="where the output is written before being copied to --output (default: the system "
                             "temporary directory)")
    parser.add_argument("--format", choices=("csv", "parquet"), default="csv",
                        help="csv (default): one CSV file at --output. parquet: a parquet dataset in the "
                             "--output directory, partitioned by --partition-by (see point_samples_parquet.py)")
    parser.add_argument("--partition-by", default=None,
                        help="parquet: comma-separated partition folders, highest level first (default: the "
                             "--id-field alone, e.g. genus,treeid); rows within each are sorted by date")
    parser.add_argument("--staging-dir", default=None,
                        help="parquet: where the per-scene samples are staged before being arranged into the "
                             "partitions (default: <output>/_staging); kept until the dataset is complete")
    parser.add_argument("--buckets", type=int, default=512,
                        help="parquet: groups of points the staging is split into, each arranged into the final "
                             "partitions in one pass (more buckets: less memory per pass)")
    parser.add_argument("--flush-rows", type=int, default=8_000_000,
                        help="parquet: staged rows a process buffers in memory before writing them out")
    parser.add_argument("--layout", choices=("full", "slim"), default="full",
                        help="parquet: full (default) -- every column in every point file; slim -- point files "
                             "with the per-point columns only, and the scene-and-band fields once in "
                             "<output>/_scene_bands.parquet (join on scene_id, band)")
    parser.add_argument("--finalize-only", action="store_true",
                        help="parquet: only arrange already-staged samples into the final partitions")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.composite and not args.normalization_dir:
        parser.error("--composite needs --normalization-dir")
    if any("=" not in g for g in args.group_dir):
        parser.error("--group-dir takes COMPOSITE=FOLDER")
    group_dirs = dict(g.split("=", 1) for g in args.group_dir)

    if args.format == "parquet" and args.finalize_only:
        import point_samples_parquet as psp
        psp.finalize(args, [f.strip() for f in args.extra_fields.split(",") if f.strip()], workers=args.workers)
        return
    if args.format == "parquet" and args.workers > 1 and not args.shard:
        # Each shard stages its scenes; then the dataset is arranged here.
        import subprocess
        import point_samples_parquet as psp
        raw = list(sys.argv[1:] if argv is None else argv)
        passthrough, skip = [], False
        for a in raw:
            if skip:
                skip = False
                continue
            if a == "--workers":
                skip = True
                continue
            if not a.startswith("--workers="):
                passthrough.append(a)
        procs = [subprocess.Popen([sys.executable, "-W", "ignore", os.path.abspath(__file__), *passthrough,
                                   "--shard", f"{k}/{args.workers}"]) for k in range(1, args.workers + 1)]
        failed = [k for k, proc in enumerate(procs, start=1) if proc.wait() != 0]
        if failed:
            raise SystemExit(f"shard(s) {failed} failed; rerun to continue (staged scenes are kept)")
        psp.finalize(args, [f.strip() for f in args.extra_fields.split(",") if f.strip()], workers=args.workers)
        return
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
    ctx = SceneContext(args, points, assignment, group_dirs)
    print(f"{len(points)} points, {len(tasks)} scenes", flush=True)

    if args.format == "parquet":
        import point_samples_parquet as psp
        psp.sample_to_staging(ctx, tasks, args, extra_fields)
        if not args.shard:
            psp.finalize(args, extra_fields)
        return

    columns = output_columns(args.id_field, extra_fields, bool(args.normalization_dir))
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
            sample = ctx.sample_scene(task)
            if sample is not None:
                for record in sample.csv_records(args.id_field):
                    writer.writerow(record)
                    n_rows += 1
            if i % 20 == 0 or i == len(tasks):
                print(f"[progress] {i}/{len(tasks)} scenes, {n_rows} rows, {time.time() - start:.0f} s", flush=True)
    shutil.copy(tmp_path, args.output)
    os.remove(tmp_path)
    os.rmdir(tmp_dir)
    print(f"DONE: {n_rows} rows written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
