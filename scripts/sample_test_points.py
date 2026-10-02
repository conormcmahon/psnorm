#!/usr/bin/env python3
"""Sample a set of test points (a point layer in a GeoPackage, with a
'class' attribute) against one psnorm run's outputs, producing one
long-format CSV row per (point, scene, band):

  - the original (pre-correction) DN, read from the scene's own raw
    analytic image
  - the corrected DN, read from that scene's psnorm `normalized/` output
  - the exclusion-flags bitmask value at that pixel (see masking.py's
    EXCLUDE_* bit constants) -- same for every band, repeated per row
  - whether that scene flagged the point as an IR-MAD invariant-target
    *candidate* (Phase A) -- NaN for the reference scene itself, which
    has no self-comparison
  - whether the point was retained as a Phase B *consensus* candidate --
    a property of the point alone (computed once against the reference
    grid), repeated identically across every scene's row for that point

Every point is sampled against every scene that has a psnorm `normalized/`
output (the reference scene plus every successfully-corrected target),
regardless of whether that particular scene's footprint actually covers
the point -- out-of-footprint or nodata pixels come back as NaN, which
keeps a consistent per-point time axis (useful for spotting coverage gaps)
rather than silently dropping scenes.

Usage:
    .venv/bin/python scripts/sample_test_points.py \\
        --points /path/to/test_points.gpkg \\
        --run-dir /path/to/psnorm_output \\
        --input-dir /path/to/raw/files \\
        --output /path/to/sampled_points.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import sys
import tempfile

from osgeo import gdal, ogr, osr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psnorm import io

gdal.UseExceptions()
ogr.UseExceptions()

BAND_NAMES = ["blue", "green", "red", "nir"]


def load_points(gpkg_path: str) -> list[dict]:
    """[{"id":, "class":, "lon":, "lat":}, ...] from the first layer of
    `gpkg_path` -- assumed to be WGS84 points with a 'class' text field.

    Copies the GeoPackage (and any `-wal`/`-shm` sidecar files) to a local
    temp directory before opening it. A GeoPackage left mid-transaction by
    an editor (e.g. QGIS without an explicit save/checkpoint) keeps its
    real feature data in the `-wal` journal rather than the main file --
    opening it in place off a flaky/network-ish mount can fail to replay
    that journal (a disk I/O error falls back to an IMMUTABLE open that
    silently ignores the WAL and reports 0 features) even though the data
    is intact. A plain local copy sidesteps whatever lock/journal quirk the
    source location has and lets SQLite replay the WAL normally.
    """
    tmp_dir = tempfile.mkdtemp(prefix="psnorm_gpkg_")
    local_path = os.path.join(tmp_dir, os.path.basename(gpkg_path))
    for src in glob.glob(gpkg_path + "*"):  # main file plus -wal/-shm if present
        shutil.copy2(src, os.path.join(tmp_dir, os.path.basename(src)))

    ds = ogr.Open(local_path)
    layer = ds.GetLayer(0)
    points = []
    layer.ResetReading()
    for feat in layer:
        geom = feat.GetGeometryRef()
        points.append({
            "id": feat.GetFID(),
            "class": feat.GetField("class"),
            "lon": geom.GetX(),
            "lat": geom.GetY(),
        })
    ds = None
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return points


_LOG_SCENE_RE = re.compile(r"^\s{2}(\S+):", re.MULTILINE)


def scene_ids_from_run_log(log_path: str) -> set[str]:
    """Every scene ID mentioned in a psnorm run log (lines of the form
    "  {scene_id}: {status}"), as a plain sequential text scan.

    This is the authoritative scene list for a run -- NOT `os.listdir()` on
    an output directory, and NOT `io.discover_scenes()` (glob-based) on the
    input directory. Both of those go through directory listings, and this
    external drive has repeatedly been shown to serve stale/incomplete
    directory listings (confirmed concretely on this exact run: `os.listdir`
    on `normalized/` found 435 of 728 real files; `io.discover_scenes` on
    the raw input folder found 3813 of the true 4023 scenes). A log file
    written progressively during the run itself is just a text file read
    top to bottom, with no directory-listing step involved, so it doesn't
    share that failure mode.
    """
    with open(log_path) as f:
        text = f.read()
    return set(_LOG_SCENE_RE.findall(text))


def _reproject_points(points: list[dict], dst_wkt: str) -> None:
    """Adds "x"/"y" (in `dst_wkt`) to each point dict in place."""
    src = osr.SpatialReference()
    src.ImportFromEPSG(4326)
    dst = osr.SpatialReference()
    dst.ImportFromWkt(dst_wkt)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src, dst)
    for p in points:
        x, y, _ = transform.TransformPoint(p["lon"], p["lat"])
        p["x"], p["y"] = x, y


def _pixel_offset(info: io.RasterInfo, x: float, y: float) -> tuple[int, int] | None:
    gt = info.geotransform
    col = int((x - gt[0]) / gt[1])
    row = int((gt[3] - y) / -gt[5])
    if 0 <= col < info.width and 0 <= row < info.height:
        return col, row
    return None


class RasterSampler:
    """Opens one raster once and samples arbitrary single pixels from it
    (each a tiny windowed read, not a full-scene load) -- cheap even for a
    ~9000x3500 scene since only the handful of requested points are ever
    actually read off disk."""

    def __init__(self, path: str):
        self.info = io.get_raster_info(path)
        self._dataset, self._bands = io.open_bands(path)
        self._nodata = [b.GetNoDataValue() for b in self._bands]

    def sample(self, x: float, y: float, band: int = 1) -> float | None:
        """`band` is 1-based. None if the point falls outside this
        raster's extent or lands on that band's nodata value."""
        offset = _pixel_offset(self.info, x, y)
        if offset is None:
            return None
        col, row = offset
        value = self._bands[band - 1].ReadAsArray(col, row, 1, 1)[0, 0]
        nodata = self._nodata[band - 1]
        if nodata is not None and value == nodata:
            return None
        return float(value)

    def close(self):
        self._dataset = None
        self._bands = None


def sample_points(
    points_path: str, run_dir: str, input_dir: str, output_csv: str, *, run_log: str | None = None, log=print,
) -> str:
    points = load_points(points_path)
    log(f"Loaded {len(points)} points ({len(set(p['class'] for p in points))} classes).")

    # Authoritative scene-ID list from the run's own log file -- a plain
    # sequential text read, not a directory listing. NOT os.listdir() on the
    # normalized/ dir and NOT io.discover_scenes() (glob-based) on the input
    # dir: both go through directory listings, and this external drive has
    # repeatedly been shown to serve stale/incomplete ones (confirmed on
    # this exact run: os.listdir found 435/728 real normalized files;
    # io.discover_scenes found 3813/4023 real raw scenes). Each candidate
    # is still checked with a direct os.path.exists() below.
    if run_log is None:
        candidates = sorted(glob.glob(os.path.join(run_dir, "*.log")))
        if not candidates:
            raise FileNotFoundError(f"No --run-log given and no *.log file found in '{run_dir}'.")
        run_log = candidates[0]
        log(f"No --run-log given; using '{run_log}' (found in run dir).")
    all_scene_ids = scene_ids_from_run_log(run_log)
    log(f"Log file lists {len(all_scene_ids)} distinct scene IDs.")

    normalized_dir = os.path.join(run_dir, "normalized")
    scene_ids = sorted(
        sid for sid in all_scene_ids
        if os.path.exists(os.path.join(normalized_dir, f"{sid}_normalized.tif"))
    )
    log(f"Found {len(scene_ids)} scenes with a normalized output "
        f"(checked {len(all_scene_ids)} log-listed scenes directly, not via directory listing).")

    # Reproject points into the run's own CRS (every psnorm scene here
    # shares one CRS/grid -- confirmed for this dataset elsewhere -- so a
    # single reprojection, taken from any one scene, applies to all of them).
    sample_info = io.get_raster_info(os.path.join(input_dir, f"{scene_ids[0]}_3B_AnalyticMS_SR_harmonized_clip.tif"))
    _reproject_points(points, sample_info.crs)

    consensus_sampler = RasterSampler(os.path.join(run_dir, "consensus", "consensus_mask.tif"))
    for p in points:
        v = consensus_sampler.sample(p["x"], p["y"])
        p["consensus"] = None if v is None else bool(v)
    consensus_sampler.close()

    fieldnames = [
        "point_id", "class", "lon", "lat", "scene_id", "date", "band",
        "original_dn", "corrected_dn", "flags_bitmask", "is_irmad_candidate", "is_consensus_candidate",
    ]
    n_rows = 0
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, scene_id in enumerate(scene_ids):
            raw_path = os.path.join(input_dir, f"{scene_id}_3B_AnalyticMS_SR_harmonized_clip.tif")
            normalized_path = os.path.join(normalized_dir, f"{scene_id}_normalized.tif")
            flags_path = os.path.join(run_dir, "masks", f"{scene_id}_flags.tif")
            candidate_path = os.path.join(run_dir, "candidates", f"{scene_id}_candidate.tif")

            if not (os.path.exists(raw_path) and os.path.exists(flags_path)):
                log(f"  WARNING: skipping '{scene_id}' -- missing raw image or flags raster.")
                continue

            date = io.parse_acquisition_time(scene_id)
            raw = RasterSampler(raw_path)
            corrected = RasterSampler(normalized_path)
            flags = RasterSampler(flags_path)
            candidate = RasterSampler(candidate_path) if os.path.exists(candidate_path) else None

            for p in points:
                flag_value = flags.sample(p["x"], p["y"])
                candidate_value = candidate.sample(p["x"], p["y"]) if candidate is not None else None
                for b, band_name in enumerate(BAND_NAMES, start=1):
                    writer.writerow({
                        "point_id": p["id"],
                        "class": p["class"],
                        "lon": p["lon"],
                        "lat": p["lat"],
                        "scene_id": scene_id,
                        "date": date.date().isoformat() if date else "",
                        "band": band_name,
                        "original_dn": raw.sample(p["x"], p["y"], b),
                        "corrected_dn": corrected.sample(p["x"], p["y"], b),
                        "flags_bitmask": None if flag_value is None else int(flag_value),
                        "is_irmad_candidate": None if candidate_value is None else bool(candidate_value),
                        "is_consensus_candidate": p["consensus"],
                    })
                    n_rows += 1

            raw.close()
            corrected.close()
            flags.close()
            if candidate is not None:
                candidate.close()

            if (i + 1) % 50 == 0 or i == len(scene_ids) - 1:
                log(f"  ...{i + 1}/{len(scene_ids)} scenes sampled ({n_rows} rows so far)")

    log(f"Wrote {n_rows} rows to {output_csv}")
    return output_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", required=True)
    parser.add_argument("--run-dir", required=True, help="a psnorm_output directory (masks/candidates/consensus/normalized)")
    parser.add_argument("--input-dir", required=True, help="the raw scene folder the run was built from")
    parser.add_argument("--run-log", default=None, help="the run's own log file (authoritative scene list); "
                                                          "defaults to whichever *.log file is found in --run-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sample_points(args.points, args.run_dir, args.input_dir, args.output, run_log=args.run_log)


if __name__ == "__main__":
    main()
