"""Which version of each scene -- raw or registered -- later steps use.

scripts/register_to_solstice_composites.py writes, for one warp model's
output directory, a registration_manifest.csv (one row per scene) and
alignment_metrics/registration_metrics_<n>band.csv (planet_georeg_opencv's
before/after metrics). build_preferred_imagery reads those back, checks
each scene against registration.RegistrationAcceptance, and lists the file
every later step should read for it: the registered scene if it was
accepted (and exists), otherwise the raw one. write_preferred_imagery
saves that as preferred_imagery.csv next to the manifest.

The pipeline (pipeline.run_pipeline's preferred_imagery_manifest) then
swaps each scene's analytic_path for its preferred file via
apply_preferred_imagery. Only the analytic raster is swapped: a registered
scene has no UDM2 or metadata of its own, so those keep coming from the raw
scene's folder. (A registered scene is on the raw scene's own pixel grid,
so the raw UDM2 lines up with it to within the registration's correction.)
"""

from __future__ import annotations

import csv
import os
from dataclasses import replace

from . import io, registration, sensors

PREFERRED_IMAGERY_FILENAME = "preferred_imagery.csv"


def _fields(acceptance: registration.RegistrationAcceptance) -> list[str]:
    per_band = [f"{stat}_{band}_{when}_all" for band in acceptance.bands
                for stat in ("r2", "rmse") for when in ("before", "after")]
    return [
        "scene_id", "product", "acquired", "preferred_source", "preferred_analytic_path",
        "raw_analytic_path", "udm2_path", "registered_path", "warp_model", "registration_status",
        "accepted", "failed_checks", "ransac_inliers", "affine_scale_x", "affine_scale_y", *per_band,
    ]


def _read_rows(path: str, key: str) -> dict:
    """{row[key]: row} from a CSV (a later row for the same key wins, as
    when a scene was re-run with --retry-failed)."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {row[key]: row for row in csv.DictReader(f)}


def _read_metrics(output_dir: str) -> dict:
    metrics = {}
    metrics_dir = os.path.join(output_dir, "alignment_metrics")
    if os.path.isdir(metrics_dir):
        for name in sorted(os.listdir(metrics_dir)):
            if name.startswith("registration_metrics_") and name.endswith("band.csv"):
                metrics.update(_read_rows(os.path.join(metrics_dir, name), "moving_filename"))
    return metrics


def build_preferred_imagery(
    output_dir: str,
    acceptance: registration.RegistrationAcceptance = registration.RegistrationAcceptance(),
) -> list[dict]:
    """One row per scene in `output_dir`'s registration_manifest.csv (see
    the module docstring), in manifest order. `preferred_source` is
    "registered" or "raw"; `failed_checks` lists why a scene was not
    accepted (registration.registration_failed_checks, plus
    "registration_<status>" for a registration that did not complete and
    "registered_file_missing" for an accepted one whose output is not on
    disk -- a run that only saves accepted scenes never writes the rest)."""
    # Keyed by raw file name, not scene_id: a 4-band and an 8-band product
    # of the same capture share a scene_id.
    manifest = {os.path.basename(row["analytic_path"]): row
                for row in _read_rows(os.path.join(output_dir, "registration_manifest.csv"), "analytic_path").values()}
    metrics = _read_metrics(output_dir)
    rows = []
    for entry in manifest.values():
        scene_id = entry["scene_id"]
        raw_path = entry["analytic_path"]
        registered_path = entry.get("output_path") or ""
        band_names = sensors.BAND_PROFILES.get(int(entry["band_count"]), []) if entry.get("band_count") else []
        scene_metrics = {**metrics.get(os.path.basename(raw_path), {}),
                         "affine_scale_x": entry.get("affine_scale_x"), "affine_scale_y": entry.get("affine_scale_y")}
        if entry["status"] != "ok":
            failed = [f"registration_{entry['status']}"]
        else:
            failed = registration.registration_failed_checks(
                scene_metrics, band_names, entry.get("guided_ransac_inliers"), acceptance)
            if not failed and not (registered_path and os.path.exists(registered_path)):
                failed = ["registered_file_missing"]
        accepted = not failed
        row = {
            "scene_id": scene_id, "product": entry.get("product", ""), "acquired": entry.get("acquired", ""),
            "preferred_source": "registered" if accepted else "raw",
            "preferred_analytic_path": registered_path if accepted else raw_path,
            "raw_analytic_path": raw_path, "udm2_path": registration.udm2_path_for(raw_path) or "",
            "registered_path": registered_path if registered_path and os.path.exists(registered_path) else "",
            "warp_model": entry.get("warp_model", ""), "registration_status": entry["status"],
            "accepted": str(accepted).lower(), "failed_checks": ";".join(failed),
            "ransac_inliers": entry.get("guided_ransac_inliers", ""),
            "affine_scale_x": entry.get("affine_scale_x", ""), "affine_scale_y": entry.get("affine_scale_y", ""),
        }
        for band in acceptance.bands:
            b = band_names.index(band) if band in band_names else None
            for stat in ("r2", "rmse"):
                for when in ("before", "after"):
                    row[f"{stat}_{band}_{when}_all"] = (
                        scene_metrics.get(f"{stat}_band{b}_{when}_all", "") if b is not None else "")
        rows.append(row)
    return rows


def write_preferred_imagery(
    output_dir: str,
    acceptance: registration.RegistrationAcceptance = registration.RegistrationAcceptance(),
    path: str | None = None,
) -> tuple[str, list[dict]]:
    """build_preferred_imagery, written to `path` (default:
    preferred_imagery.csv in `output_dir`). Returns (path, rows)."""
    rows = build_preferred_imagery(output_dir, acceptance)
    path = path or os.path.join(output_dir, PREFERRED_IMAGERY_FILENAME)
    # Written in place rather than via a rename, which is unreliable on
    # the mounts this project's outputs live on.
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_fields(acceptance))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def load_preferred_imagery(path: str) -> dict[str, str | None]:
    """{raw analytic file name: registered path to use instead, or None to
    keep the raw scene} from a preferred_imagery.csv. Keyed by file name
    rather than full path so the manifest still applies when the raw
    archive is reached through a different mount point."""
    with open(path, newline="") as f:
        return {
            os.path.basename(row["raw_analytic_path"]):
                row["preferred_analytic_path"] if row["preferred_source"] == "registered" else None
            for row in csv.DictReader(f)
        }


def apply_preferred_imagery(scenes: list[io.Scene], manifest_path: str, log=lambda msg: None) -> list[io.Scene]:
    """`scenes` with each analytic_path replaced by its registered version
    where `manifest_path` (a preferred_imagery.csv) prefers it. UDM2 and
    metadata paths stay those of the raw scene; scenes the manifest keeps
    raw, or does not list, are returned unchanged."""
    preferred = load_preferred_imagery(manifest_path)
    out, n_registered, n_unlisted = [], 0, 0
    for scene in scenes:
        name = os.path.basename(scene.analytic_path)
        if name not in preferred:
            n_unlisted += 1
        elif preferred[name]:
            scene = replace(scene, analytic_path=preferred[name])
            n_registered += 1
        out.append(scene)
    log(f"Preferred imagery ({manifest_path}): {n_registered} of {len(scenes)} scenes use their registered version"
        + (f"; {n_unlisted} are not in the manifest and use their raw version" if n_unlisted else ""))
    return out
