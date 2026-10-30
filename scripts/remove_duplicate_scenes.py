#!/usr/bin/env python3
"""Find scenes that are in the raw archive more than once -- the same
capture (scene_id) as both a 4-band and an 8-band product, as happens over
the period the two products overlap -- and remove the copy with fewer
bands, provided every one of its bands is also in the copy kept. Run it
after ingesting new data; with nothing to remove it only reports.

The rest of psnorm assumes one file per scene_id (every per-scene output
is named by it, and io.discover_scenes refuses an archive with duplicates),
so this keeps the archive in that shape rather than every step having to
choose between copies.

A scene's files are its analytic raster and every sidecar next to it named
"<scene_id>_*" (UDM2, metadata XML/JSON, .aux.xml). Folders are read with
io.list_folder, which also finds the files a broken directory listing
leaves out, wherever the folder's order manifest names them. Duplicates that cannot
be resolved this way -- equal band counts, a band in the smaller copy that
the larger lacks, or both copies in one folder -- are reported and left
alone.

With --registered-dir (repeatable; e.g. the affine_transform and
tps_transform trees of scripts/register_to_solstice_composites.py), a
removed scene's registration outputs go too: its registered raster, its
rows in registration_manifest.csv, alignment_metrics/registration_metrics_
<n>band.csv and preferred_imagery.csv (each CSV backed up first, as
<name>.before_dedup_<time>).

Nothing is deleted without --apply. Every removed file is logged to
--log (appended; default <raw_root>/removed_duplicate_scenes.csv).

Usage:
    python scripts/remove_duplicate_scenes.py /mnt/e/E_Coast_PlanetScope/philadelphia/raw \\
        --registered-dir /mnt/e/E_Coast_PlanetScope/philadelphia/affine_transform \\
        --registered-dir /mnt/e/E_Coast_PlanetScope/philadelphia/tps_transform [--apply]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psnorm import io, sensors  # noqa: E402


def find_analytic_files(raw_root: str) -> dict[str, list[str]]:
    """{scene_id: [analytic paths]} for every 4- or 8-band analytic file
    anywhere under raw_root."""
    found = defaultdict(list)
    for folder in io.walk_folders(raw_root, recursive=True):
        for name in io.list_folder(folder):
            for suffix in io.ANALYTIC_SUFFIXES:
                if name.endswith(suffix) and not name.startswith("._"):
                    found[name[: -len(suffix)]].append(os.path.join(folder, name))
    return found


def plan_removals(found: dict[str, list[str]], band_names=sensors.detect_band_names):
    """([(scene_id, path to remove, path kept)], [(scene_id, paths, reason)])
    for every scene_id with more than one analytic file."""
    removals, unresolved = [], []
    for scene_id, paths in sorted(found.items()):
        if len(paths) < 2:
            continue
        bands = {p: band_names(p) for p in paths}
        keep = max(paths, key=lambda p: len(bands[p]))
        if sum(len(bands[p]) == len(bands[keep]) for p in paths) > 1:
            unresolved.append((scene_id, paths, f"{len(bands[keep])} bands in more than one copy"))
            continue
        for path in paths:
            if path == keep:
                continue
            missing = [b for b in bands[path] if b not in bands[keep]]
            if missing:
                unresolved.append((scene_id, paths, f"{os.path.basename(path)} has bands {missing} that "
                                                    f"{os.path.basename(keep)} lacks"))
            elif os.path.dirname(path) == os.path.dirname(keep):
                unresolved.append((scene_id, paths, "both copies in one folder (their sidecars cannot be told "
                                                    "apart)"))
            else:
                removals.append((scene_id, path, keep))
    return removals, unresolved


def scene_files(scene_id: str, analytic_path: str) -> list[str]:
    """The analytic file and its sidecars ("<scene_id>_*" in its folder)."""
    folder = os.path.dirname(analytic_path)
    return [os.path.join(folder, n) for n in io.list_folder(folder) if n.startswith(scene_id + "_")]


def _filter_csv(path: str, drop, stamp: str) -> int:
    """Rewrite the CSV at `path` without the rows for which drop(row) is
    true, after backing it up. Written in place (no rename: unreliable on
    the mounts these outputs live on). Returns how many rows were dropped."""
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    kept = [r for r in rows if not drop(r)]
    if len(kept) == len(rows):
        return 0
    shutil.copy2(path, f"{path}.before_dedup_{stamp}")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    return len(rows) - len(kept)


def registered_outputs(registered_dir: str, names: set[str]) -> list[str]:
    """The registered rasters recorded in registered_dir's manifest for raw
    analytic files named in `names`."""
    manifest = os.path.join(registered_dir, "registration_manifest.csv")
    if not os.path.exists(manifest):
        return []
    with open(manifest, newline="") as f:
        return sorted({r["output_path"] for r in csv.DictReader(f)
                       if os.path.basename(r["analytic_path"]) in names and r.get("output_path")
                       and os.path.exists(r["output_path"])})


def clean_registered_dir(registered_dir: str, names: set[str], stamp: str) -> dict[str, int]:
    """Drop the rows for raw analytic files named in `names` from
    registered_dir's manifest, alignment metrics and preferred imagery."""
    counts = {"registration_manifest.csv": _filter_csv(
        os.path.join(registered_dir, "registration_manifest.csv"),
        lambda r: os.path.basename(r["analytic_path"]) in names, stamp)}
    for path in sorted(glob.glob(os.path.join(registered_dir, "alignment_metrics", "registration_metrics_*band.csv"))):
        counts[os.path.relpath(path, registered_dir)] = _filter_csv(
            path, lambda r: r["moving_filename"] in names, stamp)
    counts["preferred_imagery.csv"] = _filter_csv(
        os.path.join(registered_dir, "preferred_imagery.csv"),
        lambda r: os.path.basename(r["raw_analytic_path"]) in names, stamp)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_root")
    parser.add_argument("--registered-dir", action="append", default=[],
                        help="a registration output directory to remove the duplicates' outputs and rows from "
                             "(repeatable)")
    parser.add_argument("--apply", action="store_true", help="delete the files (default: only report)")
    parser.add_argument("--log", default=None, help="CSV of removed files (default: <raw_root>/removed_duplicate_scenes.csv)")
    args = parser.parse_args()

    found = find_analytic_files(args.raw_root)
    n_dup = sum(len(p) > 1 for p in found.values())
    print(f"{sum(map(len, found.values()))} analytic files, {len(found)} scene IDs, {n_dup} found more than once.")
    removals, unresolved = plan_removals(found)
    for scene_id, paths, reason in unresolved:
        print(f"  UNRESOLVED {scene_id}: {reason}: {paths}")
    by_folder = defaultdict(int)
    for _, path, keep in removals:
        by_folder[(os.path.relpath(os.path.dirname(path), args.raw_root),
                   os.path.relpath(os.path.dirname(keep), args.raw_root))] += 1
    for (folder, kept_in), n in sorted(by_folder.items()):
        print(f"  {n:4d} in {folder} (kept: {kept_in})")
    print(f"{len(removals)} duplicate(s) to remove, {len(unresolved)} unresolved.")

    names = {os.path.basename(path) for _, path, _ in removals}
    files = [(scene_id, f) for scene_id, path, _ in removals for f in scene_files(scene_id, path)]
    outputs = {d: registered_outputs(d, names) for d in args.registered_dir}
    size_gb = sum(os.path.getsize(f) for _, f in files) / 1e9
    print(f"Raw files: {len(files)} ({size_gb:.1f} GB). Registered outputs: "
          + (", ".join(f"{len(o)} in {d}" for d, o in outputs.items()) or "none checked"))
    if not removals:
        return
    if not args.apply:
        print("Dry run: nothing deleted (pass --apply).")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log or os.path.join(args.raw_root, "removed_duplicate_scenes.csv")
    new_log = not os.path.exists(log_path)
    kept_by_id = {scene_id: keep for scene_id, _, keep in removals}
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_log:
            writer.writerow(["removed_at", "scene_id", "removed_file", "kept_analytic_path"])
        for scene_id, path in files:
            os.remove(path)
            writer.writerow([stamp, scene_id, path, kept_by_id[scene_id]])
        for d, paths in outputs.items():
            for path in paths:
                scene_id = next(s for s, p, _ in removals if os.path.basename(p) == os.path.basename(path))
                os.remove(path)
                writer.writerow([stamp, scene_id, path, kept_by_id[scene_id]])
    for d in args.registered_dir:
        counts = clean_registered_dir(d, names, stamp)
        print(f"  {d}: rows removed: " + ", ".join(f"{k} {n}" for k, n in counts.items()))
    print(f"Removed {len(files)} raw file(s) and {sum(map(len, outputs.values()))} registered output(s); "
          f"logged to {log_path}")


if __name__ == "__main__":
    main()
