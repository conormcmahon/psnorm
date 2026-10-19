#!/usr/bin/env python3
"""Run the psnorm pipeline with per-year composites (e.g. the solstice
composites from scripts/build_solstice_composite_reference.py) as the
references, instead of reference scenes.

Every scene is assigned to one composite, then each composite's scenes are
normalized together against it (one pipeline.run_pipeline call per
composite, output in <--output>/<composite name>/):

  - A scene uses the composite whose solstice is nearest its acquisition
    time, among the composites that can correct *all* of its bands (see
    psnorm.composite_reference.composite_is_eligible). With --composite-year,
    that year's composite instead, wherever it is eligible.
  - A composite holds a band only where that band is valid over most of its
    area (--min-band-coverage): composites built before 8-band data
    existed carry coastal_blue/green_i/yellow/rededge as empty layers.
  - An 8-band scene additionally only uses a composite whose acquisition
    window (solstice +- --window-days) starts on or after the onset of
    8-band data -- the earliest 8-band scene found, or --eight-band-onset.

Scenes are discovered recursively under --input (e.g. the raw archive's
<product>/<period>/ folders, both 4- and 8-band). The assignment is written
to <--output>/composite_assignment.csv; --dry-run stops there.

Usage:
    python scripts/run_composite_reference_pipeline.py \\
        --input /mnt/e/E_Coast_PlanetScope/philadelphia/raw \\
        --composite-dir /mnt/e/E_Coast_PlanetScope/philadelphia/summer_solstice_composites \\
        --preferred-imagery-manifest /mnt/e/E_Coast_PlanetScope/philadelphia/affine_transform/preferred_imagery.csv \\
        --output /mnt/e/E_Coast_PlanetScope/philadelphia/normalized_composite_reference [--dry-run]
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_multi_reference_pipeline import (  # noqa: E402
    _run_pipeline_kwargs, add_pipeline_arguments, log, workers_from_args,
)

from psnorm import composite_reference as cref  # noqa: E402
from psnorm import io as pio  # noqa: E402
from psnorm import pipeline, preferred_imagery  # noqa: E402


def write_assignment(path, groups, composites, band_names):
    by_path = {c.path: c for c in composites}
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "acquired", "n_bands", "analytic_path", "composite", "composite_year",
                         "days_from_solstice"])
        rows = []
        for composite_path, scenes in groups.items():
            c = by_path[composite_path]
            for s in scenes:
                rows.append([s.scene_id, s.acquired.isoformat(), len(band_names[s.scene_id]), s.analytic_path,
                             c.name, c.year, round((s.acquired - c.solstice) / timedelta(days=1), 1)])
        writer.writerows(sorted(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="folder searched recursively for scenes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--composite-dir", required=True)
    parser.add_argument("--composite-prefix", default="solstice", help="composites are <prefix>_<year>.tif")
    parser.add_argument("--window-days", type=float, default=14,
                        help="the --window-days the composites were built with (default 14)")
    parser.add_argument("--hemisphere", default="N", choices=["N", "S"])
    parser.add_argument("--composite-year", type=int, default=None,
                        help="use this year's composite for every scene it is eligible for")
    parser.add_argument("--eight-band-onset", default=None,
                        help="YYYY-MM-DD; default: the earliest 8-band scene's acquisition date")
    parser.add_argument("--min-band-coverage", type=float, default=cref.DEFAULT_MIN_BAND_COVERAGE)
    parser.add_argument("--composite-stat", default="median")
    parser.add_argument("--composite-cache-dir", default=None,
                        help="where the composites' single-statistic caches are (default: "
                             ".composite_reference_cache next to them)")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="only run these composites' groups (the assignment still considers all)")
    parser.add_argument("--dry-run", action="store_true", help="write the assignment and stop")
    add_pipeline_arguments(parser)
    args = parser.parse_args()

    scenes = pio.discover_scenes(args.input, recursive=True)
    if not scenes:
        raise SystemExit(f"No scenes found under '{args.input}'.")
    log(f"Discovered {len(scenes)} scenes.")
    if args.preferred_imagery_manifest:
        scenes = preferred_imagery.apply_preferred_imagery(scenes, args.preferred_imagery_manifest, log=log)
    band_names = cref.scene_band_names(scenes)

    composites = cref.find_solstice_composites(args.composite_dir, prefix=args.composite_prefix,
                                               hemisphere=args.hemisphere, window_days=args.window_days)
    if not composites:
        raise SystemExit(f"No {args.composite_prefix}_<year>.tif composites in '{args.composite_dir}'.")
    composite_bands = {}
    for c in composites:
        coverage = cref.band_coverage(c.path, stat=args.composite_stat, cache_dir=args.composite_cache_dir)
        composite_bands[c.path] = cref.present_bands(coverage, args.min_band_coverage)
        log(f"  {c.name}: window from {c.window_start:%Y-%m-%d}, bands {composite_bands[c.path]}")

    onset = (datetime.strptime(args.eight_band_onset, "%Y-%m-%d") if args.eight_band_onset
             else cref.eight_band_onset(band_names, scenes))
    log(f"8-band data onset: {onset:%Y-%m-%d}" if onset else "No 8-band scenes.")

    groups = cref.assign_scenes_to_composites(scenes, band_names, composites, composite_bands, onset=onset,
                                              fixed_year=args.composite_year, log=log)
    os.makedirs(args.output, exist_ok=True)
    assignment_path = os.path.join(args.output, "composite_assignment.csv")
    write_assignment(assignment_path, groups, composites, band_names)
    for c in composites:
        counts = Counter(len(band_names[s.scene_id]) for s in groups[c.path])
        log(f"  {c.name}: {len(groups[c.path])} scenes ({counts[4]} 4-band, {counts[8]} 8-band)")
    log(f"Wrote {assignment_path}")
    if args.dry_run:
        return

    workers = workers_from_args(args)
    summary = {"eight_band_onset": onset.isoformat() if onset else None, "groups": {}}
    for c in composites:
        group = groups[c.path]
        if not group or (args.years and c.year not in args.years):
            continue
        log(f"=== {c.name}: {len(group)} scenes ===")
        output_dir = os.path.join(args.output, c.name)
        try:
            result = pipeline.run_pipeline(
                args.input, output_dir, reference_composite=c.path, composite_stat=args.composite_stat,
                composite_cache_dir=args.composite_cache_dir, min_composite_band_coverage=args.min_band_coverage,
                scenes=group, workers=workers, log=log, **_run_pipeline_kwargs(args),
            )
        except Exception as exc:
            log(f"  {c.name} FAILED: {exc!r}")
            summary["groups"][c.name] = {"error": repr(exc)}
            continue
        summary["groups"][c.name] = {
            "n_scenes": len(group), "band_names": result.band_names,
            "n_consensus_pixels": result.n_consensus_pixels, "output_dir": result.output_dir,
            "status_counts": dict(Counter(r.status for r in result.scene_results)),
        }
        # Written after every group, so a long run's progress is on disk.
        with open(os.path.join(args.output, "composite_reference_manifest.json"), "w") as f:
            json.dump(summary, f, indent=2)
    log("Done.")


if __name__ == "__main__":
    main()
