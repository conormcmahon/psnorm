#!/usr/bin/env python3
"""Sweep `ncp_threshold` x `invariance_frequency_threshold` against an
already-completed psnorm run, using each target scene's *saved* IR-MAD fit
(candidates/{target_id}_irmad_fit.json — see model_io.save_irmad_fit) to
reclassify invariant candidates at each ncp_threshold in one O(pixels) pass
per target per threshold, instead of re-running IR-MAD's full O(pixels *
iterations) iterative fit for every grid point. Phase B's aggregation (and
its threshold) is done in-memory for every grid cell too, so nothing here
touches disk except the saved fits/flags rasters it reads once and the
final summary CSV.

Loop order is scene-outer, threshold-inner: each target scene's window is
read from disk exactly once (normalize.load_reclassification_cache) and
reused for every ncp_threshold in the grid (normalize.reclassify_from_cache)
— NOT once per (scene, threshold) pair. This matters beyond raw speed: on a
slow or flaky filesystem, minimizing total disk reads directly reduces how
much surface area there is for a stall or transient failure to hit. Peak
memory stays bounded to one scene's window at a time (freed before the next
scene), not the whole run's worth of windows at once.

**Progress visibility**: run this with `python -u` (or set
PYTHONUNBUFFERED=1) if you redirect stdout to a file/log and want to
tail -f it live — this script's own prints are flushed explicitly
(`flush=True`) regardless, but a non-`-u` interpreter can still buffer
output from libraries it calls. A progress line is printed every
`--progress-every` scenes (default 20) specifically so a hang is visible as
"no new line for N minutes" rather than silence indistinguishable from
"still working" — if you don't see a new progress line for several minutes
past what --progress-every scenes should take, something is actually stuck
(most likely a blocking disk read on this environment's known-flaky
external-drive mount — see README's "GPU support" section / this repo's own
incident history), not just slow.

For each (ncp_threshold, invariance_frequency_threshold) pair, reports:
  n_consensus_pixels        size of the resulting global consensus set
  n_scenes_feasible         how many target scenes have >= --min-fit-pixels
                             consensus pixels inside their own ref_window --
                             a *proxy* for "would Phase C give this scene a
                             real (non-identity-fallback) fit", cheap to
                             compute here because it only needs the raw
                             consensus-pixel count, not an actual RANSAC +
                             regression run. It will over-count slightly
                             relative to a real Phase C run, since RANSAC
                             can still trim a scene below min_fit_pixels
                             even when the raw count clears it -- treat this
                             as a fast first-pass signal for narrowing the
                             grid, not a substitute for validating the
                             chosen configuration with a real run.

Usage:
    .venv/bin/python -u scripts/sweep_thresholds.py \\
        --input /path/to/scenes --output /path/to/completed_psnorm_output \\
        --ncp-thresholds 0.3,0.4,0.5,0.6,0.7 \\
        --freq-thresholds 0.2,0.3,0.4,0.5 \\
        --summary-csv sweep_summary.csv

    # Minimal-subset smoke test before committing to a full sweep:
    .venv/bin/python -u scripts/sweep_thresholds.py --limit 20 ...

Requires the target run to have been produced with fit-saving enabled
(current pipeline.py always saves it) — scenes from before that existed
won't have a `*_irmad_fit.json` and are skipped with a note.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psnorm import backend, io, masking, model_io, normalize, sensors  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def discover_target_records(output_dir: str, input_dir: str, *, limit: int | None = None, log=_log):
    """One lightweight record per target scene with a saved fit: paths,
    windows, band indices, and the small fitted model -- everything
    run_sweep needs *except* the `evaluated` mask and pixel data, which are
    deliberately loaded lazily, one scene at a time, inside run_sweep's own
    loop rather than here. Doing it here and stashing the mask on the record
    would mean every one of (up to several thousand) scenes' masks stays
    resident for the entire run just because `records` itself does -- on a
    dataset with large per-scene overlap windows that's enough to exhaust
    memory well before the actual (intentionally one-scene-at-a-time)
    pixel-cache loop does. Keep `records` itself cheap (paths and small
    scalars only) so its total size never depends on window size.

    Deliberately does *not* use glob/listdir over candidates/ to find which
    scenes have a saved fit — on some network/virtual filesystems directory
    listings can lag behind what's actually on disk (a file written moments
    ago not yet appearing in a fresh listing, even though it opens fine by
    path). Every scene from `io.discover_scenes(input_dir)` is checked by
    direct path existence instead, which doesn't have that failure mode.
    """
    log(f"Discovering scenes in {input_dir}...")
    t0 = time.time()
    scenes = io.discover_scenes(input_dir)
    log(f"  {len(scenes)} scenes discovered in {time.time() - t0:.1f}s")

    records = []
    reference_id = None
    skipped_no_fit = 0

    t0 = time.time()
    for i, scene in enumerate(scenes):
        if limit is not None and len(records) >= limit:
            log(f"  --limit {limit} reached, stopping discovery early")
            break
        if i and i % 500 == 0:
            log(f"  ...checked {i}/{len(scenes)} scenes ({time.time() - t0:.1f}s elapsed, {len(records)} usable so far)")

        target_id = scene.scene_id
        stats_path = os.path.join(output_dir, "candidates", f"{target_id}_stats.json")
        fit_path = os.path.join(output_dir, "candidates", f"{target_id}_irmad_fit.json")
        if not os.path.exists(stats_path):
            continue  # not every scene reaches Phase A candidate detection (e.g. no overlap)
        if not os.path.exists(fit_path) or not model_io.irmad_fit_is_valid(fit_path):
            skipped_no_fit += 1
            continue

        stats = model_io.load_json(stats_path)
        reference_id = stats["reference_id"]
        ref_window = tuple(stats["ref_window"])
        tgt_window = tuple(stats["tgt_window"])

        reference_scene = next((s for s in scenes if s.scene_id == reference_id), None)
        if reference_scene is None:
            continue

        reference_band_names = sensors.detect_band_names(reference_scene.analytic_path)
        target_band_names = sensors.detect_band_names(scene.analytic_path)
        common_bands = [b for b in reference_band_names if b in target_band_names]
        ref_indices = [sensors.band_index(reference_band_names, b) for b in common_bands]
        tgt_indices = [sensors.band_index(target_band_names, b) for b in common_bands]

        ref_flags_path = os.path.join(output_dir, "masks", f"{reference_id}_flags.tif")
        tgt_flags_path = os.path.join(output_dir, "masks", f"{target_id}_flags.tif")
        fit = model_io.load_irmad_fit(fit_path)

        records.append(dict(
            target_id=target_id, reference_path=reference_scene.analytic_path,
            target_path=scene.analytic_path, ref_window=ref_window, tgt_window=tgt_window,
            ref_indices=ref_indices, tgt_indices=tgt_indices,
            ref_flags_path=ref_flags_path, tgt_flags_path=tgt_flags_path, fit=fit,
        ))

    log(f"Discovery done in {time.time() - t0:.1f}s: {len(records)} usable target scenes, "
        f"{skipped_no_fit} skipped (no saved fit)")
    return records, reference_id, skipped_no_fit


def run_sweep(records, ref_shape, ncp_grid, freq_grid, *, min_observations, min_fit_pixels, device,
              progress_every=20, log=_log, raster_output_dir=None, reference_info=None):
    """Scene-outer, threshold-inner: each scene's window (pixel data *and*
    its `evaluated` search mask) is loaded from disk exactly once per scene
    regardless of len(ncp_grid), used, and released before the next scene —
    at any instant only one scene's data is resident, not the whole run's
    worth (see discover_target_records' docstring for why that distinction
    matters here). Returns the summary rows (list of dicts), ready to write
    to CSV.

    `raster_output_dir` (with `reference_info`, required together) writes
    one subdirectory per `ncp_threshold` — `{raster_output_dir}/ncp_{v}/` —
    containing `times_evaluated.tif`/`times_invariant.tif`/`frequency.tif`
    (shared across every `freq_grid` value for that ncp_threshold, since
    frequency doesn't depend on it) and one `consensus_mask_freq{f}.tif` per
    freq_grid value. This is what makes the expensive scene-reclassification
    loop above a one-time cost even if you later want rasters for *more*
    freq_grid values than were swept here: reload times_evaluated/
    times_invariant and threshold fresh, no rescan needed.
    """
    times_evaluated_by_ncp = {ncp: np.zeros(ref_shape, dtype=np.uint16) for ncp in ncp_grid}
    times_invariant_by_ncp = {ncp: np.zeros(ref_shape, dtype=np.uint16) for ncp in ncp_grid}
    per_scene_windows = []  # (target_id, ref_window) -- same for every ncp_threshold, so tracked once

    n_records = len(records)
    t_start = time.time()
    for i, rec in enumerate(records):
        if i and i % progress_every == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (n_records - i) / rate if rate > 0 else float("nan")
            log(f"  scene {i}/{n_records} ({elapsed:.1f}s elapsed, {rate:.2f} scenes/s, ETA {eta:.0f}s)")

        xoff, yoff, xsize, ysize = rec["ref_window"]
        ref_flags = masking.load_flags_window(rec["ref_flags_path"], rec["ref_window"])
        tgt_flags = masking.load_flags_window(rec["tgt_flags_path"], rec["tgt_window"])
        evaluated = (ref_flags == 0) & (tgt_flags == 0)
        evaluated_u16 = evaluated.astype(np.uint16)

        cache = normalize.load_reclassification_cache(
            rec["reference_path"], rec["target_path"], rec["ref_window"], rec["tgt_window"],
            rec["ref_indices"], rec["tgt_indices"], evaluated, device=device,
        )

        for ncp_threshold in ncp_grid:
            candidate_mask, _n_evaluated, _n_invariant = normalize.reclassify_from_cache(
                cache, rec["fit"], rec["ref_indices"], ncp_threshold, xsize, ysize,
            )
            times_evaluated_by_ncp[ncp_threshold][yoff : yoff + ysize, xoff : xoff + xsize] += evaluated_u16
            times_invariant_by_ncp[ncp_threshold][yoff : yoff + ysize, xoff : xoff + xsize] += (
                evaluated & candidate_mask
            ).astype(np.uint16)
        per_scene_windows.append((rec["target_id"], rec["ref_window"]))
        del cache, evaluated, evaluated_u16, ref_flags, tgt_flags  # one scene's data at a time, not the whole run's

    log(f"All {n_records} scenes reclassified across {len(ncp_grid)} ncp_thresholds in {time.time() - t_start:.1f}s")

    rows = []
    for ncp_threshold in ncp_grid:
        times_evaluated = times_evaluated_by_ncp[ncp_threshold]
        times_invariant = times_invariant_by_ncp[ncp_threshold]
        with np.errstate(invalid="ignore", divide="ignore"):
            frequency = np.where(
                times_evaluated > 0,
                times_invariant.astype(np.float32) / np.maximum(times_evaluated, 1),
                np.nan,
            ).astype(np.float32)

        ncp_dir = None
        if raster_output_dir is not None:
            ncp_dir = os.path.join(raster_output_dir, f"ncp_{ncp_threshold}")
            os.makedirs(ncp_dir, exist_ok=True)
            io.write_single_band_raster(times_evaluated, reference_info, os.path.join(ncp_dir, "times_evaluated.tif"), gdal.GDT_UInt16)
            io.write_single_band_raster(times_invariant, reference_info, os.path.join(ncp_dir, "times_invariant.tif"), gdal.GDT_UInt16)
            io.write_single_band_raster(frequency, reference_info, os.path.join(ncp_dir, "frequency.tif"), gdal.GDT_Float32)
            log(f"  wrote times_evaluated/times_invariant/frequency.tif to {ncp_dir}")

        for invariance_frequency_threshold in freq_grid:
            consensus_mask = (times_evaluated >= min_observations) & (frequency > invariance_frequency_threshold)
            n_consensus_pixels = int(consensus_mask.sum())

            if ncp_dir is not None:
                consensus_path = os.path.join(ncp_dir, f"consensus_mask_freq{invariance_frequency_threshold}.tif")
                io.write_single_band_raster(consensus_mask.astype(np.uint8), reference_info, consensus_path, gdal.GDT_Byte)

            n_scenes_feasible = 0
            for target_id, (xoff, yoff, xsize, ysize) in per_scene_windows:
                n_here = int(consensus_mask[yoff : yoff + ysize, xoff : xoff + xsize].sum())
                if n_here >= min_fit_pixels:
                    n_scenes_feasible += 1

            rows.append(dict(
                ncp_threshold=ncp_threshold,
                invariance_frequency_threshold=invariance_frequency_threshold,
                min_observations=min_observations,
                n_consensus_pixels=n_consensus_pixels,
                n_target_scenes=n_records,
                n_scenes_feasible=n_scenes_feasible,
                fraction_scenes_feasible=n_scenes_feasible / n_records if n_records else 0.0,
            ))
            log(
                f"ncp={ncp_threshold} freq={invariance_frequency_threshold}: "
                f"consensus={n_consensus_pixels} px, feasible scenes={n_scenes_feasible}/{n_records}"
            )

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="original scene folder (for re-reading pixel values)")
    parser.add_argument("--output", required=True, help="completed psnorm_output dir (candidates/, masks/)")
    parser.add_argument("--ncp-thresholds", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--freq-thresholds", default="0.2,0.3,0.4,0.5")
    parser.add_argument("--min-observations", type=int, default=2)
    parser.add_argument("--min-fit-pixels", type=int, default=500)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N target scenes (smoke test)")
    parser.add_argument("--progress-every", type=int, default=20, help="print a progress line every N scenes")
    parser.add_argument("--save-rasters", default=None,
                         help="directory to write times_evaluated/times_invariant/frequency.tif (per ncp_threshold) "
                              "and consensus_mask.tif (per ncp_threshold x freq_threshold) into -- see run_sweep's "
                              "docstring for layout. Omit to skip raster output (CSV summary only).")
    args = parser.parse_args()

    ncp_grid = [float(v) for v in args.ncp_thresholds.split(",")]
    freq_grid = [float(v) for v in args.freq_thresholds.split(",")]
    summary_csv = args.summary_csv or os.path.join(args.output, "threshold_sweep.csv")

    device = backend.resolve_device(args.device)
    _log(f"Device: {device}")

    records, reference_id, skipped_no_fit = discover_target_records(args.output, args.input, limit=args.limit)
    if not records:
        raise SystemExit("No target scenes with saved IR-MAD fits found — nothing to sweep.")

    ref_info = io.get_raster_info(records[0]["reference_path"])
    ref_shape = (ref_info.height, ref_info.width)

    if args.save_rasters:
        os.makedirs(args.save_rasters, exist_ok=True)
        _log(f"Rasters (times_evaluated/times_invariant/frequency/consensus_mask) will be written to: {args.save_rasters}")

    start = time.time()
    rows = run_sweep(
        records, ref_shape, ncp_grid, freq_grid,
        min_observations=args.min_observations, min_fit_pixels=args.min_fit_pixels, device=device,
        progress_every=args.progress_every,
        raster_output_dir=args.save_rasters, reference_info=ref_info,
    )

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _log(f"\nDone in {time.time() - start:.1f}s. Summary written to: {summary_csv}")


if __name__ == "__main__":
    main()
