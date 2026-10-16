#!/usr/bin/env python3
"""Register every raw PlanetScope scene to its temporally nearest summer-
solstice median composite (psnorm.registration.register_scene_models),
writing registered scenes plus registration metadata.

One warp model (--registration-warp-model into --output-dir) or two
(--also-warp-model into --also-output-dir): with two, each scene is read
and matched once and both warps are applied to the same matched keypoints,
which costs about a third more than one model rather than twice as much.
Each output directory is complete in itself and laid out as below.

For each scene under --raw-root's philadelphia4b/philadelphia8b period
folders (the layout scripts/build_solstice_composite_reference.py reads),
the reference is the composite in --composite-dir (solstice_<year>.tif)
whose solstice instant is closest to the scene's acquisition time.

Outputs, under --output-dir:
  <product>/<period folder>/<same file name as the input>   registered scene
      (same data type, band order/descriptions and nodata as the input,
      so the tree can stand in for the raw archive)
  registration_manifest.csv   one row per scene: reference used, status,
      warp strategy, cloud-mask clear fraction, timings, sizes, the key
      quality numbers, and the fitted affine transform (the six matrix
      entries in the scene's own pixels, plus its shift at the scene
      centre, scale, rotation and shear -- see
      planet_georeg_opencv.tps_warp.describe_affine). For the affine
      model that transform is the warp applied; for TPS it is the global
      fit the control points were selected with.
  alignment_metrics/registration_metrics_<n>band.csv   planet_georeg_
      opencv's own per-scene metrics (keypoint counts, per-band R^2/RMSE/
      scale/intercept before and after, mean shift), via its
      write_alignment_metrics_csv; split by band count because its columns
      are per band index
  run_params.json   the RegistrationParams and command line of the latest start
  run_params_history.jsonl  the same for every start

Resumable: scenes already in a model's manifest are skipped for that model
(pass --retry-failed to re-run the ones that failed).

Parallelism is memory-aware: a scene starts only while the estimated
memory of everything running fits --memory-budget-gb (a scene too big for
the budget on its own runs alone). If a worker is killed anyway (e.g. by
the kernel's OOM killer), the scenes it was running are retried one at a
time; a scene that also crashes alone is recorded as "worker_crashed".

Usage:
    python scripts/register_to_solstice_composites.py \\
        --raw-root /mnt/e/E_Coast_PlanetScope/philadelphia/raw \\
        --composite-dir /mnt/e/E_Coast_PlanetScope/philadelphia/summer_solstice_composites \\
        --output-dir /mnt/e/E_Coast_PlanetScope/philadelphia/affine_transform \\
        --registration-warp-model affine \\
        --also-warp-model tps \\
        --also-output-dir /mnt/e/E_Coast_PlanetScope/philadelphia/tps_transform
"""

from __future__ import annotations

import os

# Per-process memory, set before GDAL/numpy load (worker processes inherit
# the environment): this process holds two GDAL copies (osgeo's and
# rasterio's bundled one), each defaulting to a block cache of 5% of RAM;
# and glibc otherwise keeps freed multi-MB arrays mapped, inflating RSS.
os.environ.setdefault("GDAL_CACHEMAX", "64")
os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", "1048576")
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "1048576")

import argparse
import concurrent.futures as cf
from concurrent.futures.process import BrokenProcessPool
import csv
import fcntl
import glob
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timedelta

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_solstice_composite_reference import (  # noqa: E402
    EIGHT_BAND_SUFFIX, FOUR_BAND_SUFFIX, UDM2_SUFFIX, summer_solstice,
)
from planet_georeg_opencv import band_metrics, nd_index, tps_warp  # noqa: E402

from psnorm import compositing, io, registration  # noqa: E402

PRODUCTS = {"philadelphia4b": FOUR_BAND_SUFFIX, "philadelphia8b": EIGHT_BAND_SUFFIX}

MAX_CONSECUTIVE_SOLO_CRASHES = 3
LOOKAHEAD = 200       # how far past a non-fitting head scene to look for one that fits
MAX_HEAD_SKIPS = 12   # how many scenes may jump ahead of it before it gets priority

# Set by main() for the worker processes: how many threads each may use.
WORKER_THREADS_ENV = "PSNORM_REGISTER_WORKER_THREADS"

MANIFEST_FIELDS = [
    "scene_id", "product", "acquired", "analytic_path", "output_path", "reference_path",
    "reference_year", "days_from_solstice", "status", "message", "strategy", "warp_model",
    "band_count", "udm2_mask_used", "udm2_clear_fraction", "width", "height", "megapixels",
    "input_bytes", "seconds", "coarse_ransac_inliers", "guided_ransac_inliers",
    "mean_kp_dist_before", "mean_kp_dist_after", "avg_dx_px", "avg_dy_px",
    # Detection scale of the composite's stored keypoints the scene was
    # matched against; empty if the composite's keypoints were detected for
    # that scene alone (as in every run before the keypoint cache existed).
    "reference_keypoint_scale",
    # The fitted affine transform (see the module docstring).
    *tps_warp.AFFINE_METRIC_KEYS,
    # The warp models this scene was registered with in one pass (e.g.
    # "affine+tps"); `seconds` is the time for all of them together.
    "models_together",
]


# --------------------------------------------------------------------------
# Work list
# --------------------------------------------------------------------------

def find_composites(composite_dir: str, hemisphere: str) -> list[tuple[int, datetime, str]]:
    out = []
    for path in glob.glob(os.path.join(composite_dir, "solstice_*.tif")):
        m = re.fullmatch(r"solstice_(\d{4})\.tif", os.path.basename(path))
        if m:
            year = int(m.group(1))
            out.append((year, summer_solstice(year, hemisphere), path))
    if not out:
        raise SystemExit(f"No solstice_<year>.tif composites in {composite_dir}")
    return sorted(out)


def nearest_composite(acquired: datetime, composites):
    year, solstice, path = min(composites, key=lambda c: abs(acquired - c[1]))
    return year, (acquired - solstice) / timedelta(days=1), path


def build_tasks(raw_root, output_dir, composites, products):
    tasks = []
    for product in products:
        for scene in compositing.discover_product_scenes(raw_root, product, PRODUCTS[product], UDM2_SUFFIX):
            if scene.acquired is None:
                print(f"  WARNING: no acquisition time in '{scene.scene_id}'; skipping", flush=True)
                continue
            info = io.get_raster_info(scene.analytic_path)
            year, days, ref_path = nearest_composite(scene.acquired, composites)
            period = os.path.basename(os.path.dirname(scene.analytic_path))
            tasks.append({
                "scene_id": scene.scene_id, "product": product, "acquired": scene.acquired.isoformat(),
                "analytic_path": scene.analytic_path, "udm2_path": scene.udm2_path,
                "output_path": os.path.join(output_dir, product, period, os.path.basename(scene.analytic_path)),
                "output_relpath": os.path.join(product, period, os.path.basename(scene.analytic_path)),
                "reference_path": ref_path, "reference_year": year, "days_from_solstice": round(days, 2),
                "width": info.width, "height": info.height, "band_count": info.band_count,
                "megapixels": info.width * info.height / 1e6,
                "input_bytes": os.path.getsize(scene.analytic_path),
            })
    tasks.sort(key=lambda t: t["acquired"])
    return tasks


def large_last(tasks, large_mpx):
    """Chronological order, except scenes over large_mpx go last -- so a
    big scene never holds up the queue waiting for memory, and the big
    ones run together at the end when little else is competing."""
    return [t for t in tasks if t["megapixels"] <= large_mpx] + [t for t in tasks if t["megapixels"] > large_mpx]


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

class FileLock:
    """Inter-process exclusive lock (flock on a local file), used so only one
    worker at a time is inside SIFT detection -- see pgo.register_local's
    sift_lock. Holds only a path, so it can be created in each worker."""

    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        self._file = open(self.path, "w")
        fcntl.flock(self._file, fcntl.LOCK_EX)

    def __exit__(self, *exc):
        fcntl.flock(self._file, fcntl.LOCK_UN)
        self._file.close()


def _reset_peak_rss() -> None:
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")  # resets this process's peak-RSS (VmHWM) counter
    except OSError:
        pass


def _peak_rss_gb() -> float | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    return None


def register_one(task: dict, params: registration.RegistrationParams, sift_lock_path: str | None = None) -> dict:
    """Runs in a worker process: registers one scene with every warp model
    in task["warp_outputs"] ({model: output path}). Never raises: failures
    come back as a status/message in the returned rows (one per model)."""
    _reset_peak_rss()
    t0 = time.time()
    threads = int(os.environ.get(WORKER_THREADS_ENV) or 0)
    if threads:
        cv2.setNumThreads(threads)
    models = list(task["warp_outputs"])
    base_row = {k: task[k] for k in MANIFEST_FIELDS if k in task}
    rows = {model: {**base_row, "warp_model": model, "output_path": task["warp_outputs"][model]} for model in models}
    metrics_out = {model: None for model in models}
    try:
        # Each output is written band by band as the scene is warped, and
        # only if registration succeeds (register_scene_models re-reads
        # it, and rewrites it once if that fails).
        for path in task["warp_outputs"].values():
            os.makedirs(os.path.dirname(path), exist_ok=True)
        scenes = registration.register_scene_models(
            task["analytic_path"], task["reference_path"], task["warp_outputs"], params,
            target_udm2_path=task["udm2_path"], sift_lock=FileLock(sift_lock_path) if sift_lock_path else None,
        )
        for model, reg in scenes.items():
            result, row = reg.result, rows[model]
            row.update(
                status=result.status, message=result.message, strategy=reg.strategy,
                udm2_mask_used=reg.target_mask_used,
                udm2_clear_fraction=None if reg.target_clear_fraction is None else round(reg.target_clear_fraction, 4),
                reference_keypoint_scale=reg.reference_keypoint_scale,
            )
            for key in ("coarse_ransac_inliers", "guided_ransac_inliers"):
                row[key] = result.diagnostics.get(key)
            for key in ("mean_kp_dist_before", "mean_kp_dist_after", "avg_dx_px", "avg_dy_px"):
                value = result.metrics.get(key)
                row[key] = None if value is None or not np.isfinite(value) else round(float(value), 4)
            for key in tps_warp.AFFINE_METRIC_KEYS:
                # Full precision: over a scene thousands of pixels across,
                # the sixth decimal of a matrix entry is a hundredth of a pixel.
                value = result.metrics.get(key)
                row[key] = None if value is None or not np.isfinite(value) else float(value)
            metrics_out[model] = {
                "band_map": reg.band_map,
                "metrics": {k: (float(v) if v is not None else np.nan) for k, v in result.metrics.items()},
                "coarse": result.diagnostics.get("coarse_ransac_inliers", 0),
                "guided": result.diagnostics.get("guided_ransac_inliers", 0),
            }
    except registration.RegistrationFailed as exc:
        for row in rows.values():
            row.update(status="registration_failed", message=str(exc))
    except Exception as exc:  # recorded per scene rather than aborting the batch
        for row in rows.values():
            row.update(status="error", message=f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=3)!r}")
    seconds = round(time.time() - t0, 1)
    for row in rows.values():
        # The whole scene's time, for every model run with it (see
        # models_together): matching is shared, so it cannot be split.
        row["seconds"] = seconds
        row["models_together"] = "+".join(models)
    peak = _peak_rss_gb()
    return {"rows": rows, "library_metrics": metrics_out, "peak_rss_gb": None if peak is None else round(peak, 3)}


# --------------------------------------------------------------------------
# Parent: bookkeeping
# --------------------------------------------------------------------------

class Recorder:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.manifest_path = os.path.join(output_dir, "registration_manifest.csv")
        self.metrics_dir = os.path.join(output_dir, "alignment_metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        self._add_missing_manifest_columns()

    def _add_missing_manifest_columns(self) -> None:
        """Bring a manifest written before MANIFEST_FIELDS gained a column
        up to the current header (new columns empty for existing rows), so
        appended rows line up. Rewritten in place after a backup copy --
        not via a rename, which is unreliable on the output mount."""
        if not os.path.exists(self.manifest_path):
            return
        with open(self.manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames == MANIFEST_FIELDS:
                return
            rows = list(reader)
        shutil.copy(self.manifest_path, self.manifest_path + f".before_{datetime.now():%Y%m%d_%H%M%S}")
        with open(self.manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def previously_recorded(self) -> dict[str, str]:
        if not os.path.exists(self.manifest_path):
            return {}
        with open(self.manifest_path, newline="") as f:
            return {r["scene_id"]: r["status"] for r in csv.DictReader(f)}

    def record(self, row: dict, library_metrics: dict | None, peak_rss_gb) -> None:
        """Append one scene's row for this recorder's warp model.
        `peak_rss_gb` is None for all but one of a scene's models, so the
        scene's memory is logged once."""
        new_file = not os.path.exists(self.manifest_path)
        with open(self.manifest_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)

        # Measured per-scene peak memory, for tuning the scheduler's estimate.
        if peak_rss_gb is not None:
            memory_path = os.path.join(self.output_dir, "worker_memory.csv")
            new_memory_file = not os.path.exists(memory_path)
            with open(memory_path, "a") as f:
                if new_memory_file:
                    f.write("scene_id,band_count,megapixels,peak_rss_gb,seconds,status,models_together\n")
                f.write(f"{row['scene_id']},{row['band_count']},{row['megapixels']:.2f},"
                        f"{peak_rss_gb},{row.get('seconds')},{row['status']},{row.get('models_together')}\n")

        lm = library_metrics
        if lm is None:
            return
        pairs = nd_index.get_matched_band_pairs(lm["band_map"])
        path = os.path.join(self.metrics_dir, f"registration_metrics_{row['band_count']}band.csv")
        band_metrics.write_alignment_metrics_csv(
            path, band_metrics.build_metric_keys(pairs),
            os.path.basename(row["analytic_path"]), os.path.basename(row["reference_path"]),
            lm["coarse"], 0, 0, lm["guided"], 0, 0, lm["metrics"],
        )


def record_outcome(recorders: dict, outcome: dict) -> None:
    """Record a register_one outcome: each model's row in that model's
    output directory (`recorders` is {model: Recorder})."""
    for i, (model, row) in enumerate(outcome["rows"].items()):
        recorders[model].record(row, outcome["library_metrics"].get(model),
                                outcome.get("peak_rss_gb") if i == 0 else None)


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


class Progress:
    def __init__(self, tasks, interval_s):
        self.total_n = len(tasks)
        self.total_mpx = sum(t["megapixels"] for t in tasks)
        self.total_gb = sum(t["input_bytes"] for t in tasks) / 1e9
        self.done_n = self.done_mpx = self.done_gb = 0.0
        self.status_counts: dict[str, int] = {}
        self.start = self.last = time.time()
        self.interval_s = interval_s

    def add(self, task, status):
        self.done_n += 1
        self.done_mpx += task["megapixels"]
        self.done_gb += task["input_bytes"] / 1e9
        self.status_counts[status] = self.status_counts.get(status, 0) + 1

    def maybe_report(self, running: int, force=False):
        now = time.time()
        if not force and now - self.last < self.interval_s:
            return
        self.last = now
        elapsed = now - self.start
        rate = self.done_mpx / elapsed if elapsed > 0 else 0.0
        remaining = (self.total_mpx - self.done_mpx) / rate if rate > 0 else float("nan")
        eta = (datetime.now() + timedelta(seconds=remaining)).strftime("%a %H:%M") if rate > 0 else "?"
        statuses = ", ".join(f"{k}={v}" for k, v in sorted(self.status_counts.items()))
        print(
            f"[progress] {int(self.done_n)}/{self.total_n} scenes ({statuses}) | "
            f"{self.done_gb:.1f}/{self.total_gb:.1f} GB input, {self.done_mpx:,.0f}/{self.total_mpx:,.0f} Mpx "
            f"({100 * self.done_mpx / max(self.total_mpx, 1e-9):.1f}%) | elapsed {fmt_duration(elapsed)} | "
            f"{rate * 60:,.0f} Mpx/min | {running} running | "
            f"remaining ~{fmt_duration(remaining) if rate > 0 else '?'} (ETA {eta})",
            flush=True,
        )


# --------------------------------------------------------------------------
# Scheduler
# --------------------------------------------------------------------------

_last_cache_drop = 0.0


def drop_fs_caches(reason: str) -> None:
    """Drop the kernel's cached directory entries/inodes (needs passwordless
    sudo; silently does nothing without it), at most once a minute.

    On the virtiofs mounts this project runs on, a file occasionally turns
    unreadable through one path ("Bad file descriptor", shown as ????? by
    ls) while its data is intact -- a stale cached entry, cleared by
    dropping these caches. It has hit a composite and a module of the
    Python environment itself (which made every new worker die on import).
    Only cached metadata is discarded; nothing on disk changes.
    """
    global _last_cache_drop
    if time.time() - _last_cache_drop < 60:
        return
    _last_cache_drop = time.time()
    result = subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 2 > /proc/sys/vm/drop_caches"],
                            capture_output=True)
    print(f"[info] dropped filesystem metadata caches ({reason}); "
          f"{'ok' if result.returncode == 0 else 'not permitted'}", flush=True)


class CompositeReadiness:
    """Whether a reference composite is safe to read: not mid-rewrite by
    scripts/rewrite_composites.py (which overwrites composites in place --
    reading one then, including building its median cache, would read a
    half-written file). With require_rewritten, a composite still in the old
    strip layout also counts as not ready (the rewrite hasn't reached it
    yet). Re-checked at most every recheck_s seconds per composite."""

    def __init__(self, require_rewritten: bool, recheck_s: float = 120):
        self.require_rewritten = require_rewritten
        self.recheck_s = recheck_s
        self.state: dict[str, tuple[bool, float]] = {}

    def ready(self, path: str) -> bool:
        ok, checked = self.state.get(path, (False, 0.0))
        if ok or time.time() - checked < self.recheck_s:
            return ok
        ok = not os.path.exists(path + ".rewrite.tif")
        if ok and self.require_rewritten:
            try:
                from osgeo import gdal
                ds = gdal.Open(path)
                ok = ds.GetMetadataItem("INTERLEAVE", "IMAGE_STRUCTURE") == "BAND" and \
                    ds.GetRasterBand(1).GetBlockSize()[1] > 1
            except RuntimeError:
                ok = False
                drop_fs_caches(f"could not open {os.path.basename(path)}")
        self.state[path] = (ok, time.time())
        if ok:
            print(f"[info] reference composite now available: {os.path.basename(path)}", flush=True)
        return ok


def estimate_gb(task, base_gb, gb_per_mpx):
    return base_gb + gb_per_mpx * task["megapixels"]


def read_limits(limits_file, workers, budget_gb):
    """(workers, budget_gb), overridden by `limits_file` if it exists: a JSON
    object with either or both keys, e.g. {"workers": 3, "memory_budget_gb": 6}.
    Lets the limits be tuned against observed memory without restarting."""
    try:
        with open(limits_file) as f:
            limits = json.load(f)
        return int(limits.get("workers", workers)), float(limits.get("memory_budget_gb", budget_gb))
    except (OSError, ValueError, TypeError):
        return workers, budget_gb


def run(tasks, params, recorders, progress, workers, budget_gb, base_gb, gb_per_mpx, readiness, sift_lock_path,
        limits_file=None, max_workers=8):
    pending = list(tasks)
    start_workers, start_budget = workers, budget_gb
    solo_retry: list[dict] = []  # scenes whose worker died; rerun one at a time
    crashed_once: set[str] = set()

    def new_pool():
        # Sized for the most workers a limits file may ask for (each idle
        # pooled process holds ~0.2 GB, so not generously); how many run at
        # once is decided by the submission loop below.
        # A fresh process per scene: a reused worker's memory grew from
        # scene to scene (measured: 5-7 GB peaks at sizes that take ~2 GB in
        # a fresh process), which the memory budget cannot account for.
        return cf.ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=1)

    pool = new_pool()
    running: dict[cf.Future, tuple[dict, float]] = {}
    consecutive_solo_crashes = 0
    head_skips = 0
    error_retried: set[str] = set()  # scenes re-queued once after an in-worker exception
    try:
        while pending or solo_retry or running:
            if limits_file:
                new_limits = read_limits(limits_file, start_workers, start_budget)
                new_limits = (min(new_limits[0], max_workers), new_limits[1])
                if new_limits != (workers, budget_gb):
                    workers, budget_gb = new_limits
                    print(f"[info] limits now: workers={workers}, memory budget={budget_gb} GB", flush=True)
            # Submit while the memory budget allows (always at least one).
            if solo_retry:
                if not running:
                    task = solo_retry.pop(0)
                    running[pool.submit(register_one, task, params, sift_lock_path)] = (task, budget_gb)
            else:
                while len(running) < workers:
                    # The first ready scene (reference composite available)
                    # is the head of the queue. If it doesn't fit in the
                    # memory budget next to what's running, look a little
                    # further for one that does rather than idling -- but
                    # only MAX_HEAD_SKIPS times in a row, after which
                    # nothing new starts until the head scene has run, so a
                    # big scene can't be starved by a stream of small ones.
                    in_use = sum(gb for _, gb in running.values())
                    idx = head = None
                    for i, t in enumerate(pending):
                        if not readiness.ready(t["reference_path"]):
                            continue
                        if head is None:
                            head = i
                        need = estimate_gb(t, base_gb, gb_per_mpx)
                        if not running or in_use + need <= budget_gb:
                            idx = i
                            break
                        if head_skips >= MAX_HEAD_SKIPS or i - head >= LOOKAHEAD:
                            break
                    if idx is None:
                        break
                    head_skips = 0 if idx == head else head_skips + 1
                    task = pending.pop(idx)
                    need = estimate_gb(task, base_gb, gb_per_mpx)
                    running[pool.submit(register_one, task, params, sift_lock_path)] = (task, need)

            if not running:
                time.sleep(30)  # everything left is waiting on a composite
                progress.maybe_report(0)
                continue
            done, _ = cf.wait(list(running), timeout=30, return_when=cf.FIRST_COMPLETED)
            broken = False
            for fut in done:
                task, _ = running.pop(fut)
                try:
                    outcome = fut.result()
                    consecutive_solo_crashes = 0
                    first_row = next(iter(outcome["rows"].values()))
                    if first_row["status"] == "error" and task["scene_id"] not in error_retried:
                        # Possibly an input that went unreadable (see
                        # drop_fs_caches): clear the caches and try once more
                        # before recording the scene as failed.
                        error_retried.add(task["scene_id"])
                        drop_fs_caches(f"scene {task['scene_id']} raised: {first_row['message'][:120]}")
                        pending.insert(0, task)
                        continue
                except BrokenProcessPool:
                    broken = True
                    if task["scene_id"] in crashed_once:
                        consecutive_solo_crashes += 1
                        if consecutive_solo_crashes >= MAX_CONSECUTIVE_SOLO_CRASHES:
                            # Scene after scene dying even alone is an environment
                            # failure (e.g. a library file that stopped being
                            # readable), not a property of the scenes: stop
                            # rather than record the whole queue as crashed.
                            print(f"[error] {consecutive_solo_crashes} scenes in a row crashed their worker even when "
                                  f"run alone; stopping. Nothing was recorded for the latest one ({task['scene_id']}).",
                                  flush=True)
                            raise SystemExit(3)
                        outcome = {"rows": {model: {**{k: task[k] for k in MANIFEST_FIELDS if k in task},
                                                    "status": "worker_crashed", "warp_model": model,
                                                    "output_path": path,
                                                    "message": "worker process died even when run alone "
                                                               "(likely out of memory)"}
                                            for model, path in task["warp_outputs"].items()},
                                   "library_metrics": {}}
                    else:
                        crashed_once.add(task["scene_id"])
                        solo_retry.append(task)
                        continue
                record_outcome(recorders, outcome)
                progress.add(task, "/".join(dict.fromkeys(row["status"] for row in outcome["rows"].values())))
            if broken:
                # Every other in-flight future of a broken pool fails too.
                for fut, (task, _) in list(running.items()):
                    crashed_once.add(task["scene_id"])
                    solo_retry.append(task)
                running.clear()
                pool.shutdown(wait=False, cancel_futures=True)
                drop_fs_caches("a worker process died")
                pool = new_pool()
                print(f"[warning] a worker process died; retrying {len(solo_retry)} scene(s) one at a time", flush=True)
            progress.maybe_report(len(running))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    progress.maybe_report(0, force=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--composite-dir", required=True)
    parser.add_argument("--output-dir", required=True,
                        help="output directory for --registration-warp-model")
    parser.add_argument("--also-warp-model", default=None, choices=["affine", "tps"],
                        help="a second warp model to apply to the same matched keypoints in the same pass")
    parser.add_argument("--also-output-dir", default=None, help="output directory for --also-warp-model")
    parser.add_argument("--products", default=",".join(PRODUCTS), help="comma-separated (default: both)")
    parser.add_argument("--hemisphere", default="N", choices=["N", "S"])
    parser.add_argument("--workers", type=int, default=4)
    # Measured with five 19-megapixel scenes at once on 24 cores: 166 s per
    # scene with every library using all cores (the thin-plate-spline fit
    # alone went from 2 s to 64 s), 78 s with 4 threads each.
    parser.add_argument("--threads-per-worker", type=int, default=4,
                        help="threads each worker's numerical libraries (OpenCV, BLAS, GDAL compression) may use; "
                             "workers x this should not exceed the machine's cores. 0 leaves the libraries' defaults")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="most workers a --limits-file may raise --workers to while running")
    # Measured over a 42-scene sample of this archive (fresh process per
    # scene, SIFT on 1024-pixel tiles, affine and TPS in one pass):
    #   peak GB ~= 0.46 + 0.092 * scene Mpx, at most 1.1 GB above that
    # The estimate below sits a little above the fit; concurrent scenes
    # rarely peak together, so the budget can be most of the machine's RAM.
    parser.add_argument("--memory-budget-gb", type=float, default=8.0,
                        help="estimated memory of concurrently running scenes may not exceed this")
    parser.add_argument("--base-gb", type=float, default=1.0, help="memory estimate: fixed part per scene")
    parser.add_argument("--gb-per-mpx", type=float, default=0.095, help="memory estimate: per megapixel")
    parser.add_argument("--sift-lock-file", default="",
                        help="local file used as a lock so only one worker at a time is inside SIFT detection. "
                             "Only worth setting with --registration-sift-tile-size none, when SIFT needs up to "
                             "about 4 GB at once (which --memory-budget-gb must then leave room for); with tiled "
                             "SIFT it would only make workers wait for each other")
    parser.add_argument("--large-mpx", type=float, default=60.0,
                        help="scenes larger than this run after all others (they need the machine nearly to themselves)")
    parser.add_argument("--limits-file", default=None,
                        help="optional JSON file re-read while running, overriding --workers / --memory-budget-gb, "
                             'e.g. {"workers": 3, "memory_budget_gb": 6}')
    parser.add_argument("--task-cache", default=None,
                        help="JSON file caching the discovered scene list (discovery reads every scene's header, "
                             "minutes on a slow mount); delete it if the raw archive or composites change")
    parser.add_argument("--limit", type=int, default=None, help="only the first N pending scenes (for testing)")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--require-rewritten-composites", action="store_true",
                        help="defer scenes whose reference composite is still in the old strip layout "
                             "(use while scripts/rewrite_composites.py is still working through them)")
    parser.add_argument("--progress-interval-s", type=float, default=120)
    registration.add_registration_arguments(parser)
    args = parser.parse_args()

    if args.threads_per_worker > 0:
        # Inherited by the worker processes, which read these as they
        # import their libraries (and in register_one, for OpenCV).
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "GDAL_NUM_THREADS", WORKER_THREADS_ENV):
            os.environ[name] = str(args.threads_per_worker)

    params = registration.registration_params_from_args(args)
    if (args.also_warp_model is None) != (args.also_output_dir is None):
        parser.error("--also-warp-model and --also-output-dir go together")
    if args.also_warp_model == params.warp_model:
        parser.error(f"--also-warp-model is the same as --registration-warp-model ({params.warp_model})")
    # {warp model: its output directory}; the first is the primary one.
    output_dirs = {params.warp_model: args.output_dir}
    if args.also_warp_model:
        output_dirs[args.also_warp_model] = args.also_output_dir
    if len({os.path.abspath(d) for d in output_dirs.values()}) != len(output_dirs):
        parser.error("each warp model needs its own output directory")

    for model, output_dir in output_dirs.items():
        os.makedirs(output_dir, exist_ok=True)
        run_record = {"registration_params": asdict(replace(params, warp_model=model)), "argv": sys.argv,
                      "warp_models_this_run": list(output_dirs), "started": datetime.now().isoformat()}
        with open(os.path.join(output_dir, "run_params.json"), "w") as f:
            json.dump(run_record, f, indent=2)
        # run_params.json describes the latest start only; the history keeps
        # every start, since an output directory can be filled over several
        # runs whose parameters differ.
        with open(os.path.join(output_dir, "run_params_history.jsonl"), "a") as f:
            f.write(json.dumps(run_record) + "\n")

    composites = find_composites(args.composite_dir, args.hemisphere)
    print(f"Composites: {', '.join(str(y) for y, _, _ in composites)}", flush=True)
    if args.task_cache and os.path.exists(args.task_cache):
        with open(args.task_cache) as f:
            tasks = json.load(f)
        # Cached entries are independent of the output directory.
        for t in tasks:
            t["output_path"] = os.path.join(args.output_dir, t["output_relpath"])
        print(f"Loaded {len(tasks)} scenes from task cache {args.task_cache}", flush=True)
    else:
        tasks = build_tasks(args.raw_root, args.output_dir, composites, args.products.split(","))
        if args.task_cache:
            with open(args.task_cache, "w") as f:
                json.dump(tasks, f)

    recorders = {model: Recorder(output_dir) for model, output_dir in output_dirs.items()}
    recorded = {model: recorder.previously_recorded() for model, recorder in recorders.items()}
    # A scene runs with the models it is not yet recorded for (or, with
    # --retry-failed, not yet recorded as ok for).
    pending = []
    for t in tasks:
        missing = [
            model for model in output_dirs
            if recorded[model].get(t["scene_id"]) is None
            or (args.retry_failed and recorded[model][t["scene_id"]] != "ok")
        ]
        if missing:
            pending.append({**t, "warp_outputs": {
                model: os.path.join(output_dirs[model], t["output_relpath"]) for model in missing}})
    pending = large_last(pending, args.large_mpx)
    n_unrecorded = len(pending)
    if args.limit is not None:
        pending = pending[: args.limit]
    by_models = {}
    for t in pending:
        key = "+".join(t["warp_outputs"])
        by_models[key] = by_models.get(key, 0) + 1
    print(f"{len(tasks)} scenes found; {len(tasks) - n_unrecorded} already recorded for every model; "
          f"{len(pending)} to run ({', '.join(f'{n} for {k}' for k, n in by_models.items()) or 'none'})"
          f"{' (--limit)' if len(pending) < n_unrecorded else ''}. Params: {params}", flush=True)

    progress = Progress(pending, args.progress_interval_s)
    run(pending, params, recorders, progress, args.workers, args.memory_budget_gb, args.base_gb, args.gb_per_mpx,
        CompositeReadiness(args.require_rewritten_composites), args.sift_lock_file or None,
        limits_file=args.limits_file, max_workers=max(args.max_workers, args.workers))
    print(f"DONE {datetime.now().isoformat()}", flush=True)


if __name__ == "__main__":
    main()
