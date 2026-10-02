#!/usr/bin/env python3
"""Run the psnorm pipeline against a wide AOI using MULTIPLE automatically
selected references, so every scene gets corrected against whichever
reference actually overlaps it (a single reference's own footprint may not
reach every scene across a wide area-of-interest).

Usage:
    .venv/bin/python scripts/run_multi_reference_pipeline.py --input DIR --output DIR [--scene-log FILE] \\
        [--no-omnicloudmask] [--workers N] [--resume no|yes|validate] [--device auto|cpu|gpu] \\
        [--ncp-threshold F] [--invariance-frequency-threshold F] \\
        [--dsm-path PATH] [--dsm-height-units m|ft] [--max-slope-deg F] \\
        [--roughness-window-radius-px N] [--roughness-max-deg F] \\
        [--use-shadow-mask] [--max-building-height-m F] [--shadow-ray-step-m F] \\
        [--mask-erode-px N] [--mask-dilate-px N]
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psnorm import io as pio
from psnorm import pipeline

MAX_ATTEMPTS = 150
RETRY_DELAY_S = 20.0
RECONCILIATION_PASSES = 2


def log(msg):
    print(msg, flush=True)


def _run_pipeline_kwargs(args):
    return dict(
        use_omnicloudmask=not args.no_omnicloudmask,
        resume=args.resume,
        device=args.device,
        ncp_threshold=args.ncp_threshold,
        invariance_frequency_threshold=args.invariance_frequency_threshold,
        dsm_path=args.dsm_path,
        dsm_height_units=args.dsm_height_units,
        max_slope_deg=args.max_slope_deg,
        roughness_window_radius_px=args.roughness_window_radius_px,
        roughness_max_deg=args.roughness_max_deg,
        use_shadow_mask=args.use_shadow_mask,
        max_building_height_m=args.max_building_height_m,
        shadow_ray_step_m=args.shadow_ray_step_m,
        mask_erode_px=args.mask_erode_px,
        mask_dilate_px=args.mask_dilate_px,
    )


def _run_group_with_retries(args, workers, all_scenes, reference, group_scenes):
    """Run (or resume) the pipeline for one reference group, retrying on
    transient I/O errors (mount hiccups, transient permission/file-descriptor
    faults) rather than aborting the whole multi-reference run over a
    single group's flaky attempt."""
    ref_id = reference.scene_id
    output_dir = os.path.join(args.output, f"ref_{ref_id}")
    scene_ids = {s.scene_id for s in group_scenes}
    attempt = 1
    while attempt <= MAX_ATTEMPTS:
        log(f"=== reference group {ref_id}: attempt {attempt}/{MAX_ATTEMPTS} ({len(group_scenes)} scenes) ===")
        try:
            result = pipeline.run_pipeline(
                args.input, output_dir,
                reference_path=reference.analytic_path,
                scenes=all_scenes,
                scene_ids=scene_ids,
                workers=workers,
                log=log,
                **_run_pipeline_kwargs(args),
            )
            return result
        except Exception as exc:
            log(f"  attempt {attempt} FAILED: {exc!r}; retrying after {RETRY_DELAY_S}s")
            time.sleep(RETRY_DELAY_S)
            attempt += 1
    log(f"  giving up on reference group {ref_id} after {MAX_ATTEMPTS} attempts")
    return None


def _best_overlap_reference(scene, candidates, bounds_map):
    sb = bounds_map[scene.scene_id]
    best_ref, best_area = None, 0.0
    for r in candidates:
        rb = bounds_map[r.scene_id]
        ox0, oy0 = max(sb[0], rb[0]), max(sb[1], rb[1])
        ox1, oy1 = min(sb[2], rb[2]), min(sb[3], rb[3])
        area = max(0.0, ox1 - ox0) * max(0.0, oy1 - oy0)
        if area > best_area:
            best_ref, best_area = r, area
    return best_ref


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-log", default=None,
                         help="read candidate scene IDs from a prior run's log instead of listing --input's "
                              "directory (see psnorm.io.scene_ids_from_log)")
    parser.add_argument("--no-omnicloudmask", action="store_true")
    parser.add_argument("--workers", default="cpu")
    parser.add_argument("--resume", default="validate", choices=["no", "yes", "validate"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--ncp-threshold", type=float, default=0.70)
    parser.add_argument("--invariance-frequency-threshold", type=float, default=0.5)
    parser.add_argument("--dsm-path", default=None)
    parser.add_argument("--dsm-height-units", default="m", choices=["m", "ft"])
    parser.add_argument("--max-slope-deg", type=float, default=5.0)
    parser.add_argument("--roughness-window-radius-px", type=int, default=2)
    parser.add_argument("--roughness-max-deg", type=float, default=10.0)
    parser.add_argument("--use-shadow-mask", action="store_true")
    parser.add_argument("--max-building-height-m", type=float, default=150.0)
    parser.add_argument("--shadow-ray-step-m", type=float, default=None)
    parser.add_argument("--mask-erode-px", type=int, default=1)
    parser.add_argument("--mask-dilate-px", type=int, default=1)
    parser.add_argument("--max-references", type=int, default=30)
    args = parser.parse_args()

    workers = args.workers
    if workers in ("cpu", "none", None):
        workers = None if workers == "none" else "cpu"
    else:
        workers = int(workers)

    if args.scene_log:
        scene_ids = pio.scene_ids_from_log(args.scene_log)
        log(f"Read {len(scene_ids)} candidate scene IDs from '{args.scene_log}'.")
        scenes = pio.discover_scenes_from_ids(args.input, scene_ids, log=log)
        log(f"Discovered {len(scenes)} scenes total.")
    else:
        scenes = pio.discover_scenes(args.input)
        log(f"Discovered {len(scenes)} scenes total.")
    if not scenes:
        raise ValueError(f"No scenes found for input '{args.input}'.")

    log("Reading every scene's footprint (this is the only pass over all scenes' headers)...")
    bounds = pipeline.scene_bounds_map(scenes)

    references = pipeline.select_multi_references(
        scenes, scene_bounds=bounds, max_references=args.max_references, log=log,
    )
    log(f"Selected {len(references)} reference(s) covering the AOI.")

    groups = pipeline.assign_scenes_to_references(scenes, references, scene_bounds=bounds, log=log)

    group_results = {}
    for reference in references:
        group_scenes = groups.get(reference.scene_id, [])
        if not group_scenes:
            continue
        result = _run_group_with_retries(args, workers, scenes, reference, group_scenes)
        if result is not None:
            group_results[reference.scene_id] = result

    tried = {ref_id: {ref_id} for ref_id in group_results}
    for pass_n in range(1, RECONCILIATION_PASSES + 1):
        reassignments = {}  # new_ref_id -> set of scene_ids to add
        leftover_scene_objs = {}
        for ref_id, result in list(group_results.items()):
            reference = next(r for r in references if r.scene_id == ref_id)
            leftover = [
                sr for sr in result.scene_results
                if sr.status == "skipped_insufficient_overlap"
            ]
            if not leftover:
                continue
            by_id = {s.scene_id: s for s in scenes}
            for sr in leftover:
                scene = by_id.get(sr.scene_id)
                if scene is None:
                    continue
                already_tried = tried.setdefault(sr.scene_id, {ref_id})
                already_tried.add(ref_id)
                candidates = [r for r in references if r.scene_id not in already_tried]
                new_ref = _best_overlap_reference(scene, candidates, bounds)
                if new_ref is None:
                    continue
                reassignments.setdefault(new_ref.scene_id, set()).add(sr.scene_id)
                leftover_scene_objs[sr.scene_id] = scene
                tried[sr.scene_id] = already_tried | {new_ref.scene_id}

        if not reassignments:
            break

        n_scenes = sum(len(v) for v in reassignments.values())
        log(f"Reconciliation round {pass_n}: reassigning {n_scenes} scene(s) across "
            f"{len(reassignments)} batch(es)...")

        for new_ref_id, scene_ids_to_add in reassignments.items():
            reference = next(r for r in references if r.scene_id == new_ref_id)
            existing = groups.get(new_ref_id, [])
            existing_ids = {s.scene_id for s in existing}
            merged = list(existing) + [
                leftover_scene_objs[sid] for sid in scene_ids_to_add if sid not in existing_ids
            ]
            groups[new_ref_id] = merged
            result = _run_group_with_retries(args, workers, scenes, reference, merged)
            if result is not None:
                group_results[new_ref_id] = result

    final_by_scene = {}
    for result in group_results.values():
        for sr in result.scene_results:
            final_by_scene[sr.scene_id] = sr.status

    status_counts = Counter(final_by_scene.values())
    manifest = {
        "references": [r.scene_id for r in references],
        "groups": {
            ref_id: {
                "reference_scene_id": result.reference_scene_id,
                "n_scenes": len(result.scene_results),
                "n_consensus_pixels": result.n_consensus_pixels,
                "output_dir": result.output_dir,
            }
            for ref_id, result in group_results.items()
        },
        "status_counts": dict(status_counts),
    }
    manifest_path = os.path.join(args.output, "multi_reference_manifest.json")
    os.makedirs(args.output, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"Wrote manifest: {manifest_path}")
    log(f"Final status counts: {dict(status_counts)}")


if __name__ == "__main__":
    main()
