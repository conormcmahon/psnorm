"""Orchestration: discover scenes, auto-select a reference, then three
barrier-separated phases —

  A. per-target invariant-target *candidate* detection (IR-MAD vs. the
     reference only — never target-vs-target), persisted to disk
  B. cross-scene consensus: a candidate location is kept only if it was
     flagged invariant in more than `invariance_frequency_threshold` of the
     comparisons that actually evaluated it (masked comparisons don't count)
  C. final per-target regression fit against the consensus set, then
     full-scene application (unmasked — every output pixel is corrected)

— and finally reports R^2/RMSE agreement between chronologically adjacent
scene pairs before vs after normalization.
"""

from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Literal

import numpy as np

from . import apply, backend, consensus, io, lidar_masks, masking, metrics, model_io, normalize, registration, sensors
from .normalize import BandModel

# --------------------------------------------------------------------------
# Reference-scene selection
# --------------------------------------------------------------------------


def _scene_centroid_lat_lon(scene: io.Scene) -> tuple[float, float] | None:
    geom = scene.metadata().get("geometry")
    if not geom or geom.get("type") != "Polygon" or not geom.get("coordinates"):
        return None
    ring = geom["coordinates"][0]
    lats = [pt[1] for pt in ring]
    lons = [pt[0] for pt in ring]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _acquired_utc(scene: io.Scene) -> datetime | None:
    """Prefer metadata.json's full-precision `acquired` timestamp; fall back
    to the filename-parsed timestamp (see io.parse_acquisition_time)."""
    acquired_str = scene.metadata().get("properties", {}).get("acquired")
    if acquired_str:
        try:
            return datetime.fromisoformat(acquired_str.replace("Z", "+00:00"))
        except ValueError:
            pass
    return scene.acquired


def _solar_geometry_rad(dt_utc: datetime, lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """(declination_rad, hour_angle_rad, zenith_rad) via the standard NOAA
    solar position formulas (Meeus). Shared by solar_zenith_angle_deg and
    solar_position_deg since both need the same intermediate quantities."""
    day_of_year = dt_utc.timetuple().tm_yday
    hour_utc = dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600

    gamma = 2 * math.pi / 365 * (day_of_year - 1 + (hour_utc - 12) / 24)
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    time_offset = eqtime + 4 * lon_deg  # minutes; dt_utc is already UTC
    true_solar_time = hour_utc * 60 + time_offset
    hour_angle = math.radians((true_solar_time / 4) - 180)

    lat = math.radians(lat_deg)
    cos_zenith = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    zenith = math.acos(max(-1.0, min(1.0, cos_zenith)))
    return decl, hour_angle, zenith


def solar_zenith_angle_deg(dt_utc: datetime, lat_deg: float, lon_deg: float) -> float:
    """Approximate solar zenith angle (degrees). Used only as a fallback
    when a scene's metadata.json doesn't carry `sun_elevation` directly."""
    _decl, _hour_angle, zenith = _solar_geometry_rad(dt_utc, lat_deg, lon_deg)
    return math.degrees(zenith)


def solar_position_deg(dt_utc: datetime, lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """(azimuth_deg, elevation_deg) -- azimuth measured clockwise from true
    north, per the standard NOAA solar-position formula. Used as a fallback
    for lidar_masks.py's shadow ray-trace when a scene's metadata.json
    doesn't carry `sun_azimuth` directly."""
    decl, hour_angle, zenith = _solar_geometry_rad(dt_utc, lat_deg, lon_deg)
    lat = math.radians(lat_deg)
    denom = math.cos(lat) * math.sin(zenith)
    if abs(denom) < 1e-9:
        azimuth_deg = 180.0  # sun at zenith/nadir: azimuth is undefined, pick an arbitrary value
    else:
        arg = max(-1.0, min(1.0, (math.sin(lat) * math.cos(zenith) - math.sin(decl)) / denom))
        base_deg = math.degrees(math.acos(arg))
        azimuth_deg = (base_deg + 180.0) % 360.0 if hour_angle > 0 else (540.0 - base_deg) % 360.0
    return azimuth_deg, 90.0 - math.degrees(zenith)


def sun_zenith_angle_for_scene(scene: io.Scene) -> float | None:
    """Sun zenith angle (degrees) for `scene`: metadata.json's
    `sun_elevation` if present (90 - elevation), else computed from
    acquisition time + scene centroid lat/lon. None if neither is available.
    """
    elevation = scene.metadata().get("properties", {}).get("sun_elevation")
    if elevation is not None:
        return 90.0 - float(elevation)

    acquired = _acquired_utc(scene)
    centroid = _scene_centroid_lat_lon(scene)
    if acquired is None or centroid is None:
        return None
    lat, lon = centroid
    return solar_zenith_angle_deg(acquired, lat, lon)


def sun_position_for_scene(scene: io.Scene) -> tuple[float, float] | None:
    """(azimuth_deg, elevation_deg) for `scene`: metadata.json's
    `sun_azimuth`/`sun_elevation` if both present, else computed from
    acquisition time + scene centroid lat/lon. None if neither source is
    available. Acquisition time is always UTC (Planet's `acquired` field and
    psnorm's filename-parsed timestamps both are), which solar_position_deg
    requires."""
    props = scene.metadata().get("properties", {})
    azimuth, elevation = props.get("sun_azimuth"), props.get("sun_elevation")
    if azimuth is not None and elevation is not None:
        return float(azimuth), float(elevation)

    acquired = _acquired_utc(scene)
    centroid = _scene_centroid_lat_lon(scene)
    if acquired is None or centroid is None:
        return None
    lat, lon = centroid
    return solar_position_deg(acquired, lat, lon)


def select_reference(
    scenes: list[io.Scene],
    *,
    reference_path: str | None = None,
    zenith_percentile: float = 10.0,
    zenith_percentile_step: float = 10.0,
    min_reference_area_km2: float = 20.0,
    use_water_mask: bool = True,
    ndwi_threshold: float = 0.0,
    use_vegetation_mask: bool = True,
    ndvi_threshold: float = 0.2,
    use_omnicloudmask: bool = True,
    omnicloud_downsample_factor: int = 5,
    omnicloud_kwargs: dict | None = None,
    log=lambda msg: None,
) -> io.Scene:
    """Auto-select the reference scene: filter to the lowest
    `zenith_percentile` of sun zenith angles (closest to overhead sun —
    minimizes shadow length/BRDF variation), then within that subset pick
    the one with the greatest clear-sky fraction *among scenes with at
    least `min_reference_area_km2` of unmasked (not nodata/water/cloud)
    area*.

    A reference with only a small sliver of usable land is a bad choice
    even if its sun angle and UDM2 clear_percent look good: it leaves few
    pixels for IR-MAD to find invariant targets in, and what little usable
    area there is tends to sit at the edge of the strip where registration
    error (there being no registration step yet — see registration.py) is
    worst. If no scene within `zenith_percentile` clears the area bar, the
    percentile is widened by `zenith_percentile_step` (repeating up to
    100%) until one does.

    `reference_path` overrides all of this and is used as-is.
    """
    if reference_path is not None:
        for s in scenes:
            if s.analytic_path == reference_path:
                return s
        raise ValueError(f"reference_path '{reference_path}' not found among discovered scenes.")

    zeniths = {s.scene_id: sun_zenith_angle_for_scene(s) for s in scenes}
    candidates = [s for s in scenes if zeniths[s.scene_id] is not None]
    if not candidates:
        raise ValueError(
            "Cannot auto-select a reference scene: no scene has usable "
            "sun-angle data (metadata.json sun_elevation, or acquisition "
            "time + geometry). Pass reference_path explicitly."
        )

    zenith_values = np.array([zeniths[s.scene_id] for s in candidates])

    def clear_fraction_safe(s: io.Scene) -> float:
        try:
            return masking.scene_clear_fraction(s)
        except Exception:
            return -1.0

    area_cache: dict[str, float] = {}

    def unmasked_area_km2(s: io.Scene) -> float:
        if s.scene_id not in area_cache:
            band_names = sensors.detect_band_names(s.analytic_path)
            flags = masking.compute_exclusion_flags(
                s.analytic_path, s.udm2_path, band_names,
                use_water_mask=use_water_mask, ndwi_threshold=ndwi_threshold,
                use_vegetation_mask=use_vegetation_mask, ndvi_threshold=ndvi_threshold,
                use_omnicloudmask=use_omnicloudmask, omnicloud_downsample_factor=omnicloud_downsample_factor,
                omnicloud_kwargs=omnicloud_kwargs,
            )
            info = io.get_raster_info(s.analytic_path)
            pixel_area_m2 = abs(info.geotransform[1] * info.geotransform[5])
            area_cache[s.scene_id] = float((flags == 0).sum()) * pixel_area_m2 / 1e6
        return area_cache[s.scene_id]

    percentile = zenith_percentile
    while True:
        threshold = float(np.percentile(zenith_values, percentile))
        pool = sorted(
            (s for s, z in zip(candidates, zenith_values) if z <= threshold),
            key=clear_fraction_safe, reverse=True,
        )
        for s in pool:
            area = unmasked_area_km2(s)
            if area >= min_reference_area_km2:
                if percentile != zenith_percentile:
                    log(f"  no scene within the {zenith_percentile}th sun-zenith percentile had "
                        f">= {min_reference_area_km2} km^2 unmasked area; widened to {percentile}th percentile")
                return s
        if percentile >= 100.0:
            break
        percentile = min(100.0, percentile + zenith_percentile_step)

    raise ValueError(
        f"No scene has at least {min_reference_area_km2} km^2 of unmasked "
        f"(non-nodata/water/cloud) area, even after widening the sun-zenith "
        f"percentile filter to 100%. Lower min_reference_area_km2 or pass "
        f"reference_path explicitly."
    )


def scene_bounds_map(scenes: list[io.Scene]) -> dict[str, tuple[float, float, float, float]]:
    """{scene_id: (x0,y0,x1,y1)} for every scene, read once so
    `select_multi_references`/`assign_scenes_to_references` (and any
    caller chaining both) don't each re-open every scene's raster header."""
    return {s.scene_id: io._bounds(io.get_raster_info(s.analytic_path)) for s in scenes}


def select_multi_references(
    scenes: list[io.Scene],
    *,
    zenith_percentile: float = 10.0,
    zenith_percentile_step: float = 10.0,
    min_reference_area_km2: float = 20.0,
    use_water_mask: bool = True,
    ndwi_threshold: float = 0.0,
    use_vegetation_mask: bool = True,
    ndvi_threshold: float = 0.2,
    use_omnicloudmask: bool = True,
    omnicloud_downsample_factor: int = 5,
    omnicloud_kwargs: dict | None = None,
    min_overlap_km2: float = 1.0,
    max_references: int = 30,
    scene_bounds: dict[str, tuple[float, float, float, float]] | None = None,
    log=lambda msg: None,
) -> list[io.Scene]:
    """Repeated application of `select_reference`'s own selection criteria
    -- sun-zenith percentile screen, then rank by clear-sky fraction within
    that screen, then require `min_reference_area_km2` of actual unmasked
    (non-nodata/water/cloud/vegetation) area, widening the percentile if
    nothing clears that bar -- to an AOI wider than any one scene's own
    strip (e.g. two or more adjacent/overlapping satellite tracks side by
    side, where no single reference's footprint reaches every target).

    Unlike a plain greedy footprint set-cover, this does NOT pick whichever
    candidate covers the most remaining scenes -- it picks the *best-
    quality* candidate (by the same sun-angle + clear-sky + unmasked-area
    bar `select_reference` uses) among those that cover at least one
    remaining scene, repeating against whatever's still uncovered until
    every scene has a reference, `max_references` is hit, or nothing left
    covers anything further (logged as a warning). A low-quality reference
    (heavy cloud, oblique sun angle, mostly water/nodata) makes a poor
    IR-MAD target pool regardless of how much area it covers -- that's
    exactly the failure mode a coverage-only greedy selection can produce,
    so quality is the primary criterion here, coverage only a filter on
    which quality-ranked candidates are even relevant to consider.

    The sun-zenith percentile screen is a free, metadata-only filter
    (`sun_zenith_angle_for_scene` reads metadata.json or derives from
    acquisition time + centroid -- no pixel data touched), applied *before*
    the expensive per-scene mask computation (`compute_exclusion_flags`,
    which actually opens and analyzes each candidate's pixels for
    NDVI/water/cloud) -- so that expensive step only ever runs against the
    scenes the cheap filter already judged relevant, not the full input
    population. Results are cached across every outer iteration (picking
    reference #2 doesn't re-examine scenes already scored while picking
    reference #1), so no candidate's mask is computed more than once here
    regardless of how many references end up being selected. Note this
    in-memory computation is separate from (and not shared with) the
    per-scene flags raster the pipeline itself later computes and caches to
    disk (`_compute_and_save_flags`) -- that duplication is intentional and
    was a deliberate tradeoff discussed with the user: the real per-scene
    flags computation happens during the run regardless of what reference
    selection does, so skipping the equivalent check here to save compute
    saves nothing in the end, while measurably degrading which references
    get picked (fewer/worse consensus targets from a poorly-chosen
    reference) -- not a trade worth making for a cost that isn't actually
    avoided.
    """
    if not scenes:
        return []

    ids = [s.scene_id for s in scenes]
    by_id = {s.scene_id: s for s in scenes}
    bounds_map = scene_bounds if scene_bounds is not None else scene_bounds_map(scenes)
    bounds = np.array([bounds_map[sid] for sid in ids])  # (N, 4): x0,y0,x1,y1
    zeniths = {s.scene_id: sun_zenith_angle_for_scene(s) for s in scenes}
    zenith_values = np.array([zeniths[sid] if zeniths[sid] is not None else np.inf for sid in ids])

    clear_fraction_cache: dict[str, float] = {}

    def clear_fraction_safe(s: io.Scene) -> float:
        if s.scene_id not in clear_fraction_cache:
            try:
                clear_fraction_cache[s.scene_id] = masking.scene_clear_fraction(s)
            except Exception:
                clear_fraction_cache[s.scene_id] = -1.0
        return clear_fraction_cache[s.scene_id]

    area_cache: dict[str, float] = {}

    def unmasked_area_km2(s: io.Scene) -> float:
        if s.scene_id not in area_cache:
            try:
                band_names = sensors.detect_band_names(s.analytic_path)
                flags = masking.compute_exclusion_flags(
                    s.analytic_path, s.udm2_path, band_names,
                    use_water_mask=use_water_mask, ndwi_threshold=ndwi_threshold,
                    use_vegetation_mask=use_vegetation_mask, ndvi_threshold=ndvi_threshold,
                    use_omnicloudmask=use_omnicloudmask, omnicloud_downsample_factor=omnicloud_downsample_factor,
                    omnicloud_kwargs=omnicloud_kwargs,
                )
                info = io.get_raster_info(s.analytic_path)
                pixel_area_m2 = abs(info.geotransform[1] * info.geotransform[5])
                area_cache[s.scene_id] = float((flags == 0).sum()) * pixel_area_m2 / 1e6
            except Exception:
                area_cache[s.scene_id] = -1.0
        return area_cache[s.scene_id]

    def overlap_km2_vec(idx: int) -> np.ndarray:
        """Overlap area (km^2) between scene `idx` and every scene, vectorized."""
        ax0, ay0, ax1, ay1 = bounds[idx]
        ox0 = np.maximum(ax0, bounds[:, 0])
        oy0 = np.maximum(ay0, bounds[:, 1])
        ox1 = np.minimum(ax1, bounds[:, 2])
        oy1 = np.minimum(ay1, bounds[:, 3])
        w = np.maximum(0.0, ox1 - ox0)
        h = np.maximum(0.0, oy1 - oy0)
        return (w * h) / 1e6

    remaining = np.ones(len(ids), dtype=bool)
    selected = np.zeros(len(ids), dtype=bool)
    references: list[io.Scene] = []

    while remaining.any() and len(references) < max_references:
        picked_idx, picked_covered = None, None
        percentile = zenith_percentile
        while True:
            threshold = float(np.percentile(zenith_values, percentile))
            pool = sorted(
                (i for i in range(len(ids)) if not selected[i] and zenith_values[i] <= threshold),
                key=lambda i: -clear_fraction_safe(by_id[ids[i]]),
            )
            for i in pool:
                covered = remaining & (overlap_km2_vec(i) >= min_overlap_km2)
                if not covered.any():
                    continue  # doesn't help with what's left uncovered -- not worth the expensive area check
                if unmasked_area_km2(by_id[ids[i]]) >= min_reference_area_km2:
                    picked_idx, picked_covered = i, covered
                    break
            if picked_idx is not None or percentile >= 100.0:
                break
            percentile = min(100.0, percentile + zenith_percentile_step)

        if picked_idx is None:
            # Nothing left clears the quality bar anywhere in the AOI (rare) --
            # fall back to whichever remaining candidate covers the most
            # uncovered scenes, same last-resort behavior as before, rather
            # than leaving those scenes with no reference at all.
            best_idx, best_count, best_covered = -1, -1, None
            for i in range(len(ids)):
                if selected[i]:
                    continue
                covered = remaining & (overlap_km2_vec(i) >= min_overlap_km2)
                count = int(covered.sum())
                if count > best_count:
                    best_idx, best_count, best_covered = i, count, covered
            if best_idx < 0:
                break
            picked_idx, picked_covered = best_idx, best_covered
            log(f"  WARNING: no candidate met the quality bar for the remaining uncovered area; "
                f"falling back to best-coverage candidate {ids[picked_idx]}.")

        references.append(by_id[ids[picked_idx]])
        selected[picked_idx] = True
        remaining &= ~picked_covered
        log(f"  reference #{len(references)}: {ids[picked_idx]} "
            f"(clear_fraction={clear_fraction_cache.get(ids[picked_idx], float('nan')):.3f}, "
            f"unmasked_area={area_cache.get(ids[picked_idx], float('nan')):.1f}km^2) "
            f"covers {int(picked_covered.sum())} scenes ({int(remaining.sum())} scene(s) still uncovered)")

    if remaining.any():
        uncovered = [ids[i] for i in np.nonzero(remaining)[0]]
        log(f"  WARNING: {len(uncovered)} scene(s) have no overlapping reference and will "
            f"be left un-normalized: {sorted(uncovered)[:10]}{' ...' if len(uncovered) > 10 else ''}")

    return references


def assign_scenes_to_references(
    scenes: list[io.Scene], references: list[io.Scene], *,
    scene_bounds: dict[str, tuple[float, float, float, float]] | None = None,
    log=lambda msg: None,
) -> dict[str, list[io.Scene]]:
    """{reference_scene_id: [scenes assigned to it]} -- every scene
    (including the references themselves) is assigned to whichever
    reference's bounding box it overlaps *most*, so each scene is corrected
    against exactly one reference even when several references' footprints
    overlap it. A scene that overlaps no reference at all is dropped (with
    a warning) rather than assigned arbitrarily.
    """
    bounds_map = scene_bounds if scene_bounds is not None else scene_bounds_map(scenes + references)
    groups: dict[str, list[io.Scene]] = {r.scene_id: [] for r in references}
    dropped = []
    for s in scenes:
        sb = bounds_map[s.scene_id]
        best_ref, best_area = None, 0.0
        for r in references:
            rb = bounds_map[r.scene_id]
            rid = r.scene_id
            ox0, oy0 = max(sb[0], rb[0]), max(sb[1], rb[1])
            ox1, oy1 = min(sb[2], rb[2]), min(sb[3], rb[3])
            area = max(0.0, ox1 - ox0) * max(0.0, oy1 - oy0)
            if area > best_area:
                best_ref, best_area = rid, area
        if best_ref is None:
            dropped.append(s.scene_id)
        else:
            groups[best_ref].append(s)
    if dropped:
        log(f"  WARNING: {len(dropped)} scene(s) overlap no reference and were dropped: "
            f"{sorted(dropped)[:10]}{' ...' if len(dropped) > 10 else ''}")
    return groups


# --------------------------------------------------------------------------
# Shared config + path conventions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    reference_scene_id: str
    reference_analytic_path: str
    reference_flags_path: str
    reference_search_mask_path: str
    reference_band_names: tuple[str, ...]
    output_dir: str
    use_water_mask: bool
    ndwi_threshold: float
    use_vegetation_mask: bool
    ndvi_threshold: float
    use_omnicloudmask: bool
    omnicloud_downsample_factor: int
    omnicloud_kwargs: dict | None
    max_iter: int
    conv_threshold: float
    ncp_threshold: float
    invariance_frequency_threshold: float
    min_observations: int
    min_overlap_pixels: int
    resume: Literal["no", "yes", "validate"]
    block_rows: int
    target_downsample_factor: int  # 1 == disabled; see downsample_targets/downsample_resolution_m
    outlier_relative_threshold: float | None
    use_log_transform: bool
    slope_bounds: tuple[float, float] | None
    min_fit_pixels: int
    device: str  # already-resolved "cpu"/"gpu" (see backend.resolve_device) -- Phase A's IR-MAD fit + classify only
    dsm_path: str | None  # None (default) disables every LiDAR-derived mask below entirely
    dsm_height_units: str  # "m" or "ft" -- vertical unit of the DSM's own pixel values
    max_slope_deg: float
    roughness_window_radius_px: int
    roughness_max_deg: float
    use_shadow_mask: bool  # separate opt-in from dsm_path -- shadow ray-tracing is far more expensive than slope/roughness
    max_building_height_m: float
    shadow_ray_step_m: float | None  # None == one DSM native pixel; coarsen for speed
    shadow_downsample_factor: int  # 1 == disabled; working resolution for the shadow ray-trace specifically
    shadow_angle_bucket_deg: float  # sun (azimuth, elevation) rounding granularity for the on-disk shadow-mask cache
    mask_erode_px: int  # search-mask morphology (see masking.erode_dilate_bitmask) -- 0 disables
    mask_dilate_px: int
    force_registration: bool  # see registration.py -- is_grid_aligned alone misses PlanetScope's sub-pixel geolocation wobble
    registration_warp_model: Literal["tps", "affine"]  # see registration.register_to_reference
    registration_buffer_m: float  # reference search margin around each target's extent, see registration.register_to_reference


@dataclass
class SceneResult:
    scene_id: str
    status: Literal["fitted", "resumed", "reference", "skipped_registration", "skipped_insufficient_overlap", "error"]
    message: str = ""
    model_path: str | None = None
    normalized_path: str | None = None
    spectral_coverage: list | None = None  # list[normalize.BandSpectralCoverage], see write_spectral_coverage_report
    n_identity_fallback_bands: int = 0  # bands where the fit was rejected (too few pixels / implausible slope)


def _flags_path(output_dir: str, scene_id: str) -> str:
    return os.path.join(output_dir, "masks", f"{scene_id}_flags.tif")


def _search_mask_path(output_dir: str, scene_id: str) -> str:
    """The eroded/dilated exclusion bitmask (masking.erode_dilate_bitmask)
    actually used for target/consensus search -- kept separate from
    `_flags_path` so reference selection, clear-percent reporting, and the
    adjacent-pair metrics report keep reading the raw, unmodified flags."""
    return os.path.join(output_dir, "masks", f"{scene_id}_search_mask.tif")


def _lidar_flag_path(output_dir: str, scene_id: str, kind: str) -> str:
    return os.path.join(output_dir, "masks", f"{scene_id}_{kind}_flag.tif")


def _lidar_cache_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "masks", "_lidar_cache")


def _lidar_slope_cache_path(output_dir: str) -> str:
    return os.path.join(_lidar_cache_dir(output_dir), "slope_excluded.tif")


def _lidar_roughness_cache_path(output_dir: str) -> str:
    return os.path.join(_lidar_cache_dir(output_dir), "roughness_excluded.tif")


def _lidar_shadow_bucket_key(azimuth_deg: float, elevation_deg: float, bucket_deg: float) -> str:
    az = round(azimuth_deg / bucket_deg) * bucket_deg
    el = round(elevation_deg / bucket_deg) * bucket_deg
    return f"az{az:.1f}_el{el:.1f}"


def _lidar_shadow_cache_path(output_dir: str, azimuth_deg: float, elevation_deg: float, bucket_deg: float) -> str:
    key = _lidar_shadow_bucket_key(azimuth_deg, elevation_deg, bucket_deg)
    return os.path.join(_lidar_cache_dir(output_dir), f"shadow_excluded_{key}.tif")


def _candidate_mask_path(output_dir: str, target_id: str) -> str:
    return os.path.join(output_dir, "candidates", f"{target_id}_candidate.tif")


def _candidate_stats_path(output_dir: str, target_id: str) -> str:
    return os.path.join(output_dir, "candidates", f"{target_id}_stats.json")


def _irmad_fit_path(output_dir: str, target_id: str) -> str:
    return os.path.join(output_dir, "candidates", f"{target_id}_irmad_fit.json")


def _model_path(output_dir: str, scene_id: str) -> str:
    return os.path.join(output_dir, "models", f"{scene_id}_model.json")


def _normalized_path(output_dir: str, scene_id: str) -> str:
    return os.path.join(output_dir, "normalized", f"{scene_id}_normalized.tif")


def _ransac_inliers_path(output_dir: str, scene_id: str, band_name: str) -> str:
    return os.path.join(output_dir, "ransac_inliers", f"{scene_id}_{band_name}.tif")


def _compute_and_save_flags(scene: io.Scene, band_names: list[str], config: PipelineConfig) -> tuple[str, str]:
    """Compute (or reuse) a scene's exclusion-flags raster at its own full
    extent and save it under output_dir/masks/, along with the eroded/
    dilated *search* mask target/consensus search actually reads (see
    _search_mask_path). Shared by the reference and every target so
    OmniCloudMask/NDWI/LiDAR masks run exactly once per scene, regardless of
    how many pairs that scene participates in. Returns (flags_path,
    search_mask_path).
    """
    path = _flags_path(config.output_dir, scene.scene_id)
    search_path = _search_mask_path(config.output_dir, scene.scene_id)
    if config.resume != "no" and os.path.exists(path) and os.path.exists(search_path):
        return path, search_path

    info = io.get_raster_info(scene.analytic_path)
    lidar = _lidar_masks_for_scene(scene, info, config)
    lidar_slope = lidar["slope"] if lidar else None
    lidar_rough = lidar["roughness"] if lidar else None
    lidar_shadow = lidar.get("shadow") if lidar else None

    flags = masking.compute_exclusion_flags(
        scene.analytic_path, scene.udm2_path, band_names,
        use_water_mask=config.use_water_mask, ndwi_threshold=config.ndwi_threshold,
        use_vegetation_mask=config.use_vegetation_mask, ndvi_threshold=config.ndvi_threshold,
        use_omnicloudmask=config.use_omnicloudmask,
        omnicloud_downsample_factor=config.omnicloud_downsample_factor,
        omnicloud_kwargs=config.omnicloud_kwargs,
        lidar_slope_excluded=lidar_slope, lidar_rough_excluded=lidar_rough, lidar_shadow_excluded=lidar_shadow,
    )
    masking.save_flags_raster(flags, info, path)

    if lidar_slope is not None:
        masking.save_flags_raster(lidar_slope, info, _lidar_flag_path(config.output_dir, scene.scene_id, "horizontality"))
    if lidar_rough is not None:
        masking.save_flags_raster(lidar_rough, info, _lidar_flag_path(config.output_dir, scene.scene_id, "roughness"))
    if lidar_shadow is not None:
        masking.save_flags_raster(lidar_shadow, info, _lidar_flag_path(config.output_dir, scene.scene_id, "unshadowed"))

    search_flags = masking.erode_dilate_bitmask(
        flags, masking.ALL_EXCLUDE_BITS, erode_px=config.mask_erode_px, dilate_px=config.mask_dilate_px,
    )
    masking.save_flags_raster(search_flags, info, search_path)
    return path, search_path


def _prepare_lidar_masks(scenes: list[io.Scene], reference_scene: io.Scene, config: PipelineConfig, log) -> None:
    """Precompute (or reuse, if already cached from a prior run) every
    DSM-derived exclusion mask this run will need, once, before the
    parallel Phase A dispatch. No-op if config.dsm_path is None.

    Writes GeoTIFFs under output_dir/masks/_lidar_cache/ on the DSM's own
    native grid/CRS -- slope/roughness depend only on terrain so one
    raster covers every scene; shadow depends on sun position so one raster
    per rounded (azimuth, elevation) bucket needed by any scene this run
    covers. Each worker process is spawned fresh (see _fork_context) and
    can't share the open GDAL dataset or in-memory arrays built here, so
    they're written to disk and each worker independently re-opens the tiny
    crop it needs and warps it onto its own scene's grid -- see
    _compute_and_save_flags.
    """
    if config.dsm_path is None:
        return
    cache_dir = _lidar_cache_dir(config.output_dir)
    os.makedirs(cache_dir, exist_ok=True)

    slope_cache_path = _lidar_slope_cache_path(config.output_dir)
    roughness_cache_path = _lidar_roughness_cache_path(config.output_dir)
    need_slope_roughness = config.resume == "no" or not (
        os.path.exists(slope_cache_path) and os.path.exists(roughness_cache_path)
    )

    shadow_buckets_needed: dict[str, tuple[float, float]] = {}
    if config.use_shadow_mask:
        for scene in scenes:
            position = sun_position_for_scene(scene)
            if position is None or position[1] <= 0:
                continue
            azimuth, elevation = position
            key = _lidar_shadow_bucket_key(azimuth, elevation, config.shadow_angle_bucket_deg)
            shadow_buckets_needed.setdefault(key, (azimuth, elevation))
        if config.resume != "no":
            shadow_buckets_needed = {
                key: pos for key, pos in shadow_buckets_needed.items()
                if not os.path.exists(
                    _lidar_shadow_cache_path(config.output_dir, pos[0], pos[1], config.shadow_angle_bucket_deg)
                )
            }

    if not need_slope_roughness and not shadow_buckets_needed:
        log(f"  LiDAR masks: reusing cached slope/roughness/shadow rasters under {cache_dir}")
        return

    dataset = lidar_masks.open_dsm_mosaic(config.dsm_path)
    px_m, py_m = lidar_masks.dsm_pixel_size_m(dataset)

    ref_info = io.get_raster_info(reference_scene.analytic_path)
    # The shared DSM window must cover every scene whose exclusion mask
    # needs LiDAR coverage, not just the reference -- a target scene
    # extending beyond the reference's own footprint (as most do, being
    # different overpasses/strips) would otherwise get zero LiDAR flagging
    # in the part outside the reference, not because the DSM lacks data
    # there but because this window was never read that far. Scenes with no
    # geometric overlap with the reference at all are skipped regardless of
    # LiDAR (see _detect_candidates_for_scene), so they're excluded from
    # the union to keep this window from ballooning to cover unrelated
    # scenes/strips from other days.
    scene_infos = [ref_info]
    for scene in scenes:
        if scene.scene_id == reference_scene.scene_id:
            continue
        try:
            tgt_info = io.get_raster_info(scene.analytic_path)
        except Exception:
            continue
        if io.overlap_window(ref_info, tgt_info) is not None:
            scene_infos.append(tgt_info)
    log(f"  LiDAR DSM window covers {len(scene_infos)}/{len(scenes)} scenes "
        f"(every scene overlapping the reference)...")

    if need_slope_roughness:
        # Row-blocked (see compute_and_cache_slope_roughness) so peak memory
        # is bounded by block_rows x width, not by the whole union window --
        # important now that the window can span many scenes' combined
        # extent rather than just the reference's own footprint.
        roughness_buffer_m = config.roughness_window_radius_px * max(px_m, py_m) * 2
        out_info = lidar_masks.compute_and_cache_slope_roughness(
            dataset, scene_infos, height_units=config.dsm_height_units, buffer_m=roughness_buffer_m,
            max_slope_deg=config.max_slope_deg, roughness_window_radius_px=config.roughness_window_radius_px,
            roughness_max_deg=config.roughness_max_deg,
            slope_path=slope_cache_path, roughness_path=roughness_cache_path,
        )
        if out_info is None:
            log(f"  WARNING: DSM at '{config.dsm_path}' does not overlap any scene "
                f"-- horizontality/roughness masks disabled for this run.")
        else:
            log(f"  slope/roughness cache: {out_info.width}x{out_info.height} DSM pixels "
                f"({px_m:.2f}x{py_m:.2f}m native)")

    if shadow_buckets_needed:
        # Each sun-angle bucket gets its own cached raster (see
        # _lidar_shadow_cache_path) -- this is the mechanism that actually
        # handles "different scenes have different sun angles", independent
        # of how large the shared window is. Row-blocked the same way as
        # slope/roughness (see compute_and_cache_shadow), just with a much
        # larger halo -- the full shadow search radius, not a couple
        # pixels -- since a whole-array shadow computation over a
        # multi-scene union window is exactly what caused a real OOM in
        # production.
        if config.shadow_downsample_factor > 1:
            log(f"  NOTE: shadow_downsample_factor={config.shadow_downsample_factor} is not "
                f"supported by the row-blocked shadow computation (avoids reintroducing a "
                f"cross-block grid-phase-alignment hazard) -- computing at native DSM "
                f"resolution instead.")
        for bucket_i, (key, (azimuth, elevation)) in enumerate(shadow_buckets_needed.items(), start=1):
            log(f"  computing shadow exclusion for sun bucket {key} "
                f"({bucket_i}/{len(shadow_buckets_needed)}, az={azimuth:.1f} deg, el={elevation:.1f} deg)...")
            shadow_info = lidar_masks.compute_and_cache_shadow(
                dataset, scene_infos, height_units=config.dsm_height_units,
                azimuth_deg=azimuth, elevation_deg=elevation,
                max_building_height_m=config.max_building_height_m, ray_step_m=config.shadow_ray_step_m,
                shadow_path=_lidar_shadow_cache_path(config.output_dir, azimuth, elevation, config.shadow_angle_bucket_deg),
                log=log,
            )
            if shadow_info is None:
                log(f"  WARNING: DSM at '{config.dsm_path}' does not overlap any scene "
                    f"-- shadow mask for bucket {key} disabled.")


def _lidar_masks_for_scene(scene: io.Scene, scene_info: io.RasterInfo, config: PipelineConfig) -> dict[str, np.ndarray] | None:
    """Warp this scene's crop of each cached DSM-derived exclusion raster
    (see _prepare_lidar_masks) onto `scene_info`'s exact grid. Returns None
    if config.dsm_path is None (LiDAR masks disabled) or the cache is
    missing (DSM didn't overlap the reference extent -- already warned
    about in _prepare_lidar_masks)."""
    if config.dsm_path is None:
        return None
    slope_cache_path = _lidar_slope_cache_path(config.output_dir)
    roughness_cache_path = _lidar_roughness_cache_path(config.output_dir)
    if not (os.path.exists(slope_cache_path) and os.path.exists(roughness_cache_path)):
        return None

    out = {
        "slope": lidar_masks.warp_mask_to_scene(slope_cache_path, scene_info),
        "roughness": lidar_masks.warp_mask_to_scene(roughness_cache_path, scene_info),
    }

    if config.use_shadow_mask:
        position = sun_position_for_scene(scene)
        if position is not None and position[1] > 0:
            azimuth, elevation = position
            shadow_cache_path = _lidar_shadow_cache_path(config.output_dir, azimuth, elevation, config.shadow_angle_bucket_deg)
            if os.path.exists(shadow_cache_path):
                out["shadow"] = lidar_masks.warp_mask_to_scene(shadow_cache_path, scene_info)
    return out


def _existing_outputs_reusable(paths: list[str], resume: str) -> bool:
    if resume == "no":
        return False
    if not all(os.path.exists(p) for p in paths):
        return False
    if resume == "yes":
        return True
    return all(model_io.model_is_valid(p) if p.endswith(".json") else True for p in paths)


def _target_model_reusable(model_path: str, normalized_path: str, config: PipelineConfig) -> bool:
    """Like _existing_outputs_reusable, but additionally invalidated when
    the saved model was fit under a *different* invariance_frequency_
    threshold/min_observations than the current config -- both change which
    pixels are in the consensus mask Phase C fits against (see
    normalize.NormalizationModel), and Phase B always recomputes the
    consensus mask fresh every run (no resume check of its own), so a saved
    model from a differently-thresholded consensus set is stale even though
    the files themselves still exist and parse. Only meaningful for target
    scenes -- the reference's own model is always the fixed identity
    transform regardless of these thresholds."""
    if not _existing_outputs_reusable([model_path, normalized_path], config.resume):
        return False
    try:
        model = model_io.load_model(model_path)
    except Exception:
        return False
    return (
        model.invariance_frequency_threshold == config.invariance_frequency_threshold
        and model.min_observations == config.min_observations
    )


def _process_reference_scene(scene: io.Scene, band_names: list[str], config: PipelineConfig) -> SceneResult:
    model_path = _model_path(config.output_dir, scene.scene_id)
    normalized_path = _normalized_path(config.output_dir, scene.scene_id)
    if _existing_outputs_reusable([model_path, normalized_path], config.resume):
        return SceneResult(scene.scene_id, "resumed", model_path=model_path, normalized_path=normalized_path)

    model = normalize.identity_model(band_names, scene.scene_id)
    model_io.save_model(model, model_path)
    apply.apply_model(scene.analytic_path, model, normalized_path, band_names)
    return SceneResult(scene.scene_id, "reference", model_path=model_path, normalized_path=normalized_path)


# --------------------------------------------------------------------------
# Phase A: per-target invariant-candidate detection
# --------------------------------------------------------------------------


@dataclass
class CandidateOutcome:
    scene_id: str
    status: Literal["detected", "skipped_registration", "skipped_insufficient_overlap", "error"]
    message: str = ""
    ref_window: io.Window | None = None
    tgt_window: io.Window | None = None
    target_search_mask_path: str | None = None
    candidate_mask_path: str | None = None


def _detect_candidates_for_scene(scene: io.Scene, config: PipelineConfig) -> CandidateOutcome:
    reference_band_names = list(config.reference_band_names)
    try:
        target_band_names = sensors.detect_band_names(scene.analytic_path)
    except ValueError as exc:
        return CandidateOutcome(scene.scene_id, "error", message=str(exc))

    common_bands = [b for b in reference_band_names if b in target_band_names]
    if not common_bands:
        return CandidateOutcome(
            scene.scene_id, "error",
            message=f"no shared bands between reference {reference_band_names!r} and target {target_band_names!r}",
        )

    ref_info = io.get_raster_info(config.reference_analytic_path)
    tgt_info = io.get_raster_info(scene.analytic_path)

    if config.force_registration or not registration.is_grid_aligned(ref_info, tgt_info):
        try:
            aligned_path = registration.register_to_reference(
                scene.analytic_path, config.reference_analytic_path,
                warp_model=config.registration_warp_model, buffer_m=config.registration_buffer_m,
            )
        except (NotImplementedError, registration.RegistrationFailed) as exc:
            return CandidateOutcome(scene.scene_id, "skipped_registration", message=str(exc))
        target_analytic_path = aligned_path
        tgt_info = io.get_raster_info(aligned_path)
    else:
        target_analytic_path = scene.analytic_path

    window = io.overlap_window(ref_info, tgt_info)
    if window is None:
        return CandidateOutcome(scene.scene_id, "skipped_insufficient_overlap", message="no geometric overlap with reference")
    ref_window, tgt_window = window

    _target_flags_path, target_search_mask_path = _compute_and_save_flags(scene, target_band_names, config)

    candidate_path = _candidate_mask_path(config.output_dir, scene.scene_id)
    stats_path = _candidate_stats_path(config.output_dir, scene.scene_id)
    fit_path = _irmad_fit_path(config.output_dir, scene.scene_id)

    existing_ncp_threshold = None
    if os.path.exists(stats_path):
        try:
            existing_ncp_threshold = model_io.load_json(stats_path).get("ncp_threshold")
        except Exception:
            existing_ncp_threshold = None

    if (
        config.resume != "no"
        and os.path.exists(candidate_path)
        and os.path.exists(stats_path)
        and os.path.exists(fit_path)
        and existing_ncp_threshold == config.ncp_threshold
    ):
        return CandidateOutcome(
            scene.scene_id, "detected", ref_window=ref_window, tgt_window=tgt_window,
            target_search_mask_path=target_search_mask_path, candidate_mask_path=candidate_path,
        )

    ref_flags_window = masking.load_flags_window(config.reference_search_mask_path, ref_window)
    tgt_flags_window = masking.load_flags_window(target_search_mask_path, tgt_window)
    search_mask = (ref_flags_window == 0) & (tgt_flags_window == 0)
    n_search = int(search_mask.sum())
    if n_search < config.min_overlap_pixels:
        return CandidateOutcome(
            scene.scene_id, "skipped_insufficient_overlap",
            message=f"only {n_search} pixels pass the search mask (< {config.min_overlap_pixels})",
        )

    ref_indices = [sensors.band_index(reference_band_names, b) for b in common_bands]
    tgt_indices = [sensors.band_index(target_band_names, b) for b in common_bands]

    # Fast path: a valid fit already exists (from a prior run, possibly at a
    # different ncp_threshold -- the fit itself doesn't depend on it, only
    # classification does). Reclassify in one O(pixels) pass instead of
    # redoing IR-MAD's O(pixels * iterations) iterative fit -- see
    # normalize.reclassify_invariant_pixels. This assumes the *other*
    # fit-affecting parameters (log_transform, downsample_factor, max_iter,
    # conv_threshold) match whatever produced the saved fit; pass
    # resume="no" to force a full refit if you've changed any of those.
    if config.resume != "no" and model_io.irmad_fit_is_valid(fit_path):
        try:
            fit = model_io.load_irmad_fit(fit_path)
            invariant_mask, n_evaluated, n_invariant = normalize.reclassify_invariant_pixels(
                config.reference_analytic_path, target_analytic_path, ref_window, tgt_window,
                ref_indices, tgt_indices, search_mask, fit,
                ncp_threshold=config.ncp_threshold, block_rows=config.block_rows,
                downsample_factor=config.target_downsample_factor, log_transform=config.use_log_transform,
                device=config.device,
            )
        except Exception as exc:
            return CandidateOutcome(scene.scene_id, "error", message=str(exc))

        masking.save_flags_raster(invariant_mask, io.windowed_raster_info(ref_info, ref_window), candidate_path)
        model_io.save_json(
            {
                "reference_id": config.reference_scene_id, "target_id": scene.scene_id,
                "ref_window": list(ref_window), "tgt_window": list(tgt_window),
                "n_evaluated": n_evaluated, "n_invariant": n_invariant,
                "ncp_threshold": config.ncp_threshold, "irmad_rho": [float(v) for v in fit.rho],
                "irmad_converged": fit.converged, "irmad_iterations": fit.n_iterations,
            },
            stats_path,
        )
        return CandidateOutcome(
            scene.scene_id, "detected", ref_window=ref_window, tgt_window=tgt_window,
            target_search_mask_path=target_search_mask_path, candidate_mask_path=candidate_path,
        )

    try:
        result = normalize.detect_invariant_candidates(
            config.reference_analytic_path, target_analytic_path,
            config.reference_scene_id, scene.scene_id,
            ref_window, tgt_window, ref_indices, tgt_indices, common_bands,
            search_mask,
            max_iter=config.max_iter, conv_threshold=config.conv_threshold,
            ncp_threshold=config.ncp_threshold, block_rows=config.block_rows,
            downsample_factor=config.target_downsample_factor,
            log_transform=config.use_log_transform,
            device=config.device,
        )
    except Exception as exc:
        return CandidateOutcome(scene.scene_id, "error", message=str(exc))

    masking.save_flags_raster(result.invariant_mask, io.windowed_raster_info(ref_info, ref_window), candidate_path)
    model_io.save_json(
        {
            "reference_id": result.reference_id, "target_id": result.target_id,
            "ref_window": list(result.ref_window), "tgt_window": list(result.tgt_window),
            "n_evaluated": result.n_evaluated, "n_invariant": result.n_invariant,
            "ncp_threshold": result.ncp_threshold, "irmad_rho": result.irmad_rho,
            "irmad_converged": result.irmad_converged, "irmad_iterations": result.irmad_iterations,
        },
        stats_path,
    )
    # Saved separately from stats.json (which is a diagnostics summary) since
    # this is the actual fitted model -- see model_io.save_irmad_fit for why
    # persisting it is what makes re-sweeping ncp_threshold later cheap.
    model_io.save_irmad_fit(result.irmad_fit, fit_path)

    return CandidateOutcome(
        scene.scene_id, "detected", ref_window=ref_window, tgt_window=tgt_window,
        target_search_mask_path=target_search_mask_path, candidate_mask_path=candidate_path,
    )


# --------------------------------------------------------------------------
# Phase C: final regression fit + apply, against the consensus mask
# --------------------------------------------------------------------------


def _finalize_target_scene(scene: io.Scene, outcome: CandidateOutcome, config: PipelineConfig) -> SceneResult:
    model_path = _model_path(config.output_dir, scene.scene_id)
    normalized_path = _normalized_path(config.output_dir, scene.scene_id)
    if _target_model_reusable(model_path, normalized_path, config):
        # Re-derive the fallback count from the saved model rather than
        # defaulting to 0 -- summary.md's tally otherwise silently omits
        # every resumed scene's fallback bands (only freshly-fitted scenes
        # would ever contribute to it), undercounting on any run with
        # resumed scenes in the mix.
        n_identity_fallback_bands = sum(1 for b in model_io.load_model(model_path).bands if b.identity_fallback)
        return SceneResult(scene.scene_id, "resumed", model_path=model_path, normalized_path=normalized_path,
                            n_identity_fallback_bands=n_identity_fallback_bands)

    reference_band_names = list(config.reference_band_names)
    target_band_names = sensors.detect_band_names(scene.analytic_path)
    common_bands = [b for b in reference_band_names if b in target_band_names]
    ref_indices = [sensors.band_index(reference_band_names, b) for b in common_bands]
    tgt_indices = [sensors.band_index(target_band_names, b) for b in common_bands]

    consensus_mask_path = os.path.join(config.output_dir, "consensus", "consensus_mask.tif")
    consensus_window = masking.load_flags_window(consensus_mask_path, outcome.ref_window) != 0

    try:
        fitted, inlier_rasters = normalize.fit_regression_from_mask(
            config.reference_analytic_path, scene.analytic_path,
            config.reference_scene_id, scene.scene_id,
            outcome.ref_window, outcome.tgt_window, ref_indices, tgt_indices, common_bands,
            consensus_window,
            invariance_frequency_threshold=config.invariance_frequency_threshold,
            min_observations=config.min_observations, block_rows=config.block_rows,
            downsample_factor=config.target_downsample_factor,
            outlier_relative_threshold=config.outlier_relative_threshold,
            slope_bounds=config.slope_bounds, min_fit_pixels=config.min_fit_pixels,
            return_inlier_masks=True,
        )
    except ValueError as exc:
        return SceneResult(scene.scene_id, "error", message=str(exc))

    # One binary raster per band showing RANSAC's final inlier pixels for
    # that band's fit (saved regardless of identity_fallback -- see
    # fit_regression_from_mask's docstring for why a rejected band's inliers
    # are still worth inspecting).
    ref_info = io.get_raster_info(config.reference_analytic_path)
    for band_name, inlier_mask in inlier_rasters.items():
        masking.save_flags_raster(
            inlier_mask, io.windowed_raster_info(ref_info, outcome.ref_window),
            _ransac_inliers_path(config.output_dir, scene.scene_id, band_name),
        )

    # Spectral-range coverage diagnostic: always computed from native-
    # resolution pixel values (see compute_spectral_coverage), independent
    # of whether downsampling was used to find the targets/fit the model.
    spectral_coverage = normalize.compute_spectral_coverage(
        scene.analytic_path, outcome.tgt_window, tgt_indices, common_bands,
        consensus_window, block_rows=config.block_rows,
    )

    # Bands the target has but the reference doesn't (e.g. SuperDove's extra
    # bands vs a 4-band harmonized reference) pass through unchanged rather
    # than being dropped, so no spectral information is lost on output.
    full_bands = [
        fitted.band(name) if name in common_bands
        else BandModel(band_name=name, slope=1.0, intercept=0.0, r2=float("nan"), n_invariant_pixels=0)
        for name in target_band_names
    ]
    full_model = replace(fitted, band_names=target_band_names, bands=full_bands)

    model_io.save_model(full_model, model_path)
    apply.apply_model(scene.analytic_path, full_model, normalized_path, target_band_names, block_rows=config.block_rows)
    n_identity_fallback_bands = sum(1 for b in full_model.bands if b.identity_fallback)
    return SceneResult(scene.scene_id, "fitted", model_path=model_path, normalized_path=normalized_path,
                        spectral_coverage=spectral_coverage, n_identity_fallback_bands=n_identity_fallback_bands)


def _finalize_pair(pair: tuple[io.Scene, CandidateOutcome], config: PipelineConfig) -> SceneResult:
    scene, outcome = pair
    return _finalize_target_scene(scene, outcome, config)


# --------------------------------------------------------------------------
# Parallel execution helpers
# --------------------------------------------------------------------------


def _resolve_workers(workers) -> int | None:
    if workers is None:
        return None
    if workers == "cpu":
        return os.cpu_count() or 1
    return int(workers)


def _fork_context():
    import multiprocessing as mp

    # spawn is used unconditionally (not just on non-Linux): forking a
    # process that has already imported torch/GDAL can inherit broken
    # internal thread-pool state from either library, which is a much
    # costlier failure mode (silent hangs) than spawn's slightly higher
    # per-worker startup cost.
    return mp.get_context("spawn")


def _map_parallel(fn, items, workers, *extra_args, on_result=None):
    """[fn(item, *extra_args) for item in items], parallelized across
    `workers` processes. The returned list is always in the same order as
    `items`, regardless of completion order.

    `on_result(item, result)`, if given, fires as soon as *that* item's
    result is available -- in completion order, which on a large batch
    (hundreds of scenes) can be a very different order than `items` itself.
    Without it, a caller has no visibility into a batch's progress until
    every single item has finished: the previous version of this function
    only ever returned once the whole list was ready, so on a big multi-
    hour batch there was no sign of life (not even in the log) between
    "starting" and "done" -- which reads identically to a hang. Passing
    `on_result` (used by run_pipeline's Phase A/C batches) lets the caller
    log/count as items complete instead of only once, at the very end.
    """
    n_workers = _resolve_workers(workers)
    if n_workers is None or n_workers <= 1 or len(items) <= 1:
        results = []
        for item in items:
            r = fn(item, *extra_args)
            if on_result is not None:
                on_result(item, r)
            results.append(r)
        return results
    ctx = _fork_context()
    results = [None] * len(items)
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {executor.submit(fn, item, *extra_args): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            if on_result is not None:
                on_result(items[i], results[i])
    return results


# --------------------------------------------------------------------------
# Adjacent-pair R^2/RMSE report
# --------------------------------------------------------------------------


@dataclass
class PairReport:
    scene_a: str
    scene_b: str
    time_a: str
    time_b: str
    n_pixels: int
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    excluded_reason: str | None = None


def _agreement_to_dict(agreement: metrics.AgreementResult) -> dict:
    result = {"pooled": {"r2": agreement.pooled_r2, "rmse": agreement.pooled_rmse}}
    for band in agreement.bands:
        result[band.band_name] = {"r2": band.r2, "rmse": band.rmse}
    return result


def _compute_pair_report(
    pair: tuple[tuple[io.Scene, SceneResult], tuple[io.Scene, SceneResult]],
    output_dir: str,
    min_overlap_pixels: int,
) -> PairReport:
    (scene_a, res_a), (scene_b, res_b) = pair
    time_a = scene_a.acquired.isoformat() if scene_a.acquired else ""
    time_b = scene_b.acquired.isoformat() if scene_b.acquired else ""

    info_a = io.get_raster_info(scene_a.analytic_path)
    info_b = io.get_raster_info(scene_b.analytic_path)
    if not io.grids_aligned(info_a, info_b):
        return PairReport(scene_a.scene_id, scene_b.scene_id, time_a, time_b, 0, excluded_reason="not grid-aligned")

    window = io.overlap_window(info_a, info_b)
    if window is None:
        return PairReport(scene_a.scene_id, scene_b.scene_id, time_a, time_b, 0, excluded_reason="no geometric overlap")
    window_a, window_b = window

    band_names_a = sensors.detect_band_names(scene_a.analytic_path)
    band_names_b = sensors.detect_band_names(scene_b.analytic_path)

    flags_a = masking.load_flags_window(_flags_path(output_dir, scene_a.scene_id), window_a)
    flags_b = masking.load_flags_window(_flags_path(output_dir, scene_b.scene_id), window_b)
    joint_mask = (flags_a == 0) & (flags_b == 0)
    n_valid = int(joint_mask.sum())
    if n_valid < min_overlap_pixels:
        return PairReport(scene_a.scene_id, scene_b.scene_id, time_a, time_b, n_valid,
                           excluded_reason=f"only {n_valid} jointly-valid pixels (< {min_overlap_pixels})")

    before = metrics.agreement_between(scene_a.analytic_path, window_a, band_names_a, scene_b.analytic_path, window_b, band_names_b, joint_mask)
    after = metrics.agreement_between(res_a.normalized_path, window_a, band_names_a, res_b.normalized_path, window_b, band_names_b, joint_mask)

    return PairReport(
        scene_a=scene_a.scene_id, scene_b=scene_b.scene_id, time_a=time_a, time_b=time_b,
        n_pixels=n_valid, before=_agreement_to_dict(before), after=_agreement_to_dict(after),
    )


def _build_adjacency_report(
    ordered: list[tuple[io.Scene, SceneResult]],
    output_dir: str,
    min_overlap_pixels: int,
    workers: Literal["cpu"] | int | None,
) -> list[PairReport]:
    pairs = list(zip(ordered, ordered[1:]))
    return _map_parallel(_compute_pair_report, pairs, workers, output_dir, min_overlap_pixels)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass
class PipelineResult:
    reference_scene_id: str
    band_names: list[str]
    scene_results: list[SceneResult]
    pair_reports: list[PairReport]
    consensus_paths: dict
    n_consensus_pixels: int
    output_dir: str


def run_pipeline(
    input_folder: str,
    output_folder: str,
    *,
    reference_path: str | None = None,
    band_names: list[str] | None = None,
    zenith_percentile: float = 10.0,
    zenith_percentile_step: float = 10.0,
    min_reference_area_km2: float = 20.0,
    use_water_mask: bool = True,
    ndwi_threshold: float = 0.0,
    use_vegetation_mask: bool = True,
    ndvi_threshold: float = 0.2,
    use_omnicloudmask: bool = True,
    omnicloud_downsample_factor: int = 5,
    omnicloud_kwargs: dict | None = None,
    max_iter: int = 30,
    conv_threshold: float = 0.99,
    ncp_threshold: float = 0.70,
    invariance_frequency_threshold: float = 0.5,
    min_observations: int = 2,
    min_overlap_pixels: int = 500,
    resume: Literal["no", "yes", "validate"] = "validate",
    workers: Literal["cpu"] | int | None = "cpu",
    block_rows: int = io.DEFAULT_BLOCK_ROWS,
    downsample_targets: bool = False,
    downsample_resolution_m: float = 15.0,
    outlier_relative_threshold: float | None = 0.05,
    use_log_transform: bool = True,
    slope_bounds: tuple[float, float] | None = (0.8, 1.2),
    min_fit_pixels: int = 500,
    device: Literal["auto", "cpu", "gpu"] = "auto",
    dsm_path: str | None = None,
    dsm_height_units: str = "m",
    max_slope_deg: float = 5.0,
    roughness_window_radius_px: int = 2,
    roughness_max_deg: float = 10.0,
    use_shadow_mask: bool = False,
    max_building_height_m: float = 150.0,
    shadow_ray_step_m: float | None = None,
    shadow_downsample_factor: int = 1,
    shadow_angle_bucket_deg: float = 1.0,
    mask_erode_px: int = 1,
    mask_dilate_px: int = 1,
    force_registration: bool = False,  # attempt registration.register_to_reference even when is_grid_aligned is already True
    registration_warp_model: Literal["tps", "affine"] = "tps",
    registration_buffer_m: float = 500.0,
    scene_ids: set[str] | None = None,
    scenes: list[io.Scene] | None = None,
    log=print,
) -> PipelineResult:
    if scenes is not None:
        log(f"Using {len(scenes)} caller-supplied scenes (bypassing directory discovery).")
    else:
        scenes = io.discover_scenes(input_folder)
        if not scenes:
            raise ValueError(f"No scenes found in '{input_folder}'.")
        log(f"Discovered {len(scenes)} scenes.")

    if scene_ids is not None:
        scenes = [s for s in scenes if s.scene_id in scene_ids]
        missing = scene_ids - {s.scene_id for s in scenes}
        if missing:
            log(f"  WARNING: {len(missing)} requested scene_id(s) not found in '{input_folder}': "
                f"{sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}")
        if not scenes:
            raise ValueError(f"None of the requested scene_ids were found in '{input_folder}'.")
        log(f"Restricted to {len(scenes)} scenes (scene_ids filter, e.g. one multi-reference group).")

    resolved_device = backend.resolve_device(device)
    log(f"Device: {resolved_device}" + (f" (auto-detected from device={device!r})" if device == "auto" else ""))
    if resolved_device == "gpu" and workers not in (1, None):
        log(
            f"  note: workers={workers!r} with device='gpu' means multiple CPU processes will "
            f"each independently use the one GPU for Phase A -- they'll contend for it rather "
            f"than getting N-way speedup the way CPU-only workers do. Consider workers=1 (or a "
            f"small number) when device='gpu'/'auto' resolves to gpu."
        )

    reference_scene = select_reference(
        scenes, reference_path=reference_path, zenith_percentile=zenith_percentile,
        zenith_percentile_step=zenith_percentile_step, min_reference_area_km2=min_reference_area_km2,
        use_water_mask=use_water_mask, ndwi_threshold=ndwi_threshold,
        use_vegetation_mask=use_vegetation_mask, ndvi_threshold=ndvi_threshold,
        use_omnicloudmask=use_omnicloudmask, omnicloud_downsample_factor=omnicloud_downsample_factor,
        omnicloud_kwargs=omnicloud_kwargs, log=log,
    )
    log(f"Reference scene: {reference_scene.scene_id}")

    if band_names is None:
        band_names = sensors.detect_band_names(reference_scene.analytic_path)
    log(f"Reference band names: {band_names}")

    for sub in ("masks", "candidates", "consensus", "models", "normalized", "ransac_inliers"):
        os.makedirs(os.path.join(output_folder, sub), exist_ok=True)

    # Downsampling (if enabled) only coarsens the pixel values IR-MAD/the
    # regression see when searching for targets and fitting correction
    # factors (see io.read_block_flat) -- apply.py always writes the final
    # corrected scene at its native resolution, unaffected by this.
    if downsample_targets:
        native_pixel_size_m = abs(io.get_raster_info(reference_scene.analytic_path).geotransform[1])
        target_downsample_factor = max(1, round(downsample_resolution_m / native_pixel_size_m))
        log(f"Downsampling target search + correction-factor fitting to "
            f"~{target_downsample_factor * native_pixel_size_m:.1f}m "
            f"(factor={target_downsample_factor}) from native {native_pixel_size_m:.1f}m")
    else:
        target_downsample_factor = 1

    # Reference's own exclusion flags are computed once, up front, since
    # every target comparison reads them.
    reference_flags_path = _flags_path(output_folder, reference_scene.scene_id)
    reference_search_mask_path = _search_mask_path(output_folder, reference_scene.scene_id)
    config = PipelineConfig(
        reference_scene_id=reference_scene.scene_id,
        reference_analytic_path=reference_scene.analytic_path,
        reference_flags_path=reference_flags_path,
        reference_search_mask_path=reference_search_mask_path,
        reference_band_names=tuple(band_names),
        output_dir=output_folder,
        use_water_mask=use_water_mask,
        ndwi_threshold=ndwi_threshold,
        use_vegetation_mask=use_vegetation_mask,
        ndvi_threshold=ndvi_threshold,
        use_omnicloudmask=use_omnicloudmask,
        omnicloud_downsample_factor=omnicloud_downsample_factor,
        omnicloud_kwargs=omnicloud_kwargs,
        max_iter=max_iter,
        conv_threshold=conv_threshold,
        ncp_threshold=ncp_threshold,
        invariance_frequency_threshold=invariance_frequency_threshold,
        min_observations=min_observations,
        min_overlap_pixels=min_overlap_pixels,
        resume=resume,
        block_rows=block_rows,
        target_downsample_factor=target_downsample_factor,
        outlier_relative_threshold=outlier_relative_threshold,
        use_log_transform=use_log_transform,
        slope_bounds=slope_bounds,
        min_fit_pixels=min_fit_pixels,
        device=resolved_device,
        dsm_path=dsm_path,
        dsm_height_units=dsm_height_units,
        max_slope_deg=max_slope_deg,
        roughness_window_radius_px=roughness_window_radius_px,
        roughness_max_deg=roughness_max_deg,
        use_shadow_mask=use_shadow_mask,
        max_building_height_m=max_building_height_m,
        shadow_ray_step_m=shadow_ray_step_m,
        shadow_downsample_factor=shadow_downsample_factor,
        shadow_angle_bucket_deg=shadow_angle_bucket_deg,
        mask_erode_px=mask_erode_px,
        mask_dilate_px=mask_dilate_px,
        force_registration=force_registration,
        registration_warp_model=registration_warp_model,
        registration_buffer_m=registration_buffer_m,
    )
    if dsm_path is not None:
        log(f"Preparing LiDAR masks from DSM '{dsm_path}' "
            f"(max_slope_deg={max_slope_deg}, roughness_max_deg={roughness_max_deg}, "
            f"use_shadow_mask={use_shadow_mask})...")
    _prepare_lidar_masks(scenes, reference_scene, config, log)
    _compute_and_save_flags(reference_scene, band_names, config)
    log(f"Reference exclusion flags: {reference_flags_path}")

    reference_result = _process_reference_scene(reference_scene, band_names, config)
    log(f"  {reference_scene.scene_id}: {reference_result.status}")

    targets = [s for s in scenes if s.scene_id != reference_scene.scene_id]

    log(f"Phase A: detecting invariant-target candidates against {len(targets)} target scenes "
        f"(target vs. reference only)...")
    n_done = [0]

    def _log_candidate_progress(scene, outcome):
        n_done[0] += 1
        log(f"  [{n_done[0]}/{len(targets)}] {scene.scene_id}: {outcome.status}"
            + (f" ({outcome.message})" if outcome.message else ""))

    candidate_outcomes = _map_parallel(
        _detect_candidates_for_scene, targets, workers, config, on_result=_log_candidate_progress,
    )

    detected = [(s, o) for s, o in zip(targets, candidate_outcomes) if o.status == "detected"]

    log(f"Phase B: aggregating consensus across {len(detected)} candidate sets "
        f"(frequency > {invariance_frequency_threshold}, min_observations={min_observations})...")
    ref_info = io.get_raster_info(reference_scene.analytic_path)
    records = [
        consensus.ConsensusRecord(
            target_id=scene.scene_id, ref_window=outcome.ref_window, tgt_window=outcome.tgt_window,
            candidate_mask_path=outcome.candidate_mask_path,
            reference_search_mask_path=reference_search_mask_path, target_search_mask_path=outcome.target_search_mask_path,
        )
        for scene, outcome in detected
    ]
    consensus_result = consensus.aggregate_consensus(
        (ref_info.height, ref_info.width), records,
        invariance_frequency_threshold=invariance_frequency_threshold, min_observations=min_observations,
    )
    consensus_dir = os.path.join(output_folder, "consensus")
    consensus_paths = consensus.save_consensus(consensus_result, ref_info, consensus_dir)
    n_consensus_pixels = int(consensus_result.consensus_mask.sum())
    log(f"  consensus invariant targets: {n_consensus_pixels} pixels")

    reference_result.spectral_coverage = normalize.compute_spectral_coverage(
        reference_scene.analytic_path, (0, 0, ref_info.width, ref_info.height),
        [sensors.band_index(band_names, b) for b in band_names], band_names,
        consensus_result.consensus_mask, block_rows=block_rows,
    )

    log(f"Phase C: final regression fit + full-scene apply, {len(detected)} scenes...")
    n_done_c = [0]

    def _log_finalize_progress(pair, result):
        n_done_c[0] += 1
        log(f"  [{n_done_c[0]}/{len(detected)}] {pair[0].scene_id}: {result.status}"
            + (f" ({result.message})" if result.message else ""))

    target_results = _map_parallel(_finalize_pair, detected, workers, config, on_result=_log_finalize_progress)

    skipped_results = [
        SceneResult(scene.scene_id, outcome.status, message=outcome.message)
        for scene, outcome in zip(targets, candidate_outcomes) if outcome.status != "detected"
    ]
    all_results = [reference_result] + target_results + skipped_results
    results_by_id = {r.scene_id: r for r in all_results}

    usable = [
        (s, results_by_id[s.scene_id])
        for s in scenes
        if results_by_id[s.scene_id].status in ("fitted", "resumed", "reference")
    ]
    usable.sort(key=lambda pair: pair[0].acquired or datetime.min)

    log(f"Building adjacency report over {len(usable)} usable scenes...")
    pair_reports = _build_adjacency_report(usable, output_folder, min_overlap_pixels, workers)

    result = PipelineResult(
        reference_scene_id=reference_scene.scene_id,
        band_names=band_names,
        scene_results=all_results,
        pair_reports=pair_reports,
        consensus_paths=consensus_paths,
        n_consensus_pixels=n_consensus_pixels,
        output_dir=output_folder,
    )
    write_report(result, output_folder)
    write_spectral_coverage_report(all_results, output_folder)
    return result


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def write_report(result: PipelineResult, output_dir: str) -> None:
    csv_path = os.path.join(output_dir, "adjacent_pair_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_a", "scene_b", "time_a", "time_b", "band", "n_pixels",
            "r2_before", "r2_after", "delta_r2", "rmse_before", "rmse_after", "delta_rmse",
            "excluded_reason",
        ])
        for pair in result.pair_reports:
            if pair.excluded_reason is not None:
                writer.writerow([pair.scene_a, pair.scene_b, pair.time_a, pair.time_b, "", pair.n_pixels,
                                  "", "", "", "", "", "", pair.excluded_reason])
                continue
            for band_name, before in pair.before.items():
                after = pair.after[band_name]
                writer.writerow([
                    pair.scene_a, pair.scene_b, pair.time_a, pair.time_b, band_name, pair.n_pixels,
                    before["r2"], after["r2"], after["r2"] - before["r2"],
                    before["rmse"], after["rmse"], after["rmse"] - before["rmse"],
                    "",
                ])

    summary_path = os.path.join(output_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write(_render_summary(result))


def write_spectral_coverage_report(scene_results: list[SceneResult], output_dir: str) -> str:
    """One row per (scene, band) for every scene with computed
    spectral-coverage diagnostics (the reference scene, and every "fitted"
    target -- see normalize.compute_spectral_coverage): how well that
    scene's invariant-target pixels span its own real reflectance range in
    each band, a check against a regression that extrapolates outside the
    range it was actually fit on. Scenes reused via `resume` ("resumed"
    status) or that never got a fit ("error"/"skipped_*") have no fresh
    coverage data and are omitted.
    """
    csv_path = os.path.join(output_dir, "spectral_coverage_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_id", "band", "n_target_pixels", "target_variance", "whole_image_variance",
            "variance_ratio", "whole_image_p10", "whole_image_p90", "target_min", "target_max",
            "coverage_fraction",
        ])
        for result in scene_results:
            if not result.spectral_coverage:
                continue
            for band in result.spectral_coverage:
                writer.writerow([
                    result.scene_id, band.band_name, band.n_target_pixels, band.target_variance,
                    band.whole_image_variance, band.variance_ratio, band.whole_image_p10,
                    band.whole_image_p90, band.target_min, band.target_max, band.coverage_fraction,
                ])
    return csv_path


def _pooled_section(title: str, pairs: list[PairReport]) -> list[str]:
    lines = [f"## {title}", ""]
    if not pairs:
        lines.append("_no pairs in this group_")
        lines.append("")
        return lines

    delta_r2 = [p.after["pooled"]["r2"] - p.before["pooled"]["r2"] for p in pairs
                if not math.isnan(p.after["pooled"]["r2"]) and not math.isnan(p.before["pooled"]["r2"])]
    delta_rmse = [p.after["pooled"]["rmse"] - p.before["pooled"]["rmse"] for p in pairs]
    r2_before = [p.before["pooled"]["r2"] for p in pairs if not math.isnan(p.before["pooled"]["r2"])]
    r2_after = [p.after["pooled"]["r2"] for p in pairs if not math.isnan(p.after["pooled"]["r2"])]
    rmse_before = [p.before["pooled"]["rmse"] for p in pairs]
    rmse_after = [p.after["pooled"]["rmse"] for p in pairs]

    lines.append(f"- Pairs: {len(pairs)}")
    lines.append(f"- Mean R² before: {np.mean(r2_before):.4f}, after: {np.mean(r2_after):.4f}, mean ΔR²: {np.mean(delta_r2):+.4f}")
    lines.append(f"- Mean RMSE before: {np.mean(rmse_before):.2f}, after: {np.mean(rmse_after):.2f}, mean ΔRMSE: {np.mean(delta_rmse):+.2f}")
    n_improved_r2 = sum(1 for d in delta_r2 if d > 0)
    n_improved_rmse = sum(1 for d in delta_rmse if d < 0)
    lines.append(f"- Pairs with improved R²: {n_improved_r2}/{len(delta_r2)}; improved RMSE: {n_improved_rmse}/{len(delta_rmse)}")
    lines.append("")
    return lines


def _per_band_table(band_names: list[str], pairs: list[PairReport]) -> list[str]:
    lines = ["| band | mean R² before | mean R² after | mean ΔR² | mean RMSE before | mean RMSE after | mean ΔRMSE |",
             "|---|---|---|---|---|---|---|"]
    for band in band_names:
        r2_before = [p.before[band]["r2"] for p in pairs if band in p.before and not math.isnan(p.before[band]["r2"])]
        r2_after = [p.after[band]["r2"] for p in pairs if band in p.after and not math.isnan(p.after[band]["r2"])]
        rmse_before = [p.before[band]["rmse"] for p in pairs if band in p.before]
        rmse_after = [p.after[band]["rmse"] for p in pairs if band in p.after]
        if not r2_before:
            continue
        lines.append(
            f"| {band} | {np.mean(r2_before):.4f} | {np.mean(r2_after):.4f} | "
            f"{np.mean(r2_after) - np.mean(r2_before):+.4f} | {np.mean(rmse_before):.2f} | "
            f"{np.mean(rmse_after):.2f} | {np.mean(rmse_after) - np.mean(rmse_before):+.2f} |"
        )
    lines.append("")
    return lines


def _render_summary(result: PipelineResult) -> str:
    status_counts: dict[str, int] = {}
    for r in result.scene_results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    compared = [p for p in result.pair_reports if p.excluded_reason is None]
    excluded = [p for p in result.pair_reports if p.excluded_reason is not None]
    # PlanetScope revisits often capture a wide AOI as several strip segments
    # seconds apart, which end up chronologically adjacent AND overlapping
    # (little to no radiometric drift to correct) alongside the — usually
    # rarer — pairs that are actually different overpasses/days (where
    # atmosphere/sun-angle/sensor-state drift is the thing normalization is
    # meant to fix). Both match "temporally adjacent pair", so both are
    # reported, plus this breakdown so the more/less interesting comparisons
    # aren't blended into one number.
    same_day = [p for p in compared if p.time_a[:10] == p.time_b[:10]]
    cross_day = [p for p in compared if p.time_a[:10] != p.time_b[:10]]

    lines = ["# psnorm run summary", ""]
    lines.append(f"- Reference scene: `{result.reference_scene_id}`")
    lines.append(f"- Bands: {', '.join(result.band_names)}")
    lines.append(f"- Scenes: {len(result.scene_results)} total — " + ", ".join(f"{k}: {v}" for k, v in status_counts.items()))
    lines.append(f"- Consensus invariant targets: {result.n_consensus_pixels} pixels (reference grid)")
    lines.append(f"- Adjacent pairs: {len(result.pair_reports)} total, {len(compared)} compared ({len(same_day)} same-day, {len(cross_day)} cross-day), {len(excluded)} excluded")
    fallback_scenes = [r for r in result.scene_results if r.n_identity_fallback_bands > 0]
    n_fallback_bands = sum(r.n_identity_fallback_bands for r in fallback_scenes)
    lines.append(f"- Identity-fallback bands (fit rejected as too few pixels / implausible slope): "
                 f"{n_fallback_bands} band(s) across {len(fallback_scenes)} scene(s) — see model.json fallback_reason")
    lines.append("")

    if excluded:
        lines.append("## Excluded pairs")
        for p in excluded:
            lines.append(f"- {p.scene_a} vs {p.scene_b}: {p.excluded_reason}")
        lines.append("")

    if fallback_scenes:
        lines.append("## Scenes with identity-fallback bands")
        for r in fallback_scenes:
            lines.append(f"- {r.scene_id}: {r.n_identity_fallback_bands} band(s)")
        lines.append("")

    if compared:
        lines += _pooled_section("Pooled agreement — all compared pairs", compared)
        lines += _pooled_section("Pooled agreement — same-day pairs (seconds apart, same overpass)", same_day)
        lines += _pooled_section("Pooled agreement — cross-day pairs (different overpasses)", cross_day)

        band_names = list(result.band_names)
        lines.append("## Per-band mean Δ — all compared pairs")
        lines += _per_band_table(band_names, compared)
        if cross_day:
            lines.append("## Per-band mean Δ — cross-day pairs only")
            lines += _per_band_table(band_names, cross_day)

    return "\n".join(lines)
