"""Per-scene exclusion masking: nodata, water (NDWI), vegetation (NDVI),
Planet's UDM2 quality mask, and an independent OmniCloudMask pass, combined
into one bitmask raster per scene so each exclusion source stays
individually distinguishable on disk (see EXCLUDE_* below) rather than
collapsing into a single opaque boolean.

This mask is only ever used to decide which pixels are *eligible to be
searched for invariant targets* (see irmad.py/normalize.py/pipeline.py) — the
final radiometric correction in apply.py is applied to every pixel of the
output scene regardless of what's flagged here, masked or not.

UDM2 is Planet's own per-pixel cloud/shadow/snow/haze classification,
delivered alongside every scene — cheap to use and a reasonable first pass,
but its cloud detector is known to miss thin cirrus/haze in particular.
OmniCloudMask is an independently-trained deep-learning cloud/shadow
detector (Wright et al., validated on PlanetScope among other sensors) that
catches many of the thin-cloud/haze cases UDM2 misses; `use_omnicloudmask`
controls whether it's included at all (it pulls in torch). Water is excluded
separately via NDWI since water surfaces are frequently unstable/non-
Lambertian and make poor invariant targets regardless of cloud status.
Vegetation is excluded via NDVI for the same reason: canopy reflectance
drifts with phenology/moisture/growth on timescales far shorter than the
multi-scene baseline invariant targets are meant to hold a fit stable
across, so IR-MAD flagging a vegetated pixel as unchanged *for one
particular pair* doesn't make it a trustworthy long-term target.
"""

from __future__ import annotations

import warnings

import numpy as np
from osgeo import gdal
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

from . import io, sensors

gdal.UseExceptions()

EXCLUDE_NODATA = 1
EXCLUDE_WATER = 2
EXCLUDE_UDM2 = 4
EXCLUDE_OMNICLOUD = 8
EXCLUDE_VEGETATION = 16
EXCLUDE_LIDAR_SLOPE = 32     # not flat/level (see lidar_masks.compute_horizontality_excluded)
EXCLUDE_LIDAR_ROUGH = 64     # locally rough surface orientation (see lidar_masks.compute_roughness_excluded)
EXCLUDE_LIDAR_SHADOW = 128   # likely shadowed at this scene's sun position (see lidar_masks.compute_shadow_excluded)

ALL_EXCLUDE_BITS = [
    EXCLUDE_NODATA, EXCLUDE_WATER, EXCLUDE_UDM2, EXCLUDE_OMNICLOUD,
    EXCLUDE_VEGETATION, EXCLUDE_LIDAR_SLOPE, EXCLUDE_LIDAR_ROUGH, EXCLUDE_LIDAR_SHADOW,
]


def read_udm2_valid_mask(udm2_path: str, window: io.Window) -> np.ndarray:
    """UDM2 `clear` band (band 1) == 1 over `window`, as a 2D boolean array."""
    dataset, bands = io.open_bands(udm2_path, band_indices=[1])
    xoff, yoff, xsize, ysize = window
    clear = bands[0].ReadAsArray(xoff, yoff, xsize, ysize)
    dataset = None
    return clear == 1


def read_nodata_valid_mask(analytic_path: str, window: io.Window) -> np.ndarray:
    """True where every band of `analytic_path` is not the nodata value,
    over `window`."""
    info = io.get_raster_info(analytic_path)
    arr = io.read_window_bands(analytic_path, window)  # (bands, y, x)
    if info.nodata is None:
        return np.ones(arr.shape[1:], dtype=bool)
    return np.all(arr != info.nodata, axis=0)


def compute_ndwi_water_mask(
    analytic_path: str,
    window: io.Window,
    band_names: list[str],
    *,
    ndwi_threshold: float = 0.0,
) -> np.ndarray | None:
    """True where NDWI = (green - nir) / (green + nir) > `ndwi_threshold`
    (McFeeters 1996 convention: water surfaces are typically NDWI > 0).
    0.0 is a reasonable default for atmospherically-corrected reflectance
    inputs; NDWI is a ratio so it's insensitive to a uniform DN scale
    factor (e.g. reflectance*10000 vs. raw fraction). Returns None (no
    pixels excluded) if this band set lacks green/nir.
    """
    indices = sensors.ndwi_band_indices(band_names)
    if indices is None:
        warnings.warn(
            f"NDWI water mask needs green/nir bands; band set "
            f"{band_names!r} doesn't have both — skipping for "
            f"'{analytic_path}'.",
            RuntimeWarning,
        )
        return None
    green_idx, nir_idx = indices
    arr = io.read_window_bands(analytic_path, window, band_indices=[green_idx, nir_idx])
    green, nir = arr[0], arr[1]
    denom = green + nir
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = np.where(denom != 0, (green - nir) / denom, 0.0)
    return ndwi > ndwi_threshold


def compute_ndvi_vegetation_mask(
    analytic_path: str,
    window: io.Window,
    band_names: list[str],
    *,
    ndvi_threshold: float = 0.2,
) -> np.ndarray | None:
    """True where NDVI = (nir - red) / (nir + red) > `ndvi_threshold`.

    Vegetation is a poor invariant target even when IR-MAD flags it as
    statistically unchanged between one particular pair of scenes: canopy
    reflectance drifts with phenology, moisture, and growth on timescales
    much shorter than the multi-scene baseline these targets are meant to
    hold radiometric fits stable across. 0.2 is the same default used by
    spectralmatch's PIF vegetation filter. Returns None (no pixels
    excluded) if this band set lacks red/nir.
    """
    indices = sensors.ndvi_band_indices(band_names)
    if indices is None:
        warnings.warn(
            f"NDVI vegetation mask needs red/nir bands; band set "
            f"{band_names!r} doesn't have both — skipping for "
            f"'{analytic_path}'.",
            RuntimeWarning,
        )
        return None
    red_idx, nir_idx = indices
    arr = io.read_window_bands(analytic_path, window, band_indices=[red_idx, nir_idx])
    red, nir = arr[0], arr[1]
    denom = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(denom != 0, (nir - red) / denom, 0.0)
    return ndvi > ndvi_threshold


def run_omnicloudmask(
    analytic_path: str,
    window: io.Window,
    band_names: list[str],
    *,
    nodata: float | None,
    downsample_factor: int = 5,
    omnicloud_kwargs: dict | None = None,
) -> np.ndarray | None:
    """OmniCloudMask clear mask (class 0 == clear) over `window`, or None if
    this band set doesn't have the red/green/nir bands OmniCloudMask needs.

    Cloud/shadow are large-scale features, so inference runs on a
    `downsample_factor`-reduced read (GDAL resamples during the read itself,
    so full resolution is never actually loaded) and the predicted mask is
    nearest-neighbor upsampled back to `window`'s native resolution. This
    mirrors spectralmatch's own `down_sample_m` option on
    create_cloud_mask_with_omnicloudmask, for the same reason: CPU inference
    over a full 3 m PlanetScope strip is far slower than the mask actually
    needs to be to be useful. downsample_factor=1 disables downsampling.
    """
    rgb_indices = sensors.cloudmask_band_indices(band_names)
    if rgb_indices is None:
        warnings.warn(
            f"OmniCloudMask needs red/green/nir bands; band set "
            f"{band_names!r} doesn't have all three — skipping for "
            f"'{analytic_path}'.",
            RuntimeWarning,
        )
        return None

    try:
        from omnicloudmask import predict_from_array
    except ImportError as exc:
        raise RuntimeError(
            "use_omnicloudmask=True but the 'omnicloudmask' package is not "
            "installed. Install it (`pip install omnicloudmask`, pulls in "
            "torch) or pass use_omnicloudmask=False."
        ) from exc

    xoff, yoff, xsize, ysize = window
    buf_w = max(1, xsize // downsample_factor)
    buf_h = max(1, ysize // downsample_factor)

    dataset, bands = io.open_bands(analytic_path, band_indices=list(rgb_indices))
    rgn = np.stack(
        [b.ReadAsArray(xoff, yoff, xsize, ysize, buf_xsize=buf_w, buf_ysize=buf_h) for b in bands],
        axis=0,
    ).astype(np.float32)
    dataset = None

    kwargs = dict(omnicloud_kwargs or {})
    kwargs.setdefault("no_data_value", nodata if nodata is not None else 0)
    pred = np.squeeze(predict_from_array(rgn, **kwargs))
    clear_small = pred == 0
    return io.nearest_resize(clear_small, (ysize, xsize))


def compute_exclusion_flags(
    analytic_path: str,
    udm2_path: str | None,
    band_names: list[str],
    *,
    use_water_mask: bool = True,
    ndwi_threshold: float = 0.0,
    use_vegetation_mask: bool = True,
    ndvi_threshold: float = 0.2,
    use_omnicloudmask: bool = True,
    omnicloud_downsample_factor: int = 5,
    omnicloud_kwargs: dict | None = None,
    lidar_slope_excluded: np.ndarray | None = None,
    lidar_rough_excluded: np.ndarray | None = None,
    lidar_shadow_excluded: np.ndarray | None = None,
) -> np.ndarray:
    """Bitmask (uint8) over the *entire* scene combining every exclusion
    source, each as its own bit (EXCLUDE_NODATA/WATER/UDM2/OMNICLOUD/
    VEGETATION/LIDAR_SLOPE/LIDAR_ROUGH/LIDAR_SHADOW) so a saved flags raster
    keeps each reason individually recoverable. 0 means clear/eligible for
    invariant-target search.

    Computed once per scene (not once per target-reference pair) — this is
    the shared basis both invariant-target detection and the adjacent-pair
    metrics report read from, and it's what gets persisted as the "final
    cloud mask" output.

    The three `lidar_*_excluded` arrays are optional, pre-warped-to-this-
    scene's-grid boolean masks from lidar_masks.py (pipeline.py computes
    these once, since they're DSM/sun-position derived rather than something
    this function can compute from the analytic image alone) — pass None
    (the default) for any/all of them to leave that exclusion reason out
    entirely, which is what happens whenever no DSM was supplied at all.
    """
    info = io.get_raster_info(analytic_path)
    window = (0, 0, info.width, info.height)
    flags = np.zeros((info.height, info.width), dtype=np.uint8)

    flags |= (~read_nodata_valid_mask(analytic_path, window)).astype(np.uint8) * EXCLUDE_NODATA

    if use_water_mask:
        water = compute_ndwi_water_mask(analytic_path, window, band_names, ndwi_threshold=ndwi_threshold)
        if water is not None:
            flags |= water.astype(np.uint8) * EXCLUDE_WATER

    if use_vegetation_mask:
        vegetation = compute_ndvi_vegetation_mask(analytic_path, window, band_names, ndvi_threshold=ndvi_threshold)
        if vegetation is not None:
            flags |= vegetation.astype(np.uint8) * EXCLUDE_VEGETATION

    if udm2_path is not None:
        not_clear = ~read_udm2_valid_mask(udm2_path, window)
        flags |= not_clear.astype(np.uint8) * EXCLUDE_UDM2
    else:
        warnings.warn(
            f"No UDM2 found for '{analytic_path}' — proceeding without "
            f"Planet's cloud/shadow/snow/haze mask.",
            RuntimeWarning,
        )

    if use_omnicloudmask:
        ocm_clear = run_omnicloudmask(
            analytic_path, window, band_names, nodata=info.nodata,
            downsample_factor=omnicloud_downsample_factor, omnicloud_kwargs=omnicloud_kwargs,
        )
        if ocm_clear is not None:
            flags |= (~ocm_clear).astype(np.uint8) * EXCLUDE_OMNICLOUD

    if lidar_slope_excluded is not None:
        flags |= lidar_slope_excluded.astype(np.uint8) * EXCLUDE_LIDAR_SLOPE
    if lidar_rough_excluded is not None:
        flags |= lidar_rough_excluded.astype(np.uint8) * EXCLUDE_LIDAR_ROUGH
    if lidar_shadow_excluded is not None:
        flags |= lidar_shadow_excluded.astype(np.uint8) * EXCLUDE_LIDAR_SHADOW

    return flags


def erode_dilate_bitmask(flags: np.ndarray, bit_values: list[int], *, erode_px: int = 1, dilate_px: int = 1) -> np.ndarray:
    """Morphologically open (erode then dilate) each exclusion reason in
    `bit_values` independently within the `flags` bitmask, then re-combine.
    Used to build a *search* mask (see pipeline._compute_and_save_flags,
    saved separately from the raw `_flags.tif`) that avoids edge effects at
    exclusion-region boundaries -- e.g. a mixed pixel straddling a cloud
    edge, or a misregistered building edge -- without disturbing the raw
    exclusion flags other consumers (reference selection, clear-percent
    reporting, the adjacent-pair metrics report) read.

    Applied per-plane (rather than to the collapsed "excluded for any
    reason" boolean) so every pixel's exclusion reason stays individually
    recoverable, including pixels added back by dilation -- a newly-added
    pixel keeps exactly the reason(s) whose own shape grew into it, instead
    of an ambiguous generic "morphology" bit.

    erode_px/dilate_px=0 disables that step (a pure dilation or pure
    erosion, or a no-op if both are 0) for whichever plane(s) it's applied
    to. Bits not listed in `bit_values` pass through unmodified.
    """
    out = np.zeros_like(flags)
    structure = generate_binary_structure(2, 1)
    touched_bits = 0
    for bit in bit_values:
        touched_bits |= bit
        plane = (flags & bit) != 0
        if erode_px > 0:
            plane = binary_erosion(plane, structure=structure, iterations=erode_px, border_value=0)
        if dilate_px > 0:
            plane = binary_dilation(plane, structure=structure, iterations=dilate_px, border_value=0)
        out |= plane.astype(flags.dtype) * bit
    dtype_mask = np.iinfo(flags.dtype).max
    untouched_mask = flags.dtype.type((~touched_bits) & dtype_mask)
    out |= flags & untouched_mask
    return out


def save_flags_raster(flags: np.ndarray, raster_info: io.RasterInfo, output_path: str) -> str:
    """Write a uint8 bitmask raster to `output_path`, georeferenced like
    `raster_info`."""
    return io.write_single_band_raster(flags.astype(np.uint8), raster_info, output_path, gdal.GDT_Byte)


def load_flags_window(path: str, window: io.Window) -> np.ndarray:
    dataset, bands = io.open_bands(path, band_indices=[1])
    xoff, yoff, xsize, ysize = window
    arr = bands[0].ReadAsArray(xoff, yoff, xsize, ysize)
    dataset = None
    return arr


def scene_clear_fraction(scene) -> float:
    """Fraction of the scene that's clear. Prefers metadata.json's
    `clear_percent` (cheap, Planet-computed); falls back to the mean of the
    UDM2 `clear` band."""
    meta = scene.metadata()
    clear_percent = meta.get("properties", {}).get("clear_percent")
    if clear_percent is not None:
        return clear_percent / 100.0

    if scene.udm2_path is None:
        raise ValueError(
            f"Scene '{scene.scene_id}' has neither metadata.json "
            f"clear_percent nor a UDM2 file — cannot estimate clear fraction."
        )
    dataset, bands = io.open_bands(scene.udm2_path, band_indices=[1])
    clear = bands[0].ReadAsArray()
    dataset = None
    return float(np.mean(clear == 1))
