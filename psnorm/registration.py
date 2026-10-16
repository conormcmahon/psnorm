"""Image co-registration between a target scene and the reference scene.

IR-MAD (irmad.py) requires the reference and target to be readable on a
shared pixel grid -- `pipeline.py` gets this for free when the two rasters
are already grid-aligned (see `io.grids_aligned`), which holds for
same-collection PlanetScope strips sharing a common ortho tiling grid.
`register_to_reference` is the real warp/align step, built on top of the
sensor/CRS-agnostic `planet_georeg_opencv` library (SIFT keypoint matching
over normalized-difference index channels, RANSAC, and either a local TPS
warp or a global affine warp -- see that library's own docs for how the
matching/warping actually works). This module only adds the PlanetScope-
and psnorm-specific pieces on top: band-name resolution (via
`psnorm.sensors`), UDM2 cloud masking, composite-reference support (via
`psnorm.composite_reference`, for registering a tile against a multi-
statistic composite instead of a single reference scene), picking which
of the library's two warp strategies keeps the result grid-aligned with
the reference, and writing the result in the target's own data type.

`is_grid_aligned` only checks CRS/pixel-size/pixel-integer-origin-offset --
it does NOT detect genuine sub-pixel/few-pixel geolocation error, which is
PlanetScope's well-known weak point even between same-collection strips.
Callers that want registration to actually correct that error (as opposed
to only reprojecting mismatched grids onto each other) should call
`register_to_reference` unconditionally rather than gating it on
`is_grid_aligned` -- see pipeline.py's `force_registration` option.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import asdict, dataclass, fields, replace

import numpy as np
import planet_georeg_opencv as pgo
import rioxarray as rxr
from osgeo import gdal, gdal_array
from planet_georeg_opencv import tps_warp
from planet_georeg_opencv.register import CV2_INTERPOLATION

from . import composite_reference, io, masking, sensors

gdal.UseExceptions()


class RegistrationFailed(Exception):
    """Raised when register_to_reference cannot produce an aligned raster
    (insufficient overlap, too few keypoint matches, RANSAC failure, etc),
    as opposed to a configuration error (e.g. a bad path)."""


@dataclass(frozen=True)
class RegistrationParams:
    """Every setting that controls registration.

    psnorm-specific:
      warp_model     "affine" (one global transform) or "tps" (local,
                     non-rigid); see pgo.register_local. Only applies when
                     the target and reference already share a grid; the
                     not-grid-aligned case is always a global affine onto
                     the reference's grid.
      buffer_m       Reference area searched for keypoints = the target's
                     extent plus this margin on every side (metres). For a
                     composite reference it also bounds how much of the
                     composite is read.
      reference_stat Composite statistic used as the reference (e.g.
                     "median"); ignored for a plain-scene reference.
      use_udm2_mask  Exclude pixels that the target's (and a plain-scene
                     reference's) UDM2 does not mark clear from keypoint
                     matching and metrics. The warped output still covers
                     every pixel.
      match_bands    Logical band names used for keypoint matching (all
                     bands are still warped). The default is the four
                     bands every PlanetScope product shares, as in the
                     original PlanetScope registration scripts; None uses
                     every band the target and reference share, which for
                     8-band scenes means 28 index channels instead of 6
                     (roughly 4-5x the time and keypoint memory).

      reference_keypoint_cache
                     For a composite reference, detect the composite's
                     keypoints once (stored next to it, see
                     composite_reference.ensure_reference_keypoints)
                     instead of again for every scene registered against
                     it. The stored keypoints cannot depend on the scene,
                     so matching channels are then scaled over a fixed
                     value range and the composite is detected at one of
                     reference_keypoint_scales; results differ slightly
                     from false (detect on both images for every scene).
      reference_keypoint_scales
                     Detection scales the composite's keypoints are stored
                     at (1.0 = the composite's own resolution). Each scene
                     uses the one nearest the scale its own keypoints are
                     detected at, min(1, max_sift_dimension / its larger
                     side), since SIFT matches best between similar scales.

      reference_cache_dir / reference_cache_compression
                     Where a composite reference's derived files (its
                     single-statistic copy and stored keypoints) are kept
                     -- default: next to the composite -- and the
                     compression of a newly built copy, "LZW" or "NONE".
                     Fast local storage and "NONE" make each scene's
                     reference window quicker to read; results are the
                     same. See composite_reference.ensure_single_stat_cache.
      sift_tile_size / sift_tile_overlap
                     Run SIFT detection on tiles of this many detection
                     pixels (plus the overlap on every side) rather than
                     on a scene's whole detection image, which cuts its
                     working memory from up to about 4 GB to about 0.4 GB
                     so more scenes can be registered at once; "none"
                     detects on the whole image (the library's default),
                     which gives slightly different keypoints near tile
                     edges. See pgo.register_local.

    Passed through to planet_georeg_opencv.register_local unchanged (the
    defaults are that library's own; see its docstring):
      min_pixel_count, min_keypoints, lowe_ratio_threshold,
      distance_threshold_pixels, ransac_reproj_threshold,
      keypoint_match_distance_threshold_m, blur_kernel_base_size,
      blur_kernel_reference_resolution, mask_erosion_blur_multiple,
      max_sift_dimension, tps_max_control_points, tps_coarse_grid_size,
      resampling_method, seed. The not-grid-aligned case
      (pgo.register_to_grid) uses the subset that applies to it and keeps
      that function's own lowe_ratio_threshold default.
    """

    warp_model: str = "affine"
    buffer_m: float = 500.0
    reference_stat: str = "median"
    use_udm2_mask: bool = True
    match_bands: tuple[str, ...] | None = ("blue", "green", "red", "nir")
    reference_keypoint_cache: bool = True
    reference_keypoint_scales: tuple[float, ...] = (1.0, 0.7, 0.5)
    reference_cache_dir: str | None = None
    reference_cache_compression: str = "LZW"
    sift_tile_size: int | None = 1024
    sift_tile_overlap: int = 128

    min_pixel_count: int = 1000
    min_keypoints: int = 10
    lowe_ratio_threshold: float = 1000.0
    distance_threshold_pixels: float = 50.0
    ransac_reproj_threshold: float = 2.0
    keypoint_match_distance_threshold_m: float = 15.0
    blur_kernel_base_size: int = 3
    blur_kernel_reference_resolution: float = 3.0
    mask_erosion_blur_multiple: float = 2.0
    max_sift_dimension: int = 4000
    tps_max_control_points: int | None = 2000
    tps_coarse_grid_size: int = 300
    resampling_method: str = "bilinear"
    seed: int = 0

    def __post_init__(self):
        if self.warp_model not in ("tps", "affine"):
            raise ValueError(f"warp_model must be 'tps' or 'affine', got {self.warp_model!r}")
        if not self.reference_keypoint_scales or not all(0 < s <= 1 for s in self.reference_keypoint_scales):
            raise ValueError(
                f"reference_keypoint_scales must be scales in (0, 1], got {self.reference_keypoint_scales!r}")
        if self.reference_cache_compression.upper() not in ("LZW", "NONE"):
            raise ValueError(
                f"reference_cache_compression must be 'LZW' or 'NONE', got {self.reference_cache_compression!r}")

    def register_local_kwargs(self) -> dict:
        psnorm_only = {"warp_model", "buffer_m", "reference_stat", "use_udm2_mask", "match_bands",
                       "reference_keypoint_cache", "reference_keypoint_scales",
                       "reference_cache_dir", "reference_cache_compression"}
        kwargs = {k: v for k, v in asdict(self).items() if k not in psnorm_only}
        kwargs["warp_model"] = self.warp_model
        kwargs["fixed_buffer_m"] = self.buffer_m
        return kwargs

    def register_to_grid_kwargs(self) -> dict:
        keys = ("min_pixel_count", "min_keypoints", "distance_threshold_pixels", "ransac_reproj_threshold",
                "blur_kernel_base_size", "blur_kernel_reference_resolution", "mask_erosion_blur_multiple",
                "max_sift_dimension", "seed")
        return {k: getattr(self, k) for k in keys}


# Integer settings that also accept "none" on the command line.
_NONE_ALLOWED = ("tps_max_control_points", "sift_tile_size")


def add_registration_arguments(parser) -> None:
    """Add `--registration-<field>` options (one per RegistrationParams
    field, defaulting to that field's default) to an argparse parser.
    Read them back with registration_params_from_args."""
    group = parser.add_argument_group("registration (see psnorm.registration.RegistrationParams)")
    for f in fields(RegistrationParams):
        flag = "--registration-" + f.name.replace("_", "-")
        default = f.default
        if isinstance(default, tuple) and isinstance(default[0], float):
            help_text = f"comma-separated numbers (default: {','.join(str(v) for v in default)})"
        elif isinstance(default, tuple):
            help_text = f"comma-separated band names, or 'all' (default: {','.join(default)})"
        elif isinstance(default, bool):
            help_text = f"true/false (default: {str(default).lower()})"
        else:
            help_text = f"default: {default}" + ("; 'none' for no limit" if f.name in _NONE_ALLOWED else "")
        group.add_argument(flag, dest="registration_" + f.name, default=None, help=help_text)


def _parse_param(name: str, text: str):
    default = next(f.default for f in fields(RegistrationParams) if f.name == name)
    lowered = text.strip().lower()
    if name == "reference_cache_dir":
        return text
    if isinstance(default, tuple) and isinstance(default[0], float):
        return tuple(float(t) for t in text.split(",") if t.strip())
    if isinstance(default, tuple) or default is None:
        return None if lowered in ("all", "none") else tuple(t.strip() for t in text.split(",") if t.strip())
    if isinstance(default, bool):
        if lowered not in ("true", "false"):
            raise ValueError(f"--registration-{name.replace('_', '-')} must be true or false, got {text!r}")
        return lowered == "true"
    if lowered == "none" and name in _NONE_ALLOWED:
        return None
    return type(default)(text)


def registration_params_from_args(args) -> RegistrationParams:
    """RegistrationParams from options added by add_registration_arguments
    (unset options keep their defaults)."""
    overrides = {
        f.name: _parse_param(f.name, getattr(args, "registration_" + f.name))
        for f in fields(RegistrationParams)
        if getattr(args, "registration_" + f.name, None) is not None
    }
    return replace(RegistrationParams(), **overrides)


@dataclass
class SceneRegistration:
    """register_scene's output: the library result plus what psnorm
    resolved to produce it."""
    result: "pgo.RegistrationResult"
    strategy: str  # "local" (grid-aligned, warped onto its own grid) or "grid" (onto the reference's grid)
    band_names: list[str]
    band_map: list[int]
    target_mask_used: bool
    target_clear_fraction: float | None
    # Detection scale of the stored composite keypoints used, or None when
    # the reference's keypoints were detected for this scene alone.
    reference_keypoint_scale: float | None = None
    # Where register_scene wrote the registered raster, if asked to.
    output_path: str | None = None


def is_grid_aligned(reference_info: io.RasterInfo, target_info: io.RasterInfo, tol: float = 1e-6) -> bool:
    """True if `target_info` can be directly overlapped with
    `reference_info` (see `io.overlap_window`) with no resampling."""
    return io.grids_aligned(reference_info, target_info, tol=tol)


_ANALYTIC_SUFFIX_RE = re.compile(r"_AnalyticMS[^/\\]*_clip\.tif$")


def udm2_path_for(analytic_path: str) -> str | None:
    """The UDM2 sidecar PlanetScope delivers next to an analytic clip
    (`..._3B_AnalyticMS_SR[_8b]_harmonized_clip.tif` ->
    `..._3B_udm2_clip.tif`), or None if the name doesn't follow that
    pattern or the file isn't there."""
    if not _ANALYTIC_SUFFIX_RE.search(analytic_path):
        return None
    path = _ANALYTIC_SUFFIX_RE.sub("_udm2_clip.tif", analytic_path)
    return path if os.path.exists(path) else None


def _clear_mask(udm2_path: str, info: io.RasterInfo) -> np.ndarray:
    return masking.read_udm2_valid_mask(udm2_path, (0, 0, info.width, info.height))


def _resolve_udm2(explicit: str | None, raster_path: str, use_mask: bool) -> str | None:
    if not use_mask:
        return None
    path = explicit or udm2_path_for(raster_path)
    if path is None:
        warnings.warn(f"No UDM2 found for '{raster_path}'; registering it without a cloud mask.", RuntimeWarning)
    return path


class _RegisteredWriter:
    """Writes a registered raster one band at a time, in the target's own
    data type, nodata value, band descriptions and metadata (LZW, tiled,
    band-interleaved, compressed on several threads). The file is only
    created when the first band arrives."""

    def __init__(self, target_path: str, output_path: str, *, width: int | None = None, height: int | None = None,
                 geotransform=None, projection: str | None = None):
        self.output_path = output_path
        self._target_path = target_path
        self._grid = (width, height, geotransform, projection)
        self._src = self._out = None

    def _create(self) -> None:
        self._src = gdal.Open(self._target_path)
        src_band = self._src.GetRasterBand(1)
        self._dtype_code = src_band.DataType
        self._nodata = src_band.GetNoDataValue()
        self._np_dtype = np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(self._dtype_code))
        width, height, geotransform, projection = self._grid
        self._width = width or self._src.RasterXSize
        self._height = height or self._src.RasterYSize
        self._geotransform = geotransform or self._src.GetGeoTransform()
        self._projection = projection or self._src.GetProjection()
        # INTERLEAVE=BAND because the bands arrive one at a time: in
        # GDAL's default pixel-interleaved layout every tile holds all
        # bands, so each band written re-compresses and re-appends every
        # tile already flushed from the block cache. With a 64 MB cache
        # that made an 8-band, 19-megapixel output take 29 s and 642 MB,
        # against under 1 s and 131 MB band-interleaved (the layout
        # PlanetScope's own analytic files use).
        options = ["COMPRESS=LZW", "TILED=YES", "INTERLEAVE=BAND", "BIGTIFF=IF_SAFER", "NUM_THREADS=" + io.gdal_thread_count()]
        if np.issubdtype(self._np_dtype, np.integer):
            options.append("PREDICTOR=2")
        self._out = gdal.GetDriverByName("GTiff").Create(
            self.output_path, self._width, self._height, self._src.RasterCount, self._dtype_code, options=options)
        self._out.SetGeoTransform(self._geotransform)
        self._out.SetProjection(self._projection)
        self._out.SetMetadata(self._src.GetMetadata())

    def write_band(self, index: int, warped: np.ndarray) -> None:
        """Write 0-indexed band `index` from `warped` (float, NaN where
        there is no data), which is converted in place."""
        if self._out is None:
            self._create()
        # Rounding and clipping leave NaN as NaN, which is then replaced
        # by the fill.
        fill = self._nodata if self._nodata is not None else 0
        missing = np.isnan(warped)
        if np.issubdtype(self._np_dtype, np.integer):
            info = np.iinfo(self._np_dtype)
            np.rint(warped, out=warped)
            np.clip(warped, info.min, info.max, out=warped)
        warped[missing] = fill
        del missing
        values = warped.astype(self._np_dtype)
        band = self._out.GetRasterBand(index + 1)
        band.WriteArray(values)
        if self._nodata is not None:
            band.SetNoDataValue(self._nodata)
        band.SetDescription(self._src.GetRasterBand(index + 1).GetDescription())

    def close(self) -> None:
        self._out = None
        self._src = None

    def verify(self) -> None:
        """Re-read every band (raises if one does not decode): on the
        virtiofs mount this project writes to, a file written while the
        machine ran out of memory once came back with an undecodable
        compressed block."""
        check = gdal.OpenEx(self.output_path, gdal.OF_RASTER | gdal.OF_READONLY, open_options=["NUM_THREADS=" + io.gdal_thread_count()])
        for i in range(1, check.RasterCount + 1):
            check.GetRasterBand(i).ReadAsArray()

    def discard(self) -> None:
        """Close and remove a partly written output."""
        created = self._out is not None
        self.close()
        if created and os.path.exists(self.output_path):
            os.remove(self.output_path)


def _reference_keypoints_for(reference_path: str, band_names: list[str], target_info: io.RasterInfo,
                             reference_window, params: RegistrationParams):
    """The stored keypoints of composite `reference_path` (built on first
    use) that fall in `reference_window`, at the stored detection scale
    nearest the one the target's own keypoints are detected at. Returns
    (keypoints, scale)."""
    target_scale = min(1.0, params.max_sift_dimension / max(target_info.width, target_info.height))
    scale = min(params.reference_keypoint_scales, key=lambda s: abs(math.log(s / target_scale)))
    keypoint_dir = composite_reference.ensure_reference_keypoints(
        reference_path, band_names, stat=params.reference_stat, detection_scale=scale,
        blur_kernel_base_size=params.blur_kernel_base_size,
        blur_kernel_reference_resolution=params.blur_kernel_reference_resolution,
        mask_erosion_blur_multiple=params.mask_erosion_blur_multiple,
        cache_dir=params.reference_cache_dir, cache_compression=params.reference_cache_compression,
    )
    return composite_reference.load_reference_keypoints(keypoint_dir, reference_window.rio.bounds()), scale


def register_scene(
    target_path: str,
    reference_path: str,
    params: RegistrationParams = RegistrationParams(),
    *,
    target_udm2_path: str | None = None,
    reference_udm2_path: str | None = None,
    output_path: str | None = None,
    sift_lock=None,
) -> SceneRegistration:
    """Run registration of `target_path` against `reference_path` with
    params.warp_model; see register_to_reference for the behaviour. The
    returned result's status may be a failure (never raises for one).

    With `output_path`, a successful registration is also written there
    (see write_registered for the format) and the returned
    SceneRegistration's output_path is set. For a grid-aligned target each
    band is written as it is warped, so the scene is warped once; without
    `output_path` nothing is written, and write_registered can apply the
    result afterwards at the cost of warping the matching bands again.

    UDM2 paths default to the sidecar next to each raster (see
    udm2_path_for) when params.use_udm2_mask is set. A composite reference
    has no UDM2 (its statistics are already built from clear pixels only).

    `sift_lock` is passed to pgo.register_local (see there): an optional
    inter-process lock so parallel registrations don't all hold SIFT's
    working memory at once (only worth having with
    params.sift_tile_size=None, when that memory is several GB).
    """
    return register_scene_models(
        target_path, reference_path, {params.warp_model: output_path}, params,
        target_udm2_path=target_udm2_path, reference_udm2_path=reference_udm2_path, sift_lock=sift_lock,
    )[params.warp_model]


def register_scene_models(
    target_path: str,
    reference_path: str,
    warp_outputs: dict,
    params: RegistrationParams = RegistrationParams(),
    *,
    target_udm2_path: str | None = None,
    reference_udm2_path: str | None = None,
    sift_lock=None,
) -> dict:
    """register_scene for several warp models in one pass: `warp_outputs`
    maps each model ("affine", "tps") to the path its registered raster is
    written to (or None to write nothing for it), and params.warp_model is
    ignored. Reading the scene, keypoint detection and matching -- most of
    the work -- happen once; each model's result and output are exactly
    what register_scene gives for that model alone.

    Returns {model: SceneRegistration}. A target that is not grid-aligned
    with its reference is always registered with one global affine onto
    the reference's grid (see register_to_reference), so there every
    model gets that same result.
    """
    if not warp_outputs:
        raise ValueError("warp_outputs must name at least one warp model")
    target_info = io.get_raster_info(target_path)
    target_names = sensors.detect_band_names(target_path)
    wanted = [n for n in target_names if params.match_bands is None or n in params.match_bands]

    target_udm2 = _resolve_udm2(target_udm2_path, target_path, params.use_udm2_mask)
    target_mask = _clear_mask(target_udm2, target_info) if target_udm2 else None
    reference_mask = None

    reference_info = io.get_raster_info(reference_path)
    grid_aligned = is_grid_aligned(reference_info, target_info)
    fixed_keypoints = keypoint_scale = None
    if composite_reference.is_composite(reference_path):
        reference_input = composite_reference.extract_reference_window(
            reference_path, target_info, wanted, stat=params.reference_stat, buffer_m=params.buffer_m,
            cache_dir=params.reference_cache_dir, cache_compression=params.reference_cache_compression,
        )
        reference_names = wanted
        if params.reference_keypoint_cache and grid_aligned and len(wanted) >= 2:
            fixed_keypoints, keypoint_scale = _reference_keypoints_for(
                reference_path, wanted, target_info, reference_input, params)
    else:
        reference_input = reference_path
        reference_names = sensors.detect_band_names(reference_path)
        reference_udm2 = _resolve_udm2(reference_udm2_path, reference_path, params.use_udm2_mask)
        if reference_udm2:
            reference_mask = _clear_mask(reference_udm2, reference_info)

    band_map = [
        reference_names.index(name) if name in wanted and name in reference_names else -1
        for name in target_names
    ]
    if sum(1 for b in band_map if b != -1) < 2:
        raise RegistrationFailed(
            f"Fewer than two shared matching bands between target {target_names!r} and reference "
            f"{reference_names!r} (match_bands={params.match_bands!r}); cannot compute a "
            f"normalized-difference index to register on."
        )

    writers = {}
    if grid_aligned:
        strategy = "local"
        # return_registered=False: each model's warp is streamed band by
        # band, to its writer here or by write_registered from
        # result.remap later.
        writers = {model: _RegisteredWriter(target_path, path) for model, path in warp_outputs.items() if path}

        def write_band(model, band, warped):
            if model in writers:
                writers[model].write_band(band, warped)

        library_settings = params.register_local_kwargs()
        library_settings.pop("warp_model")
        try:
            results = pgo.register_local_models(
                target_path, reference_input, band_map, tuple(warp_outputs),
                moving_valid_mask=target_mask, fixed_valid_mask=reference_mask,
                return_registered=False, on_warped_band=write_band if writers else None,
                fixed_keypoints=fixed_keypoints, sift_lock=sift_lock, **library_settings,
            )
        except BaseException:
            for writer in writers.values():
                writer.discard()
            raise
        for writer in writers.values():
            writer.close()
    else:
        strategy = "grid"
        result = pgo.register_to_grid(
            target_path, reference_input, band_map,
            moving_valid_mask=target_mask, fixed_valid_mask=reference_mask,
            **params.register_to_grid_kwargs(),
        )
        results = {model: result for model in warp_outputs}

    scenes = {}
    for model, output_path in warp_outputs.items():
        result = results[model]
        # The library's diagnostic clips are only for plotting; dropping
        # them frees the reference window.
        result.diagnostics.pop("moving_clip", None)
        result.diagnostics.pop("fixed_clip", None)
        scene = SceneRegistration(
            result=result, strategy=strategy, band_names=target_names, band_map=band_map,
            target_mask_used=target_mask is not None,
            target_clear_fraction=float(target_mask.mean()) if target_mask is not None else None,
            reference_keypoint_scale=keypoint_scale,
        )
        if output_path and result.status == "ok":
            model_params = replace(params, warp_model=model)
            if model in writers:
                try:
                    writers[model].verify()
                except RuntimeError:
                    # One more attempt covers a transient write failure
                    # (see _RegisteredWriter.verify); write_registered
                    # raises if its own output does not verify either.
                    write_registered(scene, target_path, output_path, model_params)
            else:
                write_registered(scene, target_path, output_path, model_params)
            scene.output_path = output_path
        scenes[model] = scene
    return scenes


def write_registered(registration: SceneRegistration, target_path: str, output_path: str,
                     params: RegistrationParams = RegistrationParams()) -> str:
    """Write a successful registration to `output_path` in the target's own
    data type, nodata value, band descriptions and metadata (LZW, tiled),
    then re-read it to confirm every band decodes (raises if not). A
    grid-aligned ("local") result is warped and written one band at a
    time, so the whole warped raster is never held in memory."""
    result = registration.result
    if result.status != "ok":
        raise ValueError(f"cannot write a failed registration ({result.status})")

    if registration.strategy == "local":
        moving = rxr.open_rasterio(target_path, masked=True, cache=False).squeeze()
        map_x, map_y = result.remap
        # Every band is resampled with the same (map_x, map_y).
        bands = tps_warp.iter_remapped_bands(moving, map_x, map_y, CV2_INTERPOLATION[params.resampling_method])
        writer = _RegisteredWriter(target_path, output_path)
    else:
        registered = result.registered
        bands = (registered.values[i] for i in range(registered.shape[0]))
        writer = _RegisteredWriter(
            target_path, output_path, width=registered.shape[2], height=registered.shape[1],
            geotransform=registered.rio.transform().to_gdal(), projection=registered.rio.crs.to_wkt(),
        )
    try:
        for i, warped in enumerate(bands):
            writer.write_band(i, warped)
            del warped
        writer.close()
    except BaseException:
        writer.discard()
        raise
    writer.verify()
    return output_path


def register_to_reference(
    target_path: str,
    reference_path: str,
    *,
    output_path: str | None = None,
    params: RegistrationParams = RegistrationParams(),
    target_udm2_path: str | None = None,
    reference_udm2_path: str | None = None,
) -> str:
    """Co-register `target_path` onto `reference_path`.

    `reference_path` may be a plain analytic scene (resolved via
    `sensors.detect_band_names`) or a compositing.py-produced multi-
    statistic composite (detected via `composite_reference.is_composite`
    and resolved by extracting just the `{band}_{params.reference_stat}`
    bands, windowed to `target_path`'s extent + params.buffer_m -- see
    composite_reference.py for why this never reads more than a small
    window of a composite that may be tens of GB on disk).

    Band correspondence is resolved by logical band name (blue/green/red/
    nir/...), restricted to params.match_bands, not by position.

    Two warp strategies are available in planet_georeg_opencv, and the
    right one depends on whether `target_path` and `reference_path`
    already share a grid:
      - If they do (the common case for scenes/composites built from the
        same raw archive and ortho tiling grid): `pgo.register_local`, a
        self-referential warp (params.warp_model) that corrects
        `target_path` onto ITS OWN pixel grid -- the output's geotransform
        is literally unchanged, so it trivially stays grid-aligned with
        the reference.
      - If they don't (different CRS/pixel size/origin phase):
        `pgo.register_to_grid`, a global affine warp onto the reference's
        own full pixel grid, so the result satisfies `is_grid_aligned`
        against it afterward as this function's contract requires.

    See RegistrationParams for every setting.

    Returns the path to the registered raster (`output_path`, or a
    derived default next to `target_path` if not given), written in the
    target's own data type with its band count, order, descriptions and
    nodata value.

    Raises:
        RegistrationFailed: if registration could not be completed (e.g.
            no geographic overlap, too few keypoint matches, or fewer
            than two shared bands between target and reference).
    """
    if output_path is None:
        base, ext = os.path.splitext(target_path)
        output_path = f"{base}_registered{ext}"
    registration = register_scene(
        target_path, reference_path, params,
        target_udm2_path=target_udm2_path, reference_udm2_path=reference_udm2_path, output_path=output_path,
    )
    result = registration.result
    if result.status != "ok":
        raise RegistrationFailed(f"register_to_reference('{target_path}'): {result.status} -- {result.message}")
    return registration.output_path
