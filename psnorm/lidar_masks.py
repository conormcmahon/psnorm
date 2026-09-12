"""Optional LiDAR/DSM-derived exclusion masks: horizontality (flat, level
ground), roughness (locally-variable surface orientation -- a proxy for
vegetation/building-edge contamination), and shadow-casting (ray-traced
against each scene's own sun position). Nothing in this module is invoked
unless a DSM path is supplied to the pipeline -- see PipelineConfig.dsm_path.

Vegetation and shadowed pixels are already excluded elsewhere (NDVI, UDM2,
OmniCloudMask -- see masking.py) using purely spectral evidence. These masks
add independent, purely-geometric evidence: a target is a better invariant
target if it also sits on flat, locally-uniform, currently-sunlit ground,
since sloped/rough surfaces have strong BRDF effects that spectral checks on
a single pair of images can't detect, and their spectral "invariance" between
any one pair is coincidental rather than physically stable.

Every mask here is computed against the DSM's own native grid/CRS -- not
resampled down to the coarser PlanetScope grid first, which would blur out
exactly the small building/canopy edges these masks exist to catch -- then
warped onto each scene's grid with max-resampling, so a single flagged DSM
pixel conservatively excludes whichever (coarser) PlanetScope pixel it falls
in.

Slope and roughness depend only on terrain, not on scene time, so
pipeline.py computes them once per run and reuses the result for every
scene. Shadow masks depend on per-scene sun position, but scenes captured in
the same overpass share near-identical sun angles, so pipeline.py caches
them to disk keyed on a rounded (azimuth, elevation) bucket -- this is a
practical first version of the "lookup table keyed on sun angle" reuse
scheme; see compute_shadow_mask's docstring for the more general fix
(precomputed per-pixel horizon angles) worth implementing if this needs to
scale to many years of imagery.
"""

from __future__ import annotations

import glob
import math
import os
import time
from dataclasses import dataclass, replace

import numpy as np
from osgeo import gdal, osr
from scipy.ndimage import binary_erosion, map_coordinates, uniform_filter

from . import io

gdal.UseExceptions()

_HEIGHT_UNIT_TO_M = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "ft": 0.3048, "feet": 0.3048, "foot": 0.3048, "us survey foot": 1200.0 / 3937.0,
}


def height_unit_factor(units: str) -> float:
    key = units.strip().lower()
    if key not in _HEIGHT_UNIT_TO_M:
        raise ValueError(f"Unknown dsm_height_units {units!r}; expected one of {sorted(_HEIGHT_UNIT_TO_M)}")
    return _HEIGHT_UNIT_TO_M[key]


@dataclass(frozen=True)
class DsmWindow:
    """A DSM read, cropped to some area of interest, on the DSM's own native
    grid/CRS (not resampled/reprojected) with heights already converted to
    meters. `valid` is False wherever the source DSM had nodata/NaN."""

    heights_m: np.ndarray
    valid: np.ndarray
    info: io.RasterInfo


def open_dsm_mosaic(dsm_path: str) -> gdal.Dataset:
    """Open `dsm_path` as a single GDAL dataset. If it's a directory, mosaic
    every *.tif/*.tiff inside via a VRT (a pointer to the underlying tiles,
    no data copied or resampled) so a tiled DSM delivery -- which may use a
    completely different tiling scheme than the PlanetScope imagery -- can
    be passed as-is without a separate merge step."""
    if os.path.isdir(dsm_path):
        tiles = sorted(glob.glob(os.path.join(dsm_path, "*.tif"))) + sorted(glob.glob(os.path.join(dsm_path, "*.tiff")))
        if not tiles:
            raise ValueError(f"No .tif/.tiff tiles found in DSM directory '{dsm_path}'.")
        vrt_path = os.path.join(dsm_path, "_psnorm_dsm_mosaic.vrt")
        gdal.BuildVRT(vrt_path, tiles)
        dataset = gdal.Open(vrt_path, gdal.GA_ReadOnly)
    else:
        dataset = gdal.Open(dsm_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"Could not open DSM at '{dsm_path}'.")
    return dataset


def dsm_pixel_size_m(dataset: gdal.Dataset) -> tuple[float, float]:
    """(x, y) pixel size in meters, converting from whatever linear unit the
    DSM's own horizontal CRS uses (feet, US survey feet, meters, ...) --
    read from the CRS itself, not assumed. For a compound CRS (horizontal +
    a vertical datum, common for LiDAR-derived DSMs), GDAL's
    GetLinearUnits() returns the horizontal component's unit, which is what
    pixel spacing is measured in."""
    srs = dataset.GetSpatialRef()
    factor = srs.GetLinearUnits() if srs is not None else 1.0
    gt = dataset.GetGeoTransform()
    return abs(gt[1]) * factor, abs(gt[5]) * factor


def _bounds_to_dsm_crs(scene_info: io.RasterInfo, dsm_crs_wkt: str) -> tuple[float, float, float, float]:
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(scene_info.crs)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromWkt(dsm_crs_wkt)
    dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src_srs, dst_srs)

    gt = scene_info.geotransform
    x0, y1 = gt[0], gt[3]
    x1 = x0 + scene_info.width * gt[1]
    y0 = y1 + scene_info.height * gt[5]
    corners = [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
    transformed = [transform.TransformPoint(x, y)[:2] for x, y in corners]
    xs = [p[0] for p in transformed]
    ys = [p[1] for p in transformed]
    return min(xs), min(ys), max(xs), max(ys)


def _union_bounds_to_dsm_crs(scene_infos: list[io.RasterInfo], dsm_crs_wkt: str) -> tuple[float, float, float, float]:
    """The union bounding box, in the DSM's CRS, of every scene in
    `scene_infos` -- each may be in its own CRS (transformed independently),
    though in practice every PlanetScope scene here shares one UTM zone."""
    all_bounds = [_bounds_to_dsm_crs(info, dsm_crs_wkt) for info in scene_infos]
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)
    return minx, miny, maxx, maxy


def read_dsm_window_for_scene(
    dataset: gdal.Dataset, scene_info: io.RasterInfo, *, height_units: str = "m", buffer_m: float = 0.0,
) -> DsmWindow | None:
    """Read the region of the DSM covering `scene_info`'s extent (padded by
    `buffer_m` on every side, so a shadow ray search near the scene's edge
    still has real DSM coverage to sample from) on the DSM's own native
    grid/CRS -- a crop, not a reprojection. Returns None if the DSM doesn't
    overlap the scene at all."""
    return read_dsm_window_for_scenes(dataset, [scene_info], height_units=height_units, buffer_m=buffer_m)


def _dsm_read_window(
    dataset: gdal.Dataset, scene_infos: list[io.RasterInfo], buffer_m: float,
) -> tuple[int, int, int, int, io.RasterInfo] | None:
    """The DSM-native pixel window (read_xoff, read_yoff, xsize, ysize) and
    corresponding RasterInfo covering the union of every scene in
    `scene_infos`, padded by `buffer_m`. Pure bounds arithmetic -- no pixel
    data touched -- shared by both the whole-array reader (read_dsm_window_
    for_scenes, used where random access across the area is unavoidable,
    e.g. the shadow ray-trace) and the row-blocked slope/roughness writer
    (compute_and_cache_slope_roughness, which never materializes the full
    window in memory). Returns None if the union doesn't overlap the DSM.
    """
    dsm_crs = dataset.GetProjectionRef()
    minx, miny, maxx, maxy = _union_bounds_to_dsm_crs(scene_infos, dsm_crs)

    srs = dataset.GetSpatialRef()
    unit_factor = srs.GetLinearUnits() if srs is not None else 1.0
    buffer_native = buffer_m / unit_factor if unit_factor else buffer_m
    minx, miny, maxx, maxy = minx - buffer_native, miny - buffer_native, maxx + buffer_native, maxy + buffer_native

    gt = dataset.GetGeoTransform()
    xoff = int(math.floor((minx - gt[0]) / gt[1]))
    yoff = int(math.floor((maxy - gt[3]) / gt[5]))
    xend = int(math.ceil((maxx - gt[0]) / gt[1]))
    yend = int(math.ceil((miny - gt[3]) / gt[5]))

    raster_w, raster_h = dataset.RasterXSize, dataset.RasterYSize
    read_xoff, read_yoff = max(0, xoff), max(0, yoff)
    read_xend, read_yend = min(raster_w, xend), min(raster_h, yend)
    xsize, ysize = read_xend - read_xoff, read_yend - read_yoff
    if xsize <= 0 or ysize <= 0:
        return None

    window_gt = (
        gt[0] + read_xoff * gt[1], gt[1], gt[2],
        gt[3] + read_yoff * gt[5], gt[4], gt[5],
    )
    info = io.RasterInfo(
        path=dataset.GetDescription(), width=xsize, height=ysize, band_count=1,
        crs=dsm_crs, geotransform=window_gt, nodata=None,
    )
    return read_xoff, read_yoff, xsize, ysize, info


def _read_dsm_block(band, height_units: str, xoff: int, yoff: int, xsize: int, ysize: int) -> tuple[np.ndarray, np.ndarray]:
    raw = band.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float32)
    nodata = band.GetNoDataValue()
    valid = np.isfinite(raw)
    if nodata is not None and not math.isnan(nodata):
        valid &= raw != nodata
    factor = height_unit_factor(height_units)
    heights_m = np.where(valid, raw * factor, 0.0).astype(np.float32)
    return heights_m, valid


def read_dsm_window_for_scenes(
    dataset: gdal.Dataset, scene_infos: list[io.RasterInfo], *, height_units: str = "m", buffer_m: float = 0.0,
) -> DsmWindow | None:
    """Read the region of the DSM covering the *union* of every scene in
    `scene_infos` (padded by `buffer_m` on every side), on the DSM's own
    native grid/CRS, as one in-memory array. Pass every scene whose
    exclusion mask will need LiDAR coverage -- not just the reference --
    since a scene extending beyond the reference's own footprint would
    otherwise get zero LiDAR flagging there, not because the DSM lacks data
    but because this window was never read that far. Returns None if the
    DSM doesn't overlap any of them.

    Only use this for computations that genuinely need random access across
    the whole window (the shadow ray-trace's directional sampling). For
    slope/roughness, which only ever need a small local neighborhood per
    pixel, use compute_and_cache_slope_roughness instead -- it processes in
    row-blocks and never holds the full window in memory, which matters
    once the window covers many scenes' combined extent rather than one.
    """
    window = _dsm_read_window(dataset, scene_infos, buffer_m)
    if window is None:
        return None
    read_xoff, read_yoff, xsize, ysize, info = window
    heights_m, valid = _read_dsm_block(dataset.GetRasterBand(1), height_units, read_xoff, read_yoff, xsize, ysize)
    return DsmWindow(heights_m=heights_m, valid=valid, info=info)


def compute_and_cache_slope_roughness(
    dataset: gdal.Dataset, scene_infos: list[io.RasterInfo], *,
    height_units: str = "m", buffer_m: float = 0.0,
    max_slope_deg: float, roughness_window_radius_px: int, roughness_max_deg: float,
    slope_path: str, roughness_path: str, block_rows: int = 1024,
) -> io.RasterInfo | None:
    """Like read_dsm_window_for_scenes + compute_horizontality_excluded +
    compute_roughness_excluded + masking.save_flags_raster combined, but
    processed in row-blocks (each padded by a `roughness_window_radius_px`
    halo of real neighboring DSM data, trimmed off after computing) and
    written incrementally to `slope_path`/`roughness_path` -- so peak memory
    is bounded by block_rows x width, not by the full window, regardless of
    how many scenes' combined extent the window covers. Returns the output
    RasterInfo (matching the window actually processed), or None if the DSM
    doesn't overlap any of `scene_infos`.
    """
    window = _dsm_read_window(dataset, scene_infos, buffer_m)
    if window is None:
        return None
    read_xoff, read_yoff, xsize, ysize, info = window
    band = dataset.GetRasterBand(1)
    px_m, py_m = dsm_pixel_size_m(dataset)

    # Write to temp paths and rename into place only on full success -- see
    # compute_and_cache_shadow's identical comment: a crash partway through
    # must not leave a partially-written file at the final path, which a
    # resumed run's os.path.exists check would wrongly treat as complete.
    slope_tmp_path = slope_path + ".tmp"
    roughness_tmp_path = roughness_path + ".tmp"
    driver = gdal.GetDriverByName("GTiff")
    slope_ds = driver.Create(slope_tmp_path, xsize, ysize, 1, gdal.GDT_Byte, options=["COMPRESS=LZW"])
    slope_ds.SetGeoTransform(info.geotransform)
    slope_ds.SetProjection(info.crs)
    rough_ds = driver.Create(roughness_tmp_path, xsize, ysize, 1, gdal.GDT_Byte, options=["COMPRESS=LZW"])
    rough_ds.SetGeoTransform(info.geotransform)
    rough_ds.SetProjection(info.crs)

    halo = max(1, roughness_window_radius_px)
    for block_start, n_rows in io.iter_row_blocks(ysize, block_rows):
        block_end = block_start + n_rows
        pad_top = min(halo, block_start)
        pad_bottom = min(halo, ysize - block_end)
        padded_yoff = read_yoff + block_start - pad_top
        padded_ysize = pad_top + n_rows + pad_bottom

        heights_m, valid = _read_dsm_block(band, height_units, read_xoff, padded_yoff, xsize, padded_ysize)

        slope_excluded = compute_horizontality_excluded(heights_m, valid, px_m, py_m, max_slope_deg=max_slope_deg)
        roughness_excluded = compute_roughness_excluded(
            heights_m, valid, px_m, py_m,
            window_radius_px=roughness_window_radius_px, max_roughness_deg=roughness_max_deg,
        )

        core = slice(pad_top, pad_top + n_rows)
        slope_ds.GetRasterBand(1).WriteArray(slope_excluded[core].astype(np.uint8), xoff=0, yoff=block_start)
        rough_ds.GetRasterBand(1).WriteArray(roughness_excluded[core].astype(np.uint8), xoff=0, yoff=block_start)

    slope_ds.GetRasterBand(1).FlushCache()
    rough_ds.GetRasterBand(1).FlushCache()
    slope_ds = None
    rough_ds = None
    os.replace(slope_tmp_path, slope_path)
    os.replace(roughness_tmp_path, roughness_path)
    return info


def compute_and_cache_shadow(
    dataset: gdal.Dataset, scene_infos: list[io.RasterInfo], *,
    height_units: str = "m", azimuth_deg: float, elevation_deg: float,
    max_building_height_m: float, local_avg_radius_px: int = 1, ray_step_m: float | None = None,
    shadow_path: str, block_rows: int = 256, log=lambda msg: None,
) -> io.RasterInfo | None:
    """Like read_dsm_window_for_scenes + compute_shadow_excluded +
    masking.save_flags_raster combined, but processed in row-blocks (each
    padded by a halo covering the *full shadow search radius* -- tens to
    hundreds of DSM pixels, driven by max_building_height_m/tan(elevation),
    vs. a couple pixels for roughness) and written incrementally to
    `shadow_path`. See compute_and_cache_slope_roughness for the same
    pattern with a much smaller halo; without this, a single whole-array
    shadow computation over a multi-scene union window is what caused a
    real OOM in production (the window read alone reached 7.66GB before the
    ray-march's own temporaries were even allocated).

    Blocks only split the window by rows, not columns -- each block still
    spans the full width, so no column halo is needed (a ray's horizontal
    reach is already covered) and only the row (vertical) halo, bounded by
    the same worst-case search distance, has to pad each block.

    Does not support `shadow_downsample_factor` (see PipelineConfig) --
    combining a block halo with resolution downsampling would reintroduce
    the same cross-block grid-phase-alignment hazard already solved
    elsewhere in this codebase for io.read_block_flat, and shadow masking's
    scope is deliberately kept to native-resolution processing for now.
    Downsampling is still applied by the (non-blocked) caller path when the
    union window is small enough not to need blocking at all.

    `log` (default a no-op) is called once per block with a progress
    string -- this is a genuinely long-running, previously-opaque
    computation (a single native-resolution bucket has been observed to
    take multiple hours over a several-dozen-scene union window), so
    surfacing per-block timing matters for judging whether an unattended
    run is actually progressing or stuck.
    """
    max_distance_m = max_building_height_m / math.tan(math.radians(elevation_deg))
    window_bounds = _dsm_read_window(dataset, scene_infos, max_distance_m)
    if window_bounds is None:
        return None
    read_xoff, read_yoff, xsize, ysize, info = window_bounds
    band = dataset.GetRasterBand(1)
    px_m, py_m = dsm_pixel_size_m(dataset)

    halo = int(math.ceil(max_distance_m / min(px_m, py_m))) + 1

    # Write to a temp path and rename into place only on full success, so a
    # crash partway through (this can run for hours -- exactly the sort of
    # long unattended stretch where the sandbox's recurring transient
    # failures are likeliest to land) leaves no file at `shadow_path` at
    # all, rather than a partially-written one that a resumed run's
    # os.path.exists check would wrongly treat as already complete.
    tmp_path = shadow_path + ".tmp"
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(tmp_path, xsize, ysize, 1, gdal.GDT_Byte, options=["COMPRESS=LZW"])
    out_ds.SetGeoTransform(info.geotransform)
    out_ds.SetProjection(info.crs)

    blocks = list(io.iter_row_blocks(ysize, block_rows))
    start_time = time.time()
    for i, (block_start, n_rows) in enumerate(blocks):
        block_end = block_start + n_rows
        pad_top = min(halo, block_start)
        pad_bottom = min(halo, ysize - block_end)
        padded_yoff = read_yoff + block_start - pad_top
        padded_ysize = pad_top + n_rows + pad_bottom

        heights_m, valid = _read_dsm_block(band, height_units, read_xoff, padded_yoff, xsize, padded_ysize)
        shadow_excluded = compute_shadow_excluded(
            heights_m, valid, px_m, py_m, azimuth_deg, elevation_deg,
            max_building_height_m=max_building_height_m, local_avg_radius_px=local_avg_radius_px, ray_step_m=ray_step_m,
        )

        core = slice(pad_top, pad_top + n_rows)
        out_ds.GetRasterBand(1).WriteArray(shadow_excluded[core].astype(np.uint8), xoff=0, yoff=block_start)

        elapsed = time.time() - start_time
        done = i + 1
        eta_s = elapsed / done * (len(blocks) - done)
        log(f"    block {done}/{len(blocks)} ({elapsed:.0f}s elapsed, ~{eta_s:.0f}s remaining for this bucket)")

    out_ds.GetRasterBand(1).FlushCache()
    out_ds = None
    os.replace(tmp_path, shadow_path)
    return info


def downsample_height_window(window: DsmWindow, factor: int) -> DsmWindow:
    """Block-average `window` by `factor`x`factor` native DSM pixels -- used
    to coarsen the working resolution specifically for the shadow ray-trace
    (see compute_shadow_excluded's cost discussion), independent of the
    resolution slope/roughness are computed at. A block's height is the
    mean of its valid sub-pixels (0.0, and marked invalid, if none are
    valid); a block is valid if any of its sub-pixels were."""
    if factor <= 1:
        return window
    h, w = window.heights_m.shape
    h2, w2 = h // factor, w // factor
    if h2 < 1 or w2 < 1:
        return window
    cropped_h, cropped_w = h2 * factor, w2 * factor
    heights = window.heights_m[:cropped_h, :cropped_w].reshape(h2, factor, w2, factor)
    valid = window.valid[:cropped_h, :cropped_w].reshape(h2, factor, w2, factor)
    valid_count = valid.sum(axis=(1, 3))
    heights_sum = np.where(valid, heights, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        heights_mean = np.where(valid_count > 0, heights_sum / np.maximum(valid_count, 1), 0.0)
    gt = window.info.geotransform
    new_gt = (gt[0], gt[1] * factor, gt[2], gt[3], gt[4], gt[5] * factor)
    info = replace(window.info, width=w2, height=h2, geotransform=new_gt)
    return DsmWindow(heights_m=heights_mean.astype(np.float32), valid=valid_count > 0, info=info)


def compute_slope_deg(heights_m: np.ndarray, pixel_size_x_m: float, pixel_size_y_m: float) -> np.ndarray:
    """Surface-normal angle from vertical (degrees), via a central-difference
    gradient of height. 0 deg = perfectly level."""
    dzdy, dzdx = np.gradient(heights_m, pixel_size_y_m, pixel_size_x_m)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def compute_roughness_deg(
    heights_m: np.ndarray, pixel_size_x_m: float, pixel_size_y_m: float, *, window_radius_px: int = 2,
) -> np.ndarray:
    """Local variation in surface orientation (degrees), via the Vector
    Ruggedness Measure (Sappington et al. 2007): average the unit surface-
    normal vector over a (2*window_radius_px+1) window, and take the angular
    deviation of that average from a perfectly-aligned result. 0 deg = every
    normal in the window points the same way (smooth, even if sloped);
    higher = normals point in increasingly divergent directions within the
    window (small bumps/pits -- the signature of vegetation canopy and
    building/tree edges, as opposed to a merely-sloped but smooth surface,
    which VRM does NOT flag since a slope's normals are still all parallel).
    """
    dzdy, dzdx = np.gradient(heights_m, pixel_size_y_m, pixel_size_x_m)
    norm = np.sqrt(dzdx**2 + dzdy**2 + 1.0)
    nx, ny, nz = -dzdx / norm, -dzdy / norm, 1.0 / norm
    size = 2 * window_radius_px + 1
    mean_nx = uniform_filter(nx, size=size, mode="nearest")
    mean_ny = uniform_filter(ny, size=size, mode="nearest")
    mean_nz = uniform_filter(nz, size=size, mode="nearest")
    resultant_length = np.clip(np.sqrt(mean_nx**2 + mean_ny**2 + mean_nz**2), 0.0, 1.0)
    return np.degrees(np.arccos(resultant_length))


def compute_horizontality_excluded(
    heights_m: np.ndarray, valid: np.ndarray, pixel_size_x_m: float, pixel_size_y_m: float, *, max_slope_deg: float = 5.0,
) -> np.ndarray:
    """True where a pixel should be EXCLUDED for not being flat/level.
    Pixels with no DSM coverage (`~valid`) are never excluded by this mask
    (fail-open, matching how a missing UDM2/OmniCloudMask result elsewhere
    in masking.py just skips that check rather than excluding everything)."""
    slope = compute_slope_deg(heights_m, pixel_size_x_m, pixel_size_y_m)
    return valid & (slope > max_slope_deg)


def compute_roughness_excluded(
    heights_m: np.ndarray, valid: np.ndarray, pixel_size_x_m: float, pixel_size_y_m: float, *,
    window_radius_px: int = 2, max_roughness_deg: float = 10.0,
) -> np.ndarray:
    """True where a pixel should be EXCLUDED for sitting in a locally rough
    (likely vegetated or edge-contaminated) area. `max_roughness_deg=10.0`
    is a starting default with no strong physical basis (unlike
    max_slope_deg, which directly encodes "level ground") -- expect to
    retune it after looking at the saved *_roughness_flag.tif diagnostic
    raster against a real scene. A window near invalid DSM data has its
    validity eroded by `window_radius_px` first, since VRM computed there
    is contaminated by the filled-zero placeholder used for the gradient."""
    roughness = compute_roughness_deg(heights_m, pixel_size_x_m, pixel_size_y_m, window_radius_px=window_radius_px)
    size = 2 * window_radius_px + 1
    valid_eroded = binary_erosion(valid, structure=np.ones((size, size), dtype=bool), border_value=0)
    return valid_eroded & (roughness > max_roughness_deg)


def compute_shadow_excluded(
    heights_m: np.ndarray, valid: np.ndarray, pixel_size_x_m: float, pixel_size_y_m: float,
    azimuth_deg: float, elevation_deg: float, *,
    max_building_height_m: float = 150.0, local_avg_radius_px: int = 1, ray_step_m: float | None = None,
) -> np.ndarray:
    """True where a pixel is likely in shadow at the given sun position.

    A vectorized ray-march shared across every pixel at once (sun direction
    is fixed for the whole array, so this is N whole-array shift-and-compare
    passes rather than a per-pixel loop). For each pixel, `local_avg_radius_px`
    denoises the DSM height used as that pixel's own surface (avoids single-
    pixel DSM noise spuriously self-shadowing). A straight-line ray toward
    the sun rises at tan(elevation) per unit distance; any DSM height sampled
    along that ray -- out to `max_building_height_m / tan(elevation)`, the
    farthest a max_building_height_m obstruction could still reach -- that
    exceeds the ray's height at that distance blocks the sun.

    Cost scales as O(pixels x samples), where samples ~ max_building_height_m
    / tan(elevation) / ray_step_m -- tens to hundreds of samples per pixel at
    native DSM resolution and a typical building height, hence `ray_step_m`
    (coarsen the marching step) as an explicit escape hatch. The principled
    fix if this needs to run over many scenes/years is a different
    algorithm entirely: precompute, once per pixel and independent of any
    particular scene, the horizon angle (highest obstruction angle above
    flat) in a fixed set of azimuth bins (e.g. every 5 deg) -- a "horizon
    angle" datacube, the standard technique for solar-shading rasters (see
    GRASS GIS r.horizon). Testing a given scene's sun position against it is
    then a single interpolated lookup per pixel instead of a ray-march, and
    the datacube itself doesn't depend on date/time at all so it's shared
    across every scene and every year for free. Not implemented here to keep
    this first pass's scope bounded -- worth building if shadow masking
    needs to scale past a handful of test scenes.
    """
    if elevation_deg <= 0:
        return valid.copy()

    heights_filled = np.where(valid, heights_m, 0.0)
    size = 2 * local_avg_radius_px + 1
    local_height = uniform_filter(heights_filled, size=size, mode="nearest") if local_avg_radius_px > 0 else heights_filled

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    d_row_per_m = -math.cos(az) / pixel_size_y_m
    d_col_per_m = math.sin(az) / pixel_size_x_m

    max_distance_m = max_building_height_m / math.tan(el)
    step_m = ray_step_m or min(pixel_size_x_m, pixel_size_y_m)
    n_steps = max(1, int(math.ceil(max_distance_m / step_m)))

    row_grid, col_grid = np.meshgrid(
        np.arange(heights_m.shape[0], dtype=np.float64), np.arange(heights_m.shape[1], dtype=np.float64), indexing="ij",
    )
    tan_el = math.tan(el)
    shadowed = np.zeros(heights_m.shape, dtype=bool)
    for step in range(1, n_steps + 1):
        distance_m = step * step_m
        ray_height = local_height + distance_m * tan_el
        sample_row = row_grid + distance_m * d_row_per_m
        sample_col = col_grid + distance_m * d_col_per_m
        sampled = map_coordinates(heights_filled, [sample_row, sample_col], order=1, mode="constant", cval=-1e9)
        shadowed |= sampled > ray_height

    return shadowed & valid


def warp_mask_to_scene(source_path: str, scene_info: io.RasterInfo) -> np.ndarray:
    """Reproject the boolean mask raster at `source_path` (on the DSM's own
    native grid) onto `scene_info`'s exact grid, using max-resampling so
    that a single True DSM pixel anywhere inside a coarser PlanetScope
    pixel's footprint conservatively marks that whole output pixel True,
    rather than being diluted/aliased away by an averaging or
    nearest-neighbor resample.

    Takes a file path (not a pre-loaded array) and lets gdal.Warp read it
    directly: GDAL's warp engine only touches the portion of the source
    that geographically overlaps the destination grid, rather than
    Python reading the entire source into memory first. That distinction
    matters once the source is a cache raster shared across many scenes'
    combined extent -- pre-loading the whole thing on every call, once per
    scene, across several concurrent worker processes, is what caused a
    real OOM in production (see pipeline._lidar_masks_for_scene).
    """
    driver = gdal.GetDriverByName("MEM")
    dst_ds = driver.Create("", scene_info.width, scene_info.height, 1, gdal.GDT_Byte)
    dst_ds.SetGeoTransform(scene_info.geotransform)
    dst_ds.SetProjection(scene_info.crs)

    gdal.Warp(dst_ds, source_path, resampleAlg=gdal.GRA_Max)
    out = dst_ds.GetRasterBand(1).ReadAsArray()
    dst_ds = None
    return out.astype(bool)
