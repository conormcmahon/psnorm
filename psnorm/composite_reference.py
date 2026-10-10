"""Extracting a small, scene-shaped reference window out of a multi-
statistic composite (see compositing.py / scripts/build_solstice_composite_reference.py).

A composite covers a whole AOI (tens of thousands of pixels per side,
tens of GB on disk) and carries one band per (canonical band, statistic)
pair (e.g. "blue_median", "nir_p75") rather than a scene's plain band
layout. Registering a single tile against it should never require reading
more than a small window of the composite -- this module is the
composite-specific counterpart to psnorm.sensors (which only knows about
plain per-scene band layouts) and is what keeps that size/shape mismatch
out of the (sensor/CRS-agnostic) planet_georeg_opencv library.
"""

from __future__ import annotations

import math
import os
import shutil

import numpy as np
import rioxarray  # noqa: F401 -- registers the .rio accessor used below
import xarray as xr
from osgeo import gdal

from . import io

gdal.UseExceptions()

_STAT_SUFFIXES = ("min", "p25", "median", "p75", "max", "mean", "stddev")

# Creation options for the single-stat cache built by ensure_single_stat_cache:
# tiled + band-interleaved, as opposed to a composite's own strip + pixel-
# interleaved layout (see that function's docstring for why this matters).
_CACHE_CREATION_OPTIONS = [
    "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "INTERLEAVE=BAND", "COMPRESS=LZW",
]


def composite_band_names(path: str) -> list[str]:
    """Return each band's GDAL Description, e.g. 'blue_median'."""
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"Could not open raster: {path}")
    names = [dataset.GetRasterBand(b).GetDescription().strip() for b in range(1, dataset.RasterCount + 1)]
    dataset = None
    return names


def is_composite(path: str) -> bool:
    """True if `path` looks like a compositing.py-produced multi-statistic
    raster (band descriptions end in a known statistic suffix), as opposed
    to a plain analytic scene psnorm.sensors.detect_band_names can handle
    directly."""
    return any(name.rsplit("_", 1)[-1] in _STAT_SUFFIXES for name in composite_band_names(path) if name)


def ensure_single_stat_cache(composite_path: str, *, stat: str = "median", cache_dir: str | None = None) -> str:
    """Build (if not already present) and return the path to a small,
    windowed-read-friendly derivative of `composite_path` holding only its
    `_{stat}` bands.

    compositing.py writes composites as GDAL's default strip-organized,
    pixel-interleaved GeoTIFF (one compressed strip per row, spanning the
    full raster width across ALL bands). That's fine for reading the whole
    raster at once, but it means reading even a single narrow column
    requires decompressing every band of every strip it touches -- for a
    60-band, tens-of-GB composite, a few-hundred-pixel-wide window can take
    over a minute. This builds a one-time, tiled + band-interleaved copy
    containing only the requested statistic's bands (e.g. 8 bands instead
    of 60), so later windowed reads via extract_reference_window only
    touch the tiles they actually need. The one-time build cost is
    amortized across every tile later registered against this composite.

    The cache lives in a `.composite_reference_cache` subdirectory next to
    `composite_path` by default. Returns the existing cache path
    immediately if it's already there -- this function does not detect a
    composite that has since changed on disk.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(composite_path), ".composite_reference_cache")
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(composite_path))[0]
    cache_path = os.path.join(cache_dir, f"{base}_{stat}.tif")
    if os.path.exists(cache_path):
        return cache_path

    suffix = f"_{stat}"
    names = composite_band_names(composite_path)
    band_indices = [i for i, name in enumerate(names, start=1) if name.endswith(suffix)]
    if not band_indices:
        raise ValueError(f"'{composite_path}' has no '{suffix}' bands.")

    tmp_path = cache_path + f".tmp{os.getpid()}"
    dataset = gdal.Translate(
        tmp_path, composite_path, format="GTiff", bandList=band_indices, creationOptions=_CACHE_CREATION_OPTIONS
    )
    dataset = None

    # Not os.replace(): on at least one network/virtual filesystem seen in
    # this environment (a virtiofs mount), a same-directory rename leaves
    # the destination name listed in the directory but permanently
    # unstatable/unopenable (ls shows it, stat/open raise ENOENT) --
    # os.replace's usual atomicity isn't available there anyway, so a
    # plain copy (verified to work) is used instead.
    shutil.copy2(tmp_path, cache_path)
    os.remove(tmp_path)
    return cache_path


def extract_reference_window(
    composite_path: str,
    target_info: io.RasterInfo,
    band_names: list[str],
    *,
    stat: str = "median",
    buffer_m: float = 500.0,
    cache_dir: str | None = None,
) -> xr.DataArray:
    """Read just the window of `composite_path` covering `target_info`'s
    extent plus `buffer_m` on each side (in the composite's CRS units --
    metres for UTM; rounded up to whole pixels and clamped to the
    composite's own bounds), for the `{name}_{stat}` band of each entry in
    `band_names`. PlanetScope geolocation is expected to be within roughly
    this distance of correct, so composite areas further away can't hold
    true matches and aren't worth computing keypoints for.

    Reads through ensure_single_stat_cache's tiled derivative rather than
    `composite_path` directly, so repeated calls against the same
    composite stay fast (see that function's docstring) and memory stays
    proportional to the target scene's own size regardless of how large
    the composite is.

    Returns an in-memory DataArray with one band per `band_names` entry
    (in that order, with composite nodata converted to NaN), CRS/transform
    set from the extracted window. It carries no band-name metadata of its
    own -- callers build a band_map positionally against `band_names`.

    Raises ValueError if any requested `{name}_{stat}` band is missing, or
    if the window doesn't overlap the composite at all.
    """
    cache_path = ensure_single_stat_cache(composite_path, stat=stat, cache_dir=cache_dir)

    composite_names = composite_band_names(cache_path)
    suffix = f"_{stat}"
    name_to_band_idx = {
        name[: -len(suffix)]: i
        for i, name in enumerate(composite_names, start=1)
        if name.endswith(suffix)
    }
    missing = [n for n in band_names if n not in name_to_band_idx]
    if missing:
        raise ValueError(f"'{composite_path}' has no '{suffix}' band for: {missing!r}")

    composite_info = io.get_raster_info(cache_path)
    window = io.overlap_window(composite_info, target_info)
    if window is None:
        raise ValueError(f"'{composite_path}' does not geographically overlap the target scene.")
    composite_window, _ = window

    margin_x_px = math.ceil(buffer_m / abs(composite_info.geotransform[1]))
    margin_y_px = math.ceil(buffer_m / abs(composite_info.geotransform[5]))
    xoff, yoff, xsize, ysize = composite_window
    xoff2 = max(0, xoff - margin_x_px)
    yoff2 = max(0, yoff - margin_y_px)
    xsize2 = min(composite_info.width - xoff2, (xoff - xoff2) + xsize + margin_x_px)
    ysize2 = min(composite_info.height - yoff2, (yoff - yoff2) + ysize + margin_y_px)

    dataset = gdal.Open(cache_path, gdal.GA_ReadOnly)
    arrays = []
    for name in band_names:
        band = dataset.GetRasterBand(name_to_band_idx[name])
        arr = band.ReadAsArray(xoff2, yoff2, xsize2, ysize2).astype(np.float32)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        arrays.append(arr)
    dataset = None

    windowed_info = io.windowed_raster_info(composite_info, (xoff2, yoff2, xsize2, ysize2))
    gt = windowed_info.geotransform
    xs = gt[0] + (np.arange(xsize2) + 0.5) * gt[1]
    ys = gt[3] + (np.arange(ysize2) + 0.5) * gt[5]

    result = xr.DataArray(
        np.stack(arrays, axis=0),
        coords={"band": np.arange(1, len(band_names) + 1), "y": ys, "x": xs},
        dims=["band", "y", "x"],
    )
    result.rio.write_crs(composite_info.crs, inplace=True)
    result.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    return result
