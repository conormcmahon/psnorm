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

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import rioxarray  # noqa: F401 -- registers the .rio accessor used below
import xarray as xr
from osgeo import gdal

from . import io, sensors

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


def ensure_single_stat_cache(composite_path: str, *, stat: str = "median", cache_dir: str | None = None,
                             compression: str = "LZW") -> str:
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

    `compression` is the GeoTIFF compression of a newly built cache: "LZW"
    (about 1-1.6 GB for a 93-megapixel, 8-band composite) or "NONE" (about
    3 GB, but a scene-sized window then reads in about a second instead of
    5-6 s of decompression). The pixel values are the same either way, and
    an existing cache file is used as it is, whatever its compression.

    The cache lives in a `.composite_reference_cache` subdirectory next to
    `composite_path` by default; `cache_dir` puts it elsewhere -- worth
    doing when the composites are on slow storage, since a window read
    for a large scene touches hundreds of MB of this file (measured: 20 s
    of a 126 s scene when it was read uncached from a 90 MB/s disk).
    Returns the existing cache path
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
        tmp_path, composite_path, format="GTiff", bandList=band_indices,
        creationOptions=[o for o in _CACHE_CREATION_OPTIONS if not o.startswith("COMPRESS=")]
        + [f"COMPRESS={compression.upper()}", "BIGTIFF=IF_SAFER"],
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
    cache_compression: str = "LZW",
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
    cache_path = ensure_single_stat_cache(composite_path, stat=stat, cache_dir=cache_dir,
                                          compression=cache_compression)

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

    # Decoded on several threads: the window is a few hundred compressed
    # tiles per band.
    dataset = gdal.OpenEx(cache_path, gdal.OF_RASTER | gdal.OF_READONLY, open_options=["NUM_THREADS=" + io.gdal_thread_count()])
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


# --------------------------------------------------------------------------
# Reference keypoints, detected once per composite
# --------------------------------------------------------------------------

# Bump when the stored keypoints would differ for the same inputs (a change
# to how planet_georeg_opencv prepares or detects them, or to this layout).
_KEYPOINT_CACHE_VERSION = 5
# Keypoints are stored sorted by the cell (of this many composite pixels)
# they fall in, so the keypoints of a window are a few contiguous runs.
_KEYPOINT_CELL_PX = 512
# Detection tile (in detection pixels): bounds SIFT's working memory during
# the one-time build to about 1 GB whatever the composite's size.
_KEYPOINT_TILE_SIZE = 2000


def _keypoint_cache_dir(cache_path: str, band_names: list[str], settings: dict) -> str:
    key = json.dumps({"bands": list(band_names), **settings, "version": _KEYPOINT_CACHE_VERSION}, sort_keys=True)
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    scale_tag = f"{settings['detection_scale']:.3f}".replace(".", "p")
    return f"{os.path.splitext(cache_path)[0]}.keypoints_s{scale_tag}_{digest}"


def ensure_reference_keypoints(
    composite_path: str,
    band_names: list[str],
    *,
    stat: str = "median",
    detection_scale: float = 1.0,
    blur_kernel_base_size: int = 3,
    blur_kernel_reference_resolution: float = 3.0,
    mask_erosion_blur_multiple: float = 2.0,
    cache_dir: str | None = None,
    cache_compression: str = "LZW",
) -> str:
    """Detect (if not already stored) the SIFT keypoints of the whole
    composite's `{name}_{stat}` bands, for every normalized-difference
    channel formed from `band_names` in that order, and return the
    directory they are stored in (see load_reference_keypoints).

    Registering a scene against a composite otherwise repeats the
    composite's half of keypoint detection for every scene, over much the
    same pixels each time; this does it once per composite (per band list,
    detection scale and blur settings -- each combination is its own
    directory, next to the single-stat cache from ensure_single_stat_cache).
    See planet_georeg_opencv.detect_reference_keypoints for what the
    settings mean and how these keypoints differ from ones detected for a
    single scene pair.

    Safe to call from several processes at once: the build is serialized
    by a lock, and a directory only counts as built once its `meta.json`
    is there (written last).
    """
    cache_path = ensure_single_stat_cache(composite_path, stat=stat, cache_dir=cache_dir,
                                          compression=cache_compression)
    settings = {
        "detection_scale": float(detection_scale),
        "blur_kernel_base_size": int(blur_kernel_base_size),
        "blur_kernel_reference_resolution": float(blur_kernel_reference_resolution),
        "mask_erosion_blur_multiple": float(mask_erosion_blur_multiple),
    }
    out_dir = _keypoint_cache_dir(cache_path, band_names, settings)
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        return out_dir

    # Deferred: psnorm.registration is the module that otherwise owns the
    # dependency on the registration library.
    import planet_georeg_opencv as pgo

    lock_path = os.path.join(tempfile.gettempdir(), "psnorm_keypoints_" + os.path.basename(out_dir) + ".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if os.path.exists(meta_path):  # built by another process while this one waited
                return out_dir

            names = composite_band_names(cache_path)
            suffix = f"_{stat}"
            missing = [n for n in band_names if n + suffix not in names]
            if missing:
                raise ValueError(f"'{composite_path}' has no '{suffix}' band for: {missing!r}")
            band_indices = [names.index(n + suffix) for n in band_names]

            reference = pgo.detect_reference_keypoints(
                cache_path, band_indices, tile_size=_KEYPOINT_TILE_SIZE, **settings,
            )
            info = io.get_raster_info(cache_path)
            cells_across = math.ceil(info.width / _KEYPOINT_CELL_PX)
            cells_down = math.ceil(info.height / _KEYPOINT_CELL_PX)

            # Written to local temporary files and copied into place (not
            # renamed -- see ensure_single_stat_cache), meta.json last.
            os.makedirs(out_dir, exist_ok=True)
            counts = {}
            with tempfile.TemporaryDirectory() as tmp:
                for (a, b), features in reference.channels.items():
                    name = f"{band_names[band_indices.index(a)]}-{band_names[band_indices.index(b)]}"
                    cell = ((features.points[:, 1] // _KEYPOINT_CELL_PX).astype(np.int64) * cells_across
                            + (features.points[:, 0] // _KEYPOINT_CELL_PX).astype(np.int64))
                    order = np.argsort(cell, kind="stable")
                    # cell_starts[c] .. cell_starts[c + 1] are cell c's rows.
                    cell_starts = np.searchsorted(cell[order], np.arange(cells_across * cells_down + 1))
                    for part, array in (("points", features.points[order]),
                                        ("descriptors", features.descriptors[order]),
                                        ("cells", cell_starts)):
                        tmp_file = os.path.join(tmp, f"{name}.{part}.npy")
                        np.save(tmp_file, array)
                        shutil.copy(tmp_file, os.path.join(out_dir, f"{name}.{part}.npy"))
                    counts[name] = int(len(order))
                meta = {
                    **settings, "bands": list(band_names), "stat": stat, "source": os.path.basename(cache_path),
                    "geotransform": list(info.geotransform), "crs": info.crs, "width": info.width,
                    "height": info.height, "cell_px": _KEYPOINT_CELL_PX, "cells_across": cells_across,
                    "cells_down": cells_down, "keypoints": counts,
                    "value_ranges": {f"{band_names[band_indices.index(a)]}-{band_names[band_indices.index(b)]}": list(r)
                                     for (a, b), r in reference.value_ranges.items()},
                    "version": _KEYPOINT_CACHE_VERSION,
                }
                tmp_meta = os.path.join(tmp, "meta.json")
                with open(tmp_meta, "w") as f:
                    json.dump(meta, f, indent=2)
                shutil.copy(tmp_meta, meta_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return out_dir


def load_reference_keypoints(keypoint_dir: str, bounds: tuple[float, float, float, float]):
    """The stored keypoints (see ensure_reference_keypoints) that can fall
    inside `bounds` = (left, bottom, right, top) in the composite's CRS,
    as a planet_georeg_opencv.ReferenceKeypoints whose channels are keyed
    by position in the stored band list: (i, j) is the channel of
    band_names[i] and band_names[j]. That matches a reference image whose
    bands are those names in that order, such as extract_reference_window
    returns.

    Only the cells of the stored grid that touch `bounds` are read, so
    time and memory follow the size of the window, not of the composite;
    the result may include keypoints a little outside `bounds`.
    """
    import planet_georeg_opencv as pgo
    from affine import Affine
    from rasterio.crs import CRS

    with open(os.path.join(keypoint_dir, "meta.json")) as f:
        meta = json.load(f)
    transform = Affine.from_gdal(*meta["geotransform"])
    left, bottom, right, top = bounds
    col_a, row_a = ~transform * (left, top)
    col_b, row_b = ~transform * (right, bottom)
    cell_px, cells_across, cells_down = meta["cell_px"], meta["cells_across"], meta["cells_down"]
    cell_col0 = min(max(int(min(col_a, col_b) // cell_px), 0), cells_across - 1)
    cell_col1 = min(max(int(max(col_a, col_b) // cell_px), 0), cells_across - 1)
    cell_row0 = min(max(int(min(row_a, row_b) // cell_px), 0), cells_down - 1)
    cell_row1 = min(max(int(max(row_a, row_b) // cell_px), 0), cells_down - 1)

    bands = meta["bands"]
    channels = {}
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            name = f"{bands[i]}-{bands[j]}"
            cell_starts = np.load(os.path.join(keypoint_dir, f"{name}.cells.npy"))
            points = np.load(os.path.join(keypoint_dir, f"{name}.points.npy"), mmap_mode="r")
            descriptors = np.load(os.path.join(keypoint_dir, f"{name}.descriptors.npy"), mmap_mode="r")
            point_runs, descriptor_runs = [], []
            for cell_row in range(cell_row0, cell_row1 + 1):
                start = cell_starts[cell_row * cells_across + cell_col0]
                stop = cell_starts[cell_row * cells_across + cell_col1 + 1]
                if stop > start:
                    point_runs.append(np.array(points[start:stop]))
                    descriptor_runs.append(np.array(descriptors[start:stop]))
            del points, descriptors
            channels[(i, j)] = pgo.reference_keypoints.DetectedFeatures(
                np.concatenate(point_runs) if point_runs else np.zeros((0, 2), np.float32),
                np.concatenate(descriptor_runs) if descriptor_runs else np.zeros((0, 128), np.uint8),
            )
    return pgo.ReferenceKeypoints(
        channels=channels, transform=transform, crs=CRS.from_wkt(meta["crs"]),
        value_ranges={(i, j): tuple(meta["value_ranges"][f"{bands[i]}-{bands[j]}"])
                      for i in range(len(bands)) for j in range(i + 1, len(bands))},
        detection_scale=meta["detection_scale"],
    )


# --------------------------------------------------------------------------
# Summer solstice date (Meeus low-precision mean solstice/equinox formula)
# --------------------------------------------------------------------------

def _jde_to_datetime(jde: float) -> datetime:
    """Julian Ephemeris Day -> UTC calendar datetime (Meeus, Astronomical
    Algorithms ch.7's standard JD-to-Gregorian-calendar algorithm)."""
    jde = jde + 0.5
    z = math.floor(jde)
    f = jde - z
    if z < 2299161:
        a = z
    else:
        alpha = math.floor((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - math.floor(alpha / 4)
    b = a + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)
    day = b - d - math.floor(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    day_int = int(math.floor(day))
    seconds = round((day - day_int) * 86400)
    return datetime(int(year), int(month), day_int) + timedelta(seconds=seconds)


def summer_solstice(year: int, hemisphere: str) -> datetime:
    """UTC instant of the summer solstice for `hemisphere` ('N' -> the
    June solstice, 'S' -> the December solstice) in `year`, via Meeus's
    low-precision mean-solstice polynomial (Astronomical Algorithms
    ch.27, valid 1000-3000 CE). This omits the ~24-term periodic
    correction Meeus adds for sub-hour precision -- the mean value alone
    is already accurate to well under a day (checked here against the
    known June 2018/2024 solstices), which is all a two-week selection
    window needs."""
    if hemisphere not in ("N", "S"):
        raise ValueError(f"hemisphere must be 'N' or 'S', got {hemisphere!r}")
    yp = (year - 2000) / 1000.0
    if hemisphere == "N":
        jde = 2451716.56767 + 365241.62603 * yp + 0.00325 * yp**2 + 0.00888 * yp**3 - 0.00030 * yp**4
    else:
        jde = 2451900.05952 + 365242.74049 * yp - 0.06223 * yp**2 - 0.00823 * yp**3 + 0.00032 * yp**4
    return _jde_to_datetime(jde)


# --------------------------------------------------------------------------
# Composites as normalization references (pipeline.run_pipeline's
# reference_composite; scripts/run_composite_reference_pipeline.py)
# --------------------------------------------------------------------------

# A band counts as present in a composite when it is valid over at least
# this fraction of the area its best-covered band is valid over. The
# composites built before 8-band data existed carry the 8-band-only bands
# (coastal_blue, green_i, yellow, rededge) as all-nodata layers.
DEFAULT_MIN_BAND_COVERAGE = 0.9
# Bands only the 8-band (SuperDove) product has.
EIGHT_BAND_ONLY = tuple(b for b in sensors.BAND_PROFILES[8] if b not in sensors.BAND_PROFILES[4])
# Band coverage is measured on a decimated read this many pixels across.
_COVERAGE_SAMPLE_PX = 1024


@dataclass(frozen=True)
class ReferenceComposite:
    """One per-year composite (e.g. solstice_2023.tif) and the acquisition
    window it was built from: `solstice` +- `window_days` (see
    scripts/build_solstice_composite_reference.py's --window-days)."""
    path: str
    year: int
    solstice: datetime
    window_days: float

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def window_start(self) -> datetime:
        return self.solstice - timedelta(days=self.window_days)


def find_solstice_composites(folder: str, *, prefix: str = "solstice", hemisphere: str = "N",
                             window_days: float = 14) -> list[ReferenceComposite]:
    """Every `<prefix>_<year>.tif` in `folder`, by year."""
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{4}})\.tif")
    out = []
    for name in os.listdir(folder):
        m = pattern.fullmatch(name)
        if m:
            year = int(m.group(1))
            out.append(ReferenceComposite(os.path.join(folder, name), year, summer_solstice(year, hemisphere),
                                          float(window_days)))
    return sorted(out, key=lambda c: c.year)


def band_coverage(composite_path: str, *, stat: str = "median", cache_dir: str | None = None,
                  cache_compression: str = "LZW") -> dict[str, float]:
    """{band: fraction of the composite's pixels where `{band}_{stat}` is
    valid (not nodata, finite)}, from a decimated read of
    ensure_single_stat_cache's file; saved as JSON next to that file so it
    is measured once per composite."""
    cache_path = ensure_single_stat_cache(composite_path, stat=stat, cache_dir=cache_dir,
                                          compression=cache_compression)
    json_path = os.path.splitext(cache_path)[0] + ".band_coverage.json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    suffix = f"_{stat}"
    dataset = gdal.Open(cache_path, gdal.GA_ReadOnly)
    scale = _COVERAGE_SAMPLE_PX / max(dataset.RasterXSize, dataset.RasterYSize)
    buf_x = max(1, round(dataset.RasterXSize * min(1.0, scale)))
    buf_y = max(1, round(dataset.RasterYSize * min(1.0, scale)))
    coverage = {}
    for b in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(b)
        name = band.GetDescription().strip()
        if not name.endswith(suffix):
            continue
        arr = band.ReadAsArray(buf_xsize=buf_x, buf_ysize=buf_y, resample_alg=gdal.GRIORA_NearestNeighbour)
        valid = np.isfinite(arr)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            valid &= arr != nodata
        coverage[name[: -len(suffix)]] = float(valid.mean())
    dataset = None
    # Written in place (see ensure_single_stat_cache on renames here).
    with open(json_path, "w") as f:
        json.dump(coverage, f, indent=2)
    return coverage


def present_bands(coverage: dict[str, float], min_band_coverage: float = DEFAULT_MIN_BAND_COVERAGE) -> list[str]:
    """The bands of `coverage` (see band_coverage) valid over at least
    `min_band_coverage` of the area its best-covered band is, in
    `coverage`'s (the composite's) band order."""
    best = max(coverage.values(), default=0.0)
    if best <= 0:
        return []
    return [b for b, c in coverage.items() if c >= min_band_coverage * best]


def ensure_reference_raster(
    composite_path: str,
    band_names: list[str],
    out_dir: str,
    *,
    stat: str = "median",
    cache_dir: str | None = None,
    cache_compression: str = "LZW",
) -> str:
    """A VRT in `out_dir` over ensure_single_stat_cache's file holding the
    `{name}_{stat}` band of each of `band_names`, in that order, described
    by the plain band name ("red", not "red_median") -- so the pipeline can
    read the composite like any scene (sensors.detect_band_names, band
    indices). No pixels are copied."""
    cache_path = os.path.abspath(ensure_single_stat_cache(composite_path, stat=stat, cache_dir=cache_dir,
                                                          compression=cache_compression))
    names = composite_band_names(cache_path)
    missing = [b for b in band_names if f"{b}_{stat}" not in names]
    if missing:
        raise ValueError(f"'{composite_path}' has no '_{stat}' band for: {missing!r}")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(composite_path))[0]
    vrt_path = os.path.join(out_dir, f"{base}_{stat}_reference.vrt")
    dataset = gdal.Translate(vrt_path, cache_path, format="VRT",
                             bandList=[names.index(f"{b}_{stat}") + 1 for b in band_names])
    for i, b in enumerate(band_names, start=1):
        dataset.GetRasterBand(i).SetDescription(b)
    dataset = None
    return vrt_path


def scene_band_names(scenes: list[io.Scene]) -> dict[str, list[str]]:
    """{scene_id: band names}. A file named like a PlanetScope 4- or 8-band
    analytic clip (io.FOUR_BAND_SUFFIX / io.EIGHT_BAND_SUFFIX, also kept by
    registered copies) is taken from its name; anything else is opened
    (sensors.detect_band_names)."""
    out = {}
    for scene in scenes:
        if scene.analytic_path.endswith(io.EIGHT_BAND_SUFFIX):
            out[scene.scene_id] = list(sensors.BAND_PROFILES[8])
        elif scene.analytic_path.endswith(io.FOUR_BAND_SUFFIX):
            out[scene.scene_id] = list(sensors.BAND_PROFILES[4])
        else:
            out[scene.scene_id] = sensors.detect_band_names(scene.analytic_path)
    return out


def eight_band_onset(scene_band_names: dict[str, list[str]], scenes: list[io.Scene]) -> datetime | None:
    """The earliest acquisition among `scenes` that have any EIGHT_BAND_ONLY
    band (`scene_band_names`: {scene_id: band names}), or None if none do."""
    times = [s.acquired for s in scenes
             if s.acquired is not None and any(b in EIGHT_BAND_ONLY for b in scene_band_names[s.scene_id])]
    return min(times, default=None)


def composite_is_eligible(
    composite: ReferenceComposite,
    composite_bands: list[str],
    target_bands: list[str],
    onset: datetime | None,
) -> bool:
    """Whether `composite` (whose present_bands are `composite_bands`) can
    correct every band of a target with `target_bands`: it must hold all of
    them, and for a target with 8-band-only bands its acquisition window
    must start on or after the 8-band product's `onset` -- so it is built
    entirely from the period when 8-band data existed."""
    if not set(target_bands) <= set(composite_bands):
        return False
    if onset is not None and any(b in EIGHT_BAND_ONLY for b in target_bands):
        return composite.window_start >= onset
    return True


def assign_scenes_to_composites(
    scenes: list[io.Scene],
    scene_band_names: dict[str, list[str]],
    composites: list[ReferenceComposite],
    composite_bands: dict[str, list[str]],
    *,
    onset: datetime | None,
    fixed_year: int | None = None,
    log=lambda msg: None,
) -> dict[str, list[io.Scene]]:
    """{composite path: [scenes it is the reference for]}. Each scene gets
    the eligible composite (composite_is_eligible) whose solstice is
    nearest its acquisition; with `fixed_year`, that year's composite
    wherever it is eligible, else the nearest eligible one (logged).
    `composite_bands` is {composite path: present_bands}. Scenes with no
    eligible composite, or no acquisition time, are left out (logged)."""
    groups: dict[str, list[io.Scene]] = {c.path: [] for c in composites}
    fixed = next((c for c in composites if c.year == fixed_year), None) if fixed_year is not None else None
    if fixed_year is not None and fixed is None:
        raise ValueError(f"No composite for year {fixed_year} among {[c.year for c in composites]}")
    unassigned, moved_off_fixed = [], 0
    for scene in scenes:
        bands = scene_band_names[scene.scene_id]
        eligible = [c for c in composites if composite_is_eligible(c, composite_bands[c.path], bands, onset)]
        if not eligible or scene.acquired is None:
            unassigned.append(scene.scene_id)
            continue
        if fixed is not None and fixed in eligible:
            chosen = fixed
        else:
            chosen = min(eligible, key=lambda c: abs(scene.acquired - c.solstice))
            moved_off_fixed += fixed is not None
        groups[chosen.path].append(scene)
    if moved_off_fixed:
        log(f"  {moved_off_fixed} scene(s) cannot use the {fixed_year} composite and use their nearest eligible one")
    if unassigned:
        log(f"  WARNING: {len(unassigned)} scene(s) have no eligible composite (or no acquisition time) and are "
            f"left out: {sorted(unassigned)[:10]}{' ...' if len(unassigned) > 10 else ''}")
    return groups
