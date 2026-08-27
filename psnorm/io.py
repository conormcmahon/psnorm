"""Scene discovery, raster metadata, grid-alignment/overlap arithmetic, and
chunked block reading.

Nothing here loads a full scene into memory: `iter_row_blocks`/`read_block_flat`
stream a bounded number of rows at a time (used by irmad.py and apply.py), and
`overlap_window` works entirely off each raster's geotransform so two scenes
with different origins/extents (e.g. adjacent strip segments of the same AOI
clip, as in the test dataset) never need to be resampled just to find where
they overlap.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Iterator

from osgeo import gdal

gdal.UseExceptions()

DEFAULT_BLOCK_ROWS = 512


@dataclass(frozen=True)
class RasterInfo:
    path: str
    width: int
    height: int
    band_count: int
    crs: str
    geotransform: tuple
    nodata: float | None


def get_raster_info(path: str) -> RasterInfo:
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"Could not open raster: {path}")
    info = RasterInfo(
        path=path,
        width=dataset.RasterXSize,
        height=dataset.RasterYSize,
        band_count=dataset.RasterCount,
        crs=dataset.GetProjectionRef() or "",
        geotransform=dataset.GetGeoTransform(),
        nodata=dataset.GetRasterBand(1).GetNoDataValue(),
    )
    dataset = None
    return info


def windowed_raster_info(info: RasterInfo, window: Window) -> RasterInfo:
    """`info` describing just `window` within `info`'s own raster: same
    CRS/pixel size/nodata, but the geotransform origin shifted to the
    window's own upper-left corner and width/height set to the window's
    size. Pass this (not the unwindowed `info`) to write_single_band_raster
    whenever the array being written only covers a sub-region of the
    raster `info` was read from -- otherwise the output is sized correctly
    but geolocated at the *parent* raster's origin instead of the window's,
    silently shifting it on disk.
    """
    xoff, yoff, xsize, ysize = window
    gt = info.geotransform
    windowed_gt = (gt[0] + xoff * gt[1], gt[1], gt[2], gt[3] + yoff * gt[5], gt[4], gt[5])
    return replace(info, width=xsize, height=ysize, geotransform=windowed_gt)


@dataclass(frozen=True)
class Scene:
    scene_id: str
    analytic_path: str
    udm2_path: str | None
    metadata_path: str | None
    acquired: datetime | None

    def metadata(self) -> dict:
        if self.metadata_path is None:
            return {}
        with open(self.metadata_path) as f:
            return json.load(f)


_TIMESTAMP_RE = re.compile(r"^(\d{8})_(\d{6})")


def parse_acquisition_time(scene_id: str) -> datetime | None:
    """Parse the `YYYYMMDD_HHMMSS...` timestamp PlanetScope encodes at the
    start of every scene id/filename."""
    match = _TIMESTAMP_RE.match(scene_id)
    if not match:
        return None
    date_str, time_str = match.groups()
    try:
        return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def discover_scenes(
    folder: str,
    *,
    analytic_suffix: str = "_3B_AnalyticMS_SR_harmonized_clip.tif",
    udm2_suffix: str = "_3B_udm2_clip.tif",
    metadata_suffix: str = "_metadata.json",
) -> list[Scene]:
    """Find analytic scenes in `folder` and pair each with its UDM2 mask and
    metadata.json sidecar (present if found, `None` otherwise).

    Skips macOS AppleDouble resource-fork files (`._*`), which are not valid
    rasters but otherwise match the same glob pattern.
    """
    pattern = os.path.join(folder, f"*{analytic_suffix}")
    analytic_paths = sorted(
        p
        for p in glob.glob(pattern)
        if not os.path.basename(p).startswith("._")
    )

    scenes = []
    for analytic_path in analytic_paths:
        folder_dir = os.path.dirname(analytic_path)
        base = os.path.basename(analytic_path)
        scene_id = base[: -len(analytic_suffix)]

        udm2_path = os.path.join(folder_dir, scene_id + udm2_suffix)
        metadata_path = os.path.join(folder_dir, scene_id + metadata_suffix)

        scenes.append(
            Scene(
                scene_id=scene_id,
                analytic_path=analytic_path,
                udm2_path=udm2_path if os.path.exists(udm2_path) else None,
                metadata_path=metadata_path if os.path.exists(metadata_path) else None,
                acquired=parse_acquisition_time(scene_id),
            )
        )
    return scenes


def grids_aligned(a: RasterInfo, b: RasterInfo, tol: float = 1e-6) -> bool:
    """True if `a` and `b` share a CRS, pixel size, and a pixel-integer
    origin offset (i.e. can be overlapped by geotransform arithmetic alone,
    with no resampling needed)."""
    if a.crs != b.crs:
        return False
    gt_a, gt_b = a.geotransform, b.geotransform
    # Only north-up, non-rotated grids are supported.
    if any(abs(v) > tol for v in (gt_a[2], gt_a[4], gt_b[2], gt_b[4])):
        return False
    if abs(gt_a[1] - gt_b[1]) > tol or abs(gt_a[5] - gt_b[5]) > tol:
        return False
    dx = (gt_a[0] - gt_b[0]) / gt_a[1]
    dy = (gt_a[3] - gt_b[3]) / gt_a[5]
    return abs(dx - round(dx)) < tol and abs(dy - round(dy)) < tol


Window = tuple[int, int, int, int]  # (xoff, yoff, xsize, ysize)


def _bounds(info: RasterInfo) -> tuple[float, float, float, float]:
    x0 = info.geotransform[0]
    y1 = info.geotransform[3]
    x1 = x0 + info.width * info.geotransform[1]
    y0 = y1 + info.height * info.geotransform[5]  # geotransform[5] < 0
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def overlap_window(a: RasterInfo, b: RasterInfo) -> tuple[Window, Window] | None:
    """Matching pixel windows into `a` and `b` covering their geographic
    intersection, or None if they don't overlap. Assumes grids_aligned(a, b)
    — callers must check that first (or route through registration)."""
    ax0, ay0, ax1, ay1 = _bounds(a)
    bx0, by0, bx1, by1 = _bounds(b)
    minx, maxx = max(ax0, bx0), min(ax1, bx1)
    miny, maxy = max(ay0, by0), min(ay1, by1)
    if minx >= maxx or miny >= maxy:
        return None

    def to_window(info: RasterInfo) -> Window:
        px = info.geotransform[1]
        py = -info.geotransform[5]
        xoff = round((minx - info.geotransform[0]) / px)
        yoff = round((info.geotransform[3] - maxy) / py)
        xsize = round((maxx - minx) / px)
        ysize = round((maxy - miny) / py)
        return xoff, yoff, xsize, ysize

    wa, wb = to_window(a), to_window(b)
    if wa[2] <= 0 or wa[3] <= 0:
        return None
    return wa, wb


def iter_row_blocks(height: int, block_rows: int = DEFAULT_BLOCK_ROWS) -> Iterator[tuple[int, int]]:
    """Yield (row_offset, n_rows) chunks covering [0, height)."""
    for y in range(0, height, block_rows):
        yield y, min(block_rows, height - y)


def open_bands(path: str, band_indices: list[int] | None = None):
    """Open `path` and return (dataset, [band objects]). Caller must keep the
    dataset alive as long as the bands are read from (GDAL band objects are
    only valid while their dataset is open)."""
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"Could not open raster: {path}")
    if band_indices is None:
        band_indices = list(range(1, dataset.RasterCount + 1))
    bands = [dataset.GetRasterBand(i) for i in band_indices]
    return dataset, bands


def nearest_resize(arr, shape: tuple[int, int]):
    """Nearest-neighbor resize of a 2D array to `shape` (rows, cols), with
    no interpolation library dependency."""
    import numpy as np

    src_h, src_w = arr.shape
    tgt_h, tgt_w = shape
    row_idx = np.minimum((np.arange(tgt_h) * src_h) // tgt_h, src_h - 1)
    col_idx = np.minimum((np.arange(tgt_w) * src_w) // tgt_w, src_w - 1)
    return arr[row_idx][:, col_idx]


def read_block_flat(
    bands, xoff: int, yoff: int, xsize: int, n_rows: int, downsample_factor: int = 1,
    x_phase: int | None = None, y_phase: int | None = None,
):
    """Read an (n_rows*xsize, len(bands)) float64 tile, one column per band.

    `downsample_factor` > 1 reads each band downsampled (GDAL-averaged over
    `downsample_factor`x`downsample_factor` native pixels, reducing per-pixel
    sensor/registration noise) and then nearest-neighbor "blows it back up"
    to the original (n_rows, xsize) block shape, so every pixel within one
    coarse cell carries the same smoothed value. This keeps every caller's
    window/shape/masking bookkeeping completely unaffected by downsampling —
    only the *values* seen by IR-MAD/regression are coarsened, not the grid
    they're indexed on.

    The coarse-cell grid is anchored to absolute pixel positions
    `{x_phase, x_phase + downsample_factor, ...}` / `{y_phase, y_phase +
    downsample_factor, ...}` in *this raster's own* pixel coordinates —
    NOT to wherever `xoff`/`yoff` happens to start. `x_phase`/`y_phase`
    default to `xoff % downsample_factor` / `yoff % downsample_factor`
    (self-consistent for a single window read in isolation), but a caller
    reading *multiple* windows that need their coarse grids to land at
    consistent absolute positions relative to each other — e.g. every
    target scene's window onto the same reference scene, whose windows all
    start at different `xoff`/`yoff` (see io.overlap_window) — must compute
    `x_phase`/`y_phase` once (typically from the reference window's own
    offset) and pass the *same* values into every read, reference and
    target alike, so a fixed pixel offset between the two is preserved.
    Without this, each window's coarse grid would be phased independently,
    and aggregating many targets' contributions onto the reference grid
    (Phase B's consensus/frequency, or a saved candidate/inlier mask) would
    show finer-than-`downsample_factor` variation — a patchwork of
    differently-phased coarse grids rather than one consistent grid.
    """
    import numpy as np
    from osgeo import gdal

    tile = np.empty((n_rows * xsize, len(bands)), dtype=np.float64)
    if downsample_factor <= 1:
        for k, band in enumerate(bands):
            arr = band.ReadAsArray(xoff, yoff, xsize, n_rows)
            tile[:, k] = arr.ravel()
        return tile

    if x_phase is None:
        x_phase = xoff % downsample_factor
    if y_phase is None:
        y_phase = yoff % downsample_factor

    # Snap the read down/left to the phase-aligned coarse-grid boundary at
    # or before (xoff, yoff), and round the size up to the next whole
    # number of coarse cells past (xoff+xsize, yoff+n_rows) -- this is
    # always >= the originally requested window, cropped back down below.
    read_xoff = xoff - x_phase
    read_yoff = yoff - y_phase
    read_xsize = -(-(x_phase + xsize) // downsample_factor) * downsample_factor
    read_ysize = -(-(y_phase + n_rows) // downsample_factor) * downsample_factor

    raster_xsize = bands[0].XSize
    raster_ysize = bands[0].YSize
    # A shared phase borrowed from another window (see docstring) can push
    # read_xoff/read_yoff below 0 -- e.g. this window's own xoff is smaller
    # than the phase value inherited from the reference window. Clamp both
    # edges independently and pad each side actually missing (off either
    # end of the raster) with zeros, the same treatment already used for
    # the trailing edge.
    clip_left = max(0, -read_xoff)
    clip_top = max(0, -read_yoff)
    read_xoff_c = max(read_xoff, 0)
    read_yoff_c = max(read_yoff, 0)
    avail_xsize = max(0, min(read_xsize - clip_left, raster_xsize - read_xoff_c))
    avail_ysize = max(0, min(read_ysize - clip_top, raster_ysize - read_yoff_c))
    ds_xsize = max(1, avail_xsize // downsample_factor) if avail_xsize > 0 else 1
    ds_ysize = max(1, avail_ysize // downsample_factor) if avail_ysize > 0 else 1

    for k, band in enumerate(bands):
        if avail_xsize > 0 and avail_ysize > 0:
            small = band.ReadAsArray(
                read_xoff_c, read_yoff_c, avail_xsize, avail_ysize,
                buf_xsize=ds_xsize, buf_ysize=ds_ysize,
                resample_alg=gdal.GRIORA_Average,
            ).astype(np.float64)
            big = nearest_resize(small, (avail_ysize, avail_xsize))
        else:
            big = np.zeros((0, 0), dtype=np.float64)
        # Pad back out to the full (read_ysize, read_xsize) phase-aligned
        # canvas: clip_left/clip_top zeros for pixels off the raster's
        # leading edge, and whatever's left short of read_xsize/read_ysize
        # for pixels off its trailing edge.
        pad_top, pad_left = clip_top, clip_left
        pad_bottom = max(0, read_ysize - clip_top - big.shape[0])
        pad_right = max(0, read_xsize - clip_left - big.shape[1])
        if pad_top or pad_bottom or pad_left or pad_right:
            big = np.pad(big, ((pad_top, pad_bottom), (pad_left, pad_right)))
        cropped = big[y_phase : y_phase + n_rows, x_phase : x_phase + xsize]
        tile[:, k] = cropped.ravel()
    return tile


def read_window_bands(path: str, window: Window, band_indices: list[int] | None = None):
    """Read `window` from every requested band as a (bands, ysize, xsize)
    float64 array. Intended for overlap windows, which are bounded by
    geographic intersection and typically much smaller than a full scene."""
    dataset, bands = open_bands(path, band_indices)
    xoff, yoff, xsize, ysize = window
    arr = [b.ReadAsArray(xoff, yoff, xsize, ysize) for b in bands]
    dataset = None
    import numpy as np

    return np.stack(arr, axis=0).astype(np.float64)


def write_single_band_raster(array, info: RasterInfo, output_path: str, gdal_dtype: int) -> str:
    """Write a 2D array as a single-band GeoTIFF sharing `info`'s
    georeferencing (CRS + geotransform). Used for exclusion-flag, invariant
    -candidate, and consensus rasters — every mask/count artifact this
    package persists shares this one writer."""
    from osgeo import gdal

    height, width = array.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, width, height, 1, gdal_dtype, options=["COMPRESS=LZW"])
    dataset.SetGeoTransform(info.geotransform)
    dataset.SetProjection(info.crs)
    band = dataset.GetRasterBand(1)
    band.WriteArray(array)
    band.FlushCache()
    dataset = None
    return output_path
