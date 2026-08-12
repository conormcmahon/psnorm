"""Band-name detection so the pipeline is not hardcoded to one PlanetScope product.

PlanetScope imagery ships with different band counts/orders depending on the
sensor generation and ordering option: Dove-C and Dove-R deliver 4 bands
(blue, green, red, nir); SuperDove (PSB.SD) can be ordered either as a
4-band "harmonized" product (same 4 logical bands as Dove-C/R, for drop-in
compatibility) or as its native 8-band product. Every other module in this
package works purely in terms of a list of logical band names, so adding a
new product only means extending BAND_PROFILES / _DESCRIPTION_ALIASES below.
"""

from __future__ import annotations

import warnings

from osgeo import gdal

gdal.UseExceptions()

# Canonical band order for each band count psnorm knows how to handle.
BAND_PROFILES: dict[int, list[str]] = {
    4: ["blue", "green", "red", "nir"],
    8: [
        "coastal_blue",
        "blue",
        "green_i",
        "green",
        "yellow",
        "red",
        "rededge",
        "nir",
    ],
}

# Maps free-text GDAL band Description strings (as PlanetScope GeoTIFFs set
# them) to the canonical logical names above.
_DESCRIPTION_ALIASES: dict[str, str] = {
    "coastal blue": "coastal_blue",
    "coastal_blue": "coastal_blue",
    "blue": "blue",
    "green i": "green_i",
    "green_i": "green_i",
    "green": "green",
    "yellow": "yellow",
    "red": "red",
    "red edge": "rededge",
    "rededge": "rededge",
    "red_edge": "rededge",
    "nir": "nir",
}


def detect_band_names(path: str) -> list[str]:
    """Return the logical band name for each band in the raster at `path`.

    Prefers each band's GDAL Description (PlanetScope sets these, e.g.
    "blue"/"green"/"red"/"nir"). Falls back to BAND_PROFILES keyed by band
    count if descriptions are missing or unrecognized, with a warning since
    that's a guess about band order rather than a read fact.
    """
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"Could not open raster: {path}")
    band_count = dataset.RasterCount

    descriptions = [
        dataset.GetRasterBand(b).GetDescription().strip().lower()
        for b in range(1, band_count + 1)
    ]
    dataset = None

    resolved = [_DESCRIPTION_ALIASES.get(d) for d in descriptions]
    if all(resolved):
        return resolved

    if band_count not in BAND_PROFILES:
        raise ValueError(
            f"'{path}' has {band_count} bands with no recognized band "
            f"descriptions, and no default band profile is registered for "
            f"{band_count} bands. Add one to psnorm.sensors.BAND_PROFILES."
        )
    warnings.warn(
        f"'{path}': band descriptions missing/unrecognized "
        f"({descriptions!r}); assuming the default {band_count}-band "
        f"PlanetScope order {BAND_PROFILES[band_count]!r}.",
        RuntimeWarning,
    )
    return list(BAND_PROFILES[band_count])


def band_index(band_names: list[str], logical_name: str) -> int | None:
    """1-based index of `logical_name` in `band_names`, or None if absent."""
    try:
        return band_names.index(logical_name) + 1
    except ValueError:
        return None


def cloudmask_band_indices(band_names: list[str]) -> tuple[int, int, int] | None:
    """(red, green, nir) 1-based band indices for OmniCloudMask, or None if
    this band set doesn't have all three (e.g. a red/NIR-only product)."""
    red = band_index(band_names, "red")
    green = band_index(band_names, "green")
    nir = band_index(band_names, "nir")
    if red is None or green is None or nir is None:
        return None
    return red, green, nir


def ndwi_band_indices(band_names: list[str]) -> tuple[int, int] | None:
    """(green, nir) 1-based band indices for NDWI, or None if this band set
    doesn't have both."""
    green = band_index(band_names, "green")
    nir = band_index(band_names, "nir")
    if green is None or nir is None:
        return None
    return green, nir


def ndvi_band_indices(band_names: list[str]) -> tuple[int, int] | None:
    """(red, nir) 1-based band indices for NDVI, or None if this band set
    doesn't have both."""
    red = band_index(band_names, "red")
    nir = band_index(band_names, "nir")
    if red is None or nir is None:
        return None
    return red, nir
