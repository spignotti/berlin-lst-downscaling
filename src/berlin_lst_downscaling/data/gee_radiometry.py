"""Radiometric scaling and harmonization for Landsat and Sentinel-2.

All Landsat Collection 2 bands are scaled from raw DN to physical units
(TOA reflectance for SR bands, Kelvin for ST_B10).

Sentinel-2 L2A in the ``_HARMONIZED`` collection is already corrected
for the baseline-04.00 offset, but a helper is provided in case the
raw collection is used.

All operations are server-side GEE operations — safe inside ``.map()``.
"""

import ee
from omegaconf import DictConfig


def _carry_band(image: ee.Image, result: ee.Image, band_name: str) -> ee.Image:
    """Add *band_name* from *image* to *result* if it exists.

    Server-side safe for use inside ``.map()``.
    """
    has_band = (
        image.bandNames()
        .filter(ee.Filter.eq("item", band_name))
        .size()
        .gt(0)
    )
    return ee.Image(ee.Algorithms.If(has_band, result.addBands(image.select(band_name)), result))


def apply_landsat_scaling(image: ee.Image, cfg: DictConfig) -> ee.Image:
    """Apply Collection 2 scaling factors to Landsat SR and LST bands.

    * SR bands: ``DN * scale + add`` → reflectance [0, 1]
    * ST_B10: ``DN * scale + add`` → Kelvin

    The ``cloud_mask`` band is carried through from the masking step.

    All operations are server-side — safe inside ``.map()``.

    Args:
        image: Raw Landsat C2 image (integer DNs) or masked image.
        cfg: Pipeline config containing ``landsat.radiometry`` sub-config.

    Returns:
        Image with scaled SR + LST bands and cloud_mask.
    """
    rad = cfg.landsat.radiometry
    sr_bands = cfg.landsat.bands_sr
    lst_band = cfg.landsat.band_lst

    # Scale SR bands
    sr_stack = image.select(sr_bands).float()
    sr_scaled = sr_stack.multiply(rad.sr_mult).add(rad.sr_add)
    clip_lo, clip_hi = rad.reflectance_clip
    sr_scaled = sr_scaled.max(clip_lo).min(clip_hi)

    # Scale LST
    lst = image.select(lst_band).float()
    lst_scaled = lst.multiply(rad.st_mult).add(rad.st_add)

    result = ee.Image.cat([sr_scaled, lst_scaled]).float()
    result = _carry_band(image, result, "cloud_mask")
    return result


def apply_sentinel2_scaling(image: ee.Image, cfg: DictConfig) -> ee.Image:
    """Apply reflectance scaling to Sentinel-2 L2A bands.

    The ``_HARMONIZED`` collection already accounts for the baseline-04.00 offset.
    Bands are multiplied by ``0.0001`` to convert to reflectance [0, 1].

    All operations are server-side — safe inside ``.map()``.

    Args:
        image: Sentinel-2 image with unscaled uint16 bands.
        cfg: Pipeline config containing ``sentinel2.radiometry`` sub-config.

    Returns:
        Image with scaled bands in float32, plus SCL and cloud_mask.
    """
    rad = cfg.sentinel2.radiometry
    all_bands = cfg.sentinel2.bands_10m + cfg.sentinel2.bands_20m
    scl_band = cfg.sentinel2.band_scl

    clip_lo, clip_hi = rad.reflectance_clip
    band_stack = image.select(all_bands).float()
    scaled = band_stack.multiply(rad.reflectance_scale).max(clip_lo).min(clip_hi)

    result = scaled.float()
    result = result.addBands(image.select(scl_band))
    result = _carry_band(image, result, "cloud_mask")
    return result
