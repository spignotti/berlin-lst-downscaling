"""Cloud and shadow masking functions for Landsat and Sentinel-2.

All functions operate on ``ee.Image`` objects (server-side GEE operations).
Masks are stored as a ``cloud_mask`` flag band (1 = clear, 0 = cloud/shadow),
never applied destructively.
"""

import ee
from omegaconf import DictConfig


def mask_landsat(image: ee.Image, cfg: DictConfig) -> ee.Image:
    """Generate a cloud/shadow mask for Landsat Collection 2 using QA_PIXEL.

    Bits used:
        * 3: cloud
        * 1: dilated cloud
        * 4: cloud shadow
        * 2: cirrus
    Also flags saturated pixels via QA_RADSAT.

    Args:
        image: Input Landsat image with QA_PIXEL and QA_RADSAT bands.
        cfg: Pipeline config containing ``landsat.cloud`` sub-config.

    Returns:
        Image with an added ``cloud_mask`` band (1 = clear, 0 = bad).
        The mask is dilated by ``dilation_pixels`` pixels (square kernel).
    """
    qa = image.select("QA_PIXEL")
    radsat = image.select("QA_RADSAT")
    bits = cfg.landsat.cloud.bits
    dilation = cfg.landsat.cloud.dilation_pixels

    # Unpack individual mask components (bit = 0 means clear)
    cloud = qa.bitwiseAnd(1 << bits.cloud).eq(0)
    dilated_cloud = qa.bitwiseAnd(1 << bits.dilated_cloud).eq(0)
    shadow = qa.bitwiseAnd(1 << bits.shadow).eq(0)
    cirrus = qa.bitwiseAnd(1 << bits.cirrus).eq(0)

    # Saturated: any band with saturation flagged (bit value > 0)
    saturated = radsat.gt(0)

    # Composite clear mask
    clear = cloud.And(dilated_cloud).And(shadow).And(cirrus).And(saturated.Not())

    # Dilate bad pixels: focal_min shrinks clear areas → grows bad areas
    if dilation > 0:
        clear = clear.focal_min(dilation, kernelType="square")

    return image.addBands(clear.rename("cloud_mask"))


def prepare_sentinel2_collection(
    collection: ee.ImageCollection, cfg: DictConfig
) -> ee.ImageCollection:
    """Join S2_SR with S2_CLOUD_PROBABILITY and apply cloud masking.

    Extracts config values to plain types before ``.map()`` to avoid
    ee.List promotion issues.

    Returns a collection where each image has a ``cloud_mask`` band
    (1 = clear, 0 = cloud/shadow) at 10m resolution.
    """
    cloud_col = ee.ImageCollection(cfg.sentinel2.cloud.s2cloudless_collection)
    scl_band = str(cfg.sentinel2.band_scl)
    threshold = cfg.sentinel2.cloud.threshold
    scl_mask_classes = list(cfg.sentinel2.cloud.scl_mask)

    join_filter = ee.Filter.equals(leftField="system:index", rightField="system:index")
    inner_join = ee.Join.saveFirst("cloud_prob")

    joined = ee.ImageCollection(inner_join.apply(collection, cloud_col, join_filter))

    # Build plain (list, list) for remap — avoids ee.List inside .map()
    mask_from = scl_mask_classes
    mask_to = [0] * len(scl_mask_classes)

    def _apply_mask(feature: ee.Image) -> ee.Image:
        prob_img = ee.Image(feature.get("cloud_prob"))
        scl = feature.select(scl_band)
        prob = prob_img.select("probability")

        # s2cloudless probability threshold
        prob_mask = prob.lt(threshold)

        # SCL class mask — plain lists are safe here (extracted before .map())
        scl_mask = scl.remap(mask_from, mask_to, 1)

        # Combined mask (1 = clear)
        clear = prob_mask.And(scl_mask)
        return feature.addBands(clear.rename("cloud_mask"))

    return joined.map(_apply_mask)
