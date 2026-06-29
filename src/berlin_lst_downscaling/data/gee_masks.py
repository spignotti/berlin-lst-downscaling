"""Cloud and shadow masking for Landsat and Sentinel-2 GEE exports.

* Landsat: QA_PIXEL-based masking is inlined in
  ``gee_scenes.prepare_landsat_collection`` (to extract config values
  before ``.map()``). See that function for the canonical implementation.
* Sentinel-2: ``prepare_sentinel2_collection`` joins S2_SR with
  S2_CLOUD_PROBABILITY and applies s2cloudless + SCL class masking.

All masks are stored as a ``cloud_mask`` flag band (1 = clear, 0 = cloud/shadow),
never applied destructively.
"""

import ee
from omegaconf import DictConfig


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
    shadow_projection = bool(cfg.sentinel2.cloud.shadow_projection)
    shadow_dilation_px = int(cfg.sentinel2.cloud.shadow_dilation_px)
    shadow_nir_threshold = int(cfg.sentinel2.cloud.shadow_nir_threshold)

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
        feature = feature.addBands(clear.rename("cloud_mask"))

        # Project cloud shadows (optional, Phase 2)
        if shadow_projection:
            feature = _add_shadow_projection(feature, shadow_dilation_px, shadow_nir_threshold)

        return feature

    return joined.map(_apply_mask)


def _add_shadow_projection(
    image: ee.Image, dilation_px: int, nir_threshold: int
) -> ee.Image:
    """Project cloud shadows via dilation + dark NIR threshold.

    Dilates the cloud mask to cover the potential shadow zone (~2.5 km
    at 10 m with default 250 px radius), then flags dark NIR pixels inside
    that zone as shadow.  Non-directional dilation is used — the NIR
    threshold eliminates false positives (dark vegetation, water).

    Args:
        image: S2 image with ``cloud_mask`` (1=clear, 0=cloud) at 10 m.
        dilation_px: FocalMax kernel radius in pixels.
        nir_threshold: NIR reflectance raw-DN threshold (e.g. 500 ≈ 0.05).

    Returns:
        Image with ``cloud_mask`` band updated to include projected shadows.
    """
    cloud_mask = image.select("cloud_mask")
    nir = image.select("B8")

    # Dilate cloud area in all directions to cover potential shadow zone
    cloud = cloud_mask.eq(0)
    shadow_zone = cloud.focalMax(dilation_px, "square", "pixels")

    # Shadow = (in dilated zone) AND (dark NIR) AND (not original cloud)
    nir_dark = nir.lt(nir_threshold)
    new_shadow = shadow_zone.And(nir_dark).And(cloud.Not())

    # Merge into mask: 0 = cloud/shadow
    updated_mask = cloud_mask.where(new_shadow, 0)
    return image.addBands(updated_mask.rename("cloud_mask"), overwrite=True)
