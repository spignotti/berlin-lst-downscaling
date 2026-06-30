"""Scene listing, filtering, and export preparation for GEE sources.

All functions assume GEE has been initialized (see ``gee_client.initialize``).
"""

from typing import Any, cast

import ee
from omegaconf import DictConfig

from berlin_lst_downscaling.data.boundary import buffered_bbox_wgs84
from berlin_lst_downscaling.data.gee_masks import prepare_sentinel2_collection

# ── Listing ──────────────────────────────────────────────────────────────────


def list_landsat_scenes(cfg: DictConfig, year: int | None = None) -> ee.ImageCollection:
    """List all Landsat 8/9 scenes for the configured AOI and time window.

    Args:
        cfg: Pipeline config.
        year: If set, filter to a single year. If None, use the full range.

    Returns:
        An ``ee.ImageCollection`` with ALL scenes (no cloud filtering —
        pixel masking happens later).
    """
    months = cfg.ard.time.months

    if year is not None:
        start = f"{year}-{months[0]:02d}-01"
        end = f"{year}-{months[-1]:02d}-01"
        end = _advance_month(end)  # exclusive upper bound
    else:
        start = f"{cfg.ard.time.start_year}-{months[0]:02d}-01"
        end = f"{cfg.ard.time.end_year}-{months[-1]:02d}-01"
        end = _advance_month(end)

    cols = [ee.ImageCollection(c) for c in cfg.landsat.collections]
    combined = cols[0]
    for c in cols[1:]:
        combined = combined.merge(c)

    paths = list(cfg.landsat.scene_filter.wrs_paths)
    rows = list(cfg.landsat.scene_filter.wrs_rows)

    filtered = []
    for path in paths:
        for row in rows:
            filtered.append(
                combined.filter(
                    ee.Filter.And(
                        ee.Filter.eq("WRS_PATH", int(path)),
                        ee.Filter.eq("WRS_ROW", int(row)),
                    )
                )
            )

    if not filtered:
        return ee.ImageCollection([]).filterDate(start, end)

    merged = filtered[0]
    for collection in filtered[1:]:
        merged = merged.merge(collection)

    return merged.filterDate(start, end)


def list_sentinel2_scenes(cfg: DictConfig, year: int | None = None) -> ee.ImageCollection:
    """List all Sentinel-2 L2A scenes for the configured AOI and time window.

    This returns the intersecting tiles for the AOI time window. The
    per-datatake mosaic is built later in ``prepare_sentinel2_collection_wrapped``
    so the final export can cover the full AOI even when it spans multiple
    MGRS tiles.
    """
    bbox = buffered_bbox_wgs84(cfg.ard.aoi.boundary_file)
    months = cfg.ard.time.months

    if year is not None:
        start = f"{year}-{months[0]:02d}-01"
        end = f"{year}-{months[-1]:02d}-01"
        end = _advance_month(end)
    else:
        start = f"{cfg.ard.time.start_year}-{months[0]:02d}-01"
        end = f"{cfg.ard.time.end_year}-{months[-1]:02d}-01"
        end = _advance_month(end)

    collection = ee.ImageCollection(cfg.sentinel2.collection)

    return collection.filterBounds(ee.Geometry.Rectangle(bbox)).filterDate(start, end)


def _advance_month(ym_str: str) -> str:
    """Given ``YYYY-MM-DD``, return the first of the next month (exclusive end)."""
    year, month, _ = ym_str.split("-")
    y, m = int(year), int(month)
    if m == 12:
        return f"{y + 1}-01-01"
    return f"{y}-{m + 1:02d}-01"


# ── Collection-level preparation (mask + scale) ──────────────────────────────


def prepare_landsat_collection(
    collection: ee.ImageCollection, cfg: DictConfig
) -> ee.ImageCollection:
    """Apply cloud masking and radiometric scaling to a Landsat collection.

    Extracts all config values to plain Python types before the ``.map()``
    to avoid ee.List promotion issues with closure-captured lists.
    """

    # ── Extract all config values to plain types before .map() ───────
    bits = {
        "cloud": cfg.landsat.cloud.bits.cloud,
        "dilated_cloud": cfg.landsat.cloud.bits.dilated_cloud,
        "shadow": cfg.landsat.cloud.bits.shadow,
        "cirrus": cfg.landsat.cloud.bits.cirrus,
    }
    dilation = cfg.landsat.cloud.dilation_pixels
    lst_band = str(cfg.landsat.band_lst)
    rad = cfg.landsat.radiometry
    st_mult = float(rad.st_mult)
    st_add = float(rad.st_add)
    lst_low_k = float(rad.lst_plausible_kelvin[0])
    lst_high_k = float(rad.lst_plausible_kelvin[1])

    def _process(img: ee.Image) -> ee.Image:
        # ── Cloud mask (inlined from gee_masks.mask_landsat) ──
        qa = img.select("QA_PIXEL")
        radsat = img.select("QA_RADSAT")

        cloud = qa.bitwiseAnd(1 << bits["cloud"]).eq(0)
        dilated_cloud = qa.bitwiseAnd(1 << bits["dilated_cloud"]).eq(0)
        shadow = qa.bitwiseAnd(1 << bits["shadow"]).eq(0)
        cirrus = qa.bitwiseAnd(1 << bits["cirrus"]).eq(0)
        saturated = radsat.gt(0)
        clear = cloud.And(dilated_cloud).And(shadow).And(cirrus).And(saturated.Not())
        if dilation > 0:
            clear = clear.focal_min(dilation, kernelType="square")

        # ── LST radiometric scaling ──
        lst = img.select(lst_band).float()
        lst = lst.multiply(st_mult).add(st_add)

        # ── LST plausibility flag (1=plausible, 0=outside range) ──
        lst_plausible = lst.gte(lst_low_k).And(lst.lte(lst_high_k))

        # Chain addBands on the original image to preserve properties
        # (ee.Image.cat drops properties like system:time_start)
        result = (
            img.addBands(lst, overwrite=True)  # scaled LST (replaces original ST_B10)
            .addBands(clear.rename("cloud_mask"))
            .addBands(lst_plausible.rename("lst_plausible"))
            .select(lst_band, "cloud_mask", "lst_plausible")
        )
        return result

    return collection.map(_process)


def prepare_sentinel2_collection_wrapped(
    collection: ee.ImageCollection, cfg: DictConfig
) -> ee.ImageCollection:
    """Apply cloud masking and scaling to a Sentinel-2 collection.

    Uses the join-based approach from ``gee_masks.prepare_sentinel2_collection``
    to associate cloud probability data, then applies scaling and mosaics all
    tiles belonging to the same datatake (shared ``system:time_start``).

    Extracts all config values to plain types before ``.map()``.
    """
    masked = prepare_sentinel2_collection(collection, cfg)

    all_bands = list(cfg.sentinel2.bands_10m) + list(cfg.sentinel2.bands_20m)
    scl_band = str(cfg.sentinel2.band_scl)
    rad = cfg.sentinel2.radiometry
    scale = float(rad.reflectance_scale)
    clip_min = float(rad.reflectance_clip[0])
    clip_max = float(rad.reflectance_clip[1])

    def _scale(img: ee.Image) -> ee.Image:
        # Scale bands
        band_stack = img.select(*all_bands).float()
        scaled = band_stack.multiply(scale).max(clip_min).min(clip_max)

        # Chain addBands on the original image to preserve properties
        result = img.addBands(
            scaled.float(), overwrite=True
        ).select(*all_bands, scl_band)  # scaled bands (replaces originals)

        # Carry through cloud_mask if present (from join-based masking)
        has_mask = img.bandNames().filter(ee.Filter.eq("item", "cloud_mask")).size().gt(0)
        result = ee.Image(
            ee.Algorithms.If(has_mask, result.addBands(img.select("cloud_mask")), result)
        )
        return result

    scaled = masked.map(_scale)
    return _mosaic_sentinel2_datatakes(scaled)


# ── Per-scene export image construction ──────────────────────────────────────


def prepare_landsat_export_lst(image: ee.Image, cfg: DictConfig) -> ee.Image:
    """Select bands for the 100m LST export task.

    Returns an image with ST_B10 + cloud_mask + lst_plausible,
    all cast to float32 for export compatibility.
    """
    lst_band = str(cfg.landsat.band_lst)
    return image.select([lst_band, "cloud_mask", "lst_plausible"]).toFloat()


def prepare_sentinel2_export(image: ee.Image, cfg: DictConfig) -> ee.Image:
    """Select bands for the 10m Sentinel-2 export task.

    Returns an image with all S2 predictor bands + SCL + ``cloud_mask``,
    all cast to float32 for export compatibility.
    """
    all_bands = list(cfg.sentinel2.bands_10m) + list(cfg.sentinel2.bands_20m)
    scl = str(cfg.sentinel2.band_scl)
    return image.select([*all_bands, scl, "cloud_mask"]).toFloat()


def _mosaic_sentinel2_datatakes(collection: ee.ImageCollection) -> ee.ImageCollection:
    """Mosaic all Sentinel-2 tiles that share the same datatake timestamp."""
    times = ee.List(collection.aggregate_array("system:time_start")).distinct().sort()

    def _mosaic(time_ms: Any) -> ee.Image:
        # Sort once for both deterministic mosaic overlap priority (last
        # image wins on overlap) and stable scene_id extraction.
        group = collection.filter(ee.Filter.eq("system:time_start", time_ms)).sort("system:index")
        first = ee.Image(group.first())
        scene_id = ee.String(first.get("system:index")).split("_").get(0)
        mosaic = group.mosaic()
        return cast(
            Any,
            mosaic.set(
                {
                    "system:time_start": time_ms,
                    "system:index": scene_id,
                    "scene_id": scene_id,
                }
            ),
        )

    return cast(Any, ee.ImageCollection(times.map(_mosaic)))
