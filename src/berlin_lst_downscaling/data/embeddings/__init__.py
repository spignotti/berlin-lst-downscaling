"""TESSERA embedding arm — cloud-native annual embeddings for Berlin.

This package is the TESSERA ablation arm only (Feng et al. 2025,
dataset v1.1-dclimate, CC0). Inputs are Sentinel-1/Sentinel-2 only, so
there is no Landsat/thermal target leakage. Hand-built features stay the
main model; ``features/v6`` never depends on this package.
"""

from berlin_lst_downscaling.data.embeddings.cells import (
    aggregate_tessera_valid_to_100m,
    common_cell_set,
    intersect_eligibility,
    main_arm_cell_set,
    select_common_cell_ids,
    tessera_arm_cell_set,
)
from berlin_lst_downscaling.data.embeddings.guards import (
    assert_tessera_eval_year,
    assert_tessera_not_2026,
)
from berlin_lst_downscaling.data.embeddings.loader import (
    build_tessera_model_inputs,
    load_patch_embedding,
    load_scene_year_embedding,
    select_kept_features,
)
from berlin_lst_downscaling.data.embeddings.product import (
    TESSERA_PRODUCT_PREFIX,
    product_cog_uri,
    product_completion_uri,
    product_provenance_uri,
    product_uri_for_year,
)
from berlin_lst_downscaling.data.embeddings.reproject import (
    reproject_to_canonical,
)
from berlin_lst_downscaling.data.embeddings.tessera import (
    TESSERA_BBOX,
    TESSERA_DATASET_VERSION,
    TESSERA_N_CHANNELS,
    TESSERA_STORE_URI,
    TESSERA_SUPPORTED_YEARS,
    TESSERA_ZONE,
    TESSERA_ZONE_CRS,
    TesseraProvenance,
    dequantize_tessera,
    tessera_invalid_mask,
    validate_tessera_year,
)

__all__ = [
    "TESSERA_BBOX",
    "TESSERA_DATASET_VERSION",
    "TESSERA_N_CHANNELS",
    "TESSERA_PRODUCT_PREFIX",
    "TESSERA_STORE_URI",
    "TESSERA_SUPPORTED_YEARS",
    "TESSERA_ZONE",
    "TESSERA_ZONE_CRS",
    "TesseraProvenance",
    "aggregate_tessera_valid_to_100m",
    "assert_tessera_eval_year",
    "assert_tessera_not_2026",
    "build_tessera_model_inputs",
    "common_cell_set",
    "main_arm_cell_set",
    "load_patch_embedding",
    "load_scene_year_embedding",
    "dequantize_tessera",
    "intersect_eligibility",
    "product_completion_uri",
    "product_cog_uri",
    "product_provenance_uri",
    "product_uri_for_year",
    "reproject_to_canonical",
    "select_common_cell_ids",
    "tessera_arm_cell_set",
    "select_kept_features",
    "tessera_invalid_mask",
    "validate_tessera_year",
]
