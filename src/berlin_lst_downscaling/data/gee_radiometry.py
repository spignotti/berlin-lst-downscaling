"""Radiometric scaling for Landsat and Sentinel-2 GEE exports.

Scaling is inlined in ``gee_scenes`` (``prepare_landsat_collection``,
``prepare_sentinel2_collection_wrapped``) to extract config values
before ``.map()``. See those functions for the canonical implementations.

Reference:
  * Landsat C2: SR = DN * 2.75e-5 + (-0.2), ST_B10 = DN * 0.00341802 + 149.0
  * Sentinel-2 L2A _HARMONIZED: multiply by 0.0001 (reflectance 0-1)
"""
