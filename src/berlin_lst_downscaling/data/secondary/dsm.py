"""Digital Surface Model (DSM) products — derived from terrain, LoD2, and vegetation height.

Produces three DSM products per geometry vintage:

- ``building_dsm`` — terrain + LoD2 maximum building height per cell
- ``vegetation_dsm`` — terrain + vegetation maximum canopy height per cell
- ``combined_dsm`` — pixelwise maximum of building and vegetation DSM

Each DSM is keyed by its input-vintage combination.  Derived config
hashes include upstream source/config hashes so changed inputs
invalidate descendants.

Processing
----------
1. Read upstream COGs from the canonical product paths.
2. Compute component DSMs: terrain + obstruction height.
3. Compute combined DSM as component maximum.
4. Return :class:`PreparedSecondaryProduct` for each; the pipeline
   finaliser writes the four final artifacts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

import numpy as np
import rasterio
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr
from odc.geo.geobox import GeoBox

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.secondary.product import (
    PreparedSecondaryProduct,
    vintage_interval,
)

# ── contract ───────────────────────────────────────────────────────────


def contract_for_building_dsm() -> Contract:
    """Return the output Contract for building DSM COGs."""
    return Contract(
        source="building_dsm",
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name="building_dsm",
                dtype="float32",
                nodata=float("nan"),
                description="Terrain + max building height (m above sea level)",
                unit="m",
                valid_range=(-10.0, 600.0),
            ),
        ),
        tiling=TilingSpec(),
        schema_version=1,
        flag_mode="none",
    )


def contract_for_vegetation_dsm() -> Contract:
    """Return the output Contract for vegetation DSM COGs."""
    return Contract(
        source="vegetation_dsm",
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name="vegetation_dsm",
                dtype="float32",
                nodata=float("nan"),
                description="Terrain + max canopy height (m above sea level)",
                unit="m",
                valid_range=(-10.0, 600.0),
            ),
        ),
        tiling=TilingSpec(),
        schema_version=1,
        flag_mode="none",
    )


def contract_for_combined_dsm() -> Contract:
    """Return the output Contract for combined DSM COGs."""
    return Contract(
        source="combined_dsm",
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name="combined_dsm",
                dtype="float32",
                nodata=float("nan"),
                description="Max of building and vegetation DSM (m above sea level)",
                unit="m",
                valid_range=(-10.0, 600.0),
            ),
        ),
        tiling=TilingSpec(),
        schema_version=1,
        flag_mode="none",
    )


def config_hash_for_dsm(
    dsm_type: str,
    terrain_hash: str,
    lod2_hash: str,
    vh_hash: str,
) -> str:
    """Return a stable config hash for a derived DSM product."""
    raw = f"dsm:{dsm_type}:t={terrain_hash}:l={lod2_hash}:v={vh_hash}"
    return sha256(raw.encode()).hexdigest()[:12]


# ── upstream product readers ──────────────────────────────────────────


def _read_band_by_desc(uri: str, desc: str) -> np.ndarray:
    """Read a band from a COG by its description string.

    Raises ValueError if the band is not found.
    """
    with rasterio.open(uri) as src:
        descriptions = src.descriptions or ()
        for i, d in enumerate(descriptions, 1):
            if d == desc:
                return src.read(i).astype(np.float32)
    raise ValueError(f"Band '{desc}' not found in {uri} (bands: {list(descriptions)})")


# ── prepare functions ─────────────────────────────────────────────────


def prepare_building_dsm(
    terrain_uri: str,
    lod2_height_uri: str,
    output_root: str,
    run_id: str,
    item_key: str,
    upstream_hashes: dict[str, str],
    *,
    grid=None,
) -> PreparedSecondaryProduct:
    """Compute building DSM = terrain + max LoD2 building height.

    Parameters
    ----------
    terrain_uri :
        URI of the finalized terrain_height COG.
    lod2_height_uri :
        URI of the LoD2 morphology COG (reads ``building_height_max`` band).
    """
    grid = grid or canon_grid_10m()

    terrain_arr = _read_band_by_desc(terrain_uri, "terrain_height")
    height_arr = _read_band_by_desc(lod2_height_uri, "building_height_max")

    # Building DSM = terrain + building height (where buildings exist)
    building_dsm = np.where(
        ~np.isnan(height_arr) & (height_arr > 0),
        terrain_arr + height_arr,
        terrain_arr,  # no building → terrain only
    ).astype(np.float32)

    vintage_year = int(item_key.split("_")[0]) if item_key.isdigit() else 2021

    ds = _make_ds(grid, "building_dsm", building_dsm)
    valid = building_dsm[~np.isnan(building_dsm)]

    c_hash = config_hash_for_dsm(
        "building",
        upstream_hashes.get("terrain", ""),
        upstream_hashes.get("lod2", ""),
        upstream_hashes.get("vh", ""),
    )

    return PreparedSecondaryProduct(
        source="building_dsm",
        item_key=item_key,
        category="morphology",
        dataset=ds,
        contract=contract_for_building_dsm(),
        nominal_interval=vintage_interval(vintage_year),
        source_metadata={
            "terrain_uri": terrain_uri,
            "lod2_height_uri": lod2_height_uri,
            "upstream_hashes": upstream_hashes,
            "retrieved_at": datetime.now(UTC).isoformat(),
        },
        qa_stats={
            "valid_frac": (
                round(float(len(valid)) / building_dsm.size, 4) if building_dsm.size > 0 else 0.0
            ),
            "min": float(valid.min()) if len(valid) > 0 else None,
            "max": float(valid.max()) if len(valid) > 0 else None,
            "shape": list(building_dsm.shape),
        },
        config_hash=c_hash,
    )


def prepare_vegetation_dsm(
    terrain_uri: str,
    vh_max_uri: str,
    output_root: str,
    run_id: str,
    item_key: str,
    upstream_hashes: dict[str, str],
    *,
    grid=None,
) -> PreparedSecondaryProduct:
    """Compute vegetation DSM = terrain + max canopy height."""
    grid = grid or canon_grid_10m()

    terrain_arr = _read_band_by_desc(terrain_uri, "terrain_height")
    vh_arr = _read_band_by_desc(vh_max_uri, "vegetation_height_max")

    veg_dsm = np.where(
        ~np.isnan(vh_arr) & (vh_arr > 0),
        terrain_arr + vh_arr,
        terrain_arr,
    ).astype(np.float32)

    ds = _make_ds(grid, "vegetation_dsm", veg_dsm)
    valid = veg_dsm[~np.isnan(veg_dsm)]

    c_hash = config_hash_for_dsm(
        "vegetation",
        upstream_hashes.get("terrain", ""),
        upstream_hashes.get("lod2", ""),
        upstream_hashes.get("vh", ""),
    )

    vintage_year = int(item_key.split("_")[0]) if item_key.isdigit() else 2021

    return PreparedSecondaryProduct(
        source="vegetation_dsm",
        item_key=item_key,
        category="morphology",
        dataset=ds,
        contract=contract_for_vegetation_dsm(),
        nominal_interval=vintage_interval(vintage_year),
        source_metadata={
            "terrain_uri": terrain_uri,
            "vh_max_uri": vh_max_uri,
            "upstream_hashes": upstream_hashes,
            "retrieved_at": datetime.now(UTC).isoformat(),
        },
        qa_stats={
            "valid_frac": round(float(len(valid)) / veg_dsm.size, 4) if veg_dsm.size > 0 else 0.0,
            "min": float(valid.min()) if len(valid) > 0 else None,
            "max": float(valid.max()) if len(valid) > 0 else None,
            "shape": list(veg_dsm.shape),
        },
        config_hash=c_hash,
    )


def prepare_combined_dsm(
    building_dsm_uri: str,
    vegetation_dsm_uri: str,
    output_root: str,
    run_id: str,
    item_key: str,
    upstream_hashes: dict[str, str],
    *,
    grid=None,
) -> PreparedSecondaryProduct:
    """Compute combined DSM = max(building_dsm, vegetation_dsm)."""
    grid = grid or canon_grid_10m()

    bldg_arr = _read_band_by_desc(building_dsm_uri, "building_dsm")
    veg_arr = _read_band_by_desc(vegetation_dsm_uri, "vegetation_dsm")

    combined = np.fmax(bldg_arr, veg_arr).astype(np.float32)

    ds = _make_ds(grid, "combined_dsm", combined)
    valid = combined[~np.isnan(combined)]

    c_hash = config_hash_for_dsm(
        "combined",
        upstream_hashes.get("terrain", ""),
        upstream_hashes.get("lod2", ""),
        upstream_hashes.get("vh", ""),
    )

    vintage_year = int(item_key.split("_")[0]) if item_key.isdigit() else 2021

    return PreparedSecondaryProduct(
        source="combined_dsm",
        item_key=item_key,
        category="morphology",
        dataset=ds,
        contract=contract_for_combined_dsm(),
        nominal_interval=vintage_interval(vintage_year),
        source_metadata={
            "building_dsm_uri": building_dsm_uri,
            "vegetation_dsm_uri": vegetation_dsm_uri,
            "upstream_hashes": upstream_hashes,
            "retrieved_at": datetime.now(UTC).isoformat(),
        },
        qa_stats={
            "valid_frac": round(float(len(valid)) / combined.size, 4) if combined.size > 0 else 0.0,
            "min": float(valid.min()) if len(valid) > 0 else None,
            "max": float(valid.max()) if len(valid) > 0 else None,
            "shape": list(combined.shape),
        },
        config_hash=c_hash,
    )


# ── helpers ──────────────────────────────────────────────────────────


def _make_ds(grid: GeoBox, band_name: str, data: np.ndarray) -> xr.Dataset:
    """Build a canonical-grid xr.Dataset from a numpy array."""
    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    ds = xr.Dataset(
        {band_name: (("y", "x"), data)},
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)
    return ds


__all__ = [
    "config_hash_for_dsm",
    "contract_for_building_dsm",
    "contract_for_combined_dsm",
    "contract_for_vegetation_dsm",
    "prepare_building_dsm",
    "prepare_combined_dsm",
    "prepare_vegetation_dsm",
]
