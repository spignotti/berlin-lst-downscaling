"""Source-registered synthetic fixtures for the secondary pipeline.

Each registered source has a small fixture factory that produces a
:class:`PreparedSecondaryProduct` with source-specific band names and
value ranges — but no upstream downloads.  The local ``smoke-secondary-all``
nox session iterates the registry and finalises every product, exercising
the full pipeline (COG + STAC + provenance + completion marker + ledger +
QA report) for each source.

Real provider inputs are validated in the cloud run, not here.  The
fixture smoke deliberately stays free of upstream network I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
import rioxarray  # noqa: F401 — registers rio accessor
import xarray as xr

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.ard.contract import BandSpec, Contract, TilingSpec
from berlin_lst_downscaling.data.secondary.product import (
    PreparedSecondaryProduct,
    vintage_interval,
)

FixtureFactory = Callable[[str, str], PreparedSecondaryProduct]


def registry() -> dict[str, FixtureFactory]:
    """Return ``{source_name: factory}`` for every registered fixture source.

    The pipeline iterates this registry when running in ``fixture`` mode.
    Adding a new source to the secondary pipeline should also add a small
    fixture here so that ``smoke-secondary-all`` covers it.
    """
    return {
        "imperviousness": _fixture_imperviousness,
        "vegetation_height": _fixture_vegetation_height,
        "terrain_height": _fixture_terrain_height,
        "lod2_morphology": _fixture_lod2_morphology,
    }


# ── factories ──────────────────────────────────────────────────────


def _fixture_imperviousness(
    output_root: str, run_id: str,
) -> PreparedSecondaryProduct:
    """Fixture product for the imperviousness source.

    Random uniform values in [0, 100] on the canonical 10 m grid.
    No upstream download.
    """
    return _make_fixture(
        source="imperviousness",
        item_key="2021",
        category="morphology",
        band_name="imperviousness",
        band_description="Fixture sealing degree (percent, synthetic)",
        vmin=0.0,
        vmax=100.0,
        config_hash="fixture:imperviousness:v1",
        nominal_interval=vintage_interval(2021),
        seed=1,
    )


def _fixture_vegetation_height(
    output_root: str, run_id: str,
) -> PreparedSecondaryProduct:
    """Fixture product for the vegetation-height source (2 bands).

    Random uniform values in [0, 150] on the canonical 10 m grid.
    No upstream download.
    """
    from berlin_lst_downscaling.data.secondary.vegetation_height import (
        contract_for_vegetation_height,
    )

    grid = canon_grid_10m()
    rng = np.random.default_rng(2)

    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0

    mean_data = rng.uniform(0.0, 150.0, size=(grid.shape.y, grid.shape.x)).astype(
        np.float32,
    )
    max_data = mean_data + rng.uniform(0.0, 20.0, size=mean_data.shape).astype(
        np.float32,
    )

    ds = xr.Dataset(
        {
            "vegetation_height_mean": (("y", "x"), mean_data),
            "vegetation_height_max": (("y", "x"), max_data),
        },
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    contract = contract_for_vegetation_height()

    valid = mean_data[~np.isnan(mean_data)]
    return PreparedSecondaryProduct(
        source="vegetation_height",
        item_key="2020",
        category="morphology",
        dataset=ds,
        contract=contract,
        nominal_interval=vintage_interval(2020),
        source_metadata={
            "kind": "synthetic_fixture",
            "seed": 2,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "note": "Synthetic fixture — no upstream download.",
        },
        qa_stats={
            "valid_frac": float(len(valid)) / mean_data.size if mean_data.size > 0 else 0.0,
            "min": float(valid.min()) if len(valid) > 0 else None,
            "max": float(valid.max()) if len(valid) > 0 else None,
            "shape": list(mean_data.shape),
        },
        config_hash="fixture:vegetation_height:v2",
    )


def _fixture_terrain_height(
    output_root: str, run_id: str,
) -> PreparedSecondaryProduct:
    """Fixture product for the terrain_height source.

    Random uniform values in [20, 80] on the canonical 10 m grid.
    No upstream download.
    """
    return _make_fixture(
        source="terrain_height",
        item_key="2021",
        category="morphology",
        band_name="terrain_height",
        band_description="Fixture terrain elevation (m, synthetic)",
        vmin=20.0,
        vmax=80.0,
        config_hash="fixture:terrain_height:v1",
        nominal_interval=vintage_interval(2021),
        seed=3,
    )


def _fixture_lod2_morphology(
    output_root: str, run_id: str,
) -> PreparedSecondaryProduct:
    """Fixture product for the lod2_morphology source (4 bands).

    Four-band synthetic: height mean [0, 50], std [0, 10], BCR [0, 1], max [0, 60].
    No upstream download.
    """
    from berlin_lst_downscaling.data.secondary.lod2 import contract_for_lod2_morphology

    grid = canon_grid_10m()
    rng = np.random.default_rng(4)

    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0

    mean_data = rng.uniform(0.0, 50.0, size=(grid.shape.y, grid.shape.x)).astype(
        np.float32,
    )
    std_data = rng.uniform(0.0, 10.0, size=(grid.shape.y, grid.shape.x)).astype(
        np.float32,
    )
    bcr_data = rng.uniform(0.0, 1.0, size=(grid.shape.y, grid.shape.x)).astype(
        np.float32,
    )
    max_data = mean_data + rng.uniform(0.0, 10.0, size=mean_data.shape).astype(
        np.float32,
    )

    ds = xr.Dataset(
        {
            "building_height_mean": (("y", "x"), mean_data),
            "building_height_std": (("y", "x"), std_data),
            "building_coverage_ratio": (("y", "x"), bcr_data),
            "building_height_max": (("y", "x"), max_data),
        },
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    contract = contract_for_lod2_morphology()

    return PreparedSecondaryProduct(
        source="lod2_morphology",
        item_key="2024",
        category="morphology",
        dataset=ds,
        contract=contract,
        nominal_interval=vintage_interval(2024),
        source_metadata={
            "kind": "synthetic_fixture",
            "seed": 4,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "note": "Synthetic fixture — no upstream download.",
        },
        qa_stats={
            "valid_frac": float(np.sum(~np.isnan(mean_data))) / mean_data.size,
            "min_height": float(np.nanmin(mean_data)),
            "max_height": float(np.nanmax(mean_data)),
            "shape": list(mean_data.shape),
        },
        config_hash="fixture:lod2_morphology:v2",
    )


# ── helpers ────────────────────────────────────────────────────────


def _make_fixture(
    *,
    source: str,
    item_key: str,
    category: str,
    band_name: str,
    band_description: str,
    vmin: float,
    vmax: float,
    config_hash: str,
    nominal_interval: tuple[str, str],
    seed: int,
) -> PreparedSecondaryProduct:
    """Build a synthetic product on the canonical 10 m grid."""
    grid = canon_grid_10m()
    rng = np.random.default_rng(seed)
    data = rng.uniform(vmin, vmax, size=(grid.shape.y, grid.shape.x)).astype(
        np.float32,
    )

    xs = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    ys = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    ds = xr.Dataset(
        {band_name: (("y", "x"), data)},
        coords={"x": xs, "y": ys},
    )
    ds = ds.rio.write_crs(str(grid.crs))
    ds = ds.rio.write_transform(grid.transform)

    contract = Contract(
        source=source,
        target_crs="EPSG:25833",
        output_bands=(
            BandSpec(
                name=band_name,
                dtype="float32",
                nodata=float("nan"),
                description=band_description,
            ),
        ),
        tiling=TilingSpec(),
        schema_version=1,
        flag_mode="none",
    )

    valid = data[~np.isnan(data)]
    return PreparedSecondaryProduct(
        source=source,
        item_key=item_key,
        category=category,
        dataset=ds,
        contract=contract,
        nominal_interval=nominal_interval,
        source_metadata={
            "kind": "synthetic_fixture",
            "seed": seed,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "note": "Synthetic fixture — no upstream download.",
        },
        qa_stats={
            "valid_frac": float(len(valid)) / data.size if data.size > 0 else 0.0,
            "min": float(valid.min()) if len(valid) > 0 else None,
            "max": float(valid.max()) if len(valid) > 0 else None,
            "shape": list(data.shape),
        },
        config_hash=config_hash,
    )


__all__ = [
    "FixtureFactory",
    "registry",
]
