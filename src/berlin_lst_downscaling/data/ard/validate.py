"""COG structural validation — readability, CRS, bands, grid alignment.

Provides two entry points:

* ``validate_cog`` — main data COG
* ``validate_flag_cog`` — separate flag band COG

Both return a :class:`ValidationResult` with ``ok=True/False`` and a list of
error messages.  They are designed to be called right after writing the COG
(see ``pipeline.py``) or as a standalone pass over a completed ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import rasterio
from odc.geo.geobox import GeoBox

from berlin_lst_downscaling.data.ard.contract import Contract

# Minimum valid-pixel fraction to pass the "not all-NaN" check
_MIN_VALID_FRAC = 0.01


@dataclass
class ValidationResult:
    """Result of a single COG validation."""

    scene_id: str = ""
    source: str = ""
    ok: bool = True
    errors: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)


def validate_cog(
    uri: str,
    contract: Contract,
    expected_grid: GeoBox,
) -> ValidationResult:
    """Validate a main-data COG at *uri* against *contract* and *expected_grid*.

    Checks
    ------
    1. File is openable by ``rasterio.open``
    2. CRS is ``EPSG:25833``
    3. Band count matches ``len(contract.output_bands)``
    4. At least ``_MIN_VALID_FRAC`` pixels are not NaN
    5. Width / height match *expected_grid.shape*
    6. Transform origin (upper-left corner) matches *expected_grid.transform*
    """
    result = ValidationResult()

    # ── 1. Openable + 2–6: read metadata ─────────────────────────────────
    try:
        with rasterio.open(uri) as src:
            crs = str(src.crs).upper() if src.crs else "None"
            count = src.count
            width = src.width
            height = src.height
            transform = src.transform
            # Sample check for all-NaN — read a few blocks
            _check_nan(src, result)
    except Exception as exc:
        result.fail(f"Could not open or read COG: {exc}")
        return result

    # ── 2. CRS ─────────────────────────────────────────────────────────────
    expected_crs = "EPSG:25833"
    if crs != expected_crs:
        result.fail(f"CRS mismatch: got {crs!r}, expected {expected_crs!r}")

    # ── 3. Band count ──────────────────────────────────────────────────────
    expected_bands = len(contract.output_bands)
    if count != expected_bands:
        result.fail(
            f"Band count mismatch: got {count}, "
            f"expected {expected_bands} ({[b.name for b in contract.output_bands]})"
        )

    # ── 5. Shape ───────────────────────────────────────────────────────────
    ex, ey = expected_grid.shape.x, expected_grid.shape.y
    if width != ex or height != ey:
        result.fail(
            f"Shape mismatch: got ({width}, {height}), "
            f"expected ({ex}, {ey})"
        )

    # ── 6. Origin alignment ────────────────────────────────────────────────
    ex_off = expected_grid.transform.xoff
    ey_off = expected_grid.transform.yoff
    gx_off = transform.xoff
    gy_off = transform.yoff
    if abs(gx_off - ex_off) > 0.01 or abs(gy_off - ey_off) > 0.01:
        result.fail(
            f"Origin mismatch: got ({gx_off:.1f}, {gy_off:.1f}), "
            f"expected ({ex_off:.1f}, {ey_off:.1f})"
        )

    return result


def validate_flag_cog(
    uri: str,
    expected_grid: GeoBox,
) -> ValidationResult:
    """Validate a separate flag-band COG at *uri*.

    Checks
    ------
    1. File is openable by ``rasterio.open``
    2. CRS is ``EPSG:25833``
    3. Single band, uint8 dtype
    4. Width / height match *expected_grid.shape*
    5. Transform origin matches *expected_grid.transform*
    """
    result = ValidationResult()

    try:
        with rasterio.open(uri) as src:
            crs = str(src.crs).upper() if src.crs else "None"
            count = src.count
            dtype = src.dtypes[0] if src.dtypes else "None"
            width = src.width
            height = src.height
            transform = src.transform
    except Exception as exc:
        result.fail(f"Could not open or read flag COG metadata: {exc}")
        return result

    # CRS
    if crs != "EPSG:25833":
        result.fail(f"Flag COG CRS mismatch: got {crs!r}, expected 'EPSG:25833'")

    # Single uint8
    if count != 1:
        result.fail(f"Flag COG band count: got {count}, expected 1")
    if dtype != "uint8":
        result.fail(f"Flag COG dtype: got {dtype!r}, expected 'uint8'")

    # ── 4. Shape ───────────────────────────────────────────────────────────
    ex, ey = expected_grid.shape.x, expected_grid.shape.y
    if width != ex or height != ey:
        result.fail(
            f"Flag COG shape mismatch: got ({width}, {height}), "
            f"expected ({ex}, {ey})"
        )

    # Origin
    ex_off = expected_grid.transform.xoff
    ey_off = expected_grid.transform.yoff
    if abs(transform.xoff - ex_off) > 0.01 or abs(transform.yoff - ey_off) > 0.01:
        result.fail(
            f"Flag COG origin mismatch: got ({transform.xoff:.1f}, "
            f"{transform.yoff:.1f}), expected ({ex_off:.1f}, {ey_off:.1f})"
        )

    return result


# ── internal helpers ──────────────────────────────────────────────────


def _check_nan(src: rasterio.DatasetReader, result: ValidationResult) -> None:
    """Check that the first band is not completely NaN.

    Reads the full first band into memory (COGs are clipped to Berlin,
    typically < 2k × 2k pixels at 100m).
    """
    try:
        band = src.read(1)
        total = band.size
        if total == 0:
            result.fail("Band 1 has zero pixels")
            return
        n_valid = int(np.sum(~np.isnan(band)))
        if n_valid / total < _MIN_VALID_FRAC:
            result.fail(
                f"Band 1 is {100.0 * (1 - n_valid / total):.1f}% NaN "
                f"(only {n_valid}/{total} valid pixels, "
                f"minimum {_MIN_VALID_FRAC:.0%})"
            )
    except Exception as exc:
        result.fail(f"NaN check failed for band 1: {exc}")


def format_validation_report(
    results: list[ValidationResult],
) -> str:
    """Format a validation report for console output."""
    total = len(results)
    ok = sum(1 for r in results if r.ok)
    lines = [
        f"Validation report: {ok}/{total} passed",
        "",
    ]
    for r in results:
        label = "OK" if r.ok else "FAIL"
        source_tag = f"[{r.source}]" if r.source else ""
        lines.append(f"  {label:4s} {source_tag} {r.scene_id}")
        for err in r.errors:
            lines.append(f"         - {err}")
    return "\n".join(lines)


__all__ = [
    "ValidationResult",
    "validate_cog",
    "validate_flag_cog",
    "format_validation_report",
]
