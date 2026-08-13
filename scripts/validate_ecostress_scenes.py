# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyarrow>=24.0.0",
#     "rasterio>=1.4.3",
#     "numpy",
#     "google-cloud-storage>=3.12.0",
# ]
# ///
"""Validate the six published ECOSTRESS validation scenes (read-only probe).

Reads the canonical v3 manifest bundle and the published ARD ledger, then
checks each of the six ECOSTRESS validation granules end-to-end on the
**published** artefacts (COG + flag COG):

  - manifest row present with ``role=validation`` and ledger row ``done``
  - COG readable, EPSG:25833, single float32 band, canonical 70 m grid
    (shape + origin via ``canon_grid_70m``)
  - flag COG readable, EPSG:25833, single uint8 band, same grid
  - LST NaN/flag consistency: a pixel is NaN if and only if the flag's
    fill bit is set (the ECOSTRESS mask contract — ``data/ard/masking.py``)
  - flag bit fractions (fill, cloudy, shadow, cirrus, saturated, snow/ice)
  - LST value range over valid pixels (informational, not a gate)

This is a pure probe: it prints a per-scene table and a summary, exits
non-zero only for structural/contract failures, and never writes any file,
GCS object, mask, or report.

Note on ECOSTRESS v002 semantics: cloud state lives in the product's
``cloud`` layer, not in the QC bits (NASA LP DAAC). The published flag COG
already folds cloud + water + QC-fill into the fill/cloudy bits, so the
flag fractions below are reported per published bit — a separate
water-vs-QC split is not recoverable from the published flag artefact.

Usage
-----
    uv run python scripts/validate_ecostress_scenes.py \
        --manifest gs://berlin-lst-data/manifests/v3/<bundle>-r2/manifest.parquet \
        --ledger gs://berlin-lst-data/ard/full/<cutoff>/ledger.parquet
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass, field

import numpy as np
import pyarrow.parquet as pq

from berlin_lst_downscaling.common.grid import canon_grid_70m
from berlin_lst_downscaling.data.ard.contract import Contract, contract_for_source
from berlin_lst_downscaling.data.ard.validate import (
    ValidationResult,
    validate_cog,
    validate_flag_cog,
)
from berlin_lst_downscaling.data.io import exists, read_bytes
from berlin_lst_downscaling.data.selection.schema import ECOSTRESS_VALIDATION_IDS

_GRID = canon_grid_70m()

# Physical LST range (Kelvin) — informational bounds for the report only.
_LST_RANGE_K = (150.0, 400.0)

# Flag bit names in Contract bit order.
_FLAG_BITS: tuple[tuple[int, str], ...] = (
    (Contract.FLAG_FILL, "fill"),
    (Contract.FLAG_CLOUDY, "cloudy"),
    (Contract.FLAG_SHADOW, "shadow"),
    (Contract.FLAG_CIRRUS, "cirrus"),
    (Contract.FLAG_SATURATED, "saturated"),
    (Contract.FLAG_SNOW_ICE, "snow_ice"),
)


@dataclass
class SceneCheck:
    """Result of one ECOSTRESS scene's structural/contract validation."""

    scene_id: str
    ok: bool = True
    errors: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)


def _read_table(uri: str):
    """Read a Parquet table from a local path or GCS URI."""
    return pq.read_table(io.BytesIO(read_bytes(uri)))


def _manifest_eco_rows(manifest_uri: str) -> dict[str, dict]:
    """Return ``{scene_id: row}`` for every ECOSTRESS row in the manifest."""
    table = _read_table(manifest_uri)
    cols = table.to_pydict()
    rows: dict[str, dict] = {}
    for i in range(table.num_rows):
        if cols["source"][i] != "ecostress":
            continue
        rows[str(cols["scene_id"][i])] = {
            "role": str(cols["role"][i]),
            "year": int(cols["year"][i]),
        }
    return rows


def _ledger_eco_rows(ledger_uri: str) -> dict[str, dict]:
    """Return ``{scene_id: row}`` for every ECOSTRESS row in the ARD ledger."""
    table = _read_table(ledger_uri)
    cols = table.to_pydict()
    rows: dict[str, dict] = {}
    for i in range(table.num_rows):
        if cols["source"][i] != "ecostress":
            continue
        rows[str(cols["scene_id"][i])] = {
            "status": str(cols["status"][i]),
            "path_cog": cols["path_cog"][i],
            "path_flag": cols["path_flag"][i],
            "aoi_clear_frac": cols.get("aoi_clear_frac", [None] * table.num_rows)[i],
        }
    return rows


def _check_artifacts(scene_id: str, path_cog: str, path_flag: str) -> SceneCheck:
    """Run the structural COG/flag checks plus the NaN/flag contract."""
    check = SceneCheck(scene_id=scene_id)
    contract = contract_for_source("ecostress")

    if not path_cog or not path_flag:
        check.fail(f"ledger row missing path_cog/path_flag: cog={path_cog!r} flag={path_flag!r}")
        return check
    for label, uri in (("COG", path_cog), ("flag", path_flag)):
        if not exists(uri):
            check.fail(f"missing {label}: {uri}")
    if not check.ok:
        return check

    cog_result = validate_cog(path_cog, contract, _GRID)
    _merge(check, "COG", cog_result)
    flag_result = validate_flag_cog(path_flag, _GRID)
    _merge(check, "flag", flag_result)
    if not check.ok:
        return check

    # ── pixel-level contract: dtype + NaN ⟺ fill bit ─────────────────────
    import rasterio

    with rasterio.open(path_cog) as src:
        if src.dtypes[0] != "float32":
            check.fail(f"COG dtype: got {src.dtypes[0]!r}, expected 'float32'")
        lst = src.read(1).astype(np.float32)
    with rasterio.open(path_flag) as src:
        flag = src.read(1).astype(np.uint8)

    isnan = np.isnan(lst)
    fill = (flag & Contract.FLAG_FILL) != 0
    if not np.array_equal(isnan, fill):
        mismatch = int(np.sum(isnan != fill))
        check.fail(f"NaN/fill mismatch on {mismatch} pixels (NaN must equal flag fill bit)")
    return check


def _merge(check: SceneCheck, label: str, result: ValidationResult) -> None:
    """Fold a ValidationResult into the scene check."""
    for err in result.errors:
        check.fail(f"{label}: {err}")


def _flag_fractions(flag: np.ndarray) -> dict[str, float]:
    """Return per-bit pixel fractions over the whole COG grid."""
    total = max(int(flag.size), 1)
    return {name: float(np.sum((flag & bit) != 0)) / total for bit, name in _FLAG_BITS}


def _lst_range(lst: np.ndarray) -> tuple[float | None, float | None]:
    """Return (min, max) of valid LST values, or (None, None) if none valid."""
    valid = lst[~np.isnan(lst)]
    if valid.size == 0:
        return None, None
    return float(valid.min()), float(valid.max())


def _pct(value: float | None) -> float:
    """Return *value* as a percentage, or NaN when absent."""
    return value * 100.0 if value is not None else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Published ECOSTRESS validation probe")
    parser.add_argument("--manifest", required=True, help="v3 manifest.parquet URI")
    parser.add_argument("--ledger", required=True, help="ARD ledger.parquet URI")
    args = parser.parse_args()

    expected = sorted(ECOSTRESS_VALIDATION_IDS)

    # ── load sources ─────────────────────────────────────────────────────
    try:
        manifest_rows = _manifest_eco_rows(args.manifest)
        ledger_rows = _ledger_eco_rows(args.ledger)
    except Exception as exc:
        print(f"Error: cannot read manifest/ledger: {exc}", file=sys.stderr)
        return 1

    failures: list[str] = []
    table: list[dict] = []

    for scene_id in expected:
        manifest = manifest_rows.get(scene_id)
        if manifest is None:
            failures.append(f"{scene_id}: missing from manifest (ecostress row)")
            continue
        if manifest["role"] != "validation":
            failures.append(
                f"{scene_id}: manifest role={manifest['role']!r}, expected 'validation'"
            )

        ledger = ledger_rows.get(scene_id)
        if ledger is None:
            failures.append(f"{scene_id}: missing from ARD ledger")
            continue
        if ledger["status"] != "done":
            failures.append(f"{scene_id}: ledger status={ledger['status']!r}, expected 'done'")

        check = _check_artifacts(scene_id, ledger["path_cog"], ledger["path_flag"])
        if not check.ok:
            failures.extend(check.errors)

        # ── report metrics (informational, never a gate) ──────────────────
        import rasterio

        fractions: dict[str, float] = {}
        lst_min = lst_max = None
        try:
            with rasterio.open(ledger["path_cog"]) as src:
                lst = src.read(1).astype(np.float32)
            with rasterio.open(ledger["path_flag"]) as src:
                flag = src.read(1).astype(np.uint8)
            fractions = _flag_fractions(flag)
            lst_min, lst_max = _lst_range(lst)
        except Exception as exc:
            fractions = {}
            if check.ok:  # only surface if structural checks passed
                failures.append(f"{scene_id}: metric read failed: {exc}")

        row = {
            "scene_id": scene_id,
            "status": ledger["status"],
            "ok": check.ok,
            "lst_min_k": lst_min,
            "lst_max_k": lst_max,
            "ledger_clear_frac": ledger["aoi_clear_frac"],
        }
        row.update({f"flag_{name}": fractions.get(name) for _, name in _FLAG_BITS})
        table.append(row)

    # ── console output ───────────────────────────────────────────────────
    print(f"ECOSTRESS validation probe — {len(expected)} expected granules")
    print(f"  manifest rows: {len(manifest_rows)} | ledger rows: {len(ledger_rows)}")
    print()
    header = (
        f"{'scene_id':<44} {'ok':<4} {'lst_min':>8} {'lst_max':>8} {'clear%':>7} "
        f"{'fill%':>6} {'cloud%':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in table:
        print(
            f"{row['scene_id']:<44} {str(row['ok']):<4} "
            f"{row['lst_min_k'] if row['lst_min_k'] is not None else float('nan'):>8.1f} "
            f"{row['lst_max_k'] if row['lst_max_k'] is not None else float('nan'):>8.1f} "
            f"{(row['ledger_clear_frac'] or float('nan')) * 100.0:>7.1f} "
            f"{_pct(row['flag_fill']):>6.1f} "
            f"{_pct(row['flag_cloudy']):>7.1f}"
        )

    # Physical-range outliers are reported, not failures.
    for row in table:
        lst_min = row["lst_min_k"]
        lst_max = row["lst_max_k"]
        if lst_min is not None and not (_LST_RANGE_K[0] <= lst_min <= _LST_RANGE_K[1]):
            print(f"  note: {row['scene_id']} LST min {lst_min:.1f} K outside physical range")
        if lst_max is not None and not (_LST_RANGE_K[0] <= lst_max <= _LST_RANGE_K[1]):
            print(f"  note: {row['scene_id']} LST max {lst_max:.1f} K outside physical range")

    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("All 6 ECOSTRESS scenes passed structural/contract checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
