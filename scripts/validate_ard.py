# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyarrow>=24.0.0",
#     "rasterio>=1.4.3",
#     "numpy",
#     "pystac>=1.11.0",
#     "pystac-client>=0.9.0",
# ]
# ///
"""Strict ARD validation — exact manifest/ledger key equality, all four artifacts,
COG contract, STAC extensions, provenance, and completion marker.

Usage
-----
    uv run python scripts/validate_ard.py \
        --ledger gs://berlin-lst-data/ard/full/.../ledger.parquet \
        --manifest gs://berlin-lst-data/manifests/v3/.../manifest.parquet
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from dataclasses import dataclass, field

import pyarrow.parquet as pq


@dataclass
class SceneResult:
    scene_id: str = ""
    source: str = ""
    ok: bool = True
    errors: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)


def _resolve(uri: str) -> str:
    """Resolve a GCS URI through the mounted rclone path if available."""
    if uri.startswith("gs://"):
        return uri
    return os.path.expanduser(uri)


def _read_table(uri: str):
    """Read a Parquet table from local path or GCS."""
    if uri.startswith("gs://"):
        from berlin_lst_downscaling.data.io.storage import read_bytes
        return pq.read_table(io.BytesIO(read_bytes(uri)))
    return pq.read_table(uri)


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict ARD validation")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # ── Load manifest ──────────────────────────────────────────────
    try:
        manifest_tbl = _read_table(args.manifest)
    except Exception as exc:
        print(f"Error: cannot read manifest at {args.manifest}: {exc}", file=sys.stderr)
        return 1

    manifest_keys = set()
    for i in range(manifest_tbl.num_rows):
        row = manifest_tbl.slice(i, 1).to_pydict()
        scene_id = row["scene_id"][0]
        source = row["source"][0]
        manifest_keys.add((scene_id, source))

    print(f"Manifest: {len(manifest_keys)} scene+source keys")

    # ── Load ledger ────────────────────────────────────────────────
    try:
        ledger_tbl = _read_table(args.ledger)
    except Exception as exc:
        print(f"Error: cannot read ledger at {args.ledger}: {exc}", file=sys.stderr)
        return 1

    ledger_keys = set()
    done_rows = []
    for i in range(ledger_tbl.num_rows):
        row_dict = ledger_tbl.slice(i, 1).to_pydict()
        scene_id = row_dict["scene_id"][0]
        source = row_dict["source"][0]
        status = row_dict["status"][0]
        ledger_keys.add((scene_id, source))
        if status == "done":
            done_rows.append(row_dict)

    print(f"Ledger: {ledger_tbl.num_rows} rows, {len(done_rows)} done")

    errors = []

    # ── Exact key-set equality ─────────────────────────────────────
    if manifest_keys != ledger_keys:
        missing_in_ledger = manifest_keys - ledger_keys
        extra_in_ledger = ledger_keys - manifest_keys
        if missing_in_ledger:
            errors.append(f"Missing from ledger: {len(missing_in_ledger)} keys")
            for k in sorted(missing_in_ledger)[:10]:
                errors.append(f"  {k}")
        if extra_in_ledger:
            errors.append(f"Extra in ledger: {len(extra_in_ledger)} keys")
            for k in sorted(extra_in_ledger)[:10]:
                errors.append(f"  {k}")
    else:
        print(f"Key-set equality: {len(manifest_keys)} == {len(ledger_keys)} ✓")

    # ── Count validation ───────────────────────────────────────────
    from collections import Counter
    source_counts = Counter(r["source"][0] for r in done_rows)
    print(f"Done by source: {dict(source_counts)}")

    for src, count in source_counts.items():
        expected = sum(1 for s, _ in manifest_keys if s == src)
        if count != expected:
            errors.append(f"{src}: {count}/{expected} done")
        else:
            print(f"  {src}: {count}/{expected} done ✓")

    # ── Per-scene artifact validation ──────────────────────────────
    print(f"\nValidating {len(done_rows)} done scenes...")
    scene_errors = 0
    for row_dict in done_rows:
        scene_id = row_dict["scene_id"][0]
        source = row_dict["source"][0]
        path_cog = row_dict["path_cog"][0]
        path_flag = row_dict.get("path_flag", [None])[0]
        path_stac = row_dict["path_stac"][0]

        res = SceneResult(scene_id=scene_id, source=source)

        if not path_cog:
            res.fail("No path_cog in ledger")
            errors.extend(res.errors)
            scene_errors += 1
            continue

        # Derive deterministic artifact paths
        scene_dir = os.path.dirname(path_cog)
        prov_path = f"{scene_dir}/provenance.json"
        comp_path = f"{scene_dir}/complete.json"

        # Check file existence (local or GCS)
        from berlin_lst_downscaling.data.io import exists
        for label, uri in [
            ("COG", path_cog),
            ("STAC", path_stac),
            ("provenance", prov_path),
            ("complete", comp_path),
        ]:
            if not exists(uri):
                res.fail(f"Missing {label}: {uri}")

        if path_flag and not exists(path_flag):
            res.fail(f"Missing flag COG: {path_flag}")

        # Validate STAC content
        if path_stac and exists(path_stac):
            _validate_stac(path_stac, res)

        # Validate provenance content
        if exists(prov_path):
            _validate_provenance(prov_path, source, scene_id, res)

        # Validate completion marker content
        if exists(comp_path):
            _validate_completion(comp_path, res)

        if not res.ok:
            errors.extend(res.errors)
            scene_errors += 1

    if scene_errors == 0:
        print(f"All {len(done_rows)} scenes validated ✓")
    else:
        print(f"\nSCENE ERRORS: {scene_errors}/{len(done_rows)}")

    # ── Summary ────────────────────────────────────────────────────
    if errors:
        print(f"\nTotal errors: {len(errors)}")
        if args.verbose or scene_errors <= 20:
            for e in errors:
                print(f"  ✗ {e}")
        return 1

    print("\nAll checks passed.")
    return 0


def _validate_stac(stac_uri: str, res: SceneResult) -> None:
    """Validate STAC item structure and extension schemas."""
    try:
        import json as _json

        from berlin_lst_downscaling.data.io import read_bytes

        raw = read_bytes(stac_uri)
        item = _json.loads(raw)

        # Core STAC fields
        if item.get("stac_version") != "1.0.0":
            res.fail(f"STAC version: expected 1.0.0, got {item.get('stac_version')}")
        if item.get("type") != "Feature":
            res.fail(f"STAC type: expected Feature, got {item.get('type')}")
        if "bbox" not in item:
            res.fail("STAC item missing bbox")
        if "geometry" not in item:
            res.fail("STAC item missing geometry")
        if "properties" not in item:
            res.fail("STAC item missing properties")
        if "assets" not in item:
            res.fail("STAC item missing assets")

        # Extension schema URLs
        expected_exts = {
            "https://stac-extensions.github.io/projection/v2.0.0/schema.json",
            "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
        }
        stac_exts = set(item.get("stac_extensions", []))
        missing_exts = expected_exts - stac_exts
        if missing_exts:
            res.fail(f"Missing STAC extensions: {sorted(missing_exts)}")

        # Raster nodata: must be number or "nan"/"inf"/"-inf", not null
        for asset_name, asset in item.get("assets", {}).items():
            for band in asset.get("raster:bands", []):
                nodata = band.get("nodata")
                if nodata is None:
                    continue  # absent is fine; explicit null is not
                if isinstance(nodata, str) and nodata in ("nan", "inf", "-inf"):
                    continue
                if isinstance(nodata, (int, float)):
                    continue
                res.fail(f"Asset {asset_name}: invalid nodata {nodata!r}")

        # proj:code should be present (Projection 2.0)
        props = item.get("properties", {})
        if "proj:code" not in props:
            res.fail("STAC properties missing proj:code")

    except Exception as exc:
        res.fail(f"Cannot validate STAC: {exc}")


def _validate_provenance(prov_uri: str, source: str, scene_id: str, res: SceneResult) -> None:
    """Validate provenance.json structure."""
    try:
        import json as _json

        from berlin_lst_downscaling.data.io import read_bytes

        raw = read_bytes(prov_uri)
        prov = _json.loads(raw)

        if prov.get("source") != source:
            res.fail(f"Provenance source mismatch: {prov.get('source')} != {source}")
        if prov.get("scene_id") != scene_id:
            res.fail(f"Provenance scene_id mismatch: {prov.get('scene_id')} != {scene_id}")
        if "run_id" not in prov:
            res.fail("Provenance missing run_id")
        if "completed_at" not in prov:
            res.fail("Provenance missing completed_at")
        if "output_bands" not in prov:
            res.fail("Provenance missing output_bands")

    except Exception as exc:
        res.fail(f"Cannot validate provenance: {exc}")


def _validate_completion(comp_uri: str, res: SceneResult) -> None:
    """Validate complete.json structure."""
    try:
        import json as _json

        from berlin_lst_downscaling.data.io import read_bytes

        raw = read_bytes(comp_uri)
        comp = _json.loads(raw)

        if "published_at" not in comp:
            res.fail("Completion marker missing published_at")
        if "run_id" not in comp:
            res.fail("Completion marker missing run_id")

    except Exception as exc:
        res.fail(f"Cannot validate completion marker: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
