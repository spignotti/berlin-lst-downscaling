# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyarrow>=24.0.0",
#     "numpy",
#     "google-cloud-storage>=3.12.0",
# ]
# ///
"""Independent validator for Stage-2 feature-stack QA report bundles.

Read-only probe: re-reads the published source ledgers and the report
bundle at ``--run-prefix`` and verifies

- report/table schema and internal consistency (counts, fractions,
  histogram totals, findings-vs-ok, expected exclusions),
- the **no-raster invariant** (no ``.tif`` artifact under the run prefix:
  Stage-2 must not produce or modify any validity/selection mask),
- source fingerprints recomputed from the ``inputs`` section, including
  the features ledger,
- profile-table consistency (per-scene 28-channel coverage, valid-pixel
  counts, fixed-bin histogram totals).

The validator never writes anything and never re-scans COG pixels; it is
an independent consistency gate over the published evidence. It does not
share implementation with ``data/qa/stage2_features.py``.

Usage
-----
    uv run python scripts/validate_qa_stage2_features.py \
        --run-prefix gs://berlin-lst-data/qa/stage2_features/<run-id>
    uv run python scripts/validate_qa_stage2_features.py \
        --run-prefix data/smoke/qa-stage2/<run-id>
"""

from __future__ import annotations

import argparse
import csv
import io
import json

import pyarrow.parquet as pq

from berlin_lst_downscaling.common.util import sha256_bytes
from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_CHANNEL_NAMES,
    FEATURE_CHANNELS,
)
from berlin_lst_downscaling.data.io import exists, read_bytes
from berlin_lst_downscaling.data.qa.inventory import INFERENCE_EXCLUSION_REASON

_BUCKET_LABELS = ("0-25", "25-50", "50-75", "75-90", "90-99", "99-100", "100")
_EXPECTED_EXCLUSIONS = frozenset({INFERENCE_EXCLUSION_REASON})

_N_CHANNELS = len(FEATURE_CHANNELS)


def _read_json(uri: str) -> dict:
    return json.loads(read_bytes(uri))


def _read_table(uri: str):
    return pq.read_table(io.BytesIO(read_bytes(uri)))


def _read_csv_rows(uri: str) -> list[dict]:
    text = read_bytes(uri).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _list_objects(prefix: str) -> list[str]:
    """List object keys under a local dir or GCS prefix."""
    if prefix.startswith("gs://"):
        from google.cloud import storage  # type: ignore[import-untyped]

        client = storage.Client()
        bucket_name, _, key = prefix.removeprefix("gs://").partition("/")
        bucket = client.get_bucket(bucket_name)
        return [
            f"gs://{bucket_name}/{blob.name}"
            for blob in bucket.list_blobs(prefix=key.rstrip("/") + "/")
        ]
    import os

    return [
        os.path.join(prefix, name)
        for name in sorted(os.listdir(prefix))
        if os.path.isfile(os.path.join(prefix, name))
    ]


def _check_no_raster(prefix: str, errors: list[str]) -> None:
    objects = _list_objects(prefix)
    masks = [o for o in objects if o.endswith(".tif")]
    if masks:
        errors.append(
            f"no-raster invariant violated: {len(masks)} .tif artifact(s) under {prefix}"
        )


def _check_source_fingerprints(summary: dict, errors: list[str], warnings: list[str]) -> None:
    """Recompute source fingerprints from the declared inputs section.

    ``geometry_mapping`` is a JSON file whose fingerprint is the raw
    SHA-256 prefix, matching ``load_geometry_mapping`` semantics.
    """
    inputs = summary.get("inputs", {})
    expected = summary.get("fingerprints", {})
    for label, uri in (
        ("manifest", inputs.get("manifest_uri")),
        ("ard_ledger", f"{inputs.get('ard_root', '').rstrip('/')}/ledger.parquet"),
        (
            "static_sources_ledger",
            f"{inputs.get('static_sources_root', '').rstrip('/')}/ledger.parquet",
        ),
        (
            "static_derived_ledger",
            f"{inputs.get('static_derived_root', '').rstrip('/')}"
            "/_state/static/derived/ledger.parquet",
        ),
        (
            "dynamic_ledger",
            f"{inputs.get('dynamic_root', '').rstrip('/')}/_state/dynamic/ledger.parquet",
        ),
        (
            "features_ledger",
            f"{inputs.get('features_root', '').rstrip('/')}/_state/features/ledger.parquet",
        ),
        ("geometry_mapping", inputs.get("geometry_mapping_uri")),
    ):
        if not uri or not uri.endswith((".parquet", ".json")):
            warnings.append(f"source {label}: no verifiable URI in report inputs")
            continue
        try:
            actual = sha256_bytes(read_bytes(uri))[:16]
            declared = expected.get(label)
            if declared != actual:
                errors.append(
                    f"source fingerprint mismatch {label}: declared {declared!r}, "
                    f"recomputed {actual!r}"
                )
            else:
                print(f"  {label}: fingerprint OK ({actual})")
        except Exception as exc:  # noqa: BLE001 — read-only probe
            warnings.append(f"source {label}: cannot re-read {uri}: {exc}")


def _check_scenes_table(summary: dict, parquet_uri: str, csv_uri: str, errors: list[str]) -> None:
    table = _read_table(parquet_uri)
    n_rows = table.num_rows
    scene_meta = summary.get("scenes", {})

    if n_rows != scene_meta.get("total_pairings"):
        errors.append(
            f"scenes.parquet rows {n_rows} != total_pairings {scene_meta.get('total_pairings')}"
        )

    cols = table.to_pydict()
    required = {
        "scene_id",
        "year",
        "s2_scene_id",
        "geometry_id",
        "assessed",
        "exclusion_reason",
        "target_valid_cells",
        "all_100_cells",
        "full_support_cells",
        "support_mean_frac",
        "support_histogram",
        "feature_valid_px",
        "inside_aoi_px",
        "feature_valid_frac_of_aoi",
        "errors",
    }
    missing = required - set(cols)
    if missing:
        errors.append(f"scenes.parquet missing columns: {sorted(missing)}")
        return

    assessed = [bool(a) for a in cols["assessed"]]
    if sum(assessed) != scene_meta.get("assessed"):
        errors.append(
            f"scenes.assessed rows {sum(assessed)} != summary assessed {scene_meta.get('assessed')}"
        )
    if (n_rows - sum(assessed)) != scene_meta.get("excluded"):
        errors.append(
            f"scenes excluded rows {n_rows - sum(assessed)} != summary excluded "
            f"{scene_meta.get('excluded')}"
        )

    for i in range(n_rows):
        if not assessed[i]:
            continue
        tv = int(cols["target_valid_cells"][i])
        a100 = int(cols["all_100_cells"][i])
        fs = int(cols["full_support_cells"][i])
        mean = float(cols["support_mean_frac"][i])
        fvp = int(cols["feature_valid_px"][i])
        inside = int(cols["inside_aoi_px"][i])
        if tv < 0 or a100 < 0 or fs < 0 or fvp < 0 or inside < 0:
            errors.append(f"scene {cols['scene_id'][i]}: negative counts")
        if a100 > tv or fs > tv:
            errors.append(
                f"scene {cols['scene_id'][i]}: support cells exceed target-valid "
                f"({a100}/{fs} > {tv})"
            )
        if not (0.0 <= mean <= 1.0):
            errors.append(f"scene {cols['scene_id'][i]}: support_mean_frac {mean} outside [0,1]")
        hist = json.loads(cols["support_histogram"][i])
        if sum(hist.values()) > tv:
            errors.append(
                f"scene {cols['scene_id'][i]}: support histogram sum "
                f"{sum(hist.values())} > target_valid {tv}"
            )
        unknown = set(hist) - set(_BUCKET_LABELS)
        if unknown:
            errors.append(f"scene {cols['scene_id'][i]}: unknown histogram buckets {unknown}")
        frac = float(cols["feature_valid_frac_of_aoi"][i])
        if not (0.0 <= frac <= 1.0):
            errors.append(
                f"scene {cols['scene_id'][i]}: feature_valid_frac_of_aoi {frac} outside [0,1]"
            )
        if inside > 0 and fvp > inside:
            errors.append(
                f"scene {cols['scene_id'][i]}: feature_valid_px {fvp} > inside_aoi_px {inside}"
            )

    # CSV consistency
    try:
        csv_rows = _read_csv_rows(csv_uri)
        if len(csv_rows) != n_rows:
            errors.append(f"scenes.csv rows {len(csv_rows)} != scenes.parquet rows {n_rows}")
        if csv_rows and set(csv_rows[0].keys()) != required:
            errors.append(f"scenes.csv columns differ: {set(csv_rows[0].keys()) ^ required}")
    except Exception as exc:  # noqa: BLE001 — read-only probe
        errors.append(f"scenes.csv unreadable: {exc}")


def _check_profiles_table(
    summary: dict, parquet_uri: str, csv_uri: str, scenes_parquet_uri: str, errors: list[str]
) -> None:
    """Verify the profile table covers every assessed scene with all channels."""
    table = _read_table(parquet_uri)
    cols = table.to_pydict()
    required = {
        "scene_id",
        "channel_index",
        "channel_name",
        "family",
        "unit",
        "valid_px",
        "min",
        "max",
        "mean",
        "std",
        "histogram",
    }
    missing = required - set(cols)
    if missing:
        errors.append(f"profiles.parquet missing columns: {sorted(missing)}")
        return

    scenes_table = _read_table(scenes_parquet_uri)
    scene_cols = scenes_table.to_pydict()
    assessed_scenes = [
        str(scene_cols["scene_id"][i])
        for i in range(scenes_table.num_rows)
        if bool(scene_cols["assessed"][i])
    ]

    # group profile rows per scene
    by_scene: dict[str, list[int]] = {}
    for i in range(table.num_rows):
        sid = str(cols["scene_id"][i])
        by_scene.setdefault(sid, []).append(i)
    missing_scenes = [s for s in assessed_scenes if s not in by_scene]
    if missing_scenes:
        errors.append(f"profiles missing scenes: {missing_scenes}")

    for sid, idxs in by_scene.items():
        if len(idxs) != _N_CHANNELS:
            errors.append(f"scene {sid}: profile rows {len(idxs)}, expected {_N_CHANNELS}")
            continue
        seen = {int(cols["channel_index"][i]) for i in idxs}
        if seen != set(range(1, _N_CHANNELS + 1)):
            errors.append(f"scene {sid}: channel_index set {sorted(seen)} incomplete")
        names = {str(cols["channel_name"][i]) for i in idxs}
        if names != set(FEATURE_CHANNEL_NAMES):
            errors.append(f"scene {sid}: channel names {sorted(names)} != contract")

        for i in idxs:
            vp = int(cols["valid_px"][i])
            if vp < 0:
                errors.append(f"scene {sid}: channel {cols['channel_name'][i]} negative valid_px")
            hist = json.loads(cols["histogram"][i])
            if sum(hist.values()) != vp:
                errors.append(
                    f"scene {sid}: channel {cols['channel_name'][i]} histogram sum "
                    f"{sum(hist.values())} != valid_px {vp}"
                )
            unknown = [k for k in hist if not k.isdigit()]
            if unknown:
                name = cols["channel_name"][i]
                errors.append(f"scene {sid}: channel {name} bad bin keys {unknown}")

    # CSV consistency
    try:
        csv_rows = _read_csv_rows(csv_uri)
        if len(csv_rows) != table.num_rows:
            errors.append(
                f"profiles.csv rows {len(csv_rows)} != profiles.parquet rows {table.num_rows}"
            )
    except Exception as exc:  # noqa: BLE001 — read-only probe
        errors.append(f"profiles.csv unreadable: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independent validator for Stage-2 feature-stack QA report bundles"
    )
    parser.add_argument("--run-prefix", required=True, help="QA run directory (local or gs://)")
    parser.add_argument(
        "--skip-source-verify",
        action="store_true",
        help="Do not re-read source ledgers to verify fingerprints",
    )
    args = parser.parse_args()

    prefix = args.run_prefix.rstrip("/")
    summary_uri = f"{prefix}/summary.json"
    scenes_parquet = f"{prefix}/scenes.parquet"
    scenes_csv = f"{prefix}/scenes.csv"
    profiles_parquet = f"{prefix}/profiles.parquet"
    profiles_csv = f"{prefix}/profiles.csv"

    errors: list[str] = []
    warnings: list[str] = []

    for uri, label in (
        (summary_uri, "summary.json"),
        (scenes_parquet, "scenes.parquet"),
        (scenes_csv, "scenes.csv"),
        (profiles_parquet, "profiles.parquet"),
        (profiles_csv, "profiles.csv"),
    ):
        if not exists(uri):
            errors.append(f"missing artifact {label}: {uri}")
    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    _check_no_raster(prefix, errors)
    print(f"Validating Stage-2 feature QA bundle: {prefix}")

    summary = _read_json(summary_uri)
    scene_meta = summary.get("scenes", {})

    # ── schema / identity ──────────────────────────────────────────────
    if summary.get("pipeline") != "qa-stage2-features":
        errors.append(f"pipeline identity: {summary.get('pipeline')!r}")
    if not summary.get("run_id"):
        errors.append("summary missing run_id")
    if not isinstance(summary.get("ok"), bool):
        errors.append("summary 'ok' must be a bool")

    total = scene_meta.get("total_pairings")
    assessed = scene_meta.get("assessed")
    excluded = scene_meta.get("excluded")
    if total is None or assessed is None or excluded is None:
        errors.append("summary scenes section incomplete")
    else:
        if assessed + excluded != total:
            errors.append(f"assessed {assessed} + excluded {excluded} != total {total}")
        reasons = scene_meta.get("exclusion_reasons", {})
        if sum(reasons.values()) != excluded:
            errors.append(f"exclusion_reasons sum {sum(reasons.values())} != excluded {excluded}")
        unknown_reasons = set(reasons) - _EXPECTED_EXCLUSIONS
        if unknown_reasons:
            errors.append(f"unexpected exclusion reasons: {sorted(unknown_reasons)}")

    if not summary.get("fingerprints"):
        errors.append("summary missing fingerprints")

    # ── findings ↔ ok ──────────────────────────────────────────────────
    findings = summary.get("findings", [])
    if summary.get("ok") and findings:
        errors.append(f"summary ok=True but {len(findings)} findings present")
    if not summary.get("ok") and not findings:
        errors.append("summary ok=False but no findings recorded")

    # ── aggregate ──────────────────────────────────────────────────────
    aggregate = summary.get("aggregate", {})
    if aggregate.get("assessed_scenes") != assessed:
        errors.append(
            f"aggregate assessed_scenes {aggregate.get('assessed_scenes')} "
            f"!= scenes assessed {assessed}"
        )
    hist = aggregate.get("support_histogram", {})
    if sum(hist.values()) > aggregate.get("target_valid_cells", -1):
        errors.append(
            f"aggregate support histogram sum {sum(hist.values())} > "
            f"target_valid_cells {aggregate.get('target_valid_cells')}"
        )
    unknown = set(hist) - set(_BUCKET_LABELS)
    if unknown:
        errors.append(f"aggregate: unknown histogram buckets {unknown}")

    _check_scenes_table(summary, scenes_parquet, scenes_csv, errors)
    _check_profiles_table(summary, profiles_parquet, profiles_csv, scenes_parquet, errors)

    sources_verified = not args.skip_source_verify
    if sources_verified:
        _check_source_fingerprints(summary, errors, warnings)

    # ── report ─────────────────────────────────────────────────────────
    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  ✗ {e}")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")
    if errors:
        return 1

    if sources_verified:
        print(
            f"OK: {total} pairings ({assessed} assessed, {excluded} excluded), "
            f"{len(findings)} findings, no raster artifacts, sources verified."
        )
    else:
        print(
            f"OK: {total} pairings ({assessed} assessed, {excluded} excluded), "
            f"{len(findings)} findings, no raster artifacts (source verification skipped)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())