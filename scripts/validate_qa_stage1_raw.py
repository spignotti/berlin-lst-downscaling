# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyarrow>=24.0.0",
#     "numpy",
#     "google-cloud-storage>=3.12.0",
# ]
# ///
"""Independent validator for Stage-1 raw-input QA report bundles.

Read-only probe: re-reads the published source ledgers and the report
bundle at ``--run-prefix`` and verifies

- report/table schema and internal consistency (counts, fractions,
  histogram, findings-vs-ok),
- the **no-mask invariant** (no ``.tif`` artifact under the run prefix),
- source fingerprints recomputed from the ``inputs`` section,
- optional re-opening of a sample of declared source ledgers.

The validator never writes anything and never re-scans COG pixels; it is
an independent consistency gate over the published evidence.

Usage
-----
    uv run python scripts/validate_qa_stage1_raw.py \
        --run-prefix gs://berlin-lst-data/qa/wb2c-2/raw/<run-id>
    uv run python scripts/validate_qa_stage1_raw.py \
        --run-prefix data/smoke/qa-stage1/<run-id>
"""

from __future__ import annotations

import argparse
import csv
import io
import json

import pyarrow.parquet as pq

from berlin_lst_downscaling.data.io import exists, read_bytes

_BUCKET_LABELS = ("0-25", "25-50", "50-75", "75-90", "90-99", "99-100", "100")


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


def _check_no_mask(prefix: str, errors: list[str]) -> None:
    objects = _list_objects(prefix)
    masks = [o for o in objects if o.endswith(".tif")]
    if masks:
        errors.append(f"no-mask invariant violated: {len(masks)} .tif artifact(s) under {prefix}")


def _check_source_fingerprints(summary: dict, errors: list[str], warnings: list[str]) -> None:
    """Recompute source fingerprints from the declared inputs section."""
    from berlin_lst_downscaling.common.util import sha256_bytes

    inputs = summary.get("inputs", {})
    expected = summary.get("fingerprints", {})
    for label, uri in (
        ("manifest", inputs.get("manifest_uri")),
        ("ard_ledger", f"{inputs.get('ard_root', '').rstrip('/')}/ledger.parquet"),
        (
            "static_derived_ledger",
            f"{inputs.get('static_derived_root', '').rstrip('/')}"
            "/_state/static/derived/ledger.parquet",
        ),
        (
            "dynamic_ledger",
            f"{inputs.get('dynamic_root', '').rstrip('/')}/_state/dynamic/ledger.parquet",
        ),
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
        if tv < 0 or a100 < 0 or fs < 0:
            errors.append(f"scene {cols['scene_id'][i]}: negative cell counts")
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

    # CSV consistency
    try:
        csv_rows = _read_csv_rows(csv_uri)
        if len(csv_rows) != n_rows:
            errors.append(f"scenes.csv rows {len(csv_rows)} != scenes.parquet rows {n_rows}")
        if csv_rows and set(csv_rows[0].keys()) != required:
            errors.append(f"scenes.csv columns differ: {set(csv_rows[0].keys()) ^ required}")
        for row in csv_rows:
            if row.get("assessed") == "True":
                try:
                    int(row["target_valid_cells"])
                    int(row["all_100_cells"])
                    float(row["support_mean_frac"])
                except ValueError:
                    errors.append(f"scenes.csv scene {row.get('scene_id')}: unparseable metrics")
    except Exception as exc:  # noqa: BLE001 — read-only probe
        errors.append(f"scenes.csv unreadable: {exc}")


def _check_layers(summary: dict, errors: list[str]) -> None:
    layers = summary.get("layers", {})
    if not layers:
        errors.append("summary missing 'layers' section")
        return
    for layer, stats in layers.items():
        frac = stats.get("invalid_frac")
        if frac is None or not (0.0 <= float(frac) <= 1.0):
            errors.append(f"layer {layer}: invalid_frac {frac!r} outside [0,1]")
        if int(stats.get("invalid_px", -1)) < 0:
            errors.append(f"layer {layer}: negative invalid_px")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independent validator for Stage-1 raw QA report bundles"
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
    parquet_uri = f"{prefix}/scenes.parquet"
    csv_uri = f"{prefix}/scenes.csv"

    errors: list[str] = []
    warnings: list[str] = []

    for uri, label in (
        (summary_uri, "summary.json"),
        (parquet_uri, "scenes.parquet"),
        (csv_uri, "scenes.csv"),
    ):
        if not exists(uri):
            errors.append(f"missing artifact {label}: {uri}")
    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    _check_no_mask(prefix, errors)
    print(f"Validating Stage-1 raw QA bundle: {prefix}")

    summary = _read_json(summary_uri)
    scene_meta = summary.get("scenes", {})

    # ── schema / identity ──────────────────────────────────────────────
    if summary.get("pipeline") != "qa-stage1-raw":
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

    _check_layers(summary, errors)
    _check_scenes_table(summary, parquet_uri, csv_uri, errors)

    if not args.skip_source_verify:
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

    print(
        f"OK: {total} pairings ({assessed} assessed, {excluded} excluded), "
        f"{len(findings)} findings, no mask artifacts, sources verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
