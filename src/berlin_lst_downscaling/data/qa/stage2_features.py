"""Stage-2 feature-stack QA core — blockwise verification of published stacks.

Verifies every published 24-band feature stack of the training universe
against the feature-stack contract and computes a **diagnostic** feature
support statistic at 100 m (valid Landsat target cell plus its
``feature_valid`` 10 m subpixels). No validity mask, no selection
artifact, and no resampling is produced — the report bundle is the only
output. Publication of the ``training_eligible@100m`` selection mask is a
WB2c-4 (training-data preparation) decision.

Reuses the Stage-1 grid/window/inventory primitives (see the ``# decision:``
comment at import). Range semantics derive from the feature-stack contract
(``data/features/contracts.py``), never re-derived.

Performance design:

- Pixel scans use 2560x2560 10 m tiles — an exact multiple of both the
  COG 512 px block size and the 10x10 aggregation factor. The 24-band
  float32 feature COG is read per tile (~630 MB peak for one tile).
- The ``feature_valid`` mask is read once per tile and checked for
  ``mask == 1 ⇔ all 24 channels finite and in declared range``.
- Landsat target/flag are read once per scene at 100 m (tiny).
"""

from __future__ import annotations

import csv as _csv
import io as _io
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.windows import Window

from berlin_lst_downscaling.common.util import sha256_bytes
from berlin_lst_downscaling.data.features.contracts import (
    FEATURE_CHANNEL_NAMES,
    FEATURE_CHANNELS,
    FeatureChannel,
)
from berlin_lst_downscaling.data.features.paths import (
    feature_cog,
    feature_completion,
    feature_mask_cog,
    feature_provenance,
    feature_stac,
    ledger_path,
)
from berlin_lst_downscaling.data.io import atomic_write, log_event, read_bytes
from berlin_lst_downscaling.data.qa.contracts import LST_RANGE_K
from berlin_lst_downscaling.data.qa.inventory import (
    INFERENCE_EXCLUSION_REASON,
    ResolvedScene,
    build_inventory,
)

# decision: reuse the Stage-1 gate's grid/window primitives directly instead of
# re-implementing them — the Stage-2 brief mandates identical gate logic, and
# duplicated offset/tile math would drift from Stage-1. Alternative: private
# copies (rejected: drift risk).
from berlin_lst_downscaling.data.qa.stage1_raw import (
    _check_metadata_10m,
    _tile_windows,
    _window_offset,
    analysis_grid_10m,
)

_logger = logging.getLogger(__name__)

_GRID_CRS = "EPSG:25833"

_EXPECTED_EXCLUSIONS = frozenset({INFERENCE_EXCLUSION_REASON})

_SUPPORT_BINS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
_BUCKET_LABELS = ("0-25", "25-50", "50-75", "75-90", "90-99", "99-100", "100")


# ── report types ─────────────────────────────────────────────────────


@dataclass
class SceneMetrics:
    """Per-scene Stage-2 results (one row in the scene table)."""

    scene_id: str
    year: int
    s2_scene_id: str
    geometry_id: str
    assessed: bool
    exclusion_reason: str | None
    target_valid_cells: int = 0
    all_100_cells: int = 0
    full_support_cells: int = 0
    support_mean_frac: float = 0.0
    support_histogram: dict[str, int] = field(default_factory=dict)
    feature_valid_px: int = 0
    inside_aoi_px: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class ChannelProfile:
    """One (scene, channel) profile row — fixed-bin histogram + summary stats."""

    scene_id: str
    channel_index: int  # 1-based band number in the COG
    channel_name: str
    family: str
    unit: str
    valid_px: int
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    histogram: dict[str, int]


@dataclass
class Stage2Report:
    """Complete Stage-2 QA report (serialised to summary.json + tables)."""

    run_id: str
    timestamp: str
    grid: dict
    inputs: dict
    fingerprints: dict
    total_pairings: int
    assessed: int
    excluded: int
    exclusion_reasons: dict[str, int]
    aggregate: dict
    findings: list[str]
    scenes: list[SceneMetrics]
    profiles: list[ChannelProfile]

    @property
    def ok(self) -> bool:
        return len(self.findings) == 0


# ── sidecar verification (no pixel reads) ────────────────────────────


def _check_sidecars(
    scene_id: str,
    cog_uri: str,
    stac_uri: str,
    prov_uri: str,
    comp_uri: str,
) -> tuple[list[str], dict]:
    """Verify complete/provenance/STAC sidecars; return (errors, coverage)."""
    errors: list[str] = []
    coverage: dict = {}
    try:
        comp = json.loads(read_bytes(comp_uri))
        if not comp.get("published_at"):
            errors.append(f"{scene_id}: complete.json missing published_at")
    except Exception as exc:  # noqa: BLE001 — report, never crash the run
        errors.append(f"{scene_id}: complete.json unreadable: {exc}")

    try:
        prov = json.loads(read_bytes(prov_uri))
        if tuple(prov.get("channel_order", ())) != FEATURE_CHANNEL_NAMES:
            errors.append(f"{scene_id}: provenance channel_order mismatch")
        for key in ("config_hash", "coverage", "mask_semantics", "vegetation_dsm_policy"):
            if key not in prov:
                errors.append(f"{scene_id}: provenance missing {key!r}")
        coverage = prov.get("coverage", {})
        for key in ("feature_valid_px", "inside_aoi_px", "outside_aoi_px"):
            if key not in coverage:
                errors.append(f"{scene_id}: coverage missing {key!r}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{scene_id}: provenance unreadable: {exc}")

    try:
        stac = json.loads(read_bytes(stac_uri))
        dt = stac.get("properties", {}).get("datetime")
        if not dt or "T" not in str(dt):
            errors.append(f"{scene_id}: STAC datetime missing or not RFC 3339: {dt!r}")
        assets = stac.get("assets", {})
        if "data" not in assets or "feature_valid" not in assets:
            errors.append(f"{scene_id}: STAC missing data/feature_valid assets")
        else:
            raster_bands = assets["data"].get("raster:bands", [])
            if len(raster_bands) != len(FEATURE_CHANNELS):
                errors.append(
                    f"{scene_id}: STAC data raster:bands {len(raster_bands)}, "
                    f"expected {len(FEATURE_CHANNELS)}"
                )
            fv_bands = assets["feature_valid"].get("raster:bands", [])
            if not fv_bands or fv_bands[0].get("data_type") != "uint8":
                errors.append(f"{scene_id}: STAC feature_valid asset not uint8")
            if assets["data"].get("href") != cog_uri:
                errors.append(f"{scene_id}: STAC data href does not match COG URI")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{scene_id}: STAC unreadable: {exc}")
    return errors, coverage


# ── fixed-bin channel profiles ───────────────────────────────────────


def _profile_histogram(vals: np.ndarray, spec: FeatureChannel, n_bins: int) -> dict[str, int]:
    """Equal-width histogram over the channel's declared valid_range."""
    if spec.valid_range is None or vals.size == 0:
        return {}
    lo, hi = spec.valid_range
    if hi <= lo:
        return {}
    idx = np.digitize(vals, np.linspace(lo, hi, n_bins + 1)) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    return {str(i): int(c) for i, c in enumerate(counts) if c}


# ── per-scene scan ───────────────────────────────────────────────────


def _scan_stack(
    scene: ResolvedScene,
    features_root: str,
    analysis_10: GeoBox,
    analysis_100: GeoBox,
    n_profile_bins: int,
) -> tuple[SceneMetrics, list[ChannelProfile], list[str]]:
    """Run the blockwise scan for one assessable feature stack.

    Returns (metrics, profiles, findings).
    """
    findings: list[str] = []
    errors: list[str] = []

    scene_id = scene.scene_id
    cog_uri = feature_cog(features_root, scene_id)
    mask_uri = feature_mask_cog(features_root, scene_id)
    stac_uri = feature_stac(features_root, scene_id)
    prov_uri = feature_provenance(features_root, scene_id)
    comp_uri = feature_completion(features_root, scene_id)

    # ── metadata + sidecar checks (canonical grids, no pixel reads) ────
    errors += _check_metadata_10m(cog_uri, n_bands=len(FEATURE_CHANNELS), dtype="float32")
    errors += _check_metadata_10m(mask_uri, n_bands=1, dtype="uint8")
    sidecar_errors, coverage = _check_sidecars(
        scene_id, cog_uri, stac_uri, prov_uri, comp_uri
    )
    errors += sidecar_errors

    # ── Landsat target (100 m, full read — tiny) ───────────────────────
    st = flag = None
    ls_off = (0, 0)
    try:
        with rasterio.open(scene.landsat_cog) as src:
            st = src.read(1).astype(np.float32)
            ls_off = _window_offset(src, analysis_100, 100.0)
        with rasterio.open(scene.landsat_flag) as src:
            flag = src.read(1).astype(np.uint8)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{scene_id}: cannot read Landsat target: {exc}")

    H100, W100 = analysis_100.shape.y, analysis_100.shape.x
    target_valid = np.zeros((H100, W100), dtype=bool)
    if st is not None and flag is not None:
        co, ro = ls_off
        y0, y1 = max(ro, 0), min(ro + H100, st.shape[0])
        x0, x1 = max(co, 0), min(co + W100, st.shape[1])
        sl_st = st[y0:y1, x0:x1]
        sl_fl = flag[y0:y1, x0:x1]
        target_valid[y0 - ro : y1 - ro, x0 - co : x1 - co] = (
            (sl_fl == 0) & (sl_st >= LST_RANGE_K[0]) & (sl_st <= LST_RANGE_K[1])
        )

    # ── present-count grid (edge-truncated 100 m cells) ────────────────
    H10, W10 = analysis_10.shape.y, analysis_10.shape.x
    iy = np.arange(H100, dtype=np.int64)[:, None]
    ix = np.arange(W100, dtype=np.int64)[None, :]
    present = np.minimum(10, H10 - 10 * iy) * np.minimum(10, W10 - 10 * ix)

    support = np.zeros((H100, W100), dtype=np.int32)
    feature_valid_px = 0
    agg: dict[int, dict] = {}

    try:
        with rasterio.open(cog_uri) as cog, rasterio.open(mask_uri) as msk:
            cog_off = _window_offset(cog, analysis_10, 10.0)
            mask_off = _window_offset(msk, analysis_10, 10.0)
            for (r0, c0, r1, c1), _ in _tile_windows(analysis_10):
                bh, bw = r1 - r0, c1 - c0
                w_cog = Window(c0 + cog_off[0], r0 + cog_off[1], bw, bh)  # type: ignore[call-arg]
                bands = cog.read(window=w_cog)  # (24, bh, bw) float32
                w_msk = Window(c0 + mask_off[0], r0 + mask_off[1], bw, bh)  # type: ignore[call-arg]
                mask = msk.read(1, window=w_msk)

                if not set(np.unique(mask)).issubset({0, 1}):
                    findings.append(
                        f"{scene_id}: mask values {np.unique(mask)} not in {{0,1}} "
                        f"(tile {r0},{c0})"
                    )

                # validity equivalence: mask == 1 ⇔ all 24 finite + in range
                finite_all = np.all(np.isfinite(bands), axis=0)
                in_range = np.ones((bh, bw), dtype=bool)
                for i, spec in enumerate(FEATURE_CHANNELS):
                    if spec.valid_range is None:
                        continue
                    lo, hi = spec.valid_range
                    in_range &= (bands[i] >= lo) & (bands[i] <= hi)
                claim = mask == 1
                mismatch = claim != (finite_all & in_range)
                if np.any(mismatch):
                    n = int(np.sum(mismatch))
                    findings.append(
                        f"{scene_id}: feature_valid disagrees with finite/in-range on {n} px "
                        f"(tile {r0},{c0})"
                    )
                if np.any(claim & ~finite_all):
                    n = int(np.sum(claim & ~finite_all))
                    findings.append(f"{scene_id}: mask==1 with non-finite values on {n} px")

                valid = claim
                feature_valid_px += int(np.sum(valid))

                # ── channel profiles over valid pixels ─────────────────
                for i, spec in enumerate(FEATURE_CHANNELS):
                    acc = agg.setdefault(
                        i, {"count": 0, "sum": 0.0, "sum2": 0.0, "mn": None, "mx": None, "hist": {}}
                    )
                    vals = bands[i][valid]
                    if vals.size == 0:
                        continue
                    acc["count"] += int(vals.size)
                    acc["sum"] += float(np.sum(vals, dtype=np.float64))
                    acc["sum2"] += float(np.sum(vals * vals, dtype=np.float64))
                    acc["mn"] = (
                        float(vals.min())
                        if acc["mn"] is None
                        else min(acc["mn"], float(vals.min()))
                    )
                    acc["mx"] = (
                        float(vals.max())
                        if acc["mx"] is None
                        else max(acc["mx"], float(vals.max()))
                    )
                    for k, c in _profile_histogram(vals, spec, n_profile_bins).items():
                        acc["hist"][k] = acc["hist"].get(k, 0) + c

                # ── aggregate to 100 m (exact nested 10x10) ───────────
                ph, pw = -(-bh // 10) * 10, -(-bw // 10) * 10
                if ph != bh or pw != bw:
                    m = np.zeros((ph, pw), dtype=np.uint8)
                    m[:bh, :bw] = valid
                else:
                    m = valid.astype(np.uint8)
                block_sum = m.reshape(ph // 10, 10, pw // 10, 10).sum(axis=(1, 3)).astype(
                    np.int32
                )
                br0, bc0 = r0 // 10, c0 // 10
                support[br0 : br0 + ph // 10, bc0 : bc0 + pw // 10] += block_sum
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{scene_id}: pixel scan failed: {exc}")

    # ── finalise channel profiles ──────────────────────────────────────
    profiles: list[ChannelProfile] = []
    for i, spec in enumerate(FEATURE_CHANNELS):
        acc = agg.get(i)
        if acc is None or acc["count"] == 0:
            profiles.append(
                ChannelProfile(
                    scene_id=scene_id,
                    channel_index=i + 1,
                    channel_name=spec.name,
                    family=spec.family,
                    unit=spec.unit,
                    valid_px=0,
                    min=None,
                    max=None,
                    mean=None,
                    std=None,
                    histogram={},
                )
            )
            continue
        n = acc["count"]
        mean = acc["sum"] / n
        var = max(acc["sum2"] / n - mean * mean, 0.0)
        profiles.append(
            ChannelProfile(
                scene_id=scene_id,
                channel_index=i + 1,
                channel_name=spec.name,
                family=spec.family,
                unit=spec.unit,
                valid_px=n,
                min=acc["mn"],
                max=acc["mx"],
                mean=round(mean, 6),
                std=round(float(np.sqrt(var)), 6),
                histogram=dict(sorted(acc["hist"].items(), key=lambda kv: int(kv[0]))),
            )
        )

    # ── support statistics over target-valid cells ────────────────────
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(present > 0, support / np.maximum(present, 1), np.nan)
    cells = target_valid & (present > 0)
    target_valid_cells = int(np.sum(target_valid))
    all_100_cells = int(np.sum(target_valid & (present == 10 * 10) & (support == 100)))
    full_support_cells = int(np.sum(target_valid & (support == present)))
    support_fracs = frac[cells]
    support_mean = float(np.nanmean(support_fracs)) if support_fracs.size else 0.0
    hist: dict[str, int] = {}
    if support_fracs.size:
        idx = np.digitize(support_fracs, _SUPPORT_BINS) - 1
        counts = np.bincount(idx, minlength=len(_BUCKET_LABELS))
        for label, count in zip(_BUCKET_LABELS, counts, strict=False):
            if count:
                hist[label] = int(count)

    inside_aoi_px = int(coverage.get("inside_aoi_px", 0) or 0)
    metrics = SceneMetrics(
        scene_id=scene_id,
        year=scene.year,
        s2_scene_id=scene.s2_scene_id,
        geometry_id=scene.geometry_id,
        assessed=True,
        exclusion_reason=None,
        target_valid_cells=target_valid_cells,
        all_100_cells=all_100_cells,
        full_support_cells=full_support_cells,
        support_mean_frac=round(support_mean, 6),
        support_histogram=hist,
        feature_valid_px=feature_valid_px,
        inside_aoi_px=inside_aoi_px,
        errors=errors,
    )
    findings.extend(f"{scene_id}: {e}" for e in errors)
    return metrics, profiles, findings


# ── report serialisation ─────────────────────────────────────────────


def _summary_dict(report: Stage2Report) -> dict:
    return {
        "pipeline": "qa-stage2-features",
        "run_id": report.run_id,
        "timestamp": report.timestamp,
        "ok": report.ok,
        "grid": report.grid,
        "inputs": report.inputs,
        "fingerprints": report.fingerprints,
        "scenes": {
            "total_pairings": report.total_pairings,
            "assessed": report.assessed,
            "excluded": report.excluded,
            "exclusion_reasons": report.exclusion_reasons,
        },
        "aggregate": report.aggregate,
        "findings": report.findings,
    }


def _scene_rows(report: Stage2Report) -> list[dict]:
    rows = []
    for s in report.scenes:
        frac = round(s.feature_valid_px / s.inside_aoi_px, 6) if s.inside_aoi_px else 0.0
        rows.append(
            {
                "scene_id": s.scene_id,
                "year": s.year,
                "s2_scene_id": s.s2_scene_id,
                "geometry_id": s.geometry_id,
                "assessed": s.assessed,
                "exclusion_reason": s.exclusion_reason,
                "target_valid_cells": s.target_valid_cells,
                "all_100_cells": s.all_100_cells,
                "full_support_cells": s.full_support_cells,
                "support_mean_frac": s.support_mean_frac,
                "support_histogram": json_dumps(s.support_histogram),
                "feature_valid_px": s.feature_valid_px,
                "inside_aoi_px": s.inside_aoi_px,
                "feature_valid_frac_of_aoi": frac,
                "errors": json_dumps(s.errors),
            }
        )
    return rows


_SCENE_FIELDNAMES = [
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
]


def _profile_rows(report: Stage2Report) -> list[dict]:
    return [
        {
            "scene_id": p.scene_id,
            "channel_index": p.channel_index,
            "channel_name": p.channel_name,
            "family": p.family,
            "unit": p.unit,
            "valid_px": p.valid_px,
            "min": p.min,
            "max": p.max,
            "mean": p.mean,
            "std": p.std,
            "histogram": json_dumps(p.histogram),
        }
        for p in report.profiles
    ]


_PROFILE_FIELDNAMES = [
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
]


def _write_table(rows: list[dict], parquet_uri: str, csv_uri: str, fieldnames: list[str]) -> None:
    table = pa.Table.from_pylist(rows)
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    atomic_write(parquet_uri, buf.getvalue().to_pybytes(), overwrite=True, if_generation_match=0)
    csv_buf = _io.StringIO()
    writer = _csv.DictWriter(csv_buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(csv_uri, csv_buf.getvalue().encode("utf-8"), overwrite=True, if_generation_match=0)


def write_report(report: Stage2Report, output_root: str) -> dict[str, str]:
    """Persist summary.json, scenes.parquet/csv, profiles.parquet/csv."""
    prefix = f"{output_root.rstrip('/')}/{report.run_id}"
    uris: dict[str, str] = {}

    summary_uri = f"{prefix}/summary.json"
    atomic_write(
        summary_uri,
        json_dumps(_summary_dict(report)),
        overwrite=True,
        if_generation_match=0,
    )
    uris["summary"] = summary_uri

    scene_rows = _scene_rows(report)
    scenes_parquet = f"{prefix}/scenes.parquet"
    scenes_csv = f"{prefix}/scenes.csv"
    _write_table(scene_rows, scenes_parquet, scenes_csv, _SCENE_FIELDNAMES)
    uris["scenes_parquet"] = scenes_parquet
    uris["scenes_csv"] = scenes_csv

    profile_rows = _profile_rows(report)
    profiles_parquet = f"{prefix}/profiles.parquet"
    profiles_csv = f"{prefix}/profiles.csv"
    _write_table(profile_rows, profiles_parquet, profiles_csv, _PROFILE_FIELDNAMES)
    uris["profiles_parquet"] = profiles_parquet
    uris["profiles_csv"] = profiles_csv

    return uris


def json_dumps(data) -> str:
    return json.dumps(data, indent=2, default=str)


# ── orchestration ────────────────────────────────────────────────────


def run_stage2_features(cfg, *, run_id: str) -> Stage2Report:
    """Run the Stage-2 feature-stack QA gate for the configured universe.

    ``cfg`` carries the Stage-1 inputs plus ``features_root``,
    ``expected_scene_count`` (invariant, skipped for smoke), ``profile_bins``
    (fixed histogram bin count), ``output_root``, optional ``bbox`` and
    ``scene_ids`` (smoke restriction).
    """
    timestamp = datetime.now(UTC).isoformat()

    manifest_uri = str(cfg.manifest_uri)
    ard_root = str(cfg.ard_root)
    static_sources_root = str(cfg.static_sources_root)
    static_derived_root = str(cfg.static_derived_root)
    dynamic_root = str(cfg.dynamic_root)
    geometry_mapping_uri = str(cfg.geometry_mapping_uri)
    features_root = str(cfg.features_root)
    expected_scene_count = int(cfg.get("expected_scene_count", 0) or 0)
    n_profile_bins = int(cfg.get("profile_bins", 16) or 16)
    bbox = tuple(cfg.get("bbox", None)) if cfg.get("bbox") else None
    scene_ids = [str(s) for s in cfg.get("scene_ids", []) or []]

    log_event(
        _logger,
        logging.INFO,
        "stage2_start",
        run_id=run_id,
        features_root=features_root,
    )

    inventory = build_inventory(
        manifest_uri=manifest_uri,
        ard_root=ard_root,
        static_sources_root=static_sources_root,
        static_derived_root=static_derived_root,
        dynamic_root=dynamic_root,
        geometry_mapping_uri=geometry_mapping_uri,
        scene_ids=scene_ids,
    )
    if not inventory.ok:
        log_event(_logger, logging.ERROR, "stage2_inventory_failed", errors=inventory.errors)
        # Fail closed: a broken inventory cannot produce trustworthy QA.
        raise RuntimeError(f"Stage-2 inventory failed: {inventory.errors}")

    # Features ledger fingerprint — the canonical root is the Stage-2 input.
    fingerprints = dict(inventory.fingerprints)
    fingerprints["features_ledger"] = sha256_bytes(read_bytes(ledger_path(features_root)))[:16]

    analysis_10 = analysis_grid_10m(bbox)
    analysis_100 = analysis_10.zoom_out(10)

    findings: list[str] = []
    scenes_out: list[SceneMetrics] = []
    profiles_out: list[ChannelProfile] = []
    agg_target = agg_all100 = agg_full = agg_valid_px = 0
    agg_hist: dict[str, int] = {}

    for scene in inventory.scenes:
        if not scene.assessable:
            reason = scene.exclusion_reason or "excluded"
            if reason not in _EXPECTED_EXCLUSIONS:
                findings.append(f"{scene.scene_id}: {reason}")
            scenes_out.append(
                SceneMetrics(
                    scene_id=scene.scene_id,
                    year=scene.year,
                    s2_scene_id=scene.s2_scene_id,
                    geometry_id=scene.geometry_id,
                    assessed=False,
                    exclusion_reason=reason,
                )
            )
            continue

        metrics, profiles, scene_findings = _scan_stack(
            scene, features_root, analysis_10, analysis_100, n_profile_bins
        )
        findings.extend(scene_findings)
        # Surface precise inventory-level diagnosis (e.g. missing COG/flag
        # path in the ARD ledger) alongside the scan findings.
        if scene.errors:
            findings.extend(f"{scene.scene_id}: {e}" for e in scene.errors)
        scenes_out.append(metrics)
        profiles_out.extend(profiles)

        if metrics.assessed:
            agg_target += metrics.target_valid_cells
            agg_all100 += metrics.all_100_cells
            agg_full += metrics.full_support_cells
            agg_valid_px += metrics.feature_valid_px
            for label, count in metrics.support_histogram.items():
                agg_hist[label] = agg_hist.get(label, 0) + count

    assessed_scenes = [s for s in scenes_out if s.assessed]
    if expected_scene_count and not scene_ids and len(assessed_scenes) != expected_scene_count:
        findings.append(
            f"expected {expected_scene_count} assessed scenes, got {len(assessed_scenes)}"
        )

    aggregate = {
        "analysis_grid": {
            "crs": _GRID_CRS,
            "res_10m": {"width": analysis_10.shape.x, "height": analysis_10.shape.y},
            "res_100m": {"width": analysis_100.shape.x, "height": analysis_100.shape.y},
            "origin": [analysis_10.transform.xoff, analysis_10.transform.yoff],
            "bbox_subset": bbox is not None,
        },
        "assessed_scenes": len(assessed_scenes),
        "feature_valid_px": agg_valid_px,
        "target_valid_cells": agg_target,
        "all_100_cells": agg_all100,
        "full_support_cells": agg_full,
        "support_histogram": dict(sorted(agg_hist.items())),
    }

    report = Stage2Report(
        run_id=run_id,
        timestamp=timestamp,
        grid={
            "crs": _GRID_CRS,
            "res_10m": [analysis_10.shape.x, analysis_10.shape.y],
            "res_100m": [analysis_100.shape.x, analysis_100.shape.y],
            "origin": [analysis_10.transform.xoff, analysis_10.transform.yoff],
        },
        inputs={
            "manifest_uri": manifest_uri,
            "ard_root": ard_root,
            "static_sources_root": static_sources_root,
            "static_derived_root": static_derived_root,
            "dynamic_root": dynamic_root,
            "geometry_mapping_uri": geometry_mapping_uri,
            "features_root": features_root,
        },
        fingerprints=fingerprints,
        total_pairings=inventory.total_pairings,
        assessed=inventory.assessed,
        excluded=inventory.excluded,
        exclusion_reasons=inventory.exclusion_reasons,
        aggregate=aggregate,
        findings=sorted(set(findings)),
        scenes=scenes_out,
        profiles=profiles_out,
    )

    log_event(
        _logger,
        logging.INFO,
        "stage2_done",
        run_id=run_id,
        assessed=inventory.assessed,
        excluded=inventory.excluded,
        findings=len(report.findings),
        ok=report.ok,
    )
    return report


__all__ = [
    "ChannelProfile",
    "SceneMetrics",
    "Stage2Report",
    "run_stage2_features",
    "write_report",
]