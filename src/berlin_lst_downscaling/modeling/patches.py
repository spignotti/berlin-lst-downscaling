"""Real contract-conforming patch reader (WB3, issues #18/#19).

Reads the published WB3 patch index and resolves, per accepted window, the
four tensors fixed by ``docs/pseudo-pair-tensor-contract.md``:

- ``features`` (28 channels at 10 m, train-only scaled, finite),
- ``lst_prior`` (1000 m block-expanded native LST at 10 m),
- ``target_100m`` (native Landsat LST, Kelvin),
- ``mask_100m`` (``training_eligible@100m``).

Design decisions
----------------
- **One source of truth per input.** Patch windows, splits, and anchors come
  from the patch index; the 100 m eligibility mask from the ``training/v1``
  release; the Landsat target and flag from the ARD ledger. Nothing is
  re-derived from scene IDs where a published reference exists.
- **Global alignment.** Prior blocks are formed on the global canonical
  EPSG:25833 100 m lattice (groups of ten rows/columns), *before* a patch is
  cropped, so a block value is identical on both sides of a patch boundary.
  ``_canonical_offset`` mirrors ``data/training/index.py:_global_row_col``.
- **No silent substitutions.** Missing or duplicated ledger rows, grid
  misalignment, hash drift, and index/mask disagreements raise. A patch that
  legitimately cannot be built (no prior block, target outside the scene
  footprint, empty eligibility window) is recorded as an exclusion with a
  reason and reported, never patched over with a fabricated value.
- **Model input, not source data.** The train-only scaler is applied exactly
  as fitted and invalid predictor pixels are then neutralized to ``0`` for
  the model tensor. The published stack, the eligibility mask, and the
  native target are untouched.

The naive baseline consumes :class:`RealSample` directly (physical Kelvin
prior via ``lst_prior_k``), so both arms share this reader, the mask, and the
pooling in ``modeling/metrics.py``.
"""

from __future__ import annotations

import io
import json
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import rasterio
import torch
from odc.geo.geobox import GeoBox
from rasterio.windows import Window

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.features.contracts import FEATURE_CHANNEL_NAMES
from berlin_lst_downscaling.data.features.paths import feature_cog
from berlin_lst_downscaling.data.io import exists, read_bytes
from berlin_lst_downscaling.data.qa.contracts import LST_RANGE_K
from berlin_lst_downscaling.data.training.contracts import (
    CANON_GRID_ORIGIN_X,
    CANON_GRID_ORIGIN_Y,
    CELL_SIZE_M,
    split_for_year,
    training_policy_hash,
)
from berlin_lst_downscaling.data.training.patch_index import (
    PATCH_CELLS,
    PATCH_SIZE_10M,
)
from berlin_lst_downscaling.data.training.paths import (
    manifest_parquet,
    patch_index_completion,
    patch_index_parquet,
    scaler_json,
)
from berlin_lst_downscaling.modeling.contracts import (
    N_FEATURE_CHANNELS,
    NO_PRIOR_REASON,
    PRIOR_AFFINE_OFFSET_K,
    PRIOR_AFFINE_SCALE_K,
    PRIOR_BLOCK_CELLS,
    PRIOR_BLOCK_PX_10M,
    REAL_PATCH_CELLS,
    REAL_PATCH_PX,
    RealBatch,
    RealSampleMeta,
)

_logger = logging.getLogger(__name__)

# The index geometry and the contract geometry must be one thing.
if (PATCH_CELLS, PATCH_SIZE_10M) != (REAL_PATCH_CELLS, REAL_PATCH_PX):  # pragma: no cover
    raise AssertionError(
        f"patch geometry drift: index ({PATCH_CELLS}, {PATCH_SIZE_10M}) != "
        f"contract ({REAL_PATCH_CELLS}, {REAL_PATCH_PX})"
    )

_ARRAY_10M: int = PRIOR_BLOCK_PX_10M  # 1000 m at 10 m
_LANDSAT_SOURCE = "landsat-c2-l2"
_LANDSAT_RES_M = float(CELL_SIZE_M)

# Exclusion reasons (reported, never silently dropped).
NO_LANDSAT_REASON = "ard_landsat_unresolved"
NO_PRIOR_BLOCK_REASON = NO_PRIOR_REASON
PRIOR_OUTSIDE_REASON = "prior_window_outside_scene"
TARGET_OUTSIDE_REASON = "target_window_outside_scene"
EMPTY_MASK_REASON = "empty_eligibility_window"


# ── global canonical alignment ────────────────────────────────────────


def _canonical_offset(transform, res: float) -> tuple[int, int]:
    """Return ``(col0, row0)``: the global canonical index of pixel ``(0, 0)``.

    Mirrors ``data/training/index.py:_global_row_col`` so any raster written
    on a canonical-aligned grid maps to the same global cell identity.
    """
    col0 = round((transform.xoff - CANON_GRID_ORIGIN_X) / res)
    row0 = round((CANON_GRID_ORIGIN_Y - transform.yoff) / res)
    return col0, row0


# ── configuration ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class RealSourceConfig:
    """Published roots the reader resolves against."""

    patch_index_root: str
    training_root: str
    features_root: str
    ard_root: str


@dataclass(frozen=True)
class PatchRef:
    """One accepted window from the published patch index."""

    patch_id: str
    scene_id: str
    s2_scene_id: str
    split: str
    year: int
    row: int  # global canonical 100 m anchor row
    col: int  # global canonical 100 m anchor col
    n_eligible: int
    eligibility_mask: str


@dataclass(frozen=True)
class ScalerChannel:
    """Per-channel scaler statistics as published in ``scaler.json``."""

    index: int
    name: str
    transform: str
    mean: float | None
    std: float | None


@dataclass(frozen=True)
class ScalerSpec:
    """The published train-only scaler, applied at read time."""

    channel_order: tuple[str, ...]
    channels: tuple[ScalerChannel, ...]
    policy_hash: str

    def apply(self, bands: np.ndarray) -> np.ndarray:
        """Scale ``bands`` ``(28, H, W)`` in place and return it.

        Non-finite pixels are left non-finite (the boundary zero-fill owns
        them). A channel with a degenerate ``std`` of zero is left unscaled
        rather than producing infinities.
        """
        if bands.shape[0] != len(self.channels):
            raise ValueError(f"expected {len(self.channels)} bands, got {bands.shape[0]}")
        for i, ch in enumerate(self.channels):
            if ch.transform == "identity":
                continue
            band = bands[i]
            finite = np.isfinite(band)
            if not finite.any():
                continue
            vals = band[finite].astype(np.float64)
            if ch.transform == "log1p_zscore":
                vals = np.log1p(vals)
            if ch.mean is not None and ch.std:
                vals = (vals - ch.mean) / ch.std
            band[finite] = vals
        return bands


# ── loading ───────────────────────────────────────────────────────────


def _read_parquet(uri: str):
    return pq.read_table(io.BytesIO(read_bytes(uri)))


def load_scaler(training_root: str) -> ScalerSpec:
    """Read and validate the published train-only scaler.

    Raises on channel-order or policy-hash drift so a stale release cannot
    silently feed the model differently scaled inputs.
    """
    uri = scaler_json(training_root)
    payload = json.loads(read_bytes(uri))
    order = tuple(str(n) for n in payload.get("channel_order", ()))
    expected = training_policy_hash()
    policy_hash = str(payload.get("policy_hash", ""))
    if order != FEATURE_CHANNEL_NAMES:
        raise RuntimeError(f"scaler channel order drift at {uri}: {order}")
    if policy_hash != expected:
        raise RuntimeError(
            f"scaler policy hash {policy_hash!r} != current training policy {expected!r} "
            f"at {uri} — the published release is stale"
        )
    channels = tuple(
        ScalerChannel(
            index=int(c["channel_index"]),
            name=str(c["channel_name"]),
            transform=str(c["transform"]),
            mean=None if c.get("mean") is None else float(c["mean"]),
            std=None if c.get("std") is None else float(c["std"]),
        )
        for c in payload["channels"]
    )
    if len(channels) != N_FEATURE_CHANNELS:
        raise RuntimeError(
            f"scaler publishes {len(channels)} channels, expected {N_FEATURE_CHANNELS}"
        )
    return ScalerSpec(channel_order=order, channels=channels, policy_hash=policy_hash)


def load_patch_refs(
    cfg: RealSourceConfig,
    *,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    scene_ids: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> list[PatchRef]:
    """Read the published patch index and return refs for the given splits.

    Verifies the completion marker and the release policy hash, and checks
    that every row's ``split`` agrees with the temporal contract. Rows keep
    the index's deterministic ``(scene_id, row, col)`` order.
    """
    marker_uri = patch_index_completion(cfg.patch_index_root)
    if not exists(marker_uri):
        raise RuntimeError(
            f"patch index {cfg.patch_index_root!r} has no completion marker "
            f"({marker_uri}) — the index is incomplete or unpublished"
        )
    marker = json.loads(read_bytes(marker_uri))
    expected = training_policy_hash()
    if str(marker.get("source_policy_hash", "")) != expected:
        raise RuntimeError(
            f"patch index source policy hash {marker.get('source_policy_hash')!r} != "
            f"current training policy {expected!r} — the index is stale"
        )

    table = _read_parquet(patch_index_parquet(cfg.patch_index_root))
    total = int(marker.get("patches_total", -1))
    if total != table.num_rows:
        raise RuntimeError(
            f"patch index marker reports {total} patches but the table holds {table.num_rows}"
        )

    manifest = _read_manifest_rows(cfg.training_root)
    wanted = set(splits)
    restrict = set(scene_ids) if scene_ids else None
    refs: list[PatchRef] = []
    for r in table.to_pylist():
        year = int(r["year"])
        split = str(r["split"])
        if split_for_year(year) != split:
            raise RuntimeError(
                f"{r['scene_id']}: index split {split!r} contradicts the temporal "
                f"contract for year {year}"
            )
        if split not in wanted:
            continue
        scene_id = str(r["scene_id"])
        if restrict is not None and scene_id not in restrict:
            continue
        _verify_manifest_row(manifest, r)
        refs.append(
            PatchRef(
                patch_id=str(r["patch_id"]),
                scene_id=scene_id,
                s2_scene_id=str(r["s2_scene_id"]),
                split=split,
                year=year,
                row=int(r["row"]),
                col=int(r["col"]),
                n_eligible=int(r["n_eligible"]),
                eligibility_mask=str(r["eligibility_mask"]),
            )
        )
        if limit is not None and len(refs) >= limit:
            break
    return refs


def _read_manifest_rows(training_root: str) -> dict[str, dict]:
    """Read the ``training/v1`` scene manifest keyed by scene ID."""
    uri = manifest_parquet(training_root)
    if not exists(uri):
        raise RuntimeError(f"training release manifest missing: {uri}")
    table = _read_parquet(uri)
    rows: dict[str, dict] = {}
    for row in table.to_pylist():
        scene_id = str(row["scene_id"])
        if scene_id in rows:
            raise RuntimeError(f"duplicate manifest row for scene {scene_id!r} in {uri}")
        rows[scene_id] = row
    return rows


def _verify_manifest_row(manifest: dict[str, dict], row: dict) -> None:
    """Check one index row against its ``training/v1`` manifest row.

    The index derives its splits and mask references from the manifest, so a
    disagreement means the two published artifacts have drifted apart.
    """
    scene_id = str(row["scene_id"])
    entry = manifest.get(scene_id)
    if entry is None:
        raise RuntimeError(f"{scene_id}: patch index row has no manifest entry")
    if int(entry["year"]) != int(row["year"]) or str(entry["split"]) != str(row["split"]):
        raise RuntimeError(
            f"{scene_id}: index ({int(row['year'])}, {row['split']!r}) disagrees with the "
            f"manifest ({int(entry['year'])}, {entry['split']!r})"
        )
    manifest_mask = str(entry.get("eligibility_mask") or "")
    if manifest_mask and manifest_mask != str(row["eligibility_mask"]):
        raise RuntimeError(
            f"{scene_id}: index eligibility mask {row['eligibility_mask']!r} != manifest "
            f"{manifest_mask!r}"
        )


def load_landsat_uris(ard_root: str, scene_ids: set[str]) -> dict[str, tuple[str, str]]:
    """Resolve ``scene_id -> (landsat_cog, landsat_flag)`` from the ARD ledger.

    Only ``status == "done"`` Landsat rows count. Missing rows are absent
    from the result (the caller records the exclusion); duplicated rows are
    a hard error, never a silent last-wins.
    """
    uri = f"{ard_root.rstrip('/')}/ledger.parquet"
    cols = _read_parquet(uri).to_pydict()
    sources = cols["source"]
    scene_col = cols["scene_id"]
    status_col = cols["status"]
    cog_col = cols.get("path_cog", [None] * len(sources))
    flag_col = cols.get("path_flag", [None] * len(sources))
    out: dict[str, tuple[str, str]] = {}
    seen: set[str] = set()
    for i, source in enumerate(sources):
        if str(source) != _LANDSAT_SOURCE:
            continue
        scene_id = str(scene_col[i])
        if scene_id not in scene_ids:
            continue
        if str(status_col[i]) != "done":
            continue
        if scene_id in seen:
            raise RuntimeError(f"duplicate {_LANDSAT_SOURCE} ledger row for scene {scene_id!r}")
        cog = str(cog_col[i] or "")
        flag = str(flag_col[i] or "")
        if not cog or not flag:
            raise RuntimeError(f"landsat {scene_id}: missing COG/flag path in the ARD ledger")
        seen.add(scene_id)
        out[scene_id] = (cog, flag)
    return out


# ── per-scene prior ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ScenePrior:
    """A scene's 1000 m native-valid mean LST grid on global block indices."""

    scene_id: str
    landsat_cog: str
    landsat_flag: str
    col0: int  # global canonical 100 m col of Landsat pixel (0, 0)
    row0: int
    height: int
    width: int
    block_col0: int
    block_row0: int
    block_means: np.ndarray  # (n_blocks_row, n_blocks_col) float32, NaN if unavailable
    native_valid_cells: int

    def _window_indices(
        self, row: int, col: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(rows, cols, ri, ci)``: block indices for the patch window."""
        rows = (row * 10 + np.arange(REAL_PATCH_PX)) // _ARRAY_10M
        cols = (col * 10 + np.arange(REAL_PATCH_PX)) // _ARRAY_10M
        return rows, cols, rows - self.block_row0, cols - self.block_col0

    def window(self, *, row: int, col: int) -> np.ndarray | None:
        """Return the 160x160 10 m prior for the patch at global 100 m anchor.

        ``None`` when any required 1000 m block carries no native-valid cell
        or lies wholly outside the scene footprint; ``unavailable_reason``
        reports which.
        """
        _, _, ri, ci = self._window_indices(row, col)
        if (
            ri.min() < 0
            or ci.min() < 0
            or ri.max() >= self.block_means.shape[0]
            or ci.max() >= self.block_means.shape[1]
        ):
            return None
        values = self.block_means[np.ix_(ri, ci)]
        if not np.isfinite(values).all():
            return None
        return values

    def unavailable_reason(self, *, row: int, col: int) -> str:
        """Return why :meth:`window` is unavailable: footprint vs no observations."""
        _, _, ri, ci = self._window_indices(row, col)
        outside = (
            ri.min() < 0
            or ci.min() < 0
            or ri.max() >= self.block_means.shape[0]
            or ci.max() >= self.block_means.shape[1]
        )
        return PRIOR_OUTSIDE_REASON if outside else NO_PRIOR_BLOCK_REASON


def _build_scene_prior(scene_id: str, landsat_cog: str, landsat_flag: str) -> ScenePrior:
    """Build the scene-wide 1000 m block means from native Landsat pixels."""
    with rasterio.open(landsat_cog) as src:
        if str(src.crs) != "EPSG:25833":
            raise RuntimeError(f"{scene_id}: Landsat COG CRS is {src.crs!r}, expected EPSG:25833")
        if not np.isclose(src.transform.a, _LANDSAT_RES_M) or not np.isclose(
            abs(src.transform.e), _LANDSAT_RES_M
        ):
            raise RuntimeError(f"{scene_id}: Landsat COG resolution is not {_LANDSAT_RES_M} m")
        if src.transform.b or src.transform.d:
            raise RuntimeError(f"{scene_id}: Landsat COG transform is rotated")
        st = src.read(1).astype(np.float32)
        col0, row0 = _canonical_offset(src.transform, _LANDSAT_RES_M)
        height, width = src.height, src.width
    with rasterio.open(landsat_flag) as fsrc:
        if (fsrc.height, fsrc.width) != (height, width):
            raise RuntimeError(f"{scene_id}: Landsat flag shape differs from the LST COG")
        if _canonical_offset(fsrc.transform, _LANDSAT_RES_M) != (col0, row0):
            raise RuntimeError(f"{scene_id}: Landsat flag grid differs from the LST COG")
        flag = fsrc.read(1)

    # native-valid = the same expression as training_eligible@100m target validity.
    valid = np.isfinite(st) & (flag == 0) & (st >= LST_RANGE_K[0]) & (st <= LST_RANGE_K[1])

    global_rows = np.arange(height, dtype=np.int64)[:, None] + row0
    global_cols = np.arange(width, dtype=np.int64)[None, :] + col0
    brow = global_rows // PRIOR_BLOCK_CELLS
    bcol = global_cols // PRIOR_BLOCK_CELLS
    block_row0 = int(brow.min())
    block_col0 = int(bcol.min())
    ri = (brow - block_row0).astype(np.int64)
    ci = (bcol - block_col0).astype(np.int64)
    nbr = int(ri.max()) + 1
    nbc = int(ci.max()) + 1

    flat = (ri * nbc + ci).ravel()
    sel = valid.ravel()
    counts = np.bincount(flat[sel], minlength=nbr * nbc)
    sums = np.bincount(flat[sel], weights=st.ravel()[sel].astype(np.float64), minlength=nbr * nbc)
    means = np.full(nbr * nbc, np.nan, dtype=np.float64)
    ok = counts > 0
    means[ok] = sums[ok] / counts[ok]

    return ScenePrior(
        scene_id=scene_id,
        landsat_cog=landsat_cog,
        landsat_flag=landsat_flag,
        col0=col0,
        row0=row0,
        height=height,
        width=width,
        block_col0=block_col0,
        block_row0=block_row0,
        block_means=means.reshape(nbr, nbc).astype(np.float32),
        native_valid_cells=int(valid.sum()),
    )


# ── samples ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RealSample:
    """One contract-conforming patch."""

    meta: RealSampleMeta
    features: np.ndarray  # (28, 160, 160) float32, scaled, finite
    lst_prior_k: np.ndarray  # (1, 160, 160) float32, physical Kelvin
    target_100m: np.ndarray  # (1, 16, 16) float32, native Kelvin
    mask_100m: np.ndarray  # (1, 16, 16) bool


def prior_model_channel(prior_k: np.ndarray) -> np.ndarray:
    """Map the physical Kelvin prior to the model's fixed affine channel."""
    return ((prior_k - PRIOR_AFFINE_OFFSET_K) / PRIOR_AFFINE_SCALE_K).astype(np.float32)


class RealPatchReader:
    """Resolve patch refs into :class:`RealSample` objects.

    Excluded patches are counted in :attr:`exclusions` by reason and skipped;
    misaligned grids, hash drift, and index/mask disagreements raise.
    """

    def __init__(
        self,
        cfg: RealSourceConfig,
        *,
        analysis_10: GeoBox | None = None,
        scaler: ScalerSpec | None = None,
    ) -> None:
        self.cfg = cfg
        self.analysis_10 = canon_grid_10m() if analysis_10 is None else analysis_10
        self.analysis_100 = self.analysis_10.zoom_out(10)
        self._dcol10, self._drow10 = _canonical_offset(self.analysis_10.transform, 10.0)
        self._dcol100, self._drow100 = _canonical_offset(self.analysis_100.transform, 100.0)
        self.scaler: ScalerSpec = load_scaler(cfg.training_root) if scaler is None else scaler
        self.exclusions: Counter[str] = Counter()
        self._landsat: dict[str, tuple[str, str]] = {}
        self._landsat_queried: set[str] = set()
        self._priors: dict[str, ScenePrior] = {}

    # ── scene resolution (cached) ─────────────────────────────────────

    @property
    def analysis_offsets(self) -> tuple[int, int, int, int]:
        """Return ``(col10, row10, col100, row100)`` canonical offsets.

        The global canonical index of the analysis grid's pixel ``(0, 0)`` at
        10 m and at 100 m. Public so an independent validator can re-derive
        the same alignment without reaching into private state.
        """
        return self._dcol10, self._drow10, self._dcol100, self._drow100

    def scene_prior(self, scene_id: str) -> ScenePrior | None:
        """Return the resolved scene prior, or ``None`` if the scene is unresolved."""
        return self._prior_for(scene_id)

    def _landsat_uris(self, scene_ids: set[str]) -> dict[str, tuple[str, str]]:
        """Return ledger URIs for the scenes, reading the ledger for new ones.

        Scenes already looked up are never re-queried, so a scene with no
        ``done`` Landsat row is remembered as absent rather than re-read.
        """
        missing = scene_ids - self._landsat_queried
        if missing:
            self._landsat.update(load_landsat_uris(self.cfg.ard_root, missing))
            self._landsat_queried |= missing
        return {sid: self._landsat[sid] for sid in scene_ids if sid in self._landsat}

    def preload(self, scene_ids: set[str]) -> None:
        """Resolve the Landsat COG/flag for every scene once, before iterating."""
        self._landsat_uris(scene_ids)

    def _prior_for(self, scene_id: str) -> ScenePrior | None:
        if scene_id in self._priors:
            return self._priors[scene_id]
        uris = self._landsat_uris({scene_id}).get(scene_id)
        if uris is None:
            return None
        prior = _build_scene_prior(scene_id, uris[0], uris[1])
        self._priors[scene_id] = prior
        return prior

    # ── reading ───────────────────────────────────────────────────────

    def read_patch(self, ref: PatchRef) -> RealSample | None:
        """Read one patch, or return ``None`` after recording an exclusion."""
        prior = self._prior_for(ref.scene_id)
        if prior is None:
            self.exclusions[NO_LANDSAT_REASON] += 1
            return None

        prior_k = prior.window(row=ref.row, col=ref.col)
        if prior_k is None:
            self.exclusions[prior.unavailable_reason(row=ref.row, col=ref.col)] += 1
            return None

        target, mask = self._read_target_and_mask(ref, prior)
        if target is None or mask is None:
            return None

        features, filled = self._read_features(ref)
        return RealSample(
            meta=RealSampleMeta(
                patch_id=ref.patch_id,
                scene_id=ref.scene_id,
                split=ref.split,
                year=ref.year,
                row=ref.row,
                col=ref.col,
                filled_feature_pixels=filled,
            ),
            features=features,
            lst_prior_k=prior_k[None].astype(np.float32),
            target_100m=target[None].astype(np.float32),
            mask_100m=mask[None],
        )

    def _read_target_and_mask(
        self, ref: PatchRef, prior: ScenePrior
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        r0 = ref.row - prior.row0
        c0 = ref.col - prior.col0
        outside = (
            r0 < 0
            or c0 < 0
            or r0 + REAL_PATCH_CELLS > prior.height
            or c0 + REAL_PATCH_CELLS > prior.width
        )
        if outside:
            self.exclusions[TARGET_OUTSIDE_REASON] += 1
            return None, None
        with rasterio.open(prior.landsat_cog) as src:
            window = Window.from_slices(
                (r0, r0 + REAL_PATCH_CELLS), (c0, c0 + REAL_PATCH_CELLS)
            )
            target = src.read(1, window=window).astype(np.float32)

        mask_path = ref.eligibility_mask
        with rasterio.open(mask_path) as msk:
            off = _canonical_offset(msk.transform, _LANDSAT_RES_M)
            if off != (self._dcol100, self._drow100):
                raise RuntimeError(
                    f"{ref.scene_id}: eligibility mask {mask_path} is not aligned with the "
                    f"analysis grid (offset {off} != {(self._dcol100, self._drow100)})"
                )
            mr0 = ref.row - self._drow100
            mc0 = ref.col - self._dcol100
            mwindow = Window.from_slices(
                (mr0, mr0 + REAL_PATCH_CELLS), (mc0, mc0 + REAL_PATCH_CELLS)
            )
            mask = msk.read(1, window=mwindow)
        if int(mask.max(initial=0)) > 1:
            raise RuntimeError(f"{ref.scene_id}: eligibility mask has values outside {{0, 1}}")
        n_mask = int((mask == 1).sum())
        if n_mask != ref.n_eligible:
            raise RuntimeError(
                f"{ref.patch_id}: index reports n_eligible={ref.n_eligible} but the "
                f"published mask holds {n_mask} — index/mask drift"
            )
        if n_mask == 0:
            self.exclusions[EMPTY_MASK_REASON] += 1
            return None, None
        return target, mask == 1

    def _read_features(self, ref: PatchRef) -> tuple[np.ndarray, int]:
        uri = feature_cog(self.cfg.features_root, ref.scene_id)
        r0 = ref.row * 10 - self._drow10
        c0 = ref.col * 10 - self._dcol10
        with rasterio.open(uri) as src:
            off = _canonical_offset(src.transform, 10.0)
            if off != (self._dcol10, self._drow10):
                raise RuntimeError(
                    f"{ref.scene_id}: feature stack {uri} is not aligned with the analysis "
                    f"grid (offset {off} != {(self._dcol10, self._drow10)})"
                )
            if src.count != N_FEATURE_CHANNELS:
                raise RuntimeError(
                    f"{ref.scene_id}: feature stack has {src.count} bands, "
                    f"expected {N_FEATURE_CHANNELS}"
                )
            outside = (
                r0 < 0
                or c0 < 0
                or r0 + REAL_PATCH_PX > src.height
                or c0 + REAL_PATCH_PX > src.width
            )
            if outside:
                raise RuntimeError(
                    f"{ref.patch_id}: feature window ({r0}, {c0}) is outside the stack "
                    f"({src.height}, {src.width})"
                )
            fwindow = Window.from_slices(
                (r0, r0 + REAL_PATCH_PX), (c0, c0 + REAL_PATCH_PX)
            )
            bands = src.read(window=fwindow).astype(np.float32)

        self.scaler.apply(bands)
        filled = int((~np.isfinite(bands)).sum())
        if filled:
            bands[~np.isfinite(bands)] = 0.0
        return bands, filled

    def iter_samples(self, refs: list[PatchRef]) -> Iterator[RealSample]:
        """Yield samples for the resolvable refs, skipping recorded exclusions."""
        self.preload({ref.scene_id for ref in refs})
        for ref in refs:
            sample = self.read_patch(ref)
            if sample is not None:
                yield sample


def collate_real_batch(samples: list[RealSample]) -> RealBatch:
    """Stack samples into a :class:`RealBatch` (prior normalized for the model)."""
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    features = torch.from_numpy(np.stack([s.features for s in samples]))
    prior = torch.from_numpy(
        np.stack([prior_model_channel(s.lst_prior_k) for s in samples])
    )
    target = torch.from_numpy(np.stack([s.target_100m for s in samples]))
    mask = torch.from_numpy(np.stack([s.mask_100m for s in samples]))
    return RealBatch(
        features=features,
        lst_prior=prior,
        target_100m=target,
        mask_100m=mask,
        metadata=[s.meta for s in samples],
    )


__all__ = [
    "EMPTY_MASK_REASON",
    "NO_LANDSAT_REASON",
    "NO_PRIOR_BLOCK_REASON",
    "PRIOR_OUTSIDE_REASON",
    "TARGET_OUTSIDE_REASON",
    "PatchRef",
    "RealPatchReader",
    "RealSample",
    "RealSourceConfig",
    "ScalerChannel",
    "ScalerSpec",
    "ScenePrior",
    "collate_real_batch",
    "load_landsat_uris",
    "load_patch_refs",
    "load_scaler",
    "prior_model_channel",
]
