"""LoD2 source-coverage resolution for the feature stack.

The ``lod2_morphology`` source COGs write NaN for every cell without a
building footprint — including cells that are simply unbuilt. To
distinguish "no building" (a known zero) from "source data missing" (a
true gap) at feature-composition time, this module reconstructs the
per-vintage 1 km tile coverage from immutable published provenance:

- 2021 / 2022: tile keys parsed from the raw archive manifests
  (``raw_manifest_<vintage>.json`` member names).
- 2017: the product rasterises LoD2-2021 geometry filtered against
  LoD1-2017 footprints, so a cell is covered only when both tile sets
  cover it — coverage = LoD1-2017 tile keys ∩ LoD2-2021 tile keys.
- 2024: tile extents from the published LoD provenance
  (``source_metadata.tiles[].easting/northing``).

Finite source values always win over the coverage classification: the
coverage mask is only consulted where all four LoD bands are NaN.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np
from odc.geo.geobox import GeoBox

from berlin_lst_downscaling.common.util import sha256_bytes
from berlin_lst_downscaling.data.io import read_bytes

# Vintages the feature stack can consume (2017/2021/2022 historical
# archives, 2024 ATOM-feed product).
COVERAGE_VINTAGES: tuple[int, ...] = (2017, 2021, 2022, 2024)

# Expected member/tile counts per coverage artifact. The 2017 archive is
# the LoD1-2017 stock; the LoD2 geometry for 2017 comes from the 2021
# archive (see docs/data-sources-and-contracts.md historical vintages).
_EXPECTED_MEMBERS: dict[int, int] = {2017: 1006, 2021: 928, 2022: 928}
_EXPECTED_TILES_2024 = 923

_LOD1_TILE_RE = re.compile(r"LoD1_(\d{3})_(\d{4})_")
_LOD2_TILE_RE = re.compile(r"LoD2_(?:\d+_)?(\d{3})_(\d{4})_")


@dataclass(frozen=True)
class LoDCoverageArtifact:
    """Resolved immutable evidence for one vintage's tile coverage."""

    vintage: int
    uris: tuple[str, ...]  # artifact URIs hashed into ``fingerprint``
    fingerprint: str  # sha256[:16] over the artifact bytes (stable order)
    tile_keys: tuple[str, ...]  # sorted, de-duplicated "E_N" 1 km tile keys
    cog_uri: str  # published lod2_morphology source COG for this vintage
    cog_fingerprint: str  # sha256[:16] over the COG bytes (content identity)


def _tile_key(member: str, *, lod1: bool) -> str:
    """Return the "E_N" tile key of a LoD1/LoD2 archive member name."""
    rx = _LOD1_TILE_RE if lod1 else _LOD2_TILE_RE
    m = rx.search(member)
    if not m:
        raise ValueError(f"cannot parse tile key from member {member!r}")
    return f"{m.group(1)}_{m.group(2)}"


def _manifest_members(uri: str) -> tuple[str, ...]:
    """Return the sorted member names of a published raw archive manifest."""
    payload = json.loads(read_bytes(uri))
    members = payload.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"raw archive manifest {uri} has no members list")
    return tuple(sorted(str(x) for x in members))


def _fingerprint(uris: tuple[str, ...]) -> str:
    """Stable short SHA-256 over the artifact bytes in *uris* order."""
    return sha256_bytes(b"\0".join(read_bytes(u) for u in uris))[:16]


def _cog_fingerprint(uri: str) -> str:
    """Streamed short SHA-256 over the published LoD COG bytes.

    Content identity of the actual source raster: an in-place COG
    replacement with unchanged manifests/provenance would otherwise go
    undetected by the feature config hash (the static-source ledger rows
    carry no checksum).
    """
    if uri.startswith("gs://"):
        from berlin_lst_downscaling.data.io.storage import _gcs_client, _parse_gs_uri

        bucket_name, key = _parse_gs_uri(uri)
        blob = _gcs_client().bucket(bucket_name).blob(key)
        h = hashlib.sha256()
        with blob.open("rb") as f:
            while True:
                chunk = f.read(1 << 16)
                if not chunk:
                    break
                h.update(chunk.encode() if isinstance(chunk, str) else chunk)
        return h.hexdigest()[:16]
    h = hashlib.sha256()
    with open(uri, "rb") as f:
        while True:
            chunk = f.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def _cog_uri(root: str, vintage: int) -> str:
    return f"{root}/ard/static/sources/lod2_morphology/{vintage}/lod2_morphology_{vintage}.tif"


def resolve_lod_coverage_artifacts(static_sources_root: str) -> dict[int, LoDCoverageArtifact]:
    """Resolve the per-vintage LoD coverage artifacts from published evidence.

    Raises ``ValueError`` on missing/malformed artifacts or unexpected
    member/tile counts — a run must never proceed on guessed coverage.
    """
    root = static_sources_root.rstrip("/")
    out: dict[int, LoDCoverageArtifact] = {}

    # ── 2021 / 2022: raw archive manifests ────────────────────────────
    for vintage in (2021, 2022):
        uri = f"{root}/ard/static/sources/lod_vintages/raw_manifest_{vintage}.json"
        members = _manifest_members(uri)
        if len(members) != _EXPECTED_MEMBERS[vintage]:
            raise ValueError(
                f"raw manifest {uri}: {len(members)} members, "
                f"expected {_EXPECTED_MEMBERS[vintage]}"
            )
        keys = tuple(sorted({_tile_key(m, lod1=False) for m in members}))
        if len(keys) != _EXPECTED_MEMBERS[vintage]:
            raise ValueError(f"raw manifest {uri}: {len(keys)} unique tile keys")
        cog = _cog_uri(root, vintage)
        out[vintage] = LoDCoverageArtifact(
            vintage=vintage,
            uris=(uri,),
            fingerprint=_fingerprint((uri,)),
            tile_keys=keys,
            cog_uri=cog,
            cog_fingerprint=_cog_fingerprint(cog),
        )

    # ── 2017: LoD1-2017 stock ∩ LoD2-2021 geometry ────────────────────
    lod1_uri = f"{root}/ard/static/sources/lod_vintages/raw_manifest_2017.json"
    lod1_members = _manifest_members(lod1_uri)
    if len(lod1_members) != _EXPECTED_MEMBERS[2017]:
        raise ValueError(
            f"raw manifest {lod1_uri}: {len(lod1_members)} members, "
            f"expected {_EXPECTED_MEMBERS[2017]}"
        )
    lod1_keys = {_tile_key(m, lod1=True) for m in lod1_members}
    lod2_2021 = out[2021]
    keys_2017 = tuple(sorted(lod1_keys & set(lod2_2021.tile_keys)))
    if not keys_2017:
        raise ValueError("2017 LoD coverage: empty LoD1 ∩ LoD2 tile intersection")
    cog_2017 = _cog_uri(root, 2017)
    out[2017] = LoDCoverageArtifact(
        vintage=2017,
        uris=(lod1_uri, lod2_2021.uris[0]),
        fingerprint=_fingerprint((lod1_uri, lod2_2021.uris[0])),
        tile_keys=keys_2017,
        cog_uri=cog_2017,
        cog_fingerprint=_cog_fingerprint(cog_2017),
    )

    # ── 2024: published LoD provenance tile receipts ──────────────────
    prov_uri = f"{root}/ard/static/sources/lod2_morphology/2024/provenance.json"
    prov = json.loads(read_bytes(prov_uri))
    tiles = (prov.get("source_metadata") or {}).get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise ValueError(f"LoD provenance {prov_uri} has no source_metadata.tiles")
    keys_2024 = []
    for tile in tiles:
        e = tile.get("easting")
        n = tile.get("northing")
        if not isinstance(e, int) or not isinstance(n, int):
            raise ValueError(f"LoD provenance {prov_uri}: tile missing easting/northing")
        keys_2024.append(f"{e // 1000}_{n // 1000}")
    keys_2024 = tuple(sorted(set(keys_2024)))
    if len(keys_2024) != _EXPECTED_TILES_2024:
        raise ValueError(
            f"LoD provenance {prov_uri}: {len(keys_2024)} tiles, "
            f"expected {_EXPECTED_TILES_2024}"
        )
    cog_2024 = _cog_uri(root, 2024)
    out[2024] = LoDCoverageArtifact(
        vintage=2024,
        uris=(prov_uri,),
        fingerprint=_fingerprint((prov_uri,)),
        tile_keys=keys_2024,
        cog_uri=cog_2024,
        cog_fingerprint=_cog_fingerprint(cog_2024),
    )

    return out


def rasterize_lod_coverage(artifact: LoDCoverageArtifact, grid: GeoBox) -> np.ndarray:
    """Return a bool mask (True = covered by a source tile) on *grid*.

    A grid pixel is covered when its centre falls inside a listed 1 km
    tile — the same centre-in-geometry convention the LoD rasterizer uses.
    """
    key_codes = np.array(
        [int(e) * 100_000 + int(n) for e, n in (k.split("_") for k in artifact.tile_keys)],
        dtype=np.int64,
    )
    centers_x = grid.transform.xoff + 5.0 + np.arange(grid.shape.x) * 10.0
    centers_y = grid.transform.yoff - 5.0 - np.arange(grid.shape.y) * 10.0
    col_code = (centers_x // 1000.0).astype(np.int64) * 100_000
    row_code = (centers_y // 1000.0).astype(np.int64)
    code = col_code[None, :] + row_code[:, None]
    return np.isin(code, key_codes)


__all__ = [
    "COVERAGE_VINTAGES",
    "LoDCoverageArtifact",
    "rasterize_lod_coverage",
    "resolve_lod_coverage_artifacts",
]