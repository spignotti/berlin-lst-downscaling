# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Compare two feature-stack releases (V2 baseline vs V3 candidate).

Read-only probe over two feature-stack roots (local or ``gs://``) that
verifies the V3 semantic-availability correction:

- every V2-valid pixel (``feature_valid == 1``) must be V3-valid with
  **identical** 28-band values (bit-exact float32, NaN == NaN),
- every newly V3-valid pixel (V3 valid, V2 invalid) must have the four
  LoD channels (bands 11-14) equal to zero — the known no-building cells,
- reports per-scene and aggregate valid-pixel growth.

The candidate may be a canonical-aligned bbox subset of the baseline
(smoke runs): window offsets are derived from the two GeoBox transforms.

The script writes nothing. It imports only the channel contract names —
never the composer or pipeline implementation.

Usage
-----
    uv run python scripts/compare_feature_releases.py \\
        --baseline-root gs://berlin-lst-data/features/v2 \\
        --candidate-root gs://berlin-lst-data/features/v3
    uv run python scripts/compare_feature_releases.py \\
        --baseline-root data/smoke/features \\
        --candidate-root data/smoke/features
"""

from __future__ import annotations

import argparse
import io

import numpy as np
import pyarrow.parquet as pq
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.windows import Window

from berlin_lst_downscaling.data.features.contracts import FEATURE_CHANNEL_NAMES
from berlin_lst_downscaling.data.io import read_bytes

_LOD_BAND_SLICE = slice(10, 14)  # building_height_mean/std/coverage/max (0-based)
_TILE = 1024


def _read_table(uri: str):
    return pq.read_table(io.BytesIO(read_bytes(uri)))


def _ledger_scene_ids(root: str) -> list[str]:
    """Return done feature-stack scene ids from a release ledger."""
    ledger = f"{root.rstrip('/')}/_state/features/ledger.parquet"
    try:
        table = _read_table(ledger)
    except Exception as exc:
        raise SystemExit(f"cannot read ledger {ledger}: {exc}") from None
    cols = table.to_pydict()
    ids: list[str] = []
    for i in range(table.num_rows):
        if (
            str(cols["source"][i]) == "feature_stack"
            and str(cols["status"][i]) == "done"
        ):
            ids.append(str(cols["period_or_vintage"][i]))
    return sorted(set(ids))


def _scene_uris(root: str, scene_id: str) -> tuple[str, str]:
    """Return (cog_uri, mask_uri) for one scene in a release root."""
    base = f"{root.rstrip('/')}/{scene_id}"
    return f"{base}/{scene_id}.tif", f"{base}/{scene_id}.feature_valid.tif"


def compare_scene(
    scene_id: str,
    base_cog: str,
    base_mask: str,
    cand_cog: str,
    cand_mask: str,
    errors: list[str],
) -> dict:
    """Compare one scene pair; append findings to *errors*."""
    stats = {"v2_valid_px": 0, "v3_valid_px": 0, "new_valid_px": 0}
    with (
        rasterio.open(base_cog) as bc,
        rasterio.open(base_mask) as bm,
        rasterio.open(cand_cog) as cc,
        rasterio.open(cand_mask) as cm,
    ):
        bg = GeoBox.from_rio(bc)
        cg = GeoBox.from_rio(cc)
        col_off = round((cg.transform.xoff - bg.transform.xoff) / 10.0)
        row_off = round((bg.transform.yoff - cg.transform.yoff) / 10.0)
        if col_off < 0 or row_off < 0:
            errors.append(f"{scene_id}: candidate grid not inside baseline grid")
            return stats
        if cc.count != len(FEATURE_CHANNEL_NAMES):
            errors.append(f"{scene_id}: candidate band count {cc.count}")
            return stats

        h, w = cc.height, cc.width
        n_value_mismatch = 0
        n_bad_growth = 0
        for r0 in range(0, h, _TILE):
            r1 = min(r0 + _TILE, h)
            for c0 in range(0, w, _TILE):
                c1 = min(c0 + _TILE, w)
                cwin = Window(c0, r0, c1 - c0, r1 - r0)  # type: ignore[call-arg]
                bwin = Window(  # type: ignore[call-arg]
                    c0 + col_off, r0 + row_off, c1 - c0, r1 - r0
                )
                if (
                    bwin.col_off < 0
                    or bwin.row_off < 0
                    or bwin.col_off + bwin.width > bc.width
                    or bwin.row_off + bwin.height > bc.height
                ):
                    errors.append(f"{scene_id}: baseline window out of bounds")
                    return stats
                bdata = bc.read(window=bwin)
                bmask = bm.read(1, window=bwin)
                cdata = cc.read(window=cwin)
                cmask = cm.read(1, window=cwin)

                v2 = bmask == 1
                v3 = cmask == 1
                stats["v2_valid_px"] += int(np.sum(v2))
                stats["v3_valid_px"] += int(np.sum(v3))
                growth = v3 & ~v2
                stats["new_valid_px"] += int(np.sum(growth))

                # value identity at V2-valid pixels (NaN == NaN)
                eq = (cdata == bdata) | (np.isnan(cdata) & np.isnan(bdata))
                n_value_mismatch += int(np.sum(v2 & ~np.all(eq, axis=0)))

                # newly valid pixels must be known no-building cells
                lod_zero = np.all(cdata[_LOD_BAND_SLICE] == 0.0, axis=0)
                n_bad_growth += int(np.sum(growth & ~lod_zero))

        if n_value_mismatch:
            errors.append(
                f"{scene_id}: {n_value_mismatch} V2-valid px differ between releases"
            )
        if n_bad_growth:
            errors.append(
                f"{scene_id}: {n_bad_growth} newly valid px without zero LoD bands"
            )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline-root", required=True, help="V2 release root")
    parser.add_argument("--candidate-root", required=True, help="V3 release root")
    parser.add_argument(
        "--scene-ids",
        default=None,
        help="Comma-separated scene ids (default: all done scenes in the baseline ledger)",
    )
    args = parser.parse_args()

    if args.scene_ids:
        scene_ids = [s.strip() for s in args.scene_ids.split(",") if s.strip()]
    else:
        scene_ids = _ledger_scene_ids(args.baseline_root)
    if not scene_ids:
        print("No scenes to compare.")
        return 1

    errors: list[str] = []
    agg = {"v2_valid_px": 0, "v3_valid_px": 0, "new_valid_px": 0}
    for scene_id in scene_ids:
        b_cog, b_mask = _scene_uris(args.baseline_root, scene_id)
        c_cog, c_mask = _scene_uris(args.candidate_root, scene_id)
        stats = compare_scene(scene_id, b_cog, b_mask, c_cog, c_mask, errors)
        for k in agg:
            agg[k] += stats[k]
        print(
            f"{scene_id}: v2-valid {stats['v2_valid_px']:,} | "
            f"v3-valid {stats['v3_valid_px']:,} | new-valid {stats['new_valid_px']:,}"
        )

    if errors:
        print("FINDINGS:")
        for e in errors:
            print(f"  ✗ {e}")
        return 1
    print(
        f"OK: {len(scene_ids)} scenes — v2-valid {agg['v2_valid_px']:,} px unchanged, "
        f"v3-valid {agg['v3_valid_px']:,} px, new-valid {agg['new_valid_px']:,} px "
        "(LoD-zero)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())