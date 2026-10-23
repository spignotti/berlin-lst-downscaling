"""Shared, deterministic Sentinel-2 mosaic discovery and membership contract.

Discovery is only for selection and legacy v3 ARD seeded by an exact item.
For reproducible selection/ARD pixels, build a fresh v4 bundle: ARD must
resolve every recorded ``s2_items`` href with its expected ID, never search
again. Membership is independent of processing baseline; ARD scales each
item with its own metadata before compositing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from shapely.geometry import box, shape

from berlin_lst_downscaling.data.acquisition.pc_client import get_catalog


def s2_mosaic_enabled(cfg: Any) -> bool:
    """Opt in only; absent/disabled mosaic configuration keeps legacy behavior."""
    return bool(cfg.get("sentinel2", {}).get("mosaic", {}).get("enabled", False))


def s2_mosaic_group(item: Any, *, bbox: Sequence[float]) -> dict[str, str]:
    """Identity at AOI-center solar time, excluding datatake baseline suffix.

    ESA datatake IDs end in ``_Nxx.xx`` (processing baseline), which is
    provenance rather than acquisition identity. All other components,
    including spacecraft, sensing time and orbit, remain significant.
    Missing acquisition identity fails closed; dates alone cannot group passes.
    """
    raw_dt = item.properties.get("datetime")
    if not raw_dt:
        raise ValueError(f"S2 item {item.id!r} missing datetime")
    dt = datetime.fromisoformat(str(raw_dt).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"S2 item {item.id!r} datetime must have a timezone")
    longitude = (bbox[0] + bbox[2]) / 2
    solar_day = (dt.astimezone(UTC) + timedelta(hours=longitude / 15)).date()
    datatake = item.properties.get("s2:datatake_id")
    if not isinstance(datatake, str) or not datatake.strip():
        raise ValueError(f"S2 item {item.id!r} missing s2:datatake_id")
    datatake = re.sub(r"_N\d{2}\.?\d{2}$", "", datatake.strip())
    return {"solar_day": solar_day.isoformat(), "datatake_id": datatake}


def s2_item_membership(items: Sequence[Any]) -> list[dict[str, str]]:
    """Return sorted exact references; reject missing or ambiguous identities."""
    refs: dict[str, str] = {}
    for item in items:
        href = item.get_self_href()
        if not isinstance(href, str) or not href.strip():
            raise ValueError(f"S2 item {item.id!r} missing item_href")
        if item.id in refs and refs[item.id] != href:
            raise ValueError(f"S2 item {item.id!r} has conflicting item_href values")
        refs[item.id] = href
    if not refs:
        raise ValueError("S2 mosaic membership is empty")
    return [{"item_id": sid, "item_href": refs[sid]} for sid in sorted(refs)]


def discover_s2_mosaic_items(
    seed_item: Any,
    *,
    bbox: Sequence[float],
    collection: str = "sentinel-2-l2a",
    catalog: Any = None,
) -> list[Any]:
    """Find ALL AOI-bbox-intersecting items in the seed's solar day/datatake.

    No tile list, item limit, cloud gate, baseline gate or single-tile fallback.
    The padded UTC search is post-filtered using exact geometry and identity.
    The exact seed is retained even if omitted from search; results are unique
    and sorted by item ID. Search/metadata errors propagate to the caller.
    A seed whose geometry misses the AOI bbox raises: callers iterating raw
    STAC bbox hits must skip such envelope-only seeds before calling.
    """
    group = s2_mosaic_group(seed_item, bbox=bbox)
    aoi = box(bbox[0], bbox[1], bbox[2], bbox[3])
    if not seed_item.geometry or not shape(seed_item.geometry).intersects(aoi):
        raise ValueError(f"S2 seed {seed_item.id!r} does not intersect AOI bbox")
    day = datetime.fromisoformat(group["solar_day"]).date()
    cat = catalog if catalog is not None else get_catalog()
    search = cat.search(
        collections=[collection],
        bbox=tuple(bbox),
        datetime=f"{day - timedelta(days=1)}/{day + timedelta(days=1)}",
    )
    members = {seed_item.id: seed_item}
    for item in search.items():
        if not item.geometry or not shape(item.geometry).intersects(aoi):
            continue
        if s2_mosaic_group(item, bbox=bbox) != group:
            continue
        if item.id == seed_item.id:
            # The caller's exact seed wins over its search copy.
            continue
        if item.id in members and item.get_self_href() != members[item.id].get_self_href():
            raise ValueError(f"S2 item {item.id!r} has conflicting item_href values")
        members[item.id] = item
    ordered = [members[sid] for sid in sorted(members)]
    s2_item_membership(ordered)
    return ordered


__all__ = [
    "discover_s2_mosaic_items",
    "s2_item_membership",
    "s2_mosaic_enabled",
    "s2_mosaic_group",
]
