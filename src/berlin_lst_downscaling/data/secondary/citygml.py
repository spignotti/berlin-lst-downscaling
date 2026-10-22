"""Reusable CityGML readers for static LoD morphology vintages.

Parses Berlin GDI CityGML 1.0 and 2.0 building footprints and heights
for both LoD1 (block-extruded) and LoD2 (ground surface + roof) tiles.
Designed for memory-bounded streaming of large XML members.

LoD2 height policy
------------------
A CityGML building is a tree: ``Building`` → ``consistsOfBuildingPart``
→ ``BuildingPart``. Berlin stores ``measuredHeight`` on the part when
the building is split. Each emitted unit is one part, or the building
itself when no part can be emitted.

Height for a unit, in order:

1. The unit's own direct ``measuredHeight`` child, when it is finite
   and greater than 0.
2. Else the vertical span of that unit's own coordinates
   (maximum Z − minimum Z), ignoring nested parts, when the span is
   greater than 1 mm.
3. Else, for a part only, the parent building's direct
   ``measuredHeight``. The parent's Z span is not copied onto a part.
4. Else the unit is excluded and counted. It is never stored as 0 m.

Footprints come from the unit's own ``GroundSurface`` polygons. When
any part is emitted, the parent footprint is not emitted as well, so
the parent's ground surface is not added on top of its parts.

LoD1
----
LoD1 building = ``measuredHeight`` + the lowest horizontal face of the
``lod1Solid`` block. LoD1 has no ``GroundSurface``; the footprint is
derived by scanning every wall/roof polygon for the face with the
lowest Z coordinate and taking its (x, y) ring.

The module returns plain dataclasses so downstream code can pick the
shape it needs without depending on the XML library.

Library search
--------------
PyPI packages ``citygml``, ``pycitygml``, and ``citygml-tools`` all
returned 404 on the project's Index client.  The Berlin LoD XML files
are simple 1.0 building models — stdlib ``xml.etree.ElementTree`` plus
``shapely`` and ``numpy`` cover every parse path.  Adopt stdlib +
existing geometry stack; reject a new dependency.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

_logger = logging.getLogger(__name__)

# CityGML namespaces — version detected from the document content.
_CITYGML_NS_MAP: dict[int, dict[str, str]] = {
    1: {
        "gml": "http://www.opengis.net/gml",
        "bldg": "http://www.opengis.net/citygml/building/1.0",
        "core": "http://www.opengis.net/citygml/1.0",
    },
    2: {
        "gml": "http://www.opengis.net/gml",
        "bldg": "http://www.opengis.net/citygml/building/2.0",
        "core": "http://www.opengis.net/citygml/2.0",
    },
}

# ── data classes ──────────────────────────────────────────────────────


@dataclass
class Building:
    """A single building or building part extracted from CityGML.

    ``measured_height`` is metres above ground. It is ``None`` only for
    the historical parser's rejected records; the current parser does
    not emit a unit whose height could not be resolved, and it never
    substitutes 0.
    """

    building_id: str
    footprint: Polygon | MultiPolygon | None
    measured_height: float | None
    height_source: str = "measured"  # measured | z_extent | inherited


@dataclass
class Lod2ParseStats:
    """Counters for one LoD2 parse. Missing heights are counted, not zeroed."""

    source_buildings: int = 0
    emitted: int = 0
    parts_emitted: int = 0
    dropped_missing_height: int = 0
    dropped_missing_footprint: int = 0
    height_from_measured: int = 0
    height_from_z_extent: int = 0
    height_inherited: int = 0

    def add(self, other: Lod2ParseStats) -> None:
        """Add *other*'s counters into this object."""
        self.source_buildings += other.source_buildings
        self.emitted += other.emitted
        self.parts_emitted += other.parts_emitted
        self.dropped_missing_height += other.dropped_missing_height
        self.dropped_missing_footprint += other.dropped_missing_footprint
        self.height_from_measured += other.height_from_measured
        self.height_from_z_extent += other.height_from_z_extent
        self.height_inherited += other.height_inherited

    def as_dict(self) -> dict[str, int]:
        """Return the counters as a plain dictionary."""
        return {key: int(value) for key, value in asdict(self).items()}


@dataclass
class Footprint:
    """A LoD1 footprint (lowest horizontal face of the block)."""

    building_id: str
    footprint: Polygon | MultiPolygon | None


# ── namespace + version detection ──────────────────────────────────────


def detect_citygml_version(content: bytes) -> int:
    """Detect CityGML version from document content bytes."""
    if b"citygml/building/2.0" in content:
        return 2
    if b"citygml/building/1.0" in content:
        return 1
    if b"core:CityModel" in content:
        if b"/1.0" in content:
            return 1
        if b"/2.0" in content:
            return 2
    return 1  # default


# ── GML polygon helpers ──────────────────────────────────────────────


def _parse_gml_ring(coords_text: str, srs_dim: int = 2) -> list[tuple[float, float]]:
    """Parse a GML ``posList`` into (x, y) tuples (3D also supported)."""
    tokens = coords_text.strip().split()
    coords: list[tuple[float, float]] = []
    step = srs_dim
    for i in range(0, len(tokens) - step + 1, step):
        try:
            x, y = float(tokens[i]), float(tokens[i + 1])
            coords.append((x, y))
        except (ValueError, IndexError):
            continue
    return coords


def _parse_gml_ring_3d(coords_text: str, srs_dim: int = 3) -> list[tuple[float, float, float]]:
    """Parse a GML ``posList`` into (x, y, z) tuples."""
    tokens = coords_text.strip().split()
    coords: list[tuple[float, float, float]] = []
    step = srs_dim
    for i in range(0, len(tokens) - step + 1, step):
        try:
            x, y, z = float(tokens[i]), float(tokens[i + 1]), float(tokens[i + 2])
            coords.append((x, y, z))
        except (ValueError, IndexError):
            continue
    return coords


def _parse_polygon(polygon_elem: ET.Element, ns: dict[str, str]) -> Polygon | None:
    """Parse a ``gml:Polygon`` (exterior + holes) into a Shapely Polygon."""
    gml_ns = ns["gml"]

    ext_ring = polygon_elem.find(f".//{{{gml_ns}}}exterior//{{{gml_ns}}}posList")
    if ext_ring is None or not ext_ring.text:
        return None

    srs_dim = _resolve_srs_dimension(ext_ring)
    ext_coords = _parse_gml_ring(ext_ring.text, srs_dim)
    if len(ext_coords) < 4:
        return None

    holes: list[list[tuple[float, float]]] = []
    for interior in polygon_elem.findall(f".//{{{gml_ns}}}interior"):
        pos_list = interior.find(f".//{{{gml_ns}}}posList")
        if pos_list is not None and pos_list.text:
            srs_dim_i = _resolve_srs_dimension(pos_list)
            hole_coords = _parse_gml_ring(pos_list.text, srs_dim_i)
            if len(hole_coords) >= 4:
                holes.append(hole_coords)

    try:
        poly = Polygon(ext_coords, holes)
        if poly.is_valid and not poly.is_empty:
            return poly
        poly = poly.buffer(0)
        if poly.is_valid and not poly.is_empty and isinstance(poly, Polygon):
            return poly
    except Exception:
        return None
    return None


def _resolve_srs_dimension(pos_list: ET.Element) -> int:
    """Return the srsDimension to use for a ``posList``.

    Berlin LoD CityGML files commonly omit ``srsDimension`` even though
    every coordinate is 3D; default to 3 in that case so polygons stay
    consistent with the rest of the file.
    """
    raw = pos_list.get("srsDimension")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            return 3
    return 3


def _polygon_z_extent(polygon_elem: ET.Element, ns: dict[str, str]) -> tuple[float, float] | None:
    """Return ``(min_z, max_z)`` across all vertices of *polygon_elem*."""
    gml_ns = ns["gml"]
    pos_lists = polygon_elem.findall(f".//{{{gml_ns}}}posList")
    if not pos_lists:
        return None
    z_values: list[float] = []
    for pl in pos_lists:
        if not pl.text:
            continue
        # Berlin LoD1 CityGML blocks use 3D posList without an explicit
        # ``srsDimension`` attribute; default to 3 so we still recover Z.
        srs_dim_attr = pl.get("srsDimension")
        srs_dim = int(srs_dim_attr) if srs_dim_attr else 3
        if srs_dim != 3:
            continue
        coords = _parse_gml_ring_3d(pl.text, srs_dim)
        for _, _, z in coords:
            z_values.append(z)
    if not z_values:
        return None
    return min(z_values), max(z_values)


# ── LoD2 building extractor ──────────────────────────────────────────


def _gml_id(elem: ET.Element, ns: dict[str, str]) -> str:
    """Return the ``gml:id`` of *elem*, or an empty string."""
    return elem.get(f"{{{ns.get('gml', '')}}}id", elem.get("gml:id", "")) or ""


def _is_under(node: ET.Element, ancestors: list[ET.Element]) -> bool:
    """True when *node* is *ancestor* or a descendant of one of them."""
    for ancestor in ancestors:
        for found in ancestor.iter():
            if found is node:
                return True
    return False


def _direct_building_parts(elem: ET.Element, ns: dict[str, str]) -> list[ET.Element]:
    """Return ``BuildingPart`` children that are not nested in another part."""
    part_tag = f"{{{ns['bldg']}}}BuildingPart"
    parts = [part for part in elem.iter(part_tag) if part is not elem]
    nested: set[int] = set()
    for part in parts:
        for inner in part.iter(part_tag):
            if inner is not part:
                nested.add(id(inner))
    return [part for part in parts if id(part) not in nested]


def _direct_measured_height(elem: ET.Element, ns: dict[str, str]) -> float | None:
    """Return a direct ``measuredHeight`` child when it is finite and > 0.

    ``find`` does not descend into ``BuildingPart``. A missing, zero, or
    negative value returns ``None`` so the caller can try the next
    fallback. Zero is not a height.
    """
    for tag in (
        f"{{{ns['bldg']}}}measuredHeight",
        "bldg:measuredHeight",
    ):
        height_elem = elem.find(tag)
        if height_elem is None or not height_elem.text:
            continue
        try:
            height = float(height_elem.text.strip())
        except ValueError:
            continue
        if np.isfinite(height) and height > 0.0:
            return float(height)
    return None


def _own_z_extent(elem: ET.Element, ns: dict[str, str]) -> float | None:
    """Return max Z − min Z of *elem*, excluding nested building parts.

    A flat ring (span ≤ 1 mm) is not a height. Absolute ellipsoidal Z
    cancels in the difference, which is the building height.
    """
    part_tag = f"{{{ns['bldg']}}}BuildingPart"
    parts = [part for part in elem.iter(part_tag) if part is not elem]
    z_values: list[float] = []
    for pos_list in elem.iter(f"{{{ns['gml']}}}posList"):
        if parts and _is_under(pos_list, parts):
            continue
        if not pos_list.text:
            continue
        srs_dim = _resolve_srs_dimension(pos_list)
        if srs_dim != 3:
            continue
        for _, _, z_value in _parse_gml_ring_3d(pos_list.text, srs_dim):
            if np.isfinite(z_value):
                z_values.append(z_value)
    if len(z_values) < 2:
        return None
    span = max(z_values) - min(z_values)
    if span <= 1e-3:
        return None
    return float(span)


def _own_ground_polygons(elem: ET.Element, ns: dict[str, str]) -> list[Polygon]:
    """Return ``GroundSurface`` polygons that belong to *elem*, not its parts."""
    part_tag = f"{{{ns['bldg']}}}BuildingPart"
    parts = [part for part in elem.iter(part_tag) if part is not elem]
    out: list[Polygon] = []
    ground_tag = f"{{{ns['bldg']}}}GroundSurface"
    polygon_tag = f"{{{ns['gml']}}}Polygon"
    for ground in elem.iter(ground_tag):
        if parts and _is_under(ground, parts):
            continue
        for poly_elem in ground.iter(polygon_tag):
            poly = _parse_polygon(poly_elem, ns)
            if poly is not None:
                out.append(poly)
    return out


def _merge_footprint(polygons: list[Polygon]) -> Polygon | MultiPolygon | None:
    """Union ground polygons into one footprint, or return None."""
    if not polygons:
        return None
    try:
        merged = unary_union(polygons)
    except Exception:
        return None
    if not isinstance(merged, (Polygon, MultiPolygon)):
        return None
    if merged.is_empty:
        return None
    if not merged.is_valid:
        merged = merged.buffer(0)
        if (
            not isinstance(merged, (Polygon, MultiPolygon))
            or merged.is_empty
            or not merged.is_valid
        ):
            return None
    return merged


def _resolve_height(
    elem: ET.Element,
    ns: dict[str, str],
    parent_measured: float | None,
) -> tuple[float | None, str | None]:
    """Resolve one unit's height. Never returns 0."""
    measured = _direct_measured_height(elem, ns)
    if measured is not None:
        return measured, "measured"
    z_span = _own_z_extent(elem, ns)
    if z_span is not None:
        return z_span, "z_extent"
    if parent_measured is not None and np.isfinite(parent_measured) and parent_measured > 0.0:
        return float(parent_measured), "inherited"
    return None, None


def _record_height_source(stats: Lod2ParseStats, source: str) -> None:
    if source == "measured":
        stats.height_from_measured += 1
    elif source == "z_extent":
        stats.height_from_z_extent += 1
    elif source == "inherited":
        stats.height_inherited += 1


def _emit_lod2_units(
    elem: ET.Element,
    ns: dict[str, str],
    stats: Lod2ParseStats,
    *,
    parent_measured: float | None,
) -> list[Building]:
    """Emit part units, or this element when no part can be emitted."""
    parts = _direct_building_parts(elem, ns)
    own_measured = _direct_measured_height(elem, ns)
    inherit_from = own_measured if own_measured is not None else parent_measured
    if parts:
        units: list[Building] = []
        for part in parts:
            units.extend(_emit_lod2_units(part, ns, stats, parent_measured=inherit_from))
        if units:
            return units

    height, source = _resolve_height(elem, ns, parent_measured)
    footprint = _merge_footprint(_own_ground_polygons(elem, ns))
    if footprint is None:
        stats.dropped_missing_footprint += 1
        return []
    if height is None or source is None:
        stats.dropped_missing_height += 1
        return []

    stats.emitted += 1
    if elem.tag == f"{{{ns['bldg']}}}BuildingPart":
        stats.parts_emitted += 1
    _record_height_source(stats, source)
    return [
        Building(
            building_id=_gml_id(elem, ns),
            footprint=footprint,
            measured_height=height,
            height_source=source,
        )
    ]


def _top_level_buildings(root: ET.Element, ns: dict[str, str]) -> list[ET.Element]:
    """Return ``bldg:Building`` elements that are not nested in another building."""
    building_tag = f"{{{ns['bldg']}}}Building"
    buildings = list(root.iter(building_tag))
    nested: set[int] = set()
    for building in buildings:
        for inner in building.iter(building_tag):
            if inner is not building:
                nested.add(id(inner))
    return [building for building in buildings if id(building) not in nested]


def parse_lod2_buildings_with_stats(
    xml_bytes: bytes,
) -> tuple[list[Building], Lod2ParseStats]:
    """Parse LoD2 buildings, including parts, and return drop counters.

    See the module docstring for the height and footprint policy.
    """
    stats = Lod2ParseStats()
    version = detect_citygml_version(xml_bytes)
    ns = _CITYGML_NS_MAP[version]
    try:
        root = ET.fromstring(xml_bytes)  # noqa: S314 — trusted CityGML
    except ET.ParseError as exc:
        _logger.warning("citygml parse error: %s", exc)
        return [], stats

    buildings: list[Building] = []
    for building_elem in _top_level_buildings(root, ns):
        stats.source_buildings += 1
        buildings.extend(_emit_lod2_units(building_elem, ns, stats, parent_measured=None))
    return buildings, stats


def describe_lod2_height_locations(xml_bytes: bytes) -> dict[str, int]:
    """Count where ``measuredHeight`` is stored, before the fallback.

    ``buildingpart_only_measured_height`` is the number of top-level
    buildings that have no direct positive ``measuredHeight`` and at
    least one ``BuildingPart`` that does. ``missing_measured_height``
    counts top-level buildings with no direct positive
    ``measuredHeight``, whether or not a part has one.
    ``parts_missing_measured_height`` counts parts with no direct
    positive ``measuredHeight``.
    """
    version = detect_citygml_version(xml_bytes)
    ns = _CITYGML_NS_MAP[version]
    try:
        root = ET.fromstring(xml_bytes)  # noqa: S314 — trusted CityGML
    except ET.ParseError as exc:
        _logger.warning("citygml parse error: %s", exc)
        return {
            "source_buildings": 0,
            "buildingpart_only_measured_height": 0,
            "missing_measured_height": 0,
            "parts_missing_measured_height": 0,
            "parts": 0,
        }

    part_tag = f"{{{ns['bldg']}}}BuildingPart"
    buildings = _top_level_buildings(root, ns)
    part_only = 0
    missing = 0
    parts_missing = 0
    part_count = 0
    for building in buildings:
        own = _direct_measured_height(building, ns)
        parts = [part for part in building.iter(part_tag)]
        part_count += len(parts)
        part_heights = sum(1 for part in parts if _direct_measured_height(part, ns) is not None)
        parts_missing += sum(1 for part in parts if _direct_measured_height(part, ns) is None)
        if own is None:
            missing += 1
            if part_heights > 0:
                part_only += 1
    return {
        "source_buildings": len(buildings),
        "buildingpart_only_measured_height": part_only,
        "missing_measured_height": missing,
        "parts_missing_measured_height": parts_missing,
        "parts": part_count,
    }


def parse_lod2_buildings(xml_bytes: bytes) -> list[Building]:
    """Parse every LoD2 building or building part from raw XML bytes.

    This is the current parser. Heights that live only on
    ``BuildingPart`` are kept. Missing heights are omitted, not stored
    as 0. Use :func:`parse_lod2_buildings_with_stats` for the counters
    and :func:`parse_lod2_buildings_legacy` to reproduce the previous
    behaviour for diagnostics.
    """
    buildings, _stats = parse_lod2_buildings_with_stats(xml_bytes)
    return buildings


def _legacy_extract_lod2_building(
    building_elem: ET.Element,
    ns: dict[str, str],
) -> Building | None:
    """Historical v5 extractor: direct ``measuredHeight`` on ``Building`` only.

    ``BuildingPart`` is ignored. A missing or non-positive
    ``measuredHeight`` drops the whole building, including part
    footprints. Kept so diagnostics can compare height distributions.
    """
    bid = _gml_id(building_elem, ns)
    height = None
    for tag in (
        f"{{{ns['bldg']}}}measuredHeight",
        "bldg:measuredHeight",
    ):
        height_elem = building_elem.find(tag)
        if height_elem is not None and height_elem.text:
            try:
                height = float(height_elem.text)
            except ValueError:
                continue
            break
    if height is None or height <= 0:
        return None

    ground_polys: list[Polygon] = []
    for ground in building_elem.iter(f"{{{ns['bldg']}}}GroundSurface"):
        for poly_elem in ground.iter(f"{{{ns['gml']}}}Polygon"):
            poly = _parse_polygon(poly_elem, ns)
            if poly is not None:
                ground_polys.append(poly)
    footprint = _merge_footprint(ground_polys)
    if footprint is None:
        return None
    return Building(
        building_id=bid,
        footprint=footprint,
        measured_height=height,
        height_source="measured",
    )


def parse_lod2_buildings_legacy(xml_bytes: bytes) -> list[Building]:
    """Parse with the v5 rules: no parts, no geometry fallback.

    Buildings whose only ``measuredHeight`` is on a ``BuildingPart`` are
    absent from the result. This function is for diagnostics. The
    morphology products use :func:`parse_lod2_buildings`.
    """
    version = detect_citygml_version(xml_bytes)
    ns = _CITYGML_NS_MAP[version]
    try:
        root = ET.fromstring(xml_bytes)  # noqa: S314 — trusted CityGML
    except ET.ParseError as exc:
        _logger.warning("citygml parse error: %s", exc)
        return []
    buildings: list[Building] = []
    for building_elem in root.iter(f"{{{ns['bldg']}}}Building"):
        building = _legacy_extract_lod2_building(building_elem, ns)
        if building is not None:
            buildings.append(building)
    return buildings


# ── LoD1 footprint extractor ────────────────────────────────────────


def _lod1_footprint(building_elem: ET.Element, ns: dict[str, str]) -> Polygon | MultiPolygon | None:
    """Extract the lowest horizontal face of a LoD1 building block.

    LoD1 buildings are extruded footprints with no ``GroundSurface`` — the
    footprint is recovered by scanning every polygon inside ``lod1Solid``
    for **horizontal** faces (all vertices at the same Z), then picking
    the face with the lowest Z.  Walls are vertical and span the full
    block height, so they are filtered out by the equal-Z check.
    """
    solids = list(building_elem.iter(f"{{{ns['bldg']}}}lod1Solid"))
    if not solids:
        return None

    gml_ns = ns["gml"]
    horizontal_faces: list[tuple[float, Polygon]] = []
    for solid_elem in solids:
        for polygon_elem in solid_elem.iter(f"{{{gml_ns}}}Polygon"):
            z_extent = _polygon_z_extent(polygon_elem, ns)
            if z_extent is None:
                continue
            min_z, max_z = z_extent
            # Tolerance covers benign float jitter but rejects walls
            # (which span a building's full height, typically several m).
            if (max_z - min_z) > 1e-3:
                continue
            poly = _parse_polygon(polygon_elem, ns)
            if poly is None:
                continue
            horizontal_faces.append((min_z, poly))

    if not horizontal_faces:
        return None

    horizontal_faces.sort(key=lambda pair: pair[0])
    base_z = horizontal_faces[0][0]
    base_faces: list[Polygon] = [poly for z, poly in horizontal_faces if abs(z - base_z) <= 1e-3]

    try:
        merged = unary_union(base_faces)
    except Exception:
        return None
    if isinstance(merged, (Polygon, MultiPolygon)):
        return merged
    return None


def _extract_lod1_footprint(
    building_elem: ET.Element,
    ns: dict[str, str],
) -> Footprint | None:
    """Extract a LoD1 footprint and building id."""
    bid = building_elem.get(
        f"{{{ns.get('gml', '')}}}id",
        building_elem.get("gml:id", ""),
    )
    footprint = _lod1_footprint(building_elem, ns)
    if footprint is None or footprint.is_empty or not footprint.is_valid:
        return None
    return Footprint(building_id=bid, footprint=footprint)


def parse_lod1_footprints(xml_bytes: bytes) -> list[Footprint]:
    """Parse every ``bldg:Building`` footprint from raw LoD1 XML bytes."""
    version = detect_citygml_version(xml_bytes)
    ns = _CITYGML_NS_MAP[version]

    try:
        root = ET.fromstring(xml_bytes)  # noqa: S314 — trusted XML
    except ET.ParseError as exc:
        _logger.warning("citygml parse error: %s", exc)
        return []

    out: list[Footprint] = []
    for building_elem in root.iter(f"{{{ns['bldg']}}}Building"):
        foot = _extract_lod1_footprint(building_elem, ns)
        if foot is not None:
            out.append(foot)
    return out


# ── ZIP and file path helpers ───────────────────────────────────────


_XML_RE = re.compile(r"\.(gml|xml)$", re.IGNORECASE)


def iter_xml_members(zip_path: str | Path) -> list[str]:
    """Return the names of every XML/GML member in a ZIP archive."""
    with zipfile.ZipFile(zip_path) as z:
        return [n for n in z.namelist() if _XML_RE.search(n)]


def read_xml_member(zip_path: str | Path, member: str) -> bytes:
    """Read a single XML/GML member from a ZIP archive (no full extract)."""
    with zipfile.ZipFile(zip_path) as z:
        with z.open(member) as f:
            return f.read()


def parse_lod2_buildings_from_file(path: str | Path) -> list[Building]:
    """Parse every ``bldg:Building`` from an XML/GML file path."""
    return parse_lod2_buildings(Path(path).read_bytes())


def parse_lod1_footprints_from_file(path: str | Path) -> list[Footprint]:
    """Parse every LoD1 footprint from an XML/GML file path."""
    return parse_lod1_footprints(Path(path).read_bytes())


def parse_lod2_buildings_from_zip_member(
    zip_path: str | Path,
    member: str,
) -> list[Building]:
    """Parse every ``bldg:Building`` from a ZIP archive member."""
    return parse_lod2_buildings(read_xml_member(zip_path, member))


def parse_lod1_footprints_from_zip_member(
    zip_path: str | Path,
    member: str,
) -> list[Footprint]:
    """Parse every LoD1 footprint from a ZIP archive member."""
    return parse_lod1_footprints(read_xml_member(zip_path, member))


# ── shared z helper (kept for legacy import paths) ──────────────────


def _z_extents(polygons: list[Polygon]) -> np.ndarray:
    """Return ``min_z`` per polygon — placeholder for future helpers."""
    del polygons  # placeholder; not used externally yet
    return np.array([], dtype=np.float64)


__all__ = [
    "Building",
    "Footprint",
    "Lod2ParseStats",
    "describe_lod2_height_locations",
    "detect_citygml_version",
    "iter_xml_members",
    "parse_lod1_footprints",
    "parse_lod1_footprints_from_file",
    "parse_lod1_footprints_from_zip_member",
    "parse_lod2_buildings",
    "parse_lod2_buildings_from_file",
    "parse_lod2_buildings_from_zip_member",
    "parse_lod2_buildings_legacy",
    "parse_lod2_buildings_with_stats",
    "read_xml_member",
]
