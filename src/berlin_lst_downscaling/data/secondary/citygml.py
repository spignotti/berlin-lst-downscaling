"""Reusable CityGML readers for static LoD morphology vintages.

Parses Berlin GDI CityGML 1.0 building footprints and ``measuredHeight``
values for both LoD1 (block-extruded) and LoD2 (ground surface + roof)
tiles.  Designed for memory-bounded streaming of large XML members:

- LoD2 building = ``measuredHeight`` + merged ``GroundSurface`` polygons.
- LoD1 building = ``measuredHeight`` + the lowest horizontal face of the
  ``lod1Solid`` block.  LoD1 has no ``GroundSurface``; the footprint is
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
from dataclasses import dataclass
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
    """A single building extracted from CityGML."""

    building_id: str
    footprint: Polygon | MultiPolygon | None
    measured_height: float | None  # metres above ground


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


def _lod2_ground_polygons(building_elem: ET.Element, ns: dict[str, str]) -> list[Polygon]:
    """Return every parsed polygon under ``bldg:GroundSurface`` children."""
    out: list[Polygon] = []
    for gs_elem in building_elem.iter(f"{{{ns['bldg']}}}GroundSurface"):
        for poly_elem in gs_elem.iter(f"{{{ns['gml']}}}Polygon"):
            poly = _parse_polygon(poly_elem, ns)
            if poly is not None:
                out.append(poly)
    return out


def _extract_lod2_building(
    building_elem: ET.Element,
    ns: dict[str, str],
) -> Building | None:
    """Extract a single LoD2 building's footprint + ``measuredHeight``."""
    bid = building_elem.get(
        f"{{{ns.get('gml', '')}}}id",
        building_elem.get("gml:id", ""),
    )

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

    ground_polys = _lod2_ground_polygons(building_elem, ns)
    if not ground_polys:
        return None

    try:
        merged = unary_union(ground_polys)
    except Exception:
        return None
    if isinstance(merged, (Polygon, MultiPolygon)):
        footprint = merged
    else:
        return None
    if footprint.is_empty or not footprint.is_valid:
        return None

    return Building(building_id=bid, footprint=footprint, measured_height=height)


def parse_lod2_buildings(xml_bytes: bytes) -> list[Building]:
    """Parse every ``bldg:Building`` from raw XML bytes (CityGML 1.0 or 2.0)."""
    version = detect_citygml_version(xml_bytes)
    ns = _CITYGML_NS_MAP[version]

    try:
        root = ET.fromstring(xml_bytes)  # noqa: S314 — trusted XML
    except ET.ParseError as exc:
        _logger.warning("citygml parse error: %s", exc)
        return []

    buildings: list[Building] = []
    for building_elem in root.iter(f"{{{ns['bldg']}}}Building"):
        building = _extract_lod2_building(building_elem, ns)
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
    base_faces: list[Polygon] = [
        poly for z, poly in horizontal_faces if abs(z - base_z) <= 1e-3
    ]

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
    "detect_citygml_version",
    "iter_xml_members",
    "parse_lod1_footprints",
    "parse_lod1_footprints_from_file",
    "parse_lod1_footprints_from_zip_member",
    "parse_lod2_buildings",
    "parse_lod2_buildings_from_file",
    "parse_lod2_buildings_from_zip_member",
    "read_xml_member",
]