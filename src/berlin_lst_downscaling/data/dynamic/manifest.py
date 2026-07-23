"""Dynamic manifest reader — Landsat anchors from the v3 bundle.

The dynamic pipeline reads the canonical v3 manifest bundle through
``selection.validate.load_bundle`` and filters to anchor scenes.
The pairings and report artifacts are validated together so a
single bundle load is trusted for both ARD and downstream coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from berlin_lst_downscaling.data.io.storage import exists, read_bytes
from berlin_lst_downscaling.data.selection.validate import load_bundle

# Default study period — can be overridden via years parameter
_DEFAULT_YEARS = list(range(2017, 2026))


@dataclass
class DynamicScene:
    """A Landsat anchor scene selected for dynamic product generation."""

    scene_id: str
    source: str
    role: str
    platform: str
    year: int
    day_of_year: int
    acquisition_datetime: datetime
    item_href: str | None
    cloud_cover: float | None
    solar_azimuth: float | None
    solar_elevation: float | None


@dataclass
class ManifestReport:
    """Result of loading and filtering a manifest for dynamic processing."""

    scenes: list[DynamicScene]
    total_rows: int
    manifest_hash: str
    errors: list[str]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0 and len(self.scenes) > 0


def load_landsat_anchors(
    manifest_uri: str,
    years: list[int] | None = None,
    scene_ids: list[str] | None = None,
    dataset_role: str | None = None,
) -> ManifestReport:
    """Load Landsat anchor scenes from the canonical v3 manifest bundle.

    Parameters
    ----------
    manifest_uri :
        URI to ``manifest.parquet``. Sibling ``pairings.parquet`` and
        ``manifest_report.json`` are loaded and validated together via
        :func:`selection.validate.load_bundle`.
    years :
        Restrict to these years.  Defaults to 2017–2025.
    scene_ids :
        Restrict to these scene IDs (skips year filter).
    dataset_role :
        If given, attach this role to every returned scene. Stored on
        ``DynamicScene.role`` for downstream ledger/STAC propagation.

    Returns
    -------
    ManifestReport
        Filtered scenes and metadata for dynamic pipeline consumption.
    """
    if years is None and not scene_ids:
        years = _DEFAULT_YEARS

    errors: list[str] = []

    if not exists(manifest_uri):
        return ManifestReport(
            scenes=[], total_rows=0, manifest_hash="",
            errors=[f"Manifest not found: {manifest_uri}"],
        )

    bundle, validation = load_bundle(manifest_uri, require_item_href=True)
    if not validation.ok:
        return ManifestReport(
            scenes=[], total_rows=0, manifest_hash="",
            errors=validation.errors,
        )
    table = bundle.manifest_table

    import pyarrow as pa
    import pyarrow.compute as pc

    _pceq = pc.equal  # type: ignore[attr-defined]
    _pcand = pc.and_  # type: ignore[attr-defined]
    _pcin = pc.is_in  # type: ignore[attr-defined]

    source_col = table.column("source")
    role_col = table.column("role")
    if scene_ids:
        mask = _pcand(
            _pceq(source_col, "landsat-c2-l2"),
            _pceq(role_col, "anchor"),
        )
    else:
        year_col = table.column("year")
        mask = _pcand(
            _pcand(
                _pceq(source_col, "landsat-c2-l2"),
                _pceq(role_col, "anchor"),
            ),
            _pcin(year_col, value_set=pa.array(years)),
        )
    filtered = table.filter(mask)

    if scene_ids:
        filtered = filtered.filter(
            _pcin(filtered.column("scene_id"), value_set=pa.array(scene_ids))
        )

    total_rows = bundle.manifest_table.num_rows
    manifest_hash = sha256(read_bytes(manifest_uri)).hexdigest()[:16]

    scenes: list[DynamicScene] = []
    for i in range(filtered.num_rows):
        row = filtered.slice(i, 1)
        d = row.to_pydict()

        dt_raw = d["acquisition_datetime"][0]
        dt = dt_raw if dt_raw.tzinfo else dt_raw.replace(tzinfo=UTC)
        dt = dt.astimezone(UTC)

        scenes.append(
            DynamicScene(
                scene_id=str(d["scene_id"][0]),
                source=str(d["source"][0]),
                role=dataset_role or str(d["role"][0]),
                platform=str(d["platform"][0]),
                year=int(d["year"][0]),
                day_of_year=dt.timetuple().tm_yday,
                acquisition_datetime=dt,
                item_href=d.get("item_href", [None])[0],
                cloud_cover=d.get("cloud_cover", [None])[0],
                solar_azimuth=d.get("solar_azimuth", [None])[0],
                solar_elevation=d.get("solar_elevation", [None])[0],
            )
        )

    return ManifestReport(
        scenes=scenes,
        total_rows=total_rows,
        manifest_hash=manifest_hash,
        errors=errors,
    )


__all__ = [
    "DynamicScene",
    "ManifestReport",
    "load_landsat_anchors",
]
