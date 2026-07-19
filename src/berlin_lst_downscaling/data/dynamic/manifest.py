"""Dynamic manifest reader — select Landsat anchors from v3 manifest bundle.

The v3 manifest bundle contains manifest.parquet + pairings.parquet.
This module reads only the manifest and filters to Landsat anchor rows,
which are the input for dynamic product generation.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

import pyarrow.parquet as pq

from berlin_lst_downscaling.data.io.storage import exists, read_bytes


@dataclass
class DynamicScene:
    """A Landsat anchor scene selected for dynamic product generation."""

    scene_id: str
    source: str
    role: str
    platform: str
    year: int
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
) -> ManifestReport:
    """Load Landsat anchor scenes from a v3 manifest bundle.

    Parameters
    ----------
    manifest_uri :
        URI to manifest.parquet (local or ``gs://``).
    years :
        If given, restrict to these years.  None = all years.

    Returns
    -------
    ManifestReport
        Filtered scenes and metadata for dynamic pipeline consumption.
    """
    errors: list[str] = []

    if not manifest_uri.endswith(".manifest.parquet"):
        # Accept raw path; derive if needed
        pass

    if not exists(manifest_uri):
        return ManifestReport([], 0, "", [f"Manifest not found: {manifest_uri}"])

    try:
        raw = read_bytes(manifest_uri)
        table = pq.read_table(io.BytesIO(raw))
    except Exception as exc:
        return ManifestReport([], 0, "", [f"Failed to read manifest: {exc}"])

    total_rows = table.num_rows

    # Compute manifest fingerprint
    manifest_hash = sha256(raw).hexdigest()[:16]

    # Filter to Landsat anchors
    import pyarrow as pa
    import pyarrow.compute as pc

    # pyarrow.compute stubs — every ``pc.<name>`` access needs the type-ignore
    _pceq = pc.equal  # type: ignore[attr-defined]
    _pcand = pc.and_  # type: ignore[attr-defined]
    _pcin = pc.is_in  # type: ignore[attr-defined]

    source_col = table.column("source")
    role_col = table.column("role")

    mask = _pcand(
        _pceq(source_col, "landsat-c2-l2"),
        _pceq(role_col, "anchor"),
    )
    filtered = table.filter(mask)

    # Filter by year if requested
    if years is not None:
        year_col = filtered.column("year")
        year_mask = _pcin(year_col, value_set=pa.array(years))
        filtered = filtered.filter(year_mask)

    # Convert to DynamicScene objects
    scenes: list[DynamicScene] = []
    for i in range(filtered.num_rows):
        row = filtered.slice(i, 1)
        d = row.to_pydict()

        try:
            dt_raw = d["acquisition_datetime"][0]
            if dt_raw.tzinfo is None:
                dt = dt_raw.replace(tzinfo=UTC)
            else:
                dt = dt_raw.astimezone(UTC)
        except Exception:
            errors.append(f"Row {i}: invalid acquisition_datetime")
            continue

        scenes.append(
            DynamicScene(
                scene_id=str(d["scene_id"][0]),
                source=str(d["source"][0]),
                role=str(d["role"][0]),
                platform=str(d["platform"][0]),
                year=int(d["year"][0]),
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
