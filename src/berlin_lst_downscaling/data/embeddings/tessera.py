"""TESSERA v1.1-dclimate acquisition for the Berlin bbox.

Reads the public TESSERA store (``s3://tessera-embeddings/v1.1/dclimate.icechunk/``,
zone group ``33N``) through :mod:`geotessera` (``>=0.11``). The Berlin bbox
lies entirely in UTM 33, native CRS ``EPSG:32633``. Supported years are
2017–2025; 2026 is refused because the calendar-year store has no 2026
embedding and the arm is never used for 2026 inference.

The quantized ``int8 + float32 scale`` path is preferred:
``dequantized = int8 * scale`` per pixel. Provenance records the dataset
version, zone, year, ``years_complete``, library version, and the source
chunk/tile ids reported by the reader.

This module never touches GCS writes; the product writer lives in
:mod:`berlin_lst_downscaling.data.embeddings.product`. All store reads are
lazy (``geotessera`` is imported inside the read call) so offline unit
tests can inject fakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from berlin_lst_downscaling.common.config import BERLIN_BBOX

# ── store identity (fixed by the task) ────────────────────────────────

TESSERA_DATASET_VERSION: str = "v1.1-dclimate"
TESSERA_STORE_URI: str = "s3://tessera-embeddings/v1.1/dclimate.icechunk/"
TESSERA_ZONE: str = "33N"
TESSERA_ZONE_CRS: str = "EPSG:32633"
TESSERA_N_CHANNELS: int = 128

# Berlin WGS84 bbox; entirely inside UTM 33.
TESSERA_BBOX: tuple[float, float, float, float] = BERLIN_BBOX

# Annual embeddings for the training/validation/test contract.
TESSERA_SUPPORTED_YEARS: tuple[int, ...] = tuple(range(2017, 2026))

# Default quality threshold applied to the per-pixel quality band when the
# reader provides one. Pixels below this threshold become TESSERA-invalid
# (NaN after dequantization), never zero-filled. The value is a documented
# default, not a tuned scientific choice; the operator config may override
# it and the provenance records the effective value.
TESSERA_QUALITY_THRESHOLD_DEFAULT: float = 0.0


class TesseraYearError(ValueError):
    """A TESSERA year outside the 2017–2025 ablation contract."""


def validate_tessera_year(year: int) -> int:
    """Validate a TESSERA embedding year; refuse 2026 and out-of-range years.

    Parameters
    ----------
    year :
        Calendar year of the annual embedding.

    Returns
    -------
    int
        The validated year.

    Raises
    ------
    TesseraYearError
        If ``year`` is 2026 (no calendar-year embedding; the arm is never
        used for 2026 inference) or outside 2017–2025.
    """
    if not isinstance(year, int) or isinstance(year, bool):
        raise TesseraYearError(f"TESSERA year must be an int in 2017–2025, got {year!r}")
    if year == 2026:
        raise TesseraYearError(
            "TESSERA has no 2026 embedding (calendar-year store); "
            "the TESSERA arm is never used for 2026 inference"
        )
    if year not in TESSERA_SUPPORTED_YEARS:
        raise TesseraYearError(f"TESSERA year {year} outside the 2017–2025 ablation contract")
    return year


def geotessera_version() -> str:
    """Return the installed ``geotessera`` version, or ``unknown``."""
    try:
        from importlib.metadata import version

        return version("geotessera")
    except Exception:  # noqa: BLE001 — version lookup must never fail a run
        return "unknown"


@dataclass(frozen=True)
class TesseraProvenance:
    """Provenance recorded for one annual TESSERA read/product year."""

    dataset_version: str = TESSERA_DATASET_VERSION
    zone: str = TESSERA_ZONE
    zone_crs: str = TESSERA_ZONE_CRS
    store_uri: str = TESSERA_STORE_URI
    year: int = 0
    years_complete: tuple[int, ...] = field(default_factory=tuple)
    library_version: str = "unknown"
    source_ids: tuple[str, ...] = field(default_factory=tuple)
    quantized: bool = True
    quality_threshold: float = TESSERA_QUALITY_THRESHOLD_DEFAULT

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable provenance document."""
        return {
            "dataset_version": self.dataset_version,
            "zone": self.zone,
            "zone_crs": self.zone_crs,
            "store_uri": self.store_uri,
            "year": self.year,
            "years_complete": list(self.years_complete),
            "library_version": self.library_version,
            "source_ids": list(self.source_ids),
            "quantized": self.quantized,
            "quality_threshold": self.quality_threshold,
        }


class TesseraReader(Protocol):
    """Minimal reader surface used by :func:`read_tessera_region`.

    Production passes ``None`` (lazy ``GeoTesseraZarr``); tests inject a
    fake implementing one of these methods. The quantized path is
    preferred when available.
    """

    def read_region_quantized(
        self, bbox: tuple[float, float, float, float], year: int
    ) -> tuple[np.ndarray, np.ndarray, Any, str]: ...

    def read_region(
        self, bbox: tuple[float, float, float, float], year: int
    ) -> tuple[np.ndarray, Any, str]: ...


@dataclass
class TesseraRead:
    """One annual TESSERA read over the Berlin bbox."""

    quantized: np.ndarray | None  # int8 (H, W, 128) when the quantized path ran
    scale: np.ndarray | None  # float32 (H, W) or (H, W, 1) scale per pixel
    embedding: np.ndarray | None  # float32 (H, W, 128) dequantized, NaN = invalid
    valid: np.ndarray  # bool (H, W): True where the pixel is TESSERA-valid
    transform: Any  # rasterio Affine of the native read
    crs: str  # native CRS (EPSG:32633)
    provenance: TesseraProvenance


def dequantize_tessera(quantized: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize int8 embeddings with per-pixel float32 scales.

    ``dequantized = int8 * scale`` with per-pixel broadcasting. Pixels
    with a non-finite scale become NaN (never zero-filled).

    Parameters
    ----------
    quantized :
        int8 array ``(H, W, 128)``.
    scale :
        float32 scale array ``(H, W)`` or ``(H, W, 1)`` / ``(H, W, 128)``.

    Returns
    -------
    numpy.ndarray
        float32 ``(H, W, 128)`` with NaN where ``scale`` is non-finite.
    """
    if quantized.dtype != np.int8:
        raise ValueError(f"quantized TESSERA must be int8, got {quantized.dtype}")
    if quantized.ndim != 3 or quantized.shape[2] != TESSERA_N_CHANNELS:
        raise ValueError(
            f"quantized TESSERA must have shape (H, W, 128), got {tuple(quantized.shape)}"
        )
    scale_arr = np.asarray(scale)
    if scale_arr.ndim == 2:
        scale_arr = scale_arr[:, :, np.newaxis]
    if scale_arr.shape[:2] != quantized.shape[:2]:
        raise ValueError(
            "scale spatial shape "
            f"{tuple(scale_arr.shape[:2])} != quantized {tuple(quantized.shape[:2])}"
        )
    scale_f32 = scale_arr.astype(np.float32, copy=False)
    out = quantized.astype(np.float32) * scale_f32
    finite = np.isfinite(scale_f32)
    if finite.ndim == 3 and finite.shape[2] != 1 and finite.shape[2] != TESSERA_N_CHANNELS:
        raise ValueError(f"unexpected scale shape {tuple(scale_arr.shape)}")
    if finite.ndim == 3:
        pixel_finite = np.all(finite, axis=2)
    else:  # pragma: no cover — defensive; 2-D was expanded above
        pixel_finite = finite
    out[~pixel_finite] = np.nan
    return out.astype(np.float32)


def tessera_invalid_mask(
    scale: np.ndarray,
    quality: np.ndarray | None = None,
    *,
    quality_threshold: float = TESSERA_QUALITY_THRESHOLD_DEFAULT,
) -> np.ndarray:
    """Return the TESSERA-invalid mask (True = gap, never filled).

    A pixel is invalid when its scale is non-finite or, when a quality
    band is present, its quality is below ``quality_threshold`` or
    non-finite. Valid pixels are False.

    Parameters
    ----------
    scale :
        Per-pixel float32 scale ``(H, W)`` (or broadcastable).
    quality :
        Optional per-pixel quality ``(H, W)``; ``None`` disables the check.
    quality_threshold :
        Minimum accepted quality value.
    """
    scale_arr = np.asarray(scale)
    if scale_arr.ndim == 3:
        scale_arr = scale_arr.reshape(scale_arr.shape[0], scale_arr.shape[1], -1)
        invalid = ~np.all(np.isfinite(scale_arr), axis=2)
    else:
        invalid = ~np.isfinite(scale_arr)
    invalid = np.asarray(invalid, dtype=bool)
    if quality is not None:
        quality_arr = np.asarray(quality, dtype=np.float64)
        if quality_arr.shape != invalid.shape:
            raise ValueError(
                f"quality shape {tuple(quality_arr.shape)} != scale spatial {tuple(invalid.shape)}"
            )
        invalid = invalid | ~(np.isfinite(quality_arr) & (quality_arr >= quality_threshold))
    return invalid


def _native_read(
    reader: Any,
    bbox: tuple[float, float, float, float],
    year: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, Any, str, bool, tuple[str, ...]]:
    """Run the preferred quantized path, falling back to the float path."""
    if hasattr(reader, "read_region_quantized"):
        result = reader.read_region_quantized(bbox, year=year)
        if len(result) == 4:
            quantized, scale, transform, crs = result
            source_ids: tuple[str, ...] = ()
        elif len(result) == 5:
            quantized, scale, transform, crs, meta = result
            source_ids = _source_ids_from_meta(meta)
        else:  # pragma: no cover — reader contract violation
            raise ValueError(
                "read_region_quantized must return (quantized, scale, transform, crs[, meta])"
            )
        embedding = dequantize_tessera(np.asarray(quantized, dtype=np.int8), np.asarray(scale))
        return (
            np.asarray(quantized, dtype=np.int8),
            np.asarray(scale),
            embedding,
            transform,
            str(crs),
            True,
            source_ids,
        )
    mosaic, transform, crs = reader.read_region(bbox, year=year)
    embedding = np.asarray(mosaic, dtype=np.float32)
    return None, None, embedding, transform, str(crs), False, ()


def _source_ids_from_meta(meta: Any) -> tuple[str, ...]:
    """Extract source chunk/tile ids from a reader metadata payload."""
    if meta is None:
        return ()
    if isinstance(meta, dict):
        for key in ("source_ids", "chunk_ids", "tile_ids", "chunks", "tiles"):
            raw = meta.get(key)
            if raw:
                return tuple(str(item) for item in list(raw))
        return ()
    if isinstance(meta, (list, tuple)):
        return tuple(str(item) for item in meta)
    return (str(meta),)


def _years_complete(reader: Any) -> tuple[int, ...]:
    """Best-effort ``years_complete`` probe for provenance (never fails)."""
    for attr in ("years_complete", "years", "available_years"):
        try:
            raw = getattr(reader, attr, None)
            years: Any = raw() if callable(raw) else raw
            if years:
                return tuple(int(y) for y in list(years))  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001, S112 — provenance probe must not fail reads
            continue
    return ()


def read_tessera_region(
    year: int,
    bbox: tuple[float, float, float, float] | None = None,
    *,
    reader: TesseraReader | None = None,
    quality: np.ndarray | None = None,
    quality_threshold: float = TESSERA_QUALITY_THRESHOLD_DEFAULT,
) -> TesseraRead:
    """Read the annual TESSERA embedding for the Berlin bbox and one year.

    Prefers the quantized int8+scale path; falls back to the dequantized
    ``read_region`` path when the reader has no quantized method. The
    returned ``embedding`` carries NaN at TESSERA-invalid pixels.

    Parameters
    ----------
    year :
        Calendar year in 2017–2025. 2026 raises :class:`TesseraYearError`.
    bbox :
        WGS84 ``(west, south, east, north)``; defaults to the Berlin bbox.
    reader :
        Injected reader (tests); ``None`` builds the lazy production
        ``GeoTesseraZarr`` client.
    quality :
        Optional per-pixel quality band aligned to the native read; when
        given, sub-threshold pixels become invalid.
    quality_threshold :
        Minimum accepted quality value.
    """
    validate_tessera_year(int(year))
    region = tuple(bbox) if bbox is not None else TESSERA_BBOX
    if len(region) != 4:
        raise ValueError(f"bbox must be (west, south, east, north), got {region!r}")

    if reader is None:
        try:
            from geotessera.store import GeoTesseraZarr  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "geotessera>=0.11 is required for TESSERA reads; install it or inject a fake reader"
            ) from exc
        reader = GeoTesseraZarr()

    quantized, scale, embedding, transform, crs, used_quantized, source_ids = _native_read(
        reader, (float(region[0]), float(region[1]), float(region[2]), float(region[3])), int(year)
    )
    if scale is not None:
        invalid = tessera_invalid_mask(
            np.asarray(scale), quality, quality_threshold=quality_threshold
        )
    else:
        invalid = ~np.all(np.isfinite(embedding), axis=2)
        if quality is not None:
            quality_arr = np.asarray(quality, dtype=np.float64)
            if quality_arr.shape != invalid.shape:
                raise ValueError("quality shape does not match the native read")
            invalid = invalid | ~(np.isfinite(quality_arr) & (quality_arr >= quality_threshold))
    valid = ~np.asarray(invalid, dtype=bool)
    if embedding is not None:
        embedding = np.asarray(embedding, dtype=np.float32)
        embedding[~valid] = np.nan

    provenance = TesseraProvenance(
        year=int(year),
        years_complete=_years_complete(reader),
        library_version=geotessera_version(),
        source_ids=tuple(source_ids),
        quantized=bool(used_quantized),
        quality_threshold=float(quality_threshold),
    )
    return TesseraRead(
        quantized=quantized,
        scale=None if scale is None else np.asarray(scale),
        embedding=embedding,
        valid=np.asarray(valid, dtype=bool),
        transform=transform,
        crs=str(crs) if crs else TESSERA_ZONE_CRS,
        provenance=provenance,
    )


__all__ = [
    "TESSERA_BBOX",
    "TESSERA_DATASET_VERSION",
    "TESSERA_N_CHANNELS",
    "TESSERA_QUALITY_THRESHOLD_DEFAULT",
    "TESSERA_STORE_URI",
    "TESSERA_SUPPORTED_YEARS",
    "TESSERA_ZONE",
    "TESSERA_ZONE_CRS",
    "TesseraProvenance",
    "TesseraRead",
    "TesseraReader",
    "TesseraYearError",
    "dequantize_tessera",
    "geotessera_version",
    "read_tessera_region",
    "tessera_invalid_mask",
    "validate_tessera_year",
]
