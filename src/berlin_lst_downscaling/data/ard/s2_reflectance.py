"""Sentinel-2 L2A reflectance harmonization across processing baselines.

ESA introduced ``BOA_ADD_OFFSET = -1000`` DN in Processing Baseline 04.00
(25 January 2022). Physical bottom-of-atmosphere reflectance is

    (DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE

with ``QUANTIFICATION_VALUE = 10000``. Before baseline 04.00 the offset
is 0, so ``DN / 10000`` is already the reflectance. Planetary Computer
serves the digital numbers unharmonized. The offset is chosen per scene
from that scene's STAC metadata. The acquisition date is not a fallback:
a reprocessed scene can carry a new baseline on an old date, and two
items from the same day can differ.

Nodata
------
Non-finite DN stays non-finite. Digital number 0 is a valid code. After
baseline 04.00 it is reflectance ``-0.1``, which the published ``[0, 1]``
contract then clips to 0. SCL class 0 (fill) is the nodata mask and is
applied by :func:`berlin_lst_downscaling.data.ard.masking.mask_s2` after
scaling. A raster ``nodata: 0`` field is recorded and not applied, because
it collides with the offset's lowest code.

Where the correction runs
-------------------------
The correction is applied to the raw DN in ``mask_s2``, before the
``[0, 1]`` clip and before the SWIR resample. Doing it later, on ARD
that was stored as ``clip(DN / 10000, 0, 1)``, is not exact: any DN
above 10000 was already clipped to 1, so subtracting 0.1 yields 0.9
instead of ``clip((DN - 1000) / 10000, 0, 1)``. Those pixels cannot be
reconstructed. Features/v6 therefore reads republished S2 ARD (schema
version 9) and does not apply a second offset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

QUANTIFICATION_VALUE: int = 10000
BOA_ADD_OFFSET_PRE_04: int = 0
BOA_ADD_OFFSET_FROM_04: int = -1000
# Processing Baseline 04.00 is the first baseline that stores the offset.
BASELINE_OFFSET_INTRODUCED: float = 4.0

_REFLECTANCE_ASSETS: tuple[str, ...] = ("B02", "B03", "B04", "B08", "B11", "B12")
_OFFSET_PROPERTY_KEYS: tuple[str, ...] = ("s2:boa_add_offset", "boa_add_offset")


@dataclass(frozen=True)
class S2ReflectanceScale:
    """Per-scene scale that turns L2A digital numbers into reflectance.

    ``boa_add_offset`` is the ESA digital-number offset (0 or -1000).
    ``decision`` names which metadata field won.
    """

    processing_baseline: str | None
    boa_add_offset: int
    quantification_value: int
    decision: str
    dn_nodata: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready provenance payload. Marks the raster as harmonized."""
        return {
            "processing_baseline": self.processing_baseline,
            "boa_add_offset": int(self.boa_add_offset),
            "quantification_value": int(self.quantification_value),
            "decision": self.decision,
            "dn_nodata": self.dn_nodata,
            "dn_nodata_policy": "scl_fill_only",
            "reflectance_harmonized": True,
        }


def parse_processing_baseline(value: object) -> float:
    """Parse ``s2:processing_baseline`` (``"04.00"``, ``"05.10"``) to a float.

    Raises
    ------
    ValueError
        If *value* is missing or not a baseline number. No date is consulted.
    """
    if value is None:
        raise ValueError("s2:processing_baseline is missing")
    text = str(value).strip()
    if not text:
        raise ValueError("s2:processing_baseline is empty")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"s2:processing_baseline {value!r} is not a number") from exc


def _dn_offset_from_property(value: object) -> int:
    """Interpret an explicit offset property as a digital-number offset.

    Values with absolute value below 1 are reflectance-unit offsets
    (STAC ``offset`` of -0.1). Larger magnitudes are already DN (-1000).
    """
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"BOA_ADD_OFFSET {value!r} is not numeric") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"BOA_ADD_OFFSET {value!r} is not finite")
    if abs(numeric) < 1.0:
        return int(round(numeric * QUANTIFICATION_VALUE))
    return int(round(numeric))


def _dn_offset_from_raster_band(band: Mapping[str, Any]) -> int | None:
    """Return the DN offset implied by one ``raster:bands`` entry, if any."""
    if "offset" not in band or band["offset"] is None:
        return None
    offset = float(band["offset"])
    scale = band.get("scale")
    if scale is None:
        raise ValueError("raster:bands offset is set but scale is missing")
    scale_f = float(scale)
    if scale_f == 0.0 or not np.isfinite(scale_f) or not np.isfinite(offset):
        raise ValueError(f"raster:bands scale/offset {scale!r}/{offset!r} is unusable")
    return int(round(offset / scale_f))


def _baseline_offset(baseline: float) -> int:
    """Map a processing baseline to the ESA BOA digital-number offset."""
    if baseline >= BASELINE_OFFSET_INTRODUCED:
        return BOA_ADD_OFFSET_FROM_04
    return BOA_ADD_OFFSET_PRE_04


def resolve_s2_reflectance_scale(
    properties: Mapping[str, Any],
    *,
    raster_bands: Sequence[Mapping[str, Any]] | None = None,
) -> S2ReflectanceScale:
    """Choose the BOA offset from scene metadata.

    Explicit ``BOA_ADD_OFFSET`` / ``s2:boa_add_offset`` wins, then an
    agreeing ``raster:bands`` scale and offset, then
    ``s2:processing_baseline`` (baseline >= 04.00 → -1000 DN, otherwise
    0). If an explicit offset and the baseline rule disagree, this
    raises. If none of the fields are present, this raises. The scene
    date is never used.

    Parameters
    ----------
    properties :
        STAC item properties.
    raster_bands :
        Optional ``raster:bands`` entries from the reflectance assets.
        SCL is not one of them. Bands that omit ``offset`` are ignored.
        Bands that set it must agree with each other.
    """
    baseline_text = properties.get("s2:processing_baseline")
    baseline_value: float | None
    if baseline_text is None or str(baseline_text).strip() == "":
        baseline_value = None
        baseline_stored: str | None = None
    else:
        baseline_value = parse_processing_baseline(baseline_text)
        baseline_stored = str(baseline_text).strip()

    explicit: int | None = None
    for key in _OFFSET_PROPERTY_KEYS:
        if key in properties and properties[key] is not None:
            explicit = _dn_offset_from_property(properties[key])
            break

    raster_offsets: list[int] = []
    dn_nodata: int | None = None
    if raster_bands:
        for band in raster_bands:
            parsed = _dn_offset_from_raster_band(band)
            if parsed is not None:
                raster_offsets.append(parsed)
            if dn_nodata is None and band.get("nodata") is not None:
                try:
                    dn_nodata = int(band["nodata"])
                except (TypeError, ValueError):
                    dn_nodata = None
    raster_offset: int | None = None
    if raster_offsets:
        unique = set(raster_offsets)
        if len(unique) != 1:
            raise ValueError(f"reflectance assets disagree on BOA offset: {sorted(unique)}")
        raster_offset = raster_offsets[0]

    if explicit is not None and raster_offset is not None and explicit != raster_offset:
        raise ValueError(
            f"BOA_ADD_OFFSET {explicit} disagrees with raster:bands offset {raster_offset}"
        )

    chosen: int | None = explicit if explicit is not None else raster_offset
    decision: str
    if chosen is not None:
        decision = "boa_add_offset" if explicit is not None else "raster_band_offset"
    elif baseline_value is not None:
        chosen = _baseline_offset(baseline_value)
        decision = "processing_baseline"
    else:
        raise ValueError(
            "Sentinel-2 reflectance offset needs s2:processing_baseline or "
            "BOA_ADD_OFFSET on the scene. The acquisition date is not used."
        )

    if baseline_value is not None:
        expected = _baseline_offset(baseline_value)
        if chosen != expected:
            raise ValueError(
                f"metadata offset {chosen} disagrees with processing baseline "
                f"{baseline_stored} (expected DN offset {expected})"
            )

    return S2ReflectanceScale(
        processing_baseline=baseline_stored,
        boa_add_offset=int(chosen),
        quantification_value=QUANTIFICATION_VALUE,
        decision=decision,
        dn_nodata=dn_nodata,
    )


def _asset_raster_bands(asset: Any) -> list[Mapping[str, Any]]:
    """Return ``raster:bands`` from a STAC asset or a mapping."""
    extra: Any
    if isinstance(asset, Mapping):
        extra = asset.get("raster:bands")
        if extra is None:
            extra = asset.get("extra_fields", {})
            if isinstance(extra, Mapping):
                extra = extra.get("raster:bands")
    else:
        extra_fields = getattr(asset, "extra_fields", {}) or {}
        extra = extra_fields.get("raster:bands") if isinstance(extra_fields, Mapping) else None
    if not extra:
        return []
    return [band for band in extra if isinstance(band, Mapping)]


def reflectance_scale_from_item(item: Any) -> S2ReflectanceScale:
    """Resolve the offset for one STAC item.

    Reads ``s2:processing_baseline`` and ``BOA_ADD_OFFSET`` from
    properties, and ``raster:bands`` scale/offset from the six
    reflectance assets when those assets are present.
    """
    properties = getattr(item, "properties", None)
    if properties is None and isinstance(item, Mapping):
        properties = item.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ValueError("STAC item has no properties")
    assets = getattr(item, "assets", None)
    if assets is None and isinstance(item, Mapping):
        assets = item.get("assets", {})
    bands: list[Mapping[str, Any]] = []
    if isinstance(assets, Mapping):
        for name in _REFLECTANCE_ASSETS:
            asset = assets.get(name)
            if asset is not None:
                bands.extend(_asset_raster_bands(asset))
    return resolve_s2_reflectance_scale(properties, raster_bands=bands or None)


def reflectance_scale_from_items(items: Sequence[Any]) -> S2ReflectanceScale:
    """Resolve one offset for the items that make a scene.

    A Berlin Sentinel-2 scene is one STAC item. If several items are
    passed they must carry the same digital-number offset. A mosaic of
    mixed baselines cannot be scaled with one factor.
    """
    if not items:
        raise ValueError(
            "Sentinel-2 reflectance offset requires the scene STAC item. "
            "Refusing to scale without metadata."
        )
    scales = [reflectance_scale_from_item(item) for item in items]
    offsets = {scale.boa_add_offset for scale in scales}
    if len(offsets) != 1:
        raise ValueError(
            "scene STAC items disagree on BOA_ADD_OFFSET "
            f"({sorted(offsets)}); refusing a single scale"
        )
    return scales[0]


def scale_s2_digital_numbers(dn: np.ndarray, scale: S2ReflectanceScale) -> np.ndarray:
    """Convert L2A digital numbers to harmonized reflectance in ``[0, 1]``.

    Non-finite inputs stay NaN. Finite DN, including 0, is scaled with
    ``(DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE`` and then clipped to
    the published reflectance contract. The clip is the contract bound,
    applied after the offset so the negative atmospheric-correction tail
    and the specular tail stay inside ``[0, 1]``.
    """
    values = np.atleast_1d(np.asarray(dn, dtype=np.float64))
    missing = ~np.isfinite(values)
    physical = (values + float(scale.boa_add_offset)) / float(scale.quantification_value)
    np.clip(physical, 0.0, 1.0, out=physical)
    physical[missing] = np.nan
    scaled = physical.astype(np.float32, copy=False)
    if np.ndim(dn) == 0:
        return scaled.reshape(())
    return scaled


def recover_harmonized_reflectance_from_clipped_ard(
    stored: np.ndarray,
    boa_add_offset: int,
    quantification_value: int = QUANTIFICATION_VALUE,
) -> np.ndarray:
    """Best-effort recovery from ARD stored as ``clip(DN / 10000, 0, 1)``.

    This is **not** the production path. It matches
    :func:`scale_s2_digital_numbers` only when every finite DN was inside
    ``[0, quantification_value]``. DN above that was clipped to 1 before
    the offset existed in the file, and the lost headroom cannot be
    rebuilt. NaNs are preserved.
    """
    values = np.asarray(stored, dtype=np.float64)
    missing = ~np.isfinite(values)
    shift = float(boa_add_offset) / float(quantification_value)
    recovered = values + shift
    np.clip(recovered, 0.0, 1.0, out=recovered)
    recovered[missing] = np.nan
    return recovered.astype(np.float32, copy=False)


__all__ = [
    "BASELINE_OFFSET_INTRODUCED",
    "BOA_ADD_OFFSET_FROM_04",
    "BOA_ADD_OFFSET_PRE_04",
    "QUANTIFICATION_VALUE",
    "S2ReflectanceScale",
    "parse_processing_baseline",
    "recover_harmonized_reflectance_from_clipped_ard",
    "reflectance_scale_from_item",
    "reflectance_scale_from_items",
    "resolve_s2_reflectance_scale",
    "scale_s2_digital_numbers",
]
