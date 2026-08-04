"""Shared strict COG validation and GDAL option checks."""

from __future__ import annotations

from typing import Any


class StrictCogResult:
    """Structured result from strict COG validation.

    ``valid=True`` only when there are zero errors and zero warnings.
    """

    def __init__(
        self,
        valid: bool,
        errors: tuple[str, ...],
        warnings: tuple[str, ...],
        source: str = "",
    ):
        self.valid = valid and not errors and not warnings
        self.errors = errors
        self.warnings = warnings
        self.source = source

    def __bool__(self) -> bool:
        return self.valid

    def __repr__(self) -> str:
        return (
            f"StrictCogResult(valid={self.valid}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "source": self.source,
        }


def validate_strict_cog(uri: str) -> StrictCogResult:
    """Validate COG layout using rio-cogeo in fail-closed strict mode."""
    from rio_cogeo.cogeo import cog_validate

    from berlin_lst_downscaling.data.profiling.inspection import gdal_uri

    try:
        valid, errors, warnings = cog_validate(gdal_uri(uri), strict=True, quiet=True)
        return StrictCogResult(
            valid=valid,
            errors=tuple(f"COG strict: {e}" for e in errors),
            warnings=tuple(f"COG strict warning: {w}" for w in warnings),
            source=uri,
        )
    except FileNotFoundError:
        return StrictCogResult(
            valid=False,
            errors=("rio-cogeo not found in PATH",),
            warnings=(),
            source=uri,
        )
    except Exception as exc:
        return StrictCogResult(
            valid=False,
            errors=(f"COG strict validation failed: {exc}",),
            warnings=(),
            source=uri,
        )


def validate_gdal_cog_options(
    creation_options: dict[str, Any],
    overview_resampling: str,
    has_overviews: bool,
) -> list[str]:
    """Validate that required GDAL COG creation options are present."""
    errors: list[str] = []
    required_keys = ["BLOCKSIZE", "COMPRESS", "PREDICTOR", "BIGTIFF"]
    if has_overviews:
        required_keys.append("OVERVIEW_RESAMPLING")

    for key in required_keys:
        if key not in creation_options:
            errors.append(f"Missing required COG option: {key}")

    return errors


__all__ = [
    "validate_strict_cog",
    "validate_gdal_cog_options",
    "StrictCogResult",
]
