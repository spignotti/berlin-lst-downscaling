"""GDAL GCS credentials for rasterio reads of gs:// COGs.

``google-cloud-storage`` (used by :mod:`data.io.storage`) authenticates
via Application Default Credentials automatically, but ``rasterio.open``
on a ``gs://`` URI goes through GDAL's ``/vsigs/`` handler, which only
sees GDAL-level credentials (``GS_*`` / ``GOOGLE_APPLICATION_CREDENTIALS``).
On machines where the operator authenticated with ``gcloud auth`` but never
exported GDAL variables, every COG open fails with the
"GS_SECRET_ACCESS_KEY+GS_ACCESS_KEY_ID, GS_OAUTH2_REFRESH_TOKEN or
GOOGLE_APPLICATION_CREDENTIALS ... not defined" error.

:func:`ensure_gdal_gcs_auth` points GDAL at the ADC file when no GDAL GCS
credential is already set. It only sets the env var to a file path — it
never reads, prints, commits, or copies secret material.
"""

from __future__ import annotations

import os

# Env vars GDAL's GCS driver accepts (see the validator failure message).
_GDAL_GCS_KEYS = (
    "GS_SECRET_ACCESS_KEY",
    "GS_ACCESS_KEY_ID",
    "GS_OAUTH2_REFRESH_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
)

_ADC_CANDIDATES = (
    # Explicit override wins when it exists.
    "$GOOGLE_APPLICATION_CREDENTIALS",
    "~/.config/gcloud/application_default_credentials.json",
)


def has_gdal_gcs_credentials(env: dict | os._Environ = os.environ) -> bool:
    """Return True when any GDAL GCS credential is already set and non-empty."""
    get = env.get if hasattr(env, "get") else dict(env).get
    if get("GS_SECRET_ACCESS_KEY") and get("GS_ACCESS_KEY_ID"):
        return True
    if get("GS_OAUTH2_REFRESH_TOKEN"):
        return True
    if get("GOOGLE_APPLICATION_CREDENTIALS"):
        return True
    return False


def default_adc_candidate() -> str | None:
    """Return the first existing ADC file path, or None.

    Checks ``$GOOGLE_APPLICATION_CREDENTIALS`` (when it points at an
    existing file) then the well-known gcloud ADC location. Returns the
    expanded absolute path; never reads the file.
    """
    candidates: list[str] = []
    override = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if override:
        candidates.append(os.path.expanduser(override))
    candidates.append(os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def gdal_gcs_env_patch(env: dict, adc_candidate: str | None) -> dict[str, str]:
    """Pure helper: env updates needed for GDAL GCS reads.

    Returns ``{}`` when credentials are already present or when no ADC
    file exists; otherwise ``{"GOOGLE_APPLICATION_CREDENTIALS": path}``.
    Never touches the process environment (unit-testable).
    """
    if has_gdal_gcs_credentials(env):
        return {}
    if adc_candidate and os.path.isfile(os.path.expanduser(adc_candidate)):
        return {"GOOGLE_APPLICATION_CREDENTIALS": os.path.expanduser(adc_candidate)}
    return {}


def ensure_gdal_gcs_auth() -> str | None:
    """Point GDAL at ADC when no GDAL GCS credential is set.

    Returns the ADC path that was exported, or None when nothing changed
    (credentials already present or no ADC file found). Only sets the env
    var; never reads or logs secret content.
    """
    if has_gdal_gcs_credentials(os.environ):
        return None
    candidate = default_adc_candidate()
    if candidate is None:
        return None
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = candidate
    return candidate


__all__ = [
    "default_adc_candidate",
    "ensure_gdal_gcs_auth",
    "gdal_gcs_env_patch",
    "has_gdal_gcs_credentials",
]
