"""Offline defaults for the unit suite.

Imported before test modules. Keeps PROJ and W&B from attempting a network
call if a test imports a module that would otherwise check those services.
"""

from __future__ import annotations

import os

import pytest

from berlin_lst_downscaling.modeling import ssim_range

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("PROJ_NETWORK", "OFF")


FILLED_TEST_FINGERPRINTS: dict[str, str] = {
    "index_policy_hash": "test-index-policy-hash",
    "source_policy_hash": "test-source-policy-hash",
}
FILLED_TEST_DATA_RANGE_K = 40.0


def write_filled_ssim_artifact(path, *, fingerprints=None):  # noqa: ANN001, ANN201
    """Write a synthetic FILLED SSIM data-range artifact (test-only values)."""
    import yaml

    doc = {
        "schema_version": 1,
        "version": "v1",
        "status": "filled",
        "quantity": "ssim_data_range",
        "split": "train",
        "years": list(range(2017, 2024)),
        "derived_at": "2026-01-01T00:00:00+00:00",
        "p_low": 0.5,
        "p_high": 99.5,
        "percentile_method": "linear",
        "low_k": 280.0,
        "high_k": 320.0,
        "data_range_k": FILLED_TEST_DATA_RANGE_K,
        "patch_index_fingerprints": dict(fingerprints or FILLED_TEST_FINGERPRINTS),
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _filled_ssim_artifact(tmp_path_factory, monkeypatch):  # noqa: ANN001, ANN202
    """Test-only filled artifact; production reads the committed artifact.

    Tests that need the committed placeholder re-monkeypatch
    ``ssim_range.SSIM_DATA_RANGE_ARTIFACT`` to the real path.
    """
    path = write_filled_ssim_artifact(tmp_path_factory.mktemp("ssim_artifact") / "v1.yaml")
    monkeypatch.setattr(ssim_range, "SSIM_DATA_RANGE_ARTIFACT", path)
    return path


@pytest.fixture
def write_ssim_artifact():  # noqa: ANN201
    """Return the writer for a synthetic filled artifact (``path, fingerprints=None``)."""
    return write_filled_ssim_artifact
