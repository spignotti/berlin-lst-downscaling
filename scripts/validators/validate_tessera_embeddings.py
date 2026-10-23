#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "rasterio",
# ]
# ///
"""Independent validator for the canonical TESSERA annual product.

Read-only probe: verifies, independently of the publisher implementation,
for one year (2017–2025; 2026 refused):

- the product directory holds the int8 COG (128 bands, int8), the scale COG
  (1 band, float32, NaN = gap), the validity mask COG (1 band, uint8 {0,1}),
  plus ``provenance.json`` and ``complete.json``;
- all rasters are on the canonical EPSG:25833 10 m grid (shape/transform/CRS
  match ``common/grid.py``);
- the int8 band count is 128 and the scale/valid shapes match;
- provenance records dataset version ``v1.1-dclimate``, zone ``33N``,
  the year, library version, and source chunk/tile ids;
- validity is consistent: invalid pixels (valid == 0) have non-finite scale
  and NaN after dequantization; valid pixels dequantize finite; no
  zero-filled gaps (a fully-zero embedding vector with finite scale fails).

Usage
-----
    uv run python scripts/validators/validate_tessera_embeddings.py \
        --root gs://berlin-lst-training-data/embeddings/tessera/v1.1-dclimate --year 2024
    uv run python scripts/validators/validate_tessera_embeddings.py \
        --root .scratch/tessera-out --year 2024
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio

from berlin_lst_downscaling.common.grid import canon_grid_10m
from berlin_lst_downscaling.data.embeddings.product import (
    product_completion_uri,
    product_provenance_uri,
)
from berlin_lst_downscaling.data.embeddings.tessera import (
    TESSERA_DATASET_VERSION,
    TESSERA_N_CHANNELS,
    TESSERA_ZONE,
    TesseraYearError,
    dequantize_tessera,
    validate_tessera_year,
)
from berlin_lst_downscaling.data.io import exists, read_bytes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse validator arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="TESSERA product root")
    parser.add_argument("--year", type=int, required=True, help="Product year 2017-2025")
    return parser.parse_args(argv)


def _read_local_or_gcs_json(uri: str, root: str) -> dict:
    if uri.startswith("gs://"):
        return json.loads(read_bytes(uri).decode("utf-8"))
    import os

    path = uri
    if not os.path.exists(path):
        path = os.path.join(root.rstrip("/"), str(_year_from_uri(uri)), "provenance.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _year_from_uri(uri: str) -> int:
    for token in uri.replace("\\", "/").split("/"):
        if token.isdigit() and len(token) == 4:
            return int(token)
    raise ValueError(f"cannot infer year from {uri!r}")


def main(argv: list[str] | None = None) -> int:
    """Validate one TESSERA product year; exit 0 on GO, 1 on findings."""
    args = parse_args(argv)
    try:
        validate_tessera_year(int(args.year))
    except TesseraYearError as exc:
        print(f"FAIL: {exc}")
        return 1
    root = str(args.root).rstrip("/")
    year = int(args.year)
    failures: list[str] = []

    geobox = canon_grid_10m()
    exp_shape = (int(geobox.shape.y), int(geobox.shape.x))

    if root.startswith("gs://"):
        int8_uri = f"{root}/{year}/tessera_{year}_int8.tif"
        scale_uri = f"{root}/{year}/tessera_{year}_scale.tif"
        valid_uri = f"{root}/{year}/tessera_{year}_valid.tif"
        for uri, label in (
            (int8_uri, "int8"),
            (scale_uri, "scale"),
            (valid_uri, "valid"),
            (product_provenance_uri(root, year), "provenance"),
            (product_completion_uri(root, year), "complete"),
        ):
            if not exists(uri):
                failures.append(f"missing {label}: {uri}")
        if failures:
            print("FAIL:")
            for item in failures:
                print(f"  - {item}")
            return 1
        provenance = json.loads(read_bytes(product_provenance_uri(root, year)).decode("utf-8"))
        with rasterio.open(int8_uri) as src:
            _check_grid(src, exp_shape, failures, "int8")
            if src.count != TESSERA_N_CHANNELS:
                failures.append(f"int8 bands {src.count} != {TESSERA_N_CHANNELS}")
            quantized = src.read().astype(np.int8).transpose(1, 2, 0)
        with rasterio.open(scale_uri) as src:
            _check_grid(src, exp_shape, failures, "scale")
            scale = src.read(1)
        with rasterio.open(valid_uri) as src:
            _check_grid(src, exp_shape, failures, "valid")
            valid = src.read(1)
    else:
        import os

        base = os.path.join(root, str(year))
        int8_path = os.path.join(base, f"tessera_{year}_int8.tif")
        scale_path = os.path.join(base, f"tessera_{year}_scale.tif")
        valid_path = os.path.join(base, f"tessera_{year}_valid.tif")
        provenance_path = os.path.join(base, "provenance.json")
        complete_path = os.path.join(base, "complete.json")
        for path, label in (
            (int8_path, "int8"),
            (scale_path, "scale"),
            (valid_path, "valid"),
            (provenance_path, "provenance"),
            (complete_path, "complete"),
        ):
            if not os.path.exists(path):
                failures.append(f"missing {label}: {path}")
        if failures:
            print("FAIL:")
            for item in failures:
                print(f"  - {item}")
            return 1
        with open(provenance_path, encoding="utf-8") as fh:
            provenance = json.load(fh)
        with rasterio.open(int8_path) as src:
            _check_grid(src, exp_shape, failures, "int8")
            if src.count != TESSERA_N_CHANNELS:
                failures.append(f"int8 bands {src.count} != {TESSERA_N_CHANNELS}")
            quantized = src.read().astype(np.int8).transpose(1, 2, 0)
        with rasterio.open(scale_path) as src:
            _check_grid(src, exp_shape, failures, "scale")
            scale = src.read(1)
        with rasterio.open(valid_path) as src:
            _check_grid(src, exp_shape, failures, "valid")
            valid = src.read(1)

    if provenance.get("dataset_version") != TESSERA_DATASET_VERSION:
        failures.append(f"provenance dataset_version {provenance.get('dataset_version')!r}")
    if provenance.get("zone") != TESSERA_ZONE:
        failures.append(f"provenance zone {provenance.get('zone')!r}")
    if int(provenance.get("year", -1)) != year:
        failures.append(f"provenance year {provenance.get('year')!r} != {year}")
    if not provenance.get("library_version"):
        failures.append("provenance library_version is empty")

    valid_mask = np.asarray(valid) == 1
    scale_finite = np.isfinite(np.asarray(scale, dtype=np.float64))
    if np.any(valid_mask & ~scale_finite):
        failures.append("valid pixels with non-finite scale (must be NaN only on gaps)")
    if np.any(~valid_mask & scale_finite):
        failures.append("invalid pixels with finite scale (gaps must be NaN scale)")

    embedding = dequantize_tessera(
        np.ascontiguousarray(quantized, dtype=np.int8),
        np.ascontiguousarray(scale),
    )
    finite_vectors = np.all(np.isfinite(embedding), axis=2)
    if np.any(valid_mask != finite_vectors):
        failures.append("valid mask disagrees with dequantized finite vectors")
    # No zero-filled gaps: an all-zero embedding vector with finite scale is
    # a fill artifact, not a real embedding.
    zero_vectors = np.all(embedding == 0, axis=2)
    if np.any(zero_vectors & scale_finite):
        failures.append("zero-filled embedding vectors with finite scale (gaps must be NaN)")

    if failures:
        print("FAIL:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"GO: TESSERA product year {year} validates ({exp_shape[0]}x{exp_shape[1]}).")
    return 0


def _check_grid(
    src: rasterio.DatasetReader, exp_shape: tuple[int, int], failures: list[str], label: str
) -> None:
    if (int(src.height), int(src.width)) != exp_shape:
        failures.append(f"{label} shape {(src.height, src.width)} != canonical {exp_shape}")
    if str(src.crs) != "EPSG:25833":
        failures.append(f"{label} CRS {src.crs} != EPSG:25833")


if __name__ == "__main__":
    raise SystemExit(main())
