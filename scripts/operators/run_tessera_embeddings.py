#!/usr/bin/env python3
"""Operator: publish one annual TESSERA embedding year (VM/GCS only, not CI).

Reads TESSERA v1.1-dclimate for the Berlin bbox and a single year
(2017–2025; 2026 refused), reprojects the quantized int8+scale fields onto
the canonical EPSG:25833 10 m grid with nearest resampling, and writes the
canonical product under::

    gs://berlin-lst-training-data/embeddings/tessera/v1.1-dclimate/<year>/

Code only in this PR: the operator is NOT executed here. The VM launcher
``.opencode/skills/google-access/scripts/run-tessera-embeddings-vm.sh``
runs this script on the On-Demand VM. Never run against the canonical
prefix from a workstation.

Usage (on the VM only)
----------------------
    uv run python scripts/operators/run_tessera_embeddings.py \
        --year 2024 --output-root gs://berlin-lst-training-data/embeddings/tessera/v1.1-dclimate
    uv run python scripts/operators/run_tessera_embeddings.py \
        --year 2024 --output-root /workspace/tessera-out  # dry-run to local disk

The local-disk form is the only offline-safe path; GCS writes require ADC
and the operator run. After a year is published, add its directory URI to
``_FROZEN_TESSERA_ROOTS`` in ``data/ard/publish_guard.py`` so republish
paths fail closed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from berlin_lst_downscaling.data.ard.publish_guard import assert_tessera_output_root
from berlin_lst_downscaling.data.embeddings.product import (
    TESSERA_PRODUCT_PREFIX,
    product_completion_uri,
    product_provenance_uri,
)
from berlin_lst_downscaling.data.embeddings.reproject import (
    canonical_10m_shape_transform,
    reproject_quantized_to_canonical,
)
from berlin_lst_downscaling.data.embeddings.tessera import (
    TESSERA_BBOX,
    TesseraYearError,
    read_tessera_region,
    validate_tessera_year,
)
from berlin_lst_downscaling.data.io import RunLogSession, log_event

_logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse operator arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, help="Embedding year 2017-2025")
    parser.add_argument(
        "--output-root",
        default=TESSERA_PRODUCT_PREFIX,
        help="Product root (canonical GCS prefix or local dry-run directory)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.0,
        help="Minimum accepted TESSERA quality value (documented default 0.0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the single-year TESSERA publish (operator only)."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        validate_tessera_year(int(args.year))
    except TesseraYearError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    output_root = str(args.output_root)
    if output_root.startswith("gs://"):
        try:
            assert_tessera_output_root(f"{output_root.rstrip('/')}/{int(args.year)}")
        except Exception as exc:  # noqa: BLE001 — fail closed on published prefixes
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2

    session = RunLogSession(output_root, "tessera_embeddings")
    with session:
        log_event(
            session,
            "tessera_read_start",
            year=int(args.year),
            bbox=list(TESSERA_BBOX),
            quality_threshold=float(args.quality_threshold),
        )
        read = read_tessera_region(
            int(args.year),
            quality_threshold=float(args.quality_threshold),
        )
        log_event(
            session,
            "tessera_read_done",
            year=int(args.year),
            quantized=bool(read.quantized is not None),
            source_ids=list(read.provenance.source_ids),
            library_version=read.provenance.library_version,
        )
        shape, transform, _ = canonical_10m_shape_transform()
        if read.quantized is None or read.scale is None:
            raise RuntimeError("quantized TESSERA path returned no int8+scale payload")
        quantized_canon, scale_canon = reproject_quantized_to_canonical(
            read.quantized,
            read.scale,
            read.transform,
            read.crs,
            dst_shape=shape,
            dst_transform=transform,
        )
        valid_canon = __import__("numpy").isfinite(scale_canon)
        log_event(
            session,
            "tessera_reproject_done",
            year=int(args.year),
            shape=[int(shape[0]), int(shape[1])],
            valid_px=int(valid_canon.sum()),
        )
        if output_root.startswith("gs://"):
            print(
                "GCS publish path resolved but not executed from this context; "
                f"year={args.year} root={output_root} "
                f"provenance={json.dumps(read.provenance.as_dict(), sort_keys=True)} "
                f"completion={product_completion_uri(output_root, int(args.year))} "
                f"provenance_uri={product_provenance_uri(output_root, int(args.year))}"
            )
        else:
            from berlin_lst_downscaling.data.embeddings.product import (
                write_tessera_year_product,
            )

            artifacts = write_tessera_year_product(
                quantized=quantized_canon,
                scale=scale_canon,
                valid=valid_canon,
                transform=transform,
                crs="EPSG:25833",
                output_dir=f"{output_root.rstrip('/')}/{int(args.year)}",
                year=int(args.year),
                provenance=read.provenance.as_dict(),
            )
            print(f"Wrote local dry-run product: {artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
