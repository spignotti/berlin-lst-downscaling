#!/usr/bin/env python3
"""Read-only feature-correctness diagnostics.

Prints a cost estimate, then writes ``summary.json`` and ``summary.md``
only when ``--yes`` is passed. ``--estimate-only`` prints the estimate
and exits without writing. ``--metadata-only`` runs the Sentinel-2
processing-baseline check from STAC properties and a single local LoD2
tile. It refuses the eligibility-gap raster scan.

This process never starts a virtual machine or a cloud training job, and
it never writes a feature, ARD, or training release. Uploads are accepted
only under ``gs://berlin-lst-training-data/evaluation/diagnostics/<run-id>/``.

Usage
-----
uv run python scripts/operators/diagnose_feature_correctness.py \\
    --scope s2-baseline --scenes-json scenes.json --estimate-only

uv run python scripts/operators/diagnose_feature_correctness.py \\
    --scope s2-baseline --scope lod2-parser \\
    --metadata-only --yes \\
    --scenes-json scenes.json --lod2-tile tile.gml \\
    --output-dir /tmp/feature-diagnostics
"""

from __future__ import annotations

from berlin_lst_downscaling.data.diagnostics.feature_correctness import main

if __name__ == "__main__":
    raise SystemExit(main())
