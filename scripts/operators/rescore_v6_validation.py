"""Rescore the 27 frozen v6 best checkpoints on the 2024 validation split only.

Inference only: no optimizer step, no new checkpoint, and no test (2025) ref,
metric, or export. Writes one per-patch validation export per checkpoint plus a
manifest, uploaded create-only (exports first, manifest last). Runs inside the
Vertex job (``VERTEX_PROFILE=val-rescore``); not run in CI.

Usage
-----
    uv run python scripts/operators/rescore_v6_validation.py --inventory-only
    uv run python scripts/operators/rescore_v6_validation.py \\
        --output-root data/runs/v6-val-rescore/<label> \\
        --source-sha <40-hex> --image-digest sha256:<64-hex> \\
        --manifest-uri gs://.../manifest.json --device cuda

``--inventory-only`` prints the 27 checkpoint -> export pairs and exits without
touching GCS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from berlin_lst_downscaling.modeling.v6_val_rescore import rescore_inventory, run_rescore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--image-digest")
    parser.add_argument("--manifest-uri")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--inventory-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.inventory_only:
        for run in rescore_inventory():
            print(
                f"{run.arm_id} seed{run.seed} {run.run_label} "
                f"{run.checkpoint_uri} -> {run.export_uri}"
            )
        return 0
    required = ("output_root", "source_sha", "image_digest", "manifest_uri")
    missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"required unless --inventory-only: {', '.join(missing)}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ERROR: --device cuda requested but CUDA is not available")
    manifest = run_rescore(
        output_root=args.output_root,
        source_sha=args.source_sha,
        image_digest=args.image_digest,
        manifest_uri=args.manifest_uri,
        device=args.device,
    )
    manifest_bytes = (args.output_root / "manifest.json").read_bytes()
    runs = manifest["runs"]
    print(f"manifest sha256: {hashlib.sha256(manifest_bytes).hexdigest()}")
    print(f"runs rescored: {len(runs)}; split: validation; evaluate_test: false")
    print(json.dumps(manifest["patch_cache"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
