# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Generation-guarded retirement of a superseded feature release.

Retires the exact ``features/v2/`` prefix after the V3 release is
accepted. Two phases, both with an explicit inventory hash:

1. ``plan`` — lists every object under the allowlisted prefix and builds
   an inventory hash over ``(name, generation, size, md5)``. Writes the
   plan evidence OUTSIDE the deleted prefix
   (``qa/retirements/v2/<timestamp>/plan.json``) and prints the hash.
2. ``delete --confirm-inventory <hash>`` — re-lists the live prefix,
   recomputes the hash, and refuses to delete a single object unless it
   matches the approved hash exactly. Each object is deleted with its
   recorded generation as a precondition, so a concurrent replacement
   aborts the retirement (412) instead of deleting the wrong generation.

The prefix is a fixed allowlist: the tool never deletes outside
``features/v2/`` and never touches ``qa/`` evidence. ``verify`` confirms
the prefix is empty afterwards.

Usage
-----
    uv run python scripts/retire_feature_release.py plan
    uv run python scripts/retire_feature_release.py delete --confirm-inventory <hash>
    uv run python scripts/retire_feature_release.py verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime

from google.api_core.exceptions import GoogleAPIError, PreconditionFailed
from google.cloud import storage

_BUCKET = "berlin-lst-data"
_PREFIX = "features/v2/"
_EVIDENCE_ROOT = "qa/retirements/v2"


def _client() -> storage.Client:
    return storage.Client()


def _inventory_hash(rows: list[dict]) -> str:
    """SHA-256 over the sorted (name, generation, size, md5) rows."""
    payload = json.dumps(
        sorted((r["name"], r["generation"], r["size"], r["md5"]) for r in rows),
        sort_keys=True,
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _list_rows() -> list[dict]:
    bucket = _client().get_bucket(_BUCKET)
    rows = []
    for blob in bucket.list_blobs(prefix=_PREFIX):
        rows.append(
            {
                "name": blob.name,
                "generation": blob.generation,
                "size": blob.size,
                "md5": blob.md5_hash,
            }
        )
    return rows


def _write_evidence(payload: dict, label: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    uri = f"gs://{_BUCKET}/{_EVIDENCE_ROOT}/{ts}/{label}.json"
    bucket = _client().get_bucket(_BUCKET)
    bucket.blob(f"{_EVIDENCE_ROOT}/{ts}/{label}.json").upload_from_string(
        json.dumps(payload, indent=2)
    )
    print(f"  Evidence: {uri}")
    return uri


def _cmd_plan() -> int:
    rows = _list_rows()
    total_bytes = sum(r["size"] for r in rows)
    inventory_hash = _inventory_hash(rows)
    payload = {
        "prefix": _PREFIX,
        "timestamp": datetime.now(UTC).isoformat(),
        "object_count": len(rows),
        "total_bytes": total_bytes,
        "inventory_hash": inventory_hash,
        "objects": rows,
    }
    _write_evidence(payload, "plan")
    print(f"  Prefix: {_PREFIX}")
    print(f"  Objects: {len(rows)} | Bytes: {total_bytes:,}")
    print(f"  INVENTORY HASH: {inventory_hash}")
    print("Confirm deletion with: --confirm-inventory <hash>")
    return 0


def _cmd_delete(args) -> int:
    rows = _list_rows()
    inventory_hash = _inventory_hash(rows)
    if inventory_hash != args.confirm_inventory:
        print(
            f"ERROR: live inventory hash {inventory_hash} != confirmed "
            f"{args.confirm_inventory} — aborting, nothing deleted."
        )
        return 1

    bucket = _client().get_bucket(_BUCKET)
    deleted: list[dict] = []
    for row in rows:
        blob = bucket.blob(row["name"])
        try:
            blob.delete(if_generation_match=row["generation"])
            deleted.append(row)
        except PreconditionFailed:
            print(
                f"ABORT: {row['name']} changed generation since the plan "
                f"(expected {row['generation']}) — stopping, nothing further deleted."
            )
            return 1
        except GoogleAPIError as exc:
            print(f"ABORT: deletion failed for {row['name']}: {exc}")
            return 1

    receipt = {
        "prefix": _PREFIX,
        "inventory_hash": inventory_hash,
        "deleted_count": len(deleted),
        "deleted_bytes": sum(r["size"] for r in deleted),
        "completed_at": datetime.now(UTC).isoformat(),
        "deleted_objects": deleted,
    }
    _write_evidence(receipt, "receipt")
    print(f"  Deleted {len(deleted)} objects under {_PREFIX}")
    return 0


def _cmd_verify() -> int:
    rows = _list_rows()
    if rows:
        print(f"ERROR: {len(rows)} objects remain under {_PREFIX} (first: {rows[0]['name']})")
        return 1
    print(f"OK: {_PREFIX} is empty.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    delete_p = sub.add_parser("delete")
    delete_p.add_argument("--confirm-inventory", required=True, help="Approved inventory hash")
    sub.add_parser("verify")
    args = parser.parse_args()

    if args.command == "plan":
        return _cmd_plan()
    if args.command == "delete":
        return _cmd_delete(args)
    return _cmd_verify()


if __name__ == "__main__":
    raise SystemExit(main())