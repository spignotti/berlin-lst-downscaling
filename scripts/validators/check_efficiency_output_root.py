"""Check efficiency output-root claim and stale-artifact guards locally."""

from __future__ import annotations

import os
import runpy
import tempfile
from pathlib import Path
from unittest.mock import patch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHECK_ROOT = runpy.run_path(
    str(_PROJECT_ROOT / "scripts/validators/measure_training_efficiency.py")
)["_assert_fresh_output_root"]


def main() -> int:
    temporary_parent = Path(os.environ["TMPDIR"]) / "opencode"
    if not temporary_parent.is_dir():
        print(f"Temporary parent does not exist: {temporary_parent}")
        return 1
    try:
        with tempfile.TemporaryDirectory(
            dir=temporary_parent, prefix="efficiency-root-check-"
        ) as d:
            root = Path(d) / "new-run"
            _CHECK_ROOT(root, "new-run")
            if not root.is_dir() or any(root.iterdir()):
                raise RuntimeError("missing output root was not created empty")
            try:
                _CHECK_ROOT(root, "new-run")
            except FileExistsError:
                pass
            else:
                raise RuntimeError("unclaimed existing output root was accepted")

            log_root = Path(d) / "claimed-run"
            log = log_root / "logs/modeling/vertex-entrypoint.txt"
            log.parent.mkdir(parents=True)
            log.write_text("synthetic entrypoint output\n", encoding="utf-8")
            with patch.dict(os.environ, {"VERTEX_EFFICIENCY_OUTPUT_CLAIMED": "claimed-run"}):
                _CHECK_ROOT(log_root, "claimed-run")
                (log_root / "data_scope.json").write_text("{}", encoding="utf-8")
                try:
                    _CHECK_ROOT(log_root, "claimed-run")
                except FileExistsError:
                    pass
                else:
                    raise RuntimeError("claimed output root with prior results was accepted")

            symlink_root = Path(d) / "linked-run"
            symlink_root.symlink_to(log_root, target_is_directory=True)
            with patch.dict(os.environ, {"VERTEX_EFFICIENCY_OUTPUT_CLAIMED": "linked-run"}):
                try:
                    _CHECK_ROOT(symlink_root, "linked-run")
                except FileExistsError:
                    pass
                else:
                    raise RuntimeError("symlink output root was accepted")
    except Exception as exc:
        print(f"OUTPUT-ROOT SELF-CHECK FAILED: {exc}")
        return 1
    print("OUTPUT-ROOT SELF-CHECK OK: empty claim, allowed log, stale data, and symlink cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
