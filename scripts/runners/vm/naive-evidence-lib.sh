#!/usr/bin/env bash
# Bound naive evidence helpers (sourced; functions only, no side effects).
#
# Record on the box, upload with a remote sha256 check, pass explicitly to the validator.
# The caller must define:
#   ssh_cmd "<remote cmd>"          run a remote command (stdin is not forwarded)
#   ssh_upload_cmd "<remote cmd>"   same, but forwards the caller's stdin
# The naive comparison can never be skipped by a launcher.

naive_local_sha256() { # <file> -> hex
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

naive_evidence_record() { # <report>: writes <report>.sha256, prints path/sha/bind args
  local report="$1" sha
  sha=$(naive_local_sha256 "$report")
  printf '%s  %s\n' "$sha" "$(basename "$report")" > "$report.sha256"
  echo "  Naive evidence: $report"
  echo "  Naive sha256:   $sha"
  echo "  Bind RF with:   --naive-report '$report' --naive-sha256 $sha"
}

naive_evidence_check_remote() { # <sha> <remote_path>; uses the caller's ssh_cmd
  local got
  got=$(ssh_cmd "sha256sum '$2'" | cut -d' ' -f1) || got=""
  if [[ -z "$got" || "$got" != "$1" ]]; then
    echo "ERROR: remote naive evidence sha256 '${got:-missing}' != bound $1" >&2
    return 1
  fi
}

naive_evidence_push() { # <local> <sha> <remote_path>; uses ssh_upload_cmd (stdin) and ssh_cmd
  if [[ "$(naive_local_sha256 "$1")" != "$2" ]]; then
    echo "ERROR: local naive evidence changed after verification" >&2
    return 1
  fi
  ssh_upload_cmd "mkdir -p '$(dirname "$3")' && cat > '$3'" < "$1" \
    || { echo "ERROR: naive evidence upload failed" >&2; return 1; }
  naive_evidence_check_remote "$2" "$3"
}

rf_validation_remote_cmd() { # <app_dir> <seconds> <report> [<naive_remote>] -> command string
  local cmd="cd '$1' && timeout --signal=TERM --kill-after=60s ${2}s uv run python scripts/validators/validate_random_forest.py --report '$3' --max-patches 5"
  if [[ -n "${4:-}" ]]; then cmd="$cmd --naive-report '$4'"; fi
  printf '%s' "$cmd"
}
