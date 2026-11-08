#!/usr/bin/env bash
# Bound naive evidence helpers (sourced; functions only, no side effects).
#
# Record a sha256 manifest of the naive evidence on the box, bind the manifest sha256, upload the
# report bytes with a remote sha256 check (report sha taken from the bound manifest), and pass the
# upload explicitly to the validator.
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

naive_evidence_record() { # <evidence_dir>: writes naive_evidence_manifest.sha256, prints bind args
  local dir="$1" manifest="$1/naive_evidence_manifest.sha256" rel sha
  local contexts=("$dir"/logs/baseline/*.context.json)
  if (( ${#contexts[@]} != 1 )) || [[ ! -f "${contexts[0]}" ]]; then
    echo "ERROR: expected exactly one logs/baseline/*.context.json in $dir" >&2; return 1
  fi
  for rel in baseline_report.json validation.txt; do
    [[ -f "$dir/$rel" ]] || { echo "ERROR: evidence file missing: $dir/$rel" >&2; return 1; }
  done
  local lines=""
  for rel in baseline_report.json validation.txt "logs/baseline/$(basename "${contexts[0]}")"; do
    sha=$(naive_local_sha256 "$dir/$rel") || return 1
    [[ "$sha" =~ ^[0-9a-f]{64}$ ]] || { echo "ERROR: could not hash $dir/$rel" >&2; return 1; }
    lines+="$sha  $rel"$'\n'
  done
  printf '%s' "$lines" > "$manifest" || return 1
  sha=$(naive_local_sha256 "$manifest") || return 1
  echo "  Naive evidence: $dir/baseline_report.json"
  echo "  Naive manifest: $manifest (covers report, validation.txt, provenance context)"
  echo "  Naive sha256:   $sha (sha256 of the manifest)"
  echo "  Bind RF with:   --naive-report '$dir/baseline_report.json' --naive-sha256 $sha"
  echo "  Do not delete or modify $dir before the RF steps (rf-smoke, rf-full) have finished."
}

naive_manifest_report_sha() { # <manifest> <bound_sha> -> report sha from the bound manifest
  local got line
  [[ -f "$1" ]] || { echo "ERROR: naive manifest missing: $1" >&2; return 1; }
  got=$(naive_local_sha256 "$1")
  if [[ "$got" != "$2" ]]; then echo "ERROR: naive manifest sha256 $got != bound $2" >&2; return 1; fi
  line=$(awk '$2 == "baseline_report.json" && NF == 2 { print $1 }' "$1")
  if [[ ! "$line" =~ ^[0-9a-f]{64}$ ]]; then echo "ERROR: manifest has no single baseline_report.json entry" >&2; return 1; fi
  printf '%s\n' "$line"
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
