#!/usr/bin/env bash
# Fail when `gcloud builds submit .` would not upload a file the Vertex image needs.
#
# Compares `gcloud meta list-files-for-upload .` (the filter .gcloudignore
# applies) with the git-tracked files under src/, scripts/, and configs/ plus
# pyproject.toml, uv.lock, and Dockerfile.vertex, and fails if `.env*` (except
# .env.example), root data/, or .git would be uploaded. Local only; needs no auth.
# Run from anywhere inside the repository before every `gcloud builds submit`.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [[ ! -f .gcloudignore ]]; then
  echo "missing .gcloudignore: gcloud would derive the upload filter from .gitignore" >&2
  exit 1
fi

uploaded="$(mktemp)"
required="$(mktemp)"
trap 'rm -f "$uploaded" "$required"' EXIT

gcloud meta list-files-for-upload . | sed 's|^\./||' | sort -u >"$uploaded"
{
  git -c core.quotePath=false ls-files -- src scripts configs
  printf '%s\n' pyproject.toml uv.lock Dockerfile.vertex
} | sort -u >"$required"

missing="$(comm -23 "$required" "$uploaded")"
if [[ -n "$missing" ]]; then
  echo "files missing from the Cloud Build upload:" >&2
  printf '%s\n' "$missing" >&2
  exit 1
fi

forbidden="$(grep -E '(^|/)\.env(\..*)?$|^data/|^\.git(/|$)' "$uploaded" | grep -vE '(^|/)\.env\.example$' || true)"
if [[ -n "$forbidden" ]]; then
  echo "files that must not be uploaded:" >&2
  printf '%s\n' "$forbidden" >&2
  exit 1
fi
echo "build context OK: $(wc -l <"$required") required files uploaded"
