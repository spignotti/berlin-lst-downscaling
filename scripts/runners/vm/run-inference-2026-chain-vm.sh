#!/usr/bin/env bash
# Warm-season 2026 inference chain on berlin-lst-vm.
#
# One session, in order: selection, ARD seed, S2 mosaic, Landsat gaps,
# dynamic v6, feature stacks. New prefixes only. The published
# features/v6 training release is not a target. Stops on the first
# remote failure (the steps are chained with &&).
#
# Landsat rows, and the six fixed 2018 ECOSTRESS validation rows the
# selection contract always writes, are seeded from the published ARD v10
# ledger (seed_ard_v10_reuse.py, existing CLI). S2 mosaic rows are not
# seeded: the mosaic schema hash is per scene, so every 2026 S2 scene is
# reprocessed. The Landsat pass does not run ECOSTRESS.
#
# The VM is stopped after a recorded pipeline or validation exit. A
# connection loss or poll timeout exits 2 from vm-runner-common.sh and
# leaves the VM running, same as the other launchers.
#
# Usage:
#   run-inference-2026-chain-vm.sh [branch]

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/_logging.sh"
source "$(cd "$(dirname "$0")" && pwd)/vm-runner-common.sh"

BRANCH="${1:-main}"
PIPELINE_LABEL="Inference 2026 chain"
MARKER_CONFIG="inference_2026"
MANIFEST_ROOT="gs://berlin-lst-training-data/manifests/v4-inference/2026-cutoff-20260930T235959Z"
MANIFEST_URI="${MANIFEST_ROOT}/manifest.parquet"
ARD_ROOT="gs://berlin-lst-training-data/ard/inference/2026-s2-mosaic-v10"
DYNAMIC_ROOT="gs://berlin-lst-training-data/dynamic/inference/2026-v6"
FEATURES_ROOT="gs://berlin-lst-training-data/features/v6-inference-2026"
SEED_LEDGER="gs://berlin-lst-training-data/ard/full/2017-2026-s2-mosaic-v10/ledger.parquet"
STATIC_SOURCES_ROOT="gs://berlin-lst-training-data/static/sources/lod2-area-exact-v2"
STATIC_DERIVED_ROOT="gs://berlin-lst-training-data/static/derived/lod2-area-exact-v2"
SELECTION_CONFIG="inference_2026_v4"
S2_CONFIG="sentinel2_mosaic_v10_inference_2026"
LANDSAT_CONFIG="landsat_v10_inference_2026"
DYNAMIC_CONFIG="v6_inference_2026"
FEATURES_CONFIG="v6_inference_2026"
EXTRA_MARKER_JSON="  \"manifest_uri\": \"$MANIFEST_URI\",
  \"ard_root\": \"$ARD_ROOT\",
  \"dynamic_root\": \"$DYNAMIC_ROOT\",
  \"features_root\": \"$FEATURES_ROOT\",
"
# A rerun after an interrupted chain reuses the immutable selection bundle;
# every later step is idempotent and resumes from its ledger. The S2 mosaic
# and dynamic steps grow in memory over many scenes and can be OOM-killed
# (exit 137); a fresh process resumes from the ledger, up to a fixed count.
# Fully clear 2026 scenes failed the staging write with the default GDAL block
# cache ("error writing a dirty block", one lost block of band 21, caught by
# pair validation); a 6000 MB cache writes them cleanly.
SELECTION_CMD="uv run python scripts/runners/build_manifest.py --config-dir configs/selection --config-name ${SELECTION_CONFIG} && "
if gcloud storage ls "${MANIFEST_ROOT}/pairings.parquet" >/dev/null 2>&1; then
  echo "Selection bundle exists: ${MANIFEST_ROOT} (reused, not rebuilt)"
  SELECTION_CMD=""
fi
REMOTE_CMD="${SELECTION_CMD}uv run python scripts/operators/seed_ard_v10_reuse.py --source-ledger ${SEED_LEDGER} --manifest-uri ${MANIFEST_URI} --dest-root ${ARD_ROOT} --gap-out ${ARD_ROOT}/seed_gaps.json && (for attempt in 1 2 3 4 5; do uv run python scripts/runners/run_ard.py --config-name ${S2_CONFIG} && exit 0; done; exit 1) && uv run python scripts/runners/run_ard.py --config-name ${LANDSAT_CONFIG} && (for attempt in 1 2 3; do uv run python scripts/runners/run_dynamic.py --config-name ${DYNAMIC_CONFIG} && exit 0; done; exit 1) && GDAL_CACHEMAX=6000 uv run python scripts/runners/run_features_isolated.py --config-name ${FEATURES_CONFIG}"
POLL_MAX_SECONDS=172800

vm_init_run "inference-2026"
echo "$PIPELINE_LABEL | Branch: $BRANCH | Run: $WRAP_RUN_ID"
echo "  Selection: $SELECTION_CONFIG"
echo "  ARD:       $S2_CONFIG + $LANDSAT_CONFIG"
echo "  Dynamic:   $DYNAMIC_CONFIG"
echo "  Features:  $FEATURES_CONFIG"
echo "  Manifest:  $MANIFEST_URI"
echo "  Seed:      $SEED_LEDGER (Landsat and ECOSTRESS; S2 is reprocessed)"
echo "  ARD out:   $ARD_ROOT"
echo "  Statics:   $STATIC_SOURCES_ROOT $STATIC_DERIVED_ROOT"
echo "  Dynamic:   $DYNAMIC_ROOT"
echo "  Features:  $FEATURES_ROOT"

echo "Running release preflight..."
if ! uv run python scripts/operators/preflight_republish_v6.py \
  --step inference_2026 \
  --config-dir configs/features \
  --config-name "$FEATURES_CONFIG"; then
  echo "ERROR: release preflight failed — not launching."
  exit 1
fi

vm_start_and_wait_ssh
vm_push_deploy
vm_write_marker "$MARKER_CONFIG"
vm_launch_detached
vm_poll
vm_finish

VALIDATION_OK=1
if [[ "$PIPELINE_EXIT" == "0" ]]; then
  echo "Validating inference 2026 roots ..."
  # GDAL (/vsigs/) needs its own GCS credential while google-cloud-storage
  # uses ADC automatically. Point GDAL at the operator ADC file when no GDAL
  # GCS credential is set (path only — no secret is printed or copied).
  if [[ -z "${GS_SECRET_ACCESS_KEY:-}" && -z "${GS_OAUTH2_REFRESH_TOKEN:-}" && -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
    if [[ -f "$HOME/.config/gcloud/application_default_credentials.json" ]]; then
      export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"
    fi
  fi
  uv run python scripts/validators/validate_ard.py \
    --ledger "${ARD_ROOT}/ledger.parquet" \
    --manifest "$MANIFEST_URI" \
    && uv run python scripts/validators/validate_dynamic.py \
      --output-root "$DYNAMIC_ROOT" \
      --expected-role inference \
      --expected-scenes-from-manifest "$MANIFEST_URI" \
      --include-inference-scenes \
      --check-bands \
      --era5-channel-set v6 \
    && uv run python scripts/validators/validate_feature_stacks.py \
      --root "$FEATURES_ROOT" \
      --aoi data/boundaries/aoi_10m.tif \
      --expected-scenes-from-manifest "$MANIFEST_URI" \
      --ard-ledger "${ARD_ROOT}/ledger.parquet" \
      --include-inference-scenes \
    && VALIDATION_OK=0 || VALIDATION_OK=1
fi

vm_stop

if [[ "$VALIDATION_OK" -eq 0 && "$PIPELINE_EXIT" == "0" ]]; then
  echo "SUCCESS: inference 2026 chain published and validated."
  echo "  Features: $FEATURES_ROOT"
else
  echo "FAIL: inference 2026 chain or validation failed."
  echo "  Features:   $FEATURES_ROOT"
  echo "  Remote log: $REMOTE_LOG"
  exit 1
fi
