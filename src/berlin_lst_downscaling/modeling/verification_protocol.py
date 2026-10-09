"""Fixed identity and bounds for the single Stage-1 setup verification job."""

VERIFICATION_SESSION_ID = "stage1-setup-verification-v1"
VERIFICATION_RUN_LABEL = "stage1-setup-verification-v1"
VERIFICATION_CONFIG_NAME = "stage1_verification"

VERIFICATION_TIMEOUT_SECONDS = 2700
VERIFICATION_MAX_WAIT_SECONDS = 1800
VERIFICATION_MAX_HOURLY_RATE_USD = 1.0
VERIFICATION_MAX_JOB_EXPOSURE_USD = 1.25
VERIFICATION_MAX_NONCOMPUTE_USD = 2.0
# Worker stops starting measurements after this many server seconds, keeping
# the remainder for evidence upload and finalization.
VERIFICATION_MEASUREMENT_BUDGET_SECONDS = 2400

VERIFICATION_MACHINE_TYPE = "n1-standard-4"
VERIFICATION_BOOT_DISK_SIZE_GB = 100
VERIFICATION_BOOT_DISK_TYPE = "pd-ssd"
# Calibration artifacts share the approved QA evidence root, never retained QA.
VERIFICATION_BUCKET = "berlin-lst-training-data"
VERIFICATION_ARTIFACT_PREFIX = "qa/modeling/vertex-smoke"

# Bounded real-data union: the already-measured performance IDs plus the
# already-measured learning IDs. No refill, no new cohorts.
VERIFICATION_MAX_TRAIN_REFS = 512
VERIFICATION_MAX_VALIDATION_REFS = 192

# Full-size synthetic pressure (train/validation split mirrors the request).
VERIFICATION_SYNTHETIC_TRAIN = 6873
VERIFICATION_SYNTHETIC_VALIDATION = 1342
# At most four warm-up updates per window; 3 windows x 32 updates x 2 precisions.
VERIFICATION_MAX_SYNTHETIC_UPDATES = 216
VERIFICATION_BALLAST_STEP_BYTES = 64 * 1024**2
VERIFICATION_BALLAST_CAP_BYTES = 8 * 1024**3
# Pressure counts only with an observed file-cache budget at most this
# fraction of the cache payload and backing reads of at least this fraction.
VERIFICATION_FILE_BUDGET_FRACTION = 0.75
VERIFICATION_BACKING_READ_FRACTION = 0.10
VERIFICATION_MEMORY_RESERVE_BYTES = 2 * 1024**3
VERIFICATION_MIN_MEM_AVAILABLE_BYTES = 1024**3

# Numerical tolerances (docs/archive/stage1-training-readiness.md).
VERIFICATION_IDENTITY_TOLERANCE_K = 1e-3
VERIFICATION_RELOAD_TOLERANCE_K = 1e-3
VERIFICATION_MAE_TOLERANCE_K = 1e-3
VERIFICATION_SSIM_TOLERANCE = 1e-6
VERIFICATION_SYNTHETIC_SSIM_TOLERANCE = 1e-3
# Mixed precision is selected only above this conservative-high lifecycle gain
# over cached FP32 under identical assumptions.
VERIFICATION_MIN_AMP_GAIN = 0.10

# Retained J4 artifact identity, asserted from the downloaded artifact.
VERIFICATION_J4_SSIM_WINDOWS = 5849
VERIFICATION_J4_VALID_CELLS = 16112
VERIFICATION_J4_BEST_MAE_K = 1.1623233556747437
