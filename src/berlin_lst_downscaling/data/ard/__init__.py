"""ARD processing — write-out pipeline (contracts, masking, COG/STAC, ledger, orchestration).

Phase A: Landsat C2-L2 + Sentinel-2 L2A via PC STAC (ECOSTRESS Phase B).
"""

# Submodules are imported as they are created (Tasks 2–6).
# No forward imports here — they'd break lint/typecheck before modules exist.
