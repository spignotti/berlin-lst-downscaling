# Berlin LST Downscaling

Cloud-native land-surface-temperature downscaling pipeline for Berlin. It
combines 100 m thermal observations with fine-resolution predictors to build
a training-ready dataset for high-resolution LST modelling on a canonical
10 m grid (EPSG:25833).

## Architecture

```
Data sources
     │
     ▼
Manifest + ARD ──► Static / dynamic context ──► Feature stacks ──► Modelling (WIP)
```

## Inputs

- Landsat-8/9 Collection 2 Level-2 thermal data (100 m), the anchor target
- Sentinel-2 L2A surface reflectance (10 m), the primary predictor
- ECOSTRESS L2T LSTE v002 (70 m), independent validation
- Berlin 3D city morphology (LoD2 building geometry, terrain height,
  vegetation height, imperviousness) from the Berlin GDI, fixed per-scene
  vintage
- ERA5-Land meteorology via the Copernicus CDS API

## Pipeline

- **Manifest-driven selection.** Versioned, immutable manifest bundles define
  the scene universe and pair each Landsat anchor with its Sentinel-2 partner.
- **ARD processing.** Analysis-ready COGs with NaN-no-data semantics on the
  canonical grid.
- **Static context.** Official archive sources become 10 m geometry products:
  building and vegetation DSMs, shadow horizons, and sky-view factor, per
  geometry vintage.
- **Dynamic context.** Per-anchor ERA5-Land fields and building/vegetation
  shadow masks.
- **Feature stacks.** Per-anchor 28-band 10 m stacks (spectral indices,
  semantic morphology predictors, meteorology, shadows) with co-registered
  validity masks, ready for the training stage.
- **QA gates.** Read-only, mask-free validation of every published input,
  plus independent validators per pipeline stage.

Every product publishes four co-located artifacts: the COG, a STAC Item,
`provenance.json`, and a `complete.json` completion marker written last as
the publication gate. Ledgers in Google Cloud Storage are the reproducibility
basis; the uv lockfile and per-run config fingerprints pin the environment.

## Status

Preprocessing is complete: manifest selection, ARD, static and dynamic
context, the per-anchor feature stacks, and the Stage-2 feature-stack QA
gate are delivered and validated. Training-data preparation (including the
`training_eligible@100m` selection mask) and model training are next.

## Setup

```bash
uv sync
uv run nox -s lint typecheck
```

Local smoke gates (`uv run nox -s smoke-*`) need Google Cloud ADC and, for
the dynamic pipeline, a Copernicus CDS API key. Heavy production runs execute
on a GCP VM against GCS.

## Documentation

- `docs/data-sources-and-contracts.md` — sources, canonical grid, manifest
  and product contracts.