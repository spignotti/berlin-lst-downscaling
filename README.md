# berlin-lst-downscaling

> Reproducible deep-learning pipeline that downscales Landsat land surface temperature from ~100 m to 10 m for Berlin via a fixed 2D U-Net and a 5-stage feature ablation.

100 m thermal data from Landsat TIRS is too coarse for block-level urban heat analysis. This project produces a 10 m LST time series plus a published model, using urban-context features (spectral indices, morphology, shadow/solar geometry, meteorology) evaluated through cumulative ablation.

## Setup

```bash
uv sync
```

## Development

```bash
uv run nox -s fix        # lint and format
uv run nox -s test       # run tests
uv run nox               # full validation
```
