# Thesis figures

Figures for the thesis are built by script here and written to `figures/out/` (git-ignored), as SVG (Typst cannot embed PDF images in PDF/A mode).
Shared style, palette and export live in `src/berlin_lst_downscaling/figures/`.

## Conventions

- One script per figure: `fig_<chapter><nn>_<slug>.py` (for example `fig_0401_holdout_map.py`).
  `fig_00_style_check.py` is the style check and not a thesis figure.
- Start with `style.use_thesis_style()`, use the palette and cmap constants from
  `berlin_lst_downscaling.figures.style`, size with `TEXT_WIDTH_IN` / `HALF_WIDTH_IN`, and
  write with `style.save_figure(fig, "<script name without .py>")`.
- Create figures with `figsize=(TEXT_WIDTH_IN, h)` and place them at 100% text width in Typst:
  the SVG keeps the figure size (constrained layout, no tight cropping), so text stays at 8.5 pt.
- Create maps and images with `rasterized=True`: they are embedded at 300 dpi. Text is converted
  to paths (`svg.fonttype: path`) and lines stay vector.
- Inputs only from committed tables or from documented GCS / W&B validation exports.
  Never the test 2025 block. No inputs from local scratch files.
- Output only to `figures/out/`. Do not commit generated files.
- Colours follow `docs/thesis/06-ausarbeitung-design.md`: max 4 to 5 categorical colours;
  green, orange and red only for semantic meaning.

## Build

```bash
uv run nox -s figures          # runs every scripts/figures/fig_*.py in name order
```

## Sync into the Typst repo

From the Typst repo (`masterarbeit-text`), pass the path of the code repo checkout that
holds `figures/out/` (the default is `../berlin-lst-downscaling`):

```bash
scripts/sync-figures.sh ~/code/berlin-lst-downscaling
```

This rsyncs `*.svg` (and `*.pdf`, `*.png`) into the Typst repo's `figures/`.
