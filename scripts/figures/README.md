# Thesis figures

Figures for the thesis are built by script here and written to `figures/out/` (git-ignored), as SVG.
Shared style, palette and export live in `src/berlin_lst_downscaling/figures/`.

## Conventions

- One script per figure: `fig_<chapter><nn>_<slug>.py` (for example `fig_0401_holdout_map.py`).
  `fig_00_style_check.py` is the style check and not a thesis figure.
- Start with `style.use_thesis_style()`, use the palette and cmap constants from
  `berlin_lst_downscaling.figures.style`, size with `TEXT_WIDTH_IN` / `HALF_WIDTH_IN`, and
  write with `style.save_figure(fig, "<script name without .py>")`.
- Create figures with `figsize=(TEXT_WIDTH_IN, h)` and place them at 100% text width in the thesis:
  the SVG keeps the figure size (constrained layout, no tight cropping), so text stays at 8.5 pt.
- Create maps and images with `rasterized=True`: they are embedded at 300 dpi. Text is converted
  to paths (`svg.fonttype: path`) and lines stay vector.
- Inputs only from committed tables or from documented GCS / W&B validation exports.
  Never the test 2025 block. No inputs from local scratch files.
- Output only to `figures/out/`. Do not commit generated files.
- Colours come from the palette constants in `style.py`: max 4 to 5 categorical colours;
  green, orange and red only for semantic meaning.

## Build

```bash
uv run nox -s figures          # runs every scripts/figures/fig_*.py in name order
```

## Use in the thesis

The thesis is written in LaTeX in a separate private workspace; it takes the files from
`figures/out/` as they are.
