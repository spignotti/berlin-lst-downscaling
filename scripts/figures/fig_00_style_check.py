"""Style check for the thesis figure infrastructure. Synthetic data only.

Shows the palette, a line plot with the default colour cycle and a small image
with the LST colour map and a colourbar in kelvin. Not a thesis figure.

Usage: uv run python scripts/figures/fig_00_style_check.py
Output: figures/out/fig_00_style_check.svg
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from berlin_lst_downscaling.figures import style


def main() -> None:
    style.use_thesis_style()

    fig = plt.figure(figsize=(style.TEXT_WIDTH_IN, 4.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.5, 2.0], width_ratios=[1.3, 1.0])

    # Palette swatches
    ax_sw = fig.add_subplot(grid[0, :])
    swatches = [
        ("INK", style.INK),
        ("MUTED", style.MUTED),
        ("ACCENT", style.ACCENT),
        ("ACCENT_DARK", style.ACCENT_DARK),
        ("BORDER", style.BORDER),
        ("GREEN", style.GREEN),
        ("ORANGE", style.ORANGE),
        ("RED", style.RED),
    ]
    for i, (label, colour) in enumerate(swatches):
        ax_sw.add_patch(Rectangle((i, 0.25), 0.9, 0.75, facecolor=colour, edgecolor="none"))
        ax_sw.text(i + 0.45, 0.05, label, ha="center", va="top", fontsize=6.5)
    ax_sw.set_xlim(0, len(swatches))
    ax_sw.set_ylim(-0.25, 1.0)
    ax_sw.axis("off")
    ax_sw.set_title("Palette (green, orange, red: semantic use only)", loc="left")

    # Line plot with the default colour cycle
    ax_line = fig.add_subplot(grid[1, 0])
    x = np.linspace(0, 24, 200)
    for k, name in enumerate(["Series A", "Series B", "Series C", "Series D"]):
        ax_line.plot(x, 290 + 6 * np.sin((x - 4 * k) / 24 * 2 * np.pi) + 2 * k, label=name)
    ax_line.set_xlabel("Hour of day (synthetic)")
    ax_line.set_ylabel("Temperature (K)")
    ax_line.set_ylim(283, 314)
    ax_line.legend(loc="upper center", ncols=2, columnspacing=1.0, handlelength=1.4)
    ax_line.set_title("Line plot", loc="left")

    # Small image with the LST colour map
    ax_img = fig.add_subplot(grid[1, 1])
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:24, 0:24]
    field = (
        300 + 8 * np.exp(-(((xx - 9) ** 2 + (yy - 14) ** 2) / 60)) + rng.normal(0, 0.4, (24, 24))
    )
    image = ax_img.imshow(
        field, cmap=style.CMAP_LST, interpolation="nearest", rasterized=True, vmin=298, vmax=310
    )
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title("Image, magma", loc="left")
    colourbar = fig.colorbar(image, ax=ax_img)
    colourbar.set_label("LST (K)")
    colourbar.outline.set_linewidth(0.6)

    path = style.save_figure(fig, "fig_00_style_check")
    plt.close(fig)
    print(path.resolve())


if __name__ == "__main__":
    main()
