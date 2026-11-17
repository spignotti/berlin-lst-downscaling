"""Offline tests for the thesis figure style and PDF export."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from berlin_lst_downscaling.figures import style


@pytest.fixture
def thesis_style():
    with matplotlib.rc_context():
        style.use_thesis_style()
        yield matplotlib.rcParams


def test_style_values_are_loaded(thesis_style):
    rc = thesis_style
    assert rc["font.size"] == 8.5
    assert rc["font.sans-serif"][:2] == ["Source Sans 3", "DejaVu Sans"]
    assert rc["axes.spines.top"] is False
    assert rc["axes.spines.right"] is False
    assert rc["axes.grid"] is False
    assert rc["pdf.fonttype"] == 42
    assert rc["svg.fonttype"] == "path"
    assert rc["savefig.dpi"] == 300
    assert rc["figure.constrained_layout.use"] is True
    assert rc["savefig.bbox"] is None  # no tight cropping
    assert to_hex(rc["axes.edgecolor"], keep_alpha=False).upper() == style.MUTED
    assert to_hex(rc["text.color"], keep_alpha=False).upper() == style.INK


def test_prop_cycle_matches_palette(thesis_style):
    cycle = [c["color"].upper() for c in thesis_style["axes.prop_cycle"]]
    assert cycle == [style.ACCENT, style.INK, style.MUTED, style.ACCENT_DARK]


def test_width_constants():
    assert style.TEXT_WIDTH_IN == pytest.approx(155 / 25.4)
    assert style.HALF_WIDTH_IN == pytest.approx(style.TEXT_WIDTH_IN / 2)


def test_save_figure_writes_text_free_svg_of_text_width(thesis_style, tmp_path):
    fig = Figure(figsize=(style.TEXT_WIDTH_IN, 2.0))
    ax = fig.subplots()
    ax.plot([0, 1, 2], [1, 3, 2])
    ax.set_xlabel("x")
    ax.set_title("title")
    out_dir = tmp_path / "nested" / "out"
    path = style.save_figure(fig, "dummy", out_dir=out_dir)
    assert path == out_dir / "dummy.svg"
    svg = path.read_text(encoding="utf-8")
    assert len(svg) > 1000
    assert "<text" not in svg  # svg.fonttype: path
    root = re.search(r"<svg[^>]*>", svg)
    assert root is not None
    width = re.search(r'width="([\d.]+)pt"', root.group(0))
    assert width is not None
    assert float(width.group(1)) == pytest.approx(style.TEXT_WIDTH_IN * 72, abs=0.5)


def test_static_source_sans_instance_is_used_for_regular():
    if not any(f.name == style.FONT_FAMILY for f in fm.fontManager.ttflist):
        pytest.skip("Source Sans 3 is not installed")
    style._register_static_instances()
    for weight, font_style in [(400, "normal"), (700, "normal"), (400, "italic"), (700, "italic")]:
        prop = fm.FontProperties(family=style.FONT_FAMILY, weight=weight, style=font_style)
        found = Path(fm.findfont(prop, fallback_to_default=False))
        assert "[" not in found.name, f"{weight} {font_style} resolved to variable font {found}"
