"""Offline tests for the thesis figure style and PDF export."""

from __future__ import annotations

import matplotlib
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
    assert rc["svg.fonttype"] == "none"
    assert rc["savefig.dpi"] == 300
    assert rc["savefig.bbox"] == "tight"
    assert to_hex(rc["axes.edgecolor"], keep_alpha=False).upper() == style.MUTED
    assert to_hex(rc["text.color"], keep_alpha=False).upper() == style.INK


def test_prop_cycle_matches_palette(thesis_style):
    cycle = [c["color"].upper() for c in thesis_style["axes.prop_cycle"]]
    assert cycle == [style.ACCENT, style.INK, style.MUTED, style.ACCENT_DARK]


def test_width_constants():
    assert style.TEXT_WIDTH_IN == pytest.approx(155 / 25.4)
    assert style.HALF_WIDTH_IN == pytest.approx(style.TEXT_WIDTH_IN / 2)


def test_save_figure_writes_nonempty_pdf(thesis_style, tmp_path):
    fig = Figure(figsize=(style.HALF_WIDTH_IN, 2.0))
    ax = fig.subplots()
    ax.plot([0, 1, 2], [1, 3, 2])
    ax.set_xlabel("x")
    out_dir = tmp_path / "nested" / "out"
    path = style.save_figure(fig, "dummy", out_dir=out_dir)
    assert path == out_dir / "dummy.pdf"
    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 1000
