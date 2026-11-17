"""Shared style, palette and export helper for the thesis figures.

Figure scripts call :func:`use_thesis_style` once, draw with the palette
constants, and write the result with :func:`save_figure`. Colour roles follow
``docs/thesis/06-ausarbeitung-design.md``: green, orange and red carry
semantic meaning only.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.style
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Palette (hex, as in the design document)
INK = "#14181F"
MUTED = "#5B6370"
ACCENT = "#3358D4"
ACCENT_DARK = "#273EA5"
BORDER = "#DFE3E8"
GREEN = "#2F9E63"
ORANGE = "#C86A1F"
RED = "#D93A3F"

# Colour maps (names, resolved by matplotlib)
CMAP_LST = "magma"  # land surface temperature and other thermal maps
CMAP_SEQ = "cividis"  # other sequential quantities
CMAP_DIV = "RdBu_r"  # diverging, e.g. signed errors. PROPOSAL, not yet agreed.

# A4 text block of 155 mm; HALF_WIDTH_IN is exactly half of it (no gutter).
TEXT_WIDTH_MM = 155.0
TEXT_WIDTH_IN = TEXT_WIDTH_MM / 25.4
HALF_WIDTH_IN = TEXT_WIDTH_IN / 2

DPI = 300
STYLE_PATH = Path(__file__).with_name("thesis.mplstyle")
FONT_FAMILY = "Source Sans 3"

# Variable-font instances to register: (wght, style).
_INSTANCES = ((400, "normal"), (700, "normal"), (400, "italic"), (700, "italic"))


def _cache_name(source: str, weight: int, style: str) -> str:
    """Cache file name that changes when the source font file changes."""
    stat = Path(source).stat()
    tag = hashlib.sha256(f"{source}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:10]
    return f"SourceSans3-{style}-{weight}-{tag}.ttf"


def _make_instance(source: str, weight: int, target: Path) -> None:
    """Cut a static instance of a variable font; the file appears atomically."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        font = TTFont(source)
        instantiateVariableFont(font, {"wght": weight}, inplace=True, updateFontNames=True)
        font.save(tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _register_static_instances() -> None:
    """Make Source Sans 3 render at Regular weight, not ExtraLight.

    The installed font is a variable font whose default instance is ExtraLight
    (wght=200). Matplotlib cannot select a variation, so it would draw hairline
    text. Cut static Regular, Bold, Italic and Bold Italic instances with
    fontTools (already a matplotlib dependency), cache them in the matplotlib
    cache dir (keyed by the source font's size and mtime) and register them.
    Does nothing when the font is not installed or a static Regular is already
    available, so machines without the font fall back to DejaVu Sans.
    """
    entries = [f for f in fm.fontManager.ttflist if f.name == FONT_FAMILY]
    if any(f.weight == 400 and "[" not in Path(f.fname).name for f in entries):
        return
    variable = {
        "italic" if f.style == "italic" else "normal": f.fname
        for f in entries
        if "[wght]" in Path(f.fname).name
    }
    if not variable:
        return
    cache = Path(matplotlib.get_cachedir()) / "thesis-fonts"
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning(
            "Source Sans 3 is installed as a variable font but the cache dir %s is not "
            "writable; text may render as ExtraLight.",
            cache,
        )
        return
    for weight, style in _INSTANCES:
        source = variable.get(style)
        if source is None:
            continue
        try:
            target = cache / _cache_name(source, weight, style)
            if not target.exists():
                _make_instance(source, weight, target)
            try:
                fm.fontManager.addfont(str(target))
            except Exception:
                target.unlink(missing_ok=True)  # corrupt cache file: regenerate next run
                raise
        except Exception as exc:  # styling must never break a figure script
            logger.warning(
                "Source Sans 3 is installed as a variable font but the %s %d instance "
                "could not be prepared (%s); this weight may render as ExtraLight.",
                style,
                weight,
                exc,
            )


def use_thesis_style() -> None:
    """Activate the thesis matplotlib style for the current process."""
    _register_static_instances()
    matplotlib.style.use(str(STYLE_PATH))


def save_figure(fig: Figure, name: str, out_dir: Path = Path("figures/out")) -> Path:
    """Write ``fig`` as SVG to ``out_dir/<name>.svg`` and return the path.

    SVG, not PDF: Typst cannot embed PDF images in PDF/A mode. Text is
    converted to paths (``svg.fonttype: path``), so labels do not depend on
    installed fonts. Lines stay vector. Artists created with ``rasterized=True``
    (maps, ``imshow``, ``pcolormesh``) are embedded as 300 dpi images. The page
    size is the figure size (no tight cropping), so a figure created with
    ``figsize=(TEXT_WIDTH_IN, h)`` keeps its width and text stays at 8.5 pt
    when placed at 100% text width.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.svg"
    fig.savefig(path, format="svg", dpi=DPI)
    return path
