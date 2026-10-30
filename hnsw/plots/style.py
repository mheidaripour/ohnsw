"""Shared matplotlib theming for the OSAM plots.

One place for the palette, the axis chrome and the save helper, so every figure
in the project reads as part of the same set.

Series colours are assigned in fixed slot order and never cycled: slot 1 is
always the same entity across figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots, in assignment order.  Never cycle past slot 8: fold the
# extras into "other", or facet into small multiples.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    text_primary: str
    text_secondary: str
    muted: str
    grid: str
    axis: str
    series: list[str]


LIGHT = Theme(
    name="light",
    # Pure white, not a warm off-white: these figures are dropped into a LaTeX
    # page that *is* pure white, and a near-white panel behind them reads as a
    # grey box around every figure. It doubles as the marker separation ring and
    # the patch behind the direct labels, so all three stay page-coloured.
    surface="#ffffff",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=SERIES_LIGHT,
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=SERIES_DARK,
)

THEMES = {"light": LIGHT, "dark": DARK}


# --- publication geometry ------------------------------------------------
# The USENIX template is two-column on a 7in text block with a 0.33in gutter.
COLUMN_WIDTH_IN = 3.33   # one column -- the default for a paper figure
TEXT_WIDTH_IN = 7.0      # full width, for a figure* spanning both columns

# A highlight ring drawn around a data mark, as a multiple of the mark's own
# size.  It has to clearly *enclose* the mark rather than sit on it -- a ring
# only slightly larger than what it rings reads as a fat dot, not as a ring.
MIN_RING_SCALE = 2.2

# Layered on top of a theme by apply(paper=True).  These are the settings that
# make a figure survive being dropped into a LaTeX column at 100% scale.
PAPER_RC = {
    # Type 3 fonts are the default in matplotlib and are flagged or rejected by
    # USENIX/IEEE PDF checks.  42 = TrueType, embedded and subsettable.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Match the body text: USENIX sets Times.  Liberation Serif is the metric-
    # compatible clone that ships on Linux -- and it is a real .ttf, so it
    # embeds cleanly under fonttype 42.  Nimbus Roman is the other common Times
    # clone but ships as .otf (CFF outlines), which makes pdf readers warn
    # "mismatch between font type and embedded font file"; keep it last.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral",
                   "DejaVu Serif", "Nimbus Roman"],
    "mathtext.fontset": "stix",

    # A column-width figure is ~3.3in wide, so anything above ~8pt crowds it and
    # anything below ~6pt is unreadable in print.
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,

    # Thinner everything: strokes that read as "fine" on screen go muddy when
    # the figure is scaled down into a column.  The connector stays light so the
    # markers -- the actual measurements -- carry the figure.
    "lines.linewidth": 0.8,
    "lines.markersize": 5.5,
    "lines.markeredgewidth": 0.8,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "xtick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "legend.handlelength": 2.0,

    "figure.dpi": 200,
    "savefig.pad_inches": 0.015,
}


def apply(theme: Theme, paper: bool = False) -> None:
    """Install the theme's rcParams.  Call once before creating a figure.

    ``paper=True`` layers on the print settings: Times-compatible serif at
    column-width sizes, thin strokes, and TrueType (not Type 3) font embedding.
    """
    mpl.rcParams.update({
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "savefig.edgecolor": theme.surface,

        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
        "font.size": 11,

        # Labels and tick labels carry the same weight as the title: full
        # contrast, no greyed-out chrome.
        "text.color": theme.text_primary,
        "axes.labelcolor": theme.text_primary,
        "axes.titlecolor": theme.text_primary,
        "xtick.color": theme.text_primary,
        "ytick.color": theme.text_primary,
        "xtick.labelcolor": theme.text_primary,
        "ytick.labelcolor": theme.text_primary,

        # Full box frame in the text colour, with a light grid on both axes
        # sitting behind the data.
        "axes.edgecolor": theme.text_primary,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "grid.color": theme.grid,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "legend.frameon": False,
        "legend.labelcolor": theme.text_primary,
        "legend.handlelength": 1.6,

        "lines.linewidth": 1.3,
        # Dashed by default: the markers are the measurements, and a dashed
        # connector reads as "these are the same series" without implying that
        # the segments between them were sampled.
        "lines.linestyle": "--",
        "lines.markersize": 7,
        # Surface-coloured ring so overlapping marks stay separable.
        "lines.markeredgewidth": 1.8,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    if paper:
        mpl.rcParams.update(PAPER_RC)


def style_axes(ax, theme: Theme) -> None:
    """Close the box: all four spines visible, in the text colour."""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(theme.text_primary)
        ax.spines[side].set_linewidth(mpl.rcParams["axes.linewidth"])
    ax.grid(True, axis="both", color=theme.grid,
            linewidth=mpl.rcParams["grid.linewidth"])
    ax.set_axisbelow(True)


def series_marker(index: int) -> str:
    """Marker shape per slot -- the redundant channel beside colour."""
    return ["o", "s", "^", "D", "v", "P", "X", "*"][index % 8]


def save(fig, out_base: str | Path, formats=("png", "pdf")) -> list[Path]:
    """Save one figure to several formats; returns the paths written."""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p)
        written.append(p)
    plt.close(fig)
    return written
