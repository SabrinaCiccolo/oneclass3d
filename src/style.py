"""Shared matplotlib style for every figure this project generates.

Print, not slide: figsize is the ACTUAL final width the figure is meant to be
viewed at (no insert-time rescaling), so `font.size` below is the true
printed point size. Two width presets cover every figure here:

    WIDTH_FULL  - full text width, single-panel or stacked panels
    WIDTH_HALF  - half text width, for side-by-side pairs
"""
from __future__ import annotations

import matplotlib as mpl

WIDTH_FULL = 6.3
WIDTH_HALF = 3.05

FIGSIZE_1PANEL = (WIDTH_FULL, 3.4)
FIGSIZE_2PANEL = (WIDTH_FULL, 2.9)
FIGSIZE_3PANEL = (WIDTH_FULL, 2.6)
FIGSIZE_TALL = (WIDTH_FULL, 4.6)
FIGSIZE_HALF = (WIDTH_HALF, 3.0)

DPI = 300


def panel_figsize(n_panels: int, height: float = 3.2, max_width: float = WIDTH_FULL,
                   min_panel_width: float = 1.7) -> tuple[float, float]:
    """Total width stays fixed regardless of panel count, so a figure with N
    panels of the same kind of axes never grows the printed page.
    `min_panel_width` is a floor: past it, prefer stacking panels in the
    caller (e.g. a 2-row grid) over shrinking further -- this only picks the
    width, not the layout."""
    return (max_width, height)

# Palette. Luminance-ordered (sRGB relative luminance) so any subset chosen
# for one figure can be checked for greyscale spread. Every colour is paired
# with a distinct marker/hatch/linestyle in the plotting code -- colour is
# never the only encoding of a categorical variable.
NAVY      = "#1F3A5F"   # L=0.041 - header band / primary series
ORANGE    = "#ED7D31"   # L=0.329 - accent / highlight series
ORANGE_LT = "#F2A365"   # L=0.460 - light-orange secondary accent
GREEN     = "#2E7D32"   # L=0.155 - improvement series
RED       = "#C0392B"   # L=0.143 - alert / near-chance
GREY      = "#A6A6A6"   # L=0.381 - neutral baseline / external reference
TEXT      = "#23314A"   # body text / axis labels / titles
GRID      = "#E5E7EB"   # card border / hairlines
CHANCE    = "#9AA3AF"   # muted reference-line grey

# Categorical series for the scorer-study figures: one fixed colour + marker
# per scorer identity, never cycled.
STUDY_BLUE = "#2E5E9E"   # L=0.111
STUDY_TEAL = "#0E7C7B"   # L=0.159 - ocsvm only
SCORER_COLORS = {
    "knn": STUDY_BLUE,
    "modern_hopfield": ORANGE,
    "modern_hopfield_auto": ORANGE,   # dashed variant of ORANGE, same entity family
    "mahalanobis": GREEN,
    "kde": RED,
    "ocsvm": STUDY_TEAL,
    "centroid": ORANGE_LT,
    "classical_hopfield": GREY,
    "hybrid": "#7B4EA6",
}
SCORER_MARKERS = {"knn": "o", "modern_hopfield": "s", "modern_hopfield_auto": "^",
                  "mahalanobis": "D", "kde": "v", "ocsvm": "P", "centroid": "x",
                  "classical_hopfield": "*", "hybrid": "<"}
SCORER_DASHES = {"modern_hopfield_auto": (4, 2)}   # dashed; everything else solid

FONT_STACK = ["Roboto", "Montserrat", "Ubuntu Sans", "DejaVu Sans"]


def apply() -> None:
    """Set rcParams to this project's print look. Call once at the top of a
    figure script's plotting function, before creating the figure."""
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update({
        "font.family": FONT_STACK,
        "font.size": 9,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.labelsize": 9,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "axes.edgecolor": "#C7CDD6",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.labelcolor": TEXT,
        "ytick.labelcolor": TEXT,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "savefig.dpi": DPI,
    })
