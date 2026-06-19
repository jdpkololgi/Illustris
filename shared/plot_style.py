"""Canonical plot styling for GraphWeb_DESI.

See ../PLOT_STYLE_GUIDE.md for the full rationale and rules. This module is
the executable counterpart of that document — keep them in sync if either
changes.

Usage:
    from shared.plot_style import apply_style, finalize_axes, COSMIC_WEB_COLORS, CLASS_ORDER

    apply_style()  # once per script/notebook session
    ...
    finalize_axes(ax, title=..., xlabel=..., ylabel=...)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap

__all__ = [
    "CLASS_ORDER",
    "COSMIC_WEB_COLORS",
    "COSMIC_WEB_CMAP",
    "ACCENT_COLORS",
    "BACKGROUND",
    "TEXT_COLOR",
    "GRID_COLOR",
    "FONT_FAMILY",
    "register_fonts",
    "apply_style",
    "finalize_axes",
]

# --------------------------------------------------------------------------
# Color system (see PLOT_STYLE_GUIDE.md §1)
# --------------------------------------------------------------------------

CLASS_ORDER: list[str] = ["void", "wall", "filament", "cluster"]

COSMIC_WEB_COLORS: dict[str, str] = {
    "void": "#A1FCDD",
    "wall": "#4E84F7",
    "filament": "#EB336F",
    "cluster": "#F5C144",
}

COSMIC_WEB_CMAP = ListedColormap(
    [COSMIC_WEB_COLORS[c] for c in CLASS_ORDER], name="cosmic_web"
)

# General-purpose accent colors. Not for class encoding — see §1.3.
ACCENT_COLORS: dict[str, str] = {
    "magenta": "#FF006E",
    "blue": "#3A86FF",
    "red": "#D62828",
}

BACKGROUND = "#000000"
TEXT_COLOR = "#F2F2F2"
GRID_COLOR = "#F2F2F2"  # used with alpha, not as a solid color

# --------------------------------------------------------------------------
# Typography (see PLOT_STYLE_GUIDE.md §2)
# --------------------------------------------------------------------------

FONT_FAMILY = "IBM Plex Sans"

GRAPHWEB_FONT_DIR = os.environ.get(
    "GRAPHWEB_FONT_DIR",
    str(Path(__file__).resolve().parents[1] / "assets" / "fonts"),
)

_FONT_FILES = {
    "Regular": "IBMPlexSans-Regular.ttf",
    "Bold": "IBMPlexSans-Bold.ttf",
    "Italic": "IBMPlexSans-Italic.ttf",
}

_FONTS_REGISTERED = False


def register_fonts() -> str:
    """Register IBM Plex Sans with matplotlib's font manager if available.

    Returns the font family name actually selected — FONT_FAMILY if the
    bundled .ttf files were found, otherwise 'DejaVu Sans' as a fallback.
    Never raises; a missing font is a warning, not a crash.
    """
    global _FONTS_REGISTERED

    font_dir = Path(GRAPHWEB_FONT_DIR)
    found_any = False
    for filename in _FONT_FILES.values():
        path = font_dir / filename
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            found_any = True

    if not found_any:
        warnings.warn(
            f"IBM Plex Sans .ttf files not found in {font_dir}. "
            "Falling back to DejaVu Sans. See PLOT_STYLE_GUIDE.md §2.3 for "
            "the one-time download command.",
            stacklevel=2,
        )
        _FONTS_REGISTERED = True
        return "DejaVu Sans"

    _FONTS_REGISTERED = True
    return FONT_FAMILY


def apply_style() -> None:
    """Apply the canonical GraphWeb dark theme globally via rcParams.

    Call once per script or notebook session, before creating any figures.
    """
    family = register_fonts()

    plt.rcParams.update(
        {
            # Canvas
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "figure.figsize": (8, 6),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            # Typography
            "font.family": family,
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "mathtext.fontset": "dejavusans",
            # Text / ticks / spines
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid (off by default — opt in per-plot)
            "axes.grid": False,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.15,
            # Legend
            "legend.frameon": True,
            "legend.facecolor": BACKGROUND,
            "legend.edgecolor": GRID_COLOR,
            "legend.framealpha": 0.85,
        }
    )


def finalize_axes(
    ax,
    title: str,
    xlabel: str,
    ylabel: str,
    legend: bool = True,
    legend_loc: str = "best",
) -> None:
    """Enforce the mandatory elements from PLOT_STYLE_GUIDE.md §3.

    Sets title/axis labels and adds a legend if labeled artists exist. If
    legend=True but no labeled artists are found, warns instead of silently
    producing a figure with a missing legend.
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc=legend_loc)
        else:
            warnings.warn(
                "finalize_axes(legend=True) but no labeled artists were "
                "found on this axes — add label=... to your plot calls, or "
                "pass legend=False if this plot genuinely has none.",
                stacklevel=2,
            )
