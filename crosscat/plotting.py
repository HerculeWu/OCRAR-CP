"""Portable plotting style; Matplotlib MathText, no external TeX."""
import matplotlib as mpl
SINGLE_COLUMN_WIDTH_INCH = 3.46

def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.2,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "figure.figsize": (SINGLE_COLUMN_WIDTH_INCH, 2.6),
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "legend.frameon": False,
            "text.usetex": False,
        }
    )
