"""
visualization.shared_styles.plot_style
----------------------------------------------------------
Defines the unified visual style for all figures in MEOCI paper.

Applied to:
    Fig.7–Fig.16 (Ablation, Exit Analysis, Scalability, etc.)

Features:
    - Consistent font sizes, line widths, and colors
    - Latex-compatible rendering (if available)
    - Grid and legend style for high-quality publications
"""

import matplotlib.pyplot as plt
import matplotlib as mpl


def set_global_plot_style(dark_mode: bool = False):
    """
    Configure a unified, publication-ready plotting style.

    Args:
        dark_mode (bool): whether to enable dark background plotting.
    """
    # ------------------------------------------------------
    # 1. Font & Global Aesthetic Settings
    # ------------------------------------------------------
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.1,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    })

    # ------------------------------------------------------
    # 2. Enable LaTeX Rendering (if available)
    # ------------------------------------------------------
    try:
        mpl.rc("text", usetex=True)
    except Exception:
        mpl.rc("text", usetex=False)

    # ------------------------------------------------------
    # 3. Color Theme Configuration
    # ------------------------------------------------------
    if dark_mode:
        plt.style.use("dark_background")
        mpl.rcParams["axes.facecolor"] = "#1a1a1a"
        mpl.rcParams["figure.facecolor"] = "#1a1a1a"
        mpl.rcParams["axes.edgecolor"] = "#E0E0E0"
        mpl.rcParams["text.color"] = "#FFFFFF"
        mpl.rcParams["axes.labelcolor"] = "#FFFFFF"
        mpl.rcParams["xtick.color"] = "#FFFFFF"
        mpl.rcParams["ytick.color"] = "#FFFFFF"
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
        mpl.rcParams["axes.facecolor"] = "#FFFFFF"
        mpl.rcParams["figure.facecolor"] = "#FFFFFF"

    # ------------------------------------------------------
    # 4. Color Palette (Fixed for all figures)
    # ------------------------------------------------------
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[
            "#1f77b4",  # MEOCI - Blue
            "#ff7f0e",  # Edgent - Orange
            "#2ca02c",  # FedAdapt - Green
            "#d62728",  # LBO - Red
            "#9467bd",  # EdgeOnly - Purple
        ]
    )

    # ------------------------------------------------------
    # 5. Tick and Grid Customization
    # ------------------------------------------------------
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.major.size"] = 4
    mpl.rcParams["ytick.major.size"] = 4
    mpl.rcParams["xtick.major.width"] = 0.8
    mpl.rcParams["ytick.major.width"] = 0.8
    mpl.rcParams["grid.linewidth"] = 0.7

    # ------------------------------------------------------
    # 6. Legend Style
    # ------------------------------------------------------
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["legend.loc"] = "best"
    mpl.rcParams["legend.handlelength"] = 2.5
    mpl.rcParams["legend.handletextpad"] = 0.4
    mpl.rcParams["legend.borderpad"] = 0.3

    # ------------------------------------------------------
    # 7. Figure Size (Default)
    # ------------------------------------------------------
    mpl.rcParams["figure.figsize"] = [6.8, 4.2]


def set_ieee_style():
    """Shortcut: Apply IEEE-compatible formatting (for LaTeX papers)."""
    set_global_plot_style(dark_mode=False)
    mpl.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    })


def set_paper_style(conference: str = "IEEE"):
    """
    Apply customized visual tuning for specific publication targets.
    Args:
        conference: "IEEE", "Elsevier", "ACM", or "Nature"
    """
    set_global_plot_style(dark_mode=False)

    if conference.upper() == "ELSEVIER":
        mpl.rcParams.update({"font.family": "Arial", "font.size": 11})
    elif conference.upper() == "ACM":
        mpl.rcParams.update({"font.family": "Helvetica", "font.size": 11})
    elif conference.upper() == "NATURE":
        mpl.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.titlesize": 10})
    # IEEE default already applied


if __name__ == "__main__":
    # Quick visual test
    set_global_plot_style()
    import numpy as np
    x = np.linspace(0, 10, 100)
    plt.plot(x, np.sin(x), label="MEOCI")
    plt.plot(x, np.cos(x), label="Edgent")
    plt.legend()
    plt.title("Plot Style Test")
    plt.show()
