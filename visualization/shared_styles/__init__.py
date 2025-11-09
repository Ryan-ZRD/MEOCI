"""
visualization.shared_styles
----------------------------------------------------------
This package defines consistent plotting styles and smoothing utilities
for all visualization modules in the MEOCI project.

It ensures visual uniformity across figures (Fig.7–Fig.16)
in terms of color palette, font, grid style, and data smoothness.

Modules included:
    - plot_style.py   : Global Matplotlib configuration for publication-quality figures.
    - smoothing.py    : Data smoothing (EMA, Moving Average, Gaussian, etc.)

Usage:
    from visualization.shared_styles import (
        set_global_plot_style,
        set_ieee_style,
        smooth_curve,
        moving_average,
        exponential_moving_average,
        savitzky_golay_smoothing,
        gaussian_smoothing
    )
"""

from visualization.shared_styles.plot_style import (
    set_global_plot_style,
    set_ieee_style,
    set_paper_style,
)

from visualization.shared_styles.smoothing import (
    moving_average,
    exponential_moving_average,
    savitzky_golay_smoothing,
    gaussian_smoothing,
    smooth_curve,
    normalize,
    detrend,
)

__all__ = [
    # Global style configuration
    "set_global_plot_style",
    "set_ieee_style",
    "set_paper_style",

    # Smoothing utilities
    "moving_average",
    "exponential_moving_average",
    "savitzky_golay_smoothing",
    "gaussian_smoothing",
    "smooth_curve",
    "normalize",
    "detrend",
]
