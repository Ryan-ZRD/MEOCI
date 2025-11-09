"""
visualization.transmission_effect
---------------------------------
Module for reproducing wireless transmission effect experiments (Fig.12).

This package visualizes how varying wireless transmission rates (1–10 Mbps)
affect inference latency and task completion rate across different collaborative
inference strategies.

Includes:
    • plot_latency_vs_rate.py       → Fig.12(a): Latency vs Transmission Rate
    • plot_completion_vs_rate.py    → Fig.12(b): Completion Rate vs Transmission Rate
    • transmission_data_loader.py   → Dataset loader & validator
    • transmission_plot_utils.py    → Shared plotting utilities (colors, markers, smoothing)

Typical Usage:
    from visualization.transmission_effect import (
        plot_latency_vs_rate,
        plot_completion_vs_rate,
        load_transmission_data
    )

    df = load_transmission_data()
    plot_latency_vs_rate()
    plot_completion_vs_rate()
"""

from .plot_latency_vs_rate import plot_latency_vs_rate
from .plot_completion_vs_rate import plot_completion_vs_rate
from .transmission_data_loader import load_transmission_data, summarize_transmission_data
from .transmission_plot_utils import (
    METHOD_COLORS,
    METHOD_STYLES,
    METHOD_MARKERS,
    smooth_curve,
    draw_transmission_curves
)

__all__ = [
    "plot_latency_vs_rate",
    "plot_completion_vs_rate",
    "load_transmission_data",
    "summarize_transmission_data",
    "METHOD_COLORS",
    "METHOD_STYLES",
    "METHOD_MARKERS",
    "smooth_curve",
    "draw_transmission_curves"
]
