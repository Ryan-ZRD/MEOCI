"""
visualization.energy_constraints
--------------------------------
Module for reproducing energy constraint experiments (Fig.14a–Fig.14b)
in the MEOCI framework.

Purpose:
    Analyze the impact of energy constraints on system latency and energy
    consumption across multiple collaborative inference methods.

Contents:
    - plot_latency_vs_energy.py       → Fig.14(a): Latency vs Energy Constraint
    - plot_energy_consumption.py      → Fig.14(b): Energy Consumption vs Energy Constraint
    - energy_data_loader.py           → Data loader, smoother, and trend summarizer
    - energy_plot_utils.py            → Shared color map, style, and plot functions

Typical Usage:
    from visualization.energy_constraints import (
        plot_latency_vs_energy,
        plot_energy_consumption,
        load_energy_data,
        summarize_energy_trends
    )

    df = load_energy_data()
    summary = summarize_energy_trends(df)
    plot_latency_vs_energy()
    plot_energy_consumption()
"""

from .plot_latency_vs_energy import plot_latency_vs_energy
from .plot_energy_consumption import plot_energy_consumption
from .energy_data_loader import (
    load_energy_data,
    summarize_energy_trends,
    smooth_energy_curves,
    generate_synthetic_energy_data,
)
from .energy_plot_utils import (
    METHOD_COLORS,
    METHOD_STYLES,
    draw_energy_curve,
    annotate_figure,
    finalize_energy_plot,
)

__all__ = [
    # Plot functions
    "plot_latency_vs_energy",
    "plot_energy_consumption",

    # Data utilities
    "load_energy_data",
    "summarize_energy_trends",
    "smooth_energy_curves",
    "generate_synthetic_energy_data",

    # Plot utilities
    "METHOD_COLORS",
    "METHOD_STYLES",
    "draw_energy_curve",
    "annotate_figure",
    "finalize_energy_plot",
]
