"""
visualization.scalability
----------------------------------------------------------
Scalability analysis visualization package for MEOCI.

This submodule reproduces the scalability-related figures
(Fig.14–Fig.17) from the MEOCI paper.

Included Figures:
    - Fig.14: Dynamic Vehicle Density Impact
    - Fig.15: Latency under Varying Density
    - Fig.16: System Scalability vs Vehicle Count
    - Fig.17: Computational Stress (RSU CPU Load)

Usage Example:
    from visualization.scalability import (
        plot_dynamic_density,
        plot_latency_dynamic_density,
        plot_vehicle_scalability,
        plot_computational_stress
    )

    # Generate all scalability figures
    plot_dynamic_density("results/csv/scalability.csv", "results/plots/fig14_dynamic_density.png")
    plot_latency_dynamic_density("results/csv/scalability.csv", "results/plots/fig15_latency_density.png")
    plot_vehicle_scalability("results/csv/scalability.csv", "results/plots/fig16_vehicle_scalability.png")
    plot_computational_stress("results/csv/scalability.csv", "results/plots/fig17_computational_stress.png")
"""

from visualization.scalability.plot_dynamic_density import plot_dynamic_density
from visualization.scalability.plot_latency_dynamic_density import plot_latency_dynamic_density
from visualization.scalability.plot_vehicle_scalability import plot_vehicle_scalability
from visualization.scalability.plot_computational_stress import plot_computational_stress

# Utility functions
from visualization.scalability.scalability_utils import (
    load_scalability_data,
    moving_average,
    exponential_smoothing,
    normalize,
    standardize,
    compute_stats,
    get_color_palette,
    set_plot_labels,
    add_legend,
)

__all__ = [
    # Plot functions
    "plot_dynamic_density",
    "plot_latency_dynamic_density",
    "plot_vehicle_scalability",
    "plot_computational_stress",

    # Utils
    "load_scalability_data",
    "moving_average",
    "exponential_smoothing",
    "normalize",
    "standardize",
    "compute_stats",
    "get_color_palette",
    "set_plot_labels",
    "add_legend",
]
