"""
visualization
=============
Top-level package for reproducing all visualization results (Fig.7–Fig.16)
in the MEOCI framework.

Purpose:
    This package integrates all figure modules across experimental categories,
    providing consistent plotting, data handling, and style definitions.

Submodules:
    - ablation/                → Fig.7 (Ablation Studies)
    - exit_analysis/           → Fig.8 (Multi-Exit Probability Analysis)
    - heterogeneous/           → Fig.9 (Heterogeneous Device Performance)
    - accuracy_cdf/            → Fig.10 (Accuracy & Latency CDF)
    - vehicle_effect/          → Fig.11 (Vehicle Density Effect)
    - transmission_effect/     → Fig.12 (Transmission Rate Effect)
    - delay_constraints/       → Fig.13 (Delay Constraint Impact)
    - energy_constraints/      → Fig.14 (Energy Constraint Impact)
    - scalability/             → Fig.15–16 (System Scalability Analysis)
    - shared_styles/           → Global plot styles and smoothing utilities
    - data_csv/                → Placeholder for input CSVs (no data content)
    - export_all_figures.py    → One-click reproduction of all figures

Typical Usage:
    from visualization import export_all_figures
    export_all_figures(output_dir="results/plots", fmt="png")

    Or import specific submodules:
    from visualization.ablation import plot_ablation_convergence
"""

# ------------------------------------------------------------
# 1. Import major submodules for direct access
# ------------------------------------------------------------
from visualization.ablation import (
    plot_ablation_convergence,
    plot_ablation_delay,
)
from visualization.exit_analysis import (
    plot_exit_probability_alexnet,
    plot_exit_probability_vgg16,
)
from visualization.heterogeneous import (
    plot_latency_alexnet,
    plot_latency_vgg16,
    plot_latency_resnet50,
    plot_latency_yolov10n,
)
from visualization.accuracy_cdf import (
    plot_accuracy_comparison,
    plot_latency_cdf,
)
from visualization.vehicle_effect import (
    plot_latency_vs_vehicle,
    plot_completion_vs_vehicle,
)
from visualization.transmission_effect import (
    plot_latency_vs_rate,
    plot_completion_vs_rate,
)
from visualization.delay_constraints import (
    plot_accuracy_vs_delay_constraints,
    plot_completion_vs_delay_constraints,
)
from visualization.energy_constraints import (
    plot_latency_vs_energy,
    plot_energy_consumption,
)
from visualization.scalability import (
    plot_dynamic_density,
    plot_latency_dynamic_density,
    plot_vehicle_scalability,
    plot_computational_stress,
)
from visualization.shared_styles import (
    smoothing,
)
from visualization.export_all_figures import export_all_figures


# ------------------------------------------------------------
# 2. Define accessible names
# ------------------------------------------------------------
__all__ = [
    # --- Fig.7 ---
    "plot_ablation_convergence",
    "plot_ablation_delay",

    # --- Fig.8 ---
    "plot_exit_probability_alexnet",
    "plot_exit_probability_vgg16",

    # --- Fig.9 ---
    "plot_latency_alexnet",
    "plot_latency_vgg16",
    "plot_latency_resnet50",
    "plot_latency_yolov10n",

    # --- Fig.10 ---
    "plot_accuracy_comparison",
    "plot_latency_cdf",

    # --- Fig.11 ---
    "plot_latency_vs_vehicle",
    "plot_completion_vs_vehicle",

    # --- Fig.12 ---
    "plot_latency_vs_rate",
    "plot_completion_vs_rate",

    # --- Fig.13 ---
    "plot_accuracy_vs_delay_constraints",
    "plot_completion_vs_delay_constraints",

    # --- Fig.14 ---
    "plot_latency_vs_energy",
    "plot_energy_consumption",

    # --- Fig.15–16 ---
    "plot_dynamic_density",
    "plot_latency_dynamic_density",
    "plot_vehicle_scalability",
    "plot_computational_stress",

    # --- Shared tools ---
    "smoothing",

    # --- Batch export ---
    "export_all_figures",
]
