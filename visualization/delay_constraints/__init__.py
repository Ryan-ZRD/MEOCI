"""
visualization.delay_constraints
-------------------------------
Module for reproducing delay constraint experiments (Fig.13a–13b)
in the MEOCI framework.

Purpose:
    Evaluates how different delay constraint thresholds affect
    the accuracy and task completion rate of collaborative
    inference algorithms.

Contents:
    - plot_accuracy_vs_delay_constraints.py
    - plot_completion_vs_delay_constraints.py
    - delay_constraints_utils.py

Example Usage:
    from visualization.delay_constraints import (
        plot_accuracy_vs_delay_constraints,
        plot_completion_vs_delay_constraints,
        load_delay_constraints_data,
        summarize_delay_effects
    )

    df = load_delay_constraints_data()
    summary = summarize_delay_effects(df)
    plot_accuracy_vs_delay_constraints()
    plot_completion_vs_delay_constraints()
"""

from .plot_accuracy_vs_delay_constraints import plot_accuracy_vs_delay_constraints
from .plot_completion_vs_delay_constraints import plot_completion_vs_delay_constraints
from .delay_constraints_utils import (
    load_delay_constraints_data,
    summarize_delay_effects,
    smooth_delay_curves,
    generate_synthetic_delay_data
)

__all__ = [
    "plot_accuracy_vs_delay_constraints",
    "plot_completion_vs_delay_constraints",
    "load_delay_constraints_data",
    "summarize_delay_effects",
    "smooth_delay_curves",
    "generate_synthetic_delay_data"
]
