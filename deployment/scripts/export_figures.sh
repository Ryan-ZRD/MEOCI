#!/bin/bash
# ============================================================
# export_figures.sh
# ------------------------------------------------------------
# One-click script to reproduce all visualization figures
# (Fig.7–Fig.16) from the MEOCI framework.
# Automatically executes all plotting scripts under:
#   visualization/
# and stores figures in results/plots/
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIG_DIR="$PROJECT_ROOT/results/plots"
LOG_DIR="$PROJECT_ROOT/results/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$FIG_DIR"
mkdir -p "$LOG_DIR"

print_header() {
  echo "============================================================"
  echo "MEOCI Visualization Export Script"
  echo "============================================================"
  echo "Start Time: $(date)"
  echo "Figures will be saved under: $FIG_DIR"
  echo "------------------------------------------------------------"
}

run_plot() {
  local script_path=$1
  local description=$2
  local log_file="$LOG_DIR/$(basename ${script_path%.*})_${TIMESTAMP}.log"

  echo "[$(date +"%H:%M:%S")] Running: $description"
  if python "$PROJECT_ROOT/$script_path" > "$log_file" 2>&1; then
    echo "[$(date +"%H:%M:%S")] Completed: $description"
  else
    echo "[$(date +"%H:%M:%S")] Failed: $description"
    echo "Check log: $log_file"
  fi
  echo "------------------------------------------------------------"
}

# ============================================================
# Main Execution
# ============================================================
print_header

# 1. Ablation Studies (Fig.7–8)
run_plot "visualization/ablation/plot_ablation_convergence.py" "Ablation Convergence Curves"
run_plot "visualization/ablation/plot_ablation_delay.py" "Ablation Delay Comparison"

# 2. Exit Probability Analysis (Fig.9)
run_plot "visualization/exit_analysis/plot_exit_probability_alexnet.py" "Exit Probability - AlexNet"
run_plot "visualization/exit_analysis/plot_exit_probability_vgg16.py" "Exit Probability - VGG16"

# 3. Heterogeneous Latency (Fig.10)
run_plot "visualization/heterogeneous/plot_latency_alexnet.py" "Heterogeneous Latency - AlexNet"
run_plot "visualization/heterogeneous/plot_latency_vgg16.py" "Heterogeneous Latency - VGG16"
run_plot "visualization/heterogeneous/plot_latency_resnet50.py" "Heterogeneous Latency - ResNet50"
run_plot "visualization/heterogeneous/plot_latency_yolov10n.py" "Heterogeneous Latency - YOLOv10n"

# 4. Accuracy and Latency CDF (Fig.11)
run_plot "visualization/accuracy_cdf/plot_accuracy_comparison.py" "Accuracy Comparison"
run_plot "visualization/accuracy_cdf/plot_latency_cdf.py" "Latency CDF Plot"

# 5. Vehicle Count Effect (Fig.12)
run_plot "visualization/vehicle_effect/plot_latency_vs_vehicle.py" "Latency vs Vehicle Count"
run_plot "visualization/vehicle_effect/plot_completion_vs_vehicle.py" "Completion Rate vs Vehicle Count"

# 6. Transmission Rate Effect (Fig.13)
run_plot "visualization/transmission_effect/plot_latency_vs_rate.py" "Latency vs Transmission Rate"
run_plot "visualization/transmission_effect/plot_completion_vs_rate.py" "Completion vs Transmission Rate"

# 7. Delay and Energy Constraints (Fig.14–15)
run_plot "visualization/delay_constraints/plot_accuracy_vs_delay_constraints.py" "Accuracy vs Delay Constraints"
run_plot "visualization/delay_constraints/plot_completion_vs_delay_constraints.py" "Completion vs Delay Constraints"
run_plot "visualization/energy_constraints/plot_latency_vs_energy.py" "Latency vs Energy Budget"
run_plot "visualization/energy_constraints/plot_energy_consumption.py" "Energy Consumption Trend"

# 8. Scalability and Density Analysis (Fig.16)
run_plot "visualization/scalability/plot_dynamic_density.py" "Dynamic Density Analysis"
run_plot "visualization/scalability/plot_latency_dynamic_density.py" "Latency under Dynamic Density"
run_plot "visualization/scalability/plot_vehicle_scalability.py" "Vehicle Scalability Trend"
run_plot "visualization/scalability/plot_computational_stress.py" "Computational Stress Distribution"

echo "============================================================"
echo "All visualization scripts executed successfully."
echo "Figures saved under: $FIG_DIR"
echo "Logs stored in: $LOG_DIR"
echo "End Time: $(date)"
echo "============================================================"
