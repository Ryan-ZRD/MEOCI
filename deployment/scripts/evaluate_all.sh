#!/bin/bash
# ============================================================
# evaluate_all.sh
# ------------------------------------------------------------
# Automated evaluation script for MEOCI experiments
# Runs all core evaluation modules sequentially:
#   - Latency evaluation
#   - Energy analysis
#   - Multi-exit testing
#   - Ablation study
#   - Heterogeneity evaluation
#   - Scalability testing
#   - Parameter sensitivity analysis
# Results are saved under: ./results/
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results"
LOG_DIR="$RESULTS_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$LOG_DIR"

print_header() {
  echo "============================================================"
  echo "MEOCI Automated Evaluation Pipeline"
  echo "============================================================"
  echo "Start Time: $(date)"
  echo "Results Directory: $RESULTS_DIR"
  echo "------------------------------------------------------------"
}

run_experiment() {
  local script_name=$1
  local description=$2
  local log_file="$LOG_DIR/${script_name%.py}_${TIMESTAMP}.log"

  echo "[$(date +"%H:%M:%S")] Running: $description ($script_name)"
  if python "$PROJECT_ROOT/experiments/$script_name" > "$log_file" 2>&1; then
    echo "[$(date +"%H:%M:%S")] Completed: $description"
  else
    echo "[$(date +"%H:%M:%S")] Failed: $description"
    echo "Check log: $log_file"
  fi
  echo "------------------------------------------------------------"
}

# ============================================================
# Main execution sequence
# ============================================================
print_header

run_experiment "evaluate_latency.py" "Latency Evaluation"
run_experiment "analyze_energy.py" "Energy Consumption Analysis"
run_experiment "test_multi_exit.py" "Multi-Exit Model Evaluation"
run_experiment "ablation_study.py" "Ablation Study of ADP-D3QN Variants"
run_experiment "heterogeneity_eval.py" "Heterogeneous Device Performance"
run_experiment "scalability_test.py" "System Scalability Test"
run_experiment "parameter_sensitivity.py" "Hyperparameter Sensitivity Analysis"

echo "============================================================"
echo "All experiments executed."
echo "Logs are saved under: $LOG_DIR"
echo "Results stored under: $RESULTS_DIR"
echo "End Time: $(date)"
echo "============================================================"
