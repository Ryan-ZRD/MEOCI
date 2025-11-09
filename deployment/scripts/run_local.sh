#!/bin/bash
# ============================================================
# run_local.sh
# ------------------------------------------------------------
# One-click launcher for MEOCI docker-compose deployment
# - Builds and starts the edge and vehicle containers
# - Streams logs in real-time
# - Provides cleanup and restart options
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/deployment/docker/docker-compose.yml"
RESULTS_DIR="$PROJECT_ROOT/results"
MODE="${1:-up}"  # Default mode: up | down | restart | logs

# -----------------------------
# Utility functions
# -----------------------------

print_header() {
  echo "============================================================"
  echo "MEOCI Deployment Script"
  echo "============================================================"
}

check_docker() {
  if ! command -v docker &> /dev/null; then
    echo " Docker not found. Please install Docker first."
    exit 1
  fi

  if ! docker compose version &> /dev/null; then
    echo " Docker Compose plugin not found. Please ensure it's installed."
    exit 1
  fi
}

check_gpu() {
  echo "🔍 Checking for NVIDIA GPU availability..."
  if ! docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi > /dev/null 2>&1; then
    echo "️  GPU runtime not detected. Continuing with CPU mode..."
  else
    echo "✅ GPU detected and accessible."
  fi
}

prepare_dirs() {
  mkdir -p "$RESULTS_DIR"
  echo " Ensured results directory exists: $RESULTS_DIR"
}

# -----------------------------
# Docker Compose actions
# -----------------------------

compose_up() {
  echo " Starting MEOCI containers..."
  docker compose -f "$COMPOSE_FILE" up -d --build
  echo " Containers are starting in detached mode."
  echo " Use './deployment/scripts/run_local.sh logs' to follow logs."
}

compose_down() {
  echo " Stopping and removing containers..."
  docker compose -f "$COMPOSE_FILE" down
  echo "✅ Containers stopped and removed."
}

compose_restart() {
  echo " Restarting MEOCI deployment..."
  docker compose -f "$COMPOSE_FILE" down
  docker compose -f "$COMPOSE_FILE" up -d --build
  echo "✅ Restart complete."
}

compose_logs() {
  echo " Streaming container logs..."
  docker compose -f "$COMPOSE_FILE" logs -f --tail 50
}

# -----------------------------
# Main control logic
# -----------------------------

print_header
check_docker
prepare_dirs

case "$MODE" in
  up)
    check_gpu
    compose_up
    ;;
  down)
    compose_down
    ;;
  restart)
    check_gpu
    compose_restart
    ;;
  logs)
    compose_logs
    ;;
  *)
    echo "Usage: $0 [up|down|restart|logs]"
    exit 1
    ;;
esac

echo "============================================================"
echo " MEOCI local deployment completed."
echo "Results stored under: $RESULTS_DIR"
echo "============================================================"
