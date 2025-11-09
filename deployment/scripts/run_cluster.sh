#!/bin/bash
# ============================================================
# run_cluster.sh
# ------------------------------------------------------------
# Distributed deployment script for MEOCI across multiple nodes
# Supports:
#   - Edge node and multiple vehicle nodes
#   - Remote SSH execution
#   - Log collection and container health checks
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/deployment/docker/docker-compose.yml"
HOSTS_FILE="$PROJECT_ROOT/deployment/scripts/cluster_hosts.txt"
RESULTS_DIR="$PROJECT_ROOT/results"
MODE="${1:-start}"  # start | stop | status | logs | rebuild

print_header() {
  echo "============================================================"
  echo "MEOCI Distributed Deployment Script"
  echo "============================================================"
}

check_requirements() {
  if ! command -v ssh &> /dev/null; then
    echo "SSH not found. Please install OpenSSH client."
    exit 1
  fi
  if ! command -v docker &> /dev/null; then
    echo "Docker not found. Please install Docker first."
    exit 1
  fi
  if [ ! -f "$HOSTS_FILE" ]; then
    echo "Cluster hosts file not found: $HOSTS_FILE"
    echo "Create it with a list of nodes, e.g.:"
    echo "edge_server user@192.168.1.10"
    echo "vehicle_01 user@192.168.1.11"
    echo "vehicle_02 user@192.168.1.12"
    exit 1
  fi
}

prepare_dirs() {
  mkdir -p "$RESULTS_DIR"
}

run_remote() {
  local host_alias=$1
  local host_addr=$2
  local command=$3
  echo "------------------------------------------------------------"
  echo "[$host_alias] Executing: $command"
  ssh -o StrictHostKeyChecking=no "$host_addr" "$command"
}

deploy_on_node() {
  local host_alias=$1
  local host_addr=$2

  echo "[$host_alias] Building and starting containers..."
  ssh "$host_addr" "cd $PROJECT_ROOT && docker compose -f $COMPOSE_FILE up -d --build"
}

stop_on_node() {
  local host_alias=$1
  local host_addr=$2

  echo "[$host_alias] Stopping containers..."
  ssh "$host_addr" "cd $PROJECT_ROOT && docker compose -f $COMPOSE_FILE down"
}

status_on_node() {
  local host_alias=$1
  local host_addr=$2

  echo "[$host_alias] Checking container status..."
  ssh "$host_addr" "cd $PROJECT_ROOT && docker compose ps"
}

logs_on_node() {
  local host_alias=$1
  local host_addr=$2

  echo "[$host_alias] Tailing logs..."
  ssh "$host_addr" "cd $PROJECT_ROOT && docker compose logs --tail 50"
}

rebuild_on_node() {
  local host_alias=$1
  local host_addr=$2

  echo "[$host_alias] Rebuilding images and restarting containers..."
  ssh "$host_addr" "cd $PROJECT_ROOT && docker compose -f $COMPOSE_FILE down && docker compose -f $COMPOSE_FILE up -d --build"
}

execute_cluster() {
  while read -r alias addr; do
    [ -z "$alias" ] && continue
    case "$MODE" in
      start)
        deploy_on_node "$alias" "$addr"
        ;;
      stop)
        stop_on_node "$alias" "$addr"
        ;;
      status)
        status_on_node "$alias" "$addr"
        ;;
      logs)
        logs_on_node "$alias" "$addr"
        ;;
      rebuild)
        rebuild_on_node "$alias" "$addr"
        ;;
      *)
        echo "Invalid mode: $MODE"
        echo "Usage: $0 [start|stop|status|logs|rebuild]"
        exit 1
        ;;
    esac
  done < "$HOSTS_FILE"
}

print_header
check_requirements
prepare_dirs
execute_cluster

echo "============================================================"
echo "MEOCI cluster operation completed. Mode: $MODE"
echo "Results stored in: $RESULTS_DIR"
echo "============================================================"
