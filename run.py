import argparse
import os
import sys
import yaml
from datetime import datetime

# Experiment modules
from experiments.train_agent import train
from experiments.evaluate_latency import evaluate_model
from experiments.analyze_energy import analyze_energy
from experiments.test_multi_exit import test_multi_exit
from experiments.ablation_study import run_ablation
from experiments.heterogeneity_eval import heterogeneity_eval
from experiments.scalability_test import scalability_test
from experiments.parameter_sensitivity import parameter_sensitivity

# Monitoring tools
from deployment.monitoring import (
    start_all_monitors,
    start_meoci_dashboard,
    start_prometheus_exporter
)


def load_config(config_path: str):
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs():
    """Ensure standard directories exist."""
    dirs = ["results/logs", "results/csv", "results/plots", "saved_models"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def print_header():
    """Display MEOCI launch information."""
    print("=" * 70)
    print(" MEOCI: Multi-Exit Offloading and Cooperative Intelligence Framework")
    print(" Version: 1.0.0")
    print(" Author: Research Group @ Intelligent Edge Computing Lab")
    print("--------------------------------------------------------------")
    print(" Supported Modes: train | evaluate | visualize | monitor | ablation |")
    print("                  heterogeneity | scalability | sensitivity")
    print("=" * 70)
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="MEOCI Framework - Unified Launcher")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Training
    train_parser = subparsers.add_parser("train", help="Train ADP-D3QN agent")
    train_parser.add_argument("--config", required=True, help="Path to training config YAML")

    # Evaluation
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument("--metric", default="latency",
                             choices=["latency", "energy", "accuracy"],
                             help="Evaluation metric")
    eval_parser.add_argument("--model", default="vgg16", help="Target model name")

    # Visualization
    vis_parser = subparsers.add_parser("visualize", help="Reproduce paper figures")
    vis_parser.add_argument("--target", default="all", help="Visualization subset to export")

    # Monitoring
    mon_parser = subparsers.add_parser("monitor", help="Launch monitoring tools")
    mon_parser.add_argument("--mode", default="all", choices=["dashboard", "prometheus", "all"],
                            help="Monitoring service type")

    # Ablation
    abl_parser = subparsers.add_parser("ablation", help="Run ablation experiments")
    abl_parser.add_argument("--config", default="configs/ablation_scenarios.yaml")

    # Heterogeneity
    subparsers.add_parser("heterogeneity", help="Evaluate heterogeneous devices")

    # Scalability
    subparsers.add_parser("scalability", help="Run scalability tests")

    # Sensitivity
    subparsers.add_parser("sensitivity", help="Run parameter sensitivity analysis")

    return parser.parse_args()



def main():
    args = parse_args()
    ensure_dirs()
    print_header()

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] Experiment started at {start_time}")
    print(f"[MODE] {args.command.upper()}\n")

    if args.command == "train":
        config = load_config(args.config)
        train(config)

    elif args.command == "evaluate":
        if args.metric == "latency":
            evaluate_model(args.model)
        elif args.metric == "energy":
            analyze_energy()
        elif args.metric == "accuracy":
            test_multi_exit(args.model)
        else:
            print("[ERROR] Unsupported metric type.")

    elif args.command == "visualize":
        print(f"[INFO] Generating visualization for target: {args.target}")
        os.system(f"python visualization/export_all_figures.py --target {args.target}")

    elif args.command == "monitor":
        if args.mode == "dashboard":
            start_meoci_dashboard()
        elif args.mode == "prometheus":
            start_prometheus_exporter()
        else:
            start_all_monitors()

    elif args.command == "ablation":
        config = load_config(args.config)
        run_ablation(config)

    elif args.command == "heterogeneity":
        heterogeneity_eval()

    elif args.command == "scalability":
        scalability_test()

    elif args.command == "sensitivity":
        parameter_sensitivity()

    else:
        print("[ERROR] Unknown command. Use -h for help.")
        sys.exit(1)

    print("\n[INFO] Execution completed successfully.")



if __name__ == "__main__":
    main()
