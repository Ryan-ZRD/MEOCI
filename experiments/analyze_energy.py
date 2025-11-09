"""
experiments.analyze_energy
----------------------------------------------------------
Analyze the impact of energy consumption constraints on
inference latency under the MEOCI framework.
Reproduces Fig. 13 of the paper.
"""

import os
import csv
import torch
import numpy as np
from tqdm import tqdm

from configs import ConfigManager
from utils.logger import ExperimentLogger
from utils.seed_utils import set_global_seed
from core.environment.vec_env import VehicularEdgeEnv
from core.agent.agent_adp_d3qn import ADP_D3QNAgent
from core.model_zoo.resnet50_me import MultiExitResNet50
from core.model_zoo.yolov10_me import MultiExitYOLOv10n


def evaluate_energy_constraint(env, agent, model, energy_budget_mj, episodes=8):
    """Evaluate latency and reward under a given energy constraint."""
    model.eval()
    env.set_energy_constraint(energy_budget_mj)

    total_latency, total_energy, total_reward = [], [], []

    for _ in tqdm(range(episodes), desc=f"Energy {energy_budget_mj} mJ"):
        state = env.reset()
        done = False
        ep_latency, ep_energy, ep_reward = 0.0, 0.0, 0.0

        while not done:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, info = env.step(action)

            ep_latency += info.get("latency", 0)
            ep_energy += info.get("energy", 0)
            ep_reward += reward

            state = next_state

        total_latency.append(ep_latency)
        total_energy.append(ep_energy)
        total_reward.append(ep_reward)

    return {
        "energy_budget_mJ": energy_budget_mj,
        "avg_latency_ms": np.mean(total_latency),
        "avg_energy_mJ": np.mean(total_energy),
        "avg_reward": np.mean(total_reward),
    }


def analyze_energy(cfg_path: str):
    """Run energy constraint analysis for MEOCI."""
    cfg = ConfigManager(cfg_path).config
    set_global_seed(cfg["training"]["seed"])
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    logger = ExperimentLogger(log_dir="./results/logs/energy_analysis")

    # Define test range of energy budgets (in mJ)
    energy_budgets = [200, 400, 600, 800, 1000, 1200]

    # Choose model (ResNet50 / YOLOv10n)
    model_name = cfg["model"]["name"]
    if model_name.startswith("resnet50"):
        model = MultiExitResNet50(num_classes=cfg["model"]["num_classes"])
    elif model_name.startswith("yolov10"):
        model = MultiExitYOLOv10n(num_classes=cfg["model"]["num_classes"])
    else:
        raise ValueError(f"Unsupported model type for energy experiment: {model_name}")

    model.to(device)
    env = VehicularEdgeEnv(cfg["environment"])

    agent = ADP_D3QNAgent(
        env=env,
        model=model,
        lr=cfg["training"]["lr"],
        gamma=cfg["training"]["gamma"],
        device=device
    )

    ckpt_path = os.path.join(cfg["logging"]["save_model_dir"], f"best_{cfg['experiment']['name']}.pth")
    if os.path.exists(ckpt_path):
        agent.load_checkpoint(ckpt_path)
        print(f"[Loaded pretrained agent from {ckpt_path}]")
    else:
        print(f"[Warning] No pretrained agent found, evaluating untrained agent.")

    results = []
    for budget in energy_budgets:
        metrics = evaluate_energy_constraint(env, agent, model, budget)
        results.append(metrics)
        logger.log_dict(metrics)

    # Save results to CSV
    output_path = "./results/csv/energy_constraints.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["energy_budget_mJ", "avg_latency_ms", "avg_energy_mJ", "avg_reward"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n[Energy Analysis Completed] Results saved to {output_path}")
    print("You can now run:")
    print("python visualization/energy_constraints/plot_latency_vs_energy.py --data results/csv/energy_constraints.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze the impact of energy constraints on MEOCI inference latency")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., meoci_resnet50.yaml)")
    args = parser.parse_args()

    analyze_energy(args.config)
