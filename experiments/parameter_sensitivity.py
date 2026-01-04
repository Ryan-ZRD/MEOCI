

import os
import csv
import numpy as np
from tqdm import tqdm
import torch

from configs import ConfigManager
from utils.logger import ExperimentLogger
from utils.seed_utils import set_global_seed
from core.environment.vec_env import VehicularEdgeEnv
from core.agent.agent_adp_d3qn import ADP_D3QNAgent
from core.model_zoo.resnet50_me import MultiExitResNet50



def run_episode(agent, env, max_steps=200):
    """Run one episode in the environment and return total reward."""
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < max_steps:
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        steps += 1

    return total_reward


def evaluate_parameter(env, model, param_name, param_values, cfg, episodes=5):
    """Evaluate the impact of one hyperparameter."""
    results = {}
    for val in tqdm(param_values, desc=f"Sensitivity: {param_name}"):
        if param_name == "lr":
            agent = ADP_D3QNAgent(env, model, lr=val, gamma=cfg["training"]["gamma"], device="cpu")
        elif param_name == "gamma":
            agent = ADP_D3QNAgent(env, model, lr=cfg["training"]["lr"], gamma=val, device="cpu")
        elif param_name == "buffer_size":
            agent = ADP_D3QNAgent(
                env, model, lr=cfg["training"]["lr"], gamma=cfg["training"]["gamma"], buffer_size=int(val), device="cpu"
            )
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        total_rewards = [run_episode(agent, env) for _ in range(episodes)]
        avg_reward = np.mean(total_rewards)
        results[val] = avg_reward
    return results



def parameter_sensitivity(cfg_path: str):
    """Run parameter sensitivity analysis."""
    cfg = ConfigManager(cfg_path).config
    set_global_seed(cfg["training"]["seed"])
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    logger = ExperimentLogger(log_dir="./results/logs/parameter_sensitivity")
    env = VehicularEdgeEnv(cfg["environment"])
    model = MultiExitResNet50(num_classes=cfg["model"]["num_classes"]).to(device)

    os.makedirs("./results/csv", exist_ok=True)

    # Define test ranges
    lr_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    gamma_values = [0.90, 0.93, 0.95, 0.97, 0.99]
    buffer_values = [5000, 10000, 20000, 40000, 80000]

    print("\n[Parameter Sensitivity Analysis Starting...]\n")

    # Evaluate each parameter independently
    results_lr = evaluate_parameter(env, model, "lr", lr_values, cfg)
    results_gamma = evaluate_parameter(env, model, "gamma", gamma_values, cfg)
    results_buffer = evaluate_parameter(env, model, "buffer_size", buffer_values, cfg)

    # Save results
    output_csv = "./results/csv/parameter_sensitivity.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value", "AverageReward"])
        for k, v in results_lr.items():
            writer.writerow(["lr", k, v])
        for k, v in results_gamma.items():
            writer.writerow(["gamma", k, v])
        for k, v in results_buffer.items():
            writer.writerow(["buffer_size", k, v])

    logger.log_dict({
        "lr": results_lr,
        "gamma": results_gamma,
        "buffer_size": results_buffer,
    })

    print("\n[Parameter Sensitivity Completed]")
    print(f"Results saved -> {output_csv}")
    print("You can visualize results with:")
    print("python visualization/scalability/plot_parameter_sensitivity.py --data results/csv/parameter_sensitivity.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter sensitivity analysis for MEOCI (ADP-D3QN)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/meoci_resnet50.yaml)")
    args = parser.parse_args()

    parameter_sensitivity(args.config)
