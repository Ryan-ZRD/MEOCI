

import os
import csv
import torch
import numpy as np
from tqdm import tqdm

# Project Imports
from configs import ConfigManager
from utils.logger import ExperimentLogger
from utils.seed_utils import set_global_seed
from core.environment.vec_env import VehicularEdgeEnv
from core.agent.agent_adp_d3qn import ADP_D3QNAgent
from core.model_zoo.vgg16_me import MultiExitVGG16
from core.model_zoo.alexnet_me import MultiExitAlexNet
from core.model_zoo.resnet50_me import MultiExitResNet50



def evaluate_model(env, agent, model, num_episodes: int = 10):
    """Evaluate latency and performance metrics."""
    model.eval()
    total_latency, total_reward, total_completion, total_accuracy = [], [], [], []

    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        state = env.reset()
        done = False
        ep_latency, ep_reward = 0.0, 0.0
        ep_tasks, ep_completed = 0, 0

        while not done:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, info = env.step(action)
            ep_latency += info.get("latency", 0)
            ep_reward += reward

            if info.get("completed", False):
                ep_completed += 1
            ep_tasks += 1
            state = next_state

        acc = info.get("accuracy", 0.0)
        total_latency.append(ep_latency)
        total_reward.append(ep_reward)
        total_completion.append(ep_completed / max(1, ep_tasks))
        total_accuracy.append(acc)

    return {
        "avg_latency": np.mean(total_latency),
        "avg_reward": np.mean(total_reward),
        "completion_rate": np.mean(total_completion),
        "avg_accuracy": np.mean(total_accuracy),
    }



def evaluate_heterogeneous(cfg_path: str):
    cfg = ConfigManager(cfg_path).config
    set_global_seed(cfg["training"]["seed"])
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    logger = ExperimentLogger(log_dir="./results/logs/evaluate_latency")

    # Different device profiles (simulate heterogeneous compute)
    devices = {
        "Jetson_Nano": {"compute_power": 1.0, "bandwidth": 20},
        "Raspberry_Pi4B": {"compute_power": 0.6, "bandwidth": 15},
    }

    results = []

    for dev_name, dev_conf in devices.items():
        print(f"\n[Device: {dev_name}] Evaluating...")

        # Update env params dynamically
        cfg["environment"]["bandwidth_mbps"] = [dev_conf["bandwidth"]]
        env = VehicularEdgeEnv(cfg["environment"])

        # Load trained agent + model
        model_type = cfg["model"]["name"]
        if model_type.startswith("alexnet"):
            model = MultiExitAlexNet(num_classes=cfg["model"]["num_classes"])
        elif model_type.startswith("vgg16"):
            model = MultiExitVGG16(num_classes=cfg["model"]["num_classes"])
        elif model_type.startswith("resnet50"):
            model = MultiExitResNet50(num_classes=cfg["model"]["num_classes"])
        else:
            raise ValueError(f"Unsupported model: {model_type}")

        model.to(device)
        agent = ADP_D3QNAgent(env, model, lr=cfg["training"]["lr"], gamma=cfg["training"]["gamma"], device=device)
        agent.load_checkpoint(os.path.join(cfg["logging"]["save_model_dir"], f"best_{cfg['experiment']['name']}.pth"))

        metrics = evaluate_model(env, agent, model, num_episodes=10)
        metrics["device"] = dev_name
        results.append(metrics)

        logger.log_dict(metrics)

    # Save CSV for plotting (Fig.9)
    output_path = "./results/csv/heterogeneous_latency.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["device", "avg_latency", "completion_rate", "avg_accuracy", "avg_reward"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n✅ [Heterogeneous Evaluation] Results saved -> {output_path}")



def evaluate_vehicle_density(cfg_path: str):
    cfg = ConfigManager(cfg_path).config
    set_global_seed(cfg["training"]["seed"])
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    logger = ExperimentLogger(log_dir="./results/logs/evaluate_vehicle_density")

    vehicle_counts = [5, 10, 15, 20, 25, 30]
    results = []

    model_type = cfg["model"]["name"]
    if model_type.startswith("alexnet"):
        model = MultiExitAlexNet(num_classes=cfg["model"]["num_classes"])
    elif model_type.startswith("vgg16"):
        model = MultiExitVGG16(num_classes=cfg["model"]["num_classes"])
    elif model_type.startswith("resnet50"):
        model = MultiExitResNet50(num_classes=cfg["model"]["num_classes"])
    else:
        raise ValueError(f"Unsupported model: {model_type}")

    model.to(device)
    agent = ADP_D3QNAgent(env=None, model=model, lr=cfg["training"]["lr"], gamma=cfg["training"]["gamma"], device=device)
    agent.load_checkpoint(os.path.join(cfg["logging"]["save_model_dir"], f"best_{cfg['experiment']['name']}.pth"))

    for v_count in vehicle_counts:
        print(f"\n[Vehicles={v_count}] Evaluating...")

        cfg["environment"]["vehicles"] = v_count
        env = VehicularEdgeEnv(cfg["environment"])
        metrics = evaluate_model(env, agent, model, num_episodes=10)
        metrics["vehicles"] = v_count
        results.append(metrics)

        logger.log_dict(metrics)

    # Save CSV for plotting (Fig.10)
    output_path = "./results/csv/latency_vs_vehicle.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["vehicles", "avg_latency", "completion_rate", "avg_accuracy", "avg_reward"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n✅ [Vehicle Density Evaluation] Results saved -> {output_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate MEOCI latency and performance metrics")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., meoci_vgg16.yaml)")
    parser.add_argument("--mode", type=str, default="hetero", choices=["hetero", "vehicle"],
                        help="Evaluation mode: hetero (Fig.9) or vehicle (Fig.10)")
    args = parser.parse_args()

    if args.mode == "hetero":
        evaluate_heterogeneous(args.config)
    elif args.mode == "vehicle":
        evaluate_vehicle_density(args.config)
