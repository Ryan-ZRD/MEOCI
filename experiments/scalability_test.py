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
from core.model_zoo.alexnet_me import MultiExitAlexNet


def evaluate_vehicle_density(env, agent, model, density_list, episodes=3):
    """Measure inference latency and completion rate under different vehicle counts."""
    latency_results = {}
    completion_results = {}

    for n_vehicles in tqdm(density_list, desc="Vehicle Density Scaling"):
        env.set_vehicle_density(n_vehicles)
        total_latency, total_completion = [], []

        for _ in range(episodes):
            env.reset()
            done = False
            cumulative_latency = 0.0
            completed_tasks = 0
            total_tasks = 0

            while not done:
                action = agent.select_action(env.state, eval_mode=True)
                _, _, done, info = env.step(action)
                cumulative_latency += info.get("latency", 0.0)
                completed_tasks += info.get("completed_tasks", 0)
                total_tasks += info.get("total_tasks", 1)

            avg_latency = cumulative_latency / max(1, total_tasks)
            completion_rate = (completed_tasks / max(1, total_tasks)) * 100
            total_latency.append(avg_latency)
            total_completion.append(completion_rate)

        latency_results[n_vehicles] = np.mean(total_latency)
        completion_results[n_vehicles] = np.mean(total_completion)

    return latency_results, completion_results



def evaluate_transmission_rate(env, agent, model, rate_list, episodes=3):
    """Measure latency and completion rate under different transmission rates (Mbps)."""
    latency_results = {}
    completion_results = {}

    for rate in tqdm(rate_list, desc="Transmission Rate Scaling"):
        env.set_network_bandwidth(rate)
        total_latency, total_completion = [], []

        for _ in range(episodes):
            env.reset()
            done = False
            total_l, completed, total = 0.0, 0, 0
            while not done:
                action = agent.select_action(env.state, eval_mode=True)
                _, _, done, info = env.step(action)
                total_l += info.get("latency", 0.0)
                completed += info.get("completed_tasks", 0)
                total += info.get("total_tasks", 1)

            avg_l = total_l / max(1, total)
            comp = (completed / max(1, total)) * 100
            total_latency.append(avg_l)
            total_completion.append(comp)

        latency_results[rate] = np.mean(total_latency)
        completion_results[rate] = np.mean(total_completion)

    return latency_results, completion_results



def evaluate_delay_constraints(env, agent, model, delay_constraints, episodes=3):
    """Analyze how delay constraints affect accuracy and task completion."""
    accuracy_results = {}
    completion_results = {}

    for delay_limit in tqdm(delay_constraints, desc="Delay Constraint Scaling"):
        env.set_delay_constraint(delay_limit)
        total_acc, total_completion = [], []

        for _ in range(episodes):
            env.reset()
            done = False
            total_accuracy, completed, total = 0.0, 0, 0

            while not done:
                action = agent.select_action(env.state, eval_mode=True)
                _, _, done, info = env.step(action)
                total_accuracy += info.get("accuracy", 0.0)
                completed += info.get("completed_tasks", 0)
                total += info.get("total_tasks", 1)

            avg_acc = total_accuracy / max(1, total)
            completion = (completed / max(1, total)) * 100
            total_acc.append(avg_acc)
            total_completion.append(completion)

        accuracy_results[delay_limit] = np.mean(total_acc)
        completion_results[delay_limit] = np.mean(total_completion)

    return accuracy_results, completion_results



def scalability_test(cfg_path: str):
    """Run scalability experiments for vehicle density, rate, and delay constraints."""
    cfg = ConfigManager(cfg_path).config
    set_global_seed(cfg["training"]["seed"])
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    logger = ExperimentLogger(log_dir="./results/logs/scalability_test")
    env = VehicularEdgeEnv(cfg["environment"])
    model = MultiExitAlexNet(num_classes=cfg["model"]["num_classes"]).to(device)

    agent = ADP_D3QNAgent(
        env=env,
        model=model,
        lr=cfg["training"]["lr"],
        gamma=cfg["training"]["gamma"],
        device=device,
    )

    ckpt_path = os.path.join(cfg["logging"]["save_model_dir"], f"best_{cfg['experiment']['name']}.pth")
    if os.path.exists(ckpt_path):
        agent.load_checkpoint(ckpt_path)
        print(f"[Loaded agent from {ckpt_path}]")
    else:
        print("[Warning] No trained agent found — running in simulation mode.")

    os.makedirs("./results/csv", exist_ok=True)

    vehicle_counts = [5, 10, 15, 20, 25, 30]
    latency_v, completion_v = evaluate_vehicle_density(env, agent, model, vehicle_counts)
    with open("./results/csv/latency_vs_vehicle.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Vehicles", "Latency(ms)", "Completion(%)"])
        for n in vehicle_counts:
            writer.writerow([n, latency_v[n], completion_v[n]])

    rates = [5, 10, 15, 20, 25]
    latency_r, completion_r = evaluate_transmission_rate(env, agent, model, rates)
    with open("./results/csv/latency_vs_mbps.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Mbps", "Latency(ms)", "Completion(%)"])
        for r in rates:
            writer.writerow([r, latency_r[r], completion_r[r]])

    delay_limits = [15, 20, 25, 30]
    acc_d, completion_d = evaluate_delay_constraints(env, agent, model, delay_limits)
    with open("./results/csv/delay_constraints.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Delay(ms)", "Accuracy(%)", "Completion(%)"])
        for d in delay_limits:
            writer.writerow([d, acc_d[d], completion_d[d]])

    print("\n[Scalability Experiments Completed]")
    print("Results saved to ./results/csv/")
    logger.log_dict({
        "latency_vs_vehicle": latency_v,
        "completion_vs_vehicle": completion_v,
        "latency_vs_rate": latency_r,
        "completion_vs_rate": completion_r,
        "accuracy_vs_delay": acc_d,
        "completion_vs_delay": completion_d,
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scalability analysis for MEOCI (VEC inference)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/meoci_alexnet.yaml)")
    args = parser.parse_args()

    scalability_test(args.config)
