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



def build_env_from_cfg(cfg: dict, logger: ExperimentLogger, override: dict = None):
    raw_env_cfg = dict(cfg.get("environment", {}))

    # mapping old keys -> new keys
    if "vehicles" in raw_env_cfg and "num_vehicles" not in raw_env_cfg:
        raw_env_cfg["num_vehicles"] = raw_env_cfg.pop("vehicles")
    if "layers" in raw_env_cfg and "model_layers" not in raw_env_cfg:
        raw_env_cfg["model_layers"] = raw_env_cfg.pop("layers")
    if "exits" in raw_env_cfg and "exit_points" not in raw_env_cfg:
        raw_env_cfg["exit_points"] = raw_env_cfg.pop("exits")
    if "random_seed" in raw_env_cfg and "seed" not in raw_env_cfg:
        raw_env_cfg["seed"] = raw_env_cfg.pop("random_seed")

    allowed_keys = {"model_layers", "exit_points", "num_vehicles", "bandwidth_range", "seed"}
    env_cfg = {k: v for k, v in raw_env_cfg.items() if k in allowed_keys}

    env_cfg.setdefault("model_layers", 10)
    env_cfg.setdefault("exit_points", 4)
    env_cfg.setdefault("num_vehicles", 20)
    env_cfg.setdefault("bandwidth_range", (5, 25))
    env_cfg.setdefault("seed", cfg["training"]["seed"] if "training" in cfg else 42)

    if override:
        env_cfg.update(override)

    print("[Energy-EnvCFG] Using:", env_cfg)
    return VehicularEdgeEnv(logger=logger, **env_cfg)


def decode_action(action_idx: int, exit_points: int):

    partition_layer = int(action_idx // exit_points)
    exit_point = int((action_idx % exit_points) + 1)
    return partition_layer, exit_point



def evaluate_energy_constraint(env, agent, energy_budget_mj: float, episodes: int = 8, max_steps: int = 200):


    total_latency_sum = []
    total_latency_avg = []
    total_energy_sum = []
    total_reward = []

    for _ in tqdm(range(episodes), desc=f"EnergyBudget={energy_budget_mj}mJ"):
        state = env.reset()
        done = False
        step_count = 0

        ep_latency_sum = 0.0
        ep_energy_sum = 0.0
        ep_reward = 0.0

        while not done and step_count < max_steps:
            step_count += 1

            env_stats = env.get_env_stats()
            action_idx = agent.select_action(state, env_stats)

            action = decode_action(action_idx, env.exit_points)
            next_state, reward, done, info = env.step(action)


            step_latency = float(info.get("avg_latency", 0.0))
            step_energy = float(info.get("avg_energy", 0.0))

            ep_latency_sum += step_latency
            ep_energy_sum += step_energy
            ep_reward += float(reward)


            if ep_energy_sum > energy_budget_mj:
                done = True

            state = next_state

        steps = max(step_count, 1)
        total_latency_sum.append(ep_latency_sum)
        total_latency_avg.append(ep_latency_sum / steps)
        total_energy_sum.append(ep_energy_sum)
        total_reward.append(ep_reward)

    return {
        "energy_budget_mJ": energy_budget_mj,
        "avg_latency_sum_ms": float(np.mean(total_latency_sum)),
        "avg_latency_per_step_ms": float(np.mean(total_latency_avg)),
        "avg_energy_mJ": float(np.mean(total_energy_sum)),
        "avg_reward": float(np.mean(total_reward)),
    }



def analyze_energy(cfg_path: str, ckpt: str = "best"):
    cfg = ConfigManager(cfg_path).config
    set_global_seed(cfg["training"]["seed"])

    device = cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

    exp_name = cfg["experiment"]["name"]
    output_dir = cfg["experiment"]["output_dir"]
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, f"{ckpt}_{exp_name}.pth")

    logger = ExperimentLogger(log_dir="./results/logs/energy_analysis", enable_tensorboard=False)


    energy_budgets = [200, 400, 600, 800, 1000, 1200]


    env = build_env_from_cfg(cfg, logger)


    agent = ADP_D3QNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device,
        gamma=cfg["training"].get("gamma", 0.98),
        lr=cfg["training"].get("lr", 1e-4),
        logger=logger,
    )


    if os.path.exists(ckpt_path):
        agent.load(ckpt_path, map_location=device)
        print(f"[Energy] Loaded agent checkpoint -> {ckpt_path}")
    else:
        print(f"[Energy-Warning] Checkpoint not found: {ckpt_path}")
        print("[Energy] Will evaluate with untrained agent!")

    results = []
    for budget in energy_budgets:
        metrics = evaluate_energy_constraint(env, agent, energy_budget_mj=budget, episodes=8, max_steps=200)
        results.append(metrics)
        print("[Energy-Result]", metrics)

    output_path = "./results/csv/energy_constraints.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "energy_budget_mJ",
                "avg_latency_per_step_ms",
                "avg_latency_sum_ms",
                "avg_energy_mJ",
                "avg_reward"
            ]
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n [Energy Analysis Completed] Results saved -> {output_path}")
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze energy constraints impact (compatible with fixed env/agent)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", type=str, default="best", choices=["best", "final"],
                        help="Use best or final checkpoint in results/<exp>/checkpoints")
    args = parser.parse_args()

    analyze_energy(args.config, ckpt=args.ckpt)
