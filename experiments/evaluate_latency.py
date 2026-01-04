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

    # key mapping (compat)
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

    # defaults
    env_cfg.setdefault("model_layers", 10)
    env_cfg.setdefault("exit_points", 4)
    env_cfg.setdefault("num_vehicles", 20)
    env_cfg.setdefault("bandwidth_range", (5, 25))
    env_cfg.setdefault("seed", cfg["training"]["seed"] if "training" in cfg else 42)

    # override filtered cfg
    if override:
        env_cfg.update(override)

    print("[Eval-EnvCFG] Using:", env_cfg)
    return VehicularEdgeEnv(logger=logger, **env_cfg)


def decode_action(action_idx: int, model_layers: int, exit_points: int):

    partition_layer = int(action_idx // exit_points)
    exit_point = int((action_idx % exit_points) + 1)  # 1..exit_points
    return partition_layer, exit_point


def run_one_episode(env: VehicularEdgeEnv, agent: ADP_D3QNAgent, max_steps: int = 200):

    state = env.reset()
    done = False
    step_count = 0

    ep_reward = 0.0
    ep_latency_sum = 0.0
    ep_energy_sum = 0.0
    ep_acc_sum = 0.0

    while not done and step_count < max_steps:
        step_count += 1

        env_stats = env.get_env_stats()
        action_idx = agent.select_action(state, env_stats)

        action = decode_action(action_idx, env.model_layers, env.exit_points)
        next_state, reward, done, info = env.step(action)

        ep_reward += float(reward)
        ep_latency_sum += float(info.get("avg_latency", 0.0))
        ep_energy_sum += float(info.get("avg_energy", 0.0))
        ep_acc_sum += float(info.get("avg_accuracy", 0.0))

        state = next_state

    # per-step averages
    steps = max(step_count, 1)
    return {
        "episode_reward": ep_reward,
        "episode_latency_sum_ms": ep_latency_sum,
        "episode_latency_avg_step_ms": ep_latency_sum / steps,
        "episode_energy_avg_step": ep_energy_sum / steps,
        "episode_accuracy_avg_step": ep_acc_sum / steps,
        "steps": steps
    }


def evaluate(env, agent, num_episodes: int = 10, max_steps: int = 200):
    all_rewards = []
    all_lat_sum = []
    all_lat_avg = []
    all_energy_avg = []
    all_acc_avg = []

    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        m = run_one_episode(env, agent, max_steps=max_steps)
        all_rewards.append(m["episode_reward"])
        all_lat_sum.append(m["episode_latency_sum_ms"])
        all_lat_avg.append(m["episode_latency_avg_step_ms"])
        all_energy_avg.append(m["episode_energy_avg_step"])
        all_acc_avg.append(m["episode_accuracy_avg_step"])

    return {
        "avg_reward": float(np.mean(all_rewards)),
        "avg_latency_sum_ms": float(np.mean(all_lat_sum)),
        "avg_latency_per_step_ms": float(np.mean(all_lat_avg)),
        "avg_energy_per_step": float(np.mean(all_energy_avg)),
        "avg_accuracy": float(np.mean(all_acc_avg)),
    }



def evaluate_heterogeneous(cfg_path: str, ckpt_path: str):
    cfg = ConfigManager(cfg_path).config
    set_global_seed(cfg["training"]["seed"])
    device = cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

    logger = ExperimentLogger(log_dir="./results/logs/evaluate_hetero", enable_tensorboard=False)

    # simulate different bandwidth ranges
    devices = {
        "Jetson_Nano": {"bandwidth_range": (15, 25)},
        "Raspberry_Pi4B": {"bandwidth_range": (8, 18)},
    }

    results = []
    for dev_name, dev_conf in devices.items():
        print(f"\n[Device: {dev_name}] Evaluating...")

        env = build_env_from_cfg(cfg, logger, override={"bandwidth_range": dev_conf["bandwidth_range"]})

        agent = ADP_D3QNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            device=device,
            gamma=cfg["training"].get("gamma", 0.98),
            lr=cfg["training"].get("lr", 1e-4),
            logger=logger
        )
        agent.load(ckpt_path, map_location=device)

        metrics = evaluate(env, agent, num_episodes=10, max_steps=200)
        metrics["device"] = dev_name
        metrics["bandwidth_range"] = str(dev_conf["bandwidth_range"])
        results.append(metrics)

        print("[Result]", metrics)

    # save CSV
    out_path = "./results/csv/heterogeneous_latency.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="") as f:
        fieldnames = [
            "device", "bandwidth_range",
            "avg_latency_per_step_ms", "avg_latency_sum_ms",
            "avg_accuracy", "avg_energy_per_step", "avg_reward"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\n✅ [Heterogeneous Evaluation] Results saved -> {out_path}")
    logger.close()


def evaluate_vehicle_density(cfg_path: str, ckpt_path: str):
    cfg = ConfigManager(cfg_path).config
    set_global_seed(cfg["training"]["seed"])
    device = cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

    logger = ExperimentLogger(log_dir="./results/logs/evaluate_vehicle_density", enable_tensorboard=False)

    vehicle_counts = [5, 10, 15, 20, 25, 30]
    results = []

    for v_count in vehicle_counts:
        print(f"\n[Vehicles={v_count}] Evaluating...")

        env = build_env_from_cfg(cfg, logger, override={"num_vehicles": v_count})

        agent = ADP_D3QNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            device=device,
            gamma=cfg["training"].get("gamma", 0.98),
            lr=cfg["training"].get("lr", 1e-4),
            logger=logger
        )
        agent.load(ckpt_path, map_location=device)

        metrics = evaluate(env, agent, num_episodes=10, max_steps=200)
        metrics["vehicles"] = v_count
        results.append(metrics)

        print("[Result]", metrics)

    out_path = "./results/csv/latency_vs_vehicle.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="") as f:
        fieldnames = [
            "vehicles",
            "avg_latency_per_step_ms", "avg_latency_sum_ms",
            "avg_accuracy", "avg_energy_per_step", "avg_reward"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\n [Vehicle Density Evaluation] Results saved -> {out_path}")
    logger.close()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate MEOCI latency/performance (compatible with fixed env/agent)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/meoci_vgg16.yaml)")
    parser.add_argument("--mode", type=str, default="hetero", choices=["hetero", "vehicle"],
                        help="Evaluation mode: hetero or vehicle")
    parser.add_argument("--ckpt", type=str, default="best", choices=["best", "final"],
                        help="Which checkpoint to use (best or final)")
    args = parser.parse_args()

    cfg = ConfigManager(args.config).config
    exp_name = cfg["experiment"]["name"]
    output_dir = cfg["experiment"]["output_dir"]
    ckpt_dir = os.path.join(output_dir, "checkpoints")

    ckpt_path = os.path.join(ckpt_dir, f"{args.ckpt}_{exp_name}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[Eval] Checkpoint not found: {ckpt_path}")

    print(f"[Eval] Using checkpoint: {ckpt_path}")

    if args.mode == "hetero":
        evaluate_heterogeneous(args.config, ckpt_path)
    else:
        evaluate_vehicle_density(args.config, ckpt_path)
