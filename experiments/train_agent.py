import os
import time
import torch
import numpy as np
from tqdm import tqdm

from configs import ConfigManager
from utils.logger import ExperimentLogger
from utils.seed_utils import set_global_seed
from utils.profiler import Profiler
from utils.registry import MODEL_REGISTRY

# Core Modules
from core.environment.vec_env import VehicularEdgeEnv
from core.agent.agent_adp_d3qn import ADP_D3QNAgent


def build_env_from_cfg(cfg: dict, logger: ExperimentLogger):

    raw_env_cfg = dict(cfg.get("environment", {}))

    # key mapping
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
    env_cfg.setdefault("num_vehicles", 10)
    env_cfg.setdefault("bandwidth_range", (5, 25))
    env_cfg.setdefault("seed", cfg["training"]["seed"] if "training" in cfg else 42)

    print("[EnvCFG] Using:", env_cfg)

    env = VehicularEdgeEnv(logger=logger, **env_cfg)
    return env


def decode_action(action_idx: int, model_layers: int, exit_points: int):

    partition_layer = action_idx // exit_points
    exit_point = (action_idx % exit_points) + 1  # exit point starts from 1
    return partition_layer, exit_point


def train(cfg_path: str):
    cfg = ConfigManager(cfg_path).config

    # Seed + device
    seed = cfg["training"]["seed"]
    set_global_seed(seed)

    device = cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

    exp_name = cfg["experiment"]["name"]
    output_dir = cfg["experiment"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logger = ExperimentLogger(
        log_dir=os.path.join(output_dir, "logs"),
        exp_name="default",
        enable_tensorboard=True
    )
    profiler = Profiler(exp_name=exp_name)

    print(f"\n[MEOCI] Starting training for experiment: {exp_name}")
    print(f"[Device] Using {device}\n")


    env = build_env_from_cfg(cfg, logger)


    model_type = cfg["model"]["type"]
    if model_type in MODEL_REGISTRY.list_all():
        model = MODEL_REGISTRY.build(model_type, num_classes=cfg["model"]["num_classes"])
    else:
        from core.model_zoo.vgg16_me import MultiExitVGG16
        model = MultiExitVGG16(num_classes=cfg["model"]["num_classes"])
    model.to(device)


    agent = ADP_D3QNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device,
        gamma=cfg["training"]["gamma"],
        lr=cfg["training"]["lr"],
        batch_size=cfg["training"].get("batch_size", 64),
        target_update_freq=cfg["training"].get("update_target_freq", 50),
        buffer_capacity=cfg["training"].get("buffer_capacity", 8000),
        logger=logger
    )

    print(f"[Init] Model: {cfg['model']['name']} | Agent: {cfg['agent']['name']}")
    print(f"[Env] Vehicles={env.num_vehicles}, ExitPoints={env.exit_points}, Layers={env.model_layers}")
    print(f"[Agent] state_dim={env.state_dim}, action_dim={env.action_dim}\n")

    episodes = 200
    max_steps_per_episode = cfg["training"].get("max_steps", 200)


    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_reward = -1e9

    all_rewards, all_latencies = [], []

    profiler.start(label="training_loop")

    for episode in tqdm(range(1, episodes + 1), desc="Training"):
        state = env.reset()
        total_reward = 0.0
        total_latency = 0.0
        done = False
        step_count = 0

        while not done and step_count < max_steps_per_episode:
            step_count += 1


            env_stats = env.get_env_stats()


            action_idx = agent.select_action(state, env_stats)


            action = decode_action(action_idx, env.model_layers, env.exit_points)

            next_state, reward, done, info = env.step(action)


            td_error = 0.0
            agent.store_transition(state, action_idx, reward, next_state, done, td_error)


            agent.train_step()

            state = next_state
            total_reward += float(reward)
            total_latency += float(info.get("avg_latency", 0.0))

        all_rewards.append(total_reward)


        ep_latency_avg = total_latency / max(env.current_step, 1)
        all_latencies.append(ep_latency_avg)


        logger.log_scalar("episode_reward", total_reward, episode)
        logger.log_scalar("episode_latency_avg", ep_latency_avg, episode)

        print(f"[Episode {episode}] Reward={total_reward:.3f} | LatencyAvgStep={ep_latency_avg:.2f} ms")


        if total_reward > best_reward:
            best_reward = total_reward
            best_path = os.path.join(ckpt_dir, f"best_{exp_name}.pth")
            agent.save(best_path)
            print(f"[Checkpoint] Saved BEST agent -> {best_path} (reward={best_reward:.4f})")

    profiler.stop()
    profiler.save(fmt="csv")


    final_path = os.path.join(ckpt_dir, f"final_{exp_name}.pth")
    agent.save(final_path)
    print(f"[Checkpoint] Saved FINAL agent -> {final_path}")

    avg_reward = np.mean(all_rewards)
    avg_latency = np.mean(all_latencies)

    print(f"\n✅ [Training Complete - 10ep Test] Avg Reward={avg_reward:.3f}, Avg Latency(per-step)={avg_latency:.2f} ms")


    logger.save_csv(
        os.path.join(output_dir, "reward_curve_test10.csv"),
        ["episode", "reward"],
        list(zip(range(1, episodes + 1), all_rewards))
    )
    logger.save_csv(
        os.path.join(output_dir, "latency_curve_test10.csv"),
        ["episode", "latency_avg_step_ms"],
        list(zip(range(1, episodes + 1), all_latencies))
    )

    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ADP-D3QN Agent in MEOCI Framework (10 episode checkpoint test)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    start_time = time.time()
    train(args.config)
    end_time = time.time()

    print(f"\n[MEOCI] Total training time: {(end_time - start_time) / 60:.2f} min")
