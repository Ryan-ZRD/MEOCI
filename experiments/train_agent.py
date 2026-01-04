import os
import time
import torch
import numpy as np
from tqdm import tqdm

# Project Imports
from configs import ConfigManager
from utils.logger import ExperimentLogger
from utils.checkpoint import CheckpointManager
from utils.seed_utils import set_global_seed

from utils.profiler import Profiler
from utils.registry import MODEL_REGISTRY, AGENT_REGISTRY, ENV_REGISTRY

# Core Modules
from core.environment.vec_env import VehicularEdgeEnv
from core.agent.agent_adp_d3qn import ADP_D3QNAgent



def train(cfg_path: str):

    cfg = ConfigManager(cfg_path).config
    set_global_seed(cfg["training"]["seed"])
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    exp_name = cfg["experiment"]["name"]
    output_dir = cfg["experiment"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logger = ExperimentLogger(log_dir=os.path.join(output_dir, "logs"))
    profiler = Profiler(exp_name=exp_name)

    print(f"\n[MEOCI] Starting training for experiment: {exp_name}")
    print(f"[Device] Using {device}\n")


    env = VehicularEdgeEnv(cfg["environment"])
    model_type = cfg["model"]["type"]


    if model_type in MODEL_REGISTRY.list_all():
        model = MODEL_REGISTRY.build(model_type, num_classes=cfg["model"]["num_classes"])
    else:
        from core.model_zoo.vgg16_me import MultiExitVGG16
        model = MultiExitVGG16(num_classes=cfg["model"]["num_classes"])

    model.to(device)

    agent = ADP_D3QNAgent(
        env=env,
        model=model,
        lr=cfg["training"]["lr"],
        gamma=cfg["training"]["gamma"],
        epsilon_cfg=cfg["training"]["epsilon"],
        dual_buffer=cfg["agent"]["dual_buffer"],
        reward_fn=cfg["agent"]["reward_function"],
        device=device,
    )

    print(f"[Init] Model: {cfg['model']['name']} | Agent: {cfg['agent']['name']}")
    print(f"[Env] Vehicles={env.num_vehicles}, RSUs={env.num_rsus}")


    episodes = cfg["training"]["epochs"]
    update_target_freq = cfg["training"]["update_target_freq"]

    all_rewards, all_latencies = [], []

    profiler.start(label="training_loop")

    for episode in tqdm(range(1, episodes + 1), desc="Training"):
        state = env.reset()
        total_reward, total_latency = 0, 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.optimize_model()

            state = next_state
            total_reward += reward
            total_latency += info.get("latency", 0)

        # Update target network
        if episode % update_target_freq == 0:
            agent.update_target_network()

        all_rewards.append(total_reward)
        all_latencies.append(total_latency)

        # Logging
        logger.log_scalar("reward", total_reward, episode)
        logger.log_scalar("latency", total_latency, episode)

        if episode % cfg["evaluation"]["log_interval"] == 0:
            print(f"[Episode {episode}] Reward={total_reward:.3f} | Latency={total_latency:.2f} ms")

        # Save best model
        if cfg["evaluation"]["save_best_model"] and total_reward >= max(all_rewards):
            best_path = os.path.join(cfg["logging"]["save_model_dir"], f"best_{exp_name}.pth")
            CheckpointManager.save_checkpoint(model, best_path)
            print(f"[Checkpoint] Saved best model at episode {episode} -> {best_path}")

    profiler.stop()
    profiler.save(fmt="csv")


    avg_reward = np.mean(all_rewards[-10:])
    avg_latency = np.mean(all_latencies[-10:])
    print(f"\n✅ [Training Complete] Avg Reward={avg_reward:.3f}, Avg Latency={avg_latency:.2f} ms")

    # Save reward curve for Fig.7
    logger.save_csv(os.path.join(output_dir, "reward_curve.csv"), ["episode", "reward"], list(zip(range(episodes), all_rewards)))
    logger.save_csv(os.path.join(output_dir, "latency_curve.csv"), ["episode", "latency"], list(zip(range(episodes), all_latencies)))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ADP-D3QN Agent in MEOCI Framework")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    start_time = time.time()
    train(args.config)
    end_time = time.time()

    print(f"\n[MEOCI] Total training time: {(end_time - start_time) / 60:.2f} min")
