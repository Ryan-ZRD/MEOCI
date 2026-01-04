import os
import csv
import torch
from tqdm import tqdm

from configs import ConfigManager
from utils.logger import ExperimentLogger
from utils.seed_utils import set_global_seed
from utils.checkpoint import CheckpointManager
from core.environment.vec_env import VehicularEdgeEnv
from core.agent.agent_adp_d3qn import ADP_D3QNAgent
from core.agent.agent_baselines import D3QNAgent, AD3QNAgent, DP_D3QNAgent
from core.model_zoo.vgg16_me import MultiExitVGG16


def train_agent_variant(agent, env, episodes, update_target_freq, logger, save_dir, label):

    rewards = []
    best_reward = -float("inf")

    for ep in tqdm(range(1, episodes + 1), desc=f"Training {label}"):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.optimize_model()
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        logger.log_scalar(f"reward_{label}", total_reward, ep)

        if ep % update_target_freq == 0:
            agent.update_target_network()

        if total_reward > best_reward:
            best_reward = total_reward
            CheckpointManager.save_checkpoint(agent.q_network, os.path.join(save_dir, f"best_{label}.pth"))

    return rewards


def run_ablation(cfg_path: str):

    cfg = ConfigManager(cfg_path).config
    set_global_seed(cfg["training"]["seed"])
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    exp_dir = "./results/ablation"
    os.makedirs(exp_dir, exist_ok=True)
    logger = ExperimentLogger(log_dir=exp_dir)

    env = VehicularEdgeEnv(cfg["environment"])
    model = MultiExitVGG16(num_classes=cfg["model"]["num_classes"]).to(device)

    episodes = cfg["training"]["epochs"]
    update_target_freq = cfg["training"]["update_target_freq"]

    variants = {
        "D3QN": D3QNAgent(env, model, lr=cfg["training"]["lr"], gamma=cfg["training"]["gamma"], device=device),
        "A-D3QN": AD3QNAgent(env, model, lr=cfg["training"]["lr"], gamma=cfg["training"]["gamma"], device=device),
        "DP-D3QN": DP_D3QNAgent(env, model, lr=cfg["training"]["lr"], gamma=cfg["training"]["gamma"], device=device),
        "ADP-D3QN": ADP_D3QNAgent(env, model, lr=cfg["training"]["lr"], gamma=cfg["training"]["gamma"], device=device),
    }

    all_rewards = {}

    print("\n[Starting Ablation Study: Comparing D3QN Variants]\n")
    for label, agent in variants.items():
        print(f"→ Training variant: {label}")
        rewards = train_agent_variant(agent, env, episodes, update_target_freq, logger, exp_dir, label)
        all_rewards[label] = rewards

    # Save CSV for visualization
    csv_path = os.path.join(exp_dir, "ablation_reward.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Episode"] + list(variants.keys())
        writer.writerow(header)

        for ep in range(episodes):
            row = [ep + 1] + [all_rewards[v][ep] if ep < len(all_rewards[v]) else None for v in variants.keys()]
            writer.writerow(row)

    print(f"\n[Ablation Completed] Results saved -> {csv_path}")
    print("You can now plot convergence with:")
    print("python visualization/ablation/plot_ablation_convergence.py --data results/ablation/ablation_reward.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation study for MEOCI agent variants")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/ablation_scenarios.yaml)")
    args = parser.parse_args()

    run_ablation(args.config)
