"""
experiments.distributed_training
----------------------------------------------------------
Simulate distributed ADP-D3QN training across multiple RSU and vehicle nodes.
Reproduces distributed learning efficiency and convergence experiments
(Fig.13–Fig.15 in the MEOCI paper).
"""

import os
import time
import copy
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from configs import ConfigManager
from utils.logger import ExperimentLogger
from utils.seed_utils import set_global_seed
from utils.checkpoint import CheckpointManager
from core.environment.vec_env import VehicularEdgeEnv
from core.agent.agent_adp_d3qn import ADP_D3QNAgent
from core.model_zoo.resnet50_me import MultiExitResNet50


# ---------------------------------------------------------
# Node simulation
# ---------------------------------------------------------
class DistributedNode:
    """A simulated RSU or vehicle node participating in distributed training."""

    def __init__(self, node_id, env_config, model, lr, gamma, device):
        self.id = node_id
        self.env = VehicularEdgeEnv(env_config)
        self.model = copy.deepcopy(model)
        self.agent = ADP_D3QNAgent(self.env, self.model, lr=lr, gamma=gamma, device=device)
        self.reward_log = []

    def train_one_epoch(self, steps=100):
        """Train locally for several steps."""
        state = self.env.reset()
        total_reward = 0.0
        for _ in range(steps):
            action = self.agent.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.replay_buffer.push(state, action, reward, next_state, done)
            self.agent.optimize_model()
            state = next_state
            total_reward += reward
            if done:
                break
        self.reward_log.append(total_reward)
        return total_reward

    def get_parameters(self):
        """Return local model parameters."""
        return {k: v.cpu().clone() for k, v in self.agent.q_network.state_dict().items()}

    def set_parameters(self, new_params):
        """Update local model parameters."""
        self.agent.q_network.load_state_dict(new_params, strict=True)


# ---------------------------------------------------------
# Parameter Aggregation
# ---------------------------------------------------------
def aggregate_parameters(node_params_list):
    """Federated averaging (FedAvg) for distributed agents."""
    aggregated = copy.deepcopy(node_params_list[0])
    for key in aggregated.keys():
        for node_params in node_params_list[1:]:
            aggregated[key] += node_params[key]
        aggregated[key] = aggregated[key] / len(node_params_list)
    return aggregated


# ---------------------------------------------------------
# Distributed Training Simulation
# ---------------------------------------------------------
def distributed_training(cfg_path: str):
    """Simulate distributed ADP-D3QN training with multiple nodes."""
    cfg = ConfigManager(cfg_path).config
    set_global_seed(cfg["training"]["seed"])
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    logger = ExperimentLogger(log_dir="./results/logs/distributed_training")
    os.makedirs("./results/csv", exist_ok=True)

    base_model = MultiExitResNet50(num_classes=cfg["model"]["num_classes"]).to(device)

    # Create simulated nodes (1 RSU master + N edge clients)
    num_nodes = cfg["distributed"].get("num_nodes", 3)
    nodes = [
        DistributedNode(
            node_id=i,
            env_config=cfg["environment"],
            model=base_model,
            lr=cfg["training"]["lr"],
            gamma=cfg["training"]["gamma"],
            device=device,
        )
        for i in range(num_nodes)
    ]

    global_rewards = []
    aggregation_interval = cfg["distributed"].get("aggregation_interval", 5)
    epochs = cfg["training"].get("epochs", 50)

    print(f"\n[Distributed Training Started] Nodes={num_nodes}, Epochs={epochs}\n")

    # Training loop
    for epoch in tqdm(range(1, epochs + 1)):
        local_params = []
        local_rewards = []

        # Parallel simulation of local training
        for node in nodes:
            reward = node.train_one_epoch(steps=cfg["distributed"].get("local_steps", 100))
            local_rewards.append(reward)
            local_params.append(node.get_parameters())

        # Global aggregation every few epochs
        if epoch % aggregation_interval == 0:
            aggregated_params = aggregate_parameters(local_params)
            for node in nodes:
                node.set_parameters(aggregated_params)
            avg_reward = np.mean(local_rewards)
            global_rewards.append(avg_reward)
            print(f"[Epoch {epoch}] Aggregated across nodes | Avg Reward: {avg_reward:.2f}")

            # Save global checkpoint (for analysis)
            CheckpointManager.save_checkpoint(nodes[0].agent.q_network, f"./results/distributed/global_epoch{epoch}.pth")

    # Save results to CSV
    csv_path = "./results/csv/distributed_training.csv"
    with open(csv_path, "w", newline="") as f:
        f.write("Epoch,AverageReward\n")
        for i, r in enumerate(global_rewards):
            f.write(f"{(i + 1) * aggregation_interval},{r:.4f}\n")

    logger.log_dict({"global_rewards": global_rewards})
    print(f"\n[Distributed Training Completed] Results saved -> {csv_path}")
    print("You can visualize results with:")
    print("python visualization/scalability/plot_distributed_convergence.py --data results/csv/distributed_training.csv")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distributed training simulation for MEOCI (ADP-D3QN)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/meoci_resnet50.yaml)")
    args = parser.parse_args()

    distributed_training(args.config)
