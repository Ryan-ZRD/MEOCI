import os
import torch
import random
import numpy as np
from datetime import datetime



# Random seed for reproducibility
GLOBAL_SEED = 42

# Default device (auto-detect GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base project directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directory paths
PATHS = {
    "results": os.path.join(PROJECT_ROOT, "results"),
    "logs": os.path.join(PROJECT_ROOT, "results", "logs"),
    "csv": os.path.join(PROJECT_ROOT, "results", "csv"),
    "plots": os.path.join(PROJECT_ROOT, "results", "plots"),
    "models": os.path.join(PROJECT_ROOT, "saved_models"),
    "datasets": os.path.join(PROJECT_ROOT, "datasets"),
    "configs": os.path.join(PROJECT_ROOT, "configs"),
}


for path in PATHS.values():
    os.makedirs(path, exist_ok=True)


EXPERIMENT = {
    "exp_name": "MEOCI_Default_Run",
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "save_interval": 20,
    "log_interval": 5,
    "use_tensorboard": True,
    "wandb_logging": False,
}


HYPERPARAMS = {
    "batch_size": 64,
    "gamma": 0.98,
    "learning_rate": 1e-4,
    "target_update_freq": 50,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 5000,
    "buffer_size": 20000,
    "warmup_steps": 500,
    "max_episodes": 1000,
    "max_steps_per_episode": 200,
    "reward_scale": 1.0,
}


ENVIRONMENT = {
    "num_vehicles": 10,
    "num_edge_servers": 2,
    "bandwidth_range": (5, 25),  # Mbps
    "vehicle_speed_range": (10, 25),  # m/s
    "packet_loss_prob": 0.01,
    "latency_constraint": 25,  # ms
    "energy_constraint": 500,  # mJ
    "accuracy_threshold": 0.8,
    "dynamic_topology": True,
}


LOGGING = {
    "log_to_file": True,
    "log_file": os.path.join(PATHS["logs"], "runtime.log"),
    "log_level": "INFO",
}


def set_random_seed(seed: int = GLOBAL_SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Config] Random seed set to {seed}")


def get_experiment_name():

    return f"{EXPERIMENT['exp_name']}_{EXPERIMENT['timestamp']}"


def get_save_path(filename: str = None):

    exp_dir = os.path.join(PATHS["results"], get_experiment_name())
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir if filename is None else os.path.join(exp_dir, filename)


def print_config_summary():

    print("=" * 60)
    print(" MEOCI Framework - Configuration Summary ")
    print("=" * 60)
    print(f" Device:           {DEVICE}")
    print(f" Global Seed:      {GLOBAL_SEED}")
    print(f" Bandwidth Range:  {ENVIRONMENT['bandwidth_range']} Mbps")
    print(f" Vehicle Speed:    {ENVIRONMENT['vehicle_speed_range']} m/s")
    print(f" Latency Limit:    {ENVIRONMENT['latency_constraint']} ms")
    print(f" Energy Limit:     {ENVIRONMENT['energy_constraint']} mJ")
    print(f" Accuracy Thres.:  {ENVIRONMENT['accuracy_threshold']}")
    print("-" * 60)
    print(f" Learning Rate:    {HYPERPARAMS['learning_rate']}")
    print(f" Batch Size:       {HYPERPARAMS['batch_size']}")
    print(f" Gamma (Discount): {HYPERPARAMS['gamma']}")
    print(f" Replay Buffer:    {HYPERPARAMS['buffer_size']}")
    print("=" * 60)



set_random_seed(GLOBAL_SEED)

if __name__ == "__main__":
    print_config_summary()
