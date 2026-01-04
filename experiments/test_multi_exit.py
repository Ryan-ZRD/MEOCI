import os
import csv
import torch
import numpy as np
from tqdm import tqdm

from configs import ConfigManager
from utils.logger import ExperimentLogger
from utils.seed_utils import set_global_seed
from core.model_zoo.alexnet_me import MultiExitAlexNet
from core.model_zoo.vgg16_me import MultiExitVGG16
from core.model_zoo.resnet50_me import MultiExitResNet50
from datasets.bdd100k_loader import BDD100KDataset


@torch.no_grad()
def evaluate_multi_exit(model, dataloader, device):

    model.eval()
    num_exits = len(model.exits)
    exit_counts = np.zeros(num_exits)
    exit_correct = np.zeros(num_exits)
    total_samples = 0

    for inputs, targets in tqdm(dataloader, desc="Evaluating multi-exit"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, exit_idx = model.forward_with_exit(inputs)

        preds = outputs.argmax(dim=1)
        correct = preds.eq(targets).sum().item()

        exit_counts[exit_idx] += len(targets)
        exit_correct[exit_idx] += correct
        total_samples += len(targets)

    # Compute metrics
    exit_probs = (exit_counts / total_samples) * 100
    exit_accs = np.zeros(num_exits)
    for i in range(num_exits):
        if exit_counts[i] > 0:
            exit_accs[i] = (exit_correct[i] / exit_counts[i]) * 100

    return exit_probs, exit_accs


def test_multi_exit(cfg_path: str):
    """Run early-exit testing experiment."""
    cfg = ConfigManager(cfg_path).config
    set_global_seed(cfg["training"]["seed"])
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    logger = ExperimentLogger(log_dir="./results/logs/multi_exit")

    model_name = cfg["model"]["name"]
    if model_name.startswith("alexnet"):
        model = MultiExitAlexNet(num_classes=cfg["model"]["num_classes"])
    elif model_name.startswith("vgg16"):
        model = MultiExitVGG16(num_classes=cfg["model"]["num_classes"])
    elif model_name.startswith("resnet50"):
        model = MultiExitResNet50(num_classes=cfg["model"]["num_classes"])
    else:
        raise ValueError(f"Unsupported multi-exit model: {model_name}")

    model.to(device)

    # Load pretrained weights if available
    ckpt_path = os.path.join(cfg["logging"]["save_model_dir"], f"best_{cfg['experiment']['name']}.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[Loaded pretrained model from {ckpt_path}]")
    else:
        print(f"[Warning] Model weights not found, evaluating untrained model.")

    # Data loader (classification subset from BDD100K)
    dataloader = BDD100KDataset(
        dataset_root="./dataset/",
        batch_size=cfg["training"]["batch_size"],
        split="val",
        num_workers=4
    ).get_loader()

    exit_probs, exit_accs = evaluate_multi_exit(model, dataloader, device)

    # Save results
    output_path = f"./results/csv/exit_{model_name}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Exit_Index", "Exit_Probability(%)", "Exit_Accuracy(%)"])
        for i, (p, a) in enumerate(zip(exit_probs, exit_accs)):
            writer.writerow([i + 1, round(p, 2), round(a, 2)])

    print("\n[Multi-Exit Evaluation Completed]")
    print(f"Exit probabilities: {np.round(exit_probs, 2)}")
    print(f"Exit accuracies: {np.round(exit_accs, 2)}")
    print(f"Results saved -> {output_path}")

    logger.log_dict({"exit_prob": exit_probs.tolist(), "exit_acc": exit_accs.tolist()})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test multi-exit models for early-exit probability and accuracy")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., meoci_vgg16.yaml)")
    args = parser.parse_args()

    test_multi_exit(args.config)
