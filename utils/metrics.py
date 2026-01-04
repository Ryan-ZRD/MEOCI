

import numpy as np
import torch
from typing import Dict, List, Tuple



def compute_latency(vehicle_time: float, edge_time: float, comm_time: float) -> float:

    total = vehicle_time + comm_time + edge_time
    return round(total, 4)


def average_latency(latencies: List[float]) -> float:
    """Compute average latency in milliseconds."""
    return float(np.mean(latencies)) if latencies else 0.0


def latency_std(latencies: List[float]) -> float:
    """Compute standard deviation of latency."""
    return float(np.std(latencies)) if latencies else 0.0



def compute_energy(vehicle_power: float, edge_power: float,
                   vehicle_time: float, edge_time: float) -> float:

    energy = vehicle_power * vehicle_time + edge_power * edge_time
    return round(energy, 4)


def average_energy(energies: List[float]) -> float:
    """Mean energy consumption."""
    return float(np.mean(energies)) if energies else 0.0



def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = predictions.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    return round(acc * 100, 2)


def early_exit_rate(exit_indices: List[int], total_exits: int) -> Dict[str, float]:

    if not exit_indices:
        return {f"exit{i+1}": 0.0 for i in range(total_exits)}

    counts = np.bincount(exit_indices, minlength=total_exits)
    probs = counts / np.sum(counts)
    return {f"exit{i+1}": round(float(probs[i]), 4) for i in range(total_exits)}



def completion_rate(successes: int, total_tasks: int) -> float:
    """
    Task completion rate under delay constraints.
    """
    if total_tasks == 0:
        return 0.0
    return round(100.0 * successes / total_tasks, 2)


def qos_penalty(latency: float, delay_constraint: float) -> float:

    return max(0.0, (latency - delay_constraint) / delay_constraint)



def compute_reward(latency: float, energy: float,
                   accuracy: float, weights: Dict[str, float]) -> float:

    w1, w2, w3 = weights.get("latency", 0.5), weights.get("energy", 0.3), weights.get("accuracy", 0.2)
    reward = - (w1 * latency + w2 * energy - w3 * accuracy)
    return round(float(reward), 6)



def summarize_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["latency", "energy", "accuracy", "reward"]
    summary = {k: 0.0 for k in keys}

    for rec in records:
        for k in keys:
            if k in rec:
                summary[k] += rec[k]

    n = len(records)
    for k in summary:
        summary[k] = round(summary[k] / n, 4) if n > 0 else 0.0

    return summary



def pareto_frontier(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:

    points = sorted(points, key=lambda x: x[0])
    frontier = []
    best_energy = float("inf")

    for latency, energy in points:
        if energy < best_energy:
            frontier.append((latency, energy))
            best_energy = energy
    return frontier


if __name__ == "__main__":
    latencies = [30.5, 28.7, 32.1, 31.2]
    energies = [4.2, 3.9, 4.1, 4.0]
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8]])
    labels = torch.tensor([1, 0, 0, 1])

    acc = compute_accuracy(preds, labels)
    avg_lat = average_latency(latencies)
    avg_energy = average_energy(energies)
    reward = compute_reward(avg_lat, avg_energy, acc, {"latency": 0.5, "energy": 0.3, "accuracy": 0.2})

    print("Accuracy:", acc)
    print("Average Latency:", avg_lat)
    print("Average Energy:", avg_energy)
    print("Reward:", reward)
