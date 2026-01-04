import numpy as np
from typing import Dict, List, Tuple


class SimpleTask(dict):
    """Allow both dict and attribute access: task.flops or task['flops']."""
    def __getattr__(self, item):
        return self[item]


class WorkloadGenerator:


    def __init__(self,
                 arrival_rate: float = 2.0,                          # tasks/s
                 data_size_range: Tuple[float, float] = (2.0, 20.0),  # MB
                 flops_range: Tuple[float, float] = (0.2, 3.0),        # GFLOPs
                 delay_constraint_range: Tuple[float, float] = (50, 150),  # ms
                 random_seed: int = 42):
        self.arrival_rate = arrival_rate
        self.data_size_range = data_size_range
        self.flops_range = flops_range
        self.delay_constraint_range = delay_constraint_range
        self.random_state = np.random.RandomState(random_seed)

        self.task_id_counter = 0
        self.generated_tasks = []


    def get_current_rate(self) -> float:
        return float(self.arrival_rate)

    def generate_tasks(self,
                       num_vehicles: int,
                       total_layers: int = 10,
                       exit_points: int = 4) -> List[SimpleTask]:

        tasks = []
        for _ in range(num_vehicles):
            t = self.generate_task()

            # -------- base --------
            t["size_mb"] = float(t["data_size_MB"])
            t["flops"] = float(t["compute_flops_GFLOPs"]) * 1e9  # GFLOPs -> FLOPs
            t["total_layers"] = int(total_layers)

            # -------- vehicle_node required --------
            t["feature_size_mb_per_layer"] = t["size_mb"] / max(t["total_layers"], 1)
            t["enable_early_exit"] = True
            t["exit_layer"] = int(self.random_state.randint(0, t["total_layers"]))

            # exit accuracy: deeper exit -> higher acc (0.70 -> 0.95)
            if t["total_layers"] > 1:
                depth_ratio = t["exit_layer"] / (t["total_layers"] - 1)
            else:
                depth_ratio = 0.0
            t["exit_accuracy"] = float(0.70 + 0.25 * depth_ratio)

            # -------- edge_server required --------
            t["arrival_rate"] = float(self.arrival_rate)
            t["exit_points"] = int(exit_points)

            tasks.append(SimpleTask(t))

        return tasks


    def generate_task(self) -> Dict:
        """Generate one raw task (without compatibility alias fields)."""
        self.task_id_counter += 1
        task_id = self.task_id_counter

        data_size = self.random_state.uniform(*self.data_size_range)      # MB
        compute_flops = self.random_state.uniform(*self.flops_range)      # GFLOPs
        delay_constraint = self.random_state.uniform(*self.delay_constraint_range)  # ms

        task = {
            "id": task_id,
            "data_size_MB": round(float(data_size), 2),
            "compute_flops_GFLOPs": round(float(compute_flops), 3),
            "delay_constraint_ms": round(float(delay_constraint), 1),
        }

        self.generated_tasks.append(task)
        return task


    def poisson_arrival(self, time_window: float = 1.0) -> int:
        expected_tasks = self.arrival_rate * time_window
        return int(self.random_state.poisson(expected_tasks))

    def generate_tasks_in_window(self, time_window: float = 1.0) -> List[Dict]:
        num_tasks = self.poisson_arrival(time_window)
        return [self.generate_task() for _ in range(num_tasks)]


    def summary(self, last_n: int = 5) -> Dict:
        if len(self.generated_tasks) == 0:
            return {"num_tasks": 0, "avg_data_MB": 0, "avg_flops": 0}

        tasks = self.generated_tasks[-last_n:]
        avg_data = np.mean([t["data_size_MB"] for t in tasks])
        avg_flops = np.mean([t["compute_flops_GFLOPs"] for t in tasks])
        avg_deadline = np.mean([t["delay_constraint_ms"] for t in tasks])

        return {
            "recent_tasks": len(tasks),
            "avg_data_MB": round(float(avg_data), 2),
            "avg_flops_GFLOPs": round(float(avg_flops), 3),
            "avg_deadline_ms": round(float(avg_deadline), 1)
        }


if __name__ == "__main__":
    gen = WorkloadGenerator(arrival_rate=3.0)
    tasks = gen.generate_tasks(num_vehicles=3, total_layers=10, exit_points=4)
    for t in tasks:
        print("Task sample:", t)
        print("  size_mb:", t.size_mb)
        print("  flops:", t.flops)
        print("  total_layers:", t.total_layers)
        print("  feature_size_mb_per_layer:", t.feature_size_mb_per_layer)
        print("  exit_layer:", t.exit_layer)
        print("  exit_accuracy:", t.exit_accuracy)
        print("  arrival_rate:", t.arrival_rate)
        print("  exit_points:", t.exit_points)
