import numpy as np
from typing import Dict, List, Tuple


class WorkloadGenerator:
    """
    Workload Generation Model for Vehicular Edge Inference
    ----------------------------------------------------------
    Simulates the task arrival process and task properties
    in the vehicular edge computing environment.

    Implements:
      - Poisson task arrival (λ)
      - Task size sampling (input data)
      - Computation complexity (FLOPs)
      - Delay constraint generation
      - Task feature vector output for DRL state input
    """

    def __init__(self,
                 arrival_rate: float = 2.0,             # λ (tasks/s)
                 data_size_range: Tuple[float, float] = (2.0, 20.0),  # MB
                 flops_range: Tuple[float, float] = (0.2, 3.0),       # GFLOPs
                 delay_constraint_range: Tuple[float, float] = (50, 150),  # ms
                 random_seed: int = 42):
        self.arrival_rate = arrival_rate
        self.data_size_range = data_size_range
        self.flops_range = flops_range
        self.delay_constraint_range = delay_constraint_range
        self.random_state = np.random.RandomState(random_seed)

        self.task_id_counter = 0
        self.generated_tasks = []

    # ----------------------------------------------------------
    # Task arrival simulation (Poisson process)
    # ----------------------------------------------------------
    def poisson_arrival(self, time_window: float = 1.0) -> int:
        """
        Sample number of task arrivals within a time window (in seconds).
        N ~ Poisson(λ * Δt)
        """
        expected_tasks = self.arrival_rate * time_window
        num_tasks = self.random_state.poisson(expected_tasks)
        return num_tasks

    # ----------------------------------------------------------
    # Generate a single task
    # ----------------------------------------------------------
    def generate_task(self) -> Dict:
        """
        Generate one inference task with random attributes.
        """
        self.task_id_counter += 1
        task_id = self.task_id_counter

        data_size = self.random_state.uniform(*self.data_size_range)      # MB
        compute_flops = self.random_state.uniform(*self.flops_range)      # GFLOPs
        delay_constraint = self.random_state.uniform(*self.delay_constraint_range)  # ms

        task = {
            "id": task_id,
            "data_size_MB": round(data_size, 2),
            "compute_flops_GFLOPs": round(compute_flops, 3),
            "delay_constraint_ms": round(delay_constraint, 1),
        }

        self.generated_tasks.append(task)
        return task

    # ----------------------------------------------------------
    # Generate multiple tasks within Δt
    # ----------------------------------------------------------
    def generate_tasks_in_window(self, time_window: float = 1.0) -> List[Dict]:
        """
        Generate a batch of tasks during the current time window.
        """
        num_tasks = self.poisson_arrival(time_window)
        return [self.generate_task() for _ in range(num_tasks)]

    # ----------------------------------------------------------
    # Task feature encoding for DRL state vector
    # ----------------------------------------------------------
    @staticmethod
    def encode_task_features(task: Dict) -> np.ndarray:
        """
        Convert task attributes into normalized feature vector.
        Example output: [data_size_norm, compute_norm, deadline_norm]
        """
        data_norm = task["data_size_MB"] / 20.0
        flops_norm = task["compute_flops_GFLOPs"] / 3.0
        delay_norm = task["delay_constraint_ms"] / 150.0
        return np.array([data_norm, flops_norm, delay_norm], dtype=np.float32)

    # ----------------------------------------------------------
    # Utility: summary
    # ----------------------------------------------------------
    def summary(self, last_n: int = 5) -> Dict:
        if len(self.generated_tasks) == 0:
            return {"num_tasks": 0, "avg_data_MB": 0, "avg_flops": 0}
        tasks = self.generated_tasks[-last_n:]
        avg_data = np.mean([t["data_size_MB"] for t in tasks])
        avg_flops = np.mean([t["compute_flops_GFLOPs"] for t in tasks])
        avg_deadline = np.mean([t["delay_constraint_ms"] for t in tasks])
        return {
            "recent_tasks": len(tasks),
            "avg_data_MB": round(avg_data, 2),
            "avg_flops_GFLOPs": round(avg_flops, 3),
            "avg_deadline_ms": round(avg_deadline, 1)
        }



if __name__ == "__main__":
    generator = WorkloadGenerator(arrival_rate=3.0)
    for step in range(5):
        tasks = generator.generate_tasks_in_window()
        print(f"[t={step}] Generated {len(tasks)} tasks.")
        if tasks:
            print("Sample task:", tasks[0])
    print("Summary:", generator.summary())
