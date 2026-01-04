import os
import time
import json
import psutil
import torch
import csv
from datetime import datetime
from typing import Optional, Dict, Any


class Profiler:


    def __init__(
        self,
        log_dir: str = "./results/logs/",
        exp_name: str = "default",
        enable_gpu: bool = True,
        sampling_interval: float = 0.1,
    ):
        self.log_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.enable_gpu = enable_gpu
        self.sampling_interval = sampling_interval
        self.start_time = None
        self.records = []


    def start(self, label: str = "session"):
        self.start_time = time.time()
        self.records = []
        self.session_label = label
        print(f"[Profiler] Started session '{label}'")


    def stop(self) -> Dict[str, Any]:
        if self.start_time is None:
            raise RuntimeError("Profiler not started. Use start() before stop().")
        duration = time.time() - self.start_time
        avg_metrics = self._aggregate()
        avg_metrics["duration_s"] = duration
        print(f"[Profiler] Session '{self.session_label}' stopped after {duration:.2f}s")
        return avg_metrics


    def capture(self):

        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        gpu_util, gpu_mem = 0.0, 0.0

        if self.enable_gpu and torch.cuda.is_available():
            gpu_util = torch.cuda.utilization(0)
            gpu_mem = torch.cuda.memory_allocated(0) / 1e6

        record = {
            "timestamp": time.time(),
            "cpu(%)": cpu,
            "mem(%)": mem,
            "gpu(%)": gpu_util,
            "gpu_mem(MB)": gpu_mem,
        }
        self.records.append(record)
        time.sleep(self.sampling_interval)
        return record

    def monitor_loop(self, duration: float = 5.0):

        t_end = time.time() + duration
        while time.time() < t_end:
            self.capture()
        print(f"[Profiler] Collected {len(self.records)} samples in {duration:.1f}s")


    @staticmethod
    def estimate_energy(cpu_usage: float, gpu_usage: float, duration_s: float) -> float:

        P_cpu = 1.5 * cpu_usage / 100.0
        P_gpu = 3.0 * gpu_usage / 100.0
        energy = (P_cpu + P_gpu) * duration_s
        return energy

    def _aggregate(self) -> Dict[str, float]:
        if not self.records:
            return {"cpu_avg": 0, "gpu_avg": 0, "mem_avg": 0}
        cpu_avg = sum(r["cpu(%)"] for r in self.records) / len(self.records)
        mem_avg = sum(r["mem(%)"] for r in self.records) / len(self.records)
        gpu_avg = sum(r["gpu(%)"] for r in self.records) / len(self.records)
        gpu_mem_avg = sum(r["gpu_mem(MB)"] for r in self.records) / len(self.records)
        return {
            "cpu_avg(%)": cpu_avg,
            "mem_avg(%)": mem_avg,
            "gpu_avg(%)": gpu_avg,
            "gpu_mem_avg(MB)": gpu_mem_avg,
        }

    def save(self, fmt: str = "csv"):

        if not self.records:
            print("[Profiler] No records to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.log_dir, f"profiler_{self.session_label}_{timestamp}.{fmt}")

        if fmt == "csv":
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
                writer.writeheader()
                writer.writerows(self.records)
        elif fmt == "json":
            with open(path, "w") as f:
                json.dump(self.records, f, indent=4)
        else:
            raise ValueError("Unsupported format (use 'csv' or 'json').")

        print(f"[Profiler] Saved logs to {path}")


    def profile_function(self, fn):

        def wrapper(*args, **kwargs):
            self.start(label=fn.__name__)
            t0 = time.time()
            result = fn(*args, **kwargs)
            t1 = time.time()
            metrics = self.stop()
            latency_ms = (t1 - t0) * 1000
            energy_est = self.estimate_energy(
                metrics["cpu_avg(%)"], metrics["gpu_avg(%)"], metrics["duration_s"]
            )
            print(
                f"[Profiler] {fn.__name__}: "
                f"Latency={latency_ms:.2f} ms | "
                f"Energy≈{energy_est:.3f} J | "
                f"CPU={metrics['cpu_avg(%)']:.1f}% | GPU={metrics['gpu_avg(%)']:.1f}%"
            )
            return result
        return wrapper



if __name__ == "__main__":
    profiler = Profiler(exp_name="demo")

    @profiler.profile_function
    def dummy_inference():
        x = torch.randn(1024, 1024).cuda() if torch.cuda.is_available() else torch.randn(1024, 1024)
        for _ in range(50):
            y = torch.matmul(x, x)
        return y

    dummy_inference()
    profiler.save(fmt="csv")
