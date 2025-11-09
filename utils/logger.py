"""
utils.logger
==========================================================
Unified logging and visualization interface for MEOCI.
----------------------------------------------------------
Provides:
    - Console + File logging
    - CSV metric tracking
    - TensorBoard integration
    - Episode and step-level logging for ADP-D3QN
Used in:
    - core/agent/*
    - experiments/*
    - visualization/*
"""

import os
import csv
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """
    ExperimentLogger
    ======================================================
    A unified experiment logger for MEOCI framework.
    Supports:
        ✅ Console + file logging
        ✅ CSV metrics
        ✅ TensorBoard visualization
        ✅ JSON checkpoints
    """

    def __init__(
        self,
        log_dir: str = "./results/logs",
        exp_name: str = "default",
        enable_tensorboard: bool = True,
        overwrite: bool = False,
    ):
        """
        Args:
            log_dir (str): base log directory
            exp_name (str): experiment name (auto time-tagged)
            enable_tensorboard (bool): enable TB logs
            overwrite (bool): overwrite old logs if True
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = exp_name
        self.root = os.path.join(log_dir, f"{exp_name}_{timestamp}")

        if overwrite and os.path.exists(self.root):
            os.system(f"rm -rf {self.root}")
        os.makedirs(self.root, exist_ok=True)

        # File paths
        self.txt_log = os.path.join(self.root, "train_log.txt")
        self.csv_log = os.path.join(self.root, "metrics_log.csv")
        self.json_log = os.path.join(self.root, "episode_records.json")

        # Console + file logger
        self.logger = self._init_logger()

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.root) if enable_tensorboard else None

        # Metrics buffer
        self.metrics_buffer = []

        self.logger.info(f"🚀 Logger initialized at {self.root}")

    # ------------------------------------------------------------
    # ⚙️  Logger configuration
    # ------------------------------------------------------------
    def _init_logger(self):
        logger = logging.getLogger(self.exp_name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s]  %(message)s", "%Y-%m-%d %H:%M:%S")

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(self.txt_log)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    # ------------------------------------------------------------
    # 🧩 Log metric entry
    # ------------------------------------------------------------
    def log_metrics(self, step: int, metrics: Dict[str, Any], prefix: str = ""):
        """
        Log a dictionary of metrics to file, CSV, and TensorBoard.
        """
        log_str = " | ".join([f"{k}={v:.4f}" if isinstance(v, (float, np.floating)) else f"{k}={v}"
                              for k, v in metrics.items()])
        self.logger.info(f"[{prefix}] Step {step:05d} | {log_str}")

        # Append to buffer
        record = {"step": step}
        record.update(metrics)
        self.metrics_buffer.append(record)

        # CSV write
        self._append_csv(record)

        # TensorBoard logging
        if self.writer:
            for k, v in metrics.items():
                if isinstance(v, (float, int)):
                    self.writer.add_scalar(f"{prefix}/{k}", v, step)

    # ------------------------------------------------------------
    # 💾 Append to CSV
    # ------------------------------------------------------------
    def _append_csv(self, record: Dict[str, Any]):
        file_exists = os.path.exists(self.csv_log)
        with open(self.csv_log, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    # ------------------------------------------------------------
    # 🧠 Log episode summary
    # ------------------------------------------------------------
    def log_episode_summary(self, episode: int, summary: Dict[str, Any]):
        """
        Log per-episode summary (avg latency, energy, reward, accuracy, etc.)
        """
        self.logger.info(f"[Episode {episode:03d}] Summary: {json.dumps(summary, indent=2)}")
        with open(self.json_log, "a") as f:
            json.dump({"episode": episode, **summary}, f)
            f.write("\n")

    # ------------------------------------------------------------
    # ⏱️ Training progress bar
    # ------------------------------------------------------------
    def log_progress(self, episode: int, total: int, reward: float, latency: float, acc: float):
        """
        Print compact training progress line (for RL agent)
        """
        pct = (episode / total) * 100
        msg = f"[{episode:03d}/{total}] {pct:6.2f}% | Reward={reward:.3f} | Lat={latency:.2f}ms | Acc={acc:.2f}%"
        self.logger.info(msg)

    # ------------------------------------------------------------
    # 🧹 Close & flush
    # ------------------------------------------------------------
    def close(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()
        self.logger.info("[Logger] Closed successfully.")
        self._save_metrics_buffer()

    # ------------------------------------------------------------
    # 📦 Save buffered metrics as JSON
    # ------------------------------------------------------------
    def _save_metrics_buffer(self):
        if len(self.metrics_buffer) > 0:
            out_path = os.path.join(self.root, "metrics_buffer.json")
            with open(out_path, "w") as f:
                json.dump(self.metrics_buffer, f, indent=2)
            self.logger.info(f"[Logger] Saved {len(self.metrics_buffer)} metrics to {out_path}")

    # ------------------------------------------------------------
    # 📊 Simple plotting interface
    # ------------------------------------------------------------
    def export_plot_data(self, out_csv: Optional[str] = None):
        """
        Export all metrics to a CSV for visualization (used in Fig.7–10 plots).
        """
        if not self.metrics_buffer:
            self.logger.warning("[Logger] No metrics to export.")
            return

        if out_csv is None:
            out_csv = os.path.join(self.root, "metrics_export.csv")

        keys = list(self.metrics_buffer[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.metrics_buffer)

        self.logger.info(f"[Logger] Exported metrics to {out_csv}")


# ------------------------------------------------------------
# ✅ Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    logger = ExperimentLogger(exp_name="adp_d3qn_train", enable_tensorboard=False)
    for step in range(1, 6):
        logger.log_metrics(step, {
            "reward": np.random.uniform(-1, 1),
            "latency": np.random.uniform(20, 35),
            "accuracy": np.random.uniform(80, 95)
        }, prefix="train")

    summary = {"latency": 28.7, "energy": 4.1, "accuracy": 90.2, "reward": 0.57}
    logger.log_episode_summary(episode=1, summary=summary)
    logger.close()
