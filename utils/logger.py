import os
import csv
import json
import logging
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:

    def __init__(
        self,
        log_dir: str = "./results/logs",
        exp_name: str = "default",
        enable_tensorboard: bool = True,
        overwrite: bool = False,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = exp_name
        self.root = os.path.join(log_dir, f"{exp_name}_{timestamp}")

        if overwrite and os.path.exists(self.root):
            shutil.rmtree(self.root)

        os.makedirs(self.root, exist_ok=True)

        self.txt_log = os.path.join(self.root, "train_log.txt")
        self.csv_log = os.path.join(self.root, "metrics_log.csv")
        self.json_log = os.path.join(self.root, "episode_records.json")

        self.logger = self._init_logger()
        self.writer = SummaryWriter(log_dir=self.root) if enable_tensorboard else None

        self.metrics_buffer = []
        self.logger.info(f"Logger initialized at {self.root}")

    def _init_logger(self):
        logger = logging.getLogger(self.exp_name)
        logger.setLevel(logging.INFO)

        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s]  %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(self.txt_log, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def info(self, msg: str):
        self.logger.info(msg)

    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)
        self.logger.info(f"[Scalar] {tag}={value:.4f} @ step={step}")

    def log_metrics(self, step: int, metrics: Dict[str, Any], prefix: str = ""):
        log_str = " | ".join([
            f"{k}={v:.4f}" if isinstance(v, (float, np.floating)) else f"{k}={v}"
            for k, v in metrics.items()
        ])
        self.logger.info(f"[{prefix}] Step {step:05d} | {log_str}")

        record = {"step": step}
        record.update(metrics)
        self.metrics_buffer.append(record)

        self._append_csv(record)

        if self.writer:
            for k, v in metrics.items():
                if isinstance(v, (float, int)):
                    self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def _append_csv(self, record: Dict[str, Any]):
        file_exists = os.path.exists(self.csv_log)
        with open(self.csv_log, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    def save_csv(self, path: str, headers: List[str], rows: List[Tuple]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in rows:
                writer.writerow(list(r))
        self.logger.info(f"[Logger] Saved CSV -> {path}")

    def close(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()
        self.logger.info("[Logger] Closed successfully.")
        self._save_metrics_buffer()

    def _save_metrics_buffer(self):
        if len(self.metrics_buffer) > 0:
            out_path = os.path.join(self.root, "metrics_buffer.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self.metrics_buffer, f, indent=2, ensure_ascii=False)
            self.logger.info(f"[Logger] Saved {len(self.metrics_buffer)} metrics to {out_path}")
