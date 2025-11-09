"""
utils.checkpoint
==========================================================
Model and agent checkpoint utilities for MEOCI framework.
----------------------------------------------------------
Provides:
    - Save/load model checkpoints
    - Resume training or evaluation
    - Store optimizer/scheduler states
    - Versioned checkpoint management
Used in:
    - core/agent/*
    - experiments/*
    - deployment/*
"""

import os
import torch
import glob
from typing import Any, Dict, Optional


class CheckpointManager:
    """
    CheckpointManager
    ======================================================
    Manages model saving/loading and experiment recovery.
    Features:
        ✅ Save model/optimizer/scheduler
        ✅ Resume training from checkpoint
        ✅ Auto-versioning
        ✅ Compatibility with single/multi-GPU
    """

    def __init__(self, save_dir: str = "./saved_models", exp_name: str = "default"):
        self.save_dir = os.path.join(save_dir, exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.ckpt_pattern = os.path.join(self.save_dir, "checkpoint_epoch_*.pth")

    # ------------------------------------------------------------
    # 💾 Save Checkpoint
    # ------------------------------------------------------------
    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, Any]] = None,
        tag: Optional[str] = None,
        max_keep: int = 5
    ):
        """
        Save training checkpoint.
        Args:
            epoch (int): current epoch number
            model (torch.nn.Module): model instance
            optimizer (torch.optim.Optimizer, optional): optimizer state
            scheduler (torch.optim.lr_scheduler, optional): scheduler state
            metrics (dict, optional): training metrics
            tag (str, optional): custom name suffix (e.g. 'best', 'final')
            max_keep (int): number of old checkpoints to keep
        """
        ckpt_name = f"checkpoint_epoch_{epoch}.pth" if tag is None else f"checkpoint_{tag}.pth"
        ckpt_path = os.path.join(self.save_dir, ckpt_name)

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "metrics": metrics or {},
        }

        torch.save(state, ckpt_path)
        print(f"[Checkpoint] Saved: {ckpt_path}")

        # Clean up old checkpoints
        ckpts = sorted(glob.glob(self.ckpt_pattern), key=os.path.getmtime)
        if len(ckpts) > max_keep:
            for old_ckpt in ckpts[:-max_keep]:
                os.remove(old_ckpt)
                print(f"[Checkpoint] Removed old file: {old_ckpt}")

    # ------------------------------------------------------------
    # 🔁 Load Checkpoint
    # ------------------------------------------------------------
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ckpt_path: Optional[str] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load model and optimizer states from checkpoint.
        """
        if ckpt_path is None:
            ckpt_files = sorted(glob.glob(self.ckpt_pattern), key=os.path.getmtime)
            if not ckpt_files:
                raise FileNotFoundError(f"No checkpoint found in {self.save_dir}")
            ckpt_path = ckpt_files[-1]

        checkpoint = torch.load(ckpt_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state"], strict=strict)
        if optimizer and checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler and checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        print(f"[Checkpoint] Loaded from: {ckpt_path}")
        return checkpoint

    # ------------------------------------------------------------
    # 🔍 Resume Training
    # ------------------------------------------------------------
    def resume_training(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> int:
        """
        Resume from latest checkpoint and return last epoch.
        """
        ckpt_files = sorted(glob.glob(self.ckpt_pattern), key=os.path.getmtime)
        if not ckpt_files:
            print("[Checkpoint] No checkpoint found. Starting from scratch.")
            return 0

        latest_ckpt = ckpt_files[-1]
        state = torch.load(latest_ckpt, map_location="cpu")

        model.load_state_dict(state["model_state"], strict=True)
        if optimizer and state.get("optimizer_state"):
            optimizer.load_state_dict(state["optimizer_state"])
        if scheduler and state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])

        last_epoch = state.get("epoch", 0)
        print(f"[Checkpoint] Resumed from {latest_ckpt} (epoch {last_epoch})")
        return last_epoch

    # ------------------------------------------------------------
    # 📦 Load Model Only
    # ------------------------------------------------------------
    def load_model_only(self, model: torch.nn.Module, ckpt_path: str, strict: bool = True):
        """
        Load model weights only (for inference/evaluation).
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=strict)
        print(f"[Checkpoint] Loaded model weights from {ckpt_path}")

    # ------------------------------------------------------------
    # 🧹 Utility: List all checkpoints
    # ------------------------------------------------------------
    def list_checkpoints(self):
        ckpts = sorted(glob.glob(self.ckpt_pattern))
        if not ckpts:
            print("[Checkpoint] No checkpoints available.")
        else:
            print("[Checkpoint] Available checkpoints:")
            for path in ckpts:
                print("  •", os.path.basename(path))
        return ckpts


# ------------------------------------------------------------
# ✅ Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 2)

    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    ckpt_mgr = CheckpointManager(save_dir="./saved_models", exp_name="test_run")
    ckpt_mgr.save_checkpoint(epoch=1, model=model, optimizer=optimizer)
    ckpt_mgr.list_checkpoints()
    ckpt_mgr.resume_training(model, optimizer)
