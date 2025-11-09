"""
configs.__init__
==========================================================
Configuration Loader and Manager for MEOCI Framework.
----------------------------------------------------------
Features:
    - Load YAML / JSON configuration files
    - Support hierarchical config inheritance
    - Merge default and experiment-specific settings
    - Environment variable & CLI override
----------------------------------------------------------
Used in:
    - run.py
    - experiments/*
    - core/agent/*
    - core/environment/*
"""

import os
import json
import yaml
from typing import Any, Dict, Union

# ------------------------------------------------------------
# 🔹 Utility Functions
# ------------------------------------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data or {}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update nested dictionaries.
    Values in `override` take precedence over `base`.
    """
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


# ------------------------------------------------------------
# ⚙️ Config Loader Class
# ------------------------------------------------------------
class ConfigManager:
    """
    ConfigManager
    ======================================================
    Unified configuration handler for MEOCI framework.

    Example:
        >>> cfg = ConfigManager("configs/meoci_vgg16.yaml").config
        >>> print(cfg["training"]["epochs"])
    """

    def __init__(self, config_path: Union[str, None] = None):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        if config_path:
            self.load(config_path)

    # --------------------------------------------------------
    def load(self, config_path: str):
        """Load configuration file (YAML or JSON)."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"[Config] File not found: {config_path}")

        ext = os.path.splitext(config_path)[-1].lower()
        if ext in [".yaml", ".yml"]:
            cfg = load_yaml(config_path)
        elif ext == ".json":
            cfg = load_json(config_path)
        else:
            raise ValueError(f"[Config] Unsupported format: {ext}")

        self.config = self._process_inheritance(cfg)
        print(f"[Config] Loaded configuration: {config_path}")
        return self.config

    # --------------------------------------------------------
    def _process_inheritance(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle 'base' field inheritance:
        base:
          - configs/env_cluster.yaml
          - configs/train_hyperparams.yaml
        """
        if "base" not in cfg:
            return cfg

        merged_cfg = {}
        base_files = cfg.pop("base")
        if not isinstance(base_files, list):
            base_files = [base_files]

        for base in base_files:
            base_path = base if os.path.isabs(base) else os.path.join("configs", base)
            if not os.path.exists(base_path):
                raise FileNotFoundError(f"[Config] Base file not found: {base_path}")
            base_cfg = load_yaml(base_path)
            merged_cfg = deep_update(merged_cfg, base_cfg)

        merged_cfg = deep_update(merged_cfg, cfg)
        return merged_cfg

    # --------------------------------------------------------
    def merge(self, extra_cfg: Dict[str, Any]):
        """Merge an additional configuration dictionary (e.g., from CLI)."""
        self.config = deep_update(self.config, extra_cfg)

    # --------------------------------------------------------
    def save(self, out_path: str):
        """Save current configuration to YAML."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
        print(f"[Config] Saved configuration to {out_path}")

    # --------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve nested config values using dot notation."""
        parts = key.split(".")
        value = self.config
        for p in parts:
            if isinstance(value, dict) and p in value:
                value = value[p]
            else:
                return default
        return value

    # --------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        """Return configuration as a standard dictionary."""
        return self.config

    def __repr__(self):
        return f"<ConfigManager path={self.config_path} keys={list(self.config.keys())}>"



# ------------------------------------------------------------
# ✅ Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    # Example: load multi-base config
    cfg = ConfigManager("configs/meoci_vgg16.yaml").config
    print(cfg["training"]["lr"])

    # Override by CLI-like extra params
    extra = {"training": {"lr": 0.0005, "batch_size": 16}}
    cm = ConfigManager("configs/meoci_vgg16.yaml")
    cm.merge(extra)
    cm.save("configs/tmp_merged.yaml")
    print("Merged config:", cm.get("training.lr"))
