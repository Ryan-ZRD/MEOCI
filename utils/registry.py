"""
utils.registry
==========================================================
Registry Manager for MEOCI Framework
----------------------------------------------------------
Provides a unified way to:
    - Register algorithms, models, environments, optimizers
    - Dynamically load classes by name
    - Manage configuration-to-class mapping
----------------------------------------------------------
Used in:
    - core/agent/*
    - core/model_zoo/*
    - core/environment/*
    - core/optimization/*
"""

from typing import Dict, Type, Any, Optional, Callable
import importlib


# ==========================================================
# 🎯 Registry Class
# ==========================================================
class Registry:
    """
    A simple and extensible registry class for modular design.

    Example:
        >>> MODEL_REGISTRY = Registry("Model")
        >>> @MODEL_REGISTRY.register()
        >>> class MyModel:
        >>>     pass
        >>> model_cls = MODEL_REGISTRY.get("MyModel")
    """

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type[Any]] = {}

    def __len__(self):
        return len(self._registry)

    def __repr__(self):
        return f"<Registry({self._name}): {list(self._registry.keys())}>"

    # ------------------------------------------------------
    # 🧩 Register Function / Class
    # ------------------------------------------------------
    def register(self, name: Optional[str] = None):
        """
        Decorator for registering classes or functions.

        Example:
            @REGISTRY.register("MyAlgo")
            class MyAlgo: ...
        """
        def _register(cls_or_fn):
            key = name or cls_or_fn.__name__
            if key in self._registry:
                raise KeyError(f"[Registry:{self._name}] Duplicate key '{key}'")
            self._registry[key] = cls_or_fn
            print(f"[Registry:{self._name}] Registered: {key}")
            return cls_or_fn
        return _register

    # ------------------------------------------------------
    # 🔍 Get Registered Object
    # ------------------------------------------------------
    def get(self, name: str) -> Optional[Type[Any]]:
        """
        Retrieve a registered class or function by name.
        """
        if name not in self._registry:
            raise KeyError(
                f"[Registry:{self._name}] '{name}' not found. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    # ------------------------------------------------------
    # 🧱 Build Instance Dynamically
    # ------------------------------------------------------
    def build(self, name: str, **kwargs) -> Any:
        """
        Instantiate a registered class or function dynamically.
        """
        cls_or_fn = self.get(name)
        return cls_or_fn(**kwargs)

    # ------------------------------------------------------
    # 📜 List All Registered Keys
    # ------------------------------------------------------
    def list_all(self, verbose: bool = False):
        """
        List all registered entries.
        """
        if verbose:
            print(f"\n[Registry:{self._name}] Registered items:")
            for k in self._registry:
                print(f"  - {k}")
        return list(self._registry.keys())


# ==========================================================
# 🔧 Predefined Registries
# ==========================================================
MODEL_REGISTRY = Registry("Model")
AGENT_REGISTRY = Registry("Agent")
ENV_REGISTRY = Registry("Environment")
OPTIMIZER_REGISTRY = Registry("Optimizer")
REWARD_REGISTRY = Registry("RewardFunction")


# ==========================================================
# ⚙️ Helper: Dynamic Import Utility
# ==========================================================
def import_from_path(path: str, name: str) -> Any:
    """
    Dynamically import a class or function from a given module path.

    Args:
        path (str): module import path (e.g. 'core.agent.agent_adp_d3qn')
        name (str): class or function name

    Example:
        >>> cls = import_from_path('core.model_zoo.vgg16_me', 'MultiExitVGG16')
    """
    module = importlib.import_module(path)
    return getattr(module, name)


# ==========================================================
# ✅ Example Usage
# ==========================================================
if __name__ == "__main__":
    # Register a dummy agent
    @AGENT_REGISTRY.register("DummyAgent")
    class DummyAgent:
        def __init__(self, gamma=0.99):
            self.gamma = gamma

        def act(self):
            return "action"

    print(AGENT_REGISTRY)
    agent = AGENT_REGISTRY.build("DummyAgent", gamma=0.95)
    print(f"Agent gamma={agent.gamma}, action={agent.act()}")

    # Register and build model
    @MODEL_REGISTRY.register("FakeNet")
    class FakeNet:
        def __init__(self, layers=5): self.layers = layers
    model = MODEL_REGISTRY.build("FakeNet", layers=8)
    print(f"Model layers={model.layers}")

    # Test dynamic import
    vgg_cls = import_from_path("core.model_zoo.vgg16_me", "MultiExitVGG16")
    print(f"Imported class: {vgg_cls.__name__}")
