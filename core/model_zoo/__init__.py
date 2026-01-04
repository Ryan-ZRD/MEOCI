
from .alexnet_me import MultiExitAlexNet
from .vgg16_me import MultiExitVGG16
from .resnet50_me import MultiExitResNet50
from .yolov10_me import MultiExitYOLOv10n
from .base_multi_exit import MultiExitBase


# ------------------------------------------------------------
#  Registry for supported models
# ------------------------------------------------------------
MODEL_REGISTRY = {
    "alexnet_me": MultiExitAlexNet,
    "vgg16_me": MultiExitVGG16,
    "resnet50_me": MultiExitResNet50,
    "yolov10_me": MultiExitYOLOv10n,
}


# ------------------------------------------------------------
#  Factory method
# ------------------------------------------------------------
def get_model(model_name: str, num_classes: int = 10, **kwargs):
    """
    Dynamically instantiate a multi-exit model by name.

    Args:
        model_name (str): Name of the model (e.g., 'vgg16_me')
        num_classes (int): Number of output classes
        **kwargs: Additional model-specific parameters

    Returns:
        torch.nn.Module: Instantiated model object

    Example:
        >>> from core.model_zoo import get_model
        >>> model = get_model("resnet50_me", num_classes=10)
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"[ModelZoo] Unknown model name '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](num_classes=num_classes, **kwargs)


# ------------------------------------------------------------
#  List all available models
# ------------------------------------------------------------
def list_models(verbose: bool = False):
    """
    Returns a list of all available models.

    Args:
        verbose (bool): If True, prints descriptions.

    Returns:
        list[str]: Supported model names
    """
    if verbose:
        print("✅ Available Multi-Exit Models in MEOCI:")
        for name, cls in MODEL_REGISTRY.items():
            doc = (cls.__doc__ or "").splitlines()[0]
            print(f" - {name:15s}: {doc}")
    return list(MODEL_REGISTRY.keys())


__all__ = [
    "MultiExitBase",
    "MultiExitAlexNet",
    "MultiExitVGG16",
    "MultiExitResNet50",
    "MultiExitYOLOv10n",
    "get_model",
    "list_models",
]
