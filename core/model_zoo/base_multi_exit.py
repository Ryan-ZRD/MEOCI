import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


class MultiExitBase(nn.Module):


    def __init__(self, num_exits: int, exit_layers: List[int], num_classes: int):

        super(MultiExitBase, self).__init__()

        assert num_exits == len(exit_layers), \
            "num_exits must match number of exit_layers."

        self.num_exits = num_exits
        self.exit_layers = exit_layers
        self.num_classes = num_classes
        self.exits = nn.ModuleList()  # exit classifiers
        self.feature_layers = nn.ModuleList()  # main trunk
        self._build_model()


    def _build_model(self):

        raise NotImplementedError("Subclasses must implement _build_model().")


    def _add_exit(self, in_channels: int, feature_dim: int = 256):

        exit_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, self.num_classes)
        )
        self.exits.append(exit_branch)


    def forward(self, x: torch.Tensor, exit_threshold: float = 0.9) -> Tuple[torch.Tensor, int]:

        outputs = []
        exit_id = self.num_exits - 1  # default: last exit

        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            if i in self.exit_layers:
                idx = self.exit_layers.index(i)
                logits = self.exits[idx](x)
                outputs.append(logits)

                probs = F.softmax(logits, dim=1)
                conf, _ = torch.max(probs, dim=1)
                mean_conf = conf.mean().item()

                if mean_conf >= exit_threshold:
                    exit_id = idx
                    return logits, exit_id  # early exit

        # fallback to final exit
        final_logits = self.exits[-1](x)
        outputs.append(final_logits)
        return final_logits, exit_id


    def forward_all(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute logits from all exit points (used for multi-exit training).
        """
        outputs = []
        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            if i in self.exit_layers:
                idx = self.exit_layers.index(i)
                outputs.append(self.exits[idx](x))
        return outputs


    def compute_losses(self,
                       outputs: List[torch.Tensor],
                       targets: torch.Tensor,
                       loss_fn=nn.CrossEntropyLoss(),
                       loss_weights: List[float] = None) -> torch.Tensor:

        if loss_weights is None:
            loss_weights = [1.0 / len(outputs)] * len(outputs)

        losses = [loss_fn(out, targets) * w for out, w in zip(outputs, loss_weights)]
        total_loss = torch.stack(losses).sum()
        return total_loss


    def evaluate_exits(self, x: torch.Tensor, y: torch.Tensor) -> Dict[int, float]:
        """
        Evaluate per-exit accuracy (for analysis/plotting).
        """
        self.eval()
        results = {}
        with torch.no_grad():
            outputs = self.forward_all(x)
            for idx, logits in enumerate(outputs):
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y).float().mean().item()
                results[idx + 1] = round(acc, 4)
        return results


    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> None:
        """Print model summary."""
        print(f"Model: {self.__class__.__name__}")
        print(f"Total Parameters: {self.count_parameters():,}")
        print(f"Exit Layers: {self.exit_layers}")
        print(f"Number of Exits: {self.num_exits}")
        print(f"Output Classes: {self.num_classes}")



if __name__ == "__main__":
    class DummyNet(MultiExitBase):
        def _build_model(self):
            # Example: 6 conv layers, 3 exits
            conv_channels = [3, 16, 32, 64, 128, 256]
            for i in range(len(conv_channels) - 1):
                self.feature_layers.append(
                    nn.Sequential(
                        nn.Conv2d(conv_channels[i], conv_channels[i + 1], kernel_size=3, padding=1),
                        nn.BatchNorm2d(conv_channels[i + 1]),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2)
                    )
                )
                if i in self.exit_layers:
                    self._add_exit(in_channels=conv_channels[i + 1])

    model = DummyNet(num_exits=3, exit_layers=[1, 3, 5], num_classes=10)
    model.summary()

    x = torch.randn(8, 3, 64, 64)
    out, exit_id = model(x, exit_threshold=0.85)
    print(f"Exited at branch {exit_id}, output shape:", out.shape)
