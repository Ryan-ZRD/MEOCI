import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_multi_exit import MultiExitBase


class MultiExitAlexNet(MultiExitBase):
    """
    MultiExitAlexNet
    ==========================================================
    Multi-exit variant of AlexNet for collaborative inference
    in vehicular edge computing.

    This model introduces 4 exit branches positioned after:
      - conv2
      - conv4
      - conv5
      - fc7 (final exit)

    References:
      Section 4.2 - Multi-Exit Network Design
      Fig. 6(a) and Table II of MEOCI paper.
    """

    def __init__(self, num_classes: int = 10):
        super(MultiExitAlexNet, self).__init__(
            num_exits=4,
            exit_layers=[1, 3, 4, 7],
            num_classes=num_classes
        )

    # ------------------------------------------------------------
    # Build the network architecture with early exits
    # ------------------------------------------------------------
    def _build_model(self):
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2)
            ),  # conv1 (exit idx 0)
            nn.Sequential(
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2)
            ),  # conv2 (exit idx 1)
            nn.Sequential(
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),  # conv3
            nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),  # conv4 (exit idx 2)
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2)
            ),  # conv5 (exit idx 3)
            nn.Flatten(),
            nn.Sequential(
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            ),  # fc6
            nn.Sequential(
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            ),  # fc7 (final exit)
        ])

        # Attach exits to chosen layers
        self._add_exit(in_channels=192)    # after conv2
        self._add_exit(in_channels=256)    # after conv4
        self._add_exit(in_channels=256)    # after conv5
        self._add_exit(in_channels=4096)   # after fc7


    # ------------------------------------------------------------
    # Forward override (handle fc exits properly)
    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor, exit_threshold: float = 0.9):
        """
        Forward pass with adaptive early exit.
        """
        outputs = []
        exit_id = self.num_exits - 1
        flatten_after = 5  # index after conv5

        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            if i in self.exit_layers:
                idx = self.exit_layers.index(i)
                logits = self._compute_exit_output(x, idx)
                outputs.append(logits)

                conf = F.softmax(logits, dim=1).max(dim=1)[0].mean().item()
                if conf >= exit_threshold:
                    exit_id = idx
                    return logits, exit_id

        final_logits = outputs[-1]
        return final_logits, exit_id

    def _compute_exit_output(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """
        Handles exit head behavior for conv vs. fc layers.
        """
        if idx < 3:
            return self.exits[idx](x)
        else:
            # fc-layer exit
            if x.ndim > 2:
                x = torch.flatten(x, 1)
            return self.exits[idx](x)


# ✅ Example usage
if __name__ == "__main__":
    model = MultiExitAlexNet(num_classes=10)
    model.summary()

    x = torch.randn(4, 3, 224, 224)
    out, exit_idx = model(x, exit_threshold=0.85)
    print(f"Exited at exit-{exit_idx+1}, output shape = {out.shape}")

    # Check all exits
    outs = model.forward_all(x)
    for i, o in enumerate(outs):
        print(f"Exit-{i+1} output:", o.shape)
