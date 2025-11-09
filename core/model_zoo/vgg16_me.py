import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_multi_exit import MultiExitBase


class MultiExitVGG16(MultiExitBase):
    """
    MultiExitVGG16
    ==========================================================
    Multi-exit variant of VGG16 with 5 early-exit branches
    for adaptive collaborative inference in MEOCI.

    Exits are placed after:
      - block1 (conv2)
      - block2 (conv4)
      - block3 (conv7)
      - block4 (conv10)
      - block5 (conv13, final exit)

    References:
        Section 4.2, Fig. 6(b), Table II
        "Multi-Exit Neural Network Design"
    """

    def __init__(self, num_classes: int = 10):
        super(MultiExitVGG16, self).__init__(
            num_exits=5,
            exit_layers=[1, 3, 6, 9, 12],
            num_classes=num_classes
        )

    # ------------------------------------------------------------
    # Build the multi-exit VGG16 architecture
    # ------------------------------------------------------------
    def _build_model(self):
        cfg = [64, 64, 'M',
               128, 128, 'M',
               256, 256, 256, 'M',
               512, 512, 512, 'M',
               512, 512, 512, 'M']

        self.feature_layers = nn.ModuleList()
        in_channels = 3
        conv_idx = 0

        for v in cfg:
            if v == 'M':
                self.feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                self.feature_layers.append(nn.Sequential(conv, nn.ReLU(inplace=True)))
                in_channels = v
                conv_idx += 1
                if conv_idx in [2, 4, 7, 10, 13]:
                    self._add_exit(in_channels=v)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes),
        )

    # ------------------------------------------------------------
    # Forward with dynamic early exit
    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor, exit_threshold: float = 0.9):
        """
        Adaptive forward with early exits.
        """
        outputs = []
        exit_id = self.num_exits - 1

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

        # Final exit (block5 output)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        final_logits = self.fc(x)
        outputs.append(final_logits)
        return final_logits, exit_id

    # ------------------------------------------------------------
    # Exit computation for conv exits
    # ------------------------------------------------------------
    def _compute_exit_output(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """
        Handles convolutional exits (shared design for VGG).
        """
        return self.exits[idx](x)


# ✅ Example quick test
if __name__ == "__main__":
    model = MultiExitVGG16(num_classes=10)
    model.summary()

    x = torch.randn(4, 3, 224, 224)
    out, exit_idx = model(x, exit_threshold=0.85)
    print(f"Exited at exit-{exit_idx+1}, output shape = {out.shape}")

    # Evaluate all exits
    outs = model.forward_all(x)
    for i, o in enumerate(outs):
        print(f"Exit-{i+1} logits:", o.shape)
