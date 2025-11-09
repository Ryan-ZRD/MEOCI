import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_multi_exit import MultiExitBase


class ConvBNAct(nn.Module):
    """Standard Conv-BN-Activation block (used in YOLOv10 backbone)."""
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Basic bottleneck used in YOLOv10 backbone."""
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        hidden = out_channels // 2
        self.conv1 = ConvBNAct(in_channels, hidden, 1, 1, 0)
        self.conv2 = ConvBNAct(hidden, out_channels, 3, 1, 1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.use_add else y


class DetectionHead(nn.Module):
    """Lightweight detection head for early exits."""
    def __init__(self, in_channels, num_classes=10, num_anchors=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_anchors * (num_classes + 5), 1)
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):
        # Output shape: (B, num_anchors*(num_classes+5), H, W)
        return self.conv(x)


class MultiExitYOLOv10n(MultiExitBase):
    """
    MultiExitYOLOv10n (3 exits)
    ==========================================================
    Lightweight multi-exit variant of YOLOv10n
    designed for edge–vehicle collaborative inference.

    Exits:
      - Exit 1: After C3 feature map (shallow)
      - Exit 2: After C4 feature map (mid)
      - Exit 3: After C5 feature map (deep, full detection head)

    Reference:
        Section 4.3, Fig. 6(d), Table II
    """

    def __init__(self, num_classes: int = 10, num_anchors: int = 3):
        super(MultiExitYOLOv10n, self).__init__(
            num_exits=3,
            exit_layers=[1, 2, 3],
            num_classes=num_classes
        )
        self.num_anchors = num_anchors
        self._build_model()

    # ------------------------------------------------------------
    # Build simplified YOLOv10n backbone with 3 exit heads
    # ------------------------------------------------------------
    def _build_model(self):
        self.backbone = nn.ModuleList([
            nn.Sequential(  # Stem
                ConvBNAct(3, 32, 3, 2, 1),
                ConvBNAct(32, 64, 3, 2, 1),
                Bottleneck(64, 64)
            ),  # C1
            nn.Sequential(  # C3 feature
                ConvBNAct(64, 128, 3, 2, 1),
                Bottleneck(128, 128),
                Bottleneck(128, 128)
            ),  # C3
            nn.Sequential(  # C4 feature
                ConvBNAct(128, 256, 3, 2, 1),
                Bottleneck(256, 256),
                Bottleneck(256, 256)
            ),  # C4
            nn.Sequential(  # C5 feature
                ConvBNAct(256, 512, 3, 2, 1),
                Bottleneck(512, 512),
                Bottleneck(512, 512)
            )   # C5
        ])

        # Early-exit detection heads
        self.exits = nn.ModuleList([
            DetectionHead(128, num_classes=self.num_classes, num_anchors=self.num_anchors),  # Exit-1
            DetectionHead(256, num_classes=self.num_classes, num_anchors=self.num_anchors),  # Exit-2
            DetectionHead(512, num_classes=self.num_classes, num_anchors=self.num_anchors)   # Exit-3
        ])

    # ------------------------------------------------------------
    # Forward with early-exit control
    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor, exit_threshold: float = 0.85):
        """
        Forward pass with confidence-based early exit.
        Each exit head predicts bounding boxes + classes.
        """
        features = []
        for i, stage in enumerate(self.backbone):
            x = stage(x)
            if i > 0:  # Skip stem
                features.append(x)

        outputs = []
        exit_id = self.num_exits - 1

        for i, feat in enumerate(features):
            pred = self.exits[i](feat)
            outputs.append(pred)
            conf = self._compute_confidence(pred)
            if conf >= exit_threshold:
                exit_id = i
                return pred, exit_id

        return outputs[-1], exit_id  # Final exit

    def _compute_confidence(self, pred: torch.Tensor) -> float:
        """
        Compute mean objectness confidence for early exit decision.
        pred: [B, anchors*(num_classes+5), H, W]
        """
        b, c, h, w = pred.shape
        pred = pred.view(b, self.num_anchors, self.num_classes + 5, h, w)
        obj = torch.sigmoid(pred[:, :, 4, :, :])  # objectness score
        return obj.mean().item()

    # ------------------------------------------------------------
    # Forward all exits (for evaluation)
    # ------------------------------------------------------------
    def forward_all(self, x):
        features = []
        for i, stage in enumerate(self.backbone):
            x = stage(x)
            if i > 0:
                features.append(x)
        outputs = [self.exits[i](feat) for i, feat in enumerate(features)]
        return outputs


# ✅ Example test
if __name__ == "__main__":
    model = MultiExitYOLOv10n(num_classes=10)
    x = torch.randn(1, 3, 256, 256)
    y, exit_idx = model(x, exit_threshold=0.8)
    print(f"Exited at Exit-{exit_idx+1}, Output shape: {y.shape}")
    outs = model.forward_all(x)
    for i, o in enumerate(outs):
        print(f"Exit-{i+1} detection map:", o.shape)
