import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck
from .base_multi_exit import MultiExitBase


class MultiExitResNet50(MultiExitBase):


    def __init__(self, num_classes: int = 10):
        super(MultiExitResNet50, self).__init__(
            num_exits=6,
            exit_layers=[3, 4, 5, 6, 7, 8],
            num_classes=num_classes
        )
        self.inplanes = 64
        self._build_model()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def _build_model(self):
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ),  # layer 0
            self._make_layer(Bottleneck, 64, 3),   # conv2_x
            self._make_layer(Bottleneck, 128, 4, stride=2),  # conv3_x
            self._make_layer(Bottleneck, 256, 6, stride=2),  # conv4_x
            self._make_layer(Bottleneck, 512, 3, stride=2),  # conv5_x
        ])

        # Attach exits after key depth stages (total 6 exits)
        self._add_exit(256)    # after conv2_x
        self._add_exit(512)    # after conv3_x
        self._add_exit(1024)   # mid conv4_x
        self._add_exit(1024)   # deep conv4_x
        self._add_exit(2048)   # conv5_x-1
        self._add_exit(2048)   # final exit

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, self.num_classes)


    def forward(self, x: torch.Tensor, exit_threshold: float = 0.9):
        """
        Forward pass with adaptive early exits based on confidence.
        """
        outputs = []
        exit_id = self.num_exits - 1
        layer_count = 0

        for i, block in enumerate(self.feature_layers):
            x = block(x)
            # emulate intermediate feature points for exits
            if i == 1:
                logits = self.exits[0](x)  # after conv2_x
                outputs.append(logits)
                if self._should_exit(logits, exit_threshold, 0):
                    return logits, 0
            elif i == 2:
                logits = self.exits[1](x)
                outputs.append(logits)
                if self._should_exit(logits, exit_threshold, 1):
                    return logits, 1
            elif i == 3:
                # conv4_x mid and deep exits
                mid = self.exits[2](x)
                outputs.append(mid)
                if self._should_exit(mid, exit_threshold, 2):
                    return mid, 2

                deep = self.exits[3](x)
                outputs.append(deep)
                if self._should_exit(deep, exit_threshold, 3):
                    return deep, 3
            elif i == 4:
                logits5a = self.exits[4](x)
                outputs.append(logits5a)
                if self._should_exit(logits5a, exit_threshold, 4):
                    return logits5a, 4

        # Final exit
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        final_logits = self.fc(x)
        outputs.append(final_logits)
        return final_logits, exit_id

    def _should_exit(self, logits, threshold, idx):
        """Check softmax confidence for early exit."""
        conf = F.softmax(logits, dim=1).max(dim=1)[0].mean().item()
        return conf >= threshold

    # ------------------------------------------------------------
    # Forward all exits for training/analysis
    # ------------------------------------------------------------
    def forward_all(self, x):
        outputs = []
        for i, block in enumerate(self.feature_layers):
            x = block(x)
            if i == 1:
                outputs.append(self.exits[0](x))
            elif i == 2:
                outputs.append(self.exits[1](x))
            elif i == 3:
                outputs.append(self.exits[2](x))
                outputs.append(self.exits[3](x))
            elif i == 4:
                outputs.append(self.exits[4](x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        outputs.append(self.fc(x))
        return outputs



if __name__ == "__main__":
    model = MultiExitResNet50(num_classes=10)
    model.summary()

    x = torch.randn(2, 3, 224, 224)
    out, exit_idx = model(x, exit_threshold=0.88)
    print(f"Exited at exit-{exit_idx+1}, logits shape:", out.shape)

    outs = model.forward_all(x)
    for i, o in enumerate(outs):
        print(f"Exit-{i+1} logits:", o.shape)
