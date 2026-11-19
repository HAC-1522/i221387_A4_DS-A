import torch
import torch.nn as nn
from torchvision import models

class EfficientNetLite0(nn.Module):
    """Compact EfficientNet-Lite0-style network for CIFAR-100.
    Uses MBConv blocks, depthwise convolutions and ReLU6, without squeeze-excitation.
    This is a CIFAR-adapted implementation, not a pretrained torchvision model.
    """
    def __init__(self, num_classes=100, width=1.0):
        super().__init__()
        def c(v): return max(8, int(v * width))
        self.stem = nn.Sequential(
            nn.Conv2d(3, c(32), 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c(32)), nn.ReLU6(inplace=True)
        )
        specs = [
            # out_channels, expansion, repeats, first_stride
            (16, 1, 1, 1), (24, 6, 2, 2), (40, 6, 2, 2),
            (80, 6, 3, 2), (112, 6, 3, 1), (192, 6, 4, 2), (320, 6, 1, 1)
        ]
        blocks, in_ch = [], c(32)
        for out_ch, expansion, repeats, stride in specs:
            out_ch = c(out_ch)
            for i in range(repeats):
                s = stride if i == 0 else 1
                hidden = in_ch * expansion
                layers = []
                if expansion != 1:
                    layers += [nn.Conv2d(in_ch, hidden, 1, bias=False),
                               nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
                layers += [nn.Conv2d(hidden, hidden, 3, stride=s, padding=1,
                                     groups=hidden, bias=False),
                           nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
                           nn.Conv2d(hidden, out_ch, 1, bias=False),
                           nn.BatchNorm2d(out_ch)]
                blocks.append(nn.Sequential(*layers))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, c(1280), 1, bias=False),
            nn.BatchNorm2d(c(1280)), nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(c(1280), num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x).flatten(1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))

def build_model(name, num_classes=100):
    name = name.lower().replace("-", "").replace("_", "")
    if name in ("resnet50", "teacher"):
        m = models.resnet50(weights=None, num_classes=num_classes)
        # CIFAR adaptation: preserve spatial information at 32x32.
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m
    if name in ("mobilenetv2", "mobile"):
        return models.mobilenet_v2(weights=None, num_classes=num_classes)
    if name in ("squeezenet", "squeezenet11"):
        return models.squeezenet1_1(weights=None, num_classes=num_classes)
    if name in ("efficientnetlite0", "lite0", "efficientnet"):
        return EfficientNetLite0(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")

def get_feature_vector(model, name, x):
    """Return a pooled intermediate representation for feature distillation."""
    key = name.lower().replace("-", "").replace("_", "")
    if key in ("resnet50", "teacher"):
        z = model.conv1(x); z = model.bn1(z); z = model.relu(z)
        z = model.maxpool(z); z = model.layer1(z); z = model.layer2(z)
        z = model.layer3(z); z = model.layer4(z)
        return torch.nn.functional.adaptive_avg_pool2d(z, 1).flatten(1)
    if key in ("mobilenetv2", "mobile"):
        z = model.features(x)
        return torch.nn.functional.adaptive_avg_pool2d(z, 1).flatten(1)
    if key in ("squeezenet", "squeezenet11"):
        z = model.features(x)
        return torch.nn.functional.adaptive_avg_pool2d(z, 1).flatten(1)
    if key in ("efficientnetlite0", "lite0", "efficientnet"):
        return model.forward_features(x)
    raise ValueError(f"No feature extractor for {name}")

def feature_dim(model, name):
    key = name.lower().replace("-", "").replace("_", "")
    if key in ("resnet50", "teacher"): return 2048
    if key in ("mobilenetv2", "mobile"): return 1280
    if key in ("squeezenet", "squeezenet11"): return 512
    if key in ("efficientnetlite0", "lite0", "efficientnet"): return model.classifier.in_features
    raise ValueError(name)
