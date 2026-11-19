import torch
from torch import nn

def make_activation(name):
    mapping = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation: {name}")
    return mapping[name]()

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, conv_type, activation):
        super().__init__()
        padding = kernel // 2
        if conv_type == "depthwise":
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel, padding=padding, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        elif conv_type == "standard":
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            raise ValueError(f"Unsupported convolution type: {conv_type}")
        self.activation = make_activation(activation)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        return self.pool(self.activation(self.conv(x)))

class SearchableCNN(nn.Module):
    def __init__(self, architecture, num_classes=10):
        super().__init__()
        cfg = architecture
        blocks = []
        in_ch = 3
        for i in range(cfg["depth"]):
            out_ch = int(cfg["channels"][i])
            blocks.append(ConvBlock(
                in_ch, out_ch, int(cfg["kernels"][i]),
                cfg["conv_types"][i], cfg["activations"][i]
            ))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_ch, num_classes)
        self.architecture = cfg

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
