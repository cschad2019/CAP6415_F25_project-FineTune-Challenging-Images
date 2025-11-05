"""
model.py — Small ResNet-18-style backbone for CIFAR-10 with freeze_backbone support.

We adapt torchvision's resnet18 to CIFAR-10 by:
- Replacing the first conv (7x7, stride 2) with 3x3, stride 1
- Removing the maxpool
- Replacing the final fc to output 10 classes
"""
import torch.nn as nn
from torchvision.models import resnet18

def build_model(num_classes: int = 10, pretrained: bool = False, freeze_backbone: bool = False) -> nn.Module:
    m = resnet18(weights=None if not pretrained else "IMAGENET1K_V1")
    # CIFAR-10 friendly stem
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    # Classifier
    m.fc = nn.Linear(m.fc.in_features, num_classes)

    if freeze_backbone:
        for name, p in m.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    return m
