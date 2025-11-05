"""
model.py
────────
Model factory: builds a small CNN backbone (< 50 layers).
Default: ResNet-18 from torchvision with a replaced classification head.
"""

import torch.nn as nn
from torchvision import models

def build_model(num_classes: int = 10, pretrained: bool = False, freeze_backbone: bool = False):
    """
    Build a ResNet-18 classifier.
    - pretrained=False to avoid network downloads by default.
    - If freeze_backbone=True, we freeze all feature extractor params.
    """
    # torchvision >=0.13 uses Weights enums; keep it robust:
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
        except AttributeError:
            weights = "IMAGENET1K_V1"  # older torch compat fallback

    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    # Replace classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
