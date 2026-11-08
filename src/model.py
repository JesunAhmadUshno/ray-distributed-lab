"""ResNet-18 construction for EuroSAT (10 classes).

Identical construction to pytorch-depth-lab's src/model.py:
the baseline's parameter count for the 10-class head is
11,181,642 (its PREREG Amendment A1).
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18

EXPECTED_PARAMS = 11_181_642


def build_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    model = resnet18(weights=None)
    # Re-initialise the head under the run seed, as the baseline
    # does, so arms differ only in orchestration.
    torch.manual_seed(seed + 1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    n = sum(p.numel() for p in model.parameters())
    assert n == EXPECTED_PARAMS, f"parameter count {n} != {EXPECTED_PARAMS}"
    return model
