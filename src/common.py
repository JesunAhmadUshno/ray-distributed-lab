"""Shared recipe, seeding, environment capture, and IO helpers.

Every arm reads the frozen recipe from here. The recipe mirrors
pytorch-depth-lab's PREREG (frozen recipe and CPU sub-protocol)
at the subset scale frozen in this repo's PREREGISTRATION.md.
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
RUNS = RESULTS / "runs"

# Frozen recipe (PREREGISTRATION.md). Do not edit without a
# dated amendment to PREREGISTRATION.md.
RECIPE = {
    "model": "torchvision ResNet-18, fc replaced by Linear(512, 10)",
    "optimizer": "AdamW",
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "weight_decay": 1e-4,
    "scheduler": "CosineAnnealingLR(T_max=epochs)",
    "loss": "CrossEntropyLoss (label_smoothing=0.0)",
    "effective_batch": 128,
    "epochs": 3,
    "seeds": (42, 43, 44),
    "subset_per_class": 400,
    "subset_size": 4000,
    "warmup_steps_excluded": 5,
    "threads_per_process": 2,
    "loader_workers": 0,
    "augmentation": "RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5), train only",
    "normalisation_mean": (0.485, 0.456, 0.406),
    "normalisation_std": (0.229, 0.224, 0.225),
    "split_seed": 20261107,
}

IMAGENET_MEAN = RECIPE["normalisation_mean"]
IMAGENET_STD = RECIPE["normalisation_std"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "device": "cpu",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        # Co-tenancy evidence: the machine of record is shared.
        info["loadavg"] = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        pass
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:
        pass
    try:
        import ray

        info["ray"] = ray.__version__
    except Exception:
        pass
    return info


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return path
