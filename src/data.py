"""EuroSAT data: the baseline study's frozen split, plus this
study's frozen stratified subset.

The split code is copied from pytorch-depth-lab's src/data.py
(seed 20261107, fractions 0.8 / 0.1 / 0.1, regenerated
deterministically from the class folders), so the indices are
identical to the baseline's. The subset rule is frozen in
PREREGISTRATION.md: per class, the first 400 of that class's
train indices in the frozen split's sorted order.

Images are read from a local EuroSAT_RGB folder that is never
committed. Location resolution: the EUROSAT_IMAGE_DIR
environment variable, else the sibling pytorch-depth-lab
checkout's data folder (the verified copy both studies share).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .common import IMAGENET_MEAN, IMAGENET_STD, RECIPE, RESULTS

_DEFAULT_IMAGE_DIR = (
    Path.home()
    / "workspace/gap-closure/pytorch-depth-lab/data/EuroSAT_RGB"
)
IMAGE_DIR = Path(os.environ.get("EUROSAT_IMAGE_DIR", str(_DEFAULT_IMAGE_DIR)))
ZIP_PATH = IMAGE_DIR.parent / "EuroSAT_RGB.zip"
SPLIT_SEED = RECIPE["split_seed"]
FRACTIONS = (0.8, 0.1, 0.1)  # train, val, test
SUBSET_PER_CLASS = RECIPE["subset_per_class"]


def class_files(image_dir: Path | None = None) -> dict[str, list[str]]:
    if image_dir is None:
        image_dir = IMAGE_DIR
    out: dict[str, list[str]] = {}
    for cls_dir in sorted(p for p in image_dir.iterdir() if p.is_dir()):
        out[cls_dir.name] = sorted(f.name for f in cls_dir.glob("*.jpg"))
    return out


def build_split(image_dir: Path | None = None) -> dict:
    """Stratified split as lists of global indices into the sorted
    (class, file) enumeration used by ImageFolder. Byte-identical
    logic to the baseline study."""
    if image_dir is None:
        image_dir = IMAGE_DIR
    files = class_files(image_dir)
    classes = sorted(files)
    index_of: dict[tuple[str, str], int] = {}
    k = 0
    for cls in classes:
        for name in files[cls]:
            index_of[(cls, name)] = k
            k += 1
    rng = random.Random(SPLIT_SEED)
    split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    per_class: dict[str, dict[str, int]] = {}
    for cls in classes:
        idx = [index_of[(cls, name)] for name in files[cls]]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * FRACTIONS[0])
        n_val = int(n * FRACTIONS[1])
        parts = {
            "train": idx[:n_train],
            "val": idx[n_train : n_train + n_val],
            "test": idx[n_train + n_val :],
        }
        per_class[cls] = {k2: len(v) for k2, v in parts.items()}
        for k2, v in parts.items():
            split[k2].extend(v)
    for v in split.values():
        v.sort()
    return {"seed": SPLIT_SEED, "classes": classes, "per_class": per_class, "split": split}


def build_subset(split: dict) -> dict:
    """The frozen training subset: per class, the first
    SUBSET_PER_CLASS of that class's train indices in the frozen
    split's sorted order. The split's train list is globally
    sorted, so filtering it per class preserves that order."""
    files = class_files()
    classes = split["classes"]
    # Global index to class, from the same enumeration as the split.
    index_class: dict[int, str] = {}
    k = 0
    for cls in classes:
        for _ in files[cls]:
            index_class[k] = cls
            k += 1
    per_class: dict[str, list[int]] = {cls: [] for cls in classes}
    for idx in split["split"]["train"]:
        per_class[index_class[idx]].append(idx)
    subset: list[int] = []
    counts: dict[str, int] = {}
    for cls in classes:
        take = per_class[cls][:SUBSET_PER_CLASS]
        counts[cls] = len(take)
        subset.extend(take)
    subset.sort()
    return {"rule": f"first {SUBSET_PER_CLASS} train indices per class, frozen split order",
            "per_class": counts, "size": len(subset), "indices": subset}


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


def load_split_and_subset() -> tuple[dict, dict]:
    split_path = RESULTS / "split.json"
    subset_path = RESULTS / "subset.json"
    if split_path.exists() and subset_path.exists():
        split = json.loads(split_path.read_text())
        subset = json.loads(subset_path.read_text())
        return split, subset
    split = build_split()
    subset = build_subset(split)
    RESULTS.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2, sort_keys=True))
    subset_path.write_text(json.dumps(subset, indent=2, sort_keys=True))
    return split, subset


def make_datasets(image_dir: Path | None = None):
    """Returns (train_subset_ds, test_ds) honouring the frozen
    split and subset."""
    if image_dir is None:
        image_dir = IMAGE_DIR
    split, subset = load_split_and_subset()
    train_base = ImageFolder(image_dir, transform=train_transform())
    eval_base = ImageFolder(image_dir, transform=eval_transform())
    return (
        Subset(train_base, subset["indices"]),
        Subset(eval_base, split["split"]["test"]),
    )


def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
    sampler=None,
) -> DataLoader:
    """The frozen loader configuration: workers 0, pin off, in
    every arm (PREREGISTRATION.md)."""
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        generator=gen,
    )


def verify() -> dict:
    """Recompute dataset facts and write results/dataset.json,
    results/split.json and results/subset.json."""
    sha = hashlib.sha256()
    with open(ZIP_PATH, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    files = class_files()
    counts = {cls: len(v) for cls, v in files.items()}
    split = build_split()
    subset = build_subset(split)
    payload = {
        "dataset": "EuroSAT RGB",
        "zenodo_record": 7711810,
        "version": "v2",
        "zip_md5_record": "f46e308c4d50d4bf32fedad2d3b",
        "zip_sha256": sha.hexdigest(),
        "image_dir": str(IMAGE_DIR),
        "total_images": sum(counts.values()),
        "per_class_counts": counts,
        "image_size_checked": list(
            Image.open(IMAGE_DIR / split["classes"][0] / files[split["classes"][0]][0]).size
        ),
        "split_sizes": {k: len(v) for k, v in split["split"].items()},
        "subset_size": subset["size"],
        "subset_per_class": subset["per_class"],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dataset.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (RESULTS / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True))
    (RESULTS / "subset.json").write_text(json.dumps(subset, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
