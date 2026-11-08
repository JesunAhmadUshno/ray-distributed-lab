"""Split and subset logic on a synthetic folder tree.

The real EuroSAT tree is never touched by tests. The synthetic
tree has 10 classes of 20 images each, so the frozen fractions
give 16 train / 2 val / 2 test per class, and the subset rule
(per class, the first SUBSET_PER_CLASS train indices, capped by
what exists) takes all 16.
"""
from PIL import Image

from src import data


def _make_tree(tmp_path):
    root = tmp_path / "EuroSAT_RGB"
    for c in range(10):
        cls = root / f"Class{c:02d}"
        cls.mkdir(parents=True)
        for i in range(20):
            Image.new("RGB", (8, 8), color=(c * 20, i * 10, 0)).save(
                cls / f"img_{i:03d}.jpg"
            )
    return root


def test_split_sizes_and_determinism(tmp_path):
    root = _make_tree(tmp_path)
    s1 = data.build_split(root)
    s2 = data.build_split(root)
    assert s1 == s2
    assert len(s1["split"]["train"]) == 160
    assert len(s1["split"]["val"]) == 20
    assert len(s1["split"]["test"]) == 20
    assert s1["per_class"]["Class00"] == {"train": 16, "val": 2, "test": 2}


def test_subset_rule(tmp_path, monkeypatch):
    root = _make_tree(tmp_path)
    monkeypatch.setattr(data, "IMAGE_DIR", root)
    split = data.build_split(root)
    subset = data.build_subset(split)
    assert subset["size"] == 160  # 16 per class exist, fewer than the cap
    assert subset["indices"] == sorted(subset["indices"])
    assert set(subset["indices"]) <= set(split["split"]["train"])
    assert all(v == 16 for v in subset["per_class"].values())


def test_subset_cap_takes_first_in_split_order(tmp_path, monkeypatch):
    root = _make_tree(tmp_path)
    monkeypatch.setattr(data, "IMAGE_DIR", root)
    monkeypatch.setattr(data, "SUBSET_PER_CLASS", 5)
    split = data.build_split(root)
    subset = data.build_subset(split)
    assert subset["size"] == 50
    # Per class, the taken indices are exactly the first 5 of that
    # class's train indices in the split's sorted order.
    files = data.class_files(root)
    index_class = {}
    k = 0
    for cls in split["classes"]:
        for _ in files[cls]:
            index_class[k] = cls
            k += 1
    per_class = {cls: [] for cls in split["classes"]}
    for idx in split["split"]["train"]:
        per_class[index_class[idx]].append(idx)
    for cls in split["classes"]:
        taken = [i for i in subset["indices"] if index_class[i] == cls]
        assert taken == per_class[cls][:5]
