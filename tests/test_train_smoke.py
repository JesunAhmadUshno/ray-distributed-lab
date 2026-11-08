"""End-to-end smoke of the shared training core on synthetic
tensors. No EuroSAT data, no Ray, no DDP: this proves the
measurement code path (warmup exclusion, counting, evaluation,
JSON write) that all three arms share.
"""
import json

import torch
from torch.utils.data import TensorDataset

from src import train_core


def _tiny_datasets(n_train=1024, n_test=128):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n_train, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n_train,), generator=g)
    xt = torch.randn(n_test, 3, 32, 32, generator=g)
    yt = torch.randint(0, 10, (n_test,), generator=g)
    return TensorDataset(x, y), TensorDataset(xt, yt)


def test_execute_training_single(tmp_path, monkeypatch):
    train_ds, test_ds = _tiny_datasets()
    monkeypatch.setattr(train_core, "make_datasets", lambda: (train_ds, test_ds))
    out = tmp_path / "run.json"
    cfg = {
        "arm": "A",
        "seed": 42,
        "epochs": 1,
        "out_path": str(out),
        "checkpoint_out": None,
        "command": "pytest smoke",
    }
    payload = train_core.execute_training(cfg, rank=0, world=1, wrap_fn=None)
    assert payload is not None
    # 1024 samples at batch 128 = 8 steps; the first 5 are warmup.
    assert payload["counted_steps"] == 3
    assert payload["counted_samples"] == 3 * 128
    assert payload["throughput_samples_s"] > 0
    assert 0.0 <= payload["test_accuracy"] <= 1.0
    saved = json.loads(out.read_text())
    assert saved["arm"] == "A" and saved["seed"] == 42
