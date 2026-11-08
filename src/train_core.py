"""One training implementation, three orchestrations.

execute_training runs the frozen recipe identically for arm A
(single process), arm B (torch DDP launched by hand) and arm C
(Ray Train TorchTrainer). The only differences between arms are
the ones under study: who launches the processes, who wraps the
model, and who owns the rendezvous. Optimizer, schedule, data,
sampler contract, loader configuration, thread count and
measurement code are shared, byte for byte.

Measurement (frozen in PREREGISTRATION.md): step time is
accumulated around the forward, loss, backward and optimizer
step block only; the first 5 steps of epoch 1 are excluded from
both the time and the sample count; for distributed arms the
slower rank's accumulated step time is used, after a check that
both ranks counted the same number of steps.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data.distributed import DistributedSampler

from .common import RECIPE, env_info, seed_all, write_json
from .data import make_datasets, make_loader
from .model import build_model

WARMUP_STEPS = RECIPE["warmup_steps_excluded"]
THREADS = RECIPE["threads_per_process"]


@torch.no_grad()
def evaluate(model: nn.Module, loader) -> float:
    model.eval()
    correct, seen = 0, 0
    for x, y in loader:
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum())
        seen += y.numel()
    return correct / max(seen, 1)


def execute_training(cfg: dict, rank: int, world: int, wrap_fn=None) -> dict | None:
    """Run one frozen training run. Rank 0 writes the run JSON to
    cfg["out_path"] and returns the payload; other ranks return
    None. cfg keys: arm, seed, out_path, epochs, and optionally
    checkpoint_out."""
    torch.set_num_threads(THREADS)
    seed = int(cfg["seed"])
    epochs = int(cfg.get("epochs", RECIPE["epochs"]))
    seed_all(seed)
    device = torch.device("cpu")

    train_ds, test_ds = make_datasets()
    per_rank_batch = RECIPE["effective_batch"] // world
    sampler = None
    if world > 1:
        sampler = DistributedSampler(
            train_ds, num_replicas=world, rank=rank, shuffle=True, seed=seed
        )
    loader = make_loader(
        train_ds, per_rank_batch, shuffle=sampler is None, seed=seed, sampler=sampler
    )
    test_loader = make_loader(test_ds, RECIPE["effective_batch"], shuffle=False)

    model = build_model(seed)
    if wrap_fn is not None:
        model = wrap_fn(model)
    opt = torch.optim.AdamW(
        model.parameters(), lr=RECIPE["lr"], betas=RECIPE["betas"],
        weight_decay=RECIPE["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    rows = []
    step_time_sum = 0.0
    counted_steps = 0
    counted_samples = 0  # global samples in counted steps
    global_step = 0
    t_start = time.perf_counter()
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        ep_t0 = time.perf_counter()
        loss_sum, seen = 0.0, 0
        model.train()
        for x, y in loader:
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            dt = time.perf_counter() - t0
            global_step += 1
            if global_step > WARMUP_STEPS:
                step_time_sum += dt
                counted_steps += 1
                counted_samples += y.numel() * world
            loss_sum += loss.item() * y.numel()
            seen += y.numel()
        sched.step()
        mean_loss = loss_sum / max(seen, 1)
        if world > 1 and dist.is_available() and dist.is_initialized():
            t = torch.tensor([mean_loss * seen, float(seen)])
            dist.all_reduce(t)
            mean_loss = (t[0] / t[1]).item()
        rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": mean_loss,
                "epoch_train_wall_s": time.perf_counter() - ep_t0,
            }
        )

    if world > 1 and dist.is_available() and dist.is_initialized():
        t = torch.tensor([step_time_sum, float(counted_steps)])
        gathered = [torch.zeros_like(t) for _ in range(world)]
        dist.all_gather(gathered, t)
        counts = [int(g[1]) for g in gathered]
        if len(set(counts)) != 1:
            raise RuntimeError(f"rank step counts disagree: {counts}")
        step_time_used = max(float(g[0]) for g in gathered)
    else:
        step_time_used = step_time_sum

    payload = None
    if rank == 0:
        raw = model.module if hasattr(model, "module") else model
        test_acc = evaluate(raw, test_loader)
        throughput = counted_samples / step_time_used if step_time_used > 0 else 0.0
        payload = {
            "arm": cfg["arm"],
            "seed": seed,
            "world_size": world,
            "epochs": epochs,
            "per_rank_batch": per_rank_batch,
            "effective_batch": RECIPE["effective_batch"],
            "threads_per_process": THREADS,
            "loader_workers": RECIPE["loader_workers"],
            "counted_steps": counted_steps,
            "counted_samples": counted_samples,
            "step_time_s": step_time_used,
            "throughput_samples_s": throughput,
            "test_accuracy": test_acc,
            "epoch_rows": rows,
            "wall_s": time.perf_counter() - t_start,
            "env": env_info(),
            "command": cfg.get("command", ""),
        }
        ckpt = cfg.get("checkpoint_out")
        if ckpt:
            ckpt_path = Path(ckpt)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(raw.state_dict(), ckpt_path)
            payload["checkpoint_path"] = str(ckpt_path)
            payload["checkpoint_sha256"] = hashlib.sha256(
                ckpt_path.read_bytes()
            ).hexdigest()
        write_json(Path(cfg["out_path"]), payload)
    return payload
