"""Launchers for the three training arms.

Usage:
  python -m src.run_training --arm A --seed 42 --out PATH [--checkpoint-out PATH]
  python -m src.run_training --arm B --seed 42 --out PATH
  python -m src.run_training --arm C --seed 42 --out PATH

Arm B rendezvous follows the baseline study's DECISIONS D8
lessons for this VM: a file store unique per run, and gloo
pinned to the loopback interface. Arm C's workers receive the
repo root in their train-loop config and put it on sys.path
themselves, so imports do not depend on the worker's cwd.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path

from .common import REPO_ROOT, RESULTS
from .data import IMAGE_DIR


def _cfg(args) -> dict:
    cfg = {
        "arm": args.arm,
        "seed": args.seed,
        "out_path": str(Path(args.out).resolve()),
        "checkpoint_out": str(Path(args.checkpoint_out).resolve())
        if args.checkpoint_out
        else None,
        "command": " ".join(sys.argv),
    }
    if args.epochs is not None:
        # Smoke-run convenience only. The study driver never
        # passes this; frozen runs use RECIPE epochs, and every
        # run JSON records the epochs actually used.
        cfg["epochs"] = args.epochs
    return cfg


def run_arm_a(cfg: dict) -> None:
    from .train_core import execute_training

    execute_training(cfg, rank=0, world=1, wrap_fn=None)


def _ddp_worker(rank: int, world: int, cfg: dict) -> None:
    import torch.distributed as dist
    from datetime import timedelta

    from torch.nn.parallel import DistributedDataParallel as DDP

    from .train_core import execute_training

    dist.init_process_group(
        "gloo",
        init_method=f"file://{cfg['store_path']}",
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=10),
    )
    try:
        execute_training(cfg, rank=rank, world=world, wrap_fn=lambda m: DDP(m))
    finally:
        dist.destroy_process_group()


def run_arm_b(cfg: dict) -> None:
    import torch.multiprocessing as mp

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    # Baseline D8: the default gloo interface stalls peer traffic
    # between local ranks on this VM; pin gloo to loopback.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    store = Path(tempfile.mkdtemp(prefix="raylab_ddp_")) / "store"
    cfg["store_path"] = str(store)
    mp.spawn(_ddp_worker, args=(2, cfg), nprocs=2, join=True)
    store.unlink(missing_ok=True)


def run_arm_c(cfg: dict) -> None:
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    cfg["repo_root"] = str(REPO_ROOT)
    cfg["image_dir"] = str(IMAGE_DIR)
    cfg["run_name"] = f"armC_s{cfg['seed']}"

    def train_fn(config: dict) -> None:
        import sys as _sys

        _sys.path.insert(0, config["repo_root"])
        os.environ["EUROSAT_IMAGE_DIR"] = config["image_dir"]
        from ray.train import get_context
        from ray.train.torch import prepare_model

        from src.train_core import execute_training

        ctx = get_context()
        execute_training(
            config,
            rank=ctx.get_world_rank(),
            world=ctx.get_world_size(),
            wrap_fn=prepare_model,
        )

    trainer = TorchTrainer(
        train_fn,
        train_loop_config=cfg,
        scaling_config=ScalingConfig(
            num_workers=2, use_gpu=False, resources_per_worker={"CPU": 2}
        ),
        run_config=RunConfig(
            name=cfg["run_name"], storage_path=str(RESULTS / "ray_train")
        ),
    )
    trainer.fit()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    cfg = _cfg(args)
    try:
        if args.arm == "A":
            run_arm_a(cfg)
        elif args.arm == "B":
            run_arm_b(cfg)
        else:
            run_arm_c(cfg)
    except Exception:
        traceback.print_exc()
        return 1
    if not Path(cfg["out_path"]).exists():
        print(f"run finished without writing {cfg['out_path']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
