"""Inference study: a plain in-process loop vs Ray serving.

Frozen protocol (PREREGISTRATION.md): the frozen checkpoint
(arm A, seed 42) served two ways at equal total thread budget
(4). The request stream is the frozen test split in fixed
order, cycled; 2 client threads run closed loops; 5 warmup
batches per client are excluded; 40 measured batches per
client per cell. Batch 128 is the primary cell, batch 32 the
secondary. Latency is client send to reply receipt, pooled
across clients; throughput is total images over the makespan.

Arm R pattern: Ray Serve is primary. The actors fallback and
its trigger are frozen in PREREGISTRATION.md and recorded in
DECISIONS.md before any measured run.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .common import RECIPE, RESULTS, env_info, write_json
from .data import make_datasets, make_loader
from .model import build_model

N_CLIENTS = 2
WARMUP = 5
MEASURED = 40
CELLS = (128, 32)


def load_model(ckpt_path: str, threads: int) -> torch.nn.Module:
    torch.set_num_threads(threads)
    model = build_model(0)  # seed irrelevant: weights overwritten
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def predict_np(model: torch.nn.Module, arr: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(arr)
    return model(x).argmax(dim=1).numpy()


def materialise_test_tensor() -> torch.Tensor:
    """The full frozen test split as one tensor, in split order."""
    _, test_ds = make_datasets()
    loader = make_loader(test_ds, 256, shuffle=False)
    parts = [x for x, _ in loader]
    return torch.cat(parts)


def client_batches(tensor: torch.Tensor, batch_size: int) -> list[list[np.ndarray]]:
    """Per-client batch lists. Global batch j takes rows
    (j*bs + k) mod n in order; client c takes global batches
    c, c+2, c+4, ... Each client gets WARMUP + MEASURED."""
    n = tensor.shape[0]
    total_global = N_CLIENTS * (WARMUP + MEASURED)
    global_batches: list[np.ndarray] = []
    for j in range(total_global):
        idx = [(j * batch_size + k) % n for k in range(batch_size)]
        global_batches.append(tensor[idx].numpy())
    return [global_batches[c::N_CLIENTS] for c in range(N_CLIENTS)]


def run_cell(call_fn, batches_per_client: list[list[np.ndarray]]) -> dict:
    """Closed-loop measurement. call_fn(arr) blocks until the
    reply arrives and returns the predictions."""
    latencies: list[float] = []
    lock = threading.Lock()
    start_barrier = threading.Barrier(N_CLIENTS)
    makespan = {"t0": None, "t1": None}

    def client(batches: list[np.ndarray]) -> None:
        start_barrier.wait()
        for i, arr in enumerate(batches):
            t0 = time.perf_counter()
            call_fn(arr)
            dt = time.perf_counter() - t0
            if i >= WARMUP:
                with lock:
                    latencies.append(dt)

    threads = [
        threading.Thread(target=client, args=(b,)) for b in batches_per_client
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.perf_counter()
    total_images = MEASURED * N_CLIENTS * batches_per_client[0][0].shape[0]
    lat = np.array(sorted(latencies))
    return {
        "n_batches_measured": int(lat.size),
        "total_images": int(total_images),
        "makespan_s": t1 - t0,
        "throughput_images_s": total_images / (t1 - t0),
        "latency_p50_s": float(np.percentile(lat, 50)),
        "latency_p95_s": float(np.percentile(lat, 95)),
        "latency_mean_s": float(lat.mean()),
    }


class PlainServer:
    """One process, one worker thread, requests served one
    batch at a time from a queue. Torch threads 4: the whole
    machine, matching arm R's total budget."""

    def __init__(self, ckpt_path: str):
        self.model = load_model(ckpt_path, threads=4)
        self.q: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        torch.set_num_threads(4)
        while True:
            item = self.q.get()
            if item is None:
                return
            arr, holder = item
            holder["pred"] = predict_np(self.model, arr)
            holder["event"].set()

    def call(self, arr: np.ndarray) -> np.ndarray:
        holder = {"event": threading.Event(), "pred": None}
        self.q.put((arr, holder))
        holder["event"].wait()
        return holder["pred"]

    def close(self) -> None:
        self.q.put(None)
        self.thread.join(timeout=10)


def run_plain(ckpt_path: str, tensor: torch.Tensor) -> dict:
    server = PlainServer(ckpt_path)
    try:
        out = {}
        for bs in CELLS:
            out[str(bs)] = run_cell(server.call, client_batches(tensor, bs))
        return out
    finally:
        server.close()


def run_ray_serve(ckpt_path: str, tensor: torch.Tensor) -> dict:
    import ray
    from ray import serve

    ckpt_abs = str(Path(ckpt_path).resolve())

    @serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 2})
    class Predictor:
        def __init__(self, ckpt: str):
            self.model = load_model(ckpt, threads=2)

        def __call__(self, arr: np.ndarray) -> np.ndarray:
            return predict_np(self.model, arr)

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    try:
        # Port 8000 is held by a pre-existing service on the
        # machine of record (DECISIONS.md D3). Clients use the
        # deployment handle, so the HTTP proxy port carries no
        # study traffic; it only has to be free.
        serve.start(detached=False, http_options={"host": "127.0.0.1", "port": 8231})
        handle = serve.run(Predictor.bind(ckpt_abs), name="raylab_predictor")
        # Readiness probe outside the measured stream.
        probe = np.zeros((1, 3, 64, 64), dtype=np.float32)
        handle.remote(probe).result()

        def call(arr: np.ndarray) -> np.ndarray:
            return handle.remote(arr).result()

        out = {}
        for bs in CELLS:
            out[str(bs)] = run_cell(call, client_batches(tensor, bs))
        return out
    finally:
        try:
            serve.shutdown()
        finally:
            ray.shutdown()


def run_ray_actors(ckpt_path: str, tensor: torch.Tensor) -> dict:
    import ray

    ckpt_abs = str(Path(ckpt_path).resolve())

    @ray.remote(num_cpus=2)
    class PredictActor:
        def __init__(self, ckpt: str):
            self.model = load_model(ckpt, threads=2)

        def predict(self, arr: np.ndarray) -> np.ndarray:
            return predict_np(self.model, arr)

    ray.init(ignore_reinit_error=True, log_to_driver=False)
    try:
        actors = [PredictActor.remote(ckpt_abs) for _ in range(N_CLIENTS)]
        probe = np.zeros((1, 3, 64, 64), dtype=np.float32)
        ray.get([a.predict.remote(probe) for a in actors])
        counters = [0]
        counter_lock = threading.Lock()

        def call(arr: np.ndarray) -> np.ndarray:
            # Round-robin across the two actors. Under 2 closed
            # loop clients this approximates replica routing.
            with counter_lock:
                k = counters[0]
                counters[0] += 1
            return ray.get(actors[k % N_CLIENTS].predict.remote(arr))

        out = {}
        for bs in CELLS:
            out[str(bs)] = run_cell(call, client_batches(tensor, bs))
        return out
    finally:
        ray.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["P", "R"], required=True)
    ap.add_argument("--pattern", choices=["serve", "actors"], default="serve")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    tensor = materialise_test_tensor()
    if args.arm == "P":
        cells = run_plain(args.checkpoint, tensor)
        pattern = "plain-loop"
    elif args.pattern == "serve":
        cells = run_ray_serve(args.checkpoint, tensor)
        pattern = "ray-serve"
    else:
        cells = run_ray_actors(args.checkpoint, tensor)
        pattern = "ray-actors"
    payload = {
        "arm": args.arm,
        "pattern": pattern,
        "checkpoint": args.checkpoint,
        "protocol": {
            "clients": N_CLIENTS,
            "warmup_per_client": WARMUP,
            "measured_per_client": MEASURED,
            "total_threads": 4,
        },
        "cells": cells,
        "env": env_info(),
        "command": " ".join(os.sys.argv),
    }
    write_json(Path(args.out), payload)
    print(json.dumps(cells, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
