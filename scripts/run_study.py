"""Study driver: run every registered cell sequentially.

Order (frozen by dependency): dataset verify, versions, arm A
seeds 42/43/44 (seed 42 also writes the inference checkpoint),
arm B seeds 42/43/44, arm C seeds 42/43/44, inference arm P,
inference arm R. One heavy matrix at a time on the shared VM:
each cell is a subprocess and the next starts only after the
previous exits.

A cell whose subprocess fails or never writes its JSON gets a
synthesised failure record (status: failed) pointing at its
log, per stop rule S1: failures are reported, never dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
LOGS = RESULTS / "logs"
SEEDS = (42, 43, 44)
CELL_TIMEOUT_S = 3600


def run_cell(name: str, cmd: list[str], expect: Path) -> dict:
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{name}.log"
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=CELL_TIMEOUT_S,
                env={**os.environ, "OMP_NUM_THREADS": "2"},
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    wall = time.time() - t0
    if expect.exists():
        payload = json.loads(expect.read_text())
        payload.setdefault("driver_wall_s", wall)
        return {"name": name, "status": "ok", "rc": rc, "wall_s": wall, "record": payload}
    failure = {
        "status": "failed",
        "cell": name,
        "rc": rc,
        "wall_s": wall,
        "log": str(log_path.relative_to(REPO)),
    }
    expect.parent.mkdir(parents=True, exist_ok=True)
    expect.write_text(json.dumps(failure, indent=2, sort_keys=True))
    return {"name": name, "status": "failed", "rc": rc, "wall_s": wall, "record": failure}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", choices=["serve", "actors"], default="serve",
                    help="arm R pattern, as decided in DECISIONS.md before the runs")
    ap.add_argument("--skip-inference", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    cells = []

    cells.append(run_cell("dataset_verify", [py, "-m", "src.data"], RESULTS / "dataset.json"))
    cells.append(run_cell("versions", [py, "scripts/write_versions.py"], RESULTS / "versions.json"))

    ckpt = RESULTS / "checkpoints" / "armA_s42.pt"
    for arm in ("A", "B", "C"):
        for seed in SEEDS:
            out = RESULTS / "runs" / f"{arm.lower()}_s{seed}.json"
            cmd = [py, "-m", "src.run_training", "--arm", arm, "--seed", str(seed),
                   "--out", str(out)]
            if arm == "A" and seed == 42:
                cmd += ["--checkpoint-out", str(ckpt)]
            cells.append(run_cell(f"train_{arm}_s{seed}", cmd, out))

    if not args.skip_inference:
        out_p = RESULTS / "inference" / "p.json"
        cells.append(run_cell(
            "inference_P",
            [py, "-m", "src.serve_study", "--arm", "P", "--checkpoint", str(ckpt),
             "--out", str(out_p)],
            out_p,
        ))
        out_r = RESULTS / "inference" / "r.json"
        cells.append(run_cell(
            "inference_R",
            [py, "-m", "src.serve_study", "--arm", "R", "--pattern", args.pattern,
             "--checkpoint", str(ckpt), "--out", str(out_r)],
            out_r,
        ))

    summary = {c["name"]: c["status"] for c in cells}
    print(json.dumps(summary, indent=2))
    (RESULTS / "driver_summary.json").write_text(
        json.dumps({"cells": summary}, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
