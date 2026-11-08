# Ray Distributed Lab

## Research question

Against the CPU distributed-training baselines already measured
in pytorch-depth-lab, what does Ray's training and serving stack
cost or buy on the same workload family: scaling efficiency,
accuracy parity, and inference throughput?

The design was frozen before any run: PREREGISTRATION.md (the
first content commit of this repository) fixes the recipe
(ResNet-18 on a stratified 4,000-image EuroSAT subset, 3 epochs,
effective batch 128, seeds 42/43/44, loader workers 0 in every
arm so the framework comparison is not a loader comparison),
the three arms, the measurement definitions, and the verdict
rules. RESULTS.md is generated from the recorded runs by
scripts/make_results.py; nothing in it is hand-typed.

## Verdicts

- **H1 CONFIRMED: Ray Train matches hand-rolled DDP on
  scaling.** Median training throughput: single process 50.7
  samples/s, DDP 57.8, Ray Train 55.1. Scaling efficiencies
  against the single process: DDP 0.570, Ray Train 0.543, an
  absolute difference of 0.027 inside the frozen plus or minus
  0.10 parity band. Ray's training tax over hand-written DDP at
  world size 2 is about 5 percent of throughput, and both
  distributed arms are far from linear scaling (0.54 to 0.57)
  on this workload: two workers buy roughly 1.1x the samples
  per second of one process at twice the processes.
- **H2 CONFIRMED: accuracy parity holds, this time.** Mean
  final test accuracy: single 0.7180, DDP 0.6923 (-2.57 pp),
  Ray Train 0.7077 (-1.04 pp), both inside the frozen plus or
  minus 5.0 pp band. The band was set knowing the baseline
  study's DDP arm once diverged by 9.59 pp on validation
  accuracy; at this subset and seed set the divergence did not
  recur, and the per-seed table in RESULTS.md shows the noise
  the band had to absorb (the single-process arm alone spans
  0.660 to 0.757 across seeds).
- **H3 INCONCLUSIVE: Ray Serve is ahead, but not by the
  registered margin.** At batch 128 and equal total thread
  budget (4), Ray Serve served 218.5 images/s against the
  plain loop's 200.4 (ratio 1.090), with lower latency (p50
  1.067 s vs 1.135 s). The frozen rule confirms only at 1.25x
  and refutes at 1.00x or below, so 1.090 lands in the
  inconclusive zone by design. At batch 32 the ratio is 1.058
  (descriptive, no verdict). The honest summary: no serving
  tax was measured for the framework at this scale, and a
  small gain was, whose mechanism this study did not
  instrument.

## The machine of record moved, and the evidence is kept

The study was frozen against a 4-core VM. That VM cannot run
Ray at all: its sandbox has no working epoll, and gRPC, which
Ray's control plane requires, hangs under every poll strategy
(default, epoll, poll). The failure was established with a
minimal grpcio probe, preserved in
results/machine-of-record-failure.log, and recorded as
Amendment A1 to PREREGISTRATION.md before any measured run.
The whole study moved to BlackBoxV2 (8 cores, 14 GB RAM), the
same successor machine kafka-streaming-lab moved to for the
same class of reason, so every within-study comparison stands.
The baseline study's absolute numbers, measured on the
original VM class, are context and priors only; RESULTS.md
says so next to them. Package versions on the successor
machine match the frozen pins exactly (torch 2.15.0+cpu,
torchvision 0.29.2+cpu, ray 2.60.0, numpy 2.5.4) under
Python 3.14.4.

One threat the reader should hold onto: BlackBoxV2 is shared
with other live services. Load averages recorded at run starts
ranged from 4.6 to 8.6 on 8 cores. Arms ran sequentially in
A, B, C order, so tenant drift is not balanced across arms;
medians over 3 runs are the mitigation the design has, and
AUDIT.md charges for the rest.

## Why this matters beyond this repository

Distributed frameworks promise scale to teams that mostly run
small. The sales pitch is written at cluster scale; the bill,
in money and in operational weight, arrives at whatever scale
the team actually has. This study measured the two-worker
truth on ordinary hardware: a managed framework (Ray Train)
cost about 5 percent against hand-rolled DDP and returned the
same accuracy, and its serving stack (Ray Serve) cost nothing
measurable against a hand-written loop. For a small team that
is the decision-relevant shape of the answer: at this scale
you choose the framework for its ergonomics and its headroom,
not because the hand-rolled version is faster, because here it
essentially is not. Numbers at one machine and world size 2 do
not travel; the method, pre-registered hypotheses, frozen
measurement definitions, verdicts published either way, does.

## Reproduce

The study ran on BlackBoxV2 (Amendment A1). The same commands
work on any machine where Ray can start:

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q        # unit and smoke (fast)
EUROSAT_IMAGE_DIR=/path/to/EuroSAT_RGB .venv/bin/python scripts/run_study.py --pattern serve
.venv/bin/python scripts/make_results.py    # RESULTS.md and results/summary.json
```

EuroSAT is not committed here; obtain EuroSAT_RGB.zip from
Zenodo record 7711810 (SHA-256 in PREREGISTRATION.md) and
point EUROSAT_IMAGE_DIR at the extracted folder. The full
driver takes about an hour at this scale and runs every cell
sequentially.

## Layout

- `PREREGISTRATION.md`: frozen design, hypotheses, verdict
  rules, and Amendment A1 (machine of record).
- `RESULTS.md`: generated verdicts and raw tables.
- `KUBEFLOW-ASSESSMENT.md`: the written Kubeflow verdict
  (DEFER), against the criteria frozen in PREREGISTRATION.md.
- `AUDIT.md`: honest self-audit, graded out of 10,000.
- `DECISIONS.md`: D1 to D5, including the machine move and
  the Serve port.
- `src/`: shared training core, the three arm launchers,
  the serving study, data and model code.
- `scripts/`: the study driver, the results generator, the
  versions recorder.
- `results/`: raw per-run records, inference records,
  dataset and versions records, the driver summary, the
  machine-of-record failure log, and the Jev claim check.

Licence: Apache-2.0 (LICENSE). Component and dataset
provenance: LICENSES.md.
