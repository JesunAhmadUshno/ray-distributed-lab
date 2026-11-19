# DECISIONS: Ray Distributed Lab

Decisions taken by the builder, in order. Protocol changes
live in PREREGISTRATION.md as amendments; this file records
the reasoning and the implementation-level calls.

- D1 (2026-11-08, before any measured run): the machine of
  record moves from the Muse VM to BlackBoxV2. Diagnosis:
  Ray's GCS never becomes reachable on the Muse VM. A minimal
  grpcio probe reproduces the failure without Ray: default
  epoll1 hangs, GRPC_POLL_STRATEGY=epoll reports "No event
  engine could be initialized from epoll", and
  GRPC_POLL_STRATEGY=poll hangs or is reset by peer. gRPC is
  therefore unusable on that VM, and Ray 2.60 cannot start
  without it. Evidence preserved in
  results/machine-of-record-failure.log; protocol change
  recorded as Amendment A1 in PREREGISTRATION.md. All arms
  move together to preserve the within-study comparisons.
  Precedent: kafka-streaming-lab Amendment A1 (2026-11-08),
  whose broker could not run on the same VM class and whose
  study moved to the same successor machine.
- D2 (2026-11-08, before any measured run): env_info() in
  src/common.py also records the machine load average at run
  start. The successor machine is shared with other services;
  the load reading is co-tenancy evidence for the audit, not a
  measurement input. No definition in PREREGISTRATION.md
  changes.
- D3 (2026-11-08, before any measured run): arm R uses Ray
  Serve, the primary pattern. It starts cleanly on BlackBoxV2
  with one adjustment: its HTTP proxy binds port 8231 instead
  of the default 8000, because port 8000 on BlackBoxV2 is held
  by a pre-existing service (a python3 server.py, not ours, not
  touched). Study clients use the deployment handle, so the
  HTTP port carries no measured traffic. The frozen actors
  fallback was not triggered.
- D4 (2026-11-08, before any measured run): on BlackBoxV2 the
  study runs inside a clone of this repo at ~/ray-distributed-lab
  with the dataset at ~/raylab-data (zip SHA-256 re-verified
  there against the frozen value before extraction). The
  interpreter there is Python 3.14.4, the machine's only
  Python, created with venv --without-pip plus a bootstrapped
  pip because the system Python ships no ensurepip and is
  externally managed (PEP 668). Installed versions match the
  frozen pins exactly (torch 2.15.0+cpu, torchvision
  0.29.2+cpu, ray 2.60.0, numpy 2.5.4); the remote freeze is
  saved as requirements-lock-blackboxv2.txt.
- D5 (2026-11-08, after the runs): the claim-boundary line
  that scripts/make_results.py prints in RESULTS.md still
  named the original machine (4 cores, 7.7 GB) after Amendment
  A1 moved the study. The generator text is corrected and
  RESULTS.md regenerated from the same raw records; a diff
  confirms no number, table or verdict changed, only that
  provenance line.
- D6 (2026-11-08, after the runs): CI runs Python 3.12 only.
  The frozen numpy pin (2.5.4) requires Python 3.12 or newer,
  so the house-standard 3.11 leg cannot install the
  requirements at all (its failure is in the CI log of the
  results commit). The study's interpreters are 3.12.3 and
  3.14.4; pyproject requires-python is raised to match. No
  study code or result is affected. Precedent:
  xgboost-credit-benchmark recorded the same constraint.
- D7 (2026-11-17, correction recorded during the paper
  claim-check): AUDIT.md (Deductions) and README.md both
  quote the run-start load averages on BlackBoxV2 as "ranged
  from 4.6 to 8.6" on 8 cores. That range is wrong at the
  low end. Recomputing from the nine run records in
  results/runs/ (a_s42, a_s43, a_s44, b_s42, b_s43, b_s44,
  c_s42, c_s43, c_s44), taking the 1-minute load average
  recorded at each run start (the first element of
  env/loadavg, the field added under D2), the minimum is
  4.00 (a_s43) and the maximum is 8.64 (c_s44): the correct
  range is 4.00 to 8.64. AUDIT.md and README.md are frozen
  records and are not edited; this entry is the correction
  of record. The paper (preprint, 17 November 2026) already
  discloses this discrepancy in its limitations.
