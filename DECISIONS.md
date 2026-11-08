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
