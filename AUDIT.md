# AUDIT: Ray Distributed Lab

Self-audit against the Gap-Closure Programme standard,
written after the results were generated and recounted from
the raw records. Grade: 6,400 of 10,000.

## What was verified

- Pre-registration commit b2ff00072da438cc29e95921541aced8c815ce26
  (PREREGISTRATION.md) precedes every harness and results
  file. Amendment A1 (machine of record) was appended to it
  before any measured run; the only earlier executions were
  1-epoch smoke runs written to /tmp on both machines, which
  are not study data and are named as such in the failure log.
- The machine failure was diagnosed, not assumed: a minimal
  grpcio probe fails on the original VM under every poll
  strategy (epoll unavailable, epoll1 hangs, poll hangs or is
  reset), and the same probe passes instantly on the
  successor machine. Evidence: results/machine-of-record-
  failure.log.
- Dataset integrity on both machines: the EuroSAT zip SHA-256
  matches the frozen value (b4f5b234...b90), the regenerated
  split is 21,600 / 2,700 / 2,700 exactly as the baseline
  study's, and the subset is 4,000 by the frozen rule
  (400 per class, first in split order).
- All 13 registered cells completed: results/
  driver_summary.json shows ok for dataset verify, versions,
  9 training runs, and both inference arms. Verdicts in
  RESULTS.md reproduce from the raw records under the frozen
  rules; regenerating RESULTS.md locally after the sync
  produced identical numbers and verdicts (the only diff ever
  seen in regeneration is the DECISIONS D5 provenance line).
- Counted steps reconcile in every training run: 91 counted
  steps (96 total steps across 3 epochs minus the 5 frozen
  warmup steps), and counted samples equal steps times the
  effective batch accounting in src/train_core.py.
- Arms B and C return identical test accuracy on seeds 42 and
  43 (0.6930, 0.6774), as they should under the shared sampler
  contract, and diverge on seed 44 (0.7067 vs 0.7526). Both
  facts are printed in RESULTS.md; the divergence is charged
  below, not explained away.
- Jev claim_check on the headline verdicts: model
  jev-1.13.0, noul 0.20, route REVIEW (bands: clear 0.85+,
  block 0.10 or below). On the REVIEW route every figure was
  recomputed by hand from the raw records at full precision
  and matched RESULTS.md at its printed precision. Record:
  results/jev-claim-check.json.
- Build failures were kept, not edited out: the first harness
  commit missed three files through a failed git pathspec
  (fixed in the next commit, efb9d17, before any run); Ray
  Serve's first start on the successor machine failed on a
  port held by a pre-existing service and was fixed by a
  recorded port change (DECISIONS D3), not by touching the
  tenant.

## Deductions (total 3,600)

- 900: the machine of record changed under Amendment A1. The
  protocol was followed exactly and the evidence is complete,
  but the study was designed against pytorch-depth-lab's
  machine and no longer shares it, so the cross-study absolute
  comparison is void and only the within-study comparisons
  carry the hypotheses.
- 800: co-tenancy is uncontrolled. Load averages recorded at
  run starts on BlackBoxV2 ranged from 4.6 to 8.6 on 8 cores,
  and the driver ran arms in fixed A, B, C order, so tenant
  drift is not balanced across arms. Medians over 3 runs are
  the only mitigation, and they do not remove the bias risk
  from the throughput table.
- 500: accuracy evidence is thin. The single-process arm's
  own per-seed spread is 9.7 pp peak to peak (0.660 to 0.757),
  so H2's CONFIRMED rests on means of n = 3 against a 5.0 pp
  band. The threat was named in the pre-registration; the
  verdict is what the frozen rule computes, and the evidence
  strength is what it is.
- 400: one dataset, one small model, one machine, world
  size 2 only. External validity is narrow by design and
  stays narrow in every claim.
- 300: arms B and C diverge on seed 44's test accuracy with
  no instrumented cause. The study carries no per-step loss
  comparison between the hand DDP and Ray Train paths, so
  the difference is reported, not explained.
- 300: the plain serving baseline is one design (a single
  queue thread feeding one process). A process-pool plain
  baseline might narrow or erase the 1.090 ratio. H3's
  inconclusive zone is the honest verdict for that
  sensitivity, but the sensitivity itself is a design limit.
- 200: the inference matrix ran once per cell: 80 measured
  batches behind each percentile. No repetition pass exists
  to bound cell-level variance.
- 200: Ray Train's cluster startup sits in arm C's wall times
  but outside its step-time throughput by the frozen
  definition. Both numbers are printed, and a reader comparing
  wall clocks will see a larger Ray cost than the headline
  5 percent throughput tax. The confusion risk stands.

## What the grade is not

It is not a grade of the hypotheses. One verdict landed in
the inconclusive zone the frozen rules built for it, and the
study's most expensive event, losing its machine before the
first run, is graded as a cost even though it was handled by
the book. The grade rewards the pre-registration surviving
contact with a hostile environment with its evidence intact.

## Carried forward

A per-step loss trace shared by arms B and C, to close the
seed-44 question. A quieter machine window, or a load-gated
driver, for any rerun of the timing matrix. A process-pool
plain serving baseline, to test whether H3's 1.090 is a
framework property or a baseline-design property.
