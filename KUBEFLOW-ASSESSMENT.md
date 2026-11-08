# KUBEFLOW ASSESSMENT

A written verdict, not an install. The criteria are the frozen
ones in PREREGISTRATION.md: (1) install and maintenance weight
on a small fleet with no dedicated platform engineer and no
Kubernetes cluster today; (2) overlap with capabilities
pytorch-depth-lab and this study already prove; (3) what
Kubeflow would add that Ray plus DDP do not. Facts about
Kubeflow below are from its public documentation and release
materials as of November 2026; they are assessment inputs,
not measurements, and are labelled as such.

## Verdict: DEFER

Kubeflow is not adopted for this fleet now, and it is not
rejected outright. The conditions that would reopen the
question are named at the end.

## Criterion 1: install and maintenance weight

Kubeflow is a Kubernetes-native platform. Adopting it means
first adopting Kubernetes: a cluster to provision, secure,
back up, and upgrade, on a fleet whose largest always-on
machine is a single 8-core box shared with other services.
Kubeflow itself installs as a large set of manifests (dozens
of pods across several namespaces in a standard install), and
its upgrade path is a recurring project of its own: the
project's releases move on their own cadence, and in-place
upgrades across major versions have historically meant
careful, staged migration rather than a package bump.

The fleet's platform staff is one founder and his agents.
Every always-on system competes with that attention, and the
programme's own record (this study included) shows the cost
profile that actually bites here is not orchestration: it is
environment fragility and co-tenancy on shared machines. A
Kubernetes control plane plus Kubeflow would add the largest
single maintenance surface the fleet has ever carried, before
it trained a single model.

Assessment: the weight is real, recurring, and front-loaded.

## Criterion 2: overlap with what is already proven

Proven on this fleet, with receipts in sibling repositories:

- Distributed training at small world size: hand-rolled DDP
  (pytorch-depth-lab arm C2) and Ray Train (this study, arm
  C), measured against each other and against a single-process
  baseline under pre-registered rules.
- Model serving at small scale: Ray Serve (this study, arm R)
  and plain in-process loops, measured.
- Experiment tracking and a model registry with promotion
  gates: MLflow, in mlops-evidence-lab, including an exact
  reproduction check of a published champion model.
- Evaluation harnesses: rag-eval-harness (P06) and the Ragas
  wiring in mlops-evidence-lab.

Kubeflow's training operators overlap the first row, KServe
overlaps the second, and its metadata and pipeline stores
overlap the third. For the workloads this fleet actually runs
(measured studies, small models, one operator), the overlap
covers the need.

## Criterion 3: what Kubeflow would add

Named honestly, because a DEFER verdict that hides the upside
is worth nothing:

- Kubeflow Pipelines: scheduled and triggered multi-step
  pipelines with a web UI, run lineage, and artifact
  visualisation. Our pipelines are scripts driven by a study
  driver (scripts/run_study.py here); they have no UI, no
  schedule, and no lineage store. This is Kubeflow's strongest
  addition for us.
- Katib: a hyperparameter tuning service with several search
  algorithms. Our tuning is hand-run random search with the
  trials published (xgboost-credit-benchmark); nothing
  automated exists today.
- KServe: autoscaling, scale-to-zero, and canary serving on
  Kubernetes. Our serving is two replicas on one machine;
  nothing here needs scale-to-zero or traffic splitting.
- Multi-tenancy: namespaces and profiles for several humans
  sharing one platform. This fleet has one operator.
- Notebook servers and a central dashboard: convenience for a
  team, unused by a one-person fleet.

Each addition is real. None of them is a current bottleneck:
the gaps they fill are gaps we have measured ourselves not
hitting at this scale.

## Why DEFER and not REJECT

REJECT would claim Kubeflow cannot fit this fleet's future.
That is not supported: if the fleet grows a second human
operator, or a Kubernetes cluster exists for another reason,
or scheduled production pipelines replace one-off studies as
the daily workload, Kubeflow Pipelines and Katib answer needs
that Ray plus scripts genuinely do not. The verdict is about
now: the install and maintenance cost lands today, and the
benefits land at a scale and staffing level the fleet has not
reached.

## Conditions that reopen this verdict

Any one of these, observed rather than imagined, reopens the
assessment:

1. A Kubernetes cluster is running on the fleet for another
   reason, so Kubeflow's marginal cost drops to its own pods.
2. A second human needs self-serve training or serving
   without going through the founder.
3. A workload needs scheduled, recurring, multi-step pipelines
   in production (weekly or more often), where the study
   driver pattern no longer suffices.
4. A serving target needs autoscaling or canary rollout that
   Ray Serve on one machine cannot express.

Until one of those holds, the standing stack is: hand DDP
when control matters, Ray Train and Ray Serve when ergonomics
and headroom matter, MLflow for tracking and registry. That
stack is measured, in this repository and its siblings, which
is more than this fleet could say for a Kubeflow install on
the day it landed.
