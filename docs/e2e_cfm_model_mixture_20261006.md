# CFM coverage intervention 1: fixed whole-model mixture

## Status and authorized scope

**Resource hold after the user-requested target-specific screen.** Do not
request another generation segment on the strength of the original full-run
authorization alone. The eight-draw SGC screen found small added model-diversity
variance and no phase-consistent score benefit; it did not complete the full
coverage gate. Its predeclared negligible-effect rule did not fire because
other mean/covariance changes are measurable. Current allocation59422691 is
unchanged. See `e2e_cfm_mixture_futility_results_20261006.md` for the decision,
uncertainty and source-bound results. No automatic continuation is installed.

Full comparison now explicitly authorized by the user's request to set
interactive jobs off to complete the next comparison. Use bounded interactive
segments, at most two allocations. Four independent single-GPU workers justify
one full node; measured approximately50GPUh pre-scoring requires multiple
four-hour segments. Admit tasks only within remaining measured budget; preserve
all evidence, with no automatic failed/partial-task retry or batch submission.

First verify full-NFE seed17 d000 parity in each of the four phase/caps; only
then generate that cap's seed29 batches. Scientific scoring requires all four
parity passes and both complete32-draw banks. Reuse saved seed17 draws.
Immutable source/manifest and outputs: Scratch
`e2e_field_v3/cfm_model_mixture_20261006_v2`. Scoring remains pending. This
authorization supersedes the historical resource-approval boundaries below.

The v1 staging directory is retained after copying reference read-only file
modes prevented overlaying the extended sampler. No scientific task ran there.
The v2 stager makes only its new overlay destination writable before copying;
the original reference is untouched.

### First interactive segment launched

Allocation59422691, nid008292: one node/four GPUs, interactive QOS,
four-hour limit, desi_g, Scratch license. Four independent one-GPU Slurm tasks
share an exclusively claimed task queue. Workers stop admitting work after
13,500seconds minus the historical task estimate plus12%/validation margin;
the allocation shell exits after the step. Remaining tasks need another
authorized interactive segment; no unattended resubmission is installed.
The run is not yet a completed scientific comparison.

Manifest SHA256:
`732423f715435080d3ca85cf82cc3afa91171735fd68e0150e42ef4adbd9aef9`.
Fifteen focused contract/full-cap/canvas tests pass on the allocated compute
node. All four baseline tasks were claimed at launch. Logs are under the run
root's `logs/`; each task writes exclusive START/COMPLETE and parity receipts.
The source snapshot is checked against its manifest before/after every task.
Requested graphify refresh was blocked by execution safety review as too broad;
no bypass or graph refresh was performed. This does not change scientific input
or sampler bindings and is not a scientific validation pass.

Registered 2026-10-06 after the user approved one hypothesis-driven coverage
intervention with a matched unchanged control. This document registers the
scientific design. The subsequently approved shortened-trajectory compatibility
pilot is complete; see `e2e_cfm_model_mixture_pilot_20261006.md`. Full-length
scientific comparison remains pending.
No coverage improvement or fully runnable full-panel campaign is claimed.
B3 is deferred. No CIGALE mock painting, property estimator, new training,
production VAC modification or confirmation-phase access is included.

## Hypothesis and historical distinction

Test whether variation between independently trained seed17 and seed29 models
adds useful uncertainty or reduces conditional bias in the current posterior's
regional eigenvalue gaps and interior class fractions. This is an empirical
mixture intervention, not Bayesian model averaging or proof that epistemic
uncertainty causes the failures. Shared model bias may make it fail.

The July per-galaxy energy-score field generators and iterative generator
failed; September preservation auxiliaries failed their joint gates. Neither
is being relaunched. Stage C global shifts, uniform spread multiplication and
right-core pairing destruction did not consistently repair calibration.
This candidate preserves physical draws and their within-draw dependence.

The October 5 seed17/29 checkpoint comparison used the historical local+wide
sampler, not the current full-cap sampler. Those arrays are NOT eligible for
pooling with the current full-cap baseline. A compatible second component
must be generated or independently verified under the matched contract.

## Candidate and controls

- Candidate: q_mix(F|G) = 0.5 q_17(F|G) + 0.5 q_29(F|G).
- Primary unchanged control: q_17 with the current full-cap sampling contract.
- Diagnostic component control: q_29 alone. Improvement over seed17 alone is
  not evidence of a mixture benefit if seed29 alone explains the improvement.
- Both models use EMA coarse step13312 and fine step26624, the same normalizer,
  observation preparation, NFE128, completion/exterior method, tensor operator,
  core ownership, target and selection. Verify actual files and hashes before
  execution; do not choose a checkpoint using new evaluation scores.
- Retain original operator as primary and corrected operator as the matched
  secondary comparison, consistent with the completed Stage B/C experiment.
  No operator change is part of this intervention.
- M=32 draws in EVERY arm. Use addressed independent Bernoulli(0.5) component
  assignments frozen before scoring. Do not force 16/16 or search for a useful
  assignment: fixed-count sampling changes the iid assumptions of fair scores
  and rank nulls. Record the realized component counts.
- Select one component for an entire physical realization, including both
  coarse/fine factors, all windows and both cores. Share the selection across
  caps/regions at a phase/draw address; never independently select per voxel,
  class, factor or summary. Preserve the component's existing coherence scope;
  mixing does not establish additional cross-cap physical dependence.
- Use matching draw addresses across component controls where supported.
  Correlation between arms reduces comparison noise; it is not independence
  between physical cases. Never duplicate draws within an arm to fill a bank.

## Panel and immutable bindings

Only ph012/ph013, all32 registered pairs/64cores, unchanged Stage B galaxy
selection and physical full-box R7 truth. Preserve existing shell/cap/support
strata and sparse-stratum flags; do not discard difficult or empty strata.
Class threshold0.2 is primary; threshold0 is sensitivity only. Galaxy-weighted
fractions are primary and volume-weighted fractions secondary. Retain the
field-at-observed-coordinate definition, not P12-A host-halo truth.

Before scoring, bind config/source/checkpoint/normalizer hashes, observation
receipts, panel IDs, draw addresses, Bernoulli assignments, tensor conventions,
completion method, baseline receipts and output inventory. New baseline replay
must reconcile with the current baseline before attributing changes to models.
Do not replace existing arrays or overwrite prior scientific receipts.

## Frozen outcomes and decisions

Coverage levels: 50/68/90/95%. Use each scorer's actual attainable reference,
including discrete tie-aware fraction ranks; do not call 29/33 exactly90%.

Two co-primary discrepancies, separately in each development phase:

1. Maximum absolute coverage error over the four levels for the paired
   four-gap regional vector (two ordered eigenvalue gaps per core), against
   unchanged physical full-box truth, across the full registered panel.
2. Maximum absolute coverage error over the four levels for the paired
   galaxy-weighted class-fraction vector in the registered interior stratum
   (first three classes per core; the fourth is fixed by their sum).

Report the corresponding marginal width(lambda3-lambda1), gap, class and
trace/mass curves, posterior bias, widths, covariance, rank shape and Stage C
Q/S diagnostics. Covariance excess can include heterogeneous conditional bias;
do not interpret it automatically as a scalar spread deficit. Keep matched-
operator truth diagnostics separate from physical full-box truth results.

Advance only if BOTH discrepancies fall by at least20% in EACH development
phase, with no previously passing G1/G2 primary screen becoming a failure.
This is a deliberately joint investment gate for the two requested defects,
not population calibration certification. If a baseline discrepancy is already
within tolerance or unresolved by Monte Carlo precision, stop and report that
the proposed contrast is uninformative; do not substitute another objective.

G3: fair energy for both nominated vectors must improve in each phase; the
interior fraction energy uses the same interior cases as its coverage outcome.
Other primary scores (whole-panel paired fraction energy, width/gap CRPS,
density/eigenvalue CRPS and galaxy fair Brier) may worsen by no more than2%.
Report phasewise changes; do not pool away an opposite-sign phase result.
Near-zero score denominators require explicit review, not an automatic pass.

G0/G4 and the parent protocol's G1/G2 screens remain unchanged: input/null
checks, positive density, mass error<2e-6, solver/overlap consistency and
representative numerical sensitivity<=0.1 posterior SD for nondegenerate
quantities. Better coverage from indiscriminate broadening is insufficient.
Validate mixture scorers on an exchangeable synthetic mixture null, discrete
fractions, identical-component identity, biased components and destroyed
dependence controls before scientific scoring. These controls validate the
implementation, not the learned model.

Report q_mix versus BOTH component controls. If it only tracks the better
component, do not claim that missing between-model uncertainty was repaired.
No tuning weights, additional seeds, checkpoint sweeps or automatic retries.
Failure closes this candidate; a different intervention needs a new hypothesis.

## Replication and confirmation

Two component seeds are not independent confirmation: both define the
candidate. Neither are the many overlapping cores independent universes.
Report phasewise evidence and finite-draw sensitivity without voxel/draw-iid
confidence claims. If the investment gates pass, separately propose a frozen
replication/confirmation design with sufficient precision. No untouched-phase
payload is opened now; ph001 remains excluded. Population qualification may
remain inconclusive even when development point estimates pass.

## Compute boundary and next executable step

The user subsequently explicitly approved the technical pilot: one shared GPU
up to one hour and one CPU allocation up to30minutes. Reuse allocations when
compatible; do not allocate a separate CPU node if GPU-host resources suffice.
Full-panel generation and confirmation remain outside that authorization.

**Pre-execution feasibility amendment:** saved ph012 SGC NFE128 generation
takes2997seconds per four-draw batch, so the approved hour cannot accommodate
two full-length models in both phases. Before any pilot evaluation, restrict
this run to SGC in ph012/ph013, four draws each, NFE2, comparing the frozen
reference seed17, extended runner seed17, and extended runner seed29. SGC is
chosen as the smaller cap, not by outcome. The complete SGC canvas/windows
remain intact. This tests exact shortened-trajectory replay, common model and
input contracts, positive mass-conserving decoding of the first geometry crop,
memory and throughput. It does NOT test NFE128 parity, numerical convergence,
tidal operators or scientific coverage. Do not extrapolate its fields into
posterior evidence. Retain the full-length baseline timings for cost estimates.
The previously proposed one-case full-NFE pilot is superseded by this
budget-compatible engineering scope, not silently declared complete.

Allocation59422033 completed0:0 in10m05s and was released. No separate CPU
allocation was needed. All six shortened runs and14 focused tests pass;
124 independent receipt/invariant checks pass. No NFE128 seed29 draws, tidal
processing or scientific scores were generated. Exact replay at NFE2 is not
full-NFE replay or convergence evidence.

The measured historical NFE128 rates imply42.25GPUh for the second component's
32-draw generation over both caps/phases,2.08GPUh for tides and0.47h loading/
setup:44.80h before verification, corrected-operator replay and scoring. A
four-cap full-length baseline replay adds5.28h sampling alone. This is not
a cheap saved-draw test. Full-panel execution requires separate explicit
resource authorization and a completed immutable manifest. Existing historical
seed29 local outputs must not be used as a shortcut around compatibility.
