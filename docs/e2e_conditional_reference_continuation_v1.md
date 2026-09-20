# Gaussian reference learning curve: unchanged-training continuation

Registered 2026-09-19 before extended fitting. User requests a substantial
continuation of both fixed and amortised models before an affine learned control.
This supersedes the previous ordering of proposed diagnostic tests, not their
recorded results. No affine teacher, architecture, optimizer or loss change here.

## Design

Continue **all twelve** existing fits: VDM and CFM, seeds17/29, amortised and
fixed-observation0/1. Resume the exact4096 checkpoints, including Adam moments,
generator state and CPU/CUDA RNG states. Retain the original model, batch32,
learning rate3e-4, constant schedule, normalization, observations, fresh-data
generator and objective. No restart from scratch, learning-rate decay, clipping,
teacher supervision, best-checkpoint selection or auxiliary loss.

Log at8192/16384/32768/65536updates (16times the original final update), retaining
all checkpoints. Training samples remain fresh IID draws under the original
procedure: fixed-observation training is NOT repeated epochs over one finite
dataset. The experiment tests continued stochastic training/data exposure,
not optimization iterations isolated from the number of simulated examples.

Evaluate512draws per case at every saved checkpoint, including a new4096baseline:
VDM512/1024NFE, CFM128/256NFE. Do not compare the old VDM256NFE point directly
with a later1024NFE point and label the difference a training gain. Existing
CFM4096results supply a replay/evaluation sanity check. At65536 additionally
evaluate2048draws per case at the higher NFE, with recalibrated Monte Carlo null
thresholds. Amortised models retain all four observations; matched comparison
uses0/1. Exact-sampler and erased-covariance controls remain unchanged.

Primary metrics/tolerances remain those of the first reference: posterior mean,
voxel variance,16-probe covariance, octant coverage, with power by Fourier shell
reported separately. The larger final ensembles reduce uncertainty about a
borderline covariance pass. A reproducible reference pass requires both seeds
and both matched cases to pass at32768and65536, plus the final precision check.
Passing only the best checkpoint or an average across cases is insufficient.

For the stronger statement that under-training resolves the identified failures,
also require final posterior residual power ratios in[0.9,1.1] in every toy shell
for all matched cells. This is an additional prospectively declared diagnostic
condition, NOT a retroactive change to the original four-gate pass flag or a
DESI science tolerance. Report the amortised-only cases2/3 separately and do not
call the conditional learner qualified if they fail.

## Interpretation

- Errors fall into the reference range reproducibly: insufficient training was
  a material cause of the initial failure under this setup; not proof it is the
  only possible limitation or that the cosmological problem is solved.
- Errors keep falling but remain outside tolerance: training adequacy is still
  unresolved; do not declare an asymptote from two late checkpoints.
- Errors stabilize outside tolerance: persistent failure under this optimizer,
  objective and parameterization. This is NOT by itself proof of insufficient
  network representational capacity. An affine control/exact conditional-target
  diagnostic can then isolate further causes, after interpreting this curve.

Opus correctly prioritizes the missing learning curve. Its fixed-training
diversity caveat is also valid: posterior-only training narrows the distribution
of fields compared with prior-predictive amortised training. Consequently a worse
fixed covariance estimate cannot establish that amortisation improves covariance.
Nor do the toy oracle numbers quantitatively explain the earlier Abacus
corrector experiments: their target, sampler and covariance norms differ.
The8^3objective comparison is useful, but is not a substitute for the later
coupled-field comparison at cosmological scale. No publication or convergence
claim follows merely from the current table.

## Resource and provenance contract

Proposed allocation: one four-A100 interactive node,90minutes (6GPUh hard cap),
<=20GiB new Scratch, independent single-GPU workers and no inter-GPU training
communication. At most two total interactive allocations; use tmux to preserve
the finite shell, with Slurm enforcing the limit. No successor or resubmission.
The measured original update cost is about0.011s; each worker owns three fits,
so training-only cost is approximately34minutes per worker before evaluation,
staging and margin. Full-node concurrency still requires a technical smoke.

The user explicitly approved this four-GPU/90-minute/20GiB continuation on
2026-09-19. Source and parent checkpoints are copied
to a new hash-bound run tree; the original4096run and archived receipts remain
untouched. Parent hashes, all immutable training settings and imported checkpoint
update are verified fail-closed. Each worker checkpoints atomically and has an
85-minute deadline. Parent and worker completion records distinguish successful
execution from scientific gate success. Affine, nonlinear and Abacus fits remain
outside this run.

## Technical preflight record

Source commit5660b29, frozen run tree
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/conditional_reference_continuation_20260919_v1`.
Allocation58596902 stopped after17s because all four initial GPU replay checks
failed before scientific fitting. No continuation checkpoints were published.
The original4096results were not altered. A mature-checkpoint CPU replay is
exact, but GPU nondeterminism remains a hypothesis until measured explicitly.

The proposed retry uses a separate `_r1` run tree, records restored-state equality
and numeric replay differences, and provides a deterministic-CUDA *smoke-only*
control. Actual training retains the parent's backend settings. No equality
tolerance is relaxed. One89-minute/four-GPU retry has been requested; combined
with the first17seconds it remains below the original6GPU-hour cap. Await
specific approval; do not silently resubmit. The failed source remains frozen.

**Subsequent authorization and resolution:** the user then explicitly said to
forget the budgets and complete the task. Allocation58597933 requests four A100s
for two hours and remains in tmux during diagnosis and training. This supersedes
the pending-budget blocker above, not the frozen scientific settings. In `_r1`,
default arithmetic repeats loss/data/state exactly but differs in parameter
updates; deterministic smoke repeats all12fits with zero parameter difference.
This isolates GPU nondeterminism. Actual continuation keeps the original default
backend. The frozen85-minute worker guard remains in place; if reached, preserved
checkpoints can be resumed under the user's new completion authority. No new
scientific configuration or Abacus campaign is authorized by this execution fix.

## Completion,2026-09-20

All12fits reached65536updates;240curve ensembles and24precision ensembles pass
execution/integrity checks. Job58597933 completed0:0 in1h11m59s, scientific step
1h06m51s; allocation released.23focused tests pass. Full results and the distinction
between endpoint success, across-checkpoint qualification and power failure are in
[the closeout](e2e_conditional_reference_continuation_conclusions_20260920.md).
The continuation materially improves the models but does not promote the nonlinear
or coupled campaign. No affine/control training was included.
