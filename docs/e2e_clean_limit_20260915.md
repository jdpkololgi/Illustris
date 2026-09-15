# Fixed-field optimization and the clean-input limit

Registered 2026-09-15, following the completed diversity/normalization matrix.
User authorizes ordered follow-ons that survive SSH disconnects. Full E2E remains
paused at384; no held-out access, promotion, or automatic extension.

## Scientific controls

1. **Optimization:** resume both existing15-field/current-normalization checkpoints
   from3,072 to24,576 total updates, including AdamW/RNG. Same model, learning rate,
   clipping, v-MSE, physical corruption, conditioning, fields, Gaussian noise and
   normalization. Probe at3,072/6,144/12,288/18,432/24,576.
2. **Near-zero coverage:** fork each optimized parent into unchanged-exposure
   control and near-zero exposure,12,288 additional updates each (36,864 total).
   Probe at24,576/30,720/36,864. Only the first of six exposure bins changes:
   probability1/32 at exactly0; otherwise half uniform[0,.05], half log-uniform
   [1e-5,.05]. Second original[.025,.1] bin, higher bins and pure-noise endpoint
   every64 updates unchanged. Same Gaussian tensors/fields at each global update.
   No clean penalty, dropout, normalization, embedding or architecture change.
3. **Selection/conditioning:** only after all six branches complete, report paired
   physical metrics, target/observation associations, within-phase correlations,
   and fixed-ridge leave-one-phase-out predictions. Compare intercept, target
   mean/std, target distribution, observation features and combined predictors.
   Feature centering/scaling uses each regression fold's fit fields only. This
   is descriptive diagnosis, NOT causal isolation or selection-aware retraining.

92,160 new updates total, plus bounded smoke/replay overhead. Same15 NGC fitting
regions and12 SGC development transfer regions, in ph000/ph002/ph003; no ph001.
Stored target footprints exclude fit/transfer overlap; two transfer pairs and
wide conditioning footprints may overlap. Three phases, not27 independent
cosmologies. Current normalization deliberately retains its original development-
pool provenance; this run does not make that scaler strictly train-only.

## Estimands and stop gates

At ratio sigma, VP uses a=1/sqrt(1+sigma^2), b=sigma*a. Zero-injected-noise input
is x=a*y, not unscaled y. Physical Delta_clean=a*x-b*v-y. Clean ratios:
0,1e-5,3e-5,1e-4,3e-4,.001,.003,.005,.01,.025,.05,.2. Report RMS, signed bias,
max error, RMS/field std, RMS/nominal noise, spectral gains/errors, and per-field
small-sigma log slopes/monotonicity. No per-field statistics enter the predictor.
Noisy ratios .001,.005,.01,.025,.05,.2,1,5,20, with two paired evaluation seeds.
Preserve .05/.2 gate: high-k noise amplitude<=.2, error<=.25 of paired384 parent,
and lower-three-band gains in[.9,1.1], per field. Higher-noise guards are diagnostic.

Exact0 identity is STRUCTURAL, not learned. At0 the unchanged v-MSE has an
independent Gaussian target unidentifiable from the clean input (conditional
mean0); this stochastic endpoint loss is not identifiable denoising. The existing
log-noise embedding floor.002 stays fixed; raw time/sine/cosine still vary below
it. Failure of coverage alone can leave parameterization unresolved. At positive
noise a Bayes denoiser need not preserve every clean input exactly; the limit/rate
and accompanying signal/noise metrics are the relevant checks.

The report labels<=5% paired clean-RMS and noisy-high-k-error changes in the final
interval on both fitted/transfer fields as descriptive last-interval stability,
NOT convergence. Inspect both late optimization intervals, both seeds, field-level
changes, gains, broad-noise regressions and clipping. A stable biased plateau
is not success. No outcome automatically restarts E2E or requests more compute.

## Disconnect and restart contract

Committed source is archived into a unique Scratch clean_limit_*/source directory.
MANIFEST.json binds archived files, configuration, parent checkpoints, prepared
data receipt and reference diagnostics. Original experiments remain unchanged.
Jobs import the snapshot, not a mutable HOME checkout.

First pass unit tests and one-GPU interactive smoke: full96^3 gradients, parent
prediction parity, exact optimizer/RNG resume, endpoints and throughput. Two
actual runner eight-update SIGUSR1 tests deliberately exit75, followed by exact
comparison to uninterrupted16-update continuation. Those valid16 updates remain
the beginning of optimization seed0; the batch resumes them normally.

| Stage | Resources | Bound | Dependency |
| --- | --- | --- | --- |
| Optimization array0-1%2 | oneGPU/task, shared, desi_g | 3h/task | none |
| Coverage array0-3%2 | oneGPU/task, shared, desi_g | 2h/task | afterok: entire optimization array |
| Analysis | 8 logical CPUs on oneCPU node, debug, desi | 10min | afterok: entire coverage array |

At most two training GPUs concurrently,14GPU-hour reservation cap plus smoke.
The report is a short read-only diagnostic, not training; CPU debug was selected
after shared-CPU dry-run queue estimates extended into October. Queue forecasts
are provisional, and even debug may wait. No GPU training uses debug QOS.
Slurm owns the chain independently of SSH/desktop/agent connection. Queue waiting
time is separate. Checkpoint every512 updates and on USR1/TERM: weights, Adam,
all RNG, history, binding and checksums. Files/directories flushed before atomic
LATEST publication. Single-writer lock; immutable probes bind their checkpoints.
Incomplete generations preserved but ignored; corruption fails closed.

Slurm sends USR1 180s before walltime; finish current operation, checkpoint, exit75.
A node failure may lose work since last checkpoint. --no-requeue: no automatic
retries. Failed upstream jobs block afterok dependencies; dependent jobs may remain
pending until reviewed. SSH persistence is not immunity to walltime, hardware,
filesystem purge or outages. Scratch is not a backup; science records stay in Git.
Submission creates an exclusive intent before Slurm calls and records each ID.
Interrupted submission requires inspecting Slurm before manual partial recovery;
never delete the intent and blindly resubmit. Completed branches validate/no-op.

## Commands

Use cosmic_env, unset PYTHONPATH/PYTHONHOME/PYTHONUSERBASE/LD_PRELOAD, and set
PYTHONNOUSERSITE=1. Heavy actions only inside compute allocation.

```
python -m workflows.sbi.e2e_clean_limit_launch stage --root <new Scratch run>
# From the staged source, in one-GPU compute steps:
python -m workflows.sbi.e2e_clean_limit smoke --root <run>
python -m workflows.sbi.e2e_clean_limit train --root <run> --smoke-interrupt-at 3080
python -m workflows.sbi.e2e_clean_limit train --root <run> --smoke-interrupt-at 3088
python -m workflows.sbi.e2e_clean_limit resume-check --root <run>
# After passing gates, authorized login-node submission from frozen source:
python -m workflows.sbi.e2e_clean_limit_launch submit --root <run>
```

Recovery: inspect squeue/sacct, COMPLETE/LATEST, logs and hashes. Normal train
--root ... --replica ... --arm ... resumes committed LATEST in a newly authorized
allocation/batch; dependencies must be repaired explicitly. No live SSH needed.
Final artifact: <run>/analysis/SUMMARY.json, written only after source/checkpoint/
schedule/probe validation. Slurm COMPLETED is not scientific success. Summarize
final results into SCIENCE_LOG after completion; batch never edits the checkout.
