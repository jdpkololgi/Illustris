# Frozen VDM checkpoint, sampler and calibration decision experiment

User authorization2026-09-17: evaluate the current four literature-inspired
models before longer/diverser training; checkpoint512/2048/5120, sampler
convergence,100--256 draws on fixed held-out fields, density and especially
tidal eigenvalue/eigengap calibration. Restartable/chained compute authorized.

## Frozen estimand and panel

All four direct-density VDM fits from direct_vdm_20260916_v3, no weight updates.
Registered anchors ph003_NGC_s0_boundary_00 and ph003_SGC_s0_boundary_00 were
excluded from all ten fitting patches and normalization. Their128-fine-cell
source footprints are disjoint. They share one cosmological phase and have
already been inspected: development-heldout, NOT two independent universes or
a blind final test. No sealed ph001 access. Do not manufacture sample size from
correlated grid points. Fitting phases remain ph000/ph002.

Fields are48^3, voxel-averaged R7 density, cell6.766 Mpc/h. No additional R7
smoothing. Preserve all frozen target/conditioning transforms and per-run charts.
Produce128 draws per anchor per checkpoint per model at250 ancestral steps:
3072 fields. At final checkpoint produce32 additional draws per anchor/model
at each500 and1000steps:512 fields. Total3584. Same addressed random streams
across models/checkpoints. Initial noise and1000-grid Brownian innovations are
coupled; each coarser innovation is normalized sum of fine innovations and
retains the correct marginal Gaussian transition. Compare the first32 shared
draw identities; never compare unrelated stochastic paths as solver error.

Coverage at250steps is explicitly sampler-dependent. If refinement fails,
do NOT label that coverage a converged-model calibration result. Expand sampling
at the resolved step count only after the refinement result/resource decision.

## Tidal estimand and boundary controls

Use T_ij(k)=k_i k_j/k^2 delta(k), with isotropic DC mean/3 and no second smoothing.
Same operator on generated and true patches under periodic, zero-to2N, and
reflect-to2N closures. Evaluate central16^3 cells at stride2:512 spatially
correlated probes, ordered lambda1<=lambda2<=lambda3 and nonnegative gaps21/32.
Density and tensor trace must close. These are grid probes, not galaxy targets.

Also load the existing full-box-derived `tensor_spectral` reference from the
hash-bound ph003 source shards. Average the SIX tensor components over2^3 cells
BEFORE eigendecomposition, never average eigenvalues. Compare true patch closure
against this full-box reference to quantify missing-exterior/discretization
error. Evaluate posterior coverage against BOTH matched patch truth and full-box
truth, always separately. The latter includes boundary error; good matched-patch
coverage cannot certify physical full-volume tidal calibration. No oracle exterior
is supplied to generation or retrospectively used to correct generated draws.

## Readouts and decision logic

Per anchor/seed/checkpoint: Hann-window total-density power in the existing four
k bands, cross-correlation, means/variance/quantiles/CDF, regional density,
posterior mean error and spread. Pointwise50/68/90/95% central interval coverage,
randomized truth ranks, CRPS and interval widths for density, eigenvalues and
eigengaps; separate observed and unobserved probes. Keep two seeds separate.
Report Monte Carlo uncertainty of draw statistics, not binomial significance
from treating512 correlated probes as independent universes. Rank definition:
[Stan SBC guidance](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html).
This small fixed panel is a calibration diagnostic, NOT joint-field SBC.

Exploratory preregistered screens, not scientific release gates:

- Strong checkpoint progress: >=20% reduction in absolute log-power discrepancy
  from2048 to5120, examine CRPS/coverage alongside it and require seed consistency
  before a general conclusion. Trend disagreement means inconclusive.
- Plateau candidate: <=5% relative change in discrepancy; loss alone insufficient.
- Sampler refinement:500->1000 paired mean power change <=5% in every band;
  retain draw-paired Monte Carlo confidence intervals and interval-width changes.
- Statistics adequate but narrow/bias-shifted posteriors: conditioning/calibration
  issue, not an architecture ranking. Boundary-reference error is checked first.
- No architecture/objective change from immature/solver-unresolved evidence.

No automatic longer training occurs inside this frozen evaluation. A subsequent
training decision follows the evidence and user's authorized direction, with
phase/patch availability and exclusion geometry verified before adding data.

## Operations

`e2e_vdm_assessment stage --root <new vdm_assessment_ Scratch child>` freezes code,
all12 checkpoint hashes/bindings and panel metadata. `prepare`, then `smoke`, on
approved GPU compute verify raw cache/source shards, targets, full-size batched
sampling, scalar/batch parity and atomic chunk replay. `run --branch BRANCH`
produces independent addressed chunks; `--panel checkpoints|refinement|all`.
Rerunning the same branch verifies and skips committed chunks, preserving orphan
partial files. Per-branch single-writer lock; SIGUSR1/SIGTERM stops after current
chunk, exit75; no automatic retries. Original checkpoints and hash-bound training
sources remain unchanged. Frozen-source logs and receipts survive disconnection.

Use short interactive development/smoke compute first, then a bounded explicitly
authorized resumable evaluation schedule. Avoid an unattended login agent making
new scientific/resource decisions or open-ended allocation chaining.

### Authorized four-GPU interactive execution

User specifically prefers interactive chaining and all four node GPUs over the
proposed queued batch jobs. `e2e_vdm_assessment_interactive.py` freezes a separate
hash-bound deterministic launcher; two75-minute allocations maximum, four
independent GPU workers,10GPUh ceiling. Workers receive SIGUSR1 after70minutes
and finish their current atomic chunk; only verified exit75 resumes. Unexpected
errors, queue failure or exhausted budget stop, without batch fallback. All four
completed branches gate report generation. tmux holds the launcher across SSH
disconnects but does not extend Slurm limits. No agent is left making decisions.

Root: vdm_assessment_20260917_v2; scientific source4ff49d7.23 tests pass and
58464481 actual GPU smoke passes:43.78s/8draws/250steps, scalar/batch relativeRMS
9.90e-8. Actual pause/restart saved32 unique scientific draws with the first
receipt unchanged. SMOKE.json and RESTART_TEST.json bind the source manifest.
Report/calibration results are pending the complete assessment, not established
by these technical gates.
