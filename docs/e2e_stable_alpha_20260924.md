# Bounded development batch: stabilized alpha and downstream sensitivity

## Decision and stopping rule

User authorizes multiple experiments, not another serial one-knob loop. This
batch tests a known lever and independently measures use-case sensitivity.
One four-A100 interactive allocation, at most two hours, restartable fixed
tmux launcher; no autonomous retries, extra allocations or Abacus training.
Stop alpha/optimizer tuning after this batch regardless of outcome. If a
stochastic configuration passes across both seeds/cases/checkpoints, it is a
candidate for a separately approved fresh-observation confirmation, not a
qualified posterior. If not, report the remaining mechanism and decide between
a structural/classical baseline and closing this toy tuning effort. Do not
automatically extend updates or narrow alpha in another sweep.

Scientific value is a changed decision, a ruled-out failure mechanism, or a
reusable diagnostic. Falling loss without those does not justify more agent
tokens or compute. Larger neural campaigns require accuracy on intended
functionals plus a cost/accuracy advantage over the classical reference on a
nonlinear rung. A Gaussian neural model cannot outperform its exact oracle.

## Experiment A: stabilized alpha re-selection

Sixteen AMORTISED CFM fits: alpha={.25,.30,.35,.40}, exact/stochastic targets,
seeds={17,29}. All start fresh from matched seed-specific weights and data/bridge
streams, using unchanged train-only prior power. No unfair mix of aged parents;
no change of alpha inside a fitted coordinate system. Train 98,304 updates,
batch32, Adam3e-4 for first65,536 then cosine to3e-6 for remaining32,768.
EMA beta=.999 starts with model initialization and never affects gradients.
This differs from the prior experiment's EMA initialization at65,536; the
within-batch comparisons, not historical equality, are the causal controls.

Predict alpha .30-.35 improves top-shell learning without restoring excessive
DC power, but test .40 rather than assuming the balance point. Check exact
oracle integration for every alpha/template at256NFE before launch (1% ceiling).
Evaluate raw and EMA at81,920/98,304 with2048draws,256NFE on four DEVELOPMENT
observations; add128NFE at final. Total384ensembles. Reuse per-shell, DC
offset/scatter, bridge absorption, velocity-risk and nonspectral diagnostics.
Checkpoint raw/EMA/Adam/random state every1024 updates. Technical deterministic
replay only; scientific kernel settings stay unchanged.

Select using STOCHASTIC EMA only, minimax normalized existing gate violation
across both checkpoints, both seeds and all four cases (16cells); tie-break
average then smaller alpha. Exact targets diagnose mechanism, never select the
practical model. Keep every original gate, including every unsmoothed shell.
Passing an average or only one checkpoint is not stable success. No reused
historical confirmation case is described as sealed. This run opens none.

## Experiment B: frozen-posterior use-case sensitivity (no training)

Before A completes, evaluate prior stabilization final decay+EMA outputs and
independent exact Gaussian ensembles on the same four development observations.
Controlled alternatives preserve exact conditional means: inflate only the
highest Fourier shell by13% in variance; inflate ALL modes by13%; erase
covariance while preserving voxel variances. Oracle-null and controlled spectral
arms share residual realizations, so their difference isolates the intervention.
An independent oracle ensemble supplies the nonlinear reference/noise floor.

Measure Gaussian-smoothed fields at R={0,.5,1,2} GRID CELLS; non-overlapping
regional mean densities at widths2,4,8cells; and threshold-zero tidal-class
probabilities (number of positive tidal eigenvalues). Define tidal multiplier
k_i*k_j/k^2, with isotropic DC delta_mean/3, and verify its trace equals the
smoothed field. This is a stated periodic toy convention, not a production
T-Web replacement. Include exact Gaussian 90% interval coverage for linear
summaries, variance ratios and mean errors; nonlinear probability RMS/max error
must be compared with independent2048-draw oracle Monte Carlo discrepancies.
These are conditional distribution comparisons, not survey-wide empirical SBC.

The toy has no physical box size. It CANNOT assert that R=1cell is7Mpc/h, or
that a small smoothed error justifies ignoring a failed full-field gate. No
post-hoc tolerance relaxation. The assay determines whether smoothing reduces
the measured error and whether regional summaries expose dependencies missed
by marginals. Actual DESI requirements still need physical voxel spacing,
smoothing convention, tidal threshold and an agreed allowed change in derived
probabilities before confirmation. Keep original gates until then.

## Evidence and literature boundaries

Stabilization job58818101 completed8fits/192ensembles. At98,304updates,
stochastic constant/raw -> decay/EMA: mean .10167->.08189,
16-probe covariance .09842->.08542, DC1.09176->1.03008,
top1.12987->1.13220. Exact decay/EMA top1.11306. No stable full-gate pass.
EMA strongly reduced coherent offsets, but two observations/template cannot
prove that conditioning is learned generally or all DC uncertainty is solved.
Posterior variance share is not a measurement of loss curvature.

- Karras et al., https://arxiv.org/abs/2206.00364: separating preconditioning,
  training and sampler design materially improves generative performance;
  image-quality results do not establish conditional scientific calibration.
- Doeser & Jasche, https://arxiv.org/abs/2606.10023: comparison against HMC
  finds that accurate means/marginals/cross-correlations need not imply correct
  field uncertainty geometry. This supports joint diagnostics, not the claim
  that all conditional models fail or that our particular error is inevitable.

These papers justify disciplined validation and modular controls. They do not
justify indefinite toy tuning, numerical DESI timelines, or a universal alpha.
