# Gaussian reference: target, optimization and representation controls

Registered 2026-09-22 before fitting. User authorizes planning and execution.
No Abacus, nonlinear posterior or P12 training is authorized by this protocol.

## Corrections to interpretation

The September 20 progression entry's claims of a permanent CFM spectral floor
and an exclusion of under-training as the cause of spectral errors are too strong.
Three checkpoint slopes do not identify an asymptote. Proper scores and power
can decouple; power need not worsen monotonically. A persistent relative
amortisation penalty does not establish an irreducible conditional-map error.
The covariance diagnostic uses 16 probes, not the entire 512-dimensional field.
Use draw-count-matched nulls. Fixed CFM's 0.067 final covariance error used 2048
draws, whereas the roughly 0.18 null quoted elsewhere used 512 draws.

## Stage 1: matched CFM intervention

Reuse all six CFM parents at update 65536 (seeds 17/29; fixed observations 0/1
and amortised). Restore identical model, Adam and random-stream state for each
of three branches; run 16384 additional updates, evaluating at +8192/+16384:

1. `baseline`: unchanged stochastic targets, constant learning rate 3e-4.
2. `decay`: identical stochastic targets; cosine learning rate 3e-4 to 3e-6.
3. `teacher`: same U-Net, constant 3e-4; replace the stochastic bridge velocity
   target by E[v | z_t,y,t], computed from the exact Gaussian posterior.

Both target branches draw the SAME x, noise and times. The teacher is privileged
training supervision ONLY; exact means/covariances are not network conditions.
The population minimizer is unchanged. Adam history is intentionally retained
in all three branches; effects include adaptation to the intervention.

Fit a separate Gaussian-capable affine control from scratch on both stochastic
and exact targets, with the same two seeds and three fitting modes, 16384 steps.
Its deliberately privileged, fixed posterior eigenvectors define coordinates;
posterior eigenvalues and the dense observation-to-mean map (or fixed mean) are
LEARNED, never initialized from their true values. The time dependence is the
analytic Gaussian bridge form. This is an existence/optimization control, NOT
an equal-information architecture contest or a deployable DESI model. A pass
cannot alone localize failure to U-Net capacity rather than parameterization.

## Measurements and decisions

Use 512 draws at intermediate checkpoints, 2048 final draws, Heun 128/256 paired
NFE at the endpoint, all four observations for amortised fits, one for fixed.
Preserve gates: normalized mean error <=0.1 (or matched null p99), 16-probe
covariance error <=0.15 (or null p99), variance ratio [0.9,1.1], octant coverage
[0.85,0.95]. Additionally require EVERY shell residual-power ratio [0.9,1.1].
Report every seed/case; no endpoint or seed selection. Intermediate 512-draw
results describe trajectories, not a precision-matched endpoint comparison.
Also measure frozen-model velocity excess MSE on fresh bridge examples with
exact targets; raw stochastic training losses are not comparable across targets.

Teacher helps: target-noise optimization is implicated, not proved exclusive.
Decay helps: constant-step optimization contributes. Affine passes but U-Net
fails: Gaussian posterior learnability established under privileged structure;
U-Net representation/time conditioning/optimization remain candidate causes.
No branch passes: do not infer impossibility from this finite run.

## Stage 2: controlled resolution test

After Stage 1, compare 8^3 and 16^3 at fixed physical volume, physical prior and
observation information. Do NOT use the old `prior(n)` unmodified: its unit-voxel
normalization changes physical Fourier amplitudes with resolution. Use a common
continuous periodic prior discretized on nested grids and identical observed
locations/noise, with independently verified posterior references. Record
physical receptive-field changes. Compare common physical k and k/k_Nyquist;
newly resolved modes are reported separately. Freeze this detailed construction
and resource estimate before fitting, informed by Stage 1's failure pattern.

### Frozen resolution construction (before resolution fitting)

The prepared reference uses one continuous periodic Fourier series with integer
wavevectors in [-8,8]^3, coefficients proportional to
`[1+(|k|/1.5)^2]^-2`, normalized ONCE. Each grid sums its aliases, so covariance
on the nested 8^3 sites is identical. Both grids see exactly the same noisy
observations at the original observed sites. Independent Kalman identities and
nested posterior mean/covariance equality pass during preparation. The native
shells have different alias content: compare subsampled 16^3 draws against the
8^3 posterior as the identical-estimand test; report native shells separately.

Eight fresh exact-target CFM fits: grids 8/16, seeds 17/29, fixed observation 0
and amortised conditioning, 16384 updates, unchanged base8/levels2 U-Net and
Adam 3e-4. This isolates resolution under a common physical inference problem,
but not a single architecture mechanism: physical receptive field, condition
sparsity, target dimension and cost per update change. No direct quantitative
comparison to the original differently normalized 8^3 reference is warranted.
The 16^3 smoke measured 0.0354 seconds/update; four independent GPUs allow two
fits each with adequate evaluation time in a 90-minute allocation. Exact CFM
128/256-NFE covariance errors are below 1% on both prepared grids. Final
2048-draw native and common-observable diagnostics remain required.

### Executor reconciliation

At takeover, login39 still had a duplicate agent controlling the same task.
Per the user's explicit request, its app-server PID670715 and code-mode host
PID701449 were suspended with SIGSTOP, not killed. Slurm jobs58772775 (target
controls) and58773114 (completed technical/reference checks) were preserved.
This login34 session now owns execution. Do not resume the duplicate executor
while this task is active. No scientific source snapshot was changed.

## Adaptive affine convergence check (registered before extension)

Early endpoint results show the fresh affine fixed fits pass, while the
amortised affine exact-target fits retain about16-19%highest-shell excess after
16384steps. They receive roughly half as many examples per template. Before
interpreting this as a conditioning/representation limit, continue all FOUR
amortised affine fits (both targets, both seeds) unchanged to65536steps, in a
separate source-bound output tree. Preserve16384results and report the extension
as adaptive, not part of the original endpoint comparison. Use32768/65536curves,
final2048draws at256NFE, and learned continuous-time moments to separate Monte
Carlo/sampler error. No further U-Net training or model selection follows.

Resolution construction frozen before its fits: a unit periodic box with fixed
continuous Fourier series k_i=-8,...,8 and S(k) proportional to
[1+(|k|/1.5)^2]^-2, normalized ONCE to point variance one. Aliases are summed
when evaluating on each grid. Identical 8^3 observation locations, masks, noise
and four realized observations serve both grids. At16^3 only nested even sites
are observed: there is no increase in observation count. Verify the nested prior
AND posterior mean/covariance identities numerically before fitting. This is a
new matched resolution problem, not a continuation of the original 8^3 prior.
Train exact-target U-Nets from scratch at each resolution, both seeds, fixed
observation0 and amortised (8fits total),16384updates with8192intermediate draws.
The U-Net parameters/time embedding are identical; physical receptive fields
and sampling representation change. Report native spectra AND the SAME 8^3
subsampled field functionals; native Fourier shells have different aliasing and
must not be mistaken for identical continuum modes. These are finite-budget
diagnostics, not evidence of a resolution-dependent irreducible floor.

## Execution

Use approved interactive compute, at most two allocations; technical smoke first,
then independent fits on four GPUs if throughput warrants. Durable checkpoints
and frozen source/config hashes; no login-node fitting. Stop on nonfinite values,
ancestry mismatch or missing cells. Record scheduler and artifact receipts.
The nonlinear lognormal/Poisson rung remains gated on reproducible Gaussian
success and would require a demonstrably converged numerical reference.
