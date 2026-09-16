# Preservation-objective isolation v1

Registered 2026-09-16 UTC. User requests implementation of a loss-level experiment,
not another unchanged extension. Full E2E stays paused. The implemented commands
support frozen-source, restartable execution; the full eight-fit matrix is NOT
automatically submitted by implementation or smoke tests.

**Authorized launch, 2026-09-16 03:18UTC:** submitted array58407655 (0-7%2)
and afterok CPU report58407656. Both initially pending, not yet a scientific
result. Session `e2e-preservation` on login39 retains the launch terminal;
`tmux attach -t e2e-preservation` reconnects there. Slurm owns training, so loss
of SSH or tmux does not stop it. Exact single-use launch commands are recorded in
[launch_tmux.sh](evidence/e2e_field_v2/preservation_objective_v1/launch_tmux.sh).
Job-ID receipts and logs are in the canonical staged root below. This authorized
submission supersedes the earlier implementation-only status, not its frozen
scientific design. Full E2E remains paused; no automatic retries.

**Validation complete:** 35 focused tests and all four full96^3 GPU smoke arms
pass. Model/Adam/RNG checkpoint replay is exact; the zero-weight control exactly
matches the previous training step. Finite endpoint gradients, 60 ordinary and
24 near-clean evaluation rows, and gate checks pass. Peak GPU memory is5.334GB.
Allocation58397904 was released successfully. Frozen implementation:
`344f65202515b7b5f632ec3506e215fae65d7d9f`; canonical staged root:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/preservation_objective_v1_20260916_58397904`.
Receipt: [VALIDATION.json](evidence/e2e_field_v2/preservation_objective_v1/VALIDATION.json).
This verifies implementation, not scientific efficacy. At validation time the
eight-fit training had not been launched. Signal handling reuses the previously tested durable
workflow; the new smoke checks same-process checkpoint replay, not a fresh
scheduler-interruption test.

## Evidence motivating this test

The completed clean-limit chain58385300 ->58385303 ->58385304 ran successfully.
Two15-field seeds reached24,576 updates; each then forked into unchanged versus
near-zero exposure to36,864.22 probe-file hashes and all completion receipts
verified. Summary SHA256:
`fcbc6f1ca447190ab7e36e59400dc1f70861be2d8208ff4d30e9c0faf2ae4737`.

Transfer clean RMS at nominal sigma=.05 (physical fine-residual units):

| Checkpoint / exposure | Seed0 | Seed1 |
| --- | ---: | ---: |
| 3,072 original | .00680295 | .00953133 |
| 24,576 optimized | .00336991 | .00544333 |
| 36,864 unchanged | .005612 | .00432896 |
| 36,864 near-zero | .0034987 | .00576567 |

Optimization reduced clean distortion50.5%/42.9% and noisy high-k error68.4%/78.8%.
At24,576 and all final branches, both .05/.2 noisy gates pass all15 fitted and12
transfer regions. These gates do NOT impose a clean-input preservation tolerance.
No branch meets joint last-interval stability. Near-zero exposure versus equal-
budget controls reduces paired .001 clean RMS59.3%/83.3%, but leaves about97% of
injected high-k noise. At .05 its clean effect is -37.0%/+32.6% by seed, while
noisy high-k error changes +23.9%/-15.0%: a tradeoff, not a uniform repair.
Final .05 clean RMS is roughly0.9--1.5% of field std and also occurs on fitted
regions. Exact0 identity was structural throughout; small-sigma RMS is roughly
linear in sigma. Observation associations are seed-sensitive and non-causal;
three development phases and correlated SGC regions cannot identify selection
as the dominant cause. No new-cosmology or posterior-calibration result.

## Design: change only the supervision

Fork the two **24,576-update optimization** checkpoints (not a selected winning
seed or the two different final-exposure checkpoints). Architecture, current
normalization, optimizer/Adam state, LR1e-4, clipping1, fields and condition inputs
remain fixed. Each arm receives6,144 updates; probes at0/1,536/3,072/6,144 additional
updates. Eight fits,49,152 new updates; no adaptive extension or weight sweep.
Use the same previous **near-zero primary exposure** in every arm, with identical
Gaussian tensors/field/global-update addresses. The control must replay that
previous objective exactly; near-zero exposure is not a confounded new treatment.

| Arm | Clean-anchor weight | Response weight |
| --- | ---: | ---: |
| control | 0 | 0 |
| identity_weak | .1 | 0 |
| identity_strong | 1 | 0 |
| identity_response | .1 | .1 |

Every arm executes the same four neural forwards and backpropagates the same
expression with different registered weights. No EMA, LR decay, model change,
per-field scaling, conditioned-data augmentation, transfer fitting or new data.
Separate auxiliary Gaussian/address and time draws do not consume the primary
RNG stream. The six-bin field traversal exposes every fitting region to every
auxiliary bin. Independent auxiliary sigma bins are log-uniform within
[1e-5,1e-4], [1e-4,.001], [.001,.005], [.005,.015], [.015,.03], [.03,.05].
The auxiliary objective never samples exactly0; the primary schedule retains
its existing0 atom/pure-noise endpoint unchanged. Log-time embedding floor.002
also remains unchanged. Record clipping and all unweighted losses separately.

## Objective and physical meaning

For the existing VP chart, a=(1+sigma^2)^(-1/2), b=sigma*a and
D(x,sigma)=a*x-b*v_theta(x,sigma,c). Let y be the normalized clean fine field.

The **primary denoising loss is unchanged**:

    L_den = mean[(v(a*y+b*epsilon,t,c) - (a*epsilon-b*y))^2].

At a separately sampled low-noise time, evaluate

    v0 = v(a*y,t,c)
    v+ = v(a*y+b*rho*eta,t,c),   v- = v(a*y-b*rho*eta,t,c), rho=.25
    v_even = (v+ + v-)/2,       v_odd = (v+ - v-)/2
    L_id   = mean[(v0+b*y)^2]
    L_even = mean[(v_even-v0)^2]
    L_odd  = mean[(v_odd-a*rho*eta)^2]
    L = L_den + lambda_id*L_id + lambda_response*(L_even+L_odd).

These are analytically the clean distortion, perturbation-even drift away from
the clean prediction, and retained perturbation-odd noise, each divided by b^2
in normalized field units. Multiplying by the fixed physical fine scale squared
converts them to physical units. The implementation uses the velocity identities
directly: no division by tiny b, no inverse state scaling, no teacher/stop-gradient.
Thus the loss distinguishes **leave y alone**, **do not introduce systematic
structure drift when slightly perturbed**, and **remove the perturbation**.
An identity-only predictor satisfies L_id but fails L_odd and the ordinary noisy
gate. Positive-noise clean preservation is an explicit inductive bias, not an
identity that the exact conditional Bayes denoiser must obey everywhere.

A simple .1 clean penalty already failed in the earlier raw-skip3,072-update
architecture comparison. L_id is algebraically that same kind of penalty, not
a novel formula. This experiment tests it on the established near-identity
skip/optimized parents, separates its exposure from primary time, and includes
the explicit even/odd response arm. Failure of the simple reference is retained,
not omitted. Auxiliary mismatched-noise inputs deliberately change supervision;
this is NOT guaranteed unbiased score matching or a posterior-preserving change.

## Evaluation and joint gate

Same15 NGC fitting and12 SGC transfer regions, same physical normalization and
paired evaluation noise, no ph001. These are three development phases, not27
independent cosmologies. Normalization retains the previous development-pool
provenance. No target-derived per-field normalizer is used at inference.

Retain the full dense clean sweep0 through.2, ordinary noisy ratios
.001/.005/.01/.025/.05/.2/1/5/20, and two evaluation noise replicates. Add direct
physical noisy MSE (not an unweighted sum of band powers). For near-clean response,
test rho=.1 and.5 (not the trained.25) at sigma=.001/.01/.05, both signs and both
noise replicates. Save clean error, perturbation-even drift, odd noise left, and
total perturbed-input MSE. Exact0 is a numerical endpoint check, not a learned win.

An arm passes the preregistered **local pilot gate** only if all conditions hold
for BOTH parent seeds, BOTH fitted/transfer groups, and BOTH final probe points
(+3,072 and+6,144), against the same seed/update control:

- At .001/.01/.05, median paired clean-RMS ratio<=.75 and at least75% of fields
  improve. At every other positive clean ratio, median ratio<=1.10.
- At every ordinary noisy ratio, median paired physical-MSE ratio<=1.05 and
  every phase median<=1.10. Same limits on perturbed-input MSE for every held-back
  near-clean fraction/time. Even/odd components remain separate diagnostics.
- Existing .05/.2 gate passes on EVERY field: high-k noise amplitude<=.2,
  high-k error<=.25 of the paired384 parent, lower-three-band gains in[.9,1.1].
- Finite outputs/gradients, no provenance/endpoint/schedule failure. A smaller
  clean loss cannot compensate for a failed noisy or signal-preservation gate.

These are conservative pilot decision thresholds, not universal physical
tolerances or a convergence proof. Report every arm/seed/checkpoint. If all fail,
retain the negative result; do not select the prettiest field or extend silently.
An independent third training-seed confirmation is required before recommending
E2E reintegration; it is a follow-on design, not part of this eight-fit budget.

## Return to the full field posterior: gated and separate

After reproducible local success and independent confirmation, freeze the recipe
and propose a separately authorized conditional DIFF canary: first true-coarse,
then sampled-coarse conditioning, matched draws/NFE, physical support/power,
cross-region coherence and conditional calibration/proper scores. Those tests
can still reject an apparently good denoiser. A diffusion velocity is NOT a CFM
velocity: CFM requires an explicit path/target conversion or separately trained
matched CFM objective plus numerical endpoint/sampler checks. Do not directly
reuse a diffusion head as CFM, and do not claim either posterior is calibrated
because this local loss passes. No P12-A changes, held-out opening or promotion.

## Implementation and execution

`e2e_preservation_loss.py`: pure loss and addressed auxiliary schedule.
`e2e_preservation_experiment.py`: stage, smoke, train, report. Source snapshots
are committed archives; runtime source/data/parent hashes checked. Native durable
checkpoints preserve Adam/RNG/history every512 updates and on USR1/TERM. Single
writer, fail-closed corruption handling, immutable probes and idempotent completed
training/report. Existing completed runs and hash-bound source remain unchanged.

Use cosmic_env with Python environment contamination unset. Stage a fresh
`preservation_*` directory beneath the registered Scratch pipeline root, then
run smoke from its source on one allocated GPU. The smoke checks every objective
arm at full96^3, exact resume, zero-weight control replay against the old training
step, and ordinary/near-clean evaluation. It does NOT establish scientific efficacy.

    python -m workflows.sbi.e2e_preservation_experiment stage --root <run>
    cd <run>/source
    python -m workflows.sbi.e2e_preservation_experiment smoke --root <run>

After a matching SMOKE.json passes and a full run is requested, use the reviewed
`submit_e2e_preservation.slurm` with an eight-task array0-7%2: four arms per seed,
oneGPU/task, shared/desi_g,32 logical CPUs,2h/task cap. Final CPU diagnostic uses
debug/desi,8 CPUs,10min, afterok of the entire array. Both declare Scratch and
disable automatic requeue; stdout/stderr belong under <run>/logs. The script
checks the smoke/source binding before work. Do not submit the same array twice;
inspect Slurm and branch receipts before any explicit resume. No batch is launched
by staging or smoke. Record actual IDs/commands when authorized and submitted.
