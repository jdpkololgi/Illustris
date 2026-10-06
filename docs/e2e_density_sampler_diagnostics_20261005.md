# Density-only tidal reconstruction and sampler-factor diagnosis

## Scope and decision

User-authorized follow-up to the October 5 closure and trace/gap diagnostics.
Only reused development phases ph012/ph013 are accessed. Ph001 and fresh
ph018/ph019 remain sealed for this investigation. Existing production source,
selected checkpoints, physical target and score definitions are unchanged.

There are two independent phase realizations. The 32 pairs / 64 cores and the
overlapping production windows are correlated summaries, not independent trials.
No new fitting, posterior qualification, target redefinition or website change
is part of these tests. Oracle controls are explicitly privileged diagnostics.

## Density-only reconstruction

Inputs are the saved 6.766 Mpc/h density cells, not the underlying 3.383 Mpc/h
true subcells. Within each parent cell, evaluate constant, central-linear, and
monotonicity-limited central-linear reconstructions at eight subcell centers.
Linear slopes come from adjacent parent-cell differences; the MC limiter uses
the minimum magnitude of the central slope and twice each one-sided slope when
the one-sided differences agree in sign, and zero otherwise. A common
per-parent slope multiplier ensures positive subcells without changing their
average. No fitted coefficient, cosmological truth or posterior outcome enters
the reconstruction formula.

Apply the existing exact free-space cuboid operator at each resolution and
average fine tensors, not eigenvalues, onto the original cells. Add the local
fine-minus-coarse tensor increment to the corresponding original full-cap
tensor. This preserves the distant contribution and trace. The initial four
geometry-selected pairs use collars of 16 and 32 parent cells; the expanded
32-pair check uses the same fixed methods at collar 32. All comparisons retain
the original full-box physical target.

The constant-subcell control also changes target quadrature: it tests averaging
the fine target samples rather than evaluating only at the parent center, even
without reconstructing a density gradient. Its effect must not be described as
recovery of genuinely unresolved density modes.

Completed true-coarse results across all 64 cores:

| Method | Mean width deficit against physical truth | Original deficit recovered |
| --- | ---: | ---: |
| Original operator | 3.871% | 0% |
| Constant subcells | 3.548% | 8.36% |
| Central-linear, positive and mass-conserving | 1.366% | 64.71% |
| MC-limited linear, positive and mass-conserving | 1.540% | 60.23% |

Each reconstruction improves every tested core. For central-linear, recovery
ranges from 62.74% to 67.28%; phase-specific mean deficits are 1.341% and
1.391%. The earlier true-subcell oracle recovered 73–75% on eight cores, so
the density-only method does not fully reproduce that oracle or close the
physical target. No target-at-model-resolution fallback has been adopted.

The fixed central-linear method is additionally applied to 32 unchanged
posterior draws in each of the same four geometry-selected pairs. This bounded
application uses only each draw's own completed density, not the true-density
increments. Replay the original cap tensor, check saved-eigenvalue parity, and
compare corrected draws with both the original full-box truth and identically
reconstructed true-coarse density. This is not a production implementation or
new full-cap inference campaign.

That posterior replay completed all four pairs / eight cores / 32 draws. On
this subset only, the physical width deficit falls from 5.885% to 3.463%; mean
width SD rises slightly from 0.033243 to 0.033853. Phase-specific deficits are
2.706% to 0.219% and 8.813% to 6.451%. Against identically reconstructed true
coarse density, the corrected posterior's pooled deficit is 2.146%, with
phase-specific signed deficits -1.149% and 5.180%. The operator improvement
therefore transfers to posterior draws without removing posterior error.
This subset must not be compared directly with the earlier all-64-core 7.664%
physical deficit or used to claim an all-core coverage repair. Its width
marginal coverage is unchanged at 87.5%, based on only eight correlated cores.
Maximum saved-eigenvalue replay error is 3.49e-10, corrected trace error
8.78e-15, and independent truth-reconstruction CPU/GPU discrepancy 1.67e-16.

## Matched sampler controls

For each registered pair, choose the closest existing production window that
contains both cores, using geometry only. Extract its actual observation
channels and retain the window index and physical origins. All selected windows
have complete raw-canvas coverage. Read the exact coarse and fine initial noise
from the original seed/shape convention and crop at the same production-window
location. Fixed seed-17 coarse-13,312 / fine-26,624 EMA weights, normalizer,
TF32 setting, Heun NFE128 and 32 draw addresses are shared by all arms.

| Arm | Coarse density | Fine density / sampling |
| --- | --- | --- |
| `canvas_raw` | Saved full-cap coarse draw | Saved raw canvas density |
| `canvas_completed` | Same saved coarse draw | Saved completed density in the window |
| `canvas_coarse_local_fine` | Same saved coarse draw | Single-window fine solve, original noise |
| `local_both` | Single-window coarse solve, original noise | Single-window fine solve, original noise |
| `oracle_coarse_local_fine` | True coarse density | Single-window fine solve, original noise |

All five arms use one identical local-plus-wide tidal operator and matched
true density on identical cells. The coarse contribution is the spectral tide
of a replicated 48-cubed wide field, and the local residual contribution uses
the same historical rectangular spectral operator. This intentionally removes
the previous local/full-cap operator confound, but it is not evaluation with
the production full-cap free-space operator. Original full-box truth remains
visible alongside the matched diagnostic target.

The saved-coarse single-fine arm isolates the fine consensus/context change
conditional on the same realized coarse field. Local-both additionally changes
the coarse sampler. Full-canvas windows exchange surrounding observational
information as well as velocities, so an effect cannot be assigned purely to
velocity averaging. The oracle intervention tests coarse-conditioning sensitivity;
it does not establish that every remaining error belongs uniquely to one factor,
nor assess an implementable conditional posterior with oracle information.

Oracle regional trace is structurally almost deterministic. FP32 log-density
conditioning produces tiny quantization offsets against FP64 true density.
Raw per-case trace z-scores/ranks divide those offsets by near-zero spread and
are not meaningful coverage diagnostics. The reviewed summary suppresses those
values, retains the absolute residual and explicitly labels conditioned mass.
All other arms' trace and all arms' gap/width statistics retain the original
finite-32-draw scoring convention.

## Scale-resolved and numerical checks

Separate the coarse tensor and fine-residual tensor before eigendecomposition.
Record their traceless Frobenius RMS, regional tensor means and fine-residual
density variance. Eigenvalue widths are nonlinear and are not additive budgets.

Replay the saved diagnostic density fields for coarse and fine-residual
rectangular pseudo-spectra. Coarse bins have edges
0, 0.01, 0.02, 0.04, 0.08, 0.16, infinity in h/Mpc; fine-residual bins have
edges 0, 0.04, 0.08, 0.16, 0.32, infinity. Exclude the DC mode from these bins.
Retain truth power, draw power, posterior-mean power, unbiased posterior
variance power and squared posterior-mean error minus finite-draw mean noise.
This is a windowed diagnostic of the model's own decomposition, not proof of
an isolated global Fourier-mode defect: the finite windows and decomposition
mix spatial frequencies. Oracle-coarse variance ratios are suppressed.

Five focused tests check constant reconstruction, positive mass conservation
under large contrasts, exact interior linear reconstruction, and the shear
metric's diagonal/off-diagonal weighting, plus an analytic Fourier wave with
antithetic perturbations to check power, variance and finite-draw subtraction.
An independent SciPy CPU calculation
on a shifted production window agrees with the GPU mean eigenvalues within
5.56e-17; its core trace agrees with the original full-box target within1.12e-16.
Preparation verifies all saved source payload hashes and parent-cell alignment.

The first four-draw engineering smoke generated valid fields but its report
incorrectly invoked a 32-draw interval helper. Its outputs were preserved; the
corrected smoke writes only four-draw moments, passes, and precedes the full
32-draw runs. No failed smoke result is used as scientific coverage evidence.

## Completed sampler and scale results

All 32 pairs, five arms and 32 addressed draws completed with frozen seed-17
checkpoints. The operator-matched mean widths are 0.466598 for ph012 and
0.471175 for ph013 (pooled 0.468887). These diagnostic means must not be
substituted into the production free-space operator's earlier width budget.

| Arm | Width deficit ph012 | Width deficit ph013 | Pooled deficit | Pooled width SD |
| --- | ---: | ---: | ---: | ---: |
| Raw/completed canvas | 3.904% | 3.942% | 3.923% | 0.028618 |
| Canvas coarse + local fine | 3.018% | 3.281% | 3.150% | 0.028837 |
| Local coarse + local fine | 2.601% | 2.649% | 2.625% | 0.028679 |
| True coarse + local fine, oracle | 0.247% | 0.330% | 0.289% | 0.007357 |

Raw and completed canvas results are identical in every selected window.
Changing only the fine solve recovers approximately 20% of the matched mean
width deficit; changing both solves recovers approximately 33%. This is a
replicated sampler/context contribution, not evidence that velocity averaging
alone causes the entire deficit. The remaining 2.625% is not an operator
resolution mismatch in this common-operator comparison.

The pooled traceless tensor RMS ratios to truth are:

| Arm | Coarse tensor RMS | Fine-residual tensor RMS |
| --- | ---: | ---: |
| Canvas | 0.9783 | 0.9271 |
| Canvas coarse + local fine | 0.9783 | 0.9426 |
| Local both | 0.9858 | 0.9461 |
| Oracle coarse + local fine | 1.0000 | 0.9931 |

This shows strong coarse-conditioning sensitivity, including a response in the
fine-residual component. It does not cleanly assign the whole error to the
coarse model: oracle conditioning removes actual coarse uncertainty and its
errors, and does not isolate a training-distribution mismatch from conditional
response, field alignment or coarse/fine dependence.

Window pseudo-spectra strengthen that distinction. Across the predefined bands
and both phases, canvas coarse draw/truth power ratios are 0.923–1.023; coarse
mean-error/variance ratios after finite-draw correction are 0.861–1.087.
There is no large, universal coarse low-frequency power shortfall in this test.
Canvas fine-residual power is more broadly suppressed: 0.787–0.913 of truth;
local-both raises this to 0.824–0.944, and oracle coarse raises it to
0.957–1.007. The deficit is not confined to the lowest fine-residual bin.
Oracle fine-residual error/variance still reaches 1.206 in ph012's lowest band
and approximately 1.11–1.13 in the highest bands. Scalar spectra cannot rule out
incorrect anisotropy, phase alignment, cross-scale correlations or conditional
miscalibration even where their ratios are near one.

Calibration does not follow from the improved means. Joint four-gap nominal
90% coverage is 68.75% for canvas, fixed-coarse/local-fine and local-both in
each phase. After removing the phase mean and subtracting marginal draw-mean
noise, local-both residual-scatter eigenvalues in mean-posterior-covariance
units remain [0.436, 1.237, 1.784, 2.400] and
[0.394, 1.119, 1.497, 2.270]. Oracle coverage is 81.25%/75%, not qualification;
its corresponding eigenvalues are [0.514, 0.801, 1.031, 1.423] and
[0.432, 0.704, 1.317, 1.827]. These are descriptive dependent-pair diagnostics,
not independent binomial tests or proof of a uniquely identified covariance
error. The materially narrower oracle width SD reflects privileged conditioning.

## Research decision

The density-only reconstruction is a promising deterministic operator candidate,
not a promoted scorer. Preserve the original physical target. Its production
implementation would still need full-cap convergence, all-core posterior replay
and the registered joint/conditional calibration gates.

Opus's two controls were useful: they identify a partial sampler/context effect
and a much stronger sensitivity to coarse conditioning. They do not furnish a
complete additive causal partition into coarse model, fine model and sampler.
Prioritize a bounded coarse/fine-conditioning and cross-scale-dependence
ablation, with any sampled-conditioning training preserving the exact parent
mass contract, before selecting a low-mode architecture or simply training
longer. Neither additional training nor an architecture change is launched here.
The present evidence does not justify releasing a corrected posterior, opening
confirmation phases or replacing the global canvas with the local sampler.

The next [conditioning/decoder ablation](e2e_conditioning_ablation_20261005.md)
is complete. It separates decoder mass weighting from shape response and does
not confirm a scalar distance-driven amplification mechanism. Refer to that
follow-up before choosing a conditioning-augmentation training objective.

## Reproduction

Source scripts under `workflows/sbi`:

- `e2e_density_only_reconstruction.py`: fixed conservative reconstruction.
- `e2e_sampler_factor_diagnostic.py`: aligned input preparation and five arms.
- `e2e_sampler_scale_diagnostic.py`: density-scale replay without new inference.
- `e2e_reconstructed_posterior.py`: fixed reconstruction on saved posterior draws.
- `e2e_density_sampler_summary.py`: reviewed summaries and CPU parity check.

Artifacts live beneath the campaign Scratch root
`coherent_exterior_20261002_v1/verification`, in
`density_only_reconstruction_20261005_v1`, its `_all64` expansion,
`sampler_factor_20261005_v1`, and `reconstructed_posterior_20261005_v1`.
All reconstruction, sampler, scale and posterior-replay completion receipts
passed. CPU59382745 and GPU59382896/59383302 were released after their work
finished. The only failed engineering step was the preserved four-draw report
smoke described above; its corrected smoke and all scientific runs passed.
Unfinished production jobs were not cancelled. No more than two interactive
allocations were held at once.
