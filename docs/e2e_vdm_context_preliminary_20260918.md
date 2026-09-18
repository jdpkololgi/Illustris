# Preliminary registered density results: A/B/C

This is an early subset of the frozen experiment, not its final decision or a
replacement for D, tidal/eigengap, joint-field or conditional-stratum analyses.
All training was frozen before these scores were opened. No fitting, new draws,
checkpoint selection, normalization change, or architecture change followed.

## Evidence and scope

CPU job 58521029/nid004145 completed 0:0 in 51 seconds on 2026-09-18.
Wrapper commit 2aa4841 imports the scientific functions from frozen source
1639851. It analyzes all 384 registered A/B/C main cases (21,504 saved draws):
both seeds; all sixteen ph004 anchors at 5120/10240/20480 updates; and all
sixteen ph005 anchors at 20480. Each final phase has equal cap, shell and
support-stratum representation. These are two programme-exposed evaluation
phases, not population SBC or globally blind cosmological validation.

The first predeclared case matches the complete frozen report calculation for
CRPS, bias, RMSE, spread, widths, coverage and attainable coverage to
rtol=1e-10/atol=1e-12, with identical spectral outputs. Every input field array
was hash-verified on the compute node. A separate metadata audit verifies all
384 output receipts and their draw-receipt hashes, then independently checks
equal-anchor CRPS/bias/coverage aggregation. No audit errors.

Run root: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1`.
Artifact: `analysis/density_preview/PRELIMINARY.json`, SHA256
`ed3e401de1b236e6f946687d69bb491bc8a3f45d61549278fa802c472acd97c3`.
Parity, request, task receipts and terminal accounting are retained alongside it.
The 0.0141667 CPU-nodeh cost is inside the original eight-hour allowance.
After terminal verification, resources/DENSITY_PREVIEW_START.json registers
this completed CPU job with the unchanged controller's existing accounting scan.
Later controller totals therefore include it; do not add it a second time.

## Final-checkpoint density comparison

CRPS is the fair finite-ensemble score standardized by the frozen train-only
density scale; lower is better. Summary weights are equal over anchors and
then the two seeds and two phases. Coverage is marginal across correlated
central-core grid probes, not a whole-field or conditional-calibration claim.

| Arm | Training / information | Pooled CRPS | Mean central C90 | Seed/phase C90 range | Density-mean RMSE |
|---|---|---:|---:|---:|---:|
| A | 32 patches / 2 phases; wide summaries | 0.35214 | 79.09% | 78.16--80.66% | 0.33604 |
| B | 384 patches / 3 phases; same information | 0.31384 | 89.32% | 87.31--91.26% | 0.31177 |
| C | Same data as B; spatial wide context | 0.32443 | 87.35% | 85.74--88.44% | 0.32122 |

The average attainable target for the registered final ensembles is **90.75%**,
not exactly 90%. The earlier 32-draw ensembles have an attainable target of
87.88%, so raw coverage at different checkpoints must not be compared without
accounting for that difference.

**Diversity has a reproducible favorable density-score direction here.** B
improves pooled CRPS by 10.88% over A. All four seed/phase gains are positive:
11.24%, 11.00%, 10.89%, 10.35%. It also improves RMSE and brings central coverage
closer to the finite-ensemble target. This contrast increases both patch count
and the number of training phases; 384 patches are not 384 independent
cosmological realizations. Final registered contrast decisions and spatial
uncertainty still belong to the full report.

**Spatial wide context has not improved this density score overall.** C is
3.37% worse than B in pooled CRPS: three cells worsen by 3.01--5.62%, while one
improves by 0.53%. This is a result for the implemented encoder and budget,
not proof that wider observations carry no information.

## Checkpoint progression is mixed, not convergence

Development-phase changes from 10,240 to 20,480 updates:

| Arm / seed | CRPS improvement | Sample-power mean absolute log discrepancy, before to after |
|---|---:|---:|
| A / 0 | +0.20% | 0.3610 to 0.2325 |
| A / 1 | -4.38% | 0.2056 to 0.4923 |
| B / 0 | +6.10% | 0.0868 to 0.2307 |
| B / 1 | +8.98% | 0.2443 to 0.4039 |
| C / 0 | +4.98% | 0.2636 to 0.4238 |
| C / 1 | +4.24% | 0.2543 to 0.5395 |

B and C improve density CRPS in both seeds, but their sample-power discrepancy
worsens in both. At the final checkpoint B's aggregate sample/truth power ratios
span 0.562--0.893 across the five parent-patch bands and four seed/phase cells:
about 11--44% suppression. C spans 0.415--0.854. These are spectra of individual
posterior samples averaged across the panel, not merely the expected smoothing
of a posterior mean. No individual draw is required to reproduce its paired
truth spectrum exactly; the concern is the systematic panel-level shortfall.

Thus a reasonable central marginal coverage does not establish the correct
spatial field distribution. The much smaller sampler-refinement differences
make simply increasing sampling steps an unlikely complete explanation of
these spectral deficits, although the refinement screen itself is limited.

## Next action remains the registered full experiment

Finish D seed1, then evaluate all 688 cases with the frozen report: physical and
matched-boundary tides, eigengaps, posterior ranks/widths, conditioning strata,
fixed-mean/oracle controls, cross-core dependence, spatial uncertainty and
figures. Reconcile this preview numerically against the corresponding final
case summaries. Do not select a checkpoint, declare DESI readiness, or launch
more training from this partial report. The evidence supports a benefit from
training diversity, but does not yet establish a calibrated field posterior.
