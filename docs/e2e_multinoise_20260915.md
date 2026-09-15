# Joint-noise residual denoiser architecture comparison

## Registered plan (before execution)

Motivation: fixed-ratio residual U-Net learns fitted .05/.2 denoising, but .05
transfer high-k error remains .314--.493 of parent despite noise amplitude
.105--.146. Antithetic response localizes much of that error to persistent
distortion, not uncancelled injected noise. No prior model has demonstrated a
single joint-noise denoiser with transfer preservation. Full E2E stays paused384.

Six fresh-start arms, same three NGC fitted anchors and three SGC training-pool
transfer anchors as previous tests; no transfer fitting or held-out payloads:

| Arm | Changed mechanism/hypothesis |
| --- | --- |
| unet_residual | Original-size U-Net, bounded field-unit raw-head scaling |
| unet_film | Same base initialization, per-block noise FiLM |
| unet_film_clean | FiLM plus .1 clean-input penalty for nominal ratio <=.2; explicitly changed objective |
| unet_film_drop | FiLM plus .1 whole-context dropout and explicit presence bit |
| transformer | Patch2, width64, two four-head window-attention blocks, window4 tokens, offset0/2, adaptive LayerNorm |
| wavelet | Orthogonal eight-subband Haar analysis/inverse, width32, four residual convolution blocks with FiLM |

The transformer outputs all eight voxels per patch, not their mean. Windows
use padding/masking and cropping, never cyclic wrap. Wavelets preserve all
subbands, with no activation normalization or periodic convolution. Both retain
the same wide-context encoder architecture (fresh common seed); architecture
comparisons are NOT parameter/FLOP-matched or pure attention-only ablations.
Transformer design is inspired by [DiT](https://arxiv.org/abs/2212.09748);
wavelet design by [MWCNN](https://arxiv.org/abs/1805.07071). These are independent
small 3-D adaptations, not reproductions or claims that image benchmark results
transfer to cosmological fields.

For VP a=cos(pi*t/2), b=sin(pi*t/2), define d=sqrt(b^2+.05^2*a^2), raw correction r,
v=-r/d and D=a*x+(b/d)*r. Gain is bounded by20 at both endpoints; no division
by a or b. All zero heads begin at v=0, D=a*x. This changes raw-head scaling and
parameter-space optimization, not scalar v-MSE. It is NOT the earlier singular
D=x/a+r specialist or an exact conversion of its trained checkpoint. The clean
penalty adds .1*||v(a*y,t)+b*y||^2, equivalent to normalized clean reconstruction
error, and may trade optimal stochastic denoising for less distortion; it cannot
be promoted just because a clean-input control improves.

Budget: 3,072 updates/arm,18,432 total, AdamW lr1e-4/wd1e-4/clip1. Common model
seed91532, training seed91531 and evaluation seed91526. Continuous log-uniform
sampling within bins [.025,.1] twice, [.1,.4], [.4,2], [2,10], [10,80]; three-step
blocks expose every phase to every bin. Pure-noise endpoint every64updates.
This is a diagnostic exposure choice, not a posterior-optimal schedule claim.
Evaluate0/512/1536/3072 at .05/.2/1/5/20 (two common noise replicates), then
interpolation probes .08/.5/2/10. Original frozen384 parent baseline. Expected
1,440 curve probes +60parent +288interpolation =1,788;18checkpoints;42paired
32-step DDIM true-coarse diagnostic draws (parent+six arms x sixanchors).

Unchanged per-phase gate at BOTH .05/.2: high-k absolute noise amplitude <=.2,
high-k error <=.25paired parent, all lower-three-band gains .9--1.1. Fit and
transfer separate; every phase must pass. Broad-noise/interpolation errors and
generated spectra are regression evidence, not substituted passing criteria.
No held-out calibration, production readiness, convergence, or architecture
superiority claim from one small correlated training-pool panel and one seed.

One user-approved A100, at most two-hour interactive allocation,6,000s application
cap; full96-cube finite-gradient/memory smoke for all arms before fitting.
Stop on nonfinite/OOM/cap/provenance mismatch; no automatic extension or retries.
Original pipeline/config/data/normalization/checkpoints are immutable. New
experiment bindings and hashes separate all diagnostic checkpoints.

Entrypoints: `workflows.sbi.e2e_multinoise_test`, models in
`workflows/sbi/e2e_multinoise_models.py`, frozen config
`configs/e2e_multinoise_20260915.json`, tests `tests/test_e2e_multinoise.py`.
Large artifacts remain under the registered wide_pipeline_v1 Scratch child.

## Results

The initial six-arm experiment completed as allocation58368502 on nid001028,
source registration569bfe9:2,208.51s after preflight/data validation,18,432updates,
1,788probes,18checkpoints and42 true-coarse DDIM draws. All six full-size
forward/backward smoke checks passed (peak GPU allocation0.95--2.10GB).
Original run artifacts, checkpoints, source hashes, pairing and gates verify.
Twenty focused initial model/report/regression tests pass.

No phase-level gate passes:0/72. Each phase must meet noise <=.2, error/parent
<=.25 and lower-three-band gains .9--1.1. The following noise values are medians
over transfer anchors/noise replicates; error shows the range of the three
paired phase gates. Percentages are noise AMPLITUDES, not noise powers.

| Initial chart / backbone | Parameters | .05 noise left | .05 error/parent | .2 noise left | .2 error/parent | Transfer draw high-k power/truth |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| Plain bounded U-Net | 96,969 | 95.72% | .942--.946 | 76.11% | .725--.765 | 46.23 |
| Noise-FiLM U-Net | 112,937 | 57.97% | .680--.729 | 26.68% | .364--.383 | 15.51 |
| FiLM + clean penalty | 112,937 | 72.46% | .758--.811 | 40.97% | .449--.457 | 24.12 |
| FiLM + context dropout | 112,937 | 99.66% | .987--1.000 | 92.62% | .961--1.001 | 53.47 |
| Patch transformer | 120,592 | 87.68% | .971--1.009 | 60.48% | .565--.621 | 83.28 |
| Haar-wavelet CNN | 346,448 | 84.76% | .841--.872 | 77.26% | .693--.722 | 32.56 |

Frozen384 parent transfer draw high-k power/truth is67.12 on this exact panel.
FiLM improves it strongly but remains far from unity and distorts other bands:
transfer median four-band power ratios1.199/1.053/.781/15.508. Transformer is
not better in this bounded test: ratios.934/.431/.502/83.281. It is much faster
(118.80s vs427.67s FiLM; wavelet192.59s), but this is an update-matched comparison,
not an equal-walltime or equal-FLOP ranking of architecture families.

Noise-conditioned U-Net learning is strongly noise dependent: fitted v-MSE
initially about1 at all levels, midpoint .05/.2/1/5/20 becomes
1.001/.967/.550/.204/.325; final .633/.366/.211/.185/.327. Falling pooled loss
therefore did not initially demonstrate near-clean learning. The final curves
are still changing substantially; neither convergence nor sufficient joint-noise
capability is established. Clean penalty/dropout in these specific forms are
not supported for promotion. No transformer-family impossibility is inferred.

The post-fit read-only response module adds216 forwards: clean and antithetic
noise inputs for sixmodels x twonear-clean ratios x sixanchors. It introduces
zero updates or new pass thresholds and has a separate ten-minute cap within
the approved allocation. Reports verify exposure pairing, disjoint evaluation
noise, no transfer fitting, gate recomputation, source/checkpoint/draw hashes.
The216 response forwards completed in15.39s after preflight. At .05 transfer,
noise-even fractions are .356FiLM/.238clean-penalty/.224transformer/.160wavelet;
clean-input power/paired power is .380/.252/.224/.160. Unlike the successful
fixed-noise specialist, these joint models still leave substantial injected
noise. A smaller even fraction is not a success: the penalty reduces that
fraction while worsening total error and cancellation. Antithetic closure holds.
Evidence: `docs/evidence/e2e_field_v2/multinoise_20260915/SUMMARY.json`,
`RESPONSE.json`, `learning_curves.png`, `SLICES.json`, `transfer_slice.png`.

## Separately approved adaptive skip-path follow-on

After the six fits showed no complete gate pass, the user approved approximately
15 additional GPU minutes within the same two-hour allocation. This is adaptive
and must NOT be presented as part of the original six-arm preregistration.
The initial bounded chart retained raw scaling but replaced the specialist's
near-clean identity skip. Its failure therefore confounds noise coverage with
that skip change; it does not disprove the earlier specialist's learnability.

Freeze a skip-only three-arm contrast: FiLM U-Net, transformer, wavelet, each
fresh3,072updates with IDENTICAL weights/optimizer/exposure/evaluation seeds
to its original counterpart. Change only v from -r/d to -a*b*x-r/d, equivalently
D=a*(1+b^2)*x+(b/d)*r. The fixed velocity coefficient is bounded by1/2 at every
time and zero at both endpoints. For a clean input x=a*y and zero head,
D=(1-b^4)*y rather than (1-b^2)*y: it approximates the specialist's near-clean
identity through second order without the divergent x/a endpoint. The initial
predictor and raw regression target change; architecture and scalar v-MSE do not.
This changes neither the pass thresholds nor the old six fits/checkpoints.

Budget9,216 newupdates;864 new noise probes plus60 reused parent probes;
9checkpoints;18 new true-coarse draws;108 clean/antithetic forwards, no held-out
payloads.900s application cap, no automatic extension/retry. Compare against the
immutable original run and paired counterpart, not just the weaker frozen384
baseline. Entrypoint `workflows.sbi.e2e_skip_path_test`, frozen config
`configs/e2e_skip_path_20260915.json`, focused tests `tests/test_e2e_skip_path.py`.

### Skip-only results and interpretation

Completed with source82c4c14,761.60s application time (12m57.74s process),
9,216updates,864 new probes,9 checkpoints,18 new draws,108 response forwards.
Training exposures exactly match each original counterpart; all source,
checkpoint/draw hashes, per-phase gates, and evaluation panels verify.

| Near-identity skip model | .05 transfer noise left | .05 error/parent by phase | .2 transfer noise left | .2 error/parent range | Phase checks passed |
| --- | ---: | --- | ---: | --- | ---: |
| FiLM U-Net | 13.33% | .249 / .375 / .340 | 9.23% | .077--.090 | 9/12 |
| Transformer | 94.89% | 1.067 / 1.024 / 1.035 | 73.60% | .696--.741 | 0/12 |
| Haar-wavelet CNN | 40.51% | .412 / .392 / .408 | 14.76% | .182--.190 | 6/12 |

Both U-Net and wavelet pass ALL fitted and transfer phases at .2. U-Net passes
two fitted .05 phases and one transfer .05 phase. Fitted ph000 fails its noise
threshold (.232 amplitude despite .145 error/parent); transfer ph002/ph003 fail
error reduction despite passing noise and lower-band gain criteria. The .05
transfer lower-band gains remain close to unity, but this alone does not protect
the genuinely weaker high-k structure. No model passes the full12-check gate.
Total15/108 phase checks pass across the nine fits; no cutoff or threshold moved.

The skip-only intervention is a substantive isolation result. On identical
FiLM U-Net weights/data/optimizer/exposure, .05 transfer noise drops .580->.133,
worst-phase error/parent .729->.375. Generated transfer high-k power/truth drops
15.51->3.54. This cannot be attributed to extra network capacity or extra updates
relative to its paired counterpart. It does change initial prediction, raw target
and optimization trajectory; it is not proof that a scalar loss rescaling works.
It also helps the wavelet model, but not the tested transformer. The latter is
therefore not supported as a replacement by these small, update-matched tests.

The improved U-Net's .05 transfer residual error is now80.19% noise-even;
clean-input error power/paired power .904 and even/clean correlation .99788.
Thus the remaining problem again resembles persistent reconstruction distortion,
not primarily uncancelled injected noise. This identifies an error component,
NOT a unique cause such as normalization, regional coverage or conditioning.
The ratio of powers is not an additive bias fraction. The antithetic identities
hold numerically. Wavelet .05 even fraction46.00%; transformer12.26%.

Full generation remains inadequate even with true coarse conditioning. The
improved U-Net transfer four-band power ratios are .706/.611/.625/3.543: less
high-k excess, but underpowered lower/middle bands. Transformer ratios are
.851/.170/.228/75.215; wavelet .910/.861/1.149/24.572. No generated-coarse E2E
draws, calibration test or production promotion were performed.

Convergence is NOT established. Improved U-Net successive512-update mean losses
are1.004/.609/.448/.328/.315/.272; noise/error curves still move substantially.
Gradient clipping is frequent (93.88% U-Net,94.27% transformer,76.79% wavelet).
That is an optimization diagnostic, not proof that clipping causes the failure;
earlier clipping ablations used a different parameterization. This budget is
sufficient to isolate a useful skip mechanism and reject promotion, not to claim
the best model has reached a converged transfer-error floor.

Decision: retain near-identity+FiLM U-Net as the diagnostic reference, keep full
E2E paused at384, and do not replace it with this transformer. The next useful
controlled comparison is broader fitted-anchor coverage versus unchanged
coverage, with the same skip/model and a geometrically disjoint transfer panel;
then isolate activation normalization if persistent clean-input distortion
remains. A training-only estimated Wiener reference would also help separate
optimization failures from limitations of simple shrinkage, without tuning on
transfer truth. A bounded continuation of the best diagnostic may still improve,
but more full E2E training is not justified by these gates. None of these next
tests has been launched, and no additional allocation is authorized implicitly.

Evidence: `SKIP_SUMMARY.json`, `skip_comparison.png`, and `RUN_RECEIPT.json` in
the same evidence directory. Across both tests:27,648updates,2,652 new noise
probes (the60 reused skip baseline probes are NOT counted twice),27checkpoints,
60 true-coarse draws and324 response forwards. Allocation58368502 was released
after57m34s; all four steps and allocation COMPLETED0:0, no allocations remain.
Original384 parents, normalization/data/physics contracts and unrelated D2 work
are unchanged; training_ready=false and no calibration/physics pass asserted.
Final focused verification:23 tests pass across multi-noise models/report,
skip-path algebra/report, and the fixed-noise/EDM regression tests. Both learning
curve figures were visually checked for correct labels, thresholds and pairing.
