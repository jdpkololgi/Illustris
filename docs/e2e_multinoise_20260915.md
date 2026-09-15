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

Running as approved allocation58368502 on nid001028, source registration569bfe9.
All six full-size forward/backward smoke checks passed (peak GPU allocation
0.95--2.10GB). Actual parameter counts: plain U-Net96,969; FiLM variants112,937;
transformer120,592; wavelet346,448. Nineteen focused model/report/regression tests
pass. No completed comparison or scientific pass is claimed yet.

The post-fit read-only response module adds216 forwards: clean and antithetic
noise inputs for sixmodels x twonear-clean ratios x sixanchors. It introduces
zero updates or new pass thresholds and has a separate ten-minute cap within
the approved allocation. Reports verify exposure pairing, disjoint evaluation
noise, no transfer fitting, gate recomputation, source/checkpoint/draw hashes.

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
