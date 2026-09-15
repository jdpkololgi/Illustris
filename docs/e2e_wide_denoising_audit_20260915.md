# Wide E2E denoising localization — 2026-09-15

Status: diagnostic implementation prepared; compute execution requires scheduler
approval. No fitting, architecture/target changes, smoothing, clipping, held-out
payload access, or promotion of any model is authorized by this investigation.

## Registered comparison

Use the unchanged balanced 24-anchor training diagnostic panel from the 384-update
evaluation (ph000/002/003, four shells, two support strata). Compare the frozen
192- and 384-update checkpoints, both CFM and diffusion, both coarse and fine
stages, with two common Gaussian-noise replicates per anchor/stage and five
noise/signal amplitude ratios: 0.05, 0.2, 1, 5, 20. This gives 1,920 controlled
forward probes. Time conventions differ: CFM data is at t=1, diffusion data at
t=0. Match sigma/alpha, not time or absolute noise amplitude.

The exact clean estimates are x+(1-t)v for CFM and alpha*x-sigma*v for diffusion.
Check the error identities against the known injected noise on every probe.
Analytic-oracle unit tests also exercise both actual sampling implementations.

Use weighted demeaning and a 3-D Hann window, with full-rFFT multiplicities.
Sum window-normalized fluctuation powers in coarse k bands (0,.04],(.04,.08],
(.08,.16],>.16 and fine/local bands (0,.08],(.08,.16],(.16,.32],>.32 h/Mpc.
Report absolute error power alongside error/truth, gain, correlation, noise
leakage, and velocity-error band fractions. Window leakage prevents interpreting
the globally low-pass coarse target as exactly band-limited on each crop.

Decompose all 384 saved draws on this panel (two checkpoints, two methods, four
draws) into interpolated coarse power, residual power, and signed cross-power.
The residual is not an orthogonal high-pass field; do not discard cross-power.

Generate 48 explicitly labelled **true-coarse diagnostic** local fields at step
384, with the same fine-stage seeds as saved draw 0. This isolates dependence on
the coarse input but is an oracle control, not an inference result or correction.
For six boundary anchors in shells 0 and 3, instrument both sampler stages at
every fourth network call plus call 31. Require exact reproduction of both
saved stage fields; inspect intermediate predicted-clean spectra. Odd CFM call
31 evaluates the Heun proposal, not the final accepted state.

## Interpretation and stop conditions

High-noise pointwise error includes conditional uncertainty; low gain there is
not by itself a denoiser bug. Near-clean recovery can preserve the input almost
trivially, so inspect velocity error/noise cancellation as well. High-k ratios
can be inflated by very small truth power; use absolute errors and total-power
fractions before attributing physical importance. These are correlated training
anchors with only three phase units, not validation or posterior calibration.

Stop on provenance drift, checkpoint/draw checksum mismatch, failed algebraic
closure, or nonidentical instrumented replay. No automatic restart or additional
training. Keep the original source, checkpoints, normalization, continuation
binding, negative-domain receipts and release flags frozen. Outputs live in a
new dedicated Scratch directory; archive compact receipts/results in this repo.

Entrypoint: `python -m workflows.sbi.e2e_wide_denoising_audit --output <new-path>`.
