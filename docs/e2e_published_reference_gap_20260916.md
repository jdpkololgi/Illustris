# Published-reference gap: CAMELS VDM and Cosmo3DFlow

Read-only audit2026-09-16 after the user questioned diagnostic churn. No new
training authorized by this note. CAMELS public main inspected at
`032a19f33e471d15bf552aa175bf2039c372baf0`; code was read, not executed/copied.

## CAMELS: a genuine missing reference, not just another attention toggle

Sources: [paper](https://arxiv.org/html/2403.10648v1),
[VDM](https://github.com/victoriaono/variational-diffusion-cdm/blob/032a19f33e471d15bf552aa175bf2039c372baf0/model/vdm_model.py),
[network](https://github.com/victoriaono/variational-diffusion-cdm/blob/032a19f33e471d15bf552aa175bf2039c372baf0/model/networks.py),
[data](https://github.com/victoriaono/variational-diffusion-cdm/blob/032a19f33e471d15bf552aa175bf2039c372baf0/data/astro_dataset.py),
[training](https://github.com/victoriaono/variational-diffusion-cdm/blob/032a19f33e471d15bf552aa175bf2039c372baf0/train_model.py).

| Axis | Authors' implementation | Our current diagnostic |
| --- | --- | --- |
| Representation | log10(stellar mass+1), log10(DM), fixed per-suite mean/std | Affine normalized signed fine residual; previous normalization controls do NOT test this nonlinear log-density representation |
| Network | Four down/up levels, base48 to bottleneck768, residual blocks, bottom attention, log-SNR/time embedding, noise prediction+input skip | Three levels, base8, about113k parameters, FiLM near-identity velocity chart; not a comparably scaled reference |
| Objective | VLB: .5 gamma'(t) epsilon-MSE plus terminal-prior KL and finite-noise decoder likelihood | Fixed cosine VP, v-MSE under registered noise exposure; later auxiliary clean/response penalties were not their recipe |
| Endpoints | Learned finite linear gamma: trainable intercept and nonnegative slope | Exact VP endpoints and fixed schedule; finite-sigma identity penalty is NOT their decoder likelihood |
| Optimization | Batch12 with antithetic time sampling, AdamW1e-4, cosine warm restarts; public trainer clips at.5 | Batch1, LR1e-4 fixed, clip1; today's first-moment/clipping factorial cannot replicate their differently scaled VLB gradients |
| Sampling | Stochastic ancestral reverse transitions, typically250 steps | Deterministic DDIM or probability-flow Heun from random initial noise; not inherently invalid, but a different finite-step approximation |

The paper's main training description is60,000 batch12 updates; its ablations
are300,000. Public main uses max_epochs1000, so code default is NOT proof of the
exact published stopping point. Their LH sets contain1000 simulations per suite,
15 projections per simulation. Our15 cutouts span three phases, not15 independent
universes. This limits evidence but does not by itself establish the cause of
our spectral errors. Their conditioning is a2D projected stellar mass map atz0;
they explicitly defer3D DESI point clouds, selection/RSD/fibre collisions to future
work. That makes the task different, not the reference irrelevant.

The VLB is not a magical unbiasedness guarantee, and velocity-MSE is not invalid
merely because it differs. The important omission is that we have not tested a
faithful, adequately sized conditional VDM baseline before many custom repairs.
In their public code, the reconstruction term depends on the noisy endpoint and
schedule, not a learned D(clean,sigma)=clean penalty across positive sigma.

Evidence for their posterior claim includes sampled density PDFs, spectra,
cross-correlations, regional masses, different simulation suites and parameter
variations. These are substantially more direct than our isolated loss proxies.
I did not find a repeated-simulation joint-field SBC/rank-coverage demonstration
in this paper; its use of unbiased should not be treated as a universal theorem.
Our future comparison should retain direct sample metrics AND calibrated coverage.

Do not blindly copy public-code validation: it uses map-level random_split rather
than grouping maps by simulation, evaluates sample-MSE on the first validation
batch, and loads normalization constants without documenting their fitting split
in these files. Use simulation-grouped splits and explicit train-derived metadata.
These are reproducibility cautions, not a dismissal of their separate CV tests.

## What the wavelet work actually established

[Cosmo3DFlow](https://arxiv.org/html/2602.10172v1) reconstructs initialz127 fields
from evolvedz0 matter/halo grids in periodic Quijote boxes. It combines wavelet
CFM, a multilevel3D U-Net, scale-specific conditioning and cross-scale connections,
plus optional weight.01 log-power regularization on voxel-space velocities.
Its32^3 ablation reports VRMSE .26 baseline, .34 bare wavelets, .27 with scale/
skip additions, .25 with spectral regularization. Wavelets alone were not enough
even in the authors' experiment. Its spectral term is not automatically a proper
conditional-posterior scoring rule for our non-Gaussian late-time target.

Our executed comparison was a single-level invertible Haar transform plus flat
residual CNN under the diffusion v-loss; see e2e_multinoise_20260915.md. No paper-
specific multiscale U-Net/cross-scale design or spectral loss. With corrected skip,
it passes every fitted/transfer .2-noise phase; .05 transfer noise left40.51%
versus13.33% for paired U-Net. Its transfer generated high-k power/truth24.57
versus3.54 for the then-current U-Net. This is useful negative/partial evidence
for THAT small model, not a test invalidating Cosmo3DFlow. The planned E2E-WCFM
fine-representation challenger remained gated, not completed and discarded.

## Consequence for the next decision

Do not call our architecture probes faithful paper reproductions. Finish direct
frozen pipeline assessment without more training, then choose one reference-led
training experiment rather than a broad architecture/auxiliary-loss sweep. A
published-task reproduction followed by a clearly labelled3D/selection adaptation
would distinguish implementation failure from task/domain shift. If instead
adapting directly, label it an adaptation and retain a matched base objective.
Log full positive density, not a signed residual; do not apply periodic padding
to survey edges. Compare spectral penalties on/off and test posterior spread,
not just spectral agreement. No new allocation or fit is launched by this audit.
