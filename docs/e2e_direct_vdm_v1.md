# Direct conditional density pilot, 2026-09-16

User authorized implementing and testing the reference-led recommendation, with
bounded side checks and a focus on p(delta | galaxies, DESI survey). This is a
research pilot, not production posterior validation or a CAMELS reproduction.

User subsequently requested training now in tmux for approximately90minutes.
Priority changes: run smoke then the four-fit VDM matrix as a single90-minute
shared Slurm batch job launched from tmux. Frozen paired inference is deferred.
This uses one GPU, not four90-minute jobs. Slurm survives terminal loss; tmux is
only the retained launch terminal. Save atomic checkpoints every256updates and
on scheduler warning; no automatic requeue or resume after interruption.

## Ordered experiments

1. Frozen pipeline: improved fine control checkpoints at30720, seeds0/1; original
   diffusion coarse384. Six registered fitted/transfer anchors, two paired draws,
   true versus generated coarse. VP-Heun128 NFE throughout. Report TOTAL density
   power, cross-correlation, PDF/support, regional density, and coarse local error.
   No model updates. True coarse is a diagnostic oracle, not inference.
2. Reference-led DIRECT full-density conditional VDM: no sampled/true coarse input.
   Average96^3 delta_R7 cells to48^3 (6.766 Mpc/h), log1p physical density, affine
   normalization fitted only on ten NGC cutouts from ph000/ph002. ph003 excluded
   from fitting and all new normalization; all SGC excluded. These are development
   checks, not independent repeated-simulation calibration or sealed ph001 access.

Inputs: 12 local galaxy/survey channels pooled2, plus twelve spatially broadcast
wide-context means. No spatially misregistered wide channels; no density-derived
coarse channel. Existing deterministic observation transforms retained, followed
by refitted train-only moments. Response, LOS, redshift and counts remain inputs.
This is a voxelized mock observation model, not a complete DESI likelihood.

Paired arms share initialization, batch/field/noise order, larger base24 residual
3D U-Net, three down/up levels, bottom attention, gamma embedding and epsilon+z
output skip. Fixed linear gamma[-13.3,13.3] VLB versus learned linear gamma VLB.
Both use .5 gamma-prime epsilon-MSE + terminal KL + finite-noise Gaussian decoder
likelihood, bits/voxel. Decoder expectation is analytic, not Monte Carlo; no
positive-sigma identity penalty. AdamW1e-4, weight decay1e-5, clip.5, batch2,
5120 updates, two seeds, checkpoints512/2048/5120 plus every256. Fixed LR is a documented departure
from published cosine restarts. Zero rather than circular padding at survey edges.

Each final fit produces4 ancestral250-step draws on each of6 anchors. One paired
condition-swap draw on ph003 SGC tests condition dependence; it is not a calibration
test or an ablation separating galaxy channels from survey response.
Primary readout: total-density power/cross-correlation, density PDF/support,
regional-density spread, and fitted versus phase-excluded performance. Four draws
cannot validate posterior coverage. No training convergence claim from5120 updates.

## Decision rules

Never promote from falling loss, exact endpoint identity, or density positivity
alone. Positivity of direct log-density draws is architectural, not scientific
success. Report spectra on the adapted48 grid separately from original96 results.
If generated coarse degrades otherwise good fine draws, train/improve the coarse
posterior before changing the fine denoiser again. If VDM samples remain biased,
inspect the registered512/2048/5120 loss/checkpoints and condition response before
authorizing a larger run; this pilot cannot rule out the published method.
Full-cap continuity, tidal reconstruction and conditional calibration remain gates.

## Verification / operation

Module `e2e_direct_experiment stage --root <new direct_vdm_ Scratch child>` archives
committed sources; commands `smoke`, `paired`, `train --seed S --arm ARM` run from
that snapshot on approved GPU compute. Source and parent/data hashes checked.
Smoke validates full-size forward/backward, sampling and exact checkpoint replay.
Small tests: `python -m unittest tests.test_e2e_direct_vdm`.
Outputs exclusive-create; do not rerun into completed/partial branches. Checkpoints
are atomic/hash-bound but automatic continuation/requeue is intentionally disabled.

Literature mapping: `e2e_published_reference_gap_20260916.md`; equations from
[Ono et al.](https://arxiv.org/html/2403.10648v1) and their audited public implementation.
New code independently implements these equations, not downloaded executable code.

## Current execution record

Frozen implementation87151a15c7cc823a0036805910118dafb497dd04 in
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/direct_vdm_20260916_v2`.
Eight synthetic unit tests pass in0.553seconds, single-threaded CPU, including
finite-step analytic Gaussian sampler variance (not incorrectly requiring exact
unit variance at16 ancestral steps), target-channel exclusion and normalization
invariance to development data. No production data read by those unit checks.
Initial interactive smoke allocation58445684 queued with maintenance reservation;
full-size GPU smoke, replay and training are NOT yet passed/started. Source-only
v1 snapshot retained without training after correcting the oracle test expectation.

The two-hour request expired CANCELLED with zero runtime. Approved15-minute
allocation58445797 started nid008325. Full-size GPU smoke/replay passes:8,196,171
parameters, peak2.92GB including cached real data and replay copies,4.35seconds
model check. Full-shape synthetic benchmark32 measured updates averages.17945s;
one ancestral250-step draw11.0645s. Four fits at5120updates plus100draws predict
about80minutes compute, leaving roughly10minutes for data/checkpoint/metric I/O
within the90-minute job. This measured budget supersedes the initial2048-update
proposal. A10040GB is sufficient; do not unnecessarily constrain batch to80GB.
Final configuration must be restaged with matching smoke before submission.
