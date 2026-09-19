# Research context — Illustris

## Current coupled-field direction (2026-09-19)

The A/B/C/D diversity/context/multiscale experiment is scientifically closed:
more diverse training helps density marginals; the tested hierarchy does not
establish a calibrated joint field posterior. Authoritative results and decision
are `docs/e2e_vdm_context_results_20260918.md` and
`docs/e2e_vdm_context_joint_decision_literature_20260918.md`.

The approved follow-up currently prepares data, not scientific fits:13train/
2development/6confirmation phases, matched DESI-like observed BGS mocks and
same-phase c000 z0.2 R7-smoothed matter. Coordinates are corrected to the pinned
DESI/Abacus distance convention with the inherited selection-volume Jacobian.
All21phase products, independent audits and normalized interfaces now qualify,
covering1,792paired domains and11,776offset cases. The full13-phase normalizer,
physical gate and7synthetic GPU factor checks pass; preparation jobs are terminal.
Use `SCIENCE_LOG.md` and
`docs/e2e_coupled_preparation_closeout_20260919.md` for current readiness.

Proposed later arms are retained D, observation-matched independent I-VDM,
joint J-VDM and joint J-CFM. I/J share their coarse fit/draws; fine coupling cannot
change block-aligned regional mass. Primary joint tests must therefore separate
coarse uncertainty from fine residual dependence. Full-size technical checks and
measured resource costs pass. The next proposal is ready for review, with a
development cap of260GPUh/24CPU-nodeh/512GiB; scientific implementation/training
still needs separate approval. No real-DESI validity, HOD marginalization, evolving lightcone
or whole-survey coherence claim follows from these nominal mock products.
The older dated status sections below are historical.

## Latest field-model controls (2026-09-16)

Frozen neural DDIM/Heun and first-moment x clipping tests complete on58442539.
57 tests pass; numerical Heun128 improvement24/24, but residual spectral bias and
seed1 auxiliary-loss tradeoff persist. No E2E training restart. Contracts/results:
docs/e2e_frozen_controls_v1.md. Public-reference audit:
docs/e2e_published_reference_gap_20260916.md. Our affine residual charts/tiny U-Net
do not reproduce CAMELS log-density/full-VLB/learned-schedule recipe; tested Haar
CNN is not Cosmo3DFlow. Prior next-step notes below are historical.

## Preservation objective (2026-09-16)

The eight-fit preservation matrix completed; no modified loss passes the joint
gate. Current next step: Gaussian/log-Gaussian oracles and unchanged-checkpoint
gradient diagnostics, `docs/e2e_oracle_conflict_v1.md`. No E2E restart or repair fit.

Those diagnostics are now complete: exact-oracle gates pass; matched-NFE VP Heun
improves synthetic sampling accuracy; checkpoint gradients reveal non-universal
conflicts. Raw projection fails consistent fresh-noise non-regression through
actual Adam/clipping. No automatic loss repair or E2E restart is justified.

Clean-limit chain58385300 ->58385303 ->58385304 completed successfully. Fixed15-
field optimization improves denoising substantially and passes .05/.2 noisy gates
across both seeds, but clean distortion is not stable. Near-zero exposure helps
the tiny-noise clean limit while leaving a seed-dependent signal/noise tradeoff.
Next: explicit clean-anchor and even/odd near-clean response supervision, four
paired arms/two seeds; contract `docs/e2e_preservation_objective_v1.md`. Implement
and smoke-test completed; user then authorized launch: array58407655 and dependent
report58407656 submitted2026-09-16 via tmux `e2e-preservation` on login39. No full E2E restart,
held-out access or promotion; live validation/result status in SCIENCE_LOG.

## Role

Illustris is the simulation and methods hub for GraphWeb: cosmic-web inference
from galaxy catalogues using AbacusSummit mocks, IllustrisTNG development
workflows, graph neural networks, and simulation-based inference.

It is the source repository for shared models, target definitions, training
artifacts, and the shared `SCIENCE_LOG.md`.

## Scientific goal

Infer calibrated continuous posteriors over the ordered tidal-tensor eigenvalues

`lambda_1 <= lambda_2 <= lambda_3`

for DESI BGS galaxies. This supersedes a purely discrete T-web classification
into void, wall, filament, and cluster.

The published foundation is Kololgi et al. 2025 (RASTI,
doi:10.1093/rasti/rzag025), a graph-attention classifier trained on
IllustrisTNG-300.

## Current programme state — reconciled 2026-09-09

- **P12-A:** the untempered, uncorrected coordinate-aligned FMPE is the first
  per-galaxy posterior product. Its one-open ph001 evaluation reports green and
  all ten registered release gates pass on 4,897,905 galaxies. This is a
  registered conditional-on-mock/HOD claim, not exact conditional calibration,
  HOD marginalization, sim-to-real validation, or a coherent field posterior.
- **P12-B:** the matched point/frozen-latent/joint pilot and five-arm continuation
  are complete. Features are connected and used, but neither the pilot nor the
  continuation supports a production representation change. No further variant
  is automatically licensed.
- **D2:** seed42 passed its registered native-support-amended science and sampler
  ladder. The second training seed and combined-decision chain are submitted;
  no two-seed promotion is recorded. All galaxies were retained, and the
  registered conditional tolerance is 0.10 versus 0.05 for global/joint checks.
- **E2E-FIELD:** 160 response-screened anchors (320 nested 64/96-cell views)
  have source-verified arrays, whole-phase internal roles, train-only scaling
  and independent native tensor/eigenvalue references. Full-size engineering
  checks pass, but `training_ready=false`. The completed three-training-phase
  error audit finds FD2 sub-percent in typical eigenvalue RMS but topology-
  sensitive; floor sampling and finite-parent tides are larger typical errors.
  FD8 nearly matches the spectral reference. Separate spectral v2 products are
  verified complete (job 58115135, 44m05s). The matched cubic-target domain audit
  also completed (58120066): neither 64 nor 96 cells qualifies under the frozen
  internal tidal/topology screen. The 96-cell parent passes filling/pair limits
  but retains eigenvalue/class and topology failures. See
  `docs/e2e_field_domain_gate_20260909.md`. Final model/transform release and
  diagnostic-power gates remain blocked; no learned training
  has begun. See `docs/e2e_field_data_build_20260908.md` and
  `docs/e2e_field_error_budget_20260908.md`.
  A separate user-authorized wider-coarse/local-fine build and truth-only test
  launched as interactive CPU job 58131992 (code 5051f69): two larger extents
  with a fixed-extent resolution control, training phases only. Historical
  all-anchor criteria are diagnostics for this extension, not an automatic
  research-pilot veto. No model training or release is implied. See
  `docs/e2e_field_wide_coarse_20260909.md`. All 96 anchors / three variants and
  nine payloads (5.687 GB) are verified built. The full truth-only test completed
  0:0: the 1299.072 Mpc/h domain reduces observed median eigenvalue RMS to
  0.817/0.643/0.489% of truth scatter, about 7--8x better than local96.
  One 6.211-percentage-point largest-void outlier remains; finer coarse resolution
  does not remove it. This supports bounded research-pipeline development, not
  a strict joint-topology pass or training release. See
  `docs/e2e_field_wide_coarse_results_20260909.md` for the completed comparison
  and concrete remaining model/normalization/sampler/canary work.
  The September-10 implementation now supplies a bounded wide384_f4 loader,
  matched staged CFM/DIFF trainer, shared-draw sampler and training diagnostics.
  The September-11 approved full-size GPU smoke and training normalization now
  pass (58196582, released): exact two-update resume and repeated-draw parity
  for both objectives/stages. Only engineering fits occurred; no research
  canary/science fit or held-out wide-product access. See
  `docs/e2e_wide_gpu_smoke_20260911.md`.
  The user subsequently authorized the matched 192-update research canary;
  job 58196924 completed all four 192-update fits, verified September 14 with
  finite histories and matching final checkpoint hashes. First-to-second-pass
  objective means fell 12--23%; convergence/calibration are not established.
  Trained-draw/checkpoint/sampler evaluation completed on September 14 (58305867,
  released): 864 saved fields and 4,608 fixed-noise probes, training phases only.
  Late losses still fall 1.5--7.7%, and checkpoint drift far exceeds sampler
  refinement drift. Generated support violations and incorrect power shape
  remain. The 192-update canary was not converged or science-ready; see
  `docs/e2e_wide_evaluation_20260914.md`. The subsequent authorized continuation
  to 384 and repeated evaluation/conditioning diagnostics completed as 58309454,
  released. Support/topology improved, but all predeclared diagnostic gates fail:
  late losses fall 7.9--11.4%, fields still drift, and low-k power suppression
  worsens while high-k excess remains. Observations affect the fits; generated-
  coarse loss sensitivity is modest and does not identify the spectral cause.
  Stopped at 384; no automatic 768, held-out opening or model revision. The
  subsequently approved scale-resolved audit completed September 15 (58352703,
  released): about 99% of generated local high-k power comes from the fine
  residual, whose near-clean estimates retain 97--98% of injected high-k noise
  amplitude. True-coarse controls improve largest-scale power but leave high-k
  excess almost unchanged. Exact sampler algebra/replay passes; optimization
  versus time-conditioning/capacity/loss-weighting causes remain unresolved.
  The subsequently authorized small learning test completed as 58358688:
  six 256-update fine branches, no capability pass. Near-clean-only fitting
  reduces residual-noise amplitude most (to 86% CFM/65% DIFF at ratio .05), but
  still fails and damages other noise levels/generated middle-scale structure.
  Partial learning is evident; no architecture impossibility or plateau is
  established. Keep E2E paused at384 and investigate fine-stage optimization/
  time conditioning before a large extension. Details:
  `docs/e2e_fine_learning_20260915.md`; held-out validation remains separately gated. See
  `docs/e2e_wide_denoising_audit_20260915.md`.
  The next approved six-arm diffusion-only EDM-informed test completed as58361744
  (released):512 additional updates per arm, all72 capability checks fail.
  EDM scalar preconditioning/loss equals existing normalized VP-v here; changed
  noise exposure helps, per-block conditioning partly helps, but optimizer reset,
  relaxed clipping and the tested bottleneck RF addition do not solve denoising.
  Best fitted .05 residual noise remains74%; generated high-k power remains
  7--9x truth for EDM variants. No converged or production-ready branch. Keep
  original E2E paused at384. Report: `docs/e2e_edm_ablation_20260915.md`.
  The subsequently approved fixed-noise residual-path/high-resolution test
  completed as58363476 (released).
  The same-size field-unit residual U-Net passes all fitted phases at BOTH .05
  and .2, and transfer at .2. Basic fitted denoising is now demonstrated; the
  blanket inability claim is superseded. At .05 transfer it cancels noise but
  fails reconstruction error, dominated by a noise-even distortion closely
  matching clean-input error. No architecture passes the complete two-ratio
  fit+transfer gate. Separate specialists are not a joint-noise sampler; original
  E2E stays paused at384. `docs/e2e_fixed_noise_20260915.md` records the results,
  confounders. Joint-noise comparison58368502 completed and was released:
  six3,072-update arms plus an explicitly approved three-arm skip-only contrast.
  Near-identity+FiLM U-Net improves .05 transfer noise to13.33% and worst-phase
  error/parent to.375; two transfer phases still fail error. U-Net and wavelet
  pass all .2 checks, but no model passes the full gate; transformer does not help.
  Remaining U-Net .05 error is80.19% noise-even, matching clean-input distortion.
  Generated high-k power is still3.54x truth with biased lower/middle bands.
  Keep full E2E paused384; no convergence/calibration/production claim.
  Completed isolation58373492 compared3/6/9/15 non-overlapping fitting
  regions versus two global normalizers, two seeds, fixed3,072 updates; twelve
  fixed transfer regions. Current target scaler is already pooled/global;
  strict scaler excludes transfer moments and is held fixed across diversity.
  Physical corruption/loss matched, internal GroupNorm unchanged. Exactly0
  identity is structural; positive-nominal-noise clean probes are diagnostic.
  Diversity gives modest, seed-dependent clean preservation gains, not a collapse;
  global scaling is not a reliable remedy. No model passes the full fitted+transfer
  gate. Near-zero probes below.025 extrapolate beyond training noise coverage.
  Next isolate optimization, selection/conditioning shift and this boundary;
  no follow-on launched, allocation released. No full E2E or held-out opening.
  `docs/e2e_diversity_norm_20260915.md`; prior `docs/e2e_multinoise_20260915.md`.
  See `docs/e2e_wide_continuation_20260914.md` and its archived evidence.
  The tested finite-window residual is not a local orthogonal high-pass
  subspace; see `docs/e2e_wide_pipeline_20260910.md` for the explicit amendment,
  matched canary caps and the 1299-versus-2000 Mpc/h volume/phase limitation.
- **P13/Loa:** held pending separate authorization. P12-A's single blind opening
  does not authorize ph001 reuse by D2, P12-B or E2E.

See `docs/plan_generalisable_graphweb_vac.md`, the P12-B plans and
`docs/plan_end_to_end_conditional_field_posterior.md`. These status statements
come from the recorded terminal evidence, not continuous scheduler monitoring.

## Historical graph/NPE baseline

- Training data: AbacusSummit HOD cutsky mocks with CACTUS T-web labels.
- T-web convention: potential Hessian, `lambda_th = 0.2`, smoothing scale
  7 Mpc/h. Do not conflate this with V-web velocity shear.
- Baseline encoder: attentional Jraph Battaglia-style GraphNetwork.
- Baseline graph: union of Delaunay and radius-10 Mpc/h edges.
- Posterior model: FlowJAX normalising flow trained as NPE.
- Canonical target representation: ordered eigenvalue increments. Predict
  `lambda_1` and positive successive increments; invert only for evaluation
  and plots.
- G3 union-graph NPE performance: R2 = 0.804 / 0.846 / 0.895 for
  lambda_1 / lambda_2 / lambda_3.

## Field-level physics-grounded inference: motivation and historical evidence

The immediate research priority is field-level inference rather than further
equivariant-architecture exploration:

`graph encoder -> density grid -> fixed FFT tidal operator -> eigensolver -> eigenvalues`

The fixed layer uses

`T_ij(k) = (k_i k_j / k^2) W_7(k) delta(k)`

so the density-to-tidal mapping is physical and differentiable rather than
learned. This supports tensor consistency and supplies eigenvectors for future
intrinsic-alignment science.

Earlier deterministic evidence (not posterior-selection scores):

- T2: a 3-D U-Net on 5 Mpc voxelised galaxy counts reaches lambda_1
  R2 = 0.876 +/- 0.004 across three seeds.
- T4/F1: graph encoder -> CIC scatter -> 3-D U-Net -> fixed physics layer
  reaches R2 = 0.841 / 0.897 / 0.931.
- These deterministic scores do not establish posterior calibration. The
  current posterior status is the P12-A/P12-B/D2 summary above; full-cap field
  coherence remains a separate E2E research question.

The graph work remains central: the results indicate that representation scale
and a physics-grounded output are the main levers. A CNN is a GNN on a regular
lattice; graph encoders remain highly effective, especially in the
graph-to-field architecture.

## G4 evidence and architectural lessons

G4-PROPER is now supporting evidence rather than the active near-term gate.

- Union graphs beat either Delaunay-only or radius-only graphs.
- Delaunay void bridges and fixed-radius edges are complementary,
  geometry-anchored support for the non-local tidal operator.
- Position-only fixed-local attention recovers most curated-feature signal.
- Dynamic feature-space kNN / DGCNN harms performance: long connections should
  remain geometry-anchored, not feature-anchored.
- Tested SEGNN variants underperformed but were capacity-confounded; steerable
  models are shelved, not disproved.
- Attention is a second-order question to test cleanly inside the F-tier.

## Interpretation and safeguards

- The graph is a discrete quadrature of the non-local tidal operator
  (`1/k^2` inverse Laplacian), not merely a generic ML representation.
- NPE incorporates the Abacus training prior. Do not stack posteriors to infer
  population eigenvalue distributions without importance reweighting or
  hierarchical SBI.
- Feature-space domain shift and implicit-prior mismatch are distinct.
- TARP/SBC establish in-domain calibration, not sim-to-real validity.
- DESI closure tests remain necessary.
- Current field-level comparisons include unresolved controls: matched-estimand
  graph point estimates, DESI-like number-density re-runs, cell-size sensitivity,
  and attention on/off within the F-tier.

## Shared workflow

`SCIENCE_LOG.md` is the live source of truth. Read it before substantive work.
It wins if it conflicts with this file.

Use `[science]` for research decisions and `[code]` for implementation or run
results. Add only genuine decisions, hypotheses, results, and direction changes;
keep newest entries first.

Local desktop work is primarily science, planning, and interpretation. NERSC
work includes implementation, data processing, interactive development and
user-authorized frozen production jobs.
Synchronise through git and the science log:

1. Pull with `git pull --no-rebase` before pushing.
2. Preserve all log entries when resolving conflicts.
3. Commit substantive science-log updates; push when requested or authorized.
4. Use the NERSC allocation workflow: interactive allocations for development,
   explicitly authorized Slurm batch for frozen production. A login-node tmux
   supervisor is not a substitute for scheduler persistence. Preserve all
   scheduler approval rules and frozen source worktrees.
