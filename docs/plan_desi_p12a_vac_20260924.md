# P12-A to DESI Loa environmental-posterior VAC

Active plan, adopted 2026-09-24 following the user's approval of the roadmap
disposition and request to proceed toward DESI application. SCIENCE_LOG.md and
immutable experiment contracts control claims. This supersedes old prospective
VAC schedules, not historical evidence. V0 source/recorded-product inspection
has begun; no new compute job or inference has run under it.

## Decision and product

Proceed with frozen, untempered, uncorrected U-PATCH + P12-A through input parity,
a golden mock and a bounded Loa trial. Do not retrain merely because additional
same-family phases exist. Public science release remains gated on survey parity,
replication and misspecification evidence. Coherent-field research is separate.

Release joint ordered eigenvalue posteriors per BGS BRIGHT galaxy, conditional
on the fiducial simulator, at Gaussian R=7 Mpc/h and target epoch z=0.2. They
are not a spatially joint posterior. At threshold 0.2, derive void/sheet/filament/
knot probabilities by counting 0/1/2/3 eigenvalues above threshold in the same
joint draw; never multiply marginal probabilities. Preserve the sparse-shell
limitation and quality flags. Do not infer orientations from eigenvalues alone.

The current science-shell edges are [0.15,0.25,0.35,0.45,0.55]; V0 must verify
support against the actual frozen P12 manifest. Context redshifts and E2E cost
study ranges are not science support. Unsupported rows get flags/null values.

Blind ph001: 4,897,905 rows; C68=.66661/.67075/.66480,
C90=.89115/.89481/.89001; maximum conditional error=.03429; ten registered
gates pass, one opening, no refit. The older stricter calibration marker remains
absent; do not manufacture it or claim exact conditional calibration.

## Execution stages

| Stage | Owner | Deliverable and exit criterion | State |
|---|---|---|---|
| V0: handoff freeze | Illustris | Checkpoint/source/transforms/input/solver manifest; exact support, quality bits and coordinate audit | Started: source/recorded-product audit; numerical closure pending |
| V1: Loa crosswalk | GraphWeb_DESI | Hashed release and mock-to-Loa schema; units, distances, masks, counts, response and joins verified | Pending |
| V2: golden mock and final-checkpoint parity | Both | Existing licensed mock replay reproduces inputs/predictions/posteriors; context-growth and subdivision pass | Pending V0/V1 |
| V3: bounded Loa trial | GraphWeb_DESI | Preselected supported region including sparse/edge cases; ordered finite draws, flags, input-shift/closure and throughput report | Pending V2 |
| V4: replication and misspecification | Illustris | Additional-phase baseline tests and observation/HOD/epoch robustness; explicit release-scope or retraining decision | Prepare alongside V1 |
| V5: scale-out | GraphWeb_DESI | Restartable uniquely owned shards, hashes, full-footprint QA; provisional research output until release gates close | Pending V3 |
| V6: science release | Both | Model card, validation/systematics report, class reliability, covariance guidance and collaboration review | Pending V4/V5 |

Reversible implementation is authorized. Slurm submissions/modifications still
require explicit approval under workspace rules; prepare concrete commands and
resource requests. Existing phase-access guards remain binding. No publication,
reserved confirmation opening or unrelated E2E scheduler action is implied.

### Exact parity and observation contract

**Mandatory V0-C coordinate gate:** follow
[P12-A coordinate handoff audit](p12a_coordinate_handoff_audit_20260924.md).
The initial inspection confirms Planck18 observer inputs in recorded P10
products, differing from the native-distance convention identified in E2E.
Whether this is a valid fiducial input convention or an actual training/label
error remains unresolved. Bind frozen artifact ancestry and independently
check native host-label closure before the DESI canary.

If training inputs/labels are scientifically wrong, correct immutable products
and retrain the affected encoder/OOF pipeline, regenerate summaries, refit FMPE
and independently revalidate. A Loa-adapter-only mismatch instead requires
adapter correction and golden-mock replay. A consistent input-only fiducial
choice with valid native labels does not automatically require retraining.
Until classified, prepare the adapter but do not pass V2/V3 on assumption.

- Bind the actual production encoder(s), FMPE, OOF provenance, channel order,
  expected counts, normalizers, pooling alignment, interpolation and draw solver.
- Freeze Loa DA2/loa-v1/LSScats/v2.1/PIP or explicitly audit a newer Loa version.
  Retain Kibo-derived mock provenance; Kibo is not a deployment fallback.
- Map M, mu, completeness, support-edge distance, imaging vetoes, redshift quality
  and available systematic weights. Audit PIP/ZFAIL/SYS roles separately; do not
  insert weights or completeness channels into the frozen model without validation.
- Audit the E2E cartesian_v2 distance correction against actual P12 conventions.
  New E2E arrays are not drop-in P12 inputs. Determine whether P12 is affected;
  a required coordinate/input correction demands versioned scientific validation.
- Replay frozen training ntilde exactly, then compare Loa selection. A selection
  refit is a scientific change requiring matched mock validation, not bookkeeping.
- Every TARGETID owns one output core. Overlap reuses context, not independent
  evidence; do not multiply overlapping posteriors.
- Freeze numerical tolerances from existing contracts before scoring. Locate or
  repeat the final-checkpoint parity receipt; a historical canary is insufficient.

### Robustness, population use and release

Use phase/spatial-block uncertainty for physical SBC/TARP, C68/C90, proper scores,
widths and class reliability/Brier, stratified by shell/cap/density/response/edge.
Reuse the registered P12 acceptance contract; do not import D2's looser tolerance.

Challenge redshift failures/errors, fibre/selection realizations, imaging response,
HOD/velocity prescriptions, cosmology and evolution. Measure posterior shifts in
width units and coverage/proper-score degradation where truth exists. Material
failure blocks the affected science domain: restrict a defensible scope or register
retraining plus independent confirmation. Unbounded nuisance effects are not
marginalized uncertainties. Set quantitative intervention tolerances and phase
roles before access; this plan does not invent post-hoc numerical pass thresholds.

P12 does not generate catalogues. A real-data posterior-predictive test needs an
explicit forward observation model; otherwise label it a domain/observational
closure check. Preserve observable-input comparisons and preregistered closure.

Measure residual/class-probability error covariance versus separation and response
on mocks; propagate it into representative population statistics. Effective-N is
statistic-dependent. Independent posterior stacking ignores both spatial errors
and the simulator prior. Build a small-set joint model only if the measured
population error-bar deficit warrants it.

Ship reproducible joint draws or validated compression, quantiles, class
probabilities, information summaries, provenance and support/OOD/prior-domination
bits. Scale idempotently; require restart/identity/checksum checks and review.

## Additional mocks: evaluate first, retrain conditionally

The September-19 qualified set contains 21 paired matter/observation products:

- E2E train: ph000, ph002, ph003, ph007-ph011, ph020-ph024 (13).
- E2E development: ph012-ph013 (2).
- E2E confirmation: ph014-ph019 (6), with reservations/access gates preserved.
- ph001/ph006 excluded and ph004/ph005 historical unused in THAT build. P12
  already opened ph001, selected on ph006 and trained on ph004/ph005: E2E labels
  do not erase prior exposure or make these phases newly blind.

Sources: configs/e2e_coupled_data_v1.json and
docs/e2e_coupled_preparation_closeout_20260919.md. These are c000 z0.2, A+B10%
R7 matter and DA2 BGS_v2 altmtl/kibo-v1 observations. They add independent
structures, not demonstrated HOD/cosmology/epoch diversity.

First implement a cross-programme exposure ledger recording source phase, HOD,
epoch, release, QA, fitting/normalization/selection/prediction access and reserved
role. Eligible unreserved additional phases should test frozen P12 before release;
ph007-ph011/ph020-ph024 are candidate replication data, not automatically fresh
blind evidence. Start with metadata; preserve all reserved target guards.
Build P12-compatible full-cap inputs/galaxy labels from audited sources rather
than reuse E2E normalized paired patches with different grids/targets.

Additional-phase replication should support release, but does not delay V1-V3
for blanket retraining. Retrain only for failed baseline replication, a necessary
observation/coordinate repair, or a registered learning-curve/context comparison
showing useful gains without calibration/sparse-shell regression. Any successor
needs new OOF summaries, training-only transforms, refitted FMPE and untouched
confirmation with explicit cross-programme role allocation. Never reuse ph001
as a fresh blind test or absorb all reserved confirmation into training.

## Why patches, and what a whole-survey model changes

**Registered next-step intent (user confirmed):** after numerical crop parity,
measure residual/proper-score/calibration dependence on large-scale context on
development data. Predefine context scales, spatial resampling, effect-size
thresholds and a finite test budget before scoring. A material missing-information
result licenses a separately registered matched local versus whole-survey/coarse
context comparison, with new posterior fits and fresh confirmation. A null result
does not prove that arbitrarily long modes are irrelevant. Do not use ph001 for
this model-selection step or launch an open-ended architecture sweep.

Patches are not physically necessary. They enabled memory-bounded training,
balanced spatial sampling and blocked validation after random-split transfer
failed. U-PATCH reads context from canonical full-cap fields; output cores are
ownership units, not independent universes.

P6 exposed crop dependence from spatial GroupNorm. The adopted adapter uses
per-voxel channel LayerNorm, globally frozen input transforms and an 8-voxel
pooling phase lock. Historical-canary context convergence begins at 24 voxels
=120 Mpc (not Mpc/h), against an 80-voxel reference: prediction NRMSE .00155,
worst-core .00301 and subdivision .00166. Repeat/locate this suite for the exact
deployed checkpoints. These results do not prove physical context sufficiency.

Distinguish three issues:

1. Numerical tiling: enough halo, aligned strides and matching padding,
   normalization/interpolation can reproduce a larger finite-receptive-field
   network forward pass. Real survey boundaries still require support flags.
2. Information horizon: a larger input array alone does not enlarge a fixed local
   network's receptive field. Truly using survey-scale information requires
   global/coarse conditioning or architecture changes. Missing external tidal
   shear can remain after tiling converges; R7 smoothing does not make gravity
   local. There is no simple hard wavelength cutoff at the output-core size.
3. Joint uncertainty: whole-survey deterministic encoding plus independent
   per-galaxy FMPE draws still does not supply a spatially joint posterior.

First pass numerical parity. Separately diagnose errors/calibration against
large-scale density/shear on development mocks; truth summaries are diagnostics,
not deployment inputs. If material, register one matched local versus survey-scale
or coarse-context challenger, same targets/observations/phase roles and adequate
optimization. Refit both posterior heads and compare calibrated proper scores
and downstream errors, not R2 alone.

The whole-volume cost note used 6.766 Mpc/h cells, a different redshift interval
and rectangles for one score evaluation. It is not a measured whole-cap P12
training benchmark at 5 Mpc cells. Benchmark actual bounding-box tensors, halos,
activations/backpropagation before claiming a whole-survey replacement fits.

References: P6 in plan_generalisable_graphweb_vac.md; overlap tiling in
https://arxiv.org/abs/1505.04597. Image-border mirroring is not permission to
fabricate tracers outside the DESI mask.

## Disposition and immediate checklist

Generalisable plan: preserve P0-P12 contracts/history, restart P13 via this plan.
Old environmental roadmap: retired execution schedule. Field/multimodal and G4:
archived architecture agendas. P12-B and completed P12-F rescues: closed, no
automatic restart. D2: read-only terminal-artifact reconciliation, separate from
VAC. E2E: existing bounded work/ownership and stop rules preserved, no automatic
larger campaign. GraphWeb_DESI: U-PATCH+P12-A handoff supersedes G3 deployment.

- [x] Adopt plan, mock-use decision and patch/context audit requirements.
- [x] Begin V0 source/recorded-product coordinate audit; save evidence and decision tree.
- [ ] V0-C numerical input/host-label closure and explicit correction/retraining decision.
- [ ] V0 artifact manifest and cross-programme exposure ledger.
- [ ] V1 adapter with schema/identity tests.
- [ ] V2 frozen criteria, executable commands and measured resource request.
- [ ] V2/V3 execution and V4 validation under applicable approvals.
- [ ] V5/V6 complete; no public production-validity claim yet.
