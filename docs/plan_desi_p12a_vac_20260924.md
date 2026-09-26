# P12-A to DESI Loa environmental-posterior VAC

Active plan, adopted 2026-09-24 following the user's approval of the roadmap
disposition and request to proceed toward DESI application. SCIENCE_LOG.md and
immutable experiment contracts control claims. This supersedes old prospective
VAC schedules, not historical evidence. V0 audits, halo48 OOF/refit/revalidation and golden replay have passed.
The first real-Loa bounded provisional VAC is produced and independently checked;
full-footprint production and science release remain open (2026-09-25).

## Decision and product

Proceed with frozen U-PATCH weights and the versioned halo48 context correction,
regenerated OOF summaries and a refitted untempered P12-A posterior through
validation, a golden mock and a bounded Loa trial. Do not retrain merely because additional
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
| V0: handoff freeze | Illustris | Checkpoint/source/transforms/input/solver manifest; exact support, quality bits and coordinate audit | Bounded handoff closed; retained frozen P12 coordinate/selection convention |
| V1: Loa crosswalk | GraphWeb_DESI | Hashed release and mock-to-Loa schema; units, distances, masks, counts, response and joins verified | Source and adapter passed; new real Loa inputs built |
| V2: golden mock and final-checkpoint parity | Both | Existing licensed mock replay reproduces inputs/predictions/posteriors; context-growth and subdivision pass | Passed: eight golden cap/shell cores and halo48 context controls |
| V3: bounded Loa trial | GraphWeb_DESI | Preselected supported region including sparse/edge cases; ordered finite draws, flags, input-shift/closure and throughput report | Technical trial complete; input/cut audit done, unresolved high-z selection shift retained |
| V4: replication and misspecification | Illustris | Additional-phase baseline tests and observation/HOD/epoch robustness; explicit release-scope or retraining decision | Prepare alongside V1 |
| V5: scale-out | GraphWeb_DESI | Restartable uniquely owned shards, hashes, full-footprint QA; provisional research output until release gates close | Technically complete: 5,436,413 rows; provisional due to unresolved selection shift |
| V6: science release | Both | Model card, validation/systematics report, class reliability, covariance guidance and collaboration review | Pending V4/V5 |

Reversible implementation and all VAC production compute are explicitly
authorized by the user. Use bounded resource requests and preserve receipts. Existing phase-access guards remain binding. No publication,
reserved confirmation opening or unrelated E2E scheduler action is implied.

### Exact parity and observation contract

**Mandatory V0-C coordinate gate:** follow
[P12-A coordinate handoff audit](p12a_coordinate_handoff_audit_20260924.md).
The initial inspection confirms Planck18 observer inputs in recorded P10
products, differing from the native-distance convention identified in E2E.
The coordinate, population and lineage checks retained this as a consistent
fiducial input convention with native labels; the adapter reproduces the exact
historical radius lookup. Corrected E2E inputs remain incompatible replacements.
See the coordinate/handoff and golden reports; this gate is closed for the trial.

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

Execution instruction (2026-09-24): all VAC compute is approved. Continue the
remaining checks autonomously; return to the user for scientific choices,
blocking issues, or readiness to launch real-DESI inference. Do not repeatedly
ask for routine scheduler approval or treat a partial audit as deployment clearance.

Next ph000 audit uses the original direct Planck18/SkyCoord implementation
(not the newer interpolation). Before inspection: require <=1e-9 Mpc point
replay error, exact cap identity, full SHA256 equality of original/imported
catalogues and points, the manifest's recorded catalogue SHA, and the same
128-row/eight-stratum native x_com label test used for ph002-005. Do not assume
TARGETID-1 indexes an older parent. Existing helper tests cover native mapping
and geometry-only sampling. Bound execution to one CPU node/8CPUs/32GiB/30min;
source entry point `workflows/abacus_tweb/p12a_handoff_followup.py --mode ph000`.

Generalisable plan: preserve P0-P12 contracts/history, restart P13 via this plan.
Old environmental roadmap: retired execution schedule. Field/multimodal and G4:
archived architecture agendas. P12-B and completed P12-F rescues: closed, no
automatic restart. D2: read-only terminal-artifact reconciliation, separate from
VAC. E2E: existing bounded work/ownership and stop rules preserved, no automatic
larger campaign. GraphWeb_DESI: U-PATCH+P12-A handoff supersedes G3 deployment.

- [x] Adopt plan, mock-use decision and patch/context audit requirements.
- [x] Begin V0 source/recorded-product coordinate audit; save evidence and decision tree.
- [x] Implement bounded artifact preflight and conservative phase-ledger generator
  in GraphWeb_DESI/workflows/catalog/p12a_vac_preflight.py; five fixture tests pass.
- [x] Run live preflight and archive GraphWeb_DESI/docs/p12a_vac_preflight_report_20260924.json:
  eight candidate artifact sizes/hashes and feature/checkpoint bindings pass.
  Exit 2 is expected because DESI canary/release gates remain open. The prior
  command failure was a missing Codex sandbox-wrapper executable, not Slurm or
  Python; see the 2026-09-24 ops/handoff SCIENCE_LOG entry.
- [ ] V0-C numerical input/host-label closure and explicit correction/retraining decision.
  First training-only sampled audit implemented; nine synthetic tests pass.
  Frozen ph002 input/source inventory and concrete 30-minute CPU run request:
  `docs/p12a_coordinate_sample_run_20260924.md`. User approved all VAC compute;
  job58823054 completed ph002-005 with all partial checks passing (65,536
  observed rows and512 native labels total). Reports and limits:
  `docs/p12a_coordinate_sample_results_20260924.md`. Imported ph000 and remaining
  lineage/response/Loa checks are next; these partial audits do not close V0-C.
- [ ] V0 artifact manifest and cross-programme exposure ledger.
- [ ] V1 adapter with schema/identity tests.
- [ ] V2 frozen criteria, executable commands and measured resource request.
- [ ] V2/V3 execution and V4 validation under applicable approvals.
- [ ] V5/V6 complete; no public production-validity claim yet.

### Follow-up evidence (2026-09-24)

ph000 import/numerical closure, six checkpoint embedded-transform checks,
ph002-005 annotation-run x_com evidence and bounded live Loa source revalidation
completed. See `p12a_handoff_followup_results_20260924.md` for exact scope and
remaining gaps. This supersedes the earlier checklist's "imported ph000 next"
status; V0-C as a whole remains open. Before golden mock: resolve population
coverage, pin Loa BGS success policy (legacy >=25 is not the installed LSS >40
rule), replay both R0 derived channels and P3b posterior response, and verify
final-checkpoint behavior against recovered training/export source lineage.

### Bounded replay criteria frozen before execution

Run `workflows/sbi/p12a_frozen_mock_replay.py` on training ph002 only: select
highest-row-count core deterministically in each cap/shell, before reading any
prediction errors. Replay omitted-ph002 encoder against stored predictions
(atol=rtol=1e-5, float32 numerical agreement), ntilde against archived response
(exact float32), and P3b support/boundary against cached values (exact).
For the production full-fit encoder, compare halo24 vs48 and aligned two-way
core subdivision using existing P6_GATES, without changing the deployed halo.
This is a bounded reference-path check; it does not replace a rebuilt Loa
adapter golden test, sparse/edge controls, posterior replay or release gates.
No P12 coordinate replacement, model fitting or new phase access is authorized
by a passing result. Preserve failures without retuning tolerances.

First replay retained in P12A_FROZEN_MOCK_REPLAY_20260924.json: ntilde and P3b
response replay pass; encoder stored-output agreement fails at max~3.72e-4
with cuDNN TF32 disabled by the audit, unlike the exporter's default. Retry
with exporter-matching cuDNN TF32 enabled (same numeric thresholds), preserving
both reports. Full-fit halo24->48/subdivision failures are much larger than
that precision discrepancy. On the preselected lowest-shell core in each cap,
add diagnostic halo48->64 comparisons; do not adopt a different deployment halo
or relax original gates based on these diagnostics.

Halo48 diagnostic passed on the eight selected dense cores. Before considering
promotion, extend the same unchanged thresholds to sixteen geometry-selected
ph002 stress controls: smallest populated core (>=16 rows) per cap/shell, and
largest fraction of rows within the existing P6 support-boundary distance per
cap/shell (ties by row count then core ID). Use archived random-support distances,
not truth or errors, for selection. Test halo48->64 and aligned subdivision.
This remains a frozen-weight diagnostic; calibrated halo24 FMPE is not silently
reused with halo48 summaries. Changing context requires refreshed OOF summaries,
posterior fitting and independent confirmation before a replacement can deploy.

### New blocking result and minimal correction path

The matched-precision OOF replay is exact (14,230 ph002 rows), and sampled
selection/random-support response replays pass. However, production halo24
fails registered context-growth/subdivision gates. This is independent of the
native/fiducial distance comparison and of DESI quality-cut choice. Inference
readiness remains false. Evidence: `p12a_population_response_replay_results_20260924.md`.

A frozen-weight halo48 candidate passes the same criteria on24 dense/sparse/edge
cores. Keep original P12-A artifacts immutable. Minimal next scientific work:
1. Pin a versioned larger-context candidate and reproduce its input fields.
2. Regenerate all five OOF exports and the full-fit validation export using the
   same context policy, preserving phase roles and epoch20 checkpoint choices.
3. Refit FMPE with unchanged feature/target definitions and compare registered
   calibration/information criteria; reusing the old FMPE is not qualified.
4. Complete independent confirmation under the exposure ledger, then the Loa
   adapter golden replay. Only then declare readiness for the bounded DESI trial.
Encoder weight retraining is conditional on failure of this smaller correction;
whole-survey architecture/E2E substitution is not the default remedy.

The existing DESI catalogue passes its advertised ZWARN0/DELTACHI2>=25/GALAXY
cuts. A different installed LSS cut does not itself require retraining or force
changing this choice. Freeze real observed sample and response consistently;
record threshold and spectral-class changes separately.

Concrete candidate execution and stop rules are now in
`p12a_halo48_candidate_plan_20260924.md`. This correction is motivated by the
actual final-checkpoint gate failure, not a preference for larger patches or
whole-survey encoding. No fitting/re-export has yet been launched.

Population update: raw bright parent census now complete for all five training
phases; no RES=0 objects occur in the VAC target range. That absence is now
explained and recorded, rather than left as a missing target-population test.
Halo24 context dependence is the active scientific blocker; candidate48 has
passed24 geometry-selected dense/sparse/edge controls, with calibration pending.


## Current execution: approved halo48 correction

The candidate plan `p12a_halo48_candidate_plan_20260924.md` now records the
submitted chain: export array58827028 (six phases, concurrency2), refit58827034,
and calibration audit58827036, with after-success dependencies. Six bounded
phase benchmarks and GPU posterior reconstruction pass; 13 relevant unit tests
pass. Frozen source/manifest receipts are under `docs/evidence/p12/HALO48_*`.
V2 is not closed: scientific comparison and Loa adapter golden replay remain
required. Original halo24 evidence remains historical, not candidate validation.
The user should next be consulted for a scientific issue or when ready for
bounded DESI inference, not for routine already-authorized VAC compute.


Scheduler recovery, 2026-09-24: ph000 and ph002 exports completed successfully
(9,956,825 rows total). Remaining tasks had a stale-looking JobArrayTaskLimit
reason despite no running task; refreshing the unchanged two-task throttle
cleared it, with tasks2/3 subsequently waiting on Priority. Existing export,
refit and audit job IDs/dependencies remain in place. No repeat export or
interactive allocation needed. See `docs/evidence/p12/HALO48_SCHEDULER_RECOVERY_20260924.json`.


## Optimized resubmission (supersedes original pending export IDs)

At the user's request, pending array tasks2-5 were replaced by ordinary named
jobs: ph003 **58831285**, ph004 **58831286**, ph005 **58831287** (after ph003),
ph006 **58831289** (after ph004). Each requests one GPU,32CPUs,16GiB and90min;
two may run together. The completed ph000/ph002 products are retained.
Actual completed export times60-65min and MaxRSS<2.7GiB support the tighter
requests, with runtime and memory headroom. Scientific configuration and frozen
source hashes are unchanged. Refit **58827034** depends on all four new jobs;
audit **58827036** follows successful refit. Jobs are accepted/released but
queue priority/resource availability still determines start time. Receipt:
`docs/evidence/p12/HALO48_OPTIMIZED_RESUBMISSION_20260924.json`.


Completion update2026-09-25: all halo48 production jobs succeeded. See
`p12a_halo48_readiness_20260925.md` for measured overcoverage and remaining
conditional-acceptance/Loa-golden/confirmation-eligibility work. This supersedes
queued status; readiness remains false. No automatic encoder retraining.


2026-09-25: conditional coverage and eight-core observer golden replay pass;
Loa full-data/random content hashes verified. Supersedes pending replay status.
Next is bounded Loa input construction/QA then diagnostic inference; no further
retraining indicated. Full release and independent confirmation remain open.
See Illustris `docs/p12a_golden_conditional_results_20260925.md` and the explicit
bounded-trial handoff manifest. Real Loa inputs/posteriors are not yet produced.


### 2026-09-25 real-Loa trial completion and V5 handoff

Bounded provisional VAC now exists:5,615 rows across16 preselected cores;
5,602 supported rows each have512 ordered joint draws and derived class
probabilities,13 null/flagged. Product QA passes. Full target input catalogue
contains5,436,413 science-range galaxies with6,680,148 context galaxies.
Evidence: `docs/evidence/p12/LOA_CANARY_20260925/` and the observation report
`GraphWeb_DESI/docs/p12a_loa_canary_20260925.md`.

Next authorized work: (1) quantify count/selection and conditional feature
shifts across shell/cap; feature-envelope inclusion alone is insufficient;
(2) benchmark dense/edge production shards with full context, freeze footprint,
unique ownership, versioned flags and atomic restart contract; (3) submit bounded
batch scale-out using measured resources, validate merged TARGETID census and
shard checksums. Preserve unsupported rows/nulls and provisional designation.
No additional compute permission is needed. V4 independent replication and
misspecification, and V6 science release remain separate unfinished gates.


### Full-survey input audit and execution contract (2026-09-25)

Count/selection and conditional-input diagnostics now exist. The high-z excess
survives common pixels, stricter magnitude/DELTACHI2 and assignment accounting;
see LOA_FULL_20260925 evidence. This is an unresolved science applicability issue,
not a technical-input pass or calibration claim. No silent selection correction.
Proceed only as a provisional full-footprint model-conditional VAC, with explicit
z>=.35 selection-shift flag64 and provisional flag32 on all rows. Before release,
resolve upstream targeting/photometric/n(z)/observation differences and validate
any correction on mocks. No phase reservations changed.

Implementation now has atomic per-core completion markers, stable global-core
seeds, hash-verified resume,128 balanced parts and independent draw/identity merge
validation. Dense/edge benchmark and exact interrupted-aggregate recovery pass.
The final merge must enumerate exactly5,436,413 unique TARGETIDs, preserve null
unsupported rows, and reproduce every saved summary from its512 joint draws.

### 2026-09-25 selection diagnosis after full VAC completion

V5 technical production is complete; earlier queued/canary-only status above is
superseded. V4/V6 scientific qualification remains open. Executed the authorized
matched-sky ph002–006 census, ph006 truth/encoder/posterior comparison and paired
selection tests. Report: GraphWeb_DESI/docs/p12a_selection_diagnostics_20260925.md.
Common-sky DESI excess is8.68% overall,≈1.91–1.97x atz=.45–.55. Even raw bright
CutSky before assignment has fewer high-z galaxies than DESI successes; prioritize
upstream apparent-magnitude/luminosity-evolution/photometric-system equivalence.
No downstream accidental cut identified. The environment redshift trend exists
in ph006 truth as well; do not flatten it or claim physical evolution.

Next: pin actual upstream photometric recipe and compare magnitude/colour at
fixed z/cap; register a physically justified population/observation correction;
rebuild selection and fields consistently; representative truth-known conditional
coverage and observable-closure tests; OOF/posterior replacement if summaries
change, encoder retraining only if indicated. Independent phase eligibility and
reserved-phase protection persist. No automatic empirical DESI n(z) substitution:
paired tests demonstrate its potential impact, not its scientific validity.
Only then version/recompute a replacement VAC and assess science-release gates.


### 2026-09-25 priority clarification: prove selection alignment first

Do not assume a population correction is needed before completing the selection
crosswalk. Current exact-alignment status is **unverified**. Latest audit and
specific missing provenance: GraphWeb_DESI/docs/p12a_targeting_alignment_20260925.md.
Complete original Loa targeting/Gaia/SGA join and targeting-version replay;
pin official mock preparation/altMTL/masks and CutSky photometric recipe; compare
stagewise retention and conditional photometry. The new high-z colour difference
survives fine-z composition matching and a common numerical r limit, but passband
alignment is unresolved. Only demonstrated defects justify repairs and subsequent
rebuild/revalidation. No imposed count adjustment is proposed or authorized by
these diagnostics. Preserve completed provisional VAC and qualification flags.


### 2026-09-26 diagnostic closure update — no repairs authorized

User explicitly requests closure points1–3 only. Selected-DESI historical
DR9/1.1.1 targeting replay now passes all5,436,413rows including Gaia/SGA;
original g/r/z fluxes unchanged. Kibo and Loa are both three-year DR2 reductions;
matching actual Loa mocks still leaves high-shell1.856/1.839 count ratios.
Mock stagewise true-z parent/assignment joins completed. Report and open
provenance requirements: GraphWeb_DESI/docs/p12a_selection_closure_20260926.md.
Next is obtaining the official raw-CutSky generator code/config/receipt and
executed catalogue mask/observation recipes, then completing conditional
population/completeness interpretation. A located FA script/version mismatch
must be resolved before declaring exact production lineage. No repairs, changed
cuts, retraining or VAC replacement are authorized by the current diagnostic
request. Preserve provisional VAC; do not substitute Loa mocks silently.


### 2026-09-26 follow-up: numerical recipe and conditional photometry

Completed diagnostic fingerprint and SGC matching in z/apparent magnitude;
see GraphWeb_DESI/docs/p12a_photometry_recipe_20260926.md. Published Abacus
GAMA g/r tables reproduce actual ph006 colour to floating precision. Current
v0.1 stored magnitudes require a -0.8(z-0.1) relation beyond M+DM+Kr; archived
v0.1/old differs (~-1.6 term, older distance behavior). These are measured
identities, not authorization to alter magnitudes or assert their motivation.
The high-shell colour offset remains +0.157mag after fine z/r standardization
on southern imaging. FA version mismatch is present in five date-spaced samples.

- [x] Identify the g/r numerical conversion, interpolation and redshift argument.
- [x] Test whether fine z/r composition and mixed real PHOTSYS explain the offset.
- [x] Establish whether the archived FA version discrepancy is isolated (it is not in the five samples).
- [ ] Pin the producer receipt for stored absolute-magnitude evolution, the -0.8
  term, LF/colour distributions, and the current-versus-archived transformations.
  Audit imported ph000 photometric revision if the receipt reveals changed rules.
- [ ] Compare same-target DESI synthetic SDSS/Legacy photometry and distinguish
  passband effects from Petrosian/model flux estimation; then compare intrinsic
  LF/rest-colour populations using consistent conventions.
- [ ] Recover executed FA and full-catalogue mask receipts; script module requests
  do not certify actual versions. Keep header5.7.2.dev3588 evidence explicit.

No repair, retraining or new VAC inference is authorized by this diagnostic
step. Existing production VAC remains provisional; no arbitrary offset fitting.


### 2026-09-26: module failure reproduced and paired passband pilot

See GraphWeb_DESI/docs/p12a_passband_followup_20260926.md. A reproducibility
record means executed command/code/config/input/log evidence, not a required
special DESI file. Using the retained module tree in an isolated shell, requests
for fiberassign4.0.0 and5.0.0 fail and leave the main fba_run executable selected.
This reproduces the proposed mechanism; historical stderr remains unavailable.
Version5.7.2.dev3588 spans Feb2024–Apr2025 in Git history, consistent with retained
Oct2024 output mtimes, but does not pin a unique source revision.

- [x] Paired synthetic SDSS/DECam passbands for512 common-sky Loa VAC targets
  using FastSpecFit3.1.5/speclite0.20, replayed saved model colours <2e-5mag.
  High-shell SDSS-minus-DECam r=+.246mag, colour=-.089mag. Restricted better-fit
  subsets retain similar signs/scales. One hp00 stratified pilot, not global QA.
- [x] Locate actual SDSS Petrosian/model imaging and perform bounded unique-ID
  positional crossmatch:83 pairs,60 with g/r Petrosian SNR>=5, only6 high-z.
  Magnitudes differ substantially, but spatial, quality and dust-convention
  limitations preclude a population-wide mapping or correction.
- [ ] Extend representative imaging comparisons with harmonized extinction,
  clean flags and size/surface-brightness diagnostics; separate passband and
  flux-estimator effects before intrinsic LF/colour comparisons.
- [ ] Resolve whether the mock -0.8(z-0.1) term already implements a photometric
  adjustment or is only an absolute-magnitude convention. Do not apply a second
  mapping without this evidence. Then replay physical parent selections versus z.

Audit convention fixed before reporting: FastSpecFit output FLUX is already
de-reddened, unlike Loa LSS FLUX. Initial double-extinction diagnostic was rerun;
production inputs and VAC are unaffected.

No repair or VAC replacement. Historical FA attribution and exact raw-magnitude
semantics remain open; the new passband evidence makes selection-definition
mismatch a concrete candidate, not a quantified explanation of the full counts.


### 2026-09-26 n(z) and regeneration disposition

See GraphWeb_DESI/docs/p12a_nz_regeneration_assessment_20260926.md. Checked BGS
preparation bypasses the dark-tracer nz STATUS mask. Public BGS parent has an
LF/colour/flux-limit radial model, not a located external n(z) input. Missing
thinning cannot alone explain a deficient raw bright parent. Pin the exact
user-intended generator/option before interpreting a supplied n(z).

Conditional saved-SED test: if the existing -.8(z-.1) is an observer-magnitude
adjustment, it already offsets the filter difference (high-shell residual-.076mag).
If it defines stored evolution-corrected M, that causal interpretation is invalid.
Actual column semantics remain a prerequisite to any regeneration/extra mapping.

Next experiment: reproduce an existing exposed phase with pinned input/recipe,
then alter only an established faulty selection or physical-population layer.
Reuse existing gravity/halo products. Require joint population, clustering and
conditional coverage evidence, not only an n(z) match. No full-suite regeneration
or repair launched/authorized by this diagnostic assessment. Rebuilt training
observations require fields, OOF summaries, posterior refit and independent
revalidation; encoder retraining remains a controlled comparison decision.

Confirmed after broader search: LSS calibrate_nz_bgs accepts nzfile and
prepare_holi_bgs.py uses Loa BGS n(z). This is a separate pipeline, not found in
our Abacus path. Its random thinning cannot upsample a deficient parent and
its placeholder magnitude categories are not a validated substitute for the
Abacus luminosity/halo population. User's general recollection was correct;
no further entrypoint clarification is required to establish that capability.

### 2026-09-26 authoritative LSS Abacus follow-up

User identifies desihub/LSS as the processing authority. Upstream commit
d942b990860e016e7558c740966fea3e5ce7e7f3 inspected and snapshotted in GraphWeb
LSS_UPSTREAM.json. mkCat_amtl --nz y measures n(z)/adds weights; it does not
match counts to an input n(z). Loa Abacus wrapper distinguishes plain BRIGHT
full products from BRIGHT-02 with a z-dependent absolute-magnitude threshold.
Y3-named preparation wrapper actually calls Y1 code with downsampling n and
rbandcut 19.5. Pin executed commands, footprint and sample suffix before causal
attribution; current upstream is not proof of the historical execution.
No repair/regeneration. Details in the n(z) regeneration assessment.

Audit implementation/evidence committed in GraphWeb_DESI 1078042; 13 lightweight
P12-A tests and 15 source snapshot hash checks pass. Scientific selection closure
remains open. No catalogue repair, regeneration or inference launched this turn.
