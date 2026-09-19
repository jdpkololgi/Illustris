# Coupled-field VDM/CFM: approved data preparation

**Completed qualification,2026-09-19:** all21phase products and independent
audits pass, with1,792normalized pair domains and11,776offset cases. The exact
13-phase normalizer, physical-reference gate,7synthetic GPU cases and measured
loader/postprocessing costs pass. All preparation jobs are terminal. Final cost:
16.310833CPU-nodeh,0.141667GPUh,4,311,094,964,561newScratch bytes, within the
original caps. The [closeout](e2e_coupled_preparation_closeout_20260919.md) and
[evidence archive](evidence/e2e_coupled_20260919/README.md) are the handoff;
the [measured proposal](e2e_coupled_resource_proposal_20260918.md) requires
separate scientific-run approval. No scientific fit or confirmation prediction.

## Historical execution record and preparation contract

The following dated inventories are preserved history, not current readiness.

03:20 UTC: all21native matter fields and21condition sets are complete.015/016
finish with count-conservation errors below1e-7. Fresh-process400GiB target
recovery begins012, with014/015/016next; audited/normalized phases remain17/21.
All123focused tests pass. Final data index, requirement audit/evidence archive
and Git commits are still pending. No scientific fit or confirmation prediction.

03:06 UTC recovery update: superseding`target_recovery_v2` starts58562167 for2h
in the slot freed by the completed GPU check. The unused v1 controller is
stopped; no Slurm job is cancelled/modified. Terminal target-step OOM plus
healthy named workers gates this early start; native per-phase locks protect
015/016 concurrency. Only replacement audit/interface publishers wait for the
old parent to terminate cleanly apart from its reviewed OOM. This preserves
single-writer publication while allowing independent native/FFT work now.
All numerical kernels/data gates and the original resource caps are unchanged.

02:57 UTC:19/21native,21/21condition sets,17/21audited target sets and17/21
normalized interfaces are complete. All13training phases are qualified and
the final13-phase normalizer/26-pair actual-loader benchmark pass. GPU58559826
completes all7technical factor cases with exact fresh-process replay in8m30s;
no scientific fit occurs. The combined measured resource projection now exists.
Sampling, not training, dominates the proposed later experiment's cost.

Target step58552205.0 hits OOM while starting014's FFT after completing013.
Completed products are retained; the other lanes continue. The clean-predecessor
final-data controller will reject that failure, not restart its288GiB target
configuration. A separately reviewed/frozen`target_recovery_v1` waits for the
whole predecessor to stop and verifies its exact step identities/exit states.
Its single2hCPU request remains inside the original cap: native gridding FIRST,
then400GiB fresh-process target builds plus two8GiB qualification lanes. Numerical
kernels/previous products remain unchanged. No automatic retry or new scientific
decision occurs after disconnect. Final all-panel release and committed closeout
remain outstanding. Detailed receipts/recovery evidence are in SCIENCE_LOG.

02:20 UTC milestone: all13train-moment sets and the global normalizer are
complete. Exact recomputation from the13cached, source-bound moment receipts
matches the saved transform, SHA256
`b2b294c5de91a7659be16b87e2ea454a11464fbd4522b4c0d30d2b1effa9e43e`.
ph022's independent audit passes, taking audits to16/21. Normalized interfaces
for000/002/003/007 each pass128pairs/896offsets at about1e-7maximum density
round-trip error. Full-panel qualification, actual loader/GPU measurements,
resource-proposal finalization and committed closeout remain outstanding.
The final normalizer is real; the all-panel data-release index is still absent.

02:03 UTC inventory:18/21native fields,19/21conditions,15/21audited targets,
12/13train-statistics sets.022 target/audit is the remaining train-normalizer
dependency. Synthetic CPU postprocessing benchmark58550091.7 completes0:0
within the existing allocation, without scientific payloads or fitting; its
measured components are incorporated into the resource-proposal accounting.
All112focused tests pass. GPU and normalized-loader measurements remain pending.
Usage:13.1475CPU-nodeh,0GPUh,4,229,616,338,827newScratch bytes. Both CPU jobs
continue; the goal and final committed closeout remain incomplete.

01:40 UTC delta: native014 passes count conservation (9.5053e-8 relative error),
giving16/21 native fields. ph011 passes its128-pair/608-hash independent audit
in227.04s, SHA256 `0d3e0680897639ebc885de486652eea18c9f2a3d08156cf37ba9d9111d59671f`.
Audits now15/21; eleven train-moment sets were present at the latest check,
with011 newly eligible.022 is the last training phase awaiting audited products.
Both CPU jobs remain live. Usage at01:33UTC:12.196389CPU-nodeh,0GPUh,
4,121,430,991,491newScratch bytes; original caps and no-training scope unchanged.

01:30 UTC delta: ph024 passes the independent128-pair/608-hash audit in201.68s,
SHA256 `11feeb4dfe37648695207ce3b6ce0f5da08b270ffcbc00d881c2a4e5949ef3b7`.
Current audit/source binding is verified; audits now14/21 and train moments11/13.
The full normalizer still awaits011/022. Both CPU jobs remain live; no technical
GPU or scientific fit has started. The resource proposal now records the exact
finite-ensemble coverage references and mean/residual power accounting, without
claiming posterior performance or finalizing thresholds from confirmation data.

01:23 UTC delta: native011 completes, giving15/21 native fields; conditions
now cover17/21. Audits remain13/21 and training moments10/13. Both CPU jobs are
verified live. The later resource proposal now explicitly counts development
checkpoint, sampler-ladder and fixed/oracle-coarse panels (45,056 fields), with
eight passing accounting tests. These are proposed counts, not GPU timings or
scientific launch authority. No running immutable source snapshot was changed.

Current inventory (2026-09-19, 01:20 UTC): 14/21 native fields, 15/21 condition
sets, 13/21 independently audited target sets and 10/13 train-moment sets.
All 103 focused tests pass; all thirteen completed audit bindings are rechecked.
The real data-release index remains absent. Both CPU jobs continue; GPU timing
and the full normalizer/normalized interface are still outstanding. Latest
accounting is 11.579722 CPU-node-hours, zero GPU-hours, with the subsequent
disk check at 4,113,161,609,636 bytes under the original cap.
Mac/login04 tmux `coupled-final-data-controller` now waits for 58552205's verified
terminal state and a free slot before one bounded four-hour CPU continuation
from `final_data_continue_v1`. It finishes checkpointed products and publishes
the data-only index only after all required qualifications. It cannot authorize
scientific fitting or substitute for the technical-GPU/resource-proposal gate.
The technical-GPU controller remains separately waiting for 58550091.
The observational scope is detailed below: DESI-like selection is already
inherited from upstream observed mocks; real-DESI validation is not established.
Earlier dated inventories are historical snapshots.

Current inventory (2026-09-19,00:46UTC):all21joins/17B restores,14native fields,
15corrected condition sets,10independently audited target phases and8/13train
moment sets. No final normalizer or all-panel normalized interface yet. Physical
reference gate PASS; GPU measurement remains unrun. All93focused tests pass.
Five additional resource-accounting arithmetic/guard tests also pass; no measured
cost report is produced without the actual GPU/loader receipts. Usage at00:45UTC:
10.575833CPU-nodeh/0GPUh/3,931,253,849,858newScratch bytes. The frozen
`gpu_benchmark_v3` adds actual host-transfer/checkpoint overhead measurement.
One bounded Mac/login04 tmux controller waits for58550091's verified terminal
state and a free slot before requesting one shared A10080 for one hour. It
does not create a third allocation or authorize a scientific fit.

58544227 is terminal: planned75:0 pauses for the live product lanes; overall1:0
from the earlier documented and recovered adoption/geometry errors. A bounded
guard let attached workers checkpoint before resuming the old wrapper. After
manual terminal/recovery review,58552205 starts onnid004157 under Mac/login04
tmux `coupled-products02-controller`, frozen `products_continue_02_v3`.
Native017 resumes from file92 at93; additional013/016 native work shares existing
per-phase locks. Six wrapper-owned steps reserve472GiB/128logical CPUs, including
a serial audit/statistics tail that waits for58550091.3/.4 to stop before writing.
58550091 continues the other lanes; exactly two allocations remain active.
The new data-only qualification publisher is implemented, NOT issued: all21
audits/interfaces and13-phase normalization are mandatory. It does not replace
GPU/resource-proposal qualification or authorize scientific training.
All dated inventories below are historical snapshots.

Current inventory (23:26UTC):10native fields,10response/condition sets and7
independently audited target phases; train moments6/13. Native018/021 and
the019target/audit are newly complete. All75tests pass. Two CPU allocations
continue; no scientific fit or GPU run. Usage8.05139CPU-nodeh/0GPUh and
3,661,803,699,095newScratch bytes. The NEXT product-node snapshot is
`products_continue_02_v2`, not v1:472GiB/124logical CPUs, with a runtime actual-
allocation memory/CPU guard. The never-launched488GiB v1 exceeds the observed
487802MiB allocation and is superseded; its sources are preserved. Full-panel
normalization and interface/technical-cost checks remain outstanding.

Current inventory (23:12UTC): all21joins/17B restores,8native fields,
10corrected response/condition sets and6independently audited joint-target
sets complete. Train moments6/13; final normalization is deliberately absent.
The009 independent audit passes128pairs/608hashes. Native010and021 are verified
resumed from file33and105 checkpoints in58550091, which replaces58540511 after
its planned75:0pause. The other node58544227 builds019targets while remaining
native and survey-response work continues. Exactly two allocations; no fits.
The full physical-reference gate remains PASS, not posterior-calibration evidence.

All71 focused tests pass. GPU benchmark `gpu_benchmark_v1` is frozen but unrun.
Actual normalized-loader timing and incremental interface qualification are
implemented in `products_continue_02_v1`, also not yet run. The latter snapshot
contains the next bounded product-node wrapper (488GiB/124logical CPUs), which
owns and waits for all its steps; no successor has yet been requested for58544227.
New normalized-loader timing requires the COMPLETE13-phase transform and will
not substitute synthetic throughput or an invented cache speedup. Usage at
23:12UTC:7.58806CPU-nodeh,0GPUh,3,593,018,529,838newScratch bytes.
Earlier timestamped inventories below are historical snapshots.

Latest inventory (22:38UTC):6native,9response/condition and5audited joint-target
phases complete; train statistics5/13. New003target/audit/statistics complete.
All64 focused tests pass. A full-size synthetic CPU smoke validates8cases of
the rectangular-aware VDM/CFM engineering backbone, including spatial-context
gradient paths and fine-subspace preservation. This uses0optimizer updates;
it is neither a scientific fit nor the outstanding GPU cost/replay measurement.
The freed legacy-observation slot now builds019on58544227.10 without a new
allocation. The old audit and observation queues pause75:0as registered;
their successor is still waiting for58540511 to end. Preparation usage is
6.45417CPU-nodeh/0GPUh and3,546,153,585,698newScratch bytes.

Latest inventory (22:16UTC):6native,9response/condition and4joint-target phases
complete. The4target phases000/002/007/008 also pass independent audits and
have train-statistics receipts;003targets are building. All58 focused tests pass.
Full normalized-interface qualification is implemented for all1,792pairs and
11,776registered offset cases, but awaits the complete13-phase normalizer.
An actual targetless-reader IO smoke passes all128ph007 pairs in6.84s; that
smoke is explicitly not normalized/global readiness. The single frozen
`coupled-prep03-controller` onlogin04 waits for planned termination of58540511
before one bounded four-hour CPU continuation; it cannot create a third
allocation or automatically retry an unexpected failure. Current charged usage:
5.74806CPU-nodeh/0GPUh;3,465,220,532,698newScratch bytes.

Latest inventory (22:02UTC): all21 exact source joins and all17 approved B
transfers complete;6native phases,7corrected response/condition phases and3
full joint-target phases complete. The latter000/007/008 independently pass
128pair/608hash audits each. Native CRC qualification remains phase-specific.
Restartable train-only normalization statistics now exist for those3phases,
but publication requires ALL13training phases. No partial usable normalizer,
scientific fit or claim of full data readiness is permitted. All54 focused
tests pass. The bounded4CPU/8GiB normalization queue reuses58540511 under tmux
onlogin04; it does not request a third allocation. At22:00UTC usage is
5.21361CPU-nodeh/0GPUh and3,396,942,351,204newScratch bytes.

Latest milestone (21:46UTC): the full registered physical-reference gate passes
for both operators on64training cores, with about88% median eigenvalue-error
reduction. [Report and provenance](e2e_coupled_physical_gate_20260918.md).
Both007/008 have complete128-pair targets and independently audited608hashes.
This is not global training readiness; remaining phase products, normalization
and actual GPU cost/restart checks are still required.

Status: full-panel construction and qualification in progress; not data-qualified and
not a training launch. The separate machine authority is
[`configs/e2e_coupled_data_v1.json`](../configs/e2e_coupled_data_v1.json).
Mac-led continuations58550091 and58552205 are active; the corrected coordinate contract
passes independent host checks on007/008/020 and fullph007 observation builds.
The downstream coordinate authority is separately versioned at
`configs/e2e_coupled_coordinates_v2.json`; old Cartesian fields are excluded.

The user approved thirteen training phases (000,002,003,007–011,020–024), two
development phases (012–013), and six confirmation phases (014–019). Historical
004–005 are unused in new fits; 001/006 remain sealed. Eight training phases
were a reserve choice, not a demonstrated diversity plateau; thirteen is not
a proven asymptote either. The new panel supplies 1,664 training pair domains
and 128 held-out pair domains, each containing two adjacent owned science cores.

Approved caps:4.5TiB newScratch,64CPU-nodeh,4GPUh technical checks only, at most
two simultaneous allocations, CPU jobs at most four hours. Approved B restores
008–024 require approximately2.60TiB. No scientific training or full-particle
restore is authorized. Source/provenance remain in Git; large artifacts live at
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1`.

The later arms are D-reference, observation-matched independent I-VDM, jointly
evolving J-VDM, and J-CFM at both scales. I/J-VDM share coarse fits and draws;
the two-seed panel therefore has fourteen distinct factor fits. D:new is a
package comparison; I:J isolates fine residual coupling; J-VDM:J-CFM compares
objective/path/sampler packages under matched architecture/data, not a pure
loss-only ablation. Exact block conservation means fine
coupling cannot fix block-aligned regional-mass uncertainty: the coarse posterior
must pass that test independently. Motivation remains the
[joint-field literature decision](e2e_vdm_context_joint_decision_literature_20260918.md).

Required deliverables before completion:

1. All-phase source inventories, full exact CutSky/forFA/successful-LSS joins,
   independent host-coordinate checks, explicit HOD/epoch/selection provenance.
2. Verified A+B field/halo particles (all34slabs/kind), POSIX CRCs and ASDF
   headers, native2048-cubed TSC counts with conservation and resumable receipts.
3. Target-free canonical observed counts/response, retaining the previous DA2
   recipe and all18 angular random catalogues. Preserve and audit the inherited
   P10 selection function; new model normalizers use only the thirteen current
   training phases with equal phase weights.
4. Support-only stratified pair geometry, no owned-core overlap in periodic
   source space, separate observation/target shards, positive exact coarse/fine
   charts, and full-box spectral R7 reference tides on owned science cores.
5. Train-only normalization and targetless condition readers; confirmation
   target QA is allowed, predictive scoring requires frozen model choices.
6. Synthetic/operator/round-trip tests, real-payload hashes, numerical and
   physical-reference qualification, measured memory/runtime, and the downstream
   four-arm resource/draw proposal for separate approval.

User-requested closeout (2026-09-19): after all preparation requirements pass,
commit the goal-related code, tests, configurations and scientific records on
the current branch, verify the commit contents, and leave them ready for the
user to push. Do not push remotely or include unrelated workspace changes.

### Which DESI observational effects are already included?

The E2E condition is NOT the complete/idealized CutSky galaxy field. We reuse
the upstream DESI DA2 SecondGenMocks AbacusSummitBGS_v2 `altmtl/kibo-v1`
`BGS_BRIGHT_full_HPmapcut.dat.fits` products. Preparation verifies the exact
CutSky -> forFA BRIGHT parent -> successful-LSS TARGETID/RA/DEC/RSD-redshift
joins, retaining finite positive `Z_not4clus` with `ZWARN==0`. forFA BRIGHT
matching uses r<19.5. The upstream potential-assignment and fiber-assignment
assets are inventoried; this goal does not rerun or independently reconstruct
every upstream survey-simulation operation. The final catalogue's selection is
inherited, not newly simulated by a uniform thinning of ideal galaxies.

The condition builder deposits the successful galaxies at their observed/RSD
positions, with context0.10<=z<0.60 and the inherited0.585<=z<0.595 sentinel
exclusion. All18 matching random catalogues supply GOODHARDLOC-selected angular
support/response at NSIDE256; the existing HPmapcut product is trusted, and
MASKBITS are audited without inventing an additional veto-bit selection.
The inherited P10 training-only radial selection and corrected volume Jacobian
provide expected counts. Local/wide channels include counts, support, angular
response, expected counts, sampling density, line of sight and radial context.
These are conditioning features, not a separately calibrated survey likelihood.
Counts are not inverse-completeness weighted to recreate an ideal galaxy field.

Example checked live:ph007's9,155,700-row full product gives7,339,301successful
rows and6,245,349rows in the registered context. Its successful redshifts agree
with CutSky RSD redshifts within2e-7; no additional spectroscopic error kernel
is introduced or claimed here. The paired target is the SAME-phase real-space
z0.2 c000 matter field, reconstructed from A+B10% particles, not a degraded
galaxy target. Matter density is smoothed once atR7 and supplies the reference
density/coarse/tidal products. Matter/host truth is used for pairing/QA and
training targets, never as an observation input at inference.

Thus DESI-like observational selection is already inside this experiment.
What remains outside the current capability claim is demonstrated real-DESI
transfer, an independently validated instrument/selection likelihood, numerical
HOD-parameter marginalization, and an evolving lightcone target. The fixed-z0.2
snapshot supports a nominal mock-conditional test across observational redshift
shells; it is not matter evolution across those shells. Survey-realism closure
and astrophysical nuisance tests remain necessary before a real-DESI claim.

The target is fixed-epoch z0.2 c000 matter, not an evolving matter lightcone.
The phone's distance audit supersedes the inherited Planck18 approximation:
new Cartesian products use the pinned DESI/Abacus distance table in Mpc/h,
under `configs/e2e_coupled_coordinates_v2.json`, with inherited selection
densities transformed by the radial volume Jacobian. Raw spacing is3.383Mpc/h.
The original v1 source/particle authority remains unchanged. See the
[session reconciliation](e2e_coupled_session_reconciliation_20260918.md).
Fine6.766Mpc/h cells form a64×48×48 joint parent from two
48³parents; owned16³cores begin at(16,16,16) and(32,16,16). Wide48³cells are
27.064Mpc/h. Pair selection reads only support/coordinates, never counts/truth.
Both cores require≥25%support; interior means both≥95%. Domain midpoints must
be≥162.384Mpc/h apart and owned cores from distinct domains cannot overlap.

Smooth density once atR7, derive spectral tides (not the old FD2 trace), retain
patch DC and density positivity, and require relative2e−6 round-trip/coarse-mass
agreement. Transfer completion is not particle QA. Numerical trace closure is
not physical-reference qualification. Stop on source drift, failed pairing,
failed CRC/headers, incomplete slabs, wrong units/epoch, leakage or resource caps;
do not silently replace A+B with A-only or shrink the registered science panel.

Historical initial evidence (superseded by the progress record below):21phase metadata inventories pass; seven focused contract/
join/transfer-parser tests pass. CPU58538574/nid004142 is auditing full joins in
tmux from immutable source. ph007 has7,339,301verified successful observed rows.
A separate1,357-central,three-slab check gives median0.510/p95=1.453Mpc/h host
offset, supporting inherited axes/offset but not zero mapping error.

## Verified progress at 21:12 UTC

| Required product | Committed state | Still required |
| --- | --- | --- |
| Exact catalogue joins | All21 phases | Independent final source/payload audit |
| B restoration008–024 | All17; transfer58539034 completed | CRC/header verification before each native build |
| Native2048 A+B density | 007 plus explicitly requalified002/003 | Fresh000 replay and remaining17 phases |
| Corrected cap responses | 000,007,008,009,010 | Remaining16 phases |
| Support-only pair geometry | 000,007,008,009 | Remaining17 phases |
| Condition shards | 007,008,009 | Remaining18 phases |
| R7 density/coarse/full-box tensor pairs | All128 pairs in007 | Remaining20 phases |
| Train-only normalization | Implemented, not fitted | All13 training phases complete and verified |
| Physical-reference gate | One-pair/seven-offset numerical smoke only | Predeclared64-core gate on007/008 |
| Training resource proposal | CPU product measurements available | Technical GPU benchmark and complete readiness audit |

All46 focused tests pass. First full target construction:603.0s, peak257.5GiB,
trace error1.78e-15, coarse/fine relative mass error9.83e-16. The native density
particle-conservation tolerance remains2e-6. This establishes numerical
consistency for the first phase, not posterior performance or global readiness.

The old002/003 arrays were adopted without copying after all136 A+B inputs,
packed counts, original manifests and actual payload hashes were requalified.
The first two adoption steps failed due to a manifest-field location and a
floating-point reduction-tree mismatch, respectively. Corrected step58544227.6
completed0:0 for both. Whole-array float64 reduction exactly reproduces the old
manifest; changing the reduction tree moved the sum by0.0016 particles out of
33billion. Failed logs remain; no physical acceptance tolerance was relaxed.
ph000's missing original exact-count provenance requires a fresh native replay.

Geometry acceleration is exact: every originalph007 candidate diagnostic and
selected row is reproduced in14.75s, compared with54m37s for the slow build.
ph000 then failed the original greedy packing at120/128 pairs. The separately
frozen support-only feasibility policy (`e2e_coupled_geometry_search_v1.json`)
tries the original order, scarcity-first order, then at most16 deterministic
scarcity-first permutations of the SAME candidate pool. Its first seeded retry
finds128ph000 pairs, passing the unchanged8-per-stratum, support,162.384Mpc/h
separation and source-nonoverlap checks. No counts or truth are inspected and no
already-committed geometry is replaced. The failed receipt remains preserved.

The independent `e2e_coupled_product_audit.py` re-reads actual SHA256 payloads and
source catalogues/randoms, receipt relationships, targetless input schemas, native
conservation, owned-core count/support extraction, all context mass identities,
and full-box trace. Its firstph007 run passes in58540511.12:604 actual hashes
(121,419,972,713bytes) and all128 pairs in254.77s. Receipt:
`cartesian_v2/product_audit/ph007/PRODUCT_AUDIT_1789766211380398266.json`, SHA256
`011c388bcae0f6fc4654263c428a54081a154fd13bfb9452199816ba40980f4e`.
It retains prior full particle CRC evidence and rechecks those
official CRC bindings plus source metadata, rather than claiming a new full
particle CRC rescan. A phase audit is explicitly not global training readiness.

At21:18UTCph000's128 condition shards also complete and010 geometry passes.
The full registered physical gate now waits in step58540511.13, tmux
`coupled-physical-gate`, frozen `physical_gate_v1`, with4CPU/8GiB and an80-minute
step limit. It waits at most one hour forph008 targets before returning an
explicit dependency pause if necessary. A smoke result cannot bypass this gate.

Persistent workers:58540511 `coupled-prep-mac02` and `coupled-legacy-native`;
58544227 `coupled-products-01`, `coupled-native-pool`, and recovered
`coupled-conditions-recovery`. Original frozen wrappers may retain nonzero exit
statuses from the recovered adoption/geometry attempts: final qualification
requires receipts plus recovery step exits, not just a top-level scheduler label.
Remaining-time checks and checkpoint pauses stay in force. Exactly two allocations;
no scientific fitting, full-particle restore or GPU run. Accounted21:12UTC usage:
3.59139CPU-nodeh,0GPUh,3,215,664,836,011newScratch bytes. All original caps apply.

Mac reconnected tologin04 during this turn; its backend213499 holds the task.
The oldlogin30 task backend is gone, but its deterministic allocation controllers
and tmux sessions remain live. Newly launched recovery and physical-gate tmux
sessions are onlogin04. Reuse the existing Slurm IDs rather than requesting
duplicate jobs because a tmux session is not visible on another login node.

## Common data interface and fair I/J domain comparison (21:36 UTC)

`e2e_coupled_views.py` now packages pinned global train-normalized observations
and positive coarse/fine charts. It checks the exact thirteen-phase normalization
panel and equal phase weights; fixed support/LOS identity scalings and the
zero-mean fine subspace cannot change silently. Fit-batch access rejects
development/confirmation/sealed phases before opening any data. Observation-only
inference reads the pinned normalizer and condition shards, never follows its
historical target-source paths, and does not require target payloads to be mounted.
The normalizer itself remains unfitted until the full training panel is available.

The I and J arms must receive the SAME full rectangular observation view plus the
same wide observation view, relative context displacement, and sampled coarse
field. Do not crop away the I arm's observation information while calling it a
pure residual-dependence control. The D-reference retains its separately disclosed
conditioning restriction and is a package baseline, not a pure objective control.

For the primary I/J field diagnostics, use the SAME64x48x48 domain and joint
physical operator. Assemble the two independent48-cubed parent draws using a
block-aligned ownership seam: joint x<32 comes from left parent x<32; joint x>=32
comes from right parent x>=16. Do not average the stochastic overlap, which would
artificially narrow uncertainty. Tests prove exact reassembly of the joint truth
and preservation of every shared coarse block mass for independently generated
residuals. The per-parent physical operator remains a D-reference/diagnostic
control, not an unacknowledged change of estimator between I and J.

The independent arm uses two48-cubed latent parent evaluations per pair, whereas
the joint arm uses one64x48x48 evaluation. Keep owned-core/pair exposure matched,
report that extra auxiliary-overlap exposure and actual compute, and normalize
objectives per independent latent degree of freedom. Do not claim equal GPU cost
or identical numbers of latent voxels merely from equal update counts. Exact
training exposure and benchmarked costs belong to the later approval proposal.

All52 focused tests pass, including float32 normalized round trips for all seven
offsets and role/identity-scale guards. The strengthened phase audit now also
checks both factor-four response caches, pair-to-response/geometry receipt links,
and exact I/J truth reassembly. A bounded audit queue on58540511, frozen
`product_audit_v2`, re-audits completed phases and consumes later products as they
arrive. This is preparation QA only, including allowed numerical confirmation QA;
no confirmation predictions, learned checkpoint selection or scientific fits.

## Explicit source provenance and remaining claim boundary

All registered catalogue sources use the nominal
`CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_phNNN.fits`
branch, paired exactly with DA2 `AbacusSummitBGS_v2/forFAN_nomask.fits` and
`altmtlN/kibo-v1/mockN/LSScats/BGS_BRIGHT_full_HPmapcut.dat.fits`.
The similarly named `ph000_hod_sample_1` through `_10` are different populations
and are NOT used. Phase labels are not model features. Matter is c000/z0.2,
not an evolving matter lightcone; observer redshift changes selection/RSD.

The source inventories do not contain numerical HOD parameter cards. Additional
read-only primary/extension header checks onph007's original CutSky and cubic-box
FITS also find no HOD/cosmology/phase parameter provenance. Thus the verified
statement is **one identified nominal mock-release branch with exact paired
file identities**, not a recovered numerical HOD parameter hash or marginalized
HOD ensemble. Preserve this limitation in the later training proposal and DESI
claims; do not label undocumented numerical HOD parameters as independently
verified. This does not prevent constructing the approved empirical paired set.

DESI's official data model distinguishes complete, probabilistic FFA and altmtl
catalogues; altmtl uses the survey's assignment algorithm/passes/hardware. Our
actual DA2 paths and full exact joins, not a generic directory name, establish
which catalogue flavor is consumed. The inherited random-response map is still
not an explicit full fibre/redshift-success likelihood.
[DESI mock data model](https://desidatamodel.readthedocs.io/en/latest/DESI_ROOT/survey/catalogs/RELEASE/mocks/AbacusSummit/index.html).

## Joint product implementation and launch (historical 19:52 UTC)

The new modules are `e2e_coupled_condition_products.py`,
`e2e_coupled_conditions.py`, `e2e_coupled_target_products.py`, and
`e2e_coupled_normalization.py`. Conditions contain12 local/joint channels and12
wide channels, without matter, phase-ID features or observational periodic wrap.
Counts/expectations retain the previous sum-transform-average wide-grid recipe.
Number-density and radius channels now explicitly carry h-based physical units;
the legacy number-density log floor is converted to the same physical value.

Targets store positive physical density, a56³extended coarse grid and two16³
owned-core full-box tensors, separately from conditions. Density and six tensor
components commit independently; interrupted stages cannot appear complete.
Exact coarse/fine mass agreement is checked under all seven registered context
translations. Trace/round-trip checks are numerical gates, not the pending
physical-reference representation gate. No new FFT/reference job has started yet.

Before any new physical error evaluation, `configs/e2e_coupled_physics_v1.json`
fixes the training-only gate:ph007/ph008, ordinal00 in all16strata, both cores.
Both independent-parent and joint-parent wide-plus-fine operators must improve
the median per-core eigenvalue RMSE over parent-only reconstruction by at least
25% for each eigenvalue, with no median regression in either phase. This retains
the prior consistent-expanded-density-v2 pilot threshold; it is not a posterior
coverage claim. Other context translations and tail/gap errors are reported
separately and cannot replace the zero-offset primary test.

The normalizer requires all thirteen complete training phases, equally weighted,
and never fits per-field scalings. It preserves identity-scale support/LOS channels
and the zero-mean fine residual chart. It has not been fit. Confirmation conditions
can be read targetlessly for preparation QA; held-out context augmentation and
unregistered phase paths are rejected. All32 focused tests pass.

Frozen source `source_snapshots/joint_products_v1` drives step58540511.6:
4CPU/16GiB in the existing allocation, tmux `coupled-products-smoke`. It waits
forph007 geometry, qualifies two real shards against raw owned-core counts/support,
then completes the phase only if those checks pass. Log:
`condition_smoke_58540511.log`. Geometry58540511.5 is still running. No additional
allocation was requested. At19:52UTC:14joined phases,12Btransfer receipts,
onephase with corrected cap responses; full products are incomplete.

A bounded geometry timing check in58540511 measured2.763s for100 calls to
`domain_positions`, versus0.095s on the login node. The current implementation
re-reads/hashes the same immutable coordinate authority repeatedly per candidate.
Hoist that validation outside the candidate loop before scaling geometry to all
phases, with equivalence tests; do not modify the source of the currently running
ph007 geometry job before its receipt is committed. This is an implementation
overhead, not a change to candidate sampling, support thresholds or science gates.

Bulk restores use the official transfer-only
[xfer QOS](https://docs.nersc.gov/jobs/examples/#xfer-qos): DTN01 has no HPSS
binaries. All payload computation, CRC checking, deposition and FFTs remain on
compute nodes. All later products and downstream cost measurements are pending.
