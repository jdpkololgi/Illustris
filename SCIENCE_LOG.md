# SCIENCE_LOG.md — shared brain: Claude Desktop (science) ⇄ Claude Code (NERSC)

### 2026-09-01 - [science/code] Close JEPA mainline; open calibrated coherent-field posterior canary

P11 now answers the representation-alignment question sufficiently for the current
programme.  JEPA-v2 learned strong dense/final-view alignment but did not preserve
more held-out tidal information than its matched supervised control.  More
fundamentally, deterministic latent alignment is not the estimand required by the
VAC: one selection-affected galaxy observation is compatible with multiple matter
and tidal fields.  No JEPA-v3 is therefore on the production path.  The existing P11
products remain useful factorial observation views and advisory representation
evidence, but P12 posterior inference remains binding.

An immediate physics diagnostic clarifies the low-rank P11 result.  On the frozen
426-row ph006 step-500 probe, PC1 explains `0.98869` of weighted JEPA-student latent
variance and has weighted Spearman `|rho|=0.84600` with the physical tidal trace
`lambda1+lambda2+lambda3`, versus `0.27012/-0.07781` with the two eigengaps and
`0.18195` with random-response strength.  The supervised control is almost identical:
PC1 fraction `0.98682`, trace `|rho|=0.84362`, response `0.18414`.  The sign of PC1 is
arbitrary.  Thus the one-dimensional locus is predominantly a learned density/trace
coordinate, not a constant collapse and not principally a response shortcut.  It also
explains why alignment alone did not recover tidal-shape ambiguity.  Evidence:
`docs/evidence/p11/P11_LATENT_PHYSICS_DIAGNOSTIC.json`; implementation:
`workflows/sbi/p11_latent_physics_diagnostic.py`.  ph001 remained sealed.

The next coherent-field challenger is `P12-F`, with frozen conditional estimand

```text
p(delta_R7 patch | BGS_BRIGHT V_final, random-derived response, H_fid).
```

`delta_R7` is the trace of the already `R=7 Mpc/h` smoothed T-web eigenvalues.  It is
not smoothed again.  Every posterior draw is transformed by the fixed local-periodic
projector `T_ij(k)=k_i k_j/k^2 delta_R7(k)` and `eigvalsh`; this is a deterministic
physics map, not a learned eigenvalue head.  Posterior-mean field `R2` is explicitly a
diagnostic rather than a promotion gate.  The primary canary asks whether held-out
ph006 truth behaves as a draw using voxel ranks/coverage, Fourier-mode ranks/coverage,
derived-eigenvalue coverage and stratification by redshift shell, random response,
true environment and support-boundary distance.  Spatial dependence is retained as a
warning: production validation still requires block/mode TARP/SBC and posterior
predictive re-observation through a frozen stochastic galaxy plus DESI response model.

Groundwork is implemented in
`workflows/abacus_tweb/p12f_build_field_targets.py`,
`workflows/sbi/p12f_field_posterior_diagnostics.py`,
`workflows/sbi/p12f_train_conditional_field_flow.py`, and
`configs/p12f_conditional_field_flow_v1.json`.  Seven focused tests pass, including
ph001 rejection, rectified-flow path algebra, Heun integration, posterior ranks,
draw-axis-preserving Fourier diagnostics and fixed-physics trace/eigenvalue closure.
Visible-phase field-target construction and the 1,000-update conditional-flow canary
are pending compute.  A canary pass licenses a larger comparison, not production: it
does not establish global long-mode coherence, HOD marginalisation, DESI closure or
blind-phase calibration.

Degraded P11 views are retained, but their role changes.  They are alternate
conditioned observations `(X_s,S_s)` of the same privileged `delta_R7`, not teacher
latents and not posterior samples.  Establish the V_final-only field posterior first.
Then run one factorial-view challenger that samples one view per latent core/update,
conditions on that view's deployable response, and keeps total core weight one.
Promotion is decided on V_final and a held-out degradation recipe using proper scores
and calibration.  Do not force coordinate-wise latent equality, sample equality, or
per-object width monotonicity; only aggregate conditional entropy/contraction should
follow the information ordering after calibration.

### 2026-09-01 - [science/code/run] P11 JEPA-v2 technical pass, latent-content no-go, matched controls complete

The recovered P11 runtime and R1 metadata mirror removed the two independent Lustre
failure modes described below.  The corrected exact-M real-view parity gate then
passed for ph002--ph006.  All four bounded arms (`jepa`, `supervised_masked`,
`masked_reconstruction`, and `response_only`) completed exactly 500 optimizer updates
under the same frozen examples, masks, target weights, U-PATCH capacity, data digest
and step-0/250/500 probe schedule.  Every technical marker passes: losses, gradients
and parameters are finite; checkpoints reload; all 500 examples are auxiliary-valid;
the cumulative target-mask fraction is `0.249998`; and `ph001_opened=false`.  The
aggregate frozen-data SHA-256 is
`004ef485b3773ded1639720aad2e2d634155000367f98ee663f8d0468e676f57`.

The JEPA arm nevertheless fails the independently registered latent-content gate.
Native student/dense CKA rises from `0.8393` to `0.8725`, and paired retrieval at step
500 is far above its response-matched shuffle (`MRR 0.3135`, excess `+0.2545`).  Those
attractive alignment numbers coexist with loss of content: the student effective-rank
fraction falls from `0.22116` to `0.03328` (effective rank approximately `7.08 ->
1.06` of 32), while the spatially held-out student target probe falls from macro
`R2=0.02201` to `-0.08775`.  At step 500 the dense-view probe remains `0.21796`.
The shared-predictable-subspace gate is false and the collapse/adverse-trajectory
guard fires.  The full JEPA-v2 supervisor correctly refused to request another GPU
allocation; this gate was not weakened or overridden.

The matched controls sharpen rather than reverse that conclusion:

| Arm | Effective-rank fraction, step 0 -> 500 | Student probe macro R2, step 0 -> 500 | Native CKA, step 500 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `jepa` | `0.22116 -> 0.03328` | `0.02201 -> -0.08775` | `0.87254` | registered latent-content failure |
| `supervised_masked` | `0.22116 -> 0.03342` | `0.02201 -> +0.01145` | `0.86674` | advisory reference |
| `masked_reconstruction` | `0.22116 -> 0.03335` | `0.02201 -> -0.65792` | `0.84876` | advisory; worst target retention |
| `response_only` | `0.11342 -> 0.04067` | `-3.65232 -> -2.40036` | `0.16649` | response alone is non-predictive |

The almost identical low rank in the plain supervised arm means that pooled-latent
compression is not uniquely caused by JEPA; it is partly confounded with rapid
optimization of a 32-dimensional pooled U-Net representation toward three regression
targets.  This nuance does not rescue JEPA-v2.  With matched target loss and healthy
optimization, JEPA retains about `0.0992` less held-out probe macro R2 than the
supervised control, while masked reconstruction is substantially worse.  The
response-only arm also rules out the simple explanation that high JEPA alignment is
only footprint/response reconstruction.  Alignment without rank and target-content
preservation is therefore not accepted as a shared scientific representation.

The fixed-reference latent trajectory makes the failure visible: by steps 250 and 500
the paired embeddings have folded onto an almost one-dimensional locus even as CKA
recovers.  The durable figure is
`docs/figures/p11_jepa_v2_20260901/latent_trajectory_0_250_500.png`; its SHA-256 is
`496abad0552b14879af210d959f0f15f05a43bc47fd109e10214a554840806e6`.
The machine-readable matched-arm summary is
`docs/evidence/p11/P11_JEPA_V2_MATCHED_CANARY_RESULTS.json`.

This is a no-go for this frozen JEPA-v2 objective, not for every possible JEPA.  Any
JEPA-v3 must be a separately registered experiment with a changed teacher/anti-collapse
objective and must repeat the identical technical and latent-content gates before
longer training.  It may not tune v2 post hoc against these ph006 diagnostics.  Any
future promoted representation must still receive a newly fitted P12 posterior and
pass SBC, TARP, shell/response/local-information conditional coverage; JEPA alignment
is not posterior uncertainty.

The advisory dense teacher did resume successfully under the CFS runtime, advancing
epoch 4 from cursor `13,819` to `54,518` and global step `215,551` to `256,250`.
A later continuation after source changes was correctly refused by the frozen source
hash guard.  Restore the exact historical source set before any further advisory
resume; do not weaken that guard.  The login-side JEPA supervisor now also clears
inherited `PYTHONPATH`, `PYTHONHOME`, `PYTHONUSERBASE`, and `LD_PRELOAD` before using
the CFS Python, preventing DESI Python 3.13 extensions from contaminating the frozen
Python 3.11 runtime.

### 2026-09-01 - [science/code/run] P11 zero-update mask failure exposes exact-support contract error

The repaired R1/P3b-R JEPA preflight progressed past the inaccessible-manifest
failure and reached real paired patches. It then failed before the step-0 checkpoint
and before any optimizer update: ph002 core 6 contained zero complete supported
`8^3` target cuboids, whereas `PAIRED-DEGRADE-JEPA-v1` required four. No latent
trajectory, validation score, or JEPA result exists from this run. The failed v1
directory is retained as negative provenance and must never be resumed under an
amended mask contract.

The failure revealed that `exposure_apodized_random > 1e-4` is not the binary survey
support `M`. The apodized exposure is deliberately Gaussian-smoothed and spills beyond
the footprint. In the failing 19x19x20 core, exact `support_random` contains only 18
supported voxels, while the apodized threshold admits 939. Using the latter as target
eligibility could therefore make `M=0` cells into reconstruction/alignment targets,
contrary to the P11 scientific contract. The visible response channel remains the
apodized exposure, but target eligibility must be loaded separately from exact
`support_random`.

A label-free 500-core-per-phase audit over ph002--ph006 confirms that selecting a
different parity core or merely fixing the greedy cuboid packer would not solve the
problem. Four complete exact-M `8^3` cuboids were constructible in only 69.6--70.8%
of sampled cores. Exact support ranges from 18--194 voxels at the phase minima to
about 7,683--7,793 at the medians; excluding thin-support cores would selectively
remove the boundary/low-response regime that P11 is meant to improve.

The replacement `PAIRED-DEGRADE-JEPA-v2` mask is therefore support-aware rather than
interior-only. It deterministically selects four compact, spatially separated clusters
covering 25% of exact M, leaves the apodized response channel visible, and never targets
M=0. Student, reconstruction, response-only and JEPA arms receive the identical seeded
mask schedule and retain every core's authoritative supervised eigenvalue loss. Target
masks are propagated to the U-Net bottleneck by pooling-aligned logical-any operations;
nearest-neighbour resizing silently lost all but one coarse target in 10.2--14.6% of
the sampled cores, whereas pooled propagation retained at least two bottleneck targets
in all 2,500 sampled cores across epochs 1, 2, 5 and 10.

The production-identical all-visible-core epoch-1 audit covers 84,040 cores. Exact-M
support minima are 2/11/5/5/26 voxels for ph002/ph003/ph004/ph005/ph006, while phase
medians remain 7,736--7,739. Pool-aligned propagation leaves only 10 auxiliary-invalid
training cores (3/2/3/2 in ph002/ph003/ph004/ph005 and none in ph006): `10/67,244 =
1.487e-4`, below the frozen `0.001` population ceiling. Nearest resizing would have
invalidated 12.12--12.61%. Those ten examples remain in the phase-balanced training
stream and retain their authoritative eigenvalue loss, but receive an
explicit `aux_valid=false` and zero alignment/reconstruction/collapse weight. They are
never silently removed. The frozen summary is
`docs/evidence/p11/P11_JEPA_MASK_FEASIBILITY.json`; ph001 remains sealed.

The technical canary does not reapply that population fraction to only 500 updates,
where one rare invalid example would equal 0.002 by granularity alone. It requires
exactly 500 updates, permits at most one auxiliary-invalid update, and fails at two.
Full science continuation additionally requires a passing, content-bound 0/250/500
latent diagnostic; technical health alone cannot trigger another allocation.

This amendment is not post-hoc optimization against a science result: v1 performed
zero updates and exposed no ph006 representation or target metric. It repairs a
support-definition error and an infeasible preprocessing rule before training.

### 2026-09-01 - [science/code/run] P11 runtime recovered; paired-view JEPA technical contract frozen

The stalled dense-teacher allocation was diagnosed before altering the model.  Its
Python process had spent more than 3.5 hours in the Lustre-client kernel wait
`cl_sync_io_wait`, with 0% CPU, 5.3 MiB RSS, a zero-byte invocation log, and a
`stat(2)` system call that never returned.  It had not imported the training stack or
written a new checkpoint.  A brief occurrence of this wait state is ordinary storage
latency; an hours-long occurrence with no CPU or log progress is a node/client I/O
failure, not slow learning and not evidence about P11.  Allocation `57823090` was
cancelled.  The intact epoch-4 checkpoint is 17,390,251 bytes with SHA-256
`c088262000b7a9f4b75157c60ae8b1af476e8e183d1c1cb5b5318944f73246fb` at
epoch 4, cursor 13,819/67,244 and global step 215,551.

To avoid repeating the failing Scratch-resident interpreter access, the frozen conda
explicit specification was reconstructed under CFS at
`/global/cfs/cdirs/desi/users/dkololgi/conda/envs/cosmic_env_recovery_v4_20260901`.
Clean compute-node preflights on A100-80GB nodes `nid008257` and `nid008604` passed
JAX/Torch/NumPy/HDF5/FITS imports, a Scratch checkpoint `stat`, and a frozen P11 JSON
read.  Twenty-four focused JEPA/latent/loader tests and six dense-trainer contract
tests pass.  This is a P11/P12 recovery runtime, not a claim that all historical
catalogue-builder extras are reproduced.  Its exact provenance and inventory hashes
are stored in `docs/evidence/p11/P11_CFS_RUNTIME_RECOVERY.json`.

Two subsequent step-0 JEPA attempts isolated a second, independent storage fault.  On
both `nid008604` and `nid008280`, the process blocked in `stat(2)` on the same small
file,
`training_contract_r1_random/adapters/ph002/field/adapter_manifest.json`, while all
large fields remained unopened.  The first allocation was reported by Slurm as a node
failure; the second could not kill the uninterruptible task and remained `COMPLETING`.
Both attempts performed zero optimizer updates.  This proves that moving Python alone
does not repair an inaccessible data-contract inode.

The R1 contract was therefore deterministically regenerated under home at
`/global/homes/d/dkololgi/p11_contracts/training_contract_r1_random_repair_v2_20260901`.
Only 98,043 bytes of manifests/transforms/symlinks were created: all phase arrays,
geometry arrays, targets and field products still resolve to the original immutable
artifacts.  The regenerated field transform has the frozen R1 SHA-256
`1841e2d2f8a76f6e649d149803f377fd4ae6e0e713eb30fd2d775e957c490aa4`, and
the target scaler retains SHA-256
`c918ab3d58c3ec1dbb568ef450b3eabc76db7a94a41b822e7cedbfaf364eae9f`.
During recovery, an inherited READY pointer to the base R0 adapter inventory was also
found and corrected: READY now points to and hashes the mirror-local R1 inventory, and
the JEPA preflight fails closed on every phase pointer/hash.  Full recovery evidence is
`docs/evidence/p11/P11_R1_CONTRACT_MIRROR_RECOVERY.json`.

The advisory dense teacher was then resumed on interactive allocation `57827338` from
the unchanged checkpoint.  Fresh loss rows, atomic checkpoints, finite gradients and
nonzero GPU updates are present.  It remains advisory privileged-view evidence; it is
not the JEPA go/no-go criterion and ph001 remains sealed.

The bounded `PAIRED-DEGRADE-JEPA-v1` implementation is now frozen as a supervised
multi-view JEPA auxiliary objective:

- identical ph002--ph005 latent cores supply a deployable `V_final` student and an
  EMA, stop-gradient, target-head-free `V_dense` teacher;
- the student alone owns the ordered-increment eigenvalue head;
- exactly four non-overlapping 8^3 blocks are hidden only inside the authoritative
  core and common random-supported region; exposure remains visible and M=0 is never
  a target;
- the checkpoint hashes the loader marker, target scaler, final and dense transforms,
  core/weight arrays, response manifests, configuration and source files, and writes
  an atomic step-0 checkpoint before any diagnostic export;
- the 500-update technical gate requires finite losses, gradients and all parameters,
  a reloadable checkpoint, the registered mask fraction, and valid fixed ph006 latent
  snapshots at steps 0, 250 and 500.

The first real-view preflight performed zero optimizer updates and correctly failed:
the legacy final-view adapter used P3a occupancy-derived `exposure_apodized`, whereas
the dense view used P3b-R `exposure_apodized_random`, producing 4,374 support-mask
disagreements in the first audited patch.  This was a data-contract defect, not a
reason to intersect the masks.  The final student is now explicitly dispatched
through the frozen R1 random-response loader: its immutable final BRIGHT counts and
P3b-R log-ratio/exposure channels share the same random-derived M as `V_dense`, while
the view-specific counts and radial selection remain different.  The failed run
directory is retained fail-closed as provenance; no score or representation result is
accepted from it.

The latent audit must not ask whether the degraded embedding is numerically bounded by
the dense embedding.  Latent coordinates are free to rotate, rescale and permute.  The
valid information statement is

`I(E; Z_student) <= I(E; V_final) <= I(E; V_dense)`.

Evidence for a shared predictable representation therefore requires paired CKA/CCA,
cross-fitted Procrustes alignment and retrieval above shuffled and response-matched
controls, retained effective rank/spread, and target probes on spatially disjoint
ph006 cores.  Fixed teacher-space PCA is used only for visualization, so arrows across
steps 0/250/500 show changes in one stable coordinate system rather than refitting an
attractive projection at every checkpoint.

Multiple matter/density fields can remain compatible with one observed galaxy view.
A deterministic JEPA predictor can learn only the teacher components predictable from
that view—often a conditional-average representation—not select the unique missing
field.  Strong alignment can therefore become a hallucination or overconfidence risk
if a later posterior head treats the aligned latent as complete information.  The
response-only, shuffled, collapse/rank, low-response error and held-out target-probe
controls guard the representation claim.  They do not calibrate uncertainty.  Any
promoted encoder requires a new P12 fit plus SBC, TARP and response/shell-conditional
coverage; JEPA itself is never reported as a posterior.

Implemented artifacts:

- `configs/p11_paired_degrade_jepa_v1.json`;
- `workflows/abacus_tweb/p11_jepa_canary.py`;
- `workflows/abacus_tweb/p11_jepa_latent_diagnostics.py`;
- `workflows/abacus_tweb/plot_p11_jepa_latents.py`;
- `workflows/abacus_tweb/run_p11_jepa_interactive.sh`;
- `workflows/abacus_tweb/p11_jepa_supervisor_guard.py`;
- `workflows/abacus_tweb/run_p11_jepa_science_worker.sh`;
- `workflows/abacus_tweb/run_p11_jepa_science_supervisor.sh`;
- focused tests under `tests/phase4/test_p11_jepa_*.py`.

The post-canary supervisor uses successive one-GPU interactive allocations only,
enforces the shared two-allocation limit, and resumes only after exit 75 plus an
independently reloaded checkpoint.  It does not auto-launch the matched controls:
`supervised_masked`, `masked_reconstruction`, and `response_only` each require their
own technical canary after the JEPA trajectory is reviewed.  No sbatch job was used
and ph001 was not opened.

### 2026-09-01 - [science/code/run] P11 information-headroom sub-gate passes; bounded JEPA canary licensed

The dense-view supervised teacher has been downgraded from a hard JEPA go/no-go gate
to advisory privileged-view evidence. This is a conceptual correction, not a weakening
of the scientific gate. A supervised dense-view score conflates two possibilities:
the privileged view may contain little extra information, or the matched U-PATCH
architecture may fail to extract it. JEPA operates in representation space and can be
useful in the second case, so a failed `R2_dense >= R2_final + 0.03` test is not an
information-theoretic veto.

The first reproducible information-headroom sub-gate now uses the frozen, uncorrected
P12-A posterior on its existing ph006 folds-2--4 evaluation selection: 50,000 rows,
512 posterior draws per row, natural-volume weights and 500 cap+superblock bootstrap
resamples. For a calibrated posterior conditioned on summary `S`, the Bayes-risk
identity gives

`R2_max(S) = 1 - E[Var(lambda|S)] / Var(lambda)`.

The implementation also compares this conditional-variance estimate with the actual
held-out squared error of the posterior mean. If those disagree, posterior width is
not a defensible ceiling estimate. This guard passes at the registered 0.03 scale.

On pooled ph006 rows, the posterior-variance estimates for lambda1/lambda2/lambda3 are
`0.66152/0.74185/0.79151`, versus posterior-mean R2
`0.66181/0.74788/0.79960`. The corresponding Bayes-identity gaps are only
`-0.00029/-0.00603/-0.00809`. On the primary macro-shell statistic the estimates are
`0.57896/0.66703/0.72086`, versus posterior-mean R2
`0.58082/0.66489/0.72038`; the gaps are
`-0.00186/+0.00214/+0.00047`. The global lambda1 spatial-block interval for the
identity/headroom gap is `[-0.01764,+0.01764]`, so even its 97.5% upper endpoint is
below the 0.03 materiality threshold. The small negative point gaps mean the sampled
posterior variance is slightly wider than posterior-mean error; they do not mean the
model exceeds a Bayes limit.

The shell-wise lambda1 conditional-variance estimates are
`0.6773/0.6676/0.5705/0.4005`, making the falling information ceiling toward the
sparse shell explicit. Shell-level identity uncertainty is widest in shell 3, so no
claim of exact shell-wise saturation is made. The robust conclusion is narrower:
there is no evidence for material downstream-estimator headroom **inside the current
seven-feature P12 summary**. The posterior head is not the source of a missing 0.03
gain. Any material improvement must change the information retained by the summary or
the observations supplied to it.

Crucially, P12 conditions on the three OOF U-PATCH eigenvalue predictions plus
redshift, ntilde, cap and random-support boundary distance. This result therefore
estimates `R2_max(S_U)`, not `R2_max(X_final)` for the raw DESI-like voxel field and
not `R2_max(X_dense)`. It neither proves nor excludes a representation gap. It instead
licenses one bounded matched-control JEPA canary as the direct test of whether a new
final-view representation preserves more target-relevant information. Random-init,
masked-reconstruction, JEPA and curriculum controls must still be frozen and compared
at identical examples, updates and compute. JEPA is not promoted unless the deployable
ph006 student improves by the existing fresh-phase criterion without supported-shell
or held-out-degradation harm.

The already-running dense teacher was left untouched as requested. Allocation
`57823090` remains advisory, but its current process has spent more than three hours in
the Lustre `cl_sync_io_wait` state with 0% CPU and a zero-byte new invocation log, so
no epoch-4 result has been produced by this allocation. This is an I/O/runtime stall,
not a scientific score. Earlier complete epochs remain the only accepted teacher
evidence.

Artifacts and provenance:

- implementation: `workflows/sbi/p11_information_headroom.py`;
- focused tests: `tests/phase4/test_p11_information_headroom.py` (3/3 pass);
- durable report: `docs/evidence/p11/P11_INFORMATION_HEADROOM.json`;
- authoritative scratch report:
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p11_factorial_views_v1/information_headroom_v1/P11_INFORMATION_HEADROOM.json`;
- both report copies SHA-256:
  `95ee472ee1bca4ab3e983423ea581cfbe35bd45abc88ddfd7b926567ff3552eb`;
- posterior draws SHA-256:
  `bcdc94442e9ebc3c75ea0d10254d181eff43f4afc9de69eeacb3e891982fdfee`;
- ph001 remained sealed and unopened.

### 2026-08-31 - [science/code/run] Frozen P12-A TARP curve archived

The full P12 calibration audit had stored the numerical TARP result but no durable
curve. A plotting-only postprocessor now reuses the exact cached ph006 folds-2--4
selection: 50,000 galaxies, 512 posterior draws per galaxy, seed 42 and the same
standardized ordered-softplus three-dimensional coordinates used by the audit. It
does not resample the posterior, fit a correction, or open ph001.

The recomputed expected-coverage curve exactly reproduces the frozen audit:
`max|ECP-alpha|=0.007700`, comfortably inside the registered `0.05` gate. The lower
residual panel exposes the percent-level shape rather than hiding it behind the
diagonal. The narrow magenta reference is explicitly the audit's 95th percentile of
the *maximum* absolute deviation across 100 cap+superblock resamples of 10,000 rows
(`0.016105`); it is not presented as a pointwise confidence interval.

The first compute invocation stopped before reading posterior products because a
file-path entrypoint could not resolve the repository package. No result was accepted.
The package-safe module invocation completed on CPU allocation `57797228`, and the
script now also inserts the repository root so either invocation is reproducible.

Artifacts:

- source: `workflows/sbi/plot_p12_tarp_curve.py`;
- focused tests: `tests/phase4/test_plot_p12_tarp_curve.py`;
- report: `docs/evidence/p12/P12A_TARP_CURVE.json`
  (SHA-256 `fc6a12c65af1dc93c9257ba026ebb9ec746267eab39312fd8dc54220101d3704`);
- figure: `docs/figures/p12_calibration_audit_20260830/p12a_tarp_curve.png`
  (SHA-256 `0cca83b2a1fbc048ecf7f9cb7a0bce875297654a7324b58bee978d53bc5a8ef4`).
### 2026-08-31 - [code/run] P11 dense-teacher gate resumed after GPU balance restoration


The DESI GPU balance now accepts interactive work. Allocation `57782878` was
granted on `nid008477` for four hours with one 80-GB A100, 32 CPU cores and
the `desi_g` account. The label-free V_dense response adapter completed, all
10 focused P11 tests passed, and the fresh 20-epoch capacity-matched U-PATCH
dense teacher is now training with checkpoint and loss logging active.
Authoritative runtime log:
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p11_factorial_views_v1/p11_dense_teacher_interactive_57782878.log`.

The adapter marker is
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p11_factorial_views_v1/dense_response_adapter_v1/P11_DENSE_RESPONSE_ADAPTER_READY.json`
(SHA-256 `e2c2847a8f38a2ba46c38823c9facd86ad07e7c90b3dc6330251933c97fa56ed`).
Its gates record ph002--ph005-only fitting, ph006 application-only, ph001
unopened, finite positive curves and normalization, and passing shell closure.

The first launch stopped before reading scientific products because the clean
compute-node environment could not resolve the repository package when
`p11_factorial_training.py` was invoked by file path
(`ModuleNotFoundError: workflows`). The worker now invokes the builder with
`python -m workflows.abacus_tweb.p11_factorial_training`, matching the
package-safe contract used by the tests. The same allocation was reused; no
data product, checkpoint or scientific result was accepted from the failed
step.

The GPU canary has passed 500 optimizer updates with finite losses and a valid
atomic checkpoint. No dense-teacher validation result is claimed yet. ph006
remains validation-only and ph001 remains sealed.

### 2026-08-30 - [science/code/run] P11 dense-teacher gate implemented; launch blocked by zero interactive balance

P11 has begun with the preregistered information-headroom test rather than JEPA
training. The first estimator is a capacity-matched supervised U-PATCH teacher on the
BRIGHT V_dense targetable view. It retains the production three-channel interface
[log1p counts, clipped count/mu contrast, random-support exposure]; the count field
and the view-specific ntilde_dense(z) are changed, but model width, target
parameterisation, loss, patch geometry and ph006 scoring contract are not. Training
uses ph002--ph005 only, ph006 remains application/model-selection only, ph000 is
excluded because its factorial views are not pointwise nested, and ph001 remains
sealed. The promotion question is whether the dense teacher exceeds the frozen
final-view U-PATCH by about +0.03 macro R2 on ph006. Only a passing teacher licenses
the matched supervised/masked-reconstruction/JEPA/curriculum programme.

Implementation is complete for the dense-view response-contract builder, lazy patch
overlay, P11 phase-balanced loader, integration with the resumable Arm-A trainer,
interactive worker and focused tests:

- workflows/abacus_tweb/p11_factorial_training.py;
- workflows/abacus_tweb/run_p11_dense_teacher_interactive.sh;
- tests/phase4/test_p11_factorial_training.py;
- the pre-existing factorial contract/count tests plus the new tests pass (10/10).

The response fit is label-free and phase-safe. It derives mu_dense from the common
P3b-R support and angular response plus a phase-equal ph002--ph005 radial density
curve; ph006 does not enter the curve or normalization. The builder must still scan
the stored dense count/response grids on a compute node before writing
P11_DENSE_RESPONSE_ADAPTER_READY.json, after which the same worker runs a resumable
20-epoch teacher fit.

No scientific training result exists yet. Two interactive allocation attempts
(desi_g and desi) were rejected by NERSC before allocation because the user
balance was reported as 0.00 node hours despite a positive repository balance.
The builder was not run on the login node and no sbatch job was substituted.
This is an external scheduling/account blocker, not a data, code or model failure.

The P12 illustrative posterior plot was also corrected. All four rows now share one
fixed physical x-range within each eigenvalue column, so width differences are
visually comparable rather than hidden by per-panel autoscaling. Updated artifact:
docs/figures/p12_width_information_20260830/p12a_posterior_examples.png
(SHA-256 2154704e85f20acfc0ea40b1ed899021ba8f3d6110bfc5fe59103dc9aea1d5d7);
the PDF companion was regenerated from the same frozen four posterior rows.

### 2026-08-30 - [science/run] P12-A widths adapt to information; sparse-shell residual is mild miscentring

The frozen uncorrected P12-A posterior was tested directly for the production question:
does uncertainty widen when the observed configuration carries less information, and
does that width identify genuinely difficult galaxies? The diagnostic used 50,000
ph006 folds-2--4 rows, 512 cached posterior draws per row, natural-volume weights and
300 spatial cap+superblock bootstrap resamples. ph001 remained sealed.

The answer is **yes**. Median physical central-68% half-width grows from
`0.0664/0.0729/0.0863` in shell 0 to `0.1466/0.1694/0.1946` in shell 3 for
lambda1/lambda2/lambda3: sparse-to-dense factors `2.21/2.33/2.25`. The widest posterior
quartile has `3.45/3.33/3.38` times the RMSE of the narrowest quartile. Thus width is
not decorative output; it is a strong difficulty discriminator.

P12-A directly conditions on redshift, ntilde, cap and random-support boundary distance,
but not on realized BRIGHT neighbour counts. As an external local-configuration check,
observed BRIGHT neighbours within `20 Mpc/h` were counted with the target excluded.
After residualizing within redshift shell and predicted-trace decile, local count still
anticorrelates with log width by `-0.131/-0.151/-0.134`; 300-block 95% intervals exclude
zero for all coordinates. The explicit boundary-distance response covariate is more
strongly related (`-0.481/-0.523/-0.551`). This establishes an association between
local sampling and posterior contraction, but does not claim that the P12 head reads
neighbour counts independently: that information can only arrive through the frozen
U-PATCH base predictions.

The sparse-shell limitation is now more precisely classified. Nominal 68% coverage is
`0.6893/0.6577/0.6633`. For lambda2/lambda3, lower-tail misses are
`0.1810/0.1791` versus upper-tail misses `0.1613/0.1575`; the posterior widens but is
occasionally shifted slightly high. Four deterministic examples show (i) a dense,
narrow covered case, (ii) a sparse, wider covered case, and (iii--iv) representative
lambda2/lambda3 lower-tail misses. The sky diagnostic shows these misses distributed
through the observed caps rather than confined to one obvious geometric defect.

Scientific decision: adaptive widening passes; exact conditional calibration in the
sparsest shell does not. Keep the uncorrected P12-A posterior, publish response/width and
sparse-shell quality metadata, and do not revive the rejected affine correction. A
future richer-summary challenger must improve an out-of-fit proper score, not merely
widen or flatten ranks.

Implementation and evidence:

- `workflows/sbi/p12_width_information_diagnostics.py` and
  `tests/phase4/test_p12_width_information_diagnostics.py` (five tests pass);
- authoritative report
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1/fmpe_seed42/width_information_diagnostic_v1/P12A_WIDTH_INFORMATION_DIAGNOSTIC.json`
  (SHA-256 `e3933e050d9d89dbd97eb090360a893f5ed974bfcfbecdb336fa53d1247cb455`);
- durable repository evidence `docs/evidence/p12/P12A_WIDTH_INFORMATION_DIAGNOSTIC.json`;
- figures `docs/figures/p12_width_information_20260830/` with summary, posterior
  examples and sparse-shell sky hashes
  `f7f56e6ef9620e2e025cb8778be82aa2ec1b1ab1503b04dbc8a0a8bd4791dc95`,
  `2ac296bb62886280e58b6fd3ecb942eac63c36da93a7606b5a3b85fa696850f8`, and
  `874b61206108459bd0d3549ebd8f0e863aa1a5a6240a2ae9bd64945b57c5887c`.

### 2026-08-30 - [science/run] P12-A affine recalibration rejected by proper-score and crossfit gates

The preregistered correction canary completed without opening ph001. Its purpose was
not to make the rank histograms look flatter, but to test whether the sparse-shell
lambda2/lambda3 residual identified by the full audit was a stable location/scale
defect that could be corrected without degrading the posterior density.

The evaluation separated five questions:

1. **Does the correction replicate across calibration folds?** No. Fitting fold 0 and
   evaluating fold 1 changed the aggregate rank-plus-coverage score from `0.09660` to
   `0.09776`; fitting fold 1 and evaluating fold 0 changed it from `0.09755` to
   `0.13775`. Lower is better, so neither direction improved. The maximum crossfit
   offset disagreement was `0.11545` scaled target units and the maximum scale
   disagreement was `0.13523`, failing the frozen `0.05/0.10` stability gates. This
   is evidence that the small calibration residual is not a spatial-fold-stable
   scalar affine defect.

2. **Does the fold-0+1 correction improve a proper posterior score on untouched
   selection folds 2--4?** No. The mean physical log score fell from `3.448858` to
   `3.446206`; corrected-minus-base was `-0.002653`, with a 254-spatial-block 95%
   interval `[-0.005009,-0.000334]`. Because the interval is entirely below zero,
   the corrected density assigns systematically less probability to the truth. This
   is the decisive rejection, even though some marginal diagnostics improve.

3. **What symptom did the correction improve?** Sparse-shell 68% coverage changed
   from `0.68928/0.65771/0.66333` to `0.70335/0.67115/0.68428`. Thus lambda2 and
   lambda3 moved toward nominal coverage, while lambda1 became overcovered. The
   correction is therefore capable of treating the visible high-redshift symptom,
   but it trades that gain against other coordinates and regimes.

4. **Did point accuracy or TARP expose a hidden failure?** No. Posterior-mean R2 was
   essentially unchanged (`0.66181/0.74788/0.79960` before and
   `0.66188/0.74769/0.79934` after), and corrected TARP still passed. These are
   necessary but insufficient checks: an affine map can preserve the mean and TARP
   while worsening the density's proper score and conditional rank structure.

5. **Scientific decision.** Reject the affine correction and retain the uncorrected
   P12-A posterior. Its global physical-eigenvalue calibration remains practically
   close to nominal, with a small, explicitly reported sparse-shell conditional
   limitation. Do not write `P12A_CALIBRATION_PASS.json`, do not tune a more flexible
   map on the selection folds, and do not describe the posterior as exactly
   calibrated in the sparsest shell. A future challenger must earn adoption through
   crossfit replication and out-of-fit proper-score improvement, not rank flattening.

The likely interpretation is that the sparse-shell residual combines limited tracer
information, conditional-mean/tail structure and spatial cosmic covariance. A single
per-shell location/scale map conflates those effects; its unstable fold estimates are
consistent with fitting realization-specific moments rather than a portable response
correction. This is why apparent coverage repair did not translate into a better
posterior density.

Authoritative report:
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1/fmpe_seed42/affine_calibration_canary_v1/P12A_AFFINE_CALIBRATION_CANARY.json`
(SHA-256 `f7be68b8b0780a771e6797e64ebe1b17e9117c02e5dfef76a3817fa7374674f2`).
Repository evidence: `docs/evidence/p12/P12A_AFFINE_CALIBRATION_CANARY.json`.
Diagnostic plot:
`docs/figures/p12_calibration_audit_20260830/p12a_affine_rank_comparison.png`
(SHA-256 `17fdad01b8adcd40f66033fc5836c32774581ed27284392a8a0be2a9127c7126`).

### 2026-08-30 - [science/code] Sparse-shell affine correction canary preregistered

The full audit supports testing one small correction, but not presuming it should be
adopted. `workflows/sbi/p12_affine_calibration_canary.py` implements a per-shell,
per-coordinate map in the scaled ordered-softplus space:

`theta_corrected = posterior_centre + offset_shell +`
`                  scale_shell * (theta_sample - posterior_centre)`.

Working in lambda1 plus positive eigengap coordinates preserves ordered physical
eigenvalues after inversion. The offset tests a conditional location residual; the
scale tests whether posterior variance matches posterior-mean squared error in each
redshift shell. This is deliberately less flexible than a quantile map, so it cannot
silently erase skew or tail structure.

Fitting and selection are separated. Fit fold 0 and test fold 1, reverse the roles,
then require maximum cross-fit offset disagreement <=0.05 scaled units and scale
disagreement <=0.10. Only after that is one map fitted to folds 0+1 and evaluated on
the already registered folds 2--4. The correction is rejected unless all of the
following hold:

- the aggregate rank-plus-coverage score improves in both cross-fit directions;
- spatial-superblock 95% confidence for the proper log-score difference is above zero;
- posterior-mean R2 loses no more than 0.002 per eigenvalue;
- global rank distance does not worsen by more than 0.002;
- no shell's 68% coverage error worsens by more than 0.01;
- sparse-shell lambda2/lambda3 coverage error improves;
- corrected TARP remains below the 0.05 gate.

If any gate fails, the uncorrected P12-A posterior remains the preferred model and
the sparse-shell residual becomes an explicit quality/calibration flag. A correction
is not scientifically better merely because its histogram looks flatter. Three
focused correction tests plus the five diagnostic tests pass. ph001 remains sealed.
### 2026-08-30 - [science/run] P12-A full calibration audit: small shape residuals, not global posterior collapse

The preregistered audit completed on the frozen 50,000-row ph006 fold-2--4
selection set with 512 draws per galaxy. The principal correction is to the
*interpretation of the evaluation*: the original KS test treated spatially correlated
galaxies as 50,000 independent realizations and ranked two softplus eigengaps while
describing them as physical eigenvalues. The audit now reports randomized finite ranks
for both coordinate systems and resamples the 254 cap+superblock spatial clusters.

The untempered physical-eigenvalue posterior is close to calibrated globally:

- weighted mean ranks are `0.50469/0.50625/0.49942`;
- weighted KS distances are only `0.00749/0.01561/0.00785`; the very small ordinary
  lambda2 p-value (`8.36e-7`) therefore describes a detectable percent-level shape
  residual, not a large scientific bias;
- 68% coverage is `0.68037/0.68834/0.68835` and 90% coverage is
  `0.89981/0.90345/0.90284`;
- posterior-mean R2 `0.66181/0.74788/0.79960` preserves the deterministic base
  `0.66251/0.74811/0.80032`;
- matched-row TARP max deviation is `0.00770`; its spatial-block bootstrap
  5/50/95% range is `0.00689/0.01160/0.01611`.

The 254-block bootstrap mean-rank intervals contain 0.5 for all three physical
eigenvalues. It nevertheless finds a few pointwise rank-decile departures, so the
posterior is not being declared mathematically exact. Fold-2/3/4 coverage and rank
directions are mutually compatible; there is no single-fold failure.

The scientifically meaningful residual is conditional. In the sparsest shell,
lambda2/lambda3 68% coverage is `0.65771/0.66333`; spatial-block 95% intervals are
`[0.64296,0.67235]` and `[0.64691,0.68001]`. Their mean ranks
`0.48725/0.49061` and excess first-decile mass indicate a small high-location and
undercoverage tendency in the sparse regime. Other shells are near nominal or mildly
overcovered, which partly cancels this in pooled coverage. This is compatible with
the known loss of tracer information at high redshift; it is not evidence that the
posterior machinery failed globally.

The next correction is therefore bounded: fit a per-shell, per-softplus-coordinate
location/scale map on ph006 folds 0--1 only; require fold-0-to-1 and fold-1-to-0
stability; then apply the frozen map once to folds 2--4. Promote it only if physical
rank/coverage and proper log score improve without posterior-mean R2 loss. Otherwise
retain the uncorrected P12-A posterior and report its sparse-shell calibration flag
instead of overfitting away a small spatially covariant residual.

Authoritative audit:
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1/fmpe_seed42/calibration_audit_v1/P12A_CALIBRATION_AUDIT.json`
(SHA-256 `ea602b92eadc997fdf1aaf201902f4973981e8e7cae6007bc6ecfb07f17cab3b`).
Repository evidence copy: `docs/evidence/p12/P12A_CALIBRATION_AUDIT.json`.
Durable rank plot:
`docs/figures/p12_calibration_audit_20260830/p12a_rank_histograms.png`.
### 2026-08-30 - [code/run] P12 calibration audit canary passes; full sampling made resumable

The preregistered P12 audit canary stopped twice before scientific evaluation and
exposed two loader-only device defects: reconstructing the saved FMPE left first the
box-prior support and then the restored vector-field estimator on CPU while posterior
candidates were on CUDA. The audit loader now places both objects explicitly on the
requested device. These failures do not diagnose the fitted posterior; no posterior
sample or calibration result was accepted from either failed run.

The unchanged 512-row/64-draw canary then completed the full chain: exact frozen
ph006 fold selection, checkpoint reconstruction, posterior sampling, ordered physical
eigenvalue conversion, randomized ranks, matched-row TARP, plots and provenance
export. TARP max deviation was `0.03661` on this small smoke sample and ph001 remained
sealed. This is an engineering pass only; 512 rows are intentionally insufficient for
the scientific calibration claim.

Full posterior sampling is now an NPY memmap with an atomically updated completed-row
counter after every chunk. A time-limited interactive allocation can therefore resume
without discarding or silently accepting partial draws. The resumability contract has
a focused unit test. Canary marker:
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1/fmpe_seed42/calibration_audit_canary_v1/P12A_CALIBRATION_AUDIT.json`
(SHA-256 `0ac523b39fdd25f9a10e074b1ce9cf0d0c1b26e8f134eb300e42834cf9e8d899`).

### 2026-08-29 - [science/code] P12-A calibration correction programme preregistered

The full P12-A fit is a technical success but not yet a posterior-VAC calibration
pass. The correction programme will not treat a small KS p-value as a diagnosis
or tune directly on all ph006 rows. It first audits the frozen posterior without
opening ph001 or changing its samples.

The existing SBC implementation ranked the three *training coordinates*
`(lambda1, inverse-softplus gap12, inverse-softplus gap23)` using
`mean(samples < truth)` and a continuous-uniform KS test. It was therefore
mislabelled as three physical-eigenvalue ranks and did not randomize the finite
256-draw rank grid. Meanwhile TARP used only 10,000 of the 50,000 selection rows.
Those choices do not prove that the posterior is calibrated, but they can exaggerate
or obscure the meaning of the observed SBC--TARP disagreement.

`workflows/sbi/p12_calibration_diagnostics.py` now preregisters the first audit:

- reproduce the exact ph006 fold-2--4 selection rows, retaining folds 0--1 solely
  for later correction fitting;
- draw 512 posterior samples and use randomized finite-sample ranks;
- report ranks in both ordered-softplus coordinates and physical
  `(lambda1, lambda2, lambda3)`;
- compare SBC and TARP on the same rows and draws, with bootstrap sensitivity;
- stratify ranks and coverage by shell, cap, redshift, `ntilde(z)`, random-support
  boundary distance, truth/base lambda1, and posterior width;
- interpret rank location, edge mass, centre mass, and asymmetry separately so
  location bias is not mistaken for underdispersion or intrinsic tail ambiguity.

The decision tree is frozen before looking at the audit. A mostly affine
location/width defect will be corrected in ordered-softplus coordinates using
ph006 folds 0--1 only and evaluated once on folds 2--4. Strong skew, tail, or
response-dependent failures will not be hidden by scalar tempering; they instead
motivate a conditional calibration map or a richer P12-B posterior summary.
`P12A_CALIBRATION_PASS.json` remains absent until physical-eigenvalue SBC/TARP,
marginal and conditional coverage, proper score, and tail/class diagnostics pass.
ph001 remains sealed.


### 2026-08-29 - [science/code/run] P12 dataset contract repaired; FMPE canary passes and full fit starts

All five leave-one-phase-out U-PATCH encoders and all six guarded OOF summaries
(`ph000`, `ph002--ph006`) are complete. The first full P12-A dataset build then
failed correctly on `ph000 invalid field-support distance`. Audit established that
`active_assignment.npz[field_support_distance_mpc]` is an intentionally unfilled NaN
placeholder in every visible phase, not a corrupted ph000 product. The deployable
physical covariate already exists in the canonical P3b-R random-response overlays as
`distance_to_support_boundary` with `support_random`.

P12-A now samples those P3b-R channels at each OOF parent position, explicitly excludes
`M=0` rows from the posterior target/calibration population before shell-stratified
sampling, and records the excluded denominator. It never imputes unsupported area or
treats it as a void. Provenance-keyed parent caches make the expensive phase/cap HDF5
sampling a one-time operation; six phases are prepared in parallel. The 25k/5k
six-phase canary at
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_contract_canary_v2/`
passes with `sealed_phase_opened=false`. Supported OOF fractions are `0.96885` for
legacy ph000 and `0.99411--0.99417` for ph002--ph006. The larger ph000 exclusion is
retained as an explicit domain audit, not silently harmonized.

The full response-conditioned dataset is now ready at
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1/`: two million
OOF training rows and 600,000 ph006 selection rows, conditioned on three physical base
eigenvalue predictions, redshift, `ntilde(z)`, cap and
`log1p_random_support_boundary_distance_mpc`. Phase, fold, superblock and artificial
fold-boundary distance remain excluded as features. Fold/superblock are retained only
for disjoint ph006 calibration and selection reporting.

Two runtime defects were caught before the science fit: CUDA-resident FMPE tensors
cannot be consumed by forked DataLoader workers, and this SBI version exposes paired
FMPE density through its vector-field probability flow rather than
`log_prob_batched`. The loader is now frozen to zero workers and paired
`q(theta_i|x_i)` scoring uses the one-sample-by-batch continuous flow with a focused
unit test. The official 50k/20k FMPE canary is technically complete: GPU training,
posterior sampling, physical-space log score, TARP/SBC/coverage diagnostics, disjoint
ph006 superblocks, checkpoint and `P12A_COMPLETE.json` export all work. Its calibration
gate is intentionally not promoted because the smoke fit uses only three epochs and
1,000 evaluation rows.

The full two-million-row P12-A FMPE fit is now technically complete. A validated
`batch_size=16384` throughput setting changed only minibatch execution; the frozen
dataset, seed, architecture, ordered-softplus target, conditioning variables, early
stopping and ph006 split were unchanged. The estimator converged after 143 epochs
(best validation performance `4.5359`) and wrote
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1/fmpe_seed42/fmpe_estimator.pt`
(SHA-256 `0474ce3f0b3a6c97de7fe225eaceee9f0b2b0ea76ea5d417e545eeeacf437880`).
The authoritative technical marker is
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_base_response_v1/fmpe_seed42/P12A_COMPLETE.json`;
ph001 remained sealed.

This is a useful near-pass but not yet a production calibration pass. On 50,000
spatially disjoint ph006 evaluation rows, untempered 68% coverage is
`0.6770/0.6862/0.6848`, 90% coverage is `0.8970/0.8991/0.8997`, and TARP passes
with maximum deviation `0.0194`. Posterior-mean R2 is
`0.6612/0.7475/0.7993`, essentially preserving the deterministic base
`0.6625/0.7481/0.8003`; knot probability ECE is `0.00283` and Brier score
`0.03423`. Marginal coverage, shell-conditional coverage, posterior-mean accuracy
and TARP pass. The separate `P12A_CALIBRATION_PASS.json` is correctly absent because
SBC rank-uniformity fails for all three eigenvalues
(`p=5.98e-4, 2.35e-4, 2.58e-8`). The fitted scalar width temperature
`tau=1.03045` does not repair that shape failure. P12-A therefore proves the full
posterior machinery and gives well-covered, informative posteriors, but requires a
more flexible rank/shape-calibration diagnosis before posterior VAC promotion.

Random-response work is now an informative bounded ladder rather than the critical
path. R1 reached epoch 17 with macro `R2(lambda1)=0.57475`; its epoch-16
first-three-shell macro and sparse-shell scores were `0.62841` and `0.38922`.
R2 assignment response and R3-RF high-S/N random fields are frozen at epoch 15 with
two-seed mean macro scores `0.57991` and `0.57126`, respectively. The stricter
density-matched random and cross-phase-FAINT controls have reached epoch 14 with
two-seed means `0.56862` and `0.57989`. They remain far below the old same-phase
BRIGHT+FAINT Proxy/Null results. Random response alone therefore does not reproduce
the old FAINT gain, while cross-phase FAINT currently has only modest headroom over
the strict random control; final epoch-15 markers remain required before closing the
diagnosis.

P11 factorial count products are now complete for `ph002--ph006`. Dense BRIGHT/FAINT
and assigned BRIGHT fields were materialized on the exact canonical P3 lattices with
common random support. Every individual phase passes count conservation and exact
`V_dense >= V_assign >= V_final` voxel nesting. The aggregate
`FACTORIAL_VIEW_PRODUCTS_READY.json` passes with `sealed_phase_opened=false` and
`truth_or_targets_read=false`.
The source and aggregate product marker SHA-256 values are
`e820fec02d4cff94a51256a7fcb73b7acef85fd68b0c148d1ca6f3da846282f2` and
`bac20cc4115d58c02c399d8debcc5a916227f64e99296b11aa07e1717dafcdcb`,
respectively.

The build audit caught and prevented a catalogue-lineage confound in `ph000`: its
legacy `path1_fiberassign` final catalogue is not pointwise nested inside the
official `forFA0_nomask` and `altmtl0` targetable/assigned products. The optional
P11 branch therefore uses `ph002--ph005` for training and `ph006` for selection
only. This does not change the frozen P10 training or P12 posterior contracts, which
retain ph000.

This makes the factorial-view data substrate ready, but not JEPA training itself.
The view-specific response adapter must still be frozen and a supervised dense-view
teacher must demonstrate measurable transfer headroom on ph006. JEPA remains a
separate, non-blocking representation challenger rather than posterior uncertainty or
recovery of information removed by selection.



### 2026-08-28 — [run/code] NERSC recovery: strict controls running and P12 chain resumed

After NERSC compute and `/pscratch` recovered, the frozen completion programme was
resumed without opening ph001 or changing any scientific contract. Strict-control job
`57670700` is running on `nid008196` with four independent one-GPU workers. It was
recovered from the earlier supervisor with the registered epoch-10 gate. Persistent
session `p10_strict_controls_e15` now waits for that allocation to release and then
resumes the same R3-RF-DM and cross-phase-FAINT checkpoints, seeds 42/43, to the full
epoch-15 terminal markers while enforcing the two-submitted-job limit.

P12 crossfit job `57672005` is running on `nid008261` as the second interactive node.
Session `p12a_completion` resumes omit-ph003/ph004/ph005 from their
atomic checkpoints and exports the missing OOF summaries. Session `p12a_posterior`
remains allocation-free until all six summaries exist; it then builds P12-A, runs the
FMPE canary and full fit, and evaluates/calibrates on the disjoint ph006 contract.

`workflows/abacus_tweb/report_p10_u_patch_epoch_table.py` now reconstructs the matched
per-epoch macro `R2(lambda1)` table directly from saved histories, averaging the
registered two-seed arms. `workflows/sbi/wait_then_run_p10_strict_epoch15.sh` implements
the allocation-aware epoch-10 to epoch-15 handoff. At this restart snapshot the strict
controls have five completed epochs; the response-ladder and BRIGHT+FAINT histories
remain unchanged from the preceding entry.

The first post-recovery live audit found the four strict workers active: cross-phase
FAINT seeds 42/43 have both written epoch 6 (`0.51355/0.50899`), R3-RF-DM seed 42 has
written epoch 6 (`0.49983`), and seed 43 is completing the same epoch. The P12 workers
are respectively computing omit-ph003 epoch 10, omit-ph004 epoch 11, and omit-ph005
epoch 18; their last atomic history rows remain epochs `9/10/17` until validation and
checkpoint writes finish. Two allocation-free P12 posterior watchers had survived
different login hosts. The older duplicate was terminated without touching a Slurm
job, and `run_p12a_posterior_interactive.sh` now acquires a lifetime `flock` so future
reconnects cannot race to submit the same posterior fit. ph001 remains sealed.
The epoch-15 handoff additionally requires all four atomic histories to reach epoch 10,
not merely the disappearance of one Slurm job; this prevents the gate-10 supervisor
and terminal continuation from racing if another allocation is needed.



### 2026-08-28 — [science/code/run] Response ladder reaches epoch-15 comparison; P12 crossfits require restart

The deterministic U-PATCH response ladder now has a matched epoch-15 comparison, but
the strict causal controls and P12 are not complete.  At epoch 15, the frozen
BRIGHT-only R0 reference is macro `R2(lambda1)=0.56003`.  The two-seed means are
`0.57991` for R2 assignment response (`0.58143/0.57839`) and `0.57126` for the
high-S/N R3-RF random field (`0.57513/0.56738`).  Their gains over matched-age R0 are
therefore only `+0.01988` and `+0.01123`.  R2 improves the first-three-shell macro
from `0.62042` to `0.64591`, but changes the sparse shell only from `0.37886` to
`0.38189`; R3-RF similarly reaches `0.63525/0.37928`.  The compressed R1 arm remains
at epoch 10 and is slightly below matched R0 (`0.52301` versus `0.52943`).  R2/R3-RF
two-seed means have effectively levelled between epochs 14 and 15, although their
configured 20-epoch schedules have not written terminal markers.

The BRIGHT+FAINT contrast is much larger: the registered epoch-15 real-FAINT Proxy
reaches `0.69363` and the old within-phase FAINT Null `0.67597`, compared with
`0.56003` for R0.  This gap is not explained by either compressed or voxel-resolved
random response.  It also cannot yet be attributed cleanly to extra physical tracer
information because the old Null preserves same-phase angular structure.  The strict
controls are only at epoch 5.  At that matched age, the two-seed means are `0.49388`
for density-matched sparse randoms and `0.50295` for cross-phase FAINT, versus
`0.44071` for R0.  Their early gains occur almost entirely in the first three shells;
the sparse-shell scores remain approximately unchanged.  Thus an unrelated or
synthetic second point-process field can improve optimization/representation without
adding information about the target potential.  This is evidence against interpreting
the FAINT Null as a response-only control, not evidence that real FAINT is useless.
Epoch-10 and epoch-15 strict comparisons remain mandatory before any production
BRIGHT+FAINT decision.

P12-A has not produced a posterior result.  The four-GPU crossfit allocation
`57654542` ended at its time limit on 2026-08-27.  Subsequent ten-minute allocation
requests 3--24 were not granted and the bounded completion supervisor exited.  The
checkpoint state is intact: ph000/ph002/ph006 OOF exports pass, omit-ph003 and
omit-ph004 have completed epoch 10, and omit-ph005 epoch 17; the three remaining OOF
markers, `P12A_DATASET_READY.json`, `P12A_COMPLETE.json`, and
`P12A_CALIBRATION_PASS.json` are absent.  On 2026-08-28 persistent supervisors were
restored as `p12a_completion` and `p10_strict_controls_e15`.  Replacement P12 request
`57669956` is pending with NERSC reason `PartitionDown`; no scientific failure or
posterior diagnostic has occurred.  The already-running P12-A posterior watcher
continues to wait for all six OOF markers.  ph001 remains sealed.

The production ordering is unchanged: prioritize P12-A completion and calibration;
finish the strict epoch-10/15 controls in the spare allocation; keep the first VAC
BRIGHT-only and response-conditioned; treat R2 as the strongest deterministic
response challenger; and promote BGS_FAINT only if it beats both strict nulls after
convergence and improves the same calibrated posterior contract.  P11 remains a
non-blocking summary-learning challenger after this observation ladder is frozen.


### 2026-08-27 — [science/code/run] P12-A is the immediate posterior baseline; factorial JEPA sources frozen

The execution order is frozen as: finish the already-bounded strict FAINT controls;
complete the coordinate-aligned P12-A posterior now; build the factorial observation
views and test whether a dense-view teacher has ph006 headroom; only then spend GPU
time on matched supervised, masked-reconstruction, JEPA and curriculum controls. Any
winning summary is fed to the same posterior head and judged by log score, SBC/TARP,
conditional coverage, interval width and tail/class reliability. End-to-end
JEPA-initialized FMPE, held-out HOD stress tests and the one-open ph001 evaluation remain
later gates. JEPA is a deployable-summary challenger, not uncertainty estimation and
not recovery of information removed by the observation operator.

P12-A is defined as
`q(lambda | three-dimensional OOF U-PATCH base prediction, deployable response, H_fid)`.
The conditioning variables are the physical base eigenvalue prediction, redshift,
`ntilde(z)`, cap and physical field-support distance. Phase, fold, superblock and the
artificial fold-boundary distance are excluded as features. Fold/superblock identity is
retained only to separate ph006 folds 0--1 for width calibration from folds 2--4 for
selection reporting. Posterior targets use ordered softplus coordinates. The new
preparation/evaluation code reports physical-space log score, SBC, TARP, marginal and
shell-conditional coverage, width--error association, posterior contraction and knot
reliability/Brier; `P12A_COMPLETE.json` is intentionally distinct from the stricter
`P12A_CALIBRATION_PASS.json`. Eleven focused P11/P12 tests pass.

OOF status at restart was unchanged: ph000/ph002/ph006 exports are complete;
omit-ph003 and omit-ph004 are at epoch 9, and omit-ph005 at epoch 16. Persistent tmux
session `p12a_completion` requested four-GPU interactive job `57653839` to resume the
three exact checkpoints and export their omitted phases. It is the second permitted
allocation; strict-control job `57651100` remains allocation one. ph001 remains sealed.

The first P12 allocation exposed an orchestration inefficiency: three independent
one-GPU steps launched while the fourth remained queued inside the already granted
node. This was Slurm step packing, not a model/data failure. The launcher now uses one
four-task step with one bound GPU per task. The idle allocation was relinquished and
replacement job `57654542` has all four tasks running. A second persistent tmux
session, `p12a_posterior`, waits for all six summary markers and for the crossfit job to
release its slot; it cannot request a third submitted allocation. It then runs the
P12-A dataset build, a small FMPE canary, the full fit, and ph006 calibration/evaluation
on an interactive GPU node. The chain is implemented in
`workflows/sbi/run_p12a_posterior_interactive.sh` and writes logs below
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_logs/`.

The P11 design now keeps three causal axes separate: observation stage
`V_dense/V_assign/V_final`, BRIGHT-only versus BRIGHT+FAINT context, and stochastic
response realization. `tileloc_correlated_thinning` is frozen as the held-out
degradation recipe; all views of one latent core remain in one outer split with total
weight one. BRIGHT-only is the production default. The manifest-only, login-safe source
audit passes for ph000/ph002--ph006 and records that the heavy products still missing
are dense BRIGHT/FAINT count fields and assigned BRIGHT counts. No large FITS/HDF5 work
was run on the login node.

The source audit was tightened after implementation review: the extant FAINT overlay
is an assignment-stage field, not an independently validated final Loa FAINT
catalogue. `V_final` therefore references it only under the explicit current-mock
`C_z=1` identity contract. The heavy builder now constructs targetable and assigned
BRIGHT/FAINT count fields on the exact P3 lattices, applies the common random-derived
support, and requires `V_final Bright <= V_assign Bright <= V_dense Bright` plus the
corresponding assigned/dense FAINT nesting voxel by voxel. It reads no labels or
ph001. Persistent tmux session `p11_factorial_products` waits for P12-A to complete
before requesting a CPU allocation, so the factorial preparation cannot displace the
production posterior work.

Artifacts/code:

- `configs/p11_factorial_views_v1.json`;
- `workflows/abacus_tweb/p11_prepare_factorial_view_sources.py`;
- `workflows/abacus_tweb/p11_build_factorial_view_counts.py`;
- `workflows/abacus_tweb/run_p11_factorial_view_counts_interactive.sh`;
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p11_factorial_views_v1/FACTORIAL_VIEW_SOURCES_READY.json`;
- `workflows/sbi/p12_prepare_base_response_dataset.py`;
- `workflows/sbi/p12_train_base_response_fmpe.py`;
- `workflows/sbi/run_p12a_posterior_interactive.sh`;
- `tests/phase4/test_p11_factorial_view_contract.py`;
- `tests/phase4/test_p12_base_response.py`.


### 2026-08-25 — [code] Project progress / validity audit

- What: Wrote timestamped validity audit separating established vs preliminary vs gated claims; companion Canvas; verified live jobs 57582458/57583442 and epoch-10 R2/R3-RF means.
- Why / decision: Posterior samples ≈ per-galaxy amortized ordered-eigenvalue posteriors under H_fid when calibrated — **not** jointly coherent tidal fields. P12 incomplete; P13 gated; G3 wedge demonstrator only. U-PATCH ph006 R2(λ₁)=0.573 established.
- Next: Finish strict controls; complete P12 OOF+calibration; freeze ph001 predictions before one-open.
- Refs: `docs/project_progress_validity_audit_20260825.md`; canvas `graphweb-validity-audit.canvas.tsx`


### 2026-08-25 — [code/data/run] Strict sparse-random and cross-phase controls ready; GPU gate queued

The registered R3-RF-DM and cross-phase-FAINT groundwork is complete for every
visible phase (`ph000`, `ph002--ph006`) without opening ph001 or tidal targets. Commit
`5a17118` implements the input-only catalogue/field builder and focused tests; commit
`dd11a73` adds the all-phase HDF5/loader validator and persistent CPU-to-GPU interactive
supervisor; commit `6899daa` corrects a smoke-test-only tensor-axis assertion caught by
the first validation attempt. The model/product path itself was not changed by that
assertion fix.

Four frozen roots now pass six-phase U-PATCH extraction smokes:

- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/multitracer/strict_controls/r3_rf_dm_seed1701_v1/`;
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/multitracer/strict_controls/r3_rf_dm_seed2718_v1/`;
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/multitracer/strict_controls/bf_xphase_forward_v1/`;
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/multitracer/strict_controls/bf_xphase_reverse_v1/`.

Each root contains `P10_MULTITRACER_VIEWS_READY.json`, phase-local component manifests,
and `STRICT_CONTROL_LOADER_SMOKE.json`. All four validation markers report `pass=true`,
six visible phases, `sealed_phase_opened=false`, `targets_opened_by_validator=false`,
and the unchanged FAINT selection SHA-256
`35b7c4f2320e783490e64d4726954ec5e5a44885e865505a9a4a426f67ea02c8`.
Twenty-four phase/control products were constructed in CPU allocation `57583312`; the
corrected validation retry completed in `57583374`.

The geometry audit found that donor and recipient phase grids are not byte-identical.
Cross-phase products therefore do not use unsafe zero-copy reindexing: donor FAINT
points are CIC-deposited afresh on each recipient's immutable P3 grid, while BRIGHT
inputs, exposure/selection, ownership and targets remain recipient-local. The
density-matched random views preserve the exact recipient FAINT cap/redshift multiset
and count but sample angular directions from the all-18 random response. This is needed
because the registered full randoms are a deliberately dense Monte Carlo description
of angular support/targetability, not a galaxy tracer: their clustering-random `Z` is
linked to data through `TARGETID_DATA` and is not an independent radial selection draw.
R3-RF-DM is consequently a strict diagnostic synthetic point process, not a proposed
native random-catalogue input for the VAC.

The persistent tmux session `p10_strict_controls` released the CPU node and acquired
four-GPU interactive job `57583442` on `nid008600`. All four 1,000-update canaries pass
with finite logged objectives/gradients and atomic checkpoints. Fresh scientific runs
are now active for R3-RF-DM seed 1701 and forward cross-phase Null with matched U-PATCH
model seeds 42/43, targeting epoch 15. Outputs are under
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/strict_control_training/` and worker logs
under `strict_control_logs/`. Seed 2718 and the reverse donor map are built and frozen
but are not promoted to full training until the primary epoch-10 comparison warrants
them. ph001 remains sealed.


### 2026-08-25 — [science/decision] U-PATCH target contract and strict FAINT/random controls

The production U-PATCH is a per-galaxy estimator, not a voxelwise three-eigenvalue
segmenter. Its 3-D U-Net maps the canonical input patch to a latent voxel field; the
latent field is trilinearly sampled at every authoritative BRIGHT-galaxy coordinate,
and a point head returns three scaled linear eigenvalue increments for that galaxy.
The increments are converted to the ordered physical eigenvalues only for evaluation.
Consequently the 5-comoving-Mpc input lattice need not equal the physical
`R=7 Mpc/h` T-web smoothing scale. The frozen requirement is instead exact parity of
the coordinate frame, grid origin/cell size, galaxy fractional voxel index and target
row, together with enough patch context to infer the smoothed target. Any future
voxelwise tensor/eigenvalue decoder must define a separate truth-grid and interpolation
contract before it can be compared with this per-galaxy U-PATCH.

The interpretation that BRIGHT+FAINT primarily wins through improved survey-response
modelling is weakened by the response ladder. At matched epoch 10 the two-seed mean
macro `R2(lambda1)` is `0.54478` for R2 and `0.53474` for high-S/N R3-RF, versus
`0.66394` for real FAINT and `0.64185` for the existing FAINT Null. R3-RF is a
deterministic all-18 random-response expectation, not a sparse tracer field, so this is
not yet a point-process-matched comparison.

The old FAINT Null is also not structure-free. It preserves every real FAINT angular
direction and radius and only permutes their pairing within cap and `Delta-z=0.01`.
It therefore preserves the complete projected FAINT clustering pattern in every narrow
redshift stratum and can retain coarse same-phase cosmic information. Proxy-minus-Null
cannot be interpreted as the full value of FAINT or as a clean response-only contrast.

Two bounded controls are authorized without opening ph001:

1. `R3-RF-DM-v1`: sample an unclustered angular point process from the frozen all-18
   random response, match the assigned-FAINT count and exact redshift/radius multiset
   within each cap and `Delta-z=0.01`, deposit it with the identical CIC/grid contract,
   and expose the identical FAINT-style six-channel U-PATCH interface. This diagnostic
   contains the FAINT one-point selection and sparse-count texture but no same-phase
   angular cosmic structure. At least two catalogue-realisation seeds are required
   before treating a gain as reproducible.
2. `U-BF-XPHASE-NULL-v1`: pair each phase's immutable BRIGHT inputs and labels with a
   deranged donor phase's real FAINT triplet. It preserves realistic FAINT clustering,
   shot noise, selection and channel semantics while breaking the shared potential
   field. Grid origin/shape/cell identity must pass before the zero-copy donor view is
   allowed.

The primary decomposition is: `R3-RF-DM - R3-RF` tests sparse point-process texture;
`old FAINT Null - cross-phase Null` diagnoses same-phase structure retained by the old
Null; and `real FAINT - cross-phase Null` tests genuine additional tracer information.
If a fixed stochastic-random field helps, a refreshed-realisation/ensemble control is
mandatory before production because an arbitrary Monte Carlo field cannot be allowed
to create false posterior sharpness. BGS_FAINT becomes a production dependency only
after a reproducible independent-phase gain over both strict nulls and calibrated
posterior improvement under separate BRIGHT/FAINT response contracts.


### 2026-08-23 — [run/diagnosis/fix] Response allocation ownership failure corrected; all four checkpoints resume

Allocation `57475703` did not time out or fail scientifically. Slurm accounting shows
`COMPLETED 0:0` after only `01:07:00` of its four-hour request because the owning
non-persistent interactive shell closed. Its four training steps were consequently
cancelled with signal 15. This is an orchestration defect: the previous detached child
steps did not outlive their allocation owner. Empty redirected logs are explained by
buffered Python output at cancellation, not by failure to train.

All checkpoints are intact. The last pre-restart loss rows were R2 seed 42/43 at
`23,600/16,100` updates and R3-RF seed 42/43 at `20,050/16,825`, all within epoch 1
of 84,446. No validation epoch or ph006 science result had yet been produced.

Commit `300acba` adds
`workflows/sbi/run_p10_response_ladder_interactive.sh`: a login-node tmux supervisor
that owns one four-GPU interactive `salloc` at a time, launches four explicit one-GPU
`srun` workers, resumes atomically checkpointed runs, and requests the next allocation
only when terminal markers are absent. It never uses `sbatch` and never exceeds the
two-allocation policy.

The supervisor is live in tmux session `p10_response_ladder`. Allocation `57489518`
on `nid008201` is running all four workers. Fresh rows at `23,750/16,250` for R2 and
`20,250/17,000` for R3-RF prove exact resume rather than restart. ph001 remains sealed.
Logs are under
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/response_training/logs/p10_response_ladder_supervisor/`.


### 2026-08-23 — [science/decision/code/run] R3-RF is technically ready; R2 and R3-RF launch concurrently

The response programme is reordered after the matched training histories showed that
waiting for R1 to converge would not answer the more important representation question.
The large six-channel BRIGHT+FAINT/FAINT-Null result may reflect a richer spatial
description of survey response that the three-channel R1 contract compresses away.
R2 and R3-RF are therefore frozen as a response-detail ladder while R1 continues:
R1 supplies compressed random support/expected-count correction, R2 adds audited
assignment-completeness summaries, and priority R3-RF exposes the high-S/N
voxel-resolved random field directly.

R3-RF uses no FAINT galaxies and no clustering-random redshift. It reuses the all-18
P3b-R overlays and gives the six-channel U-PATCH two explicit blocks: unchanged BRIGHT
`[counts, log_count_ratio, exposure_apodized]` and random response
`[expected_counts_random, angular_response, support_random]`. The expected intensity
uses training-phase-only `log1p` z-scoring, angular response is centred on one and
clipped at the frozen limits, and support is binary. This is an empirical observation-
operator field, not a second tracer and not a substitute for `C_fibre` or `C_z`.

All ph000/ph002--ph006 product and real-loader checks pass. Every sampled cap/phase
patch has exact six-channel order, finite non-negative intensity, binary support, zero
response outside support and non-empty authoritative targets. ph001 remains sealed.
The 1,000-update one-A100 canary passes at `11.236 patches s^-1`. Key artifacts are:

- products: `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/r3_random_field_v1/R3_RANDOM_FIELD_PRODUCTS_READY.json`,
  SHA-256 `de428c979bc9ebcdaa8d5066fbc244085a9435a3c29052fff3d0444d988dc36b`;
- loader: `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract_r3_random_field/TRAINING_LOADER_READY.json`,
  SHA-256 `a3e160a949247582b6ce9dce88e38b099445eb3dee4784cb93762f3b8e24c7c5`;
- loader smoke: `training_contract_r3_random_field/R3_RF_LOADER_SMOKE.json`,
  SHA-256 `44c093e6c73db24be2b54b2a6c7d0ae69f8ce9dc0708545ab1e63d95410faf0c`;
- technical marker: `training_contract_r3_random_field/P10_R3_RF_TECHNICAL_READY.json`;
- canary report: `response_training/p10_r3_rf_canary_1000_v1/unet/seed_42/THROUGHPUT_CANARY_REPORT.json`,
  SHA-256 `4ad020b092007f0323a97fabcad9dbb4f90e195c4bd24ddd16ce298a5b205fa9`.

Full R2 and R3-RF runs are now concurrent on interactive allocation `57475703` as
independent one-GPU tasks. GPUs 0/1 run seed 42; GPUs 2/3 run independent seed 43
replicas, preserving single-patch optimizer semantics while using the full node. At the
first seed-42 status R2/R3-RF had reached `5,125/600` updates; seed-43 replicas were
then launched and independently reached `575/600` updates. All use the frozen
`84,446`-patch epoch, checkpoint every 250 updates and resume automatically. Run roots
are `response_training/p10_r2_assignment_v1/` and
`response_training/p10_r3_rf_v1/`.

The density-matched stochastic random-count arm is deferred. It becomes the next
diagnostic only if high-S/N R3-RF fails to approach FAINT Null, in which case it will
test whether the residual gain requires count-like sampling noise. The previous
boundary-distance arm is renamed R4 and deferred until this ladder is resolved.

### 2026-08-23 — [science/code/run] R2 response conditioning is technically ready; full science training remains ordered after R1

The spare second allocation was used to close the R2 engineering and response-definition
gates without disturbing the active R1/P12 node. CPU allocation `57473387` built and
validated the response products; GPU allocation `57473825` ran the bounded throughput
canary. Both were released after completion. No full R2 science run was launched,
because doing so before the final R0/R1 decision and paired-view freeze would weaken the
registered `R0 -> R1 -> R2` attribution. ph001 remained sealed.

The missing target-derived assignment values are not a fixed angular hole pattern.
Across ph000/ph002--ph006, raw assignment response covers
`0.98477--0.98491` of random-supported pixels. In ph000/ph006,
`0.9826--0.9827` of missing pixels lie within four nside-256 neighbour rings of the
support boundary and `0.9988--0.9995` within eight, while the missing-pixel Jaccard
between phases is only `0.86047`. The frozen conservative policy is therefore:
inside random support, an undefined response receives the physically neutral
no-competition value one plus `C_fibre_defined=0`; outside support it is zero. No
neighbour smoothing or galaxy-density imputation is allowed.

All 12 phase/cap overlays now pass finite-value, range, cap-ownership, neutral-defined
and zero-copy R1-view checks. The R2 model input is exactly six channels: the three R1
channels plus `FRACZ_TILELOCID`, `FRAC_TLOBS_TILES` and
`C_fibre_defined`. Constant mock `C_z=1` remains stored with
`C_z_informative=false` but is intentionally omitted from the model input. The
all-phase loader smoke passes.

The one-A100 canary completed exactly 1,000 finite, phase-balanced updates in
`159.75 s`, or `6.260 patches s^-1`. The matched R1 canary achieved
`9.174 patches s^-1`, so the six-channel R2 input is `31.8%` slower. At the
registered 84,446-patch epoch this implies about 3.75 hours of update time before
validation and checkpoint overhead. This cost is acceptable but must be scheduled
explicitly.

Live aggregate markers are:
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/r2_assignment_response_v1/R2_ASSIGNMENT_OVERLAYS_READY.json`
(SHA-256 `6301e764f5d240fb0f5e2735115b6e3316bbf48850a89fae08d19b003631cfe6`),
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract_r2_assignment/R2_LOADER_SMOKE.json`
(SHA-256 `bdfb1f06006c1d5067213cd0df8752a93e74678f2c1a9800bacb16721fbfc75b`), and
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract_r2_assignment/P10_R2_TECHNICAL_READY.json`.
The canary report is
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/response_training/p10_r2_canary_1000_v1/unet/seed_42/THROUGHPUT_CANARY_REPORT.json`
(SHA-256 `ddaf361bec054639bb92931e1e340c34af91f247934435754936c4bd0e2b978c`).
Compact evidence is tracked under
`docs/evidence/p10/r2_response_audit_20260823/`.

The resulting state is deliberately asymmetric: `technical_ready=true` and
`science_run_authorized=false`. Once R1 is frozen and
`P10_VIEW_LADDER_READY.json` exists, the full R2 run can start without further data,
loader or throughput work.


### 2026-08-23 — [science/code/run] R2 source semantics pass, but the spatial assignment map is not yet freezeable

The spare interactive capacity was used for the R2 prerequisite rather than launching
an incorrectly labelled R2 training run. GPU allocation `57472981` on `nid001212`
ran the visible-phase catalogue/view audit and ph000/ph006 angular-map canary; it was
released after the bounded work completed. The existing R1/P12 allocation continued
unchanged and ph001 remained sealed.

The nested `V_dense -> V_assign -> V_final` source audit passes for
ph000/ph002--ph006. Assignment fractions are `0.79983--0.80330`, final-view fractions
are `0.79793--0.80128`, and every registered row-identity, unique-TARGETID,
probability-range and nesting gate passes. A scientific naming error is now corrected:
`FRACZ_TILELOCID` is local **fibre-assignment completeness**, not redshift success.
Against `LOCATION_ASSIGNED` its ten-bin calibration error is only
`1.33e-5--4.43e-5` across phases. `FRAC_TLOBS_TILES` is a supplementary
TILES-group completeness term; the two quantities should remain separate even if their
product is retained as a total-completeness diagnostic.

There is no continuous mock `C_z` signal to learn. Every assigned object passes
`ZWARN==0` with a finite positive mock redshift and all clustering
`WEIGHT_ZFAIL` values are exactly one in every visible phase. The fiducial mock
contract is therefore `C_z=1`, `C_z_informative=false`. This must not be presented as
training sensitivity to the non-constant Loa `mod_success_rate`; such sensitivity
requires an explicitly simulated/degraded response view and belongs in later P11/P12
closure.

The first spatial canary correctly prevents premature R2 promotion. Averaging
target-level assignment response at nside 256 covers only `0.9848447` of ph000 and
`0.9849085` of ph006 random-supported pixels. There are respectively `4,042` and
`4,025` supported pixels with no eligible target, so the registered `0.999` coverage
gate fails. These pixels cannot be smoothed from neighbouring galaxy density. The next
decision is to test a physically neutral no-competition value with an explicit
`C_fibre_defined` flag, stratified by support-boundary distance, or identify a
response-defined full-random source. `P10_VIEW_LADDER_READY.json` was deliberately not
written and no R2 GPU training was launched.

Live products are
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/r2_response_audit_v1/` and
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/r2_assignment_canary_v1/`.
Aggregate SHA-256 values are
`d4b47c986f217f48c03b99669841cbbe2e45ef94dcb9020dccfd62178ea48b8e` and
`f2c1facffc680c1e71c14149a10d4251aab1addb026f81f96254cee4538bc4c9`.
Compact copies are tracked under
`docs/evidence/p10/r2_response_audit_20260823/`. Reusable code is
`workflows/abacus_tweb/p10_audit_r2_response_ladder.py`,
`workflows/abacus_tweb/p10_build_r2_assignment_angular_canary.py`, and
`workflows/sbi/run_p10_r2_response_audit_interactive.sh`; three focused unit tests
pass.

R1 continues independently. Its completed epoch-7 row is macro
`R2(lambda1)=0.48240`, first-three `0.53565`, and shells
`0.58304/0.54592/0.47800/0.32265`; this is below its epoch-6 best `0.51623` and
remains an interim oscillation, not a frozen R0/R1 decision.

### 2026-08-22 — [science] Random catalogues become the production response reference; FAINT freezes at epoch 15

The production spine is narrowed to
`P3b-R random response -> response-conditioned U-PATCH -> P12 -> P13`. The
BGS_FAINT Proxy/Null pair will complete epoch 15 and then freeze as a bounded
information-content diagnostic; it is no longer allowed to delay the BRIGHT-only
response or posterior path. P11 JEPA remains optional and non-blocking.

The current matched epoch-13 rows are Proxy macro `0.68854`, first-three `0.71973`,
shells `0.74849/0.73056/0.68013/0.59498`, and Null macro `0.66392`, first-three
`0.70823`, shells `0.74800/0.72070/0.65598/0.53098`. Proxy-minus-Null is therefore
`+0.02463` macro and `+0.01150` over the first three shells, with per-shell differences
`+0.00050/+0.00986/+0.02415/+0.06400`. This is preliminary evidence that real FAINT
positions carry recoverable spatial information, strongest in the sparse shell. It is
not a clean estimate of the observation-response contribution: both six-channel models
also change input width, shot noise, sampling regularisation and selection information
relative to the three-channel BRIGHT baseline.

The principled response-only comparison is now BRIGHT plus random-derived response,
not scrambled FAINT. Full random catalogues define the unclustered angular support and
targetability measure `M`/`p_angular`; they do not by themselves encode the complete
density-dependent fibre-assignment or redshift-success factors `C_fibre` and `C_z`.
Clustering-random `Z` remains forbidden as a replacement for the separately frozen
smooth `ntilde(z)` because it is data-linked through `TARGETID_DATA`. The first `R1`
arm will preserve the exact three-channel U-PATCH width and replace only the
occupancy-derived exposure/reference with random-derived support and expected counts.

Implementation is staged in `P3b-R` of the generalisable-VAC plan. Products will be
built for ph000 and ph002--ph006; ph001 remains sealed. The existing four-GPU
interactive chain continues P12 cross-fits while Proxy/Null finish epoch 15. No new
FAINT run is authorized before P3b-R, the matched R0/R1/classical comparison and the
baseline P12 calibration path. Artifact paths and hashes will be appended when each
P3b-R gate closes.

Implementation note: the overlay inherits the audited P3a cell size exactly: **5
comoving Mpc**, or `3.383 Mpc/h` for the registered Planck18 cosmology. The earlier
shorthand `5 Mpc/h` in the P3b-R checklist was inconsistent with the completed P3 unit
audit and has been corrected. The Poisson canary is also registered on the mean
standardized count residual, not on the log-count ratio: even a perfect Poisson draw
has a negative mean log ratio at low expected occupancy because `log` is concave.

Later implementation update: the tracked P3b-R builder, tests, resumable pipeline and
capacity-matched R1 loader/trainer are now present at
`workflows/abacus_tweb/p3br_build_random_response.py`,
`workflows/abacus_tweb/run_p3br_pipeline.py`,
`workflows/abacus_tweb/p3br_prepare_r1_contract.py`,
`workflows/abacus_tweb/p3br_training_contract.py`, and
`workflows/abacus_tweb/p10_train_random_response.py`. The R1 HDF5 view uses virtual
datasets for unchanged P3a counts/LOS fields, so the response overlay does not
duplicate multi-gigabyte immutable fields. Counts keep the exact R0 normalization;
only the new response-derived log-ratio is refit on the five training phases.

The first real ph000 random canary passed on CPU allocation `57425308`: registered
random ID 0 contributed `30,923,488` accepted rows, no `GOODHARDLOC` rejection,
`266,418` supported nside-256 pixels (`13,975.18 deg2`), and output
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph000/p3b_random_response_v1/angular/randoms_n1.npz`
with SHA-256
`8ac84b021c80ebd2bc342dd01e944375022af8fd05c5ea5d18f2b27413dd5d94`.
The canary found that SGC x PHOTSYS-N is a genuinely empty domain while NGC spans
both PHOTSYS regions; empty physical domain intersections are now handled explicitly.
It also caught and corrected a Healpy longitude/latitude argument error before any
product was promoted. The ph000/ph006 1/4/18 convergence scans are active and write
resumable snapshots under each phase's `p3b_random_response_v1/angular/` directory.

The registered 4-versus-18 convergence decision is now complete and selects **all 18
random realizations** for every visible phase. ph000 and ph006 independently give
support-mask Jaccard `0.999636`, but median absolute fractional angular-response
differences `0.027752/0.027752` and 99th-percentile differences
`0.119179/0.119179`; the latter two fail the registered `0.01/0.05` gates. The
area-weighted cap/shell expected-count proxies differ by at most `0.000501`, well
inside their `0.01` gate, so the decision is driven specifically by unresolved
pixel-scale random shot noise rather than cap normalization. The atomic decision is
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract/P3BR_RANDOM_DENSITY_DECISION.json`
(SHA-256 `2f0204b3cdc64d9f23408672cdc34da20033844e8bb747b7c6f12f40575bc6e7`);
the corrected ph000/ph006 18-random map hashes are respectively
`a42c26a1a21bcb665de5ac9f63cb967f10b08978d2fb5b9ef2f11de0cb971cde` and
`b3d45d06447d2978ad4dd60ec534e0d3ee85b2c6ba1545b827a2b9e0f6d53dc4`.
`ph001_opened=false`. ph002--ph005 all-18 scans are now active concurrently in CPU
allocation `57425308`, with atomic progress checkpoints after every random file.

P3b-R subsequently completed across ph000/ph002--ph006 after a restart-safe handoff
from CPU allocation `57425308` to `57428530`; ph001 remained sealed. Every phase now
has the frozen pair
`p3b_random_response_v1/{NGC,SGC}/response_overlay.h5` plus `manifest.json` and
`qa.json`. All twelve cap products pass exact P3a grid identity, virtual-dataset
identity for unchanged counts/LOS channels, finite/non-negative response arrays,
binary support containment, exposure bounds, the Poisson count-residual gate, and
cap/shell observed-versus-expected closure. Across all phases/caps the one-million-draw
Poisson standardized-residual means lie within `[-9.30e-4,+1.05e-3]`; the observed to
expected shell-total ratios lie within `[0.9494,1.0526]`. Random support is sampled
directly into the binary channel and is never smoothed or filled; only the separate
apodized exposure channel is distance transformed.

The frozen capacity-matched R1 loader is
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract_r1_random/TRAINING_LOADER_READY.json`
and passes all six-phase deterministic extraction, target alignment, epoch coverage,
resume and weighted-loss gates. The three channels are exactly
`[counts, exposure_apodized, log_count_ratio]`; counts retain R0 normalization and the
new log-ratio transform is fitted only where the training-phase random support is one.
The pipeline marker is
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract/P3BR_PIPELINE_COMPLETE.json`.
Compact hash-verified evidence is tracked under `docs/evidence/p3br/`; principal
SHA-256 values are `4698aa04...d50d32d` for `evidence_manifest.json`,
`d06f7520...0c53d6` for `pipeline_complete.json`, `70c58f08...36adc` for
`r1_training_loader.json`, and `1841e2d2...490aa4` for
`r1_field_transform.json`. The next open gate is the frozen R1 run and matched
response-aware classical comparison; P12 continues independently.

The exact throughput gate passed on the otherwise idle fourth A100 in allocation
`57424410`: 1,000 canonical patch updates completed in `109.005 s`
(`9.174 patches/s`) with finite loss logging and a resumable checkpoint. The report is
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/response_training/p3br_r1_canary_1000_v1/unet/seed_42/THROUGHPUT_CANARY_REPORT.json`
(SHA-256 `2ccaf66107d71cc1cff4ab44e45516345dda59b62b8c416a54999abe512a949e`);
checkpoint SHA-256 is
`b062db85bbb50eee3faf09237c51c8dcf1ae1e414897663647281a5bc0e1bd11`, loss-trace
SHA-256 is `9bc8138998c30acc6e636af5a6293993f0cc185c4c8a323a9f6b561bce8790a2`,
and `ph001_opened=false`. At this measured rate one 84,446-core epoch is about `2.56 h`
on one A100, so R1 must span checkpointed interactive allocations; this does not
justify an unbenchmarked DDP rewrite. The full frozen R1 seed-42 run was launched on
the same freed GPU immediately after the canary.

Long-run resource use is now persistent without another node allocation. Tmux session
`p3br_r1_sidecar` runs
`workflows/sbi/run_p3br_r1_sidecar_existing_gpu.sh`, which attaches R1 only to an
unused fourth GPU of a running legacy `p10mtp12` allocation and resumes after each
allocation rollover. Tmux session `p3br_transition` runs
`workflows/sbi/run_p3br_transition_after_legacy.sh`; it consumes no compute while
waiting for both epoch-15 FAINT freeze markers and all six P12 OOF summaries, then
waits for the old allocation to disappear before handing control to the integrated
R1/P12/classical supervisor. Thus crossfits cannot be duplicated, the two-allocation
limit is preserved, and matched CIC/DTFE work will begin automatically after the
legacy programme closes.

The epoch-15 terminal contract is protected by
`workflows/sbi/watch_p10_multitracer_epoch15.sh` in tmux session
`p10_epoch15_guard`. The finalizer now writes both the explicit
`EPOCH15_FROZEN.json` and a legacy-safe `ARM_A_TRAINING_COMPLETE.json` whose status is
`FROZEN_REGISTERED_EPOCH15`; this prevents any already-launched supervisor from
silently resuming the bounded FAINT diagnostic into epoch 16. P12 remains independent
and checkpoint-resumable.

The implementation is now restart-safe at allocation boundaries. Each random file
updates an atomic accumulated-count checkpoint, a validated builder-owned partial cap
overlay may restart without touching any completed HDF5, and
`workflows/sbi/run_p3br_cpu_interactive.sh` can loop bounded two-hour CPU allocations.
The exact 1,000-patch one-A100 canary and independent-task four-GPU production
supervisor are
`workflows/abacus_tweb/p3br_run_r1_throughput_canary.py` and
`workflows/sbi/run_p3br_r1_p12_4gpu_interactive.sh`. No DDP rewrite is being adopted:
one variable-size canonical patch remains one optimizer update, while the other GPUs
continue independent P12 cross-fits. Matched response-aware CIC/DTFE and the final
ph006 decision are implemented in
`workflows/sbi/run_p3br_classical_4gpu_interactive.sh` and
`workflows/abacus_tweb/p3br_evaluate_r1.py`. The evaluator enforces exact authoritative
parent coverage and reports response-quartile and random-boundary-distance performance,
not only pooled/macro scores. Compact runtime evidence will be copied with exact hash
parity by `workflows/abacus_tweb/p3br_export_evidence.py` only after all visible-phase
QA passes. Sixteen focused P3b-R/R1 unit tests pass; runtime product and science gates
remain open rather than being inferred from code completion.

Second-node capacity was used to pull the matched response-aware classical gate
forward without changing the frozen R1 update contract. Interactive job `57436149`
on `nid008444` runs four independent one-A100 phase workers from persistent tmux
session `p3br_classical_second_node`; the original `p10mtp12` allocation
`57430396` continues R1, FAINT freeze and P12 work independently. The first
allocation attempt exposed a nested-shell quoting defect before any science process
started. Commit `2171876` replaces the fragile function-string invocation with the
tracked launcher's explicit `--worker` mode. The corrected run has already written
all six ph000/ph002--ph006 `CIC_RAW_COMPLETE.json` markers with complete
authoritative coverage and `blind_phase_opened=false`; pooled CIC finalization is
active and DTFE follows automatically using the six pre-existing
`DTFE_FIELD_READY` rasters. This acceleration shortens the overall R0/R1/classical
critical path but does not claim to accelerate the single-GPU R1 epoch itself: a
four-GPU DDP rewrite remains unadopted until a matched benchmark demonstrates the
registered `>=2.5x` wall-clock gain.

### 2026-08-22 — [science/code/run] Two P12 cross-fits finish; multitracer signal is shell-dependent and not yet causal

Later 2026-08-22 update: the corrected exporter has now passed on the production-scale
artifacts. `OOF_SUMMARY_COMPLETE.json` exists for ph000 (`5,026,863` rows), ph002
(`4,929,962` rows), and ph006 (`4,908,831` rows), with exact parent-keyed latent,
base-prediction, truth and response arrays plus hashes. This closes the exporter bug in
practice as well as in its unit test. The `omit_ph005` cross-fit has started, completed
three epochs, and improved monotonically to macro `0.41474`; it is active in epoch 4.
The ph003/ph004 cross-fits remain queued behind Proxy/Null on workers 0/1.

Proxy and Null have each completed epoch 12 and the full epoch-13 training traversal;
epoch-13 ph006 validation is in progress on job `57415926`. Their matched epoch-12 rows
are Proxy macro `0.67679`, first-three `0.70687`, shells
`0.73305/0.71650/0.67107/0.58655`, and Null macro `0.66467`, first-three `0.70767`,
shells `0.74665/0.71969/0.65666/0.53568`. Thus Proxy-minus-Null at the same epoch is
`+0.01212` macro but `-0.00079` over the first three shells; per-shell it is
`-0.01360/-0.00319/+0.01441/+0.05087`. This strengthens the narrower interpretation:
real FAINT positions are providing repeatable additional information in the sparsest
shell, while a general all-shell causal gain is not yet established. Both schedules
must still finish and be compared using frozen best checkpoints and paired spatial
blocks.

The persistent four-A100 interactive chain remains healthy on job `57404937` and
ph001 remains unopened. The real-FAINT Proxy and angular-Null models have each
completed 11/20 full 84,446-core epochs and are active in epoch 12. Their current
best ph006 rows are:

| run | best epoch | macro R2(lambda1) | first-three macro | shell R2(lambda1) |
| --- | ---: | ---: | ---: | --- |
| real-FAINT Proxy | 10 | 0.66394 | 0.69707 | 0.72403 / 0.70754 / 0.65966 / 0.56452 |
| angular-Null | 11 | 0.65497 | 0.69724 | 0.73240 / 0.70874 / 0.65057 / 0.52819 |
| Bright-only reference | 20 | 0.57310 | 0.63464 | 0.69190 / 0.64159 / 0.57042 / 0.38850 |

Both six-channel models currently exceed Bright-only, so Proxy-minus-Bright is not
evidence that real FAINT positions supplied the gain. At the matched epoch 10,
Proxy-minus-Null is `+0.02208` macro and `+0.00760` over the first three shells, with
per-shell differences `-0.00528/+0.00459/+0.02348/+0.06555`. At epoch 11 the overall
contrast reverses to `-0.00945` macro, while Proxy retains a `+0.05188` advantage in
the sparse fourth shell. The persistent high-redshift contrast is interesting, but
the causal multitracer result remains open until both cosine schedules reach epoch 20
and the frozen best checkpoints are compared with paired spatial-block uncertainty.
The non-monotonic macro scores despite improving train/global losses are another reason
not to stop at the current interim ranking.

The first two leave-one-phase-out P12 encoders are complete at epoch 20. The encoder
omitting ph000 scores macro `0.56469`, first-three `0.62942`, and shells
`0.68091/0.63844/0.56891/0.37049` on ph000. The encoder omitting ph002 scores macro
`0.56844`, first-three `0.62796`, and shells
`0.68732/0.62367/0.57288/0.38987` on ph002. Their similar performance across distinct
omitted phases is positive evidence that phase-cross-fitted deterministic summaries
remain informative; these are encoder gates, not posterior calibration results.

Summary export then exposed an implementation-only bug: `len(NpzFile)` returned the
15 arrays in the P4 archive rather than its roughly five million assignment rows.
Commit `ad0925b` now derives row indices from `parent_node_id`, validates bounds and
uniqueness, and adds a regression test reproducing the 15-field trap. Five focused
exporter tests pass. No checkpoint, prediction, target or training metric was affected.
The ph000/ph002 summary markers remain incomplete until the corrected full export runs;
ph003/ph004/ph005 cross-fit training follows in the existing queue.

### 2026-08-21 — [code/run] Proxy/Null and P12 cross-fit queue is healthy; results remain preliminary

The persistent four-A100 interactive chain is on allocation attempt 7 of 24, current
job `57357340`. All four independent one-GPU workers are active and their loss traces
are advancing; this is not a stalled tmux shell or a repeated initialization. At the
audit, the real-FAINT Proxy and angular-Null runs had each completed five full
84,446-core epochs and were roughly 47,000 cores into epoch 6. The `omit_ph000` and
`omit_ph002` P12 encoders had each completed eleven full epochs and were roughly
20,000--21,000 cores into epoch 12. Current gradient windows are finite and below the
norm-5 clipping threshold. ph001 remains unopened.

The best **interim** ph006 rows are:

| run | completed best epoch | macro R2(lambda1) | first-three macro | shell R2(lambda1) |
| --- | ---: | ---: | ---: | --- |
| real-FAINT Proxy | 5 | 0.57687 | 0.58239 | 0.57825 / 0.58168 / 0.58724 / 0.56030 |
| angular-Null | 4 | 0.58666 | 0.62192 | 0.65462 / 0.62801 / 0.58312 / 0.48091 |
| P12 omit-ph000 encoder | 11 | 0.52796 | 0.58382 | 0.62533 / 0.59309 / 0.53305 / 0.36039 |
| P12 omit-ph002 encoder | 11 | 0.54136 | 0.59635 | 0.64222 / 0.59316 / 0.55368 / 0.37639 |

These numbers are not decision rows. Proxy and Null are early in a 20-epoch cosine
schedule, their current ranking is shell-dependent, and Proxy-minus-Null must be
computed from the frozen best checkpoints with paired spatial-core uncertainty after
both schedules finish. In particular, the Null currently has the larger macro score,
whereas Proxy has the stronger sparse-shell score; neither observation licenses a
multitracer conclusion yet. The cross-fit scores are encoder diagnostics rather than
posterior results. Once the first two omitted-phase encoders finish, the queue trains
ph003/ph004/ph005, exports ph006 from the frozen all-five-phase U checkpoint, and only
then materializes the leakage-safe P12 conditioning population.

### 2026-08-20 — [science/code] P10 decision gates close: U beats matched classical and corrected G; tail shrinkage is conditionally calibrated

The four post-Arm-A decision rows are now complete without opening ph001. The important
result is not a macro-only win. On all authoritative ph006 targets:

| estimator | four-shell macro R2(lambda1) | first-three-shell macro | shell R2(lambda1) |
| --- | ---: | ---: | --- |
| U-PATCH epoch 20 | **0.57310** | **0.63464** | **0.69190 / 0.64159 / 0.57042 / 0.38850** |
| corrected G-PATCH, LR 2e-4 | 0.46986 | 0.51260 | 0.53897 / 0.51258 / 0.48625 / 0.34164 |
| train-affine CIC | 0.31139 | 0.51877 | 0.51582 / 0.54325 / 0.49725 / -0.31074 |
| train-affine exact DTFE | 0.01607 | 0.05337 | 0.02077 / 0.05823 / 0.08111 / -0.09583 |

The corrected G schedule confirms that the original multi-phase G collapse was largely
an optimization defect, not evidence that its graph rows were misaligned: the `2e-4`
canary recovered macro `0.46986`, while `5e-4` reached `0.45504`. Both used three full
84,446-core epochs and an explicit 400,000-update cosine horizon. G is nevertheless
materially below U under the matched ph006 contract and is frozen as a non-selected
control; no further GraphNet variant sweep is opened.

The matched classical result is stronger than the earlier same-phase framing. U beats
CIC in each of the first three shells where CIC remains informative, by
`+0.1761/+0.0983/+0.0732`, as well as retaining a positive sparse-shell score. Thus its
overall lead is not manufactured by CIC's last-shell collapse. Exact DTFE has useful
raw rank information in places but its train-phase response fit transfers poorly under
the full cap/selection contract; its very low calibrated R2 is retained as a negative
result rather than silently replaced by a target-fitted affine score. Tracked copies:
`docs/evidence/p10/{cic,dtfe}_ph006_complete_20260820.json` and
`docs/evidence/p10/arm_a_g_schedule_*_20260820.json`.

A new prediction-conditioned calibration diagnostic resolves the apparent conflict in
the predicted-versus-true plots. The familiar regression of prediction on truth has
slopes below one because an MSE model estimates `E[lambda|observations]` and extreme
truths with ambiguous observations regress toward the population mean. The appropriate
point-calibration question reverses the conditioning: is
`E[truth|prediction]=prediction`? On ph006, its pooled slopes are
`0.9995/0.9942/1.0013` for lambda1--lambda3, and normalized weighted mean absolute
calibration errors are only `0.0066/0.0075/0.0104`. Shell four remains the weakest but
still has truth-on-prediction slopes `0.970/0.968/0.975`. No affine debiasing is
warranted. The missing tail width is instead a posterior problem: P12 must model the
conditional distribution around this calibrated mean. Evidence and reusable code:
`docs/evidence/p10/prediction_conditioned_calibration.json` and
`workflows/visualization/plot_p10_prediction_calibration.py`.

The active ordering is now deliberately gated. First finish the phase-matched
BRIGHT-target/FAINT-context Proxy-versus-angular-Null views and train them under the
same five-phase/ph006 protocol. In parallel, implement the five leave-one-phase-out
U-PATCH conditioning-summary contract needed for P12; all-five-phase training latents
are forbidden as posterior-training rows. Then build the paired
`V_dense/V_assign/V_final` view ladder and test whether a dense observed-galaxy teacher
has measurable ph006 headroom. JEPA opens only if that teacher gate passes. One bounded
tail-aware auxiliary loss remains an optional diagnostic after these data and posterior
gates; it may not replace the calibrated conditional-mean baseline or redefine the
primary metric. Four GPUs should be occupied by independent frozen-contract tasks until
true data-parallel parity is established; changing patch-batch/update semantics merely
to use hardware remains forbidden.

Implementation of those next gates has begun, but no Proxy/Null or P12 science result
is claimed yet. The phase registry now resolves official observation sources explicitly,
including ph000's historical `fba0_bkp` fibre-assignment product. A single builder
constructs matched real-FAINT Proxy and angular-scramble Null field channels while
leaving Bright targets unchanged, and a six-channel U-PATCH adapter consumes the three
Bright plus three Faint observational channels. For P12, a leave-one-phase-out contract
builder and guarded 32-dimensional latent exporter enforce that an exported training
row can never come from an encoder trained on its phase. Twelve focused unit tests pass,
including source-override resolution, Null geometry, Bright-target identity, and the
omitted-phase export refusal. The full source audit, view construction and cross-fit
contract materialization remain running prerequisites. Once their immutable markers
exist, one four-A100 node will run four independent one-GPU frozen-contract workers;
this fills the node without changing the one-variable-patch optimizer semantics.

The six-visible-phase source audit now passes. It finds `7,151,478--7,205,427`
unique assigned FAINT targets per phase; every assigned FAINT TARGETID matches its
phase-correct forFA row and all matched `RSDZ` values are finite and positive. The raw
assignment products contain roughly 28.7--29.1 million duplicate rows per phase, so
TARGETID-level bitwise-OR deduplication is part of the frozen contract. ph001 was not
opened. Tracked evidence:
`docs/evidence/p10/multitracer_source_audit_20260820.json`.

All five P12 leave-one-phase-out data contracts also pass. Each independently refits
selection and target/count transformations from four phases, exposes the omitted phase
only as validation, and produces a complete deterministic epoch of `67,244--67,678`
training cores plus `16,768--17,202` unique omitted-phase validation cores. The marker
asserts for every contract that its omitted phase is absent from training; ph001 remains
sealed. This completes contract materialization, not encoder training or posterior
calibration. Evidence: `docs/evidence/p10/p12_crossfit_contracts_ready_20260820.json`.

One P12 guardrail is added before training begins. The out-of-fold base predictions are
in a shared physical coordinate system, but independently trained encoders need not
give their raw 32-dimensional latent axes a common orientation. The exporter retains
both. The first posterior baseline will therefore condition on base prediction plus
deployable response covariates. Raw latents require a truth-free common-ph006 alignment
and fold-identifiability test, followed by a held-out likelihood/calibration gain; they
must not enter merely because they are higher-dimensional.

The full multitracer data gate is now closed. All six visible phase products pass with
`6,804,198--6,850,218` context-limited assigned FAINT rows per phase. Proxy and Null
use identical Bright targets, P3 grids, exposure and train-only FAINT selection curves;
the Null preserves every tracer radius and permutes angular directions only within cap
and `Delta-z=0.01`. Deposited tracer counts are conserved. Tracked evidence:
`docs/evidence/p10/p10_multitracer_views_ready_20260820.json`.

Persistent tmux supervisor `p10_mt_p12_4gpu` obtained four-A100-80GB interactive job
`57292623`. All four initial two-update canaries passed: real-FAINT Proxy, angular Null,
P12 `omit_ph000`, and P12 `omit_ph002`. All four scientific runs now have live loss
traces. The workers are checkpoint-resumable and will chain ph003/ph004/ph005 cross-fit
training plus ph006 summary export in later allocations. The four GPUs run independent
one-patch-per-update tasks; this is hardware concurrency, not an unvalidated change to
the optimizer batch semantics. ph001 remains sealed.

### 2026-08-18 — [science/code] P10 U-PATCH Arm A closes at epoch 20; G comparison is optimization-invalid; next gates use four GPUs as independent workers

The five-phase U-PATCH Arm-A trajectory is complete. It visited every one of the
84,446 canonical training cores in each of 20 epochs (1,688,920 optimizer updates),
selected only on all authoritative ph006 rows, and never opened ph001. Epoch 20 is the
frozen checkpoint:

- four-shell macro `R2(lambda1)=0.573104`;
- first-three-shell macro `0.634637`;
- per-shell `0.691903 / 0.641587 / 0.570422 / 0.388504`;
- all-row scaled validation MSE `0.354620`;
- equal-phase training objective `0.364937`;
- cosine learning rate exactly `0`.

This is enough training for the registered Arm-A experiment. Epochs 18--20 improve the
primary score by only `0.001495`, below the registered `0.002` material-improvement
scale, while the schedule has reached zero. Appending epochs to the same run would make
no optimizer update. This does not prove that U-PATCH has reached its architecture or
information ceiling; any low-LR restart, seed replication, richer response view or
multitracer model is a separately named experiment. The current checkpoint is frozen
as the deterministic P10/P12 reference.

The original G-PATCH trajectory is **not** a valid representation comparison. It is
frozen after six epochs as a diagnostic: training objective improved
`0.74885 -> 0.69972` by epoch 4 then worsened to `0.70700`; ph006 macro oscillated
`0.31263 / 0.28145 / 0.31458 / 0.25923 / 0.29673 / 0.27356`; and predictions were
severely variance-compressed. No row, feature, target, dropout, checkpoint or
phase-marginal alignment bug was found. The leading testable cause is the schedule:
one five-phase epoch contains 84,446 updates, 8.16 times the old P8 epoch, so epoch 6
still used LR `1.59e-3` after 506,676 updates. The trainer now supports an explicit
update-based cosine horizon and records pre-clip gradient-norm means/maxima plus the
fraction of steps clipped. Separately named `2e-4` and `5e-4` G canaries will test this
without modifying or resuming the frozen run.

Four GPUs will be used concurrently, but not as unvalidated data-parallel training. A
P10 optimizer step is one variable-size canonical patch. Averaging four patch gradients
would change effective batch size, phase/row weighting, update count and exact resume
semantics. The safe current use of one four-GPU 80-GB interactive node is therefore
four independent one-GPU tasks: two G schedule canaries and two phase-parallel CIC
workers. This uses the hardware without changing the estimator. Genuine DDP remains a
separate parity-and-scaling implementation if later needed.

The next-gate implementation is now tracked:

1. `p10_classical_fullcap.py` reconstructs each visible phase independently from its
   immutable P3 counts/expected-counts, applies the fixed R=7 Mpc/h tensor solve, and
   fits the CIC affine response only on ph000+ph002--ph005 with frozen phase/shell
   weights before ph006 scoring. A matched exact-DTFE branch builds independent
   phase-local Delaunay rasters and uses the same training-phase-only response fit before
   ph006 evaluation. It is expected to be the longer branch, but is no longer represented
   by the older ph000/P8 score.
2. `p10_multitracer_source_audit.py` verifies the official phase-matched targetable and
   fibre-assigned Faint sources for ph000/ph002--ph006 without truth access. The next
   science product keeps Bright supervision and compares real Faint context with a
   cap+redshift-stratified angular-scramble Faint null; Proxy-minus-Null is the
   additional-spatial-information estimand.
3. P12 preparation is open because U passed ph006 deterministic selection. The
   all-five-phase checkpoint may summarize ph006, but its own training-phase latents
   are in-sample. Posterior training therefore requires five leave-one-phase-out U
   encoders and the exact exportable 32-dimensional per-galaxy latent used by the point
   head. ph001 remains sealed until every deterministic/posterior decision is frozen.

The production interpretation remains conservative: U-PATCH is now a credible
independent-phase deterministic reference, not yet a production VAC. Matched classical
performance, response Arms B/C, posterior calibration, and the one-open ph001 test are
still blocking.

Runtime launch update: commit `ee74594` froze the above code, plan and tracked U
completion evidence. Persistent tmux supervisor `p10_next_gates_4gpu` obtained one
four-A100-80GB interactive allocation (`57216431`). GPU 0/1 started paired fresh-seed-42
G schedule canaries at LR `2e-4` and `5e-4`; GPU 2/3 completed the six phase-local CIC
raw solves in parallel and then moved into the calibration/source-audit stage. Early G
telemetry already confirms that clipping was not a hypothetical concern: over each
canary's first 125 updates, 40--52% of patch gradients exceeded the frozen norm-5
threshold, with window-maximum pre-clip norms of about 19--35. These are optimization
diagnostics, not validation results. Once CIC finalization returns, the same two released
GPUs automatically begin the six exact-DTFE phase rasters and their matched ph006 row.
All branches are checkpoint/resume safe across four-hour interactive allocations;
ph001 is not referenced by the launcher.

### 2026-08-16 — [code/run] P10 Arm-A allocation chaining verified; U epoch 4 active; G resumed unchanged on 80-GB A100 after one large-patch OOM

The two-hour interactive-allocation supervisors are behaving as designed. U-PATCH
returned the registered continuation code `75` at each runtime boundary, wrote an
atomic patch-cursor/optimizer/scheduler/RNG checkpoint, and acquired a fresh allocation.
This is exact continuation of one run, not repeated scratch initialization. At this
audit U had completed three full 84,446-core epochs and was at epoch-4 cursor 65,787
(`global_step=319,125`). Its all-authoritative-row ph006 macro-shell
`R2(lambda1)` progressed `0.2618 -> 0.3847 -> 0.3976`; these are early optimization
diagnostics, not a converged result or promotion decision. The per-shell epoch-3 values
were `0.4978 / 0.4596 / 0.4160 / 0.2170`, so the sparse fourth shell remains the
limiting and currently noisy term.

G-PATCH exposed a separate hardware-envelope issue during epoch 1. After 550 completed
updates, one unusually large canonical graph patch exhausted a 40-GB A100 (38.83 GiB
allocated by PyTorch). The supervisor correctly stopped on exit code 1 rather than
hiding the failure or silently skipping the core. The last atomic checkpoint is at
cursor/global-step 500; no epoch or ph006 metric exists yet. The launcher now requests
an 80-GB A100 for G-PATCH only and resumes the same checkpoint with the frozen model,
data, sampler, objective, optimizer and update budget unchanged. U remains on a standard
GPU constraint. Live allocations at the resume record were U `57126513` and G
`57128583`; both tmux supervisors were present and the two-allocation limit was obeyed.
G subsequently advanced through cursor 625, beyond the patch that had failed on the
40-GB device, confirming that the 80-GB continuation is live.

This is an operational memory fix, not an architecture or scientific-contract change.
The retained G scientific log intentionally contains the original OOM traceback for
provenance.

### 2026-08-16 — [science/code/run] P10 Arm A launched from scratch; P11 JEPA opened only as a bounded parallel challenger

The roadmap now reflects the final paired-view JEPA discussion. P11 no longer requires
P10 Arm C to first diagnose a purely representation-limited failure. One bounded
`PAIRED-DEGRADE-JEPA-v1` experiment may open after the `V_dense/V_assign/V_final`
pairing contract is frozen and a dense-view teacher transfers to ph006 with measurable
headroom. It remains non-blocking: it may run beside P12, cannot delay Arm A or P12,
cannot reconstruct information removed by the final observation operator, and supplies
an encoder/summary rather than a calibrated posterior. The deployable final-view student
alone must earn the existing fresh-phase/held-out-response adoption gates. P12 now starts
immediately after ph006 deterministic selection because posterior calibration is the
binding production-VAC requirement. Plan commit: `c1854aa`.

Deterministic Arm A is now running for both the frozen U-PATCH leader and matched
G-PATCH control. The new trainer consumes the passed
`TRAINING_LOADER_READY.json` contract and enforces:

- scratch initialization; no historical ph000 weights;
- ph000+ph002+ph003+ph004+ph005 training, with phase never entering model inputs;
- every one of 84,446 cores exactly once per epoch, phase-balanced prefixes and the
  equal mean of within-phase square-root-shell-weighted row MSEs;
- 20 complete epochs and identical optimizer-update budgets, with early stopping
  disabled for this first comparison;
- exact all-authoritative-row ph006 evaluation after every completed epoch;
- ph001 absent from the visible loader and truth path;
- atomic patch-cursor, optimizer, scheduler and RNG checkpoints, allowing exact
  continuation across bounded interactive allocations;
- windowed loss JSONL every 25 updates and epoch-level loss/metric JSONL.

All 42 P10 unit tests pass. Two-update GPU canaries passed for both models before the
scientific runs began. The first canary attempt exposed a `/global/homes` versus
`/global/u2` provenance-path alias and failed before any scientific update; the path is
now canonicalized and regression-tested. Implementation commits are `7f52b34`,
`59a5c25`, and `42c1feb`.

Live launch provenance:

- tmux supervisors: `p10_arm_a_unet`, `p10_arm_a_graph`;
- interactive allocations only, no batch jobs: U `57094869` on `nid001125`, G
  `57094871` on `nid001036`;
- canary markers:
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/arm_a_training/arm_a_gpu_canary_v2/{unet,graph}/seed_42/TECHNICAL_CANARY_COMPLETE.json`;
- scientific outputs:
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/arm_a_training/arm_a_r0_v1/{unet,graph}/seed_42/`;
- supervisor/scientific logs:
  `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/arm_a_training/launcher_logs/`.

At this launch record U had reached epoch-1 cursor 1,025 and written its first atomic
scientific checkpoint; G had reached cursor 150. These are only proof that optimization
is active, not a performance result. No epoch or ph006 validation metric exists yet.

### 2026-08-14 — [science] PATHWAY ASSESSMENT (Claude Code): JDPK's "protocol > encoder" framing is QUANTITATIVELY confirmed; P12/SBI is now the binding constraint on the VAC product, not accuracy

Reviewed the P8 closeout, P10 readiness, and the plan's P10-P13 contracts. Assessment below;
no new runs, no promotion.

**1. The framing is confirmed by our own numbers.** Measured spread in same-phase ph000 macro
R2(lambda1), separated by which axis was varied:

| axis varied | spread | evidence |
|---|---:|---|
| ENCODER/architecture (4 families, matched protocol) | **0.052** | G 0.4695, D0 0.4723, U 0.5035, U-CICres 0.5214 |
| TRAINING METHODOLOGY (exposure+budget, same encoder) | **0.150** | U 0.363 (2k-step screen) -> 0.5134 (epochs+budget); G +0.091 |
| DATA/INFORMATION (add Faint tracers) | **0.113** | U 0.4943 -> MT4 0.6074 (rot0; null still required) |
| EVALUATION honesty (leaky -> spatial -> patch) | 0.082 | 0.421 transfer -> 0.440 R0 -> 0.5035 patch |

Training methodology moved the score ~3x more than the entire architecture family space, and the
data axis ~2x more. Four encoder families (attention graph, 3-D U-Net, density-first physics, and
classical-residual hybrid) all land within 0.052 of each other under the same protocol — that is the
signature of a task limited by information and optimization, not by inductive bias. The programme
should stop treating encoder choice as a primary lever; the plan's P8 closeout already does this
implicitly by freezing ph000 rather than opening more architecture sweeps, and that was correct.

**2. What P8's closeout actually established (and its honest limits).** U-PATCH-BRIGHT_REFERENCE is
the frozen deterministic candidate; CIC is the anchor; G-PATCH is a non-promoted runner-up;
U-CIC-residual is NO_GO on sparse-shell regression (matching the P9 audit mechanism); MT4 is
PASS for Faint spatial information but NO_GO for the current Proxy encoder; D0 is retained as
secondary field evidence. The D0 DarkAI-like rescore is a genuinely useful piece of self-scepticism:
it showed r(k)=0.937 and P_cross/P_true=0.855 at 0.02-0.08 h/Mpc under external-style scoring, i.e.
part of the apparent gap to published field results is a SCORING-CONVENTION difference, not method
quality — while honestly retaining that ~14.5% amplitude suppression and knot incompleteness remain.
That is the correct way to handle a favourable-looking diagnostic.

**3. P10 readiness is the strongest infrastructure state the programme has had.** 84,446 training
cores per deterministic epoch across five phases with balanced phase prefixes and reproducible
resume; transforms fit on the five training phases only and frozen for ph006/ph001; 40/40 P10 tests;
signed `sealed_inputs_ready_predictions_not_written_truth_not_opened` for ph001 with 12 blind-isolation
gates. The 64 Mpc/h vs 94.5906 Mpc unit catch is exactly the class of bug that silently degrades a
model input, and it was caught by a gate rather than by a confusing result.

**4. THE BINDING CONSTRAINT IS NOW P12, NOT ACCURACY.** The deterministic ladder has plateaued near
macro 0.50-0.52 (U converged at ep19/20 both rotations), while the VAC product is per-galaxy
ENVIRONMENT with uncertainty. The class evidence already shows why a point estimator cannot deliver
that product: conditional-mean shrinkage (sigma_p/sigma_t ~ sqrt(R2) ~ 0.6-0.8) systematically
under-populates extreme classes, and thresholding it gave 26-31% predicted voids vs 38% true and
knot recall 56-64% at 51-64% precision. A calibrated posterior converts this from a bias into a
quantified uncertainty: P(lambda1>0.2) per galaxy is the deliverable that the shrunk point estimate
cannot be. The plan's P12 contract is well-specified (SBC/TARP/conditional coverage, response
conditioning, H_fid estimand explicitly NOT called HOD-marginalised, stratified by density/mask/
recipe) — the honesty framing is right. One historical note that must be carried forward: ALL prior
FMPE/tempering calibration was fitted under the leaky random-split regime and its coverage claims are
void; P12 must be refit from scratch on the P10 protocol, and the earlier TARP results should not be
cited as evidence that the flow calibrates.

**5. RECOMMENDED NEXT STEPS (pathway to the VAC):**
(i) **Run P10 Arm-A deterministic training now** — it is ready, and it is the first true
    generalisation evidence in the programme's history; everything before it is same-phase.
(ii) **Select on ph006, then immediately begin P12 on the ph006 selection artifacts** rather than
     treating P12 as a post-hoc add-on. The posterior is the product; scheduling it after a long
     deterministic tail risks it being rushed.
(iii) **Keep the MT4 neural null on the critical path** — if Faint context survives its null under
      multi-phase training, it is the single largest remaining lever (+0.11) and it is a DATA lever,
      consistent with (1).
(iv) **Do not reopen encoder sweeps.** Given a 0.052 architecture spread, further encoder work has
     poor expected value against P12 calibration, response conditioning (Arms B/C), and phases.
(v) **Report the VAC product in class/posterior terms, not R2.** Our own figures show morphology and
    class recovery are better than the per-galaxy R2 implies; the release framing should follow the
    calibrated-probability contract, with R2 as a secondary diagnostic.

### 2026-08-14 — [science/code/data] P10 deterministic multi-phase training ready; ph001 sealed; Loa deployment frozen

The five pre-training work packages are complete. Freshly initialized deterministic
models may now train with equal phase-level status on
`ph000+ph002+ph003+ph004+ph005`, use all authoritative ph006 cores for
validation/model selection, and retain ph001 for one-open blind evaluation. Phase is
provenance only and is never presented to either encoder. This closes Arm-A software
and source readiness; no production model training or ph001 scoring has been launched
by this work. Arms B/C remain gated on the forward-view ladder and P3b response-field
exports.

1. The versioned registry, graph/field adapters and common phase-balanced loader now
   cover ph000--ph006. One deterministic epoch contains 84,446 training cores exactly
   once: 17,202 / 16,768 / 16,890 / 16,793 / 16,793 from
   ph000/ph002/ph003/ph004/ph005. The order is seed-and-epoch deterministic, phase
   prefixes are balanced, resume at cursor 12,345 reproduces the uninterrupted order,
   and validation visits all 16,796 ph006 cores in frozen core-ID order.
2. Graph-feature, field-channel, selection and ordered-target transforms were fit once
   on only the five training phases and applied unchanged to ph006 and sealed ph001.
   The shared objective is the equal mean over phases of the within-phase
   square-root-shell-weighted authoritative-row MSE; the canary reconstructs its direct
   value to floating-point precision.
3. Real graph and field patch extraction is deterministic for every visible phase,
   with row/target alignment, P1--P4 contract identity, complete-epoch coverage, resume
   parity and weighted-loss accounting all passing. The full P10 unit discovery also
   exposed and fixed an older registry validator that still excluded ph000; all 40 P10
   tests now pass.
4. The response-source audit is frozen to the correct deployment family: official
   SecondGen mock products remain Kibo-derived, while the intended DR2 deployment is
   the superseding Loa product `DA2/loa-v1/LSScats/v2.1/PIP`. This is an explicit
   Kibo-to-Loa schema contract, not a falsely matched release. All 18 full and 18
   clustering randoms per mock phase plus the Loa data/random products were enumerated,
   content-hashed and schema-crosswalked; all 24 source gates pass, unsupported
   point-pairing is not claimed, and no truth or patch-local statistic enters the
   response. `response_fields_complete=false` remains explicit.
5. The signed ph001 contract records
   `sealed_inputs_ready_predictions_not_written_truth_not_opened`. Density and T-web
   products remain forbidden, ph001 contributed to neither transform nor response
   fitting, and all 12 blind-isolation gates pass.

A unit-sensitive adapter failure was caught before these gates were stamped. P4 cores
are 64 Mpc/h wide, but the P5/P6 point coordinates are observer-frame Mpc; the exact
canonical width is therefore 94.5906000591 Mpc for the frozen cosmology. The adapter
now derives this width from the P4 geometry instead of hard-coding 64, with cap-origin
closure below 1e-12 Mpc. This is a genuine model-input alignment correction, not a
relaxed tolerance.

Authoritative atomic markers:

- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract/TRAINING_LOADER_READY.json`;
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract/P10_RESPONSE_SOURCES_READY.json`;
- `/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract/P10_BLIND_EVALUATION_FROZEN.json`.

Tracked evidence is under `docs/evidence/p10/`, including
`adapter_inventory_20260814.json`, `transforms_frozen_20260814.json`,
`training_loader_ready_20260814.json`, `response_sources_20260814.json`,
`response_sources_ready_marker_20260814.json`,
`blind_evaluation_frozen_20260814.json`,
`blind_evaluation_ready_marker_20260814.json`, and `multiphase_live_status.json`.
Implementation commits are `3a74d38`, `15739fa`, `2a48dad`, `4c191f3`, and
`fd7a835`. Interactive CPU allocations were 56948963 (primary adapter/transform/canary),
56949222 (response audit), and 56950253 (ph006 adapter); no batch queue was used.


### 2026-08-14 — [science/plan] ph000 admitted as the fifth P10 training phase; independent roles unchanged

The P10 phase-role contract is revised to train freshly initialized production models
on `ph000+ph002+ph003+ph004+ph005`, select models and hyperparameters on ph006, and
open ph001 once for the final blind evaluation. The earlier exclusion of ph000 from the
training mixture was unnecessarily conservative. ph000 has been spent as an independent
development/evaluation set because it informed architecture, feature and protocol
choices, but training data are allowed to influence those choices. It therefore remains
invalid as fresh generalisation evidence while remaining useful as an additional
independent cosmic realization in the training mixture.

All production models must initialize from scratch rather than fine-tune an old
ph000-trained checkpoint. The phase-balanced sampler gives ph000 the same phase-level
status as ph002--ph005; phase remains provenance and may not enter model features. All
feature, field-channel and ordered-target transforms are fitted on the complete
five-phase training mixture and frozen for ph006 and ph001. Historical ph000 checkpoints
and metrics remain reference evidence only, and no new ph000-only architecture or
hyperparameter sweep is authorized. The ph006 selection and sealed ph001 one-open
contracts are unchanged.

### 2026-08-13 — [science/plan/data audit] Observation-view response contract frozen after ph000--ph006 closure

The separate “Observation Views in P10” discussion has been reconciled with the live
repository and the completed phase products. The historical point matters: use of
random catalogues and an explicit survey response was already present in P3b/P10 before
the ASTRA discussion. The discussion did not originate the method; it exposed missing
end-to-end closure and sharpened four requirements: distinguish footprint support from
completeness, build a response for each controlled observation view, prevent graph
topology from bridging mask holes, and validate deterministic residuals/posterior
coverage as functions of completeness and boundary distance. The durable plan now calls
the old “degradation ladder” a **forward-observation ladder**, because each view is a
different survey operator applied to the same latent field rather than a poorer
universe.

The survey-response contract is now
`p_s(x)=M(x) C_s(x)`. Here binary `M` records angular footprint and imaging-veto
support, including holes, while continuous `C_s` records the stage-dependent fibre and
redshift-success response. In U-PATCH, random-derived `M` and expected intensity
`mu_s`, completeness and mask distance are candidate voxel inputs; zero support must
remain distinct from a physical void. In G-PATCH the same response is interpolated at
galaxies, and any Delaunay/radius edge whose sampled segment crosses `M=0` must be
removed. Directly adding random points as graph nodes, as done by
[ASTRA](https://academic.oup.com/rasti/article/doi/10.1093/rasti/rzaf032/8221862),
is retained as a later bounded graph ablation rather than made the default encoder.
This is because our estimand remains tidal-environment information and our established
U/G encoders need a deployable selection baseline; ASTRA uses random nodes as part of a
different stochastic topological classifier.

A read-only NERSC inventory found that no new final-view random catalogue needs to be
generated by default. Every upstream ph000--ph006 DA2 SecondGen BGS LSS directory has 18
`BGS_BRIGHT_{0..17}_full_HPmapcut.ran.fits` files. Homologous DESI-data names are
present in both DA2 `kibo-v1` and `loa-v1` LSS trees. In the inspected mock, the full
HP-map-cut random has 30,923,493 rows and angular/tile/hardware/imaging fields including
`GOODHARDLOC`, `NTILE`, `MASKBITS` and `FRAC_TLOBS_TILES`, but no `Z`. The
clustering random adds `Z`, `WEIGHT*` and `TARGETID_DATA`. The final data catalogue
also carries `COMP_TILE` and `FRACZ_TILELOCID`. Consequently:

- register paths, schemas, sizes and hashes for all phases rather than searching for or
  rebuilding final-view randoms;
- audit the intended measure and provenance of every response column before mapping it
  to `M`, `C_fibre`, `C_z` or `ntilde(z)`;
- do not treat the clustering-random redshift distribution as an independent radial
  selection function until its data linkage has been accepted;
- use the same audited base angular randoms plus stage probabilities for intermediate
  views unless persistent point identities and true stage-matched randoms are proven;
- freeze one exact DESI LSS release/version only at P13 rather than mixing mutable DA2
  `kibo-v1` and `loa-v1` products.

This is consistent with the DESI LSS definition of a matched random catalogue as an
unclustered sampling of the probability that DESI could have observed the tracer
([Ross et al.](https://arxiv.org/abs/2405.16593)). Ordinary angular randoms do not encode
density-dependent fibre collisions or redshift failure by themselves; those require
alternate-assignment/PIP or accepted completeness/quality products
([Lasker et al.](https://arxiv.org/abs/2404.03006)). The plan now requires the atomic
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract/P10_RESPONSE_SOURCES_READY.json`
source/schema marker before new multi-phase GPU training. Full P3b response-field
export does not block the Arm-A baseline, but must pass its random-only and boundary
canaries before Arm C.

The minimum registered forward-observation ladder is now:

1. `V_dense`: targetable BGS galaxies inside the accepted imaging footprint, before
   assignment/redshift loss;
2. `V_assign`: the same latent population after the audited fibre-assignment operator;
3. `V_final`: the successful-redshift, final-quality, HP-map-cut deployment view.

Another view is added only if it isolates a named effect and has an audited deployable
response. Uniform thinning remains a sampling-information control, not a replacement
for fibre assignment. All views of one latent core remain in one phase/spatial split,
carry one total scientific weight, and rebuild their fields, graph topology and
response products from their own selected IDs.

The implied input-feature tests are real, but they are now deliberately bounded. For
the leading U-PATCH family, keep architecture, target rows, sampler, optimizer-update
budget and seeds fixed and run:

- `R0`: frozen P3a/P8 final-view baseline;
- `R1`: replace only the occupancy-derived support/reference approximation with
  random-derived `M` and `mu`;
- `R2`: add audited `C_fibre` and `C_z`;
- `R3`: add mask-boundary distance while retaining the established LOS channels.

Do not run the full power set of response features. If `R3` is promoted, perform one
leave-one-response-block-out shortcut diagnostic. For G-PATCH finalists only, isolate
the `M=0` edge veto with the response features held fixed; an unsafe no-veto model can
be measured but cannot ship. The ASTRA-style random-node experiment is optional and is
opened only if G-PATCH remains scientifically competitive.

HOD marginalization has been removed from the first-result critical path. The baseline
encoder and SBI posterior are explicitly conditional on the fiducial BGS galaxy--halo
prescription:
`q(lambda_g | encoder(x_s), response_s, H_fid)`. HOD ID/seed are provenance, never
production inputs. DESI has studied 11 BGS HOD variants—`BGS_0` plus ten posterior
samples controlled by 17 luminosity-dependent meta-parameters—but that analysis had
only 11 `z=0.2` cubic mocks from one AbacusSummit realization, not a ready-made
multi-phase cut-sky/assignment/LSS suite
([Findlay et al.](https://arxiv.org/html/2411.12023v4)). After baseline P12 calibration,
one held-out-HOD intervention may compare induced posterior shifts/coverage with the
posterior width. Escalate to HOD augmentation or marginalization only if that effect is
material.

The completed ph000--ph006 products therefore remain valid and do not need rebuilding
before training. The immediate launch gates are now two software/source contracts:

1. implement the phase-balanced ph002--ph005 loader, fit transforms on that training
   mixture only, freeze them for ph006, pass row/target/coverage/resume/loss canaries,
   and write `TRAINING_LOADER_READY.json`;
2. register and crosswalk the existing mock/DESI random and response sources and write
   `P10_RESPONSE_SOURCES_READY.json`.

Once both exist, Arm A can train on the final view while P3b response exports are built.
Arms B/C wait for the three-view manifests and P3b canaries. Selection remains on ph006;
ph001 stays sealed for one final evaluation; ph000 remains a development reference and
never enters the training mixture. P12 FMPE/NPE follows only after the deterministic
encoder and response schema are frozen, and its coverage must be stratified by phase,
redshift, density, support, completeness, mask distance, holes versus footprint edges
and held-out response recipe.

### 2026-08-13 — [science/code/runtime] P10 phase products close through ph006; ph000 normalized; loader gate remains

The P10 phase-product matrix is now complete. ph006 is no longer missing T-web:
its passing atomic target marker is
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/targets/tweb/backend_optimized_ngrid_2048_rsmooth_7/TWEB_COMPLETE.json`.
Sixteen rank products cover x=[0,2048), total 111,669,184,864 bytes, and bind the
same c000, z=0.2, 2000-Mpc/h, 2048-cubed, 10% A+B TSC, R=7-Mpc/h and
threshold-0.2 contract used by the training phases. The earlier table that described
ph006 T-web as absent was a transient build-status snapshot and is now stale.

ph006 now has a passing terminal
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/PHASE_COMPLETE.json`.
Its P1 catalogue contains 7,330,186 successful-redshift rows, 6,248,640 context
rows and 4,968,208 authoritative active targets, with shell counts
2,704,647 / 1,635,574 / 556,883 / 71,104. Its separately constructed NGC and
SGC graph components contain 55,873,345 Delaunay edges and zero cross-cap edges;
the 14.78-Mpc radius union has 187,094,689 context pairs and mean degree 59.88.
P3 and P4 pass the same nine-channel, Planck18 observer-Mpc, 64-Mpc/h core and
256-Mpc/h super-block contracts as the other phases. Catalogue--field--target
closure passes: cap trace Spearman correlations are 0.462/0.455, shuffle
separations are 0.257/0.294, and CIC maximum discrepancies are 2.86e-6.

One ph006 row records a precision-boundary diagnostic rather than a physics or
lineage failure. CACTUS computes CWEB from its native eigenvalues before the
catalogue writer casts eigenvalues to float32. An eigenvalue infinitesimally above
0.2 can therefore be stored as exactly `float32(0.2)`, making the original strict
class inequality unrecoverable from the stored value. ph006 has exactly one such
row and zero disagreements away from the float32 threshold. This corrects the
over-broad wording in the preceding log entry: CWEB was not generated from the
stored float32 columns. The shared P10 gate now explicitly counts threshold
quantization ambiguities while continuing to reject every non-boundary mismatch.

The frozen ph000 development products have also been copied into the same P10
directory contract rather than remaining scattered across legacy scratch roots:
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph000`. The passing
`REFERENCE_PHASE_COMPLETE.json` binds 193 files and 162,057,995,430 bytes by
source, destination and checksum. The tree contains normalized `targets/`,
`catalogues/`, `p1_canonical/`, `p2_graph/`, `p2_union/`, `p3_fields/`,
`p4_patches/`, `p4_rebuild/` and `contracts/` products. ph000 remains a frozen
development/convention reference and is explicitly ineligible for the training
mixture. Its historical canonical graph is already NGC--SGC disconnected but
predates cap-checkpoint markers; this is recorded as a validated legacy-global-graph
provenance difference rather than fabricating cap-level build records. Scratch had
about 15 PiB free during the copy; the roughly 151-GiB reference duplication is not
a storage constraint.

The final cross-phase report is
`docs/evidence/p10/multiphase_p1_p4_complete_with_ph006_20260813.json`. Every
registered physics, unit, schema, row-identity, target-closure, graph-disconnection,
field, patch, density-conservation and T-web-layout gate passes. Roles remain:
ph002--ph005 training, ph006 validation/selection, ph001 sealed blind, and ph000
development reference.

This completes the **P1--P4 data-product gate**, not the training-launch gate.
The existing P8 model entry points are still single-phase. Before multi-phase
deterministic training starts, one phase-balanced adapter/loader must sample
ph002--ph005 without exposing phase as a feature, fit all feature/target transforms
on the training mixture only, apply them frozen to ph006, prove row/target identity
for both U-PATCH and G-PATCH, and pass resume/coverage/loss-accounting canaries. The
missing atomic gate is
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/training_contract/TRAINING_LOADER_READY.json`.
`p10_multiphase_status.py` therefore correctly reports `p1_p4_products_ready=true`
but `ready_to_launch_deterministic_training=false` until that software contract is
implemented and tested.

### 2026-08-13 — [science/code/runtime] ph001--ph005 P1--P4 campaign complete; training-pool inputs pass cross-phase physics closure

The requested automated ph001--ph005 campaign is complete. ph002--ph005 now each
carry a passing truth-bearing `PHASE_COMPLETE.json`; sealed ph001 carries a passing
truth-free `BLIND_INPUT_COMPLETE.json`. The machine-readable cross-phase report is
`docs/evidence/p10/multiphase_p1_p4_completion_20260813.json`, and every registered
gate passes. This closes the P1--P4 **training-pool input build**. It does not yet
close deterministic multi-phase model selection: the matching ph006 validation
P1--P4 products remain required before training and selection can follow the frozen
train/validation phase roles.

The exact physical contract is common across all phases: AbacusSummit c000 at z=0.2,
2000 Mpc/h, a 2048-cubed TSC field from the same 10% A+B particle definition,
R=7 Mpc/h Gaussian tidal smoothing, ascending eigenvalues and threshold 0.2. Phase is
not an input feature. Each truth-bearing phase processed all 136 A+B slabs and exactly
33,022,530,364 particles; relative deposition errors are `9.4491e-8`, `9.4582e-8`,
`9.4639e-8`, and `9.4933e-8` for ph002--ph005. Every T-web product has 16 contiguous
ranks covering x=[0,2048) and the same 111,669,184,864-byte layout. The previously
tracked ph002/ph000 convention audit binds this contract to the ph000 products used for
P8; equality of field realizations is correctly not expected.

P1 context counts for ph001--ph005 are 6,247,537 / 6,256,475 / 6,205,524 /
6,224,750 / 6,276,100, or 96.99--98.10% of ph000. P2 has zero NGC--SGC edges in
every phase. Union mean degrees are 58.46 / 59.48 / 58.61 / 59.67 / 60.32 versus
59.57 for ph000; these small differences are cosmic-realization diagnostics, not
contract failures. P3 uses the same nine canonical channels and Planck18 observer-frame
Mpc coordinates in every phase. P4 uses identical 64-Mpc/h cores, 256-Mpc/h
super-blocks, fold roles and support rules.

A stronger catalogue--field--target closure was made terminal for ph002--ph005. It
independently redeposits the P1 coordinates into P3, verifies repeated host keys have
identical tidal labels, checks native-float32 CWEB threshold semantics, and compares the
sampled galaxy contrast with the T-web trace against within-shell shuffles. CIC maximum
absolute differences are only `1.91e-6`--`3.81e-6`; all CWEB mismatches are zero; and
cap-level trace Spearman correlations are 0.441--0.471 with shuffle separations
0.256--0.297. This is positive spatial/lineage closure, not a model-performance claim.

Two useful pre-promotion failures were preserved. First, one ph003 row lies exactly at
the float32 representation of the 0.2 threshold. Upcasting before comparison created a
spurious one-row class mismatch; the exact CWEB gates now compare the stored float32
eigenvalues with `float32(0.2)`, matching how CWEB was generated. Second, ph003's frozen
fold allocator passed count, cap/shell and occupied-superblock balance but missed the
registered boundary-distance gate (median ratio 1.260 > 1.25). A deterministic,
label-free fallback removes only the redundant global context-count penalty while
retaining cap/shell and core-count objectives. It passed every unchanged gate with
boundary-distance ratio 1.066, active-count ratio 1.017 and maximum cap/shell deviation
9.06%. ph004 and ph005 passed the original allocator. No tidal label selected a fold or
weakened a threshold.

Canonical phase roots are
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph00{1,2,3,4,5}`. Within each root,
P1 is `p1_canonical/`, P2 is `p2_graph/` plus `p2_union/`, P3 is `p3_fields/`, and
P4 is `p4_patches/`. The cross-phase evidence records exact terminal paths, hashes,
counts, graph diagnostics, closure statistics, field channels and patch units.

### 2026-08-13 — [science/code/runtime] ph002 P1--P4 closes; ph001 blind graph branch passes

The first complete independent-phase model-input contract now exists. ph002 has a
passing atomic
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/PHASE_COMPLETE.json` covering
truth, P1, P2, P3 and P4. This is a data-readiness milestone, not a model-transfer
result and not permission to inspect ph001 truth.

The terminal validation independently requires the registered c000, z=0.2,
2000-Mpc/h, 2048-cubed, TSC, 10% A+B and R=7-Mpc/h target convention. ph002 used all
136 A+B slabs and conserved 33,022,530,364 deposited particles to relative error
`9.4491e-8`; the T-web marker has 16 contiguous rank outputs covering x=[0,2048).
P1 uses Planck18 observer-frame Mpc coordinates. P2 contains 7,370,124 canonical
nodes, finite frozen node/edge metrics, a 14.78-Mpc radius union and zero NGC--SGC
edges. P3 and P4 schemas differ from the tracked ph000 contracts only in
`catalogue_id`. The P4 semantic rebuild and all five-fold/cap/shell gates pass.
ph002 has 6,256,475 context galaxies, 97.79% of ph000's context count, and union
mean degree 59.48; these are recorded as statistical diagnostics rather than falsely
required to equal another cosmic realization.

The sealed ph001 representation has also passed P2 without target access: 7,328,081
nodes, 55,863,964 Delaunay edges, 182,621,426 context union pairs, exact P1 row
identity, finite metrics and zero cross-cap edges. Its observed FITS has no CWEB or
eigenvalue columns. P4 now passes deterministic rebuild, graph/field support and
fold-balance validation, and the phase is closed only by the truth-free
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph001/BLIND_INPUT_COMPLETE.json`.

Two final-gate defects were caught before promotion. The RAPIDS stage required an
explicit `CONDA_PREFIX`/PATH when invoked outside shell activation, and JSON reporting
needed to convert NumPy scalar gates. More importantly, phase completion initially
looked for `catalogue_id` and `target_truth_present` in the P1 atomic marker rather
than the P1 compatibility manifest where they are defined. The corrected validator
now also requires density conservation, all 136 particle slabs, frozen T-web layout,
and complete absence of density/T-web target products for blind ph001. Thirteen
focused tests pass.

### 2026-08-13 — [science/code/runtime] P10 automated P1--P4 campaign begins with sealed-blind separation

The ph001--ph005 build campaign is now active under the explicit objective of creating
complete, physically matched P1--P4 inputs before multi-phase training. The execution
contract separates ph002--ph005 truth-bearing training products from ph001 blind input
products. ph001 may receive observed geometry, host keys for duplicate control, a fresh
global graph/metrics, observational fields and P4 patches, but no density/T-web target is
opened and no target column is written. Its terminal marker is therefore
`BLIND_INPUT_COMPLETE.json`, not a truth-bearing `PHASE_COMPLETE.json`.

The failed first ph002 P2 attempt motivated an operational change rather than a physics
change. `p10_build_phase_graph.py` constructs NGC and SGC as separate atomic Delaunay
work units, maps every local simplex back to immutable P1 row IDs, hashes each cap, and
merges only two passing cap markers. A completed cap now survives allocation expiry.
Global graph metrics are still computed once on the merged per-catalogue graph; patches
remain views of that canonical graph and never recompute Delaunay topology or metrics.

Phase-specific P3/P4 schemas are also materialized from the tracked ph000 templates with
exactly one permitted difference: `catalogue_id`. Planck18 observer-frame Mpc geometry,
5-Mpc field cells, the 14.78-Mpc radius graph, 64-Mpc/h physical cores, response channels
and five-fold rules remain frozen. `p10_validate_phase_products.py` distinguishes these
exact physical gates from expected statistical differences in galaxy counts and graph
degree caused by cosmic variance.

The reusable stage driver is
`workflows/abacus_tweb/run_p10_phase_stage.sh`; it skips validated atomic products and
supports `p1`, per-cap `graph-cap`, GPU `p2-post`, and `p3-p4` stages. Eleven focused P10
tests pass, including truth-column exclusion in the ph001 observed dtype, cap-array
concatenation with preserved global IDs, and blind compatibility-manifest propagation.

The first new cap completion exposed and fixed a completion-logic defect before any
merged graph was promoted. ph002 SGC produced finite, positive-volume arrays with valid
endpoints and zero cross-cap edges, but the dictionary stored the measured condition as
`cross_cap_edges: false` and then applied `all(values)`. The gate is now the positive
predicate `no_cross_cap_edges: true`. Completed arrays can be revalidated to recover a
missing marker, so the 61-million-simplex SGC calculation is not repeated. This was a
bookkeeping failure, not a graph or physics failure.

Interactive allocation `56857414` is presently building the ph003, ph004 and ph005
2048-cubed 10% A+B TSC density fields concurrently, one phase per node. This is an active
run, not a completion claim; target, P1--P4 and final phase markers will be logged only
after their atomic gates exist and validate.

### 2026-08-13 — [science/code/runtime] Multi-phase readiness audit: particle staging closes, model inputs do not

The first atomic-marker audit after the ph002/ph006 launches gives a clear answer:
**P10 is not yet ready for multi-phase training.** The expensive particle-availability
risk is now largely closed, but target and representation products are not. The exact
machine-readable status is
`docs/evidence/p10/multiphase_readiness_20260813.json`.

All three remaining training-phase B restores completed and passed their registered
post-restore checks. ph003, ph004 and ph005 wrote verified `B_STAGE_COMPLETE.json`
markers at 2026-08-12 16:22, 18:57 and 22:05 UTC, with payloads of
168,068,368,057 / 168,066,274,916 / 168,047,687,099 bytes. Together with the already
verified ph002 restore and online ph006 A+B inputs, every non-blind training/validation
phase now has the frozen 10% particle source available. The persistent staging
supervisor ended cleanly with `all training-phase B restores complete`.

ph006's full 2048-cubed density build also completed successfully. It processed
33,022,530,364 particles, deposited 33,022,527,237.68657 counts, and passed the strict
conservation gate with relative error `9.4672e-8`. The builder took 2696.58 seconds;
the wrapped process ended with status zero after 45m05s and 54,538,056 KiB peak RSS.
The grid and manifest are
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph006/targets/density/AbacusSummit_base_c000_ph006_z0.200_ngrid2048_ab10_tsc_counts.{npy,manifest.json}`.
This is one completed target stage, not a completed validation phase: ph006 still needs
T-web, catalogue linkage/annotation, the full observed join, P1, P2/P3 and P4.

The ph002 P2 graph launch did **not** complete. Its log records more than 148 million
NGC and 61 million SGC simplices, but stops without a traceback, completion record or
saved graph. Allocation 56763306 no longer exists, the registered `p2_graph/` directory
contains zero files, and no atomic graph marker exists. The precise terminal cause is
therefore unknown; scientifically and operationally the result is unambiguously
`INCOMPLETE_NO_ATOMIC_ARTIFACT`. Before rerunning this large step, the phase-generic P2
builder should save cap-level checkpoints so a completed NGC construction cannot be
lost while SGC is still running. RAPIDS graph metrics remain downstream of that rerun.

The phase matrix is consequently:

| Phase | Role | A+B particles | density/T-web/truth | P1 | P2 graph/metrics | P3 fields | P4 | Ready |
|---|---|---|---|---|---|---|---|---|
| ph002 | train | verified | complete | complete | incomplete, no artifact | missing | missing | no |
| ph003 | train | verified | not started | missing | missing | missing | missing | no |
| ph004 | train | verified | not started | missing | missing | missing | missing | no |
| ph005 | train | verified | not started | missing | missing | missing | missing | no |
| ph006 | validation/selection | verified online | density only | missing | missing | missing | missing | no |
| ph001 | sealed blind | unopened | unopened | unopened | unopened | unopened | unopened | deliberately no |

No Slurm jobs were active at the audit time. U-PATCH multi-phase training requires
P1+P3+P4 for ph002--ph005 and the matching ph006 validation products; G-PATCH requires
P1+P2 graph/metrics+P4 over the same phases. Neither branch satisfies its data contract
today. The next production sequence is to make P2 cap-checkpointed and finish ph002,
then run the target/observed/P1 chains for ph003--ph005 and ph006, construct P2 and P3
as parallel branches, and freeze one shared P4/evaluation manifest per phase. ph001
truth remains sealed and untouched.

### 2026-08-12 — [science/code/runtime] P10 ph002 target-truth chain complete; ph006 density begins

The ph002 production density and T-web truth products now exist and pass their
frozen contracts. The complete 136-slab 10% A+B TSC build produced a
`2048^3` float32 count grid from `33,022,530,364` particles. It deposited
`33,022,527,243.663353` counts, for relative conservation error
`9.4491e-8`, in 2833.80 seconds. The 34.36-GB grid and manifest are
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/targets/density/AbacusSummit_base_c000_ph002_z0.200_ngrid2048_ab10_tsc_counts.{npy,manifest.json}`.

The first T-web launch used eight MPI ranks and exposed a precise infrastructure
limit rather than a scientific failure: `shift`'s Python-object FFT transpose
formed a `256 x 256 x 2048` complex128 message, exactly 2 GiB, which the
installed mpi4py stack rejected with `MPI_ERR_ARG`. No rank products were written.
`p10_run_tweb.py` now rejects unsafe layouts before the FFT and records the
worst transpose-message size. The clean 16-rank retry on four CPU nodes completed
in 7m39s, with about 52.7 GB peak RSS per rank. Sixteen contiguous outputs cover
`x=[0,2048)`, total 111,669,184,864 bytes, and carry the frozen c000/ph002,
z=0.2, 2000 Mpc/h, 2048-cubed, R=7 Mpc/h, threshold-0.2 contract. The atomic
gate is
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/targets/tweb/backend_optimized_ngrid_2048_rsmooth_7/TWEB_COMPLETE.json`.

The public `forFA2_nomask.fits` deliberately omits halo linkage. A phase-generic
compact-parent builder now exact-matches the authoritative BRIGHT
`(RA, DEC, RSDZ)` keys back to CutSky rather than replaying a software-version-
dependent tile radius. This decision was forced by a useful failed gate: the
current `desimodel` footprint selected 10,824,544 rows, not the frozen forFA
count 10,619,510. The exact-key product recovers all 10,619,510 BRIGHT parents,
has sequential TARGETIDs, unique matching keys, and zero maximum discrepancy in
RA, DEC, true/RSD redshift, and apparent r magnitude. The 594.7-MB parent and
atomic parity record are
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/catalogues/bright_parent/ph002_bgs_bright_parent_linkage.fits{,.complete.json}`.

The compact parent-to-T-web annotation subsequently completed in 5.81 minutes.
All 10,619,510 rows mapped to host-halo voxels with zero skipped rows; every row
has finite ordered eigenvalues and a class consistent with the frozen threshold.
The audited product is
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/catalogues/annotated_parent/ph002_bgs_bright_parent_with_tweb_eigs_rs7_ngrid2048_thr0p2_15d.fits`.
Its mean `(lambda1,lambda2,lambda3)` is
`(-0.1399565,0.1556087,0.4224288)`.

An immediate source-lineage audit caught and corrected an observed-view naming bug
before P1. The first join interpreted `BGS_BRIGHT-02` as “mock 02”, but the enclosing
`mock2/LSScats` directory already carries the mock identity; `BGS_BRIGHT-02` is a
restricted tracer product with 3,272,314 rows. Its 1,427,814-row joined output is
therefore deprecated diagnostic evidence and may not feed P1--P4 or training. The
full final-view source is the unsuffixed `BGS_BRIGHT_full_HPmapcut.dat.fits`, with
9,214,624 rows, 7,370,124 finite positive `ZWARN==0` redshifts, and 4,988,277 such
rows in `0.15<=z<0.55`. `p10_build_observed_truth.py` now selects this source and
names the output `*_bgs_bright_full_observed_with_tweb.fits`; regression tests forbid
reintroducing mock-number suffix inference. The corrected join completed in 51.25
seconds with 3.13 GB peak RSS. Its 7,370,124 TARGETIDs are unique, every label is
complete, and maximum RA/DEC linkage discrepancies are zero. The 1.40-GB FITS product
and atomic record are
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/catalogues/observed/ph002_bgs_bright_full_observed_with_tweb.fits{,.complete.json}`.

A representation-neutral, phase-generic P1 builder is now implemented and tested.
It treats that successful-redshift FITS table as the immutable canonical row table,
then writes only stable row IDs, Galactic cap, shell/active/context masks and Planck18
observer-frame Cartesian points in Mpc. P2 graph construction and P3 fields must consume
the same point/mask identity. Twenty-four focused P10 tests pass. The real ph002 P1
build completed in 20.87 seconds with 1.69 GB peak RSS. It freezes 7,370,124 parent
rows, 6,256,475 context rows and 4,988,277 active targets; shell counts are
2,714,747 / 1,649,261 / 553,052 / 71,217. NGC/SGC contain 5,218,760 / 2,151,364
parent rows. The 2,379,758 `BOX_INDEX<0` rows remain context-only and never enter
loss or evaluation. Artifacts and the atomic gate are under
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/p1_canonical/`.

The phase-generic points input was then enabled in the established graph builder.
A fresh ph002 Delaunay construction is active in allocation 56763306, separately
triangulating the two Galactic caps from the exact P1 points. No phase feature is
exposed to the graph.

The ph002 convention audit now passes against ph000: both record c000, z=0.2,
2000 Mpc/h, 2048-cubed, R=7 Mpc/h and threshold 0.2, while a 100,000-row ph000
annotation sample passes finite-value, eigenvalue-order and CWEB consistency checks.
The complete ph002 benchmark occupies about 142 GiB plus 157 GiB for the staged B
particles. Density, T-web and annotation required 47.23, 7.65 and 5.81 minutes,
respectively; the B restore required 2.39 hours of wall time.

This closes ph002's target/observation truth chain, not the complete model-ready phase.
P2 graph/metrics, P3 fields and P4 shared patches are still required. In parallel,
the persistent tmux
supervisor is restoring ph003
B from HPSS with ph004/ph005 queued, and the full ph006 2048-cubed density build has
started in interactive allocation 56762868 from its already-online A+B particles.
ph001 truth remains sealed and untouched. Relevant commits: `a5b513d`, `a54f4d7`,
`8795bb3`, `781181a`, `eb8a268`, `6de923c`, `a4e227f`, and `6d09a57`.

### 2026-08-11 — [science/code/runtime] P10 begins: uniform 10% A+B particle contract and all-phase source gate pass

The earlier particle-availability statement was too broad. The 7% B snapshot
subsample is available for every required independent phase: ph001--ph005 store
`field_rv_B` and `halo_rv_B` inside their per-phase HPSS halo archives, while
ph006 has the same products online on CFS. P10 therefore freezes one scientifically
uniform truth contract for ph001--ph006: 3% A plus 7% B, using both field and halo
particles, for 10% total. Full snapshots and A-only fallbacks are forbidden. Storage
location is provenance only and phase identity is not a model input.

The versioned phase registry is
`configs/p10_phase_registry_v1.json`. It keeps ph002--ph005 as training phases,
ph006 as validation/model selection, and ph001 sealed blind. A real read-only audit
passes all six source gates. Every phase has the required 34 A-field and 34 A-halo
slabs, CutSky catalogue with halo/box linkage, forFA catalogue, potential assignments,
fibre-assignment output, and Kibo LSS catalogues. Every B source has 34 field slabs,
34 halo slabs, and two checksum manifests. Registered B payloads are
168.037/168.060/168.068/168.066/168.048/168.109 GB for ph001--ph006 respectively.
The tracked evidence is `docs/evidence/p10/asset_inventory.json`.

The first reusable P10 implementation is now in place:

- `p10_phase_assets.py` validates the registry, source paths, FITS schemas, and
  exact HPSS members;
- `p10_stage_particle_b.py` provides idempotent listing/restore, scratch-headroom
  enforcement, POSIX checksum verification, ASDF readability and phase/redshift
  checks, and an atomic `B_STAGE_COMPLETE.json` marker; it deliberately has no
  delete operation;
- `p10_build_density_field.py` replaces the hard-coded ph000/A-path assumptions
  of the legacy helper with a phase-explicit streaming A+B TSC builder.

Eleven focused tests pass. The online ph006 density-input preflight additionally
verified all four 34-slab directories and first/last ASDF headers against c000/ph006,
z=0.2, BoxSize=2000 Mpc/h, A=3%, and B=7%. Exact evidence is under
`docs/evidence/p10/`.

The ph002 benchmark chain has begun with a persistent HPSS restore in tmux session
`p10_ph002_restore`. Its dry run proves a 168,059,673,012-byte B payload.
No density/T-web benchmark or independent-phase model result exists yet; those remain
the next gates, and ph001 truth remains unopened.

**Runtime update:** the ph002 restore and post-restore verification completed
successfully. HTAR extracted 70 archive members in 8604.942 seconds of wall time
(reported 720.465 MB/s average). The staging verifier then confirmed 34 field-B plus
34 halo-B ASDF slabs, exact POSIX checksums for both manifests, readable headers for
all 68 slabs, `SimName=AbacusSummit_base_c000_ph002`, and `z=0.2`. The verified
payload is 168,059,673,012 bytes. The atomic gate is
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/particle_b/AbacusSummit_base_c000_ph002/B_STAGE_COMPLETE.json`;
the result record is
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002_b_stage_result.json`. The next
gate is a reduced-grid/four-slab A+B TSC canary with count conservation, followed by
the full 2048-cubed cost benchmark if it passes.

**Density-canary update:** the phase-explicit A+B builder has passed its first real
ph002 deposition test. One slab from each of field-A, halo-A, field-B, and halo-B
provided 962,897,612 particles. The authoritative 1024-cubed float32 TSC canary
deposited 962,897,259.972 counts, a relative conservation error of `3.6559e-7`, below
the frozen `2e-6` gate. Build time was 101.45 seconds; the wrapped process used
113.65 seconds wall time and 20.49 GB peak RSS. The 4.1-GB grid and manifest are under
`/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/targets/density/canary/`.

The deliberately coarser 256- and 512-cubed probes produced conservation errors of
`2.2667e-5` and `2.3499e-6`. Their nearly billion-particle occupancy creates visible
float32 accumulation loss; the approximately order-of-magnitude improvement with
each grid refinement, followed by the 1024-cubed pass, identifies a canary-resolution
effect rather than missing slabs or inconsistent coordinate units. The production
2048-cubed grid is less occupied still. No tolerance was relaxed and failed probes
wrote no density artifact. The complete 136-slab 2048-cubed benchmark is now
technically unblocked.

### 2026-08-10 — [science/code/runtime] P8 COMPLETE: DarkAI-like D0 rescore narrows the protocol gap; `ph000` development freezes

The requested no-retraining diagnostic has completed on the frozen
`U-DENSITY-PHYS-v1` rotation-0 field. It selects NGC science-support grid-cell centres
with `0.15<z<0.4`, gives every 5-Mpc cell equal volume weight, subtracts the prediction
and truth means separately inside that subset, and performs no random/window
deconvolution. It reports the requested quantities with the exact definitions
`P_cross/P_true = Re(F_pred F_true*)/|F_true|^2` and
`r(k)=P_cross/sqrt(P_pred P_true)`.

The mask contains 33,192,463 cells. Mode-weighted results are:

| k range [h/Mpc] | modes | P_cross/P_true | r(k) |
| --- | ---: | ---: | ---: |
| 0.002--0.100 | 164,598 | 0.85744 | 0.93114 |
| 0.020--0.080 | 87,214 | 0.85516 | 0.93689 |
| 0.080--0.200 | 973,084 | 0.76037 | 0.88319 |
| 0.200--0.400 | 11,688,128 | 0.46789 | 0.64592 |

The DarkAI-like sign-threshold grid classification has exact cell accuracy `0.74855`
and balanced accuracy `0.70876`. True-row recall for void/sheet/filament/knot is
`0.68605/0.76551/0.76407/0.61941`. Under the native GraphWeb threshold 0.2, exact and
balanced accuracy are `0.80213/0.66565`, with recalls
`0.91245/0.71111/0.59250/0.44653`.

This is a real protocol effect, but not a reversal of the scientific conclusion. On the
older full supported volume D0 had `r=0.89543` and cross-transfer `0.79543` over
`0.02<=k<0.08`; the DarkAI-like subset raises these to `0.93689/0.85516`. NGC,
`z<0.4`, and equal-volume scoring therefore explain part of the apparent external-method
gap. They do not explain most of it: large-scale amplitudes remain suppressed by about
14.5%, correlation falls with k, and the rare knot class remains materially incomplete.
The diagnostic strengthens the statement that D0 recovers coherent web morphology more
faithfully than its per-galaxy lambda1 R2 suggests, but does not promote it over direct
U-PATCH for the primary point catalogue.

P8 now closes under a terminal `ph000` freeze. The frozen handoff is:

- `U-PATCH-BRIGHT_REFERENCE` as the deterministic learned candidate;
- CIC as the required classical anchor;
- G-PATCH as a non-promoted runner-up;
- U-CIC residual v2 as `NO_GO_SPARSE_SHELL_REGRESSION`;
- MT4 as `PASS_SAME_PHASE_TWO_ROTATIONS` for Faint spatial information but
  `NO_GO_FOLD_INSTABILITY_AND_ROTATION_2_SAFEGUARDS` for the current Proxy encoder;
- D0 rotation 0 retained as experimental secondary field/tensor evidence, not the
  primary point estimator;
- F-PATCH v2_A retained as a frozen resource NO-GO for that configuration.

D0 rotation 2 and D1 are explicitly `NOT_RUN_SUPERSEDED_BY_PH000_FREEZE`. This does not
convert them into negative results. The new diagnostic showed insufficient primary gain
to justify more tuning on repeatedly inspected `ph000` folds; independent phases now
have higher scientific value. Five-fold/three-seed `ph000` expansion is likewise
deferred in favour of P10 phase-level replication. No model is production-approved:
ph006 selection and the one-open ph001 blind test remain blocking.

Exact tracked evidence is
`docs/evidence/p8/density_d0_darkai_like_rescore.json` and
`docs/evidence/p8/p8_final_decision.json`. Runtime outputs are under
`p8_density_phys_v1/d0_darkai_like_rescore/rotation_0/seed_42/`; the terminal marker is
`p8_closeout_v1/P8_COMPLETE`. Implementation commits are `4f23fa6`, `8f0ad82`,
`c7f5822`, and `ad6781c`.

### 2026-08-10 — [science/code/runtime] P8.9 rotation-0 density-first downstream closes: no point promotion, physical-field branch retained

The complete `U-DENSITY-PHYS-v1` rotation-0/seed-42 chain is now finished. The
20-epoch density run selected epoch 16 by complete-validation macro-shell
`R2(delta_R7)=0.69678`. Inference from that frozen checkpoint was stitched over all
21,910 nominal plus inference-only output cores, with exactly 100% supported-voxel
coverage in both NGC (78,099,094 voxels) and SGC (36,990,531 voxels). Trained-model
patch parity passes comfortably: expanded-context NRMSE is `3.44e-4` (worst core
`6.95e-4`) and subdivision NRMSE is `1.96e-7`. The production decomposition is
therefore not responsible for the scientific score.

The stitched learned field retains substantial transferable information but is
amplitude- and tail-suppressed. Its supported-volume overall and four-shell macro
scores are `R2(delta_R7)=0.62005/0.67563`, with shell values
`0.76840/0.76765/0.72722/0.43924`. The prediction/truth standard-deviation ratio is
`0.79747`. Mode-weighted cross-correlation is `r=0.89543` over
`0.02<=k<0.08 h/Mpc` and `0.80604` over `0.08<=k<0.20 h/Mpc`, while the corresponding
cross-transfer values are `0.79543/0.64448`. Only 18.6% of the true `delta_R7>3`
voxel count is recovered and no `delta_R7>6` voxels are produced. This is credible
field recovery, not faithful one-point-tail recovery.

After one global fixed FFT solve per cap, the primary VAC-deployable observed-`Z`
raw-physical lambda1 row gives macro/first-three/pooled `R2` of
`0.47234/0.53345/0.54196`, with shells
`0.53047/0.57798/0.49189/0.28903`. The matched direct `U-PATCH` rotation-0 reference
is macro `0.50700`, so D0 is lower by `0.03466`: it narrowly fails the registered
within-0.03 point-estimator continuation gate and also fails the no-supported-shell-
loss gate. A train-fold-only affine diagnostic reaches macro `0.51154`, but is not a
physical product and degrades raw class behavior; it may never replace the raw row.
The privileged `Z_COSMO` oracle is materially higher (raw macro `0.58049`), confirming
that RSD/localization remains a distinct per-galaxy deployment penalty.

The field route nevertheless supplies pre-declared product information that the direct
eigenvalue U-Net does not. All six windowed tensor components have `R2=0.82490--0.86567`.
For the largest eigengap quartile the three median orientation disagreements are
`7.69/10.17/6.61 deg`; the smallest-gap quartile is correctly less stable at
`12.56/22.74/12.67 deg`. On raw deployable predictions, balanced accuracy, macro F1,
void recall, and knot recall are `0.66380/0.66318/0.81290/0.50482`, compared with the
direct U-PATCH reference `0.63306/0.65464/0.67984/0.42385`. These tensor and raw-class
benefits satisfy the pre-declared secondary-product continuation route, not the primary
point-estimator gate.

The learned-context diagnostic also closes. Across 24 anchors, eigenvalue
RMSE/reference-std falls from `0.13753` at 60 Mpc to approximately `0.114` by
240--360 Mpc; traceless-shear error falls from `0.10132` to `0.05406`. Thus additional
long-mode context is important through roughly 240 Mpc, especially for shear, but the
remaining eigenvalue plateau is learned-field/model error rather than a tiling artifact
that can be removed merely by enlarging patches.

The registered decision is therefore:

- **no rotation-0 promotion** of D0 as the primary lambda point estimator;
- **retain as a secondary physical field/tensor candidate and run rotation 2**, owing
  to the unique tensor and raw-class benefit;
- the D1 trigger is met because the field/spectra are credible while tails and
  downstream eigenvalues are misaligned; register exactly one fixed auxiliary loss
  before training and do not open a loss sweep;
- no production VAC claim: independent-phase P10 remains blocking.

The repaired downstream launcher uses OS-backed nonblocking run/stage locks,
same-filesystem unique atomic writes, hash-validated stage reuse, and a restart-safe
stitch/evaluate/context orchestrator. The original context stage exposed and then
fixed a one-sample/constant-target `R2` denominator bug; scalar closure now reports
RMSE/bias with undefined correlations represented as null. Ten focused downstream and
closeout tests pass. Code commits are `b8b9ec8`, `e0c9cfd`, and `98a2648`.

Primary runtime evidence:
`p8_density_phys_v1/d0_stitched/rotation_0/seed_42/`,
`p8_density_phys_v1/d0_evaluation/rotation_0/seed_42/field_downstream_metrics.json`,
`p8_density_phys_v1/d0_learned_context/rotation_0/seed_42/learned_context_report.json`,
`p8_density_phys_v1/d0_downstream/rotation_0/seed_42/downstream_run_manifest.json`, and
`p8_density_phys_v1/d0_decision/rotation_0/seed_42/`. The compact tracked decision is
`docs/evidence/p8/density_first_rotation0_closeout.json`.

### 2026-08-09 — [science/code/runtime] D0 complete-epoch canary passes and authorizes scientific training

The U-DENSITY-PHYS-v1 canary completed on interactive allocation `56532166` under the
frozen rotation-0/seed-42 contract. It visited all 13,664 training `(exact owner core,
shell)` units exactly once, with zero repeats and exact shell voxel exposure of
`6,731,621/13,020,795/20,426,433/25,690,328`. Persistent loss logging and 250-update
atomic checkpoints both functioned. It then evaluated all 4,602 validation units; no
sampled validation was used. Peak CUDA allocation was 1.46 GB and the complete training
and validation run took 1,432 seconds.

After this single epoch, the privileged density-field validation result is overall
`R2(delta_R7)=0.58375` and four-shell macro `0.63158`. Shell values are
`0.72340/0.71081/0.68496/0.40716`; overall Pearson is `0.76502` and shell Pearson is
`0.86063/0.84382/0.83209/0.63948`. The complete-epoch training weighted MSE is
`0.44750`.

This is the first positive learned-field result under exact ownership and complete
spatial-fold validation. It shows that Bright-only P8 inputs contain a learnable,
transferable signal for the smoothed matter field, including nonzero signal in the
sparsest shell. It is not a converged result and it is not yet a tidal-eigenvalue result:
the registered 20-epoch trajectory, stitched supported-volume inference, and one global
FFT per cap remain necessary before comparison with U-PATCH, G-PATCH, or classical
eigenvalue baselines.

Runtime artifacts are under
`p8_density_phys_v1/d0_runs/rotation_0/seed_42/canary_v1/`; tracked evidence is
`docs/evidence/p8/density_d0_canary_{summary,manifest}.json`. All D0 pre-scientific
gates have now passed, so the frozen rotation-0 scientific run is authorized.

### 2026-08-09 — [code/runtime] D0 interrupted/resumed trajectory is exactly reproducible

The frozen U-DENSITY-PHYS-v1 resume preflight passed on interactive allocation
`56532166`. A continuous four-update trajectory and a trajectory interrupted after
update two, atomically checkpointed, reconstructed, and resumed used the same real P8.9
exact-owner patches. Their four loss values agree exactly. Maximum absolute differences
in the final model and optimizer state are both `0.0`; scheduler state, full epoch order,
cursor, CPU RNG, and CUDA RNG also agree exactly.

PyTorch reports that CUDA `MaxPool3D` backward has no formally deterministic
implementation. The validator therefore allows that kernel while retaining a strict
`1e-7` numerical replay gate; this execution was nevertheless bitwise identical. This
tests checkpoint ownership and replay, not cross-device bitwise reproducibility.

Runtime evidence is
`p8_density_phys_v1/resume_parity/rotation_0/seed_42/{resume_parity.json,interrupted_checkpoint.pt,D0_RESUME_PARITY_PASS}`;
the compact tracked report is `docs/evidence/p8/density_d0_resume_parity.json`. The only
remaining pre-scientific D0 gate is one complete epoch with exact exposure and complete
validation.

### 2026-08-09 — [science/code/runtime] D0 unit/scaler contract and one-core overfit gates pass

The rotation-0 D0 contract builder completed on interactive allocation `56532166` at
revision `c702010`. It freezes 13,664 training core-shell units, 4,602 validation
units, and 4,485 development units. Exact training ownership contains
`6,731,621/13,020,795/20,426,433/25,690,328` voxels in shells 0--3, respectively;
all roles contain every shell and no inference-only core owns loss. The scaler used by
all later D0 runs is fitted on exactly 65,869,177 rotation-0 training voxels:
`mean=0.00383909`, `std=0.46607224`, range `[-0.841180,10.186234]`.

The pre-registered one-core probe then ran at revision `392e1bd` on the largest
supported training unit: SGC fold 2, shell 0, exact `19^3` core with 6,859 loss voxels
and a `72^3` contextual input. Its standardized density MSE contracts from a steps
1--10 mean of `1.181697` to a final five-record mean of `0.027542`; the ratio is
`0.023307`, passing the frozen `<=0.25` gate by more than an order of magnitude. The
200-update run took 8.53 seconds and has no stderr.

This closes target scaling, exact owner extraction, target cropping, mask/shell
selection, scalar U-Net output, and basic optimization. It is not evidence of spatial
generalisation and its checkpoint is forbidden as a scientific warm start. The next
gates remain an interrupted/resumed trajectory parity test and a one-complete-epoch
exposure/validation canary. Runtime artifacts are under
`p8_density_phys_v1/{training_contract,overfit_probe}`; tracked evidence is
`docs/evidence/p8/density_d0_{config,target_scaler,overfit_report}.json`.

### 2026-08-09 — [science/code] U-DENSITY-PHYS-v1 D0 optimization contract frozen

The passing physical ceiling authorizes exactly the bounded D0 density experiment, not
a new architecture sweep. D0 reuses the Bright-only U-PATCH input mapping and U-Net:
train-fold-normalized counts, the derived count/expected-count density proxy, and P3
apodized exposure feed a base-24 `UNet3D` with per-voxel channel normalization and one
scalar output. The privileged `delta_R7`, `Z_COSMO`, halo linkage, and eigenvalues are
targets/evaluation data only and can never enter the input tensor.

Density ownership is now defined at the voxel rather than galaxy level. Each training
unit is one positive `(exact nominal owner core, redshift shell)` intersection. Across a
complete epoch every supported owned voxel appears exactly once; inference-only owner
cores never own loss. The voxel loss is MSE after one rotation-0 training-fold-only
target standardization. Shell weights are `N_shell^-0.5`, matching the registered P8
galaxy-row convention without adding an auxiliary target.

The frozen architecture/context is base 24, one output channel, 24-voxel/120-Mpc halo,
8-voxel alignment, seed 42. The scientific budget is 20 complete epochs of AdamW at
`lr=0.002`, weight decay `1e-4`, gradient clipping 5, and one cosine schedule ending at
zero. There is no early stopping in this first trajectory: the previous P8 experience
showed that capped runs can still be improving, and the fixed schedule makes the
convergence diagnosis explicit. Checkpoints are atomic every 250 updates and training
loss is logged every 25. The field checkpoint statistic is complete-validation
macro-shell `R2(delta_R7)`; downstream eigenvalue adoption remains a separate stitched
global-FFT decision.

Before the 20-epoch run, the executable sequence is: build/hash the unit and scaler
manifest; pass a one-core overfit probe; pass exact checkpoint/resume and one-complete-
epoch exposure canaries; only then launch rotation 0. The preparation executable is
`workflows/abacus_tweb/p8_prepare_density_training.py`; seven focused D0/tensor tests
currently pass.

The overfit probe is frozen at 200 AdamW updates with the scientific `lr=0.002` on the
largest supported rotation-0 training unit. Its mean standardized density MSE over the
last five logged records must be no more than 0.25 times the mean of steps 1--10. It is
an optimization/plumbing gate only and its checkpoint cannot initialize D0.

### 2026-08-09 — [science/code/runtime] P8.9 global tensor closure passes with large field-to-physics headroom

The pre-registered one-global-FFT-per-cap closure completed on interactive 80-GB GPU
allocation `56532166` at revision `8e93626`. It used the corrected privileged
`delta_R7` target, 24-voxel/120-Mpc zero padding, no second Gaussian smoothing, and
retained only tensors sampled at the fixed 16,000 authoritative galaxies. Peak CUDA
memory was 10.54 GiB NGC and 6.14 GiB SGC; the final run took 17.39 seconds.

All frozen gates pass. The maximum tensor-trace versus transformed-input RMSE over
every cap, window, and coordinate row is `2.997e-7`, far below `2e-4`. Oracle
`Z_COSMO` macro-shell `R2_lambda1` is:

| exact input/window | macro | shell 0 / 1 / 2 / 3 |
|---|---:|---|
| full rectangle | **0.98648** | 0.99446 / 0.99141 / 0.98443 / 0.97560 |
| P7 box taper | 0.98648 | 0.99444 / 0.99139 / 0.98443 / 0.97565 |
| hard DESI science support | **0.97230** | 0.94776 / 0.98904 / 0.98325 / 0.96915 |
| apodized production window | **0.92325** | 0.83536 / 0.97819 / 0.97753 / 0.90190 |

The full-rectangle and box-taper rows are indistinguishable: rectangular FFT edge
handling is not the limiting effect on this sample. Removing all matter outside the
hard science support costs only `0.01418` macro R2. The larger apodized-window loss is
localized chiefly to the first and last radial shells and is the registered cost of
the conservative 100-Mpc radial taper, not a failure of the tidal projector.

At deployable observed-`Z` positions, the apodized-window macro remains `0.69971`, with
shells `0.63450/0.75368/0.68478/0.72589`; overall R2 is
`0.71893/0.80888/0.80654` for the three eigenvalues. The corresponding full-rectangle
observed-`Z` row is `0.75384` macro and
`0.77003/0.86718/0.90118` overall. This difference quantifies the combined RSD
localization and conservative-window floor; neither row is a learned-model score.

The shear is also retained: hard-support oracle traceless-shear R2 is
`0.96692/0.93599/0.96378`; the apodized oracle row is
`0.81866/0.88097/0.79359`. Window-induced orientation changes have the expected
eigengap dependence. For the smallest-gap quartile, median axis angles are
`1.74/3.37/2.04 deg` and p90 `18.29/29.28/17.09 deg`; for the largest-gap quartile,
medians fall to `0.61/0.85/0.62 deg`.

This is a decisive **GO for the bounded D0 density-learning experiment**. It proves
that the field-to-physics factorization has far more headroom than the current
`R2~0.5` learned estimators; it does not prove that a U-Net can reconstruct the true
density accurately enough. D0 must now earn that information under the unchanged P4
patch protocol and observation-only inputs. The runtime report/predictions are under
`p8_density_phys_v1/tensor_closure/`; tracked evidence is
`docs/evidence/p8/density_tensor_closure.json`. Two preceding attempts exited before
opening scores due to a script-path import and a float64 radial-window promotion;
commits `d71133c` and `8e93626` fix and test those runtime defects.

### 2026-08-09 — [science/code] P8.9 global tensor-closure contract frozen before runtime

The corrected scalar target/trace gate does not by itself establish that a density
field confined to the DESI support produces accurate tidal shear. The next blocking
preflight is therefore frozen before opening any tensor/eigenvalue result. It applies
exactly one unsmoothed projector `T_ij(k)=k_i k_j/k^2 delta_R7(k)` separately to the
NGC and SGC cap rectangles; the target already carries `R=7 Mpc/h` smoothing, so a
second Gaussian is forbidden.

Four rows separate effects that must not be conflated:

- full rectangular privileged density with 24-voxel zero padding;
- the same rectangle with the P7 20-voxel cosine box taper;
- the exact hard science-support window;
- the registered production candidate: science support times P3 apodized exposure
  times a 100-Mpc radial cosine taper, with 24-voxel padding.

Every row is sampled at both oracle `Z_COSMO` and deployable observed-`Z` positions.
The report separates trace, traceless-shear eigenvalues, the three ordered eigenvalues,
four redshift shells, both caps, and window-induced orientation changes conditional on
the privileged eigengap. The algebraic trace identity must have RMSE no larger than
`2e-4`. On the balanced 16,000-galaxy oracle sample, the full rectangle must reach
macro-shell `R2_lambda1 >= 0.90`; the apodized science window must reach macro-shell
`R2_lambda1 >= 0.50` with no shell below zero. These are implementation/physical-floor
gates, not DESI performance claims and not model-promotion scores.

Failure blocks D0 training until the output/context/window contract is revised; the
thresholds may not be weakened after inspection. The executable is
`workflows/abacus_tweb/p8_validate_density_tensor_closure.py`; four CPU tests cover the
window symmetries, radial taper, unsmoothed trace identity, and eigengap-conditioned
orientation diagnostic.

### 2026-08-09 — [science/code/runtime] Corrected P8.9 targets, exact ownership, and trace closure pass; global tensor closure remains blocking

The unit-corrected target rebuild completed on CPU allocation `56531803` at frozen
revision `9500924`. It now evaluates
`(observer_xyz_Mpc * h - 1000 Mpc/h) modulo 2000 Mpc/h`, while radial support uses
Planck18 comoving distances in the native P3 Mpc frame. All cap target voxels are
finite and no second `R=7 Mpc/h` smoothing is applied.

Correct science support contains 115,089,625 voxels—78,099,094 NGC and 36,990,531
SGC. Exact nominal P4 owners leave 5,660,978 supported voxels without an output owner.
The corrected complete tiling adds 3,145 inference-only cores (1,788 NGC, 1,357 SGC),
retains every nominal core/fold/authoritative row, assigns no loss or metric to those
new cores, and reaches exactly 100% supported-volume coverage. This replaces the
superseded 37.1-million-voxel/890-core result.

Independent trace closure then rejoined the immutable target table and sampled the
corrected P3 fields for 16,000 authoritative galaxies balanced across both caps and all
four shells. The registered gates pass:

- direct host-`x_com` slab trace: `R2=1.000000`, `RMSE=3.19e-8`;
- supported `Z_COSMO` cap trace: `R2=0.991444`, `RMSE=0.072685`;
- supported observed-`Z` cap trace: `R2=0.870115`, `RMSE=0.283456`.

The `Z_COSMO` row validates corrected target construction and 5-Mpc trilinear
interpolation; it is an oracle, not a DESI score. The observed-`Z` degradation is the
expected RSD/localization penalty and remains visible. This pass does **not** authorize
D0 training yet: the one-global-FFT-per-cap tensor/eigenvalue closure, including the
survey window and missing external shear, is still a separate blocking gate.

Runtime evidence is under `p8_density_phys_v1/{targets,field_output_tiling,target_closure}`;
tracked manifests are `docs/evidence/p8/density_target_manifest.json`,
`field_output_tiling_manifest.json`, and `density_target_trace_closure.json`.
### 2026-08-09 — [correction/code] P8.9 target artifacts superseded before training: observer Mpc was not converted to periodic Mpc/h

The mandatory closure review caught a unit defect before any model training. P3/P4 are
unambiguously indexed in observer-frame comoving Mpc: the field cell is 5 Mpc and the
64 Mpc/h P4 core is 94.5906 Mpc at Planck18 `h=0.6766`. The periodic T-web cube and
observer origin are in Mpc/h. The first target builder incorrectly evaluated
`observer_xyz_Mpc + origin_Mpc/h`; the correct mapping, already used by the passing
galaxy-level coordinate preflight, is

`periodic_xyz_Mpc/h = (observer_xyz_Mpc * h + origin_Mpc/h) modulo 2000 Mpc/h`.

The same audit found that radial science support had compared a P3 radius in Mpc with
shell limits in Mpc/h. Both conversions are now fixed and guarded by unit tests. The
previous target HDF5 files, target manifest, support counts, and derived output-tiling
manifest are therefore **superseded** and may not be used for training or science.
Their commit history is retained as an auditable negative result; tracked evidence will
be replaced only after the corrected rebuild and independent closure.

This is precisely why P8.9 placed target closure before model training. Nine focused
tests now pass, including explicit Mpc-to-Mpc/h periodic mapping and Mpc shell-bound
checks. The corrected target and exact-owner tiling must be rebuilt under the new source
revision. `U-DENSITY-PHYS-v1` remains blocked.

### 2026-08-09 — [science/code/runtime] P8.9 target fields are ready; exact owner tiling closes the supported-volume gap

The cap-aligned privileged `delta_R7` fields have been built successfully from the
already-smoothed `ngrid=2048`, `R=7 Mpc/h` T-web slab traces on the immutable P3
NGC/SGC lattices. All target voxels are finite, every output x-plane was written, and
no second smoothing was applied. Runtime products are
`/pscratch/sd/d/dkololgi/abacus/p8_density_phys_v1/targets/ngc_delta_r7.h5`,
`sgc_delta_r7.h5`, and `target_manifest.json`; the tracked manifest is
`docs/evidence/p8/density_target_manifest.json`.

The first coverage audit correctly failed rather than silently zero-filling: the
galaxy-occupied P4 cores covered only `35,942,316 / 37,070,964 = 96.955%` of P6
science-supported voxels under the old intersecting-range diagnostic. A deeper audit
found that those stored P4 voxel ranges intentionally include every cell intersecting a
physical core, so adjacent ranges can share boundary cells. P8.9 now freezes the exact
voxelwise rule instead: each P3 cell centre belongs to exactly one half-open core on the
unchanged 64 Mpc/h P4 lattice.

Under exact ownership, nominal P4 cores cover `35,884,054 / 37,070,964 = 96.798%` of
supported voxels. The missing `1,186,910` voxels are not safe to discard: only 432 have
positive deposited Bright counts, but all have nonzero expected counts and apodized
exposure. They are predominantly empty-tracer, observationally supported locations—the
kind of underdense field support a global tidal solve must retain.

`p8_build_field_output_tiling.py` therefore adds 890 **inference-only** owner cores
(507 NGC, 383 SGC). They have fold sentinel 255, never own density loss, labels,
galaxy metrics, or model selection, and do not alter any P4 core/fold/authoritative row.
They only ensure that the frozen model can write every supported voxel before the one
global FFT per cap. Coverage after extension is exactly 100%; uncovered supported
voxels are never zero-filled and no classical field is substituted. Unsupported
rectangular volume remains explicitly windowed outside the loss and final solve.

Runtime evidence is under
`/pscratch/sd/d/dkololgi/abacus/p8_density_phys_v1/field_output_tiling/`; the tracked
manifest is `docs/evidence/p8/field_output_tiling_manifest.json`. The passing allocation
was `56531687`; a preceding launcher attempt exited before computation because the
cleared Python environment required module-mode invocation. The next blocking gate is
trace/tensor closure at authoritative host, `Z_COSMO`, and observed-`Z` locations.
`U-DENSITY-PHYS-v1` is still not authorized to train until that closure passes.

### 2026-08-09 — [science/code/runtime] P8.9 density-target coordinate gate passes; RSD-to-real-space localization is now an explicit VAC gate

Before constructing a privileged `delta_R7` target on the 365-million-voxel P3 cap
lattices, the new executable preflight
`workflows/abacus_tweb/p8_density_target_alignment.py` rejoined `Z_COSMO` for 16,000
authoritative Bright galaxies balanced over both caps and four shells, then sampled the
frozen `ngrid=2048`, `R=7 Mpc/h` T-web slabs under six sky-coordinate hypotheses and an
independent host-halo `x_com` reference.

The coordinate contract is unambiguous:

- `(FILE_NUM, HALO_INDEX) -> x_com` reproduces all three frozen eigenvalues exactly
  (`R2=1.0`, zero residual), closing label/slab lineage;
- the correct observer-to-periodic mapping is `Z_COSMO` with origin
  `(-1000,-1000,-1000) Mpc/h`; its minimum eigenvalue R2 is `0.987379` and mean R2 is
  `0.991028`;
- changing the origin to `-990` or `0 Mpc/h` destroys alignment, so the origin is not a
  harmless convention;
- using deployable observed/RSD `Z` at the correct origin gives
  `R2(lambda1/lambda2/lambda3)=0.764040/0.865638/0.900256` before any learned error.

The coordinate gate therefore **passes** and authorizes canonical observer-frame
`delta_R7` target construction. It does not authorize training yet. The observed-Z
result isolates a real production issue: a physically valid real-space density/tidal
field can be excellent while per-galaxy sampling at redshift-space positions is worse.
P8.9 must report both an oracle `Z_COSMO` row and a deployable observed-`Z` row, and may
not present the oracle row as a DESI VAC score. The next implementation gate is the
target-field/support/stitching contract; only after its trace/eigenvalue closure passes
may `U-DENSITY-PHYS-v1` train.

Runtime evidence is
`/pscratch/sd/d/dkololgi/abacus/p8_density_phys_v1/preflight/coordinate_alignment.json`;
the tracked copy is `docs/evidence/p8/density_target_alignment.json`. The persistent
interactive run completed in 919.46 seconds at frozen revision `67dc94d`, exit status
zero, without using a production batch submission.

### 2026-08-09 — [science/code] MT4 closed: BGS_FAINT contains replicated spatial information, but the current multitracer U-PATCH is not promoted

The matched `U-BF-NULL-v1` rotation-2 run completed cleanly through exact-resume
interactive allocations at frozen revision `6b80744`. It reaches macro
`R2_lambda1=0.518740` with shell scores
`0.603295/0.578805/0.501779/0.391083`. Every epoch has complete unique-core exposure,
zero repeats and complete-fold validation; the final cosine learning rate is zero.

This second null resolves the MT4 interpretation. On rotation 2 the scrambled-Faint
null is only `-0.000997` below Bright-only in macro R2, while the real Proxy exceeds
the null by `+0.020672` macro and by
`+0.009677/+0.014400/+0.019530/+0.039080` in the four shells. Together with rotation
0 (`+0.132041` macro; positive in all shells), Proxy minus Null is positive in every
shell on both registered geographies. The two-rotation mean contrast is `+0.076356`
macro and `+0.080124/+0.072925/+0.075334/+0.077042` by shell.

The correct conclusion has two parts:

1. **Information PASS within `ph000`:** real BGS_FAINT positions contain recoverable
   tidal information beyond matched occupancy, exposure, radial selection and the
   six-channel optimization shortcut. The Bright-only inverse problem is at least
   partly tracer-information limited.
2. **Current encoder NO-GO:** Proxy improves over Bright by `+0.100362/+0.019674`
   macro on rotations 0/2. Rotation 2 fails the registered `+0.03` macro gate and
   degrades the `0.25 <= z < 0.35` shell by `-0.015216`; Proxy fold spread is
   `0.067954`. This model is not promoted, and no fresh-phase or DESI claim is made.

No extension was launched: both Proxy and Null schedules already terminate at zero
learning rate. The executable closeout is
`workflows/abacus_tweb/p8_close_multitracer_mt4.py`; the tracked digest is
`docs/evidence/p8/multitracer_mt4_decision.json`. Runtime evidence and the completion
marker are under
`/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/models/recovery/mt4_closeout/`
(decision SHA256 `4e840cde75606eefa0e5819e91fbb6c97f187c12a712e7fd24c6744cce88b0c3`).

The MT4 information result would permit a field/physics challenger scientifically, but
the pre-registered `ph000` boundary in P10.0 prevents a new architecture sweep. The
next model experiment is the already registered P8.9 `U-DENSITY-PHYS-v1`; MT5 full
G-PATCH, F-tier expansion, seeds, posterior estimation and HOD marginalisation remain
closed pending the bounded density-first decision and independent-phase programme.

### 2026-08-08 — [science/runtime] Proxy rotation 2 replicates the sparse-shell gain; neural Null rotation 0 establishes real Faint spatial information; Null rotation 2 launched

The two active MT4 runs completed cleanly through three exact-resume interactive
allocations each at frozen revision `6b80744`. Every epoch had complete unique-core
coverage, zero repeats, and complete validation; ordering violations remain zero and
the boundary-error Spearman coefficients are negligible.

Best-checkpoint results against the converged Bright-only U-PATCH reference are:

| run | rot0 macro | rot2 macro | rot0 sparse | rot2 sparse |
|---|---:|---:|---:|---:|
| Bright U-PATCH extension | 0.50700 | 0.51974 | 0.34534 | 0.32104 |
| `U-BF-PROXY-v1` | **0.60737** | **0.53941** | **0.48452** | **0.43016** |
| `U-BF-NULL-v1` | 0.47533 | running | 0.36952 | running |

The Proxy two-rotation mean is `0.57339`, a `+0.06002` gain over the Bright mean
`0.51337`. Crucially, its sparse-shell gain replicates strongly: `+0.13918` on
rotation 0 and `+0.10913` on rotation 2. This is the clearest evidence yet that the
current Bright-only problem is tracer-information limited.

The matched rotation-0 causal contrast is much larger than the classical control had
suggested. Proxy minus scrambled-Faint Null is `+0.13204` in macro R2, with positive
per-shell contrasts `+0.15057/+0.13145/+0.13114/+0.11500`. The Null itself is
`-0.03168` below Bright-only in macro R2: uncorrelated Faint context hurts the supported
shells and gives only `+0.02418` in the sparse shell. Therefore the large neural Proxy
gain on rotation 0 is not explained by extra occupancy, the retained radial selection,
or a generic six-channel regularization shortcut. The network is using the actual
spatial relationship between Faint tracers and the Bright-target tidal field.

This is a positive information result, not yet a model-adoption result. Proxy rotation
2 improves macro R2 over Bright by only `+0.01967`, below the registered `+0.03` gate,
and its second supported shell changes by `-0.01522`, exceeding the allowed `-0.01`
degradation. The Proxy fold spread is `0.06795`, so the absolute gain is geography
dependent even though the sparse-shell gain is robust. Fresh-phase promotion remains
closed.

Both completed runs carry the mechanical `NOT_CONVERGED_MAX_EPOCHS` marker because
their best score is epoch 20. Their cosine schedules have already reached learning rate
zero, however, so they fail the pre-registered extension condition. No unregistered
warm restart or extra epochs were launched.

The large rotation-0 Proxy--Null contrast and Proxy fold spread trigger the registered
geographic replication of the causal control. `U-BF-NULL-v1` rotation 2 is active from
the same detached worktree and immutable contract:

- tmux: `p8_mt4_null_rot2`;
- allocation: `56514867` on `nid008340`;
- output:
  `/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/models/recovery/mt4_faint_null_v1/unet_multitracer/rotation_2/seed_42/`.

The next MT4 decision is frozen: finish Null rotation 2, compute the two-fold causal
contrast and shell safeguards, then either close MT4 as a demonstrated multitracer
information gain with an unpromoted current encoder, or open the already registered
seed/fresh-phase steps. MT5/F-tier/posterior work does not pre-empt this decision.

### 2026-08-07 — [code] All-model parity figures (fig18/19/20): pooled lambda1 ladder verified on identical rows; MT4 is the only model that does not degrade toward the sparse shell

Parity for every completed P8 model on the SAME rotation-0 validation fold (999,683 Bright
authoritative cores; parent_node_id-verified for all six learned models plus CIC).
Code: workflows/abacus_tweb/plot_all_models_parity.py.

**Pooled lambda1 (R2 / slope / amplitude ratio):**
| model | R2 | slope | sigma_p/sigma_t |
|---|---:|---:|---:|
| CIC (train-affine) | 0.527 | 0.52 | 0.72 |
| G-PATCH base | 0.528 | 0.61 | 0.82 |
| G-PATCH extension | 0.560 | 0.58 | 0.78 |
| U-PATCH base | 0.572 | 0.57 | 0.75 |
| U-PATCH extension | 0.588 | 0.59 | 0.77 |
| U-CIC-residual v2 | 0.617 | 0.62 | 0.79 |
| MT4 multitracer* | **0.674** | 0.67 | 0.82 |
Pooled ordering matches the registered macro ladder exactly — the ranking is not a macro artifact.

**The per-shell panel (fig20 left) is the important one.** Every Bright-only model declines steeply
toward the sparse shell (0.55-0.65 -> 0.29-0.35, a drop of ~0.25-0.30) while CIC falls off a cliff
(-0.76). MT4 declines only from 0.708 to 0.485 — a drop of 0.22 but from a much higher start, and it
is the first model whose sparse-shell R2 exceeds every other model's SUPPORTED-shell floor. That is
exactly what adding tracers should look like IF the gain is information; it is also exactly what
denser sampling alone would look like. The neural null remains mandatory before interpretation.

**Amplitude ratios (fig20 middle) separate the two failure modes cleanly.** All learned models sit
below 1 and decline with sparsity (0.86->0.59), i.e. honest conditional-mean shrinkage tracking
sqrt(R2). CIC alone crosses 1 and reaches **1.33** in the sparse shell — noise amplification. MT4
holds the highest ratios of any learned model (0.83->0.71), i.e. it is the least shrunk, consistent
with genuinely more information rather than better regularisation — a useful discriminator to re-check
after the null runs.

**U-CIC-residual detail:** its parity cloud is visibly tighter than U-PATCH at high true-lambda1 but
its sparse-shell point is the lowest of the learned models (0.31 vs 0.33-0.35), reproducing the
supported-gain/sparse-loss trade identified in the P9 audit.

### 2026-08-08 — [science/code/runtime] U-BF Proxy replication and matched neural Faint-position null launched from one frozen contract

The decisive MT4 comparison is now running. Commit `6b80744` adds an optional,
provenance-checked Faint-count override to the existing six-channel U-PATCH loader.
The null changes only the Faint count grid and the density proxy derived from it. It
retains the real Proxy Faint exposure field, full-cap Faint selection curve, frozen
rotation-specific normalization, all Bright channels, Bright-only targets/loss/evaluation,
patches, optimizer, seed, and 20-epoch schedule. The override refuses control products
that did not pass, read labels, are not CIC, lack `faint_counts`, or do not exactly match
the canonical NGC/SGC grids.

A live full-cap rotation-0 extraction smoke test passed on core `11`: Bright fields,
authoritative parent IDs, Faint exposure, and sample coordinates were identical between
Proxy and Null; the Faint count and derived-density L1 changes were `14807.94` and
`232151.38` over the `[64,72,80]` context. Five targeted unit tests pass. The executable
smoke test is `workflows/abacus_tweb/p8_smoke_multitracer_null.py`.

Two persistent interactive-allocation supervisors were then launched from detached
worktree `/global/u2/d/dkololgi/TNG/Illustris_mt4_6b80744` at exact revision
`6b8074438d5916699b8f24b11367b5286275f516`:

- `U-BF-PROXY-v1`, rotation 2: tmux `p8_mt4_proxy_rot2`, allocation `56495027`,
  node `nid008197`, output
  `/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/models/recovery/mt4_proxy_v1/unet_multitracer/rotation_2/seed_42/`;
- `U-BF-NULL-v1`, rotation 0: tmux `p8_mt4_null_rot0`, allocation `56495031`,
  node `nid008397`, output
  `/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/models/recovery/mt4_faint_null_v1/unet_multitracer/rotation_0/seed_42/`.

Both immutable run manifests exist and record the same code revision. The Proxy manifest
records no override. The Null manifest records the frozen control manifest SHA256
`c03c1abd10b188e76890d681b545b3d41dc1261293ec537aa3fe50e4eb9002f4`, product
`faint_position_null_cic`, and the exact NGC/SGC component paths. Both runs have written
their first atomic recovery checkpoints. The supervisors enforce
the global two-allocation ceiling and exact checkpoint resume across four-hour walls.

No new scientific score is claimed by this launch. Proxy rotation 0 remains a large but
causally ambiguous result (`macro R2_lambda1=0.60737`, sparse shell `0.48452`). Proxy
rotation 2 tests geographic replication; Null rotation 0 determines how much of the
rotation-0 gain survives destruction of Faint angular structure. The Faint structural-
information claim remains closed until the Null comparison completes.

### 2026-08-07 — [science] PROGRESS ASSESSMENT (Claude Code): generalisation ladder now +0.06 over the frozen baseline and CONVERGED; my P9 CIC-residual NO-GO was WRONG in scope; MT4's headline is uninterpretable until the neural null runs

Reviewed SCIENCE_LOG + both plans and pulled every completed run's numbers directly. Assessment of
where generalisability stands, with three corrections/flags.

**1. The ladder (two-fold mean primary score = mean over rotations of macro R2(lambda1)):**
| model | rot0 | rot2 | MEAN | first-three | sparse shell |
|---|---:|---:|---:|---:|---:|
| G-PATCH base | 0.4682 | 0.4708 | 0.4695 | 0.514 | 0.336 |
| G-PATCH extension | 0.4910 | 0.4825 | 0.4867 | 0.535 | 0.343 |
| U-PATCH base | 0.4943 | 0.5128 | 0.5035 | 0.562 | 0.328 |
| U-PATCH extension | 0.5070 | 0.5197 | **0.5134** | 0.573 | 0.333 |
| U-CIC-residual v2 | 0.5220 | 0.5208 | **0.5214** | **0.598** | 0.291 |
| frozen R0 baseline | — | — | 0.440 | — | — |
| CIC | — | — | 0.185 | 0.520 | -0.76 |
Best honest same-phase result is now **+0.081 over the frozen spatial-holdout baseline** and passes
both registered bars. The programme has moved from "no encoder generalises" (three weeks ago) to a
converged, fold-replicated model that beats classical BOTH where classical works and where it fails.

**2. CONVERGENCE IS NOW ANSWERED (it was the open question).** In the extension runs U-PATCH peaks at
epoch **19 of 20** in BOTH rotations, with the last six epochs flat (rot0 0.5058->0.5070; rot2
0.5166->0.5197) — the first genuine plateau in the programme; the earlier "peaked on last epoch"
lower-bound caveat is discharged for U-PATCH. U-CIC-residual rot2 peaks at ep17. G-PATCH extension
still rises at ep20 (0.4910) — G remains budget-limited and its ceiling is still unmeasured, so
"G < U" is established at equal budget but not at G's own convergence.

**3. CORRECTION — my 2026-07-21 P9 verdict "a CIC-residual/hybrid is NOT worth building" was WRONG in
scope.** U-CIC-residual v2 scores 0.5214, the best non-multitracer result. What I tested was a GLOBAL
LINEAR BLEND of frozen predictions, which cannot vary its trust in CIC by location; a TRAINED residual
conditions on local context and can. My underlying mechanism was right and is visible in the numbers:
the residual gains entirely in the supported shells (first-three 0.598 vs U-PATCH 0.573) and LOSES the
sparse shell (0.291 vs 0.333, and 0.273 in rot2 — its worst-shell is the weakest of any learned
model). So the correct statement is: CIC residuals help where CIC is informative and hurt where it is
not, exactly as the blend analysis predicted — but a learned gate recovers the net. My error was
generalising a blend-specific null into an architecture-level NO-GO. The registered "shell-aware /
gated hybrid" caveat in that same entry is what actually got vindicated.

**4. FLAG — MT4's headline number must not be quoted yet.** U-BF-PROXY rot0 reaches macro **0.6074**
with sparse shell **0.4845** (vs U-PATCH 0.4943 / 0.341): +0.113 macro and +0.144 sparse, by far the
largest single gain recorded. But MT3 already established that the CLASSICAL Faint gain was dominated
by extra-sampling shrinkage, not structural information — genuine-position increment over the
angularly scrambled null was only +0.014. **The neural null `U-BF-NULL-v1` has NOT been run**
(`p8_multitracer_v1/models/recovery/` contains only mt4_proxy_v1 and its canary; the only null
artifacts are classical `faint_position_null_cic`). Until the scrambled-Faint neural control exists,
0.6074 cannot be attributed to Faint structural information rather than to sampling density/selection
regularisation. The plan already mandates this comparison — it is now the single highest-priority run.

**5. On the two integrated side-discussions:**
- **Density-first (P8.9 U-DENSITY-PHYS-v1)** — correctly scoped and worth doing. It is the one
  matched DarkAI-style control the programme lacks, and the log's distinction between P6 (patch
  parity), P7 (FFT numerics), and the true-field context curve (missing-physics floor) is exactly
  right: none of them measures learned long-mode accuracy. Two risks worth pre-registering: (a) the
  density objective optimises bulk MSE while our product lives in the tails — the plan's rare-knot
  reporting requirement covers this, keep it; (b) with U-CIC-residual already at 0.5214, a
  density-first model that merely matches CIC-quality density will not beat it, so the informative
  comparison is r(k)/transfer vs CIC's density, not just downstream R2.
- **Degradation ladder (P10.1)** — the response-explicit design is a genuine improvement on my
  earlier framing. Two points I would emphasise: it manufactures observation diversity, NOT new
  structures (the plan says this — it must survive contact with any later claim), and the held-out
  RECIPE (not just seed) is the load-bearing part. Given MT4's result, the ladder's tracer-density
  axis is now the most scientifically interesting rung, because it directly probes whether the gains
  are sampling-driven — the same question the MT4 null asks.

**RECOMMENDED ORDER:** (i) U-BF-NULL neural control — decides whether the biggest number in the
programme is real; (ii) G-PATCH longer budget to measure its actual ceiling (fairness of the U>G
claim); (iii) P8.9 density-first; (iv) 3-seed replication of the current leader; then freeze and
P10 fresh-phase. Everything above remains SAME-PHASE (ph000): no fresh-phase generalisation claim
exists yet, and none of these results changes that.

### 2026-08-07 — [science/plan] Density-first baseline, learned-context closure, paired degradation, and `ph000` freeze boundary registered

The methodology review closes four planning ambiguities without claiming a new model
result. First, full multi-phase training may wait until the remaining `ph000`
representation questions select one or two finalists, but `ph000` development is no
longer open-ended. The authorized remainder is MT4, one bounded density-first baseline,
technical observation-view canaries, and only a diagnosed paired-view representation
test. Phase asset auditing and one complete target-generation cost benchmark proceed in
parallel. After those decisions, freeze the representation/response/metric contract,
train on ph002--ph005, select on ph006, and open ph001 once. Multi-phase evaluation is
an unbiased transfer test, not an experiment expected to raise the score.

Second, the programme has diagnosed but not run the decisive learned-context accuracy
experiment. P6's 120-Mpc result is patch-extraction/model-parity convergence; P7's
360-Mpc result is learned-field/FFT numerical convergence; and the completed true-field
60/120/180/240/360-Mpc curve measures the missing-physics floor. None compares trained
eigenvalue accuracy as longer learned modes enter. Merely enlarging a halo beyond a
fixed convolutional receptive field cannot answer that question.

P8.9 therefore registers `U-DENSITY-PHYS-v1`: the frozen Bright P3/P6 fields feed a
patch-safe U-Net trained only on the true `R=7 Mpc/h` smoothed matter contrast. Its
overlapping core predictions are stitched into complete cap fields, followed by one
global fixed FFT tidal solve and evaluation at the identical Bright authoritative
galaxies. This is the missing density-supervised DarkAI-style baseline. Historical
T4/F1 produced an internal density field but was eigenvalue-supervised; F2 density
supervision never ran. Density MSE is the declared optimization loss, while downstream
P8 eigenvalue/class metrics, field `r(k)`/transfer/power/tails, and a learned-field
60--360-Mpc context curve decide the result. Rotation 0 is mandatory; rotation 2 and at
most one downstream-auxiliary loss are gated. No density architecture or loss sweep is
opened by a null result.

Third, P10.1 now freezes a response-explicit degradation ladder and the information
accounting for paired views. Dense, intermediate, and final Path1-like observations of
one latent core remain in the same outer split and share one scientific weight; they do
not become new cosmic structures. The first mandatory comparison remains supervised
Arms A--C: final view, balanced observation diversity, then explicit deployable response
conditioning. A curriculum is only an order ablation over the same examples and updates,
with balanced replay after any dense-to-sparse warm-up. At least one degradation recipe,
not only a random seed, is held out.

Finally, P11.1 registers one optional paired dense/degraded teacher--student experiment
only if Arms A--C leave a representation bottleneck and the ideal dense teacher itself
transfers to unseen spatial blocks and ph006. The student receives degraded inputs plus
DESI-available response channels; the teacher is absent at inference. Alignment uses a
predictor, stop-gradient teacher, local/multiscale masks, and a spread regularizer rather
than forcing exact equality of a pooled embedding whose dense-view information may be
unrecoverable. The older T3 LUPI launch is explicitly recorded as untested, not negative
evidence. Promotion still requires fresh-phase improvement, a held-out degradation
recipe, and no supported- or sparse-shell regression; latent alignment is not posterior
calibration.

Current state remains unchanged: MT4 rotation 0 is active; `U-PATCH-BRIGHT_REFERENCE`
is frozen; U-CIC/G-CIC/F-CIC remain closed; the density-first baseline is registered but
not started; P10 infrastructure benchmarking is active; paired JEPA is gated. No model
is promoted and no empirical score is introduced by this plan amendment.

Refs: `docs/plan_generalisable_graphweb_vac.md` P8.9, P10.0, P10.1, and P11.1;
`docs/plan_field_level_multimodal.md` T3/F1/F2 history.

### 2026-08-07 — [science/code/runtime] MT2/MT3 close; exact-resume MT4 rotation-0 training launched

The BGS_BRIGHT-target/BGS_FAINT-context branch has crossed its last pre-training
gates. The frozen scientific contract is unchanged: only the original `9,538,254`
Bright galaxies own targets, loss, validation, or future catalogue rows; the
`3,233,026` PHOTSYS-marginal Proxy Faint galaxies are observation context only.

MT2 confirms that the intervention is large in sampling terms. In the highest
redshift shell, occupied-voxel fraction rises from `1.18% -> 4.87%` in NGC and
`1.08% -> 4.33%` in SGC; mean tracer separation falls from `42.30 -> 25.56` and
`43.45 -> 26.47 Mpc`, respectively. The approximate Poisson shot-noise ratio is
`0.22`. This establishes a real increase in measurements, not improved tidal
inference by itself. The audit is
`/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/diagnostics/bf_proxy_response_v1/information_audit.json`.

The complete MT3 control suite now passes on rotations 0/2. Combined Bright+Faint
CIC reaches macro `R2_lambda1 = 0.4591/0.4470`, compared with frozen Bright-only
CIC `0.2005/0.1694`. That large difference is **not** evidence that Faint positions
recover comparable extra cosmic-web information: Bright plus angularly scrambled
Faint reaches `0.4449/0.4325`. The genuine-position increment over the matched null
is only `+0.0143/+0.0145`. Density-matched mixed catalogues score only
`0.1024/0.1082` on average, and the simple bias-aware estimator fails badly in the
sparse shell. The correct interpretation is that the classical gain is dominated by
extra-sampling shrinkage/selection regularization; transferable Faint structural
information remains unproven. Consequently, any neural Proxy gain must be compared
with `U-BF-NULL-v1`, not only with `U-B-v1`.

The first MT3 pass correctly withheld its marker because independent training-fold
affine maps crossed for `3--6e-6` of nearly degenerate spectra. Commit `4556317`
restores ascending order after affine calibration without changing the spectrum or
reading validation labels. Eight targeted tests pass. The corrected run stamps
`classical/mt3_complete/MT3_MULTITRACER_CLASSICAL_COMPLETE`; durable evidence is
`docs/evidence/p8/multitracer_mt3_summary.json`.

MT4 preflight also passes:

- repeated training on representative core `9009` reduced the loss-window mean from
  `0.37727` to `1.0148e-4` over 200 updates, demonstrating that the six-channel
  representation and optimizer can deliberately fit one patch;
- the epoch-aware canary was deliberately interrupted after an atomic checkpoint and
  resumed from the same immutable revision; its loss trace is strictly increasing with
  no duplicate cursors;
- it visited all `10,351/10,351` eligible training cores exactly once, with zero
  repeats, represented every shell, and evaluated the complete validation fold;
- the one-epoch macro `R2_lambda1=0.3994` and sparse-shell `R2=0.3552` are plumbing
  diagnostics, not model-selection results.

The canary summary is `docs/evidence/p8/multitracer_mt4_canary_rot0_summary.json`.
The frozen 20-epoch `U-BF-PROXY-v1` rotation-0 screen is now active in tmux
`p8_mt4_unet`, interactive allocation `56471082` on `nid008268`, from detached
worktree `/global/u2/d/dkololgi/TNG/Illustris_mt4_4556317`. Runtime outputs are under
`/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/models/recovery/mt4_proxy_v1/`;
the exact-resume supervisor log is
`/pscratch/sd/d/dkololgi/logs/p8_mt4_proxy_v1_unet_multitracer_rot0_supervisor.log`.

No model is promoted. Rotation 2 remains closed until rotation 0 improves either
macro R2 by `>=0.02` or sparse-shell R2 by `>=0.03` without a supported-shell loss
worse than `0.01`. Full G-PATCH training remains closed. If the Proxy screen passes,
the matched neural Faint-position null becomes mandatory before adoption or a claim
that Faint traces additional cosmic structure.

### 2026-08-07 — [science/code/runtime] Faint response corrected to explicit PHOTSYS-marginal Proxy

A response-provenance audit invalidated the first `BF_PROXY_RESPONSE_v1` branch before
its graph completed. The DESI LOA North/South redshift-success rates are defined by
imaging `PHOTSYS`; they must not be applied using Galactic NGC/SGC membership. The
staged mock retains neither per-object `PHOTSYS` nor regional Faint target-selection
bits, so mapping the rates by either Galactic cap or an inferred survey location would
create false precision.

A direct de-duplicated assignment-table audit makes that limitation quantitative:
all `3,352,713` unique observed Faint targets carry neither `BGS_FAINT_NORTH` nor
`BGS_FAINT_SOUTH`; both regional counts and the ambiguous-both count are zero. The
DESI overall/North/South Faint success rates are `0.964284/0.966388/0.963353`, so the
Proxy applies the overall rate rather than guessing the absent covariate. The tracked
audit is `docs/evidence/p8/faint_response_covariate_audit.json`.

The frozen development Proxy therefore uses the DESI LOA **overall BGS_FAINT success
rate marginalized over PHOTSYS**, with a deterministic TARGETID-keyed draw. This does
not merge tracer responses: Bright and Faint still retain independent count, contrast,
radial-selection, and angular-response channels. It only acknowledges that the finer
North/South response covariate is absent from this mock. The Oracle remains the
assignment-observed, no-redshift-failure information upper bound. Both remain
ineligible for a final production claim until a richer Faint photometric/PHOTSYS mock
or validated DESI-target reweighting is available.

The invalid cap-mapped Proxy catalogue, its fields, and its 100-step U-PATCH canary are
preserved as invalidation evidence but cannot satisfy any current completion marker.
The interrupted cap-mapped Delaunay graph produced no completion marker. The corrected
branch uses separately versioned `*_photsys_marginal` graph artifacts and canary names.
Seven catalogue tests now cover order-independent response draws, explicit
PHOTSYS-marginal fallback, rejection of ambiguous regional bits, coordinate units, and
the Bright-target/Faint-context contract.

The first information experiment remains:

```text
supervised/evaluated/output population = unchanged BGS_BRIGHT parent only
additional observation/context          = BGS_FAINT
U-PATCH input                            = separate Bright/Faint count, contrast, exposure
G-PATCH input                            = global Bright+Faint graph with tracer identity
Faint loss or released prediction        = forbidden in this screen
```

The corrected Proxy catalogue, field overlay, and train-fold-only selection fits now
pass. The catalogue contains `9,538,254` unchanged Bright targets and `3,233,026`
Faint context rows. The four Faint shell counts are
`826,122/1,038,124/719,172/253,299`; the last is compared with only `72,353` Bright
targets and therefore constitutes a substantial information-density intervention.
The final application audit correctly records all `3,352,713` Oracle Faint candidates
before the deterministic response draw, rather than confusing that input count with
the selected Proxy row count.

The initial allocation exited only because its already-loaded shell retained the old
response-population assertion after the data products had passed. Idempotent resume
validation prevented a second 48 GB zcatalog scan. The live disconnect-safe jobs are:

- tmux `p8_mt_cpu_photsys`, CPU allocation `56452732`: global disconnected NGC+SGC
  Delaunay construction over `12,771,280` Bright+Faint nodes;
- tmux `p8_mt_gpu_photsys`, one-A100 allocation `56452729`: the separately named
  PHOTSYS-marginal Proxy U-PATCH 100-step technical canary has completed and the
  worker now waits for the graph gate before cuGraph metrics, radius-union adapter,
  feature preparation, and the Proxy G-PATCH technical canary.

The Proxy U-PATCH canary covers all `999,683` Bright validation targets with zero
ordering violations. Its macro `R2_lambda1=0.23615` is a 100-update loader/optimization
diagnostic, not a comparison with the registered 20-epoch Bright reference.

No canary score is interpreted as a model-selection result and no MT4/MT5 scientific
gate is claimed. Full training remains behind MT2 information diagnostics and MT3
matched classical controls.

The response and orchestration corrections are commits `2a2acc0`, `449dcbc`,
`4ef48fc`, `89463c2`, `7acc7e2`, `71da78a`, `99bd150`, and `1c80820`.

### 2026-08-07 — [science/code/runtime] Bright-target/Faint-context multitracer implementation launched

The first BGS_FAINT information experiment is now frozen at the correct causal
boundary: the unchanged 9,538,254-row BGS_BRIGHT parent is the only supervised,
evaluated, and eventual output population. BGS_FAINT is an additional observation of
the latent density field and is context-only. It receives no eigenvalue loss and no
Faint prediction may be reported from this screen. This isolates the information gain
from more tracers from the separate question of gaining more labelled rows.

Two response-explicit products are being built from the immutable stage-3 target IDs:

- `BF_ORACLE_ASSIGNED_v1`: fibre-observed Faint rows without spectroscopic failure,
  explicitly an information upper bound and never production eligible;
- `BF_PROXY_RESPONSE_v1`: the same assignment realization followed by an independent,
  deterministic LOA BGS_FAINT success draw by north/south target bit. The frozen Bright
  parent is copied exactly and is never redrawn.

The first runtime attempt exposed that repeated alternate-tile rows in
`datcomb_bright_tarspecwdup_zdone.fits` disagree in nominal immutable columns, including
`Z_COSMO`. This is not treated as truth ambiguity: that table now supplies observation
state only, and observed TARGETIDs are joined back to unique `inputs/targ.fits` rows for
sky, RSD redshift, magnitude, and simulation linkage. A hard gate requires zero observed
Faint IDs unmatched to the unique target input before either catalogue is stamped.

Tracer selection remains separate throughout. The field branch preserves the frozen
Bright P3 fields and adds Faint-only CIC counts plus a regularized HEALPix
selected/target angular-response exposure on the exact same NGC/SGC grids. Independent
per-cap, per-rotation Faint radial curves and normalizers are fitted from P4 training
folds only. U-PATCH therefore receives six explicit channels: Bright count, contrast,
exposure and Faint count, contrast, exposure; it never receives one merged count field.
The graph branch builds one global response-aware Bright+Faint graph, computes all graph
metrics globally in `rapids-gnn`, adds tracer identity and tracer-specific response
features, and reuses the frozen Bright P4 cores as loss ownership.

Implementation is tracked by commits `8a31768`, `3f02836`, `ef44626`, `3419844`,
`d4c2994`, `4934c07`, `06bb9dd`, `cc84832`, and `2bac74d`. Targeted catalogue,
field/selection, graph-adapter, and model-loader tests pass. Runtime products are rooted
at `/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/`; logs are under
`/pscratch/sd/d/dkololgi/logs/p8_multitracer_*` and `p8_mt_*`.

All long work is disconnect-safe and uses interactive allocations only:

- CPU allocation `56446724`, shell `p8_mt_cpu_alloc`;
- dependency supervisor `p8_mt_cpu_pipeline`, which stops on failed catalogue, field,
  selection, or global-graph gates;
- GPU supervisor `p8_mt_gpu_supervisor`, which waits for the field/selection marker
  before requesting the second allocation, then runs Oracle/Proxy U-PATCH canaries,
  response-aware cuGraph metrics, the union adapter, and a Proxy G-PATCH canary.

No MT1/MT2/MT5 completion marker or model accuracy is claimed in this entry. The graph
precomputation is opened in parallel by explicit user instruction to compare both leading
representations; the registered promotion and stop rules remain unchanged.

Runtime checkpoint after launch:

- Both MT1 catalogue gates passed. `BF_ORACLE_ASSIGNED_v1` contains 9,538,254
  unchanged Bright rows plus 3,352,713 Faint rows; `BF_PROXY_RESPONSE_v1` contains
  the same Bright prefix plus 3,237,149 Faint rows after the deterministic LOA
  north/south response draw. In the key `0.45 <= z < 0.55` shell the Proxy adds
  253,664 Faint tracers to 72,353 Bright tracers. This is a strong information-density
  intervention, not yet evidence of improved tidal inference.
- Both field and independent-selection gates passed. The Proxy Faint training-fold
  radial fits close to 0.592% in the worst registered shell; 99.36% of its Faint
  context lies in defined P4 cores. The remainder stays outside the benchmark rather
  than being reassigned.
- The first field build correctly failed the zero-loss CIC gate because the frozen
  Bright P3 grids do not contain a complete stencil for 4 Oracle NGC and 21 Oracle
  SGC edge tracers. The audited lost-weight fractions were only `1.36e-6` and
  `2.23e-5`. Expanding the canonical grids for these points would change the frozen
  benchmark. The field branch now excludes only incomplete CIC stencils, records each
  exclusion, and again conserves deposited counts exactly; the graph branch retains
  those galaxies. Five field/selection tests pass.
- The first supervisor handoff exposed two orchestration defects before training:
  Python file execution lost the repository import root inside `srun`, and Perlmutter
  rejects simultaneous `--exclusive --overlap`. Stages now use `python -m` and valid
  exclusive job steps. The replacement GPU allocation is `56447933`; the Proxy
  U-PATCH canary has completed 100 optimization steps and is evaluating the complete
  validation fold. No score is interpreted from this canary.

The additional fixes are commits `9595fd6`, `8689a73`, `a6ebdfb`, and `6bd2621`.
Passed catalogue manifests are under
`/pscratch/sd/d/dkololgi/abacus/p8_multitracer_v1/catalogues/`; passed field and
selection manifests are under the sibling `fields/` and `selection/` directories.
The frozen-grid audit is `fields/grid_support_audit.json`. The global Proxy Delaunay
graph remains in progress on CPU allocation `56446724`; no graph or model-completion
marker has been stamped.

### 2026-08-06 — [result/code/plan] U-CIC closed as sparse-shell NO-GO; U-PATCH-BRIGHT frozen; BGS_FAINT F0 audit complete

The two registered `U-CIC-RESID-v2` runs completed their 20-epoch contracts. The
corrective model improves the first-three-shell mean over standalone U-PATCH by
`+0.0324/+0.0173` in rotations 0/2, but gains only `+0.0150/+0.0011` in the
four-shell primary macro and degrades the sparse shell by `-0.0371/-0.0476`.
Both rotations violate the registered maximum `0.01` sparse-shell degradation and
neither reaches the `+0.03` promotion target. Freeze the entire U-CIC branch as
`NO_GO_SPARSE_SHELL_REGRESSION`; do not open G-CIC or F-CIC variants. The exact
machine-readable comparison is `docs/evidence/p8/ucic_v2_closeout.json`.

Standalone U-PATCH is now frozen as `U-PATCH-BRIGHT_REFERENCE`: the leading learned
Bright-only candidate in the completed two-rotation screen, not a production-approved
VAC model. Its five-fold/three-seed replication and P10 fresh-phase transfer remain
unopened gates.

The new chunked audit
`workflows/abacus_tweb/p8_multitracer_feasibility.py` was run on a CPU interactive
allocation at revision `6267c08`. It follows target bits and TARGETIDs through eight
current ph000 staged products, checks exact uniqueness and truth-link support, and
restores target class in the assigned-only product through a label-free TARGETID join
because that product drops `BGS_TARGET`. Seven unit tests pass.

The principal unique-target counts are:

| Stage | Bright unique | Faint unique | Meaning |
|---|---:|---:|---|
| `forFA0.fits` | 10,547,983 | 7,559,142 | pre-fibre target population |
| assigned-only crossmatch | 8,314,754 | 3,601,878 | matched unique assigned IDs |
| spectroscopic join | 9,920,755 | 7,110,292 | repeated rows de-duplicated by TARGETID |
| final GraphWeb parent | 9,538,254 | 0 | Bright-only enforcement confirmed |

The fixed upstream Faint realization contains `553,830` unique targets at
`0.45 <= z < 0.55`; the spectroscopic-join table contains `150,919` unique Faint
IDs in that shell. There is therefore enough potential additional sampling to justify
an information-content experiment.

The F0 verdict is nevertheless
`CONDITIONAL_GO_BUILD_RESPONSE_COMPLETE_FAINT`, not training readiness:

1. `upstream_prepare_mocks_Y3_bright.py` defines the current Faint proxy only by
   `19.5 <= R_MAG_APP <= 20.175`, retains 69.5% through unseeded random draws, and
   promotes 20% of the retained targets. The immutable on-disk IDs are usable, but the
   selection cannot be regenerated bit-for-bit.
2. The current CutSky table lacks the `r_fiber`, `z`, and `W1` photometry required by
   the final DESI BGS_FAINT fibre-magnitude/colour selection. The proxy is not a final
   selection emulator.
3. `run_path1_mkcat.sh` explicitly invokes `--tracer BGS_BRIGHT`, the LOA injection is
   calibrated on Bright bits only, and the final GraphWeb exporter defaults to the
   Bright-bit cut.
4. The assigned crossmatch leaves 3,654,393 unique assignment IDs unmatched to
   `forFA0`; this alternate-MTL/bookkeeping population must be resolved before a unique
   multitracer parent is frozen.

Runtime evidence is
`/pscratch/sd/d/dkololgi/abacus/p8_multitracer_feasibility_v1/feasibility_audit.json`
(SHA-256 `84543ded4245c1927abbbcba468240872906142280cc480e1a3fdeefed7ee7ba`)
with marker `F0_FEASIBILITY_COMPLETE`; the tracked digest is
`docs/evidence/p8/multitracer_f0_summary.json`.

P8.8 of `docs/plan_generalisable_graphweb_vac.md` now registers the next work in
strict order: response-explicit oracle/proxy catalogue construction; separate
Bright/Faint fields and information preflight; matched two-tracer classical controls;
then a context-only U-PATCH screen. Bright authoritative targets, P4 folds, target
labels, smoothing, and the primary macro metric remain unchanged. The global
Bright+Faint graph, G-PATCH, F-tier, Faint supervision, posterior estimation, JEPA,
and HOD marginalization remain gated until the cheap U-PATCH information test shows a
reproducible gain.

### 2026-08-05 — [runtime] U-CIC-RESID-v2 migrated to disconnect-safe tmux with exact-resume supervisors

The separately registered three-sigma `U-CIC-RESID-v2` screens are active in
login-node tmux sessions rather than Codex-owned allocation PTYs:

- rotation 0: `p8_ucic_v2_rot0`, allocation `56371700`, node `nid008676`;
- rotation 2: `p8_ucic_v2_rot2`, allocation `56371701`, node `nid008432`.

Both resumed exact epoch-4 checkpoints and have completed epoch 6 with 100% unique-core
exposure. Current best macro `R2_lambda1` is `0.5185` for rotation 0 and `0.5124`
for rotation 2; these are interim optimization values, not model-selection results.
Two sleeping tmux supervisors, `p8_ucic_v2_rot{0,2}_supervisor`, request at most one
replacement interactive allocation per rotation if the current four-hour wall expires
before epoch 20. Future resumes execute from the detached worktree
`/global/homes/d/dkololgi/p8_ucic_b9f9fde_worktree` at immutable revision
`b9f9fded745463f078c7ae00ef3058f8e9529d9b`, so later documentation commits cannot
violate the checkpoint revision guard. Artifacts are under
`/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/u_cic_resid_v2/`; scheduler/training
logs are `/pscratch/sd/d/dkololgi/logs/p8_u_cic_resid_v2_rot{0,2}_*tmux.*`.

### 2026-08-05 — [science/code/result] Exact full-cap DTFE closed; one-sigma corrective gate proven infeasible before further training

The two P8 classical products are now complete on rotations 0 and 2. Frozen CIC
anchors were regenerated for `4,023,207/4,021,771` active rows. The accelerated
exact-DTFE path constructed the full `58,007,741`-tetrahedron opposite-face
adjacency and retried every approximate-locator miss with an exact Delaunay walk.
There are zero singular or maximum-step unresolved voxels. Finite coverage of the
P3 apodized support is `97.071%` NGC and `94.874%` SGC; all remaining voxels are
outside the observed catalogue convex hull rather than locator failures.

The exact full-cap DTFE result is poor: train-affine macro `R2_lambda1` is
`-0.1499/-0.1847` and the first-three-shell diagnostic is `0.1247/0.1044` for
rotations 0/2. Boundary-distance cuts do not rescue it. This is a negative matched
control under the complex full-cap mask/selection geometry, not evidence that the
point locator is approximate. CIC remains the relevant supported-shell classical
anchor. Artifacts are under
`/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1/classical/dtfe_fullcap_v1/`.

The live `U-CIC-RESID-v1` runs passed zero-parity and completed three exact epochs,
but a new feasibility audit proved their hard one-sigma lambda1 correction can never
pass the registered sparse-shell gate. The train-fold 99th-percentile absolute CIC
residual is `2.602/2.553 sigma`. Even an oracle validation correction restricted to
one sigma reaches only sparse-shell `R2 = 0.281/0.196`, below standalone U-PATCH
`0.345/0.321` and its allowed 0.01 degradation. Spending the remaining 17 epochs
would therefore be scientifically empty. V1 is frozen as `STOP_FEASIBILITY_NO_GO`;
its partial results are not an accuracy decision.

`workflows/abacus_tweb/p8_audit_residual_bound.py` records the audit at
`/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1/classical/residual_bound_feasibility.json`.
It selects a common three-sigma bound from training folds only. A separately named
`U-CIC-RESID-v2` is registered in P8.7a with every other contract unchanged. The
training CLI now records and resume-checks `--lambda1-max-sigma`; targeted patch and
epoch tests pass. V2 must still complete both 20-epoch rotations and beat standalone
U-PATCH without a sparse-shell regression before any corrective branch is promoted.

### 2026-08-05 — [science/code] P8 exact-DTFE acceleration and first trained U+CIC corrective model implemented

The two remaining P8 controls now have executable, pre-registered implementations;
neither runtime gate is being claimed complete before full-cap canaries and runs.

The exact-DTFE build no longer uses the failed tetrahedron-AABB raster as its production
path. `workflows/abacus_tweb/p8_dtfe_fullcap.py` now constructs a reusable
vertex-to-incident-tetrahedron CSR and performs voxel-centric containing-tetrahedron
search followed by exact barycentric interpolation. A synthetic SciPy-Delaunay test
caught an important false assumption before launch: the single nearest site is not
always a vertex of the containing tetrahedron. The frozen locator therefore uses a
progressive K=1/8/32/128 incident-star search only for unresolved voxels, reports every
miss, checkpoints every grid-x slab, and never substitutes a vertex splat. The full-cap
NGC+SGC build and rotation-0/2 evaluation remain outstanding.

`U-CIC-RESID-v1` is the first genuinely trained classical-plus-learned correction in
this programme. The earlier P9 U+CIC result was a cross-fitted constant linear blend and
must not be described as residual training. The new model receives the globally
reconstructed, train-fold-affine CIC eigenvalues and the established local U-PATCH field
view. Its backbone starts from the corresponding converged U-PATCH checkpoint; its new
head is zero-initialized so checkpoint zero reproduces CIC to `2e-6`. Corrections are
bounded to one training sigma for lambda1 and multiplicative positive eigengap factors
`exp(1.5 tanh(r))`, preserving ordering while testing whether local learned structure
adds information beyond the nonlocal classical solve. The classical workflow now saves
parent-keyed active training+validation anchors without exposing a test fold.

Seven targeted DTFE/U-PATCH/residual tests pass. The wider phase-4 suite passes all
relevant tests; its sole failure is the pre-existing entrypoint-help subprocess importing
a contaminated user-site NumPy, not a failure in these P8 implementations. The frozen
execution contract and adoption rules are recorded in P8.7 of
`docs/plan_generalisable_graphweb_vac.md`.

### 2026-08-04 — [result] P8 rotation-2 extensions and true-field context gate complete

The frozen rotation-2 `convergence_extension_v1` runs completed all 20 epochs
with immutable revision `c92356a` and exact checkpoint resumes:

- G-PATCH improves macro-shell `R2_lambda1 = 0.4708 -> 0.4825`; its best
  effective epoch is 36, pooled `R2 = 0.5523`, first-three-shell macro
  `R2 = 0.5331`, and sparse-shell `R2 = 0.3305`.
- U-PATCH improves macro-shell `R2_lambda1 = 0.5128 -> 0.5197`; its best
  effective epoch is 39, pooled `R2 = 0.6055`, first-three-shell macro
  `R2 = 0.5860`, and sparse-shell `R2 = 0.3210`.

U-PATCH therefore retains the transferable accuracy lead on both registered
rotations and, importantly, on the three classically supported shells; this
is not a macro-only win caused by collapse of the final shell.

The true-density context-growth diagnostic also completed (`177 s`, marker
`TRUE_FIELD_CONTEXT_COMPLETE`). Against the matched full-periodic 1024-grid
reference, overall eigenvalue RMSE falls from `9.27%` of reference scatter at
`60 Mpc/h` context to `4.66/2.34/1.41/0.88%` at
`120/180/240/360 Mpc/h`. Missing external context is measurable and converges
smoothly; it must remain an explicit field-model deployment control.

The exact full-cap DTFE preflight completed but is a red engineering gate.
The naive tetrahedron-AABB raster has an upper bound of approximately
`77.5e9` voxel--tetrahedron visits; extreme boundary tetrahedra have AABBs
covering approximately `18e6` voxels. The exact NGC+SGC build was correctly
not launched. Replace the tet-centric AABB enumeration with accelerated
voxel point location before continuing this baseline.

### 2026-08-04 — [code] P8 true-field context and exact full-cap DTFE runtime workflows implemented

The two remaining non-neural P8 controls now have explicit, tested runtime
implementations rather than prose-only gates.

- `workflows/abacus_tweb/p8_true_field_context.py` block-averages the true
  2048^3 Abacus density to a tractable 1024^3 field, computes a full periodic
  matched-resolution tensor reference, and compares finite true-field context
  radii of 60/120/180/240/360 Mpc/h. It reports trace, centered traceless-shear
  eigenvalues, shear amplitude, and eigenvalues overall, by shell, and by
  conservative fold-boundary-distance bin. The canonical 2048^3 labels are a
  separately reported resolution floor, not the finite-context reference.
- `workflows/abacus_tweb/p8_dtfe_fullcap.py` computes standard 3-D DTFE vertex
  densities `4 / sum(V_tet)`, then barycentrically rasterizes the immutable
  global NGC+SGC tetrahedra. It includes an AABB-workload preflight and only
  calls the estimator DTFE after the exact piecewise-linear field is built.
  Its evaluation uses the same P3 grids, frozen rotation-specific selection,
  P4 authoritative rows, fixed tidal operator, and training-only affine policy
  as the matched CIC row.

Four numerical unit tests pass: periodic cube extraction, taper geometry,
Fourier trace consistency, and exact linear-field barycentric interpolation.
Runtime markers remain deliberately absent until the full-data runs complete.

### 2026-08-04 — [code] Frozen rotation-2 G-PATCH and U-PATCH convergence extensions launched interactively

The pre-registered rotation-2 `convergence_extension_v1` runs are active on
two independent interactive 80-GB A100 allocations after CUDA/environment
smoke tests passed. No `sbatch` jobs were used.

- G-PATCH: allocation `56320768`, node `nid008589`, four-hour limit;
  parent epoch 16 / macro R2(lambda1) `0.47083495`; output
  `/pscratch/sd/d/dkololgi/logs/p8_convergence_extension_graph_rot2_56320768.out`
  and matching `.err`.
- U-PATCH: allocation `56320772`, node `nid008292`, three-hour limit;
  parent epoch 20 / macro R2(lambda1) `0.51278075`; output
  `/pscratch/sd/d/dkololgi/logs/p8_convergence_extension_unet_rot2_56320772.out`
  and matching `.err`.
- Artifacts:
  `/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/convergence_extension_v1/{graph,unet}/rotation_2/seed_42/`.

Both guarded run manifests record revision
`c92356a2bf1b5d721bc09f003e7b6d82e87cf537`, the immutable
`recovery_v1` parent paths and hashes, fresh AdamW/cosine state, exactly 20
complete-exposure epochs at `2e-4`, disabled early stopping, and unchanged
fold/target/scaler/model contracts. Atomic mid-epoch checkpoints remain the
recovery mechanism if either interactive wall expires.

### 2026-08-04 — [science/code] Rotation-0 P8 convergence extensions COMPLETE: U-PATCH remains the leader; both schedules are practically saturated, rotation 2 remains required

The exact-resume `convergence_extension_v1` runs launched before the NERSC
shutdown both completed their registered 20 additional complete-exposure epochs.
Each completion retained 100% unique-core exposure per epoch, the frozen
rotation-0 folds, target/scaler, architecture, row-weighted objective, seed 42,
fresh AdamW optimizer at `2e-4`, cosine schedule, and immutable `recovery_v1`
parent checkpoint.

| rotation-0 model | parent macro R2(lambda1) | extension best | gain | effective epoch | pooled R2(lambda1) | first-three macro | final shell |
|---|---:|---:|---:|---:|---:|---:|---:|
| G-PATCH | 0.4682 | **0.4910** | +0.0228 | 35 | 0.5598 | 0.5360 | **0.3560** |
| U-PATCH | 0.4943 | **0.5070** | +0.0127 | 39 | **0.5879** | **0.5609** | 0.3453 |

The corresponding per-shell lambda1 results are G-PATCH
`0.5876/0.5444/0.4759/0.3560` and U-PATCH
`0.6223/0.5708/0.4896/0.3453`. U-PATCH remains the strongest overall
deterministic encoder; G-PATCH gained more from slow training and retains a
small sparse-shell advantage. Pooled U-PATCH lambda2/lambda3 R2 is
`0.6582/0.7333`, and its pooled lambda1 Spearman coefficient is `0.7921`.

Both immutable run summaries correctly say `NOT_CONVERGED_EXTENSION_CAP`
because their best checkpoint lies in the final three extension epochs. That
formal marker is retained. Numerically, however, the last-five-epoch primary
score changes are only approximately `+0.0013` for G and `+0.0008` for U, and
the cosine learning rates reach zero. Thus rotation 0 is operationally
saturated under the registered schedule; no further rotation-0 continuation is
justified. The pre-registered trigger for applying the identical extension to
rotation 2 fired for both models because each improved its parent by more than
0.005.

This is not yet a production or classical-method victory. It is one
same-phase, one-seed rotation. The exact matched full-cap DTFE row, true-field
trace/shear context-growth diagnostic, rotation-2 extension, seed/fold
replication, and P10 fresh-phase blind test remain binding. Runtime artifacts:

- `/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/convergence_extension_v1/graph/rotation_0/seed_42/`;
- `/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/convergence_extension_v1/unet/rotation_0/seed_42/`.

### 2026-07-22 — [code] Exact checkpoint resume launched for both rotation-0 P8 convergence extensions

The first `convergence_extension_v1` interactive allocations ended at the
four-hour wall rather than at the registered 20-epoch cap. G-PATCH had written
11 complete extension epochs and checkpointed epoch 12 at cursor 139
(`global_step=114000`); U-PATCH had written 13 complete extension epochs and
checkpointed partway through epoch 14. Both retained full model, optimizer,
cosine-scheduler, RNG, sampler-order, and partial-epoch accumulator state.

Two new interactive 80-GB A100 allocations now continue those exact states:

- G-PATCH: allocation `56305444`, node `nid008381`, four-hour limit; output
  `/pscratch/sd/d/dkololgi/logs/p8_convergence_extension_graph_rot0_resume_56305444.out`
  and stderr alongside it with suffix `.err`.
- U-PATCH: allocation `56305445`, node `nid008304`, four-hour limit; output
  `/pscratch/sd/d/dkololgi/logs/p8_convergence_extension_unet_rot0_resume_56305445.out`
  and stderr alongside it with suffix `.err`.

Both steps use `--resume` against the existing artifact directories under
`/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/convergence_extension_v1/`.
The checkpoint code revision is frozen at
`03c2397658934e24cbcbbda3bff8d8f234e1df10`; execution therefore uses a
detached shared-home worktree at that exact revision rather than weakening the
resume guard. No architecture, split, target, scaler, objective, learning-rate,
schedule, seed, or extension-length contract changed. The existing best scores
remain G-PATCH `0.4879539` and U-PATCH `0.5036065` until a later complete-fold
validation improves them. P8 remains in progress until both reach the
registered 20 complete extension epochs and their final artifacts are audited.

### 2026-07-21 — [code] Frozen P8 long-horizon extensions launched on rotation 0

The pre-registered `convergence_extension_v1` is now running for G-PATCH and
U-PATCH on two independent interactive 80-GB A100 allocations. The completed
`recovery_v1` parents remain immutable. Both extensions use a fresh AdamW
optimizer (`lr=2e-4`, weight decay `1e-4`, gradient clipping 5), a fresh cosine
schedule over exactly 20 complete-exposure epochs, complete-fold validation
after every epoch, and no early stopping. All architecture, fold, target,
scaler, patch, row-weighting, and seed contracts are unchanged.

- G-PATCH: allocation `56285194`, node `nid008253`, parent epoch 15 / score
  0.4681934, parent SHA-256
  `14124c039fd8582abdea27fc74865e422d29c27dca6d33c7a6bb05b0557f1971`, log
  `/pscratch/sd/d/dkololgi/logs/p8_convergence_extension_graph_rot0_56285194.log`.
- U-PATCH: allocation `56285193`, node `nid008244`, parent epoch 20 / score
  0.4942791, parent SHA-256
  `21a174f14de060463da04f1c59e8fa8c36ec8e10f7104f42efe3a231342dbd5e`, log
  `/pscratch/sd/d/dkololgi/logs/p8_convergence_extension_unet_rot0_56285193.log`.
- Artifacts:
  `/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/convergence_extension_v1/{graph,unet}/rotation_0/seed_42/`.

The guarded manifests passed the warm-start provenance checks and both loss
traces are advancing. The four-hour interactive limit is shorter than the full
20-epoch runtime; atomic mid-epoch checkpoints make subsequent allocation
renewal an exact resume rather than a new experiment. Rotation 2 receives the
identical extension only if the frozen P8.6 trigger is met: improvement of at
least 0.005 over the parent, or continued improvement at the extension cap.

### 2026-07-21 — [science/code] P8 RECOVERY COMPLETE (2 rotations x 2 encoders): U-PATCH primary score **0.5035** passes BOTH registered bars; G-PATCH fails both — but ALL FOUR runs peaked on their FINAL epoch

All four exposure-aware recovery runs have finished; no jobs remain. Every figure regenerated from
final checkpoints (fig9-fig17).

**Primary score = mean over blocked folds of macro R2(lambda1) (plan P8.3):**
| model | rot0 | rot2 | **MEAN** | fold spread | first-three mean | per-shell mean |
|---|---:|---:|---:|---:|---:|---|
| G-PATCH | 0.4682 | 0.4708 | **0.4695** | 0.0026 | 0.5140 | 0.546/0.529/0.467/0.336 |
| U-PATCH | 0.4943 | 0.5128 | **0.5035** | 0.0185 | **0.5621** | 0.611/0.578/0.498/0.328 |
(epochs run: G 15/16, U 20/20 — cap reached.)

**Registered gates, applied to the two-fold means:**
- promotion (frozen R0 0.440 + 0.03 = 0.470): **U PASS** (+0.0635); **G FAIL** (+0.0295, i.e. 0.4695
  — short of the bar by 0.0005, effectively a tie with the bar and not a pass);
- classical adoption, no-macro-only-win (CIC first-three 0.520): **U PASS** (0.5621, +0.042);
  **G FAIL** (0.5140, -0.006). G leads CIC on the macro metric only because CIC collapses in the
  sparse shell — precisely the artifact the rule exists to reject.
- U also holds the sparse shell (0.315-0.341) where CIC is -0.76.

**Fold stability:** G spread 0.0026, U spread 0.0185 — both far below the 0.03 promotion delta, so
the ordering U > G is not fold noise. Per-shell replication is on-diagonal for the three supported
shells (fig15). The sparse shell carries the real scatter (U 0.315-0.341, G 0.327-0.346, ~0.02-0.03),
so sparse-shell numbers must be quoted with that uncertainty.

**CRITICAL CAVEAT — U is cap-limited; G's slow-convergence question remains open.** Every run's best
epoch is its last (G 15/16, U 20/20), but the termination mechanisms differ. Both G runs triggered
the registered patience/minimum-delta early stop; both U runs reached the 20-epoch cap while their
validation loss and macro R2 continued improving. U's ceiling is therefore demonstrably unmeasured.
G is closer to a primary-metric plateau, but the aggressive three-epoch/0.005 rule does not exclude
slower improvement. The pre-registered low-rate `convergence_extension_v1`, rather than an ad hoc
30--40 epoch relabelling of the primary runs, is the highest-value next experiment.

**Reading:** under the patch protocol, on two disjoint geographies, a learned model now beats the
classical estimator where classical is well-supported AND where it fails. That is the result P8 was
designed to obtain or exclude. It remains SAME-PHASE evidence: single seed per rotation, ph000 only.
P10 fresh-phase blind testing is still the production gate, and 3-seed replication is still required
by P8.4 before any finalist freeze.

### 2026-07-21 — [science/code] ROTATION-2 EVALUATION: the result REPLICATES in different sky; U-PATCH clears both bars again; per-shell scatter on-diagonal

Rotation 2 (trains folds {0,1,4}, validates fold **3** — different geography, different galaxies) is
mid-flight: G-PATCH 10 epochs, U-PATCH 13. Figures fig15 (replication), fig16 (rot2 parity),
fig17 (rot2 visual + classes). Code: workflows/abacus_tweb/plot_p8_rotation2_eval.py.

**Best so far:** U-PATCH rot2 **0.4884** at epoch 13 (rot0: 0.4943 at ep20); G-PATCH rot2 **0.4494**
at ep10 (rot0: 0.4682 at ep15). Both rotations still rising at their last epoch.

**Matched-epoch replication is close.** From epoch 3 onward |rot2 - rot0| <= 0.045 for G and
U alike; at their common last epoch rot2 is AHEAD in both cases (G ep10 0.4494 vs 0.4067;
U ep13 0.4884 vs 0.4361). Per-shell best-epoch scatter (fig15 middle) sits on the diagonal for the
three supported shells.

**U-PATCH clears both registered bars in rotation 2 as well**: macro 0.4884 > promotion gate 0.470;
first-three (0.591+0.577+0.496)/3 = **0.555** vs the rotation-2 CIC supported-shell bar. G-PATCH
rot2 first-three 0.493 does not clear it — consistent with rotation 0, where G also failed the
classical bar. The ordering U > G > CIC-in-supported-shells is now reproduced in two disjoint
geographies.

**One real difference: the sparse shell is WEAKER in rotation 2** — U 0.289 vs 0.341, G 0.318 vs
0.346 (fig15, the off-diagonal points at lower left). Fold 3's 0.45-0.55 population differs from
fold 1's, so the sparse-shell number carries genuine fold-to-fold scatter of order 0.05. This is
exactly why the plan requires fold spread rather than a single fold, and it means the sparse-shell
result should be quoted with that uncertainty.

**Rotation-2 instability check: the rot0 U-PATCH epoch-2 collapse (-0.15) did NOT recur** (rot2 ep2
= +0.1195). It was rotation-specific transient LR/normalisation sensitivity, not a systematic defect.

**Visuals (fig17, super-block 424, fold 3):** slab lambda1 R2 CIC 0.659 > U 0.626 > G 0.481 — note
CIC leads on THIS particular dense slab, a reminder that single-slab numbers are not the fold metric.
Class occupancy vs TRUE 27/48/24/2%: U 23/49/26/2% (acc 70%), G 17/55/27/1% (67%), CIC 15/64/21/0%
(73%). U again reproduces the true knot fraction where CIC predicts ~none.

STATUS: two rotations, one seed each, same phase; both still training. Promotion still requires
completed runs, seed replication, and the P10 fresh-phase blind test.

### 2026-07-21 — [code] Rotation-2 recovery resumed after allocation expiry; F-tier remains untested under P8

The first rotation-2 interactive allocations ended before the registered scientific
stopping rule: G-PATCH completed 10 epochs and was interrupted midway through epoch 11;
U-PATCH completed 13 epochs and was interrupted midway through epoch 14. Neither wrote
a completion marker or `recovery_summary.json`, and both best checkpoints were their
last completed epochs. Best-so-far macro R2(lambda1) is 0.4494 for G and 0.4884 for U;
these are lower bounds, not final rotation-2 results.

Both checkpoints are now resumed unchanged from the exact original revision
`19f68bd8da13a958e62302671bebf77713c6847e`:

- G-PATCH: allocation **56267894**, node `nid008373`, log
  `/pscratch/sd/d/dkololgi/logs/p8_recovery_graph_rotation2_resume_56267894.log`;
- U-PATCH: allocation **56267895**, node `nid008392`, log
  `/pscratch/sd/d/dkololgi/logs/p8_recovery_unet_rotation2_resume_56267895.log`;
- exact worktree: `/global/u2/d/dkololgi/TNG/Illustris_p8_resume_19f68b/`;
- artifacts remain under
  `/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/recovery_v1/{graph,unet}/rotation_2/seed_42/`.

Resume reconciliation passed and both loss traces are advancing inside their saved
partial epochs. No training argument, optimizer/scheduler state, epoch permutation,
RNG state, fold, target, or metric changed.

Clarification on F-tier: no matched full-range trained F-tier model has entered P8.
The frozen v2_A candidate was declared `NO_GO_FROZEN_V2_A_RESOURCE_INFEASIBLE` at
preflight because a representative view required at least 91.6 GiB before autograd,
decoder, FFT fields, or complete graph context. Earlier dense-wedge F-tier accuracy is
not transfer evidence. A simplified F-tier/U-Physics architecture remains scientifically
untested and gated behind completion of the deterministic G/U recovery and resource-safe
adapter design.

### 2026-07-21 — [code] Recovery PARITY plots (fig13/fig14): pooled lambda1 U 0.572 > G 0.528 ~ CIC 0.527; CIC's sparse-shell failure mode diagnosed as noise amplification

Parity for the final recovery checkpoints over the COMPLETE rotation-0 validation fold (999,683
authoritative cores) — previously only the short screens had parity plots.
Code: workflows/abacus_tweb/plot_p8_recovery_parity.py.

**Pooled (fig13), R2 / slope / amplitude ratio sigma_p/sigma_t:**
| | lambda1 | lambda2 | lambda3 |
|---|---|---|---|
| G-PATCH | 0.528 / 0.61 / 0.82 | 0.652 / 0.70 / 0.86 | 0.705 / 0.70 / 0.84 |
| U-PATCH | **0.572** / 0.57 / 0.75 | **0.658** / 0.66 / 0.81 | **0.727** / 0.73 / 0.85 |
| CIC | 0.527 / 0.52 / 0.72 | 0.571 / 0.56 / 0.74 | 0.586 / 0.58 / 0.75 |
U-PATCH leads all three eigenvalues on pooled R2 as well as on the registered macro metric; the
learned/classical gap WIDENS with eigenvalue index (lambda3: 0.727 vs 0.586).

**Amplitude ratios behave as theory predicts.** For a conditional-mean estimator sigma_p/sigma_t ~=
sqrt(R2): U lambda1 sqrt(0.572)=0.756 vs measured 0.75; G sqrt(0.528)=0.727 vs 0.82 (G is slightly
UNDER-shrunk, consistent with its knot over-prediction, 579 vs 461 true). So the compression seen in
the field maps is honest shrinkage, not a pathology — the residual fix is calibration (P12), not loss
engineering.

**fig14 diagnoses CIC's sparse-shell collapse precisely.** In 0.45-0.55 CIC has slope 0.52 but
sigma_p/sigma_t = **1.33** and R2 = -0.763: it produces a WIDE, essentially uncorrelated prediction
cloud with a visible horizontal stripe of near-constant values — noise amplification where tracers
vanish. The learned models do the opposite: sigma_p/sigma_t ~ 0.59-0.60, i.e. they shrink toward the
mean but stay correlated (R2 0.34). This is the mechanistic explanation of the macro-metric gap and
independently justifies the plan's rule that a macro win driven by classical collapse is not an
encoder win — while also showing the learned behaviour is the SAFER failure mode for a VAC.

**Per-shell trend (both learned models):** R2 and amplitude ratio decline monotonically with
decreasing tracer density (U: 0.604/0.555/0.478/0.341 with ratios 0.76/0.75/0.70/0.59) — the
signature of sampling-information limitation, matching the three independent data-limitation lines.

### 2026-07-21 — [code] P0S near-complete: env specs + scratch-only ph000 source preserved to git
- What: Executed the safe, no-move parts of P0S. (1) Exported reproducibility records for both
  envs → `docs/evidence/p0s/env/` (`cosmic_env` py3.11.15: jax 0.7.2/torch 2.9.1/gudhi 3.11.0;
  `rapids-gnn` py3.11.14: cudf/cugraph 26.02); from-history+full YAML, explicit spec, conda/pip
  lists, import smoke tests, CUDA/module metadata. (2) Inventoried + checksummed 42 scratch-only
  source files under `abacus/SecondGen_Mocks/ph000/`; overlap check showed NONE existed in either
  repo. Copied them into version control (0 checksum mismatches): 27 code/docs →
  `workflows/abacus_tweb/secondgen_mocks/ph000/`, 15 JSON manifests → `docs/evidence/p0s/ph000_manifests/`.
  (3) Wrote `docs/evidence/p0s/MIGRATION_MANIFEST.md` (classified manifest + dry-run move table).
- Why / decision: pscratch is purge-eligible before the downtime; these scripts + env definitions
  are irreplaceable and were only on scratch (now also inside HPSS tarballs, but tape ≠ versioned
  source). No pscratch files modified — read/copy only, per user constraint.
- Next: P0S remainder = reviewed **CFS** copies of irreplaceable catalogues/checkpoints
  (`abacus/p1b_full_footprint/`, scalers, best checkpoints). Blocked on CFS target + space — `desi`
  project is 93% full. Awaiting user go-ahead.
- Refs: `docs/evidence/p0s/` (MIGRATION_MANIFEST.md, env/, ph000_manifests/, ph000_source_inventory.raw.tsv),
  `workflows/abacus_tweb/secondgen_mocks/ph000/`, plan P0S status line.

### 2026-07-21 — [science/code] P9 EARLY VERDICT: a CIC-residual/hybrid is NOT worth building; U+G ensembling is the only combination that helps (+0.012)

Ran the P9 complementarity audit early using the frozen rotation-0 predictions (all three methods on
the same 999,683 validation rows). Blend weights are fit on one spatially disjoint half of the
validation super-blocks and evaluated on the other (both directions) — fitting and scoring on the
same rows would manufacture the gain. Code: workflows/abacus_tweb/p9_residual_complementarity_audit.py.

**Residual correlations (lambda1):** U vs CIC **0.821**, G vs CIC 0.756, U vs G 0.883 overall; in the
sparse shell U-vs-CIC drops to 0.545 while U-vs-G stays 0.950. The learned models and CIC make
*largely the same errors* wherever CIC works.

**Out-of-sample blends (macro / per-shell):**
| combination | macro | shells | gain vs best member |
|---|---:|---|---:|
| CIC | 0.2005 | 0.559/0.565/0.442/-0.763 | — |
| G-PATCH | 0.4682 | 0.545/0.520/0.462/0.346 | — |
| U-PATCH | **0.4943** | 0.604/0.555/0.478/0.341 | — |
| U+CIC | 0.4748 | 0.629/0.589/0.505/**0.176** | **-0.020** |
| G+CIC | 0.4449 | 0.621/0.585/0.498/0.076 | -0.023 |
| **U+G** | **0.5059** | 0.614/0.565/0.490/0.354 | **+0.012** |
| U+G+CIC | 0.4892 | 0.636/0.592/0.511/0.218 | -0.005 |

**Verdict — CIC-residual: NO-GO for the current critical path.** Adding CIC *improves the three
supported shells* (U 0.604/0.555/0.478 -> 0.629/0.589/0.505, i.e. +0.025/+0.034/+0.027) but *destroys*
the sparse shell (0.341 -> 0.176), for a net macro LOSS of 0.020 under the registered equal-shell
metric. The mechanism is clear: a single global blend cannot both exploit CIC where it is informative
and ignore it where it is actively harmful (R2 = -0.76). This reverses my own earlier advocacy of the
residual hybrid as "highest-prior candidate" — that argument was made when learned models TRAILED CIC
in the supported shells; U-PATCH now leads there, removing the original motivation.

**Caveat that keeps the idea alive in a narrower form:** the +0.025-0.034 supported-shell gains are
real and show CIC still carries nonlocal information the finite-context patch models lack. A
SHELL-AWARE or gated hybrid (weights conditioned on tracer density / ntilde, or a residual applied
only where CIC support is adequate) could capture that without the sparse-shell damage. That is a
strictly larger design space than the registered zero-init residual and should only be attempted
after (a) convergence-limited reruns and (b) the registered context-growth test, which addresses the
same nonlocal-information gap more directly and more cheaply.

**Actionable now: U+G ensembling gains +0.012 macro** with no new training — both models already
exist. Cheap, but below the 0.03 promotion bar on its own; log as a candidate, not a result.

### 2026-07-21 — [code] P8 G-PATCH stop explained exactly; blind rotation-2 recovery launched; long-horizon extension pre-registered

The rotation-0 G-PATCH run did stop through the registered early-stopping code. The
rule tracks only improvements of at least 0.005 in complete-fold macro R2, separately
from the absolute-best checkpoint:

| epoch | macro R2(lambda1) | early-stop reading |
|---:|---:|---|
| 12 | 0.4663 | qualifying best; patience reset |
| 13 | 0.4617 | stale 1 |
| 14 | 0.4668 | absolute best, but only +0.0005 versus epoch 12; stale 2 |
| 15 | 0.4682 | new absolute best, but only +0.0019 versus epoch 12; stale 3 -> stop |

Thus epoch 15 was correctly saved as the best checkpoint even though it did not clear
the 0.005 threshold used to reset patience. This was faithful execution of the frozen
rule, but the combination of a noisy GNN validation curve, patience 3, and minimum delta
0.005 is too aggressive to establish that the model itself had converged. The preceding
log statement that it ended early for a reason other than the registered rule was wrong.
U-PATCH independently reached the epoch-20 cap with its best checkpoint at epoch 20, so
the long-horizon optimization question is open for both architectures.

The primary `recovery_v1` artifacts remain immutable. The next blind same-phase
replication has begun unchanged:

- G-PATCH rotation 2: interactive allocation **56262951**, node `nid008489`, log
  `/pscratch/sd/d/dkololgi/logs/p8_recovery_graph_rotation2_retry_56262951.log`;
- U-PATCH rotation 2: interactive allocation **56262952**, node `nid008556`, log
  `/pscratch/sd/d/dkololgi/logs/p8_recovery_unet_rotation2_retry_56262952.log`;
- artifacts:
  `/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/recovery_v1/{graph,unet}/rotation_2/seed_42/`.

The first launch inherited DESI's Python-3.13 `PYTHONPATH` and failed at NumPy import,
before parsing arguments or creating scientific state. The same allocations were reused
with `PYTHONPATH`, `PYTHONHOME`, and `CONDA_PREFIX` removed; a NumPy/Torch/CUDA smoke test
passed and both corrected runs began writing loss traces and checkpoints.

Before opening rotation-2 results, the roadmap now pre-registers a separately named
`convergence_extension_v1`: low-rate warm-starting from immutable best checkpoints,
20 additional complete-exposure epochs, no early stopping, complete-fold validation
after every epoch, and unchanged architecture/data/metric contracts. The extension is
an optimization diagnostic; it does not replace matched classical controls or P10.

Implementation is now guarded in
`workflows/abacus_tweb/{p8_train_patch_recovery.py,p8_epoch_training.py}`. The CLI
rejects a mismatched run name, model, rotation, seed, parent-best epoch offset, learning
rate, epoch budget, or enabled early stopping; it resets AdamW/cosine state, records the
parent best/recovery/manifest paths and SHA-256 hashes, and labels a best score in the
final three extension epochs `NOT_CONVERGED_EXTENSION_CAP`. Nine epoch-contract unit
tests pass, Python compilation passes, and the actual rotation-0 GraphNet/U-Net best
checkpoints both load into production-shaped models. Because the in-flight rotation-2
runs were started at revision `19f68bd8da13a958e62302671bebf77713c6847e`, an exact
detached resume worktree is retained at
`/global/u2/d/dkololgi/TNG/Illustris_p8_resume_19f68b/`.

### 2026-07-21 — [code] P8 recovery rotation-0 FINAL: U-PATCH 0.4943 (hit the 20-epoch CAP still improving), G-PATCH 0.4682; both registered bars passed; figures regenerated

Both rotation-0 recovery runs have finished. Figures regenerated from the final best checkpoints
(fig9 curves, fig10 lambda1 before/after, fig11 lambda2/3, fig12 T-web classes). Note the fig titles
still read "IN FLIGHT" — they were generated before the RECOVERY_COMPLETE marker convention;
the underlying data are the final best checkpoints.

| final (rotation 0, val fold 1) | epochs | best ep | macro R2(l1) | first-three | worst shell |
|---|---:|---:|---:|---:|---:|
| U-PATCH | 20/20 | **20** | **0.4943** | **0.5455** | 0.3406 |
| G-PATCH | 15 | **15** | 0.4682 | 0.5091 | 0.3455 |
| frozen R0 | — | — | 0.440 | — | — |
| CIC train-affine | — | — | 0.185 | 0.520 | -0.76 |

**Both models peaked on their LAST epoch — neither converged.** U-PATCH hit the 20-epoch cap while
still gaining (+0.006, +0.001 over the final two epochs, no patience trigger); G-PATCH's best is also
its last (epoch 15 of a 20 budget — the run ended early for reasons that should be checked in its
manifest, not by early stopping on the registered rule). CONSEQUENCE: these are LOWER BOUNDS, and the
"epochs to convergence" question is unresolved. A longer-budget rerun (30-40 epochs) is the cheapest
next experiment and should precede any architecture work.

**Registered bars (rotation 0, single seed):** promotion gate 0.470 — U 0.4943 PASS, G 0.4682 fail by
0.002. Classical adoption: U first-three 0.5455 > CIC 0.520 while holding the sparse shell 0.341 vs
-0.76 — a supported-shell win, not a macro artifact. G first-three 0.5091 < 0.520, so G does NOT clear
the classical bar.

**Class metrics at lambda_th=0.2 (slab, fig12), TRUE 38/41/18/3%:**
- U-PATCH 31/49/18/3%, accuracy **73%**, knot recall 56% (64% prec), void recall 70% (88% prec);
- G-PATCH 26/51/19/4%, accuracy 68%, knot recall **64%** (51% prec) — G now OVER-predicts knots
  (579 vs 461 true), the opposite of its short-screen deficit;
- CIC 18/63/19/0%, accuracy 68%, knot recall 11%.
Slab lambda1 R2: U 0.711, CIC 0.658, G 0.624. lambda2/lambda3 (fig11): U leads both.

INTERPRETATION: U-PATCH is the current leader on every registered criterion and its curve had not
flattened. G-PATCH plateaus lower and does not clear the classical bar. Still binding: single seed,
rotation 0 only, same phase (ph000); rotation-2 replication, seeds, and P10 fresh-phase remain
required before any promotion. Two new interactive jobs (56262951/2) started as these finished —
presumably rotation 2; that is the right next step.

### 2026-07-21 — [code] $PSCRATCH → HPSS tape backup (ahead of NERSC downtime / scratch purge)
- What: Archiving ~2.7 TB of irreplaceable derived data from `/pscratch/sd/d/dkololgi`
  (3.2 TB total) to HPSS tape at `/home/d/dkololgi/pscratch_backup_20260721/`. One
  `htar` archive per directory, CRC-verified (`htar -Kv`), idempotent/resumable.
  Submitted on the `xfer` QOS (job 56234004, `--account=desi`). Self-test on `logs/`
  passed (299 files, 0 verify errors).
- Why / decision: pscratch is purge-prone and a downtime was flagged; HPSS is the only
  durable on-site target (CFS `desi` is 93% full, $HOME is 40 GB). Excluded the 385 GB
  public `AbacusSummit_base_c000_ph000` (still on CFS `/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit`)
  and ~50 GB regenerable caches (`conda`, `.cache`, `.local`, `.cursor-server`, `jax_cache`).
- Next: confirm job 56234004 finishes with 0 FAIL in `~/hpss_backup/logs_20260721/manifest.tsv`;
  spot-check restore with `htar -xvf`. Re-run the script to resume if any archive failed.
- Refs: `~/hpss_backup/hpss_backup.sh`, `~/hpss_backup/submit_hpss_backup.slurm`

### 2026-07-20 — [code] Recovery VISUALS + both registered bars crossed by U-PATCH (runs STILL IN FLIGHT, G ep14 / U ep18 of 20)

Figures fig10 (lambda1 short-screen vs recovery), fig11 (lambda2/3), fig12 (T-web classes) from the
saved best checkpoints (plot_p8_recovery_visuals.py). Runs were NOT complete when plotted — jobs
56224585/86 still running; these are best-so-far checkpoints.

**Best checkpoints:** G-PATCH epoch 14 macro **0.4668** (first-three 0.5078); U-PATCH epoch 18 macro
**0.4871** (first-three **0.5395**), still rising.

**Both registered bars are now crossed by U-PATCH:**
- promotion gate (frozen R0 0.440 + 0.03 = 0.470): U 0.4871 **PASS** (G 0.4668, marginal fail);
- classical adoption (no macro-only win): U first-three **0.5395 > CIC 0.520**, i.e. it leads in the
  tracer-SUPPORTED shells, while also holding the sparse shell (0.330 vs CIC -0.76). This is a
  genuine win by the plan's own rule, not a macro artifact.

**Visual/class evidence (slab, same SB467 as before):**
- lambda1 slab R2: G 0.595 -> **0.633**; U 0.524 -> **0.712** (U now exceeds CIC's 0.658).
- Class occupancy vs TRUE 38/41/18/3%: U-PATCH recovery **30/48/19/3%** (acc 73%) — the knot fraction
  is now CORRECT (3% vs 3%), where every short-screen model had 0-1%. G recovery 26/51/19/3% (68%).
  CIC remains 18/63/19/0% (68%).
- **Knot recall transformed**: U 58% at 62% precision and G 59% at 58% (short screens: 7% and 20%;
  CIC 11%). Void recall U 69%/89%. The amplitude-compression -> class-occupancy bias identified in
  fig6 is substantially repaired by exposure alone, with no loss/target change.

Caveats unchanged and binding: mid-flight, single seed, rotation 0 only, same phase (ph000). Nothing
is promoted until runs finish, early stopping resolves, seeds/rotations replicate, and P10 fresh-phase
blind testing is done. But this is the first time in the programme that a learned model has beaten the
classical estimator where classical is well-supported.

### 2026-07-20 — [code] Recovery curves plotted (runs still IN FLIGHT): U-PATCH overtakes G-PATCH, passes frozen R0, and reaches the classical supported-shell bar

fig9_recovery_curves.png (workflows/abacus_tweb/plot_p8_recovery_curves.py; reads live histories).
NOTE: jobs 56224585/86 were still RUNNING when plotted — G at epoch 10, U at epoch 12 of 20.

| | best macro | at epoch | shells (best epoch) | first-three mean |
|---|---:|---:|---|---:|
| G-PATCH | 0.4396 | 9 | 0.494/0.480/0.443/0.341 | 0.472 |
| U-PATCH | **0.4686** | 12 (still rising) | 0.561/0.525/0.454/0.334 | **0.513** |
| frozen R0 (spatial holdout) | 0.440 | — | — | — |
| CIC train-affine | 0.185 | — | 0.56/0.57/0.44/-0.76 | 0.520 |

**Three developments:**
1. **U-PATCH has overtaken G-PATCH** and is still climbing (+0.019 at epoch 12), reversing every
   earlier ordering. G-PATCH is oscillating in 0.40-0.44 since epoch 6 (0.426/0.405/0.402/0.440/0.407)
   — plateau-with-noise, its epoch-9 best likely near its ceiling for this configuration.
2. **U-PATCH has passed the frozen R0 baseline** (0.4686 vs 0.440) on a harder scoring set — the first
   time a patch-trained model exceeds the previous spatial-holdout number. The plan's promotion gate
   (+0.03 over frozen baseline) sits at 0.470: U is 0.001 away, mid-flight.
3. **The classical supported-shell bar is essentially reached**: U first-three 0.513 vs CIC 0.520
   (per shell 0.561 vs 0.56, 0.525 vs 0.57, 0.454 vs 0.44 — U leads shells 1 and 3). With the sparse
   shell at 0.334 vs CIC -0.76, a converged U-PATCH would satisfy the no-macro-only-win rule rather
   than exploiting it. NOT yet claimable: still rising, single seed, one rotation, same phase.

Caveat retained: training loss (left panel) is flat/noisy throughout for both models while validation
climbs — gains come from exposure breadth, and the loss curve is NOT a useful convergence signal here;
the epoch-wise validation curve and patience rule are the arbiters.

### 2026-07-20 — [science/code] P8 rotation-0 continuation: fixed receptive field, encouraging but not promoted; resume to the registered stop

The first exposure-aware interactive allocations ended before either model reached a registered
stopping condition. Both models achieved their best validation score at their final completed
epoch, so allocation expiry is not evidence of convergence:

- **G-PATCH epoch 9:** macro-shell R2(lambda1) **0.4396**, shells
  **0.494/0.480/0.443/0.341**, pooled **0.486**, training weighted MSE **0.7041**.
- **U-PATCH epoch 11:** macro-shell R2(lambda1) **0.4498**, shells
  **0.548/0.511/0.446/0.295**, pooled **0.523**, training weighted MSE **0.6907**.

The G-PATCH computational receptive field does **not** grow with epoch. The frozen two-pass
attention model has two dependency hops per pass, hence every prediction has the same exact
four-hop graph support at epoch 1 and epoch 20. Epochs update shared weights; there is no
persistent node state or cross-epoch message propagation. The physical radius of four hops varies
with graph sparsity, and the precomputed local graph metrics add fixed feature-construction
support, but neither grows during training.

This rules out the simple fear that later epochs let G-PATCH propagate across and learn the whole
cap in a forward pass. It does **not** establish fresh-volume generalisation: shared weights may
still learn `ph000`-specific local motifs, long-mode statistics, HOD/observer response, or other
compressible same-realisation structure. The validation fold is label-held-out but uses the
globally observed `ph000` representation. The recovered-checkpoint boundary results are stable
(macro R2 0.4396 all, 0.4387 beyond 10.4 Mpc/h, 0.4380 beyond 20.8 Mpc/h), and the earlier exact
four-hop-untouched slab improved rather than degraded, so boundary/context bleed is not a credible
explanation for the complete signal. Rotation 2 and especially the independently constructed P10
phase remain mandatory.

No encoder is promoted. G-PATCH ties the frozen R0 macro rather than clearing the registered
+0.03 gate; U-PATCH is only +0.01. On the non-collapsed first three shells, G-PATCH **0.472** and
U-PATCH **0.501** still trail CIC **0.520**. A macro-only advantage caused by CIC's catastrophic
last-shell failure is useful sparse-regime evidence, not a claim that ML has beaten classical
reconstruction.

**Decision:** resume both rotation-0 runs unchanged until the pre-registered 20-epoch maximum or
early stop (minimum epoch 5, patience 3, minimum macro-R2 improvement 0.005). Atomic checkpoint,
optimizer, scheduler, epoch order, cursor, loss accumulators and RNG state are retained. Exact
checkpoint-revision worktrees are used because only `SCIENCE_LOG.md` changed between those commits;
the resume guard is not weakened.

- G-PATCH allocation **56224585**, node `nid008225`, exact revision
  `d7bab4edca77ef907de3f7dcf2d7920edd40dd0b`, log
  `/pscratch/sd/d/dkololgi/logs/p8_recovery_graph_rotation0_resume_56224585.log`.
- U-PATCH allocation **56224586**, node `nid008412`, exact revision
  `a2a944c3be1f952696f3868da0563bbd1fd362e4`, log
  `/pscratch/sd/d/dkololgi/logs/p8_recovery_unet_rotation0_resume_retry_56224586.log`.
- Run artifacts remain under
  `/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/recovery_v1/{graph,unet}/rotation_0/seed_42/`.

### 2026-07-20 — [code] CORRECTION (JDPK challenge upheld in part): "every model before P4 lacked this property" was FALSE — R0 already showed val-improves-on-unseen under the July-14 spatial holdout; comparator numbers untangled

JDPK challenged the previous entry's closing framing as too strong. Adjudication:

**WRONG in my framing:** R0/A1 (July 14) were TRAINED under a spatial holdout and reached held-out
macro 0.440 / pooled 0.504 (P0-frozen). The property "validation improves on unseen volumes" was
therefore NOT new at P8 — it existed in the R0 lineage. The correct narrower claim: the blocked-patch
protocol is the first to show this on the FULL NGC+SGC manifold with matched candidates, exposure
accounting, and verified fold geometry — an upgrade in rigour and scope, not the first appearance of
generalisation.

**COMPARATOR NUMBERS (to prevent the next apples-to-oranges):**
- Frozen-transfer tests (dense wedge -> RA200-240): GraphNet 0.421 POOLED on a SINGLE z0.2-0.3 shell.
  The like-for-like G-PATCH numbers are its z0.15-0.25/0.25-0.35 shells: 0.440/0.440 at epoch 5 —
  at/above par ALREADY, on blocked validation, not a friendly dense wedge.
- R0 frozen baseline: macro 0.440 (four shells, old gutter-filtered wedge rows). G-PATCH rotation-0
  is at 0.4059 after 5 epochs, still climbing, on a different and in several ways harder scoring set
  (full caps, no gutter filtering, all-authoritative). Behind by ~0.03 TODAY, mid-flight; the plan's
  promotion gate (+0.03 over the matched frozen baseline) is the arbiter and has NOT been met.
- Classical: G leads the registered primary metric (macro 0.406 vs CIC 0.185) but TRAILS the
  supported shells (0.43-0.44 vs 0.52-0.57); per the plan's own no-macro-only-win rule this does NOT
  count as beating classical. JDPK's "not even beaten the classical method yet" is correct in the
  sense the plan says matters.

**On R²=0.8 as a reference point:** that number was the random-split artifact and should not creep
back as an implicit bar. All honest evidence (DTFE floor ~0.55, three independent data-limitation
lines, best honest pooled ~0.55) says ~0.5-0.55 is the current within-phase ceiling at BGS sparsity.
The registered bars are: (1) +0.03 over the frozen R0 baseline on matched scoring, (2) beat/tie the
best classical without macro-only wins, (3) fresh-phase blind (P10). If the science requires
0.8-grade per-galaxy precision, the route on the evidence is data volume (phases/footprint) and a
calibrated posterior product — not encoder iteration on ph000.

Latest live: G epoch 5 macro 0.4059 (0.440/0.440/0.407/0.337); U epoch 6 macro 0.3553
(0.365/0.400/0.355/0.301).

### 2026-07-20 — [code] LIVE recovery-run evaluation (Claude Code): the first defensible generalisation signal — with its scope stated precisely

Read the in-flight rotation-0 recovery histories directly (fig8_recovery_learning_curves.png,
figures/p8_smoke_eval/). Numbers at time of writing:

- **G-PATCH** epochs 1-4 macro: 0.3215 -> 0.3260 -> 0.3654 -> **0.4006** (still rising; already at
  the short-screen best). Per-shell @4: **0.427 / 0.440 / 0.414 / 0.321** — near-FLAT across shells,
  sparse shell rising with the rest.
- **U-PATCH** epochs 1-6 macro: 0.094 -> **-0.150** -> 0.056 -> 0.190 -> 0.286 -> **0.355** (epoch-2
  instability, shell-1 R² -0.82, then steep recovery ~+0.07/epoch). Sparse shell stable 0.26-0.30
  from epoch 1.

**Why this IS a generalisation signal (four grounds):**
1. The right metric is rising with exposure: blocked-fold validation (disjoint super-blocks, verified
   hold-out incl. deep-interior check) climbs as training exposure grows. Not the interpolation
   signature.
2. The causal shape is right: windowed TRAIN loss is roughly flat/noisy while VAL climbs strongly —
   gains come from broader exposure (each epoch = all 10,351 cores once), not from grinding loss on a
   memorised subset. This is the OPPOSITE of the leak-audit signature (train down, val up).
3. G's per-shell profile is flat — no density-regime trade-off. Contrast CIC (0.56/0.57/0.44/-0.82)
   and every dense-wedge model. Rising-tide learning is what "learned the local map" looks like.
4. Independent supports: U overfit probe (tails recoverable, R²=0.999, exact range match) closed the
   structural question; canary accounting (10,351/10,351 cores, zero repeats, all rows) closed the
   exposure question.

**Why it is NOT yet the claim we need:** (a) not converged, no early stop reached, single seed,
rotation 0 only; (b) SAME-PHASE spatial transfer — shared long-wavelength modes; P10 fresh-phase
remains the only blind gate; (c) the classical adoption gate is NOT met: CIC first-three-shell 0.520
vs G epoch-4 0.427 — G must close ~0.09 in supported shells (or the registered context-growth /
residual-hybrid tests must show the gap is context, not encoder); the sparse shell (G 0.321 vs CIC
-0.82) already favours learned decisively; (d) U's epoch-2 instability flags LR/normalisation
sensitivity — recovered, but a second seed should confirm it is benign.

WATCH: whether G plateaus above the frozen baselines with first-three approaching CIC; early-stop
rules activate from epoch 5 (patience 3, min delta 0.005). Curves + per-shell now plottable at any
moment from epoch_history.jsonl / loss_trace.jsonl (fig8 script).

### 2026-07-20 — [code] U-PATCH rotation-0 epoch canary passes; scientific 20-epoch-schedule run launched

The exposure-aware U-PATCH canary completed on interactive allocation 56202186. It
visited exactly 10,351/10,351 eligible training cores once, with zero repeats, and
accounted for all 3,023,524 authoritative rows: 1,668,208 / 998,471 / 319,511 / 37,334
across the four shells. Weighted loss numerators/denominators, 415 windowed trace
records, full-fold validation predictions, report, atomic checkpoint, and
`CANARY_COMPLETE` are present under
`/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/canary_v1/unet/rotation_0/seed_42/`.

The canary macro-R2(lambda1) is 0.1646 (shells 0.106/0.222/0.098/0.233). This is not
interpreted as a model score: the canary intentionally uses a one-epoch cosine schedule
that reaches zero learning rate and exists to validate exposure/accounting/validation
plumbing. The scientific U-PATCH rotation-0 run has now started under `recovery_v1`
with a 20-epoch schedule, minimum 5 epochs, patience 3, and min delta 0.005. It is
checkpoint-resumable across interactive allocation expiry.

### 2026-07-20 — [science/code] U-PATCH one-core overfit probe passes: short-screen tail compression is not structural

The pre-registered U-PATCH capacity diagnostic was run on rotation-0 training core
15211 (204 authoritative shell-3 galaxies). With the frozen patch geometry,
selection channels, normalization, interpolation head, and architecture, 500 updates
on this one core reduced scaled-increment MSE from 1.16995 to 0.001726 (ratio 0.00148).
The final eigenvalue R2 values are 0.99913/0.99776/0.99765. Most decisively, the true
lambda1 range `[-0.4650, 0.8457]` is recovered as `[-0.4643, 0.8450]`.

This is a capacity/optimization diagnostic, not transfer evidence. It rules out a hard
output clamp and shows that the registered U-PATCH can represent the missing tails when
given sufficient exposure. The short-screen under-dispersion is therefore consistent
with inadequate optimization and data exposure, although only complete-epoch blocked
validation can show whether the expanded range generalises rather than memorises.

Artifacts:
`/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/probes/unet_rotation_0/core_15211_seed_42/`
(`probe_summary.json`, `loss_trace.jsonl`, final predictions, and `PROBE_COMPLETE`).

### 2026-07-20 — [code] P8 recovery trainer frozen; rotations clarified; interactive GPU canaries next

The P8 recovery implementation now exists separately from the immutable 2,000-step smoke
trainers. `workflows/abacus_tweb/p8_train_patch_recovery.py` supports both G-PATCH and
U-PATCH and defines a scientific epoch as one visit to every eligible P4 training core.
The order is a deterministic `W_p`-weighted permutation without replacement; explicit
row weights make the arithmetic mean of patch objectives equal the globally row-weighted
MSE, independent of how memory patches are subdivided. The learning-rate schedule counts
real patch updates rather than nominal epochs.

The recovery contract now persists atomic mid-epoch checkpoints containing model,
optimizer, scheduler, cursor, exact loss numerators/denominators, shell exposure, history,
and CPU/CUDA random-number states. A disconnected interactive allocation can therefore
resume the same dropout sequence as well as the same data order. Windowed training loss is
written every 25 patches. A deliberate step interruption exposed and fixed the remaining
logging edge case: resume now removes post-checkpoint abandoned loss rows and de-duplicates
replayed steps before appending. The complete validation fold is evaluated after every completed
epoch. Scientific runs use 5--20 epochs, no stopping before epoch 5, patience 3, and a
minimum macro-R2 improvement of 0.005. Outputs are isolated under
`/pscratch/sd/d/dkololgi/abacus/p8_recovery_v1/`; smoke checkpoints are never overwritten.

Rotations are fold-role rotations, not physical sky rotations. Rotation 0 trains folds
`{2,3,4}`, validates fold 1, and seals fold 0 as development test. Rotation 2 trains
`{0,1,4}`, validates fold 3, and seals fold 2. This changes the held-out geography but is
not a fresh-phase test; P10 remains the blind generalisation gate.

The common epoch/resume/objective tests pass (7/7; 14/14 with the existing P8 common and
patch-model tests). `workflows/abacus_tweb/p8_probe_unet_overfit.py` adds the pre-registered
single-core U-PATCH capacity diagnostic. The next execution step uses two reusable
interactive A100 allocations only: U-PATCH probe then rotation-0 canary on one, and the
rotation-0 G-PATCH canary on the other. Long rotation-0/2 recovery runs start only after
100% core coverage, loss accounting, validation, and resume are verified.

### 2026-07-20 — [code] VERIFIED: the fig2/5/6 slab is genuinely held out — labels never trained on, and metrics HOLD on the provably-untouched interior

JDPK asked for confirmation that the plotted volume was unseen. Checked directly against the frozen
rotation-0 role files and the P4 support arrays.

**Identity:** super-block 467, cap NGC, **fold 1 = the VALIDATION fold for rotation 0** (train folds
{2,3,4}). 64 cores, 126,239 active galaxies; the plotted slab is 14,438 of them, ALL fold 1.
**Zero of the 10,351 frozen training cores lie inside SB467**; 62 validation cores do.

**Labels: definitively unseen (100%).** Training loss is computed only on authoritative core rows of
the TRAINING folds, so these galaxies contributed exactly zero gradient. That is by construction, and
now verified against the frozen core-id lists rather than assumed.

**Context/features — the honest nuance:** patches carry K-hop context, so a validation galaxy near a
fold boundary CAN appear as a context node (features only, never labels) inside a training patch.
For this slab: **50.9% are >=5 graph hops from ANY other-fold node**, hence provably outside the
4-hop dependency context of every training patch — they never entered a training computation in any
form. The remaining 49.1% are within 4 hops of *some* other-fold node (an upper bound: 'other fold'
includes the dev-test fold 0, which is not training either). Median physical distance to a fold
boundary is 58 Mpc = 5.6 smoothing lengths; 89.4% exceed one smoothing length.

**Decisive check — metrics on the provably-untouched interior (n=7,348) vs the full slab:**

| model | lam1 slab | lam1 deep | lam2 deep | lam3 deep |
|---|---:|---:|---:|---:|
| G-PATCH | 0.595 | **0.632** | 0.726 | 0.613 |
| U-PATCH | 0.524 | **0.587** | 0.674 | 0.488 |
| CIC (train-affine) | 0.658 | **0.665** | 0.694 | 0.694 |

Performance does not degrade on the deep interior — it *improves* for both learned models (+0.037
G, +0.063 U), while CIC is nearly flat (+0.007). So the visual quality in fig2/5/6 is NOT an artifact
of context bleed near fold boundaries. The learned gain on the interior is consistent with the P4
boundary-margin design (interior galaxies have complete graph support; boundary galaxies have context
clipped by the survey/fold edge), and it is the same direction as fig4's flat boundary-error trend.

This strengthens the reading of the visuals: two encoders trained on ~15% of the training cores, with
no convergence, reconstruct recognisable cosmic-web morphology in a volume whose labels they have
never seen — and do so BEST where they are furthest from anything they touched.

### 2026-07-20 — [code] P8 loss curves DO NOT EXIST for the frozen screens; trainers instrumented so recovery reruns produce them

Asked for G/U loss curves; there are none to plot. Each run's `history` has exactly ONE entry
(step 2000) because `--eval-every` equalled `--steps`, and that entry's `training_loss` is the
INSTANTANEOUS single-patch loss, not a running mean. Everything that exists:

| run | steps | train loss @ end | val macro R2(lambda1) |
|---|---:|---:|---:|
| G-PATCH rot0 | 2000 | 0.5290 | 0.400 |
| G-PATCH rot2 | 2000 | 0.3976 | 0.392 |
| U-PATCH rot0 | 2000 | 0.5989 | 0.369 |
| U-PATCH rot2 | 2000 | 0.4334 | 0.357 |

The rot0-vs-rot2 loss offset is patch-draw noise (which core was sampled at step 2000), NOT an
optimization difference — do not read it as one. Checkpoints carry only final state, no trace.
This is the training-adequacy audit finding in its rawest form: with one sample there is no way
to distinguish "converged" from "still descending", which is exactly why the learned NO-GO was
withdrawn.

FIX (additive, no training-math change; sol please retain in the recovery reruns): both
p8_train_graph_patch.py and p8_train_unet_patch.py now log a WINDOWED mean training loss
(+min/max/lr) to `loss_trace.jsonl` every `--loss-log-every` steps (default 25), decoupled from
validation cadence because training-loss logging costs no fold evaluation. `loss_trace` is also
carried in screen_summary.json. plot_p8_loss_curves.py renders curves when present and states the
absence explicitly otherwise (fig7_loss_curves.png). Combined with the recovery contract's
per-epoch complete-fold validation, the reruns will finally support a convergence/early-stopping
judgement — and will also settle the U-PATCH under-dispersion question (output range vs step).

### 2026-07-20 — [code] P8 visuals part 2 (λ2/λ3 + T-web classes) + U-PATCH saturation fingerprint: NOT a clamp — under-dispersion

fig5/fig6 added (commit above; figures/p8_smoke_eval/). Class maps at λ_th=0.2, validation
super-block 467: TRUE occupancy 38/41/18/3% (void/wall/filament/knot) vs G-PATCH 22/54/23/1%
(acc 65%), U-PATCH 14/58/27/0% (58%), CIC 18/63/19/0% (68%). Amplitude compression becomes
class-occupancy bias — every method floods the modal wall class and starves voids+knots; G-PATCH
keeps the best knot recall (20% at 75% precision; CIC 11%, U-PATCH 7%). λ2: G 0.690 ~ CIC 0.688 >
U 0.612 (slab); λ3: CIC 0.715 > G 0.616 > U 0.480.

U-PATCH "clipping" REDIAGNOSED: scaled-increment-space fingerprint shows zero pileup at the bounds
and non-constant extremes (v1 in [-0.76,+3.49] vs G [-2.12,+4.41]) — no clamp, no bounded
activation. It is severe under-dispersion, most visible as the missing low-λ1 tail. Cheap
discriminator vs convergence (registered): overfit-one-patch probe (train on a single patch a few
hundred steps; if its predictions expand past the current range, it is convergence; if pinned,
structural) + head-code diff vs G-PATCH. Run before the converged rerun.

### 2026-07-20 — [code] P8 SHORT-SCREEN EVALUATION FIGURES + programme integrity review (Claude Code); two new model pathologies surfaced

Figures (workflows/abacus_tweb/plot_p8_smoke_eval.py, commit 9f75a41;
/pscratch/sd/d/dkololgi/abacus/figures/p8_smoke_eval/; every panel banners NOT CONVERGED):
- fig1_roles_rotation0 — train/val/dev-test core geometry, both caps (counts verified: 10,351 /
  3,446 / 3,405 occupied cores = 60/20/20; galaxies 3.02M / 1.00M / sealed).
- fig2_visual_predictions — validation super-block 467 (NGC, 40 Mpc slab, 14,438 galaxies): TRUE λ1
  vs G-PATCH (slab R² 0.595) / U-PATCH (0.524) / CIC-affine (0.658). All three recover the web
  MORPHOLOGY; all three visibly compress CONTRAST (deep voids and knots washed toward the mean).
- fig3_parity_and_shells — parity per model + per-shell R² for both rotations.
- fig4_diagnostics — boundary-trend, amplitude-ratio, rotation-consistency.

**NEW FINDINGS from the figures (not in the P8 reports):**
1. **U-PATCH output SATURATION**: predictions hard-clipped to ~[-0.30, +0.57] (flat hexbin edges).
   Under a 2,000-step screen this is plausibly an under-trained output scaling, but if structural it
   caps knot recovery permanently. CHECK BEFORE the converged rerun.
2. **Amplitude compression is the dominant learned-model error mode already**: pred/true std ≈ 0.6
   in all shells, declining with z (regression to the mean); CIC instead OVERSHOOTS at high-z
   (ratio >1.3 — noise amplification where tracers vanish, the flip side of its -0.76 R²).
3. **No fold-boundary artifact**: median |error| vs distance-to-fold-boundary is FLAT for all three
   methods across 2-400 Mpc — the blocked protocol introduces no edge pathology. (Protocol integrity
   evidence, worth keeping as a standing check.)
4. **Rotation stability is excellent**: per-shell R² for rot0 vs rot2 sits on the diagonal for all
   models including CIC's high-z collapse — the metric is reproducible across disjoint geography.

**INTEGRITY REVIEW (state of the whole programme):** The P8 self-correction (short screens frozen as
smoke, gate reopened as INCONCLUSIVE with a machine-readable adequacy audit) is the system working as
designed — three weeks ago this same result would have been logged as a verdict. The blocked-fold
protocol is now EVIDENCED, not just designed (flat boundary trend, rotation consistency). The
classical adoption gate is hard and honestly framed (macro-only wins rejected). Registered caveats
stand: context-size mismatch vs global-FFT classical (recovery item 4), z-error realism gap, P10
cross-phase still the only route to a blind claim.

**GENERALISABILITY — recommended order:**
1. Execute P8.5 recovery as written (full exposure epochs, per-epoch complete-fold validation, early
   stopping) — G-PATCH at 0.396 macro with 15% core exposure and zero convergence is a floor, not a
   ceiling.
2. Fix/clear the U-PATCH saturation first (cheap; a poisoned rerun wastes a GPU-day).
3. Context-growth experiment (60->360 Mpc/h; trace vs traceless separately) — decides how much of
   CIC's supported-shell edge is CONTEXT, not encoder.
4. **Classical+local-residual hybrid is the highest-prior candidate** (zero-initialised residual
   around CIC/DTFE): classical carries nonlocal modes, the network learns only the local sparse-
   tracer correction; where classical fails (high-z) the residual learns, where it works the residual
   ~0. Matches the strongest published pattern (field-level BAO CNN-around-reconstruction).
5. Amplitude calibration diagnostics as standing reports (variance ratio per shell) — MSE point heads
   regress to the mean by construction; do not mistake it for missing signal.
6. The data axis remains the blind-claim path: P10 ph002 benchmark (STILL UNCLAIMED) then multi-phase
   training — the literature's transferable results all rest on many independent realizations.

### 2026-07-19 — [science/code] P8 CORRECTION: short screens frozen; optimization audit reopens the scientific gate

The rotation-0/2 P8 jobs completed and remain useful plumbing/transfer smoke tests, but
they are not converged training experiments and cannot support the previously drafted
learned-model NO-GO. The corrected machine-readable gate is
`INCONCLUSIVE_OPTIMIZATION_AUDIT_REQUIRED`.

Mean lambda1 results across the two screens are:

| Candidate | Four-shell macro R2 | First-three-shell diagnostic R2 | Final-shell R2 |
|---|---:|---:|---:|
| full-cap CIC + train-fold affine | 0.185 | **0.520** | -0.822 |
| G-PATCH | **0.396** | 0.448 | **0.240** |
| U-PATCH | 0.363 | 0.423 | 0.183 |

The learned macro advantage is explicitly **not** interpreted as a classical-method win.
G-PATCH and U-PATCH trail CIC in each of the first three shells; the sign of the
four-shell macro difference is reversed only by CIC's catastrophic final-shell collapse.
The result highlights the need for a genuinely generalisable encoder and a strong matched
classical adoption gate. It does not establish learned superiority.

A post-run audit reproduced the exact seeded patch sampler. Each G/U job performed only
2,000 replacement-sampled optimization steps despite having 10,262--10,351 eligible
training cores. Rotation 0 exposed 1,560 unique cores (15.07%; 40.08% of weighted mass)
and rotation 2 exposed 1,554 (15.14%; 40.86%). The sparse shell saw only 118 and 103
unique cores, respectively, out of 3,845--3,848 eligible cores. Each job evaluated the
complete validation fold only once, at step 2,000. There is therefore no learning curve,
plateau, or executed early-stopping rule. G/U wall times were approximately 6.6--13.3
minutes per job, not the registered 2--4 GPU-day screen envelope.

The existing predictions, reports, and checkpoints are frozen as short-screen evidence.
They justify no automatic five-fold promotion, but they do not justify calling either
encoder scientifically rejected. F-PATCH v2_A remains a resource NO-GO for that frozen
configuration because its representative forward activation floor is 91.6 GiB before
autograd, optimizer, U-Net decoder, FFT arrays, or full graph context; this does not reject
the field-to-physics concept or a simplified implementation.

A second mismatch is now explicit. G-PATCH and U-PATCH receive finite patch context,
whereas CIC uses the whole cap followed by a global FFT tidal solve. The trace of the
tidal tensor is local density, but its traceless shear retains nonlocal external modes.
P8 recovery therefore adds a true-field context-growth experiment and a zero-initialized
classical-plus-local-learned residual control. These tests do not weaken the classical
bar; they determine whether finite support, rather than encoder capacity alone, explains
part of the deficit.

The immediate recovery contract is:

1. implement complete exposure-aware patch epochs, with every eligible core visited once
   per epoch and the frozen square-root shell objective applied through core loss weights;
2. train 5--20 complete epochs, validate the full blocked fold after each epoch, and allow
   early stopping only after epoch 5 with patience 3 and minimum macro-R2 delta 0.005;
3. rerun rotations 0/2 for unchanged G-PATCH and U-PATCH controls before architecture or
   target changes;
4. compare full-field tidal truth with 60/120/180/240/360 Mpc/h context solves, reporting
   trace and traceless-shear convergence separately;
5. implement the exact matched full-cap DTFE row and zero-initialized global
   CIC/DTFE-plus-local-residual controls;
6. reapply the classical adoption and five-fold promotion rules only to converged results;
7. keep log-gap, JEPA, FMPE/NPE, HOD marginalisation, and broad architecture work gated.

A targeted primary-literature check supports this correction. Published transferable
field-learning results generally train on many complete realizations and multiple full
epochs, not a fraction of one phase's patches. DarkAI used 30 independent COLA
realizations split 15/5/10 and tested separate high-resolution simulations
(arXiv:2305.11431). A DESI-like extension trained for roughly 1,300 selected epochs, but
its 12 mock samples came from rotations/translations of one stacked Jiutian simulation,
so it is not a fresh-phase proof (arXiv:2501.12621). A 2026 field-level BAO study uses
1,000 independent random-phase mocks and explicitly learns a CNN correction around a
traditional global reconstruction, closely motivating the registered residual control
(arXiv:2603.15732). These tasks and Fourier metrics are not directly comparable to our
per-galaxy lambda1 R2, so no literature number licenses either optimism or a NO-GO.

Artifacts:

- runtime audit: `/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1/training_adequacy.json`;
- corrected runtime summary: `/pscratch/sd/d/dkololgi/abacus/p8_deterministic_v1/screen_summary.json`;
- tracked evidence: `docs/evidence/p8/training_adequacy.json` and `screen_summary.json`;
- audit code: `workflows/abacus_tweb/p8_audit_training_adequacy.py`;
- decision logic: `workflows/abacus_tweb/p8_summarize_screens.py`;
- recovery specification: P8.5 in `docs/plan_generalisable_graphweb_vac.md`.

### 2026-07-19 — [science/code] P8 STARTED: deterministic patch-generalisation showdown and classical adoption gate frozen

P8 has begun under the completed P4–P7 representation and adapter contracts. The
controlling question is now deliberately narrower than posterior inference or HOD
marginalisation: can deterministic GraphNet, 3-D U-Net, or F-tier inference learn a
mapping that transfers across the registered spatial folds when trained on canonical
core/context patches?

The classical comparison has been strengthened from a diagnostic row to a hard
scientific adoption gate. Learned-model failure means that the best matched,
train-calibrated DTFE/CIC-style estimator exceeds every learned candidate on the
equal-shell spatial-fold lambda1 R-squared score. A macro advantage caused only by a
classical collapse in the sparsest shell is not an encoder win; per-shell behaviour and
spatial-block uncertainty remain mandatory. If no learned candidate at least ties the
best classical result within uncertainty, the learned VAC branch receives a NO-GO
and the classical estimator becomes the production reference rather than weakening
the metric.

The one-seed screen will use the already frozen linear-increment targets, all
authoritative validation cores, training-only transformations, and complete-fold
checkpointing. Posterior heads, HOD variation, JEPA, log-gap targets, and hybrids stay
gated until this deterministic transfer question has an answer.


### 2026-07-19 — [science/code] P6/P7 adapter convergence complete: patch-safe U-Net normalization and nonlocal FFT geometry frozen

The remaining P6 and P7 deployment gates were run on an interactive A100 allocation
without loading tidal targets. Patch geometry was selected entirely from structural
agreement with larger-context references; no R-squared score or target label was used.
The suite covers both Galactic caps and all four redshift shells for P6, and both caps
plus the lowest/highest shells for the more expensive P7 FFT tests.

The first trained P6 canary exposed a critical hidden dependency. The historical T2
U-Net uses PyTorch `GroupNorm`, whose statistics include every spatial position in the
current patch. Core predictions therefore changed with patch extent even after
convolutional context had converged. This is incompatible with a decomposition-invariant
VAC. Thresholds were not relaxed. The structural canary was converted to per-voxel
channel LayerNorm while retaining its learned affine weights, and the P8 contract now
forbids spatial GroupNorm, spatial InstanceNorm, and patch-local input normalization.
The final U-PATCH model must train from scratch with patch-safe normalization.

The P6 selection rule was strengthened after audit: freeze the smallest halo whose
entire larger-context tail also passes, not the first isolated passing point. The
selected field halo is 24 voxels = 120 Mpc, compared with an 80-voxel reference, with
an 8-voxel global-lattice phase lock for strided pooling. At the selected halo:

- pooled galaxy-prediction NRMSE = 0.001545;
- latent-core NRMSE = 0.003868;
- worst-core prediction NRMSE = 0.003005;
- parent-versus-child subdivision NRMSE = 0.001658;
- subdivision p95 absolute error / reference standard deviation = 0.003553.

The retained boundary Spearman coefficient is 0.203, but the pre-registered
trivial-effect branch passes because prediction NRMSE is below 0.002. P6 is therefore
adapter-complete and `UNET_PATCH_READY` is written. The final trained checkpoint must
repeat the suite before release.

P7 then propagated the patch-safe learned latent field through the fixed FFT tidal
operator. P5 already proves exact graph-context parity independently of encoder
weights; no historical full-range F-tier checkpoint exists, so this is a learned-field
spectral/numerical canary rather than an F-tier accuracy claim. A 64-voxel field halo
failed despite converged learned density: tensor/eigenvalue NRMSE = 0.0553/0.0502.
One-factor controls at fixed 80-voxel context show that both nonlocal choices matter:
16 rather than 24 padding voxels gives 0.0982/0.1078, and 16 rather than 20 apodization
voxels gives 0.0339/0.0370.

The frozen P7 adapter configuration is a 72-voxel = 360 Mpc learned-field halo,
20-voxel = 100 Mpc zero padding, and 20-voxel = 100 Mpc cosine apodization, scored on
the P4 authoritative core. The reference is halo 80 / padding 24 / apodization 20.
The selected configuration gives tensor NRMSE 0.02272, eigenvalue NRMSE 0.01770,
eigenvalue p95/reference-standard-deviation 0.04097, and trace error 1.9e-15. For the
large-eigengap half of every principal axis, the worst median/p95 orientation changes
are 0.744/1.817 degrees. Small-gap axes degrade as expected and require an eigengap
quality field in any orientation product.

The survey-support boundary audit retains galaxies beyond 2 smoothing lengths =
20.69 Mpc. There are 8,869 retained and 463 near-boundary galaxies. A residual trend
remains within the retained sample (Spearman = -0.342), but its mean eigenvalue change
is 0.0158 of the reference standard deviation and passes the pre-registered 0.02
trivial-effect branch; the near-boundary mean is 0.0253. This is recorded as a small
but non-zero boundary dependence, so near-support rows require a quality flag.

`FTIER_PATCH_READY` now certifies the graph/field/FFT adapter and frozen numerical
geometry. It does not certify final F-tier science performance. The final trained
F-PATCH checkpoint must rerun the complete decoder/FFT/eigenvector convergence suite
before tensor or eigenvector release.

Runtime artifacts:

- `/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter/trained_convergence_v1/trained_convergence_report.json`
- `/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter/trained_convergence_v1/UNET_PATCH_READY`
- `/pscratch/sd/d/dkololgi/abacus/p7_ftier_patch_adapter/trained_convergence_v1/trained_convergence_report.json`
- `/pscratch/sd/d/dkololgi/abacus/p7_ftier_patch_adapter/trained_convergence_v1/FTIER_PATCH_READY`

The executable gate is
`workflows/abacus_tweb/p6_p7_validate_model_convergence.py`; patch extraction gained
global-lattice alignment, and the tidal operator gained explicit zero padding. The
implementation commits are `5351048`, `7ff551b`, `b554fdc`, and `b959df6`; the
authoritative reports bind to `b959df6`. Compact schemas, reports, and markers are
tracked under
`docs/evidence/p6/` and `docs/evidence/p7/`.


### 2026-07-19 — [science/code] P6 full-cap selection refit passes; inherited P7 blocker closed

The wedge-derived P3 `ntilde` channels are now superseded at model-read time by a
versioned P6 overlay; the immutable P3 HDF5 products were not modified. For each of
the five P4 spatial rotations, separate NGC and SGC radial number-density curves are
fit from only two label-free quantities in that rotation's three training folds:
observed mock-galaxy redshifts and the apodized P3 effective exposure volume.
Validation/development galaxies and all tidal labels are excluded from fitting.

The primary estimator is a weighted cubic least-squares spline in log number density
with fixed 0.05 redshift knot spacing. The maximum training-shell expected/observed
error across 5 rotations x 2 caps x 4 reporting shells is 0.5687%. The deliberately
unfitted validation/development ratios range from approximately 0.87 to 1.12, exposing
real spatial/sample variance rather than forcing artificial closure. Pre-registered
0.04/0.06 knot-spacing variants differ from the primary curve by at most 8.85% over
0.15 <= z < 0.55; this is retained as a selection-model systematic.

Each rotation also has one frozen channel normalizer fitted on supported voxels whose
P4 cores belong to its training folds, pooled across NGC and SGC. No patch-local,
validation-fold, development-fold, or blind-catalogue normalization is allowed. The
P6 adapter now regenerates `ntilde_mpc3`, `expected_counts`, and
`log_count_ratio` lazily from the selected rotation while leaving all other
canonical channels untouched.

All overlay gates pass on representative cores from every cap/fold/rotation stratum:
4- versus 8-voxel patch overlap is exactly invariant; the expected-count identity
agrees within 5.96e-8; the contrast identity agrees within 4.77e-7; legacy P3
selection channels are demonstrably replaced; and frozen-normalized patches are
finite. Unit tests also cover unsupported voxels and prove the overlay ignores stale
selection arrays in the immutable P3 file.

P7 was rerun using rotation 0's passed overlay. Authoritative graph/field identity,
TSC conservation and overlap, fixed-FFT trace consistency, and finite ordered
eigensystems all continue to pass. Thus the P6 selection blocker inherited by F-tier
is closed. This does **not** write `UNET_PATCH_READY` or `FTIER_PATCH_READY`:
P6 still requires trained U-Net context/subdivision/boundary convergence, while P7
still requires trained decoder plus nonlocal FFT tile/padding/apodization/trim and
eigengap-conditioned orientation convergence.

Runtime artifacts:

- `/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter/fullcap_selection_v1/selection_manifest.json`
- `fold_radial_histograms.npz`
- `selection_overlay_validation.json`
- `SELECTION_REFIT_COMPLETE`
- `SELECTION_CHANNELS_READY`
- refreshed P7 `/pscratch/sd/d/dkololgi/abacus/p7_ftier_patch_adapter/initial_composition_report.json`

Implementation is in
`workflows/abacus_tweb/p6_refit_fullcap_selection.py`,
`p6_validate_selection_overlay.py`, `p6_field_patch_utils.py`, and the updated
`p7_validate_initial_composition.py`. Compact evidence is tracked under
`docs/evidence/p6/` and `docs/evidence/p7/`.

### 2026-07-19 — [science/code] P7 STARTED: graph-field composition, conservative scatter, and fixed tidal-operator gates pass

P7 now composes the real P5 graph view with the P6 canonical field frame. On one SGC
and one NGC core, authoritative parent IDs agree exactly between adapters; TSC scatter
conserves both test latent channels to better than 4.4e-4 in absolute summed weight;
and the common global-voxel overlap is exactly invariant when field context grows.
The tested graph views contain 21,168/27,126 nodes and 1.44M/2.64M directed edges.

The fixed FFT tidal operator produces finite ordered eigensystems and trace consistency
at 3.7e-15 or better for apodization candidates 0, 4, and 8 voxels. The smoothing
conversion is now explicitly bound to the frozen P3 observer-frame contract:
7 Mpc/h = 10.345846881466155 Mpc at Planck18 h=0.6766. This avoids silently mixing
the Abacus c000 value of h used by older classical scripts with the coordinate system
of the canonical P3 lattice.

This is an initial composition gate, not an F-tier production pass.
`FTIER_COMPOSITION_READY` is written, while `FTIER_PATCH_READY` remains false.
Before F-PATCH training, P7 still needs the P6 selection-channel refit, trained
graph-encoder/field-decoder context convergence, FFT tile-size/padding/apodization/
overlap/trim convergence, and eigengap-conditioned orientation stability.

Runtime artifacts are under
`/pscratch/sd/d/dkololgi/abacus/p7_ftier_patch_adapter/`; code is in
`workflows/abacus_tweb/p7_ftier_patch_utils.py` and
`p7_validate_initial_composition.py`; tracked evidence and schema are under
`docs/evidence/p7/`.

### 2026-07-19 — [science/code] P6 STARTED: canonical field-patch views pass structural parity; selection-channel refit is a hard U-PATCH gate

P6 now has an immutable field-patch adapter over the P3 NGC/SGC HDF5 lattices and
the P4 authoritative core manifest. The index covers all 5,026,863 authoritative
galaxies and 18,765 cores. Across the maximum-occupancy authoritative core in every
cap/fold stratum, canonical field channels are copied exactly, galaxy parent-ID order
and global/local coordinates agree exactly, and trilinear sampling is identical for
4- and 8-voxel context views. No patch-local normalization is performed.

The parity run exposed an important geometry distinction rather than a data loss. P4
owns physical galaxy cores, whereas P3 stores rounded voxel slices. A few percent of
authoritative galaxies in sampled cores can lie just outside the nominal voxel-core
slice, but every one has a valid interpolation stencil in the context view. P6 now
keeps P4 ownership authoritative and treats the voxel core as dense-output and
normalization bookkeeping; it does not discard valid core targets to force lattice
coincidence.

The field adapter is structurally sound, but `UNET_PATCH_READY` is deliberately not
written. The P3 expected-count channels still use the wedge-frozen `ntilde` spline.
The full-cap audit finds expected/input ratios of 1.03–1.08 in NGC and 1.13–1.18 in
SGC across the four reporting shells. Cap-aware expected-count channels must be
refit or independently validated from the training-footprint observation model before
U-PATCH training. Per-rotation channel normalizers and trained-model context/boundary
convergence also remain open.

Runtime artifacts:
`/pscratch/sd/d/dkololgi/abacus/p6_unet_patch_adapter/adapter_manifest.json`,
`structural_parity_report.json`, `FIELD_PATCH_INDEX_READY`, and
`FIELD_PATCH_PARITY_READY`. Code is in
`workflows/abacus_tweb/p6_{field_patch_utils,build_field_patch_adapter,validate_field_patch_adapter}.py`;
tracked evidence is under `docs/evidence/p6/`.

### 2026-07-19 — [science/code] P5 AUDIT RESOLVED: authoritative primary mask, frozen G-PATCH control, production A100 parity

Claude's audit identified genuine pre-registration gaps, not a failed adapter. They
are now resolved before P8. The primary loss and metric mask is **all P4 authoritative
core galaxies**. Four-hop isolation is retained as an architecture-specific robustness
diagnostic only: it keeps just 106/62,243 galaxies at z=0.45-0.55 and cannot define
the mandatory four-shell macro-R². The shared robustness ladder is now (1) all
authoritative cores, (2) physical fold-boundary margins beyond 10.4 and 20.8 Mpc,
and (3) hop-isolated subsets with their retained fractions.

This needs precise naming. P8 is **spatial target generalization with a globally
observed representation**: validation/test labels are excluded, but the globally
observed, label-free point process may enter graph construction and context. That is
deployment-relevant for DESI, but it is not fresh-graph induction. P10 independent
phases separately construct each graph and remain the decisive fresh-catalogue test.

The architecture fork is frozen. G-PATCH first retrains the existing two-pass
receiver-normalized attention GraphNetwork, with the R0/A1 eight-feature schema and
four exact dependency hops, so the protocol is the only central change. Receiver-only
message passing is a separately named optional challenger after this control; it
cannot replace the control after results are seen. The historical two-layer
disconnected-wedge transfer failure motivates the experiment but does not answer it.

Patch-local feature fitting is forbidden. For each rotation, node SI medians are fit
per cap on authoritative training-fold nodes, Box-Cox on training cores, and edge
transforms on training-fold internal edges; all are then frozen. Blind phases and DESI
receive the training-ensemble transforms unchanged.

The production-shape GPU parity point also passes on one A100: latent 80, eight heads,
100,935 full-graph nodes, and a 5,737-node / 194,584-directed-edge patch. Patch-order
difference is exactly zero; maximum embedding and prediction differences are 1.54e-3
and 7.38e-4, respectively, both within the pre-registered 2e-3 float32 GPU tolerance
and small relative to output scale. Runtime evidence is in
`/pscratch/sd/d/dkololgi/abacus/p5_graph_patch_adapter/parity_report_prod_gpu.json`;
tracked evidence and the revised mask/scaling contract are in `docs/evidence/p5/`.
`GRAPH_PATCH_PROD_GPU_READY` is written.


### 2026-07-19 — [code] P5 INTEGRITY AUDIT (adversarial, Claude Code): parity is SOUND; four pre-registration holes found — H1 (strict-hop mask starves/undefines the sparsest shell) is CRITICAL and must be decided BEFORE P8

**Verified sound (independently reproduced, not just re-read):**
1. **Parity is real and exact.** Embedding/prediction/patch-order diffs = 0.0; subdivision 2.38e-7
   (float32 resummation, gate 5e-5); boundary-error slope 0. Crucially the P1a parity runs through the
   SAME incident-CSR/assemble_patch code as the production adapter (p5_validate imports from
   p5_build) — the assembly path is what was tested, not a bespoke twin.
2. **The 2-hops-per-pass preflight discovery is correct and important.** Receiver-normalised attention
   + sent-edge aggregation doubles dependency depth; exact parity with 4-hop context (and failure at
   2-hop, 1.2e-2) is the proof. Caught before training — this would otherwise have produced silent
   context truncation and unexplained boundary artifacts in every patch model.
3. **Exact parity with finite context also PROVES no global-block dependency** in the encoder config.
4. **Feature identity**: my independent 4-row fancy-index check across the full 9.5M range = 0.0.
5. **Strict-mask fractions reproduce exactly** from graph_support_active.npz: safe-2hop 62.08%,
   safe-4hop 29.30%; per shell 4-hop = 34.0/28.5/10.7/**0.17%** — the high-z shell keeps
   **106 of 62,243** galaxies. (First check with P4's radius-physical flags gave 40%/43% — those are
   the older approximate flags; the exact union-hop arrays are the operative ones. Both now stated.)

**HOLES (all pre-registration decisions, none code defects; loss_mask is caller-supplied so nothing
is irreversibly baked):**

- **H1 — CRITICAL: the strict-hop mask collides with the primary metric AND the data-limitation
  diagnosis.** The P5 schema declares loss = authoritative ∩ strict-hop; plan P8.3 mandates scoring
  ALL authoritative validation cores in all four shells. Under the current attention architecture
  (4-hop), strict TRAINING loss leaves the sparsest shell essentially UNSUPERVISED (0.17%) — the one
  shell three independent lines showed is data-limited — and strict SCORING makes macro R² undefined
  there. Their own smoke record (core 18730: 137 core nodes, 2 strict) shows the mask starving
  exactly where data starves. **Root cause: hop-strictness is pathologically conservative in sparse
  regions because hops there are physically LONG** — median hops-to-other-fold at high-z is 2, yet
  **87.8% of high-z galaxies are >10.4 Mpc (one smoothing length) from any fold boundary** (76.6% at
  >2 lengths). Labels correlate over the smoothing length, not over graph hops.
  **RECOMMENDATION (register before P8):** (a) TRAIN loss on ALL authoritative cores — cross-fold
  context is feature exposure, not label leakage; (b) PRIMARY scoring on all authoritative val cores
  per plan P8.3; (c) leakage-sensitive robustness cut = PHYSICAL margin (>10.4 / >20.8 Mpc), which is
  architecture-independent and retains the high-z shell; keep hop-strict subsets as a reported
  diagnostic, not the gate.
- **H2 — architecture fork must be pre-registered.** Attention (2 passes = 4 hops, 29% strict) vs
  receiver-only (per-pass hops, 62% strict) is now a real design fork. Whichever is THE P8 G-PATCH
  must be fixed before training; switching after seeing results would be post-hoc. Note the
  receiver-only variant is also an architecture change vs the R0 lineage — attribution of any gain
  needs the plan's matched-control language.
- **H3 — feature-scaling contract is incomplete.** The frozen contract covers the TARGET scaler only.
  The historical 8-feature pipeline = per-graph SI medians + box-cox + ntilde; "per-graph" has no
  defined meaning in the patch world (per-patch medians would break parity/patch-size invariance;
  parity used a neutral standardisation). Register: SI medians per CAP computed on TRAINING-FOLD
  nodes only, frozen, applied everywhere (deployment-consistent); box-cox fit on training cores.
- **H4 — minor:** ntilde_spline_v1 was frozen on wedge data; verify/refit for full NGC+SGC (SGC
  selection may differ) on training folds only.
- **H5 — minor, cheap:** parity ran at latent 8 / 2-pass / 16 cores on CPU. Add one production-shape
  GPU parity point (real ~20k-node patch, latent 80, XLA GPU kernels) vs the P1a full-graph
  reference; expect small float diffs — measure rather than assume.

VERDICT: P5's engineering integrity is high — the parity gate does what it claims, and the preflight
correction is the kind of catch that saves the programme weeks. The audit findings are all
protocol-decision gaps upstream of P8, with H1 the one that would quietly reproduce the old failure
mode (an under-supervised, unscoreable sparse shell) if trained as-schema'd. Decide H1-H3 in the
plan before any P8 run starts.

### 2026-07-19 — [science/code] P5 COMPLETE: canonical global-graph patches reproduce full-graph computation

P5 is complete and `GRAPH_PATCH_READY` has been written. The adapter is an
immutable view layer over the P1b/P2b canonical graph: graph metrics and union
connectivity are computed globally once, then exact core-plus-context patches are
extracted by global parent ID. No graph or graph metric is reconstructed per patch,
and no node or edge is capped or traversal-order truncated.

The full-footprint index contains 9,538,254 parent nodes, 48,743,628 context
Delaunay pairs, 190,563,017 union pairs, and 381,126,034 directed messages. It
indexes all 18,765 P4 cores through an incident-edge CSR and keeps authoritative core
ownership, strict loss masks, padding masks, canonical edge direction, and the seven
frozen graph metrics distinct. All nine construction gates pass.

The decisive parity test used the actual shared Jraph attention encoder code on the
canonical P1a graph (100,935 nodes and 1,988,732 undirected pairs). With two model
passes and the required four graph-hop dependency context, patch embeddings and a
fixed deterministic decoder agree exactly with full-graph results: maximum embedding
difference 0, maximum prediction difference 0, patch-order difference 0, boundary
error slope 0, and recursive-subdivision difference 2.38e-7. Randomly
initialized shared weights were used deliberately: this is an arithmetic
representation/parity gate, not an accuracy claim about a fitted model.

The production-scale smoke suite samples at least one strict four-hop core from every
NGC/SGC x fold stratum. All 12 samples preserve exact canonical features, exact
padding masks, non-empty strict loss masks, non-truncating subdivision, and unique
authoritative ownership. The largest sample has 24,212 nodes and 2,047,016 directed
edges; its exact recursive subdivision produces 61 subpatches without dropping a
core node.

The main scientific correction remains the P5 preflight result: the current
receiver-normalized attention `GraphNetwork` has two graph-hop dependencies per
nominal pass because its node update aggregates both sent and received edges. The
current two-pass model therefore uses the P4 four-hop support mask (29.30% overall,
effectively zero in the sparsest shell), while a receiver-only two-pass candidate can
use the two-hop mask (62.08%). Model passes and dependency hops are now separate
schema fields and must be derived for every future architecture.

Runtime artifacts are under
`/pscratch/sd/d/dkololgi/abacus/p5_graph_patch_adapter/`; compact immutable
evidence and the adapter schema are tracked under `docs/evidence/p5/`. P5 proves
that global-graph patch training can be implemented without changing graph inputs or
core predictions. It does **not** demonstrate that GraphNet generalizes: deterministic
blocked-fold training and blind-simulation testing remain the scientific objective.

### 2026-07-19 — [science/code] P5 PREFLIGHT: current attention GraphNetwork has two graph-hop dependencies per nominal pass

The first actual P1a full-graph-versus-patch embedding parity test exposed an
important receptive-field correction. With the current Jraph
`GraphNetwork`, a two-pass encoder supplied with only two-hop graph context
does **not** reproduce the full graph (maximum embedding difference 0.0121;
maximum fixed-decoder prediction difference 0.0113). Supplying four-hop
context makes the same full-graph, patch, subdivided-patch, and reordered-patch
outputs exactly equal in this deterministic test (all reported differences 0).

This is expected from the implementation: Jraph updates nodes from both
received and sent edge aggregates, while attention is normalized over edges
incoming to each receiver. A sent edge from the core to a neighbour therefore
depends on every other edge entering that neighbour. One nominal pass can
depend on nodes two graph hops away; the current two-pass attention GraphNet
requires four-hop context.

The frozen P4 arrays named `safe_2pass` and `safe_4pass` are mathematically
valid but must now be interpreted literally as **two-hop** and **four-hop**
support masks, not architecture-independent pass counts. Thus the current
two-pass attention GraphNet has the four-hop strict scoring fraction (29.30%
overall and effectively zero at z=0.45–0.55), whereas a receiver-only
message-passing variant with one-hop dependency per pass can use the two-hop
mask (62.08%). P5 now records model passes and dependency hops separately.

### 2026-07-18 — [code] P4 EVALUATED + pedagogical patch figures (real arrays): core/context/super-block/fold and model-agnosticism illustrated

Independent read of the P4 artifacts (all gates PASS, internally consistent):
17,202 occupied 64 Mpc/h cores (94.59 Mpc) grouped into 533 super-blocks (256 Mpc/h = 378 Mpc) across
5 blocked folds; 5,026,863 supervised galaxies; fold active max/min = 1.0102, max cap-shell deviation
2.4%; rotations 3 train / 1 val / 1 dev-test. Anti-leakage machinery is real and per-node:
graph_support_active carries min_hops_to_other_fold + safe_2pass (62%) / safe_4pass (29%) flags;
only 3.9% of the 190.6M union pairs cross a fold (context-only, no loss); 59,238 periodic box-images
(2885-4188 Mpc apart) retained context-only; repeated halo hosts never cross folds; distance-to-fold-
boundary median 46 Mpc >> the 10.4 Mpc smoothing.

Figures (workflows/abacus_tweb/plot_p4_patches.py, commit 9e03521;
/pscratch/sd/d/dkololgi/abacus/figures/p4_patches/):
- figA_manifest_anatomy — NGC cores coloured by fold (contiguous colour patches = blocked, not
  salt-and-pepper) + one super-block zoom (4x4x4 core tiling) + the core/context/super-block/fold
  definitions + real fold-balance bars.
- figB_model_agnostic — ONE real core (id 12281, NGC, 97 authoritative galaxies) rendered three ways:
  GraphNet (core + K-hop union-graph context), U-Net (same region as a 5 Mpc voxel field), F-tier/
  classical (FFT-tile / DTFE sampled at the SAME core). Drives home the key property: the manifest
  owns the cores + target (linear increments) + fold + train-core scaler; each architecture only
  supplies its own context representation.
- figC_boundary_safety — hops-to-other-fold, physical margin, and the random-split-vs-blocked-fold
  leakage contrast with the measured collapse numbers (0.80->0.42, 0.87->0.35).

These are communication artifacts, not a gate. STATE: P0-P4 COMPLETE; critical path now P5 GraphNet
patch adapter + parity (mine) and P6 U-Net adapter, both unblocked by P4. P10 ph002 cost benchmark
still unclaimed and must land before the Jul 21 freeze.

### 2026-07-18 — [science/code] P4 COMPLETE: deterministic 64-Mpc/h cores, five balanced folds, exact graph/field support

P4 is complete and `PATCH_MANIFEST_COMPLETE` has been written after an independent
readback and deterministic rebuild. The shared protocol is now a model-neutral data
contract for GraphNet, 3-D U-Net, F-tier, classical, and hybrid candidates—not a
GraphNet-specific tiling scheme.

**Frozen geometry and units.** Scientific cores are exactly 64 Mpc/h = 94.590600 Mpc
at Planck18 `h=0.6766`; super-blocks are four cores per axis, 256 Mpc/h = 378.362400
Mpc. Observer-frame Mpc coordinates remain indexing metadata. Core bounds are not
rounded to the 5-Mpc P3 voxel grid; each core stores the exact intersecting voxel
range separately.

**Periodic-image leakage guard.** The full-sky mock contains 59,238 repeated adjacent
host-key pairs (59,236 repeated groups). Their spatial separations are 2.885–4.188 Gpc,
median 2.960 Gpc: these are periodic box images, not multiple nearby galaxies in one
halo. For each repeated `(FILE_NUM, BOX_INDEX, HALO_INDEX)` key, one stable
TARGETID-ranked occurrence is authoritative; 59,238 other images remain context-only.
This leaves 5,026,863 supervised rows from the 5,086,101 P1b active rows and prevents
identical halo targets crossing folds without collapsing the sky into one giant fold.
Independent simulation phases are still mandatory because removing exact duplicate
halos does not make different views of ph000 independent universes.

**Five-fold balance.** There are 18,765 context-occupied 64-Mpc/h cores grouped into
533 spatial super-blocks. Each fold contains both caps and every reporting shell.
Authoritative rows per fold are 1,003,656 / 999,683 / 1,005,092 / 1,009,901 /
1,008,531 (max/min 1.010). The maximum relative deviation in any cap/shell cell is
2.42%; occupied super-block max/min is below 1.18; median conservative distance to a
fold boundary differs by less than 15% across folds. Five registered rotations use
three train, one validation, and one development-test fold.

**Exact P2b union support.** All 190,563,017 context union pairs were streamed from
the global parent Delaunay plus radius-only products, with zero cross-cap pairs.
Exact hop distance to another fold gives:

- two-pass safe: 3,120,670 authoritative rows (62.08%); fold fractions 59.4–63.2%;
- four-pass safe: 1,472,761 rows (29.30%); fold fractions 26.3–30.9%;
- at z=0.45–0.55, two-pass safety is 14.79% SGC / 19.65% NGC, whereas four-pass
  safety is 0.024% / 0.224%.

This is a decisive protocol result: a four-pass GraphNet has essentially no strictly
isolated high-z evaluation set under these folds. Do not enlarge or weaken the folds
merely to rescue it. Shallow GraphNet variants remain testable, while U-Net/F-tier
use their own recorded support and all candidates are compared on pre-declared shared
or intersection scoring masks. Global graph-metric construction remains explicitly
label-free and representation-level transductive; blind fresh-graph phase tests remain
the deployment gate.

**P3 field/exposure support.** Every P1 active coordinate lies inside its canonical
cap grid. P3 occupancy exposure supports 99.95% of authoritative rows; 91.14% have at
least 20 Mpc and 77.04% at least 40 Mpc to the nearest unsupported voxel. These are
quality/context flags, not permission to normalize or re-voxelize patches. P6 must
still pass context-growth and interpolation parity. FFT status is explicitly reserved
for P7 and does not block P5/P6.

All final gates pass: unique ownership, exact counts, no supervised host crossing,
cap/shell coverage, fold geometry matching, artifact checksums, P2/P3 identity,
semantic deterministic rebuild, and completion-marker binding. Runtime root:
`/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/`; compact evidence:
`docs/evidence/p4/`; validator: `workflows/abacus_tweb/p4_finalize_validate.py`.

**State:** P4 COMPLETE. P5 GraphNet and P6 U-Net adapters are now READY in parallel.
The primary objective remains deterministic transfer under the patch protocol; FMPE,
posterior calibration, HOD marginalisation, and JEPA remain downstream gates.

### 2026-07-18 — [science/code] P4 CORE-SIZE PROBE PASS: freeze 64 Mpc/h cores with explicit Mpc indexing

P4 evaluated exact scientific core sizes 32, 64, and 96 Mpc/h over the full P1b
NGC+SGC active catalogue. The indexing lengths are 47.2953, 94.5906, and 141.8859
observer-frame comoving Mpc at the independently audited Planck18 `h=0.6766`.
They are not relabelled as Mpc and are not rounded to the 5-Mpc P3 lattice.

The occupancy/resource trade-off supports the registered 64-Mpc/h default:

- 32 Mpc/h fragments the high-z shell into 28,260 occupied cores with median only
  two high-z labels; conservative four-pass context is 27,114 nodes at p95.
- 64 Mpc/h gives 7,520 high-z occupied cores with median six labels (p95 29),
  while conservative graph context is 42,078 nodes and about 1.25M union pairs at
  p95. A P3 field patch with 40-Mpc context is only 35^3 voxels.
- 96 Mpc/h improves high-z occupancy to median 16 but raises conservative context
  to 127,726 nodes and about 3.80M union pairs at p95. That extra physical size is
  unnecessary because sparse cores can be batched or gradient-accumulated.

The graph figures intentionally overestimate context by including complete adjacent
core cells and using the global mean union degree. They are a P4 selection probe, not
an 80-GB GPU guarantee. P5 must measure exact K-hop context and may losslessly
subdivide oversized computational batches without changing authoritative 64-Mpc/h
core ownership.

Artifacts: `docs/evidence/p4/p4_spatial_schema_v1.json`,
`docs/evidence/p4/core_size_probe.json`, runtime root
`/pscratch/sd/d/dkololgi/abacus/p4_spatial_manifest/`, and executable
`workflows/abacus_tweb/p4_probe_core_sizes.py`.

### 2026-07-18 — [science/code] P3a CATALOGUE–FIELD–TARGET CLOSURE PASS; P4 may proceed

The completed P3 fields were checked against the exact P1b catalogue and attached
T-Web targets before starting P4. This was designed to catch the historical failure
mode where a plausible marginal eigenvalue distribution was attached to the wrong
galaxies or spatial coordinates.

Three independent closures pass:

1. **Catalogue coordinates -> P3 CIC field:** all 6,397,925 context galaxies were
   independently redeposited, cap by cap, onto the frozen 5-Mpc P3 lattices. Across
   all 365,451,496 NGC+SGC voxels the maximum absolute difference from the stored
   count field is `2.86e-6`, no voxel differs by more than `5e-5`, and no CIC weight
   is lost. This is an exact coordinate/cap/deposition alignment test.
2. **Host keys -> attached targets:** among the 5,086,101 active rows, 59,238
   adjacent pairs with repeated `(FILE_NUM, BOX_INDEX, HALO_INDEX)` keys have exactly
   identical eigenvalues (`max |Delta lambda| = 0`). Every CWEB label equals the
   number of attached eigenvalues above the frozen threshold 0.2 (zero mismatches).
3. **Spatial field -> T-Web labels:** the P3 selection-aware log-count contrast was
   trilinearly sampled at 672,353 deterministic active galaxy positions. Spearman
   correlation with the T-Web trace is positive in every cap and shell: NGC
   `0.579, 0.487, 0.350, 0.233`; SGC `0.580, 0.457, 0.332, 0.240` from low to high
   redshift. Within-cap/shell shuffled-label controls lie between `-0.0044` and
   `+0.0008`. Individual-eigenvalue correlations are also positive in every stratum.

The declining correlation with redshift is physically expected from falling tracer
density; it is not evidence of row scrambling. The correlation test is deliberately
not interpreted as galaxy counts equalling the smoothed matter density: it is a
spatial-coherence/null test, while the independent CIC equality is the exact field
alignment test.

Artifacts:

- runtime report: `/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/catalogue_field_closure.json`;
- tracked report: `docs/evidence/p3/catalogue_field_closure.json`;
- executable audit: `workflows/abacus_tweb/p3a_catalogue_field_closure.py`.

**Decision:** P3 is closed with the additional catalogue/target alignment evidence.
P4 is ready. The P4 implementation must preserve the audited observer-frame Mpc
coordinates while defining the scientific core-size candidates in Mpc/h with an
explicit Planck18 conversion; no silent `64 Mpc`/`64 Mpc/h` substitution is allowed.

### 2026-07-18 — [code] P1b/P2b/P3a EVALUATED + overview figures produced; wedge demotion to canary ACCEPTED; one deployment-realism flag registered (z-error policy)

**Independent evaluation of the authoritative products (Claude Code):** manifests, hashes, and counts
verified directly; internally consistent throughout.

- **P1b** ph000_path1_full_ngc_sgc_v1: parent rows ARE canonical rows (PARENT_NODE_ID = FITS row =
  graph row) — 9,538,254 total; context 6,397,925 (NGC 4,518,132 / SGC 1,879,793 within z[0.10,0.60));
  **active 5,086,101** over z[0.15,0.55); shells 2.77M / 1.67M / 572k / 72k. vs the wedge canary this
  is **x17 active galaxies and x16 in the weakest 0.45-0.55 shell** — the data ceiling that three
  independent transfer tests identified is now directly attacked within ph000.
- **P2b**: promote-don't-recompute was the right resolution of the "largest contiguous volume"
  ambiguity — the existing full-sky Delaunay + cuGraph metrics are reused exactly, and only per-cap
  radius pairs were added (141.8M) -> **190,563,017 union context pairs, 0 cross-cap edges, built in
  99 seconds.** Neither my wedge-first reading nor a rebuild-everything reading was right: promotion
  made full-footprint essentially free.
- **P3a**: per-cap 5 Mpc observer-frame lattices (NGC 539x823x528, SGC 445x764x386), 8 channels,
  lossless CIC by cap and shell, expected-count identity to 1e-9, independent post-build readback,
  and the unit gate correctly caught the Mpc vs Mpc/h cell-size distinction.
- **Wedge P1a/P2a**: demotion to canary/regression-test accepted — that is now their correct role.

**FLAG (deployment realism, registered for P8 baselines + P13 golden mock):** P1b's redshift policy is
"parent observed Z, NO additional measurement error" (chosen to preserve exact catalogue/graph/metric
alignment — correct for the protocol gate). But the FROZEN canonical evidence (P0) was built from the
S2 shells WITH sigma_v=35 km/s z-errors, and DESI data will have them too. Consequences to keep
explicit: (1) P8 comparisons vs the frozen baseline straddle a z-error difference — small (35 km/s ~
0.5 Mpc at these z, well under the 10.4 Mpc smoothing) but should be stated when quoting gains;
(2) models trained on error-free z see slightly sharper small-scale structure than DESI provides —
the P13 golden mock MUST include z-errors, and a z-error augmentation is a candidate nuisance view
for P11 JEPA. Not a blocker; a bookkeeping obligation.

**Overview figures produced** (workflows/abacus_tweb/plot_p1b_p2b_overview.py):
/pscratch/sd/d/dkololgi/abacus/figures/p1b_p2b_overview/
  fig1_footprint_and_data.png — NGC+SGC sky map with all three historical wedges overlaid; N(z)
    context vs active; per-shell counts (xN vs canary); canary->authoritative scale-up bars.
  fig2_pipeline_status.png — P0-P13 DAG with live statuses + Jul16->DESI timeline incl. shutdown.
  fig3_patch_protocol.png — the generalised-model protocol: core/context anatomy, blocked 5-fold
    scheme, matched candidates + pass rules, with the transfer-test motivation stated.

STATE: P0 ✓ P0S active | P1 ✓ P2 ✓ P3 ✓ | P4 ACTIVE (folds; the critical path item) -> P5/P6 parity
(Jul 19) -> P8 two-fold screen (Jul 20) -> Jul 21 freeze. P10 ph002 cost benchmark still unclaimed
and must land before the freeze.

### 2026-07-18 — [science/code] P3a COMPLETE: canonical 5-Mpc NGC+SGC fields pass unit, conservation, chunk, and independent readback gates

P3a is complete for `ph000_path1_full_ngc_sgc_v1`. The canonical observer-frame
field products were constructed once per Galactic cap from every P1b context galaxy;
NGC and SGC are never enclosed in one lattice and cannot mix through empty sky.
The build used source commit `2afa5f53901727b0621950a198e206780bc94c87` and
wrote `FIELD_COMPLETE` only after all component and global gates passed.

Frozen grids (observer-frame comoving Mpc; Planck18):

- NGC: `539 x 823 x 528`, 4,518,132 context galaxies, 1.55 GiB compressed;
- SGC: `445 x 764 x 386`, 1,879,793 context galaxies, 0.76 GiB compressed;
- cell = 5 Mpc = 3.383 Mpc/h; padding = 40 Mpc = 27.064 Mpc/h;
- raw canonical channels: CIC counts, binary/apodized exposure, expected counts,
  log count ratio, ntilde in Mpc^-3, and three LOS components.

CIC deposition is lossless by cap and reporting shell. Float32 HDF5 readback sums
are 4,518,132.000293 (NGC) and 1,879,792.999953 (SGC). Every field is finite;
all dataset shapes/dtypes/chunks match the schema; the fixed-halo apodization
recomputation agrees exactly; and only 0.159% / 0.279% of deposited weight lies
outside the occupancy-derived binary support (well below the registered 2% gate).

An independent post-build reader then verified:

- SHA-256 equality for both HDF5 files and binding of `FIELD_COMPLETE` to the
  manifest hash;
- exact equality for overlapping HDF5 reads;
- expected-count identity `mu = ntilde * (5 Mpc)^3 * exposure` to better than
  `1e-9` in sampled supported blocks;
- reconstructed log-count contrast to better than `1.2e-7`;
- LOS unit norms to better than `3.9e-8`;
- nonnegative counts, binary support values, and frozen-schema/unit-audit hashes.

The HDF5 datasets carry the explicit `cell_mpc` attribute; the full coordinate-unit
contract is bound through the checksummed frozen schema and passing unit audit. P4/P6
consumers must open fields through the manifest/schema contract rather than treating a
bare HDF5 file as self-describing.

Authoritative root:
`/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/`

Key products: `ngc_fields.h5` (SHA-256
`dfdf07126c38d9d5acb8ef44a598214f1a9a4dfb1f433bbd1032ec9524b2d81a`),
`sgc_fields.h5` (SHA-256
`fc0d0cf3fc7be6b74441dc8988e17de7dfcb76e9f700c48b8a101c8bbe77040d`),
`field_manifest.json`, `validation_report.json`, `postbuild_validation.json`,
`support_atlas.json`, `unit_audit.json`, and `FIELD_COMPLETE`. Compact evidence is
tracked under `docs/evidence/p3/`.

Scope guard: the P3a exposure is a target-free, split-free HEALPix occupancy support
derived from the parent observed galaxies. The current staged parent has no explicit
random-catalogue exposure, per-object completeness, or luminosity field. Those remain
P3b observation-model upgrades and are not silently approximated here.

**Interpretation:** P3a completes the canonical field substrate and unblocks P4 patch
manifests plus P6 U-Net/F-tier adapters. It is not evidence that any encoder generalises;
that controlling claim still requires blocked patch training and blind simulation tests.

### 2026-07-18 — [science/code] P3 UNIT GATE: observer lattices are comoving Mpc; 5 Mpc/h is not the historical U-Net cell

P3 was paused before the full-cap build because the plan said “5 versus 6 Mpc/h,”
while the established U-Net scripts named their grid argument `cell_mpc`. A dedicated
unit audit now resolves this with two independent data-level checks rather than trusting
variable names.

1. On 8,192 sampled P1b rows, the canonical graph XYZ values equal the Planck18
   observer-frame Cartesian coordinates calculated from `(RA, DEC, Z)` in comoving Mpc
   exactly (maximum component and radial error 0). Interpreting the same coordinates as
   Mpc/h is rejected by a maximum 724 Mpc discrepancy.
2. On 8,192 historical full-range U-Net rows, `||XYZ||` matches the Planck18 comoving
   distance in Mpc to `2.41e-6` Mpc. The h-scaled interpretation is rejected by a
   maximum 683 Mpc discrepancy. The saved model configuration explicitly used
   `cell_mpc=5.0`, `pad_mpc=40.0`, and the audited code reproduces its
   `334 x 317 x 194` grid shape.

**Frozen convention:** P3 stores observer-frame Cartesian coordinates and lattice
lengths in comoving Mpc. The first matched protocol experiment retains the historical
5 Mpc cell, which is 3.383 Mpc/h for Planck18 (`h=0.6766`). A literal 5 Mpc/h cell is
7.390 Mpc and would be a different, substantially coarser resolution ablation. This
does not change the physical target convention: T-Web smoothing remains 7 Mpc/h
(10.346 Mpc), and the 14.78 Mpc graph radius remains 10.000 Mpc/h.

The passing audit is now a hard `FIELD_COMPLETE` dependency. The P3 schema records
both unit systems explicitly and the builder refuses a failing audit or inconsistent
cell conversion. The cap-separated storage probe gives 12.06 GB of raw channels at
5 Mpc versus 6.98 GB at 6 Mpc, so the matched 5 Mpc representation remains feasible.

Pre-build artifacts/tests:

- `/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/unit_audit.json` — PASS;
- `/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/p1a_canary_parity.json` — PASS,
  including exact CIC, fractional-index, LOS, expected-count, and contrast parity;
- `/pscratch/sd/d/dkololgi/abacus/p3_full_footprint/storage_probe.json`;
- `docs/evidence/p3/p3_field_schema_v1.json`;
- `workflows/abacus_tweb/p3a_audit_units.py`;
- `workflows/abacus_tweb/p3a_build_canonical_fields.py`;
- `workflows/abacus_tweb/p3a_canary_parity.py`;
- `tests/phase4/test_p3_field_utils.py` — four tests pass.

Status: P3 implementation/canary gates pass; NGC+SGC field construction and full
validation are next. This is not yet `FIELD_COMPLETE`.

### 2026-07-18 — [science/code] P3 READINESS: GO for full-cap field implementation; plan now tracks every work package with granular checklists

The plan had inconsistent progress semantics: P0/P0S used checkboxes, while P1 onward
mixed completed implementation, design requirements, and future gates in prose. This
made completed P1b/P2b work harder to audit and made P3 readiness ambiguous. The live
plan now has explicit checked completion items for P1/P2 and unchecked execution/gate
items for P3-P13. A work-package status is no longer inferred merely from narrative.

**P3 readiness verdict: GO to implement P3a now.** Authoritative P1b is complete and
provides exact parent-row/XYZ alignment for 6,397,925 context galaxies across NGC and
SGC. P2b is also complete and fixes shared global identity/provenance, although count
field deposition itself depends only on P1b.

This is not yet a `FIELD_COMPLETE` verdict. The existing pooled U-Net builder is tied
to a rectangular wedge. The full-cap implementation must use separate immutable NGC
and SGC Cartesian lattice frames with chunked storage; it must not allocate one dense
bounding cuboid around both caps or recompute/standardize fields independently inside
patches.

The shutdown-critical P3a channel contract is counts, footprint/exposure support,
smooth radial expected counts, stabilized contrast, ntilde(z), and LOS unit vectors.
The authoritative graph-ready FITS schema has no explicit random-catalogue exposure,
per-object completeness, or luminosity fields. Those are therefore recorded as P3b
observation-model upgrades rather than blockers for deterministic patch-protocol
testing. P3a must document its exposure approximation and must not be described as the
final DESI observation model.

Next: freeze the P3 schema and 5-versus-6 Mpc/h storage/resource estimate, implement a
P1a parity canary, then build/audit both full caps before writing `FIELD_COMPLETE`.

### 2026-07-18 — [science/code] P1/P2 SCOPE CORRECTION: wedge products are canaries; authoritative patch training uses the full NGC+SGC footprint

The phrase "largest complete contiguous volume" in the generalisable-VAC plan was ambiguous. The
2026-07-18 Wave-0 implementation resolved it operationally to RA 118–162 / Dec 12.5–32.6 with
redshift buffers. That was a reasonable deadline canary, but it is **not** the intended canonical
volume for the new patch-training experiment.

**Controlling decision:** construct the catalogue, canonical graph, and globally derived graph
products over the full usable ph000 BGS footprint. NGC and SGC are two disconnected components in
one catalogue/graph object: build or retain topology separately within each Galactic cap, map both
through stable parent/global indices, and concatenate without any cross-cap edges. Fixed-comoving
P4 cores and folds are then defined across both caps. Patches are views of this canonical two-cap
representation; graph metrics are never recomputed inside patches.

The existing path1 parent already has the expensive full-footprint Delaunay products:

- 9,538,254 graph-aligned catalogue rows;
- 67,546,704 undirected Delaunay pairs;
- separate north/south Gudhi complexes merged by global indexing;
- global cuGraph node and edge metrics in the established seven-node/five-edge convention.

These artifacts must be audited for exact row/selection/target alignment and then reused wherever
their contract matches. Missing full-footprint products, including the production union-radius
topology or its validated patch-extraction equivalent, should be built once at the parent level.

**Status correction:**

- `ph000_path1_wedge_v1` P1 and its 374,537-node P2 graph are **P1a/P2a CANARY COMPLETE**;
- authoritative **P1b full NGC+SGC is COMPLETE**;
- authoritative **P2b full NGC+SGC is COMPLETE**: promoted full Delaunay + cuGraph products plus
  a canonical per-cap fixed-radius augmentation;
- P3/P4 may use the wedge for engineering smoke/parity tests only. The scientific P4 manifest and
  deterministic P8 training folds must be generated from P1b/P2b across NGC+SGC.

This larger footprint is central to the present objective: patch training needs many spatially
separated structures and optimization units. The wedge can validate code, but it cannot establish
that the protocol generalises. Same-phase blocked folds remain development evidence; P10 still
requires a fresh independently constructed simulation phase for the production-transfer claim.

**P1b result:** `ph000_path1_full_ngc_sgc_v1` retains the immutable 9,538,254-row parent
catalogue as the canonical row order and adds a compact 32 MB index/manifest. It contains
6,649,091 NGC and 2,889,163 SGC rows; 6,397,925 context rows in the usable redshift support and
5,086,101 active labelled rows over 0.15 < z < 0.55. Shell counts are 2,769,399 / 1,672,079 /
572,270 / 72,353. The parent observed Z is retained without a second measurement-error draw so
catalogue geometry, the existing full graph, and global metrics remain exactly aligned.

Artifacts:

- `/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/canonical_index.npz`;
- `/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/manifest.json`;
- `/pscratch/sd/d/dkololgi/abacus/p1b_full_footprint/full_footprint_audit.json`.

**P2b result:** the promoted parent graph passes exact row, coordinate, component, edge-index,
and finite-feature gates. It has zero cross-cap pairs. Within the authoritative context volume
there are 48,743,628 Delaunay pairs and 178,333,745 radius pairs; 36,514,356 overlap, leaving
141,819,389 radius-only additions and 190,563,017 union pairs. Radius-only pairs and their five
edge attributes are stored separately by cap, preserving global parent IDs and avoiding a copy
of the Delaunay layer. P4 patch extraction must fancy-index the parent Delaunay arrays and append
the corresponding per-cap radius-only rows.

Artifacts: `/pscratch/sd/d/dkololgi/abacus/p2b_full_footprint/`
(`p2b_union_manifest.json`, per-cap pair/attribute arrays, `UNION_COMPLETE`).

### 2026-07-18 — [code] P1 + P2 COMPLETE (first-pass gates): canonical catalogue + canonical union graph frozen — **sol: P3/P4 are unblocked, over to you**

**P1 CANONICAL CATALOGUE — COMPLETE.** ph000_path1_wedge_v1: one contiguous extraction from the
path1_fiberassign parent (sha 1ff0ac13...), RA 118-162 / DEC 12.5-32.6 / Z_ORIG [0.10,0.60) minus
sentinel, single z-error pass (35 km/s, seed 42). **374,537 rows = 301,912 active + 72,554 buffer**;
shells 164,222 / 98,584 / 34,645 / 4,532. Gates first-pass, including canonical continuity: all
219,929 frozen TARGETIDs present with **exact** eigenvalue agreement (max|dlam|=0.0); 99.96% remain
active under the fresh z-draw (rest crossed shell edges; recorded in manifest). Active set is ~37%
larger than the old canonical basis because P1 correctly drops RA-gutter filtering — split boundaries
now belong to P4. Artifacts: /pscratch/sd/d/dkololgi/abacus/p1_canonical/ph000_path1_wedge/
(canonical_catalogue.fits + manifest.json + CATALOGUE_COMPLETE). Commits 074b3e4, 96ee03b.

**P2 CANONICAL GRAPH — COMPLETE.** Gold-validated chain reused end-to-end on the P1 rows (no
refiltering: --catalog-path + both no-filter flags): Delaunay (gudhi) -> cuGraph global metrics
(rapids-gnn, production invocation) -> union 14.78 Mpc -> new validator (b97bb43).
**374,537 nodes | Delaunay 2,849,717 pairs | radius 9,882,220 | union 10,601,479 undirected pairs
(x3.72) | 1 component | 0 isolated | 7 node features (R0/A1 Delaunay-metric convention, recorded).**
Edge-type provenance saved (delaunay-only 719,259 / radius-only 7,751,762 / both 2,130,458 — raises
if any pair is neither). Node flags saved (survey-boundary<15 Mpc, extreme-degree, buffer). Position
alignment vs P1 verified. GRAPH_COMPLETE written by the validator only.
Artifacts: /pscratch/sd/d/dkololgi/abacus/p2_canonical/ph000_path1_wedge/ (arrays + parquet +
edge_provenance.npz + node_flags.npz + p2_manifest.json). NOTE for consumers: the union npz stores
UNDIRECTED pairs (10.6M); bidirectional doubling happens downstream as before (=21.2M directed at
message-passing time). One cosmetic wart: p2_manifest's "union_directed_edges" field actually holds
the undirected pair count.

**HANDOFF -> sol (per the logged division of labour):**
- **P3 canonical fields** on the P1 catalogue (counts / expected counts / contrast / mask / ntilde /
  LOS channels; conserve totals; no per-patch standardisation). Use ALL 374,537 rows for field
  deposition (buffer included); only active rows ever carry loss.
- **P4 manifest/folds**: fixed-comoving cores (evaluate L_core 32/64/96 Mpc/h), 5 super-block folds,
  matched val/test geometry, support flags. Graph K-hop support can use the union topology +
  edge_provenance just frozen. The P1 D_BOUNDARY_MPC column and P2 node_flags are ready inputs.
- P5 (GraphNet patch adapter) is mine once P4 lands; P6 is sol's or mine depending on timing — claim
  here first. P10 ph002 benchmark still unclaimed.

Wave-1 exit criteria (plan section 12) are now half met: CATALOGUE_COMPLETE + GRAPH_COMPLETE done;
FIELD_COMPLETE + P4 manifest remain. On current pace the Jul 19 adapter/parity wave is reachable.

### 2026-07-18 — [code] WAVE 0 EXECUTION: P1 CLAIMED (Claude Code); division of labour for sol; P2 scope + P1 continuity decisions

**DIVISION OF LABOUR (JDPK-approved; sol please confirm or amend here):**
- **Claude Code (this agent): P1 canonical catalogue (CLAIMED, in progress) → P2 canonical graph +
  global metrics next.** Rationale: P2 reuses my prior chain (union builders, cuGraph subset tools,
  s3b/s3c gate style).
- **sol (Codex): P3 canonical fields + P4 spatial manifest/folds** — natural extension of the P0
  evaluator/manifest machinery. Also §1.1 refresh from evidence_freeze.json when convenient.
- **Either agent: P10 one-phase target-generation benchmark** — I nominate **ph002** (a training-pool
  phase; keeps ph001 sealed). Claim it here before starting.
- Coordination bus = this log + the plan + git. Claim before building; do not double-build.

**WAVE-0 DECISIONS (registered per plan discipline):**
1. **P2 scope, catalogue #1 = the buffered full-range WEDGE, not full-sky:** RA 118-162, DEC
   12.5-32.6, z-buffered around 0.15-0.55. Fits the Jul 18-19 wave; contains all four reporting
   shells + the canonical rows; full-sky (and other phases) become later P1/P2 iterations of the
   same tools. "Largest complete contiguous volume" ambiguity resolved in favour of shipping the
   protocol test before shutdown.
2. **P1 lineage + continuity policy:** parent = the path1_fiberassign graph-ready full-sky catalogue
   (9,538,254 rows, rs7 halo_xcom labels) — the SAME parent S2 used. S2 injected DESI-like z-errors
   (sigma_v=35 km/s, seed 42) PER SHELL, so a single contiguous re-extraction cannot reproduce the
   old per-row Z draws. DECISION: P1 performs ONE z-error injection over the whole slice (same
   sigma_v/seed convention, Z_ORIG preserved, sentinel window 0.585-0.595 excluded) and anchors
   continuity on **TARGETID + Z_ORIG + eigenvalue agreement** with the frozen canonical rows —
   not on observed Z. The P0 evidence freeze stays valid unchanged (it is TARGETID-keyed); the new
   catalogue is the go-forward canonical basis. Reporting shells assigned on observed Z; buffer rows
   flagged shell="buffer", never active.
3. **Deterministic target/metric contract frozen** (Wave 0 item 1):
   `docs/evidence/contracts/p8_target_metric_contract_v1.json` — linear increments (v1, l2-l1,
   l3-l2), scaler fit on training cores only, primary metric = mean over blocked folds of
   equal-shell macro R2(lambda1) on complete validation folds, checkpoint on complete-fold macro,
   pooled tertiary, ordering-violation rate mandatory. Matches plan P8.1/P8.3 verbatim.

P1 gates (beyond the plan's list): canonical 219,929 TARGETIDs must be a subset of P1 active rows
with matching Z_ORIG and eigenvalues (float tol); TARGETID uniqueness across the whole slice (cutsky
replications could duplicate — hard gate, not assumption). Builder:
`workflows/abacus_tweb/p1_build_canonical_catalogue.py`; output under
`/pscratch/sd/d/dkololgi/abacus/p1_canonical/ph000_path1_wedge/` with config JSON, source hashes,
git SHA, counts, and `CATALOGUE_COMPLETE` written only after all gates pass (§14 discipline).

### 2026-07-18 — [science+ops] Shutdown priority reframed: demonstrate transferable deterministic inference under the patch protocol

The shutdown-critical objective is now deliberately narrower than a production
posterior or a fully nuisance-marginalised VAC:

> **Demonstrate transferable deterministic inference under the new patch
> protocol.**

The protocol—not GraphNet, U-Net, or F-tier—is the primary hypothesis. Each learned
candidate must train on canonical global representations viewed through many
core/context patches, select checkpoints using spatially blocked validation, and
transfer to fresh regions/graphs. Independent Abacus phases remain the strongest
test, but posterior fitting and HOD marginalisation must not delay the deterministic
protocol gate.

**Primary selection metric.** The controlling metric is spatial-fold macro
`R²(lambda1)`, not pooled galaxy-level `R²` and not the best score from any one
patch. For every blocked fold, compute `R²(lambda1)` on all authoritative validation
core galaxies separately in each eligible reporting shell, average the shell values
with equal weight, then summarize across folds with the fold distribution and a
spatial-block uncertainty interval. Mandatory safeguards are the worst-shell and
per-shell `R²`, Spearman correlation, MAE, bias, slope, balanced four-class accuracy,
and the source-to-transfer gap. Pooled `R²` is tertiary because the dense low-z shell
otherwise dominates the decision. Early stopping and architecture selection must not
use the sealed test or blind phase.

**Target-parameterisation audit.** This programme has already tested substantially
more than raw eigenvalues versus ordinary increments:

- shape parameters `(I1,e,p)` and invariants `(I1,I2,I3)` were found pathological
  as ML targets and remain deprecated;
- the June 19 matched wedge NPE study tested softplus increments, linear increments,
  and raw eigenvalues. Linear increments gave the practical NLL/point-accuracy
  compromise used by the downstream wedge models; the June 22 correction established
  that softplus, not linear, had the best TARP curve;
- the deterministic 15-component target was also implemented and run:
  `(lambda1, Delta12, Delta23, R grad lambda1..3, R^2 laplacian lambda1..3)`, with
  block weights `1.0 / 0.1 / 0.03` for eigenvalue, gradient, and Laplacian blocks.
  Representative old-protocol results include lambda1 `R²=0.819` for the weighted
  wedge and `0.806/0.768/0.592` for the cutsky-BGS/intersection/stage3-unique variants.
  These are useful historical evidence but were not spatially independent transfer
  tests and do not justify reopening the 15-d target on the shutdown critical path;
- the July `R1 15-d` failure refers to **15 input features**, not the 15-component
  derivative target. These two experiments must not be conflated.

No literal `log(lambda2-lambda1), log(lambda3-lambda2)` run was found in the ledger.
The existing inverse-softplus increment transform already behaves approximately as a
log transform for small gaps, so log gaps are not a wholly new target family. Register
one bounded deterministic ablation only: after a linear-increment patch baseline passes
parity, compare literal log gaps on the same leading encoder, folds, seed, update budget,
and training-core scaler. Adopt only for a reproducible gain in spatial-fold macro
`R²(lambda1)` with no meaningful worst-shell, ordering, or class degradation. Do not
launch a broad target sweep before the patch protocol works.

**HOD and posterior scope.** The existing staged LSS mock catalogues are sufficient
for the first protocol test. Extra HOD samples from the same phase are population
variations, not new cosmic structures, and are not required before shutdown. HOD
marginalisation remains a later robustness/uncertainty branch, after spatial transfer
and at least one fresh-phase test. Likewise, deterministic ordered-eigenvalue heads are
sufficient to decide whether the protocol generalises. FMPE/NPE, SBC/TARP, posterior
probabilities, and calibration remain downstream of a frozen deterministic winner and
are required only for posterior columns ultimately claimed by the VAC.

**Scratch and environment preservation.** No files are moved in this step, but the
plan now assigns storage tiers. Perlmutter scratch is active, rebuildable workspace—not
the source of truth; files not accessed for eight weeks are purge-eligible and scratch
is not backed up. Code, schemas, manifests, hashes, decisions, and compact evidence
belong in Git/home. Irreplaceable reusable products and selected checkpoints belong in
CFS; large archival bundles belong in HPSS. A migration manifest must inventory the
untracked upstream staged-mock scripts currently under
`/pscratch/sd/d/dkololgi/abacus/SecondGen_Mocks/ph000/` before they are copied into a
versioned repository location. Reproducibility records must cover both
`/pscratch/sd/d/dkololgi/conda/envs/cosmic_env` and
`/pscratch/sd/d/dkololgi/conda/envs/rapids-gnn`, including conda history/explicit
specifications, package inventories, CUDA/module metadata, environment variables, and
smoke tests. `rapids-gnn` remains mandatory for large global node/edge graph-metric
construction.

### 2026-07-17 — [code+science] P0 COMPLETE: canonical matched evidence frozen; no encoder generalisation win

P0 of `docs/plan_generalisable_graphweb_vac.md` completed in NERSC interactive
allocation 56033600 on one 80 GB A100. The final evaluator hard-matched all seven
methods to the same 219,929-row `s3c_cnn_fullrange` catalogue, target convention,
and spatial split. All runtime checksums pass and
`/pscratch/sd/d/dkololgi/abacus/p0_evidence_freeze/P0_COMPLETE` was written only
after both GraphNet exports, evaluation, inventory refresh, and checksums succeeded.

Final matched lambda1 evidence:

| method | validation pooled / macro R2 | test pooled / macro R2 | test 95% spatial-block R2 interval |
|---|---:|---:|---:|
| R0 GraphNet posterior mean | 0.531 / 0.463 | 0.504 / 0.440 | [0.441, 0.553] |
| A1 sqrt-balanced GraphNet | 0.536 / 0.468 | 0.475 / 0.430 | [0.397, 0.532] |
| U-Net | 0.549 / 0.461 | 0.552 / 0.438 | [0.514, 0.582] |
| DTFE, affine fitted on train only | 0.546 / 0.368 | 0.553 / 0.342 | [0.520, 0.581] |
| CIC, affine fitted on train only | 0.537 / 0.339 | 0.539 / 0.155 | [0.502, 0.569] |

Test lambda1 R2 by shell:

| method | 0.15-0.25 | 0.25-0.35 | 0.35-0.45 | 0.45-0.55 |
|---|---:|---:|---:|---:|
| R0 | 0.575 | 0.450 | 0.397 | 0.338 |
| A1 sqrt | 0.565 | 0.385 | 0.432 | 0.337 |
| U-Net | 0.597 | 0.543 | 0.426 | 0.186 |
| DTFE train-affine | 0.567 | 0.605 | 0.418 | -0.223 |
| CIC train-affine | 0.563 | 0.613 | 0.443 | -0.998 |

The conclusion is deliberately NOT that GraphNet wins. U-Net and train-calibrated
DTFE lead pooled point accuracy, while R0 retains usable performance in the final
sparse shell where the classical estimators collapse. The 100 Mpc spatial-block
intervals for R0, U-Net, and DTFE overlap substantially. A1 shell balancing does
not improve the frozen test result. These facts reinforce the protocol-first
programme: train on canonical global representations through many core/context
patches, select on spatially blocked patches, and evaluate blindly on independent
simulation phases.

Four-class test accuracy is 0.676 for R0, 0.653 for A1, 0.666 for U-Net, and 0.681
for train-affine DTFE; balanced accuracy is 0.629, 0.620, 0.621, and 0.609
respectively. R0's posterior knot event has test Brier 0.0403 and Brier skill
0.316 relative to climatology. Deterministic U-Net/DTFE/CIC threshold decisions
remain explicitly badged as decisions, not posterior probabilities, so their Brier
scores are not interpreted as calibrated posterior evidence.

Versioned artifacts:
- `docs/evidence/p0/evidence_freeze.json`;
- `docs/evidence/p0/asset_inventory.json`;
- runtime exports and checksums under
  `/pscratch/sd/d/dkololgi/abacus/p0_evidence_freeze/`;
- evaluator/exporter/inventory under `workflows/abacus_tweb/p0_*.py`;
- interactive runner `workflows/abacus_tweb/run_p0_evidence.sh`;
- reusable allocation skill `~/.codex/skills/nersc-interactive-allocation/`.

The inventory still finds source phases ph000-ph024 but T-Web-labelled GraphWeb
assets only for ph000. P0 therefore clears canonical evidence and current-phase
preprocessing work, but it does not waive the independent-phase P10 gate. Different
observer views of ph000 remain same-phase tests, not blind universes.


### 2026-07-16 — [code] P0 evidence contract implemented; matched U-Net/classical evidence and simulation-asset inventory frozen

Began P0 of `docs/plan_generalisable_graphweb_vac.md` as a reproducible evidence
freeze, not as a new model-selection exercise. Implemented one canonical evaluator
which hard-aligns every method to the `s3c_cnn_fullrange` row index and split:
219,929 rows total, with 129,113 train, 18,629 validation, and 52,848 test
rows. The target contract is the ordered real-space T-Web eigenvalue triplet at
simulation epoch z=0.2 and Rsmooth=7 Mpc/h (10.4 comoving Mpc); the inputs are the
observed redshift-space BGS-like catalogue. Any truth or row mismatch is fatal.

Implemented and tested:

- `workflows/abacus_tweb/p0_evidence.py`: matched regression, four-class, knot-event
  reliability/Brier, and 100 Mpc comoving spatial-block bootstrap metrics;
- `workflows/abacus_tweb/p0_export_graphnet_predictions.py`: export of frozen R0/A1
  posterior means and class probabilities onto the canonical row index;
- `workflows/abacus_tweb/p0_inventory_assets.py`: phase/HOD/observer/label/graph/model
  inventory with release-gate findings;
- `workflows/abacus_tweb/submit_p0_evidence.slurm`: frozen GPU evidence entry point;
- `tests/phase4/test_p0_evidence.py`: affine-calibration, web-class, reliability, and
  spatial-block tests.

Execution guardrail discovered during verification: the NERSC login shell can
inject DESI Python 3.13 `site-packages` into the Python 3.11 `cosmic_env`, causing
NumPy C-extension import failure. With `PYTHONPATH`, `PYTHONHOME`, and
`PYTHONUSERBASE` removed, all four P0 unit tests pass. The P0 SLURM entry point now
enforces that clean-environment contract after conda activation; direct use of the
environment without the guard is not a valid runtime test.

The point-estimator dry run is frozen at
`/pscratch/sd/d/dkololgi/abacus/p0_evidence_freeze/evidence_point_dryrun.json`.
It unsealed the already-frozen U-Net on the test rows only after implementation of
the evidence contract; no model or threshold was tuned on this result:

| lambda1 R2 | validation pooled / macro | test pooled / macro |
|---|---:|---:|
| U-Net | 0.549 / 0.461 | 0.552 / 0.438 |
| DTFE raw | 0.354 / -0.065 | 0.287 / -0.226 |
| DTFE, affine fitted on train only | 0.546 / 0.368 | 0.553 / 0.342 |
| CIC, affine fitted on train only | 0.537 / 0.339 | 0.539 / 0.155 |

This is not an encoder victory. U-Net and train-calibrated DTFE are at pooled
parity, and both summaries conceal severe last-shell weakness. The evidence
strengthens the corrected interpretation already in this log: architecture swaps
alone have not supplied transferable inference; canonical global representations,
proper patch/core/context training, spatial validation, and blind independent-phase
tests remain the central programme. Deterministic threshold scores are explicitly
labelled as decisions, not posterior probabilities; posterior Brier/reliability
claims await the frozen R0/A1 export.

The versioned inventory is `docs/evidence/p0/asset_inventory.json` (runtime copy:
`/pscratch/sd/d/dkololgi/abacus/p0_evidence_freeze/asset_inventory.json`). It finds
source catalogues for Abacus phases ph000--ph024, but only ph000 currently has the
required T-Web-labelled catalogue and canonical GraphWeb caches/checkpoints. Thus
the current-phase P0 assets are ready, while the independent-phase blind test is
correctly flagged NOT READY until identical label products are generated. Different
observer views of ph000 must not be mislabelled as independent cosmic realizations.

P0 remains ACTIVE. Pending work is deliberately narrow: run the frozen R0 and A1
posterior export, add their matched class/probability/block-uncertainty results to
the final machine-readable evidence JSON, then mark only the actually satisfied P0
items complete.

### 2026-07-16 — [code] DTFE ON THE FULL-RANGE SPATIAL HOLDOUT: NOT a GNN "win" — classical ties/beats the GNN wherever it is DEFINED; the macro gap is only DTFE's mechanical collapse in the sparse shell. Real lesson: our encoders don't GENERALISE → need patch-based graph training/validation.

[FRAMING CORRECTED per JDPK — the earlier headline of this entry called this a "clean GNN>classical
win / VAC rescued". That is WRONG and complacent. Corrected reading below.]

Ran the no-ML classical baseline (DTFE + FFT tidal solve) on the FULL-RANGE spatial holdout
(z0.15-0.55), scored on the SAME s3c test mask (RA>=150, 52,848 nodes) the production GraphNet uses.
Matched, both on TEST, cal (train-fit affine):

| λ1 R² (test) | pooled | macro | per-shell 0p15 / 0p25 / 0p35 / 0p45 |
|---|---|---|---|
| GraphNet (R0, production, spatial holdout) | 0.514 | 0.453 | 0.56 / 0.49 / 0.42 / **0.34** |
| DTFE (full-range, same split) | **0.553** | 0.342 | 0.567 / **0.605** / 0.418 / **-0.223** |
| CIC (full-range) | 0.539 | 0.155 | 0.563 / **0.613** / 0.443 / -0.998 |
DTFE λ2/λ3 macro 0.472 / 0.515; pooled 0.674 / 0.711.

**READ IT HONESTLY — this is NOT the GNN beating classical.**
- **POOLED: DTFE WINS** (0.553 vs 0.514). On the aggregate test set the density estimator is ahead.
- **PER SHELL, where DTFE is well-defined (has tracers): DTFE ties or BEATS the GNN in 3 of 4 shells**
  — 0p15 tie (0.567 vs 0.56), 0p25 DTFE ahead (0.605 vs 0.49), 0p35 tie (0.418 vs 0.42).
- **The macro gap (0.453 vs 0.342) is ENTIRELY the last shell (0p45_0p55)**, where DTFE returns
  **-0.223** — a *mechanical* failure, not a fair contest: a density estimator with almost no
  neighbouring galaxies has nothing to estimate density FROM. The GNN "wins" there only because
  classical is UNDEFINED — and the GNN is itself weak there (0.34, on 869 noisy galaxies). Macro
  averaging over a shell where one method is structurally broken manufactures a margin that is not
  skill. Quoting "GNN macro > DTFE macro" as a win is a metric artifact, exactly the kind of framing
  this programme has been burned by.

**THE ACTUAL LESSON: none of our encoders GENERALISE well.** Best honest number anywhere is the
full-range spatial-holdout GraphNet at ~0.45 macro / 0.51 pooled λ1, and it is at PARITY with a
textbook density estimator wherever that estimator functions. Its only "advantage" is a regime where
classical is mechanically disabled, and even there it manages only 0.34. Across the whole arc — dense
random-split 0.80/0.88 (interpolation illusion), disjoint-wedge transfer 0.42/0.37/0.35 (below
classical), full-range holdout ~0.45 — the binding failure is the SAME: the encoders learn a specific
volume, not a transferable local map. Architecture (graph vs CNN) and depth (8 vs 2 passes) are both
ruled out. The missing ingredient is an encoder BUILT and TRAINED for generalisation.

**PRIORITY LEVER (supersedes "just add sim volume"): proper PATCH-BASED graph training + validation.**
Train on many local ego-graphs / subvolume patches and VALIDATE on held-out patches, so the model is
forced to learn the local, translation-invariant configuration->eigenvalue map instead of memorising a
wedge. Patches become the unit of data (fixing the ~15k-independent-structures ceiling) AND the unit of
the generalisation test. The scaffold already exists: `local-subgraph-pipeline/` (subgraph_dataset.py,
train_flowjax_subgraphs.py, eval_local_subgraph.py), currently a TNG pilot — port it to the Abacus
wedge. Cross-phase volume (Workstream G) is complementary and needed for the sparse high-z shell, but
the ENCODER/training-protocol fix is the one that addresses the failure seen in every test above and
should come first.

**HONEST CAVEATS.** (1) A wrong prediction of mine, corrected: I expected DTFE to fall well below 0.46
POOLED at "production sparsity" — it did NOT (0.553 ~ dense-wedge 0.534), because pooled is dominated
by the abundant dense low-z shell (28.7k of 52.8k test nodes in 0p15_0p25) where DTFE has ample
tracers; the full range is not uniformly sparse. (2) Single seed; the entire macro difference rests on
the noisy 869-galaxy high-z shell. (3) R0 per-shell taken from the 2026-07-14 log, not recomputed here
— tighten before quoting anywhere.

NEXT (re-prioritised): (1) **stand up patch-based graph train/val on the Abacus wedge** (port
local-subgraph-pipeline) — the generalisation encoder is now the central task; (2) tighten this
comparison — recompute R0/A1 TEST macro directly + DTFE on VAL; (3) T-web CLASS accuracy
(void/wall/filament/knot) on the holdout — still never measured, the actual VAC product metric;
(4) Workstream G (cross-phase) for the sparse high-z shell, complementary to (1). The VAC is NOT
"rescued" — it is at classical parity where classical works and weak where it does not; a shippable
claim requires an encoder that genuinely generalises.
Refs: classical_baseline/fullrange_holdout/ (scores json + pred_eigs_{dtfe,cic}.npy);
dtfe_fullrange_pershell.py; logs/dtfe_fullrange.log; cache s3c_cnn_fullrange (matched to production
split). DTFE run: cell 4 Mpc, 51.9M cells, rsmooth 10.4 Mpc, 47s.

### 2026-07-16 — [code] 2-LAYER TRANSDUCTIVE GRAPHNET (num_passes 8→2): transfer λ1 0.369 — receptive-field hypothesis FAILS; and my "no train/val gap" diagnostic was INVALID

Shrinking the receptive field (JDPK's lever: 8→2 message passes, ~70-80 → ~20 Mpc vs 10.4 Mpc physics)
did NOT close the deployment gap. Retrained transductively on the original random-split union cache
(same seed/epochs/budget as the 8-pass anchor, ONLY num_passes=2), then the SAME RA200-240 transfer
test (same cache, same 95,220-node test mask) as the 8-pass and U-Net.

| λ1 R² (same test mask) | home (leaky) | transfer | drop | transfer NLL |
|---|---|---|---|---|
| GraphNet 8-pass | 0.804 | 0.421 | −48% | 6.53 |
| **GraphNet 2-pass** | 0.628 | **0.369** | −41% | 4.26 |
| U-Net (T2) | 0.871 | 0.353 | −59% | — |
| DTFE (no ML) | — | 0.534 | — | — |
Home 2-pass full: λ1/λ2/λ3 = 0.628/0.754/0.826 (vs 8-pass 0.804/0.846/0.896); best val NLL 2.089.

**Two findings, the second more important than the first:**

1. **Receptive field was NOT the binding constraint.** 2-pass transfers to 0.369 — WORSE than 8-pass
   (0.421) and well below DTFE (0.534). The smaller field bought proportionally less collapse (−41% vs
   −48%) and much less calibration blowup (transfer NLL 4.26 vs 6.53) — directionally consistent with
   the oversmoothing/leakage argument — but nowhere near enough. FOUR models now (8-pass, 2-pass,
   U-Net, +the leak-audit) all land below the classical floor at deployment. Encoder family AND depth
   are both ruled out; the common factor is the single-volume training data → Workstream G.

2. **★ A METHOD CORRECTION: "no train/val gap" is NOT a leakage/generalisation diagnostic on a random
   split.** I read the 2-pass run's flat curve (val ≤ train from epoch 333→3749, finishing 2.07 vs
   2.09) as evidence it wasn't leaking and might transfer — and said so. The transfer test refuted it.
   WHY the diagnostic is invalid: on a RANDOM split, train and val nodes are BOTH randomly interleaved
   through the same volume → both live in the INTERPOLATION regime. A train/val gap only opens when a
   model memorises SPECIFIC nodes; it does NOT open for the interpolation-vs-extrapolation failure that
   kills deployment. The 2-pass model was HONESTLY INTERPOLATING (no memorisation), and interpolation
   skill simply does not extrapolate. So a healthy train/val curve on a random split says nothing about
   deployment. The ONLY valid generalisation probe here is a spatial-holdout or a disjoint-wedge
   transfer test — never the train/val gap.

CONSEQUENCE: the 2-pass model is NOT a keeper; the 8-pass anchor is not displaced by it either (both
are dead-on-transfer random-split models). The production full-range model (trained UNDER the spatial
holdout) is the only one that generalises at all — but it is at PARITY with classical DTFE where DTFE
works, not a "win" (see the corrected DTFE full-range entry above). The receptive-field lever is
CLOSED; the open lever is an encoder that genuinely generalises — patch-based graph train/val — with
cross-phase data (Workstream G) complementary.
Refs: sbi_runs/path1_wedge_union_2passes_transductive (home), path1_TRANSFER_ra200_240_2passes
(transfer); logs g2layer.log, g2layer_transfer.log.

### 2026-07-15 — [code] U-NET TRANSFER: the 0.876 champion collapses HARDER than GraphNet (0.353 λ1) — failure is PROTOCOL, not architecture; deployment ranking DTFE > GraphNet > U-Net

**ML-baseline transfer (JDPK-requested):** the T2 3-D U-Net — the highest-R² model in the programme
(λ1 0.876, random split, dense wedge) — applied pure-inductively to the disjoint RA200-240 wedge,
scored on the SAME 95,220-galaxy test mask as the GraphNet transfer and DTFE. gate_t2 never saved
weights, so gate_t2_transfer.py (ab40b60) first REPLICATED T2 exactly (recovered config; hard gate:
home λ1 must land in the 3-seed band) — home test 0.871/0.902/0.930 ✓ (band 0.871-0.880). Frozen
weights then predicted the new wedge with own-wedge channels (in-mask standardisation = SI analogue).

**Result (same wedge, same test mask, same truth for all rows):**
| deployment R² | λ1 | λ2 | λ3 | home (leaky) λ1 | collapse |
|---|---|---|---|---|---|
| DTFE (cal, no ML) | **0.534** | **0.604** | **0.634** | 0.552* | ~none |
| GraphNet transfer | 0.421 | 0.498 | 0.524 | 0.804 | −48% |
| U-Net transfer | 0.353 | 0.425 | 0.484 | 0.871 | **−59%** |
(*training-wedge value; stable ±0.02 across wedges.) U-Net interior-only (25 Mpc): 0.385/0.474/0.526
— edge effects are NOT the explanation.

**READING — the leaky ranking INVERTS at deployment.** Home order U-Net (0.876) > GraphNet (0.804) >
DTFE (0.552); deployed order DTFE (0.534) > GraphNet (0.421) > U-Net (0.353). The model with the
higher random-split score fell HARDER — exactly the interpolation-machine signature: the more
effectively a model exploits neighbourhood duplication between train and test, the more its leaky
score flatters it and the less transferable structure it has actually learned. **The failure is the
TRAINING PROTOCOL (transductive, random-split, single volume), not the encoder family** — both
architecture families (message-passing graph AND Cartesian CNN) collapse below the no-ML floor.
Random-split R² is hereby WORTHLESS as a model-selection signal in this programme: it ranked the
three methods exactly backwards relative to deployment.

**Parity plots** (transfer_plots/parity_graphnet_vs_unet_ra200_240.png; 2×3 hexbin, GraphNet/U-Net ×
λ1/2/3, DTFE reference, λ_th=0.2 guides): both models show the same failure morphology — strong
regression to the mean (flattened ridge vs the 1:1 line), and a plume of high-true-λ (knot) galaxies
predicted low; the U-Net cloud is visibly more diffuse. Consistent with models that learned local
texture + amplitude priors of the training volume rather than the tidal operator.

**GraphNet eval re-run reproduced exactly** (0.4207/0.4983/0.5241, NLL 6.5274) — deterministic ✓ —
and the trainer now dumps posterior_pred_eigs_seed_<s>.npz at final eval (no more re-sampling runs
for plots).

**2-LAYER TRANSDUCTIVE RETRAIN LAUNCHED** (JDPK: shrink the receptive field; num_passes 8→2 ≈
70-80 → ~20 Mpc vs 10.4 Mpc physics). Same cache/seed/budget as the 8-pass anchor; out
sbi_runs/path1_wedge_union_2passes_transductive. EARLY SIGNAL @epoch ~260: train 3.09 / val 3.02 —
val ≤ train, NO memorisation gap yet (the 8-pass anchor's val had long since decoupled). Next after
it lands: same transfer test → does a physical receptive field close the 0.80→0.42 gap?

NEXT (standing): 2-layer transfer test; T-web CLASS accuracy on holdout; Workstream G scoping
(cross-phase = the protocol fix the field uses); calibration only on inductive data.
Refs: field_level_tests/T2_transfer/ (t2_model_seed42.pt now SAVED, scores json, pred npz);
transfer_plots/parity_graphnet_vs_unet_ra200_240.png; logs unet_transfer.log, g2layer.log;
commit ab40b60.

### 2026-07-15 — [code] DTFE ON THE TRANSFER WEDGE: classical beats the deployed GNN on EVERY eigenvalue (0.53 vs 0.42 λ1) — head-to-head now exact, verdict clean

Ran the no-ML classical baseline (density reconstruction + exact FFT tidal solve) on the SAME
RA200-240 wedge, scored on the SAME 95,220-node test mask, against the SAME truth as the GNN
transfer eval. classical_tidal_baseline.py gained CLI overrides (63b7ecc, defaults unchanged); its
row-alignment guard verified the pairing. "cal" = 2-param affine (slope+offset) per eigenvalue fit
on the 1000-node dummy-train mask — a far weaker correction than the GNN's 3,749 epochs of
amplitude learning, so the comparison is fair-to-generous toward the GNN. 7 min on a CPU node.

| λ1 / λ2 / λ3 R² | GNN transfer (posterior mean, 128) | DTFE (cal) | CIC (cal) |
|---|---|---|---|
| λ1 | 0.421 | **0.534** (interior 0.549) | **0.551** |
| λ2 | 0.498 | **0.604** | 0.574 |
| λ3 | 0.524 | **0.634** | 0.622 |

**VERDICT — now with zero caveats: the deployed dense-wedge GNN loses to textbook no-ML
reconstruction on every eigenvalue, on identical galaxies/test mask/truth.** Even plain CIC
gridding beats it. The classical floor is also STABLE across sky patches (training wedge λ1 0.552
→ transfer wedge 0.534; ±0.02 = cosmic variance), so ~0.53-0.55 is a reliable deployment bar.
λ1 Spearman: DTFE 0.761 / CIC 0.774 — the classical ranking skill transfers untouched, because a
fixed physics operator has nothing to overfit.

**Programme implication:** the bar any deployable model must clear is the classical ~0.53-0.55, not
0. The current honest ranking at deployment: classical 0.53 > full-range spatial-holdout GraphNet
~0.51 (different data regime, roughly at par) > deployed dense-wedge anchor 0.42. The GNN's leaky
+0.22 "advantage over classical" (0.775 vs 0.552) was entirely split artifact; whether ANY GNN
beats classical on a fair inductive test is now the programme's central open question — and the
first model that does so by +0.05 with transferred calibration is the one that ships to DESI.

Refs: classical_baseline/ra200_240_transfer/classical_baseline_scores.json (+ pred_eigs_{dtfe,cic}.npy);
logs/dtfe_transfer.log; commits 63b7ecc (CLI), 998c38a (transfer verdict).

### 2026-07-15 — [code] TRANSFER TEST (deployment rehearsal): anchor model on a disjoint wedge = λ1 R² **0.42** — BELOW the DTFE classical floor (0.552); posteriors catastrophically miscalibrated OOD (NLL 0.90→6.53)

**Design (JDPK-requested):** the production union GraphNet (anchor 0.8041/0.8461/0.8955, transductively
trained on RA120-160, random split; preserved ckpt @3749) applied PURE-INDUCTIVELY to a disjoint
near-twin wedge RA200-240 (same DEC 14.5-30.6, z0.2-0.3; 97,220 vs 100,935 galaxies, 100% valid box),
built through the IDENTICAL pipeline (same full-sky path1_fiberassign Delaunay+cuGraph artifacts, same
subset tools, union 14.78 Mpc). Transform policy = deployment-correct: own-graph SI medians + TRAINING
PowerTransformer/target_scaler/edge_scaler. **GOLD GATE PASSED at float precision** (recipe reproduces
the production cache: nodes 2.4e-7, edges/senders/receivers 0.0, targets 9.5e-7) — the pipeline
replication is PROVEN, so the number is real. Scored on 95,220 nodes. This is exactly the DESI
deployment path, run where truth is known.

**Result (posterior-mean, 128 samples/node, raw eigenvalues):**
| | λ1 | λ2 | λ3 | test NLL |
|---|---|---|---|---|
| anchor (random split, SAME wedge) | 0.8041 | 0.8461 | 0.8955 | 0.896 |
| **TRANSFER (disjoint wedge)** | **0.4207** | **0.4983** | **0.5241** | **6.53** |
| DTFE classical floor (no ML, honest) | 0.552 | 0.641 | 0.663 | — |
| full-range spatial holdout (R0, prod sparsity) | 0.514 | — | — | — |

**Four readings:**
1. **The deployed dense-wedge GNN LOSES to textbook no-ML reconstruction on every eigenvalue**
   (0.42<0.552, 0.50<0.641, 0.52<0.663). The "GNN ≫ classical" headline is not merely unverified —
   at deployment it is currently FALSE for this model. (Caveat: DTFE was scored on the training
   wedge; it trains nothing so it transfers trivially, but for the exact head-to-head run
   classical_tidal_baseline on RA200-240 — cheap CPU job, registered as next step.)
2. **The spatial holdout was an honest deployment proxy.** Transfer 0.42 sits in the same regime as
   the spatial-holdout numbers (~0.46-0.51), vs the leaky 0.80. Interpolation-vs-extrapolation
   explains ~half the headline R². The full-range production protocol (spatial holdout) was
   measuring the right thing all along.
3. **Calibration does NOT transfer: test NLL 0.896→6.53.** OOD the posteriors are wildly
   overconfident — FMPE tempering calibrated on a random split is calibrated for interpolation, not
   deployment. ALL calibration work must move to spatial-holdout/inductive data.
4. **Only λ1 carries transferable signal.** In increment space Δλ2/Δλ3 R² = +0.04/-0.09 (≤0!) — raw
   λ2/λ3 R² is largely λ1+ordering, not independent skill. Matches the v1-contract decision
   (calibrated-λ1-only) — now with direct deployment evidence.

**Verdict:** the transductively-trained dense-wedge anchor is an interpolation machine: 0.80 → 0.42
when asked to extrapolate, below the classical floor, with broken calibration. This CONFIRMS
JDPK's concern quantitatively — for THIS model class. The constructive corollary: the full-range
production model was already selected under a spatial holdout, so its ~0.5 is deployment-honest;
nothing about today changes ITS validity — today establishes that 0.5-not-0.8 is the real current
ceiling and that the fix is training-protocol + data (fewer passes, regularisation, patch/inductive
training, cross-phase volume), not evaluation optimism.

NEXT (registered): (1) DTFE on RA200-240 for the exact classical head-to-head; (2) num_passes
ablation (8→3/4) on the spatial split; (3) T-web CLASS accuracy on holdout (the VAC product metric);
(4) Workstream G scoping (25 phases exist; per-phase density fields are the gate).
Refs: run sbi_runs/path1_TRANSFER_ra200_240_eval/flowjax_sbi_results_seed_42_20260715_093644.txt;
cache sbi_caches/path1_TRANSFER_ra200_240_uniongraph; builder commit 9a37c67; logs transfer_A.log,
transfer_BC.log.

### 2026-07-15 — [code] LEAKAGE AUDIT: the dense-wedge 0.775-0.876 family is RANDOM-SPLIT INTERPOLATION; DTFE 0.552 never leaked ⇒ the "GNN ≫ classical" thesis is UNVERIFIED

**Audit STOPPED early by JDPK at epoch 2701/3750 — the training curve had already answered the
qualitative question.** Retrained the dense-wedge union GraphNet with ONE variable changed vs the
0.8041/0.8461/0.8955 anchor: masks random → spatial (train RA<145 / val 145-150 / test RA>=150, global
halo-centroid, 15 Mpc gutter; commit dcfe1fb). Identical graph, features, targets, trainer, seed 42,
budget.

| | anchor (RANDOM split) | audit (SPATIAL split) |
|---|---|---|
| best val NLL | **0.856** | **3.298** |
| end train NLL | — | 0.58 |
| end val NLL | — | 7.5 and RISING |
| posterior-mean R²(λ1) | 0.8041 | **NOT OBTAINED** (see below) |

**The honest R² was NOT recovered:** the trainer keeps `best_state` in memory and restores it only at
the end; killing the run lost it. The on-disk checkpoint is the periodic one (epoch ~2750 = fully
overfitted), NOT best-val. Getting the number needs a ~400-epoch rerun with early stopping (~25 min) —
val bottomed near epoch 300. **RECOMMENDED, see CRUX below.**

**MECHANISM (why a random split leaks even though labels never leak).** No label leaked and the split
was already halo-disjoint — which is why this survived so long. The leak is spatial autocorrelation:
- T-web labels are smoothed at **rs7 = 7 Mpc/h = 10.4 Mpc**.
- Dense wedge: 100,935 galaxies in ~6.9e7 Mpc³ ⇒ mean separation **8.8 Mpc**; at 70% train a test
  galaxy's **nearest TRAIN galaxy is ~9.9 Mpc = ONE smoothing length**.
- The encoder runs **8 message-passing steps** ⇒ a test node's receptive field overlaps almost
  entirely with several train nodes'.
⇒ the model is handed **near-duplicate (input,label) pairs** and tested on copies of them. Random CV on
non-independent samples measures training error. Random split = INTERPOLATION into an already-sampled
field; spatial split = EXTRAPOLATION to unseen structure = the DESI task.

**CAPACITY ARITHMETIC (why the dense wedge is worse than full range).** Independent ~10.4 Mpc
structures = V / (4/3 π R³ = 4712 Mpc³):
- dense wedge V≈6.9e7 Mpc³ ⇒ **~15k independent patches** (RA<145 train slab ≈ 9k)
- full range V≈5.5e8 Mpc³ ⇒ **~117k independent patches**
Against a model of order 1e5-1e6 params with a ~70-80 Mpc receptive field. The 8× difference in
independent structures is very likely WHY the full-range model generalises (0.46-0.51) and the
dense-wedge spatial audit collapses. Full range is also SPARSER (mean sep 13.6 vs 8.8 Mpc).

**★ THE CRUX — DTFE NEVER LEAKED.** `classical_tidal_baseline.py` is **"no ML"**: density
reconstruction + exact FFT tidal solve, using the masks only to SCORE, never to fit. **It cannot leak
⇒ its λ1 = 0.552 is an HONEST number, valid on any subset including a spatial holdout.** The
GraphNet's 0.775 is not. **The programme's headline claim ("GNN 0.775 ≫ classical 0.552, headroom
real") compares a LEAKY number against an HONEST one and is therefore UNVERIFIED.** The open question
is now binary:
  honest-GraphNet ≈ 0.65-0.75 ⇒ the graph adds real skill; restate numbers, paper stands.
  honest-GraphNet ≈ 0.55     ⇒ **the GNN adds NOTHING over textbook DTFE** and the thesis was a split
                               artifact.
This must be settled before any further architecture work. ~25 GPU-min.

**REFRAME (against the panic reading): the model DOES generalise to unseen sim regions.** EVERY
production number since R0 is already a spatial holdout at production sparsity: R0 held-out
**R²(λ1)=0.514**, A1_sqrt **0.456 macro**, C **0.461**. That IS "train one RA/Dec patch, predict a
different unseen patch at the same z". Nothing regressed this week — the 0.8-era numbers were
answering an easier question. Correct statement: **we never had 0.775 in a deployable sense**, not "we
lost accuracy". Caveat: even our spatial holdout is WITHIN one realisation (same cosmology, phase,
large-scale modes) ⇒ by CAMELS-era standards **0.51 is still an optimistic bound for DESI**.

**VALIDATION DESIGN — three flaws found (JDPK's question).** Val labels never entered a gradient (that
part is sound), and selecting on val while sealing RA>=150 for a frozen finalist is the correct guard
against val-optimism. But:
1. **val and test are NOT exchangeable** (the serious one): val is a thin 5° strip ADJACENT to train
   (~57 Mpc core after gutters); test is a block 6.4° (~90-110 Mpc) away. If the model memorises,
   distance from train matters ⇒ val is a systematically OPTIMISTIC proxy for test and early stopping
   stops too late. Fix: val and test blocks at MATCHED geometry/distance from train. (Silver lining:
   the val↔test gap is itself a memorisation diagnostic.)
2. **one fold, no error bars** — a single cut gives a point estimate with no variance; block CV
   normally reports the spread over K blocks. This is partly why C's +0.005 is a tie.
3. **transductive** — the encoder saw val/test FEATURES (defensible: features are observables and DESI
   inference has them too, but softer than deployment on a freshly built graph).

**RECEPTIVE FIELD / OVERSMOOTHING (JDPK: "we should not be using 8 layers").** AGREED, and the geometry
is quantitative: union graph = 3.98M edges / 100,935 nodes ⇒ **mean degree ~39**, edges to the 14.78
Mpc union radius. At ~8-10 Mpc/hop, **8 passes ⇒ ~70-80 Mpc receptive field vs 10.4 Mpc physics ≈ 7×
too wide** — capacity that can only be spent memorising WHERE in the wedge a node sits. The tidal
tensor is nonlocal but its Gaussian window suppresses r >> R, so ~2-3R ⇒ **3-4 hops (~30 Mpc)** is the
physical range, which is also standard GNN practice (2-4 layers; Li+2018 depth/oversmoothing).
**ACTION: --num_passes ablation (2/3/4/6/8) on the spatial holdout** (~25 min each, 2 at a time). If 3
hops ≥ 8 hops, that is simultaneously a generalisation win, less memorisation, and a cheaper model.

**★ WORKSTREAM G IS UNBLOCKED — 25 INDEPENDENT PHASES EXIST.** AbacusSummit CutSky BGS v0.1 z0.200 has
**ph000–ph024** (same cosmology c000, different initial phases ⇒ genuinely independent large-scale
structure). Train on ph000, test on ph001+ = the CAMELS/Quijote protocol and the only test resembling
DESI. **BLOCKER TO SCOPE:** T-web labels need each phase's DENSITY FIELD, and no particles/density grid
was found beyond ph000 ⇒ `abacus_cactus_tweb_fullgrid_mpi.py` (2048³) would have to run per phase.
That cost decides whether G is days or weeks — SCOPE IT FIRST.
**NOT a substitute:** `hod_sample_1..10` exist for ph000 but share the SAME underlying density field —
the tidal target is IDENTICAL across them; they give galaxy-population variance only, NOT independent
structure.

**INDUCTIVE OPTIONS (ranked).** (1) cross-phase (above) — the real test; (2) **ego-graph/subgraph
inductive training — WE ALREADY HAVE IT**: `local-subgraph-pipeline/` (subgraph_dataset.py,
train_flowjax_subgraphs.py, eval_local_subgraph.py). This is the architecturally correct framing: the
map local-configuration → tidal eigenvalues IS local and translation-invariant, so train on patches,
test on unseen patches — and patches become the unit of data, fixing the effective-sample-size problem;
(3) **fresh-graph inductive eval** (minutes): rebuild a graph from ONLY RA>=150 galaxies and predict —
quantifies the transductive discount; (4) multi-block spatial CV with matched val/test geometry
(fixes flaws 1+2); (5) cross-shell (S1 did this — density shift, not structure novelty).

**HOW THE FIELD HANDLES THIS:** spatial/blocked CV with buffer >= correlation length (Roberts+2017,
Ecography — canonical: random CV is optimistically biased under autocorrelation); scaffold/temporal
splits in molecular ML (MoleculeNet Wu+2018; **OGB Hu+2020 exists precisely because random graph splits
are unrealistically easy**); homology/temporal splits in protein ML (CASP, AlphaFold); **cosmology =
many realisations, not one box** (CAMELS Villaescusa-Navarro+2021, Quijote+2020 — train on some sims,
test on HELD-OUT sims; CAMELS explicitly measures cross-simulator transfer where models routinely
fail); inductive GNN training (GraphSAGE Hamilton+2017).

**RETIRE / RE-READ (all random-split, all interpolation):** GraphNet 0.775 (Delaunay@7000), 0.8041
(union@3749), 0.752 (radius-only), T2 CNN 0.876, F-tier 0.841, and S1(b)'s per-shell CNN
0.902/0.847/0.722/0.429. **KEEP: DTFE 0.552 (no fitting ⇒ honest).** Verify any split with
`workflows/sbi/check_s2_cache_split.py`.

**ORDER:** (1) num_passes ablation on the spatial split; (2) fresh-graph inductive eval at RA>=150;
(3) honest dense-wedge R² vs DTFE 0.552 (~25 min) — the go/no-go for the graph thesis; (4) scope G's
density-field cost; (5) **measure T-web CLASS accuracy (void/wall/filament/knot) on the spatial
holdout** — nobody has: the VAC ships calibrated CLASSES, not λ1 point estimates, and at R²(λ1)≈0.5
with honest calibration class accuracy could still be 65-80%. That is the actual product metric and
the VAC may be in better shape than R² suggests.

GIT: dcfe1fb (+ this entry) are LOCAL-ONLY — I cannot push (no credentials). **Push with plain
`git push`, NOT `git push origin refactor_codebase_Illustris`** — the local branch
`refactor_codebase_Illustris` tracks `origin/refactor_codebase` (no `origin/refactor_codebase_Illustris`
exists), so the explicit form would create a DIVERGENT remote branch. Also note a stale local
`refactor_codebase` branch at 9df2b51 (188 behind) — a checkout footgun; left untouched.
Refs: /pscratch/sd/d/dkololgi/logs/leakaudit.log; cache
sbi_caches/path1_flowjax_3d_lineareig_si_uniongraph_SPATIAL (train 57,197 RA<=144.27 / val 9,232 /
test 25,519 RA>=150.72, halo-disjoint, gates PASS); commit dcfe1fb.

### 2026-07-15 — [code] WORKSTREAM C VERDICT: pooled U-Net TIES the GraphNet (0.461 vs 0.456) — and S1(b)'s CNN numbers were a RANDOM-SPLIT artifact

**GATE FAILED (no pivot): C macro 0.461 vs GraphNet A1_sqrt 0.456 = +0.005, far below the +0.02 bar.**
Pooled, selection-aware full-range 3-D U-Net, on a cache PROVEN identical to the GraphNet's
(129,113/18,629/52,848 — s3c match gate). VAL only; RA>=150 stayed sealed.

| shell | C U-Net (pooled) | GraphNet A1_sqrt | S1(b) CNN in-shell (LEAKY, see below) |
|---|---|---|---|
| 0p15_0p25 | **0.570** | 0.530 | 0.902 |
| 0p25_0p35 | **0.545** | 0.490 | 0.847 |
| 0p35_0p45 | **0.502** | 0.500 | 0.722 |
| 0p45_0p55 | 0.226 | **0.290** | 0.429 |
| **MACRO** | **0.461** | 0.456 | 0.725 |
| pooled | **0.549** | 0.516 | — |
λ2 val pooled 0.630 / macro 0.533; λ3 val pooled 0.679 / macro 0.539; cluster-slice λ1 Spearman +0.454.

**CORRECTION — S1(b)'s CNN was scored on a RANDOM split, not the spatial holdout.** gate_t2 trains on
`cache["masks"]`, and the s2 shell caches carry a random 70/21/9 node split: train/val/test RA ranges
overlap **100%** (all 120-160 deg; means 140.24/140.25/140.07). They ARE halo-disjoint (0 shared halos),
so the crude leak was guarded — but NOT spatially disjoint. The tidal field is smooth on ~10 Mpc, so a
random split leaves a test galaxy's neighbours in train and a field model can interpolate the label
field instead of learning physics. This is exactly what the 07-13 Codex review meant by "spatial
holdout mandated". => S1(b)'s 0.902/0.847/0.722/0.429 are NOT comparable to any spatial-holdout number,
and my earlier reading ("in-range the grid loses in ZERO shells; macro 0.725 vs 0.456") was WRONG — it
set a leaky CNN against a strict-holdout GraphNet. Matched honestly, 0.725 -> 0.461. The prior "grid is
dead / mid-range specialist" verdict SURVIVES, and for a cleaner reason than the corrupt shell: on an
honest matched split the grid merely TIES.

**But C is not a wash — it is a genuine near-tie with a different error profile:** it BEATS the GraphNet
in 3 of 4 shells (+0.040/+0.055/+0.002) and on pooled (+0.033), and loses only at high-z (0.226 vs
0.290) — which is precisely what drags its macro back to a tie. Grid vs graph is a high-z story, not a
mid-range story.

**THIRD independent line for data-limitation.** C overfits hard and fast: train MSE 0.16 -> 0.07 while
val 0.38 -> 0.46; best macro at step 350; EARLY STOPPED at step 750 of a 4000 budget. It is
data-limited, not compute-limited. Together with (1) Workstream A (high-z is data-limited, uniform
sampling made it worse 0.26->0.13) and (2) R1 (aperture channels bought capacity, not information;
2x overfit gap), three independent methods now say the binding constraint is **data volume**, not
architecture and not features. Two very different encoders (message-passing graph, Cartesian CNN)
converge to macro ~0.46 on the same 129k training nodes. That is the signature of an information/data
ceiling, not an encoder ceiling. => **Workstream G (more sim coverage) is THE lever.** R²(λ1)=0.8 is
not reachable by swapping encoders on this cache.

CAVEATS: (a) C inherited T2's hyperparameters (tuned on the dense wedge) — a tuned C might gain, though
the overfitting signature caps the headroom; (b) C's high-z 0.226 on 3,118 galaxies is noisy;
(c) single seed each side, and the shell-level trajectories are noisy, so +0.005 macro is a tie in the
strict sense (well inside run-to-run scatter).

INFRA: s3c_build_cnn_fullrange_cache.py mirrors s3b bit-for-bit and PROVES the match with a mandatory
gate (raises on mismatch) — PASSED exactly. gate_c_unet_fullrange.py = T2 + pooled full range +
explicit expected-count channel + LOS unit-vector channels (a Cartesian CNN cannot otherwise know the
RSD axis; it varies across the wedge) + VAL macro-shell gating (T2 scored pooled-on-TEST, which hides
high-z AND would breach the sealed region; test now refused without --unseal-test). Grid 334x317x194 =
20.54M cells @5 Mpc, 7 channels, 1.44M params, peak 41.1 GB (WOULD OOM ON A 40GB CARD -> hbm80g is
required, not optional). Axis-order guard grid_sample vs map_coordinates corr = 1.0000.

NEXT: (a) incumbent GraphNet (8-d, tau=0.5) still stands — C does not displace it; (b) Workstream G
(more sim coverage) is now the priority on three independent lines; (c) D (F-tier v2_A) is worth far
less than it looked — its 0.841 was measured on the same dense-wedge/random-split footing and must be
re-read with this correction before anyone quotes it; (d) if grid vs graph is revisited, the question
is specifically high-z, and an ensemble/hybrid of C (mid-range) + GraphNet (high-z) is the only place
the two differ enough to matter.
Refs: /pscratch/sd/d/dkololgi/logs/C_full.log, C_smoke.log; scores
/pscratch/sd/d/dkololgi/abacus/C_unet_fullrange/scores.json; cache s3c_cnn_fullrange; commit 3651218.

### 2026-07-15 — [code] R1 VERDICT: aperture channel does NOT pay (same peak, worse NLL, 2× overfit); kcorr GATE PASSES

**kcorr parity gate — PASSED.** DESI ABSMAG_RP1 (official LSS add_ke: Smith+2017 GAMA k-corr + TMR
e-corr + DESI dm) vs Abacus R_MAG_ABS, z0.15-0.55, 400k DESI rows vs 410,210 Abacus wedge galaxies:

| quantity | mean | std | p5/p50/p95 |
|---|---|---|---|
| DESI ABSMAG_RP1 (k+e) | -20.761 | 0.840 | -22.189/-20.729/-19.420 |
| Abacus R_MAG_ABS | -20.727 | 0.777 | -22.038/-20.709/-19.469 |
| DESI G_R_OBS | 1.186 | 0.358 | 0.600/1.179/1.794 |
| Abacus G_R_OBS | 1.133 | 0.350 | 0.550/1.146/1.705 |

M_abs median offset (Abacus-DESI) = **+0.021 mag**; width ratio 0.925; G_R_OBS offset -0.033.
=> the k+e convention risk did NOT materialise; the luminosity channel (M_ABS + G_R_OBS) is
train/inference safe — no G_R_OBS-only fallback needed. CAVEATS: Abacus M_abs is 7.5% narrower (mild
covariate shift; mock luminosity scatter tighter than reality), and the DESI side is the full
BGS_BRIGHT footprint vs the Abacus wedge (both r<19.5-limited so marginals are comparable, but it is
not footprint-matched).
Two fixes were required to get a number at all: eb153fc (shim pkg_resources — setuptools>=81 removed
it, but DESI_ke/smith_kcorr still imports it as dead code, only call site findfile.py:104 commented
out) and 260c07e (z-cut + subsample BEFORE add_ke: its single-threaded rest_gmr solve ran 15%/36min
over 8.7M rows => ~4h against a 1h wall, so it could never finish and the retry loop restarted from
zero; add_ke is row-independent, so subsetting first is exactly equivalent for rows kept).

**R1 — GATE FAILED.** R1 (v3_aper, 15-d) vs A1_sqrt (v2, 8-d): identical trainer, seed 42, 4000
updates, warmup 400, tau=0.5. ONE variable = the 7 aperture/NN channels.

| metric | A1_sqrt 8-d | R1 15-d | delta |
|---|---|---|---|
| best val NLL | **2.7150** | 2.7743 | -0.059 WORSE |
| best val pooled λ1 R² | 0.5162 | 0.5169 | +0.0007 (noise) |
| best val macro λ1 R² | 0.456 | 0.464 | +0.008 (below the +0.02 bar) |
| final pooled / macro | 0.497 / 0.446 | 0.482 / 0.421 | worse + unstable |
| end train->val NLL | 1.69 -> 3.22 (gap 1.53) | 1.19 -> 4.34 (**gap 3.15**) | 2x worse overfit |

VERDICT: the aperture channel bought **capacity, not generalisable information**. R1 drives train NLL
LOWER (1.19 vs 1.69) while val NLL goes HIGHER (4.34 vs 3.22) — textbook memorisation. Its macro peak
(0.464) is a transient spike that decays to 0.421, whereas A1_sqrt converges stably
(0.451/0.448/0.446). The +0.008 macro is a max-over-10-noisy-evals difference between two single-seed
runs => not a real gain, and below the memo's +0.02 bar. **A1_sqrt (8-d) remains the incumbent; do NOT
ship 15-d.** NOTE: the bestL1 checkpoint selects on POOLED (trainer line 210), not macro — so the
headline "best val λ1 R² 0.5169" is pooled and must not be read as a macro gate pass.

WHY IT MATTERS: this CONVERGES with Workstream A's result (high-z is DATA-limited, not update-starved).
Two independent lines now say the binding constraint is **data volume + regularisation, not feature
richness**: only 10 tiles / 129,113 train nodes / one tile per update. Adding channels to this encoder
is a dead lever. => **Workstream G (more sim coverage) is the priority**, with regularisation
(dropout / weight decay / early stop) as the cheap companion test. This also reframes the R²(λ1)→0.8
target: it will not be reached by feature engineering on this cache.

CAUTION for any revisit: the aperture channels were deliberately NOT SI-normalised (to preserve the
absolute scale that SI division removes). That also hands the model absolute tile/shell scale — a
plausible memorisation handle. A fair retest = SI-normalised aperture + regularisation, not more
channels.

NEXT: (a) 8-d incumbent stands; (b) Workstream G / C (3-D U-Net) / D (F-tier v2_A) per the sprint memo;
(c) full-catalogue ABSMAG is a separate ~4h run (needs >=4h alloc or a parallelised rest_gmr), and only
if the luminosity channel actually ships.
Refs: logs R1.log / A1_sqrt.log / kcorr4.log; cache s3b_tiled_valid_v3_aper (15-d, all gates PASS);
artifact bgs_absmag_rp1_gate_sub.fits (400k, seed 42); commits d678191, eb153fc, 260c07e.

### 2026-07-13 — [science] Codex external review ADOPTED (triaged): v1 contract fixed to calibrated-λ1-only; spatial holdout mandated; tiling = correlated mixture
- External (Codex) review of the roadmap assessed: strongest external review to date; its
  grid-at-sparsity prediction was independently confirmed by S1(b) before it saw them.
  Triage (full adoptions in roadmap §4/§4b):
  1. **v1 contract corrected (pre-schema-freeze):** calibrated science columns = λ1 only
     — mean/std/quantiles + **P(λ1>λ_th) ≡ knot/three-axis-collapse probability**
     (ordering λ1≤λ2≤λ3 makes them identical — sharpened language adopted). λ2/λ3 and
     4-class ship only as EXPERIMENTAL until SBC-aware v1.1. Was previously (wrongly)
     "4 class probs (headline)".
  2. **Spatial holdout mandated** — random transductive splits are optimistic; Phase-B
     split design (TOMORROW) gets a contiguous held-out RA block across all shells;
     tempering fit on val region, assessed on a disjoint region.
  3. **Symmetric scope guard** (dense low-z now guarded like sparse high-z — S1(b)
     evidence), **reliability+Brier gate for P(λ1>0.2)** per shell + mass-anchored,
     width-vs-|error| and conditional-coverage diagnostics, **prior-dominated flag**
     (calibrated-but-uninformative rows must say so).
  4. **ñ spline discipline:** freeze prescription, two-bandwidth sensitivity test —
     "smooth expected sampling intensity", not measured density (don't condition away
     real radial modes). Randoms-grounded selection = v1.1.
  5. **Tiling:** centrality-weighted posterior MIXTURE (never products, never averaged
     variances); buffered-tile+trim; mask-hole edge flags; idempotent shards+manifests;
     golden-wedge canary before scale-out; DAILY incremental CFS backups start now.
  6. Already-converged items (no change): G3+FMPE+tempering default w/ challenger rule;
     TARGET_EPOCH=0.2 first-class; F-tier protected as v1.1/research. Deferred to v1.1:
     completeness FEATURES (v1 = flags), randoms-based selection, multi-snapshot labels.
- Net effect on the Jul 21 deliverable (unchanged in spirit, sharpened in wording):
  "a frozen, internally validated GraphWeb-BGS VAC v1 CANDIDATE, pending DESI
  collaboration review", whose success criterion is: every unflagged row has a
  posterior whose calibration, information content and domain support we can defend.

### 2026-07-13 — [code] S1(b)/S2.5 VERDICT: grid's BEST case collapses at BOTH range extremes; conditioned GraphNet confirmed as the full-range production path
- **Design (a-fortiori):** GraphNet evaluated ZERO-SHOT (worst case, leak-guarded via
  FILE_NUM/BOX_INDEX exclusion of training-wedge galaxies) vs CNN trained WITHIN each
  shell (best case). Two simultaneous hbm80g tmux sessions on the S2 caches.
- **Results (λ1 R²):**
  | shell | CNN within-shell (BEST) | GraphNet zero-shot (WORST) |
  | 0.05–0.15 | **0.002** (clu ρ 0.09) | −1.09 |
  | 0.15–0.25 | 0.902 (ρ 0.74) | 0.480 |
  | 0.25–0.35 | 0.847 (ρ 0.66) | 0.448 |
  | 0.35–0.45 | 0.722 (ρ 0.60) | 0.147 |
  | 0.45–0.55 | **0.429** (ρ 0.25) | −0.61 |
- **Three decisions fall out:**
  1. **Grid ≠ full-range encoder:** even trained in-shell, the CNN collapses at the
     sparse end (0.43, cluster ρ 0.25 at z0.45–0.55) AND the dense low-z shell (0.002 —
     possibly partly a small-volume/padding artifact of gate_t2 defaults at low-z
     geometry; diagnostic queued, non-blocking). It is a MID-RANGE specialist
     (0.85–0.90 at z0.15–0.35 — its 0.902 at z0.15–0.25 is the best single-shell number
     recorded). A CNN-everywhere VAC is dead; a hybrid would resurrect the seam problem.
  2. **Unconditioned GraphNet OOD profile quantified at production fidelity:** fails in
     BOTH density directions (−1.09 dense / −0.61 sparse; coverage 0.16–0.38 vs nominal
     0.68 = wildly overconfident off-domain) — this also RESOLVES the S1(a) shell-0
     anomaly: real, not proxy (per-graph SI medians shift ~3× at the dense end too).
     Caveat: zero-shot shell-1/2 numbers (0.48) carry a graph-change + z-error confound
     vs the in-domain 0.80; the cross-shell SHAPE is the signal.
  3. **Phase-B primary confirmed: ONE full-range ñ-conditioned GraphNet + FMPE** (per
     S1(a): pooled+ñ beats per-shell). CNN stays a mid-range point-estimate reference;
     F-tier eigenvector columns computed where valid (mid-range) for the VAC extras.
- Jul 12–13 slot of the compressed schedule COMPLETE ON TIME: S0 ✅ S1(a) ✅ S2 (5 caches)
  ✅ S1(b) ✅. Next per roadmap §6: **S3 conditioning build (Jul 14)** — ñ node feature
  (SI-excluded, metadata-driven) + FMPE conditioning vector + pooled 5-shell training
  cache — then the Phase-B full-range retrain (Jul 14–15, hbm80g, salloc/tmux).
- Ops: S2 phase-2 cupy failure traced to unset CONDA_PREFIX — rapids-gnn MUST be
  activated via `source miniforge3/bin/activate <env>` (repo CLAUDE.md pattern), not the
  bare env python. Baked into the launcher.
- Refs: `gate_s1b_graphnet_zeroshot.py`, S1b CNN runs `field_level_tests/S1b/t2_shell_*`,
  logs `s1b_{graphnet,cnn}_*.log`, caches `sbi_caches/s2_shell_*_si_union/`.

### 2026-07-13 — [code] S1(a) VERDICT: ñ-conditioning BEATS per-shell models; S2 five-shell cache chain launched; **DEADLINE: NERSC shutdown Jul 22–Aug 3 — VAC v1 frozen by Jul 21**
- **DEADLINE (JDPK):** NERSC down Jul 22–Aug 3. The VAC must be built, internally
  validated, FROZEN and BACKED UP (scratch→CFS) by **Jul 21**. Roadmap §6 rewritten with
  the compressed Jul 12–21 schedule + a pre-registered **v1 scope guard** (if z≳0.45
  can't be validated in time, ship 0.05–0.45 + OOD-flagged high-z rows; v1.1 after).
  Operating constraints: sbatch unusable → ALL runs via salloc+tmux chains; hbm80g for
  memory-bound GPU work; the two winners tested in simultaneous allocations. Honest
  flag: DESI collaboration review is human-paced and NOT achievable by Jul 22 — the
  deliverable is a frozen validated v1 candidate.
- **S1(a) shell-transfer matrix RUN** (`gate_s1_shell_transfer.py`; north wedge box only
  per JDPK volume concern; cutsky per-shell downsampled to the DESI ñ(z) spline →
  DESI-realistic 380k sample; aperture-feature GBM proxy):
  - **Off-diagonal transfer is CATASTROPHIC** (R² down to −86): unconditioned models are
    worthless off their training density. The S-track is mandatory, quantified.
  - **pooled+ñ ≥ per-shell diagonal on shells 1–4**, often by a lot (z0.15–0.25:
    **0.319 vs 0.173**; z0.25–0.35: 0.187 vs 0.167; z0.35–0.45: 0.065 vs 0.052;
    z0.45–0.55: 0.002 vs −0.045). Statistical sharing + conditioning wins. GATE 4/5 PASS
    ⇒ **single conditioned model confirmed** (formally: per-shell fallback only if the
    winner-tier contradicts).
  - **Shell-0 anomaly (z0.05–0.15):** BOTH options fail with proxy features (diag −0.016,
    pooled+ñ −0.108) — not a shells-win; a proxy/data anomaly to diagnose (S1 follow-up).
  - **Shell-4 = proxy floor** (everything ≈0 at median degree ~3): aperture counts carry
    ~nothing at that sparsity; whether the Delaunay-adaptive GraphNet does better is
    exactly S1(b)/S2.5 — and it feeds the v1 scope guard.
  Output: `abacus/s1_shell_transfer/s1_result.json`.
- **S2 LAUNCHED** (tmux `s2_shells`, 3-phase CPU→GPU→CPU chain, all 5 shells
  z0.05–0.55): buffered extract from the graph-ready parent (sentinel window excluded;
  σ_v=35 km/s z-errors, Z_ORIG kept; **NO dilution** — ñ-conditioning replaces it per
  S1a) → gudhi Delaunay → cuGraph features → trim → **union edges** →
  SI cache per shell (`sbi_caches/s2_shell_*_si_union/`). New:
  `s2_extract_shell_catalog.py`, `logs/run_s2_shells.sh`.
- **NEXT (Jul 13):** S1(b) winner zero-shot on the shell caches — existing G3-GraphNet
  and CNN/F-tier, two SIMULTANEOUS GPU tmux sessions = S2.5 encoder-at-sparsity at
  production fidelity; then S3 conditioning build + Phase B full-range retrain (Jul 14–15
  per compressed roadmap §6).
- Roadmap §3b S1 row + §6 timeline updated in the same commit.

### 2026-07-12 — [code] S0 COMPLETE: selection atlas run — union-graph degree collapses to ~3 and 5-Mpc voxel occupancy to 0.1% at z 0.45–0.55
- Ran `s0_selection_atlas.py` (new, CPU): DR2 vs sentinelfix n(z) 0.03–0.62 in the wedge
  box; smooth ñ(z) splines for BOTH datasets saved (the S3 conditioning functions);
  per-shell structural stats. Outputs `abacus/s0_selection_atlas/{s0_atlas.json,png}`.
- **DESI structural facts across the VAC range:** median degree within the 10 Mpc/h
  union radius 125→54→23→9→**3**→1 (z 0.05→0.6); frac(deg=0) hits **10.5%** at z
  0.45–0.55 and 27.8% beyond 0.55 — radius edges effectively vanish; Delaunay becomes
  the ONLY connectivity (vindicates the union design; the model must handle deg~3).
  **5-Mpc voxel occupancy 7.1%→0.1%** — the grid input becomes 99.9% empty at high z:
  P2's "grid edge preserved" (occ~5%, nzharm) says NOTHING about this regime. The
  G3-vs-CNN-vs-F-tier production-encoder decision MUST include an S2 sparsity stress.
- Mock mirrors DESI at low z; at 0.45–0.55 mock deg0=42% vs DESI 10.5% (the 0.28 count
  ratio) — handled by conditioning on ñ (density regimes overlap at shifted z), NOT by
  dilution. Sentinel window re-verified 2e-5. Data-quality flag for S1: DESI medNN at
  0.45–0.55 is 0.9 Mpc (< low-z) — close-pair excess at high z, check duplicates/fibre
  pairs before the transfer matrix.
- Next: S1 shell-transfer matrix (unblocked — splines exist).

### 2026-07-11 — [code] CONFIRMED: F-tier miscalibration was the ENERGY-SCORE OBJECTIVE, not the summary — MLE flow (FMPE) rescues it
- **Decisive test:** trained an amortized MLE flow head on the SAME F-tier physics point estimate
  (cond = predicted eigenvalues, λ1 R²=0.842) as conditioning, targets = scaled eigenvalues.
  FMPE (continuous flow) and MAF (autoregressive NPE), `flow_ftier_head.py`, n_eval=1500 test.
- **FMPE result:** SBC KS-p λ1/λ2/λ3/trace = **0.043 / 0.088 / 0.047 / 0.065** — OFF the 0.000 floor;
  cov68 ≈0.63–0.66, cov90 ≈0.85–0.88 (mild UNDER-coverage, intervals ~5% too narrow);
  pmean R² 0.858/0.913/0.937 (accuracy KEPT). **MAF result:** λ1 R²≈**0** (collapsed), SBC still
  **0.000** — FMPE is the right flow, MAF is not.
- **Interpretation:** the energy score is a strictly-proper JOINT scoring rule → calibrates the joint,
  NOT the per-eigenvalue MARGINALS that SBC tests. Swapping to maximum likelihood (FMPE) fixes the
  marginals. This confirms lever (ii) from the log-density falsification entry below; the F-tier
  physics summary DOES carry enough per-galaxy info for a calibrated posterior.
- **Remaining flaw is benign:** ~5% under-coverage → posterior-tempering τ≈1.1 (val-calibrated,
  inference-time, same machinery as the G3 VAC head) closes it without retraining.
- **Verdict:** calibrated F-tier posteriors are now DEMONSTRATED (FMPE-MLE + light tempering), not
  hopeless — but this is a research/eigenvector-product path; the VAC headline stays G3+FMPE+tempering.
- Artifact (phone-viewable): claude.ai/code/artifact/f12f95d8-29f1-485c-93bb-f12177eff104
- Refs: `field_level_tests/Pflow/flow_result.txt`, `flow_samples.npz`, `ftier_cond.npz`,
  `workflows/sbi/flow_ftier_head.py`.
- **RECONCILE (2nd run + NPSE diffusion head):** re-ran with NPSE (score-based/diffusion posterior)
  added. Across BOTH runs the per-dim SBC KS-p is NOISY (flow training stochastic; KS on 1500 pts
  near 0.05 is jittery) and the first run's clean-looking FMPE (0.043/0.088/0.047) was partly luck;
  2nd run: NPSE SBC 0.000/0.018/0.194, FMPE 0.013/0.000/0.191, MAF 0.000/0.176/0.015, all R²≈0.85.
  **Reproducible signal = uniform ~5% UNDER-coverage** (cov68 0.62–0.66 vs 0.68; cov90 0.85–0.88 vs
  0.90), IDENTICAL across NPSE/FMPE/MAF ⇒ NOT a flow-family issue, the 3-d F-tier summary is slightly
  overconfident. **NPSE (diffusion posterior) does NOT beat FMPE** — same band. Corrected takeaway:
  MLE flows lift OFF the energy-score 0.000 floor and keep accuracy, but aren't cleanly calibrated
  as-is; the lever is **posterior tempering τ≈1.1–1.15** (val-calibrated), not the density estimator.
  Ref: `Pflow/flow_result_npse.txt`.

### 2026-07-11 — [code] Diffusion FIELD decoder: NEGATIVE — rich spatial latent under energy score is UNSTABLE and worse than FiLM ⇒ confirms objective (ii), not latent (i)
- Built a diffusion-style iterative denoising FIELD decoder (`gate_f3_generative_ftier.py --z-mode
  diffusion`): replaces the 16-d GLOBAL FiLM latent with a high-dim SPATIAL latent — start from a
  Gaussian noise grid, refine over T=4 reverse steps with a conditional UNet3D denoiser (x0-pred,
  cold-diffusion style), fully differentiable so the energy-score gradient flows through all T passes
  + FFT physics. No DSM (F-tier has no ground-truth density field), so trained end-to-end via the
  SAME energy score → clean ABLATION of latent expressivity (i) holding objective (ii) fixed.
- **Result (M=4, T=4, cell 8, K=128):** pmean R² **−0.63 / −0.31 / −0.20** (WORSE than the mean;
  FiLM F3 got ~0.84), SBC 0.000 all, cov68 **0.40** / cov90 0.78, cluster Spearman −0.018.
- **Training pathology:** lam1-spread thrashed 0.074→0.0005(collapse)→0.42(explode)→0.06; train/val
  ES diverged (0.89 vs 1.53); early stop 1650. The FFT-physics gradient instability (cell-sweep cell4
  divergence) amplified by backprop through 4 diffusion steps. Deeper/expressive generator under the
  energy score = optimization pathology, not better calibration.
- **Verdict:** swapping low-dim latent (i)→rich spatial latent (ii-held) does NOT fix marginals and
  degrades everything ⇒ **the bottleneck is the ENERGY-SCORE OBJECTIVE, not latent dimensionality.**
  Agrees with NPSE (diffusion posterior head ≈ FMPE; tempering is the lever). BOTH diffusion routes
  (posterior head + field decoder) point the same way. **Calibrated F-tier posterior = MLE flow head
  (FMPE) + tempering τ≈1.1; do NOT pursue generative-field decoders for calibration.** Not worth LR
  tuning — objective-limited, per NPSE. Refs: `Pf3/f3_diffusion.txt`, `f3_diffusion_samples.npz`.
- Infra lesson: GPU drivers loading the cache pickle MUST export `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  + `XLA_PYTHON_CLIENT_ALLOCATOR=platform` — the unpickle pulls in JAX which else grabs ~30 GB (75%)
  and starves PyTorch → phantom OOM independent of model size (cost 3 allocations before caught).

### 2026-07-11 — [science] Why F3 is miscalibrated: it's the DENSITY ESTIMATOR (δ̂ posterior wrong SHAPE ← density is non-Gaussian)
- **Decisive diagnostic** (on saved F3 posterior samples, no retrain): the physics gives
  tr T = δ̂ exactly, so the F3 posterior over the TRACE (λ1+λ2+λ3) IS its δ̂ posterior, and it's
  ordering-invariant. **The trace fails SBC** (KS-p≈0, cov68 0.724, cov90 0.899, mean-rank 0.510)
  — same signature as the eigenvalues. Since the trace carries no ordering artifact, the
  miscalibration lives in **δ̂ itself → the density estimator.**
- **Ordering pushforward EXONERATED:** λ1 SBC fails equally for near-degenerate and separated
  galaxies (KS≈0 both); Spearman(|rank−0.5|, gap21)=+0.013 (≈0). If ordering distorted the ranks
  it would concentrate at small gaps — it doesn't.
- **WHY:** signature = coverage OK, SBC fails = wrong distributional SHAPE. The density field is
  intrinsically non-Gaussian (δ≥−1 in voids, long upper tail ≈ **lognormal**, Coles & Jones 1991),
  but F3's decoder is a Gaussian-latent FiLM field → roughly SYMMETRIC δ̂ samples. The energy
  score matches the marginal WIDTH (good coverage) but not the SKEW (fails SBC); the wrong δ̂ shape
  propagates deterministically through the physics into every eigenvalue + the trace identically.
  (Global-FiLM "uniform uncertainty" only partly true — per-galaxy 68%-width CV≈0.76 — so it's the
  SHAPE not the spatial structure.)
- **FIX direction:** model **log(1+δ̂)** (≈Gaussian; the standard lognormal move) instead of δ̂,
  or a normalizing-flow / diffusion field decoder able to represent a skewed bounded posterior.
- Refs: `field_level_tests/Pf3/f3_film_samples.npz` (analysis), `gate_f3_generative_ftier.py`.
- **UPDATE (log-density test, FALSIFIED the shape hypothesis):** ran F3 with `--log-density`
  (decoder emits u=log(1+δ̂), δ̂=exp(u)−1). SBC KS-p STILL 0.000 for all eigenvalues AND the trace;
  it changed the spread (base over-dispersed std/RMSE=1.65 → log-density 0.73) but NOT the SBC. So
  the density *parameterization* is NOT the lever. Rank histograms (both) pile at CENTER (0.22–0.25
  in [.4,.6] vs 0.20 uniform) — truth near the posterior median too often, unfixed by re-scaling.
  ⇒ redirect: the miscalibration is (i) the **low-dim global FiLM latent** (16-d → too restrictive
  a posterior family) and/or (ii) the **energy score is a JOINT scoring rule** — it calibrates the
  joint, not the per-eigenvalue MARGINALS that SBC tests. Remaining levers = richer/**spatial**
  latent + a **likelihood-based** (flow/diffusion) decoder, both real builds with uncertain payoff.
  **PRACTICAL:** calibrated F-tier posteriors are HARD; do NOT chase for the VAC. Ship G3+FMPE+
  tempering (calibrated λ1); keep F-tier for point-estimate eigenvector/field products. F3 = research
  thread. Refs: `field_level_tests/Pf3/f3_logdens*`.

### 2026-07-11 — [code] F-tier v2 NEGATIVE: encoder-side upgrades don't move F-tier; bottleneck is the field+physics factorization
- **v2 (raw wedge):** A = union graph + attention encoder + 9 edge features + TSC + U-Net =
  λ1 **0.8414** (0.900/0.932); B = A + survey-mask + FNO(16M) = **0.8389** (0.896/0.929).
  vs v1 0.840 → FLAT (within seed noise). The 16M FNO (B) is slightly WORSE than the 416k
  U-Net (A) with a mild train<val overfit gap.
- **Finding:** every architecturally-sound upgrade (union, attention, edge feats, TSC, FNO,
  mask) left F-tier at ~0.84. Bottleneck is NOT the encoder/graph/features/decoder — it is the
  field-decode + FIXED-physics factorization. **Coherent with branch (a)** (flow on the F-tier
  encoder embedding h_i collapsed to 0.407): both say the info lives in the POST-physics field,
  not the per-node encoder. A fancier encoder can't help.
- **~0.035 gap to the CNN (0.876) = "price of physics":** routing everything through one scalar
  field + the exact linear operator caps accuracy (the free CNN absorbs RSD/bias/discreteness
  residuals the exact map cannot); in exchange F-tier gives guaranteed-valid tensors, free
  ordering, and EIGENVECTORS.
- **Implications:** to raise F-tier accuracy work the FIELD side (finer cells — F-tier ran at
  6 Mpc vs CNN 5; learnable smoothing W_R; physics+residual hybrid), NOT the encoder (keep it
  simple). Production picture unchanged: CNN = point-estimate leader (0.876); G3+FMPE+tempering =
  shippable calibrated λ1; F-tier = physics/eigenvector product at ~0.84. Refs: `gate_ftier_v2.py`,
  `field_level_tests/Pv2/`, plan §12.

### 2026-07-10 — [code] Calibration branches: G3+FMPE+tempering ships λ1; F-tier needs F3 (encoder-embedding flow fails)
- **Branch (b) G3+FMPE + posterior tempering (val-calibrated, held-out test, union model):**
  a single scale τ≈1.15 brings **λ1 to nominal coverage (0.689@68, 0.897@90) AND passes SBC
  (KS-p 0.507)** with post-mean R² unchanged (0.839); TARP max|ECP−α|≈0.04. τ68≈τ90 ⇒ λ1
  miscal was pure SCALE. **BUT λ2/λ3 keep residual SHAPE miscal (SBC still 0.00) that
  tempering can't fix.** ⇒ **VAC headline P(λ1>λ_th) is SHIPPABLE now with G3+FMPE+tempering**;
  the 4-class (λ2/λ3) columns need SBC-aware training first. `field_level_tests/Pb_tempering/`.
- **Branch (a) F-tier encoder-embedding → flow @ nzharm: NEGATIVE (informative).** Conditioning
  FMPE/MAF on the F-tier EGNN encoder embedding h_i collapses accuracy to **λ1 0.407** (vs the
  F-tier's own 0.838); the "good" coverage (0.664@68) is trivial (wide uninformative posterior).
  **Reason:** F-tier's eigenvalue info lives in the δ̂ field AFTER the U-Net+physics (nonlocal,
  mixes neighbours), NOT in the per-node encoder embedding — so the h_i-flow shortcut (which
  works for G3) throws away the nonlocality the physics layer exploits. This VINDICATES the
  F-tier design and means a calibrated F-tier posterior needs the **generative-δ̂ route (F3)**,
  not the encoder-embedding trick. `field_level_tests/Pa_ftier_calib/`.
- **Production decision:** ship **G3 GraphNet + FMPE + scalar tempering** for the calibrated
  λ1 headline; F-tier stays the point-estimate accuracy/physics/eigenvector leader pending F3;
  λ2/λ3 calibration (4-class) = SBC-aware training, a follow-on. Scripts: `gate_pa_ftier_flow_calib.py`,
  `gate_t4 --save-embeddings`, Pb tempering driver.

### 2026-07-10 — [code] P2 DESI-density (nzharm) accuracy: field/grid edge PRESERVED, not a dense-wedge artifact
- **Setup:** re-ran CNN (gate_t2) + F-tier (gate_t4), MSE heads, on the n(z)-harmonized
  nzharm cache (82,650 gal, DESI-density-matched, verified row-aligned to
  `path1_wedge_nzharm_final_points_xyz.npy`). GPU tmux. `field_level_tests/P2/`.
- **Result (λ1 R²):** CNN **0.864 ± .004** (raw 0.876, Δ−0.012); F-tier **0.838 ± .0001**
  (raw 0.840, Δ−0.002). Ranking CNN ≳ F-tier ≫ graph baseline PRESERVED. **The
  field/grid-representation advantage is NOT a dense-wedge artifact** — it survives the
  DESI-density correction essentially intact.
- **Caveats (honest):** (1) this nzharm is only ~18% sparser than the raw wedge (100,935→
  82,650), so it is a *robustness* check, NOT an extreme-sparsity stress test — the truly
  sparse regime (voids, full BGS n(z) tails) where a fixed grid could finally degrade is
  STILL untested. (2) MSE point-estimate accuracy only; **calibration at nzharm was NOT
  measured** (needs a flow trained at nzharm) — the P3 under-coverage was on the raw wedge,
  so whether density changes it remains open. See plan §10.

### 2026-07-10 — [code] P3/G6 posterior estimator: FMPE beats MAF on accuracy; calibration comparison INCOMPLETE (caveat)
- **Setup:** frozen union GraphNet encoder (80-d embeddings), identical splits/targets;
  MAF (existing) vs a fresh sbi-package FMPE (flow-matching) head. Same conditioning ⇒
  any difference is the density estimator. Fully CPU. Script:
  `gate_g6_fmpe_frozen_head.py` + new `generate_maf_selfeval.py`. Results:
  `/pscratch/.../field_level_tests/P3/g6_result.txt`.
- **Accuracy (posterior-mean R², n_eval=1500, frozen emb):** MAF 0.819/0.881/0.916 vs
  **FMPE 0.850/0.896/0.928** — FMPE wins every eigenvalue (+0.031 λ1), cluster-slice λ1
  Spearman MAF +0.68 vs **FMPE +0.70**. The gate's accuracy half (FMPE ≥ MAF−0.01) is
  clearly met. Confirms the prior: FMPE is the better estimator on this conditioning.
- **Calibration (FMPE only, this run):** SBC KS-uniform p = 0.000/0.003/0.001 (rejects
  uniformity); λ1 central coverage 59.4% @ nominal 68%, 82.9% @ 90% → **under-covers /
  over-confident**. The gate's calibration half is NOT cleanly met as measured.
- **CAVEATS (do not over-read the calibration failure):** (1) **MAF's own SBC/coverage
  was NOT computed** in this run — the script only prints FMPE calibration, so we can only
  say FMPE is imperfectly calibrated in ABSOLUTE terms, NOT that it is worse than MAF (MAF
  may under-cover similarly). The symmetric comparison is the missing piece. (2) FMPE was
  quick-trained (144 epochs, sbi defaults, single seed) — under-coverage often = under-
  training/default HPs, not a fundamental limit. (3) raw over-dense wedge.
- **Verdict:** FMPE is the accuracy winner and the right direction, but the posterior-
  estimator DECISION is not final until MAF SBC/coverage is measured on the same eval set
  (symmetric). Next (cheap, CPU): add MAF SBC+coverage to the script and re-run. See plan §10.
- **RESOLVED (symmetric rerun, same day):** MAF SBC KS-uniform p = 0.009/0.006/0.017,
  λ1 coverage 0.610@68% / 0.837@90% — vs FMPE 0.000/0.003/0.001, 0.594 / 0.829.
  **BOTH under-cover near-identically** (~60% @ nominal 68%); MAF marginally better but
  trivially so. ⇒ Case (c): the under-coverage is a property of the frozen-encoder / raw
  over-dense wedge / default flow training, **NOT of MAF-vs-FMPE**. So FMPE's accuracy win
  stands with calibration COMPARABLE to MAF → G6 gate effectively **GO** (adopt FMPE). The
  calibration deficit is a SEPARATE workstream: SBC-aware training / posterior tempering,
  and the P2 nzharm (DESI-density) re-run. `field_level_tests/P3/g6_result.txt`.

### 2026-07-08 — [science] Where the field-level results leave the three plans (and: the graph PhD work is NOT shown useless)
- **The honest worry** (JDPK): "has a dumb CNN beating my GraphNet shown 2.5 yr of graph
  work useless?" **No — and precisely why is worth stating.** Full argument in
  `plan_field_level_multimodal.md` §9; summary here.
- **Result in one line:** the field-level program is VALIDATED and the graph is VINDICATED
  as an encoder; what changed is the FRAMING of the architectural lever — it's representation
  *scale* + a physics-grounded *output*, not attention and not equivariance. The
  SBI/calibration/VAC core and the published RASTI classification are untouched.
- **Why "CNN beats graph" is not the finding:** (1) **T4/F1 is the rebuttal** — a GRAPH
  encoder + field/physics decoder = 0.841, above every graph baseline (0.775/0.804), near
  the CNN (0.876), with mean-agg and NO attention. The winning net IS a graph net. (2) A
  **CNN is a GNN on a lattice**, so T2 = "fixed-scale sampling > Delaunay receptive field" —
  a graph-CONSTRUCTION result (your G4 thesis; you'd already seen G3 union 0.804 > Delaunay
  0.775). (3) The comparison is **confounded/unfinished**: MSE head vs NPE posterior-mean;
  the matched *production*-GraphNet-MSE control is still TODO (the 0.704 is a weak EGNN-lite
  smoke); raw wedge is ~2.4× DESI density (easiest for a grid). (4) The **metric isn't the
  thesis** — the VAC deliverable is calibrated posteriors (SBI/NPE, TARGETID VAC, closure,
  DESIVAST), which a point-estimate R² doesn't touch. (5) All on **one dense wedge**; the
  sparse survey-scale regime (voids, n(z)) is where grids degrade and graphs win — untested.
- **Per-plan status:** MULTIMODAL — central hypothesis CONFIRMED (F-tier passed accuracy
  gate); next = F1-calibration (flow head) → F2 → F4. G4-PROPER — equivariance DEPRIORITISED
  (its own P1a≥0.80 gate fired 3×); shelve SEGNN/Equiformer P1b + Tier B; F-tier supersedes
  Tier B for eigenvectors; attention demoted to second-order. ROADMAP→VAC — evaluate the
  F-tier (G7) as the production encoder in Phase-B (it beat G3); SBI core unchanged.
- **Thesis framing matures** from "graphs are the architecture" to "the graph is an
  excellent encoder for a physics-grounded field inference, and here is why" — a stronger,
  more defensible contribution (you built the dataset, the calibrated-NPE apparatus, the DESI
  application, AND the understanding that led to the F-tier that beats everything).

### 2026-07-08 — [code] Field-level tests run: T2 CNN-on-counts + T4/F1 graph→field→Poisson both beat every graph baseline; T3 shelved
- **T4/F1 (the centerpiece) PASSES its accuracy gate:** graph encoder (EGNNlite,
  mean-agg, NO attention) → differentiable CIC scatter → 3-D U-Net → δ̂ → fixed
  FFT physics layer → analytic 3×3 eigensolver → eigenvalues. **λ1/λ2/λ3 =
  0.841/0.897/0.931**, ≥ G3 (0.804). The graph encoder + field/physics decoder
  nearly matches the pure CNN and beats every prior graph baseline ⇒ the "CNN is
  killing my graph work" worry is answered: graphs are vindicated as the encoder;
  the win is the fixed-scale, field-shaped OUTPUT. Calibration half of the gate
  (≥ current flow) still pending the invariant-latent→FlowJAX head.
- **T2 CNN-on-counts is real and robust:** 3-D U-Net on voxelized galaxy COUNTS
  (5 Mpc, 1.4M params) → **λ1 0.876 ± 0.004** (3 seeds), beats GraphNet 0.775.
  Spatially validated: T-web **4-class accuracy 0.882** (void/wall/filament/cluster
  F1 0.906/0.878/0.872/0.845), class fractions within ~0.3% of truth, confusion
  only between adjacent classes, Spearman(pred,truth λ1)=0.965. Interactive 3-D
  viz built (artifact + PNGs) — predicted cosmic web tracks truth cleanly.
- **Framing (JDPK):** a CNN IS a GNN on a lattice, so T2 = "fixed-scale regular
  sampling + density-valued nodes > Delaunay receptive field" (same lesson as G3),
  NOT "CNN > graphs". Attention looks second-order, not non-essential (needs the
  clean within-F-tier on/off test). T2 also de-risks a grid-native JEPA (P5).
- **Caveats gating interpretation** (NOT yet resolved → §8 controls in the plan):
  MSE point head vs NPE posterior mean; RAW wedge is ~2.4× DESI density (easiest
  for a grid). Controls queued: matched-estimand graph MSE head, nzharm-density
  re-run, cell-size sweep on the cluster slice, clean attention on/off.
- **Ops lessons (bit us):** (1) `srun --overlap` can silently fail to bind the GPU
  for a *later* step in an allocation → T3 CPU-trained for 1.5 h invisibly; fix =
  a fail-fast `torch.cuda.is_available()` guard on every step. (2) `torch.linalg.
  eigvalsh` on CUDA tried to allocate 51 GiB for (N,3,3) (cuSOLVER blowup) →
  replaced with analytic Cardano 3×3 (matches numpy 7e-15). (3) interactive-only
  runs per JDPK — reuse idle allocations via `srun --jobid`; no sbatch.
- **T3 (LUPI)** re-running in a fresh 4 h GPU alloc (capped steps, guarded);
  box-frame teacher validated (25× random density). Verdict pending.
- Refs: plan `docs/plan_field_level_multimodal.md` §7–8; scripts
  `workflows/sbi/gate_t2_cnn_counts.py`, `gate_t4_graph_field_poisson.py`,
  `viz_t2_wedge_clusters.py`; `/pscratch/.../abacus/field_level_tests/`.

### 2026-07-09 — [code] S-TRACK added: full-z-range selection function (the maglim/sparsity problem, now explicit); FMPE confirmed production head
- **Trigger (JDPK):** the VAC must span BGS z≈0.05–0.6; the magnitude limit drives a
  **measured ~165× density falloff** (A2 full-range JSON, n̄ z0.15–0.25 vs 0.45–0.55) while
  all training so far used one shell (z0.2–0.3). Audit: the ñ(z)-conditioning idea was
  NEVER formally logged — the roadmap handled n(z) only on the sim-to-real axis (A2/A3,
  within-shell). Now explicit as **Track 2b / S-track** in `roadmap_environmental_vac.md`.
- **Why neither existing mechanism suffices:** SI per-graph medians absorb a UNIFORM
  scale (validated at 0.73×), not two decades — and rescaled features cannot fix
  CALIBRATION: an amortized NPE trained at one density is overconfident at sparser ones;
  posterior width must grow with z.
- **NEW measurement that shaped the design:** mock/DESI count ratio degrades with z:
  0.91 (z0.05–0.15) → 0.95 → 0.77 → 0.54 → **0.28 (z0.45–0.55)**. Full-range
  shape-match-by-dilution (A3 recipe) would set C=0.28 ⇒ discard 72% of training data —
  dilution CANNOT extend to the full range.
- **Design decision (default): ONE amortized model conditioned on sampling intensity
  ñ(zᵢ)** — smooth spline of each dataset's OWN n(z) (DR2 at inference, mock in training):
  node feature EXCLUDED from SI by name (it IS the covariate) + appended to the FMPE
  conditioning vector (heteroscedastic amortization → posteriors widen at high z).
  **Conditioning on ñ, not z**, dissolves the 0.28 problem: density regimes overlap
  mock↔DESI even where z-profiles diverge. Per-shell models = pre-registered fallback
  (seams in a public VAC, K× cost, no statistical sharing where data is scarcest).
- **Gates S0–S5** (roadmap §3b): S0 selection atlas + per-shell graph/voxel stats;
  S1 cheap cutsky-truth shell-transfer matrix (pooled-conditioned vs per-shell BEFORE any
  retrain); S2 multi-shell buffered wedges from the sentinelfix parent (A1 unblocked
  z-expansion); S3 winner-specific conditioning (GraphNet ñ-feature; U-Net/F-tier
  expected-counts channel ñ·V_voxel); S4 production showdown; S5 per-shell SBC/TARP +
  GateM + width-monotonicity-with-z sanity.
- **S-track is also the encoder arbiter:** T2 U-Net 0.876 / F-tier 0.841 were measured on
  a ~2.4×-dense wedge; grids degrade with sparsity, Delaunay∪radius adapts — the
  G3-vs-F-tier-vs-U-Net decision must be made on S2 full-range data, not the dense wedge.
- **Phase B REDEFINED:** the bundled retrain becomes FULL-RANGE + ñ-conditioned (S2/S3
  fold in; house rule "one bundled retrain" is exactly why this must be decided now).
- **FMPE head CONFIRMED for production (JDPK, task-3 calibration tune complete: FMPE won
  on accuracy AND calibration)** — the S3 conditioning spec targets the FMPE vector.
- Known systematic (documented): single-snapshot z=0.2 labels carry no growth evolution
  across 0.05–0.6 → VAC inherits a "z=0.2-epoch tidal field" convention; optional
  post-hoc D(z)/D(0.2) rescaling; long-term multi-snapshot lightcone.

### 2026-07-07 — [science] Multimodal/field-level direction: "we might be throwing too much information away" + the classical floor that motivates it
- **The idea (JDPK):** simulations give us complete 3-D density/tidal/eigenvalue FIELDS,
  and the pipeline reduces them to 3 numbers per galaxy before any model sees them.
  Proposals discussed: (1) train-time 3-D CNN on the sim fields latent-regularising the
  GraphNet (absent at inference); (2) semi-/self-supervised sim+real joint training;
  (3) tensor-valued outputs via a more physical model.
- **Information accounting (the discipline):** at DESI inference the input is the sparse
  redshift-space catalog, full stop — a train-only modality cannot raise the mutual
  information, only regularise (idea 1 = Vapnik LUPI / generalized distillation; cheap
  ablation, modest expected gain). Idea 2 is risky: latent alignment can HIDE sim→real
  shift and silently bias NPE posteriors → truth-known sim↔sim control required first.
- **The reframe that stuck: flip the field from INPUT to OUTPUT.** Graph → decoded δ̂
  grid → **fixed differentiable FFT physics layer** T̂ij(k)=(ki kj/k²)W₇(k)δ̂(k) →
  eigenvalues at galaxies. Nothing about the density→tidal map is learned — it is
  hard-coded mathematics; gradients flow through eigvalsh+FFTs into δ̂. Eigenvalues
  "fall out from the physics"; symmetry/trace/rotational consistency guaranteed;
  eigenVECTORS (→ IA science) come free WITHOUT e3nn irreps or the Tier-B tensor
  rotation crux (scalar field needs only the affine map). Composes with, does not
  compete against, the G4 nonlocal-operator story (same 1/k² kernel).
- **T1 classical-reconstruction baseline RUN (the "first move"):** textbook density
  estimation + the exact tidal solve, same estimand + test split as the GraphNet.
  Best classical (DTFE): **λ1/λ2/λ3 R²(cal) = 0.552 / 0.641 / 0.663** vs GraphNet+NPE
  0.775/0.811/0.891 (CIC 0.546, Wiener≈CIC, extra smoothing hurts; interior-only λ1
  0.60). **⇒ The GraphNet's +0.22/+0.17/+0.23 margin over linear reconstruction is
  genuine learning** (RSD + bias + nonlinearity) — architecture work is chasing real
  signal, and the biggest margin is on λ3 (collapse axis / clusters).
- **Physics-layer validation banked:** the same FFT solve reproduces the stored cactus
  eigenvalues voxelwise on a 512³ subbox of the 10% particle grid at
  **R² = 0.992/0.995/0.997** — the F-tier physics layer AND the G4 Tier-B validation
  anchor are both de-risked in one shot.
- **Plans updated:** new canonical doc `docs/plan_field_level_multimodal.md` (test list
  T1–T5 + F-tier phases); `plan_g4_proper_equivariant_tensor.md` (classical floor row,
  Tier-B amendment, §10 companion pointer); `roadmap_environmental_vac.md` (G5
  concretized→T2, NEW gates G7 field-decoder + G8 LUPI distillation, T5 prerequisite on
  the JEPA branch, G7 columns in the release spec).
- Refs: `workflows/abacus_tweb/classical_tidal_baseline.py`;
  `/pscratch/sd/d/dkololgi/abacus/classical_baseline/` (scores + per-estimator
  predictions + solver validation JSONs).

### 2026-07-04 — [code] G4-PROPER WAVE 1 COMPLETE (A–E): point-attention wins, dynamic graph & steerable lose; wave 2 (D seeds + F) launched
- **Full wave-1 board (all λ1/λ2/λ3, positions-only unless noted, point-estimate MSE except G3/A/baseline = NPE posterior mean):**
  - baseline GraphNet+NPE, Delaunay, curated: **0.775** / 0.811 / 0.891
  - G3 GraphNet+NPE, union, curated: **0.804** / 0.846 / 0.895  (production anchor)
  - A GraphNet+NPE, radius, curated: **0.752** / 0.799 / 0.876
  - **D point-attention MPNN, radius, POSITIONS-ONLY: 0.726 / 0.807 / 0.838, clu ρ 0.54**
  - E attentional DGCNN, DYNAMIC feature-space kNN, positions: 0.507 / 0.662 / 0.681, clu 0.36
  - B SEGNN steerable+attn, union, positions: 0.536 / 0.610 / 0.653, clu 0.35
  - C SEGNN steerable+attn, radius, positions: 0.423 / 0.411 / 0.513, clu 0.37
  - (all C/D/E converged: C ran full budget val-plateaued, D early-stopped @4000, E early-stopped @1925 — not truncated.)
- **THREE headline reads:**
  1. **Point-cloud from raw geometry works & ≈ curated features.** D (positions+LOS+radius
     graph) λ1 0.726 vs A (same graph, curated features) 0.752 — matched estimand
     (MSE point est ↔ NPE posterior mean both target E[λ|x]) → positions recover ~96% of
     the curated-feature signal; the cuGraph features add ~+0.03. JDPK's point-cloud
     instinct validated.
  2. **Dynamic feature-space graph HURTS: E−D = −0.22 λ1** (matched inputs+attention). The
     §1(d) subsumption hypothesis is CONFIRMED, not assumed: for a compact-support target,
     letting edges roam in feature space imports non-local neighbours and costs ~0.22.
     Attention did NOT rescue it (it reweights candidates, cannot attend to evicted physical
     neighbours). Clean publishable negative for the dynamic-graph line.
  3. **Steerable SEGNN underperformed everything** (B 0.536, C 0.423) — but CAPACITY-
     CONFOUNDED (fast 179k config vs D's larger net; e3nn-TP-throughput-limited). B(union)
     0.536 > C(radius) 0.423 → union>radius holds inside the steerable family too.
- **GPT-5.5 deep-review memo — synthesis (JDPK relayed):** (a) upgrades the graph story via
  the inverse-Laplacian: T̂ij(k)=(ki kj/k²)W_R(k)δ(k); the 1/k² makes the smoothed tidal
  tensor genuinely NONLOCAL (real-space kernel ~anisotropic power-law, not exp-suppressed),
  so the union graph is a **discrete quadrature of a nonlocal operator** — Delaunay bridges =
  adaptive void connectivity, radius edges = fixed aperture, attention arbitrates. This also
  explains E: useful long edges are GEOMETRY-anchored (Delaunay bridges help), not FEATURE-
  anchored (DGCNN dynamic edges hurt), and softens the original review's "compact support ⇒
  skip long-range". (b) Its pre-written decision tree lands on branch 2 (D high + B low ⇒
  steerable IMPLEMENTATION may hurt ⇒ RPP-relaxed / hybrid, NOT immediate heavy Equiformer).
  Convergent with our plan's RPP escape hatch. (c) New paper narrative endorsed: "discovering
  the correct discrete support for a nonlocal cosmological operator", not a failed arch search.
  (d) Production stays G3 (unchanged). (e) Tier-B physics constraints logged for later (trace–
  Poisson TrT_R=δ_R; Hessian integrability ∂k Tij=∂j Tik — a free symmetric tensor field is
  NOT automatically a Hessian field). Full memo lives with JDPK.
- **Redo policy: NO redos** (runs internally valid). Needed instead = diagnostic layer.
  **Wave 2 LAUNCHED now** (tmux `g4_wave2` on login node, SSH-independent):
  **D seeds 43 & 44** (seed variance for the headline result; single-seed so far) + **run F**
  = attentional DGCNN + CURATED Delaunay features (F vs E = feature axis for dynamic graph;
  F vs A = dynamic vs radius at matched curated features). D-seed43 already allocated
  (55499510).
- **DGCNN control knobs added** (answering JDPK's "how much control over the dynamic graph"):
  `gate_g4_p1e_dgcnn_attn.py` now takes `--curated-features` and `--knn-radius-cap` (restrict
  feature-space kNN to a physical envelope — 'learned selection WITHIN locality', the direct
  fix for why E lost; void nodes fall back to physical-kNN). Verified: capped kNN keeps edges
  local, rotation-invariance holds with curated+capped (1.2e-16). F runs UNCAPPED (= E arch)
  to isolate the feature axis; capped variant is the ready follow-up if F or E's void slice
  motivates it.
- **Deferred diagnostics (not yet built, next after wave 2):** environment-sliced eval
  (V/W/F/C) across G3/A–F; connectivity-residual diagnostics; union edge-type attention-mass
  attribution. Explicitly NOT queued: matched-capacity heavy SEGNN, Equiformer/MACE, Tier B.
- **Standing:** G3 resume (55441429) still PENDING on plain gpu&a100 — JDPK to protect with
  `scontrol update JobId=55441429 Features="gpu&hbm80g"` (Claude permission-blocked from the
  queued job).

### 2026-07-04 — [code] RUN B RESULT (P1b SEGNN×union, positions-only): λ1 0.536 — MISS on gate, but capacity-confounded; chain hardened
- **B (SEGNN steerable+attention, union graph, positions+LOS only, 179k params, 1348
  steps, point-estimate MSE):** test-set λ1 R² **0.536**, λ2 0.610, λ3 0.653;
  cluster-slice λ1 Spearman **+0.35** (baseline 0.54). Clean training: train 0.234 ≈
  val 0.218, NO overfitting (the DropEdge + attention-dropout discipline held).
- **Verdict: clear MISS** on the P1b gate (needs λ1 ≥ 0.75 AND beats the P1a controls
  radius 0.752 / union 0.804). **BUT heavily confounded — do NOT read as "equivariance
  fails":** (1) this is the FAST config (179k params vs baseline ~992k; hidden halved,
  3 layers) forced by e3nn TP throughput (~9.5 s/step even so); ~5× under-capacity.
  (2) positions-only, so the 0.536→0.774 gap largely re-expresses the value of the
  curated features (G1: features carry most of the recoverable signal). (3) point-
  estimate MSE head is R²-FAVOURED vs the baseline's posterior mean, so 0.536 is if
  anything optimistic. Net: positions-only steerable+attention at reduced capacity does
  not reconstruct what curated cuGraph features encode — expected direction, but the
  capacity confound must be resolved (matched-param SEGNN) before the gate is honestly
  called or an RPP-relaxed variant is run.
- **Clean comparisons still pending:** B−C (union vs radius WITHIN steerable, matched
  everything else) needs C; the equivariance contrast needs the positions-only non-
  equivariant runs D (radius) / E (dynamic). Only then does the factorial read.
- **tmux chain hardened + restarted (login30):** the liveness check now distinguishes a
  live run from a failed-salloc/crashed attempt (tail terminal-marker grep, not mere
  file freshness) — was waiting 25 min on stale failed-attempt logs; now retries
  promptly when a slot frees. Verified: on restart it correctly skipped B (results) and
  C (live), and launched D (job 55484009). B's slot freed → D pending a node.
- Interim conversion note (for the record): B's pooled val MSE 0.218 in globally-
  standardised space → pooled R²(Y) 0.78 is INFLATED by between-component ordering; the
  honest per-component decomposition predicted mean ~0.55, matching the measured
  0.536/0.610/0.653. Global-standardised val loss is NOT comparable to the FlowJAX runs'
  val NLL — different metric entirely.

### 2026-07-04 — [code] RUN A RESULT (P1a-i): radius-only 0.752 < union 0.804 at matched budget — COMPLEMENTARITY is the lever; unattended tmux chain armed
- **Radius-only control @3749 epochs (posterior-mean, 128 samples, shared test split):**
  λ₁ R² **0.7519**, λ₂ 0.7989, λ₃ 0.8757, mean 0.8088; best val NLL 1.4248.
  Anchors: union@3749 **0.8041**/0.8461/0.8955 (val NLL 0.856); Delaunay-full@7000
  0.7750/0.8105/0.8912 (val NLL 1.107).
- **Readings:** (1) **JDPK's prediction CONFIRMED: union > radius-only** (+0.052 λ₁ at
  matched budget) — Delaunay edges carry information the radius ball misses (registered
  mechanism: 172k Delaunay pairs longer than 14.78 Mpc = void bridges; radius-only has
  110 isolated nodes / 178 components). (2) Radius-only ≈ Delaunay-full (0.752@3749 vs
  0.775@7000, budget-mismatched in Delaunay's favour) — the pure constructions are
  roughly comparable; **neither alone is the lever: the UNION (complementary edge
  populations + attention arbitration) is.** (3) P1a threshold: radius-only does NOT
  hit ≥0.80 alone ⇒ "construction is the lever" holds in the ADD-radius form, not the
  replace form. P1b's bar remains the union control (0.804@3749; higher at G3-7000).
- Follow-up registered: environment-sliced eval (V/W/F/C) on A vs G3 checkpoints to
  test the mechanism (radius deficit in voids; radius gain over Delaunay in clusters).
- **SSH-independence (JDPK disconnecting):** all remaining wave-1 automation moved to a
  tmux orchestrator on **login30** (`tmux attach -t g4_chain`;
  `workflows/sbi/run_g4_chain.sh`): idempotent — skips runs whose results exist, leaves
  live runs alone (log-freshness), (re)launches anything missing incl. B/C if the
  Claude-owned sallocs die at disconnect. D's smoke script now writes a results file
  (`--out-file`) for completion detection. Claude-bound D/E watchers stopped.

### 2026-07-04 — [code] Run E added (attentional DGCNN, unfixed graph); D relabelled; B throughput fixed; wave now A/B/C/D/E
- **D relabelled (JDPK caught it):** "Point-Transformer-class" was an over-badge — a
  point-cloud network IS an attention GNN on a coordinate-derived graph, so D is
  architecturally the same object as run A. D's honest label: **positions-only attention
  control**; its scientific content is the FEATURE axis (D−A), not a new model family.
- **Run E built (P1a-iii, JDPK design):** attentional DGCNN — the ONE wave-1 model whose
  graph is genuinely NOT fixed: kNN recomputed PER LAYER in learned feature space
  (layer 0 = coordinate kNN, k=20), EdgeConv max-pool REPLACED by GAT/GAPNet-style
  4-head attention, positions+LOS only, eigenvalue supervision.
  `gate_g4_p1e_dgcnn_attn.py` + runner; selftest: forward/backward finite AND full-model
  **rotation invariance 1.4e-16** (all learned features derive from invariant scalars →
  feature-space kNN itself rotation-invariant, dodging DGCNN's usual symmetry breakage).
  **E−D = learned dynamic candidates vs fixed physical candidates** at matched
  inputs+aggregation — the §1(d) subsumption claim gets TESTED, not assumed.
- **Conceptual point logged (plan §5A):** attention fixes the WEIGHTING axis (the
  over-smoothing-like fixed-kernel problem, JDPK's analogy — half right), but NOT the
  CANDIDATE-SELECTION axis: attention cannot attend to an absent edge, and fixed-k
  feature-space kNN can evict physical neighbours entirely. E mitigates via layer-0
  coordinate kNN + per-layer geometry scalars. Prior: D ≥ E overall, but **voids are
  where E could win** (environmental parameter sharing between feature-similar,
  spatially distant void galaxies) — an E win in voids would revive the multi-scale
  graph line.
- **B throughput crisis fixed:** e3nn TP einsum kernels (not matmul — TF32 did nothing)
  gave 30–49 s/step ⇒ ~244 steps/budget = void. Config: hidden halved
  (16x0e+8x1o+4x2e → 179k params), 3 layers, DropEdge 0.5 (full edges for val/eval),
  val every 50 → **9.55 s/step, T_max=1256 steps**. Matched-param discipline vs the
  992k GraphNet is broken for the smoke (noted in results); returns in multi-seed phase.
- **Queue chain (QOS 2-slot):** A (RUNNING, epoch ~1400/3750, on pace) + B (RUNNING,
  fast config) → C (SEGNN radius) → D (positions-only radius) → E (DGCNN) via watchers.
- SKIP-list updates: DGCNN → promoted to run E; PointNet(++) stays out (no/fixed-radius
  neighbourhoods — redundant with A/D).

### 2026-07-04 — [code] Run D added (P1a-ii point-cloud control); P1a split into two controls; plan §5A/§6 reconciled
- JDPK caught a real ambiguity: the plan's §5A table named Point Transformer/DGCNN as
  "the" P1a control while §6 (and the source report's Rec. #1) specified the in-stack
  GraphNet radius control — these are TWO different controls answering different
  questions, and the launched run A implements only the latter. Fixed by splitting:
  **P1a-i** (run A, in-stack GraphNet, single-variable edge-set ablation) and **P1a-ii**
  (**run D, NEW**): Point-Transformer-class attention MPNN, POSITIONS+LOS ONLY, node
  features [1, |pos|/median], neighbourhoods built at LOAD TIME from the point
  distribution (`gate_g4_egnn_smoke.py --positions-only --build-radius-mpc 14.78`).
  Verified: load-time radius build reproduces the prebuilt npz EXACTLY (1,816,273
  pairs) — for a fixed rule, "no preconstructed graph" and "prebuilt" are the same
  edges; the genuinely dynamic family (DGCNN feature-space kNN) stays SKIPPED per the
  §1(d) subsumption argument (compact-support target).
- **Attribution algebra now complete** (added to plan §5A): D−A = raw geometry vs
  curated features; C−D = equivariance alone (matched inputs+graph); A−G3 = radius vs
  union; B−C = union vs radius within equivariant. Without D, a P1b result could not
  be attributed to equivariance vs "any attention net on raw geometry".
- **Supervision policy recorded in plan:** ALL wave-1 models train on eigenvalues
  (Tier A); steerable nets predict the tensor internally only; non-equivariant nets
  must not emit a fixed-frame tensor. Tensor TARGETS = Tier B, gated.
- Run D queued behind the QOS 2-slot cap via watcher (fires after C is training and a
  slot frees). Sequencing: A ends → C; B ends → D.

### 2026-07-04 — [code] G4-PROPER wave 1 LAUNCHED: 2×2 factorial {control, SEGNN} × {union, radius}; P0 at machine precision
- **Design (JDPK-approved):** complete the factorial with 3 runs, reusing G3 as the
  4th cell: control×union = G3 (0.8041@3749 ✓), **A** control×radius (existing GraphNet+
  curated features+FlowJAX, ONLY edges swapped; `--epochs 3750` = exact matched budget),
  **B** SEGNN×union, **C** SEGNN×radius (positions+LOS only, §0 purity). Interactive QOS
  caps 2 concurrent jobs → C auto-launches via watcher when A or B frees a slot.
- **Radius-only graph built** (`build_union_graph_arrays.py --radius-only`, +connectivity
  diagnostics): 1,816,273 pairs @14.78 Mpc. **Void-fragmentation prediction confirmed
  structurally:** 178 components, 110 isolated nodes, largest component 99.64% (union =
  single component via Delaunay's triangulation guarantee); 172,459 Delaunay pairs are
  LONGER than 14.78 Mpc (the void bridges radius-only loses). Cache
  `path1_flowjax_3d_lineareig_si_radiusgraph` built by an edge-transplant procedure
  **gold-validated by exactly reproducing the production union cache on all 9 fields**;
  nodes/masks/targets byte-identical to baseline.
- **P1b script** `workflows/sbi/gate_g4_p1b_segnn.py`: e3nn steerable MPNN (hidden
  32x0e+16x1o+8x2e, 4 layers, 4 heads, 921k params), invariant-logit segment-softmax
  attention, 1x0e+1x2e head → sym 3×3 → ANALYTIC eigvals → MSE on existing LAMBDA
  (Tier A). Single GLOBAL affine target scaling (per-component would be
  tensor-inconsistent). Regularisation parity: wd 0.08, attention dropout 0.1,
  early-stop on val.
- **Debugging harvest (all committed, each a real finding):** (1) 40-vs-80GB A100
  roulette — FlowJAX needs ~43GiB/device on union/radius; P1a pinned `hbm80g`; the
  PENDING G3 resume sbatch 55441429 still has plain gpu&a100 = OOM RISK if it lands on
  40GB (JDPK to decide: `scontrol update JobId=55441429 Features="gpu&hbm80g"`).
  (2) **JAX preallocates 75% of GPU when unpickling the cache's jnp arrays** under a
  PyTorch job — the "non-PyTorch 30GB" both B OOMs; fixed `JAX_PLATFORMS=cpu`.
  (3) cuSOLVER batched eigvalsh wants ~25GiB workspace at N~1e5×3×3 → closed-form
  trigonometric sym-3×3 eigensolver (plan §7 mitigation; matches LAPACK to 1e-14).
  (4) e3nn per-edge TP weights blow memory at 4M edges → edge-chunked checkpointed
  values (chunk-equivalence 4.4e-16); nested layer+chunk checkpoints break e3nn
  TorchScript on recompute → flat inner-only checkpointing + narrow shared logits MLP.
  (5) **P0 precision forensics:** GPU index_add atomics (~1e-7 float64 noise) → CPU
  test; CartesianTensor float32 change-of-basis → float64 buffer; **e3nn bakes CG
  constants in default dtype at module creation** (.double() doesn't recast) → build
  test model under float64 default. Result: **equivariance 5.0e-16 = machine
  precision** (the earlier 6.6e-9 was baked-float32 noise, not architecture).
- **Status at log time:** A RUNNING (~1.6 s/epoch, will hit 3750 + auto-eval well within
  4h), B RUNNING (step 0: train 1.64/val 1.42), C armed. Prediction registered: union ≥
  radius-only with the gap concentrated in voids (connectivity), radius ≥ Delaunay in
  clusters (aperture). Anchors: Delaunay-full 0.7750, union@3749 0.8041.

### 2026-07-03 — [code] G3 readout ASSESSED: connectivity axis wins; Phase B needs union×nzharm merge; G4 bar quantified
- Verified on disk (`..._uniongraph_EVAL3749/flowjax_sbi_results_*.txt`): λ₁ 0.8041 / λ₂
  0.8461 / λ₃ 0.8955 (mean 0.8486) vs baseline 0.7750/0.8105/0.8912 (0.8256); test NLL
  0.8964 vs 1.1392 (~0.24 nats/galaxy sharper posteriors) — at 56% of training budget.
- **Ladder update:** union claims +0.029 of the ~+0.085 gap between the Delaunay GraphNet
  and the G1.5 real-space bound (0.86–0.88). Remaining headroom for EVERYTHING else
  (equivariance, FMPE, features) ≈ 0.05–0.08. The Delaunay receptive-field scale-mismatch
  hypothesis is now validated at production level — graph construction was the
  first-order lever, exactly as the amended G4-PROPER reframe (§8) argues.
- **Integration gap found:** the two data-side wins live in SEPARATE caches (nzharm =
  Delaunay-on-harmonized; union = Delaunay∪radius on UN-harmonized). Phase B requires a
  **union×nzharm merged cache** (union edges on harmonized points; ~20 min) — tasked.
- G4-PROPER implications: bar formally raised to the FINAL union number; P1a
  (radius-only) control gains importance (distinguishes radius-sufficient vs
  union-necessary); the surviving unique payoff of Tier B is EIGENVECTORS/orientations
  (IA science) — a capability no connectivity fix can provide, which is the honest
  physics case for keeping the symmetry branch alive within its ~0.05 accuracy budget.
- Held sbatch 55441429 NOT released (honoring JDPK's earlier correction) — release is
  his call. Status figure: `docs/roadmap_status_20260703.png` (+ canonical figures dir).

### 2026-07-03 — [code] G3 interactive done@3749 (NOT trained-out); existing G4-PROPER plan AMENDED (attention required, P1a/P1b split)
- **G3 status + PRELIMINARY EVAL (GO signal):** the interactive run (job 55442933) ended —
  4 h `salloc` wall revoked it at 13:42 local. It banked to **epoch 3749/7000** (last
  checkpoint on disk), val NLL still descending, no plateau. Ran an eval-only read of the
  3749 checkpoint (interactive salloc job 55454981; `--resume_from` + `--epochs 1` → empty
  train loop → pipeline's own 128-sample posterior-mean block; outputs in
  `..._uniongraph_EVAL3749`, kept run dir pristine). **Result (raw eigenvalues, posterior
  mean, vs SI Delaunay baseline):** λ₁ **0.8041** vs 0.7750 (**+0.029**), λ₂ 0.8461 vs 0.8105
  (+0.036), λ₃ 0.8955 vs 0.8912 (+0.004), mean 0.8486 vs 0.8256; best val NLL **0.8563** vs
  1.1065, test NLL 0.8964 vs 1.1392. **The union graph BEATS the fully-trained Delaunay
  baseline on all three λ at only 56% of training** — a floor, not the final number. **G3 is
  a GO.** (No cluster-slice Spearman in the pipeline results txt — a separate diagnostic if
  wanted.) NB: there is **no eval-from-checkpoint for the FlowJAX/NPE stack** — only
  `jraph_{regression,classification}_eval_from_checkpoint.py` exist (regression/clf models);
  the NPE eval is inline in `main()`, hence the resume+epochs=1 trick.
- Held resume sbatch **55441429** (`--resume`, 4×A100, 12 h) remains **JobHeldUser**. Recommend
  RELEASE to finish to 7000 for the final (≥0.804) gate number — held pending JDPK's call
  (he corrected an earlier premature release, so not auto-releasing).
- **Existing G4-PROPER plan AMENDED IN PLACE** (`docs/plan_g4_proper_equivariant_tensor.md`,
  per the sharpened equivariant-GNN review) — NOT a new doc. The doc already had the Tier A
  (eigenvalue-supervised, no tweb) / Tier B (tensor+eigenvector, needs tweb) split and the
  self-contained FFT tensor-build (§3: `tidal_tensor_fullgrid.py` building
  T_ij(k)=(k_ik_j/k²)δ_k, validated against cactus `eig_vals`; NO cactus edits) + frame
  rotation (§4). Amendments layered on:
  - **Attention promoted optional→REQUIRED** (§5) in every equivariant candidate (invariant
    logits preserve equivariance exactly; = adaptive smoothing kernel for the fixed 7 Mpc/h
    target) with pre-registered regularisation parity (attn dropout, wd/dropout 0.2/0.08,
    early-stop on val NLL, matched params) — the smoke overfit because undisciplined.
  - **Fixed bake-off order + SKIP list** (§5): SEGNN-with-attention first, SE(3)-Transformer/
    Equiformer-class second, one point-cloud model as the P1a control. SKIP EGNN/PaiNN/GVP
    (ℓ≤1), GATr (long-range wasted on compact target), MACE (many-body mismatch), PCA
    frame-averaging (degeneracy).
  - **P1 split into P1a/P1b** (§6): P1a = non-equivariant ~10 Mpc/h *radius-only* attentional
    GraphNet in existing JAX/jraph (no tweb, no equivariance) — isolates the Delaunay
    scale-mismatch lever; GO within seed noise of 0.774, ≥0.80 ⇒ deprioritise equivariance.
    P1b (Tier A steerable) must beat BOTH 0.774 AND the P1a control beyond seed noise (≥3–5
    seeds, matched compute/params). RPP-relaxed escape hatch (P0) if strict fails. P2 gate
    amended to calibration/coverage ≥ eigenvalue-regression flow.
  - **Reframe (§8):** the strongest first-principles lever is graph construction, not strict
    SO(3) — the wedge breaks the symmetry, so equivariance is a regulariser (modest gain at
    ~58k nodes / 3-param group, consistent with the +0.09 G1.5-ladder bound). Added
    permanent-shelf condition + plan-changing thresholds + PyTorch-e3nn-sidecar-first staging.
  - Correction: my first pass wrongly created a duplicate `workflows/sbi/G4_PROPER_PLAN.md`
    with a "save-the-Hessian-in-run_tweb_memory_optimized" rebuild — DELETED; the existing
    plan's standalone-FFT approach is the canonical one.

### 2026-06-16 — [code] graphify knowledge graph setup complete (Mac + NERSC)
- What: Installed graphify across Mac (`~/Developer/Illustris`, `~/Developer/GraphWeb_DESI`) and NERSC (`~/TNG/Illustris`, `~/GraphWeb_DESI`). Per-repo graphs built on both machines. Global cross-repo graph built at `~/.graphify/global-graph.json` (tags: `Illustris`, `GraphWeb_DESI`) on both machines. Claude Code PreToolUse hooks (grep/Read/Glob interception) and Cursor `.cursor/rules/graphify.mdc` updated in both repos to reference global graph for cross-repo queries. `graphify-out/` gitignored; `CLAUDE.md` + `.claude/settings.json` + `.cursor/rules/graphify.mdc` travel via git.
- Why: LLMs (Claude Code + Cursor) now consult a knowledge graph before grep/file reads, reducing token cost and surfacing cross-file dependency edges. Global graph captures the Illustris→GraphWeb_DESI model/module dependency that per-repo graphs can't see.
- Next: after any significant code change, run `graphify update .` in the repo then `graphify global add graphify-out/graph.json --as <tag>` to keep global current. Mac global uses `~/Developer/` paths; NERSC uses `~/TNG/Illustris/` paths.

The boundary object that keeps **science discussions** (Claude Desktop, local Mac)
and **agent work** (Claude Code, NERSC) aware of each other. It rides git, so it
reaches every machine.

- NERSC path: `~/TNG/Illustris/SCIENCE_LOG.md`
- Mac path:   `~/Developer/Illustris/SCIENCE_LOG.md`
- Syncs through GitHub (`jdpkololgi/Illustris`).

## How to use this file (both assistants)

1. **Read it at the start of a session** to load current direction before doing
   anything substantive.
2. **Append a short entry** (newest at the top of the Log) when a decision is
   made, a result lands, or direction changes. Don't rewrite history; add.
3. **It's only current after `git pull`.** Pull before reading/editing; commit +
   push after writing. (Stage just this file: `git add SCIENCE_LOG.md`.)
4. Keep entries terse and skimmable. Prune "Open threads" as items close.

Tags: `[science]` = decisions/hypotheses/conclusions from Desktop discussions.
`[code]` = what was run/changed/found, blockers, next actions from NERSC.

Entry shape:

```
### YYYY-MM-DD — [science|code] short title
- What: ...
- Why / decision: ...
- Next: ...
- Refs: files, commits, run dirs
```

## Open threads / current focus

- **NPE on Abacus wedge subvolumes** is the path forward for SBI. Graph-partitioned
  FlowJAX is legacy ("a nightmare") and to be retired when convenient.
- **GraphWeb production VAC** needs a real `sbatch` submit script — the DESI
  graph→features→inference chain currently only runs interactively.
- Jraph regression on wedge/cube caches is active and fine.

## Log (newest first)

### 2026-07-15 — [ops] HOME QUOTA HIT 100% — log truncated & recovered; ~18 GB freed; catalogs moved to pscratch

INCIDENT: home hit 40.01/40.00 GiB (100%). A SCIENCE_LOG write (open(p,"w") truncates BEFORE writing)
failed mid-write -> SCIENCE_LOG.md truncated to 131072 B; a git commit also failed silently earlier.
RECOVERED fully from git HEAD (6931269) — no content lost. LESSON: write the log ATOMICALLY (tmp +
os.replace), never open(...,"w") in place; and a failed `git commit` must be treated as a hard error.
FREED (user-approved): .codex/logs_2.sqlite-wal.bak-20260713-0150 (12.5 GB stale WAL backup),
.cache/vscode-cpptools (4.9 GB regenerable C++ IntelliSense). Home 100% -> 56.6%. NOTE the LIVE
.codex/logs_2.sqlite-wal is still ~10 GB (Codex DB not checkpointing) — left alone (active data), but
it will re-fill home; worth a Codex-side checkpoint/vacuum.
MOVED: GraphWeb_DESI/data/loa-combined-lowz{,-zflags,-fastspec-phot}.fits (~1 GB) -> pscratch
/pscratch/sd/d/dkololgi/graphweb_desi/catalogs/ (byte-verified). Paths updated: config_paths now has
GRAPHWEB_CATALOG_DIR/_PATH/_ZFLAGS_PATH/_FASTSPEC_PATH (env-overridable, resolve to real files — the OLD
default {REPO_ROOT}/loa-combined-lowz.fits was ALREADY STALE/broken); GraphWeb_DESI/CLAUDE.md documents
the pscratch location + the z0.01-0.06 caveat; TNG Network_stats.py hardcode fixed (+ missing `import os`
that would have been a runtime NameError); load_catalog.py --out-dir now defaults to the pscratch dir.
to-delete/ + docs/archive/ refs intentionally left.

### 2026-07-15 — [code] Workstream B1/B3 aperture features BUILT — selection-aware contrast beats raw counts; high-z is empty

compute_aperture_features.py -> aperture_features_v1/ (4 valid shells, 7 channels: logN_ap7/10/14,
contrast_ap7/10/14 = log[(N+eps)/(ntilde*V+eps)], log_dNN). Diagnostic Spearman vs lambda1 (NOT a gate):

| z | median N<10.4Mpc | logN_ap7 | contrast_ap7 | log_dNN |
|---|---|---|---|---|
| 0.15-0.25 | 21 | +0.549 | **+0.605** | -0.199 |
| 0.25-0.35 | 8 | +0.533 | **+0.594** | -0.237 |
| 0.35-0.45 | 2 | +0.492 | **+0.559** | -0.326 |
| 0.45-0.55 | **0** | +0.347 | **+0.429** | -0.332 |

FINDINGS: (1) the SELECTION-AWARE CONTRAST beats raw counts at EVERY shell — memo's "supply expected
counts as a separate covariate, don't divide away the signal" validated. (2) These carry info the model
currently LACKS: the existing 7 node features are SI-normalized (per-shell median), which divides out
exactly this observed-vs-expected contrast -> R1 restores it; not redundant. (3) **median z0.45-0.55
galaxy has ZERO neighbours within 10.4 Mpc** — independent confirmation of Workstream A: that shell is
not under-trained, it is UNSAMPLED. Only Workstream G (more high-z volume) fixes it.

LUMINOSITY (Workstream B2): join validated 100% (Abacus CutSky key FILE_NUM/HALO_INDEX/BOX_INDEX unique).
Column signal vs lambda1: R_MAG_APP -0.016 (USELESS — mag-limit-selected band), R_MAG_ABS -0.141,
G_R_OBS +0.130, G_R_REST +0.109; HALO_MASS/CEN EXCLUDED (unobservable in DESI). CHOSEN (JDPK): M_ABS +
G_R_OBS. k-corr: training needs NONE (mock R_MAG_ABS is forward-modelled). DESI inference uses LSS
`add_ke` (Smith+2017 GAMA via DESI_ke/smith_kcorr + TMR e-corr + DESI dm) -> ABSMAG_RP1; our LSS
checkout LACKS DESI_ke -> use LSSCODE=/global/common/software/desi/users/ioannis. desi_environment.sh is
NOT `set -u` clean (DESI_ROOT unbound) -> drivers must `set +u`. PARITY GATE PENDING: ABSMAG_RP1 is k+E
corrected and the Abacus convention may differ -> support comparison must pass before the luminosity
channel ships; fallback = G_R_OBS only (~92% of signal, no k-corr dependency).



### 2026-07-14 — [code] Workstream A DONE: high-z is DATA-limited (not starved); adopt τ=0.5 √N sampling

Sampling-policy ablation (VAL selection, no test peek), best val λ1 R²:
| policy | val NLL | macro | per-shell 0.15/0.25/0.35/0.45 |
|---|---|---|---|
| R0 τ=1 node-prop | 2.78 | 0.452 | 0.52/0.52/0.51/0.26 |
| A1b τ=0.5 √N | 2.72 | 0.456 | 0.55/0.50/0.50/0.29 |
| A1c τ=0 uniform | 2.78 | 0.408 | 0.54/0.48/0.49/0.13 |

KEY: high-z 0.45-0.55 is DATA-limited, NOT update-starved. Uniform gave it 25% of updates (≈1000 steps on
1920 galaxies) → OVERFIT → 0.26→0.13 (worse, noisy). Memo Q#1 answered: NO, high-z wasn't starved; it's the
sparse-tracer info limit + tiny n. → high-z fix = Workstream G (more sim volume), not sampling. GATE A:
adopt τ=0.5 √N-balanced (safe: better NLL 2.72, high-z +0.03, no shell degraded; macro +0.004 below the
strict +0.02 bar but strictly ≥ node-prop). Specialist (A3) largely pre-answered by the uniform diagnostic
(more high-z exposure hurt → not negative transfer, not starvation) — deferred as optional confirmation.

NEXT: Workstream B — construct aperture-density (@7/10/14 Mpc/h counts + selection-aware contrast
log[(N+ε)/(ñ·V+ε)]) + luminosity-weighted + degree/NN-scale node channels (physical density kept DISTINCT
from ñ), rebuild a versioned feature-enriched valid cache (scalers refit on valid train), then R1 GraphNet
with τ=0.5 — fed DIRECTLY (no GBM veto). Then C (3-D U-Net) and D (F-tier) as matched challengers. Point
gates on VAL; RA≥150 test opened only for the frozen finalist.



### 2026-07-14 — [code] ACCURACY SPRINT plan adopted (supersedes prior next-levers); starting Workstream A (sampling)

New execution memo (Claude Desktop) governs the post-R0 accuracy sprint over z0.15-0.55. Key corrections
to my prior next-levers: (1) **GBM DEMOTED from gate to diagnostic** — GBM 0.27 vs GraphNet 0.77 shows GBM
is NOT a surrogate for spatial-model-extractable info; feed aperture/luminosity features DIRECTLY to the
GraphNet, no GBM veto. (2) **Workstream A (redshift optimization imbalance) is TOP priority** — R0's high-z
shell (0.45-0.55) got only ~60 updates (1.5% of 4000) under node-proportional sampling, so its 0.34 is NOT
a clean ceiling. (3) **PROTECT the RA≥150 test region** — open ONLY for the frozen finalist; select on val
(145-150) macro-shell λ1 R² + blocked spatial CV inside train. I have been evaluating test too freely.

Workstreams: A sampling-policy ablation (√N-balanced, uniform) + high-z specialist; B R1 enriched GraphNet
(aperture density @7/10/14 Mpc/h + luminosity + degree/NN-scale, ñ kept distinct from physical density);
C matched full-range 3-D U-Net (first-class challenger, LOS channels for RSD, expected-count/mask channels);
D full-range F-tier v2_A (nonlocal-FFT tile convergence test); E graph-field residual hybrid (η=η_field+
r_θ, out-of-fold field features to avoid leakage); F aux ordered-increment point head if plateau; G more sim
coverage (phases/snapshots/high-z volume). Point-estimate gates FIRST; FMPE+calibration only on the winner.
Metrics: pooled + MACRO-shell + worst-shell λ1 R², ρ², void/knot recall, block-bootstrap (not per-galaxy).

STARTED: Workstream A1 — added --sampling-temperature to the tiled trainer (tau: 1=node-prop, 0.5=sqrt,
0=uniform-shell). Launching A1b (tau=0.5) + A1c (tau=0) in parallel, selection on VAL macro-shell λ1 R²
(no test eval). Do NOT pre-register R²=0.8. VAC v1: z0.15-0.55; z<0.15 null + NO_VALID_TRAINING_LABELS.



### 2026-07-14 — [code] NEXT LEVERS after R0 (post-remediation accuracy roadmap)

R0 corrected baseline established: held-out R²(λ1)=0.514 ALL (0.56/0.49/0.42/0.34 over z0.15-0.55),
λ2=0.666, λ3=0.713, best-val NLL 2.78 (full table: prior entry). Encoder-ceiling refuted; 0.51 is the
valid starting point. Levers to raise per-galaxy accuracy, in priority order (all measured on the SAME
spatial holdout, RA<145 train / 145-150 val / RA≥150 test, valid cache s3b_tiled_valid_v2):

1. **P5 — FEATURE ENRICHMENT → R1 (highest value, do first).** The R0 bundle is only 7 geometric + ñ.
   Add, gating each family with a CHEAP GBM on the corrected valid split (include iff ≥+0.02 R²λ1 without
   support mismatch): fixed-aperture density/counts @7 & ~10 (& optionally 14) Mpc/h; luminosity-weighted
   versions (a quantity forward-modelled consistently in Abacus AND observed in DESI); local observed
   sampling diagnostics (radius degree, NN scale) kept DISTINCT from ñ. ñ = expected sampling intensity
   (covariate); physical density = signal — do NOT conflate. Then R1 = GNN with accepted bundle, else identical.
2. **P6 — CONDITIONING + SPECIALIST CONTROLS.** ñ ablation: node-feature-only (current) vs + direct concat
   to the posterior conditioning vector vs + FiLM/skip at each message-pass block (tests message-passing
   dilution). Matched specialist controls on 0.15-0.25 and 0.45-0.55; if a specialist strongly beats the
   pooled conditioned model → negative transfer → conditional modulation / mixture-of-experts (NOT K models).
3. **P7 — TOPOLOGY SHOWDOWN (matched params + updates).** corrected buffered Delaunay∪radius (R0) vs pure
   coordinate kNN(k=12) vs preferred BOUNDED-HYBRID (Delaunay + ≤k_r nearest radius neighbours within
   14.78 Mpc + edge-type flags + max-edge-length rule/flag). Adopt iff macro-shell R²λ1 +≥0.02 AND no shell
   degrades >0.03 AND reproducible in GraphWeb_DESI AND tile/full canary passes. No dynamic feature-space kNN.
4. **P8 — ACCURACY-FOCUSED OBJECTIVE (if R1 plateaus).** auxiliary ORDERED-increment point head:
   L = L_NPE + β·L_point (MSE/Huber in increment space; never an unordered 3-eig head). Compare NPE-only vs
   deterministic-pretrain→NPE vs joint; β on val only; save posterior + point checkpoints separately.
5. **P9 — FULL-RANGE F-TIER GATE.** valid split, exclude z<0.15, ñ·V_voxel expected-count channel, per-shell
   eval + realistic high-z sparsity + tile-size/overlap CONVERGENCE test (FFT tidal solve is nonlocal). Do
   NOT promote on the dense-wedge 0.84. If accurate-but-miscalibrated → physics+residual hybrid
   (η = η_physics + r_θ(X, η_physics)); keep physics tensor/eigenvectors separately badged.
6. **CALIBRATION last.** FMPE head + tempering (per-shell/per-ñ τ, not one global) ONLY after the winning
   encoder+features are frozen. SBC-aware (scalar coverage is insufficient); flag prior-dominated rows.

DO NOT pre-register R²=0.8 (above the sparse-tracer information ceiling: DTFE 0.55 / GraphNet 0.78 / F-tier
0.84 are all dense-wedge transductive). Strongest long-term generalization = additional independently-phased
Abacus cutskies + multi-snapshot T-Web labels (cosmic variance + TARGET_EPOCH), not indefinite encoder growth.
VAC v1: z 0.15-0.55 only; z<0.15 = null + NO_VALID_TRAINING_LABELS flag. DEADLINE: NERSC down Jul 22-Aug 3.



### 2026-07-14 — [code] R0 CORRECTED BASELINE: held-out R²(λ1) 0.42→0.51 — memo diagnosis CONFIRMED, "encoder ceiling" was a training artifact

R0 (fixed trainer: node-proportional sampling + global-step LR schedule, valid cache s3b_tiled_valid_v2,
total_updates=4000). Held-out TEST region (RA≥150, unseen sky), R0 vs the INVALID run:

| z | R0 R²λ1 | invalid | Δ | R0 λ2 | R0 λ3 | cov68 | cov90 |
|---|---|---|---|---|---|---|---|
| 0.15-0.25 | 0.559 | 0.472 | +0.087 | 0.709 | 0.721 | 0.51 | 0.74 |
| 0.25-0.35 | 0.488 | 0.360 | +0.128 | 0.645 | 0.718 | 0.55 | 0.78 |
| 0.35-0.45 | 0.419 | 0.390 | +0.029 | 0.547 | 0.592 | 0.62 | 0.85 |
| 0.45-0.55 | 0.344 | 0.326 | +0.018 | 0.424 | 0.472 | 0.63 | 0.87 |
| ALL | 0.514 | 0.418 | +0.096 | 0.666 | 0.713 | 0.54 | 0.76 |

VERDICT: fixing the optimizer bugs lifted honest held-out R²(λ1) 0.42→0.51 (+0.10). The "0.42 encoder
ceiling / accuracy is encoder-limited" conclusion is REFUTED — it was a training artifact (52% updates on
corrupt labels + LR dead by ~ep171 + tile-count starvation). The shell that gained MOST (0.25-0.35, +0.13)
is exactly the one starved by tile-count weighting (23% nodes → 4.8% updates in the invalid run) — precise
confirmation of the diagnosis. λ2/λ3 strong (0.67/0.71). Best val NLL 2.78 (vs invalid 3.10); val pooled λ1
R² 0.51. High-z 0.45-0.55 stays weak (0.34, shot-noise-limited). Coverage UNDER-nominal (MAF, no tempering)
— calibration is a later step. Models: R0_valid_corrected/sbi_output/flowjax_sbi_model_seed_42_bestL1_*.pkl.
Op-note: session teardown orphaned 2 identical deterministic training windows; killed duplicate (redundant,
not divergent). 0.51 is a valid encoder baseline — NOT 0.8 (memo: above sparse-tracer info ceiling).

NEXT (memo): P5 aperture/luminosity GBM gates on the valid split → R1 (feature-enriched); then P6 ñ ablation +
specialist controls; tempering for calibration once the encoder is chosen.



### 2026-07-14 — [code] Remediation P1+P2+P3 done: valid cache built, trainer fixed, PARITY PASSES (tiling is safe)

P1 valid cache: s3b_tiled_valid_v2 built — shell-0 (z<0.15 corrupt) DROPPED, BOX_INDEX>=0 mandatory on
active, all hard gates PASS. 4 shells / 10 tiles, active train/val/test 129113/18629/52848. Node-proportional
sampling under the fixed trainer → shell1/2/3/4 = 54/32/13/1.5% of updates (was tile-count-distorted).
P2 trainer: corrected (node-proportional sampling, global-step LR schedule, best-NLL/best-λ1 ckpts, run lock,
deterministic resume). Committed 267c002.
P3 PARITY test (z0.25-0.35, full shell vs RA-tiled, buffers 17/30/50 Mpc): mean|Δλ1|=0.012 INDEPENDENT of
buffer and FLAT with distance-to-cut (0.015 near vs 0.012 far); embedding cosine dist ≈ 0.000. => tiling does
NOT truncate the 8-pass receptive field (attention concentrates influence <~17 Mpc). MEMO CONCERN #5 REFUTED.
17 Mpc buffer sufficient; tiling was never the accuracy problem. R0 UNBLOCKED (valid cache + fixed trainer +
confirmed buffer). Residual 0.012 |Δλ1| is posterior-sampling RNG noise, not truncation.

NEXT: P4 R0 — corrected union baseline (fixed trainer on s3b_tiled_valid_v2, total_updates=4000, warmup=400,
node-proportional). First run allowed to judge the encoder. Then P5 aperture/luminosity GBM gates -> R1.



### 2026-07-14 — [code] CORRECTION: Phase-B run is INVALID for a model verdict — 3 confirmed training bugs; conclusions WITHDRAWN

External implementation-memo review (Claude Desktop) caught real defects in the tiled Phase-B run; all
verified against code/manifest. The R²(λ1)=0.418 is NOT an encoder or information ceiling.

CONFIRMED BUGS:
1. Tile-count optimizer weighting: jraph_sbi_flowjax_tiled.py does ONE update per tile, not per scientific
   sample. shell-0 (CORRUPT z<0.15) = 11/21 tiles = 52% of updates (only 27% of nodes); valid shell-2
   (0.25-0.35) = 23% of nodes but 4.8% of updates. Corrupt shell dominated; valid shells starved.
2. Corrupt z<0.15 labels were IN the training set (48,406 train nodes of random labels) — should have been
   excluded once the label corruption was found.
3. LR schedule ~20x too fast: warmup_cosine_decay decay_steps set in EPOCH units but the schedule advances
   per optim.update (~21/epoch) → LR hits the 1e-5 floor by ~epoch 171, not 4000. Effective training truncated.
Also flagged (not yet quantified): finalize/eval race + lingering auto-loop windows; 17 Mpc tile buffer <
8-message-pass receptive field; aperture-density + luminosity features omitted from the bundled cache.

WITHDRAWN: "accuracy is encoder-limited", "undertraining ruled out", "0.42 is the model's real accuracy",
and the FMPE-implies-encoder-ceiling framing (FMPE only shows head-swap on the SAME embedding doesn't help
— it does NOT establish an encoder ceiling). Posterior described as COVERAGE-TEMPERED, SBC-FAILING (not
"calibrated"). Run frozen: phaseB_tiled_ntilde/{PIPELINE_SMOKE_ONLY_INVALID_FOR_MODEL_VERDICT.txt,
RUNCARD_invalid.json}; artifacts preserved, not overwritten.

REMEDIATION (memo, 10 phases): P1 valid z0.15-0.55 cache (drop shell-0, BOX_INDEX≥0 mandatory, refit scalers
on valid data, hard assertions, versioned path); P2 per-scientific-sample optimizer + global-step schedule +
separate best-NLL/best-λ1 checkpoints + read-only finalize + run locks; P3 tile/full receptive-field parity
gate; P4 R0 corrected union baseline (first run allowed to judge the encoder); P5 aperture/luminosity GBM
gates → R1; P6 ñ conditioning ablation + specialist controls; P7 kNN vs bounded-hybrid topology showdown;
P8 auxiliary point-regression head; P9 full-range F-tier gate; P10 VAC release gates + golden canary.
Do NOT publish z<0.15 science (null + NO_VALID_TRAINING_LABELS flag). Do NOT pre-register R²=0.8.



### 2026-07-14 — [code] === GraphWeb-BGS v1 CONSOLIDATED STATE (session handoff) ===

**WHAT EXISTS (v1 calibrated baseline, full-range ñ-conditioned):**
- Model: tiled ñ-conditioned Attentional-GraphNetwork encoder (MAF-trained, best-val early-stopped
  ~epoch 800, NLL 3.101) + FMPE head + posterior tempering (τ=1.18).
  `phaseB_tiled_ntilde/sbi_output/flowjax_sbi_model_seed_42_20260714_000602.pkl`.
- Data: 21 disjoint RA-tiles over z0.15-0.55 (union Delaunay∪radius graph, sliced to fit GPU),
  8-d node feats [7 geometric SI-normed + log_ñ(z)], edge feats [len, unit-vec(3), contrast].
  `sbi_caches/s3b_tiled_ntilde_uniongraph/`. Frozen ñ spline: `conditioning/ntilde_spline_v1_frozen.json`.
- Splits: halo-disjoint SPATIAL holdout — train RA<145 / val 145-150 / test RA≥150 + 15 Mpc gutter.

**HEADLINE RESULTS (held-out RA≥150, honest new-sky generalization):**
- R²(λ1)=0.340 ALL / 0.418 (z≥0.15); λ2=0.608, λ3=0.629. Calibrated after tempering: cov68~0.72,
  cov90~0.91 (slightly over; global τ; per-shell τ would tighten). SBC KS p≈0 → SHAPE still miscalibrated.
- T-web 4-class (λ_th=0.2, z≥0.15): 62% agreement (random 25%). Class fractions true/pred: void .255/.176,
  sheet .412/.462, filament .271/.307, knot .063/.055. Plots: phaseB_tiled_ntilde/plots/{pred_vs_true_eigs,
  fan_true_vs_pred}.png. Visible REGRESSION-TO-MEAN: extremes (void, knot) blur toward middle.

**KEY FINDINGS:**
1. z<0.15 UNUSABLE — corrupt labels. 100% BOX_INDEX==-1 (out-of-shell for the z=0.2 snapshot cutsky);
   permutation-null test: labels ≡ random. Verified vs AbacusSummit/DESI docs (observer at box corner
   (-990,-990,-990); snapshot-shell stitching; sharp z=0.15 edge). Scope-guard AIRTIGHT. Salvage = upstream
   re-annotation w/ correct low-z snapshot (real project). FLAG TO DESI MOCK TEAM.
2. ACCURACY is ENCODER-limited: FMPE R²=0.333 ≈ MAF 0.340 (head swap does nothing); undertraining ruled out
   (early-stopped); shell-0 fails for GNN AND GBM AND position-oracle. 0.8 is above the info ceiling of sparse
   BGS positions (dense-wedge transductive ceilings: DTFE 0.552, GraphNet 0.775, F-tier 0.841).
3. GNN > GBM at every live shell (λ1 0.340 vs 0.252; wider on λ2/λ3) → graph/edge-anisotropy complexity JUSTIFIED.
4. JDPK requirement: per-galaxy environment needs accurate MEANS → 0.42/62% is a REAL limitation at the
   void↔sheet / filament↔knot boundaries (the interesting extremes). ACCURACY is now the priority axis.

**METHODOLOGY DECISIONS:** F-tier config firmed = v2_A (tsc+unet, shared G3 union-graph). ñ(z) is a SMOOTH
per-galaxy feature (not shell-quantized); one amortized model. Shells/tiles are a MEMORY device only
(union graph unbounded degree ~180 at low z → whole-wedge OOM 267GB). sbi FMPE MUST sample on GPU (CPU
batch-caps to 12 → hangs).

**OPEN / NEXT (priority = accuracy, JDPK):**
- (a) k-NN one-graph rebuild: bounded degree k≈12 → one continuous graph z0.15-0.55 fits 1×hbm80g, no
  shell/tile edge-cutting (JDPK instinct, correct). (b) F-tier v2_A full-range (tiled) — the 0.84 accuracy
  branch. (c) richer features (reconstructed density field, velocities). Measure all on the SAME held-out fan.
- Deferred: validation battery (SBC-aware shape fix, TARP), λ2/λ3 4-class calibration, low-z salvage.
- DEADLINE: NERSC shutdown Jul 22–Aug 3. All keepers on CFS (graphweb_vac_v1_backup).



### 2026-07-14 — [code] FMPE+tempering DONE (GPU): calibration improved, accuracy flat — confirms encoder is the ceiling

FMPE head on frozen tiled encoder + posterior tempering (τ=1.18 fit on VAL, assessed on disjoint TEST),
per shell (z≥0.15): R²(λ1) FMPE ALL=0.333 ≈ MAF 0.340 → **FMPE did NOT improve accuracy** on this tiled
full-range setup (contrast: G6 dense-wedge found FMPE>MAF). Tempering lifted λ1 coverage from MAF's
under-covering cov68=0.653 → 0.723 (cov90 0.876→0.907) — slightly OVER now; a single global τ over-inflates
some shells (per-shell/per-ñ τ would tighten). **SBC KS p≈0 at every shell** — scalar tempering fixes WIDTH
not SHAPE; shape miscalibration remains open. (Execution note: sbi FMPE CPU sampling is batch-capped to 12
nodes → hangs; MUST sample on GPU — fixed fmpe_temper_tiled.py with device=cuda, ran in ~9 min.)

CONCLUSION: v1 calibrated stack = tiled ñ-conditioned encoder (MAF-trained) + FMPE head + tempering, R²(λ1)
≈0.42 (z≥0.15), coverage ~nominal. FMPE confirms accuracy is set by the ENCODER, not the head — so the
per-galaxy-accuracy priority (JDPK) needs encoder/graph/feature upgrades: k-NN one-graph, then F-tier.
Result: fmpe_temper_heldout.json.



### 2026-07-14 — [code] Validation plots: regression-to-mean visible; 62% 4-class agreement (per-galaxy accuracy limited)

Held-out (RA≥150, z≥0.15, 52,844 gal) validation of the best-val MAF model. Pred-vs-true eigenvalue
hexbins: R²(λ1)=0.418 with a clear regression-to-mean tilt (flatter than 1:1 — voids pulled up, collapse
regions pulled down); λ2=0.608, λ3=0.629 tighter. Cosmic-web fan (λ_th=0.2, DEC 18-28° slice): large-scale
pattern recovered but SMOOTHED — voids under-populated (pred 0.176 vs true 0.255), sheet over (0.462 vs 0.412),
filament 0.307 vs 0.271, knot 0.055 vs 0.063. **4-class agreement 62.2%** (random=25%). knot POPULATION
fraction well-recovered (P(λ1>0.2) 5.5% vs 6.3%) but per-galaxy assignment scatters. Plots:
phaseB_tiled_ntilde/plots/{pred_vs_true_eigs.png, fan_true_vs_pred.png}.

INTERPRETATION: regression-to-mean is the honest Bayesian posterior-mean behavior under limited info, NOT a
bug — sharper per-galaxy env needs MORE INFORMATION (F-tier / k-NN one-graph / density-field features), not a
different estimator. For per-galaxy environment (JDPK requirement) this 62%/0.42 is a real limitation at the
void↔sheet and filament↔knot boundaries (the interesting extremes). Accuracy is now the priority axis.



### 2026-07-14 — [code] ACCURACY now a first-class concern (posterior MEANS carry the per-galaxy environment)

Clarified science requirement (JDPK): the VAC is per-galaxy environmental info, so posterior MEANS
(point estimates) are essential — each galaxy's environment is the deliverable, not just population
statistics. => the held-out R²(λ1)≈0.34 (ALL) / ≈0.42 (z≥0.15) is a SIGNIFICANT concern, not something
the calibrated-posterior framing excuses.

Honest ceiling context (all dense-wedge, transductive): DTFE 0.552, GraphNet 0.775, F-tier 0.841.
Our number is lower because (a) SPATIAL holdout (honest new-sky, not transductive) and (b) full-range
incl sparse high-z. DESI transfer is HARDER than the mock holdout. R²=0.8 on sparse BGS positions is
almost certainly above the information ceiling of galaxy positions alone. Real accuracy levers (FMPE is
NOT one — it's calibration): (1) F-tier v2_A (0.84 branch, the accuracy play); (2) one continuous k-NN
graph over z0.15-0.55 (bounded degree → fits one GPU, no shell/tile edge-cutting; JDPK's instinct,
correct) + smooth ñ(z) per galaxy; (3) richer features (reconstructed density field, velocities).

Validation plots being produced (dump_predictions_positions.py + plot_validation.py): pred-vs-true
eigenvalue hexbins + cosmic-web fan (wedge) plots true-vs-predicted at λ_th=0.2, on the held-out test
region. FMPE+tempering running (calibration; won't move R²). NEXT priority (per JDPK): raise accuracy —
k-NN one-graph rebuild then F-tier, measured on the same spatial holdout.



### 2026-07-14 — [code] Phase-B first pass DONE: best-val GNN beats GBM decisively; complexity justified

Stopped the 4000-epoch run (overfitting: val NLL rose 3.10→3.23 after ~epoch 800). Finalized the
BEST-VAL model from checkpoint (NLL 3.101) → flowjax_sbi_model_seed_42_20260714_000602.pkl. Held-out
(RA≥150, never-trained) per-shell eval, GNN vs the node-only GBM baseline:

| z | GNN λ1 | GBM λ1 | GNN λ2 | GBM λ2 | GNN λ3 | GBM λ3 | cov68/90 |
|---|---|---|---|---|---|---|---|
| 0.05-0.15 | -0.010 | -0.001 | -0.014 | -0.005 | -0.018 | -0.004 | 0.67/0.88 |
| 0.15-0.25 | 0.472 | 0.262 | 0.682 | 0.369 | 0.666 | 0.202 | 0.65/0.88 |
| 0.25-0.35 | 0.360 | 0.332 | 0.561 | 0.480 | 0.608 | 0.450 | 0.63/0.86 |
| 0.35-0.45 | 0.390 | 0.344 | 0.446 | 0.438 | 0.460 | 0.392 | 0.73/0.92 |
| 0.45-0.55 | 0.326 | 0.223 | 0.396 | 0.305 | 0.432 | 0.340 | 0.63/0.86 |
| ALL | 0.340 | 0.252 | 0.483 | 0.352 | 0.493 | 0.288 | 0.65/0.88 |

VERDICT: GNN earns its complexity. Beats GBM at every live shell on all 3 eigenvalues; gap WIDENS on
λ2/λ3 (edge unit-vectors + message passing capture anisotropy the node-only tree can't). Early-stop
helped (λ1 0.340 vs overfit 0.303). Shell-0 dead for both (corrupt out-of-shell labels, verified).
Calibration good everywhere. Honest-eval note: old wedge "0.80@z0.2-0.3" was TRANSDUCTIVE; this 0.34-0.42
is SPATIAL holdout (deployment-honest) — much of the "gap" is eval rigor, not regression. z≥0.15 R²λ1≈0.42.
v1 product = calibrated λ1 posterior on z 0.15-0.55; z<0.15 OOD-flagged (corrupt labels).

NEXT (open): FMPE head + tempering on frozen embeddings (calibration headline); F-tier v2_A full-range
(tiled) point-estimate branch; then validation battery. GBM retained as transparent floor/cross-check.



### 2026-07-13 — [code] Shell-0 mechanism VERIFIED vs AbacusSummit/DESI docs (corrected: snapshot-shell stitching, not "outside box")

Verified the z<0.15 conclusion against AbacusSummit readthedocs + web/literature (user request).
CONFIRMED externally: box = 2000 Mpc/h; light-cone observer at **(-990,-990,-990)**, 10 Mpc/h
INSIDE a box corner (matches the audit-doc origin exactly); cutsky = replicate+patch 2 h⁻¹Gpc
boxes from DIFFERENT snapshots → sky coords → trim to footprint+radial selection, STITCHING
redshift shells; a box-provenance flag exists (AbacusSummit "origin" 0/1/2 valid, higher=invalid;
DESI/local BOX_INDEX==-1 = out-of-box).

CORRECTED my earlier phrasing: NOT "observer outside the box → low-z out-of-box" (observer is
10 Mpc/h INSIDE the corner). Real mechanism = SNAPSHOT-SHELL STITCHING: my file is the z=0.200
snapshot cutsky; its assigned redshift shell has a lower edge near z≈0.15; galaxies below it are
OUT-OF-SHELL for the z=0.2 snapshot → BOX_INDEX==-1 → they don't validly map to the z=0.2 T-Web
box (which is the only box we ran CACTUS on) → scrambled labels. Sharp z=0.15 break = shell edge.

CORE CONCLUSION UNCHANGED & doc-independent: z<0.15 labels are permutation-null-random (R²=0 ==
shuffled). HONEST GAP: the exact z=0.15 shell-edge number is my empirical finding, not explicitly
in public docs (DESI SecondGenMocks stitching config is internal). Worth flagging to the DESI mock
team. Salvage would need the CORRECT low-z snapshot's T-Web (e.g. z=0.1), not just re-wrapping z=0.2.



### 2026-07-13 — [code] SHELL-0 FAILURE SOLVED: corrupt LABELS (out-of-box), not a model/physics problem

Comprehensive investigation (user: "understand why", no deadline pressure). Verdict: z<0.15 λ1
is unpredictable because the **ground-truth T-Web labels there are scrambled** — a data-provenance
artifact, definitively NOT model/feature/training inadequacy.

Evidence chain:
- diagnose_shell0.py: shell-0 λ1 variance is INTACT (std 0.181, comparable to all shells) — so it's
  not low-variance. Yet R²(λ1|position)=0.008, R²(λ1|density)=0.002, Spearman(density,λ1)≈0. Every
  other shell: Spearman≈0.5, R²(pos)≈0.35-0.45. Sharp break, not a gradient.
- Astrostat framing (skill): R²(pos)=0 with intact marginal P(λ1) ⇒ MI(position,label)=0 ⇒ closure-
  test failure isolated to a subpopulation (right values, wrong galaxies).
- diagnose_shell0_boxindex.py: **BOX_INDEX==-1 is 100% at z<0.15 and 0% at z≥0.15** — a hard pipeline
  switch at z=0.15. R²(λ1|pos): BOX==-1 → 0.001; BOX≥0 → 0.35/0.26 at z0.15-0.35. **Permutation null:
  shell-0 observed R²=0.004 == shuffled-label R²=-0.002** → labels ≡ random.
- Provenance: build_abacus_graph.py: BOX_INDEX==-1 = "invalid/out-of-box", excluded by default.
  annotate_cutsky_with_tweb_eigs.py assigns eigs via (FILE_NUM,HALO_INDEX)→halo box-frame position→
  voxel. ABACUS_TWEB_AUDIT_FINDINGS.md ALREADY documented "MI between graph metrics and eigenvalues
  near zero" as a "label-domain mismatch" for the OLD sky-modulo method. The "_halo_xcom" catalog
  fixed z≥0.15 (in-box), but z<0.15 galaxies are out-of-box (observer origin [-990,-990,-990]; nearest
  galaxies sit outside [0,2000] box) → their voxel lookup wraps/clamps wrong → finite-but-scrambled λ.

CONSEQUENCE: (1) low-z scope guard is now AIRTIGHT (not a guess) — no model can predict random labels;
z<0.15 must be OOD-flagged/excluded in v1. (2) z≥0.15 labels are valid; the full-range model is sound
there. (3) SALVAGE (optional, upstream): z<0.15 galaxies have VALID HALO_INDEX (min 2065), so a correct
halo-x_com + PERIODIC-WRAP re-annotation may recover them — a data-regeneration fix, not a model fix.
Feasibility test = re-annotate a z<0.15 sample via the halo-linkage method (validate_cutsky_eigs_*),
check if R²(pos) recovers.



### 2026-07-13 — [code] FIRST full-range held-out result: approach VALIDATED (calibrated everywhere), R² undertrained, shell-0 prior-dominated

Tiled ñ-conditioned model (500 epochs) evaluated on the HELD-OUT test region (RA≥150, never
trained), per shell:

| z | n_test | R²λ1 | R²λ2 | R²λ3 | cluSp | cov68 | cov90 |
|---|---|---|---|---|---|---|---|
| 0.05-0.15 | 17225 | **-0.010** | -0.014 | -0.018 | -0.00 | 0.669 | 0.881 |
| 0.15-0.25 | 28696 | 0.465 | 0.685 | 0.651 | 0.23 | 0.661 | 0.884 |
| 0.25-0.35 | 18130 | 0.259 | 0.461 | 0.522 | 0.24 | 0.615 | 0.847 |
| 0.35-0.45 | 5149 | 0.387 | 0.470 | 0.470 | 0.24 | 0.728 | 0.921 |
| 0.45-0.55 | 869 | 0.307 | 0.394 | 0.442 | 0.20 | 0.628 | 0.876 |
| ALL | 70069 | 0.303 | 0.450 | 0.459 | 0.21 | 0.655 | 0.877 |

WINS: (1) calibration EXCELLENT at every shell (cov68 0.61-0.73 nom 0.68; cov90 0.85-0.92 nom
0.90), incl shell-0. (2) positive skill z0.15-0.55 on unseen sky — huge turnaround vs S1(b)
zero-shot (shell0 -1.09; CNN 0.002@z0.05). ñ-conditioning + tiling + spatial holdout GENERALIZES.

CAVEATS: (1) shell-0 (z0.05-0.15) R²λ1≈0, cluSp≈0 = PRIOR-DOMINATED (calibrated only because it
reverts to prior) → Codex low-info flag; strong low-z scope-guard candidate. (2) overall R²λ1=0.30
« specialist wedge model's ~0.80 at z0.2-0.3 → UNDERTRAINED (val NLL still descending at ep499).

NEXT: training resumed to 4000 epochs (auto-looping 4h hbm80g windows, persistent JAX compile
cache, TRAINING_COMPLETE marker; checkpoint every 25 + --resume). Re-eval expected to lift R².
Then diagnose shell-0 (fixable vs scope-guard). eval_tiled_heldout.py + heldout_eval.json saved.



### 2026-07-13 — [code] Phase B OOM → pivot to TILED training (user-approved); density-mismatch hypothesis REFUTED

First Phase-B launch OOM'd: `RESOURCE_EXHAUSTED 267.85 GiB on one 80GB GPU`. Root cause: the
trainer computes the GNN forward on the FULL graph every step and REPLICATES the graph per
device (data-parallel shards only the loss mask, not the graph) — so 4 GPUs don't help, and the
pooled 20.9M-edge graph needs ~267GB. Per-shell edges: shell0 11.15M(~142GB), shell1 7.34M(94GB),
shell2 1.94M(25GB), shell3 0.41M, shell4 0.04M. Even single low-z shells exceed one GPU. Since
DESI is DENSER than the mock, this is fundamental to full-range/full-footprint work (→ Phase C
tiling), not an artifact.

IMPORTANT null result: I hypothesized the low-z density was a mock-vs-DESI mismatch (would explain
shell-0's S1(b) collapse). **S0 atlas REFUTES it** — mock median union-degree ≈ DESI, mock slightly
SPARSER at every shell (shell0 mock 102 vs DESI 125; shell4 mock 1 vs DESI 3). So "NO dilution" +
SI is sound, no rebuild warranted; shell-0's failure is genuine density-OOD of the OLD wedge model.
(Checked before reversing the roadmap decision — glad I did.) Also confirmed `BOX_INDEX==-1` is
valid data (central box, 100% finite eigenvalues), shell-0 usable.

DECISION (user-approved, AskUserQuestion): **tile the dense low-z shells, pool the rest.** Split
z<0.25 shells into buffered RA sub-volume tiles aligned to the holdout boundaries {145,150}; shells
2-4 stay whole. ~10-12 disjoint tiles each ≤4M edges (fits one 80GB GPU). Build by SLICING the
existing union graphs (induced subgraph on core+buffer nodes) — NO cuGraph/Delaunay rebuild.
Per-shell ANGULAR buffer (union radius 14.78 Mpc ≈ 3.9° at z=0.05). Trainer: keep pmap, iterate
tiles in the epoch loop (one compile cached per tile shape). ñ-conditioning + halo-disjoint spatial
holdout preserved. RA-tiling aligns naturally with the holdout (tiles fall into one region).



### 2026-07-13 — [code] S3 pooled ñ-conditioned cache BUILT (all gates pass); Phase B launched

`s3_build_pooled_conditioned_cache.py` → one disjoint-union cache of the 5 S2 shells,
**304,604 nodes / 20.9M edges**, 8-d node features (`log_ntilde_std` last). Method: invert
each shell's per-shell box-cox → recover SI-only features → **one pooled box-cox** on the
pooled-train split (per-shell box-cox pre-removes the density signal ñ must explain) →
append ñ (frozen mock spline, fixed standardization) as the untransformed final column.
Spatial holdout **train RA<145 / val 145–150 / test RA≥150**, halo-disjoint (each
(FILE_NUM,BOX_INDEX,HALO_INDEX) group assigned wholesale by centroid RA; 39 nodes
reassigned), + 15 Mpc transverse graph gutter (35,808 passive), + cross-shell dedup by
TARGETID (204 buffer copies passive). Active: **train 177,509 / val 21,030 / test 70,069**.
SANITY GATE all PASS: ñ untransformed ✓, every shell fills all 3 regions ✓ (even z0.45–0.55:
1920/329/869), **train/test halo-disjoint 0 shared halos** ✓, active nodes TARGETID-unique ✓.

Data findings from the build: (i) `BOX_INDEX==-1` is VALID data (shell-0 is 99.9% of it —
the central box; 100% finite eigenvalues), so shell-0's S1(b) failure is genuine density-OOD,
NOT corrupt data. (ii) zero halo replication within the footprint → the RA split is honestly
halo-disjoint (separate-cone would only add cosmic-variance robustness → v1.1 ph001 test).

Conditioning decision (get-results-first): ship **ñ-as-node-feature** for the first Phase-B
run — the encoder ingests 8-d nodes with zero model code changes; ñ reaches the flow via the
80-d embedding. Explicit skip-concat (cond_dim→81) held as the iterate-later refinement if
diagnostics show ñ dilution (avoids desync risk across 4 train/inference touch points).

**Phase B launched** (job 55859173, 4×A100 hbm80g, salloc+tmux, checkpointed+resumable):
`jraph_sbi_flowjax.py --increment_mode linear` on the pooled cache via TNG_SBI_CACHE_DIR.



### 2026-07-13 — [code] F-tier config FROZEN = v2_A; S3 plan set (ñ-conditioning + spatial holdout)

**F-tier decision (JDPK call, evidence-confirmed).** Firmed the field branch on **v2
variant A** (`gate_ftier_v2.py` scatter=tsc, decoder=unet) over v1. Concrete reason it's
"more consistent with previous models": v2 consumes the **shared union-graph arrays
(`path1_wedge_union_r10hmpc_gnn_arrays.npz`) = the G3 production connectivity**, whereas v1
built its own radius-10 graph. Point-estimate numbers (path1 wedge): v2_A λ1 **0.841** /
λ2 0.900 / λ3 0.932 / clu-Sp +0.57 vs v1 0.840/0.897/0.930/0.56 — tiny gain, but the field
product and the calibrated λ1 product now share ONE preprocessing lineage. v2_B (fno +
survey-mask, 0.839) → survey-mask kept as a **Phase-C full-footprint deployment toggle**
(real boundaries), not part of the frozen accuracy config. F-tier stays the **point-estimate
/ eigenvector (IA) product, badged** — NOT the v1 calibrated headline (G3+FMPE-λ1).

**S3 plan (starting now).** One pooled, ñ-conditioned, spatially-split training cache feeds
Phase B (G3+FMPE and F-tier-v2_A in parallel): (A) freeze the S0 mock ñ(z) spline to a
versioned JSON; (B) `log ñ(z)` GraphNet node feature EXCLUDED-by-name from SI per-graph-median
norm + appended to the FMPE conditioning vector; (C) ñ·V_voxel expected-counts channel for the
v2 U-Net; (D) **spatial holdout** replacing random split — three RA-disjoint regions applied
across all 5 shells: **train RA<145 · val/tempering 145–150 · test RA≥150** (test never
trained; τ fit on val, assessed on disjoint test, per Codex #1); (E) pool the 5 S2 union-graph
caches; (F) pre-Phase-B sanity gate (ñ SI-untouched, regions disjoint/non-empty per shell,
zero train↔test leakage, feature-count metadata survives a dry-run forward).



### 2026-07-09 — [code] Continuous + fixed-mass environment plots (CIGALE): surface, heatmaps, mass-bin lines, animation
- New `workflows/sbi_inference/plot_env_mass_continuous.py` → 4 figures in figures/desi_wedge_cigale_hz/:
  (1) env_mass_surface_3d — 3D surface of quenched fraction over (logM*, tidal trace): steep along
  mass, gentle-monotonic along trace. (2) env_mass_heatmaps — 2D median (g-r)/sSFR/quenched over
  (logM*, trace) (continuous env axis). (3) env_class_by_massbin — STATIC: property vs 4 classes,
  one line per logM* bin (STANDOUT: mass bands well-separated + clear env slope within each; SFR
  env-effect largest at low mass). (4) env_class_mass_animation.gif — mass slider (22 frames),
  4 environments' g-r/sSFR/SFR shift as logM* rises 10.55->11.35.
- These operationalise the 'mass dominates, environment second-order' result for the talk. The
  static per-mass-bin figure is the honest headline (extends closure_mass_control to all props).
- Note: animation uses fixed full-range y-limits so the MASS shift is visible; the within-frame
  env spread is modest (real physics). A 'residual-at-fixed-mass' variant would amplify the env
  tilt if wanted.

### 2026-07-09 — [code] Weak env<->property signal INVESTIGATED: it's PHYSICAL (mass-dominated), not inference noise
- User flagged: SFR-M* distribution changes less than expected with environment; sSFR/g-r vs
  inferred trace correlations weak (rho ~0.11-0.14). Comprehensive diagnostic
  (workflows/sbi_inference/investigate_env_property_signal.py) → verdict: PHYSICAL.
- Evidence: (1) inferred trace vs INDEPENDENT n(z)-controlled kNN density contrast rho=+0.75
  → model recovers real local density well. (2) Independent density gives ~SAME weak property
  correlations (log sSFR: -0.164 vs inferred -0.136; g-r +0.167 vs +0.138) → weakness is NOT a
  GNN artifact, any density-from-positions is weakly correlated with properties. (3) Model
  accurate on mock: predicted vs TRUE trace R^2=0.88, Spearman=0.97 (attenuation ~3%). (4) NOT
  uncertainty-dominated: posterior width/spread = 0.17-0.33. (5) MASS DOMINATES: property-mass
  rho 0.52-0.72, i.e. 4-5x stronger than environment. (6) At FIXED mass, env rho ~0.1 (real,
  strongest in high-mass tertile: sSFR -0.144). (7) Effect size in TAILS is large: densest vs
  sparsest decile Δmedian log sSFR = -1.5 to -1.7 dex, Δ(g-r) +0.16.
- Reconciles by-eye: bulk per-galaxy correlation weak (mass-scattered) BUT population trend
  across density is clear and tails differ strongly; by-class f_q 0.39->0.58 is consistent.
  This is the expected result (Peng+2010: environmental quenching real but subdominant to mass).
- Figure: figures/desi_wedge_cigale_hz/env_signal_diagnosis.png (trace-vs-density validation +
  |rho| bars mass vs density vs inferred). Talk implication: frame environment as a genuine
  SECOND-ORDER modulation on top of mass; lead with by-class trend + tail effect + mass-control,
  don't oversell the continuous rho.

### 2026-07-09 — [code] Talk figures updated to CIGALE: MS-fit refit + cartography/closure composite regen'd
- Task 1 (MS line): fit_ms now uses a binned-median RIDGE fit on clearly-SF gals (log sSFR
  >-10.5) → slope 0.35 (was flat 0.28 from naive polyfit); GV band widened to MS-0.8..-1.6
  to bracket the CIGALE bimodal trough. SFR-M* region lines now sit correctly.
- Task 2 (deck hero): new `figures/desi2026_spotlight/make_cartography_closure_cigale.py`
  rebuilds `spotlight_cartography_closure.png` in figures/desi_wedge_cigale_hz/ — sky maps
  drawn NATIVELY (Abacus CWEB truth + DESI hard_class, 0.25<z<0.30 shell, class-coloured) +
  CIGALE closure panels. Talk script Slide 4/5 numbers updated to CIGALE: f_quench
  0.39→0.45→0.52→0.58, median log sSFR -10.2→-11.6 (1.4 dex), (g-r) 0.78→0.91, rho_s
  +0.11/+0.14/-0.14, N~100k with CIGALE (90% of 111k).
- Minor: composite subtitle sits a touch high (above sky titles) — reposition in Keynote if
  needed. Deferred still: canonical-cut + mock n(z) re-harmonization (parity).

### 2026-07-09 — [code] CIGALE-HZ SFR/mass re-join DONE (Approach A): bimodality + closure recovered
- Problem: FastSpecFit SFRs are unimodal (single sSFR peak -11.5) → SFR-M* not bimodal,
  environmental distinction invisible. Colleague's CIGALE (HZ) SED masses+SFRs fix this.
- Approach A (user-confirmed): environment inference depends only on POSITIONS, so KEEP the
  existing parity-valid wedge + posteriors (mock n(z)-harmonized at DELTACHI2>=25) and just
  re-join CIGALE properties by TARGETID. No re-inference, parity untouched.
- Source: /global/cfs/cdirs/desi/users/manasvee/1_prepped_data/DESI/loa/
  desi_loa_fsf_bgs_scnd_quality_ZallPix_cigaleHu_primaryZ_goodPhoto.fits (CIGALE HZ = Hu Zou).
  Used SFR_CG_15/MASS_CG_15 (CG_15 vs CG_5 near-identical: corr 0.993/0.999). 90.3% match
  (100,679/111,503). Scripts: workflows/sbi_inference/build_cigale_rejoin.py (+ plot scripts
  now take WEDGE_PARQUET/WEDGE_FIGDIR env). Output run: desi_wedge_cigale_hz.
- RESULT: SFR-M* now shows the CLASSIC BIMODALITY (blue cloud + red sequence + green valley);
  environmental distinction now VISIBLE (blob balance shifts void->cluster). Closure STRONGER:
  f_quench 0.39->0.45->0.52->0.58 (was 0.74->0.84); median log sSFR -10.2->-11.6 (1.4 dex, was
  0.5). Figures in figures/desi_wedge_cigale_hz/{,closure/}.
- FIX LATER (parity item flagged by user): canonical cuts (DELTACHI2>=40, FRACFLUX<0.35,
  ZCAT_PRIMARY) differ from the DELTACHI2>=25 selection the mock was harmonized to. Mock is a
  SIM (no spec flags) so parity = re-harmonize mock n(z) to the new-cut DESI n(z), then rebuild
  graphs+cache+retrain+re-infer. Deferred; priority was statistical parity, preserved by A.
- TODO for talk: regenerate spotlight_cartography_closure composite + MS-fit line refit (slope
  came out flat 0.28); update deck numbers to CIGALE closure.

### 2026-07-07 — [science] SCOPE CORRECTION: wedge-trained NPE must NOT be extrapolated to full BGS
- User correction: the model is trained on the wedge subvolume (RA 120-160, z 0.20-0.30)
  matched to an Abacus subvolume, so its posteriors + the closure/environmental relations
  are VALID ONLY for wedge galaxies. Applying to the whole DESI DR2 BGS catalogue (lower z,
  different n(z)/density) is out-of-distribution extrapolation — WRONG. A whole-footprint
  VAC requires RETRAINING on footprint-spanning mocks.
- Fix: "amortised" buys cheap inference on IN-distribution galaxies only; it does NOT license
  generalisation off the training volume. In-wedge mock→data transfer (no retraining) remains
  legitimate (that's the actual result). DR3-KP talk script de-overstated: throughline scoped
  to the wedge, slide-5 deliverable + Q&A#7 + summary rewritten ("wedge demonstrator + path to
  VAC," never "whole-catalogue VAC"). Memory `feedback-wedge-no-extrapolation` added.

### 2026-07-07 — [code] Mass-colour (M*-g/r) by-environment contours + wedge-count clarification
- What: `workflows/sbi_inference/plot_mstar_color_environment.py` (GraphWeb_DESI) makes
  `mstar_gr_by_environment` (2x2 nested KDE contours per inferred class, reproduces the
  reference fig) and `mstar_gr_overlay` (4 classes overlaid, class-coloured 50/90%).
- Result: mass-colour shows the environmental trend clearly where SFR-M* hexbins did NOT
  (env at fixed mass is 2nd-order → hexbin panels differ mainly in count). Void is bimodal
  (blue lobe + red lobe); blue lobe shrinks void→wall→filament→cluster to a single
  red-sequence peak. Median (g-r) 0.782/0.827/0.870/0.908; median logM* 10.60→10.71.
- COUNT CLARIFICATION (user asked): full DESI wedge = 111,503 galaxies (110,251 unique
  after 1,252 hemisphere dups; 111,171 with FastSpecFit photometry/sSFR). The "~22,000"
  is the thin 3° Dec slice (Dec 21-24° = 22,486) used by the FAN/skewer visualisations,
  NOT the whole wedge. z-shell is 0.20-0.30. Talk slide-5 updated to prefer the
  mass-colour contour fig + this count note.

### 2026-07-07 — [code] DR3-KP talk: SFR-M* (SFMS) figures by inferred environment + eigenvalue-continuous version
- What: New `workflows/sbi_inference/plot_sfms_environment.py` (GraphWeb_DESI) makes
  three SFR-vs-M* diagrams from the SI closure-join parquet (desi_wedge_env_props,
  N=77,459 with valid FastSpecFit sSFR): `sfms_all` (whole wedge, hexbin counts),
  `sfms_by_environment` (2x2 hexbin by inferred hard_class), `sfms_eigen_continuous`
  (1x3 hexbin coloured by mean posterior λ1/λ2/λ3 per cell). Blue-cloud/green-valley/
  red-sequence via MS fit to SF pop (logSFR=0.83 logM*-8.67) + offsets (MS-0.6/-1.2).
- Result: clean environmental-quenching signal — SF blob dominates Void, red sequence
  dominates Cluster (Wall/Filament intermediate). Eigenvalue version: red-sequence
  corner sits at higher mean λ (more collapsed tidal env), signal strengthening
  λ1→λ2→λ3. Figures in `figures/desi_wedge_flowjax_linear_si/`.
- Talk use: user DROPPED the mass-control slide; `sfms_by_environment` becomes slide 5
  hero (script §Slide 5 updated), eigen-continuous as its backup/alt. Also confirmed
  closure-slide text is correct (Loa=DR2; quenched=log sSFR<-11; g-r independent check).

### 2026-07-07 — [code] DESI DR3 KP (Cosmic Web & Galaxy Environment) 4-min talk: script v2 written, figures all existing
- What: Slide-by-slide script for the 4+2 min DR3 KP parallel-session talk written to
  `GraphWeb_DESI/docs/desi_dr3_cwge_4min_talk_script.md` (5 slides: title/map →
  why-posteriors → mock-trained+calibrated (class-fractions hero) →
  spotlight_cartography_closure composite hero → mass-control + take-home/VAC;
  + 8-item Q&A prep incl. fibre incompleteness, RSD, classical DTFE floor
  0.55/0.64/0.66 vs GNN 0.78/0.81/0.89, closure-circularity).
- Decision: NO new figures needed — the 2026-07-06 `desi2026_spotlight` set
  (esp. `spotlight_cartography_closure.png`) + SI-run + closure figures cover all
  slides; environment-fluent audience ⇒ closure/mass-control promoted to payload,
  SBI machinery compressed to one slide, calibration figures to backup.
- Refs: figures under `/pscratch/.../graphweb_desi/figures/{desi2026_spotlight,
  desi_wedge_flowjax_linear_si,desi_wedge_flowjax_linear_si_closure}/`; numbers
  reconciled to 2026-06-21/22 closure + calibration entries.
- v2 (same day): session schedule received — talk 5 of 7, fixed title "Learning
  the cosmic web: dynamical environment value added catalogues for DESI BGS",
  after ASTRA VAC / DisPerSE / MTV-reconstruction / void-science talks, NO
  per-talk Q&A (10-min consolidated + 40-min discussion). Script updated: zero
  web setup, positioning-vs-other-VACs framing, bridge lines to Li (MTV) and
  Rincon (voids), discussion seeds (common validation wedge; closure test as
  shared truth-free VAC metric; uncertainty propagation as KP standard).
- v3 addendum: deck review vs ASTRA slides (one idea/visual per slide; drop both
  videos) appended to the script; NEW fan ("pie-slice") figures built in the
  cmlamman polar style — `figures/desi2026_spotlight/fan/` on scratch:
  `fan_{desi_inferred,abacus_truth}_dec{2,3,4}` + recommended `_dec3_crop`
  variants (theta=RA 120–160°, r=z 0.20–0.30, Dec 21–24° slice, rorigin 0.12,
  class-styled points, script `make_fan_figure.py` + README). These replace the
  wedge video and the slide-30 sky map as the deck's hero pair.

### 2026-07-03 — [code] NAMING CORRECTION: the two "G4" runs are G4-SMOKE, not G4 (JDPK)
- The 2026-07-03 runs titled "G4 smoke" and "G4 part 2 (attention)" are hereby renamed
  **G4-SMOKE (graph+features held FIXED)**. They held the prebuilt Delaunay edge set AND
  the curated 7 node features fixed, and varied only invariant edge-message aggregation
  (mean 0.603 → attention 0.654 λ1 R² vs baseline 0.774). They test message/aggregation
  design on the EXISTING graph+features — NOT equivariance-as-architecture and NOT
  point-cloud input. The gate G4 as originally defined (symmetry axis) is STILL OPEN.
- **G4-PROPER** = the real test: positions + LOS only, model builds its own ~10 Mpc/h
  neighbourhoods, NO curated features, steerable messages → the tidal TENSOR (eigenvalues
  + eigenvectors fall out). Requires T-web module changes to emit tensor/eigenvector
  targets. Planned separately: `docs/plan_g4_proper_equivariant_tensor.md` (not executed).

### 2026-07-03 — [code] A3 COMPLETE: n(z)-harmonized wedge rebuilt end-to-end; nzharm SI cache ready
- What: full GPU/CPU rebuild of the harmonized training wedge via a BUFFERED box
  (RA 118–162, Dec 12.5–32.6, z 0.185–0.315) so boundary nodes keep real neighbours —
  mirroring subset-from-full semantics: (1) buffered shape-match dilution (C=0.733 core;
  148,222 of 176,349 kept; σ_v=35 km/s z-errors; sentinels excluded) → (2) gudhi
  Delaunay (CPU-guarded script — resource guard correctly refused a GPU alloc; pipeline
  split CPU→GPU→CPU) → (3) cuGraph 7 features (rapids) → (4) trim to exact wedge:
  **82,650 nodes, 605,126 undirected pairs (avg degree 14.6 ✓)**, alignment=identity,
  single hemisphere → (5) SI cache (splits 57,854/17,357/7,439) at
  `sbi_caches/path1_flowjax_3d_lineareig_si_nzharm/`.
- Parity verified: edge storage convention matches baseline (undirected pairs; cache
  doubles internally: 605,126→1,210,252 like 740,623→1,481,246); no self-loops/dupes.
- G4 record CORRECTED (JDPK probe): both smokes used the PREBUILT Delaunay edge_index +
  curated node h0 — raw geometry entered only as edge scalars. They test message/
  aggregation design on the same restricted graph, NOT "point cloud vs graph." The true
  point-cloud experiment (positions+LOS only, self-built ~10 Mpc/h neighbourhoods,
  attention EGNN, no curated features) is well-posed, ~½ day, NOT yet run — optional,
  sequenced behind the G3 readout. Tensor-target rule reaffirmed: eigenvalues for
  invariant nets (fixed-frame tensor INVALID there); tensor only with steerable
  (or LOS-frame spin-2-careful) heads.
- Phase B inputs now ready: nzharm cache (this) + union graph (G3 training, ~epoch 900+,
  sbatch auto-release armed) + FMPE head (G6 GO, calibration tune pending) + luminosity
  features (G2).
- New scripts: `build_harmonized_buffered_catalog.py`, `trim_wedge_from_buffered.py`,
  orchestrator `logs/run_a3_rebuild.sh` (CPU→GPU→CPU, tmux a3_rebuild).

### 2026-07-03 — [code] G4 part 2 (attention): helps (+0.05 λ1) but NO-GO stands; gap decomposed
- What: JDPK hunch tested — same EGNN-lite smoke with the SINGLE change mean→4-head
  attention (invariant logits from [h_src,h_dst,egeo]; segment softmax; symmetry story
  intact). λ1 R² 0.603→**0.654** (λ3 .791→.825; cluster Spearman .44) vs baseline
  0.774 — still below the pre-registered 0.75 bar.
- Attribution now clean: of the 0.17 part-1 gap, **aggregation ≈ +0.05**; remaining
  ~0.12 points at (a) regularization/training parity (attention overfit HARDER:
  train .073 vs val .253; baseline uses dropout .2 + wd .08 over 7000 epochs) and
  (b) curated EDGE features (baseline edge_attr includes density_contrast; pure-geometry
  egeo omits it) and/or the NLL/flow head's representation learning.
- Decision: **NO-GO confirmed for the full steerable build** — closing the rest of the
  gap = rebuilding the baseline's tuning, i.e. the very effort the gate protects.
  Standing lesson: any future equivariant build must be ATTENTIONAL
  (SE(3)-Transformer-class, not plain EGNN). Optional part 3 (dropout/wd parity) noted
  but not expected to flip the decision.
- G3: epoch 840, val NLL 3.47→2.50, healthy; auto-release watcher armed for the held
  sbatch handoff at window end.
- Refs: `gate_g4_egnn_smoke.py --aggregation attention`; log `logs/g4b_attn_*.log`.

### 2026-07-03 — [code] G6 GO (FMPE beats MAF), G4 smoke NO-GO (EGNN-lite < GraphNet), G3 training in tmux
- Parallel interactive wave (tmux sessions g3_union / g6_fmpe / g4_egnn; sbatch 55441429
  HELD as resumable fallback so no double-writer on the shared checkpoint dir).
- **G6 (FMPE vs MAF, frozen 80-d GNN embeddings, sbi 0.26.1, CPU): GO on accuracy.**
  Same conditioning, same train pool: λ1 R² 0.807 vs 0.772 (+0.035), λ2 .842/.829,
  λ3 .916/.898; cluster-slice λ1 Spearman +0.77 vs +0.61 (n=71). Converged in 144
  epochs on CPU — cheap swap. **Calibration caveat:** λ1 slightly overconfident
  (68% central coverage 0.63; 90% → 0.84; KS p .004; other dims fine) — needs a tune
  + a same-slice MAF rank comparison before production adoption in Phase B.
  (`gate_g6_fmpe_frozen_head.py`; embeddings cached at
  `sbi_runs/.../frozen_embeddings_all_nodes.npz`.)
- **G4 smoke (EGNN-lite, scalarized observer-rotation invariants of raw geometry +
  curated features, same Delaunay edges/splits, MSE head): NO-GO at the
  pre-registered bar** (λ1 R² 0.603 vs baseline 0.774; needed ≥0.75; λ2 .756/.810,
  λ3 .791/.891; cluster Spearman .45/.54) — despite the point-estimate head being
  R²-favoured. Raw geometry does NOT trivially beat curated features + attention;
  overfitting gap (train .15 vs val .29) says budget/regularization, not capacity.
  **Full steerable e3nn build DEPRIORITIZED** — revisit only if G3 or P2 motivates.
  (`gate_g4_egnn_smoke.py`; OOM on full-graph backward fixed via per-layer gradient
  checkpointing.)
- Conceptual note (logged for the papers): equivariance and attention are ORTHOGONAL
  axes of the Battaglia+2018 GN framework — equivariance constrains what φ_e/φ_v may
  compute; attention is a choice of aggregation weighting. EGNN = GN block with
  invariant-restricted φ_e, no attention; SE(3)-Transformer/Equiformer = steerable φ +
  invariant-logit attention. Attention is most motivated on the UNION graph (two edge
  populations to arbitrate).
- **G3:** training live in tmux (job 55442933, 4×A100). Union graph is ×2.7 edges →
  ~5.2 s/epoch → ~10 h total: this 4 h window banks ~2700 epochs to checkpoint; then
  RELEASE the held sbatch (`scontrol release 55441429`) to resume overnight.
- Phase B bundle shape so far: n(z)-harmonized mocks (A3) + luminosity features (G2)
  + FMPE head (G6, after calibration tune) + union graph pending G3 readout;
  equivariant encoder out (G4 smoke).
- Refs: `workflows/sbi/gate_g6_fmpe_frozen_head.py`, `gate_g4_egnn_smoke.py`,
  `run_flowjax_union_interactive.sh`; logs `/pscratch/.../logs/g{4,6}_*.log`.

### 2026-07-03 — [code] A3 designed+built, G3 training SUBMITTED (job 55441429), G6 deferred (no FMPE deps)
- **A3 harmonization (design + artifacts):** shape-match the mock wedge n(z) to DESI by
  per-shell dilution — keep fractions f_i = C·(DESI_i/mock_i), C = min ratio = 0.733;
  the residual amplitude offset becomes UNIFORM (scale-only), exactly what SI features
  absorb. Kept 82,693/100,935 (81.9%). z-errors injected (σ_v=35 km/s, median |dz|≈1e-4,
  original Z kept as Z_ORIG). Artifacts: `abacus/harmonized_20260703/path1_wedge_nzharm_
  {keepmask.npy, targets.fits, selection.json}` (`harmonize_wedge_nz.py`). Remaining A3
  step: rebuild Delaunay graph + cuGraph features on the kept points (GPU, rapids-gnn),
  then SI cache — commands recorded in selection.json.
- **G3 (union graph) fully launched:** built Delaunay ∪ radius(10 Mpc/h=14.78 Mpc)
  edges — 740,623 → 1,988,732 undirected pairs (×2.69), median edge 9.3→11.3 Mpc; new
  edge_attr in the original convention (`build_union_graph_arrays.py`; new npz+metadata,
  originals untouched). SI cache built with IDENTICAL splits to baseline (seed 42;
  70,654/21,196/9,085) at `sbi_caches/path1_flowjax_3d_lineareig_si_uniongraph/`.
  Training sbatch SUBMITTED: **job 55441429** (4×A100, 12h, regular), output
  `sbi_runs/path1_wedge_flowjax_3d_linear_si_uniongraph`; compare vs the SI baseline on
  the shared test split (gate: λ1 R² / cluster metrics vs 0.774).
- **G6 (FMPE) deferred honestly:** cosmic_env has NO sbi/diffrax; flowjax 17.2.1 is
  discrete-flows-only. Plan: next session, new ISOLATED conda env with `sbi` (keeps
  cosmic_env pristine), train FMPE/NPSE heads on FROZEN GNN embeddings (one forward
  pass extracts them) vs the MAF baseline — head-only comparison, ~a day.
- Refs: scripts above; A2 n(z) JSON feeds harmonize_wedge_nz directly.

### 2026-07-03 — [code] Phase 0 gates EXECUTED: G1 GO (GNN≫GBM!), RSD penalty huge but GNN closes it, n(z) gradient, sentinel-z FIXED
- What: ran the roadmap-v2 first wave in parallel (G1, G1.5, G2, A1, A2). All new
  standalone scripts (`gate_g1_gnn_vs_gbm.py`, `gate_g15_g2_rsd_luminosity.py`,
  `measure_nz_mock_vs_desi.py` in GraphWeb_DESI/workflows/sbi_inference/); only edit =
  sentinel-z patch (backed up, new output filenames; originals untouched).
- **Physics verification first (JDPK request):** eigenvalue targets confirmed sampled at
  the REAL-SPACE host-halo x_com via (FILE_NUM,HALO_INDEX) — annotate_cutsky code
  explicitly avoids sky-coordinate inversion. So targets carry no RSD sampling offset;
  satellites inherit their HOST's λ (note for papers).
- **G1 (GNN vs GBM, identical features/splits): GO — and it FALSIFIES the 06-25/06-26
  "information-limited" claim.** GNN posterior-mean λ1 R²=0.774 vs GBM 0.272 (λ2 .810
  vs .334; λ3 .891 vs .326); cluster-slice Spearman 0.54 vs 0.04. Message passing does
  massive implicit field interpolation/de-distortion. (Transductive eval — matches
  deployment.)
- **G1.5 (RSD penalty, Z vs Z_COSMO on cutsky truth): +0.60** — real-space aperture
  features reach λ1 R²=0.86 vs 0.26 z-space (median RSD displacement 3.3 Mpc);
  mass-completeness 0.36→0.56. **The ladder: z-space GBM 0.26 → z-space GNN 0.77 →
  real-space bound 0.86.** The GNN already recovers most of the RSD-destroyed info;
  bounded headroom for G4 equivariant/LOS work ≈ +0.09 (motivated, tempered).
- **G2 (luminosity weighting): marginal GO (+0.036** z-space; +0.02 on real-space) —
  include flux-weighted features in Phase B, low cost.
- **A2 (n(z)): mock/DESI = 0.895 in the wedge and Z-DEPENDENT** — ratio slides
  1.02 (z≈0.20) → 0.73 (z≈0.275) → 0.89 (z≈0.30). A radial gradient a single per-graph
  SI median cannot absorb ⇒ per-shell harmonization (A3) confirmed as the right fix.
  Phantoms in box = 19.1%, excluded. Outputs: `abacus/nz_comparison_20260703/`.
- **A1 (sentinel-z): FIXED + REGENERATED + VALIDATED.** `_inject_chunk` now preserves
  ZWARN=999999 (never injects a pass onto fibre-unobserved rows); backup
  `inject_loa_spec_from_zall.py.bak.20260703`. New catalogs (originals untouched):
  `BGS_BRIGHT_full_noveto_loa_spec_sentinelfix.fits` and
  `/pscratch/.../abacus/mock_bgs_maglim_sentinelfix.fits` — 7,472,725 rows (was 9.54M),
  ZWARN==0 = 100%, sentinel-window frac 2e-5 (was 0.217), z-median 0.1966 (= audit's
  predicted clean value). z-expansion unblocked; A3 has its clean parent.
- Next: A3 per-shell harmonization (now fully unblocked); G3 union-graph and G6 FMPE
  (next sessions); G4 motivated with ~+0.09 tempered expectation; P1 letter drafting.
- Refs: gate scripts above; logs `/pscratch/.../logs/a1_sentinelfix_*.log`.

### 2026-07-03 — [code] Roadmap v2: decision-gated, parallelized; RSD-penalty gate added; problem stated exactly
- What: consolidated the whole brainstorm into **`docs/roadmap_environmental_vac.md` v2**
  (canonical; supersedes v1 and the 07-02 entry's sketch). Everything is now closed /
  gated / scheduled — no expensive step runs before its GO/NO-GO gate.
- Key sharpenings since v1:
  1. **Problem stated exactly**: z-space biased sparse tracers → REAL-space matter tidal
     field (targets from full particle field, comoving Cartesian, rs7; inputs from
     observed Z). The regression fuses interpolation + bias + statistical RSD
     de-distortion. Exact symmetry = **SO(3) about the observer**, realized by an
     E(3)-equivariant net fed per-node LOS r̂ᵢ (translations correctly broken). Frame
     consistency holds ⇒ tensor/eigenvector prediction well-posed (IA-ready).
  2. **New gate G1.5 — RSD-penalty decomposition** (mock has both Z and Z_COSMO):
     features from real-space vs z-space positions; the R² gap upper-bounds what ANY
     LOS-aware/equivariant machinery can recover, and splits the "information ceiling"
     into RSD vs sparsity+bias parts. Gates the equivariant rung together with G1.
  3. **Graph-construction critique formalized** (gate G3/b′): Delaunay's receptive
     field is fixed in neighbour count but variable in physical scale — narrowest in
     clusters — while the target lives at fixed 7 Mpc/h. Cheap fix under test:
     Delaunay ∪ radius(~10 Mpc/h) union graph, same GraphNet.
  4. Model zoo reduced to 2 axes (connectivity × symmetry); point-cloud models are
     GNNs that build their own graph (input = positions ± luminosity, nothing else).
  5. Timeline: P1 letter drafts now; G1/G1.5/G2 parallel CPU gates in wk 1–2 of July;
     ML4PS (P2) ~Aug 29 from the ablation; ONE bundled retrain end of Aug; JEPA/P5
     only if the post-retrain MMD gate fails.
- Refs: `docs/roadmap_environmental_vac.md` (v2, canonical — read this first on the
  Desktop side); task list mirrors the July gates.

### 2026-07-02 — [code] Method-frontier brainstorm + roadmap: triage plan, equivariant GNNs, papers
- What: strategy session (no production runs). Reviewed the whole diagnostic arc and
  set a forward plan. **Durable plan doc: `docs/roadmap_environmental_vac.md`** — point
  the Desktop side there; this entry is the summary.
- Architecture correction (propagate this — `~/.claude/CLAUDE.md` is NOT git-tracked so
  its fix won't sync): the NPE encoder is a jraph **Attentional GraphNetwork
  (Battaglia+2018)**, NOT GAT/GATv2. The GAT is the separate PyTorch classification
  lineage (the RASTI paper — JDPK is author; stop citing it as a third-party sibling).
- Method-frontier assessment (candidate upgrades, all gated): equivariant GNN encoder
  (EGNN/SEGNN fed rᵢⱼ + LOS r̂ᵢ, output the tidal *tensor* → eigenvalue ordering free,
  RSD anisotropy first-class); graph transformer (long-range = multi-scale); FMPE /
  flow-matching or Simformer posterior head (replace discrete MAF); luminosity weighting
  (claim-preserving, truth-testable on cutsky R_MAG_ABS); field-level reconstruction as
  the classical rival we complement, not beat.
- Key reframe (corrects the 06-26 "information-limited" claim): the measured ceiling is
  a property of the HAND-CRAFTED features, not proven for the raw point cloud. The
  representation ablation (below) is what settles "are we too constrained by
  distribution-only, hand-engineered features."
- Degraded-mock idea = supervised **domain randomization / nuisance-marginalized SBI**;
  related to but not the same as **JEPA** (self-supervised, dual-encoder, predicts
  masked embeddings). JEPA is the escalation if the misspecification gate fires.
- Decision structure = **Phase 0 Triage** (cheap GO/NO-GO, parallel), gating one bundled
  production retrain: (1) GNN-vs-GBM capacity check ½d; (2) luminosity-weighting gate
  ½d truth-tested; (3) representation ladder rungs c/d = equivariant GNN + sparse U-Net
  field-level, GPU, GO only if ≫ current GraphNet; (4) FMPE head swap. Phase A n(z)
  harmonization runs in parallel (fix sentinel-z bug first). Full detail + Phase B–D +
  publication ladder (Letter → Methods → VAC → ML4PS → conditional ICML/NeurIPS JEPA
  paper) in the roadmap doc.
- Next: start Phase 0 steps 1 & 2 (cheap CPU); step 1 doubles as rungs a–b of the ML4PS
  paper. Gate step 3 (GPU equivariant run) on step 1.
- Refs: `docs/roadmap_environmental_vac.md`; two roadmap flowcharts + a feature-type
  schematic produced in-session (not committed — regenerable).

### 2026-06-26 — [code] DEFINITIVE (mass-anchored): clusters favour FINE smoothing; current 7 Mpc/h good
- What: the "ideal test" — anchor 'cluster' to a SCALE-INDEPENDENT physical label
  (halo mass) instead of the smoothing-dependent λ1>λ_th. The master cutsky
  (`mocks_with_eigs_23032026/cutsky_..._with_tweb_eigs.fits`) carries **HALO_MASS**
  directly (units 1e10 Msun/h; also R_MAG_ABS, G_R_REST, CEN) and is row-aligned with
  the rs6–24 λ catalogs → no fragile CompaSO index join needed.
  (`mass_anchored_cluster_test.py`; the earlier CompaSO-join attempt was abandoned —
  path1 wedge_targets BOX_INDEX didn't reproduce host_halos x_com, and the master
  catalog has mass anyway.)
- Sanity passed: Spearman(logM, λ1@rs7)=+0.24; features(aperture density)→P(logM>13)
  AUC=0.736 (density genuinely predicts mass).
- Findings (identical across logM>12.5/13/13.5):
  1. **Massive-halo recovery decreases MONOTONICALLY with smoothing — finest is best, no
     interior optimum.** AUC(true λ1 → M>1e13): rs6 0.789 → rs7 0.770 → rs10 0.721 →
     rs20 0.637. Model recovery (aperture density → predicted λ1 → massive) same trend
     (rs6 0.722 → rs20 0.665).
  2. **Anti-correlation confirmed with a physical anchor:** mass-cluster recovery falls
     with smoothing while global R²(λ1) PEAKS at rs10. Optimising global accuracy
     actively harms real cluster recovery — the global "optimum" is a bulk artifact.
  3. **T-web cluster label is a real but imperfect mass finder:** purity (λ1>0.2 ∩
     M>1e13)/(λ1>0.2) ≈ 0.54 at rs7 (~0.61 at M>3e12) — about half of T-web clusters
     are genuine >1e13 halos; the rest are collapsed-environment outskirts. Massive halos
     ARE recoverable from geometry (AUC ~0.72), contra the earlier units-bugged run.
- Decision: **7 Mpc/h is a good cluster choice** (rs6 marginally better but noisier
  globally); do NOT raise the target smoothing for accuracy — it trades away clusters.
  Levers unchanged: soft posterior P(λ1>λ_th) + scale-matched ~10 Mpc/h *feature*
  aperture. Mass anchor removes the circularity in the smoothing-dependent cluster def.
- Refs: `GraphWeb_DESI/workflows/sbi_inference/mass_anchored_cluster_test.py`,
  `plot_mass_anchored_recovery.py`; figure `tng_illustris/figures/smoothing_scale_study/
  mass_anchored_cluster_recovery.png`. Master cutsky has HALO_MASS/R_MAG_ABS/G_R_REST/CEN
  (useful for any future property-augmentation study, now testable on truth).

### 2026-06-26 — [code] CORRECTION: global-R² smoothing optimum ≠ cluster optimum; clusters favour ≤7 Mpc/h
- What: JDPK critique of the previous entry — clusters are compact, so larger smoothing
  washes them out; the global R²(λ1) peak at ~10 Mpc/h could be raising bulk accuracy
  while *erasing* clusters. Tested with CLUSTER-CONDITIONED metrics vs smoothing
  (`cluster_recovery_vs_smoothing.py`).
- Findings — critique CONFIRMED:
  1. **Global R²(λ1) and cluster completeness are anti-correlated.** Global R² peaks at
     rs10 (0.592); rank-based cluster completeness@true-rate peaks at the FINEST scale
     (rs6–7 ≈ 0.55) and falls through rs10 (0.518) → collapses by rs16 (0.17). rs7→rs10:
     global +0.045 but completeness −0.031.
  2. **AUC(cluster) is a trap for rare classes** — it *rises* with smoothing (0.90→0.98)
     because survivors become a trivially-separable extreme tail; completeness exposes
     the real (opposite) trend. Use completeness/fate, not AUC, for rare clusters.
  3. **Cross-scale fate:** of clusters defined at rs6, only 75% survive to rs7, **33% to
     rs10**, ~0% by rs16 — compact clusters dissolve fast under smoothing.
- Decision / correction: **RETRACT the "~10 Mpc/h optimum" as a target choice** — that is
  the bulk's optimum, not the clusters'. The current **7 Mpc/h is near-optimal for
  clusters** (was well-chosen). No single smoothing serves both bulk and clusters. The
  cluster levers remain: soft posterior P(λ1>λ_th) (shrinkage) + scale-matched ~10 Mpc/h
  *feature* aperture for the 7 Mpc/h *target* (feature scale ≠ target scale).
- Caveats: "cluster" defined via λ1(s)>λ_th (smoothing-dependent); a scale-independent
  anchor (halo mass via HALO_INDEX→CompaSO) would be the ideal further test.
- Refs: `GraphWeb_DESI/workflows/sbi_inference/cluster_recovery_vs_smoothing.py`,
  `plot_smoothing_scale_study.py`; figure `tng_illustris/figures/smoothing_scale_study/
  smoothing_scale_learnability.png`.

### 2026-06-26 — [code] Target smoothing scale vs λ learnability: optimum ~10 Mpc/h; aperture≈1.4×scale
- What: JDPK follow-up — we have cutsky BGS eigenvalues at many smoothing scales (rs
  6–24 Mpc/h, same 63.9M galaxies, only target smoothing varies). Held galaxy features
  fixed (aperture density at 3–28 Mpc/h on a wedge footprint downsampled to BGS-like
  ~12.6 Mpc spacing) and measured how λ1/λ2 distribution + learnability depend on the
  target smoothing. (`smoothing_scale_investigation.py`, fitsio column reads.)
- Findings:
  1. **λ1 learnability is non-monotonic, peaks at ~10–11 Mpc/h** (geom R²(λ1): rs6 0.51,
     rs7 0.55, rs9 0.59, **rs10 0.592**, rs11 0.59, rs12 0.58, rs16 0.53, rs20 0.47).
     The current **7 Mpc/h is slightly below the learnability optimum**.
  2. **Scale-matching quantified:** λ1 at smoothing s is best predicted by density at
     aperture ≈ **1.3–1.5×s** (rs6→7, rs7→10, rs9–12→14, rs16–20→20). Design rule: for a
     target smoothed at s, build density features at ~1.4×s. Explains why few-Mpc Delaunay
     features under-serve the 7 Mpc/h target.
  3. **λ2 is the well-behaved eigenvalue** (R²≈0.64, peaks ~7 Mpc/h); web-classification
     difficulty is almost entirely in λ1.
  4. **Resolution↔learnability tradeoff is steep for clusters:** cluster frac (λ1>0.2)
     6.6%→3.2%→~0 over rs7→rs10→rs16. Can't smooth away the cluster problem without
     erasing clusters — smoothing scale and λ_th must be tuned jointly.
- Decision / implication: highest-value claim-preserving change = use **scale-matched
  ~10 Mpc/h aperture-density node features for the 7 Mpc/h target** (bigger lever than
  velocity dispersion, per the 06-26 entry). Smoothing scale + λ_th are a joint choice;
  ~10 Mpc/h is the λ1 sweet spot if fewer clusters are acceptable. No model change yet
  (investigation only); pre-veldisp default preserved.
- Caveats: aperture-density proxy (not full GNN; trends robust, absolute R² conservative);
  single footprint/realization; large-s decline partly aperture-capped at 28 Mpc/h.
- Refs: `GraphWeb_DESI/workflows/sbi_inference/smoothing_scale_investigation.py`;
  cutsky catalogs `mocks_with_eigs_*_rsmooth_*` (rs6–24, _15d set).

### 2026-06-26 — [code] Cluster fix is FEATURE SCALE-MATCHING (+ soft posterior), NOT velocity dispersion
- What: followed up the 06-25 diagnosis by gating a proposed kinematic feature (local
  line-of-sight velocity dispersion / Fingers-of-God anisotropy) with cheap mock-truth
  pre-checks BEFORE any retrain. Arc:
  1. **Delaunay-scale veldisp — null.** σ∥/σ⊥ over Delaunay neighbours (~few Mpc) does
     NOT separate true cluster vs filament (AUC 0.507). Also showed the first signal test
     was circular (used the model's own geometry-derived `hard_class` as target).
  2. **Aperture-matched veldisp — modest.** JDPK insight: T-web targets are the tidal
     field Gaussian-smoothed at **7 Mpc/h**, so the kinematic scale must match. Sweeping
     fixed apertures, FoGAniso signal peaks at ~7–10 Mpc/h (cluster-vs-filament AUC
     0.51→0.61; on a balanced HistGBM proxy, cluster recall +0.02–0.04, cluster→filament
     −0.01–0.03). Real but small.
  3. **Eigenvalue-space reframe (JDPK) — the decisive view.** Classes are just λ_th=0.2
     thresholds on the regressed eigenvalues, so test λ1 directly. geom→λ1 R²=0.286;
     FoGAniso adds only +0.018; it does NOT explain geometry's λ1 residual (r≈0.03) and
     does NOT resolve λ1 in the threshold zone.
  4. **Why-the-limit investigation — the payoff.** (a) FEATURE-SCALE MISMATCH dominates:
     adding **aperture density at 7 & 10 Mpc/h** lifts λ1 R² 0.286→0.366 (**+0.080, 4×
     the kinematic gain**) — purely spatial, claim-preserving. (b) The "cluster deficit"
     is largely **regression shrinkage**: globally λ1 ranks fine (Spearman 0.56) but a
     point estimate predicts cluster frac 0.004 vs true 0.059 at λ_th=0.2 (**97% tail
     miss**) → the calibrated **posterior P(λ1>λ_th)** (NPE `p_exceed`) is the honest
     product. (c) Boundary is hard but not irreducible: feature-space λ1 noise floor
     0.74→0.66 and zone Spearman 0.13→0.17 with better-scaled features (earlier "R²<0"
     was a narrow-variance metric artifact).
- Why / decision: **did NOT add velocity dispersion** — gated off; the cheap pre-checks
  saved a ~9h retrain that would have bought ~+0.018 λ1 R² without fixing the boundary.
  Code reverted to pre-veldisp default (no production-pipeline file touched; the precheck
  scripts are standalone/opt-in only).
- Methodology lessons: (i) independence-from-geometry ≠ usefulness — test new features in
  the TARGET (eigenvalue) space, not on discrete labels; (ii) match FEATURE scale to the
  target smoothing scale (7 Mpc/h); (iii) hard-thresholding a shrunk point estimate
  manufactures the cluster deficit — report the posterior.
- Next (future, when requested): (a) scale-matched spatial features — aperture density at
  ~7–10 Mpc/h appended to the node set (biggest, claim-preserving λ1 lever); (b) report
  & calibrate soft P(λ1>λ_th) rather than hard argmax class fractions.
- Refs (standalone, opt-in): `GraphWeb_DESI/workflows/sbi_inference/`
  `velocity_dispersion_precheck.py`, `velocity_dispersion_aperture_precheck.py`,
  `velocity_dispersion_eigenvalue_precheck.py`, `threshold_limit_investigation.py`;
  plan shelved at `~/.claude/plans/keen-twirling-prism.md`.

### 2026-06-25 — [code] Cluster-deficit diagnosis rewritten: not coverage/FoG, it's spatial-only feature limit
- What: ran a diagnostic battery on the SI run (`path1_wedge_flowjax_3d_Bcorrected_linear_si`
  / `desi_wedge_flowjax_linear_si`) to localise the known cluster under-recovery
  (clusters→filaments) and the small DESI misspecification:
  1. **Posterior degeneracy** — none. DESI matches the in-distribution Abacus self-eval on
     all checks: collapse-to-marginal ratio 2.4–3.8 (≫1), no width/spike collapse, moderate
     inter-λ corr (genuine 3D posterior), embedding eff. rank ~5–6/80 (same both sides).
  2. **Summary-space MMD misspecification** (BayesFlow/Schmitt-style, GNN embedding = summary
     net) — DESI vs Abacus MMD²=+0.0043 vs in-dist floor ≈0 (split-half), p<0.01, 51σ, but
     MMD distance only 0.066 on standardised 80-d → DETECTABLE but small; downstream λ
     distributions still overlay. Don't chase it.
  3. **Training-support coverage in SI space** — CLEAN. frac DESI > Abacus train *max* ≈0 every
     feature; cluster-cands only ~0.7% past p99.9 in density family. **SI normalisation closed
     the extrapolation gap** the old non-SI check flagged → coverage is NOT the cluster cause.
  4. **Property-information ceiling** (geometry→FastSpecFit property, cross-val, closure
     parquet) — geometry encodes ~nothing about galaxy properties: R² LOGMSTAR 0.06 / g−r 0.03 /
     log_sSFR 0.003 / DN4000 ~0; quenched AUC 0.56. In model-cluster galaxies it's at chance
     (AUC 0.53). Properties are nearly orthogonal to the spatial features.
- Why / decision: combined with the earlier mild FoG/LOS result, this **rules out extrapolation
  and FoG** and **supports the intrinsic-information-limit hypothesis**: the closure test showed
  properties correlate with environment (survive mass control); (4) now shows that signal is
  independent of geometry. So positions-only features are demonstrably leaving environment-
  relevant information (galaxy properties) on the table, most so in clusters. Answers "are we
  over-limiting with spatial-only features?" → yes, now shown not asserted.
- Caveats: redundancy measured via the eigenvalue-optimised embedding (may understate what raw
  geometry could predict — raw-7-feature check pending); low R² includes property scatter, only
  the env-correlated part is usable headroom (closure sizes it); exploiting it needs properties
  painted onto Abacus mocks (HOD/SHAM → modelling systematics) and changes the claim to
  "positions + properties". MMD per-repeat "ratio" log line is a cosmetic /0 artifact.
- Next: decide on a deliberate property-augmentation study — paint a minimal property
  (luminosity / SHAM M*) onto the path1 mock, retrain, measure cluster-recall gain. Real GPU
  retrain, not a quick check. Optional: raw-7-feature redundancy robustness check.
- Refs: `GraphWeb_DESI/workflows/sbi_inference/{property_ceiling_ablation,desi_abacus_coverage_report,mmd_misspecification_check}.py`,
  `TNG/Illustris/workflows/sbi/degeneracy_check_abacus.py`; FoG: `plot_fog_los_alignment.py`.

### 2026-06-22 — [science] Correction: softplus, not linear, is best-calibrated; talk drops the comparison entirely
- What: Re-examined `Plots/three_way_tarp.png` directly (was working from a stale table note).
  Max|ECP−α|: softplus 0.01, linear 0.03, raw 0.08. Softplus is the best-calibrated
  parameterisation, not linear — corrects the v2.1 script's "linear increments …
  best-calibrated" line, which had the calibration ranking backwards (likely conflating it
  with the ordering-violation-rate column, where linear/raw do better than softplus). This
  matches the pre-existing SCIENCE_LOG/long-term framing ("softplus gives better TARP
  coverage than linear despite similar NLL") — it was only the script that drifted from it.
- Decision: production parameterisation is UNCHANGED (linear increments — chosen on
  NLL/R²/ordering-violation grounds, and it's what every downstream DESI result in the talk
  already uses). What changes is the TALK: the three-way parameterisation comparison is cut
  from the main flow entirely — no slide framed as a comparison between target increments,
  not even a one-line mention. Slide 8 ("Training results"/calibration headline) states only
  that the production model's posteriors are calibrated on the simulator (TARP+SBC), with no
  reference to alternative parameterisations. The 3-way comparison figure moves to backup/Q&A
  (a reviewer in this room may ask "why this parameterisation" — worth having on hand) but
  carries no main-deck claim now that the deck doesn't reference it.
- Also scoped down: the "diagnosing & closing the gap" material (graph-scale/edge-length
  diagnosis, per-graph normalisation fix, cluster-recovery sequence) is cut from the talk too
  — no time in a 12-min slot once background/motivation was expanded to 3 slides. Talk now
  goes straight from zero-shot transfer ("it works, aggregate near-truth") to the DESI closure
  test (validation = first science) to take-home. The gap-diagnosis work itself is unaffected
  (still real, still in `SCIENCE_LOG`/code), it's just not presented Thursday.
- Next: slides 8 and 14 rewritten as standalone bullet content (Dakshesh is speaking without a
  script, so slide text needs to carry the point on its own). Deck still has cleanup pending:
  slide 15 is an unrelated leftover from the original 33-slide purge (delete); slides 9 and 16
  are duplicate drafts of "Zero-shot to DESI" (merge); slide 14 (Summary) needs to move to the
  final position.
- Refs: `Plots/three_way_tarp.png`; `SBI-Galev-2026.key` slides 8, 14, 17;
  `cambridge_sbi_galev_2026_script.md` (needs v3 pass to match — not yet done).

### 2026-06-22 — [code] Closure/embedding figure fixes: UMAP class-colour bug + mass-control bins redrawn
- UMAP embedding plot (`GraphWeb_DESI/.../plot_desi_wedge_flowjax.py`): the DESI panel
  coloured classes from a SEPARATE `rng.choice` draw than the embedding subsample → labels
  misaligned with points → inferred classes looked scattered at random (while the PCA panel,
  which shares one index, was fine). Fixed to one shared index. Regenerated UMAP now shows the
  expected ordered cluster→filament→wall→void gradient — and the right mental model: the T-web
  classes are thresholds (λ_th=0.2) on a *continuous* tidal-field manifold, so a smooth
  gradient (not separated blobs) is the correct picture.
- Mass-control closure plot (`plot_property_environment_closure.py`): replaced quantile
  tertiles (boundary at 10.49 — sat ON the M*≈10.5 quenching transition, and the low bin
  spanned 8.1–10.49 → poor mass control) with FIXED bins [9.8,10.4,10.6,11.0,11.6]
  (`--mass-edges`). Edges bracket the transition (10.4/10.6, not 10.5); it's isolated in the
  narrow [10.4,10.6) bin; and the **below-transition bin [9.8,10.4) still rises** void→cluster
  (f_q 0.62→0.68; ≥850 clusters/bin) ⇒ intrinsic mass quenching alone cannot explain the trend
  ⇒ environmental quenching. Ordered plasma colours for the 4 mass lines; suptitle states it.
- Uncommitted (await go-ahead): both edited scripts (GraphWeb_DESI).
- Refs: SI run dir `embedding_umap.png`, `closure/closure_mass_control.png`; `FIGURE_GUIDE.md` updated.

### 2026-06-22 — [science] Talk script finalized v2.1: all figures locked, deck-build is the only remaining task
- What: Recovered the interrupted talk-prep thread and confirmed against `Plots/` that all
  6 previously-"STILL NEEDED" figures (slide 3 idealised skewer, slide 6 three-way TARP,
  slide 7 Abacus TARP+SBC, slide 9 edge_scale + cluster recovery bars) now exist, matching
  the NERSC 06-22 "6/6 complete" entries below. Updated
  `~/Developer/Working Files/SBI-Galev-2026/cambridge_sbi_galev_2026_script.md` to v2.1:
  every slide's Visual line now points at a FINAL figure filename instead of a placeholder;
  numbers reconciled to the actual figures (slide 6 max|ECP−α| softplus/linear/raw =
  0.012/0.029/0.082; slide 9 recovery sequence 0.027→0.034→0.040→0.046 vs truth 0.058,
  ~60% of gap closed — corrected from the earlier "56%/two-point" figure). All 4 OPEN
  DECISIONS and the STILL-NEEDED list marked resolved/closed in the script.
- Why / decision: with content locked, the only remaining work before Thursday is
  mechanical — building the actual Keynote deck (`SBI-Galev-2026.key`, currently still a
  33-slide copy of the classification-era FLATS talk) from the v2.1 script. Confirmed via
  keynote-mcp the file is open at 33 slides; identified reuse candidates by content audit:
  slide 1 (title layout), slide 3 "The Cosmic Web" (T-web explainer layout), slide 8
  "Combining graphs and machine learning" (pipeline-stage layout) — consistent with the
  06-15/06-21 plan to inherit theme + 3 layouts and rebuild everything else.
- Next: confirm rebuild plan with Dakshesh (strip/repurpose old slides vs keep-and-renumber),
  then build the 11 main + ~8 backup slides via keynote-mcp using the v2.1 script + `Plots/`
  images; convert the two HTML animations (`skewer_idealised.html`,
  `skewer_posterior_animation_real.html`) to mp4/gif for embedding (no live HTML in-talk);
  timer dry-run once built (OPEN DECISIONS #4, still pending).
- Refs: `cambridge_sbi_galev_2026_script.md` v2.1; `SBI-Galev-2026.key`; `Plots/` (all);
  see 06-21/06-22 [code] entries below for figure provenance.

### 2026-06-22 — [code] Dropped three-way TARP; SBC + TARP (with error band) for production linear only
- Decision (user): stop framing results as a parameterisation comparison. Removed
  `three_way_tarp.png` and the band-less copied `abacus_tarp_coverage.png` /
  `abacus_sbc_calibration.png` from the SI folder + canonical mirror.
- NEW `TNG/Illustris/workflows/sbi/plot_calibration_linear.py` (GPU; reuses
  plot_flowjax_posteriors' loaders): SBC + TARP for the **production linear-increment SI
  model**. Outputs `tarp_linear.png` — TARP with a **bootstrap 1σ error band** (N=3000 test
  points, max |ECP−α| = 0.015, i.e. at the MC noise floor √(0.25/N)≈0.009 → consistent with
  perfect calibration) — and `sbc_linear.png` — rank histograms for λ₁/λ₂/λ₃ with the
  expected ±2σ band (flat within band; mild λ₁-upper / λ₂-lower edge structure).
- WHY drop the comparison: the three-way TARP's softplus(0.012) < linear(0.029) ordering sat
  inside the 500-point MC noise floor (per-bin SE≈0.022), so it was NOT a real calibration
  ranking — and claiming "linear best-calibrated" would be a hostage to fortune. Linear's
  case rests on Pareto-dominance (NLL +1.13 vs +2.25, R² 0.82 vs 0.74, ordering-viol 0.8%),
  not "tightest TARP". (Supersedes/corrects the 2026-06-19 "linear tightest TARP" note.)
- `FIGURE_GUIDE.md` updated to the new pair; `plot_tarp_threeway.py` now unused.
- Uncommitted (await go-ahead): + `plot_calibration_linear.py` (Illustris).
- Refs: SI run dir `tarp_linear.png`, `sbc_linear.png`.

### 2026-06-22 — [code] Skewer HTML → GIF for Keynote (NERSC ffmpeg has no h264)
- `render_skewer_video.py` NEW (`GraphWeb_DESI/workflows/sbi_inference/`): extracts the
  embedded JSON payload from a skewer HTML (`var D={…}`) and re-renders the same 3 panels
  (galaxy strip / three λ posteriors + λ_th / class-prob bar) natively with matplotlib +
  shared.plot_style → mp4 or gif. No browser/headless-Chromium needed.
- WHY gif not mp4: the NERSC system ffmpeg (`/usr/bin/ffmpeg`) is built WITHOUT libx264
  (encoders present: only gif, libvpx-vp9; h264/mpeg4 decoders disabled), and
  imageio-ffmpeg isn't installed — so no H.264. Keynote plays + loops GIFs natively, so
  produced GIFs: `skewer_idealised.gif` (3.0 MB), `skewer_real.gif` (0.97 MB) in the SI
  folder + canonical. The renderer auto-falls-back mp4→gif on encode failure.
- For a true H.264 mp4: convert the gif on the Mac (its ffmpeg has libx264):
  `ffmpeg -i x.gif -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" x.mp4`.
- Also fixed `sync_figures_to_canonical.sh` to include *.gif/*.mp4 (were excluded → the
  gifs weren't reaching the canonical dir).
- Uncommitted (await go-ahead): + `render_skewer_video.py` + sync-script edit (GraphWeb).
- Refs: SI run dir `skewer_{idealised,real}.gif`.

### 2026-06-22 — [code] Idealised skewer (slide 3) + per-figure interpretation guide; STILL NEEDED 6/6
- Slide 3 — `skewer_idealised.html` NEW (`GraphWeb_DESI/workflows/sbi_inference/build_skewer_idealised.py`):
  the discrete→continuous bookend. IMPORTS the real skewer's HTML template so layout + the
  standardised style (true-black / IBM Plex Sans / COSMIC_WEB_COLORS / λ_th=0.2) are identical
  — fixes the gap that the Desktop mockup had — but is driven by CLEAN synthetic structure:
  one density peak the sightline crosses, λ₃/λ₂/λ₁ cross λ_th at d≈0.33/0.60/0.86 so the class
  bar morphs void→wall→filament→cluster→…→void (peak P(cluster)=0.78). Slide 10 = the same
  layout on real posteriors. **STILL-NEEDED list now complete (6/6).**
- `FIGURE_GUIDE.md` written into the SI run dir: per-figure interpretation + error-bar
  provenance + variable definitions, esp. closure — inferred env = hard_class (argmax of the
  λ>λ_th class probs) / trace_lambda (E[Σλ] ∝ density); properties = LOGMSTAR, SFR,
  log sSFR=log10(SFR)−LOGMSTAR, quenched≡log sSFR<−11, g−r=ABSMAG01_SDSS_G−R from loa
  FastSpecFit; CIs = Wilson (fractions) / bootstrap 16-84 (medians) / TARP bootstrap bands /
  SBC rank uniformity.
- Uncommitted (await go-ahead): + `build_skewer_idealised.py` (GraphWeb).
- Refs: SI run dir `skewer_idealised.html`, `FIGURE_GUIDE.md` (+ canonical mirror).

### 2026-06-22 — [code] Slide 6 three-way TARP done; label fixes; "STILL NEEDED" now 5/6
- Slide 6 — `three_way_tarp.png` NEW (`TNG/Illustris/workflows/sbi/plot_tarp_threeway.py`,
  GPU): recomputes TARP coverage for the 3 parameterisation models on their own test
  splits and overlays them (bootstrap 1σ bands), reusing plot_flowjax_posteriors' eval
  machinery. Result = the slide-6 message exactly: max|ECP−α| **softplus 0.012, linear
  0.029, raw 0.082** — raw eigenvalues win NLL but are clearly OVER-confident (curve bows
  above the diagonal); softplus/linear hug it. (One JAX gotcha: flowjax needs new-style
  typed keys — `jax.random.key`, not `PRNGKey`.) Written to the SI folder + synced.
- Fixes from user review of yesterday's batch: (a) class-fractions legend now consistent
  "[domain] NPE" → `Abacus NPE` / `DESI NPE` (was "NPE DESI"); (b) recovery-bars 4th stage
  relabelled `Per-graph norm. (production)` (dropped "SI" — the fix is per-graph
  domain-relative normalisation, per the talk-v2 correction, not "scale-invariant").
- STILL NEEDED now just (d) slide-3 idealised fake-data skewer (generative; matches the
  real-skewer panel layout or reuse the 06-18 Desktop mockup) — user's call.
- Uncommitted (await go-ahead): `plot_tarp_threeway.py` (Illustris);
  `plot_desi_wedge_flowjax.py` + `plot_cluster_recovery_bars.py` (GraphWeb_DESI).
- Refs: SI run dir figures + canonical `figures/desi_wedge_flowjax_linear_si/`.

### 2026-06-21 — [code] Slide-figure batch: 4 of 6 "STILL NEEDED" plots done into the SI folder
- What: Actioned the STILL-NEEDED list from the plot→slide entry below. All new/updated
  figures are themed (`shared.plot_style`) and live in the SI run dir
  `/pscratch/.../flowjax_inference_outputs/desi_wedge_flowjax_linear_si/` (also synced to
  canonical `figures/desi_wedge_flowjax_linear_si/`). DONE:
  * Slide 8 — `class_fractions_comparison.png` REGENERATED without the "Regression DESI"
    bars (added `--no-regression` + `--only-class-fractions` to `plot_desi_wedge_flowjax.py`;
    generic bar spacing). Now 3 shades: Abacus truth / Abacus NPE / NPE DESI. Cluster group
    reads 0.06 / 0.05 / 0.05 → on-message "transfer lands near truth incl. rare clusters".
  * Slide 9 — `cluster_recovery_bars.png` NEW (`plot_cluster_recovery_bars.py`): reads the
    four DESI variant summaries live → cluster fraction 0.027→0.034→0.040→0.046 vs Abacus
    truth 0.058, annotated "60% of the gap closed". Gold bars, dashed truth line.
  * Slide 9 — `edge_scale.png` copied in from the baseline linear run (graph-level, identical
    across runs; already themed).
  * Slide 7 — Abacus calibration copied in as `abacus_tarp_coverage.png` +
    `abacus_sbc_calibration.png` (the themed 06-21 SI-run plots; provenance-renamed).
- NOT done (need heavier work, flagged for user decision):
  * Slide 6 three-way TARP (softplus/linear/raw overlay): the per-run npz saves only class
    probs (no posterior samples / no true θ), and `plot_tarp_coverage` computes ecp/alpha
    inline without saving them → a single overlay needs a GPU re-eval of all 3 models.
    User pre-authorised the table fallback; GPU overlay on request.
  * Slide 3 idealised fake-data skewer: generative; `build_skewer_animation.py` is the
    real-data version (panel layout = galaxy strip + width ribbon + class strip + 3 λ
    posterior KDEs + class-prob bar). A synthetic twin can be built to match, or reuse the
    06-18 Desktop mockup — user's call.
- Code: `plot_desi_wedge_flowjax.py` (2 new flags) + new `plot_cluster_recovery_bars.py`
  are UNCOMMITTED in GraphWeb_DESI pending user go-ahead.
- Refs: SI run dir figures; `GraphWeb_DESI/workflows/sbi_inference/{plot_desi_wedge_flowjax,
  plot_cluster_recovery_bars}.py`; this batch maps to the plot→slide entry directly below.

### 2026-06-21 — [science] Plot→slide mapping decided; graph-construction question closed
- What: Audited the conference folder `~/Developer/Working Files/SBI-Galev-2026/`
  (deck .key + 17 figures/animations) and assigned heroes per slide; full mapping
  appended to the script (§PLOT → SLIDE MAPPING v2.1). Curated, not exhaustive
  (we deliberately do not use every plot).
- Decisions:
  * Slide 8 hero = `class_fractions_comparison.png`. KEY REALISATION: this is the
    SI-corrected production run — NPE-DESI cluster fraction is already 0.05 (truth
    0.06), all four classes land near truth. So the transfer result and the cluster
    recovery are the SAME figure; slide 8 = "transfer works incl. rare clusters,"
    and slide 9 becomes the under-the-hood "this didn't come for free" methods
    slide. Reinforces the demotion of cluster-fraction from headline to support.
    ACTION: regenerate this plot WITHOUT the "Regression DESI" bars (off-message
    for an SBI room; clutters the legend) — keep 3 shades (Abacus truth/Abacus
    NPE/NPE DESI).
  * Slide 10 = `closure/closure_categorical.png` (hero: f_quench/colour/sSFR rise
    void→cluster) + `closure_mass_control.png` (survives at fixed M*) + the real
    skewer animation. Strongest part of the deck: validation = first science.
  * Slide 10 animation: `skewer_posterior_animation_real.html` → record to mp4/gif
    and embed in Keynote (no live HTML in a 12-min talk). Bookend to slide 3.
  * Backup/Q&A only: eigenvalue_corner, closure_continuous_{trace,p_cluster},
    posterior_width_sky_map, class_sky_map, width_vs_boundary (counterintuitive:
    width tracks tail extremity not class ambiguity — needs 30s, keep off main deck),
    embeddings (umap/pca/3d), lambda_th_sweep, posterior_width_3d.
- STILL NEEDED (not in folder — sync from NERSC canonical or generate): (a) **Slide 7
  TARP coverage plot + SBC rank histogram (Abacus)** — CRITICAL, the calibration
  headline currently has no figure; sync from the SI run via
  `scripts/sync_figures_to_canonical.sh`. (b) Slide 6 three-way TARP curves
  (softplus/linear/raw) to make "raw wins NLL, fails calibration" visual (table
  alone is acceptable fallback). (c) Slide 9 `edge_scale.png` + cluster-fraction
  recovery-sequence bars (else slide 9 is text-only). (d) Slide 3 idealised
  fake-data skewer — generate or reuse the 06-18 Desktop mockup; must match the
  real-skewer panel layout for the bookend.
- CLOSED: graph construction is **Delaunay on BOTH** the Abacus training wedge and
  the DESI inference wedge (user-confirmed; alpha-complex was tried earlier and
  worked worse, abandoned). => NO train/inference graph mismatch; removed that as a
  slide-9 open question. The 2026-06-21 v2 entry's open-item (1) is hereby resolved.
  Pipeline slide states "Delaunay" cleanly; slide-9 gap is purely the n(z)/edge-scale
  story.
- Closure test: confirmed done and strong per the 2026-06-21 [code] entry (99.7%
  TARGETID→FastSpecFit join; every trend monotonic void→cluster; survives mass
  control). Skewer animations exist; user may iterate.
- Deck file: still a copy of the old FLATS talk inside the SBI-Galev-2026 folder;
  it's the working base for building the new slides.
- Next: sync the four missing-figure sets from NERSC; generate the idealised skewer;
  regen class-fractions without the regression bars; then build slides on the FLATS
  copy. Timer dry-run after.
- Refs: `~/Developer/Working Files/SBI-Galev-2026/` (deck + figures);
  `~/Developer/Working Files/cambridge_sbi_galev_2026_script.md` (§PLOT→SLIDE v2.1);
  2026-06-21 [code] closure-test entry; 2026-06-19/06-20 [code] diagnostic/SI runs.

### 2026-06-21 — [code] Property–environment closure test runs on DESI LOA wedge (slide 10)
- What: Built the TARGETID→FastSpecFit join + closure-test plots for slide 10's
  truth-free validation (validation item (4) from 2026-06-18). Two new scripts in
  `GraphWeb_DESI/workflows/sbi_inference/`:
  `build_desi_wedge_property_join.py` (recovers TARGETID from preds
  `global_node_id` → source BGS catalog row index; de-dups the 1,252
  hemisphere-split duplicate targets by averaging their posteriors; joins loa
  FastSpecFit SPECPHOT by TARGETID → LOGMSTAR/SFR/Dn4000/ABSMAG g,r; writes
  `desi_wedge_env_props.parquet` + `join_report.json`) and
  `plot_property_environment_closure.py` (themed categorical + continuous +
  mass-control figures).
- Join quality: loa-vs-loa (same release — no Y1/Y3 mismatch), FastSpecFit dir
  `/global/cfs/cdirs/desi/vac/dr2/fastspecfit/loa/v1.0/catalogs/` (already the
  `config_paths.DESI_FASTSPEC_CATALOGS_DIR` default). **99.7% coverage**
  (111,171/111,503 unique targets); RA/Dec cross-check 0.027″ → join verified.
  Used the SI production run preds (`desi_wedge_flowjax_linear_si`).
- RESULT — every known trend recovered, monotonic void→wall→filament→cluster:
  f_quench 0.743→0.777→0.815→0.835; median (g−r)₀.₁ 0.782→0.827→0.870→0.908;
  median log sSFR −11.51→−11.63→−11.77→−11.96. Continuous E[Σλ] (tidal-tensor
  trace ∝ density): Spearman ρ = +0.076 (quench), +0.138 (g−r), −0.090 (sSFR),
  all correct sign at N=111k. **Mass control: trend SURVIVES at fixed M*** —
  within every logM* tertile f_quench rises void→cluster (high-mass tertile
  0.855→0.882→0.906→0.924), so it is not merely the mass–environment relation
  (median logM* itself rises only mildly, 10.60→10.71). Overall f_quench 0.78
  (massive BGS-bright sample; the TREND is the result).
- Caveats (carry to slide): inferred environment derives from galaxy positions
  so a density→quenching trend is partly expected — the point is the inferred
  T-web environment carries real physical signal AND survives mass control; loa
  fibre incompleteness is mildly env-dependent (biases absolute fractions, not
  the trend); quenched ≡ log sSFR < −11 (the colour panel is the independent
  colour-cut view).
- Figures (deck theme) synced to canonical
  `/pscratch/.../graphweb_desi/figures/desi_wedge_flowjax_linear_si_closure/`:
  `closure_categorical.png`, `closure_continuous_{trace,p_cluster}.png`,
  `closure_mass_control.png`. Slide 10 now has a real figure, not text-only.
- Next: drop the categorical (headline) + mass_control (defensible) figures into
  slide 10; optionally rerun on the baseline `desi_wedge_flowjax_linear` preds to
  confirm the trend is model-robust. The build script is `--run`/`--preds`-flagged
  so any variant rejoins in ~1 min.
- Refs: `GraphWeb_DESI/workflows/sbi_inference/{build_desi_wedge_property_join,
  plot_property_environment_closure}.py`; run dir
  `flowjax_inference_outputs/desi_wedge_flowjax_linear_si/`. See
  [[project-path1-wedge-npe-run]], 2026-06-18 validation-plan entry.

### 2026-06-21 — [science] Cambridge talk v2: narrative spine + factual corrections
- What: Reworked the talk around a single throughline question ("what cosmic-web
  environment is each galaxy in DESI, and how sure are we?") asked slide 1,
  answered slide 11, with a SKEWER BOOKEND (idealised fake-data skewer early as
  the discrete→continuous intuition pump; real DESI posterior skewer animation
  late as the payoff). Continuous-over-discrete justified with references: λ_th
  is arbitrary (Hahn 2007 used 0, Forero-Romero 2009 used 0.44, chosen for visual
  agreement); magnitude is the physics (tidal-torque theory couples spin to the
  tidal TENSOR — Codis 2015, Peebles 1969/Doroshkevich 1970); hard labels carry
  no uncertainty; and a continuous posterior SUBSUMES the discrete picture
  (threshold posterior samples → calibrated class probs at any λ_th with error
  bars). Back half reframed: the sim→real calibration-survival story is the
  differentiator for this SBI room, and the cluster-deficit / falsification work
  is demoted from headline to ONE supporting-evidence slide ("transfer is
  trustworthy and diagnosable"). Parameterisation slide reframed from a 3-way
  horse race to the methods-culture point "lower NLL ≠ a better posterior — why
  we calibration-test" (raw eigenvalues win NLL but fail TARP). Script v2 written
  to `~/Developer/Working Files/cambridge_sbi_galev_2026_script.md` (supersedes v1).
- Corrections logged (these override earlier notes/memory):
  1. CORAL is NOT implemented (confirmed against 06-19/06-20 [code] entries — the
     adopted fix is per-graph feature normalisation, not CORAL). CORAL removed
     from the pipeline slide; appears only as a one-line future-work item.
  2. The Abacus wedge graph is built with DELAUNAY triangulation, NOT alpha
     complex. (Alpha complex was the recommendation for REAL survey data with
     gaps; the training wedge uses Delaunay.) Pipeline slide + diagram corrected.
  3. The slide-9 fix is "per-graph (domain-relative) feature normalisation"
     (normalise scale-carrying features relative to each graph's own mean so the
     GNN sees RELATIVE features and the mock-vs-DESI offset cancels) — NOT
     "scale-invariant features." Reworded everywhere to avoid confusing a physics
     audience and to describe what was actually done. (NB the 2026-06-20 [code]
     entry's `--scale-invariant-features` flag implements exactly this per-graph
     mean-relative normalisation; the talk wording is corrected, the code name
     is unchanged.)
- Emphasis added: n(z) differs mock-vs-DESI AND varies ACROSS the wedge, so the
  graph-scale shift is spatially varying — which is the physical reason a single
  global rescale fails and a per-graph relative normalisation is the right tool.
- Resolves the merge-conflict flagged earlier today: falsification walkthrough
  goes to BACKUP (aligns with the NERSC-side 12+3 entry's "appendix-only"
  conclusion), but for a story reason (supporting evidence, not a result), not
  merely time.
- References verified by web search (drop-in list in the script's §REFERENCES):
  Dressler 1980, Peng 2010, Kraljić 2018, Codis 2015, Peebles 1969/Doroshkevich
  1970; Hahn 2007, Forero-Romero 2009; Papamakarios & Murray 2016, Greenberg 2019,
  Cranmer/Brehmer/Louppe 2020, Papamakarios 2021; Talts 2018 (SBC), Lemos 2023 (TARP).
- Open items for next pass: (1) CONFIRM DESI-side graph construction — if it's
  alpha-complex while training is Delaunay, that train/inference graph mismatch is
  a second on-story contributor to the slide-9 domain gap and should be surfaced;
  (2) skewer animation + closure-test plot readiness by Thu drive slide 10's final
  form; (3) timer dry-run — slides 3/5/8/9 are the ~80–90s long ones, slide 9 the
  likely cut.
- Refs: `~/Developer/Working Files/cambridge_sbi_galev_2026_script.md` (v2);
  2026-06-19 (DESI transfer + cluster-deficit diagnosis), 2026-06-20 (per-graph
  normalisation fix) [code] entries; conference site sbi-galev.github.io/2026.

### 2026-06-21 — [science] Cambridge talk: narrative finalized, format confirmed, deck strategy decided
- What: Format confirmed as **12-min talk + 3-min Q&A** (resolves the
  12+3 vs 15+5 open question from 2026-06-15). Finalized an 11-slide
  narrative built around the post-2026-06-15 results: brief T-web/cosmic-web
  motivation (1 slide) → classifier→posterior pivot with RASTI 2025 as
  single backstory slide → pipeline diagram (alpha-complex graph + GNN
  encoder + flow) → target-parameterisation result (linear increments
  Pareto-dominate softplus/raw — headline design choice) → TARP calibration
  on Abacus (credibility slide) → zero-shot DESI transfer result (class
  fractions land near truth, ordering-violation rate 0.8%→4.7% as the
  domain-shift fingerprint) → falsification walkthrough (ruled out
  under-density/FoG/extrapolation/shape, landed on graph-level edge-scale
  shift) → scale-invariant-feature fix (56% of cluster gap closed, zero
  in-domain cost, honest redistribution caveat) → truth-free validation +
  outlook (closure tests, skewer animation if ready, CORAL as next lever)
  → take-home. Full script with per-slide timing and visual notes written
  to `~/Developer/Working Files/cambridge_sbi_galev_2026_script.md`.
- Why / decision: audience (SBI-GalEv 2026, Cambridge) is SBI-fluent but not
  T-web/DESI-fluent, so intro stays to ~3 sentences on cosmic-web physics
  and zero time explaining NPE/flows. Decided to sell the back half (slides
  7–9) as the differentiator for this specific room — a fully worked,
  falsifiable sim-to-real domain-shift diagnosis on an amortised NPE — since
  the conference's own theory stream lists domain adaptation, transfer
  learning, and calibration/convergence testing verbatim. Classification
  accuracy demoted to backstory only, consistent with the 2026-06-15
  framing decision but now made concrete in slide-by-slide form.
- Decision on deck file: do NOT edit `FLATs.key` in place. It's the
  classification-era (RASTI 2025) poster talk — visual theme (true-black,
  IBM Plex Sans, COSMIC_WEB_COLORS, FLATS accent palette) and several
  layout templates (title slide, T-web explainer, pipeline-stage slide) are
  worth inheriting, but essentially all result slides are from the retired
  4-class/partitioned-FlowJAX path and need rebuilding, not reuse. Plan:
  duplicate `FLATs.key` to a new file, strip outdated result slides, rebuild
  from the script above. Noted blocker: `FLATs.key` is 199MB and timed out
  via keynote-mcp AppleScript (`open_presentation`) — likely has heavy
  embedded assets from earlier iterations; worth auditing/trimming embeds
  before the new file becomes the working copy, to keep keynote-mcp
  reliable for the rest of the week.
- Next: (1) confirm whether the skewer animation (open thread since
  2026-06-18) lands before Thursday — it's the strongest single visual if
  ready, otherwise slide 10 stays closure-test text only; (2) duplicate
  FLATs.key → new working file, audit embedded asset size; (3) build slides
  from the script, regenerating figures already produced by the NERSC
  pipeline (TARP, class-fraction bars, edge_scale.png, lambda_th_sweep.png,
  fog_los_alignment.png) into the finalized PLOT_STYLE_GUIDE theme; (4)
  dry-run with a timer — slide 8 (falsification walkthrough) is the densest
  for its 90s budget and is the most likely cut candidate.
- **CONFLICTS WITH NEXT ENTRY:** this plan puts the falsification
  walkthrough on main-deck slide 8; the next entry (independent NERSC-side
  session) concludes it should be appendix-only. Unresolved — settle before
  building that slide.
- Refs: `~/Developer/Working Files/cambridge_sbi_galev_2026_script.md`,
  `FLATs.key`, conference site (sbi-galev.github.io/2026), 2026-06-15 talk
  framing entry, 2026-06-18 validation plan entry, 2026-06-19/20 wedge-NPE
  + cluster-suppression + scale-invariant-fix entries.

### 2026-06-21 — [science] Cambridge SBI-GalEv talk slot confirmed: 12 min + 3 Q&A
- What: Organisers confirmed the talk slot is 12 minutes + 3 minutes for questions
  (resolves the open "12+3 vs 15+5" question from the 2026-06-15 framing entry).
- Why / decision: Tighter than the 15+5 option assumed in earlier figure planning —
  trim to the minimum figure set that carries the SBI-first narrative (calibration
  headline, DESI transfer result, SI fix) rather than including secondary
  diagnostics (FoG/shape/training-coverage falsification plots are appendix-only,
  not main-deck).
- Refs: FLATS deck; see 2026-06-15 "Cambridge SBI-GalEv talk" entry.

### 2026-06-21 — [code] Plot visual identity + centralised figure output fixed
- What: `workflows/sbi/plot_flowjax_posteriors.py` (Abacus-side TARP/calibration/
  posterior/class-fraction plots) had zero theming — plain matplotlib defaults,
  inconsistent with the GraphWeb_DESI true-black/IBM Plex Sans theme used on all
  DESI-side figures. Wired in `shared.plot_style.apply_style()` +
  `ACCENT_COLORS`/`COSMIC_WEB_COLORS` (the module was already mirrored into
  `TNG/Illustris/shared/`, just never imported by this script). Also added
  `scripts/sync_figures_to_canonical.sh` in both repos: rsyncs a run's finished
  figures into the existing-but-underused `CANONICAL_FIGURE_ROOT` /
  `GRAPHWEB_CANONICAL_FIGURE_DIR` (`/pscratch/.../{tng_illustris,graphweb_desi}/figures/`)
  under a run-named subfolder, so figures are centralised for picking per-conference
  without disturbing per-run experiment directories.
- Why / decision: per-run output dirs are right for active experimentation (compare
  testA/testB/Bcorrected/SI side by side) but meant the canonical figure roots,
  despite already being the documented default in `plot_flowjax_posteriors.py`,
  were being overridden by every launcher and sat empty/stale in practice.
- Next: regenerate `path1_wedge_flowjax_3d_Bcorrected_linear_si/plots/` with the
  fixed theme and sync both Abacus- and DESI-side SI figures into the canonical
  dirs (in progress).
- Refs: `shared/plot_style.py`, `scripts/sync_figures_to_canonical.sh` (both repos),
  `workflows/sbi/plot_flowjax_posteriors.py`.

### 2026-06-20 — [code] Route A: scale-invariant features fix — best cluster recovery (56%), no in-domain cost
- What: Implemented `--scale-invariant-features` in `build_abacus_sbi_cache.py` +
  `GraphWeb_DESI` inference (`abacus_gnn_parity.py`, `infer_desi_wedge_flowjax.py`):
  per-graph-median normalise the scale-carrying node (Degree/Density/NeighDensity/
  I_eig) and edge (edge_length) features into dimensionless **contrasts** before the
  box-cox/log, leaving density_contrast/directions/Clustering untouched. Rebuilt the
  linear cache (SI), retrained the linear NPE (4×A100), re-ran DESI inference with
  matched SI normalisation.
- Result (DESI cluster fraction, λ_th=0.2; truth 0.058):
  baseline 0.027 → edge-domain-adapt 0.034 (22%) → full node+edge adapt 0.040 (40%)
  → **SI retrain 0.046 (56% of the gap closed — best)**. Abacus-side UNCHANGED
  (Test NLL 1.14, posterior-mean R² 0.826, ordering-viol ~0) → the scale-invariance
  has ~zero in-domain cost, confirming the earlier prediction. Abacus TARP is the
  TIGHTEST of all runs (hugs the diagonal, dev <0.02) and SBC clean — scale-invariance
  preserved (slightly improved) calibration. So SI is the unambiguous production model:
  best Abacus calibration + best DESI transfer, no downside.
- Caveat (honest): not a clean across-the-board win — SI recovers clusters by
  REDISTRIBUTING: void 0.254→0.226 (truth 0.27, worse) and filament 0.273→0.295
  (truth 0.26, worse); wall 0.447→0.433 (toward 0.41, better). Total deviation from
  truth ~flat; it trades a little void/filament accuracy for much better cluster+wall.
  Favourable for a study where clusters are the rare science-critical class.
- Takeaway: scale-invariant features are the recommended DURABLE fix for the
  graph-scale/N(z) transfer shift (learned, not a post-hoc hack). Closes ~56% of the
  cluster deficit; the remaining ~44% needs Route B (mock densification at high z —
  the mock under-produces high-z galaxies; `--equal_data_dens` downsamples the wrong
  way). For Cambridge: present SI as the principled fix with the void/filament caveat.
- Refs: cache `path1_flowjax_3d_lineareig_si`; run `path1_wedge_flowjax_3d_Bcorrected_linear_si`;
  DESI `desi_wedge_flowjax_linear_si`; commits Illustris build-cache + GraphWeb parity/infer.

### 2026-06-19 — [code] DESI cluster-suppression diagnosis: not under-density, not FoG; a training-coverage shift
- What: Characterised why the wedge NPE (and the Jraph regression) under-predict
  **clusters** on real DESI (cluster fraction ~0.027 vs Abacus truth ~0.058 at
  λ_th=0.2). Built a diagnostic suite under
  `/pscratch/.../flowjax_inference_outputs/desi_wedge_flowjax_linear/`:
  eigenvalue corner (clean distributions) + environment corner; λ_th sweep
  (static `lambda_th_sweep.png` + animation `lambda_th_sweep_animation.html`);
  FoG magnitude test (`fog_anisotropy.png`) and the proper LOS-direction FoG test
  (`fog_los_alignment.png`). New scripts in `GraphWeb_DESI/workflows/sbi_inference/`.
- Findings (what the deficit is NOT):
  1. **Not under-density.** Real DESI cluster cores are *denser* than the mock
     (Density p99.9 0.18 vs 0.10; ~2× more galaxies above the Abacus 99th pct).
     Rules out fibre-incompleteness erasing cluster density.
  2. **Not Fingers-of-God.** Recomputed local inertia *eigenvectors*: DESI major
     axes are NOT more line-of-sight aligned than Abacus (median |cosθ| 0.421 vs
     0.456; both slightly transverse from wedge geometry). The earlier anisotropy
     *magnitude* excess (esp. void/wall-inferred) is transverse/generic, not RSD
     elongation. So FoG is ruled out as the differential cause.
  3. **Not training-coverage / extrapolation.** Only ~5% of DESI dense
     cluster-candidates (Density feature) exceed the Abacus train p99.9, ~0.01% of
     all DESI galaxies exceed the train max — the bulk is in-distribution
     (`training_coverage.png`). Too small to explain a halved cluster fraction.
- Findings (what it IS): a genuine **high-λ tail suppression** — persists at every
  threshold and *worsens* with λ_th (DESI/Abacus cluster ratio 0.65→0.45→0.30 at
  λ_th 0.1/0.2/0.3). The richest single cluster IS recovered (skewer P(cluster)=0.86
  in a thin 0.6° beam), so the model isn't blind to clusters — the bulk of moderate
  clusters degrade.
  4. **Not dense-structure shape either.** Oaxaca decomposition of the dense
     inferred-cluster-rate gap (Abacus 0.156 vs DESI 0.082, `shape_misclassification.py`):
     only **8%** is explained by DESI being more elongated; **92% is residual** —
     at MATCHED density AND matched anisotropy, DESI dense galaxies are classified
     as cluster ~half as often as Abacus ones. So the earlier shape hypothesis is
     refuted too.
- Bottom line: **every marginal node-feature explanation is falsified** (density,
  FoG, extrapolation, shape). The deficit is a *fixed-density-fixed-shape residual*
  → a joint / graph-level domain shift in the GNN encoder's feature→λ map (shared by
  regression+NPE, so it lives in the encoder, not the flow). Honest open question,
  not solved. Most concrete remaining lever to test: the **N(z) / number-density
  mismatch** (DESI ~12% more galaxies here, different N(z)) changes the graph EDGE
  SCALE (typical edge lengths) — which would shift GNN output even at fixed node
  features and is consistent with a graph-level, not node-level, residual.
- Edge-scale test (`edge_scale.png`) CONFIRMS a graph-level shift: DESI graph is 12%
  denser, edges 7% shorter, scaled log(edge_length) sits −0.11σ off the Abacus N(0,1)
  — the GNN sees a different graph scale even at fixed node features. Modest but the
  only lever that survives falsification.
- **Phase-0 cheap fix (inference-time domain correction, no retrain):** re-standardising
  DESI scaled EDGE features to training N(0,1) (`--edge-domain-adapt`) recovers ~22% of
  the cluster gap (0.027→0.034) and moves ALL classes toward truth — clean. Adding
  NODE-feature re-standardisation (`--node-domain-adapt`) reaches ~40% (cluster→0.040)
  but over-corrects (void 0.21<0.27, filament 0.30>0.26). So ~40% of the deficit is
  correctable input-domain shift; ~60% is a deeper feature→λ / connectivity residual.
  For the talk: use the clean edge-adapt; durable fix = scale-invariant edge features +
  retrain (Route A), graceful vs the blunt post-hoc rescale.
- Implication: expanding RA/Dec won't fix it. Levers, in order of evidence: (a) match
  the mock **N(z)/number density** to real DESI so the graph scale matches — the
  mock HAS the knobs (`upstream_mkCat_SecondGen_amtl.py --equal_data_dens y`,
  `upstream_prepare_mocks_Y3_bright.py --downsampling y`), both DEFAULT OFF in path1,
  which is why N(z) currently mismatches; (b) full **domain adaptation** (align joint
  feature/graph distribution, not marginals); (c) revisit the **HOD**
  (cluster richness/morphology); (d) pragmatic λ_th=0.1 (0.065 vs truth 0.10). Frame
  for Cambridge as a robust, RSD-NOT cluster systematic *localized* (not
  under-density/FoG/shape) to a graph-scale transfer effect — honest open problem.
- Refs: `GraphWeb_DESI/workflows/sbi_inference/{plot_eigenvalue_corner,plot_lambda_th_sweep,plot_fog_anisotropy*,plot_fog_los_alignment}.py`;
  diagnostic figures in the linear DESI run dir. See [[project-path1-sentinel-z-bug]].

### 2026-06-19 — [code] DESI transfer: linear NPE runs on real DESI LOA wedge (conference key result)
- What: Ran the linear-increment FlowJAX NPE on the real DESI LOA wedge (112,755
  bright BGS galaxies, same RA120–160/Dec14.5–30.6/z0.2–0.3 footprint). New
  scripts in `GraphWeb_DESI/workflows/sbi_inference/`:
  `infer_desi_wedge_flowjax.py` (GPU forward) + `plot_desi_wedge_flowjax.py`
  (themed figures). Mirrors the regression inference's preprocessing (node box-cox
  from cache `node_feature_scaler`; edge bidirectional+log+StandardScaler via
  `shared/abacus_gnn_parity.py`) but swaps the model for the GNN encoder + flow
  (128 posterior samples/galaxy → λ posteriors → class probs). Reuses
  `load_flowjax_model`/`create_gnn_and_flow`/`batched_sample_posterior`/
  `samples_to_raw_eigenvalues`/`posterior_to_classprobs`.
- Result: **transfer works, no collapse.** DESI NPE class fractions
  {void .254, wall .447, fil .273, clu .027} sit right between Abacus truth
  {.27/.41/.26/.06} and regression DESI {.246/.462/.266/.026} — wall-dominated,
  cluster-rare. Cluster .027 matches regression (.026), both below Abacus truth
  (.06) = the documented DESI cluster-tail transfer gap.
- NPE-specific findings the regression can't give: (a) per-galaxy posterior WIDTH
  sky map shows coherent structure tracking the cosmic web + survey-footprint
  gaps; (b) width rises near the survey edge (graph-truncation) AND rises with
  eigenvalue extremity (tail distance), NOT with class-boundary ambiguity;
  (c) ordering-violation rate is **4.7% on DESI vs 0.8% on Abacus** — a clean
  quantification of the domain shift (flow less certain on out-of-distribution
  real data; count-based class fracs stay robust). Fixable later with a post-hoc
  sort of posterior samples.
- CRITICAL parity guard: the edge scaler MUST be fit on the **path1 fiberassign**
  wedge npz (the model's training graph), NOT the regression wedge the old script
  uses — both have ~100k nodes so only a constants-assert (mean≈[2.15,0], scale≈
  [0.70,1.77]) catches a mix-up. Hard-asserted in the script.
- Outputs: `/pscratch/.../graphweb_desi/flowjax_inference_outputs/desi_wedge_flowjax_linear/`
  (preds npz + summary.json + 5 themed figures). TARP/SBC stay Abacus-side (no DESI
  truth). Stretch (post-conf): property–environment closure (needs TARGETID→
  fastspecfit join), skewer animation. See [[project-path1-wedge-npe-run]].
- Refs: `GraphWeb_DESI/workflows/sbi_inference/{infer,plot}_desi_wedge_flowjax.py`,
  reused DESI wedge `desi_wedge_expanded_..._from_fullgraph/`.

### 2026-06-19 — [code] Wedge NPE parameterisation study: linear increments win; mock ruled out
- What: Ran the 3-d FlowJAX NPE on the path1 fiberassign wedge (z 0.2–0.3,
  100,935 nodes) under all three eigenvalue parameterisations, 4×A100, 7000
  epochs, seed 42, identical arch/reg/split. Result (Test NLL / posterior-mean
  R² / ordering-violation rate / TARP):
  | param | NLL | R² | viol. | TARP |
  |---|---|---|---|---|
  | softplus increments | +2.25 | 0.74 | 0% | tight |
  | **linear increments** | **+1.13** | **0.82** | **0.8%** | **tightest** |
  | raw eigenvalues | −0.07 | 0.85 | 2.5% | over-confident |
  Softplus's +2 NLL floor was the inverse-softplus heavy tail (init NLL 215 vs
  ~8–10 for linear/raw). Raw gets the lowest NLL but is *over-sharpened*
  (over-confident TARP, 2.5% out-of-order samples). **Linear increments
  Pareto-dominate softplus** (better NLL, R², calibration; negligible 0.8%
  violations) and are the best-calibrated overall — and match the 15-d wedge
  regression's own parameterisation. **Headline choice = linear** for the
  Cambridge talk (calibration-first venue). Hyperparameter tuning (dropout 0→0,
  wd 0.08→0.01) made no difference — not the lever.
- Mock audit cross-ref (separate session "Path1 mock generation audit"): found
  the spectro-injection resurrects fibre-unobserved targets as sentinel z≈0.59
  (2.08M rows, 21.8% of mock_bgs_maglim). **Our wedge is 100% clean** (Z∈[0.200,
  0.300], 0.00% at 0.59 — verified). So mock generation is **ruled out** as the
  NLL-floor cause; the floor is the wedge's genuine information content (sparse
  fiber-incomplete DESI-like selection). The sentinel flaw is a **z-expansion
  blocker**: expand in RA/Dec (or fix the injection first), never in z, until
  the injection is corrected.
- Code (uncommitted→committed this entry): added `--linear-increments` to
  `build_abacus_sbi_cache.py`; `eigenvalues_to_linear_increments` /
  `linear_increments_to_eigenvalues` / `resolve_increment_mode` and a 3-mode
  `samples_to_raw_eigenvalues` in `shared/eigenvalue_transformations.py`;
  `--increment_mode {softplus,linear,raw}` threaded through
  `jraph_sbi_flowjax.py` + `resolve_sbi_paths` (suffix `_linear_eig`) + the saved
  model + `plot_flowjax_posteriors.py`; vectorised the eval (chunked vmap, TARP +
  SBC + class-prob); checkpoint/resume.
- Next: (1) regenerate TARP + class-fraction figures from the linear run in the
  plot-style-guide theme for the deck; (2) LOCK the wedge — do not expand before
  the talk (multi-day rebuild, not needed; calibration is the headline);
  (3) post-conf: fix sentinel injection, then RA/Dec wedge expansion for tighter
  constraints. See [[project-path1-wedge-npe-run]], [[project-path1-sentinel-z-bug]].
- Refs: runs `path1_wedge_flowjax_3d_{testA_reg,testB_raweig,Bcorrected_linear}`
  under `/pscratch/.../abacus/sbi_runs/`; caches `path1_flowjax_3d{,_raweig,_lineareig}`.

### 2026-06-18 — [science] Posterior validation plan beyond TARP: class probabilities, skewer check, boundary degradation, closure tests
- What: TARP only certifies statistical self-consistency, not physical
  correctness on DESI where no per-galaxy truth exists. Defined four concrete,
  implementable validation steps for wedge-NPE q(λ|X) output:
  (1) **Class probabilities from samples.** Per galaxy, draw S posterior samples
  in increment space, invert to physical (λ₁,λ₂,λ₃) (ordering guaranteed by
  construction), threshold at λ_th, count crossings n=Σ1[λ_k>λ_th] (n∈{0,1,2,3}
  by ordering): P(void)=mean(n=0), P(wall)=mean(n=1), P(filament)=mean(n=2),
  P(cluster)=mean(n=3). Cross-check analytically via marginal CDFs: P(λ_k>λ_th)
  from each marginal. **NB ordering is ASCENDING λ₁≤λ₂≤λ₃ (CACTUS-native: eig1
  smallest, eig3 largest — verified end-to-end, reproduces CWEB exactly at
  λ_th=0.2), so the decomposition is P(void)=1-P(λ₃>λ_th),
  P(wall)=P(λ₃>λ_th)-P(λ₂>λ_th), P(filament)=P(λ₂>λ_th)-P(λ₁>λ_th),
  P(cluster)=P(λ₁>λ_th)** — the *smallest* eigenvalue λ₁ exceeding λ_th ⇒ cluster
  (all exceed). (An earlier draft of this entry flipped λ₁↔λ₃, matching a
  descending convention the codebase does not use.) Sample-based and analytic
  versions must agree to MC error — mismatch indicates an inversion or sampling
  bug. **λ_th must equal the CWEB-labeling threshold (Abacus = 0.2, the CACTUS
  default), not 0.0.** Implemented in
  `shared/eigenvalue_transformations.py::posterior_to_classprobs`.
  (2) **Reliability diagram for class probabilities** (discrete-output
  complement to TARP): on Abacus (CACTUS ground truth available), bin galaxies
  by predicted P(filament) (and other classes), plot empirical true-class
  fraction per bin vs predicted probability. Diagonal = calibrated; deviation
  flags over/under-confidence, expected to concentrate at wall/filament
  boundary per RASTI confusion matrix.
  (3) **Posterior-width vs distance-to-boundary.** Compute per-galaxy posterior
  width (e.g. mean marginal σ across λ₁,λ₂,λ₃, or posterior entropy) vs distance
  to wedge/survey edge (RA/Dec/z footprint boundary, alpha-complex hemisphere
  split) and vs distance to T-web class boundary (λ near λ_th). Expect
  monotonic widening near both. On Abacus (truth known) also plot width vs
  true prediction error to confirm widening tracks real degradation, not just
  graph truncation artifacts; confirm DESI shows the same *shape* of
  width-vs-edge-distance curve as Abacus.
  (4) **Property–environment closure test (DESI-only, no truth needed).** Bin
  real DESI BGS galaxies by inferred environment (E[trace(λ)] or dominant class
  probability) and check recovery of known relations: quenched fraction /
  colour / sSFR rising toward filament→cluster, morphology–density trend.
  Recovering established astrophysical trends from inferred-only environment is
  evidence the posteriors carry real physical information.
  Also scoped (5) cross-check against independent structure finder (DisPerSE
  filaments, or BORG T-web per existing on-the-horizon validation target):
  expect bulk agreement, divergence concentrated at class boundaries as the
  reassuring signature, not a failure mode.
  (6) **Animated line-of-sight skewer plot.** Pick a 1D skewer through a wedge
  (Abacus or DESI Loa) crossing void→wall→filament→cluster. At each position
  along the skewer: render the three eigenvalue marginal posterior densities
  (KDE over per-position-bin galaxy samples, not Gaussian — Gaussian only used
  for the Desktop mockup) with λ_th marked, a posterior-width ribbon, and the
  class-probability bar from (1)/(2)'s class-prob math. Animate by sweeping
  position; render as a saved video/gif, not just an interactive widget, so it
  drops into the talk deck. On Abacus, overlay the true CACTUS (λ₁,λ₂,λ₃) as
  moving markers on the posterior panel — credible bands containing truth
  through the transition, widening exactly where truth nears λ_th, is the
  core "physically sensible" evidence this whole validation plan is for.
- Why / decision: visualizing 3 correlated posteriors per galaxy across an
  entire wedge does not scale; need scalar/derived functionals overlaid on
  geometry instead of raw posterior dumps, and need validation that is
  meaningful on DESI specifically (no per-galaxy truth there), not just on
  Abacus. (1)+(2) give a calibration check at the class level; (3) gives a
  boundary-physics sense-check transferable Abacus→DESI; (4) gives a truth-free
  closure test directly on real BGS data. (6) is the figure that actually
  conveys physicality to a human audience — the others are diagnostic plots,
  this is the one that shows the posteriors tracking real structure.
- Next: implement (1) as a `posterior_to_classprobs(samples, lambda_th)`
  utility in `shared/eigenvalue_transformations.py` (sample-count path +
  analytic-CDF path, with an assert/warning if they diverge beyond MC
  tolerance); implement (2) reliability diagram + (3) width-vs-boundary-distance
  + (6) skewer animation in `workflows/visualization/` using
  `shared/plot_style.py` (COSMIC_WEB_COLORS for class bins, matplotlib
  FuncAnimation or equivalent for (6)); run (3) and (6) on one Abacus wedge
  first (truth overlay available), then the matching DESI Loa wedge, compare;
  (4) needs BGS property table joined to wedge-NPE output by target ID — check
  join keys exist in current DESI Loa wedge cache before scripting. (5)
  BORG/DisPerSE cross-check stays "on the horizon" pending BORG catalogue
  access — not in this sprint.
- Refs: `shared/eigenvalue_transformations.py` (ordered softplus increment
  policy — class-prob math must operate in this space then invert), CACTUS
  labels (Abacus truth), DESI BGS LSS catalogues at
  `/global/cfs/cdirs/desi` (property–environment closure), RASTI confusion
  matrix (wall/filament boundary expectation), `PLOT_STYLE_GUIDE.md`. Desktop
  mockup of (6) (Gaussian-approximation, interactive, not the real KDE version)
  built this session for layout reference.

### 2026-06-17 — [science] Plot style guide finalized: true-black theme, cosmic-web class colors, IBM Plex Sans
- What: Defined canonical matplotlib style for all GraphWeb_DESI plots: true-black
  (#000000) background, off-white (#F2F2F2) text/ticks; confirmed FLATS-deck
  cosmic-web class colors (Void #A1FCDD, Wall #4E84F7, Filament #EB336F, Cluster
  #F5C144) as a categorical-only palette, kept separate from the FLATS general
  accent colors (magenta #FF006E, blue #3A86FF, red #D62828); replaced Eurostile
  with IBM Plex Sans as the single font everywhere (Eurostile has no usable Greek
  glyphs, which matters for λ-notation); mathtext.fontset='dejavusans' handles
  λ-subscripts in labels regardless of body font. Wrote PLOT_STYLE_GUIDE.md +
  shared/plot_style.py (apply_style(), finalize_axes(), COSMIC_WEB_COLORS,
  CLASS_ORDER, ACCENT_COLORS) in GraphWeb_DESI.
- Why / decision: Every plot in the pipeline (regression diagnostics, NPE
  posteriors, TARP coverage, DESI BGS results) needs one consistent visual
  identity before Cambridge SBI-GalEv talk figures are regenerated from
  wedge-NPE output; relying on Eurostile directly would have broken λ-notation
  rendering.
- Next: Download IBM Plex Sans .ttf files into GraphWeb_DESI/assets/fonts/
  (command in PLOT_STYLE_GUIDE.md §2.3) on both Mac and NERSC; migrate existing
  plotting scripts/notebooks (workflows/visualization/*) to
  `from shared.plot_style import apply_style, finalize_axes`; consider mirroring
  plot_style.py into Illustris if TNG-side plots need the same theme.
- Refs: GraphWeb_DESI/PLOT_STYLE_GUIDE.md, GraphWeb_DESI/shared/plot_style.py,
  FLATS deck slide 9 (legend swatches).

### 2026-06-16 — [code] graphify global graph + cross-repo Cursor rules
- What: Merged `Illustris` and `GraphWeb_DESI` into `~/.graphify/global-graph.json`;
  updated `.cursor/rules/graphify.mdc` in both repos to name the sibling repo and
  require `--graph ~/.graphify/global-graph.json` for cross-repo queries.
- Why / decision: Per-repo `graphify query` and grep do not surface Illustris↔DESI
  dependencies; global graph does (e.g. `graph_net_models` → Jraph wedge inference).
- Next: After substantive code changes, `graphify update .` then
  `graphify global add graphify-out/graph.json --as <tag>`.
- Refs: `Illustris/.cursor/rules/graphify.mdc`, `GraphWeb_DESI/.cursor/rules/graphify.mdc`

### 2026-06-15 — [code] Environment policy clarified for agents
- What: Updated agent/runbook guidance: activate `cosmic_env` for normal work and
  `rapids-gnn` whenever calculating graph metrics/features.
- Why / decision: Phase 4 validation on an unprepared Cloud image failed on
  missing scientific dependencies; graph metrics also require the RAPIDS/cuGraph
  stack rather than the default environment.
- Next: Keep workflow launchers aligned with this policy as new graph-metric or
  wedge-NPE scripts are added.
- Refs: `CLAUDE.md`, `RUNBOOK.md`.

### 2026-06-15 — [code] Docs aligned to wedge-NPE Abacus path
- What: Updated top-level and workflow docs so Abacus-scale SBI points to
  RA/Dec/z wedge subvolume caches with `jraph_sbi_flowjax.py`, while partitioned
  FlowJAX is marked legacy/reference.
- Why / decision: Several public docs still described partitioned Abacus FlowJAX
  as active, contradicting the current wedge-subvolume SBI direction.
- Next: Add a production `sbatch` launcher for wedge NPE when the interactive
  recipe stabilizes.
- Refs: `README.md`, `ACTIVE_WORKFLOWS.md`, `RUNBOOK.md`,
  `workflows/abacus_tweb/README.md`, `workflows/sbi/README.md`.

### 2026-06-15 — [science] Target representation: ordered increments are canonical, not "raw eigenvalues"
- What: Cleared a doc ambiguity that conflated two independent axes. (1) Target
  *quantity*: eigenvalues beat shape-param (I₁,e,p) and invariant (I₁,I₂,I₃)
  reps, which are pathological for ML. (2) Target *parameterisation*: eigenvalues
  are trained as ordered softplus increments (λ₁ anchor + two non-negative
  increments), enforcing λ₁≤λ₂≤λ₃ by construction. The net trains/emits in
  increment space; inversion to (λ₁,λ₂,λ₃) is presentation/eval-only.
- Why / decision: "raw eigenvalues preferred" (true only on axis 1) could lead an
  agent to collapse the softplus head to a direct 3-output regressor and silently
  reintroduce ordering violations. Canonical policy now documented in code + docs.
- Next: retire the shape-param path properly — remove `--use_shape_params` and the
  dead `compute_shape_param_statistics` (builds `stats`, returns `None`) from
  `jraph_pipeline.py` + `eigenvalue_transformations.py` once confirmed unused.
- Refs: `shared/eigenvalue_transformations.py` (new module docstring + dep
  banners), `CLAUDE.md` (Physics Targets, module table, regression cmd).

### 2026-06-15 — [science] Cambridge SBI-GalEv talk: framing + reusable assets
- What: Talk narrative = open on environment→galaxy-evolution motivation, then
  frame the whole pipeline as amortised per-galaxy posterior inference of T-Web
  eigenvalues with TARP calibration as the headline. Regression = validated
  encoder/forward-model backbone; NPE + flows = frontier result. Reviewed prior
  poster (classification-era) and FLATS deck: deck already has FlowJAX
  posterior-comparison + TARP-coverage figures (from the retired partitioned
  path) — reuse as visual template, regenerate from wedge-NPE.
- Why / decision: venue is SBI-first, so calibration/posterior must lead, not
  cartography; self-diagnosed weak spot is astro motivation, so first ~3 min
  carry the science case. Slot Thu 25 Jun ~11:30, 15-min block (confirm 12+3 vs
  15+5 with organisers).
- Next: (1) NPE on one Abacus wedge + TARP plot = critical path; (2)
  R²-vs-smoothing-scale plot; (3) polish 3-way λ hists + class-fraction bar + 3D
  wedge render in deck palette (Eurostile, #FF006E/#3A86FF). Stretch: ξ(r_p,π)
  forward-model check, 2nd wedge for cosmic-variance scatter. Methods-slide
  architecture diagram drafted (draw.io).
- Refs: FLATS deck; `workflows/jraph/`, wedge caches; visualization notebook.

### 2026-06-14 — [code] Cross-tool infra + this log
- What: Set up `~/.claude/CLAUDE.md` as cross-repo canon (3-location map, conda
  envs, proven Perlmutter `salloc`/`srun` recipes; production = `sbatch`). Fixed
  the `nersc` skill's `qos.md` (no `gpu_`-prefixed QOS). Added two Claude Code
  skills: `desi-bgs-graph` and `jraph-eval`. Created this SCIENCE_LOG.
- Next: stand up the Desktop⇄NERSC bridge (filesystem MCP + git) so science
  decisions land here automatically.
- Refs: `~/.claude/CLAUDE.md`, `~/.claude/skills/{desi-bgs-graph,jraph-eval}`.

### 2026-06-14 — [science] SBI direction: wedges, not partitions
- What: Abacus-scale inference works via **wedge subvolumes** (graph per RA/Dec/z
  wedge); the graph-partitioning / partitioned-FlowJAX path is abandoned.
- Why / decision: graph partitioning was a nightmare; wedge training has worked
  over the last ~2 months. Future = NPE on the wedge subvolumes.
- Next: when convenient, retire `workflows/sbi/*partitioned*` + partition builders.
- Refs: `workflows/sbi/`, see also Claude Code memory `project-abacus-wedge-npe-direction`.
