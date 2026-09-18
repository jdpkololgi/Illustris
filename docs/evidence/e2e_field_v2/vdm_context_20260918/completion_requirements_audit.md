# Completion requirements audit: approved A/B/C/D experiment

Independent read-only verification on login35, 2026-09-18. The previous goal
turn made progress by completing and committing the requested primary-paper
review; this turn verifies execution requirements and corrects the oracle
coverage interpretation. No scientific rerun or new allocation was launched.

## Scope and current verdict

Scope is the user's 248-line planning attachment
`06ff9a35-5268-4a44-a8f0-7f54b5b46c7c/pasted-text.txt`, followed by explicit
implementation approval, the active goal, the approved
`docs/e2e_vdm_context_diversity_v1.md`, the incorporated
`docs/e2e_vdm_context_audit_proposal_20260917.md`, and the subsequent explicit
v2 tidal-operator amendment. Later requests add literature/data-aware joint
go/stop reasoning, CFM discussion and the Gemini sanity check; they do not
authorize an automatic next training campaign.

**Execution and computational assessment are complete. Overall goal completion
still awaits inspection of the final scientific report and science-log/field-
plan updates owned by the login21 continuation.** The machine completion flag
alone does not establish those written deliverables. A scientifically negative
H2 result is a completed experiment, not a reason to keep training.

All Scratch paths below are relative to:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1`.

## Requirement-to-evidence mapping

| Requirement | Current inspected evidence | Assessment |
| --- | --- | --- |
| Audit the old run, geometry, channels and normalization | Approved audit sections 1--3 identify the previous 3,584-draw result/hash; distinguish 48-cubed grid from 324.768 Mpc/h parent and 108.256 core; identify local spatial plus broadcast wide-mean inputs; preserve DC and distinguish density/tidal boundaries. | Implemented in the approved contract; not an invented local/wide contrast. |
| Minimal identifying A/B/C/D comparison | Frozen MANIFEST spec and geometry: A32 nested in B384; B/C/D same three phases; common fine architecture; D additional factor/compute declared. | Meets amended design. A is the approved balanced fresh control, not historical checkpoints relabelled. |
| Whole-phase roles and sealed data | GEOMETRY has 416 primary anchors, quotas_pass=true, target_values_used=false, roles 000/002/003 train, 004 development, 005 internal confirmation. Model/test guards reject 001/006 and symlink aliases. No sealed target was read in this closeout. | Roles/guards established; 004/005 correctly labelled programme-exposed, not globally blind. |
| Independent diversity versus augmentation | Geometry separation and quota receipts; contract identifies overlapping parents and seven offsets. | 384 cores are not 384 independent universes. Two-to-three phase change and exposure are explicit. |
| Train-only preprocessing | data/NORMALIZATION.json has 32 fit IDs from ph000/ph002, seven common offsets, per_patch_mean_removed=false. | Shared scaling and observational bin edges train-only; no per-field truth normalization. |
| Physically consistent multiscale chart | Full-size smoke, block-mass/projector/roundtrip tests, actual GPU generated-coarse inference; fine 63/64-DOF construction in frozen source. | Positive density, nonduplicated block mass and restricted factorization tested; not proof of exact joint posterior. |
| Physical tidal gate and failure preservation | Original REPRESENTATION_GATE scientific_representation_pass=false remains. V2 exact_identity_pass/scientific_representation_pass/training_launch_allowed=true, heldout_used=false, eigenvalue reductions 0.841744/0.849633/0.848787. | Approved correction passed before training. Computational 192-cubed closure did not change generated 48-cubed grid. |
| Observation-only deployment and labelled oracle | SMOKE observation_only_inference=true; source separates sampled coarse from oracle_diagnostic; model rejects training_truth in evaluation. | No deployable result uses oracle truth. Wide-only coarse conditioning and conditional fine-core independence explicitly retained as limitations. |
| Shared-parent addressing and independent fine noise | Draw ledger/address tests; DRAW_INTEGRITY verifies actual coarse/fine payload receipts and exact shared-parent bindings. | Adjacent requests reuse parent draws; fine noise remains independently addressed. Does not imply all residual fine dependence is learned. |
| Controlled optimization and two seeds | MODELS_FROZEN all_models_frozen=true, 10 branch receipts and 30 checkpoints; choice policy all20480 final, no validation-selected checkpoint. MANIFEST frozen revision 16398511d4552bd041e8d1e97cebd3e168088064. | Eight fine/two coarse factors, 20,480 updates, 40,960 presentations per factor, fixed VLB/schedule; no extra objective/architecture sweep. |
| Full-size engineering and actual restart | SMOKE passed/full_size/within_ceiling=true, forecast86.241GPUh. RESTART_TEST exact=true, actual SIGUSR1 at4, exit75, three fresh processes, resume to12. | Not just an in-memory replay. Four production wall-time handoffs retain receipts and resume exact draw IDs. |
| New-model sampler convergence | SAMPLER_GATE has all16 passed records and 48 bound input chunks; includes density, tidal/gap widths and D-wide metrics at250/500/1000. | Registered numerical screens passed; no claim of statistical calibration from this alone. |
| Complete checkpoint/anchor/draw/control panel | DRAW_LEDGER_INTEGRITY and DRAW_INTEGRITY pass:688 cases,33,280fine,7,872unique coarse,5,144 actual payloads,34,634,901,688bytes hashed. | No reduction of the approved panel or substitution of preview results. |
| Actual frozen checkpoint/source verification | CHECKPOINT_INTEGRITY verifies30 payloads (3,037,247,418bytes),745 frozen source files and14 data receipts. | Payload evidence, not filenames/completion flags alone. |
| Full metrics: power, mean/residual, correlation, one-point, mass, calibration, tides/gaps and closures | 688 case JSONs plus REGISTERED_DIAGNOSTIC_SUMMARY; case auditor checks six-feature ranks, all C50/68/90/95 levels, observed/unobserved decomposition, matched/physical closures and finite-M spectral identities. Reviewed report implementation. | Complete registered outputs. Physical versus matched-patch targets remain distinct; A--C wide matter is N/A. |
| Conditional diagnostics and spatial uncertainty | RESULTS contains288 rows over six inference-time observation variables with train-only edges;12 contrasts include500/1000Mpc/h source-block intervals. | Appropriate descriptive fixed-phase analysis, not independent-voxel SBC or cosmological population uncertainty. |
| Predeclared H1/H2 decisions | RESULTS: H1 gain10.875891%,passed=true; B:C gain-3.374494%,passed=false; C:D tidal gain3.535867%,passed=false. Source enforces pooled10%, all-cell direction/nonregression and D dependence checks. | No retrospective gate loosening or validation-selected checkpoint. |
| Final numerical reconciliation | CLOSEOUT_AUDIT passed;688case identities,384preview parity checks and32 progression cells recomputed. CASE_REPORT_AUDIT passed on58526345/step0. | Auditors' code inspected: checks match their stated scope, not an assumed universal correctness certificate. |
| Figures and bindings | All3 repository PNG hashes match FIGURES; all3 visually inspected in this continuation. RESULTS hash independently recomputed. | Readable checkpoint, field-statistics and fixed-anchor draw/mean/spread figures. Not visual proof of calibration. |
| Resources and terminal jobs | Live sacct:58524030,58524609,58526345 COMPLETED0:0 (988/2896/12seconds); allocation status has zero pending/running jobs. Supplemental ACCOUNTING includes its12seconds exactly once. | 78.0075GPUh,2.536111CPU-nodeh,122,030,674,932bytes (113.64992GiB),8GPUrequests; within112/8/300/48h caps. No new compute. |
| Literature/data-specific joint go/stop and CFM | Committed joint-decision note and Gemini primary-paper audit, with actual data-inventory limitations and corrected deterministic-oracle caveat. | Complete review; recommends one bounded coupled-state VDM/CFM comparison, not its automatic implementation. |
| Final science report, science log and field-model plan | At this audit, docs/e2e_vdm_context_results_20260918.md still says draft13:29; SCIENCE_LOG and docs/plan_end_to_end_conditional_field_posterior.md still lead with running/preview states. | **Incomplete written closeout.** Login21 owns edits per closeout_handoff; re-read the final artifacts before goal completion. |

## Verified hashes and interpretation caveat

RESULTS SHA256:
`2ca834400a766de7d6c357679f823fc9ee848f12b40ce7a562949a8186c3303f`.
REGISTERED_DIAGNOSTIC_SUMMARY SHA256:
`0d025f3654717fa0ecfba1e8f94edc01ce2b5c4afaaeb5a93fc4ccd81a0f5792`.

Do not count the frozen EXPERIMENT_COMPLETE CPU total2.5327778 as final after
the supplementary audit. The separate ACCOUNTING.json preserves that original
receipt and adds0.0033333CPU-nodeh, yielding2.5361111. Step4's cancellation on
normal report-allocation release is not an additional failed scientific run;
the JSON-only audit completed in the supplemental allocation.

The oracle regional means and density component of adjacent-core differences
are fixed by exact coarse mass. Actual case JSON shows bias/spread about1e-15
and density-difference CRPS about1e-16. The raw pooled68.75% oracle coverage
mixes deterministic and stochastic quantities and cannot be compared wholesale
to an88.24% continuous finite-ensemble target. The fixed-parent density mean
also lacks stochastic spread. Preserve frozen raw outputs; interpret the
nondegenerate tidal/gap quantities and proper scores separately. Both literature
notes have been corrected, and the user was informed.

## Remaining completion action

Inspect the completed report, its D/coarse/oracle/conditional/tidal results,
SCIENCE_LOG, field-model plan and the contract's final status once the designated
owner publishes them. Confirm they link both literature notes, retain all claim
boundaries and contain one justified next step rather than an automatic run.
Then update this verdict and mark the goal complete. Do not launch or rerun
science merely because the writing is unfinished.
