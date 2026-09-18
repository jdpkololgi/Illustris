# Controlled VDM experiment: durable closeout evidence

Date: 2026-09-18. The approved A/B/C/D experiment is complete, **not a
production field-posterior release**. Scientific interpretation:
[final report](../../../e2e_vdm_context_results_20260918.md).
The original requested deliverables are mapped in the
[independent requirements audit](completion_requirements_audit.md).

## Provenance and artifact scope

Original run root (all relative Scratch paths below):
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_pipeline_v1/vdm_context_20260917_v1`.
Frozen scientific source: `16398511d4552bd041e8d1e97cebd3e168088064`.
The source snapshot, models, data, arrays and full case reports remain there.
Scratch is not backed up. This directory archives small summaries, receipts
and figures unchanged; it is not a backup of the checkpoints or training data.
No predictive data were uploaded to a third-party service.

| Repository copy | Exact original | Purpose |
| --- | --- | --- |
| [RESULTS.json](RESULTS.json) | analysis/RESULTS.json | All registered contrasts, progression, observational strata, pair metrics and case hashes |
| [REPORT.md](REPORT.md) | analysis/REPORT.md | Frozen machine-generated report, distinct from the explanatory scientific report |
| [FIGURES.json](FIGURES.json) | analysis/FIGURES.json | Binds RESULTS to REPORT and all three PNGs |
| [checkpoint_progression.png](checkpoint_progression.png), [field_statistics.png](field_statistics.png), [posterior_fields.png](posterior_fields.png) | analysis/same filename | Visually inspected figures, copied without edits |
| [MANIFEST.json](MANIFEST.json), [MODELS_FROZEN.json](MODELS_FROZEN.json) | root/same filename | Frozen contract/source/model choices |
| [SMOKE.json](SMOKE.json), [RESTART_TEST.json](RESTART_TEST.json) | root/same filename | Full-size inference/parity and actual signal/fresh-process restart tests |
| [CHECKPOINT_INTEGRITY.json](CHECKPOINT_INTEGRITY.json) | analysis/same filename | All thirty actual checkpoint payloads, source and data bindings |
| [DRAW_INTEGRITY.json](DRAW_INTEGRITY.json) | analysis/same filename | Actual hashes of all 5,144 fine/coarse array payloads and shared-parent bindings |
| [DRAW_LEDGER_INTEGRITY.json](DRAW_LEDGER_INTEGRITY.json) | analysis/same filename | Exact 688-case task ledger, paired noise and draw identities |
| [CASE_REPORT_AUDIT.json](CASE_REPORT_AUDIT.json) | analysis/same filename | All registered case metric contracts and preview consistency |
| [CLOSEOUT_AUDIT.json](CLOSEOUT_AUDIT.json) | analysis/same filename | 688 cases, 384 preview comparisons, 32 independent progression recomputations |
| [EXPERIMENT_COMPLETE.json](EXPERIMENT_COMPLETE.json) | root/same filename | Original complete=true / production_ready=false receipt; not rewritten after supplemental audit |
| [FINAL_RESOURCE_ACCOUNTING.json](FINAL_RESOURCE_ACCOUNTING.json) | analysis/closeout_resources/ACCOUNTING.json | Final eighteen-job resource total, including supplemental audit exactly once |

The original 688 `analysis/cases/*.json` and
`analysis/REGISTERED_DIAGNOSTIC_SUMMARY.json` retain all registered field,
one-point, spectral, physical/matched tidal, eigengap, rank, interval-width,
conditional, coarse and spatial diagnostics. The latter summarizes existing
metrics only; it is not a post-result tuning experiment. Its SHA256 is
`0d025f3654717fa0ecfba1e8f94edc01ce2b5c4afaaeb5a93fc4ccd81a0f5792`.

Original `analysis/SAMPLER_GATE.json`, `data/GEOMETRY.json`, normalization,
physical v1/v2 and representation-release receipts remain in the Scratch
tree, alongside all allocation requests, logs and expected pause receipts.
The v1 physical failure is preserved; the separately approved v2 correction
passed before training. No failed result was overwritten or silently excluded.

## Key immutable hashes

| Artifact | SHA256 |
| --- | --- |
| RESULTS | `2ca834400a766de7d6c357679f823fc9ee848f12b40ce7a562949a8186c3303f` |
| MANIFEST | `67e1f9d0edc73b58c761e35f1f2dc13d4fe102e27f260104ecdb87b725a9bf10` |
| MODELS_FROZEN | `04abc713e9a1baa87288640ac10672b318ee7fd218ebcb2229eb78323bd3ea57` |
| FIGURES | `ad912f178a78a34bf6cb941d7fe5d554a39d8d7079bf97c63cdf3a7c5dfa50bc` |
| CHECKPOINT_INTEGRITY | `cd13431a793aad824a54df8a54642c6a852aff5b2771a52229d370c856a89ac0` |
| DRAW_INTEGRITY | `0474b6596ff1da135789fef63ae19cd6ff58553e28cd7f32fd1a6418c7cac88b` |
| CASE_REPORT_AUDIT | `8d2c79afd8fb61458efbdaf90ee34bacfbef3a1ee7ffbab08404a81464bfa8ca` |
| CLOSEOUT_AUDIT | `9619b580928aba23be57619b715d19bb3613b562b9a409bd0b794983a1026fa9` |
| EXPERIMENT_COMPLETE | `c0d90f6f8d6540a7216e729d3dca3114c2592d584c55447476fa1628480f50be` |
| FINAL_RESOURCE_ACCOUNTING | `bb203422a1d7121b15e6c7ed793fa70ead03e8b0781c7bf841b5a5d019a3c39e` |

The three image hashes are in FIGURES.json and were verified against both
their Scratch originals and these copies. The evidence scripts here are
bounded read-only verifiers except their exclusive new audit receipts;
`closeout_compute.py` is the completed one-shot recovery wrapper, not a new
training controller. Do not rerun these merely to display the report.

## Terminal accounting and result boundaries

Training58473428 completed0:0, 11,886 seconds. Final GPU58524030 completed0:0,
988 seconds; CPU full report58524609 completed0:0, 2,896 seconds. An optional
audit step was terminated by natural release of the report allocation before
publishing results. One JSON-only recovery58526345 completed0:0 in12 seconds
under a15-minute cap. No scientific fits, fields or samples were regenerated.
All four earlier GPU status75 exits are registered clean wall-time pauses.

Final allocated totals: **78.0075 GPUh / 2.536111111 CPU-nodeh**,
**122,030,674,932 bytes (113.64992 GiB)** at resource closeout,
**23.0253 hours** from first allocation. Approved caps were112GPUh,
8CPU-nodeh,300GiB and48hours. Eight GPU requests were used; at most two
interactive allocations concurrently. The full GPU nodes used four workers.
No job is pending or running at final inspection. Do not double-count steps
sharing an allocation or the already-included preview51seconds.

H1 passes; both H2 contrasts are not established. Small marginal gains and
stable samplers do not establish joint calibration. Both true-coarse oracle
and fixed-mean controls contain deterministic mass quantities: their generic
coverage outputs must not be compared wholesale to continuous finite-M
targets. The [joint-field/CFM/data review](../../../e2e_vdm_context_joint_decision_literature_20260918.md)
and [Gemini literature audit](../../../e2e_gemini_literature_audit_20260918.md)
inform one bounded next-protocol recommendation, not an automatic run.

Verification before closeout includes the recorded54 focused/regression
tests, full-size GPU smoke, actual SIGUSR1/restart, sixteen new-model sampler
screens, actual checkpoint/draw hashes, complete metric/reconciliation audits,
visual figure inspection, archived-byte parity, local documentation links,
Git whitespace checks and terminal scheduler/resource review. Documentation
completion adds no new scientific metric or acceptance criterion.

Final documentation validation passed: all17 archived files byte-match their
originals; four REPORT/PNG bindings match FIGURES;33 local report/evidence
links and the new log/plan links resolve; result decisions and eighteen-job
resource counts match; `git diff --check` passes. One sampler-gate link was
corrected to analysis/SAMPLER_GATE.json before the successful final check.
