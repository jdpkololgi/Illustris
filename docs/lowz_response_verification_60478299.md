# Low-z payload verification — allocation 60478299

## Result

The user-approved verification-only continuation completed on `nid200063`.
Allocation 60478299 and step 60478299.0 both completed with exit `0:0`;
allocation elapsed time was 00:03:01. The allocation released automatically.
The approved ceiling was one CPU node, 64 CPUs, desi/interactive, Scratch,
20 minutes, no rebuilds or automatic retries and no sealed-phase access.

- CFM: all 46 response HDF5 payload hashes and 23 angular-source hashes match
  their saved receipts. Independent scalar support/selection, expected-count,
  line-of-sight and nonnegative-count checks pass at 1,104 sample locations
  (989 random-supported). This is sampled identity verification, not an
  exhaustive numerical replay of every voxel.
- P12-A: both geometry arrays per phase were independently rehashed for all
  23 phases. The combined inventory covers 168,860,936 parent rows and
  167,459,101 supported parents. This does not independently qualify spatial
  folds, selection channels or all raw-grid payloads.
- The full/20-OOF selection phase lists and saved terminal receipt bindings
  pass the combined inventory checks. `summary_metadata_only=false` and
  `model_inputs_ready=false` are retained.

The existing `workflows.sbi.audit_cfm_lowz_survey.payload_audit` ran first,
followed by `workflows.sbi.summarize_lowz_response_preparation.summarize`
without metadata-only mode. No source implementation changed. The workload
was launched directly through `salloc ... srun ... timeout 1150 ... python`,
not an idle allocation shell. Isolated `cosmic_env` was used with one BLAS/OMP
thread. The saved chat's sandbox startup still failed; explicitly approved
native execution worked, with Shifter configuration ownership verified as
0:0. No permission settings, security hooks or scheduler rules were changed.

## Evidence

- Combined receipt: `docs/lowz_response_verification_60478299.json`.
- Scheduler receipt: `docs/lowz_response_verification_60478299_scheduler.txt`.
- Log: `/pscratch/sd/d/dkololgi/abacus/lowz_verify_60478299.log`.
- Stderr: `/pscratch/sd/d/dkololgi/abacus/lowz_verify_60478299.err` (empty).
- CFM audit: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_lowz_selection_20261029_v1/PAYLOAD_AUDIT.json`.

## Boundaries unchanged

The prior native/half-cell low-z volume screen still fails: worst fractional
change 2.314%, versus the predeclared 1% screen. Payload integrity is not
volume convergence or scientific qualification. No fit, target rebuild,
normalizer, model training or readiness promotion occurred.

P12 radial selection, full/OOF transforms and qualified fold/loader integration
remain unfinished. CFM selection/volume qualification, observer-crossing
domain allocation, matched targets and new normalizers/loaders remain.
Historical preparation and interruption evidence is preserved in
`docs/lowz_response_60477149.md`. The previous interrupted payload audit is now
resolved by this separately authorized continuation; the original metadata-only
inventory remains unchanged as historical evidence.
