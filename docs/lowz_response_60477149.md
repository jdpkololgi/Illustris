# Low-z survey/response preparation — allocation 60477149

Follow-up: the interrupted payload audit described below was completed by the
separately approved allocation 60478299. All CFM response hashes and sampled
identities, plus all P12 geometry-array hashes, pass. See
`docs/lowz_response_verification_60478299.md`. The volume-screen failure and
training-readiness blockers remain; this original record is preserved.

## Authority and fixed scope

On 29 October 2026 the user explicitly approved including **both P12-A and CFM**
in one CPU-node, 64-CPU, desi/interactive, Scratch, at most two-hour window.
No automatic retries, no ph018/ph019 access, release afterward. No GPU or learned
training is authorized by this preparation. R7, source snapshot z=0.2, and the
shared 20-training/3-development/2-final-test roles are unchanged. Science is
0.01 <= z < 0.55, positive-z context to0.60, sentinel0.585–0.595 excluded.

Ordinary sandboxed execution still failed before startup with bwrap mkdir
permission errors. Explicitly approved host executions succeeded; no settings,
security hooks or scheduler approval rules were changed. Allocation60477149 was
granted on nid200034. The first failed salloc tool invocation did not execute
the scheduler client or create an allocation.

## Preparation sources and products

- CFM entrypoint: `workflows.sbi.cfm_lowz_selection_response` and bounded runner
  `workflows.sbi.run_cfm_lowz_response_preparation`.
- CFM root: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_lowz_selection_20261029_v1`.
  Runner receipt/log directory: `run_60477149`. Intended stages: 23 native
  statistics, one full and20 leave-one-training-phase-out selection fits,
  23 full-selection response pairs. These OOF selection files are NOT complete
  P12 encoder contracts or OOF response-grid campaigns.
- P12 raw runner: `workflows.sbi.prepare_p12a_lowz_campaign_raw`, root
  `/pscratch/sd/d/dkololgi/abacus/p12a_lowz_20261029_v1`.
  Existing ph001/ph006 receipts, manifests and count-field payloads are checked;
  all other permitted phases are created once. Partial output fails closed.
- P12 geometry-only writer: `workflows.sbi.prepare_p12a_lowz_geometry_response`.
  Reuses exact `p12a_parent_response.sample_centres` for the two posterior survey
  features: random support and support-boundary distance. Its receipt is separate
  from the old selection-bound READY marker and explicitly has
  `selection_fitted=false`, `model_inputs_ready=false`.

CFM uses native Mpc/h; P12 uses observer-frame Planck18 Mpc. Neither selection
is interchangeable with the other. Historical outputs remain unchanged.

## Verification and launch

33 focused tests pass in isolated cosmic_env: low-z selection/response,
input contract, raw preparation, campaign phase guards and geometry serialization.
All tests use synthetic data. `git diff --check` passes.

The ph001 native CFM statistics step completed exit0 before expansion. Both
caps have positive effective volume and supported CIC count weight in every
science bin. Minimum effective volume22427.62946(Mpc/h)^3 and count155.37797.
SGC/NGC CIC input counts2142015/5185667, deposited-weight loss exactly0.
Unsupported deposited weights22419.03178/53974.53715 are reported separately.
No closure or convergence claim follows from these checks.

The main CFM and P12 raw steps each reserve32CPUs under the same allocation;
CFM uses4 workers, P12 uses2, OMP4 and OPENBLAS1. CFM timeout6000s/internal5700s;
P12 timeout3000s/internal2700s. Overall allocation remains the hard2h bound.
Top-level logs are `/pscratch/sd/d/dkololgi/abacus/cfm_lowz_60477149_campaign.log`
and `p12a_lowz_60477149_raw.log`, with adjacent `.err` files. Canary logs are
`cfm_lowz_60477149_ph001.log/.err` in the same directory.

## Current status and qualification boundaries

All deterministic builds completed: 23 CFM statistics, full+20 OOF selections,
23 CFM full-response pairs (46 HDF5), 23 P12 raw phases and 23 geometry-only
response arrays. P12 has 168,675,903 supervised rows (25,209,536 below z0.1).
Of 168,860,936 parent rows, 167,459,101 are random-supported. The geometry writer
explicitly excludes 9,469 non-context rows before sampling; radius alone cannot
recover a negative observed redshift. No truth-validity cut defines this mask.

Both volume audits completed; ph001 worst-bin native/refined differences are
2.31424% SGC and 1.87216% NGC, ph006 2.31414%/1.87187%. All four fail the fixed
1% numerical screen. The independent closure and response-arithmetic audit
passes its numerical consistency checks. Largest grouped expected/observed
residuals occur at0.01–0.03 in SGC: +76.4503% for OOF ph008, -24.1651% for
development ph006. These are phase diagnostics with cosmic-variance contributions,
not calibration scores or a license to tune away held-out density fluctuations.

The owning interactive shell auto-logged out after 30m18s, relinquishing the
allocation. This interrupted the final CFM HDF5 rehash/scalar-replay audit after
18s (CANCELLED0:15). All data, geometry, volume and closure steps completed0:0.
Scheduler allocation COMPLETED does not imply that final audit completed.
No replacement allocation or automatic retry was launched. Independent CFM
payload verification and independent P12 array rehash remain incomplete; the
writers' checksums and completion receipts are preserved. Final inventory
`docs/lowz_response_60477149.json` checks bounded JSON receipts only and says so.

The approved local AST graph refresh failed before rebuilding, with existing
`fcntl.flock` OSError524. No security/locking bypass or global-graph refresh was
performed. Scheduler evidence: `docs/lowz_response_60477149_scheduler.txt`.

Before inspecting refinement results, the bounded numerical audit is fixed to
ph001/ph006, both caps,0.01–0.15 at delta-z0.005. It integrates native and
nested half-cell volumes at fixed physical apodization. A1% per-bin fractional
change is a numerical screening criterion, not a posterior-calibration threshold.
Native replay must match the saved statistics to1e-9 relative/1e-6 absolute.
This does not establish full0.01–0.55 convergence or validate the radial smoothing
prescription. Closure diagnostics use the omitted-phase selection for every
training phase and the full training-only selection for development phases;
development residuals are not used for tuning. Payload checks rehash each CFM
HDF5 and sample fixed angular directions at z0.02/0.10/0.40 to replay support,
expected-count and LOS identities independently of the grid writer.

Raw/support response products do not repair P12's invalid old duplicate/host-fold
rule. Distinct low-z galaxies must not be discarded. Full P12 selection,
full/OOF transforms, fold-qualified sampling and real loader validation remain
separate work. CFM still requires volume/selection sensitivity and independent
phase/cap closure, observer-crossing domain allocation, matched targets and new
normalizers/loaders. Pooled fitted closure is an identity, not validation.
