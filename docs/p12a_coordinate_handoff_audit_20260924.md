# P12-A coordinate handoff: initial source and recorded-product audit

2026-09-24. V0 work has begun. This is a read-only source/metadata finding,
not a numerical revalidation of Scratch arrays, checkpoint ancestry or targets.
No reserved payload, new inference, allocation or fit was used. Graphify could
not start in this session; targeted repository inspection was used instead.

## Established by the inspected records

| Component | Evidence | Finding |
|---|---|---|
| P10 observation geometry | `workflows/abacus_tweb/p10_build_phase_index.py`, `comoving_distance_mpc` / `cartesian_points` | Observed RA/DEC/Z are converted using interpolated Astropy Planck18 distances in Mpc. |
| Recorded phase products | `docs/evidence/p10/multiphase_p1_p4_complete_with_ph006_20260813.json` | The completed audit explicitly records `observer_frame_is_planck18_mpc=true`; this is not only an unused source default. |
| Physical target contract | `configs/p10_phase_registry_v1.json` | Native c000 z0.2 matter, 2000 Mpc/h box, 2048 grid, R7, ordered eigenvalues. |
| Observed-to-parent truth join | `workflows/abacus_tweb/p10_build_observed_truth.py` | Uses TARGETID-indexed parent labels, checks sky identity, and copies host keys and eigenvalues. It does not infer a truth voxel from observed redshift here. |
| Available native annotation path | `workflows/abacus_tweb/annotate_cutsky_with_tweb_eigs.py` | Resolves FILE_NUM/HALO_INDEX into native host positions and T-Web voxels; actual run receipt/position field still needs binding for each P12 parent. |
| P12 conditioning | `workflows/sbi/p12_prepare_base_response_dataset.py` | Consumes OOF predictions/truth and response; seven features are three eigenvalue predictions, redshift, log ntilde(Mpc^-3), cap, log1p support distance(Mpc). It does not recompute physical truth coordinates. |
| E2E discrepancy | `docs/e2e_coupled_distance_convention_audit_20260918.md` | Planck18 times h differed from native ph007 central positions (median .510020 Mpc/h); the pinned DESI/Abacus table reduced this to .00003617 Mpc/h. Those are existing E2E measurements, not new P12 measurements. |

Thus P12's recorded observation convention differs from the native distance
mapping identified by E2E. This does NOT alone establish erroneous P12 labels.
A consistently applied fiducial transformation of observed RA/DEC/Z can be an
admissible input representation for learning native host-label posteriors. The
E2E physical field/FFT alignment constraint is stronger than that input-only
choice. Actual source/checkpoint/array bindings and independent label closure
must settle which situation P12 occupies.

## Mandatory check before the DESI canary

Produce `P12A_COORDINATE_AUDIT.json` with source/manifest hashes, phase/row
identities, units, distance-table identity, provenance, measurements and one of
`consistent_fiducial`, `adapter_only_mismatch`, `training_product_error`, or
`unresolved`. All required checks must be evidenced; missing is not pass.

1. Bind actual full-fit and OOF checkpoint lineage to P1 points, P3 fields,
   support-distance/selection tables, normalization and parent target arrays.
   Audit ph000's imported lineage separately. Do not assume today's source
   generated a frozen artifact. Record both distance cosmology and h conversion.
2. On preselected training-phase rows spanning caps/shells/central/satellite
   categories, reproduce stored observer points from RA/DEC/observed Z. Verify
   counts, random masks, interpolation and distances use the same convention.
3. Independently follow TARGETID to FILE_NUM/HALO_INDEX and the native position
   field actually specified by annotation receipts. Resample R7 native truth at
   that voxel and compare with stored labels. Treat unresolved-host populations
   separately; a central-only test does not validate all galaxy rows.
4. Use Z_COSMO (not RSD Z) for the native geometric closure. Preserve the
   distinction between x_com and x_L2com. Do not classify physical RSD or
   satellite offsets as a distance-convention error.
5. On identical licensed rows, compare the old input convention with the pinned
   native table, reporting shell-wise offsets, cell/interpolation changes,
   expected-count volume effects and label changes where relevant. Do not treat
   the old E2E .51 Mpc/h median as a P12 result or a calibrated tolerance.
6. Freeze numerical tolerances from precision/interpolation and existing target
   contracts before these comparisons; freeze scientific effect thresholds
   before posterior scoring. Record formulas as well as measured discrepancies.
7. Check Loa adapter replay of the chosen frozen convention. If changing radial
   coordinates, conserve dN/dz/dOmega and transform ntilde with the coordinate
   volume Jacobian; update boundary distances/grid support consistently.

## Correction/retraining decision

| Outcome | Required action |
|---|---|
| Consistent fiducial coordinates, independently correct native labels, identical mock/Loa processing | Preserve frozen P12; document the input convention. Native-distance difference alone is not a retraining trigger. |
| Error only in the new Loa/export adapter | Correct adapter to frozen training semantics, replay golden mock and repeat parity. No automatic model retraining. |
| Corrupt/misaligned training labels, mixed input/response frames, or a required scientific change to training geometry | Stop affected inference/release. Build immutable corrected products and affected transforms; retrain affected encoder/full-fit and OOF models, regenerate summaries, refit FMPE and validate on fresh confirmation. Do not patch predictions or temperature-scale away the error. |
| Unresolved provenance or physical closure | Keep canary gate closed; continue metadata/adapter work that does not assume the disputed convention. |

No outcome has yet been selected. In particular, the initial audit does not
declare P12-A invalid or clear it for deployment.

## Handoff inventory to complete next

- Freeze production/OOF encoder and FMPE paths/hashes, solver, transforms and
  original audit bindings; full artifact manifest still pending.
- Build phase exposure ledger before choosing extra mock evaluation rows:
  P12 train 000/002-005, selection 006, already-opened blind 001;
  E2E train 000/002/003/007-011/020-024, development 012/013,
  reserved confirmation 014-019. These are recorded roles, not a completed
  audit of every later payload access.
- Implement bounded compute audit only after inputs, numerical criteria and
  resource request are concrete. Existing Slurm and phase-access rules apply.
