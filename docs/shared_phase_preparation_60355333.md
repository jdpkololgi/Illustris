# Shared-phase preparation, allocation 60355333

The approved four-node CPU allocation reached its one-hour limit. Completed products and partial outputs are preserved. This is not a training-ready release. R7 and the versioned 20/3/2 policy are unchanged.

## Verified work

- ph014–ph017: new 2048³, 16-rank R7 T-Web products. ph001: qualified reuse of its historical native-density and R7 products. No redundant particle redeposition.
- Eighteen corrected P12 phase inputs staged with private metadata and immutable payload references; metadata lineage verified, not a fresh full-payload rescan.
- Four missing CFM native-density sources qualified and adopted. Four target-free observed catalogues and both-cap observation grids pass. Existing angular counts reused for ph001/005/006; ph004 random maps rebuilt.
- New qualification, labels and downstream stages are listed below. Native-density CFM targets use their own spectral operator; per-galaxy T-Web fields are not substituted.

The machine-readable [receipt inventory](shared_phase_preparation_60355333.json) binds existing metadata by SHA256. It distinguishes missing stages; scheduler completion alone is not a scientific pass.

## P12 stage inventory

| Phase | R7 field | Label certificate | P1 catalogue | P2 | P3 | P4 | Full phase |
|---|---|---|---|---|---|---|---|
| ph001 | saved | saved | pending | pending | pending | pending | pending |
| ph014 | saved | saved | pending | pending | pending | pending | pending |
| ph015 | saved | saved | pending | pending | pending | pending | pending |
| ph016 | saved | saved | saved | saved | saved | saved | saved |
| ph017 | saved | pending | pending | pending | pending | pending | pending |

## CFM stage inventory

| Phase | Native source | Cap grids | Geometry | Density target stage | Tensor stages | Full target |
|---|---|---|---|---|---|---|
| ph001 | saved | saved | pending | pending | 0/6 | pending |
| ph004 | saved | saved | saved | pending | 0/6 | pending |
| ph005 | saved | saved | saved | saved | 4/6 | pending |
| ph006 | saved | saved | saved | saved | 6/6 | saved |

## Qualification and operational limits

ph017 native unresolved-central slab-33 sample has 135/136 matching voxels, below the existing 0.995 per-group threshold. The one mismatch is original row 64026976, native particle row 155985331. Coordinate residual is 8.3957e-05 Mpc/h; reconstructed x is 7.6105e-07 Mpc/h from a voxel face, so the native and reconstructed coordinates occupy adjacent x cells. The group RMS eigenvalue difference is 0.001461, below 0.01, and its web class is unchanged. Coordinate/identity checks passed. The existing field gate was not weakened and no label certificate was issued. The original report and [diagnostic script](shared_phase_ph017_boundary_audit_60355333.py) are retained.

The initial ph001/ph014/ph015 annotation streams used memory-mapped field sampling and did not complete within the allocation. An optional resident-field mode preserves sampling arithmetic exactly; 10,000-row parity passed, with 153.4 s loading and 0.0753 s resident annotation. Its 111,669,149,696-byte field fits these allocated nodes. ph016 completed all 10,518,291 parent labels in that mode and joined 7,323,814 observed rows. Attempts to cancel only steps 4 and 6 for faster restarts were rejected by automatic approval review; the user subsequently approved both restarts at 06:34 UTC. Less than a minute remained, so no restart or cancellation was issued. Partial files are preserved. A new two-node, two-hour CPU window has been requested but is not yet approved.

Ten contract/source tests and fifteen annotation tests passed in step 60355333.36. Earlier 46-check migration regression evidence remains in the implementation handoff. Graph refresh failed with filesystem lock error 524 in the prior smoke allocation; no bypass was used.

## Remaining preparation

- Finish the incomplete newly admitted P12 catalogues; resolve ph017 qualification, then graph/field/patch validation and response inputs.
- Complete/adopt all CFM target and observation products under the new authority, including the larger training geometry for ph014–017 and whole-field interfaces.
- Refit the full-20 and twenty 19+1 OOF P12 transforms, the equal-20-phase CFM normalizer, and finish loader/runtime binding checks. Old transforms are not relabelled as new fits.
- No new learned-model fits, GPU work, final-test payload access or production release occurred. Further scheduler work needs a new explicit resource budget once this allocation ends.

Historical products and comparison evidence remain intact. New training exposure changes the interpretation of future generalization comparisons, not the validity of the historical records.
