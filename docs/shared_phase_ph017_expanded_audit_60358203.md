# ph017 fixed expanded coordinate audit

Before execution: use 96,000 equally spaced original rows (three times the initial 32,000), the same native slabs 0 and 33, identical mapping and field-sampling code, and unchanged 0.995 per-group voxel and eigenvalue-RMS criteria. This is one bounded sample-size sensitivity audit, not an iterative sample search. Preserve the initial failed audit and its diagnosed boundary row. Do not certify from this plan alone; inspect all groups and retain both outcomes.

Motivation: the initial failing unresolved-central slab-33 group contains 136 rows and one 8.3957e-05 Mpc/h coordinate residual that crosses a cell face. Its 135/136 voxel agreement fails the gate, despite passing eigenvalue RMS and unchanged web class. More deterministic samples assess the rate without removing that example or relaxing the gate. The larger evenly spaced sample is not assumed to contain all original rows; the original audit remains separate evidence.

## Outcome

All four native groups passed unchanged gates: resolved/unresolved centrals in slab 0 had 1,587/393 rows; slab 33 had 1,623/407 rows. Each had exact voxel agreement and zero sampled eigenvalue differences. The original 32,000-row audit and its one diagnosed boundary mismatch remain separate evidence. The 130-row annotation smoke passed exact metadata and eigenvalue checks, including nine unresolved satellites. Certification uses the fixed expanded audit. A fresh migration invocation stopped on an existing verified parent; the hash-checking resume path was used without deleting or overwriting that parent.
