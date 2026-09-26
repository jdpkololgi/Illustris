# Frozen mixed-checkpoint compatibility test

User authorizes coarse13312 + fine26624 against saved13312/13312 and26624/26624.
EMA weights, seeds17/29; all64prepared development/replication pairs012-015.
No parameter changes, training, classical replacement or016-019access.

4096new main draws (64pairs x2seeds x32),128NFE per factor, microbatch2.
64additional refinement draws at256NFE on the same first-pair/8draw subsets.
Same addressed RNG and observations as saved ensembles. Each new coarse chunk
is compared with its checksum-verified13312reference (rtol2e-6, atol1e-8).
Initial chunk also has exact in-process replay. Model loader verifies distinct
coarse/fine checkpoint identities, EMA states, normalization and architecture.

Use existing physical operator, training-only feature scales, scores, coverage,
widths, spectral bands and joint-dependence diagnostics without retuning.
Collector checks target hashes/32draws and coarse-controlled mass equality.
Core/block mass agreement with13312is expected ALGEBRAICALLY; it is not new
evidence of learned calibration. The open question is whether fine-sensitive
density/power/tidal/eigengap/dependence/offset-region scores gain the late fine
model's benefits without new failures from its sampled coarse conditions.
The fine model was trained on true coarse fields: compatibility not assumed.

Report by phase and seed, including sampler refinement. Do not promote on pooled
coverage or power alone. Four phases already exposed; remaining confirmation
016-019requires a later explicit release and frozen selection decision.

One four-GPU/four-hour allocation under fixed tmux launcher; prior same-size
evaluation~3h33m. Atomic per-chunk restart, no automatic renewal. No old draw
files overwritten. Root: /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_mixed_20260926_v1.
Four workers partition seeds and two-phase panels; final COMPLETE.json and
SUMMARY.md compare all three configurations, with hashes of source summaries.

Launch:58902807 on nid001160, four GPUs/four-hour cap, frozen2037ca4,
tmux `cfm_mixed_20260926` on login33. Graphify update exceeded five-second
login cap; index refresh incomplete. Initial launch, not completion evidence.
