# Frozen-model phase replication: ph014 and ph015

User explicitly authorizes opening ph014/ph015 before further targeted model
diagnostics. These two phases are now replication/development evidence; future
tuning informed by them cannot count them as untouched confirmation. ph016-019
remain sealed (64 prepared pairs). Training remains the original13phases;
preparation metadata retain the historical13/2/6 split for source integrity.
The current scientific split is13training/4development-and-replication/4sealed.

Freeze EMA checkpoints13312/26624, seeds17/29. Evaluate all16pairs per phase,
32draws each,128NFE, plus paired8draw256NFE refinement on the first pair per
phase. Ledger4096main+64refinement samples. Identical metrics, addressed RNG,
microbatch2, normalization, train-only probe scales and physical operators.
No refitting, widening, tuning or model changes. Same coverage benchmark29/33.
Report each phase and seed; overlapping regions are not independent universes.

Question: does later-checkpoint regional undercoverage recur across independent
phases while density scores/power improve? Do not decide this from pooled voxel
coverage. Assess coverage, widths, bias and proper scores together. This run
does not certify real-DESI validity or tight calibration with four total phases.

Run one4GPU node for up to4h (16GPUh cap); prior matched workload3h33m. Existing
one-hour allocation is insufficient and left untouched. Fixed tmux launcher,
atomic chunks and no automatic renewal. Output root under e2e_field_v3:
`cfm_replication_20260925_v1`. Only new panel selection expands evaluator access;
default still deniesph014/ph015, replication deniesph016-019 and alltraining.
Historical checkpoint confirmation_access=false denotes training exposure,
not this newly authorized evaluation; explicit evaluation_phases/panel fields
in BINDING.json record current access. Summary is generated on full completion.

Launch receipt: job58871642, nid001105, four GPUs/four-hour cap; frozen
de379c3, tmux `cfm_replication_20260925` on login22. Seven tests passed on
compute (0.110s). Allocation and worker logs are in the output root. Existing
allocation58871404 untouched. Graphify refresh exceeded five-second login cap;
index refresh remains incomplete, not a numerical validation failure.
