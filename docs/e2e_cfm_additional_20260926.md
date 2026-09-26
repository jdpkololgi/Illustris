# Observation-regime diagnostic and additional-phase replication

User requests matched observation-regime checks followed by further non-training
phases. Freeze mixed coarse13312/fine26624, EMA, seeds17/29, existing normalization,
physical operators and scores. No changes are selected using these new targets.

## Exposed phase result

Saved-score diagnostic in `e2e_cfm_regime_check_20260926.json` verifies all392
upstream receipts. Every phase012–015 has the SAME16 combinations of
NGC/SGC x radial shells0–3 x boundary/interior, evaluated at both seeds.
There is no composition difference in these strata to reweight away.
The013 minus012 fine-region coverage contrast is+24.22pp(NGC),+8.59pp(SGC),
+17.19pp(boundary),+15.63pp(interior); by radial shell it is
+29.69,+7.81,+34.38,-6.25pp. Thus the contrast survives both cap and support
splits, but is not universal across shells. Other phases have different shell
patterns. Broad regime composition does not explain the difference; continuous
counts/exposure and cosmic structure can still differ within these strata.
This is descriptive stratification, not causal attribution or an independent
binomial significance test: regions overlap, seeds repeat the same truth, and
only four phases are exposed. No global variance correction is justified.

## Approved prospective panel

Use ph016 and017 only:16prepared anchor pairs each, two seeds,32draws per pair,
128NFE per factor; first pair per phase/seed receives paired8draw256NFE refinement.
Total2048main draws plus32refinement draws. Ph018/019 stay untouched. Prepared
historical13/2/6 roles remain immutable; runtime release is specific to this
candidate evaluation. No training-phase change. New phase outcomes are not used
to tune this candidate; if subsequently used for tuning they become development.

Four workers (phase x seed) use all four GPUs on one interactive node, max2.5h,
10GPUh. Previous4phase assessment took3h33m on4GPUs; this has half as many cases.
Restartable chunks and source snapshot under tmux, no automatic allocation renewal.
Existing paired coarse-draw reference checks remain for012–015. For016–017 no
previous13312draws exist: instead retain exact first-chunk replay, checkpoint
binding, positivity/block closure, target receipts and sampler refinement.

Report each phase and seed, core/block/fine mass coverage and error/spread,
CRPS, marginal tidal/eigengap coverage, spectra, Brier and joint scores through
the unchanged score payload. Use finite-draw benchmark29/33, not nominal.90.
No new hard pass threshold selected after seeing phases. Conclude whether the
012/013 heterogeneity replicates, not whether any single phase equals87.88%.
Do not promote to DESI-ready or infer exact class-probability accuracy from
one truth per observation. Ph016/017 are non-training mock replications, not
a real-survey robustness test.

Verification:14focused unit tests passed in0.746s (routing, frozen phase scope,
sampler/physics controls, complete/incomplete report collector); shell syntax
and git diff checks passed. Graphify refresh exceeded the five-second login
cap; index refresh incomplete.

Launch2026-09-26: user approved the specific allocation and panel. Job58920768
granted on nid001252, four GPUs,2h30m cap, tmux `cfm_additional_20260926` on
login33. Immutable source snapshot fromdc90e51bfcbd79d9f217b4ff744665f81e62938a.
Output/log root:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_additional_20260926_v1`.
This is launch evidence, not scientific completion. Once targets are scored,
016/017 are exposed replication phases;018/019 remain sealed.
