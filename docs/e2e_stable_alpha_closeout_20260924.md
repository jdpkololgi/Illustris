# Stabilized alpha sweep closeout

All16fits reached98,304updates in job58820323. That allocation timed out in
evaluation, not training:382/384 ensembles had completed. On user approval,
the frozen worker3 replay resumed without training in idle GPU58824466 and
completed the missing two evaluations and final collector, exit0. The original
LAUNCH_EXIT_CODE=143 is retained as history; COMPLETE.json now contains384rows.

Selected stochastic EMA alpha=.30 under the specified both-checkpoint minimax
rule. At98,304 mean=.08128,16-probe covariance=.08150, DC=1.02493,
top=1.09085,7/8full-gate passes. At81,920 only3/8pass. Its final failing cell
is seed17/case0 top1.11734. No stable all-cell/both-checkpoint pass is claimed.
Increasing alpha to.35/.40 improves top to1.05274/1.01538 but raises DC to
1.10956/1.11933. Exact alpha.35passes8/8at final but7/8earlier. Final exact
alpha.40is7/8, not the incomplete interim7/7. Maximum NFE shell change=.0003168.

Opus's predicted useful shift in alpha is supported, but the DC trade-off has
not disappeared. No further optimizer/alpha tuning follows automatically.
User's revised stopping criterion is downstream relevance or a simpler equally
useful calibrated competitor, not mathematical perfection. Immediate comparison:
`e2e_classical_hybrid_20260924.md`. Existing gates remain diagnostic evidence;
physical DESI/T-Web tolerances are not inferred from this dimensionless toy.

Authoritative Scratch root:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/reference_stable_alpha_20260924_v1`.
Final COMPLETE.json/SUMMARY.md are archived under docs/evidence/e2e_stable_alpha_20260924.
