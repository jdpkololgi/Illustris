# Development-only optimization stabilization

Scope: alpha=.25, AMORTISED Gaussian CFM only. Four65536-update parents
(seeds17/29 × stochastic/exact targets); branch each into constant LR3e-4 and
cosine decay3e-4→3e-6 over32768additional updates. Eight fits, identical parent
model/Adam/random states within each pair. Same objective, batch32 and bridge.
Both branches maintain an EMA shadow, initialized at the parent, beta=.999
per optimizer update (~693-update half-life). EMA has no gradient or influence
on the training trajectory. No retrospective reconstruction of an earlier EMA.

Evaluate RAW and EMA at81920and98304updates on FOUR DEVELOPMENT cases only:
2048draws,256NFE at both checkpoints;128NFE paired at98304. Same base-draw
seeds across raw/EMA/LR branches. No alpha tuning, no checkpoint selection,
no additional held-out evaluation. This is exploratory evidence, not a new
confirmatory qualification. Existing gates remain descriptive guardrails.

Prediction: decay and/or EMA reduce the coherent per-template DC offset while
leaving within-template observation scatter approximately unchanged. This
supports (does not uniquely prove) optimizer drift. Always show both, with
Monte Carlo terms. There are only TWO development observations per template:
scatter and offset estimates are weakly replicated; do not claim the mapping
is learned generally. For n cases, raw mean-square decomposes exactly into
offset² + centered variance (ddof0). Report MC-corrected components WITHOUT
clipping: subtract sum(mean-MC-variance)/n² from offset², and the remainder
of mean MC variance from scatter². These component corrections assume
independent ensemble-mean errors across cases; evaluation uses distinct seeds
per case, pairing noise only ACROSS model branches for the same case. Retain
sampling uncertainty and the small number of observations in interpretation.

Standard output per branch/checkpoint: all-shell power, mean,16-probe
covariance, variance and octant coverage; signed DC errors and MC SE; per-template
offset/scatter and offset share; bridge-aware absorption observed versus
predicted; Gaussian DC null; frozen exact-target velocity risk; paired NFE
changes. A stability flag requires ALL gates at BOTH checkpoints, all four
cases and both seeds for a given LR/weight-view/target law. No single-checkpoint
success is a reproducible pass. EMA is not exempt from this check.

Stop at98304 regardless of outcome. No automatic sweep, retry, confirmation,
lognormal/HMC, analytic hybrid, Abacus or P12 changes. The next decision is
whether stabilization is useful enough to test separately, not 'train forever'.
Four GPUs ×75minutes maximum, using measured prior throughput; persistent fixed
tmux/Slurm launcher, restartable raw/EMA/optimizer/RNG state and exit receipt.

Speed and scope: one batched exploratory comparison and automatic diagnostic
summary replace several serial runs. Preserve basic provenance and tests without
a new sealed-data campaign. Future downstream tolerances must specify a physical
estimand; this dimensionless8³toy does not define7Mpc/h smoothing. A Wiener
solution is exact here, not guaranteed near-exact for DESI low modes. Classical
nonlinear baselines and hybrids remain separate proposals, not established fixes
or an authorized broader research goal. No literature-priority/timeline claims
from the supplied discussion are adopted without independent review.
