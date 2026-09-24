# First coupled CFM pilot segment

Allocation58831634 granted on nid001033, four A100 GPUs, interactive QOS,
desi_g account, two-hour cap. No other interactive allocation existed at launch;
Production VAC batch jobs unchanged. tmux `cfm_pilot_20260924` on login03 owns
the fixed launcher; no automatic follow-on allocation. Scientific frozen source
`d973cc9`, archived workflows/shared/configs/tests under
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_pilot_20260924_v1/source`.

All four factors (coarse/fine × seeds17/29) passed64actual-data updates, finite
losses. Unit tests4/4passed on compute in0.098s. Early step32/64 timings around
0.34–0.38s after startup; not a whole-job forecast or posterior-performance result.
First checkpoint saved for each factor; latest every256updates, fixed progression
checkpoints and segment-end saves. Raw and EMA model/optimizer/objective RNG,
spectral weights and bindings are retained. Full fresh-process replay still needs
verification before any scientific maturity claim.

Root: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_pilot_20260924_v1`.
Allocation log:`allocation.log`; factor logs:`coarse_17.log`, `fine_17.log`,
`coarse_29.log`, `fine_29.log`; factor dirs contain binding, spectrum, train log,
STATUS.json and LATEST.pt. Training logs label velocity diagnostics explicitly;
they are not calibrated posterior draws. Confirmation remains unopened.

Remaining scientific work: checkpoint/restart audit; development posterior
sampling/evaluation, achievable-effect controls, bounded scale-selection decision
and continuation proposal using measured times. No evaluator or confirmation
success is inferred from a healthy training launch. Graphify update attempted
under a5second lightweight cap; completion not established, global refresh deferred.
