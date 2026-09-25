# Final-horizon development evaluation

Training58862352 completed54m07s, all four factors at26624updates. User approves
evaluation and required compute. Compare EMA26624 against saved13312 outputs;
do not regenerate baseline or change draw identities, observations, physical
operators, metrics, chart or train-only feature scales. Only evaluator change
permits step26624; collector accepts the existing baseline directory and checks
paired target hashes, phase/pair identities and ensemble sizes.

New ledger:32development pairs x2seeds x32draws =2048main draws at128NFE,
plus32refinement draws at256NFE (same paired8draw subset per phase/seed).
Identical microbatch2, same addressed seeds; exact first-chunk replay retained.
Six sampler/metric tests pass0.124s. New run uses two GPUs for up to4h,
expected3.5h from measured prior evaluation; existing one-hour allocation is
not used or altered. tmux persistence, atomic draw chunks, no automatic renewal.
No training and no confirmation access. Frozen source snapshot and separate
output root `cfm_pilot_final_eval_20260925_v1` under the e2e_field_v3 Scratch
directory preserve earlier ensembles. Collector produces COMPLETE.json and
SUMMARY.md only after both final workers finish the full ledger.

Questions: does high-k power move toward truth; does the ph012/ph013 regional
coverage split shrink without sacrificing CRPS/sharpness; do joint scores and
tidal classes improve reproducibly across seeds; is sampler refinement still
negligible? Two development phases cannot certify general calibration. No
scientific conclusions from this comparison exist at launch.

Launch:58864736 on nid001024, frozen27bca24, tmux
`cfm_final_eval_20260925` on login22. Six compute-node tests passed0.108s.
Post-edit graphify update exceeded the five-second login cap; index refresh
remains incomplete, unrelated to numerical evaluation success.
