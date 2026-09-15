# Fine-stage noise-resolved learning test — 2026-09-15

User authorized the recommended small learning test and results reporting.
This is a separate diagnostic experiment, not continuation of the full E2E
run, a model replacement, or permission to open held-out phases.

## Frozen design

Start every arm from the same 384-update fine model **and AdamW state** for its
method. Keep the architecture, normalization, learning rate, weight decay,
gradient clipping, velocity target and MSE formula unchanged. Train only three
anchors: ph000/002/003 NGC, shell0, boundary00. True coarse conditioning is
deliberate teacher forcing to isolate the fine operation. Coarse models stay
frozen. The only arm difference is noise-time exposure:

- `uniform_time`: the original uniform-in-time distribution for each method.
- `balanced_noise`: equal cycling through sigma/alpha=.05,.2,1,5,20.
- `near_clean`: equal cycling through .05 and .2, a deliberately easier
  near-clean-only learning stress test; failure at high noise is not surprising.

Use 256 fresh-noise updates per arm and method: six fits, 1,536 updates total.
Anchor order and Gaussian noise are paired between arms and methods. Schedule
levels cross all three anchors; time and noise seeds have separate namespaces.
Baseline uniform times have the same distribution as the old objective, not
the identical original RNG sequence. Preserve original parents immutably;
all 18 new checkpoints carry a separate experiment binding and inherited history.
Checkpoint step448/512/640 means original384 plus diagnostic64/128/256, not a
claim that the 96-anchor E2E run has advanced to640.

At diagnostic updates0/64/128/256, evaluate all five noise ratios with two
fixed evaluation-noise seeds excluded from training. Evaluate both the fitted
NGC anchors and opposite-cap SGC anchors matched by phase/shell/support. These
six parents are all from the original training pool. The additional three
are transfer controls, **not validation or independent held-out phases**;
the cap change also changes conditioning/domain. There are1,440 probes.
Initial probes must be identical between all arms of a method.

Generate one fixed-seed, true-coarse fine draw per anchor from each parent
model and each final branch:48 diagnostic fields. These assess whether a
specialist near-clean fit damages the rest of generation; they are not
deployable samples or calibrated posterior checks. No density clipping,
additional smoothing, training-target changes or modified sampler is allowed.

## Predeclared learnability check

For **each phase** at both near-clean ratios, using medians over two evaluation
noise seeds, require all of:

1. Absolute signed high-k residual-noise amplitude <=.2 of input noise.
2. High-k clean-error power <=.25 of its matched parent-model baseline.
3. Signal gain between .9 and1.1 in each of the three lower-k bands.

This demands actual noise removal while preserving signal, not merely a lower
aggregate loss. Report fit and transfer checks separately and retain failures;
the thresholds are engineering criteria, not a physics/calibration release gate.
Inspect the entire noise-resolved learning curve, absolute errors and generated
power, including regressions. A bounded failure cannot prove that an architecture
is incapable in principle; a tiny-panel success cannot demonstrate generalization.

If uniform-time improves, undertraining is plausible. If balanced/near-clean
exposure improves more at equal updates, noise exposure matters under this
protocol. If none succeeds, this experiment alone cannot choose between longer
optimization, inherited optimizer effects, capacity or time-conditioning problems.
No post hoc architecture change, extra fitting or automatic extension follows.

Use one GPU for at most one hour; application work stops at45 minutes if needed.
Nonfinite training/gradients, baseline mismatch, provenance drift or budget expiry
fail closed. Original source/checkpoint bindings and sealed phases stay frozen.
Code: `workflows/sbi/e2e_fine_learning_test.py`; contract:
`configs/e2e_fine_learning_20260915.json`. Run/evidence details follow on completion.
