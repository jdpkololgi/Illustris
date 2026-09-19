# Pre-Stage-A diagnostics, 2026-09-19

Three read-only diagnostics run before committing the proposed coupled-field
Stage A budget. They add no fit, change no frozen artifact, and open no sealed
or confirmation phase. Scripts: `workflows/sbi/diagnostics_20260919/`.

| File | Experiment | Question |
| --- | --- | --- |
| `EXP1_SUMMARY.json` | sampler vs score error | how much of the fine power deficit is sampler discretisation? |
| `EXP1_SAMPLER_SCALING_B0.json` | step ladder + corrector | band powers per sampler, arm B seed0 |
| `EXP1_SAMPLER_SCALING_B0_extended.json` | corrector strength sweep | does the correction overshoot? |
| `EXP2_EFFECTIVE_SAMPLE_SIZE.json` | replicate unit | are anchors inside a phase independent replicates? |
| `EXP3_PROBE_EFFECT_SIZE.json` | gate effect size | can the proposed 10% variogram gate ever fire? |

Headline numbers, with their caveats recorded inside each JSON:

- **EXP1.** Against the registered 64-draw B/seed0 ensemble at
  `ph004_NGC_s2_interior_00` (published sample/truth power ratios
  `[0.646, 0.748, 0.766, 0.674, 0.684]`), quadrupling ancestral steps from 250 to
  1000 removes only **3.7%** of the summed band 2-5 deficit, while 2 Langevin
  corrector steps at snr 0.3 on the SAME 250-step predictor, checkpoint and seeds
  removes **49.3%** (`[0.856, 1.045, 0.878, 0.777, 0.819]`). The correction
  overshoots above snr 0.3, so the optimum is interior. The 8-draw estimator
  reproduces the published 64-draw band powers to 0.6-4.7%.
- **EXP2.** For paired arm differences, the intraclass correlation across the 16
  anchors inside a phase is `-0.06` (A->B density CRPS), `+0.15` (B->C) and
  `-0.02` (C->D). Anchors therefore behave as near-independent replicates for
  paired contrasts, giving n_eff ~ 64 of 64 rather than the phase count.
- **EXP3.** On 832 training-role coupled truth rectangles, the matched cross-core
  probe correlations are `-0.23..+0.10` raw and `-0.13..+0.08` after removing the
  shared coarse block means. The maximum variogram gain any correct posterior
  could show is **0.93% raw / 0.23% conditional on the coarse field**, against a
  registered gate of 10%.

Verify with `sha256sum -c SHA256SUMS.txt`.
