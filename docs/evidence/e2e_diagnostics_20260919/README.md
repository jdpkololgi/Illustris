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

## Follow-up, same day

Two extensions were run after the first three, and one of them materially
qualifies the EXP1 headline.

| File | Experiment | Question |
| --- | --- | --- |
| `EXP4_SAMPLER_CALIBRATION.json` | corrector vs calibration | does the corrector improve CRPS/coverage, or only inflate variance? |
| `EXP3_BLOCK_SWEEP.json` | coarse blocking sweep | does a coarser coarse cell rescue the primary gate? |
| `EXP3_PROBE_EFFECT_SIZE_block8.json` | 54.13 Mpc/h blocking | probe correlations and achievable gain |
| `EXP3_PROBE_EFFECT_SIZE_block16.json` | 108.26 Mpc/h blocking | probe correlations and achievable gain |

- **EXP4 qualifies EXP1.** The corrector setting tuned on arm B (2 steps, snr 0.3)
  does not transfer. Over four ph004 anchors at 32 draws: arm B improves density
  CRPS by 1.69% on average (3/4 anchors) and closes the C90 gap in 4/4; arm **D is
  neutral** (CRPS +0.55%); arm **C is harmed** (CRPS +9.56%, improving in 1/4).
  So the corrector is an arm-specific tunable sampler factor, not a universal
  repair, and the "half the deficit is the sampler" result from EXP1 is specific to
  arm B at one anchor. Separately, several anchors are simultaneously OVER-covered
  at voxel level (C90 0.94-0.95 against an attainable 0.879) and power-deficient --
  the marginal-versus-joint distinction visible inside a single ensemble.
- **EXP3 block sweep.** Coarsening the coarse cell raises the maximum achievable
  variogram gain from 0.233% (27.06 Mpc/h, registered) to 0.545% (54.13) and 0.797%
  (108.26) -- all still 12-40x below the 10% gate. Geometry alone does not rescue a
  relative-gain gate. But the measured reference correlations are non-trivial, so a
  DIRECT chi-square test of rho_hat against rho_ref is well powered: 3.4 sigma at
  54.13 Mpc/h blocking with ~192 paired domains, rising to 5.5 sigma at ~384, while
  the registered blocking needs ~384 to reach 3 sigma.

Verify with `sha256sum -c SHA256SUMS.txt`.
