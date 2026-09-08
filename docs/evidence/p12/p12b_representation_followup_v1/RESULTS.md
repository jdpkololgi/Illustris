# P12-B investigation: features are used, but joint optimization has no demonstrated gain

Completed 2026-09-06 on allocation 57986464. The original pilot is unchanged.
The read-only diagnostic and five-arm continuation both pass numerical/replay
gates. No ph006 payloads or ph001 data were used. All results below are exploratory
ph005-internal, one-phase/one-seed results on 1,717 rows and 56 cap+superblock
clusters, not independent confirmation or a production-calibration pass.

## What the diagnostics ruled out

The encoder is connected, changes during training, and retains feature spread.
The joint standardized feature RMS shift is 0.1713; the old deterministic point
head is exactly unchanged. Original terminal posteriors replay with zero measured
difference. The fitted frozen and joint heads rely strongly on their 32 features:
cap/shell-preserving feature shuffles raise energy by 0.08098 and 0.08693,
respectively, while the point-only negative control is exactly unchanged.
Feature permutation moves the posterior mean vector by an average Euclidean
0.2037 (frozen) / 0.2143 (joint) in physical eigenvalue coordinates. Average
posterior widths change by only about 1--2.5%, confirming substantial disturbance
of location information in the fitted heads.
They do not prove extra information beyond the cached point prediction: shuffling
creates inconsistent/off-manifold combinations of point and latent context.

Joint global clipping occurred at 90.9% of logged updates, versus 23.1% for the
head-only arms. Mean terminal head/encoder gradient norms on 32 fixed training
probes were 3.59/14.98. Adam's adaptive scaling means these statistics alone are
not a diagnosis of harmful clipping, motivating the controlled continuation.

## Matched continuation results

All arms end at 6,000 total head updates and exactly 431,146 row presentations.
The added 3,000 updates use the same continuation of the original core/noise
schedule. Head/encoder LRs remain 5e-4/1e-5; no LR sweep was run.

| Arm | Parent energy at 3000 | Energy at 6000 | Marginal 68% coverage at 6000 |
| --- | --- | --- | --- |
| point continuation | 0.138912 | 0.142606 | 0.736 / 0.738 / 0.719 |
| frozen-feature continuation | 0.138369 | 0.139953 | 0.702 / 0.744 / 0.727 |
| joint continuation | 0.139325 | 0.139702 | 0.635 / 0.670 / 0.631 |
| joint, separate group clipping | 0.139325 | 0.140616 | 0.677 / 0.715 / 0.707 |
| joint warm-start from frozen head | 0.138369 | 0.140363 | 0.624 / 0.670 / 0.635 |

Lower energy is better. Paired 2,000-bootstrap 95% intervals resample the 56
cap+superblock clusters; they are descriptive and not multiplicity-adjusted.

- Point continuation minus its parent: +0.003694 [0.001456, 0.005696].
- Frozen minus point at 6000: -0.002653 [-0.003831, -0.001473]. This relative
  advantage is not an absolute improvement over the original frozen pilot;
  the point control has deteriorated. Frozen minus its own parent is +0.001585
  [-0.000701, 0.003779].
- Joint minus frozen at 6000: -0.000251 [-0.001427, 0.000903], unresolved.
- Separate clipping minus joint continuation: +0.000914 [0.000022, 0.001677].
- Warm-start minus frozen continuation: +0.000410 [-0.000764, 0.001630], unresolved.
- Warm-start minus joint continuation: +0.000661 [0.000335, 0.001034].

No continuation demonstrates an energy improvement over its 3000-update parent.
Coverage also changes materially: longer point/frozen training shows pooled
overcoverage and broader intervals, whereas joint variants retain undercoverage in some components
and shells. Separate clipping changes that trade-off but does not improve energy.
The registered common-noise Heun64/128 checks pass for every arm, so these results
are not explained by the tested numerical integration tolerance.

## Scientific conclusion and next gate

Do not claim that joint FMPE is intrinsically ineffective or that frozen features
contain no additional information. Do reject the simple explanations that the
encoder was disconnected, its features were ignored/collapsed, or that doubling
this constant-learning-rate budget/warm-starting/removing cross-group clipping
reliably fixes the pilot. The experiment does not distinguish optimization drift
from limited-data generalization well enough to label the effect as overfitting.

No model is promoted and no further variant is automatically triggered. If the
representation branch continues, first establish a stable head-training protocol
on multi-phase data with a registered internal convergence/selection procedure,
and consider a matched residual-target head to separate inherited point-location
information from additional uncertainty information. Then require phase/seed
replication and independent confirmation. This remains per-galaxy inference;
shared U-Net features do not provide coherent field posterior draws.

Evidence: FOLLOWUP_COMPARISON.json SHA256
8296b3da2f96b0886fab72d4855ed87f8e6a16fa2f037c45ec4c4283d90d15cf.
The read-only diagnostic report SHA256 is
93e58df1d1e180e52f24de330c7c569450d0dcd5aabac5bb1f4be882383ef97e.
