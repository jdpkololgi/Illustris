# Historical checkpoints on matched ph006 galaxies

The user requested a direct test of whether the older models with stronger
reported knot recall actually outperform the current P12-A pipeline on unseen
galaxies. This investigation preserves the provisional Loa VAC and all models.

## Result

**None of the four replayed historical checkpoints beats the current posterior
in average precision, either pooled or in any of the four registered redshift
shells.** Higher old hard-label recall comes at lower precision. This is not
evidence that current knot incompleteness is harmless; it rules out these
historical checkpoints as an obvious improvement on this exposed mock sample.

[Precision–recall figure (PDF)](figures/p12a_historical_knot_20260928/precision_recall.pdf),
[PNG](figures/p12a_historical_knot_20260928/precision_recall.png),
[numeric summary and provenance](figures/p12a_historical_knot_20260928/SUMMARY.json),
[original-prediction parity](figures/p12a_historical_knot_20260928/PARITY.json),
[artifact hashes](figures/p12a_historical_knot_20260928/ARTIFACT_SHA256.json).

At 0.2 <= z < 0.3, exactly 21,052 evaluation galaxies (1,161 true knots):

| Frozen model | Original hard recall | Original hard precision | Recall at >=70% precision | Average precision |
|---|---:|---:|---:|---:|
| Current posterior | 45.7% | 69.1% | **45.2%** | **0.6093** |
| Current encoder alone | 50.0% | 65.6% | 42.9% | 0.5950 |
| July G-PATCH | 50.8% | 57.4% | 31.7% | 0.5355 |
| July U-PATCH | 44.9% | 59.8% | 32.4% | 0.5289 |
| Extended G-PATCH | 46.0% | 62.1% | 31.8% | 0.5484 |
| Extended U-PATCH | 47.2% | 59.7% | 32.5% | 0.5293 |

At the exact current hard-label precision of 69.09%, the old models' empirical
frontiers recover 32.2–34.1%, versus 45.4% for a threshold on current P(knot).
That posterior threshold result differs slightly from the four-class argmax
operating point because these are different decision rules. No threshold has
been tuned into the VAC.

Across all 50,000 galaxies (3,387 true knots), AP is 0.6167 for current posterior,
0.6050 for current encoder, 0.5315/0.5332 for July G/U, and 0.5526/0.5327 for
extended G/U. At 70% precision, recall is 43.4% current posterior versus
25.7–30.8% historical. Current posterior AP across the four shells is
0.6407/0.5994/0.5885/0.5387; curves can cross locally, especially in sparse tails,
so this is not a claim of dominance at every threshold.

For the strongest historical model by pooled AP (extended G), the paired
bootstrap AP difference from current posterior is -0.0641, with 16–84% interval
[-0.0676,-0.0576]. In the wedge range it is -0.0609, interval
[-0.0682,-0.0555]. The current encoder alone also ranks knots better than all
four historical point models here: the gain is not solely a change in hard
classification rule. Current posterior supplies a smaller additional AP gain.

Both historical recovery input reconstructions reproduce one original ph000
validation core (24 galaxies) with maximum absolute eigenvalue errors
1.33e-8 (U) and 7.54e-8 (G). Reconstructed graph node transforms match the saved
original transformed inputs. All four transfers return finite predictions for
all 50,000 distinct requested parent IDs; all have zero ordering violations.
Weighted-PR sanity checks cover row permutation, monotone score transforms,
perfect/null rankings and the all-eigenvalues knot predicate. All four full
GPU replays and final scoring exited successfully. Allocations were released.

The present model still misses roughly half the true knots at its original
hard decision. Better ranking than older models does not establish adequate
completeness, class-conditional calibration, or real-DESI accuracy. No retraining,
posterior replacement, Loa selection change or release-qualification change
follows from this result.

## Frozen scope

Use exactly the already-exposed ph006 50,000-row evaluation indices from
`p12a_halo48_candidate_20260924_v1/posterior/calibration_audit/evaluation_index.npy`
and `dataset/ph006_selection_sample.npz`. Truth, natural weights, parent IDs,
redshift bins and galaxy membership are identical across models. ph006 was
not an encoder training phase, but was used for current model selection. This
is a development comparison, **not a fresh blind confirmation**. No ph001 input
or truth is opened.

Replay the rotation-0 seed-42 P8 recovery U-PATCH and G-PATCH checkpoints, then
their `convergence_extension_v1` descendants. The immutable checkpoint path,
hash, epoch, sample/index hashes and evaluator hash are recorded per replay.
Include the current saved halo48 encoder point predictions and the current
saved 512-draw FMPE posterior. Old point models are not passed through the
current posterior: that would require separately fitted, validated posteriors.

## Input and metric contracts

Historical U-PATCH uses its own frozen ph000 rotation-0 selection curves,
normalization, target scaler and 24-voxel context. Historical G-PATCH uses its
frozen cap SI medians, Box–Cox transformer, ntilde transform and edge scalers.
No transform is fitted on ph006. Existing ph006 raw graph products supply
geometry; no graph metric is recomputed. Four reverse-dependency hops preserve
the two-pass graph's selected-row inputs. Graph patches include context nodes,
while only the requested evaluation rows are scored. Current halo48 predictions
retain their original preprocessing. Thus this compares frozen pipelines, not
a controlled change of architecture alone.

Knot truth means all three tidal eigenvalues exceed 0.2. Point-model ranking
uses the minimum of the three predicted eigenvalues, which respects this
predicate even if an unconstrained point output violates eigenvalue ordering.
Posterior ranking uses P(knot); its hard decision uses four-class argmax. Old
hard decisions threshold point eigenvalues at 0.2. These decisions need not sit
at the same recall or precision, so compare full weighted precision–recall
curves and average precision (AP), plus recall at common precision levels.
An empirical recall-at-precision frontier is an evaluation diagnostic; it does
not provide a threshold validated for future deployment.

Report the full range, 0.2–0.3 wedge range and each registered 0.1-wide shell
from 0.15 to 0.55. Use 64 paired spatial-block bootstrap replicates for AP and
AP differences from the current posterior. These intervals do not include
independent-phase variance or real-DESI transfer uncertainty.

## Reproduction and provenance

- `workflows/sbi/p12a_historical_knot_comparison.py`: restartable recovery replay.
- `workflows/sbi/p12a_historical_knot_extension.py`: same evaluator, immutable
  longer-trained checkpoints.
- `workflows/sbi/p12a_historical_knot_parity.py`: original ph000 saved-prediction
  replay and exact graph-node preprocessing check.
- `workflows/sbi/p12a_historical_knot_scores.py`: identical-row metrics and plots.
- Scratch roots: `/pscratch/sd/d/dkololgi/abacus/p12a_historical_knot_20260928_v1`
  and `/pscratch/sd/d/dkololgi/abacus/p12a_historical_knot_extension_20260928_v1`.
- GPU allocations: 59024562 and 59024856; at most two interactive allocations.

The June FlowJAX model and flow files survive under
`abacus/sbi_runs/path1_wedge_flowjax_3d_Bcorrected_linear_si`, and rotation-2
P8 extension checkpoints also survive. They are inventoried but not part of
this rotation-0 comparison. June's whole-wedge graph/feature contract is
different; feeding it the P8 patch inputs would not constitute a valid test.
