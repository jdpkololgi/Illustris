# September 9 conversation and scientific-work recovery

## What was recovered

The saved conversation record contains the later discussion missing from the
app's task reader. [The recovered transcript](2026-09-09-conversation.md)
contains 64 user-visible messages: eight user messages, seven assistant answers,
and 49 progress updates, ending at 23:29:52 UTC. It retains the original text,
including qualifications and corrections. It excludes internal instructions,
reasoning and tool traffic. This is content recovery, not an app-display repair.

The task is “Conditional field models”, ID
`01a06ce0-480c-71c1-9e40-a32a06b743a4`.
During recovery, the app reader omitted the later wide-coarse discussion, while
the saved conversation record retained it. This establishes a retrieval/display
discrepancy; it does not establish the underlying product bug or its cause.
The original conversation storage and app database were not modified.

## Methodological decisions and corrections

1. **Regeneration succeeded; the finite-domain approximation was the problem.**
   Corrected Gaussian-R7 density, spectral derivatives and consistent sampling
   do not by themselves supply matter outside a finite parent. The audit used
   true density, not learned predictions: it was not evidence of a failed E2E
   model or posterior miscalibration.
2. **Separate exploratory training from scientific-release claims.** At 22:12
   UTC the assistant acknowledged that its combined every-anchor/every-test
   rule, including zero connectivity reversals, was too restrictive as a blanket
   veto on exploratory work. Keep the original negative screen; do not relabel
   it a pass. Those internal limits are historical diagnostics for the new
   extension, not automatically binding research-pilot acceptance criteria.
3. **Return to the shared multiscale design.** Wide observed galaxy/response
   context conditions one coarse stochastic realization, with fine residuals
   conditioned on that same realization. Independently generated patches stitched
   together are not the intended coherent field posterior. The learned system
   itself had not been completed or trained.
4. **Wide conditioning alone is insufficient.** The wide coarse field must enter
   the tidal calculation, not merely the neural-network input. Distinguish
   cross-patch long-mode coordination from tides generated beyond the physical
   reconstruction domain. The earlier suggestion to retreat immediately to
   density-only training was explicitly qualified as premature.
5. **Retain the useful existing products.** Keep the 96-cubed local targets,
   independent full-box references, observations and phase splits. Add wider
   coarse density targets, matching wide observations, fine residuals without
   double-counting, and shared geometry. No broad architecture search or full
   rebuild was required.
6. **Run the bounded truth-only comparison first.** Compare wide-coarse plus
   local-fine tides to full-box truth and the old local-96 baseline, with a
   fixed-extent finer-grid control. Separate eigenvalue, class, pair, connectivity
   and void effects. Retain tensor-error cross terms. Full-box-low truth is an
   attribution oracle only, never a deployable condition.
7. **Authorization was bounded.** At 22:40 UTC the user authorized the interactive
   product build and initiation of that test. No learned fit, holdout access or
   change to P12-A/D2/P13 was included. Products support a future training
   workflow; technical completion is not an automatic training release.

The historical literature discussion is preserved in the transcript, rather
than reinterpreted or newly validated during this recovery.

## Built products and frozen provenance

Run definition: [wide-coarse prototype](../e2e_field_wide_coarse_20260909.md).
Configuration: [frozen config](../../configs/e2e_field_wide_coarse_v1.json).
Builder: [implementation](../../workflows/sbi/e2e_field_wide_coarse.py).

| Variant | Physical side (Mpc/h) | Coarse grid | Cell (Mpc/h) |
| --- | ---: | ---: | ---: |
| wide192_f4 | 649.536 | 48 cubed | 13.532 |
| wide384_f4 | 1299.072 | 96 cubed | 13.532 |
| wide384_f2 | 1299.072 | 192 cubed | 6.766 |

The third variant controls resolution at fixed extent. All use the same
96-cubed fine fields and 32-cubed science cores. The fixed low-pass projector is
`|k| <= 0.08 h/Mpc` on the once-R7 full-box density; this is a decomposition,
not a second R7 smoothing. Coarse cap observations explicitly distinguish missing
coverage from zero counts. No wide-context normalization was fitted.

- Training phases: ph000, ph002, ph003; 96 anchors and 288 anchor-variant checks.
- Nine HDF5 payloads, 5,686,861,663 bytes.
- Maximum reconstruction error: 3.070e-7; independent Fourier interpolation
  error: 6.191e-7; relative count-conservation error: 6.706e-10.
- Fifty focused tests passed before launch.
- Source commit: `5051f69`; launch record: `6486cfe`; product receipt: `0769eb6`.
- [Durable product checksums](../evidence/e2e_field_v2/wide_coarse_20260909/PRODUCTS_AND_TEST_START.json).

Scratch root:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v2/wide_coarse_20260909`.

## Newly verified terminal status

The last recovered answer correctly said the test was running at its timestamp.
It has since finished: Slurm job **58131992**, node **nid004156**, reports
**COMPLETED, exit 0:0, elapsed 36m03s**. The completion marker reports a technical
pass, no learned training and no holdout payload reads.

`PHYSICS_TEST_REPORT.json` contains all 96 anchors. Its SHA256 was rechecked during
recovery and matches the terminal receipt:
`ea4196eda37470445d3e003e0708d43d8845c5715affc06f9b6a283e40da574c`.

The existing science log and run note still describe the earlier running
snapshot. This recovery note supplies the verified newer status without
rewriting historical messages or asserting a scientific pass.
`training_ready=false` and `r0_physics_pass=false` remain unchanged. Interpretation
of the completed comparison and the next training decision remain separate work.

## App-display issue

Content is recovered to the transcript above; restoration into the task's message
display is not verified. No database edits, resets, cache deletions, scheduler
changes or external uploads were made. The official troubleshooting guidance
describes submitting feedback from the composer and reviewing logs for sensitive
content before sharing: [official troubleshooting](https://learn.chatgpt.com/docs/reference/troubleshooting).
No report or scientific transcript was sent to a third party during recovery.

### Bounded follow-up investigation

On September 10, the app reader still reported the September 9 14:31 UTC turn
as in progress and returned no newer turns or further page cursor. The saved
record contains later task-start and task-complete pairs, including the 22:40
wide-coarse request and its 23:29 completion. No rollback marker appeared in the
inspected interval. The 14:31 turn itself has no matching completion in that
interval. This is consistent with stale loaded task state or a history-reader
problem; the precise cause, including whether the unmatched early turn matters,
is not established. It is not evidence that the later message text was deleted.

No exposed tool restores original message positions inline. The documented
`thread/inject_items` operation appends model-visible prompt history; it is not
a documented display-repair mechanism. Per the user's instruction to prioritize
science if restoration is burdensome, no app-storage mutation was attempted.
See [official app-server documentation](https://learn.chatgpt.com/docs/app-server).

### Fresh-reader follow-up during pipeline implementation

A fresh local app-server independently reproduces the missing-later-messages
cutoff through both `thread/read` and `thread/turns/list`. Its last returned turn
is `01a08694-a7ca-7571-aca0-25d4cbf7c21d` (September 9, 14:31:24 UTC); it returns
ten turns and the expected saved-record path, but not the later completed turns.
The fresh process calls the task interrupted, while the desktop previously
called it in progress. This rules out a purely desktop rendering-cache
explanation; it does not identify the underlying parsing/state defect.

Both temporary diagnostic servers exited on stdin EOF. No resume, injected
messages, app restart, database edit, saved-record rewrite, deletion or external
upload was performed. The inspected supported API documents retrieval and
continuation, not a history repair operation. A desktop restart is therefore
not a verified remedy. App synchronization remains unresolved; the current
scientific handoff is in `../e2e_wide_pipeline_20260910.md` and SCIENCE_LOG.md.

The physics closeout and training-pipeline implications are now recorded in
[the results report](../e2e_field_wide_coarse_results_20260909.md).
