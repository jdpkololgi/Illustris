# Coupled CFM original-horizon continuation

User authorized unchanged continuation after paired development improvement.
Launched job58862352 on nid001100, four GPUs, 90-minute cap; tmux
`cfm_continue_20260925` on login31. Launcher commit cded6d0; training executes
the original immutable pilot source. No new architecture, objective, schedule,
training split or confirmation access. Target is26624updates for all factors.

Starting coarse17/29steps20749/20721, fine17/29steps17949/17945. All four
workers resumed beyond these steps with finite loss; four compute-node tests
passed (0.101s). Current per-step times~0.36s coarse/~0.41s fine suggest
roughly one hour for the longest worker, not a guaranteed completion time.

Original parent states copied before launch into
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_pilot_continue_20260925_v1`.
This directory also holds allocation and four worker logs. Training outputs
remain in `cfm_pilot_20260924_v1/{coarse,fine}_{17,29}`; original historical
checkpoints preserved. Completion requires each STATUS.json training_complete
and CHECKPOINT_026624.pt, not merely allocation exit0.

Next scientific step: compare final EMA draws against13312 on the same
development panel, emphasizing phase-dependent regional coverage, spectral
deficit and proper scores. This launcher performs training only. Confirmation
remains sealed. Post-edit graphify refresh exceeded the five-second login cap.
