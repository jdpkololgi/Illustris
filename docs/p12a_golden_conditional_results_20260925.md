# Halo48 conditional review and observer golden replay — 2026-09-25

The requested mock-side checks pass. Clear the candidate to enter bounded Loa
trial preparation; real Loa input fields have not been built and no DESI posterior
sampling occurred. This is not a full-footprint or public-release qualification.

## Conditional acceptance

Applied the inherited P12 blind evaluator's per-row coverage and covariate
quartile definitions to saved ph006 draws, without changing thresholds or fitting.
On50,000 already-exposed validation rows and512 untempered draws, C68 is
.69418/.69624/.70714 and C90 .90910/.90922/.91596. Maximum global error .02714
passes .03; non-sparse conditional .03576 and sparse-shell .01928 pass .06.
The natural-weight audit from the prior review is retained separately; its
maximum global excess .02975 also passes .03 but is close to the boundary.
Validation-sample quartiles and sampling differ from a full blind phase. Do not
transfer the original ph001 result to this refitted candidate or claim all nine
release gates from these three coverage decisions. No temperature correction.

## Golden replay

Eight geometry-selected ph006 cores span both caps and four redshift shells,
14,149 output galaxies. Reconstructed observer positions from RA/DEC/Z, deposited
all neighboring context galaxies in canonical shell/row order, rebuilt angular
exposure/apodization and frozen-selection contrast. All three encoder fields
(counts, exposure_apodized, log_count_ratio) match exactly; frozen full-fit halo48
predictions match stored exports. Independently sampled ntilde and P3b random
support distances agree on64 output galaxies; TARGETID, redshift and ZWARN checks
pass. Thirty-two draws per sampled galaxy reproduce the canonical-input posterior.

Initial direct Planck18 integration differed from the frozen P10 interpolation
by up to6.86e-7Mpc and failed strict parity. Matched the existing17,001-point
lookup on0-.85 and trigonometric arithmetic; coordinates and fields now agree
exactly without relaxing tolerances. Initial failed receipt and successful v2/v3
receipts are retained. This is an adapter numerical correction, not an encoder
or native-label correction. Golden mock angular support is regenerated from its
frozen support atlas; response distances are independently sampled from its
frozen random fields. Real Loa response fields are not built by this test.

## Loa sources and next action

Full data and all18 full random files (157.6GB randoms) match registered content
hashes. No clustering-random redshift is used. Pin the discussed legacy GraphWeb
quality rule: finite positive Z_not4clus, ZWARN0, DELTACHI2>=25 and GALAXY;
mock inputs retain their noiseless observed-success rule. Four adapter tests pass.

Next: construct the bounded Loa inputs with complete surrounding context, check
identity/support/input shifts, then run diagnostic posterior inference. The
handoff distinguishes readiness to start that stage from readiness to sample
before those inputs exist. Independent confirmation, class/proper-score/joint
qualification and misspecification checks remain public-release requirements.
No additional retraining is triggered by the present review.

Compute: GPU allocation58862354/nid001013; successful review and replay steps.
Sources and evidence: `evidence/p12/P12A_HALO48_BOUNDED_TRIAL_HANDOFF_20260925.json`.

Repository-index maintenance: Graphify update failed in both repositories with
filesystem flock errno524 on the compute node; existing indices were preserved.
This is not a validation failure. Allocation58862354 is released after checks.
