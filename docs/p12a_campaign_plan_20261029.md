# P12-A corrected-label / low-redshift campaign

The user requests fresh P12-A training with corrected galaxy-to-tidal labels,
the shared 20/3/2 phase split and unchanged Gaussian R7. After the input audit,
the user confirmed supervision from **z=0.01 to z=0.55**, with coverage below
z=0.1 essential for lower-mass galaxy science. Do not launch the old-range
campaign as a substitute. R7 and the 20/3/2 phase policy are unchanged.

## Brief readiness audit

The supplied `configs/p12a_20phase_inputs_bound_20261027.json` is internally
consistent: its full readiness hash, 20 OOF marker hashes, phase-policy binding,
23 parent-response marker hashes and corrected label lineage pass a metadata
audit. The completed preparation receipt also passes. See
`docs/p12a_campaign_input_audit_20261029.json`. No bulk payload scan or GPU replay
was repeated here.

However, phase contracts still list only four supervised shells spanning
0.15–0.55, and the selection/context range is 0.10–0.60. Corrected low-z labels
do not automatically extend supervision. These products are ready for the
old-range corrected model, not the requested low-redshift extension. Version
the science/context bounds for the adopted z=0.01 lower limit, selection
and support, supervision/sampling shells, transforms and all 20 OOF contracts.
Reuse compatible corrected labels/geometry/response products only after checking
their coverage. Both P1 index and training-selection builders retain hard-coded
old bounds; the change must propagate through geometry and support, not merely
the posterior table filter. Register low-z diagnostics in bins
0.01–0.03, 0.03–0.05, 0.05–0.075, 0.075–0.10 and 0.10–0.15, retaining the old
four bins above0.15. Audit mock population coverage separately from redshift
coverage. Use stellar-mass stratification only with validated comparable mass
columns; label luminosity proxies as proxies, not stellar-mass validation.

Near-observer context must use available observations below the supervision
cut where appropriate and explicitly encode missing support. Do not extrapolate
the old selection curve below0.10 or treat missing context as observed zero
density. The simulation snapshot epoch remains z=0.2; observer redshift is not
evolving-snapshot truth.

The summary exporter still rejected ph001 unconditionally. It now validates
the bound phase policy before payload IO: omitted ph001 is allowed only under
its correct OOF contract; ph018/ph019 and in-sample exports are refused. Shared
exports require halo48 and terminal epochs. Historical contracts retain ph001
protection. Sixteen focused export/input tests pass. This is a downstream guard
repair, not a new data-readiness certificate.

## Run and resource plan

`configs/p12a_campaign_plan_20261029.json` stages the model settings, dependencies
and pending choices. After new input preparation: 20 fresh OOF encoders plus one
full encoder, terminal-checkpoint halo48 exports, aligned response tables,
ordered-increment FMPE, then development calibration/evaluation. Keep ph018/019
sealed until the final evaluation protocol and model/calibration are frozen.

Initial CPU canary **approved and completed** as60475000; allocation released
after1m49s, canary step28s and exit0. See [results](p12a_lowz_canary_60475000.md).
Across ph001/ph006,2,149,478galaxies at0.01–0.1 have valid corrected labels and
angular coverage, but old3Dresponses/grids do not qualify. No further launch
is authorized by this completed allowance. Original scope: one CPU node,64CPUs,
desi/interactive, Scratch, at most one hour. Inspect low-z counts and corrected
label validity and prototype observed/context/random-support preparation on
training ph001 and development ph006. No fits, full-volume FFTs, graph building,
final-test access or automatic retries; release after success/failure. This
canary sizes the full 23-phase low-z rebuild; it does not authorize that rebuild.

Subsequent GPU request, **not authorized or launched**: one 80GB GPU, 32 CPUs,
desi_g/shared_interactive, Scratch, one hour; disposable actual-loader/restart
and halo48 timing checks, no scientific fit, no retry, release afterward.
Production training uses a separately approved measured budget. The current
old-range full epoch has 336,579 updates; all 21 encoders together require
6,731,580 updates per epoch. Blindly inheriting the old 20-epoch default would
mean 134,631,600 updates before validation/exports/FMPE. No GPU-hour estimate is
defensible without throughput; expanded low-z counts must be recomputed.

Freeze equal complete-epoch budgets and terminal checkpoints after that benchmark.
The existing FMPE entrypoint uses validation-based early stopping by default;
its fixed-budget terminal path must also be implemented/verified before the
posterior production stage. Per-phase/per-redshift evaluation must expose low-z
performance rather than hide it in pooled results. No old calibration applies.

The worktree contains the other chat's uncommitted preparation work. Preserve
it; Git HEAD alone is not a complete execution snapshot. Freeze exact source
and configuration hashes before any launch. No scheduler action occurred.
Graph maintenance after the exporter change is deferred to approved compute:
repository-wide graph updates are not login-node work under the NERSC skill.
