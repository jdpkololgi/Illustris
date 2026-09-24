# CPU batch launch receipt

Job58828297 submitted with explicit user approval: one CPU node, regular QOS,
desi account, two-hour cap,32CPU threads,8independent workers, Scratch license.
No interactive allocation used or modified. No automatic retry. No scientific
result is claimed by submission. Goal remains calibrated field inference.

Frozen committed source: `f53898a63e7711c0eb59d40e230f1c56b57baf4f`, archived
workflows/shared/configs/tests under
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/abacus_wiener_20260924_v1/source`.
Three bounded solver tests passed in2.796s locally and2.906s from frozen source.
They rerun inside batch before any scientific input. First anchor is serial
fail-closed smoke before the remaining31anchor panel. CG failures halt comparison.
Log: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/abacus_wiener_20260924_v1/job_58828297.log`.
Expected results: same root, `results/COMPLETE.json`, per-anchor JSON and draws.

Graphify refresh was attempted with a20second login-node cap; AST extraction
finished but the command timed out before completion. Global graph was not
refreshed. No extended login-node indexing or scientific sampling was attempted.
Unrelated Production VAC worktree edits remain untouched.
