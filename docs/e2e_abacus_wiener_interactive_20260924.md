# Corrected Wiener interactive launch

User approved interactive retry after the batch data-contract failure. No active
or pending interactive allocations existed at launch. Production VAC jobs untouched.

- Allocation:58830984, CPU nid004169, account desi, interactive QOS, one node,
  32CPU threads, two-hour limit, Scratch license; eight evaluation workers.
- Persistence:tmux `e2e_wiener_v2` on login03. Fixed salloc/srun launcher only;
  no autonomous decisions, retries or new fits beyond the approved comparison.
- Scientific source:frozen `ecce430`, archived workflows/shared/configs/tests.
- Launcher:`ae8cab3`, workflows/sbi/run_e2e_abacus_wiener_interactive.sh.
- Root:`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/abacus_wiener_20260924_v2`.
- Log:`interactive.log`; output:`results/`; final shell exit appended to log.

Allocation granted and compute step started. Launch is not a scientific result.
Tests rerun on compute before data access; first-anchor smoke precedes remaining
panel. Prior failed v1 products remain unchanged. No additional training or
expanded classical programme. Allocation exits when the fixed command completes.
