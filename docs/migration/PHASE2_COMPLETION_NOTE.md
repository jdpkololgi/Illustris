# Phase 2 Completion Note

Date: 2026-02-25

## Scope completed

Phase 2 targets from the refactor plan are complete for active workflows:

- `workflows/abacus_tweb/abacus_process_particles2.py`
- `workflows/abacus_tweb/abacus_cactus_tweb.py`
- `workflows/abacus_tweb/annotate_cutsky_with_tweb.py`
- `workflows/jraph/jraph_pipeline.py`
- `workflows/sbi/jraph_sbi_flowjax.py`

## What was finalized in this pass

- Canonical workflow entry scripts were hardened to resolve moved repo modules after reorganization.
- Legacy and experimental scripts were updated with safe `--help` behavior where needed.
- Optional dependencies in legacy scripts were guarded so smoke tests do not fail at import time.
- Wrapper invocation for `jraph_pipeline.py` was corrected to avoid incorrect `main()` signature calls.
- `workflows/abacus_tweb/abacus_process_particles2.py` now has an explicit CLI parser for non-destructive `--help`.

## Validation result

Illustris workflow-wide smoke test (`workflows/**/*.py --help`, excluding `__init__.py`) passed:

- Total scripts: 23
- Pass: 23
- Fail: 0
- Timeout: 0

## Notes

- Root-level script shims are intentionally kept during the migration window.
- Canonical execution paths should now be preferred for all new runs and SLURM updates.
