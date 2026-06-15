# Workflow Reorganization Notes

This document tracks the workflow-folder migration for `TNG/Illustris`.

## Target structure

- `workflows/abacus_tweb/`
- `workflows/jraph/`
- `workflows/sbi/`
- `workflows/sbi/experimental/`
- `workflows/gcn_paper/`
- `shared/`
- `legacy/`
- `scripts/`

## Migration policy

- Move one workflow at a time.
- Keep root-level compatibility wrappers while migration is in progress.
- Prefer updating imports and SLURM entries gradually after wrappers are validated.

## Wrapper deprecation window

Compatibility wrappers at repo root are temporary and should be removed after one stable cycle (at least one full workflow validation run per migrated workflow).

- Deprecation starts: 2026-02-25
- Earliest removal date: 2026-03-31
- Required validation before removal:
  - Abacus/T-Web workflow end-to-end run from `workflows/abacus_tweb/`
  - Jraph workflow training/eval smoke from `workflows/jraph/`
  - SBI FlowJAX workflow smoke from `workflows/sbi/`
  - Paper-critical GCN workflow smoke from `workflows/gcn_paper/`
