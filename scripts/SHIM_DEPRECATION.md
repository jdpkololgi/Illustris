# Compatibility Shim Deprecation

Root-level wrappers are temporary compatibility shims for the migration to `workflows/*` and `shared/*`.

## Timeline

- Deprecation notice effective: 2026-02-25
- Earliest removal date: 2026-03-31

## Removal gate

Shims may be removed only after one full validation cycle confirms all active workflows run from canonical paths:

- `workflows/abacus_tweb/`
- `workflows/jraph/`
- `workflows/sbi/`
- `workflows/gcn_paper/`

Until then, do not remove root-level wrappers.
