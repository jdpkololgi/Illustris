# Phase 3 Completion Note

Date: 2026-02-26

## Scope completed

Phase 3 target from the refactor plan ("remove overlapping SBI variants") is complete.

## What was finalized

- Canonical SBI training path remains:
  - `workflows/sbi/jraph_sbi_flowjax.py`
- Single optional experimental SBI path remains:
  - `workflows/sbi/experimental/jraph_sbi_two_stage.py`
- Overlapping SBI variants were moved to legacy:
  - `legacy/sbi/jraph_sbi_pipeline.py`
  - `legacy/sbi/jraph_sbi_flowjax_two_stage.py`
- Root-level compatibility wrappers for those moved scripts were updated to point at `legacy/sbi/*`.
- Active workflow index was updated to make canonical vs experimental vs legacy SBI paths explicit.

## Validation summary

- Legacy SBI wrappers now return correct `--help` behavior via module execution wrappers.
- Canonical SBI and experimental two-stage entrypoints retain CLI-safe startup behavior.
- Runtime guard behavior for active GPU/CPU-MPI workflows remains enforced.

## Notes

- Legacy scripts are retained for reproducibility only and are not recommended for new runs.
- New SLURM launches and documentation should reference canonical paths first.
