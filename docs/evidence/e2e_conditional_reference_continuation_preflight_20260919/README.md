# Continuation preflight stop

Slurm58596902: FAILED, exit1:0, allocated four A100s for17seconds; GPU step14s.
The GPU replay equality check failed on all four workers before any continuation
checkpoint or scientific evaluation was published. The original4096run is intact.

`failed_manifest.json` preserves the first frozen source/parent-checkpoint hashes;
`failed_allocation.log` records the failure. Source was committed as5660b29.
The checkpoint generator/data replay check passed before the numerical-step
comparison failed. The first check did not log the magnitudes; do not infer them.

An actual matureCFM checkpoint replays exactly on CPU: both losses
2.9325413703918457, maximum parameter discrepancy0, on a small synthetic input
batch. This is an implementation diagnostic, not a scientific posterior result.
Seventeen focused tests pass after adding exact state-comparison instrumentation.
GPU nondeterminism is a hypothesis, not an established cause.

The separate `_r1` snapshot adds per-fit replay-difference receipts and a
deterministic-GPU smoke-only switch. No training tolerance, model, optimizer,
data law or scientific objective changes; no affine control. A single89-minute
four-GPU retry is requested, keeping cumulative allocation below6GPUh. It has
not been submitted pending user approval.
