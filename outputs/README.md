# Scientific execution evidence

Track compact run receipts, source-hash manifests, readiness results and the
small scripts needed to interpret or reproduce the recorded checks. These
records include failed attempts; consult the newest SCIENCE_LOG.md entry and
final result receipts before treating an intermediate result as current.

Full execution source snapshots in `*/source/` and `*/source_attempt*/` remain
local and are ignored by Git. Their adjacent SOURCE_FREEZE manifests retain
file hashes and original locations. A clone alone does not contain those exact
historical snapshots. Small diagnostic source excerpts are retained when they
explain a specific failure or numerical comparison.

The two large generated `product_lifecycle*_20261019.json` inventories also remain
local. The durable product policy is `../configs/product_lifecycle_20261019.json`,
with compact audit evidence in `../docs/evidence/` and the interpretation in
`../docs/product_lifecycle_20261019.md`.

Simulation payloads, checkpoints, figures and runtime logs continue to follow
the repository ignore rules. No files were deleted during Git organization.
