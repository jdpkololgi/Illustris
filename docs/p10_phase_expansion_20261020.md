# Corrected galaxy-position preparation, ph000–ph024

Status: ph007 native targets passed in job 60054561 (9:05, 104 GiB).
The catalogue pilot 60055134 stopped at qualification (2:07). Diagnostic job
60057549 established that the two candidate particles have different positions
(1.953125 kpc/h apart) but exactly the same grid cell/eigenvalues/CWEB as the galaxy.
Particle identity remains ambiguous. Qualification must record this distinction
before a separately approved catalogue continuation. No corrected ph007 catalogue
or P1 was built; NOT training-ready. Existing target grids need no rebuild.
The original registry and running/queued source snapshots are preserved.

## Phase separation

The new `configs/p10_phase_registry_v2.json` combines established P10 training
exposure with the E2E reservation policy:

- Training: ph000, ph002–ph005, ph007–ph011, ph020–ph024 (15).
- Validation/selection: ph006, ph012, ph013 (3).
- Reserved confirmation: ph014–ph019 (6).
- ph001: separately reserved, historically exposed in P12-A; not fresh blind evidence.

The last seven phases are geometry-only in this migration. Their target payloads
have not been inspected by this expansion. This deliberately preserves existing
sealed access; it does not label confirmation products as training inputs.
Numerical preparation of reserved targets will need a separate frozen evaluation
contract. No model training is included in this request or pilot.

## Verified inventory and reuse boundaries

`docs/evidence/p10_expansion_20261020/inventory.json` records bounded path checks
for all 25 phases. CutSky, forFA, pota, fiberassign, full LSS and cubic catalogues
exist for every phase. Native-density receipts exist for ph007–ph024. For the 12
non-reserved phases in this range, receipts match the 2048-cubed, 10-percent A+B,
136-file, TSC count contract and the native payload file sizes. The ph007 full SHA256 check passed in job 60052784; the other checks
remain pending on compute. Reserved receipts were only checked for existence.

The E2E count grids can avoid another particle restore/deposition if their hashes
pass. E2E normalized patches and distance conventions are not interchangeable
with P10 products. The P10 R=7 Mpc/h CACTUS eigenvalue slabs must still be generated,
then the galaxy-position coordinate audit, all-population certificate, real CLI
smoke test, metadata reconstruction, annotation, observed join and P1 index.
ph002 already completed the corrected catalogue chain. In array 60049122, ph000
completed in 1:02:40 with 7,286,820 validated observed rows; ph003 is running and
ph004–ph006 remain queued, all using the existing immutable source snapshot.
ph000 membership differs from its legacy product, so graph/field/patch rebuilding
is required; its old geometry cannot be adopted.
No completion or training-readiness claims are inferred from those submissions.

## Reviewable ph007 pilot

`workflows/abacus_tweb/run_p10_expansion_pilot.sh` has two explicit stages, no
submission, no automatic retry, no deletion, and fresh output paths:

1. `targets`: four CPU nodes, 16 MPI ranks, 4 ranks/node, 32 CPUs/rank,
   `desi`, interactive QOS, Scratch license, maximum 30 minutes. Hash the existing
   32 GiB native density; write a provenance-bound adapter manifest; solve R=7
   targets using the established 16-rank layout. Historical measured solve was
   7m39s on four nodes, approximately 52.7 GB peak RSS/rank. The expected target
   output is about 104 GiB; reserve roughly 120 GiB including catalogue products.
2. `catalogues`: one CPU node, 64 CPUs, `desi`, interactive QOS, Scratch license,
   maximum 90 minutes, only after the target stage passes. Independently qualify
   ph007 coordinates, certify and smoke-test, build corrected parent/observed/P1,
   then refresh graphify locally on compute. The recent ph002 full-chain benchmark
   took 57m43s; this is an estimate for ph007, not a guarantee.

Allocations are sequential, released on completion/failure or walltime; no files
are removed and no retry or production expansion is authorized implicitly.
Use `--immediate=600` for each allocation. Preserve source hashes at launch.
Outputs: `/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020/ph007`.
Before either launch, inspect existing allocations and respect the two-allocation
limit. Target attempt 60052784 failed before the solve due to an omitted root-level
config_paths.py in the snapshot. Repaired attempt2 native-solver imports pass;
original source/logs are preserved. No target files were created and no cleanup
is necessary. Replacement target job 60054561 was submitted only after explicit approval;
no further retry is authorized.

## Remaining work after the pilot

- Freeze measured throughput and authorize the additional-phase build budget.
- Complete/reuse P2 graphs and RAPIDS metrics, observer fields P3 and graph/field
  patch supports P4. Exact membership, point geometry, masks, ordering and payload
  bindings must prove reuse for the original phases; otherwise build new products.
- Update schema materialization to consume the actual galaxy-v2 catalogue ID;
  the legacy helper is restricted to seven phases and v1 IDs.
- Generalize the training-contract builder and selection loaders to this explicit
  registry, with tests forbidding reserved-phase membership. Current training
  preparation remains the historical five-train/one-selection contract.
- Rebuild label arrays and caches in ordered increment space. Require current
  catalogue/P1/patch/source hashes; old readiness markers do not qualify.
- Distinguish whole-phase independence from internal P4 fold grouping. No blanket
  internal-fold rebuild is required for disjoint whole-phase learning. Consumers
  using within-phase held-out folds require their own grouping check.
- Prepare reserved geometry separately; freeze access and prediction rules before
  any evaluation target opening or scoring. Never pool reserved rows into fitting.
- Publish per-phase and aggregate readiness only after actual payload/loader checks.

Validation so far: 17 focused registry, density-contract, MPI layout and existing
phase-asset tests pass; pilot shell syntax passes. Full field/catalogue/payload
validation and graphify refresh are pending the approved compute pilot.
