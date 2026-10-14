# Exact block sampler checkpoint

## Completed after the approved temporary workaround

At 2026-10-09 14:12 UTC, the Gaussian panel and independent reconciliation are
complete: seven tests, 18 targets, 48 configurations and 1,376 receipt checks
pass. CPU59593723 has been released. The user explicitly approved the temporary
read-only NERSC module and Home/U2 output exception; cosmic_env was not modified.
See `e2e_exact_block_sampler_results_20261009.md` for results and exact provenance.
No resume is necessary. Do not relaunch into the completed output directory.
The earlier status and source hashes below are historical, pre-amendment records;
current executed hashes are bound in the completed receipt.

## Historical blocked state

Status at2026-10-09 00:42UTC (01:42Europe/London): code and prospective protocol
are present; numerical tests and campaign have NOT run. No Gaussian success,
mixing-speed improvement, scalability claim or Abacus inference follows yet.

## Implemented

- `workflows/sbi/e2e_gaussian_block_sampler.py`: exact Gaussian conditional
  subspace/patch updates; explicit innovation-covariance composition; analytic
  mean/covariance invariance, finite-sweep moment propagation and stationary
  integrated autocorrelation; geometry-based global low-mode updates.
- `workflows/sbi/e2e_exact_block_diagnostic.py`: fixed18target/48configuration
  comparison; matched actual-canvas reference, independent covariance-form
  posterior reconciliation, block constraints and stochastic sampler checks.
- `tests/test_e2e_gaussian_block_sampler.py`: seven focused tests, UNRUN.
- `workflows/sbi/e2e_exact_block_step.sh`: foreground compute-node entrypoint,
  isolated cosmic_env, tests before campaign, exclusive-create log.
- Prospective contract: `e2e_exact_block_sampler_protocol_20261009.md`.

`bash -n workflows/sbi/e2e_exact_block_step.sh` and `git diff --check` pass.
Even Python syntax inspection through the required Scratch cosmic_env could not
start within15seconds and was terminated by its bounded timeout. Do not count
that as a Python syntax/test pass. Graphify query/update also failed to complete
within their bounded timeouts; no global graph refresh is claimed.

## Resource blocker

User explicitly approved required compute. Requested CPU interactive59579059:
one node,64logical CPUs,desi,one hour,`--licenses=scratch --immediate=600`.
It remained PENDING with `Reason=Licenses`; `scontrol show licenses scratch`
reportedTotal1000000,Reserved1000000. At00:41UTC the ten-minute request timed
out without starting. `sacct` reportsCANCELLED,elapsed00:00:00; no remaining
allocation for this request. No sbatch, automatic resubmission, removal of the
required filesystem license or numerical fallback to a login node.

The runtime and outputs genuinely depend on Scratch. NERSC documents filesystem
licenses as the dependency mechanism that prevents starts during unavailability:
<https://docs.nersc.gov/jobs/best-practices/>. The local scheduler observation is
the evidence here; no outage cause or restoration time is inferred.

## Resume after Scratch is available

Recheck current allocations and reuse a compatible one. The user's compute
authorization applies to this exact diagnostic; the two-allocation limit and
bounded interactive workflow still apply. If requesting anew, use the same
one-hour CPU request and explicit Scratch dependency, not an unattended batch.

Within the approved allocation:

```bash
srun --jobid=ALLOCATION_ID --nodes=1 --ntasks=1 --cpus-per-task=64 \
  --cpu-bind=cores --mpi=none --resv-ports=0 \
  bash /global/u2/d/dkololgi/TNG/Illustris/workflows/sbi/e2e_exact_block_step.sh
```

ReplaceALLOCATION_ID with the verified allocation. The script runs all seven
focused tests first and only continues if they pass. Expect outputs in
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/exact_block_sampler_20261009_v1`;
this turn did not launch its writer. Inspect any existing outputs before a retry;
do not overwrite receipts or mix source versions. Record Slurm step exit status,
validate COMPLETE.json and its source bindings, independently check numerical
claims, write results, then release the allocation.

Source baseHEAD40d1cf33914bd061035ff8b13726a290e5d3261f, with pre-existing dirty
work preserved. New files are uncommitted. SHA256 checkpoint:

| File | SHA256 |
|---|---|
| e2e_gaussian_block_sampler.py | 37f571571ec44c231f857a0b29ae3c98e31dc965fe98c59be29174caea082402 |
| e2e_exact_block_diagnostic.py | cf362bea41402a9f1340a3ae6a7da07db683ee0a035fd6061a115ef09c14512f |
| test_e2e_gaussian_block_sampler.py | a46f204c85db36ba48279d1533ea3b41b35091f11128a25caaf74c6d90722860 |
| e2e_exact_block_step.sh | 94528c1b158d104531f84e50ca8473407cc01c70efcf5347905da06f44a499d4 |
| e2e_exact_block_sampler_protocol_20261009.md | 393115968037418fa2d340b8d1372eb1d9ad81a8a9bb95763048a29d8c5829cd |
