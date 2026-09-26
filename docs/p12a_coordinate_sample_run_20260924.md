# P12-A first numerical coordinate audit

This is the next bounded execution step in `plan_desi_p12a_vac_20260924.md`.
It is a training-phase diagnostic, not DESI production inference or completion
of the full V0-C coordinate gate.

## Execution amendment after first coverage failure

User approved all VAC compute. Job58823054 completed the original ph002 audit
in12.39s with627656KiB maximum RSS. All numerical equality checks passed on
the tested rows, but the global top-three slab rule selected only NGC (62 native
rows); `partial_checks_pass=false` correctly prevented promotion. Retain that
report under the v1 Scratch directory. The next run uses v3 source-bound input
inventory `P12A_COORDINATE_SAMPLE_INPUTS_PH002_20260924_v3.json` and a new
Scratch directory ending `_v2`; v2 inventory below is historical.

Sampling amendment, frozen before new results: greedily select at most six
halo slabs using only FILE_NUM, valid-host flags, cap and redshift. Prioritize
the number of previously uncovered cap/shell strata, then total capped-at-16
per-stratum counts, then lower slab ID. Stop when no stratum-count improvement
is possible. Sample the first <=16 rows per stratum as before. Missing strata
still fail. Numerical tolerances and label comparison rules are unchanged.
Two regression tests cover severe NGC/SGC imbalance and unavailable strata;
nine synthetic tests now pass. The existing approved allocation is reused.

## Original frozen scope and numerical criteria (sampling amended above)

Entry point: `python -m workflows.abacus_tweb.p12a_coordinate_sample_audit`.
First phase: ph002 only. Read 16,384 deterministic, evenly spaced canonical
observed rows, following the P1 completion marker's exact FITS path. Another,
older observed FITS exists in the same directory and must not be selected by glob.
The input receipt is
`docs/evidence/p12/P12A_COORDINATE_SAMPLE_INPUTS_PH002_20260924_v2.json`.
The earlier inventory without `_v2` predates input/source binding enforcement
and is retained as history, not the launch authority. Numerical mode rejects
source, receipt, file-size or modification-time changes relative to this inventory.
It repeats these identity checks after execution. Large-file hashes recorded
upstream are not represented as independently reverified by this sampled audit.

1. Reproduce stored float64 observer-frame Planck18 Cartesian coordinates and
   Galactic caps from RA/DEC/observed Z. The original interpolation grid is
   z=0..0.85 with 17,001 nodes. Require maximum Euclidean replay error <=1e-9 Mpc
   and exact cap identity. This tolerance covers floating-point evaluation order,
   not a change in cosmology or units.
2. Independently evaluate Astropy Planck18 distances at science-range sample
   redshifts. Require maximum Euclidean error <=1e-5 Mpc. The linear interpolation
   error bound is `max(abs(r''))*dz^2/8`; a conservative curvature budget of
   20,000 Mpc and dz=5e-5 gives 6.25e-6 Mpc, below this fixed threshold.
3. Independently join TARGETID-1 into the annotated parent. Require exact IDs,
   sky positions, FILE_NUM/HALO_INDEX/BOX_INDEX, float32 labels, and CWEB equality.
4. Restrict native tests to science-range BOX_INDEX>=0 rows with nonnegative
   halo keys. Select the three most populated native FILE_NUM slabs in this
   geometry-only sample (ties by slab ID), then the first <=16 sample rows per
   cap/shell. Require at least one selected row in each of the eight strata;
   report actual counts and IDs. This bounded selection is not a population
   calibration sample or a central/satellite completeness claim.
5. Load native `x_com` with uncleaned CompaSO and converted Mpc/h units, reproduce
   the annotation's float32 position cast and periodic floor voxel assignment,
   then resample the original R7/2048/2000 Mpc/h T-Web cells. Require exact
   equality after float32 label storage and exact native CWEB equality. Read
   only selected cells from uncompressed NPZ members using read-only mappings;
   reject compressed/object members instead of inflating whole grids.

All criteria above are fixed before looking at numerical discrepancies.
Any failed check stops progression and requires diagnosis; do not tune the
tolerances to pass. A successful sample does not select the final
`consistent_fiducial` outcome. Missing native data/provenance also cannot pass.
The annotation position field's actual run provenance, ph000 imported lineage,
full-fit/OOF transforms, central/satellite/unresolved strata, pinned native
distance comparison, response volumes/support, and Loa adapter replay remain
explicitly unresolved. No reserved phase or DESI payload is read here.

## Compute request and commands

Small synthetic tests and inventory-only mode run on a login node. Numerical
mode requires Slurm on a compute node because CompaSO must decode whole native
position columns even for the small final row sample. Proposed first run:
one CPU node, one task, eight requested CPUs, 32 GiB memory, 30-minute wall limit,
account desi, interactive QOS, Scratch+CFS licenses, no GPU. This is a bounded
initial budget, not a measured throughput claim. Capture time and maximum RSS;
stop on resource failure rather than automatically enlarging or retrying.
Allocation inspection at preparation found zero pending/running jobs; recheck
at launch and preserve the two-allocation limit.

The user subsequently approved all VAC compute and job58823054 completed this
audit and the amended ph002-005 runs. Results are recorded in
`p12a_coordinate_sample_results_20260924.md`. The original commands below are
retained for provenance; their output paths already exist and cannot be reused.

```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --constraint=cpu \
  --qos=interactive --time=00:30:00 --account=desi \
  --licenses=scratch,cfs --immediate=600
```

Inside that allocation, create a new output directory (fail if already present)
and run the fixed audit; keep the unified-exec allocation session alive to collect
the result and release the node promptly:

```bash
mkdir /pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_vac_coordinate_ph002_20260924_v1
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
  --output=/pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_vac_coordinate_ph002_20260924_v1/audit_%j.log \
  env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u LD_PRELOAD \
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  /usr/bin/time -v /pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python \
  -m workflows.abacus_tweb.p12a_coordinate_sample_audit \
  --phase ph002 --samples 16384 \
  --input-inventory docs/evidence/p12/P12A_COORDINATE_SAMPLE_INPUTS_PH002_20260924_v2.json \
  --output /pscratch/sd/d/dkololgi/abacus/p10_multiphase/p12a_vac_coordinate_ph002_20260924_v1/AUDIT.json
```

Exit 2 with a complete report and `numerical.partial_checks_pass=true` is expected
for this partial audit; the full coordinate/canary gate remains unresolved.
Exit 1 or a missing/invalid report is a failure. Inspect every check and sampled
stratum, record Slurm status and resource use, and archive the small report in
Git before updating the active plan. Do not infer a pass from scheduler status.

The resource syntax follows NERSC's [interactive guide](https://docs.nersc.gov/jobs/interactive/)
and [filesystem license guidance](https://docs.nersc.gov/jobs/best-practices/).
