# Abacus BGS low-redshift annotation audit — 2026-10-19

## Recovery result: resolved file-0 mapping passes held-out validation

The separately approved recovery diagnostic ran once in allocation **60000317**, node **nid200013**, account desi, 1 node / 8 CPUs per task / 32 GB / 15-minute limit. The 8-CPU diagnostic completed in **8 seconds**, exit **0:0**. The allocation was released after **33 seconds** and is COMPLETED/absent from the queue. CPU interactive reserved the full node (256 accounting CPUs); no other jobs were changed.

All 135 sampled low-z resolved centrals were independently and uniquely matched to cleaned raw haloes using position (<0.01 Mpc/h) and host mass (relative tolerance 1e-5), with no missing or ambiguous matches. Of these, 68 constrained the radius and **67 were held out**. The training and heldout sets have **zero shared raw host IDs**.

The training ranks admit one selection-radius interval:

`461.0765686035156 < R < 461.07745361328125 Mpc/h`.

The fixed selection rule chose its midpoint, **461.07701110839844 Mpc/h**, without consulting heldout rows. The candidate uses the cleaned halo array, mass >=1e11 Msun/h, the corner observer and the generator's float32 position arithmetic, preserving raw table order after selection.

- **67/67 heldout centrals recover the exact independently identified raw halo row.** This was checked directly from saved arrays, in addition to the diagnostic's rank and membership checks.
- **135/135 centrals** recover the correct mass; median position residual **3.72e-5 Mpc/h**, maximum **9.29e-5 Mpc/h**.
- **39/39 satellites** recover their stored host mass. Their galaxy-to-host offsets have median **0.258 Mpc/h**, maximum **1.690 Mpc/h**, consistent with distinct galaxy and host positions; satellite positions are not expected to coincide with host centres.

The fitted midpoint is only **0.000412 Mpc/h** above the approximate c000 z=0.16 radius. Tiny boundary-membership differences are sufficient to shift subsequent indices. The diagnostic also reproduces the generator's float32 shifting/wrapping arithmetic, so do not attribute every earlier mismatch solely to the cosmology approximation.

This establishes a passing **sampled, resolved, single-file** recovery of the filtered-index mapping. It does not certify every row, other slabs/phases, the original production export revision, or particle-based hosts. `eligible_for_production` remains false. Do not adopt this empirically inferred cutoff globally without independent-file validation and provenance recovery.

Durable receipt: `lowz_recovery_validation_20261019.json`. Full outputs, including the candidate filtered-to-raw map and independent matches: `/pscratch/sd/d/dkololgi/abacus/lowz_index_audit_20261019/recovery_job60000317/`. Production catalogues and annotation pipelines were not rewritten.

## Completed compute validation: low-z index misuse confirmed in file 0

The approved diagnostic ran exactly once in CPU interactive allocation **59999126**, node **nid200059**, account **desi**, requested 1 node / 8 CPUs per task / 32 GB / 30 minutes. The diagnostic step used 8 CPUs and completed in 15 seconds with exit 0:0. The allocation was released after 54 seconds; accounting reports COMPLETED, and it is absent from the queue. CPU interactive reserved a full node (accounting AllocCPUS=256) although the diagnostic step used the requested 8 CPUs. No other jobs were modified and no retry or second diagnostic run occurred.

Artifacts: `/pscratch/sd/d/dkololgi/abacus/lowz_index_audit_20261019/job59999126/` contains `report.json`, `run.log`, the original-row sample/reconstruction, and candidate lookup-position NPZ files. `report.json` records the executed source SHA256 and source-file size/mtime provenance. Its completed flag is true.

**Result: using delivered low-z HALO_INDEX directly in the full halo table is wrong.** This is demonstrated for both resolved and particle-based galaxies in the sampled file, not merely inferred from failed environment prediction.

| Direct raw lookup, x_L2com | Sample count | Median periodic position error (Mpc/h) | Position matches within 0.01 Mpc/h |
|---|---:|---:|---:|
| High-z resolved centrals | 217 | 0.00016559 | 217 |
| Low-z resolved centrals | 135 | 777.4613 | 0 |
| Low-z resolved satellites | 39 | 844.4602 | 0 |
| Low-z particle-based centrals | 35 | 731.1546 | 0 |

All 251 high-z controls (including satellites) match the cubic catalogue's HALO_ID exactly under the direct lookup. With cleaned masses, all 217 high-z centrals also match host mass to relative tolerance 1e-5. None of the 209 low-z direct lookups matches host mass to that tolerance, and their smallest periodic position error is 76.95 Mpc/h. Satellite offsets cannot account for these errors, and the central-only result avoids that ambiguity entirely.

The observed RA/DEC/Z_COSMO transform with observer (-1000,-1000,-1000) Mpc/h reproduces high-z x_L2com positions to at most 0.001017 Mpc/h. Using observer zero fails the same controls by approximately 1732 Mpc/h. Changing x_com to x_L2com leaves the low-z hundreds-of-Mpc errors essentially unchanged. Changing cleaned=False to cleaned=True preserves the direct-index coordinates and does not repair low-z labels. These findings identify index misuse rather than a common coordinate-origin, unit or redshift-conversion error in this sample. They do not exhaustively validate all phases, files or historical grid products.

**Partial recovery of the low-z resolved index convention:** indexing the cleaned halo table after a mass >=1e11 Msun/h cut and a radius corresponding to z=0.16 around the corner observer recovers **62/135 low-z centrals**, matching both position (<0.01 Mpc/h) and mass (relative error <1e-5). It also recovers matching masses for 20/39 resolved satellites. The central median position error falls to 0.411 Mpc/h, but the maximum remains 749.35 Mpc/h; this candidate is not a complete repair. Field-particle hosts remain unmatched.

This partial exact recovery strongly supports a filtered halo-array index for the low-z resolved branch, consistent with the public generator. It is sensitive to reproducing the exact selection: the approximate matter-only distance model changes the number of selected cleaned haloes from 1,382,628 to 1,382,657 and destroys much of the agreement. Even the much closer massive-neutrino distance approximation may change boundary membership and hence subsequent row offsets. This is a plausible remaining cause, not a proven explanation of every unmatched row. The exact production export revision, distance interpolation and full selection still have not been recovered. Do not adopt the candidate remapping or change production labels yet.

The earlier snapshot-stitching / physical-decorrelation interpretation is superseded by this direct lookup evidence. Keep the low-z exclusion until a complete, validated mapping or deliberately redefined galaxy-position label pipeline exists. Particle-based host centres require separate treatment; their positions cannot be inferred by indexing the resolved halo catalogue.

The subsequent user-requested repair investigation is recorded in `ABACUS_LOWZ_REPAIR_PLAN_20261019.md`. The geometry-anchored, training/heldout radius-recovery diagnostic is `recover_lowz_filtered_index.py`; its four synthetic tests and allocation guard passed before the separately approved recovery run reported above. Both allocations have been released. Source graph maintenance was performed locally using incremental AST extraction limited to the three new code/test files, with no model/API calls or community recomputation.

## Initial read-only conclusion (superseded by the compute results above)

The current annotation implementation has no valid explicit treatment of particle-based unresolved hosts. It applies a halo-table row lookup to every row, irrespective of `RES` or the separate low-redshift construction branch. A lookup/provenance error is a serious candidate for the low-z failure. The exact cause of the historical labels is not yet proven; the delivered catalogue's index-producing code and a direct low-z position comparison remain necessary.

The earlier claim that snapshot stitching explains the failure is unsupported and should be withdrawn. Near-zero predictive R² or rank correlation does not prove zero mutual information or physical independence.

## Direct inspection of delivered files

Original catalogue: `/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_AbacusSummit_base_c000_ph000.fits` (63,897,781 rows).

Read 2,000 deterministic, equally spaced rows using `np.linspace(0,nrows-1,2000,dtype='i8')` and fitsio row selection. This is a diagnostic sample, not a population estimate.

| Observed-redshift subset | Sample rows | RES | BOX_INDEX |
|---|---:|---|---|
| z < 0.15 | 39 | 0 | all -1 |
| z < 0.15 | 383 | 1 | all -1 |
| 0.15 <= z < 0.30 | 857 | 1 | all nonnegative |

Thus the low-z flag affects resolved galaxies too; particle galaxies alone cannot explain a failure across that whole subset. Sampled low-z HALO_INDEX values are nonnegative, which is not evidence that they index the raw halo table correctly.

Cubic catalogue: `/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit/CubicBox/BGS/v0.1/z0.200/AbacusSummit_base_c000_ph000/BGS_box_ph000.fits` (232,248,642 rows). It includes galaxy x/y/z, velocities, FILE_NUM and HALO_ID.

For all 63 sampled rows with FILE_NUM=0 and BOX_INDEX>=0, indexing the cubic catalogue by BOX_INDEX reproduces HALO_MASS exactly. Their sky directions match cubic x/y/z plus a periodic translation in {-2000,0,2000} Mpc/h on each axis: median angular discrepancy 3.52e-6 degrees, maximum 1.49e-5 degrees. Subtracting 1000 Mpc/h from each cubic coordinate instead gives median 27.97 degrees. This supports BOX_INDEX as a cubic **galaxy** row reference for this slab, not a raw halo index. Across all 1,578 nonnegative-BOX_INDEX sample rows, using BOX_INDEX as a global cubic row gives only 63 FILE_NUM matches, so a slab offset/mapping is needed outside file 0; its exact definition remains unverified.

Historical labelled file: `/pscratch/sd/d/dkololgi/abacus/mocks_with_eigs_05062026_rsmooth_7/mock_bgs_maglim_path1_fiberassign_graph_ready_with_tweb_eigs_rs7_ngrid2048_thr0p2_halo_xcom.fits` (9,538,254 rows). In a separate 1,000-row equally spaced sample, 263 rows have z<0.15; all have BOX_INDEX=-1 and finite eigenvalues. The schema lacks RES and Z_COSMO. This historical output therefore does not satisfy the current joiner's low-z masking policy. Its producing code/version must be recovered before attributing it to today's source.

## Local implementation

- `annotate_cutsky_with_tweb_eigs.py:313–380`: collects FILE_NUM/HALO_INDEX, loads `CompaSOHaloCatalog(..., cleaned=False)`, and indexes the chosen position field by HALO_INDEX whenever it lies within array bounds. Neither RES nor BOX_INDEX determines validity. A wrong but in-range index silently produces finite labels. Unresolved hosts require their particle/pseudo-halo positions, not an arbitrary resolved-halo row.
- Position-field choice matters: historical output names specify x_com, whereas the public generator uses x_L2com. This is distinct from index identity and should be tested separately.
- `annotate_cutsky_with_tweb_eigs.py:171` and particle deposition in `abacus_process_particles2.py` both wrap positions modulo box length. There is no obvious half-box discrepancy between those two source conventions. This does not verify the provenance, axes or origin of the particular saved grids.
- `join_cutsky_eigs_to_fiberassign_catalog.py:230–240` now masks BOX_INDEX=-1 to NaN. This is containment, not a repaired mapping, and excludes resolved low-z rows too.
- `export_cutsky_unique_host_halo_points.py` uses FILE_NUM/BOX_INDEX as a halo lookup. The cubic-row evidence above makes that interpretation unsafe without a demonstrated conversion.

## Public construction source and limits

Inspected [amjsmith/shared_code](https://github.com/amjsmith/shared_code/tree/a40a5d3be570a6ce9d5a0698268b04864d4d3af2/abacus_mocks), pinned to commit `a40a5d3be570a6ce9d5a0698268b04864d4d3af2`.

- `make_mocks_script.py:18–24`: snapshot redshift 0.2, separate low-z branch ending at 0.15, mass division at log10(M)=11.
- `make_catalogue_snapshot.py:64–112`: resolved haloes are loaded cleaned, then mass/radius cuts occur before galaxy population. This creates a potential filtered-index namespace; the inspected public version does not establish how the delivered HALO_INDEX preserves or remaps it.
- `hodpy/halo_catalogue.py:151–166`: resolved positions use cleaned x_L2com. At lines 304–331 unresolved hosts instead read positions, velocities and masses from a separate HDF5 catalogue built from field particles.
- `make_catalogue_snapshot.py:883–911`: combines separately populated resolved and unresolved low-z galaxies before sky conversion. The low/high split is on observed redshift (`merge_lightcone`, lines 1308–1309).
- `cut_sky_evolution.py:50–65`: converts replicated Cartesian positions to RA, DEC, cosmological redshift and then velocity-perturbed observed redshift. Recovering real-space positions requires Z_COSMO, the correct cosmology and observer/periodic transformation.

This public revision is supporting construction evidence, **not verified production provenance** for the delivered v0.1 files: its exports do not include the delivered HALO_INDEX/BOX_INDEX columns, and its example configuration uses cosmology c004 rather than this catalogue's c000. Do not copy its observer settings into a repair without validation.

## Repair and acceptance gates

1. Recover the exact v0.1 code/metadata defining FILE_NUM, HALO_INDEX, BOX_INDEX, cleaning, observer and coordinate transform. Determine whether low-z HALO_INDEX is a filtered resolved index, unresolved index, combined index, or something else.
2. Make a small stratified validation set: low-z RES=0, low-z RES=1, high-z RES=1; separate centrals and satellites. Verify host IDs/masses and periodic coordinate residuals for raw/cleaned table lookups. A bounds check is insufficient.
3. Independently reconstruct galaxy real-space positions from RA/DEC/Z_COSMO and the verified transform; compare with cubic galaxies where available. Then sample the actual saved density/eigenvalue grids. This tests coordinate mapping independently of host-index lookup.
4. Keep the estimand explicit: galaxy-position labels differ from host-centre labels for satellites. Particle-host centres may require generator intermediates; galaxy coordinates alone do not recover those centres exactly.
5. Compare repaired labels with historical labels and rerun the matched low/high-z diagnostics. Retain RES, Z_COSMO, mapping status and source/version provenance in outputs. Do not treat finite eigenvalues or improved prediction alone as proof of a correct mapping.

No scheduler jobs were launched, large halo/grid scans were run, or production code/data changed. The next direct halo/grid validation requires an approved compute allocation under the workspace rules. Existing unrelated working-tree changes were left untouched. The initial base-shell `graphify` invocation failed; the follow-up below corrects that environment diagnosis.

## Pre-compute checkpoint: reconstruction verified, halo-array comparison awaiting approval

The exact v0.1 export/index definitions have **not yet been recovered**. The nearby `docs/latest.md` is a release file inventory, not an index-definition document. The public `shared_code` repository has one branch and no tags; its abacus_mocks history and current exports do not establish the delivered uppercase index columns. The older `/global/cfs/cdirs/desi/mocks/bgs/AbacusSummit` location contains earlier catalogues, not the required export code in the inspected directory levels.

Public `hodpy/galaxy_catalogue.py:112–114` explicitly assigns `halo_ind` using the current halo-array order, after the mass/radius cuts in `make_catalogue_snapshot.py`. This establishes a filtered-array namespace in that public generator; it does not prove whether the delivered HALO_INDEX was remapped by later export code.

`CHAT_FINDINGS_SUMMARY_2026-03-31.md` records a paraphrased creator email specifying observer (-1000,-1000,-1000) Mpc/h and BOX_INDEX as a cubic galaxy index. Its statement that BOX_INDEX=-1 identifies unresolved hosts is incomplete for the delivered catalogue: the current sample demonstrates resolved low-z rows with that value as well. The original email is not present in the inspected evidence.

Prepared `audit_lowz_lookup_positions.py`, restricted to an approved Slurm job step. It samples 8,000 equally spaced catalogue rows, retains FILE_NUM=0, and compares direct raw/cleaned indices with explicitly labelled mass/radius-filtered hypotheses. It separates RES and central/satellite status, records host-mass errors, checks high-z halo IDs against the cubic catalogue, and measures periodic coordinate residuals for x_com and x_L2com under both centre/corner observer hypotheses. It writes new audit artifacts only; it does not fix production labels or claim to recover unresolved host centres.

The resulting bounded sample has 460 galaxies:

| Subset | Centrals | Satellites |
|---|---:|---:|
| Low-z RES=0 | 35 | 0 |
| Low-z RES=1 | 135 | 39 |
| High-z RES=1 | 217 | 34 |

Verified the independent RA/DEC/Z_COSMO reconstruction against all 251 corresponding high-z cubic galaxy positions, using periodic residuals in the cubic [0,2000) coordinate frame. The Astropy approximation to the published c000 CLASS parameters (h=.6736, omega_b=.02237, omega_cdm=.12, Neff=3.0328, one .06 eV neutrino, Tcmb=2.7255 K) gives median residual **0.00017755 Mpc/h**, p95 **0.00048809 Mpc/h**, maximum **0.00100834 Mpc/h**. The simpler flat matter-only background with header Omega_M=.315192 gives median 0.0138886 and maximum 0.0950061 Mpc/h. The public generator uses nbodykit/CLASS, unavailable in cosmic_env; the approximation is therefore explicitly recorded and validated empirically rather than called exact. A periodic-distance boundary test also passed.

The halo slab contains 11,442,233 rows and is 2.1 GB on disk. Only its small text header has been read so far; the large arrays remain unloaded. Cleaning metadata exists. At the approval checkpoint there are no interactive allocations; unrelated Loa batch jobs are running/pending and are not part of this request.

Proposed approval scope: one CPU interactive allocation, account `desi`, 1 node, 1 task, 8 CPUs/task, 32 GB requested memory, 30 minutes, no GPUs, ten-minute allocation wait limit. Run the bounded file-0 diagnostic once, inspect output, and release that allocation on completion/failure; no automatic retries or other job changes. Example allocation command (not executed):

```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --constraint=cpu \
  --qos=interactive --time=00:30:00 --account=desi \
  --licenses=cfs,scratch --immediate=600
```

Within that approved allocation, launch the diagnostic via srun with the same CPU count, isolated cosmic_env Python and OMP/BLAS threads capped at 8. Use a new output directory under `/pscratch/sd/d/dkololgi/abacus/lowz_index_audit_20261019/`; the script refuses to overwrite an existing output directory. The source guard, reconstruction and periodic-distance check have been validated without reading halo arrays. Full script execution and graph refresh remain pending compute approval.

Graphify correction: it is installed in cosmic_env and rapids-gnn, not the base shell. A cosmic_env query succeeded. The user-requested persistent guidance is now in `/global/u2/d/dkololgi/.codex/skills/nersc-graphify/SKILL.md` and workspace AGENTS.md; the skill validator passed.
