# Coupled-field preparation: verification and handoff

**Preparation complete,2026-09-19.** All21phase products, independent audits and
normalized interfaces qualify. The full13-phase normalizer, physical-reference
gate and measured technical checks pass. The final recovery allocation completed
0:0; no preparation job remains active. No scientific four-arm fit or
confirmation prediction is authorized. The later scientific resource proposal
is ready for separate user review, not automatically launched.

## What is being prepared

The experiment targets a nominal mock-conditional, R7-smoothed matter-field
posterior from observed BGS galaxies and survey response. It uses13training,
2development and6confirmation simulation phases, giving1,664/32/96paired domains.
Phases001/006 remain sealed;004/005 are historical and excluded from new fits.
The new common physical frame uses the pinned DESI/Abacus distance table and
the inherited selection function's volume Jacobian, not the old Planck18 frame.

The galaxy input is already survey affected: upstream DA2 `altmtl/kibo-v1`
successful-observation catalogues, matched exactly to their forFA/CutSky parent.
We build observed count, support, response, expected-count, sampling-density,
redshift and line-of-sight channels from those products and matching randoms.
We do NOT rerun every upstream survey-simulation stage or substitute ideal,
complete galaxies. The target is same-phase real-space c000 z0.2 matter from
the10%A+B particle subsample, with TSC assignment and one R7 smoothing.

This is not an evolving matter lightcone, HOD marginalization, an independently
validated observation likelihood or real-DESI transfer closure. It prepares an
adjacent-core coupling test, not a proof of globally coherent whole-survey draws.
Full observational scope is recorded in the
[preparation contract](e2e_coupled_data_preparation_20260918.md).

## Requirement-by-requirement verification

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Expanded phase panel and sealed-phase guards | Fixed13/2/6authority; explicit role/path rejection tests | Pass |
| Exact observed/mock-parent identity | All21CutSky/forFA/successful-LSS joins, bound by independent phase audits | Pass |
| B staging, particle CRCs/headers and native counts | All17B restores; all21native2048-cubed fields,136files/phase; max count error9.54e-8 | Pass |
| Coordinate and radial selection-volume convention | Same table/axes/offset on4,077resolved hosts in007/008/020; max discrepancy0.0003167Mpc/h | Pass |
| Target-free survey observation channels | All21condition sets;18randoms/phase; support-only pair geometry; actual observation-only I/O guards | Pass |
| Positive, exact-mass coarse/fine targets and independent tides | All21target sets and independent phase audits | Pass |
| Full train-only normalization | Exact13phase/equal-weight cached-moment recomputation | Pass |
| Actual normalized targetless interfaces | All21phases;1,792pairs/11,776registered pair-offset cases; round trips and role guards | Pass |
| Registered physical-reference gate |224pair/offset cases; both operators improve median eigenvalue error~88%,25%required | Pass; truth-only |
| Full-shape technical neural checks |7GPUfactor cases; exact fresh-process replay;64/128/256NFE checks | Pass; synthetic-only |
| Measured loader and analysis costs | Actual26-pair/two-pass loader;8synthetic operator probes and32/128/256draw score timings | Pass; component costs |
| Approval-ready later experiment proposal | Receipt-backed14fit/two-seed costs; finite development/confirmation stages and joint gates | Ready for review; not launch authority |
| Resources and evidence archive | All allocations terminal; original caps retained;335receipts/11,126,435bytes independently rehashed after copying | Pass |
| Repository handoff | Source/configuration/tests/reports and small evidence, not bulk fields; branch refactor_codebase_Illustris | Commit separately verified; no remote push |

The final machine-readable data qualification exists at
`cartesian_v2/data_release/DATA_PRODUCTS_QUALIFIED.json` under the preparation
root, SHA256`cf26b07c16ec0714fcd6d0c318c77b0ab3799497a00eba4195f48a2a42889ced`.
The full interface receipt is
`fe6df30cf4ff558cf85793d30d3db80129e897b395a310dddba9271369469c74`.
Its data-only flags intentionally do not certify technical/proposal completion;
the separate [CLOSEOUT.json](evidence/e2e_coupled_20260919/CLOSEOUT.json) does.
Neither artifact authorizes scientific training or claims posterior calibration.

The final metadata closeout rechecks source bindings and audited file metadata,
recomputes the exact normalizer from all13cached train-moment receipts, verifies
the physical decision, actual-loader/GPU/replay/CPU-cost receipts and original
resource ledger. It does not repeat the earlier multi-terabyte CRC scan. The
[archive manifest](evidence/e2e_coupled_20260919/MANIFEST.json) binds every copied
receipt to its original path/hash; all335copies and the proposal binding have
been independently rehashed. Final focused suite:126tests pass in4.402s.

## Resource closeout

| Resource | Actual charged or occupied | Approved cap |
| --- | ---: | ---: |
| CPU allocation time |16.310833node-hours |64node-hours |
| Technical GPU allocation time |0.141667GPU-hours |4GPU-hours |
| New preparation Scratch |4,311,094,964,561bytes (3.920918TiB) |4,947,802,324,992bytes (4.5TiB) |

All spent/failed/planned-pause generations remain charged; budgets were not
reset. The eight registered allocations include one completed HPSS transfer job,
one completed technical GPU job and six CPU jobs. Scheduler FAILED75 checkpoint
pauses and previously diagnosed failures are not erased by successful recovery.
Final58562167 completed0:0 in53m58s; its four steps also completed0:0. The
never-launched pre-OOM final-data controller correctly refused58552205's failure.
No automatic successor, pending scientific job or full-particle restore remains.
Bulk Scratch products are still purgeable, not backed up by this small archive.

## Important negative findings and recoveries

- The phone's coordinate finding was real: a cosmology-distance curve mismatch,
  not the earlier constant missing-h adapter error. Corrected products exclude
  the old Cartesian frame without discarding valid catalogue/particle products.
- Support-only greedy packing failed at120/128for000; deterministic packing
  repair retained the original candidate, support and nonoverlap requirements.
- Target step58552205.0 hit its288GiB memory limit on014's initial FFT, after013
  completed. Reported MaxRSS was271,642,300KiB; the precise allocator/cgroup cause
  was not established. Recovery58562167 uses native-first scheduling and fresh
  target processes with400GiB. Kernels, gates and completed products are unchanged.
- Old failure/checkpoint generations and logs are retained and charged to the
  original budget. No full-particle restore or scientific fit occurred.
- A proposed10%joint-energy improvement threshold failed an analytic sensitivity
  check: even an exact14-dimensional correlated Gaussian posterior improves
  population energy by only0.20/0.57/0.94% over exact-marginal independence at
  matched cross-core correlations0.5/0.8/0.99. Three analytic tests pass. Before
  scientific predictions, the proposal now uses matched cross-core fair
  variogram improvement as primary, energy nonregression as a companion, and
  retains independent power/calibration gates. This is not model performance.

## What comes next, after separate approval

The [measured resource proposal](e2e_coupled_resource_proposal_20260918.md)
recommends development first: implement the scientific workflow, fit all14factors
at two seeds through32epochs, and evaluate the registered checkpoint/sampler/
coarse-control panels. Proposed cap:260GPUh,24CPU-nodeh and512GiB new outputs.
The128draw confirmation campaign needs a later explicit decision; its proposed
128NFE ceiling is450GPUh/32CPU-nodeh, with the same cumulative output cap.

I/J share the coarse posterior exactly. Their block-aligned regional masses
must therefore agree; improved mass coverage must come from the coarse model,
not fine coupling. Primary I:J probes are chosen to annihilate block-constant
coarse contributions. Power, residual covariance, regional-mass coverage and
tidal/eigengap calibration remain separate from good one-point marginals.

No full posterior/calibration claim follows from successful preparation. The
scientific trainer, real-payload model adapter, addressed sampling and complete
evaluation workflow still require implementation/smoke under the later approval.
Create a separate scientific-run authority bound to the qualified data hashes.
Do not flip the preparation configuration's`scientific_training_authorized=false`
flag or edit its protected coordinate/normalization/operator code: doing so
invalidates existing provenance rather than authorizing training.
