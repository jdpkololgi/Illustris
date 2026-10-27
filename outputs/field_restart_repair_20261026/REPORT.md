# Field training restart repair and expanded P12-A inputs — 2026-10-26

Both numerical restart failures are repaired. The conditional density-field U-Net and P12-A U-PATCH encoder now pass their bounded restart checks without relaxing the 2e-6 parameter tolerance. This is training-input/numerical readiness, not fitted-model quality or posterior calibration.

## What failed and what changed

The original conditional density-field U-Net failed a serialized optimizer/RNG replay check. That first runner did not persist its individual difference values. The original P12-A encoder differed by 3.714342601597309e-4 after resumed versus uninterrupted two-update runs, despite matching step/cursor, epoch-address hash and scheduler state.

Deterministic GPU execution was not fully enforced. Strict PyTorch algorithms, deterministic cuBLAS/cuDNN and disabled TF32 fixed the conditional U-Net. Strict execution exposed unsupported nondeterministic backward operations in the P12-A 3-D grid sampler and max-pool. The strict path now uses equivalent separable trilinear resizing, fixed-coordinate eight-corner interpolation, and disjoint stride-two pooling with deterministic tie handling. Regression tests compare values and gradients, including boundaries, repeated coordinates, odd shapes and ties. Trainable coordinates are rejected by the fixed-coordinate helper.

The conditional trainer also needed an explicit source-revision contract and a new output root: historical source bindings correctly rejected the repair. The revision permits the named parent-to-current trainer change, rejects unrelated drift, and prevents writing revised runs into historical fit outputs. Existing models remain untouched. Failure reports are persisted before raising readiness gates.

## Verified evidence

- `ENDPOINT_READY.json`: all 656 selected training/development domains rehashed and loaded; about 11.3 GB checked. Exact replay model/EMA difference 0.0. Positive decode and maximum relative coarse-block mass error 4.44e-16; eight development cases.
- `ENDPOINT_ENTRYPOINT_REPLAY.json`: the actual trainer, independently launched as fresh-one-update, resumed-two-update and fresh-two-update, gives model/EMA differences 0.0.
- `ENCODER_READY.json` and `ENCODER_REPLAY.json`: actual P12-A trainer replay difference 0.0, scheduler equality, and successful forward checks on one geometry-selected core in each of 18 visible phases. This does not constitute a full validation epoch.
- `CROSSFIT_READY.json` and `CROSSFIT_AUDIT.json`: fifteen leave-one-training-phase-out contracts, fourteen training phases per fold, fold-specific selection/scaler fits, and all seven reserved phases excluded. Omitted-phase metrics are diagnostic only: exports require the terminal fixed-budget checkpoint.
- `PARENT_RESPONSE_PARITY.json`: compact parent-indexed support and boundary distances exactly match 64 addressed cells in the existing response grids, with separate synthetic whole-grid tests.
- `SOURCE_BINDINGS.json`: hashes of the final implementation/configuration; logs and source snapshots preserve diagnostic provenance.

## Expanded input pipeline

`configs/p12a_corrected_expanded_inputs_20261026.json` binds corrected-label data, 15 cross-fit folds, halo48 summaries, all three validation/selection phases, response inputs and a new posterior-table destination. Exports reject stale lineage/precision, incompatible geometry and incomplete or selection-chosen out-of-fold checkpoints. Posterior sample identity includes phase, cap and spatial superblock, so identical numeric block IDs in different phases cannot collide. Calibration folds 0–1, evaluation folds 2–4 and separate selection folds 15–17 are retained within each validation phase.

Twelve additional visible phases lacked P3b-R random-response products. All 324 raw random files were found (1,232,394,442,560 bytes; existence/size inventory alone is not a content audit). The original representative map build timed out at 15 minutes. A contiguous FITS record reader replaces costly indexed column gathers; a regression test verifies identical maps and rejection accounting. New phase registries hash all 18 random files, retain the existing frozen 18-realisation protocol, and check file size/mtime around map construction. Historical angular maps are copied only after metadata/payload/source-registry checks. The two response features needed by P12-A are evaluated at the exact registered voxel centres addressed by parent rows; unused full-volume response channels are not produced.

All eighteen response products passed, including payload checksums; the complete inventory is copied into `RESPONSE_INPUTS_READY.json`. It contains 132,231,271 parent rows, of which 111,631,040 have supported response values. Final regression and readiness results are in `RESULTS.json`.

## Remaining scientific work

P12-A posterior training tables cannot exist until fresh corrected-label encoders have been fitted: fifteen out-of-fold encoders plus one full-training encoder, followed by halo48 exports and aligned response-table construction. No such scientific fits were authorized or launched in this diagnostic allocation. Their fixed training budget and compute resources still require registration/approval. Historical predictions must not be substituted. Reserved ph001 and ph014–ph019 remain closed.

Allocation 60300734 on nid200273 was authorized for one 80 GB GPU, 32 CPUs, desi_g/shared_interactive, Scratch, at most 60 minutes including targeted retests, with release afterward. Final allocation states and elapsed times accompany `RESULTS.json`.

## Final verification and operational limits

The final focused suite passed 61 tests in 2.409 seconds. The first all-phase response step reached its preset 20-minute timeout (exit 124); completed products and per-realisation checkpoints were retained. A further 200-second resume used only remaining time in the same approved allocation. The final per-phase status is in RESULTS.json.

The required `graphify update .` was attempted on the compute node but failed in its filesystem lock (`fcntl.flock`, OSError 524). The global index was therefore not refreshed. No lock bypass, installation change or security change was made. Direct source hashes and test logs provide the change provenance; the knowledge graph may be stale.

## Recorded completion state

Both allocations were released: diagnostic GPU allocation 60300734 completed in 59m34s, and separately approved CPU continuation 60302320 completed in 4m58s. The CPU preparation step passed in 3m30s. Individual diagnostic failures and bounded timeout exits are preserved in ALLOCATION.txt; no other jobs were modified.

All eighteen visible-phase response products are complete and verified. The inventory is `/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020/p12a_response_inputs_20261026_v1/adapter_inventory.json`, SHA256 `7cbb0539ac0503098951291ca4499fc1a43b573fba1d5d0e1d3ba61d11e1fc77`. Completed products and partial checkpoints were reused during the CPU continuation. No scientific fits ran.

Pre-fit data preparation is complete. Conditional density-field training and P12-A encoder training are numerically ready. P12-A posterior training is still gated on the fresh scientific encoder fits, halo48 prediction exports and aligned table construction described above. Those stages cannot be replaced by the disposable two-update checks or historical predictions.
