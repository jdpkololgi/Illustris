# Full-cap shared-state density: implementation and qualification

The requested deliverable is one shared density field per NGC/SGC cap with
observed galaxies overlaid, not an atlas of independently generated tiles or
another pair-only demonstration. NGC and SGC remain disconnected survey domains.
Real selection/support holes must not be filled to make surfaces appear joined.

The user approved necessary compute. First use a bounded one-GPU allocation
for multi-axis engineering checks and an actual full-NGC residency/step
benchmark; size any production job from that measured result. No retraining,
production VAC edits, sealed mock phases, or posterior-qualification claim.

## Construction

`e2e_global_canvas.py` generalizes the previously tested overlap consensus to
arbitrary block-aligned windows. At both Heun evaluations, every network reads
the same unmodified global latent state. Window velocities are summed, divided
by their physical coverage multiplicity, and only then used to update that
state. The fine process retains its four-cell block-zero projection. Each local
fine model sees its crop of the same sampled coarse canvas. Decode uses the
same global block lattice; there is no final density averaging or smoothing.

The local networks retain their trained dimensions and conditioning conventions.
All atlas observation windows are reused (1,626 NGC; 846 SGC). Fine context
unions are larger than their old disjoint ownership cores. Uncovered cells are
not inferred; support is separately retained. Independent survey domains are
not joined across the unobserved sky.

## Engineering checks

Allocation 59177348: five focused tests pass (two-window integrator parity,
multi-axis reduction, uncovered cells, block alignment, invalid-window guards,
and global-versus-cropped density decoding).
Eight actual neighboring windows per cap, four draws, NFE128:

| Cap | Sampling seconds | Checked intersections | Maximum overlap difference | Maximum relative block-mass error |
| --- | ---: | ---: | ---: | ---: |
| NGC | 31.885 | 28 | 0 | 4.274e-16 |
| SGC | 30.653 | 28 | 0 | 4.274e-16 |

These include y/z faces, edges and corners, unlike the earlier x-pair test.
Peak allocated GPU memory was below 0.82 GiB for these small tests. Do not
extrapolate their memory footprint to the complete cap. Full-NGC benchmarking
is a separate measurement and generates no publishable posterior draw.

The full NGC benchmark passed with 1,626 conditions and four simultaneous
draws: 1,274.93 seconds preparation, 97.90 seconds for one Heun step in both
factors, 27.98 GiB peak GPU allocation, fine shape 320 x 432 x 288. Benchmark
NFE is 2, not a posterior product. The production target is NFE128; the rough
linear timing estimate is about 126 minutes including preparation, with no
guarantee of exact scaling. Bounded shared-QOS submissions use one GPU each,
3h30 for NGC and 2h30 for SGC, without automatic retries.

Submitted NGC job 59178507 and SGC job 59178509 from frozen Git revision
`2c80b4b9b829c8b678137363c2266083dd775e4d`, archived under the evidence root
as `source_2c80b4b9b829c8b678137363c2266083dd775e4d`. All five tests also
passed in that archive. Launch checks bind source hashes to the measured
benchmark. Development allocation 59177348 was relinquished after submission.
Both jobs completed successfully: NGC 2h02m50s, SGC 1h03m29s. Receipts confirm
four NFE128 draws per cap, finite positive supported density, and double-precision
block-mass errors below 4.80e-16. No retries or retraining were needed.

Independent stored-field verification on CPU allocation 59192470 passed all
3,868,333 NGC and 1,568,080 SGC catalogue positions, with all eight interpolation
corners covered. Float32 storage mass errors are below 1.69e-8. The website
package preserves 1,382,456 NGC and 611,897 SGC supported display cells on one
regular grid per cap; no per-tile mesh decomposition or seam smoothing is used.
Existing atlas selectors now target these full-cap fields. Old environment
probabilities are kept in separate galaxy-only views, not attached to new draws.

The website sparse-grid decoder passed dense-grid mesh parity: 8,272 triangles,
maximum coordinate rounding 0.000061 Mpc/h. Full-cap browser validation passed
both means and all eight draws, 100,000 galaxy overlays per cap, typed density
thresholds, fixed physical axes, cap revisits, mobile layout, and retained local
and galaxy environment views. No browser errors or external requests occurred.
The default mean surfaces at density 2 have 704,960 NGC and 303,970 SGC triangles.
Screenshots show the full fields without the former tile mesh cuts. Receipt:
`browser_qa/QA.json` under the evidence root. Native private deployment succeeded
for website source `ecca529dfce76f695636688ca95fd4cb009f3d25`, deployment
`appgdep_6abf66a1e0248191914707cbc07eb73a`, at
`https://loa-cosmic-web.astrodak99.chatgpt.site`. The NGC/SGC atlas selectors
now serve the coherent fields; the default is NGC atlas. Owner-only access was
preserved. To meet the hosting size limit, 152 unused generated chunks were
moved out of the static tree into `unused-site-assets` under the evidence root;
all referenced assets and original scientific data remain available.
CPU allocation 59192470 was relinquished after the completed update.

Evidence root:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/loa_global_canvas_20261002_v1`.

## Scientific limit

Consensus changes the learned probability flow. The prior two-window mock
calibration results do not validate the larger many-window conditioning law.
Exact coherence, mass conservation, and numerical stability are engineering
gates only. Any full-cap output remains an exploratory reconstruction, not a
calibrated full-survey posterior or a new production galaxy-environment VAC.

## Environment correction

Graphify is available inside `cosmic_env` (also reported by the user in
`rapids-gnn`). A bare-shell lookup is not an availability test. The activated
query was run successfully; an atlas-specific query found no indexed node.
An activated compute-node index refresh failed at `fcntl.flock` with filesystem
error 524; this is not evidence that Graphify is unavailable.
