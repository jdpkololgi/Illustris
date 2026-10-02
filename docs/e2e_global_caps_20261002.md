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

The website sparse-grid decoder passed dense-grid mesh parity: 8,272 triangles,
maximum coordinate rounding 0.000061 Mpc/h. Full-cap browser validation awaits
the actual inference products; no full-cap website update is published yet.

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
