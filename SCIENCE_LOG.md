# SCIENCE_LOG.md — Shared Science Log (Illustris ⇄ GraphWeb_DESI)

> **Scope:** Durable scientific results, negative results, bug discoveries, and methodological decisions **only** (newest first). Live state lives in `CONTEXT.md` (≤ 8 KB, overwritten); Slurm submissions, queue telemetry, tmux owners, and routine receipts go in `RUN_LOG.md`. Pre-November 2026 entries are archived verbatim in `docs/log_archive/SCIENCE_LOG_2026_pre_nov.md`.

## 2026-11-23 — Both Paper 3 confirmation phases have verified observation inputs

The four native and factor-four cap responses for ph018 and ph019 passed
the frozen development-window parity check, exact angular replay and count
conservation checks. Supported BGS_BRIGHT source rows at 0.01 ≤ z < 0.15:
ph018 NGC 1,713,232; ph018 SGC 692,642; ph019 NGC 1,686,768; ph019 SGC
692,086. These are catalogue rows in the input, not independent replicates
or counts of scored cells. The largest relative change across the four caps
in either total counts or random-expected counts after coarsening was
3.44e-10. Evidence: the four `RESPONSE_COMPLETE.json` receipts under
`$SCRATCH/abacus/e2e_field_v3/lowz_cfm_blind_confirmation_v1/{ph018,ph019}/{NGC,SGC}/`;
each binds the frozen release, observation and angular payloads, selection,
and two output hashes. The interrupted file was not used. No matter truth
was read and no posterior predictions were generated. Geometry registration
and observation-only prediction remain prerequisites to any blind score.

## 2026-11-23 — Paper 3 blind confirmation authorized; no test outcomes yet

The user released ph018 and ph019 for one test of the frozen seed-17 **CFM**.
The release does not change the shared training split or open either phase
for Paper 2 model selection. Register observation/random-derived geometry;
freeze the CFM and matched baseline predictions before reading matter truth.
32 draws per cap per phase, 128 total. No retuning, DESI inference, or phase
replacement. The two phases, not the draws or cells, are the independent units.
The corrected native and factor-four response products must first be built:
an older observed catalogue or angular map is not by itself a CFM-ready input.
Release: `configs/lowz_cfm_confirmation_release_v1.json`; it binds the existing
development freeze without overwriting it. Authorization is not a calibration result.

## 2026-11-23 — CFM matched observation benchmark and draw diagnostics (Paper 3)

| Metric | Split type | Phase/sky set | z range | Artifact |
|---|---|---|---|---|
| CFM fair CRPS skill over training-only smoothed counts: 16.8% density, 22.0%, 20.3%, 26.1% ascending eigenvalues | development, 344,302 supported registered cells, cell weighting | ph006/012/013, NGC+SGC | [0.01,0.55) | `lowz_finish_scorecard_61848915/SCORECARD.json` |
| Empirical 5–95 percentile coverage CFM .869/.850/.855/.847, reference .857/.865/.853/.857; CFM narrower in all four fields | same | same | [0.01,0.55) | `lowz_count_reference_61848915/MATCHED_PERCENTILE90.json` |
| Regional shear-spread error −0.97/−0.96/−1.32 posterior SD; 24/32, 25/32, 14/32 cores covered | development, historical full-core geometry, not supported-voxel metric | ph006/012/013 respectively, both caps | registered science pairs, not a per-voxel z cut | `lowz_finish_scorecard_61848915/SCORECARD.json` |
| Highest local k-bin median pair draw/truth power 1.22/1.22/1.11 | development, supported paired cores, not whole-cap P(k) | same phase order | [0.01,0.55) | `lowz_dev_density_shapes_61848915/SHAPES.json` |

Artifacts under Scratch `abacus/e2e_field_v3/`. Benchmark fitted on 20
training phases with phase-held-out residuals; per-cell benchmark is not a
coherent density-to-tides generator. Phase is the replication unit. No DESI
inference or mock retraining. Formulas, limitations and paired figure links:
`docs/paper3_lowz_cfm_development.md`.

Closure: mean signed bias in posterior SD, original / matched true-density
control / remainder, is +.161/+.107/+.054 for λ1 and −.143/−.087/−.056 for λ3
(development, supported registered cells, ph006/012/013 both caps, [0.01,0.55),
`lowz_dev_oracle_density_v1/CLOSURE_READOUT.json`). For regional shear spread
on the historical 96 cores it is −1.084/−.723/−.361 SD. True density replaces
finite model cells only; same frozen exterior, no true exterior. One exterior
seed per phase/cap; combined operator/exterior diagnostic, not a correction.

## 2026-11-22 — CFM individual-draw contrast and mock-to-Loa input check

For 48 registered core pairs (ph006/012/013, both caps, supported cells,
0.01≤z<0.55), the median across pairs of median-draw/truth spatial SD is
0.983; the corresponding 95th-minus-5th percentile ratio is 0.996. These
are local contrast checks, not recovered phases of Fourier modes or evidence
for joint calibration. Individual draws, truth, mean and SD are shown on
matched slices. Original calibration replayed exactly; no new CFM draws.

On common NSIDE256 cap/PHOTSYS random-supported pixels, Loa v2.1 quality25
BGS_BRIGHT / actual ph006 Kibo-v1 observed CFM mock counts in five shells
(.01,.15,.25,.35,.45,.55) are SGC 0.962/1.031/1.086/1.277/1.964
and NGC 1.030/1.027/1.113/1.274/1.918. One phase and one random per
release: not a selection correction, equivalence test, nor DESI calibration.
Do not treat adding low-z tracers as resolving high-z population mismatch.
Keep a matched-observation transfer gate distinct from sealed mock tests.
See `docs/paper3_lowz_cfm_development.md` for the keyed metrics and receipts.

## 2026-11-22 — coherent development CFM: marginal coverage versus regional reliability

Evidence A: `$SCRATCH/abacus/e2e_field_v3/lowz_calibration_61647276/CALIBRATION.json`.
32 independent sampler seeds per cap/phase, observation-only conditioning;
full-box truth for scoring. Central order-statistic interval (3rd to 30th
draw) has nominal coverage 27/33. Values ordered as density, λ1, λ2, λ3.

| metric | split_type | phase_set | z_range | artifact_id |
|---|---|---|---|---|
| Coverage 83.95%, 81.86%, 82.33%, 81.48%; mean widths 0.6109, 0.2367, 0.2512, 0.2933 | development, supported registered cells, cell-weighted, both caps | ph006/012/013 | [0.01,0.55) | A |
| Coverage 87.01%, 83.52%, 85.29%, 84.41%; 81,783 scored cells | same development cell weighting | ph006/012/013 | [0.01,0.15) | A |
| Fair CRPS 0.1196, 0.0474, 0.0508, 0.0620 | development, supported registered cells, cell-weighted, both caps | ph006/012/013 | [0.01,0.55) | A |

Regional density TARP at nominal 0.80 (reference width 0.25): 0.866 for
two-core means, 0.762 for eight subregions; at width 1.0: 0.847 and 0.792.
Among 248 cells assigned knot probability ≥0.8, mean prediction 0.900,
true knot frequency 0.802. These are descriptive findings across three
independent mock phases, not 344,302 independent trials. Not a full-field
calibration proof or a comparison against a matched baseline.

## 2026-11-22 — corrected encoder comparison and transformer continuation

Freeze the seed-42 window-attention epoch-1 and P12-A epoch-4 validation
scorecards in docs/evidence/p12/lowz_encoder_epoch1_window_attention.json
and lowz_encoder_epoch4_p12a.json. These cover all three physical eigenvalues,
three development phases and four shells. At epoch 1 U-Net leads on macro
lambda1 R² (0.435058 vs 0.381385); do not rank its epoch 4 against epoch 1.

Approved continuation 61786757 submitted on one 80 GB A100, 32 CPUs,
desi_g/shared, 24h, Scratch, no requeue. Pending priority at submission.
Resume in the original 61664810 output using pinned 71307f5, from the
epoch-2 pause at cursor 300912 / 346364, step 647276; no preflight replay.
Do not count historical graph job 61662008 (p10_multiphase/training_contract)
in the corrected z=0.01–0.55 table. Corrected host-cone job is 61685645.
At user request, historical job 61662008 was cancelled; Slurm accounting
confirmed CANCELLED after 04:51:39. Its products are excluded. Corrected
jobs 61685645 and 61786757 remained pending and were not modified.

## 2026-11-20 — first corrected host-cone G-PATCH production segment submitted

Slurm accepted **61685645** (initial state PENDING/Priority; no start
estimate). Launcher `8893f77` stages the benchmarked trainer `449c60d`,
explicitly calls `main(host_cone=True)`, and checks both the eight-patch
equivalence and 256-update/24-group timing receipts plus five data bindings.
Fresh seed 42 on corrected z=0.01–0.55 data; four-epoch learning-rate
schedule with only this first segment approved. One A100 (40/80 GB eligible),
32 CPUs, desi_g/shared (Slurm gpu_shared), Scratch, 24 hours; Requeue=0.
Checkpoints every 256 updates. No automatic retry, dependency, or second
segment. Log: `$SCRATCH/abacus/gpatch-lowz-s42-61685645.log`. Output root
on start: `$SCRATCH/abacus/lowz_gpatch_s42_61685645`. At the measured
rate the first segment is expected to cover ~14% of an epoch, not produce
an epoch validation score. Submission does not establish optimizer progress.

At submission, P12-A epoch 3 was running (>186k/346,364 updates); epoch 4
was waiting on it. Current corrected full-development reference is epoch 2
(λ1/λ2/λ3 macro R² 0.509229/0.567746/0.577865). Corrected large-kernel,
shifted-window and LitePT jobs remain queued. No matched multi-encoder
ranking yet; do not count historical graph runs or technical benchmarks.

## 2026-11-20 — exact host cone passes matched benchmark; prefer for next G-PATCH fit

Tmux-owned job `61683377` completed 0:0 in 14m13s and released its 80 GB
A100. Two ph000 training patches and six ph006/012/013 development groups
passed the real-data topology, feature and ownership checks before timing.
Maximum absolute difference in frozen-checkpoint scored predictions was
2.98e-7. Same seed, 256 completed updates; the registered 64→256 interval
took 339.59s vs 475.60s (0.5654 vs 0.4037 updates/s). **1.40× throughput,
28.6% less training time**, faster in all twelve matched 16-update windows.
At 346,364 updates/epoch this projects to 170.2 vs 238.3 single-A100 hours
before validation, not an observed full epoch. Single benchmark per variant,
not interleaved; cache and device effects are not isolated.

Same 24 validation groups, cores, row/node counts and 217,200 scored
galaxies against the baseline. Calls took 158.70s vs 217.39s (1.37×);
retained 72.52% of sampled directed edges. Slurm sampled 9,686 MiB GPU
(baseline 10,746 MiB), while host RSS remained ~56.1 GiB for both;
the cone run includes its additional oracle. No full-panel validation
cost, trained-model R² or ranking is established. Use the host cone for
the next **separately approved** production segment, with these bindings;
the previously prepared canonical-only launcher is not that configuration.
No production job was submitted. Full receipts:
`docs/evidence/p12/lowz_gpatch_host_cone_61683377.json` and
`docs/evidence/p12/lowz_gpatch_gpu_gate_61675946.json`.

## 2026-11-20 — matched G-PATCH host-cone benchmark allocated

User selected the one-hour host-cone benchmark, not production. Interactive
job `61683377` on `nid200304` is owned by tmux `lowz_host_cone_1120` on
`login16` (one 80 GB A100, 32 CPUs, desi_g/shared_interactive, Scratch).
Zero existing interactive slots were occupied before launch. The wrapper
ran with `--run 449c60da50e1d567aa8f007400c3fa1ee4391d35 --host-cone`;
base GPU receipt hash and corrected loader bindings passed before launch.
Scratch evidence: `$SCRATCH/abacus/lowz_gpatch_gpu_61683377/`. Read the real
patch gate before the 256-update/24-group timing comparison. No automatic
retry or production handoff; the allocation exits when the step ends.

## 2026-11-20 — first-segment launcher for the passed G-PATCH configuration

Prepared but not submitted: `workflows/abacus_tweb/lowz_gpatch_first_segment.slurm`.
The fresh seed-42 run pins receiver-streamed trainer `6b126dc` and the
canonical host extractor from diagnostic 61675946, not the optional host
cone. It preserves the four-epoch schedule and 256-update checkpoint
cadence in a new job-specific Scratch directory. A 24-hour single-GPU
segment may end with a budget pause, not an epoch score: accept exit 75
only alongside a nonempty checkpoint and internally consistent pause
receipt. No dependency chain, automatic retry or requeue. Shell syntax
and the read-only `--check` passed against the six hashed live receipts/data
bindings; no production workload has run by virtue of this check.

## 2026-11-20 — qualify the optional G-PATCH host extractor before pricing it

The corrected GPU diagnostic now accepts an explicit `--host-cone` mode;
the default remains the full canonical CPU extractor. The mode first verifies
the exact 61675946 baseline receipt and frozen 256-update checkpoint. On two
training cores (opposite caps, ph000) and six development groups (one per
phase/cap, largest non-stress group from the original timing panel), require
identical node IDs, features, ownership masks, ordered directed IDs,
endpoints and attributes on the independently reconstructed three-hop
receiver set. Require the frozen network's owned-row predictions to agree
in FP32 (rtol 2e-4, atol 3e-5). Only then time the same 256 seed-42
updates and 24 full/stress sample groups; reject changes to grouping, core
IDs, row/node counts, epoch length, or timing window before reporting speed
ratios. This compares total runtime including CPU loading; cache order and
single-run variability still apply. Stochastic optimization is not expected
to be bitwise identical between extractors. Three focused single-thread CPU
tests, Python syntax, and shell syntax passed. No real-patch result, speedup
claim, compute submission or change to the production configuration here.

## 2026-11-20 — corrected receiver-streamed G-PATCH GPU diagnostic passed

Job 61675946 completed 0:0 in 18m07s (single 80 GB A100, revision
`6b126dc`, canonical CPU extractor). All 256 optimizer updates completed;
checkpoint cursor/optimizer schedule checked and file SHA-256 independently
rechecked: 6f67ae9e3e632ae0fe3c438ee707c8f4801f347e59c673f20aadaadee7c835cf.
Updates 64→256 took 475.60s (0.40370 updates/s). Applied to 346,364 updates,
the short-window projection is **238.3 A100-hours per epoch before
validation**. Four epochs would be ~953 training-only hours at that rate;
24h buys ~34.9k updates, ~10% of an epoch. Dense patches vary greatly;
these are budget estimates, not a convergence prediction or a commitment.

All 24 sampled development groups passed independent P12-A parent-row
identity, finite output, shape and uniqueness checks (four per phase/cap
including all six max-owned stress groups). Total 217,200 owned rows;
largest patch 89,317,234 directed edges. The sampled inference calls summed
to 217.39s; deliberately nonrandom group selection is not a full validation
timing or an R² estimate. All 21,876,905 rows must still be scored for ranking.
PyTorch peaks: 4.51 GiB training, 5.48 GiB including inference; Slurm
sampled 10,746 MiB device and 58,796,972 KiB host (~56.1 GiB). Either
40 or 80 GB is reasonable for a bounded segment; full-panel memory fit is
not guaranteed. No production job or automatic continuation authorized by
this diagnostic. Full receipt: `docs/evidence/p12/lowz_gpatch_gpu_gate_61675946.json`.

## 2026-11-20 — exact host-side G-PATCH cone prototype (not the running configuration)

Receiver-streaming job 61675946 passed the former 47,196,542-edge failure
and recorded 144 updates with a cumulative 4,841,787,392 B PyTorch peak.
Sampled host RSS reached 49,702,100 KiB. The existing CPU extractor still
collects incident IDs for every four-hop node, looks up endpoint pairs,
tests membership, and duplicates all induced edge attributes in both
directions before the GPU discards irrelevant messages.

An **optional**, unbound extractor in `lowz_graph_host_cone.py` uses the
bidirectional two-pass dependency instead: retain node features through
distance four and all incoming edges at receivers through distance three.
It queries incidence only through distance three and gathers directed
IDs/attributes in blocks, preserving the canonical forward-then-reverse
order and reverse-feature convention. Only the declared owned outputs
are equivalent; outer-halo node outputs are not scientific readouts.
No neighbor sampling. On a small branched graph with a fourth-shell
lateral edge and an isolated scored node, two CPU tests passed for node
and directed-edge identities, edge attributes, both owned scores and all
parameter/input gradients (dropout off; removed edge gradients zero);
mixed caps rejected. Real mock equivalence, timing, memory and stochastic
training behavior have **not** been measured for this variant. It is not
enabled in the running test or selected for production.

## 2026-11-20 — corrected G-PATCH: 47.2M-edge failure and receiver-streamed repair

Job `61675084` failed (1:0, 5m59s). The corrected replay passed the
20,906,438-edge patch and 16 completed optimizer updates, then hit a
110,789-node/47,196,542-directed-edge patch. The concatenation of all
weighted edges requested 14.07 GiB with 13.83 GiB free on the 80 GB A100.
The first repair had checkpointed internal activations but still constructed
a patch-wide E×80 tensor. The 64→256 timing interval and all validation
groups are absent. Do not infer R² or an epoch cost from this failure.
Source/log: aedd3c7, `$SCRATCH/abacus/lowz_gpatch_gpu_61675084/probe.log`.

For dense patches the new corrected-only path sorts directed edges into
complete receiver neighborhoods. Each group computes its own exact
incoming attention normalization and contributes to both endpoint sums.
It recomputes first-layer edge states with the same dropout seed where
needed by layer two; it does not store patch-wide edge latents. The required
receiver sets are traced backward from the declared output rows, including
the incoming normalization neighbors of outgoing messages. No neighbor
sampling, change to supervised rows, gradient accumulation across optimizer
steps, or modification to the historical graph or live grid trainers.
Rows not requested from the partial readout are marked NaN. The graph
index and raw features still scale with patch size; compute-node time,
host memory and end-to-end GPU fit remain unverified.

Six one-thread CPU tests passed: reference outputs and parameter/input
gradients on partial and complete readouts; receiver closure when degree
exceeds chunk size; checkpointed dropout replay versus uncheckpointed
streaming; and an operator trace detecting patch-wide edge latents in
forward/backward (none on the tested graph). The diagnostic now preserves
16-update timings and CUDA high-water marks even on later failure.
The previous job released its allocation; no automatic retry or production.

After separate approval, Slurm allocated the receiver-streaming diagnostic:
`61675946` on `nid200301`, tmux `lowz_gp_stream_1120`, one 80 GB A100,
32 CPUs, desi_g/shared_interactive, Scratch, one hour. CPU gate and input
hashes passed before launch. Wrapper `lowz_gpatch_gpu_step.sh --run
6b126dc329474f6c80035bc3821b20fddee93b13`; artifacts under
`$SCRATCH/abacus/lowz_gpatch_gpu_61675946/`. No retries or production.

## 2026-11-20 — dense corrected G-PATCH activation OOM; bounded-activation repair

Job `61673881` failed (1:0, 2m53s). It was an **80 GB** A100. Before
the first 16-update log, the training forward requested a 6.94 GiB concatenation
of edge/sender/receiver activations; PyTorch already held 71.06 GiB and
the device reported 3.61 GiB free. No 256-update checkpoint, measured rate,
validation group, or R². See `$SCRATCH/abacus/lowz_gpatch_gpu_61673881/probe.log`.
No automatic retry. Reducing validation group size would not fix this training failure.

The corrected graph wrapper now uses the original network up to 65,536
directed edges, then chunks edge MLPs and attention logits and checkpoints
edge computations and message passes. Receiver softmax and both node sums
are still global across the patch. No change in graph, selected galaxies,
parameter shapes, objective, or supervised core per optimizer step.
In three small CPU tests, FP64 outputs and input/parameter gradients agreed
with the original in evaluation and dropout-free training (rtol 1e-9,
atol 1e-11). With dropout active, chunked checkpoint versus uncheckpointed
chunked training reproduced gradients and RNG state. Chunking does **not**
promise the original monolithic dropout realization. These tests do not
establish GPU memory fit or throughput. A new diagnostic needs approval.
For interpretation of future inference timing: the full panel uses
4,318 / 4,326 / 4,348 four-core groups in ph006/012/013; the diagnostic
oversamples dense groups and is not a random validation-time sample.

FP32 CPU mechanism check at repair `aedd3c7`: seeded (42) 64-node,
769-directed-edge random graph, full 80-channel/8-head/two-pass models,
dropout off, chunk width forced to 64 to exercise 13 chunks. The maximum
absolute prediction difference was 1.34e-7; over 136 input and parameter
gradient tensors it was 1.49e-8 (squared-output mean loss). Autograd
saved-tensor hooks around forward, deduplicating backing storage and
excluding parameters/inputs, saw 8,982,480 bytes for the original and
553,600 bytes for the repaired model (16.2× less). This measures tensors
retained for backward, **not** transient peak allocation, the GPU allocator,
or a bound for a real patch; the fresh GPU gate is still required.

After separate human approval the repaired diagnostic started: job
`61675084` on `nid200256`, tmux `lowz_gp_repair_1120`, single 80 GB
A100, 32 CPUs, desi_g/shared_interactive, one-hour cap, no automatic
retry. The step rechecked the view and transform and invokes
`lowz_gpatch_gpu_step.sh --run aedd3c76884e7d58946fbd2b5a6f1054ccd0327e`.
Log and measurements: `$SCRATCH/abacus/lowz_gpatch_gpu_61675084/`.
Allocation verified; no test result or production fit implied.

## 2026-11-20 — corrected G-PATCH loader passed; GPU diagnostic started

CPU job `61668072` completed (0:0). The read-only launch gate passed for
the corrected 23-phase view: both cap probes per phase, the exact 20-phase
training-only normalization set, and the view and transform file hashes.
Receipt: `$SCRATCH/abacus/lowz_graph_adapters_61668072/GRAPH_LOADER_CHECKED.json`.
The parent Delaunay boundary audit finds 536/485/498 supervised galaxies
directly adjacent to excluded parent rows in ph006/012/013: 1,519 of
21,876,905 development galaxies. This is a geometry diagnostic, not leakage
of validation labels or a measured effect on R².

The conditional one-hour diagnostic was allocated as `61673881` on
`nid200269`, one A100, 32 CPUs, desi_g/shared_interactive, tmux session
`lowz_gpatch_gate_1120`, wrapper revision `c84a4fca`. Output root:
`$SCRATCH/abacus/lowz_gpatch_gpu_61673881`. Scope: 256 matched training
updates and 24 sampled validation groups; no full-panel score or production
launch follows automatically. The allocation start alone is not a test pass.

## 2026-11-20 — conditional corrected G-PATCH GPU handoff

`workflows/abacus_tweb/lowz_gpatch_gpu_step.sh --check` is a read-only
pre-allocation gate: require 61668072 COMPLETED 0:0, the 23 expected
visible phases, transformed patches in both caps of each phase, the exact
training-only fit set, and matching graph/transform receipt hashes.
`--run <revision>` stages a detached source checkout and executes just the
256-update / 24-group diagnostic in the approved GPU step. Shell syntax
passed; the live pre-allocation check correctly refused the nonterminal
CPU job. This is not a positive loader pass or a GPU launch receipt.

## 2026-11-20 — which P12-A epochs can support the full matched scorecard

Epoch 1 completed all three development phases, but the original full
per-phase/per-shell/per-eigenvalue JSON is not in the production run
directory: best_validation_report.json is now epoch 2. The epoch-1 history
does preserve the primary λ1 macro (0.4350582585) and its four phase-averaged
shell scores (0.508111, 0.482167, 0.459674, 0.290280). These are usable
for a like-for-like primary-score comparison; they cannot reconstruct the
36 detailed scores. Do not fill the missing cells with best-epoch values.
The original pre-validation epoch-1 checkpoint still matches migration.json:
SHA256 6ceb15b449e0751757dfecff082af98becb7093f1c96e90d0aaac56e98bd5231,
cursor/step 346364. It permits an isolated validation-only replay should
the full epoch-1 panel become necessary. No replay is required for the
full epoch-2 comparison: live report hash 8ab9a5816146a9b13da20d638a57ef09da85d04711ae7182839b7a19bfba4b0c
matches `docs/evidence/p12/lowz_encoder_epoch2_p12a.json`.

## 2026-11-20 — measured reason to retain 80 GB for historical G-PATCH

Historical ph006 validation + 500 resumed updates (61659392.0) used
48,356 MiB peak GPU memory in Slurm. Its epoch-3 report records a cumulative
45,276,058,112 allocated CUDA bytes and a largest validation group of
8,551,158 directed edges. Both exceed-40-GB memory readings concern the
historical run, not a measurement on the corrected contracts. Leave
61662008's existing 80 GB constraint intact; no scheduler change.

The approved corrected GPU preflight still evaluates eight real groups
per development phase (four per cap). Within that count, include each
cap's largest group by owned row count and its final partial group where
present. An independent enumeration using only P12-A ownership metadata
confirmed that all six maxima are included (SGC/NGC: ph006 37,659/38,198;
ph012 26,510/35,403; ph013 27,749/49,520). Row counts are not upper bounds
on four-hop edges or VRAM. Record training throughput before inference and
record actual device capacity and per-group peak allocation; a successful
training-only timing receipt is not a pass of the grouped validation gate.

## 2026-11-20 — first corrected LitePT production segment submitted

After explicit acceptance of stochastic rather than numerically identical
restarts: job **61671100**, fresh seed 42, measured trainer fc0b407,
launcher/readout 1d198c7, 4-epoch cosine schedule. Slurm confirms a single
40/80 GB A100, 32 CPUs, desi_g/shared, Scratch, 24 h maximum and no
requeue. Initially pending for priority. No previous technical weights
were imported, no follow-on job authorized. Separately approved: one
one-hour conditional G-PATCH GPU check, only after CPU 61668072 completes
successfully and its full transformed-loader receipt verifies. No corrected
G-PATCH GPU allocation or held-out R² at this check.

## 2026-11-20 — measured GPU cost at matched P12-A scorecard epochs

For the cost-vs-R² comparison use actual allocation elapsed time, not
the epoch-history invocation timer: epoch 1's history was written by the
validation-only recovery process. Slurm allocation rows (one A100 each):
61548250 43,332 s, including the failed original validation attempt;
61581030 6,163 s, successful validation at the same 346,364 updates;
61581047 47,368 s, epoch 2 train + validation. Thus the λ1 shell/phase
macro R² points are 0.435058 at 13.749 cumulative allocated A100-h and
0.509229 at 26.906 h (13.158 h incremental). Excludes queue time,
preflights, CPU data preparation and the ongoing epoch 3; includes startup,
validation, and the actual recovery cost. These are allocation costs, not
kernel-active GPU times or retrospective best-epoch comparisons.

Grid probe 61659884 ran all three 64-update tests in **one** 225 s Slurm
step. Run-manifest to canary-marker UTC deltas are 17.39 s reference,
12.61 s large kernel, 10.46 s window attention, including checkpoint I/O
but excluding per-arm setup before manifest creation. Fixed test order,
cache effects and absence of a full validation pass preclude treating
these intervals as per-epoch cost or evidence one architecture is faster.

## 2026-11-20 — corrected low-z graph unions complete in all 23 visible phases

CPU 61664731 completed 0:0 in 52m27s. Independently rechecked the phase
set against the authoritative 20 training + 3 development assignments, all
23 phase manifest hashes against their receipts, context counts, 14.78 Mpc
radius, and addition of retained Delaunay and radius-only pair counts.
Totals: 168,851,467 context nodes and 7,507,944,999 undirected pairs summed
over phase-local graphs; context nodes are not the supervised evaluation
row count. No sealed phases. Scratch `abacus/lowz_graph_unions_61664731/ALL_UNIONS_READY.json`,
SHA256 8514f7f16bd2eb93059b33e8f00f43dd8fe942fd04f024f214851be6f818edd5.
The downstream job 61668072 can now build and validate patch indices and
training-only normalization; this union receipt is not an encoder score.

## 2026-11-20 — retain full per-epoch point-model reports for matched ranking

The G-PATCH overlay links the same base readiness marker, target transform
and phase contracts as P12-A, so a distinct graph feature transform does
not require weakening the scorecard's row/split checks. The original
trainer only retained a detailed report when the selection score improved;
the history row alone cannot reconstruct all 36 phase/shell/eigenvalue
scores for a non-best epoch. New G-PATCH and LitePT production entrypoints
now archive the existing full report at each completed development evaluation.
No change to training, selection, the sealed split, or running/queued jobs.
The reader prefers that exact epoch's archive and still cross-checks it
against the completed epoch history. Two focused tests passed. Replaying
the real P12-A epoch-2 report through the archive and reader reproduced
all 36 scores and the frozen macro triple exactly, without new inference.

## 2026-11-20 — native LitePT shared-checkpoint replay and epoch cost

GPU job 61668057 completed 0:0, 9m33s. At update 40 both cold processes
received the identical checkpoint (SHA 5e33b9c2a5b9c0a4e0de68467672ab8b9b9f499a92df3b6dd5de7d834ac43e9a).
All three reached update 48 with identical Torch CPU/CUDA RNG states,
scheduler, shell-row counts, and first resumed loss. Final update losses
(warm/B/C) were 2.082227/2.084267/2.084149. The strict model/optimizer
replay gate **failed**, even cold vs cold; this rules out a discrepancy
unique to the warm-vs-resume boundary, but does not establish that resuming
has no effect on R². No specific kernel has been identified as the cause.
Receipt: `abacus/lowz_litept_fork_61668057/FORK_DIAGNOSTIC_COMPLETE.json`.

384 native training updates took 218.434 s (1.758/s); the matched 346,364
updates project to 54.7 A100-hours/epoch **before validation**. Max allocated
tensor memory at update 512 was 16,569,569,792 bytes. The staged first-segment
launcher uses the measured source fc0b407 and the original 4-epoch LR
schedule; it starts clean, not from a diagnostic fork, and authorizes no
continuation by itself. A single 24-hour segment is not expected to produce
an epoch score. Interpret any accepted continuation as stochastic.

## 2026-11-20 — widen grid GPU eligibility; preserve LitePT voxel identities

The first 64 matched core updates of the two grid variants covered an
80³ patch (equal to the largest spatial patch across train/development).
Peak allocated tensor memory was 2,139,496,448 bytes for large-kernel CNN
and 1,559,075,328 bytes for window attention. The largest supervised
population in those updates was 20,495 vs 28,816 over the whole contract;
this is headroom evidence, not a 40 GB whole-run guarantee. With explicit
approval both still-pending jobs (61664814 / 61664810 respectively) were
changed in place to gpu&a100. Slurm verified original IDs and submit times,
one GPU/32 CPUs each, desi_g/shared, 24 hours, Scratch, Requeue=0. Prefer
40/80 GB eligibility for future A100 requests unless memory warrants 80 GB.

LitePT voxel packing now sorts occupied integer keys instead of sorting
three-column rows and sums features in input order using indexed bins.
The reference test compares coordinates, occupied-voxel count, grid order,
averaged features, and every galaxy's inverse index for float32/64, repeat
points, negative coordinates, and half/whole-Mpc cells. All arrays agreed
exactly. A single-thread, synthetic 20k-point microbenchmark took 3.58 ms vs
28.48 ms for the old packer (best of three two-call pairs). This is a CPU
preprocessing measurement only: no change to the pinned LitePT run, no
end-to-end GPU speedup or held-out score claimed.

## 2026-11-20 — graph qualification batch queued; LitePT common-checkpoint timing launched

G-PATCH adapter/normalization batch 61668072 (desi/regular, six-hour cap)
is held behind afterok:61664731 and pins c3b9712. The upstream union job
has verified 15/23 visible phases at this check. Native LitePT probe
61668057 started on nid200376 in a tmux-owned one-hour window, diagnostic
731f5d4 against original trainer fc0b407. Neither job is a production
encoder fit. Future G-PATCH GPU preflight entrypoint prices the actual first
256 matched updates and checks physical row ownership for eight grouped
validation samples in each development phase, without computing subset R².
Group-identity and final partial-group smoke checks passed; GPU not submitted.

## 2026-11-20 — corrected ph002 graph passes independent patch extraction

CPU 61666488 completed 0:0 in 13m26s (step MaxRSS 7,776,624 KiB).
The checked phase contains 329,057,667 undirected edges. SGC core 2808:
74 P12-A loss galaxies, 2,230 context nodes, 31,064 directed messages.
NGC core 12349: 452 loss galaxies, 7,049 nodes, 161,356 messages.
In each case a separate four-scan traversal of the source pairs recovered
the identical node set and induced edge IDs, and loss rows matched the
corrected P12-A core exactly. The 4.2 GiB index reuses the radius files
in place. Receipt `abacus/lowz_patch_diagnostic_61666488/ph002/PATCH_GEOMETRY_READY.json`;
union SHA a032c6ad0446bb122a127dd517f057745a9a081c1d52024a3dbf78f615ed4ea9
independently rechecked. This proves two real patches, not all-phase
qualification, training-only normalization, or an encoder validation score.

## 2026-11-20 — stage visible-phase graph campaign and bound claim about inputs

`lowz_graph_campaign.sh` requires a complete 23-union receipt and passing
ph002 patch oracle before scaling. It builds the remaining adapters, fits
training-only features, binds the immutable P12-A core contracts, then
checks transformed CPU inputs for both caps on every visible phase. It
also records how many in-context nodes have direct parent-Delaunay edges
to excluded rows: filtering a parent triangulation is not a new
context-only triangulation. Same held-out galaxies and split do not mean
identical engineered inputs: score the encoder pipelines as actually run.
No full graph campaign or training result implied by this launcher.

## 2026-11-20 — isolate checkpoint boundary from independent-start drift in LitePT

New diagnostic starts from the existing uninterrupted step 32 and pins its
original trainer and native kernels. Capture step 40 while continuing to
48; compare it with two cold resumes from the byte-identical step-40 file
at the same path, including optimizer, model, RNG, sampler and accumulator
differences. Record warm/cold separately from cold/cold, with unchanged
1e-5/1e-7 tolerance. A timed 384-update interval (128–512) excludes startup
and validation and informs a production budget. Total 496 updates counted
including duplicate continuations. Not a held-out measurement or a retry
of the previous failed independent-start comparison.

## 2026-11-20 — corrected G-PATCH normalization and isolated trainer view prepared

`lowz_graph_normalize.py` samples 100,000 nodes and canonical edges per
training phase and cap (20 phases only), fits the historical seven-feature
Box-Cox / edge transforms anew, and uses the corrected P12-A observed
redshift-bin selection for feature eight. No held-out phases used for fit;
development is application only; no sealed phase is opened. Outside the
supported radial bins the selection curve itself is zero before the log
floor. Geometry uses phase-local observational parent-graph metrics,
not tidal labels. `lowz_gpatch_train.py` binds a separate Scratch graph
view to the exact corrected core, target, and split files and uses the
unchanged Arm-A objective and complete-phase scorer. Future CSR builds
fill RAM before one sequential Scratch write; the running pilot is unchanged.
Five focused tests passed (including a small synthetic transform fit).
No real 20-phase transform fit, G-PATCH optimizer update or R² yet.

## 2026-11-20 — LitePT fixed-input repeat diagnostic and graph pilot receipts

GPU 61666487 completed 0:0 in 44s. For the four ph002 cores (69/1/7/4
supervised galaxies), both modes gave identical repeated predictions, zero
gradient elements outside rtol 1e-5 / atol 1e-7, and worst absolute gradient
differences 4.66e-8 (native) vs 4.10e-8 (deterministic attention). No
optimizer update. This is not evidence of deterministic multi-step training
or that FlashAttention caused the earlier divergence. Source and raw records:
`abacus/lowz_repeat_diagnostic_61666487/REPEAT_DIAGNOSTIC_COMPLETE.json`.
CPU patch pilot 61666488 started on nid200175; independent four-hop audit
required before scaling up.

## 2026-11-20 — freeze two independent point-model diagnostic steps

One-hour CPU ph002 CSR/closure audit and 30-minute A100 native versus
deterministic-attention repeat probe, source 2a1aa99. Private Scratch
outputs; distinct receipts for complete measurements vs scientific success.
Neither task updates model parameters or authorizes a training launch.

## 2026-11-20 — LitePT matched-loop numerical gate failed before restart

61665541 FAILED 1:0 in 4m54s. Both trajectories processed 32 updates
(second split at 16); comparison did not pass. Step-1 objectives and
gradient norms match; step-2 preclip norms differ (3.94482994 vs
3.94480443). At step 16 the objectives are 4.99367952 vs 4.96163607,
already before resume. At step 32 the first checked discrepant weight
tensor had max absolute difference 0.0148534. Neither independent-run
determinism nor restart equivalence is established; do not relax the gate
or promote to production. Logs in `abacus/lowz_litept_matched_61665541`.
New fixed-input, fixed-weight, buffer- and RNG-restored gradient probe
isolates within-process repeatability and compares the upstream default
to explicit deterministic FlashAttention backward. This is a candidate
mechanism, not an attribution; no optimizer updates or retry submitted.

## 2026-11-20 — corrected graph patch-index pilot prepared

`lowz_graph_patch_index.py` indexes the filtered parent Delaunay table and
the existing two corrected radius-only files with one incident-edge CSR.
Keeping the radius tables in place avoids a second ~7.2 GiB copy on ph002.
The pilot compares both cap patches against a separate four-scan traversal
that never reads the CSR, checks the induced edges and P12-A loss rows,
and emits no readiness marker on failure. Synthetic disconnected/cyclic
multi-table test passed; real catalogue execution still required.
Raw parent-geometry features are not fitted or training-ready.

## 2026-11-20 — freeze a common epoch-2 comparison point before best-report turnover

The corrected P12-A checkpoint at update 692728, seed 42, scored all
21,876,905 authoritative rows on ph006/012/013. Equal-weight mean across
the 12 phase-shell cells: R² 0.509229 / 0.567746 / 0.577865 for λ1/2/3.
Full per-cell scores, counts, epoch-order, source-report and data hashes:
`docs/evidence/p12/lowz_encoder_epoch2_p12a.json`. The scorecard reader
checks the complete-phase scorer's reports and rejects unmatched epochs,
row counts, and evaluation bindings. One reference score is not a ranking.

## 2026-11-20 — matched LitePT-S adapter; corrected grid and CPU submissions

CPU batch 61664731 started; pending 61662009 and 61662012 were cancelled
with zero elapsed time. Replacements: large-kernel 61664814, shifted-window
61664810, from the original 64-update checkpoints, launcher 6f862c8.
The LitePT-S training entry uses the unchanged P12-A phase/core sampler,
weights, target scaler, optimizer, schedule and complete-phase validation;
only observation packing and the encoder differ. Private LitePT Python
resolves the old factorial module's astronomy imports via read-only fallback
to cosmic_env. Registered technical comparison: 32 uninterrupted vs 16+16,
weights and optimizer rtol 1e-5 / atol 1e-7. Tmux-owned interactive
request 61665541 (one 80 GB A100, up to one hour, no retries), launcher
fc0b407; pending for resources at submission check. No held-out LitePT score.

## 2026-11-20 — corrected grid snapshot and remaining phase-union build

Confirmed that the two grid jobs' original sparse checkout at 71307f5 omits
the relative split-policy file required by TRAINING_LOADER_READY. Adding
`configs` to the checkout restores the exact bound file: both preflight
source contracts verify and the actual loader opens the 20/3 split. No
change to model source, checkpoints, or data. New serial CPU batch launcher
reuses the verified ph002 product and builds the other 22 visible unions
from the tested pilot; it only emits the aggregate receipt after verifying
all 23 component manifest hashes. Sealed phases are excluded.

## 2026-11-20 — native LitePT-S eight-update GPU preflight passed

61662989 completed 0:0 in 2m48s including isolated dependency install.
LITEPT_PREFLIGHT_PASS.json verifies eight unique ph002 training cores, 954
authoritative galaxy rows, finite losses and gradient norms, and peak torch
allocation 3,556,175,360 bytes on the sampled patches only. Used upstream
PyTorch PointROPE fallback on GPU with FlashAttention and spconv; no pretrained
segmentation weights. Scratch environment and receipts under
`abacus/lowz_litept_preflight_61662989`. No held-out scores, convergence,
production checkpoint, full-epoch throughput or all-core memory bound implied.

## 2026-11-20 — corrected ph002 graph-union gate passed

61662908 completed 0:0 in 2m33s (step MaxRSS 29904108 KiB). Independent
manifest SHA-256 matches CORRECTED_UNION_READY. In ph002, 7,369,722 context
galaxies give 56,176,778 context Delaunay pairs plus 272,880,889 new radius
pairs = 329,057,667 union pairs across the two disconnected caps. This
validates the one-phase geometry/union rebuild, not yet an exact patch
adapter, fitted transform, G-PATCH training checkpoint or validation R2.

## 2026-11-20 — corrected ph002 graph and official LitePT-S preflights launched

CPU 61662908 rebuilds the corrected ph002 union against lowz_points_61662493;
initial geometry checks passed. GPU 61662989 provisions a private Scratch venv
then attempts eight ph002 updates with official LitePT-S. Two independent
two-hour interactive windows, no resubmission. Launcher 708e609. Terminal
gates: `abacus/lowz_graph_union_61662908/CORRECTED_UNION_READY.json` and
`abacus/lowz_litept_preflight_61662989/LITEPT_PREFLIGHT_PASS.json`.
Launch is not a validation result or a held-out R2 measurement.

## 2026-11-20 — corrected point index passes matched-parent gate; neither point model trained yet

CPU job 61662493 completed 0:0 in 4m08s: Scratch
`abacus/lowz_points_61662493/INDEX_READY.json`, all 20 training and 3 development
phases. Exact development parent sets match P12-A: 7,261,727 / 7,339,923 /
7,275,255 (006/012/013), including 7,102,471 below z=0.15. All authoritative
parents unique per phase, cap checked, 12 neighborhood extractions per phase;
sealed 018/019 not read. Old ph006 union has 6,248,640 context nodes vs new
7,329,765. Rebuild corrected radius union and exact graph adapter; do not reuse
old context or normalization. Official prs-eth/LitePT source pinned at
436d04801c8151faebe66a1b2d368a9711e7e6aa. Regression wrapper uses inverse
voxel mapping and P12-A observation channels; sparse environment/GPU checks
pending. No model comparison or validation scores from point-index success.

## 2026-11-20 — three encoder production segments submitted

Launcher 53ee521: historical G-PATCH 61662008, corrected low-z large-kernel
CNN 61662009, corrected low-z window attention 61662012. Each requests one
80 GB A100, 32 CPUs, desi_g/shared, Scratch, 24 hours, no requeue or retry.
All three pending for priority at launch check; no production scores yet.
P12-A reference job 61581049 was running and was not modified.

## 2026-11-20 — G-PATCH replay and three grid smoke tests passed; production segments prepared

61659392 replayed ph006 macro lambda1 R2 0.469858376801 vs archived
0.469858376968; saved historical epoch 4 cursor 500 / global step 253838.
61659884 reached update 64 for each corrected low-z grid arm, with no
validation or ranking. Separate jobs preserve both preflight outputs.
encoder_production_segment.sh resumes from verified full optimizer, RNG and
scheduler states, with exact trainer source digests. The historical segment
ends no later than update 400000 (cosine minimum), not a convergence claim.
Any continuation beyond that point needs a separately identified nonzero
LR schedule. Both fresh grid challenger runs follow the registered four-epoch
seed-42 schedule against the existing P12-A reference; rank on matched
completed epochs, not on partial traces. First batch segments only; no
automatic retry. This entry is preparation evidence, not a launch receipt.

## 2026-11-20 — encoder comparison: historical restart and corrected grid challengers

Job 61659392: one-hour historical seed-42 G-PATCH preflight, original
37bef3a source and five training phases; replay ph006 epoch 3, then 500
epoch-4 updates. Separate Scratch output gpatch_restart_61659392. Do not
interpret launch as success or as a corrected low-z graph fit.

Grid challenger entrypoint p12a_encoder_sweep.py leaves active trainer
sources untouched. Reference, added large-kernel depthwise residual blocks,
and shifted-window attention bottleneck: matched BRIGHT inputs, loss, point
head, seed and sampler. Reduced-width CPU odd-shape forward/backward and
finite eval checks passed. Design inspirations: MedNeXt (arXiv:2303.09975),
Swin UNETR (2201.01266), DarkAI (2501.12621); these are controlled
ablations, not pretrained reproductions. LitePT (CVPR 2026) is a point
challenger pending corrected point-contract audit. No science ranking yet.
First GPU grid gate: 64 identical updates per arm, throughput/memory only;
for ranking require all three dev phases, equal-shell lambda1 R2, all
eigenvalues and shell scores, matched update and GPU-hour curves, and
replicated finalist seeds. No access to sealed evaluation truth.

## 2026-11-20 — P12-A encoder comparison: local CPU probe, not a production ranking

P10 historical re-read: at epoch 3 / 253,338 updates, ph006 four-shell
lambda1 R2 is 0.39758 for arm_a_r0_v1/unet and 0.46986 for
g_schedule_canary_lr2e4_v1/graph (G ahead in each shell). The schedules
differ; the 0.57310 U result is epoch 20, not the age-matched comparator.
The inductive G ceiling was not measured. No transductive results admitted.

Historical shell cross-check from the two `epoch_history.jsonl` files:
at identical epoch-3 order and step, G minus U on ph006 is +0.041206,
+0.053011, +0.070261, +0.124652 in increasing redshift. The largest
gap is in z=0.45–0.55 (0.341635 vs 0.216984); it accounts for 43% of
the summed four-shell improvement, but G improved in all four. U first
reached those respective G epoch-3 scores at epochs 6, 4, 6, 5. This
is evidence about early learning on the old five-phase training set,
not a convergence ranking: graph and U-Net used initial LRs 0.0002 and
0.002 and different schedules. These values exclude z<0.15.

Single-thread local-stencil probe on corrected p12a_lowz_20261029_v1
training_contract_v1. Sample 32 distinct eligible cores per phase, up to 40
unique assigned parents/core uniformly (numpy default_rng(4700+phase));
round authoritative fractional index, exclude 2-cell grid borders. Predict
physical lambda1 from a flattened 5^3 log1p(counts) cube, mean apodized
exposure over cube, and parent redshift. StandardScaler fitted on train
only. Train ph002/003/004/005 (1038/883/634/984); dev ph006/012/013
(712/864/840). Equal-weight R2 over 3 phases x 4 shells, not pooled:
ridge(alpha=100) 0.11732; ExtraTrees(24 trees, depth 9, min leaf 5,
seed 17) 0.14444; tiny Conv3d(1->8->8, valid 3^3, SiLU), plus the two
scalar features through a 16-unit SiLU head, 0.27714/0.21670 (seeds
17/42, standardized train target, AdamW .002, shuffled batch 128,
10 epochs). These inputs omit the production selection correction; this is
a spatial versus tabular micro-test, not a comparison to the production
encoder, a GNN replay, or evidence of a ceiling. No sealed phase opened.

## 2026-11-20 — approved development calibration ensemble submitted (not scored)

32 coherent draws separately for each cap of ph006/012/013. Reuse both
ph006 draw-zero fields. Gate 61646642 (SGC replay + new draw one) checks
supported-cell density and ordered eigenvalue replay and nontrivial change.
After both succeed: 61646793/61646799 run NGC/SGC draw zero for ph012/013;
after all four succeed: 61646846/61646847 supply remaining NGC/SGC draws.
61647276 depends on both large arrays; checks all 192 unique field/tide
receipts and hashes before scoring. No retries; ph018/019 remain sealed.
Sources: 5ed6c27, bae4ecc, 2033573. Output on success:
/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/lowz_calibration_61647276/CALIBRATION.json
and six cap/phase map panels. Total maximum 472.90 additional A100-hours
including gate and readout, below approved 480. Draws use 64 evaluations
per stage, sampled coarse and fresh fine noise; indexed conditioned exterior.
Cell- and observed-count-weighted marginal ranks, order-statistic coverage,
width, bias, fair CRPS, class reliability, fair Brier and class-fraction ranks;
joint density TARP and fair energy on supported core/subregion means.
These subregions are not the historical extended patch blocks. Scores apply
to registered supported locations in coherent fields, not every cap voxel;
three phases are the replication units. No baseline improvement or calibration
claim from queue state. Synthetic direct fair-CRPS and rank-count checks passed.

## 2026-11-20 — ph006 coherent draw-zero visual audit; ensemble calibration pending

Recovery 61610899 completed 0:0 in 2:57:22. NGC draw-zero class agreement
63.79% (60,617 supported scored cells), 75.68% below z=0.15 (13,945 cells);
SGC 66.39%/73.18%. Draw-zero density RMSE NGC/SGC 0.3819/0.3652.
Matched fixed central slices of all 16 registered pairs inspected;
docs/figures/lowz_ph006_draw_zero/{NGC,SGC}.png. Same mask, redshift cut,
density scale, and threshold 0.2 for truth and draw. Visible location and
class differences cannot establish underdispersion or joint calibration.
Historical CFM diagnostics included finite-draw marginal ranks/intervals,
fair CRPS, random-reference TARP for registered regional vectors, joint
energy scores, class-frequency reliability/fair Brier, randomized discrete
fraction ranks, cell vs galaxy weighting and exterior/operator controls.
Use phases as replicates; correlated selected cases are not textbook iid SBC.
No new posterior draws or scheduler submissions in this audit.

## 2026-11-19 — ph006 SGC scored; NGC-only recovery pending

The user relaxed job 61582948 to either A100 memory size. The allocation
ended FAILED (1:0) after 3:31:33 on nid001832. The 20-phase, 2,560-patch
exterior prior was fitted and frozen before inference; PRIOR.json SHA-256
1d1a2d8a6f21b13ff6336e95e0b59da4fec734481033cb3d4a3638810db1782f.
The SGC draw completed density, free-space tides and full-box scoring in
~92 minutes. Receipt: /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/lowz_cap_draw_61582948/SGC/SCORES.json.
At 53,845 supported registered cells over 0.01 <= z < 0.55, density RMSE
was 0.3652 and T-web class agreement 66.39%; for the 13,316 cells below
0.15 these were 0.2273 and 73.18%. One draw, one cap: not posterior
coverage, a full-sky evaluation, or a measured ensemble budget.

NGC failed during fine integration on the 40 GB A100; no NGC density
checkpoint survived. Authorized recovery 61610899 uses the same verified
ph006_draw_source_JwpP1b snapshot and frozen prior. Its submitted batch
script omits the prior-fit command, loops over NGC only (density, tides,
score), requests gpu&a100&hbm80g, 32 logical CPUs, desi_g/shared, Scratch,
4 hours, no requeue. Status at this entry: PENDING, elapsed zero. Do not
rerun SGC or launch the ensemble before checking NGC scores and accounting.

## 2026-11-19 — ph006 one-draw-per-cap boundary run submitted, not scored

Job 61582948: one shared 80 GB A100, 32 logical CPUs, desi_g/shared,
Scratch, six hours total, no requeue. Fixed helper snapshot
/pscratch/sd/d/dkololgi/abacus/ph006_draw_source_JwpP1b with
coarse_source_61551798 production imports. Fit the frozen three-scale
log-density prior on 20 training phases only; use one identical prior for
SGC and NGC. One coherent observation-conditioned draw per cap, 64 model
evaluations per stage. Preserve finite density cells; 32-cell conditioned
exterior, free-space cuboid tidal operator, no additional R7 smoothing.
Score the uncorrected cell-average tidal operator against periodic full-box
truth at supported registered cores, retaining low-z rows. This is not the
subcell-corrected readout or a calibration ensemble. Last check: pending.
Verify both SCORES.json files, density/tide receipts, job log and accounting
before making any result or cost claim. No ensemble authorized.

## 2026-11-19 — Corrected CFM shared-canvas engineering check, both caps

Shared 80 GB A100 job 61582135 (nid200257) completed both steps 0:0 and
released after 2:55. Frozen coarse_source_61551798 production imports; new
lowz_canvas_check.py and lowz_canvas_extent_check.py. No truth read. Receipts
under /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/lowz_canvas_check_61582135.
On ph006 NGC/SGC the trained EMA one-window coarse and endpoint outputs
equal the patch references at 8 NFE. Two-window decoded density overlaps
agree exactly over 215040/218880 cells; max relative block-mass errors
3.86e-16/4.02e-16 (upper bounds). Occupied galaxy tiles over [0.01,0.55)
plus registered cores give 7195 windows across ph006/012/013, both caps.
Timing on actual cap extents with eight active windows at four evaluations
per stage: 2.50 s combined per cap; peak allocated CUDA tensors 2.96/2.48
GiB, not the full observation cache. Sequential one-draw extrapolation at
64 evaluations per stage: ~10 A100-h per draw over six caps, ~640 A100-h
for 64 draws, excluding preparation, streamed IO, tides and scoring. No
full-cap ensemble generated, no calibration claim and no final-phase reads.

## 2026-11-19 — P12-A epoch-gated recovery windows

Cancelled 61577997 while PENDING (zero elapsed). Epoch 1 validation allocation
61581030, tmux p12a-epoch1-validation on login16, 4h shared_interactive.
Resume is restricted to epoch-1 cursor 346364; no epoch-1 optimizer updates.
On successful validation, checkpoint epoch 2 cursor zero before returning.
Epochs 2/3/4: shared batch 61581047/61581049/61581051 respectively, 14h
each, afterok chain plus per-epoch history/checkpoint guards; no requeues.
The 46 GPU-hour cap is cumulative across the four allocations. Submission
records scheduling, not a validation result.

Result: validation job 61581030 completed 0:0 in 1:42:43, at the original
step 346364. All authoritative rows were scored with zero patch failures:
ph006 7,261,727 (shell-macro lambda1 R² 0.4418), ph012 7,339,923 (0.4340),
ph013 7,275,255 (0.4294). Equal-phase mean 0.4351. Epoch 2 resumed without
repeating epoch 1; at 23:36 GMT job 61581047 had logged 178,161/346,364
epoch-2 updates. Epochs 3 and 4 remained dependency-pending.

## 2026-11-19 — Combined low-z seed-17 CFM: three-patch solver pilot complete

Both production EMA checkpoints verified; observed local/wide inputs only,
independent coarse/fine noise per draw, matched seeds across solver levels.
Job 61580056 on nid200257 completed 0:0 in 4:08 and released. Receipts:
Scratch e2e_field_v3/lowz_patch_pilot_61580056 under abacus; runner
workflows/sbi/lowz_patch_solver_pilot.py with coarse_source_61551798 imports.
One patch per development phase (006 low-z boundary, 012 low-z interior,
013 high-z interior), local+wide free-space tides with unit exterior applied
equally to draws and scoring truth. Max relative block mass error 4.41e-16
(upper bound). With matched noise, density mean absolute change fell from
0.030–0.038 to 0.014–0.0174 across successive solver doublings. At 16 coarse
steps/32 fine NFE, 4 to 8 draws shifted density coverage by 16–19 percentage
points. No calibrated coverage, baseline comparison, supported-voxel low-z
result or full-cap claim. At 32 coarse steps/64 fine NFE measured 4.95 s/draw
including local tides: 48 domains x 64 draws projects to 4.22 A100-h before
baseline and run overhead. Full run not launched; final phases sealed.

## 2026-11-19 — P12-A corrected whole-phase validation recovery

61548250 completed all 346,364 epoch-1 optimizer updates, then failed in the
legacy spatial diagnostic. Corrected assignments deliberately have fold=-1
and no internal superblock/boundary metadata. Only for the bound whole-phase
schema, mark these diagnostics unavailable and label shells from the bound P1
redshift contract. Preserve exact parent coverage and equal-phase scoring.
14 focused tests passed. Full-encoder recovery 61577997 submitted; original
checkpoint SHA256 6ceb15b449e0751757dfecff082af98becb7093f1c96e90d0aaac56e98bd5231.
One-time source migration retains the original artifact and all optimizer/RNG
state. Validation must cover ph006/012/013 before epoch 2; submission is not
evidence that this gate has passed. No omitted-phase fits or sealed-phase reads.

## 2026-11-19 — Combined low-z CFM: fine fit complete, coarse fit outstanding

Seed-17 fine endpoint job 61534556 completed 32 epochs / 15,360 updates.
That fit alone is not the observation-to-density model. The historical wide
coarse velocity trainer binds an older release and a 13-phase schedule.
For the corrected data, 20 training condition manifests each list 128
distinct paired domains. New entry point: workflows/sbi/lowz_coarse_fit.py.
It binds the fine run and normalizer, starts the coarse weights fresh, and
includes a train-address observation-only coarse-to-fine interface check.
The proposed primary coarse checkpoint follows historical epoch 16 on a
32-epoch learning-rate schedule (20,480 low-z updates at batch two).
GPU preflight 61551766 completed 0:0 in 80 seconds: 24 training steps,
then an observation-only sampled coarse-to-fine draw on ph000 yielding a
finite positive 96x64x64 density patch. Scratch receipt:
abacus/e2e_field_v3/lowz_coarse_preflight_61551766/PREFLIGHT.json.
Steps 2–24 averaged 1.03 s/update (~5.85 A100-hours projected for 20,480
updates, excluding startup/IO variation). Fresh seed-17 coarse production
job 61551798 submitted on one shared A100 with an eight-hour cap and no
requeue. Source archive SHA256:
4b8853ae4823747eb9208d33696e3b24f0a9e8885d080693a57ac5454cd18dc3.
The preflight checkpoint is disposable. Verify COARSE_FIT_COMPLETE.json
and both checkpoint hashes before evaluation handoff; no calibration claim.

Closeout: job 61551798 completed 0:0 in 2:11:38, all 20,480 updates.
The production observation-only coarse-to-fine interface check passed.
Both LATEST.pt hashes matched COARSE_FIT_COMPLETE.json under Scratch
abacus/e2e_field_v3/lowz_coarse_velocity_seed17_v1. The fine checkpoint
is in its sibling lowz_unet_endpoint_20261118_seed17 directory. Training
is complete; this interface check alone does not establish calibration.

## 2026-11-18 — Repository handoff: Bind low-z training runs and integrate ordered encoder prefetch

Group the bound input manifest, disposable profiling and restart checks, the 960-domain CFM launcher and the ordered-prefetch P12-A trainer and budget. Production submissions: CFM 61534556, P12-A 61535300; both jobs’ live constraints were relaxed to either A100 memory size. Submission does not establish convergence or posterior calibration.

## 2026-11-18 — Repository handoff: Record low-z campaign execution, recovery and verification

Group the resource and continuation plans, point-selection checks, bounded recovery workers, progress monitoring and final preparation receipts. Historical job-specific commands are provenance, not instructions to relaunch.

## 2026-11-18 — Repository handoff: Prepare low-redshift CFM targets, survey response and loaders

Group the coupled data configuration, survey selection and response construction, geometry caching, target and normalization checks, whole-field loader changes and phase qualification evidence.

## 2026-11-18 — Repository handoff: Prepare low-redshift P12-A response and encoder contracts

Group the observed-redshift contracts, galaxy identity and matched-response adapters, conservative host checks, cross-fit preparation, tests and encoder-input evidence. This is input qualification, not posterior calibration.

## 2026-11-18 — Repository handoff: Version shared-phase migration, galaxy truth and reuse audits

Group the phase registry and migration policy with the galaxy-position truth changes, reuse inventories, preparation scripts, interrupted-run evidence and loader checks for the shared 20/3/2 split.

## 2026-11-18 — Repository handoff: Preserve campaign science record and link GraphWeb handoff

Restore the accumulated October–November campaign record and durable context to version control. GraphWeb_DESI commit e2df159 groups Loa density, tidal, cube and structure diagnostics; these are presentation and sensitivity checks, not a new calibrated VAC. GraphWeb_DESI commit 0ceafe7 records the upstream low-z handoff without adopting new checkpoints or changing observational defaults.

## 2026-11-18 — P12-A integrated ordered prefetch passes mid-epoch restart

On the full low-z contract, tmux-owned A100 job 61534946 compared 24
uninterrupted updates with 11 updates followed by a fresh-process resume
to 24. Loss traces and all compared training state (including weights,
optimizer, scheduler, RNG and cursor) matched exactly. Job completed 0:0
in 3:09. Evidence: Scratch `abacus/p12a_restart_61534946/PASS.json`.
This establishes short mid-epoch restart parity, not convergence or
epoch-boundary parity. No P12-A production fit submitted. Proposed
four-epoch encoder budget: `docs/p12a_prefetch_budget_20261118.md`.

## 2026-11-18 — P12-A ordered one-core lookahead cuts measured update time

Tmux-owned single-A100 profile 61534658 completed 0:0 (93 s allocated).
Four paired blocks alternated run order. Across ph000/008/017, each arm
timed 96 updates after warm-up; per-step losses and final weight hashes
matched exactly in all four pairs. Baseline 0.23290 s/core; one-worker
ordered CPU preparation 0.13179 s/core (1.77x). Prefetch was faster in
both orderings, but second-pass reads benefit from warm data. Revised
update-only estimate: 12.68 A100-h/full encoder epoch, 253.59 A100-h
per full+20 OOF common epoch (223–294 across the four block rates).
Not a full epoch, pipeline integration, or production restart test;
no P12-A fit launched. Source: workflows/sbi/p12a_ordered_profile.py;
paired timings, addresses, losses and hashes: Scratch
abacus/p12a_ordered_profile_61534658/RESULT.json.

## 2026-11-18 — First low-z fine CFM production fit submitted

Explicit approval: one seed-17 3D U-Net clean endpoint, 32 epochs, 1 A100
(80 GB), 32 CPUs, desi_g/shared, Scratch, 3 hours, no requeue or retries.
Job 61534556 submitted, last observed pending priority; not a finished fit.
New lowz_unet_fit launcher streams two domains per update, checks 48 unique
training domains per phase across 20 phases, profiles 40 training spectra,
and writes to a fresh lowz_unet_endpoint_20261118_seed17 directory.
Focused actual-panel and spectral-mean tests passed. Historical loader untouched.
No P12-A GPU resources included. Its full+20 OOF contracts require exactly
20 times the full-only core updates per common epoch. Potential overlap of
HDF5 reads/CPU preparation with GPU work is unmeasured; do not weaken the
held-out design or spend the full multi-fit budget without a decision.

## 2026-11-18 — Low-z single-A100 disposable preflight; no production launch

User approved one hour via tmux; 61534050 on nid200301 completed 0:0 in
80 seconds allocated. Receipts: /pscratch/sd/d/dkololgi/abacus/lowz_gpu_preflight_61534050/
(cfm.json, p12.json, halo48.json). Both one-update checkpoint replays had
zero parameter difference. Training update estimates, not end-to-end timings:
P12 24 timed ph000 cores, 0.29236 s/core, 1.46 GiB peak allocated; 346364
full-fit cores or 6927280 cores across full + 20 omitted-phase fits per common
epoch (28.13 or 562.57 A100-hours respectively). Five halo-48 inference-only
patches 0.44–0.64 s; no summary file writes timed. Fine U-Net clean endpoint:
48 timed uncached two-domain batches, 0.41090 s/update, 2.79 GiB peak;
480 steps/epoch or 1.75 A100-hours per seed for 32 epochs excluding overhead.
These estimates exclude validation, production checkpoint/export I/O, sampling,
coarse-model and posterior fits. Short checks did not validate native full-run
resume. Whole-field pilot launch path still assumes 624 domains, 640 cache limit,
and 26 spectral samples rather than 40 for 20 phases. Adapt and test before
production; no scientific fit, authorization for production, or calibration claim.

## 2026-11-18 — Retraining inputs bound; U-Net clean endpoint first

User selected the low-z encoder contracts and the 3D U-Net clean-endpoint
fine CFM. New binding: configs/lowz_retraining_binding_20261118.json. Preserve
the October preparation plan for provenance; do not use its old dataset as a
runtime default. GPU scope/cost pending approval; neither fit launched.

## 2026-11-18 — Low-z retraining inputs verified; no model fit

Supersedes the November 2 launch status. Job 60684956 completed the outstanding
CFM products, train-only normalization, 23 whole-field finalizations and real
loader checks. On November 18 we checked the per-phase audit, wide-target and
finalization receipts for the 20 training and three development phases. The
loader receipt passes for all 23, with 960 training domains and no final-test
payload opened. Its hashes match the current CFM config, contract module,
whole-field loader, check implementation and normalization receipt.

P12-A: all 21 encoder contract receipt hashes (full + 20 leave-one-phase-out)
match the preparation manifest; 84 actual-loader checks, no sealed phase opened.
These checks establish input readiness, not posterior calibration, model
performance or GPU launch authority. No full scratch-array rehash was performed.
See the [verified handoff](docs/lowz_preparation_checkpoint_20261029.md).

## 2026-11-02 — CFM final preparation continuation launched

60533554 and its step completed 0:0 after 3h29m, pausing for budget before
ph012_whole_targets. 21/23 phases have paired/wide products; ph012 paired
products/audit/wide observations are complete, ph013 preparation remains.
Training-only normalization, all-phase whole-field finalization and loader
checks remain; no training-readiness claim. User explicitly approved another
one-node desi/interactive CPU window, 64-CPU steps, Scratch, up to four hours,
with no training or automatic failed-stage retries and release when done.
60684956 on nid200036 is launched from tmux cfm-finish-20261102 on login34.
Log: /pscratch/sd/d/dkololgi/abacus/lowz_continuation_60684956/cfm.log.
Six focused allowance/resume/cleanup tests passed. Historical stage receipts
and failed P12 host diagnostic are preserved. This is the fifth recorded
completion allocation; the new allowance admits no sixth allocation.


---
*Entries from 2026-10-30 and earlier (20,900+ lines) are archived verbatim in `docs/log_archive/SCIENCE_LOG_2026_pre_nov.md`.*
