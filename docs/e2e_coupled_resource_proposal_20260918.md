# Coupled-field four-arm experiment: measured resource proposal

**Ready for user review; NOT a training launch.** All21phase data products,
independent audits, normalized interfaces and the physical-reference gate now
qualify. GPU/loader/postprocessing measurements are complete. This document
specifies a finite scientific protocol, gates and measured resource request;
implementation and scientific execution require separate approval. See the
[preparation closeout](e2e_coupled_preparation_closeout_20260919.md) and its
archived evidence for the completed preparation, limitations and resource ledger.
The current preparation allowance remains64CPU-nodeh/4technicalGPUh/4.5TiB,
not an allowance for any of the scientific fits or draws described below.

## Factors and matched information

| Scientific arm | Coarse factor | Fine factor per pair |
| --- | --- | --- |
| D-reference | D coarse VDM, inherited restricted conditioner | Two independently sampled48-cubed parents |
| I-VDM | Matched coarse VDM with both joint-local and wide observations | Two independent48-cubed parents, full common observations |
| J-VDM | SAME fitted coarse VDM and SAME addressed coarse draws as I | One coupled64x48x48 residual field |
| J-CFM | Matched coarse CFM with both observation views | One coupled64x48x48 CFM residual field |

At two training seeds this is **14 distinct factor fits**, not16: D contributes4;
the shared I/J coarse plus their two fine factors contribute6; CFM contributes4.
No separate coarse fit or duplicate coarse-draw cost should be billed for J-VDM.
VDM/CFM comparisons require common spatial networks/information and fixed global
train-only transforms. D versus the new design remains a package comparison.
J-VDM:J-CFM is a matched architecture/data comparison of objective, probability
path and sampler packages. It is not a loss-only ablation: VDM uses the VP path
and ancestral sampling, while the proposed CFM uses a Gaussian-base linear path
and Heun integration. Report sampler convergence and allocated compute separately.

The data products expose13 training phases/1,664 pair domains (3,328 owned cores),
two development phases/32 domains and six confirmation phases/96 domains. These
are13/2/6 independent simulation phases, not thousands of independent universes;
overlapping wide contexts do not increase independent phase diversity. All later
arms use the same fixed-epochz0.2 nominal mock-release/selection pairing. Numerical
HOD parameter files have not been recovered from the source FITS headers; this
is not HOD marginalization or a demonstration of real-DESI validity.

A successful result would qualify the tested adjacent-core/shared-coarse
posterior on the registered mocks. It would not, by itself, establish coherence
between every separately sampled pair domain across the entire DESI footprint,
nor validate the real-survey observation likelihood. Keep that scope distinction
explicit when interpreting joint-statistic improvements.

## Exposure and estimator controls

- Match phase/pair/owned-core presentations and training seeds. I/D require two
  latent-parent evaluations per pair; their auxiliary overlap is duplicated.
  Report that extra latent exposure and actual compute instead of calling equal
  update counts equal cost. Average factor objectives per independent latent
  degree of freedom, including the block-zero-mean residual constraint.
- Give I/J the identical full joint and wide observation views; cropping the
  I observations down to its current parent would add an information confound.
  Coarse conditioning must not silently discard the joint-local observation.
- I's two parent noises must be independent conditional on the shared coarse
  draw. Cropping one global noise realization into overlapping I parents would
  introduce residual dependence and invalidate that control. Cross-arm common
  random numbers may use the block-aligned ownership assembly of independently
  drawn left/right noises for J, without sharing I's stochastic overlap.
- Use the same rectangular physical operator for primary I/J diagnostics.
  `e2e_coupled_views.assemble_independent` assigns left x<32 and right local x>=16
  without overlap averaging. Full-box truth reassembles exactly, and shared
  coarse block masses are conserved. Keep per-parent operators as named controls.
- Block-aligned regional mass is fixed by the coarse draw. Fine residual coupling
  cannot repair its coverage. Judge the improved coarse conditioner separately
  using wide/regional proper scores, bias/spread and coverage, not fine power alone.
  In the strict I:J comparison those masses, their ensemble ranks and coverage
  must AGREE draw by draw up to the numerical tolerance; agreement is an
  implementation invariant, not evidence that their common posterior is correct.
  Requiring an I:J improvement in this observable would be an impossible gate.
  Use D:new to test the coarse-conditioning package and J-VDM:J-CFM to compare
  the full objective packages; evaluate residual dependence with adjacent-core
  tidal/eigengap summaries, non-block-aligned probes and common-domain power.
- Sampler/checkpoint decisions use development only. Confirmation numerical
  product QA is permitted now; confirmation predictions require a frozen protocol.
  Report both training seeds and phase-level uncertainty; voxel counts do not
  supply independent uncertainty replicates.

## Sample-count accounting to benchmark

A128-draw ensemble on all96 confirmation pairs, both seeds and all four arms gives:

| Quantity | Count | Accounting |
| --- | ---: | --- |
| Comparable rectangular posterior fields | 98,304 | 96 x2 x128 x4 |
| Fine latent fields evolved | 147,456 | 96 x2 x128 x(2+2+1+1) |
| Distinct wide coarse draws | 73,728 | 96 x2 x128 x3 coarse families |

These are counts, **not an approved sample campaign**. A256-draw version doubles
all three. At float32 the128-draw rectangular density payload alone is about
58.0GB and the unique coarse density payload about32.6GB, before stage products,
checkpoints, temporary generations, diagnostics and optional parent controls.
Storing tidal tensors for every voxel/draw would cost far more: prefer validated
streaming metrics and bounded saved diagnostic fields, while retaining sufficient
draws for exact replay/audit. Reuse prepared inputs in place, not another particle
copy. A separate future output cap must cover all new experiment artifacts.

Costs must also include development checkpoint progression, sampler refinement,
fixed-parent/true-parent controls, analysis, warmup/checkpoint overhead and a
bounded restart reserve. The primary calibration benchmark must use the exact
finite-ensemble reference implied by its saved order-statistic/rank definition,
not assume nominal90% is exactly attainable at a given sample count. Degenerate
oracle regional masses are excluded from continuous-target coverage pooling.

### Proposed finite development panels for cost accounting

The receipt-backed estimator now explicitly charges the following proposed
panels. These counts are a bounded design for the later approval, not authority
to fit models, open confirmation predictions or select new scientific thresholds.
Checkpoint exposures are fixed in Stage A below; the final NFE is selected only
by its registered development convergence rule, never confirmation performance.

| Panel | Paired domains | Seeds | Draws | Repetitions | Rectangular fields |
| --- | ---: | ---: | ---: | ---: | ---: |
| Checkpoint progression, all four arms | 32 development | 2 | 32 | 3 checkpoints | 24,576 |
| Sampler ladder, all four arms | 16 development | 2 | 32 | 64/128/256 NFE | 12,288 |
| Fixed-mean and oracle-coarse controls, all four arms | 16 development | 2 | 32 | 2 controls | 8,192 |

Total: 45,056 comparable development fields, 67,584 fine latents and 27,648
unique coarse draws. I/J still share the same addressed coarse draw; independent
D/I fine factors still evolve two parents. The 16-domain subset must be frozen
from phase/cap/redshift/support strata before predictions, with both development
phases represented. The coarse-mean control uses the empirical mean of the saved
primary development ensemble, not a claimed exact posterior expectation; the
oracle supplies truth explicitly for diagnosis only. These controls need only
fine sampling, so no additional coarse draws are charged. Exclude their fixed
regional masses from generic continuous-target coverage pooling.

For conservatism the accounting gives no reuse discount where checkpoint and
sampler panels might overlap. It sums the measured 64-, 128- and 256-NFE costs
once each for the ladder, rather than charging three times the selected NFE.
The three scientific checkpoints are separate from frequent restart checkpoints;
`--development-checkpoints` changes only the progression panel. All eight cost
arithmetic/guard tests pass, including independent-parent and shared-coarse
accounting. Test-fixture timing values are not scientific or GPU measurements.

### Finite-ensemble interpretation to preserve in the later protocol

The existing `central_order_interval` implementation selects the central order
interval nearest the requested coverage. For a continuous calibrated ensemble,
its attainable nominal-90% reference is **87.8788% at32draws,90.6977% at128draws,
and89.8833% at256draws** (zero-based lower/upper indices1/30,5/122,12/243).
These values have been checked directly from the implementation. They are not
coverage measurements. Report the interval convention and discrete rank reference;
do not import90.75% from an earlier different ensemble size or treat the32-draw
development estimate as precisely calibrated. Deterministic coarse-oracle masses
are not continuous exchangeable-rank tests, even if ties are randomized.

For the power decomposition, let a fixed common linear window/FFT operator act
on each physical-density draw x_m, and let xbar be their M-draw mean. In each
registered band the exact finite-sample identity is

`mean_m P(x_m) = P(xbar) + mean_m P(x_m - xbar)`.

The residual term on the right uses divisor M. If posterior residual covariance
is instead estimated with the unbiased divisor M-1, the reconstruction is
`P(xbar) + (M-1)/M * P_residual_unbiased`. Do not add the unbiased residual power
without that factor or compare different ensemble sizes without naming the
convention. Under a calibrated conditional ensemble with an independent truth,
`E[P(truth-xbar)] = (1+1/M) E[P_residual_unbiased]`; account for this extra
Monte Carlo mean uncertainty when comparing residual spread to reconstruction
error. These are algebra/expectation statements, not claims that our future
posterior is calibrated or that a single pair must satisfy the expectation.

Keep three judgments separate: ensemble sample power, posterior mean/error,
and stochastic residual covariance. A better mean prediction or correct total
power can coexist with incorrect posterior dependence. Regional-mass coverage
still tests the coarse factor, while I:J mass equality remains an implementation
invariant. Evaluate fine coupling on quantities that can actually change with
that coupling. All metric normalizations, band/window conventions, tolerances
and development-based choices must be frozen before confirmation predictions.

## Completed technical measurements and data qualification

The inherited `ContextVDM` hard-codes a324.768Mpc/h fine query span. This remains
correct for each independent48-cubed parent but is WRONG for the new joint
rectangle: its span is(433.024,324.768,324.768)Mpc/h. Wide span is1299.072Mpc/h
per axis. Parent centres are displaced by(-54.128,0,0)/(54.128,0,0) from the
joint centre; every registered wide-context translation must be included too.
The benchmark must use a dedicated, tested physical-position/condition adapter,
not unchanged cubic code or a rescaled cube masquerading as the rectangle.
The matched coarse conditioner must actually consume spatial joint-local
observations as well as wide observations. Existing D coarse code alone does
not implement that proposed factor.

The new `e2e_coupled_benchmark_models.py` engineering prototype now supplies
these position/condition adapters and a common8,398,387-parameter backbone.
I/J retain identical full observation views; their fine latent domains differ.
Both VDM and CFM use the same state-residual output parameterization. VDM uses
the inherited fixed-log-SNR VLB; CFM uses a uniform-time independent-Gaussian-base
linear path and velocity MSE per independent degree of freedom. This does not
claim optimal-transport coupling or identical noise-level exposure between the
two different path objectives. The retained D architecture is still a package
baseline, not part of the strict I/J or VDM/CFM architecture match.

Meta-device shape, physical-position, projected-VLB parity and analytic CFM
checks pass. Full-size synthetic CPU forward/backward tests also pass8cases,
with nonzero spatial joint-observation gradients for coarse inference and
outside-parent observation gradients for both independent parents. All cases
retain the same parameter count; fine block means remain below7.1e-8.
These deliberately use a perturbed final head for connectivity and0optimizer
updates. Small initial context gradients establish availability only, not learned
utility. Receipt:`MODEL_INTERFACE_SMOKE_1789770865595873044.json`, SHA256
`3a29895299e741f62cfa55b64875d866411589b34583ab133f412eed8e612008`.
No scientific fit or posterior-performance claim follows from these CPU checks.
GPU timing and exact optimizer/process replay have now passed separately below.

The technical runner is frozen in `source_snapshots/gpu_benchmark_v3` and
completed as58559826 on2026-09-19. Its seven cases use batch2, nine unique synthetic updates
per factor and one additional fresh-process replay update. Sampling uses
64/128/256NFE, with two independent parent evaluations charged for I/D fine.
It measures scalar/batch agreement, optimizer/model/RNG replay and memory on the
actual registered shapes. This is not a scientific fit or a scientific sampler
convergence test. Neither a smaller grid nor CPU timing substitutes for it.
The benchmark also measures three synchronous pageable-host
transfers of the actual synthetic factor presentation shapes, without deduplicating
repeated parent-context tensors or claiming pinned-memory/overlap speedups. This
is a conservative transfer measurement, not an optimized data pipeline. Three
checkpoints measure GPU-to-CPU state extraction, serialization, fsync and SHA256
wall time. The original nine unique technical updates plus one replay update
are unchanged. A frozen one-request controller waits for58550091 to end at a
verified success or planned75:0 pause before requesting one shared A10080 GPU
for one hour. It cannot request a third allocation, retry an unexpected failure,
submit batch work or launch scientific training. The original4technicalGPUh
allowance is not reset; it charged0.141667GPUh and completed0:0 in8m30s.

Separate `e2e_coupled_loader_benchmark.py` measures the strict actual normalized
reader and CPU collation on26pairs TOTAL per pass (two per training phase), over
two passes. Each pair uses ONE preassigned offset; together the cases cover all
seven translations, not26x7views per pass. It requires the complete13-phase normalizer
and independent source audits; this measurement now passes. Both first-observed
and repeat timings include payload hashes/HDF5/normalization. No OS cache flush,
persistent-cache improvement, GPU-transfer overlap or asynchronous prefetch is
claimed. Combine actual measurements conservatively until any optimized loader
is separately implemented and measured; resident synthetic tensors alone do not
estimate end-to-end training throughput.

The complete13-phase normalizer is now published and independently rechecked
against its cached source-bound moments (SHA256
`b2b294c5de91a7659be16b87e2ea454a11464fbd4522b4c0d30d2b1effa9e43e`).
The actual measured first-observed/repeat averages are0.1172705815/0.0777026738s
per pair. Peak process RSS is596,791,296bytes. Loader receipt SHA256:
`bb4c670332a71a2516382426781597e01adfdacfcb60b1365789f73a9bcd3cf7`.
Recently read arrays may already be cached; neither pass is labelled cold-cache.
The separate full21-phase qualification now passes all1,792pairs and11,776
registered pair/offset cases. This is numerical/read-interface qualification,
not a learned-posterior result.

Use at most the already approved4GPUh and no scientific fits. Benchmark actual
full-size networks/objectives, with explicitly technical warmup/short update and
restart checks. Measure coarse/fine D, matched coarse VDM, independent/joint fine
VDM and coarse/fine CFM; count both I/D parent evaluations and the reused coarse
draw only once. Check normalization/chart decode, finite projected states,
checkpoint/RNG replay, physical offsets and memory for the registered shapes.
Do not substitute a smaller grid or infer GPU cost from CPU kernel timing.

`workflows/sbi/e2e_coupled_resource_estimate.py` now supplies tested,
receipt-backed component accounting once the GPU, normalized-loader and CPU
postprocessing measurements all exist. It
requires all seven factor cases, fresh-process replay evidence and the complete
13-phase/26-pair loader probe. It rejects partial panels, invented placeholder
costs, nonfinite timings and wrong independent-parent counts. Exposure/draw and
checkpoint-cadence arguments are explicit projections, not a training protocol
or launch authority. It counts832batch-two updates per1,664-pair epoch,14fits
across two seeds, and shared I/J coarse draws once. Serial training estimates
include the slower observed loader-pass average, host transfer and GPU updates;
periodic/final checkpoint costs are separate. Sampling kernel costs and a
training-shape transfer proxy remain separate from still-to-budget inference
IO/decode/diagnostics, startup and restart headroom. Development panels above
are now counted explicitly, including sampler refinement and fine-only controls. This
arithmetic is implemented/tested. The combined component-cost report is
`cartesian_v2/resource_estimates/0042d5a7b319e95d.json`, SHA256
`beb26f13da65882159ba145fe43e011607ded1e85fad48c412a6c424c251eb4c`.
It remains`proposal_ready=false`: measured components are not a finished
scientific protocol, data release or an end-to-end campaign measurement.

### Measured GPU rates and cost implications

Job58559826 used NVIDIA A100-SXM4-80GB, PyTorch2.9.1/CUDA12.9, float32,
deterministic math attention and batch2 on nid008700. All7cases pass exact
fresh-process replay. Scalar/batch relative RMS is below8.4e-8; three sampling
NFE points pass finite/projection checks, NOT a scientific convergence test.
Full GPU receipt SHA256:
`c9eb5cdeddc7e2f62090db1ded3f258783e25efe55efc65f398de73b66ca84c8`.

| Factor | Parameters | Update seconds, batch2 | Sample seconds/pair, NFE128 | Peak reserved GPU GiB |
| --- | ---: | ---: | ---: | ---: |
| D coarse | 8,188,395 | 0.1758 | 4.8975 | 0.799 |
| D fine, BOTH parents | 8,371,907 | 0.3582 | 9.9905 | 0.883 |
| Shared I/J coarse | 8,398,387 | 0.1819 | 5.0083 | 0.891 |
| I fine, BOTH parents | 8,398,387 | 0.3654 | 10.2052 | 0.898 |
| J fine | 8,398,387 | 0.2330 | 6.6261 | 1.105 |
| CFM coarse | 8,398,387 | 0.1822 | 4.9736 | 0.891 |
| CFM fine | 8,398,387 | 0.2330 | 6.5720 | 1.105 |

Per-case host peaks span1.66--1.79GB; Slurm step MaxRSS is4,696,248KiB. The
low tested GPU memory suggests a larger inference batch MIGHT help, but no such
speedup is credited: batch2 is the only measured shape. Independent multi-GPU
task execution likewise needs a technical throughput check before assuming
linear scaling; these are additive GPU-hours, not a four-GPU wall-time promise.

| Cost component | Measured-component projection, GPU-hours |
| --- | ---: |
| All14fits,16epochs /13,312updates per factor | 25.08 |
| All14fits,32epochs /26,624updates per factor | 50.15 |
| All14fits,64epochs /53,248updates per factor | 100.30 |
| Development panels, selected128NFE; ladder includes64/128/256 | 149.41 |
| Confirmation128draws,64 /128 /256NFE | 165.35 /329.55 /657.81 |
| Confirmation256draws,64 /128 /256NFE | 330.70 /659.09 /1,315.62 |

Training includes the slower observed reader pass, synchronous H2D and measured
checkpoint overhead at1,664-update cadence; network sampling columns exclude
the separately recorded transfer proxy and unmeasured orchestration. The128draw
confirmation transfer proxy is0.06153GPUh; development0.02706GPUh. CPU analysis
components are listed below. Increasing32to64epochs adds about50GPUh, whereas
doubling128to256confirmation draws adds about330GPUh at128NFE. Thus the next
approval should stage development before confirmation, not pre-purchase the
most expensive ensemble before the joint-dependence test shows promise.

### Completed synthetic CPU postprocessing measurement (2026-09-19)

The additional `postprocess_benchmark_v1` ran as step58550091.7 inside the
already approved node:4logical CPUs/8GiB, a12-minute hard step limit and a
10-minute internal bound. Aggregate reservations remained472GiB/132logical CPUs
against487802MiB/256allocated. No new allocation or scientific payload access
was needed. The step completed0:0 in57s; its measured work took17.50s.
Process ru_maxrss was1,442,484,224bytes; Slurm separately reports7,444,144KiB
MaxRSS. Preserve both measurements and retain8GiB for this tested worker rather
than sizing it from the smaller process-only figure.

The receipt at `technical_cpu/postprocess/5913c1ecab5fbf01/POSTPROCESS_BENCHMARK_COMPLETE.json`
has SHA256 `a119cb95fb155488e6d55741c691a25b7486bef22270c484995ab6e57d525d19`.
It records eight full64x48x48 decode/common-tidal/eigenvalue/FFT/I/O probes and
actual metric calls on8,192owned voxels at32/128/256draws. Mass/trace identities
and saved-field round trips pass. Input fields and score arrays are synthetic;
posterior-performance values are deliberately not retained. This is a component
cost benchmark, not a calibrated model, a cold-cache test or full analysis code.

Maximum observed per-field component times are: decode0.00385s; common tidal
operator0.46007s; owned eigenvalues0.04158s; FFT/quantiles0.00571s; uncompressed
HDF5 write/fsync/hash/read verification0.04308s. Pointwise five-component
eigenvalue/gap energy scoring takes2.090s per128-draw ensemble and9.268s per
256-draw ensemble. The adjacent12-summary scores are separately timed.

Using those measured components, the estimator now projects7.142four-CPU
worker-hours for all proposed development panels, plus15.890for128-draw or
33.138for256-draw confirmation. Totals are23.033or40.281worker-hours BEFORE
remaining overhead/headroom. Exclusive CPU-node-hours would equal those hours
if only one such worker group used each node; do not assume an unmeasured
many-worker speedup. This projection uses the maximum of eight observed times,
not a statistical worst-case bound. It charges two full rectangular decodes
per field as a conservative independent-parent/assembly allowance, and explicitly
labels wide scalar-score voxel scaling and wide-FFT size proxies. No shared-
coarse output-I/O discount is assumed.

Final band/window reductions, population calibration aggregation, bootstrap or
reference-point diagnostics, GPU-to-host copies, contention and orchestration
still require declared headroom. The resource estimator now requires this CPU
receipt as well as actual GPU and normalized-loader receipts before publishing
the combined report. The final focused suite passes126tests, including
cost-accounting, postprocessing, recovery, closeout and analytic score-design
controls. GPU/loader measurements and full-panel data qualification pass.

For each measured factor report parameters, peak GPU/host memory, seconds per
paired training presentation, seconds per posterior field at each tested NFE,
and exact environment/device/batch size. Compute allocated GPU-hours (not just
kernel time), add measured overhead and a declared bounded contingency. The
training-update/exposure ladder and final128-versus256 draw choice can then be
proposed quantitatively. Do not fill missing measurements with invented timings.

Measured preparation evidence so far: native2048 R7 plus128 paired targets took
603s with257.5GiB peak host RSS forph007. Its independent extended audit took
252.6s and checked608 actual hashes; neither number estimates neural training or
sampling throughput. Full-panel target construction, independent audits and
normalized-interface checks now pass, as do normalization and GPU measurements.
The registered64-core physical gate has now passed for both operators, with
about88% median eigenvalue-error improvement (25% required):
[physical gate report](e2e_coupled_physical_gate_20260918.md). This is truth-only
representation evidence, not a learned-posterior result.

## Proposed next approval: development first, confirmation separately gated

This is a concrete **proposal**, not an extension of the current preparation
authority. All-panel data qualification has passed. Implement and freeze
the scientific trainer, addressed sampler and analysis manifest before any fit;
the existing technical prototype is not presented as that finished workflow.

### Stage A: implement, fit and diagnose on development

- Request **260 allocated GPU-hours,24 CPU-node-hours and512GiB of cumulative
  NEW experiment outputs**. Reuse prepared inputs in place. This is a separate
  later allowance, not a reset or extension of today's preparation ledger.
- Fit the14registered factors on exactly1,664training pairs for **32epochs**:
  26,624batch-two updates/factor,53,248paired presentations/factor. Seeds0/1;
  common addressed pair orders and context-offset sequences across arms. Seven
  context offsets rotate deterministically; keep every phase equally exposed.
- Primary checkpoint progression is epochs8/16/32 (6,656/13,312/26,624updates).
  Restart checkpoints every2epochs plus final; preserve all three science
  checkpoints. Resume model, optimizer, data cursor and all RNG streams exactly.
- Retain the measured AdamW1e-4,weight_decay1e-5,gradient_clip0.5, no learning-rate
  search, no EMA selection and no auxiliary spectrum/preservation penalty. Use
  the tested VDM fixed-log-SNR objective/decoder0.001 and CFM independent Gaussian
  linear-path objective. Fine objectives are averaged over independent residual
  degrees of freedom. This finite exposure is a hypothesis test, not assumed
  convergence. No automatic64epoch continuation is included.
- Run the45,056-field development panel above. Checkpoint progression and
  fixed/oracle controls use128NFE; the final-checkpoint ladder measures64/128/256.
  Keep all32development pairs for progression. Freeze16ladder/control pairs:
  one per cap/redshift/support stratum, assigning012when
  `(cap_index+shell_index+support_index)%2==0`, otherwise013. UseNGC/SGC=0/1,
  shells0--3andinterior/boundary=0/1. This gives each phase8cases, both caps,
  all shells and both support classes; simple ordinal alternation would confound
  support with phase. Primary development/confirmation offset is(0,0,0).
  Neither confirmation nor sealed
  phases can choose optimizer, architecture, checkpoint or sampler.
- The component baseline is50.1503training+149.4131sampling+0.0271transfer proxy
  =**199.59GPUh**. A30% allowance gives259.47GPUh, rounded to260. It includes
  implementation smoke, warmup, checkpoint/orchestration inefficiency, one
  validated retry from an intact checkpoint, and GPU-to-host/output overhead.
  This is headroom, not a measured prediction of those missing components.
  Exceeding it stops the run; no silent resource escalation or repeated retry.
- CPU baseline is7.14four-CPU worker-hours.24CPU-nodeh allows final metric
  implementation/validation, phase-level aggregation, bootstrap/reference-point
  diagnostics and contention without assuming unmeasured many-worker scaling.
  512GiB covers retained science/restart checkpoints, development fields and
  bounded diagnostic products. Enforce the cap before writes; do not retain
  full native-box tensors or copy particle catalogues into the experiment.

Use independent one-GPU factor/draw tasks. A full four-GPU node is appropriate
only with four ready workers and a successful concurrent technical smoke; count
all allocated GPUs in the cap, including idle periods. Keep at most two total
allocations and no unbounded successor chain. Review the finalized Slurm run mode
at launch: user-approved production batch after its smoke, or an explicitly
bounded interactive protocol; tmux alone is not compute persistence. Checkpoint
and source provenance must survive either mode. No distributed optimizer rewrite
is needed. Larger inference batches may be an engineering optimization later,
but must pass addressed-draw/scalar parity and throughput checks before any
cost discount is claimed; the budget here credits only measured batch2 rates.

### Stage B: a later, explicit confirmation decision

If development shows a mature, numerically stable and scientifically promising
joint posterior, freeze checkpoint32 and the final metric/sampler manifest, then
ask for **128draws on all96confirmation pairs, both seeds and all four arms**.
Do not select just the visually best arm or drop adverse phases/strata. A common
128NFE panel costs329.55networkGPUh; propose a separate **450GPUh/32CPU-nodeh**
ceiling, keeping the SAME cumulative512GiB new-output limit. This includes more
than30% GPU headroom and about2x the15.89CPU worker-hour baseline. At64NFE, if
independently qualified on development, the corresponding GPU ceiling can be
reduced to220; that choice must be frozen before confirmation predictions.

The256-NFE and256-draw alternatives are costed above but are **not included** in
these proposed ceilings. If128vs256NFE has not stabilized, stop before opening
confirmation and review a sampler-specific change. If16to32epochs still shows
strong joint-score improvement, present the measured case for a separate64epoch
extension; do not label the model mature or change its architecture prematurely.
If no meaningful joint gain appears despite mature fits/stable sampling, stop
this matched neural comparison without buying the larger confirmation ensemble.

### Proposed scientific decision contract to implement before predictions

Freeze scales/weights/windows/probe locations from training geometry and
training truth only. Use equal phase weighting, paired arm differences, separate
seed reports and phase-cluster uncertainty; six confirmation phases are not
thousands of independent voxel replicates. Development has only two independent
phases, so its trends are screening evidence, not a calibrated significance test.

1. **Numerical invariants first.** Targetless inference, address-stable replay,
   positive density, finite trajectories, residual block means and coarse/fine
   mass agreement must pass the existing2e-6tolerances. I/J share EXACTLY the
   same coarse fits/draws. Their block-aligned regional masses/ranks must agree;
   improvement of those masses in J over I would indicate a bug, not success.
2. **Keep the three power quantities separate.** Use a separable periodic
   Tukey window (`alpha=0.25,sym=False`) on the64x48x48 rectangle and the linear
   transform`L(delta)=w*(delta-sum(w*delta)/sum(w))`. Use6.766Mpc/h cells and
   band edges`[0.02,0.035,0.05,0.075,0.10,0.15,0.22,0.30]h/Mpc`, with identical
   rFFT Hermitian multiplicities for truth and every arm. Normalize per-mode
   pseudo-power by`volume/(Nvox^2*mean(w^2))`; the common DC projection does NOT
   discard regional mass from its separate tests. Report sample power,
   posterior-mean/error power and stochastic residual power with the finite-M
   conventions above. The windowed finite-volume spectrum is a pseudo-spectrum,
   not an unbiased full-box cosmological spectrum. Never force each draw to
   match its paired truth's individual band powers. Coarse/wide power is a
   separate diagnostic from fine residual dependence.
3. **Primary coupling gain: J-VDM versus I-VDM.** Use fine-sensitive, fixed
   physical-density probes, not a score dominated by invariant regional masses.
   Within each owned16-cubed core, start with the seven separable DCT-II modes
   `(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)`, where the
   one-dimensional factor is`cos(pi*k*(i+0.5)/16)`. Subtract each4-cubed block's
   weight mean, then unit-L2 normalize the resulting weight. Its inner product
   with physical delta annihilates every block-constant coarse contribution.
   The two cores give a14-vector, scaled by TRAIN-only component standard
   deviations. Fail on zero scale or a degenerate probe; do not fit a singular
   covariance inverse. Use the seven MATCHED left/right mode pairs, equal
   weights and exponent0.5for the primary cross-core fair variogram; report all49
   cross-mode pairs as a supporting diagnostic. Require at least10% lower
   primary variogram score in each seed, with fair joint energy score no worse
   than2% and no phase-averaged marginal
   density/eigenvalue/gap CRPS degradation above5%. Register cross-core pairs
   explicitly; the generic all-pairs variogram also mixes intra-core effects.
   Require the pooled phase-cluster90% interval for the variogram gain to exclude
   zero and the upper energy-regression bound to remain below2%. Do NOT require
   a10%energy gain: the exact-Gaussian sensitivity check below shows why that
   would reject a correct dependent posterior. If the small phase panel cannot resolve it, report
   inconclusive rather than replacing phase uncertainty with voxel bootstrap.
   Relative-gain criteria require a positive reference score; if a fair finite-
   ensemble score is zero/negative, do not divide by it or redefine the gate
   after seeing confirmation. Report that comparison as unresolved and retain
   the predeclared absolute paired-score differences and uncertainty.
   Also report the12-vector of each core's mean density, three mean eigenvalues
   and two mean gaps. It has algebraic dependencies and coarse-invariant density
   components: retain it as a physical joint diagnostic, not12 independent
   modes or a stand-alone fine-coupling promotion gate. The CPU benchmark timed
   this12-summary calculation; the added14-probe score is a small UNMEASURED
   analysis addition covered by the declared CPU headroom, not a claimed timing.
4. **Coarse and calibration gates are separate.** On the common rectangle use
   eight24-cubed fine-grid mass probes starting at every combination of
   `x in(8,32), y in(0,24), z in(0,24)`. Each is162.384Mpc/h on a side and
   block-aligned; all can be evaluated exactly from the sampled coarse field.
   These locations are fixed common probes, not unchanged legacy parent octants.
   Add eight fine-sensitive12-cubed probes at
   `x in(17,33), y in(17,19), z in(17,19)`, all within owned cores but offset
   from coarse boundaries. Their overlap does not create independent evidence.
   Report the whole wide field and its eight24-cubed COARSE-grid octants as
   additional large-scale diagnostics, not pooled replicas of the local probes.
   Nominal90% coverage uses the exact90.6977% finite-M
   reference for128draws. Proposed practical pooled tolerance is5percentage
   points, with the phase-cluster90% interval containing the reference and no
   seed outside that tolerance; report50/68/95% levels as well. Report cap,
   redshift, support, observed-density strata with uncertainty and retain
   low-count strata rather than inventing a pass. No condition stratum may be
   promoted on an unresolved interval. Pointwise tidal/eigengap coverage and
   proper scores must also satisfy the marginal5%nonregression requirement.
5. **Joint-power and residual-spread gate.** Use the bands above; primary
   eligibility requires at least100Hermitian-weighted nonzero modes and
   training-mean truth pseudo-power at least1e-4times the largest training band.
   Freeze this mask before predictions; report excluded bands diagnostically.
   Fail the design screen if fewer than three eligible bands remain. Require the
   phase-averaged sample/truth power ratio within10% in those bands and the
   residual-spread/truth-minus-mean ratio within10% after the1+1/M correction,
   in both seeds, with phase-level intervals reported. Cross-core covariance,
   cross-correlation and non-block-aligned mass coverage must be reported
   even when one-point statistics look good. Numerical tolerances and these
   scientific tolerances are different quantities; none is a guarantee of
   full conditional calibration.
6. **Sampler/maturity screen.** On development require128to256NFE changes below
   2% in registered proper scores and band-power ratios, and below2percentage
   points in pooled coverage, separately for both seeds. For a64NFE choice,
   require the same64to128AND128to256conditions. Address underlying noise
   increments consistently within each objective; VDM and CFM need not share
   pathwise noise trajectories. A final16to32epoch joint-score gain above5%
   indicates unfinished learning, not a reason to open confirmation or declare
   an architecture failure. Small32-draw coverage panels have limited power;
   failure to resolve sampler drift is inconclusive, not proof of convergence.
7. **CFM decision.** Apply the same power, calibration and marginal gates to
   J-CFM. Compare its joint proper scores with J-VDM in both seeds and report
   allocated compute/NFE. A CFM win is a win of its path/objective/sampler
   package, not proof that flow matching alone repairs missing dependencies.
   Do not trade failed regional-mass coverage for better-looking spectra.

The fixed-mean and oracle-coarse panels diagnose whether failure sits in the
coarse posterior or the conditional fine factor; deterministic regional masses
are excluded from continuous-coverage tests. An optional core-permutation
diagnostic can reuse these saved FIXED-coarse draws without more sampling.
Do not shuffle cores across different sampled coarse fields and claim that only
fine dependence was removed: that also breaks the shared coarse uncertainty.
Reference-point/TARP-style diagnostics on predeclared summaries are supporting
checks, not proof of statistically optimal field posteriors. No single summary
test establishes calibration of the full DESI-conditioned random field.

An analytic metric-sensitivity control motivates this distinction. Consider
14standard-normal coordinates, seven in each core, with correlation rho between
matching core modes and all other covariances zero. The independent approximation
has EXACTLY correct marginals. Using64/128-node generalized Gauss-Laguerre
quadrature for expected Gaussian norms, the exact dependent posterior improves
population fair energy over that approximation by only0.2043%at rho=0.5,
0.5658%at0.8and0.9422%at0.99. The two quadrature orders agree to better than
1.1e-8in relative gain. A10%energy-improvement gate would therefore fail even
the oracle in this example. This is a synthetic mathematical control, not a
cosmological performance result. The matched-pair variogram is substantially
more sensitive to this constructed dependence; the generic49-pair average
dilutes it with42uncorrelated pairs. Tests preserve the known correct-marginal
counterexample and validate the population score formula before scientific use.

Separate absolute qualification from incremental promotion. If I already passes
the registered joint/power/calibration battery and J adds no reproducible gain,
the conclusion is that tested extra coupling was unnecessary at this resolution,
not that the entire field-posterior programme failed. Prefer the simpler qualified
model. If neither qualifies after mature training and stable sampling, the
negative-result stop rule below applies. A statistically unresolved difference
is not an architecture impossibility result.

These proposed thresholds are fixed BEFORE predictive results, not selected from
confirmation performance. The later implementation must materialize exact
windows/bands/probe coordinates/stratum and RNG manifests and test their algebra
before scientific approval is exercised. A failure of the matched, mature,
sampler-stable comparison triggers forward/selection-model review or a bounded
physics-likelihood posterior reference—not another marginal-only architecture
sweep. This does not authorize a survey-scale BORG or real-DESI campaign.

## Go/stop handoff

The all21-phase products and independent source/payload audits, three-phase
coordinate and selection-volume checks, exact13-phase global normalizer,
predeclared physical-reference gate and actual-artifact targetless/role-guard
checks have passed. Failures/recoveries and the original preparation resource
ledger are preserved in the separate closeout and evidence archive.

The joint-power/mean-residual decomposition, regional-mass, adjacent-core
proper-score/variogram, tidal/eigengap and conditional-coverage tolerances,
marginal nonregression and finite training/draw caps are specified above. After
approval, encode them into a frozen machine-readable manifest and test the
scientific workflow before fitting. No post-hoc relaxation or automatic longer fit after a failed
matched comparison. The decision rationale remains the
[joint-field literature recommendation](e2e_vdm_context_joint_decision_literature_20260918.md).
