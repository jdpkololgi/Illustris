# Controlled VDM diversity, spatial context, and shared matter uncertainty

Current status2026-09-17: physical gate FAILED, training_launch_allowed=false;
see docs/e2e_vdm_context_representation_failure_20260917.md. All data products
complete, no GPU fits. The operator formula below is the failed registered v1,
not an approved repaired prescription. Await explicit direction on its amendment.

Approved 2026-09-17 as the active goal: implement, execute, verify and report this
bounded experiment. No ph001/ph006 access; no old artifact overwrite.
Implementation/audit clarification: docs/e2e_vdm_context_audit_proposal_20260917.md
is incorporated under the resumed approved goal before any new predictive scores.
It clarifies pooled contrast thresholds, noncancelling component coverage,
coarse-conditioning limitations and same-draw tidal/wide refinement checks.

## Audited starting point

Previous assessment: all3584 draws complete in
`vdm_assessment_20260917_v2/analysis/RESULTS.json`, SHA256
`89cf73cab5878ed89708f9585bbc50efad6e4da201cc099af108465928932cde`.
500-to-1000-step maximum mean power change0.590327% (MC95 bound0.604442%),
maximum C90 width change0.528116%; all16 refinement screens pass. Density CRPS
worsens6/8 from2048to5120. Evaluation cost approximately6.999GPUh. Improving
correlation is not calibration; do not repeat completed controls without reason.

| Representation | Grid | Cell / side (Mpc/h) | Fundamental k (h/Mpc) |
|---|---:|---:|---:|
| Generated density parent |48^3|6.766 /324.768|0.0193467|
| Science core |16^3|6.766 /108.256|0.0580401|
| Existing wide observations |96^3|13.532 /1299.072|0.00483667|
| Proposed wide context/coarse |48^3|27.064 /1299.072|0.00483667|

Target:2^3 average of original cubic-sampled R7 Gaussian density at z0.2, log1p
then global train-only affine; no second R7 smoothing. Observer redshift0.15--0.55
changes selection, not epoch. Existing24 channels are12 spatial local +12
broadcast wide means. Wide arrangement is absent, not its footprint. No patchwise
DC removal; windowed Fourier diagnostics remove DC, so measure regional masses
separately. Distinguish unavailable observations, exterior matter and tidal
boundary errors. Correct local posteriors can marginalize unseen surroundings.
P3b random response supplies angular support/targetability, NOT audited C_fibre
or C_z. Freeze this observation contract; no real-DESI readiness claim.

## Matrix and roles

| Arm | Data | Observation information | Generated matter |
|---|---|---|---|
| A |32 balanced patches, ph000/002|local + wide summaries|central parent|
| B |384 balanced patches, ph000/002/003|same as A|central parent|
| C |same384|spatial wide context|central parent|
| D |same384|same as C|shared stochastic coarse + conditional fine|

Fresh fits, seeds0/1; old ten-NGC models are historical only. A:B tests diversity;
B:C spatial information; C:D stochastic multiscale package, confounded with the
extra coarse-model capacity/cost. Not a full factorial or many-phase learning curve.
B=3phases x2caps x4shells x2support strata x8anchors. A is nested one per stratum
per phase000/002. Geometry-only choice; central108.256 cores nonoverlap; periodic
source-box centre separation>=162.384Mpc/h withinphase including cross-cap aliases.
Parent/wide overlap within training phase is allowed/recorded, not independence.
Max65536 candidate centres per phase/cap; unmet quotas STOP, no distance relaxation.

Train000/002/003; development004; internal confirmation005; forbidden001/006.
003 has prior development use;004/005 historically programme-exposed, not globally
blind. Whole phases separate roles. New role-aware reader; preserve old train-only
guards. Separate observations from targets. All shared preprocessing fits A32 only;
D-only charts fit those same training contexts. Original products remain unchanged.

Fixed gamma[-13.3,13.3], full VLB, residual3D U-Net base24/3levels/bottleneck attention.
AdamW lr1e-4,wd1e-5,clip0.5,batch2,deterministic FP32.20480updates/factor,
40960presentations; checkpoints5120/10240/20480, durable save every256. D coarse
factor has same updates in addition to fine. Pair initialization/noise/times/order
where compatible. No EMA/AMP/auxiliary loss/spectral penalty/optimizer sweep.
Seven context offsets common to all arms:0 and +/-108.256Mpc/h along each axis;
these do not add independent fields and cover adjacent-core relative geometry.

## Models and physical contract

Retain24 current inputs. Common compact wide encoder widths8/16/32, three stride2
convolutions, projection192 and bottleneck cross-attention with physical positions.
A/B feed constant wide summaries; C/D spatial observations. Reserved local coarse
plane and wide coarse channel are zero A--C, generated D. Fine~8.37M parameters
each; D coarse12-input VDM~8.19M extra. Record actual counts.

rho=1+delta. D factorizes p(rho_c,u|X,S)=p(rho_c|X,S)p(u|rho_c,X,S).
Coarse rho is aligned4^3 fine-voxel physical-density mean across wide48^3 domain,
NOT old Fourier-lowpass products relabelled. u=logrho-blockmean(logrho); decode
rho_i=rho_c exp(u_i)/blockmean(exp(u)). Positive, block-mass consistent. Fine
diffusion projects noise/predictions/states to block-zero-mean subspace; VLB uses
63DOF per64block, including correct prior/decoder dimension. True coarse is
supervised training only; inference uses generated coarse; oracle labelled separately.
Draw address binds model/domain/draw; adjacent cores share cached coarse realization.
The coarse factor conditions on the12 wide observation channels only; the fine
factor also receives local observations. This restricted factorization assumes
wide summaries suffice for coarse mass; that assumption is not established.
Negative D results cannot rule out a more general joint conditional posterior.
Owned-core results invariant to request order; overlapping halos diagnostic only.
Shared coarse is necessary, not proof of full fine cross-patch coherence.
Implemented counts:8,371,907/fine arm; D coarse8,188,395 additional parameters.

Tides: matched-patch and fullbox physical reference; average six tensor components
BEFORE eigenvalues. D composite U T(delta_c)+T_local(delta_f-U delta_c), U block
replication; trace closes without double-counting. Exterior beyond1299 remains absent.
Training-only pre-fit gates: positive rho; roundtrip/block-mass/tensor-trace error
<=2e-6; plane/DC declared-window transfer k=.01/.02/.04/.08; >=25% median reduction
in EACH normalized eigenvalue RMSE vs parent-only closure. This median is taken
over per-anchor relative RMSE reductions on the A32 training panel. Tails/topology reported.
Failure stops for review; no automatic new grid or retrospective gate changes.

## Evaluation and interpretation

16anchors each004/005, cap x4shell x2support, fixed geometrically before truths.
Eight centralmodels:00432draws/anchor at5120/10240; bothphases64 at20480;
eight sentinels128 final; two fixed fit anchors32 at all3checkpoints. Four adjacent
core pairs (one/cap/evalphase),32jointdraws. D mean-coarse32 and oracle16 draws
on samepairs, separate attribution. Coupled solver8draws on2development anchors
permodel at250/500/1000steps, reuse250. Total33280central +7872coarse draws.
Final20480 primary, no best-checkpoint selection using confirmation. Confirmation
scores opened only after all model/protocol choices frozen.

Sample power AND power of posterior mean; residual Fourier variance, r, meanRMSE,
one-point and regionalmass; bias,widths,ranks,C50/68/90/95,fair finite-M CRPS;
ordered eigenvalues/gaps and jointenergy; crosscore covariance/differences/joint
energy/variogram of six summaries percore. Order-statistic intervals report actual
attainable coverage. Gaussian/logGaussian rank/score tests; don't silently mix old
empirical CRPS with fair CRPS. Parent325spectra, D1299low-k; unavailable wide
quantities A--C marked N/A. No requirement eachdraw matches pairedtruth P or r=1.

Deployable conditioning strata: tracer density,response,redshift,boundary; truth
environments descriptive. Equalanchor/stratum/phaseweight; report seeds/phases
separately; paired500Mpc/h source-block bootstrap,1000Mpc/h sensitivity. Two
evaluation phases limit inference; correlatedvoxels are not independent SBC trials.
Positive contrast: >=10% equally pooled primaryscore improvement, positive
direction in EACH of four seed/phase cells; >=5percentagepoint coveragegap
reduction OR within5points of finite-M target in each cell. Compute tidal gaps
per eigenvalue before averaging, never cancel under/overcoverage. Density
meanRMSE degradation<=5%, aggregate samplepower discrepancy degradation
<=.05abslogunits. A:B/B:C primary standardized densityCRPS; C:D joint physical
tidal energy plusdependence. Partial/mixed/inconclusive allowed, not allmetrics
monotone. Sampler gate:250:500<5%,500:1000power/width<2% with paired uncertainty
excluding5%; failure stops finalsampling, no automatic solversearch.
The same coupled draws must also pass tidal/eigengap width and D-wide power/width
refinement;8 draws yield a7/9 attainable central interval, not literalC90.

## Bounded authorization and milestones

Expected87--99GPUh:smoke3--4,fits12--16,main draws65--70,other diagnostics7--9.
Hard112GPUh includingidle/recovery;8CPU-nodeh;300GiB newScratch;48h elapsed from
firstallocation includingwaits. Max8GPUrequests,each<=4h,aggregatecap overrides.
One4GPU node at a time/fourworkers; sharedinteractive for tail; max2interactive
allocations includingCPU. desi_g/desi,Scratchlicense,immediate600. One bounded
infrastructure replacement withinceilings. Numeric/hash failure or forecast above
cap stops, never silently reduce protocol. Old benchmark .18s/update,5.5s per
batched250stepdraw; C/D costs unmeasured until smoke. Deterministic finite tmux
launcher, no agent/resource decisions; tmux doesn't extend Slurm walltime.

1. Correct old final result record; freeze contract/config/role ledger.
2. Build role-aware data, geometry/normalization/physical-representation gates.
3. Context/multiscale model, real checkpoint resume and shared-draw addressing.
4. Fullsize GPU smoke: parity, RNG replay, no-oracle inputs and cost forecast.
5. Execute/verify/report H1/H2 with figures; update log/fieldplan and one next step.

Completion is the whole experiment and honest assessment, not continued tuning
until positive. Failed gates preserved; no transformer/CFM/preservation detours.
