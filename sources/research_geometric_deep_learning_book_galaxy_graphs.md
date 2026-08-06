# Geometric Deep Learning book: implications for galaxy-catalogue graphs

Research lookup: 2026-07-23.
Project-plan and training-protocol integration update: 2026-07-26.

## Executive assessment

The newly released *Geometric Graphs* chapter is directly relevant to this
project, but its most useful consequence is not “replace the current encoder
with a large SE(3)-equivariant network.”

The book provides a cleaner way to separate four questions that have sometimes
been bundled together:

1. **Permutation symmetry:** galaxy row order must not matter.
2. **Spatial symmetry:** predictions should transform consistently when the
   physical catalogue, observer-frame quantities, and selection geometry are
   transformed together.
3. **Geometric expressivity:** the network needs enough angular/body-order
   information to represent anisotropic environments, not merely distances.
4. **Sampling measure:** an observed galaxy catalogue is a biased,
   non-uniformly sampled point process. Equivariance alone does not correct
   selection, incompleteness, or shot noise.

The current programme already has strong evidence that graph support,
fixed physical scale, and the physics-grounded field decoder are first-order
levers. The book strengthens that interpretation. A high-value next
geometric experiment would be a small, explicitly interpretable
`ell <= 2` multipole/tensor message block, optionally combined with a
random-catalogue or inverse-intensity quadrature correction. A matched-capacity
Equiformer/SE(3)-Transformer should remain deferred unless eigenvectors,
intrinsic alignments, or a clean lower-cost ablation gives a positive gate.

## Status of the book

The official book page currently exposes eight chapters:

1. Introduction
2. Foundations of Supervised Learning
3. Foundations of Equivariant Deep Learning
4. Foundations of Geometric Deep Learning
5. Graphs
6. Grids
7. Group Convolution on Homogeneous Spaces
8. Geometric Graphs

Chapters 1--8 constitute the released foundations and domain material. The
site still labels Part III, “Geometric Deep Learning at the Bleeding Edge,” as
TBD and says MIT Press will release the final version in 2026. Chapter 8 is
therefore the final released domain chapter, not the end of every planned part.

Official contents:
https://geometricdeeplearning.com/book/

The book asks readers to cite the earlier proto-book until the final book is
published:

- Bronstein, Bruna, Cohen & Veličković, *Geometric Deep Learning: Grids,
  Groups, Graphs, Geodesics, and Gauges*, arXiv:2104.13478.
  https://arxiv.org/abs/2104.13478

## Chapter-by-chapter relevance to this project

### Chapter 1: Introduction

The organising principle is the Geometric Deep Learning blueprint:

- choose the domain on which the signal lives;
- identify the transformations that preserve the learning problem;
- construct invariant/equivariant local operators;
- separate scales through pooling or coarsening.

For this project, the input is not merely an abstract graph. It is a finite
sample from an underlying three-dimensional matter field, observed in
redshift space through a survey selection function. This already signals that
the domain, symmetry group, and sampling measure must be specified together.

Official chapter:
https://geometricdeeplearning.com/book/introduction.html

### Chapter 2: Foundations of Supervised Learning

The chapter separates approximation, estimation, and optimisation error and
argues that high-dimensional learning needs strong regularity assumptions.
This maps cleanly onto current evidence:

- a richer encoder cannot recover information destroyed by sparse sampling or
  shot noise (estimation/information limit);
- an under-capacity or prematurely stopped SEGNN does not test the whole
  equivariant model class (approximation/optimisation confound);
- a random spatial split can make an expressive model appear successful by
  interpolation rather than generalisation (evaluation-design error).

This chapter supports the existing policy of matched-capacity controls,
spatial/fresh-phase validation, and explicit convergence diagnostics.

Official chapter:
https://geometricdeeplearning.com/book/foundations.html

### Chapter 3: Foundations of Equivariant Deep Learning

The important distinction is between:

- **invariance**, where the output does not change under a transformation; and
- **equivariance**, where the output transforms in a prescribed way.

For the present targets:

- ordered tidal-tensor eigenvalues and T-Web classes are scalar, rotation- and
  translation-invariant physical quantities;
- a tidal tensor or eigenvector prediction is rotation-equivariant;
- galaxy identities are permutation-equivariant;
- the survey observer and redshift-space line of sight break naive global
  translation symmetry.

The correct group is therefore task- and representation-dependent. For a
periodic real-space simulation box, `S_n x E(3)` is a reasonable starting
point. For an observer-centred lightcone with RSD, the exact physical symmetry
is closer to global rotations about the observer, with the line-of-sight
vectors transformed covariantly. The angular mask and selection function must
also be transformed, supplied as inputs, or treated as a symmetry-breaking
nuisance. A strict translation-invariant model is not automatically correct
for a lightcone.

Official chapter:
https://geometricdeeplearning.com/book/algebraicpriors.html

### Chapter 4: Foundations of Geometric Deep Learning

The chapter adds three priors beyond symmetry:

- locality;
- stability to small deformations;
- scale separation/multiscale processing.

These may matter more than exact equivariance here. The target uses a fixed
7 Mpc/h smoothing scale but the Delaunay graph has a density-dependent
physical receptive field. The union of Delaunay and fixed-radius edges is a
reasonable geometric response: adaptive void bridges plus a fixed physical
aperture. The field-level U-Net and graph-to-field decoder instantiate scale
separation more explicitly than a flat message-passing stack.

The chapter therefore supports the current interpretation that graph
construction and multiscale representation are first-order, while strict
equivariance is a possible regulariser or capability extension.

Official chapter:
https://geometricdeeplearning.com/book/geometricpriors.html

### Chapter 5: Graphs

The graph chapter establishes permutation symmetry, convolutional,
attentional, and general message-passing GNNs, the Graph Network formalism,
and expressivity limits related to Weisfeiler--Leman tests.

The current two-pass attentional GraphNetwork is squarely inside this
framework. Attention and spatial equivariance are orthogonal choices:
attention controls how candidate messages are weighted; equivariance
constrains what the message/update functions may compute. A union graph with
two physically distinct edge populations gives attention a concrete role:
arbitrating fixed-aperture edges and long Delaunay bridges.

The chapter's discussion of computation-graph design is also consistent with
the observed failure of dynamic feature-space kNN. Greater graph flexibility
is not useful if it replaces physically meaningful neighbours with unrelated
feature-space neighbours.

Official chapter:
https://geometricdeeplearning.com/book/graphs.html

### Chapter 6: Grids

Grids are graphs with a canonical arrangement and translation symmetry.
Convolutions and Fourier methods follow from that extra structure.

This chapter gives a principled interpretation of the successful field-level
branch. Voxelisation is not an abandonment of graph learning; it changes the
domain to a regular lattice where fixed-scale filters, pooling, and the FFT
tidal operator are natural. The current

`graph encoder -> density grid -> fixed FFT tidal operator -> eigensolver`

is a hybrid GDL design: an irregular geometric encoder feeding a regular
domain whose physics is known.

Official chapter:
https://geometricdeeplearning.com/book/grids.html

### Chapter 7: Group Convolution on Homogeneous Spaces

This chapter generalises convolution from translations to other symmetry
groups and homogeneous domains, including the sphere.

The most relevant observer-frame idealisation is a lightcone domain
`R_+ x S^2`: radius/redshift plus angular direction. A factorised
radial-by-spherical model, or spherical processing of mask/selection channels
within radial shells, is better motivated by this chapter than treating the
full observed wedge as translation-homogeneous Euclidean space.

This is an adjacent option, not an immediate replacement for the point graph.
It becomes more attractive for full-sky or large-cap field products, angular
selection modelling, and cross-survey transfer.

Relevant primary cosmology precedent:

- Perraudin et al., *DeepSphere: Efficient spherical Convolutional Neural
  Network with HEALPix sampling for cosmological applications*,
  arXiv:1810.12186. https://arxiv.org/abs/1810.12186
- Defferrard et al., *DeepSphere: a graph-based spherical CNN*,
  arXiv:2012.15000. https://arxiv.org/abs/2012.15000

Official chapter:
https://geometricdeeplearning.com/book/groups.html

## Chapter 8: Geometric Graphs

Official chapter:
https://geometricdeeplearning.com/book/geometricgraphs.html

### 1. Domain and symmetry

A geometric graph contains abstract scalar features `s_i` and coordinates
`x_i`. Its symmetry is richer than the permutation group of an abstract
graph. In the simplest Euclidean setting, the model is constrained by the
product of node permutations and continuous spatial transformations:

`S_n x E(d)`.

For scalar graph outputs, rigid transformations should leave the output
unchanged. For vectors, coordinates, or tensors, the output should rotate or
reflect with the input.

This is directly applicable to galaxy catalogues, but the observer-centred
survey geometry modifies the group. The line of sight is a covariant vector,
not a scalar nuisance, and translations are physically broken by the observer
and radial selection.

### 2. Invariant geometric GNNs

Distance-based models such as SchNet construct invariant messages from
pairwise distances. Angular models such as DimeNet introduce three-body
information through angles. These models are computationally attractive and
often adequate for scalar targets.

The book also describes counterexamples in which distances and angles do not
distinguish two point configurations. This is an expressivity result, not
evidence that every application needs high-order tensors. For parity-even
tidal eigenvalues, chirality is unlikely to be the primary bottleneck; the
more relevant missing structure is likely a controlled representation of
anisotropy and scale.

### 3. Cartesian equivariant models

EGNN-style models compute invariant scalar coefficients and multiply them by
relative displacement vectors. This yields simple, scalable coordinate/vector
updates without a full spherical-tensor algebra.

Coordinate updates are not well motivated for the current regression task:
galaxy positions are observations, not latent dynamical coordinates that the
network should move. The useful part is the construction rule—combine
invariant coefficients with covariant displacement or LOS vectors—not the
coordinate-update head.

Primary references:

- Satorras, Hoogeboom & Welling, *E(n) Equivariant Graph Neural Networks*,
  arXiv:2102.09844. https://arxiv.org/abs/2102.09844
- Brandstetter et al., *Geometric and Physical Quantities Improve E(3)
  Equivariant Message Passing*, arXiv:2110.02905.
  https://arxiv.org/abs/2110.02905

### 4. Spherical tensor models

Tensor Field Networks and SE(3)-Transformers represent scalars, vectors, and
higher tensor orders using irreducible representations. Spherical harmonics
provide direction-dependent filters; Clebsch--Gordan products couple tensor
orders without breaking equivariance. Attention weights remain invariant
while the values are equivariant.

This is the chapter's most important mathematical connection to tidal-field
inference. A symmetric rank-2 tensor decomposes into:

- an `ell = 0` trace (the isotropic/density-like component);
- an `ell = 2` symmetric-traceless component (the tidal shear);
- no `ell = 1` component when the tensor is symmetric.

The tidal target is therefore unusually well matched to a deliberately
truncated `ell <= 2` representation. A full high-order model is not required
to test whether this inductive bias is useful.

Primary references:

- Thomas et al., *Tensor Field Networks*, arXiv:1802.08219.
  https://arxiv.org/abs/1802.08219
- Fuchs et al., *SE(3)-Transformers: 3D Roto-Translation Equivariant Attention
  Networks*, arXiv:2006.10503. https://arxiv.org/abs/2006.10503
- Duval et al., *A Hitchhiker's Guide to Geometric GNNs for 3D Atomic
  Systems*, arXiv:2312.07511. https://arxiv.org/abs/2312.07511

### 5. Manifolds, meshes, and gauges

The chapter derives convolution on manifolds through tangent spaces,
geodesics, local frames, and gauge-consistent transport, then connects those
ideas to mesh CNNs.

Galaxy positions do not generally lie on a two-dimensional mesh, so mesh
convolutions are not a direct model for the 3-D cosmic web. Two ideas remain
useful:

- the observer-centred angular domain is spherical;
- a line of sight defines a local radial axis while the azimuthal basis around
  it is arbitrary, creating an `SO(2)` local-frame/gauge issue.

A local LOS frame should therefore use gauge-invariant contractions or a
gauge-equivariant construction; a hand-chosen transverse x/y basis can inject
coordinate artefacts.

### 6. AlphaFold 2 case study and the book's own caution

The AlphaFold case study uses invariant point attention and local coordinate
frames. More important for this project, the authors explicitly discuss
ablation evidence suggesting equivariance can be one incremental contribution
rather than the dominant source of performance. They also note that later
systems can retain geometric benefits through broad augmentation without
strict equivariant components.

That perspective agrees with this project's evidence: geometry matters, but
graph support, scale, optimisation, data coverage, and the output physics can
dominate the architectural symmetry choice.

## What the current project already implements

The current P8 G-PATCH has:

- a Delaunay plus fixed-radius union graph;
- eight scalar node inputs: seven graph/density/inertia summaries plus the
  frozen selection-density channel;
- five edge inputs: length, the three Cartesian components of the unit edge
  direction, and density contrast;
- two receiver-normalised attentional GraphNetwork passes;
- a three-scalar per-node output trained in the P8 frozen **linear-increment**
  space.

Relevant files:

- `workflows/abacus_tweb/p8_prepare_graph_features.py`
- `workflows/abacus_tweb/p5_build_graph_patch_adapter.py`
- `workflows/abacus_tweb/p8_train_graph_patch.py`

This model is spatially informed but is not exactly E(3)/SO(3)-invariant. The
raw `x_dir, y_dir, z_dir` components are consumed by unconstrained MLPs.
They may encode useful observer/footprint structure, but they can also enable
dependence on arbitrary global axes.

There is an important target-policy distinction. Repository-wide guidance
identifies ordered softplus increments as the preferred constrained
parameterisation for new model stacks. P8 deliberately froze the older linear
increment baseline

```text
(lambda1, lambda2 - lambda1, lambda3 - lambda2)
```

to isolate the effect of the training protocol. The P8 heads emit three
unconstrained scalars and cumulative summation does not prevent a negative
predicted gap. This is why P8 reports ordering violations. Do not describe the
current P8 outputs as softplus ordered, and do not silently change this
contract inside the in-progress comparison. After P8 closes, a bounded
ordering-enforcing head comparison must be resolved before the P10/P12
production freeze.

Important terminology: P8 “rotation 0/2” refers to fold-role rotations, not
physical rotations of the catalogue. Those replications do not test spatial
rotation invariance.

Prior direct symmetry tests in this repository are also important:

- the SEGNN positions+LOS experiment implemented exact equivariance and an
  `ell = 0 + ell = 2` tensor head;
- it underperformed the non-equivariant controls, but at substantially lower
  capacity and high computational cost;
- this is evidence against that tested configuration, not a general failure of
  geometric equivariance;
- the graph-to-field physics decoder and U-Net results make scale/output
  representation the current higher-priority path.

Relevant files and record:

- `workflows/sbi/gate_g4_p1b_segnn.py`
- `workflows/sbi/gate_g4_egnn_smoke.py`
- `SCIENCE_LOG.md`, entries dated 2026-07-03 through 2026-07-08.

## Integration with the generalisable GraphWeb VAC plan

This section connects the book to
`docs/plan_generalisable_graphweb_vac.md` and to the empirical state recorded
in `SCIENCE_LOG.md`. The evidence snapshot below follows the newest science-log
entry available at this update, dated 2026-07-22.

### What has actually transferred so far

| Generalisation axis | Current evidence | What is not yet established |
| --- | --- | --- |
| New labelled sky geography inside `ph000` | The completed two-rotation P8 recovery gives U-PATCH mean macro R2(lambda1) `0.5035`, versus G-PATCH `0.4695`. U-PATCH also beats the supported-shell CIC bar: `0.5621` versus `0.520`. | A new graph from an independent density-field phase, a new HOD family, or DESI truth. |
| Fold stability | U-PATCH fold spread is `0.0185`; G-PATCH spread is `0.0026`. The supported shells replicate more closely than the sparse shell. | Multi-seed uncertainty and all-five-fold replication. |
| Other eigenvalues | On the complete rotation-0 validation fold, pooled U-PATCH R2 is `0.572/0.658/0.727` for lambda1/lambda2/lambda3. | A registered fresh-phase score for all three components. |
| Optimisation | Every primary recovery run peaked at its final available epoch. U-PATCH reached the 20-epoch cap while improving. | A measured convergence ceiling. The registered long-horizon extensions were still in progress in the latest log and remain diagnostic rather than final evidence. |
| Sparse sampling | Learned models shrink safely and remain correlated in the sparsest shell, whereas CIC amplifies noise and reaches strongly negative R2. | Recovery of information removed by shot noise. No architecture or weighting can recreate absent tracers. |
| Model complementarity | A spatially out-of-sample U+G blend gains only `+0.012` macro. A global U+CIC blend loses `0.020`, although CIC helps the three supported shells. | A useful density-gated hybrid. The current global classical residual is a no-go, not an endorsed production branch. |
| Fresh cosmic phase | None. The asset inventory found many source phases but T-Web-labelled GraphWeb products only for `ph000`. | The P10 production-transfer claim. |
| Posterior uncertainty | None for P8: its outputs are deterministic point estimates. | Calibrated nuisance-marginalised per-galaxy uncertainty and class probabilities. |

This is stronger than the old random-split evidence: U-PATCH has learned a
mapping that transfers labels to disjoint geography and beats the registered
classical bar where that bar is well supported. It is still deliberately
narrow. Both rotations use the same `ph000` cosmic realization, and fold 4 is
common training geography. Patches create valid optimisation and spatial
holdout units; they do not create independent long-wavelength modes.

The present interpretation is therefore:

> P8 provides promising same-phase spatial target generalisation. P10 remains
> the first test of fresh-graph, fresh-phase generalisation, and P12 remains the
> first test of posterior validity.

### How the current model families fit the book

| Project model | Book domain and prior | What it gets right | Current theoretical limitation |
| --- | --- | --- | --- |
| G-PATCH | Chapter 5 message-passing Graph Network on a Chapter 8 geometric graph | Node-order permutation equivariance; physically constructed Delaunay plus fixed-radius support; local attention; irregular catalogue represented without voxelising it first | Raw Cartesian edge directions enter unconstrained MLPs; no exact physical rotation invariance; no explicit node LOS; finite message support cannot by itself recover external tidal modes |
| U-PATCH | Chapter 6 grid CNN with Chapter 4 locality and scale separation | Multiscale convolution and pooling; canonical global field before patch extraction; patch-safe channel normalisation; current best spatial-transfer result | Current P8 input is only `counts`, `exposure_apodized`, and `log_count_ratio`; the available LOS/selection products are not all passed to this model; ordinary Cartesian convolutions are not rotation equivariant; finite field context misses nonlocal shear |
| Simplified F-tier / U-Physics | Hybrid Chapter 6 grid representation plus a fixed physical operator | Conditional on the learned scalar field, the FFT Poisson/tidal operator exactly enforces the chosen discrete trace relation and Hessian integrability; long-range physics is explicit | The learned graph/grid-to-field map, voxelisation, and U-Net are not thereby exactly SO(3)-equivariant; the frozen P8 F-tier configuration was resource infeasible and no matched full-range simplified model has yet been trained; FFT overlap/context is expensive |
| CIC/DTFE plus FFT | Non-learned geometric estimator and known operator | Strong, interpretable reference; global context; exact field-to-tidal transform once a density field is supplied | Noise amplification in sparse regions; galaxy bias, RSD, selection, and discreteness are only approximately corrected |
| FMPE/NPE on a frozen encoder | Probabilistic head outside the core GDL symmetry taxonomy | Can approximate conditional eigenvalue uncertainty and class probabilities after the representation is fixed | A flow does not repair a biased or non-transferable encoder; nuisance coverage is determined by the simulator prior; calibration must be tested conditionally and on fresh phases |

Two nuances matter when applying the book:

1. A per-node scalar prediction is **permutation equivariant**, not globally
   invariant: permuting catalogue rows must permute the corresponding galaxy
   outputs. Spatially, eigenvalues are rotation-invariant scalars, whereas a
   tidal tensor or eigenvectors are rotation-equivariant outputs.
2. U-PATCH currently performs better than G-PATCH. This is direct evidence
   against assuming that a more elaborate geometric GNN is automatically the
   route to better transfer. The book supplies constraints and representations,
   not a guarantee that one architecture family wins.

### The book does not eliminate experiments; it makes them diagnostic

Performance is not usefully divided into “theory” versus “trial and error.”
Four different limitations need different interventions:

| Limitation | Diagnostic | Correct intervention |
| --- | --- | --- |
| Optimisation/exposure | Loss and complete-fold validation still improve at the cap | More complete-exposure epochs, exact resume, and multiple seeds |
| Missing training support | Performance changes on an independent phase, HOD, observer, selection, or epoch | Add the missing independent or nuisance variation to the training distribution |
| Wrong inductive bias | A matched-capacity symmetry/scale/operator ablation improves held-out domains | Change the representation or operator |
| Information loss | Real-space-oracle inputs strongly outperform deployable redshift-space inputs and all deployable models saturate with density | Return honest posterior width/OOD flags; add external observables if available; do not promise deterministic recovery |

The book helps specify admissible transformations, locality, scale, and output
type. It cannot determine the DESI conditional distribution or restore
information destroyed by sparse sampling and fingers of God. Empirical
comparison is still necessary, but it should be a small frozen factorial
experiment rather than an open-ended model zoo.

## Which training variations buy which kind of generalisation

The axes mentioned in the VAC plan are complementary, not interchangeable.

### More patches or cosmic locations within one phase

These improve optimisation coverage and test transfer to unseen geography.
They expose different local environments and survey boundaries. They do not
provide new long-wavelength modes or an independent realization of cosmic
variance. This is the axis P8 tests.

### More independent phases

Independent phases change the density field, rare structures, long modes, and
the relation between local patch context and external tidal shear. This is the
highest-priority data change for the production claim. The registered P10
roles are already appropriate:

- `ph000`: protocol development;
- `ph002`--`ph005`: multi-phase training;
- `ph006`: phase-level validation/calibration;
- `ph001`: sealed one-shot blind phase.

Training longer on `ph000` answers an optimisation question. It cannot replace
this phase split.

### More HOD and velocity-bias variations

HODs change which galaxies sample a fixed matter field, their satellite
fraction, luminosities, assembly bias, and velocities. Multiple HODs on one
phase therefore probe galaxy-bias and observation nuisance, not cosmic
variance. They become especially efficient after per-phase T-Web targets
exist, because many catalogue views can reuse one matter truth.

The right design is broad-to-narrow: train over a flexible HOD/velocity-bias
prior and hold out both new random seeds and at least one structurally
different HOD family. A sensitivity study of galaxy-clustering SBI found an
important asymmetric failure: training on a more complex HOD transferred to a
simpler HOD, while training on the simpler model failed on the more complex
one. This is a methodological precedent rather than direct proof for tidal
eigenvalues:

- Modi et al., *Sensitivity Analysis of Simulation-Based Inference for Galaxy
  Clustering*, arXiv:2309.15071.
  https://arxiv.org/abs/2309.15071

The broader SimBIG programme supplies two further protocol precedents:

- Hahn et al., *SimBIG: Mock Challenge for a Forward Modeling Approach to
  Galaxy Clustering*, arXiv:2211.00660, challenge the inference pipeline with
  test simulations that change the N-body code, halo finder, and HOD
  prescription rather than validating only on more draws from the training
  generator. https://arxiv.org/abs/2211.00660
- Hahn et al., *SimBIG: A Forward Modeling Approach To Analyzing Galaxy
  Clustering*, arXiv:2211.00723, include survey realism and observational
  systematics in an SBI forward model.
  https://arxiv.org/abs/2211.00723
- Lemos et al., *SimBIG: Field-level Simulation-Based Inference of Galaxy
  Clustering*, arXiv:2310.15256, demonstrate the field-level compression plus
  normalizing-flow pattern. https://arxiv.org/abs/2310.15256

These studies infer catalogue-level cosmological parameters, not per-galaxy
tidal eigenvalues. Their transferable lesson is the validation design:
represent the observation process in the simulator and test the posterior
against deliberately different forward models.

### More observer positions and survey realizations

Observers change wide-angle LOS geometry, cap boundaries, remapping, and which
long modes enter the survey volume. They are valuable window-function tests
but remain correlated when drawn from one periodic phase. Vary fibre
assignment, completeness, magnitude selection, redshift errors, and matched
random catalogues as labelled nuisance views rather than silently pooling
them.

### More redshifts

The current four shells test changing number density, selection, and
lightcone location while the registered physical target remains the T-Web
field at simulation epoch `z=0.2`. They do **not** demonstrate evolution across
target epochs.

Before adding snapshots, freeze the scientific estimand:

1. a coeval `z=0.2` field inferred from a lightcone; or
2. a lightcone field `T(x,z)` evaluated at each galaxy's epoch.

The second product requires targets from multiple snapshots, an explicit time
or growth-factor input, and a held-out-redshift test. Pooling snapshots without
conditioning asks one model to learn incompatible mappings and can create
negative transfer.

## What to hard-code, condition on, marginalise, or reserve as oracle truth

Providing “exact information” is useful only when the information also exists
for DESI. The following separation avoids train/deployment leakage.

| Treatment | Quantities | Reason |
| --- | --- | --- |
| Hard-code as geometry/physics | Coordinate and unit conventions; observer-centred rotation rule; eigenvalue ordering; density-to-potential-to-tidal FFT operator; tensor symmetry/trace/integrability where applicable | These are known structural constraints, not catalogue-specific labels |
| Condition on because DESI supplies them | RA/Dec, observed redshift, LOS, cap, random-catalogue intensity, completeness/exposure, mask distance, redshift uncertainty, photometry when mock parity is established | The prediction should adapt to known observing conditions |
| Draw and marginalise as nuisance | HOD and assembly bias, satellite/velocity bias, unobserved galaxies, small-scale velocities, stochastic selection, shot noise, plausible cosmology dependence | Their true values are not known per DESI galaxy; conditioning on a simulation ID would not deploy |
| Use only for diagnostics or privileged training | `Z_COSMO`, true peculiar velocities, particle density, true HOD label, true real-space positions | They quantify headroom or supervise an auxiliary teacher, but must not be required at inference |

The current P3a fields do not yet contain the production-grade random-catalogue
exposure, per-object completeness, or luminosity channels listed above. The
plan correctly treats those as P3b observation-model upgrades. They should be
introduced as a named, matched challenger rather than retroactively attributed
to the completed P3a baseline.

An HOD identifier can be used to stratify evaluation or to train an adversarial
diagnostic. It should not be a required production input unless an independently
measured DESI analogue exists. Similarly, true velocities can supervise a
velocity/displacement auxiliary task, but they cannot increase the information
available to the deployed model unless that task measurably improves a
deployable representation. Keep such an auxiliary head only if it improves
fresh-phase eigenvalue transfer or uncertainty calibration after the privileged
input is removed.

## RSD: a practical modelling ladder

RSD is not an omitted detail in the current task. The inputs use observed
redshift-space positions while labels are sampled at the real-space host-halo
position. The mapping is schematically

```text
s = x + [(v dot n) / (a H)] n,
```

with wide-angle lightcone and catalogue-construction details handled by the
mock. Small-scale velocity dispersion makes this mapping stochastic and
many-to-one; exact deterministic inversion is not possible from galaxy
positions alone.

The older project diagnostic found a large oracle gap: simple real-space
features reached lambda1 R2 about `0.86`, versus `0.26` in redshift space,
while a transductive redshift-space GNN reached about `0.77`. That result shows
both that RSD is a major effect and that message passing can learn part of its
statistical inverse. It was not a P8/P10 transfer test and must be repeated
under the current protocol before being used as a performance forecast.

### RSD level 0: repeat the oracle decomposition under the current protocol

For the same folds, phases, architecture, and budget, compare:

1. observed `Z` positions: deployable baseline;
2. `Z_COSMO` positions: non-deployable information upper bound;
3. observed positions plus observer-aware features;
4. a physics-corrected field plus a learned residual.

Report lambda errors against edge/LOS angle `mu`, radial and transverse tensor
components, number density, and satellite/FoG proxies. This distinguishes an
RSD limitation from a generic sparse-sampling or context limitation.

### RSD level 1: add exact observer-aware invariants

The current P8 models are not explicitly given all the information needed to
represent wide-angle RSD cleanly.

For G-PATCH, add edge-centred quantities such as

```text
mu_ij = rhat_ij dot nhat_edge
delta_r_parallel = delta_x_ij dot nhat_edge
r_perp = sqrt(r_ij^2 - delta_r_parallel^2)
nhat_i dot nhat_j
```

where `nhat_edge` is the normalized mean of the endpoint LOS vectors. Distances,
`mu`, `mu^2`, radial/tangential separation, and selection intensities are
global-rotation invariant while still exposing the physically preferred
radial direction. They are cleaner than asking an unconstrained MLP to infer
RSD from global `x/y/z` edge components. The historical F-tier already
computes closely related LOS-relative quantities, so port and test those
definitions before inventing a second convention. Also expose the stored
Delaunay-versus-radius edge provenance as an edge type: the two supports have
different geometric meanings, but the current five-feature P8 edge vector does
not tell the attention block which generated an edge.

For U-PATCH, the canonical fields already support LOS and radial/selection
products, but the current P8 model consumes only three channels. A controlled
challenger should add LOS-vector fields, radial distance or redshift, expected
counts/random intensity, and completeness while retaining the same U-Net and
training budget. Physical catalogue/mask rotations can be used as an
augmentation audit; a Cartesian CNN does not become exactly equivariant merely
because LOS channels are present.

### RSD level 2: physics first, learned residual second

On sufficiently large scales, use a linear or iterative RSD/reconstruction
operator and train the network only on the remaining quasi-linear and
small-scale residual. Supply both the raw observed field and the
physics-corrected field so the model can reject a poor correction in sparse
regions. Bias, growth rate, and velocity-dispersion parameters must be varied
or inferred rather than fixed to their true simulation values.

Relevant precedents:

- Maragliano et al., *From Redshift to Real Space: Combining Linear Theory
  With Neural Networks*, arXiv:2507.11462.
  https://arxiv.org/abs/2507.11462
- Chen et al., *Effective cosmic density field reconstruction with
  convolutional neural network*, arXiv:2306.10538.
  https://arxiv.org/abs/2306.10538

These works support a hybrid pattern, not direct transfer to the DESI
per-galaxy tidal-eigenvalue problem.

### RSD level 3: paired nuisance views

Render several observed catalogues from the same matter truth while varying
HOD, satellite/velocity bias, redshift error, fibre assignment, completeness,
and thinning. Train every view against the same latent field and add a
consistency term at fixed spatial query locations.

Per-galaxy node sets differ between HODs, so naïvely matching row predictions
is invalid. Consistency is easiest on a shared latent grid, fixed spatial query
points, or carefully matched host halos. This is one reason a graph-to-field
representation is attractive.

### RSD level 4: an explicit latent forward model

The physically strongest formulation infers a latent real-space density and
velocity field, populates it with a flexible galaxy-bias/HOD model, maps it to
redshift space, applies selection and noise, and compares the rendered
catalogue with the observation. Tidal tensors are then derived from latent
density samples with the fixed Poisson operator.

BORG demonstrates this hierarchical pattern with gravitational evolution,
galaxy bias, RSD, selection, and noise:

- Jasche & Wandelt, *Bayesian physical reconstruction of initial conditions
  from large scale structure surveys*, arXiv:1203.3639.
  https://arxiv.org/abs/1203.3639
- Jasche & Lavaux, *Physical Bayesian modelling of the non-linear matter
  distribution*, arXiv:1806.11117.
  https://arxiv.org/abs/1806.11117

This level gives the cleanest physical posterior but is substantially more
expensive than the present amortised pipeline. A realistic research direction
is to amortise or emulate parts of that inverse problem, not to reproduce BORG
inside the immediate P8/P10 path.

Zang et al. also show that tidal reconstruction in redshift space has strongly
angle-dependent radial noise while transverse shear is more robust. This
motivates LOS-stratified validation and possibly a transverse-shear auxiliary:

- Zang et al., *Cosmic tidal reconstruction in redshift space*,
  arXiv:2212.04294. https://arxiv.org/abs/2212.04294

## Practical multi-domain training protocol

The following extends the existing VAC plan without weakening its gates.

### Stage 0: close the current P8 question unchanged

1. Finish and audit `convergence_extension_v1`.
2. Keep the P8 linear-increment targets, folds, features, architecture, and
   objective unchanged during that test.
3. Promote only candidates that pass the registered all-fold and seed rules.
4. Run the trace-versus-traceless context-growth diagnostic before concluding
   that a finite encoder lacks capacity.

This stage measures optimisation and physical support. Adding phases or RSD
features before it closes would confound the diagnosis.

### Stage 1: establish the P10 phase-only baseline

1. Build independently hashed P1--P4 products for every phase.
2. Train on `ph002`--`ph005`, validate on `ph006`, and keep `ph001` sealed.
3. Sample phase first, then cap/shell/core, so the largest or densest phase does
   not dominate. Retain the registered square-root shell weighting within each
   phase.
4. Define a multi-phase exposure epoch explicitly: either visit every eligible
   core once in every phase, or use an equal fixed core quota per phase. Record
   the choice and effective exposure.
5. Fit every scaler/normalisation on training phases only.
6. Plot performance versus the number of training phases. This learning curve
   tells us whether the limiting resource is independent realizations.

For the first phase-count learning curve, match optimiser updates and sampled
cores across the one-, two-, and four-phase conditions. Otherwise “more
phases” is confounded with “more compute.” After that controlled comparison,
run the production candidate with complete exposure to all training phases and
report the extra cost and gain separately.

Model selection should report the mean and worst phase, mean and worst shell,
phase-to-phase spread, and spatial blocks inside each phase. Millions of
galaxies are not millions of independent validation examples.

### Stage 2: add a crossed nuisance design, not an indiscriminate pool

For each training phase, render a small balanced design such as:

```text
phase
  x HOD/assembly-bias family and seed
  x satellite/velocity-bias setting
  x observer or sky remap
  x selection/fibre/redshift-error realization
```

A full Cartesian product is unnecessary initially. Use a balanced fractional
factorial that lets each main effect be identified. Reuse the phase truth.
All HOD, observer, and selection views derived from one matter phase must stay
inside that phase's outer split; a view of `ph001` or `ph006` is not admissible
training data. Sample the latent phase uniformly before sampling its nuisance
view, so ten HOD catalogues from one phase do not count statistically as ten
independent universes.

Reserve:

- a new seed inside the training HOD family for interpolation;
- a different HOD/assembly-bias family for structural OOD;
- unseen selection and redshift-error realizations;
- at least one observer/window configuration.

Train a labelled baseline with balanced domain sampling before trying
domain-adversarial losses. If the mean improves but one domain collapses,
compare a mean-plus-worst-group or CVaR objective as a named ablation. Do not
use HOD identity as a required DESI input.

### Stage 3: run one-factor geometric and physical challengers

Against the frozen multi-phase baseline, change only one item at a time:

1. U-PATCH plus the available LOS/selection channels;
2. G-PATCH plus observer-invariant RSD features and explicit
   Delaunay/radius edge type;
3. `ell <= 2` multipole messages;
4. a simplified learned-density plus fixed-FFT tidal decoder;
5. a large-scale RSD/classical field plus learned residual.

Use matched parameters where meaningful, identical phase/nuisance batches,
identical target transform, complete exposure, and the same blind gates. The
promotion question is not “does training loss fall?” but “does the change
improve fresh-phase and held-out-nuisance performance without harming the
supported shells?”

### Stage 4: reconcile the physical target head

Once a deterministic representation wins, compare the frozen P8 linear
increments with one ordering-enforcing alternative under the same protocol.
This can be log gaps or the repository's ordered-softplus increment policy.
Choose before P12 and P10 final retraining; do not sort predictions after the
fact.

For a tensor variant, prefer one of:

1. predict a latent density field and obtain the tensor with the fixed FFT
   operator; or
2. predict trace plus an `ell=2` symmetric-traceless component and explicitly
   test trace, symmetry, integrability, and rotation equivariance.

A free six-component tensor head is not physically exact merely because it is
equivariant.

### Stage 5: fit uncertainty only after deterministic transfer

Fit FMPE/NPE on leakage-safe out-of-fold embeddings or field summaries from
the frozen winner. The simulator draws must span the nuisance prior intended
for the DESI claim. Tune/calibrate on `ph006` and evaluate `ph001` once.
At least one posterior challenge must come from an alternate forward model or
HOD family outside the NPE training generator. Coverage only on held-out draws
from the same simulator tests interpolation, not simulator-mismatch
robustness.

Separate:

- **aleatoric/nuisance uncertainty:** shot noise, HOD ambiguity, velocities,
  missing galaxies, and measurement noise represented in the simulator;
- **epistemic uncertainty:** finite phases, seeds, and model choice, assessed
  with phase-level replication or an ensemble.

A seed ensemble is not itself a posterior. If used, form and validate an
explicit mixture posterior rather than adding unrelated variances.

Required validation includes SBC/TARP, marginal and conditional coverage,
class-probability reliability, width versus realized error, posterior
contraction versus tracer density, and coverage conditioned on phase, shell,
HOD family, velocity bias, completeness, boundary, and web class. Perform
posterior predictive checks by rendering catalogue or field summaries back
through the observation model.

## A potentially distinctive GraphWeb method

Novelty should not be claimed from substituting one GNN layer for another. A
scientifically distinctive and defensible combination would be:

> an observer-aware, selection-aware geometric graph encoder that infers a
> latent real-space matter field; a fixed Poisson/FFT layer that converts field
> samples to tensors physically consistent with that learned field; and a
> calibrated posterior
> trained across independent phases and marginalised over HOD, velocity/RSD,
> and survey-selection nuisance.

Its components would be:

1. Delaunay plus fixed-scale support for sparse and dense environments;
2. selection-aware Monte Carlo message aggregation;
3. LOS-invariant edge geometry and optional `ell=0/2` internal channels;
4. a graph-to-field multiscale decoder for long-range support;
5. the exact density-to-tidal operator;
6. paired nuisance views and independent-phase training;
7. ordered eigenvalue or trace-plus-shear posterior outputs;
8. DESI support, OOD, information, boundary, and eigengap flags.

The conservative production control remains multi-phase U-PATCH plus a
calibrated posterior because it currently leads. The distinctive
graph-to-physics model must beat that control and the classical estimator on
fresh phases; its conceptual appeal cannot substitute for the result.

If validated, the responsible methodological claim would be narrower and
stronger than “a novel GNN”:

> amortised, nuisance-marginalised reconstruction of real-space tidal
> eigenvalues from DESI-like redshift-space galaxy geometry, with exact
> field-to-tidal physics and blind fresh-phase validation.

The exact novelty of that combination still requires a dedicated literature
comparison before publication.

## DESI VAC output and uncertainty contract

For every supported galaxy, the useful product should eventually include:

- posterior median/mean and intervals for ordered
  `(lambda1, lambda2, lambda3)`;
- probabilities for the four T-Web classes derived from posterior samples,
  not a separately inconsistent classifier;
- posterior information or contraction relative to the local prior;
- observed-redshift shell, `ntilde`/random intensity, completeness, boundary,
  graph/field support, and OOD flags;
- model/nuisance version and provenance.

For a tidal-tensor/orientation variant, additionally include:

- a clearly defined tensor convention and frame;
- trace and symmetric-traceless components or tensor posterior samples;
- eigenvectors only with sign/axis conventions;
- an eigengap flag and orientation uncertainty, since eigenvectors become
  physically unstable near degeneracy.

Marginal per-galaxy intervals do not describe spatial covariance between
galaxies. If downstream dynamics needs coherent fields, retain or release
joint field samples or a documented covariance approximation.

## Recommended geometric-method experiments

The priorities below are **within the geometric architecture branch**. They do
not supersede the programme order of closing P8, running P10 phases, and
calibrating P12 only after a deterministic winner.

### Priority 1: audit physical rotation sensitivity of the current model

Before changing the architecture, measure how much the current G-PATCH output
changes when all covariant inputs are rigidly rotated together:

- rotate positions and edge direction vectors;
- rotate LOS vectors wherever present;
- preserve scalar node/edge quantities and graph incidence;
- compare scalar eigenvalue predictions before and after rotation.

For a pure physical test, the observer and footprint/mask must be transformed
consistently. Report per-component prediction differences and changes in the
registered spatial/fresh-phase metrics.

This is a cheap diagnostic. A failure would not prove that equivariance will
improve accuracy, but it would show that the current model spends capacity on
global-axis dependence. Keep this separate from P8 fold-role rotations.

### Priority 2: an `ell <= 2` multipole/tensor message ablation

For each node `i`, radial basis/shell `k`, neighbour direction
`rhat_ij`, and estimated sampling intensity `q_j`, form:

```text
s_ik = sum_j b_k(r_ij) h_j / q_j
v_ik = sum_j b_k(r_ij) h_j rhat_ij / q_j
Q_ik = sum_j b_k(r_ij) h_j
       (rhat_ij rhat_ij^T - I/3) / q_j
```

Here `s`, `v`, and `Q` are monopole (`ell=0`), dipole (`ell=1`), and
quadrupole (`ell=2`) channels. Use a small number of physically chosen radial
bases around the 7 Mpc/h smoothing scale and the longer Delaunay-bridge regime.

For the existing scalar eigenvalue head, feed invariant contractions such as:

```text
||v||^2, tr(Q^2), tr(Q^3), rhat_LOS^T Q rhat_LOS, v dot rhat_LOS
```

Alternatively, keep `Q` as an equivariant internal feature and couple it with
invariant attention logits. These are internal geometric features, not the
deprecated `(I1, I2, I3)` target parameterisation. Under P8 they must feed the
frozen linear-increment head so that the ablation changes only geometry. For a
later production or posterior model, choose and freeze the ordering-enforcing
target policy before training; do not mix target and architecture changes in
one comparison.

Why this experiment is attractive:

- it directly matches the trace plus shear decomposition of a tidal tensor;
- it distinguishes isotropic density from anisotropic web geometry;
- it is much cheaper and easier to diagnose than a deep general-purpose
  steerable network;
- it can be added as a controlled feature/message ablation to the established
  pipeline.

### Priority 3: make the message sum a selection-aware quadrature

The book largely assumes that the geometric points are the domain. A survey
catalogue is instead a stochastic, biased sampling of an underlying field.
This is an orthogonal issue.

Interpret the neighbourhood aggregation as a Monte Carlo approximation to a
continuous convolution/integral and divide contributions by an estimated
sampling intensity. Candidate intensity estimates include:

- the existing frozen `ntilde(z, cap)` channel;
- local density inferred from matched DESI randoms;
- explicit data/random message channels;
- a hybrid random-derived angular/radial selection estimate.

Normalise carefully to control variance in sparse regions, and retain the
important limitation: inverse-density weighting removes a selection baseline
but cannot restore information lost to shot noise.

This should be validated under log-uniform Poisson thinning, held-out radial
density shells, and spatial/fresh-phase blocks. It should not be judged only
on the training density.

Primary references:

- Hermosilla et al., *Monte Carlo Convolution for Learning on Non-Uniformly
  Sampled Point Clouds*, arXiv:1806.01759.
  https://arxiv.org/abs/1806.01759
- Wu, Qi & Fuxin, *PointConv*, arXiv:1811.07246.
  https://arxiv.org/abs/1811.07246
- the existing project note:
  `sources/research_random_catalog_density_graph_ml.md`.

### Priority 4: retain geometry-anchored multiscale support

Do not revive unrestricted dynamic feature-space kNN. Instead compare:

- fixed physical-radius radial shells;
- Delaunay bridge edge type;
- a coarsened/field branch for long-range context;
- separate edge-type attention or gating.

This is the graph analogue of Chapter 4 scale separation and is consistent
with the non-local inverse-Laplacian physics. The active graph-to-field plus
fixed FFT operator is already the strongest version of this idea.

### Defer: a large SE(3)-Transformer/Equiformer replacement

Revisit only if one of these conditions holds:

- the physical-rotation audit exposes a large and scientifically harmful
  dependence on arbitrary axes;
- the cheap `ell <= 2` ablation yields a reproducible fresh-phase gain;
- eigenvectors/intrinsic alignments become a primary deliverable;
- a matched-capacity and matched-compute design removes the confounds of the
  existing SEGNN test.

If revisited, use invariant attention logits, retain LOS as a covariant input,
match parameters/optimisation/exposure, and compare against the field-level
physics model rather than only an older graph baseline.

## Validation gates

Any geometric upgrade should report:

1. exact numerical permutation and physical-rotation tests;
2. spatial-block and independent-phase accuracy, not random-split scores;
3. shell-wise performance across number density;
4. thinning/selection robustness;
5. calibration for the NPE product;
6. matched parameter count, exposure, optimisation status, and wall-clock
   cost;
7. tensor-physics checks if a tensor is emitted.

For a predicted tidal tensor, equivariance is not sufficient. A free symmetric
tensor may violate the Poisson trace relation or Hessian integrability. The
current fixed field-to-tidal FFT operator enforces more physics than a generic
equivariant tensor head and should remain the reference.

## Direct astronomical evidence and its limits

There is already primary evidence that symmetry-aware graph summaries can be
useful for cosmological point catalogues:

- Wu, Jespersen & Wechsler use an E(3)-invariant GNN with fixed linking-length
  graphs for the TNG galaxy--halo connection and find useful environmental
  information out to their largest tested scale, 10 Mpc.
  https://arxiv.org/abs/2402.07995
- Villanueva-Domingo & Villaescusa-Navarro use translation- and
  rotation-invariant GNNs for galaxy-catalogue cosmological inference.
  https://arxiv.org/abs/2204.13713
- Shao et al. use permutation-, translation-, and rotation-invariant GNNs on
  halo catalogues and test across several simulation codes.
  https://arxiv.org/abs/2209.06843
- Makinen et al. use graph summaries for catalogue-level information
  maximisation and noisy survey cuts.
  https://arxiv.org/abs/2207.05202

These are strong precedents for geometric graph inductive biases, not direct
validation for this problem. Most use periodic or small simulation volumes,
global catalogue targets, or different observables. They do not establish
per-galaxy recovery of 7 Mpc/h-smoothed tidal eigenvalues under DESI masks,
RSD, spatial holdout, density variation, and fresh-phase transfer.

## Bottom line

The book supports the present project direction:

- keep the union graph and physics-grounded graph-to-field decoder central;
- treat exact symmetry as one design axis, not the definition of geometry;
- represent the tidal problem with the physically relevant `ell=0` and
  `ell=2` content;
- handle the observer/LOS and selection measure explicitly;
- test physical rotations and density transfer separately;
- demand matched, fresh-phase evidence before promoting a heavier
  equivariant architecture.
