# Geometric Deep Learning book: implications for galaxy-catalogue graphs

Research lookup: 2026-07-23.

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
- an invariant three-scalar output in ordered-increment space.

Relevant files:

- `workflows/abacus_tweb/p8_prepare_graph_features.py`
- `workflows/abacus_tweb/p5_build_graph_patch_adapter.py`
- `workflows/abacus_tweb/p8_train_graph_patch.py`

This model is spatially informed but is not exactly E(3)/SO(3)-invariant. The
raw `x_dir, y_dir, z_dir` components are consumed by unconstrained MLPs.
They may encode useful observer/footprint structure, but they can also enable
dependence on arbitrary global axes.

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

## Recommended methodological experiments

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
deprecated `(I1, I2, I3)` target parameterisation; the canonical ordered
softplus-increment target should remain unchanged.

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
