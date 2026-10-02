# Observed-footprint tidal control

## Registered scope

User requests implementation of the attached observed-footprint control first,
with exterior modelling deferred. Preserve the DESI mask. Use the four saved
coherent NGC and SGC density draws; no new density sampling, mock truth access,
training, change to P12-A/VAC, or claim of total-field posterior calibration.
Threshold is lambda=0.2, inherited from the preceding discussion.

## Estimand and boundary condition

For the already-smoothed saved density, form `delta_source = support*(rho-1)`.
Support is the unchanged fraction of raw binary DESI-mask cells in each native
6.766 Mpc/h cell, not angular completeness, number density, or apodisation.
It is the existing voxel approximation to the mask, not a newly enlarged mask.
Cells with no support contribute zero density contrast. This is mean density,
not zero matter density. No further Gaussian smoothing or cap-mean subtraction
is introduced. Masking and smoothing do not commute: this control masks the
already-smoothed density, and must be compared to the same operation on truth.

Compute Hessian(Phi) with Laplacian(Phi)=delta_source in free space, summing
the NGC and SGC source contributions at both caps' physical cell centres.
Different grid origins are handled by shifted source-to-target kernels, not by
resampling either cap or joining the sky between them. Pair draw index s across
caps under their existing independent-cap approximation; this does not establish
physical cross-cap posterior covariance. The control includes gravitational
contributions from the OTHER observed cap, not just the target cap.

Use the analytic integral of Hessian(-1/(4*pi*r)) over each piecewise-constant
cubic source voxel. The self term is I/3; off-source trace is zero. FFT padding
contains every required source-to-target displacement and implements linear
convolution, not periodically repeated gravity. No periodic k=0 replacement,
unobserved coarse-model values, learned exterior, or unknown exterior shear is
silently inserted. Cuboid antiderivatives follow the standard atan/log formulas
documented in [Choclo](https://www.fatiando.org/choclo/latest/_modules/choclo/prism/_gravity.html),
with opposite sign to its gravitational-acceleration gradient and normalization
1/(4*pi). The control is exact for the stated piecewise-constant discretization,
not an assertion that the unresolved subcell density is exactly constant.

Output native tensors, ascending eigenvalues, per-draw classes, and four-draw
class frequencies at support>=0.5; outside this report mask use NaN/255.
Display tensors are 2x mean-pooled before eigendecomposition and reported only
at the ORIGINAL display mask pool2(support)>=0.5 with complete density coverage.
Save distance to non-full support as an explicit boundary diagnostic. Four draws
give class frequencies only in increments of 0.25; these are exploratory,
boundary-conditioned samples, not calibrated environment probabilities.

## Verification and compute

CPU interactive allocation59199774, one node, one-hour bound, using cosmic_env.
Eight tests pass: analytic isotropic self term; exact trace; independent 18-point
Gauss-Legendre volume integration of six kernel components; FFT/direct-sum parity
on displaced unequal grids; padding invariance; absence of periodic images;
mask irrelevance outside support; drawwise eigenvalue classification; uniform
cube symmetry; invalid-mask and singular-face guards; interior/exterior
linearity and trace-free exterior response inside a source-free region.

Actual full-NGC benchmark uses native supported bounding shape272x416x266 and
padded shape550x840x539. Kernel construction8.00s; one component/four draws plus
source FFTs12.82s. This is a numerical benchmark, not a published control field.
It supports completing all four source-target combinations within this bounded
CPU allocation. Production keeps eigenvalues in float64 so threshold labels can
be exactly reproduced from saved eigenvalues.

Inputs: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/loa_global_canvas_20261002_v1`.
Output: `/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/loa_footprint_tidal_20261002_v1`.
`COMPLETE.json` and `VERIFIED.json` both pass. Full generation took334.09s
after input preparation. Both caps have four draws, including both observed
caps' source contributions. No source-cap overlap occurs at reported native
target centres. The input density and masks were not modified.

| Cap | Reported native cells | Display cells | Double-precision Poisson trace max error | Stored float32 tensor trace max error |
| --- | ---: | ---: | ---: | ---: |
| NGC | 11,062,310 | 1,382,456 | 2.825e-14 | 2.385e-7 |
| SGC | 4,905,122 | 611,897 | 2.659e-14 | 2.385e-7 |

Saved tensor eigenvalues agree with the float64 eigenvalue datasets within
1.11e-7 after tensor float32 storage. Class labels and frequencies reproduce
the saved eigenvalues exactly; probabilities sum to one on valid cells and
are NaN outside the report mask. Display masks exactly match the preceding
coherent-density mask definition and supported-cell counts. Eight tests and
independent post-write validation pass. This is numerical validation, not
coverage/calibration or a measurement of the physical exterior error.

Outputs `NGC_TIDAL.h5` and `SGC_TIDAL.h5` contain `tensor`, `eigenvalues`,
`class` (0/1/2/3;255 outside mask), `class_probability`, `support`, `mask`,
`distance_to_nonfull_support_mpc_h`, `display_eigenvalues`,
`display_class_probability` and `display_mask`. Native tensor components are
xx,xy,xz,yy,yz,zz; eigenvalues are ascending. Thresholds other than0.2 can be
applied to the saved eigenvalues without another density or Poisson solve.
The website has not been changed in this scientific-control implementation.
Allocation59199774 was released after verification. Graphify queries worked in
cosmic_env; the required refresh again failed at filesystem `flock` error524.
The source index is stale; this does not affect the numerical receipts.

Reproduction (use a fresh output directory on a compute node):

```sh
python -m workflows.sbi.e2e_footprint_tidal_control --root <coherent-density-root> --out <new-output> --threshold .2
python -m workflows.sbi.e2e_footprint_tidal_verify --root <coherent-density-root> --out <new-output>
```

## Deferred science

On exposed mocks, compare the SAME isolated masked operator on true density
with the correctly defined full-field truth, versus boundary distance, shell,
eigenvalue and class at lambda0.2. This separates missing-exterior effects from
density-inference and discretization errors. It has NOT been run here.
Do not claim that this control alone measures the true exterior error or that
all missing modes are low-dimensional. Distant sources admit a harmonic
expansion, whose leading tidal term is symmetric and trace-free in a source-free
interior; nearby mask holes can require more spatial detail.
