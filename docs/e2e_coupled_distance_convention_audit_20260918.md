# Coupled-field coordinate audit: distance convention must be resolved

Read-only review finding, 2026-09-18. This note does not modify the frozen data
authority, running catalogue audit, transfer job, or another session's builder.

## Finding

The inherited Planck18 observer-distance conversion is not the conversion used
by the sampled Abacus c000 BGS CutSky. The axes and corner-observer translation
are supported, but the remaining displacement is systematic, radial and
redshift-dependent. A reference DESI/Abacus distance table removes it to almost
catalogue coordinate precision, without fitting a transformation to the data.

Input: the previously written 1,357 resolved central galaxies from ph007,
native slabs028,001,033, 0.15<=Z_COSMO<0.55. Compare real-space Z_COSMO positions
with their explicitly linked halo x_L2com, not RSD Z or the different x_com
definition. Saved input:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1/coordinate_audit/ph007/linked_centrals.npz`;
SHA256 `3f2dfd01bf8f1dbf04c7244b9192afbe63c73a4d52345589a6247be103da3c71`.

| Radial-distance convention | Median separation (Mpc/h) | P95 | Maximum |
| --- | ---: | ---: | ---: |
| Astropy Planck18 distance times0.6766, inherited | 0.510020 | 1.453183 | 2.169864 |
| Official tabulated DESI/Abacus, cubic interpolation | 0.00003617 | 0.00012373 | 0.00029855 |

For the inherited mapping, radial residual and Z_COSMO have correlation0.992512.
The transverse residual median is0.00003059Mpc/h and P95 is0.00012048Mpc/h.
With the tabulated distance but host x_com, the median/P95 separation is
0.0083743/0.0324824Mpc/h. This distinguishes the HOD's central x_L2com convention
from the general x_com field label convention.

The current [cosmoprimo fiducial source](https://github.com/cosmodesi/cosmoprimo/blob/main/cosmoprimo/fiducial.py)
defines DESI as AbacusSummitBase, not Astropy's Planck18 preset.
The [published table](https://raw.githubusercontent.com/cosmodesi/cosmoprimo/main/cosmoprimo/data/desi.dat)
contains redshift, E(z), and radial distance in Mpc/h. The reviewed bytes have
SHA256 `3cbbdae0d8e52350292f777e470ba6eb81889c560b9720146266aa03bc7dca35`,
3,000,237bytes and40,002numeric rows. Freeze a pinned source/version and hash
before using this as a build dependency; do not fetch mutable main at runtime.

The [BGS mock construction paper](https://academic.oup.com/mnras/article/532/1/903/7695304)
describes the corner observer and replicated snapshot construction. It also
describes unresolved galaxies seeded from field particles: these remain valid
observations, although a resolved host-halo audit cannot cover those rows.

## Reproduction without changing products

Read the existing linked_centrals NPZ and the upstream text table as numerical
data. Normalize `sky_planck18_mpc_h` to unit directions. Evaluate
`scipy.interpolate.CubicSpline(table[:,0],table[:,2])(Z_COSMO)` to obtain radial
distances. Form `sky = radius[:,None]*direction`. Compare to `host_x_L2com` with
periodic difference `(sky-host)%2000-1000`, which includes the registered
-1000Mpc/h corner translation. The table and NPZ hashes above bind this check.
No model fitting, table-parameter fitting, target rescaling, or held-out model
scoring was performed. This has been tested in one training phase, not all21.

## Required follow-through before canonical Cartesian release

1. Replicate this fixed-convention host test on additional phases. The table,
   axes and offset must be common, not fitted separately to every phase.
2. Resolve the coordinate contract explicitly. For a Cartesian physical-field
   experiment, use a common verified Mpc/h frame for the observations and matter.
   Keeping the old observer grid while merely warping matter coordinates is not
   equivalent: Fourier/tidal operators on that grid require an explicit
   coordinate-transformation treatment.
3. A coordinate correction is not a new selection/HOD model. However, the
   inherited ntilde(Mpc^-3) is a density per OLD coordinate volume. If changing
   radial coordinates, transform that density with the radial volume Jacobian
   (r_old/r_new)^2*(dr_old/dz)/(dr_new/dz), in consistent units, or equivalently
   preserve the original dN/dz/dOmega and recompute the new cell expectation.
   Do not reuse the old numeric ntilde unchanged or refit on confirmation data.
4. Bind the final convention as a new downstream product authority. The running
   parent/observed FITS builders write RA/DEC/Z, not Cartesian fields, so these
   valid joins and B transfers need not be discarded. Do not mutate the existing
   DATA_AUTHORITY.json/config hash beneath running jobs.
5. Retrain all four later arms on the same corrected products. An inherited
   architecture baseline is not a license to compare different physical frames.

This finding does NOT establish that the old distance mismatch explains the
large regional-coverage failure or all high-k power suppression. That causal
claim needs a matched test. It DOES establish that simply labeling the inherited
mapping as an approximation leaves a now-identified avoidable pairing error.

## Independent builder review notes

At review, the new observation builder uses Planck18 distances and the old
selection radius table, so it retains this mismatch. Preserve it as a comparison
implementation if useful; do not silently call its products coordinate-exact.
Its current restart policy reuses completed random files/caps but rejects
unreceipted outputs. Before long wall-time-limited production, test interruption
between NPZ/HDF5 publication and completion-receipt publication, and implement a
validated recovery or transactional generation policy. A successful arithmetic
parity test alone does not establish restart safety or all-phase data readiness.
