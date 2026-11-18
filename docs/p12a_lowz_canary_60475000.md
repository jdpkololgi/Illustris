# P12-A low-redshift canary — 2026-10-29

The approved ph001/ph006 canary completed successfully. Corrected catalogue
payload/annotation provenance, observed geometry and angular random-map hashes
passed, with finite ordered label-status checks across the two catalogues.
This is not an independent resampling of the native tidal field.

| Phase | Rows at 0.01 <= z < 0.10 | Valid labels | Angular supported | Old 3D response supported |
| --- | ---: | ---: | ---: | ---: |
| ph001 (training) | 1,075,312 | 1,075,312 | 1,075,312 | 3,213 |
| ph006 (development) | 1,074,166 | 1,074,166 | 1,074,166 | 3,035 |

All these low-z galaxies have negative legacy BOX_INDEX despite valid corrected
labels. BOX_INDEX must not become a target-validity mask again. The few nonzero
old response flags just below0.10 are compatible with sampling at voxel centres;
they do not establish low-z response qualification. Exact old support was replayed
on the small sample, not newly fitted.

At0.01–0.15, the catalogues contain2,334,047/2,352,896galaxies respectively.
Observed-only sampling selected160points per phase, balanced over five low-z
bins and both caps, with no truth-based selection. All320points have angular
support, but8/9points respectively lie outside the old Cartesian grids, and
only32/33points have old voxel response support. New grid/response geometry is
required; changing the posterior sampling cut alone cannot work.

Observed-only point/support prototypes and separately stored evaluation labels
are under `/pscratch/sd/d/dkololgi/abacus/p12a_lowz_canary_20261029_60475000/`.
The JSON companion binds sources, payloads, phase roles, counts, magnitudes,
timings and outputs. Angular-only distances are not a new deployable 3D response.
No selection curves, normalizers, full-volume targets or model were fitted.

The source tables provide R_MAG_APP and R_MAG_ABS, but no audited stellar-mass
column. The run therefore demonstrates low-z labelled-galaxy availability, not
stellar-mass completeness, mock-to-DESI population agreement or calibration.
The underlying snapshot is still z=0.2; observed z is not snapshot evolution.

## Next preparation

Reuse verified all-population labels and compatible angular maps. Build a new
version of observed/context indices and Cartesian grids reaching z=0.01, retain
available positive-z context below the supervision cut, and represent missing
near-observer/footprint support explicitly. Refit selection/effective volume
using training phases only; rebuild field/count/response channels, supervision
and spatial folds, full transforms and20OOF contracts. Do not silently extend
old selection curves. Start with the same two-phase geometry/selection prototype
to check near-observer and fold-group behavior, then budget the23-phase release.
The existing full preparation and its old-range receipts remain immutable.

## Execution and limits

Three focused synthetic scope/boundary/sampling tests passed before launch.
Allocation60475000: one CPU node,64CPU step,desi/interactive/Scratch,one-hour
maximum. Actual allocation duration1m49s; canary step28s, exit0. The exclusive
node allocation accounts256logicalCPUs; the canary step requested64. Both phase
audits took about11s each. These timings do not estimate a full field rebuild.
The allocation was released and sacct reports COMPLETED0:0.

Required graph refresh failed separately with existing flock OSError524 in2s;
no bypass or retry, and no global refresh. The data canary itself passed.
ph018/ph019 were not opened. No scientific training, GPU work, graph construction,
full-volume FFT, raw-random rescan or automatic retry occurred. Further compute
requires a separately approved preparation budget.
