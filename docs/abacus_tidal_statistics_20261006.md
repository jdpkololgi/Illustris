# Abacus tidal eigenvalue PDFs and spectra: first full-box measurement

Completed 2026-10-06. User authorized one CPU node, 64 CPUs, at most one hour.
Slurm 59431063 on nid200036 completed 0:0 in 4m37s and released its allocation.
This is an exploratory single-phase measurement, not a precision convergence
or cosmological-information result. No model training or sealed-phase access.

## Data and definitions

AbacusSummit base c000/ph000, z=0.2, periodic box 2000 Mpc/h; original
2048^3 10% particle count grid, documented 1024^3 block-mean cache, reduced by
exact block means to 256^3 and 512^3. Input cache provenance and recorded hashes
are preserved in figures/tidal_statistics_20261006/COMPLETE.json. Input hashes
were inherited from the cache manifest rather than recomputed in this run.

Normalize counts by the full-box mean and subtract one. Apply Gaussian
W(k)=exp(-k^2 R^2/2), R=7 Mpc/h, and T_ij(k)=k_i k_j delta_R(k)/k^2.
Set DC and Nyquist planes to zero, diagonalize in real space, sort ascending.
PDFs weight every mesh cell equally. Plot x=lambda/sigma_delta with the same
measured density sigma in all three marginal predictions; no marginal fits.

For spectra, subtract each eigenvalue mean. Use V/N^6 times the unnormalized
FFT products and correct real-FFT multiplicities. Signed cross spectra retained.
The random-phase control preserves each discrete density Fourier amplitude,
using phases from the FFT of independent real white noise (seed 61006).
It is a fixed-amplitude control, not an exact Gaussian ensemble.

## Findings

At 512^3, sigma_delta=0.46311854. The three means divided by sigma are
[-0.52027106, 0.01056260, 0.50970847]. The Gaussian prediction is
[-0.53523723, 0, 0.53523723]. PDFs show substantial shape differences:
sharper lambda1/lambda2 peaks and asymmetric positive tails, breaking the
Gaussian reflection relation between lambda1 and lambda3.

Histogram integrated absolute differences from the analytic density, using
bin-midpoint quadrature over -4 <= lambda/sigma <= 6, are
[0.42849, 0.34296, 0.35548] for Abacus versus
[0.00110, 0.00142, 0.00106] for the random-phase control.
These are descriptive binned distances, not significance levels or tail tests.
The largest probability fraction outside this interval is 7.37e-6.

At the 512^3 bin k=0.101024 h/Mpc:
P11=163.41, P22=345.42, P33=608.90 (Mpc/h)^3;
r12=0.67866, r13=0.60703, r23=0.89813.
The three autos therefore do not simply equal P_delta_R/9.
The measured P13 is negative in some large-scale bins; its significance is
not established from this one realization. No error bars from independent
modes/cells are claimed as a substitute for ensemble uncertainty.

## Validation and limitations

Analytic PDF integrals and means pass numerical checks. Small synthetic fields
pass trace, spectral trace closure, and correlation bounds. On the 512^3 Abacus
field, max absolute trace residual is 1.91e-6. The spectral trace sum agrees
with measured density power to 2.13e-7 relative error in the checked bins;
control and Abacus density power agree to 2.06e-7.

The normalized PDF shapes are similar at 256^3 and 512^3, but sigma changes
from 0.45041 to 0.46312. The auto-spectrum resolution diagnostic retains
several-percent differences and reaches about 11% below k=0.2 h/Mpc.
This diagnostic interpolates differently binned spectra, so some low-k
variation is binning sensitivity. It is not a clean error bound or proof that
512^3 is converged. Extra block-average cell windows remain; no assignment
window deconvolution, particle shot-noise subtraction, or multi-phase errors.
High-k behavior and sparse low-k cross-spectrum sign changes require follow-up.

## Artifacts and reproducibility

- Main PDFs: figures/tidal_statistics_20261006/pdfs_n512.png and .pdf.
- Main spectra: figures/tidal_statistics_20261006/spectra_focus_n512.png and .pdf.
  Display includes >=100 full-mode equivalents per shell and k<=0.2 h/Mpc;
  full-mode equivalents are not a count of independent complex modes.
- Full-range spectra, both resolutions, NPZ measurements, summary JSON,
  completion manifest and source hash are in the same directory.
- Resolution comparison: figures/tidal_statistics_20261006/resolution_comparison.png.
- Source: workflows/tidal_statistics/abacus_eigenvalue_statistics.py.
- Small-array plotting/summary: workflows/tidal_statistics/summarize_results.py.
- Approved launcher: workflows/tidal_statistics/run_abacus_eigenvalue_statistics.sh.
- Original run: /pscratch/sd/d/dkololgi/abacus/tidal_statistics_20261006.

Graphify query/update could not run because graphify is not installed; targeted
inspection was used. Figures were visually inspected. No new allocation or
scientific follow-up run is bundled into this completed analysis.
