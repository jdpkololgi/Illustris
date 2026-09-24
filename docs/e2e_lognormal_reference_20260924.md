# Bounded nonlinear classical comparison

User authorizes implementation/execution and scientific conclusions. CPU only,
one30-minute allocation; no neural fitting, Abacus access or P12 changes.
4^3 periodic cells, Gaussian log-density prior from existing spectral reference
(unit voxel variance). Density delta=exp(g-diag(C)/2)-1. Poisson mean is known
exposure*(1+delta). Four independent observations: two mask templates crossed
with expected count scales .5 and5, spatial completeness .5..1.5. Known bias=1,
no RSD/cosmology/selection uncertainty; not a DESI likelihood. R in grid cells.

Laplace: MAP in prior-whitened coordinates, exact Hessian inverse; gradient
stationarity <=1e-6. Correct lognormal transformation of Gaussian log-density
draws, not a Gaussian approximation directly in delta.
Classical: eight Laplace-preconditioned MALA chains, exact Metropolis-Hastings
proposal correction,2048 warmup then32768 retained steps/chain. Adapt step size
only during warmup toward .574 acceptance. Four chains then use0.7times the
adapted step size as an independent mixing/step-size replication. Dispersed
initial states. This is MALA, not an implementation of HADES/HMC.

Before comparator claims: rank-normalized folded/split Rhat<1.01 and bulk/tail
ESS>400 across all64latent and64density voxels, eight2-cell regional masses,
global mass and log posterior; finite values, and two independently stepped
groups' means agree within5combined autocorrelation-aware MCSE across these
138diagnostics. ArviZ0.22.0 supplies diagnostics. No threshold relaxation or
automatic extension. Failure blocks comparison for that case and is reported.
Check finite-difference gradient/Hessian and prior-only Gaussian sampler control.

Compare Laplace with reference: density mean/variance,90% interval reference
mass, global regional mass, and smoothed threshold-zero tidal-class probability
at R={0,.5,1}cells. Use the second independent chain group as replication/noise
baseline. For expensive summaries take every eighth retained draw from each
group (16384draws/group); ESS is measured on full unthinned chains. Four cases
cannot establish prior-wide SBC or DESI coverage. Convergence diagnostics are
not proof of exact finite-sample convergence. No mathematically perfect gate.

Scientific decision: is a cheap Laplace approximation sufficient on tested
functionals, or does classical nonlinear sampling change probabilities materially?
Only then design a neural/hybrid cost-accuracy comparison on THIS nonlinear
generator, not re-use the Gaussian-trained network on a different target.

Literature: Jasche et al.0911.2498 motivates lognormal-Poisson field inference;
Vehtari et al.1903.08008 motivates rank/folded Rhat and bulk/tail ESS checks.
Neither establishes this toy as a validated physical model of DESI galaxies.
