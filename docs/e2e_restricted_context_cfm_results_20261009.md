# Restricted exterior context and learned CFM results

The full-exterior Gaussian sampler remains correct, but none of the tested
restricted-context interfaces passes across the complete panel. Nine restricted
configurations are unstable. A single exterior-conditioned CFM pilot learns a
substantial approximation but does not meet the frozen accuracy gate, despite
adequate numerical integration refinement. No Abacus intervention is justified
by these results alone.

## What the Gaussian experiments mean

Knowing a probability distribution is not knowing the realized density field.
The synthetic experiment specifies a Gaussian prior for a density-contrast
vector x and a masked, heteroscedastic, additive Gaussian observation model.
A random true field generates the synthetic observations y. Analytic Bayesian
conditioning then supplies the exact Gaussian posterior p(x|y) as an answer
key. These observations are noisy field measurements, not galaxy catalogues.
A Gaussian prior alone would not make a realistic galaxy likelihood or its
posterior Gaussian.

Sampling means generating possible COMPLETE fields from that posterior, not
subsampling observed galaxies and not producing only a best-fit map. A block
update holds the current sampled exterior fixed, draws a new block from its
conditional posterior, and then updates other blocks. The initial exterior
can be wrong; it is not frozen forever or assumed to be the true exterior.
Correct, convergent updates explore exterior uncertainty as part of the joint
chain. Fixing one arbitrarily chosen exterior permanently would instead answer
a different, more strongly conditioned question.

The global update decomposes the current field into constant/fundamental
sine/cosine coordinates and their orthogonal complement. It redraws those seven
coordinates jointly from their posterior conditional on the complement and y.
It is not independent coarse-field sampling, posterior-width inflation, or an
update from the existing coarse network. In the preceding fine-constraint test,
the mode basis was projected into the permitted block-zero-mean subspace, so
the constant direction disappeared and coarse block masses did not change.

The preceding exact experiment established that a known target posterior can
be preserved and reached by the correct block sampler. It did not demonstrate
that our learned 3D CFM model recovers a posterior from galaxies. The new pilot
below is the first learning test of this exterior-conditioned interface, still
in a deliberately simplified Gaussian family.

## Restricted context results

The frozen panel has20 Gaussian targets,264 sampler/context/order rows, fixed
2^3-cell blocks at n4/n6/n8, two observation regimes, and periodic/nonperiodic
examples. All observations remain available through the analytic posterior;
only the sampled-field exterior is restricted. The new periodic prior holds
cell spacing and its1.5-cell correlation parameter fixed as domain size grows;
it is not the earlier fixed-domain-frequency prior. Context variants and gates
are specified in the prospective protocol.

| Exterior context | Exact invariance passes | Stationary accuracy passes | Rows |
|---|---:|---:|---:|
| Full exterior | 44 | 44 | 44 |
| None | 12 | 12 | 44 |
| One-cell collar | 20 | 20 | 44 |
| One-cell collar plus exterior mean | 20 | 20 | 44 |
| One-cell collar plus seven modes | 20 | 20 | 44 |
| Two-cell collar | 32 | 33 | 44 |

These are descriptive counts of dependent engineering cases, not statistical
success rates. Each context has both local-only and local-plus-exact-global
arms; the sparse n8 long-range target also has reversed and shifted sweeps.
At n4 a periodic one-cell collar already covers the full exterior; at n6 a
two-cell collar does so. Such passes are not evidence for large-domain locality.
The independent-field controls need no exterior information.

Every full-context row is stable and accurate. Its largest stationary worst-
direction relative covariance error is8.00e-14. Seven focused unit tests pass,
including independent covariance/precision conditional identities, explicit
dense composition, exterior-feature leakage checks and negative controls.
Independent source/panel/gate reconciliation passes2,040checks.

Nine restricted-context rows have transition spectral radius>=1. The n8
spectral/sparse one-cell-collar local chain has radius1.0288499. An independent
dense matrix composition agrees exactly with the efficient composition, ruling
out that bookkeeping error. Other unstable cases include the long-range/sparse
one-cell collar under all three tested order/tiling choices and several sparse
open-boundary variants. Exact conditionals given a shortened context need not
constitute compatible full conditionals for the desired joint law.

For the forward n8 long-range/sparse target, two-cell-collar local updates are
stable but have4.01% Frobenius covariance error and4.58% worst whitened covariance
error. Adding the exact global update reduces those to0.077% and1.41%, respectively;
the latter still misses the1% gate despite excellent registered-probe interval
probabilities. Global updates can help without repairing every missing dependency.

A two-cell collar is sufficient for the periodic spectral control because this
deliberately chosen prior has finite-range precision. A direct n8 precision
check finds far-exterior coefficients<=2.18e-15. Adding the common long-range
component introduces nonzero far-exterior precision coefficients, about
1.20e-4 in the checked block. This is a property of the synthetic model, not a
measured Markov property of cosmological density. A two-cell collar PLUS exterior
mean was not among the registered combinations and must not be conflated with
the tested one-cell-collar-plus-mean arm.

## Learned conditional CFM pilot

One fixed20,000-update run uses fresh joint prior/observation samples for the
n4 long-range/sparse family. Eight disjoint2^3blocks are conditioned on their
full exterior and noisy observations. Prior, mask and noise law are fixed;
eight held-out observation realizations test amortization over y. Context
whitening uses fixed synthetic joint covariance for numerical conditioning.
No per-observation posterior mean, covariance, precision or analytic target
velocity is supplied to the training network.

The model is a128x128 SiLU time network producing eight affine-in-state/context
velocity heads. This Gaussian-compatible structure makes learned sampler
moments exactly computable. It is a favorable diagnostic model, NOT the current
3D convolutional CFM and not evidence for non-Gaussian expressivity. Training
uses ordinary independent-endpoint linear CFM pairs and squared velocity loss,
not analytic teacher velocities. No checkpoint selection or rescue training was
performed; the final registered checkpoint controls conclusions.

Six focused tests pass. Exact conditional-CFM integration controls pass, with
maximum conditional covariance error0.0207% atNFE512. Actual learned-model Heun
calls agree with independently propagated affine endpoints to3.56e-15 on the
32-draw implementation check. Float32 training weights are evaluated in float64
for the analytic Gaussian QA.

| Final diagnostic at NFE512 | Learned local | Learned local plus exact global |
|---|---:|---:|
| Stable held-out chains | 8/8 | 8/8 |
| All stationary accuracy gates passed | 0/8 | 0/8 |
| Relative full-covariance Frobenius error | 0.597% | 0.115% |
| Worst whitened covariance error | 6.01% | 5.11% |
| Largest whitened mean-error norm across observations | 0.227 | 0.155 |
| Largest90% probe probability error | 0.700percentage points | 0.071percentage points |

Individual learned block conditionals also miss their gates: maximum whitened
mean RMS error0.0320 and maximum worst-direction covariance error1.49%. The
mean norm/RMS quantities have different definitions and are not interchangeable
percent errors. Bulk conditional covariance Frobenius errors are below1%, which
alone would hide the failed worst-direction and mean checks.

Increasing NFE256->512 changes the endpoint covariance by only0.0238% and mean
response by0.000209% in the registered relative norms. Thus the reported
integration refinement passes while the learned statistical approximation
fails. The exact construction and integration controls pass, and learned
components themselves do not yet meet tolerance: this is not evidence that an
accurate learned set of full conditionals was assembled incorrectly. Nor does
one bounded fit establish that CFM cannot learn them. Conditional approximation,
optimization and compatibility remain distinct follow-up questions.

The exact-global hybrid is an oracle intervention, not a fully learned method.
Its good probe interval probabilities cannot override failed worst-direction
joint accuracy. Independent checkpoint/source/gate reconciliation passes855checks.

## Decision and limits

Do not promote this candidate to Abacus. Preserve the successful full-information
oracle and both negative results. Before a larger fit, use the analytic pilot
to separate errors in learned conditional mean responses from innovation
covariances, then design a bounded learning/compatibility improvement. Do not
interpret the finite training budget as a proven approximation floor. Restricted
context should be justified by conditional dependence, not by visually smooth
fields or patch marginal accuracy. Any global-update learner is a further
component; the present pilot retains an exact global oracle only for diagnosis.

The new tests do not establish galaxy likelihood correctness, cosmological
posterior calibration, a qualified fine/coarse hierarchy, physical resolution
convergence, representative3D-network training, or DESI-scale cost. No Abacus
inputs, sealed phases, production sampler changes or further training were used.

## Execution and evidence

The approved temporary runtime remains read-only NERSC pytorch/2.8.0 with Python
3.12.11, NumPy2.1.2, SciPy1.16.2, PyTorch2.8.0+cu129 and h5py3.14.0. cosmic_env
and shared installations were not modified. CPU59595457/nid200006 and
single-GPU59595594/nid200468 have both been released. The complete oracle panel
took131.12seconds and the training/evaluation loop37.61seconds; these are small
synthetic diagnostic timings, not survey performance claims.

The first oracle run preserved an engineering stop on an unstable negative case.
Its handling was corrected to record instability, with no gate or target change,
and all cases rerun in a separate directory. First partial receipts remain under
`oracle`; the complete oracle uses `oracle_negativecases_v2`. A GPU step command
with an incorrect allocation ID was rejected before any workload started; smoke
and training ran in verified59595594. Graph query/update again timed out in the
Scratch-hosted tool; no graph/global-index refresh is claimed.

All artifacts are under
`/global/u2/d/dkololgi/diagnostics/restricted_context_cfm_20261009_v1`.
The completed oracle and learned pilot are `oracle_negativecases_v2` and
`cfm_train`, respectively. Each has source/runtime-bound COMPLETE.json and an
independent VALIDATION.json. COMPLETE SHA256 values:

- Oracle: `547872fe39dfd2633ed149dbee6ab7eb1a0b13171b86303b38023a24208cac1d`.
- Pilot: `7a859752f0d6aa6da02439a00dd15e2067e064a1d2f37239b78ac460afe4bfb2`.

Implementation checks total13focused tests and2,895receipt checks. These counts
validate implementation/provenance, not scientific success of failed candidates.
New code and documentation are uncommitted; unrelated working-tree changes are
preserved. Protocol: `e2e_restricted_context_cfm_protocol_20261009.md`.
