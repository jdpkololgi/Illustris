# Inference-strategy decision note: amortised posterior vs learned prior

**Status: a costed comparison for user decision. Not an authorization, not a
launch, and not a recommendation to abandon completed work.** Prepared
2026-09-19 alongside the pre-Stage-A diagnostics in
[evidence](evidence/e2e_diagnostics_20260919/README.md). Costing script:
`workflows/sbi/diagnostics_20260919/cost_non_amortised.py`; receipt
`COST_NON_AMORTISED.json`.

## The objective, stated precisely

The programme goal is a posterior over the matter field given DESI BGS-like
galaxies and their observational selection:

```
p(delta(x) | galaxies, selection)  ~  p(galaxies | delta, selection) * p(delta)
```

What the current plan estimates is narrower:

```
p(delta(x) | galaxies, HOD fixed and unrecovered, cosmology c000, one mock
             selection realization)
```

The HOD parameter files were not recovered from the source FITS headers, and the
P3b response is angular support/targetability rather than a validated fibre and
redshift-success likelihood. The galaxy-matter connection is therefore an
assumption baked into the training set, not a conditioned or marginalized
quantity. Every calibration test in the programme draws its truth from the same
generative process it trains on, so no in-distribution gate -- TARP, SBC,
coverage, variogram -- can detect misspecification of that connection. This is
the dominant unbounded error in the current strategy.

## Two routes

**Route A, amortised (current plan).** Learn a conditional generative model that
maps (galaxies, conditions) to posterior field samples, amortized over
configurations. Inference is then 128 network evaluations per draw.

**Route B, learned prior with explicit likelihood.** Learn `p(delta)` as an
unconditional score model from the same simulations. Write `p(galaxies | delta)`
explicitly: HOD/bias, RSD, angular and radial selection, fibre assignment,
redshift success. Sample the posterior with MCMC (annealed Langevin, HMC, or a
validated guidance scheme) using the learned score and the explicit likelihood.
Nuisance parameters of the likelihood become quantities to marginalize over.

## Costing

All network rates are measured, from GPU receipt `c9eb5cd...` (job 58559826,
A100-SXM4-80GB, float32, batch 2): **51.77 ms per score evaluation** on the
147,456-voxel rectangle, 0.2330 s per batch-two training update, 1.105 GiB peak
reserved. The one genuinely unknown quantity is the number of score evaluations
needed per independent posterior sample, so it is swept, not assumed.

### Training

| | 384 epochs (literature parity) |
| --- | ---: |
| Route A: 14 conditional factor fits | **289.5 GPUh** |
| Route B: one unconditional prior, two seeds | **41.4 GPUh** |

Route B is **7x cheaper to train**, because the arm matrix exists to compare
conditioning strategies that Route B does not have.

### Inference on the registered panels

| Route | panel | 2,000 evals | 5,000 | 20,000 |
| --- | --- | ---: | ---: | ---: |
| B, same confirmation panel (24,576 fields) | as proposed | 748 GPUh | 1,808 | 7,109 |
| C, reduced panel (24 fields x 64 draws) | sampler/likelihood validation only | **86 GPUh** | **152** | **483** |

Route B on the *existing* confirmation panel is unaffordable. But that panel
exists to test whether an amortized network generalizes across configurations.
A non-amortized sampler has no amortization to validate: its calibration is a
property of the sampler and the likelihood, checkable on far fewer fields and,
on a Gaussian or LPT rung, against an exact reference posterior. The honest
comparison is therefore Route A's 710 GPUh proposed ceiling against Route C's
86-483 GPUh.

### The survey-scale result, which reframes the problem

BGS BRIGHT, 14,000 deg^2, 0.1 < z < 0.4:

- comoving volume **1.76 (Gpc/h)^3**
- at 6.766 Mpc/h cells: **5.7M voxels = 39 coupled rectangles**
- a single global score evaluation needs **21.3 GiB** and **2.0 s**

**The entire BGS survey volume fits on one A100 at the working resolution.** The
cross-core coherence problem that the I/J experiment was built to probe is a
problem of stitching a domain that does not need stitching: the whole footprint
is one forward pass. The two-core coupling test is measuring a nuisance of a
decomposition that Route B does not require.

Resolution is the design lever, not the footprint:

| cell Mpc/h | voxels | GiB per field | 80 GiB A100s | s per evaluation | GPUh per sample at 5,000 evals |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6.766 | 5.7M | 21.3 | 1 | 2.0 | 2.8 |
| 4.000 | 27.5M | 103.0 | 2 | 9.7 | 13.4 |
| 3.383 | 45.5M | 170.3 | 3 | 16.0 | 22.2 |
| 2.000 | 220.0M | 824.3 | 12 | 77.2 | 107.3 |

100 posterior samples over the whole footprint at 6.766 Mpc/h cost **28 to 277
GPU node-hours** across the swept evaluation counts.

### Architecture constraint found by the costing

At footprint scale the U-Net bottleneck carries ~11,114 tokens, so the current
global self-attention becomes 1.24e8 pairs and quadratic in memory. A
footprint-scale prior needs windowed or local attention, or none. This is a
design decision to take before training a prior, not after.

## Is this still simulation-based inference?

Partly, and the honest description is more useful than the label. SBI means
inference where the likelihood is intractable, so simulations are used to learn a
posterior, a likelihood, or a ratio. Amortization is common in SBI but not
definitional -- sequential SNPE/SNLE/SNRE are non-amortized.

Route B is neither classical SBI nor classical Bayesian inference. The **prior is
implicit and learned** from simulations; the **likelihood is explicit and
written down**. The accurate name is field-level Bayesian inference with a
learned (score-based) prior.

Worth being clear about why this is not a retreat: for the field-level problem
the likelihood `p(galaxies | delta)` was never the intractable object. Given a
matter field, the galaxy field is produced by bias/HOD, RSD and selection, all of
which can be written and evaluated. What is intractable is the *marginal*
likelihood after integrating over the ~10^6-10^7-dimensional delta -- which is
why BORG samples delta jointly rather than marginalizing it. The field-level
problem needed a way to handle a huge latent space, not a way to avoid a
likelihood. Adopting an implicit likelihood was an inherited choice from the P12
per-galaxy SBI lineage, where it was correct, and has never been re-justified for
the field problem, where it is not obviously correct.

## How this differs from BORG / Manticore

BORG and Manticore place an exactly-known Gaussian prior on the **initial
conditions**, evolve them with an approximate gravity solver (LPT/PM/COLA) plus
bias, RSD and selection, and sample the ICs with HMC through that differentiable
forward model. Manticore II costs roughly 30 million CPU-hours, uses tiled
approximations, and includes spectrum and Gaussianity consistency priors plus
final-field moment regularization.

Route B replaces *(exact Gaussian IC prior + expensive gravity forward model)*
with *(learned prior directly on the late-time field)*. The differences follow
mechanically.

**Genuine advantages**

1. **No gravity solver inside the sampler.** Each score evaluation is one network
   forward pass, not a PM simulation. This is the whole cost argument: 28-277 GPU
   node-hours for 100 footprint samples against a ~234,000 CPU node-hour
   reference. Even allowing that CPU and GPU node-hours are not interchangeable
   and that Manticore solves a harder problem on a different survey, the gap is
   two to three orders of magnitude and it is the reason to care.
2. **An N-body-accurate late-time prior.** BORG's prior is exact but its forward
   model is approximate gravity. Route B's prior is approximate but trained on
   full Abacus N-body, so late-time small-scale statistics can be *better* than a
   PM-based field-level posterior. The two methods are approximate in different
   places, which makes them genuinely complementary rather than redundant.
3. **DESI BGS with the real selection machinery.** Field-level posteriors exist
   for SDSS, 2M++ and BOSS. Applying one to DESI BGS with actual altmtl fibre
   assignment and redshift-success modelling does not exist, and is an
   application contribution independent of methodology.
4. **The validation framework.** This programme's diagnostic apparatus is
   stronger than the field-level literature's. BORG-family spectral agreement is
   not wholly independent of its own consistency priors, as the earlier
   literature review noted. A posterior validated against an exact reference on a
   Gaussian rung, with registered proper scores and a measured effect-size
   analysis, would be a better-evidenced claim than the incumbents'.

**Genuine disadvantages**

1. **No initial conditions.** A late-time-field prior yields delta at the target
   epoch, not the ICs. That forecloses IC-based analyses and is the main reason
   BORG exists. The stated objective asks for `p(delta(x) | galaxies)` and does
   not mention ICs, but this should be an explicit decision rather than a silent
   loss.
2. **No joint cosmology, initially.** BORG's prior is analytically parameterized
   by cosmology; a learned prior is trained at fixed cosmology. Marginalizing
   would need a cosmology-conditioned prior, which is trainable -- AbacusSummit
   has cosmology variations -- but is additional work and an additional claim.
3. **The prior cannot be checked the way an analytic one can.** A Gaussian IC
   prior is right by construction. A learned prior is right only insofar as the
   training simulations and the fit are right, and its failures are exactly the
   failures this programme has spent months documenting.
4. **Sampler approximation error must be characterized**, not assumed. HMC on
   BORG's model is exact for that model; annealed Langevin with a learned score
   is not automatically so.

The fair summary: **Route B is not a replacement for BORG. It is the middle
ground between an amortized neural posterior and a full physical forward model,
and its case rests on cost per posterior, late-time accuracy, the DESI-specific
observation model, and better validation.**

## What the current strategy gets right, and what carries over

- **P12-A is a finished, calibrated product** on a different estimand -- per-galaxy
  joint eigenvalue posteriors, 4,897,905 rows, coverage90 0.891/0.895/0.890,
  five to seven times inside its registered gate. It does not depend on any of
  this and should not be reopened.
- The 21-phase preparation, coordinate and physics operators, normalization and
  T-Web machinery are reusable, and are *more* useful with an explicit likelihood
  than without.
- The metric, scoring and effect-size apparatus is the strongest part of the
  programme and is the seed of a methods paper either way.
- An unconditional prior trains with the existing VDM/CFM code with conditioning
  removed, so Route B is not a rewrite of the model stack.

## Recommendation

Do not choose yet. The decision hinges on one unmeasured number -- score
evaluations per independent posterior sample -- which spans a 10x range in this
note and drives everything. Buy that number before buying either route:

1. **Measure the sampler cost on a Gaussian rung**, where the exact posterior is
   computable. Train a small unconditional score prior on a GRF with the LCDM
   P(k) at reduced resolution, write the trivial Gaussian likelihood, run
   annealed Langevin, and count evaluations to reach a target Wasserstein or
   covariance error against the exact Wiener-filter posterior. This simultaneously
   fixes the cost unknown and establishes whether the sampler is calibrated at
   all. Estimated cost: a few GPU-hours.
2. **Write the explicit `p(galaxies | delta)`** for the mock selection already
   prepared. No GPU. This is the piece that makes misspecification testable and
   is required by Route B, valuable to Route A, and currently the least-developed
   part of the stack relative to its weight in the objective.
3. Only then decide whether Stage A proceeds as an amortized experiment, as a
   Route B pilot, or as a reduced version of both.

Nothing in this note authorizes a run, changes a frozen artifact or reopens a
sealed phase.
