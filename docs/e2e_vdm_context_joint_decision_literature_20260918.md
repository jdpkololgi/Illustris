# Joint-field decision: literature, CFM and available data

Research and assessment: 2026-09-18. This is a recommendation, not authorization
or launch of another experiment. The completed frozen A/B/C/D report and its
full closeout remain authoritative; this note adds the user's requested
joint-structure go/stop judgment and conditional-flow-matching assessment.

## Determination

**Yes: one bounded, explicitly joint-field follow-up is scientifically worth
doing. No: another open-ended architecture search, marginal-loss extension,
or claim of DESI readiness is not justified.** The new experiment must earn its
cost by improving spatial uncertainty and dependence, not by reproducing the
local posterior marginals already supplied by P12. P12's per-galaxy ordered
eigenvalue posterior is a different estimand, not an interchangeable density-
field baseline; independent draws from good local posteriors do not specify a
coherent joint field.

This judgment is an inference from the completed controls, not a claim that
CFM, wavelets or more simulations will necessarily solve the problem.

## What the completed experiment actually says

Sources: frozen `analysis/RESULTS.json`, its 688 case reports, and
[scientific closeout](e2e_vdm_context_results_20260918.md). All figures below
come from existing registered metrics; no refit, new target access, checkpoint
selection or new metric was introduced for this review.

- A to B improves density CRPS by 10.88% and passes the registered diversity
  contrast. B to C worsens CRPS by 3.37%. C to D improves the primary physical
  tidal energy score by only 3.54%, below the predeclared 10% threshold;
  confirmation-phase variograms regress in both seeds. The context and
  multiscale-package hypotheses are not established at this budget.
- Nevertheless, D repairs some *joint-scale* behavior. On the 64 final main
  cases, regional-mass C90 is 74.80%, versus A/B/C 56.64/46.88/31.64%; the
  attainable target is 90.75%. These are eight 162.384 Mpc/h parent octants,
  including the auxiliary halo, not the smaller owned science core.
- D's pooled fine-parent sample/truth powers in the five registered bands are
  [0.929, 0.918, 0.781, 0.737, 0.628]. The wider coarse field instead has
  [1.357, 0.949, 0.958, 1.000, 1.003], with voxel C90 89.24%. Thus neither
  blanket power suppression at every scale nor near-correct wide marginal
  power adequately describes the remaining failure. The wide lowest band
  still has excess power, and correct variance alone cannot fix conditional
  phase relationships.
- The adjacent-core controls are informative:

| D control | Joint energy (lower better) | Variogram (lower better) | Summary-difference C90 |
| --- | ---: | ---: | ---: |
| Fixed predicted coarse mean | 0.58455 | 0.01910 | 12.50% |
| Sampled shared coarse field | 0.33415 | 0.01293 | 33.33% |
| True coarse field, diagnostic only | 0.28066 | 0.00361 | 68.75% |

These are equal averages over eight seed/phase/cap cases, each containing six
summary differences; they are not 48 independent cosmological realizations.
Finite-ensemble attainable coverage is 87.88% for the first two rows and
88.24% for the 16-draw oracle. Oracle information changes the conditioning
problem: its better score localizes a bottleneck, not deployable performance
or a guaranteed lower bound for every score. Sampled versus fixed-mean coarse
improves both scores in all eight cases, but correct stochastic dependence is
still far from demonstrated. Supplying true coarse structure helps the
variogram much more than it repairs all remaining fine uncertainties.

The sampler refinement already passes. Faster or longer integration cannot
be assumed to correct this statistical error. Some checkpoint scores still
improve, but marginal and spectral trends diverge, so there is no evidence
that extending the unchanged fit will jointly converge to the desired object.

## What the relevant papers support, and what they do not

1. **Ono et al., CAMELS VDM.** Stellar-mass-to-matter inference is a useful
   conditional-generative reference, but the demonstrated task uses small
   periodic 2D projected maps and much broader simulation diversity. PDF,
   spectra, cross-correlation and regional-mass checks are relevant; the word
   unbiased is not a theorem of full conditional field calibration for masked
   3D galaxy data. Our VDM adaptation now addresses much of the original
   representation/objective gap; copying that label again is not a new test.
   [Paper](https://arxiv.org/html/2403.10648v1).
2. **Park et al., 3D Reconstruction of Dark Matter Fields with Diffusion Models.**
   This closer 3D CAMELS study reports good one-point PDFs but insufficient
   small-scale power at its highest resolution, despite 1,000 training
   simulations and 320,000 updates. Our failure is therefore not unprecedented,
   and longer training alone is not established as its remedy. Its small
   periodic stellar/galaxy setting still differs from our survey target.
   [Primary workshop paper](https://openreview.net/pdf?id=7k2Eh7OCoz).
3. **Guth et al., wavelet score models; Gerdes et al., GUD.** Normalized
   conditional wavelet factors and covariance-aware bases/noise schedules
   address multiscale conditioning rather than simply enlarging a CNN. The
   former supplies Gaussian theory and physics/image experiments; the latter
   explicitly formulates whitening and component-dependent diffusion. These
   motivate fixed train-only scale normalization, not a claim of proven DESI
   posterior calibration. A masked conditional covariance is not generally
   diagonal in a Fourier basis, even if the unconditional prior is.
   [WSGM](https://arxiv.org/html/2208.05003),
   [GUD](https://arxiv.org/html/2410.02667).
4. **Cosmo3DFlow.** Wavelet-space CFM, scale conditioning and cross-scale skips
   improve reconstruction/sampling for periodic late-time matter to initial
   conditions, using 1,800 training simulations in its standard split. This
   is not DESI galaxies to late-time matter. Its optional 0.01 log-power term
   acts on predicted/target *velocities* after inverse DWT. Its evidence is
   mainly reconstruction/power/correlation, not our joint-field coverage gate.
   The invertible Haar transform reduces spatial size but retains the total
   coefficient count. Our earlier flat Haar-CNN denoiser was not a reproduction
   of this multiscale flow model. [Paper](https://arxiv.org/html/2602.10172v1),
   [authors' implementation](https://github.com/UVA-MLSys/Cosmo3DFlow).
5. **Manticore II/BORG.** A coherent physical forward model, galaxy bias,
   redshift-space mapping and survey likelihood support large-volume inference
   from actual galaxy surveys, with external tests. This is evidence for the
   scientific programme, not for an inexpensive neural substitution. The
   analysis costs roughly 30 million CPU hours, uses tiled approximations,
   and includes spectrum/Gaussianity consistency priors and final-field moment
   regularization. Its spectral agreement is not wholly independent of these
   constraints. We should borrow the explicit observation/physics accounting,
   not propose a full BORG-scale run as a routine fallback.
   [Paper](https://arxiv.org/html/2606.10020).

The inference relevant to us: multiscale conditioning and the conditional
observation model deserve a controlled test; neither a visually plausible
field nor matching an unconditional power curve proves the posterior correct.
Adding a nonlinear per-realization power penalty can change the population
objective and distort spread. Fixed invertible preconditioning with a valid
generative objective is a cleaner first intervention than forcing every draw
to resemble its paired truth spectrum.

## Conditional flow matching deserves a fair comparison

CFM learns a transport velocity conditioned on the observation. A simple path
uses z_t=(1-t)epsilon+t*z and target velocity z-epsilon, where z is a complete
field representation and epsilon is random base noise. Drawing new epsilon
and integrating the observation-conditioned velocity gives different posterior
fields. Deterministic ODE integration **does not mean a deterministic point
prediction**. Sample-conditional regression has the same expected gradient
as the intractable marginal flow objective under the stated assumptions.
Straight per-example paths do not require the learned marginal trajectories
to be straight. [Lipman et al.](https://arxiv.org/html/2210.02747).

FMPE demonstrates that this construction can learn conditional posteriors,
not just attractive images. It permits unconstrained networks and offers
empirical inference gains. Its mass-coverage bound requires regularity not
guaranteed for an arbitrary trained network; small training MSE alone is not
a finite-capacity calibration guarantee. Its strongest posterior benchmarks
are lower-dimensional than our field, so they do not establish this result
at our scale. [Dax et al.](https://arxiv.org/html/2305.17161).

For this project the merits are:

- A genuinely different probability path/learning geometry, with potential
  improvement in how spatial modes are learned, using the same spatial model.
- Potentially fewer network evaluations per posterior draw. Our expensive
  draw campaign makes this operationally valuable *after* fidelity passes.
- Natural compatibility with a joint multiscale field representation and a
  shared stochastic parent. Neither a transformer nor a restricted invertible
  neural architecture is required for the velocity network.

The limitations are equally important. CFM cannot recover information removed
by the conditioner, manufacture independent training phases, correct an
incorrect galaxy/survey simulator, or create fine-core dependence forbidden
by a product factorization. Diffusion also admits ODE samplers: ODE versus SDE
alone is not a clean objective comparison. Density evaluation through CNF
divergence is possible in principle but is not automatically cheap or exact
numerically for a 3D field. No evidence supports claiming CFM is intrinsically
better calibrated or that the current result falsifies diffusion.

The current A/B/C/D campaign is VDM-only. Old 192/384-update CFM canaries used
a different small signed-residual pipeline and do not settle modern CFM versus
the present log-density VDM. Nor were these the programme's only flow tests:
the September 3 SCIENCE_LOG records genuine F3-L2 joint Fourier CFM recovering
the two low-mode powers to 0.9969/0.9433 and repairing much of the eigengap/shear
dependence, without passing the full calibration gate. Its conditional-Gaussian
prewhitened successor F3-L2c instead damaged eigengap dependence; target-matched
diffusion retained it better. Those are different hybrid/local targets, not
results of this new wide-domain matrix. They provide direct local evidence
both that CFM can learn useful joint structure and that whitening/conditional
Gaussian corrections are not automatically safe. Only published repository
summaries were read; no historical held-out payload was reopened.

P12-A is itself FMPE, but succeeds at a local per-galaxy posterior, not the joint
field. Consequently CFM is a justified matched challenger, not an untested
family, an already-disproved approach, or an automatic cure.

A related COT-FM paper corrects LPT-to-PM weak-lensing simulations and tests
downstream cosmological-parameter inference with TARP. Its full-field wording
refers to inference using maps; the tested posterior is over cosmological
parameters, not all latent voxels. It supports preserving conditioning and
testing downstream inference when correcting simulators, not an off-the-shelf
proof of our field posterior. [Zeghal et al.](https://arxiv.org/html/2510.24631v1).

## What simulation and DESI data we really have

The present prepared experiment uses three training phases (000/002/003),
development 004 and programme-exposed confirmation 005. Its 384 cores and
overlapping 1.3 Gpc/h contexts are not hundreds of independent cosmological
volumes. The experiment fixes cosmology, HOD and target epoch; observer redshift
changes selection, not the matter snapshot. Its P3b response is angular
support/targetability, not a validated full fibre/redshift-success likelihood.

Bounded, read-only metadata checks on September 18 establish:

- All raw `AbacusSummit_base_c000_ph007` through `ph024` directories exist in
  `/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit`. This is not yet a verified
  inventory of every required particle/redshift product. Official guidance
  says use CFS in place; some products are tape-only.
  [Abacus data access](https://abacussummit.readthedocs.io/en/latest/data-access.html).
- `AbacusSummitBGS` and `AbacusSummitBGS_v2` exist beneath
  `/global/cfs/cdirs/desi/survey/catalogs/Y1/mocks/SecondGenMocks`, with mock
  and altmtl directory indices 0--24. For example, mock7 has complete NGC/SGC
  clustering catalogues; `altmtl7/mock7/LSScats` has BGS_BRIGHT full/clustering
  catalogues and randoms. Only names were inspected, not predictive payloads.
- DESI documents complete, probabilistic FFA and full-algorithm altmtl mocks.
  These provide a feasible route to a paired selection control. The actual
  realization-to-phase/HOD mapping, coordinates, source epoch, target truth
  products and matching selection cuts must be audited first. Twenty-five
  mock indices are not automatically twenty-five independent training phases.
  [DESI mock documentation](https://desidatamodel.readthedocs.io/en/latest/DESI_ROOT/survey/catalogs/RELEASE/mocks/AbacusSummit/index.html).

The existing R7/native matter caches are not automatically ready for all those
additional phases. No large data copy, target build, new fit, or ph001/ph006
payload access was performed. Real DESI supplies observations and response
information, not paired dark-matter truth. It cannot by itself teach the
missing posterior correlations; validation requires independent simulations
and eventually external tracers/held-out observations. Clustering weights
alone are not a complete field-level selection likelihood.

## One proposed follow-up, with a stop rule

The next scientific question should be: **can a genuinely coupled spatial
generator recover conditional joint fluctuations, when the representation and
training examples expose those fluctuations?** Not which method makes the
best-looking one-point histogram.

1. **Data feasibility first, no predictive tuning.** Audit matched matter and
   complete/altmtl BGS products for additional unused phases, freeze phase-level
   train/development/confirmation roles and a common target/selection contract.
   Preserve sealed phases. Count independent wide volumes, not crop count.
   If this cannot be built within an approved bound, do not buy another large
   fit on the same three phases and call it a diversity experiment.
2. **Make one common spatial design for a matched VDM/CFM comparison.** Generate
   at least an adjacent-core pair jointly within one stochastic parent, with
   a train-only normalized, invertible multiscale representation. Preserve DC,
   positivity/mass decoding and survey boundary handling. Give each compared
   factor the same observation information, explicitly resolving the current
   coarse-conditioner restriction rather than silently assuming sufficiency.
   This design must permit residual cross-core dependence after conditioning
   on the coarse field; shared coarse noise alone cannot supply it.
3. **Use a retained D baseline and two seeds.** Retrain the baseline on exactly
   the same new phase panel if claiming a gain over D. Compare matched joint-
   design VDM and CFM at equal data exposure with allocated compute and sampling
   error reported. The design-package contrast is not a pure objective effect;
   only VDM versus CFM within the matched joint design isolates the objective.
   Avoid auxiliary spectrum penalties in the first comparison. Benchmark cost
   before proposing exact run limits; no open-ended optimizer/architecture sweep.
4. **Register a joint-only promotion criterion.** Require improvement on both
   seeds and phase-held-out confirmation in sample/residual power, regional
   masses, adjacent-core joint energy and variograms, tidal/eigengap calibration,
   and condition-stratified coverage, without unacceptable marginal regression.
   Include mean/residual power decomposition and oracle/fixed-parent controls.
   Do not require each draw's spectrum to equal its particular paired truth.
   Choose tolerances and a finite experiment budget before looking at new truth.
5. **Stop this neural route if the matched test fails.** Do not answer another
   joint failure with more marginal fits. Reassess the forward/selection model
   or a bounded physics-likelihood posterior reference before further amortized
   modelling. That does not mean launch a survey-scale BORG campaign or discard
   P12's separately useful local product.

This is a recommendation to develop a bounded protocol and resource proposal,
not to launch it automatically. The current experiment has provided a useful
negative result and a localized positive lead; it has not produced a calibrated
DESI conditional field posterior.
