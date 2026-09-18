# Added user requirement and research ownership, 2026-09-18

At about13:36UTC the user added an explicit final requirement: determine
whether another experiment set is worth attempting to repair joint dependence
and power, given that marginal structure is already available from models
including production P12. Research relevant papers and account for the actual
simulation/DESI data. Do not close the goal with marginal improvements alone.

The login35 continuation owns this bounded literature review and will write
docs/e2e_vdm_context_joint_decision_literature_20260918.md. It is not launching
new fits or changing the frozen experiment. The login21 continuation retains
ownership of the full D/results/figures/science-log/field-plan closeout per the
existing handoff. Please integrate this literature assessment with the final
D fixed-mean/oracle/dependence and wide-power evidence before the final go/stop
recommendation. The literature note will distinguish demonstrated joint
calibration from power/reconstruction tests and will not assume D's result.

At13:46UTC the user additionally requested discussion of conditional flow
matching in the conclusion. The login35 review includes FMPE/CFM foundations,
Cosmo3DFlow and fair objective/representation contrasts; it will distinguish
random-base ODE posterior sampling from deterministic point regression and
avoid claiming that a faster sampler repairs missing joint covariance.

The literature note is now written at the stated path. Recommendation: yes to
one bounded joint-field follow-up, no automatic launch or broad model search.
It incorporates completed RESULTS/pairs and 64 final D-main case JSONs:
D regional C90=.748046875; wide C90=.8923938893; pooled wide P ratios
[1.3571,.9487,.9582,.9999,1.0035], fine [.9286,.9183,.7807,.7365,.6283].
D shared stochastic versus fixed mean improves pair energy/variogram in all
eight cases; the true-coarse diagnostic improves pooled variogram .01293 to
.00361 but does not establish calibration. This supports targeting spatial
conditional dependence rather than treating all errors as marginal suppression.
The note explicitly retains F3-L2/F3-L2c historical CFM evidence from SCIENCE_LOG
and documents metadata-only availability of raw ph007--024 and BGS complete/
altmtl products; phase/HOD/epoch/truth pairings remain unaudited.

The review also includes the directly relevant Cosmo-FOLD (arXiv2601.14377):
a globally shared noisy state with shifted patch partition each denoising
step, rather than independent completed-patch stitching. Its 3D stellar-to-DM
power/bispectrum results justify a coupled-state test; periodic/same-volume
and calibration caveats prevent treating it as a ready DESI posterior method.

New user message atabout14:00UTC adds a Gemini literature sanity check, including
attachment f4cc3530-dd0a-4630-9a81-8c1f06d61e9a/pasted-text.txt (read fully).
The login35 continuation is verifying four papers2511.14667/2408.00839/
2606.00803/2603.14503 and the claimed necessity of DPS, likelihood scaling,
stochastic sampling and exact galaxy likelihoods. Do not treat Gemini's
DDIM-to-MAP claim as correct: random-base deterministic transport remains a
posterior sampler. The requested final conclusion must incorporate this audit;
no new experiments are requested or launched by this added review.

Gemini audit now written: docs/e2e_gemini_literature_audit_20260918.md.
All four references are real, but their tested estimands/calibration differ.
Conicus3D Eq7 uses noise-dependent Fourier covariance (not simply raw P(k));
its prior is independent thick lensplanes with a redshift-conditioned 2D
network, and uncertainty validation is error/spread correlation, not TARP.
DES-Y3 scales approximate guidance and tunes with TARP; galaxy inpainting
tests power-vector TARP in the opposite conditional direction with residual
bias; cluster DAPS uses a photometry-conditioned prior and marginal reliability.
The audit corrects DDIM-to-MAP, mandatory DPS/unconditional-prior, universal
Poisson and differentiable-HOD-likelihood claims. It preserves the bounded
joint VDM/CFM recommendation and notes observation double-counting hazards.
Please link both literature notes in final closeout records. No code/compute
or final-results/science-log/field-plan edits were made by this reviewer.
