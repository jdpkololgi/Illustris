# Galaxy-position repair: independent readiness review, 20 October 2026

## Decision

Proceed with staged, versioned catalogue repair for the five qualified phases
(ph000, ph002–ph005), subject to normal compute approval. Do not launch production
retraining yet. The coordinate repair is supported; the downstream replacement
contract and low-redshift training extension are not complete. Retain R=7 Mpc/h,
the existing native matter fields, and all historical products in place.

This is a readiness review, not a production execution authorization or a new
full-catalogue validation. No Slurm jobs, production replacements or fits were
launched. Existing uncommitted repair work was inspected without modifying it.

## What was checked now

- Read current SCIENCE_LOG, lifecycle register, migration protocol and repair,
  qualification, sampling, identity-join and downstream training code.
- Re-ran all 25 focused annotation/lifecycle/P10 parent/join tests in cosmic_env:
  all pass (test execution 0.144 seconds; small synthetic fixtures only).
- Independently verified 35 bindings: 16 source/test hashes, seven archived
  evidence hashes, certificate and registry hashes, five live raw-source/field
  size/mtime inventory checks and five independent-row-index hashes. All pass.
- Inspected the prior 619-row actual-source CLI smoke receipt: all rows labelled,
  including 14 unresolved satellites; exact reconstructed-voxel eigenvalues,
  metadata/independent-status checks and overwrite rejection pass in all phases.
  This smoke was not rerun today. Large catalogue/grid payloads were not rehashed.
- Tested current P10 shell logic on synthetic z=[0.08,0.12,0.20], even granting
  valid labels to every row: active=[false,false,true], context=[false,true,true].

The verified evidence is in `docs/evidence/galaxy_position_allpop_v2_20261019/`.
The certificate SHA256 is
`b1bd606a5984e6dc20ba9ec19254cbb848a93a67d2ca12266bd839361db6be15`.
These checks establish consistency of the reviewed implementation and saved
validation, not an independent repeat of the native-data campaign.

## Scientific assessment

The reported low-z defect is convincing. The original file-0 audit found large
wrong-location errors for direct halo-index lookup; filtered-namespace recovery
matched held-out resolved central identities. The replacement avoids this index
namespace entirely: fixed c000 distance, corner-observer offset, periodic native
frame and RA/DEC/Z_COSMO determine each galaxy's position. Observed Z does not
enter truth reconstruction. SOURCE_ROW is attached before selection and verified
against original coordinate/population/mass metadata, preserving satellites as
individual galaxies rather than collapsing shared hosts.

The five-phase native-reference audit is meaningful independent evidence, not
merely a coordinate round trip: original cubic galaxy positions include high-z
satellites; native halos and particles check low-z centrals. Recorded maximum
cubic positional errors are below 0.0013 Mpc/h, versus a 0.005 Mpc/h tolerance
and approximately 0.977 Mpc/h field cells. Low-z satellites do not all have
independent original XYZ references. That limitation is correctly separated in
INDEP_POS; it does not justify dropping their available galaxy positions.

Two changes must be distinguished: correction of wrong low-z locations, and
migration from host-centre to own-galaxy targets (also relevant to satellites at
higher redshift). Thus this is not evidence that all previous z>=0.15 labels
were randomly wrong, nor a reason to retain legacy models as new-target models.
Measure old/new differences on identical rows by observed and real redshift,
RES/CEN, mass, cap and phase. Do not infer improved learnability before fitting.
The fixed z=0.2 snapshot also remains a fixed-epoch mock across its radial range;
coordinate qualification does not validate light-cone evolution or DESI transfer.

## Remaining barriers before retraining

1. **Selection phase is not qualified.** The certificate and validator allow
   ph000/ph002–005 only. ph006 is the registered validation/selection phase, so
   extend the same independent coordinate audit to it before a replacement-model
   campaign. Do not infer qualification from its inclusion in a row census.
   Preserve access controls on ph001 and other reserved phases. ph001 has already
   been exposed by P12-A historically: a registry saying sealed does not make it
   fresh blind evidence for this replacement (see the September-24 VAC plan).
2. **New FITS do not yet imply new training targets.**
   `migrate_p10_galaxy_position_truth.py` builds metadata, annotation and observed
   truth only. `p10_build_phase_index.py` still defaults to the old observed
   filename; `p10_prepare_training_contract.py` reads `p1_canonical` and existing
   P4 assignments. The latter writes parent_eigenvalues from the catalogue in the
   P1 manifest without a v2 lifecycle guard. The register's narrow guards in two
   builders do not protect every training path. A dedicated versioned adapter
   must bind corrected source hashes, identities, targets, masks, scalers and
   training contracts; prohibit silent old/new mixtures.
3. **Low-z remains excluded in code.** `p10_build_phase_index.py` initializes
   valid_target from BOX_INDEX>=0 (lines 253–264), fixes science shells to
   0.15–0.55 and context to 0.10–0.60 (lines 37–39). The standard SBI cache builder
   also excludes BOX_INDEX==-1 by default. BOX_INDEX describes legacy source
   linkage, not availability of new own-position labels. For a low-z extension,
   explicitly version target validity using the v2 label contract; revise shells,
   context/boundary support, selection-response fits, weighting and evaluators
   together. The current selection fit starts at 0.15. Do not simply lower one
   redshift cut or reinterpret unchecked INDEP_POS=0 as invalid truth.
4. **Reuse needs a payload-level proof.** Compare ordered galaxy IDs, observed
   RA/DEC/Z, coordinates, membership, masks and selection arrays exactly. Preserve
   valid fold assignments where possible, while auditing legacy host-group keys:
   the low/high-z halo namespaces cannot be assumed interchangeable. Identical
   point arrays alone do not certify topology, active masks or fold eligibility.
   A range extension can require new graphs/patches/responses even when a label-
   only replacement in the old range can reuse them.
5. **Freeze the retraining chain, not just the final posterior.** Refit target
   scalers on training phases only; preserve ordered-softplus targets. Rebuild
   supervised base/encoder models and their out-of-fold summaries where they
   depend on old labels, then retrain the conditional posterior. The present
   P12-A halo48 preparation explicitly enforces legacy encoder-checkpoint parity;
   it is not a replacement-training driver. Re-evaluate accuracy and calibration
   against corrected held-out development truth, separately by redshift and
   population, and compare against a redshift/selection-conditioned prior.
6. **Freeze source and benchmark the complete path.** Most repair files are
   currently uncommitted. Their verified hashes identify what was tested, but a
   production snapshot must include those files and certificates. The existing
   real-source smoke exercises annotation; it does not demonstrate full-scale
   parent reconstruction, joins, replacement registration, target rebuild and
   training-loader parity in one chain. Measure this on one phase before choosing
   production resources. The lifecycle registration adapter currently supports
   annotation FITS receipts only; observed joins/caches/models need adapters too.

## Concrete next sequence

First qualify ph006 and freeze the implementation. Build one qualified phase
into a new galaxy_realspace_v2 root, perform full row/geometry identity and
finite/ordered/class/derivative checks, and record old/new label differences.
Validate a new P1-to-training adapter against that phase before scaling to the
remaining qualified phases. Never overwrite old paths or fit to a mixture of
legacy and corrected targets.

Use the existing redshift range as a controlled replacement baseline. Prepare
a separate, explicitly specified low-z extension with adequate observed context,
selection conditioning and registered per-bin accuracy/calibration gates. The
lower endpoint and evaluation exposure plan must be fixed before training;
the repair does not itself choose them. After data/loader/scaler and short
training-resume checks pass, start fresh supervised training and subsequent
posterior calibration under a reviewed compute budget. Regenerate VACs only
after replacement evaluation passes. This staged order avoids conflating the
label repair with a simultaneous change of selection, model and target scale.

## Consequences for our earlier geometry analysis

The October-17 mock geometry tests used legacy host-labelled truth and P12-A.
Their true-wall fractions, probability residuals and named true-wall/filament-
error examples must be recomputed under corrected galaxy labels before supporting
claims about the new target. In particular, the earlier ~83% true-wall result is
not a validated own-galaxy-truth accuracy result. Preserve it as historical.

The directly measured matter anisotropy and paired same-galaxy real/redshift
geometry remain measurements of those regions; this lookup defect does not by
itself invalidate them. Their interpretation as examples of a particular tidal
class remains provisional until reannotation. Likewise, direct-density coupled/
whole-field targets do not inherit this defect merely because they share a
project directory. Their previously failed calibration gates still stand.
