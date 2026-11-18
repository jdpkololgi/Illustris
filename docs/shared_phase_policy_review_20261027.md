# Review of the shared phase policy and required data migration

Reviewed 27 October 2026 against Illustris commit `a8f8bb1`, comparing with
`3c98209`, and the GraphWeb_DESI handoff notice at `67ef8cf`. This review covers
the new policy, intervening executable/configuration edits, current data-product
metadata and the principal downstream dependencies. It does not revalidate
historical manuscript results or scientific payloads.

## Assessment

The adopted partition is internally consistent: 20 training phases, three
development/calibration phases and two final-test phases exhaust ph000–ph024
without overlap. The promotion of previously exposed ph001 and ph014–ph017 into
training is compatible with the new campaign, provided those phases are no longer
used as independent tests of its fitted models. R = 7 h^-1 Mpc remains fixed.
Two final-test phases will give limited phase-level uncertainty; many galaxies,
spatial regions or posterior draws do not increase the number of independent
simulation phases.

The changes correctly describe a policy decision, not a completed migration.
No scientific pipeline implementation changed between the reviewed commits.
The new policy is not consumed by runtime Python code, and neither expanded
pipeline is ready to launch. The previous deterministic restart results remain
useful evidence for the numerical implementation, not certification of the new
phase split or new data bindings.

## Priority migration issues

1. **Implement a versioned runtime authority before producing new data.** The
   P10 registry validator still requires ph001 to be sealed and the previous
   15/3/7 roles. CFM still enforces the old 13/2/6 authority and excludes four
   newly required phases. Preserve those historical contracts; add a new schema
   or explicitly selected campaign contract rather than changing old constants
   or weakening guards globally. Bind the policy hash into registries, product
   receipts, normalizers, checkpoints, exports and evaluation contracts.

2. **Migrate all response and export paths, not just the top-level registry.**
   The P3b-R builder excludes ph014–ph017 and treats ph001 through the historical
   blind-access route. The all-visible P12-A response driver also hard-codes the
   old contract and output root. Its per-phase products bind the full-contract
   and selection hashes, so a new contract cannot simply adopt old completion
   markers. The newer cross-fit and posterior-table utilities already read
   roles from their supplied full contract; retain that capability and add
   policy-consistency checks rather than replacing every old default in place.

3. **Make CFM coarse conditioning and normalization compatible.** The existing
   whole-field evaluator reads `coarse_logrho_standardized` from a frozen old
   full-cap bank and passes it directly to the fine model. It also decodes it
   using the fine model's normalization chart. Reusing those standardized draws
   unchanged with a new chart would change their physical meaning. The new
   campaign needs compatible coarse checkpoint/draw bindings and exterior
   caches. If historical draws are retained as a comparator, explicitly convert
   through physical units and label them historical; that does not make the old
   coarse model a freshly trained 20-phase model. Add a rejection test for
   mismatched normalizer bindings.

4. **Carry fixed-budget policy through every model stage.** The new policy
   requires fresh initialization and terminal checkpoints. The old expanded
   P12-A configuration permits registered selection-phase checkpoint choice for
   the full encoder, while enforcing terminal checkpoints for OOF fits. The new
   configuration must enforce its chosen terminal policy for the full encoder
   as well, and explicitly cover FMPE and the selected CFM factors. Freeze the
   meaning of budget (epochs or updates), scheduler horizon and sampling balance
   before fitting; copying an old epoch count does not preserve compute cost.

Relevant code: [P10 registry validator](../workflows/abacus_tweb/p10_phase_assets.py),
[CFM authority](../workflows/sbi/e2e_coupled_contract.py),
[P3b-R producer](../workflows/abacus_tweb/p3br_build_random_response.py),
[P12-A response driver](../workflows/sbi/p12a_prepare_visible_responses.py),
[response bindings](../workflows/sbi/p12a_prepare_response_inputs.py),
[field evaluator](../workflows/sbi/e2e_wholefield_pilot_evaluate.py), and
[historical P12-A configuration](../configs/p12a_corrected_expanded_inputs_20261026.json).
These are migration requirements, not evidence that the unchanged historical
runs are newly broken.

## P12-A products

The current corrected root is
`/pscratch/sd/d/dkololgi/abacus/p10_galaxy_v2_20261020`.
For each newly admitted phase ph001/ph014/ph015/ph016/ph017, metadata inspection
found `BLIND_INPUT_COMPLETE.json` and a P1 manifest, but no corrected catalogue
migration marker or `PHASE_COMPLETE.json`. The current response inventory has
18 phases and omits all five. Thus target-free preparation exists, but the new
training role is not yet implemented in the products.

| Product | Required action |
| --- | --- |
| New registry and campaign configuration | Encode 20/3/2 roles, new roots, source/policy hashes, terminal checkpoint policy and final-test restrictions |
| Five newly admitted phases | Qualify the own-position mapping; build corrected parent/observed labels and target-bearing caches; bind them to canonical row identity and phase completion receipts |
| Existing P1/P3/P4 geometry | Reuse payloads where membership, ordering, coordinates and assignments match; reissue new lineage/role receipts; rebuild only invalidated assignments or label-bearing content |
| Existing native matter | Audit reuse before deposition/restoration; CFM native-count receipts already exist for ph014–ph017, but they are not a ready per-galaxy eigenvalue catalogue |
| Response products | Build or adopt qualified angular maps for the five added phases; create a 23-visible-phase inventory with valid parent and new contract bindings |
| Full training contract | Create a full-20 contract, with train-only selection, input moments, target transforms, phase-balanced sample inventories and actual loader checks |
| Cross-fit contracts | Create 20 contracts, each fitted on 19 training phases; exclude the omitted phase from transforms and encoder fitting |
| Model-derived products | Fit 20 OOF encoders plus one full encoder; export new halo48 base predictions; build aligned FMPE training/development tables, fresh posterior fits and calibration results |

The old 15 OOF contracts cannot become 20-phase contracts by appending five
folds: the training population and fitted transforms of every fold change.
Existing corrected labels on the original 18 visible phases do not need to be
re-annotated solely because the split changes.

Angular response geometry and parent support/distance arrays may be reusable,
but receipt reuse and byte reuse are different. Reuse requires matching raw
randoms, geometry, parent order and feature definitions, followed by new source
bindings. Do not re-read all raw random catalogues merely because a full-contract
hash changed if the physical maps can be verified and adopted safely.

Neither requested model class needs new graph features as a scientific input.
Some legacy P10 completion scripts require P2 receipts as infrastructure; preserve
or adapt verified geometry metadata rather than rerunning graph extraction by
default. Existing ph001 P1 metadata points into the legacy tree, so avoid writing
through inherited directory links during promotion.

## CFM products

The coupled root is
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1`.
Its Cartesian target directory contains the old 21 phases and lacks ph001,
ph004, ph005 and ph006. ph006 becomes development; the other three become
training. ph014–ph017 already have matter and field products, but their existing
coupled geometry is held-out sampling (16 pairs per phase), not the 128-pair
training sampling. The whole-field selector separately expects 48 training
domains versus 16 evaluation domains per phase.

| Product | Required action |
| --- | --- |
| Data, loader and evaluation authorities | Version the old fixed role lists and roots; include all 20 training phases and ph006/012/013 development |
| ph001/004/005/006 CFM interfaces | Qualify source identity, coordinate conventions, observed counts/response, geometry, native matter and target extraction for the selected representation |
| ph014–ph017 training sampling | Expand training geometry and extract corresponding conditions/targets, or explicitly register a different balanced sampling design |
| Normalization | Fit the new train-only chart with the registered phase weighting; regenerate normalized interfaces and any dependent caches |
| Whole-field/domain dataset | Build or adopt physical payloads under the new selected-domain inventory; verify positive decoding, mass conservation, source identity and loader coverage |
| Development partitions | Register separate selection, calibration and evaluation subsets across all three development phases before use; account for overlapping context and spatial dependence |
| Coarse/reference and exterior products | Bind compatible coarse models, generated conditioning draws, full-cap/exterior caches and new evaluation inventories; keep historical comparison banks separate |
| Readiness evidence | Re-run focused role/leakage, new-phase loader, binding and deterministic restart checks on the migrated setup |

Absence in the CFM root does not prove absence of reusable matter elsewhere.
P12-A/P10 target directories exist for ph004/005/006; inspect their actual native
counts and conventions before deciding to restore particles or redeposit them.
ph001 requires a bounded source inventory; this review did not establish that
no usable source exists elsewhere. For ph014–ph017 the registered native count
receipts already point into the coupled CFM matter root.

One explicit reuse decision is still needed: CFM observations inherit a selection
manifest fitted on ph000/002/003/004/005. All five are now training phases, so this
is not development/test leakage. Keeping that historical training-only selection
fixed may allow more physical observation products to be reused. If the new
campaign refits selection across all 20 phases, expected-count, log-ratio and
sampling-density channels must be rebuilt wherever affected, followed by their
moments and normalized interfaces. Document this choice separately from the
mandatory new model normalization.

## Final tests and observational handoff

ph018/019 remain excluded from fitting, transforms, selection and calibration.
In the corrected P12-A root they currently have geometry-only completion records;
corrected test truth and the test evaluator will eventually need a separate,
controlled preparation/evaluation contract. That is not a prerequisite to the
training-data build. Existing numerical preparation of CFM test matter does not
authorize model evaluation before freeze.

GraphWeb_DESI correctly records that old VAC outputs retain their original model
lineage. A new handoff requires checkpoint, transform, feature/coordinate,
phase-policy and calibration bindings plus golden-mock parity. The phase policy
alone does not require rebuilding DESI observational catalogues, and old DESI
predictions must not simply be renamed as the new campaign.

## Other intervening edits

The comparison contains 36 changed paths. Apart from the policy/documentation,
executable changes comprise six historical presentation scripts and the two
Graphify hook configurations. Six Python files parsed and all three changed JSON
files decoded successfully. Both hooks were exercised for command and source-read
notifications in a temporary fixture with a harmless Graphify stub: all four
responses were valid JSON, with zero exit status and no stderr. This is an output
smoke check, not a full Codex/Claude integration test. Presentation figures were
not regenerated; the source inventory appropriately calls their inputs historical.

One documentation correction is needed before manuscript reuse:
`docs/paper2_inference_framework_revised_20261010.md` still defines the P12-A
estimand at the linked host centre (equation 2), and its opening presents it as
current writing guidance. The active target is now the galaxy's own real-space
position. Preserve the old draft as historical and create a corrected revision,
or add an explicit supersession notice before reuse. The older abstract/outline
and presentation claims likewise belong to their original model lineage, not
new 20-phase validation.

## Recommended order

First implement the new policy-aware contracts and a per-product reuse manifest.
Then qualify missing physical inputs and expand the required geometry. Next
build the full/fold transforms and normalized loader interfaces, followed by the
migrated readiness checks. Only then start the registered R7 scientific fits;
P12-A exports and posterior tables follow its encoder fits. Keep the deferred
smoothing investigation deferred.

This review read code and completion metadata, not catalogue/field truth payloads.
It launched no scheduler jobs, rebuilt no data and changed no runtime source.
