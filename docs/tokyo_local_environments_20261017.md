# Tokyo local-environment posterior and observational selection

User correction: the current P12-A work infers local eigenvalue posteriors, not
density-field posteriors. The hand-authored many-fields illustration is retired
and was not modified or re-rendered. Its existing files are preserved.

New deliverables are under
`/global/homes/d/dkololgi/fmsu_tokyo_20261010/local_environments_selection_20261017`,
with a sibling ZIP. Three figures have4K PNG, PDF, SVG and720p previews:

- `local_environment_posterior`: the preferred figure for the current work.
- `survey_selection_training`: approved A01 option1, simulation → selection →
  observed mock, with no random-support hatching.
- `exploratory_field_reference`: earlier saved density slices with separate arrow
  endpoints and explicit identification as a separate exploratory field model.

## Local posterior

Uses the previously fixed A02 lower-class-concentration example from exposed
ph006, PARENT_NODE_ID2717758, z≈.1529663. Its512 saved joint eigenvalue draws yield
209void/303wall/0filament/0cluster at strict thresholdλ>.2, or40.8203/59.1797%.
Zero sampled classes are not a claim of exactly zero underlying probability.
One actual draw per nonempty class is shown: choose the closest draw to that
class's marginal median vector using full-posterior-SD-scaled Euclidean distance.
This deterministic representative selection uses no truth. Probabilities are
computed from all512draws, not from the two representative triplets.

The right panel uses actual observed ph006 points in a fixed ±40Mpc x/y and
±7.5Mpc depth crop about the target, same cap, no jitter/thinning:242 galaxies.
The golden ring identifies the target. These are Planck18 observed-redshift
Cartesian positions, not the c000 Mpc/h coordinates used in the separate field
figure. The full 3-D encoder context and seven FMPE conditioning features exceed
this visual crop. Only the highlighted galaxy's posterior is represented; there
is no spatially coherent environment-map sample, density reconstruction or
inferred eigenvector orientation. Threshold classes describe a local tidal
environment, not the visual appearance of the projected point distribution.

Hashes verify the cached posterior and the canonical point payload. The source
catalogue receipt and size bind the parent; selected rows alone were read for
IDs, Z, RA/Dec and ZWARN. Parent row mapping, target redshift/sky direction,
unique IDs, successful redshifts, eigenvalue order and class-count parity pass.
The complete catalogue hash is inherited, not freshly re-read.

## Observational-selection figure

Reuses the exact hash-verified A01ph002 density and718 selected galaxy positions.
The middle box describes galaxy population, brightness limit, sky coverage and
redshift success. Its steps are conceptual labels for the existing mock process;
no new thinning or fitted selection function was applied. Galaxy population/HOD
connects matter to galaxies and is distinct from the observational cuts. Source
positions are true mock real-space coordinates, with membership selected using
observed-success TARGETIDs. The display does not show redshift-space shifts.
All718 points remain, including11outside the earlier finite-random support map.
The simulated density is truth, not a reconstructed posterior field.

## Validation and reproduction

All main4K PNGs inspected for layout; arrowheads have separated destinations.
PDFs contain embedded fonts; SVGs retain editable text. Array/receipt checks,
export dimensions, package hashes and ZIP CRC pass. No inference, new posterior
draws, model training, Slurm changes, or third-party upload. Work is local.

Run in cosmic_env:
`python workflows/presentation/tokyo_local_environments_20261017.py`.
It imports the neighbouring `tokyo_many_fields_20261017.py` for the corrected
field-reference layout. The retired schematic is no longer called by either
entrypoint. Both renderer sources are included with the exports. The earlier
failed/incomplete graph refresh was not repeated for these presentation edits.
