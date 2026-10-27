# Tokyo: multiple fields and one observation

Two 16:9 figures are in `/global/homes/d/dkololgi/fmsu_tokyo_20261010/many_fields_20261017`:
`illustrative_diagram` and `simulation_posterior`, each PNG (3840×2160), PDF and
SVG. Both have three fields on the left, arrows toward one galaxy set on the
right, and black slide backgrounds. PNG previews accompany the exports.

The first is explicitly a conceptual positive-intensity sketch. It is not a
simulation, a physical forward model or a posterior compatibility calculation.
The common dot pattern and three hand-authored fields illustrate non-uniqueness.

The simulation figure uses the expanded U-Net clean-endpoint exploratory model,
seed17, epoch32/step9984, sampled coarse input, NFE128. It reuses saved draws
0,1,2 from ph012_NGC_s0_interior_00; the region, central owned z-cell8 and draw
indices were fixed before reading field values. No truth field was opened for
selection or rendering. It displays R7 matter density on 6.766 h^-1 Mpc cells,
with a shared linear colour range 0–4 and no clipped density values. Bilinear
interpolation is display-only; it adds no physical smoothing or resolution.

The actual 1,447 observed-success mock galaxies are drawn at their saved
redshift-space positions, with exact TARGETIDs, in the same xy bounds and
6.766 h^-1 Mpc slab. No point jitter or visual thinning. The complete two-core
selection contains21,590 galaxies; selected slab cells all have positive support.
Axes are relative to the same fixed origin; the panels share a2:1 spatial aspect.
Density is the real-space matter target in the observer chart, while tracers are
in redshift space; tracer peaks need not align exactly with matter peaks.

All three draws condition on the SAME complete 3-D observations, wider context
and selection channels. The plotted slice is only a visual excerpt of these
inputs. Arrows communicate the many-to-one observation/inverse-problem idea;
the figure does not forward-resimulate or prove equal likelihood for the dots.
The sampled coarse field changes between draws, as it should under the saved
conditional sampler. This is not the per-galaxy P12-A posterior and not A01's
ph002 truth patch. See `e2e_wholefield_pilot_closeout_20261014.md`: joint-field
posterior calibration remains unqualified. The figure is evidence of variation
in this learned conditional model, not proof of correct uncertainty amplitude.
No tidal tensor/eigenvectors are inferred by this renderer.

Provenance binds source draw hashes, geometry, completed evaluation/binding,
original and expanded observation receipts and the exact galaxy selection.
Observation and selection payload hashes were freshly verified. Checkpoint
hash is inherited from the saved binding. Original completion did not serialize
per-draw hashes; the displayed saved draws are freshly hashed for this export.
Checks include point-to-cell assignment, full count reconciliation, unique IDs,
matching observer chart, finite positive fields, distinct draws, common limits,
PNG dimensions, embedded PDF fonts and SVG text. Main PNGs visually inspected.
Only small cached arrays were read; no model execution, FFT, new draws, sealed
phases, scheduler actions or external upload. `graphify` was unavailable in the initial
shell. A later cosmic_env graph refresh was interrupted after AST extraction;
the graph refresh is incomplete and is not part of figure validation.

Reproduce in cosmic_env with one CPU thread:
`python workflows/presentation/tokyo_many_fields_20261017.py`.

## Simpler A01 choices (not yet produced)

1. **Simulation → survey selection → observed catalogue (recommended).** A
   matter/galaxy patch on the left, a simple central observing-process box and
   the selected points on the right. Three short labels: brightness limit,
   sky coverage, redshift success. The closing sentence: apply the same observing
   process to the training mocks. Makes the training implication explicit.
2. **Same universe, two surveys.** One underlying field, two point patterns with
   different depth/coverage. A visible footprint gap and fewer distant/faint
   objects show that observations can change without changing the universe.
3. **Ideal mock versus observed mock.** Two aligned galaxy panels. Faint grey
   points show the parent; highlighted points survive selection. Omitted patches
   and a distance fade convey masking/incompleteness without technical legends.

For data-backed versions, bind actual parent/observed IDs and selection receipts;
do not disguise illustrative deletion as the DESI selection function. Redshift
space displacement may be a separate small inset rather than crowding one slide.

The previous A01 hatching represented finite survey-random support in the
observer grid: dense hatching for no supported depth cells, dotted for partial
support. It did not mark absent matter or supply an exact completeness map.
That diagnostic is unnecessary for the introductory slide and can be removed
from its eventual redesign.
