# Tokyo presentation source inventory

These files preserve the October 17–19 presentation work and its original,
frozen numerical inputs. They are historical artifacts: the subsequent
October 19 label-provenance warning and current retraining decisions in the
shared SCIENCE_LOG take precedence. Committing these scripts does not recertify
the old model or its targets. Do not silently rerun them with corrected-label
products substituted for their recorded inputs.

Generated slide exports, arrays, receipts and ZIPs remain outside these
repositories in `/global/homes/d/dkololgi/fmsu_tokyo_20261010/`. They are not
included in these commits. Existing ignore rules exclude generated scientific
arrays/figures in the repositories. No source or scientific record is ignored.

The final corrected package is `tokyo_notation_corrections_20261019.zip` under
that output root. It contains the bold lambda_g notation and the expanded
inferred-tidal-trace axis definition; earlier packages remain preserved.

## Source groups

- `workflows/presentation/tokyo_many_fields_20261017.py` and
  `docs/tokyo_many_fields_20261017.md`: exploratory saved-field reference;
  the conceptual field sketch was retired from the main talk.
- `tokyo_local_environments_20261017.py`, `tokyo_local_geometry_20261017.py`
  and `docs/tokyo_local_environments_20261017.md`: saved per-galaxy posterior,
  survey selection and explicitly illustrative geometry.
- `tokyo_ssfr_variants_20261017.py`: historical full-survey/cached-wedge
  presentation alternatives, with their distinct sample definitions retained.
- `tokyo_revision_sources_20261018.py`, `tokyo_revisions_20261018.py` and
  `docs/tokyo_revisions_20261018.md`: source audit and A–E slide bodies,
  build layers and components. The renderer includes October 19 bold lambda_g
  corrections; set TOKYO_REVISION_OUTPUT to a prepared output directory to
  preserve earlier versions. The default remains the historical October 18 path.

Script names without a directory above are under `workflows/presentation/`.
The galaxy-properties label renderer and morphology presentation sources are
in the sibling GraphWeb_DESI repository; see its matching source inventory.

## Repository cleanup verification

All six newly tracked Python files passed syntax parsing; numerical pipelines
were not rerun. The original figures had source/export/visual validation as
recorded in their receipts. Graphify queries now work through cosmic_env;
older notes reporting a missing base-shell command describe the environment
at the time, not the current installation.
