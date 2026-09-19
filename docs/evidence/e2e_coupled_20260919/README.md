# Coupled-field preparation evidence

This archive records completed data/technical preparation, not scientific fits
or a posterior-calibration result. The registered panel is13train/2development/
6confirmation phases,1,792paired domains and11,776pair-offset interface cases.
No sealed001/006payloads or bulk fields/checkpoints are copied here.

- [CLOSEOUT.json](CLOSEOUT.json): requirement outcome, original resource caps and
  complete charged allocation ledger, qualified counts and measured proposal hash.
- [MANIFEST.json](MANIFEST.json):335receipt copies, original paths, sizes and SHA256.
- [Requirement handoff](../../e2e_coupled_preparation_closeout_20260919.md): meanings,
  failures/recoveries and limits on scientific claims.
- [Measured next proposal](../../e2e_coupled_resource_proposal_20260918.md): separate
  approval required before scientific implementation/fits.

Receipt payload total11,126,435bytes. All copied hashes, the closeout hash and
proposal binding were independently checked after archival. CLOSEOUT SHA256:
`7ba7cfe1291dc424341894f3cea180cb6da3262487d02a0c5dbea80fbc166ad3`.
MANIFEST SHA256:
`2d149e595ed89ad852685b026e77e16657c9d16d75b9a9589716cd8126c4cd51`.

The original qualified bulk products remain under
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1/cartesian_v2`;
native particle/field products and provenance also reside in the parent root.
Scratch is purgeable. The small archive documents successful qualification but
cannot reconstruct missing bulk products without the original inputs/builders.
Absolute source paths in receipt copies intentionally retain execution provenance.

Verification from the repository, with `cosmic_env` activated:

```bash
python -m unittest discover -s tests -p 'test_e2e_coupled*.py'
python -m workflows.sbi.e2e_coupled_preparation_closeout
```

The second command performs source-bound metadata closeout against existing
Scratch products, cached train moments and scheduler accounting. It requires
all owned preparation jobs terminal and original caps respected; it does not
repeat a multi-terabyte payload CRC scan. The archived independent audits and
normalized interface checks provide the original payload verification evidence.
Do not rerun `--archive` into this directory: existing evidence is protected
against replacement. Git commit verification is separate from the JSON result.
The data-only release's false scientific-authorization, posterior-calibration
and technical-closeout flags are deliberate; separate CLOSEOUT certifies only
completed preparation and proposal readiness, never scientific launch authority.
