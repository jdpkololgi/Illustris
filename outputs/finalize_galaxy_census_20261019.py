from pathlib import Path
import json,hashlib
root=Path('/global/u2/d/dkololgi/TNG/Illustris');base=Path('/pscratch/sd/d/dkololgi/abacus/galaxy_position_validation_20261019')
paths=[base/'census_job60005529/COMPLETE.json',base/'census_training_extension_job60005529/COMPLETE.json']
parts=[json.loads(p.read_text()) for p in paths]
assert all(p['completed'] for p in parts)
phases={}
for part in parts:
 assert not(set(phases)&set(part['phases']));phases.update(part['phases'])
expected={'ph000',*(f'ph{i:03d}' for i in range(2,18)),*(f'ph{i:03d}' for i in range(20,25))}
assert set(phases)==expected
n=sum(a['rows'] for a in phases.values());u=sum(a['unresolved_satellites'] for a in phases.values())
bn=sum(sum(a['groups']['bright_r_lt_19p5'].values()) for a in phases.values());bu=sum(a['groups']['bright_r_lt_19p5']['RES0_CEN0'] for a in phases.values())
high=sum(a['groups']['high_branch']['RES0_CEN0'] for a in phases.values())
receipt=dict(schema_version='full-exposed-fiducial-satellite-census-v1',completed=True,job_id=60005529,
  input_receipts=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
  phases=dict(sorted(phases.items())),rows=n,unresolved_satellites=u,fraction=u/n,percent=100*u/n,
  bright_magnitude_only=dict(rows=bn,unresolved_satellites=bu,percent=100*bu/bn,selection='finite R_MAG_APP < 19.5; not the final observed selection'),
  high_branch_unresolved_satellites=high,excluded=['ph001','ph018','ph019','HOD variants','ph006_wdens alternative'],
  scope='All rows in all 22 named exposed fiducial CutSky catalogues; input metadata bound before and after streaming.',
  phase_exposure_basis=['P10 exposed ph000/ph002-ph006','e2e_coupled_contract.TRAIN includes ph020-ph024','wholefield pilot exposed evaluation ph012-ph017; ph018/ph019 remain sealed'])
evidence=root/'docs/evidence/galaxy_position_allpop_v2_20261019'
with (evidence/'full_catalogue_census.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
report=root/'docs/unresolved_satellite_census_20261019.md'
lines=['# Exact unresolved-satellite census — 19 October 2026\n\n',
 f'Across 22 exposed fiducial catalogues: **{u:,} unresolved satellites among {n:,} galaxies ({100*u/n:.6f}%)**, approximately one in {n/u:,.0f}. This is an exact full-row count, not extrapolation from the earlier 160,000-row sample.\n\n',
 f'The magnitude-only bright subset contains {bu:,} unresolved satellites among {bn:,} rows ({100*bu/bn:.6f}%). This applies only R_MAG_APP < 19.5; it is not a count after footprint, fibre assignment and successful-observation selection.\n\n',
 'ph001/ph018/ph019 remain sealed and were not opened. HOD variants and the alternative ph006_wdens file are excluded. Additional phase counts do not expand the five-phase coordinate-qualification certificate.\n\n',
 '| Phase | Full catalogue rows | Unresolved satellites | Percent | Bright-only satellites |\n| --- | ---: | ---: | ---: | ---: |\n']
for ph,a in sorted(phases.items()):lines.append(f"| {ph} | {a['rows']:,} | {a['unresolved_satellites']:,} | {100*a['fraction']:.6f}% | {a['groups']['bright_r_lt_19p5']['RES0_CEN0']:,} |\n")
lines.append(f'\nHigh-branch (BOX_INDEX >= 0) unresolved satellites: {high:,}. Full counts reconcile exactly with low/high branches and all four RES/CEN populations. Source sizes and timestamps were unchanged across each scan. Only RES/CEN/BOX_INDEX/R_MAG_APP columns were read; no tidal-field arrays or model payloads were opened.\n\n')
lines.append('Evidence: `docs/evidence/galaxy_position_allpop_v2_20261019/full_catalogue_census.json`. Both original component receipts and their source-code hashes are retained. Approved CPU allocation 60005529; no source catalogue was modified.\n')
report.write_text(''.join(lines))
registry=json.loads((root/'configs/product_lifecycle_20261019.json').read_text())
entry=f'''## 2026-10-19 — [policy/code/census/lifecycle] All HOD satellites receive own-position labels

User explicitly approved using RA/DEC/Z_COSMO for unresolved satellites. The v2
policy samples all populations at their recorded HOD galaxy positions. HAS_LABEL
and TWEB_STATUS indicate label availability; INDEP_POS separately marks an exact
source-row reference check (0 none, 1 cubic galaxy, 2 native halo central/mass,
3 field-particle central). Code 0 never masks labels. Both low-z resolved and
unresolved satellites can be unchecked. This supersedes the earlier masking
policy; no coordinate transform refit or population-wide validation claim.

All 619 real-source smoke rows now have finite labels, including all 14 unresolved
satellites. Field-voxel reads, original metadata and validation-code matches pass;
overwrite protection remains. New outputs/certificate use all-population v2;
old v1 evidence is preserved. The certificate still covers only ph000/ph002–005,
not every counted phase. Existing science selections and sealed gates remain.

Exact census: {u:,}/{n:,} galaxies across 22 exposed fiducial CutSky catalogues
are unresolved satellites ({100*u/n:.6f}%). Magnitude-only R<19.5 subset:
{bu:,}/{bn:,} ({100*bu/bn:.6f}%); not final observed membership. High-branch
unresolved satellites: {high}. ph001/ph018/ph019 and HOD/alternative variants
excluded. Full counts, source bindings and per-phase table are in
`docs/unresolved_satellite_census_20261019.md` and the v2 evidence directory.

Product registry: 300 products/groups, 875 grouped dependency links, 53 archived
in place (52 legacy targets/groups plus one documented incomplete field shard),
10 retirement-review candidates. No files moved/deleted; all 52 legacy paths
retain their original sizes/timestamps. A version suffix or failed scientific
gate is not proof of redundancy. Known archived targets are rejected by the P10
observed-truth and Abacus SBI-cache builders. Unknown dependencies still need
review; this is not an exhaustive cross-repository consumer audit.

Coupled/whole-field CFM targets derive directly from native matter density, so
this host-index bug does not automatically invalidate those models/products.
Separate old coordinate/calibration limitations remain. Replacement registration
requires v2 completion/output hashes and exact galaxy membership/order/geometry.
No production catalogue/cache/checkpoint regeneration or retraining occurred.
GraphWeb_DESI CONTEXT now carries the training-label provenance warning.

Canonical register: `configs/product_lifecycle_20261019.json`; operating procedure:
`docs/product_lifecycle_20261019.md`; pipeline details:
`docs/galaxy_position_migration_20261019.md`; evidence:
`docs/evidence/galaxy_position_allpop_v2_20261019/`.

'''
p=root/'SCIENCE_LOG.md';s=p.read_text();first,rest=s.split('\n',1);p.write_text(first+'\n\n'+entry+rest)
with (root/'docs/galaxy_position_migration_20261019.md').open('a') as f:f.write(f'\nFull census completed: {u:,}/{n:,} galaxies ({100*u/n:.6f}%) across 22 exposed fiducial catalogues. See `docs/unresolved_satellite_census_20261019.md` for per-phase counts and selection limits.\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ['phases','input_receipts']},indent=2))
