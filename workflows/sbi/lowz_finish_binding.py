"""Bind the completed development method. Does not grant final-phase access."""
import hashlib
import json
from pathlib import Path
import lowz_operator_closure as o


def run():
    c=o.c;c.require_compute()
    repo=Path(__file__).resolve().parents[2]
    root=c.ROOT.parent
    design=repo/'configs/lowz_cfm_confirmation_design.json'
    spec=o.read(design)
    assert tuple(spec['development'])==tuple(c.DEVELOPMENT)==o.DEV
    assert spec['status']=='design_only_not_phase_access_authority'
    assert list(c.CONFIRMATION)==spec['confirmation_after_separate_freeze']
    def bind(p,expected=None):
        rec=c.file_record(p,content_hash=True)
        if expected is not None and rec['sha256']!=expected:raise ValueError(str(p))
        return rec
    closure=root/'lowz_dev_oracle_density_v1/CLOSURE_READOUT.json'
    card=root/'lowz_finish_scorecard_61848915/SCORECARD.json'
    b=root/'lowz_count_reference_61848915'
    parts=[closure,card,b/'REFERENCE_COMPLETE.json',b/'MATCHED_PERCENTILE90.json',root/'lowz_dev_density_shapes_61848915/SHAPES.json']
    assert all(o.read(p)['passed'] for p in parts)
    assert o.read(closure)['card']==c.sha256(card) and o.read(closure)['summary'][-1]['cells']==344302
    fit=o.read(b/'FIT.json');assert fit['phases']==list(c.TRAIN)
    helpers=Path(o.pilot.__file__).parent
    checks={}
    for line in (helpers/'SHA256SUMS').read_text().splitlines():
        sha,filename=line.split(); p=helpers/filename.lstrip('*')
        checks[p.name]=bind(p,sha)
    registry=dict(design=bind(design),coarse=bind(root/'lowz_coarse_velocity_seed17_v1/LATEST.pt',spec['coarse_ema_sha256']),
        fine=bind(root/'lowz_unet_endpoint_20261118_seed17/LATEST.pt',spec['fine_ema_sha256']),
        normalizer=bind(o.pilot.coord.ROOT/'normalization/NORMALIZATION_COMPLETE.json',spec['normalizer_receipt_sha256']),
        prior=bind(o.pilot.prior/'PRIOR.json',spec['exterior']['prior_sha256']),
        prior_freeze=bind(o.pilot.prior/'FREEZE.json'),baseline=bind(b/'TRAIN_ONLY.npz',fit['library']['sha256']),
        baseline_recipe=bind(b/'FIT.json'),development_results=[bind(p) for p in parts],
        training_campaign=bind(c.CONFIG),source_archive_receipt=bind(c.REPO/'SOURCE.txt'),sampler_helpers=checks,
        methods={f:bind(repo/'workflows/sbi'/f) for f in ['lowz_ensemble_scores.py','lowz_finish_replay.py','lowz_operator_closure.py',
                 'lowz_closure_readout.py','lowz_count_reference.py','lowz_dev_scorecard.py','lowz_density_shapes.py','lowz_percentile_comparison.py','lowz_finish_binding.py']})
    # Bind the actually deployed Python modules/configs, not just the source tar's name.
    paths=sorted(p for d in ('workflows','shared','configs') for p in (c.REPO/d).rglob('*') if p.suffix in ('.py','.json'))
    registry['source_files']={str(p.relative_to(c.REPO)):c.sha256(p) for p in paths}
    assert paths
    out=root/'lowz_mock_dev_method_binding_v1'
    out.mkdir(exist_ok=False)
    c.atomic_json(out/'METHOD_FROZEN.json',dict(artifacts=registry,science_scope='unaltered seed-17 CFM, mock evaluation only',
        holdout_authorized=False,holdout_inputs_verified=False,desi_authorized=False,posterior_corrected=False,
        limitations=['single training seed and snapshot','3 development phases','one oracle exterior seed per phase/cap','regional shear compression'],
        passed=True))


if __name__=='__main__':run()
