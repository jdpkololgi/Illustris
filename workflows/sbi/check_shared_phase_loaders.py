"""Data-only canaries for each prepared training/development phase; no model fit."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi import e2e_wholefield_pilot_data as data
from workflows.sbi import e2e_wholefield_pilot_train as loader
from workflows.sbi.run_shared_phase_finalize import wait_for


def check(seconds):
    c.require_compute()
    if not c.SHARED_POLICY or len(c.TRAIN)!=20:raise ValueError('Explicit 20-phase campaign required')
    deadline=time.monotonic()+seconds
    phases=list(c.TRAIN+c.DEVELOPMENT)
    normalizer=coord.ROOT/'normalization/NORMALIZATION_COMPLETE.json'
    wait_for([normalizer]+[data.ROOT/'data'/p/'COMPLETE.json' for p in phases],deadline)
    digest=c.sha256(normalizer);chart=views.load_chart(digest);checks=[]
    for phase in phases:
        proof=json.loads((data.ROOT/'data'/phase/'COMPLETE.json').read_text())
        if not proof['passed'] or proof['normalizer']!=digest:raise ValueError('Whole-field normalization binding mismatch')
        pair=proof['pairs'][0]['pair_id']
        observation=views.load_observations(phase,pair,digest)
        if not all(np.isfinite(a).all() for a in observation.values()):raise ValueError('Nonfinite observation view')
        if phase in c.TRAIN:
            _,target=views.load_training_pair(phase,pair,digest)
            decoded=views.decode_view(target['coarse_logrho'],target['fine_residual'],phase,(0,0,0),chart)
            if not np.isfinite(decoded).all() or decoded.min()<=0:raise ValueError('Invalid decoded physical target')
        arrays=loader.arrays((phase,pair),chart,training=phase in c.TRAIN)
        if not all(np.isfinite(a).all() for a in arrays):raise ValueError('Nonfinite whole-field view')
        shapes=[a.shape for a in arrays]
        if shapes!=[(1,*data.SHAPE),(12,*data.SHAPE),(12,48,48,48),(1,48,48,48)]:raise ValueError('Whole-field loader shape mismatch')
        checks.append(dict(phase=phase,pair=pair,wholefield_shapes=shapes,passed=True))
    for phase in c.CONFIRMATION:
        try:loader.arrays((phase,'closed'),{},training=True)
        except PermissionError:pass
        else:raise AssertionError('Final-test target access was not rejected')
    ids=loader.panel(c.TRAIN)
    epoch=[item for step in range(loader.EPOCH_STEPS) for item in loader.selected(ids,17,step)]
    if len(ids)!=960 or sorted(epoch)!=sorted(ids) or len(set(epoch))!=len(ids):raise ValueError('Whole-field epoch coverage failure')
    record=dict(**c.provenance(),normalizer=c.file_record(normalizer,content_hash=True),checks=checks,
                training_domains=len(ids),epoch_steps=loader.EPOCH_STEPS,final_test_payloads_opened=False,
                learned_models_fitted=False,source_sha256=c.sha256(__file__),loader_source_sha256=c.sha256(loader.__file__),
                outputs=[],**{'pass':True})
    c.atomic_json(c.ROOT/'readiness/DATA_LOADER_CANARIES.json',record)
    print(json.dumps(dict(data_loader_canaries_pass=True,phases=len(checks),training_domains=len(ids))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seconds',type=float,default=3600);a=p.parse_args()
    if not 0<a.seconds<=4800:raise ValueError('Explicit bounded wait required')
    check(a.seconds)
