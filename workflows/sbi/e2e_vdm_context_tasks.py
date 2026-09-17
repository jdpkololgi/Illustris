"""Predeclared posterior draw ledger and stable common-random-number addresses.

Constructed from geometry only. A draw's RNG excludes checkpoint/step count so
refinement and progression are coupled; cache identity includes both, and model.
"""
from workflows.sbi.e2e_vdm_context_data import address, spec
from workflows.sbi.e2e_vdm_context_products import pair_rows


def panels(rows):
    evaluation = sorted([r for r in rows if r['phase'] in ('ph004','ph005')],key=lambda r:r['anchor_id'])
    development = [r for r in evaluation if r['phase']=='ph004']
    if len(evaluation)!=32 or len(development)!=16:
        raise ValueError('common evaluation geometry incomplete')
    sentinels = []
    for phase_index,phase in enumerate(('ph004','ph005')):
        for cap_index,cap in enumerate(('NGC','SGC')):
            for support_index,support in enumerate(('interior','boundary')):
                shell=(phase_index*2+cap_index+support_index)%4
                sentinels.append(next(r['anchor_id'] for r in evaluation if r['phase']==phase
                    and r['cap']==cap and r['shell']==shell and r['support_stratum']==support))
    fit = [next(r['anchor_id'] for r in rows if r['small_train'] and r['phase']==phase
                and r['cap']==cap and r['shell']==shell and r['support_stratum']==support)
           for phase,cap,shell,support in [('ph000','NGC',1,'interior'),('ph002','SGC',3,'boundary')]]
    refinement=[next(r['anchor_id'] for r in development if r['cap']==cap
                    and r['shell']==2 and r['support_stratum']=='interior') for cap in ('NGC','SGC')]
    _,pairs=pair_rows(rows)
    return dict(evaluation=[r['anchor_id'] for r in evaluation],development=[r['anchor_id'] for r in development],
                sentinels=sentinels,fit=fit,refinement=refinement,pairs=pairs,
                selected_from='geometry only; no target/predictive scores')


def draw_seed(replica,domain,draw,factor,purpose,c=None):
    c=spec() if c is None else c
    return address(c['seed'],replica,domain,draw,factor,purpose,'posterior')


def task_seed(task,draw,factor):
    if factor not in ('fine','coarse'):
        raise ValueError('unknown stochastic factor')
    # Fine factors are conditionally independent across distinct owned cores;
    # sharing the same array noise at shifted coordinates would add spurious
    # fine dependence. Only coarse randomness belongs to the shared domain.
    owner=task['anchor'] if factor=='fine' else task['domain']
    purpose='joint' if task['purpose'].startswith('joint') else task['purpose']
    return draw_seed(task['replica'],owner,draw,factor,purpose)


def coarse_cache_key(arm,replica,checkpoint,domain,draw,steps,purpose):
    if arm!='D':
        raise ValueError('only D samples shared matter parents')
    # Caller must use domain, not child anchor: adjacent owners share this key.
    return f'D_seed{replica}/update_{checkpoint:06d}/{purpose}/{domain}/steps{steps}/draw_{draw:04d}'


def draw_tasks(rows,c=None):
    c=spec() if c is None else c
    p=panels(rows)
    tasks=[]
    def add(arm,replica,checkpoint,anchor,count,purpose,*,domain=None,steps=250,coarse_mode='sampled',start=0):
        row=dict(arm=arm,replica=replica,checkpoint=checkpoint,anchor=anchor,
            domain=anchor if domain is None else domain,start=start,count=count,purpose=purpose,
            steps=steps,coarse_mode=coarse_mode if arm=='D' else None)
        row['task_id']=f"{arm}_seed{replica}_{checkpoint}_{purpose}_{anchor}_{steps}_{coarse_mode}_{start}_{count}"
        tasks.append(row)
    for arm in c['arms']:
        for replica in c['replicas']:
            for checkpoint in c['checkpoint_updates']:
                final=checkpoint==c['updates']
                for anchor in p['evaluation'] if final else p['development']:
                    count=(c['draws_final_sentinel'] if anchor in p['sentinels'] else c['draws_final']) if final else c['draws_early_development']
                    add(arm,replica,checkpoint,anchor,count,'main')
                for anchor in p['fit']:
                    add(arm,replica,checkpoint,anchor,c['draws_fit'],'fit')
            for pair in p['pairs']:
                for anchor in pair['cores']:
                    add(arm,replica,c['updates'],anchor,c['draws_joint'],'joint',domain=pair['domain'])
                    if arm=='D':
                        for mode,count in [('fixed_mean',c['draws_mean_coarse']),('oracle_diagnostic',c['draws_oracle_coarse'])]:
                            add(arm,replica,c['updates'],anchor,count,'joint_'+mode,domain=pair['domain'],coarse_mode=mode)
            for anchor in p['refinement']:
                for steps in c['sample_refinement_steps']:
                    # Same 'main' noise identity as existing250-step final draws.
                    add(arm,replica,c['updates'],anchor,c['draws_refinement'],'main',steps=steps)
    if len({t['task_id'] for t in tasks})!=len(tasks):
        raise ValueError('duplicate task identities')
    fine_count=sum(t['count'] for t in tasks)
    parents=set()
    for t in tasks:
        if t['arm']=='D' and t['coarse_mode']=='sampled':
            for draw in range(t['start'],t['start']+t['count']):
                parents.add(coarse_cache_key('D',t['replica'],t['checkpoint'],t['domain'],draw,t['steps'],t['purpose']))
    if fine_count!=33280 or len(parents)!=7872:
        raise ValueError(f'preregistered draw budget drift: {fine_count} fine, {len(parents)} coarse')
    return dict(tasks=tasks,panels=p,central_draws=fine_count,coarse_draws=len(parents))
