"""Read-only statistical assessment of committed frozen-model draw ensembles."""
import argparse
import json
from pathlib import Path
import numpy as np
from workflows.sbi import e2e_wide_pipeline as p,e2e_durable as durable
from workflows.sbi.e2e_wide_denoising_audit import Bands
from workflows.sbi.e2e_vdm_assessment import verify,binding,read_chunk,NAMES
from workflows.sbi.e2e_vdm_assessment_math import summarize_calibration


def ensemble(root,m,branch,update,anchor,steps,count,include_fields=False):
    folder=root/branch/f'u{update}_s{steps}'/anchor;done=json.loads((folder/'COMPLETE.json').read_text())
    if done['count']!=count:raise ValueError('unexpected ensemble size')
    values={};inputs={}
    for first in range(0,count,m['spec']['microbatch']):
        receipt=folder/f'{first:06d}.json'
        if p.sha256(receipt)!=done['chunks'][str(first)]:raise ValueError('chunk receipt changed')
        ids=list(range(first,min(count,first+m['spec']['microbatch'])))
        saved=read_chunk(folder,receipt,binding(root,m,branch,update,anchor,steps,ids))
        inputs[str(receipt.relative_to(root))]=p.sha256(receipt)
        with np.load(folder/saved['file']) as f:
            for key in f.files:
                if key!='delta' or include_fields:values.setdefault(key,[]).append(f[key])
    return {k:np.concatenate(v,axis=0) for k,v in values.items()},inputs


def paired_change(a,b,seed=1):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or len(a)<2:raise ValueError('paired arrays required')
    ids=np.random.default_rng(seed).integers(0,len(a),size=(2048,len(a)))
    mean=a.mean(0);change=(b-a).mean(0)/np.maximum(np.abs(mean),1e-30)
    boot=(b[ids]-a[ids]).mean(1)/np.maximum(np.abs(a[ids].mean(1)),1e-30)
    return dict(relative_mean_change=change.tolist(),mc95=np.quantile(boot,[.025,.975],axis=0).T.tolist(),
                caveat='paired posterior-draw Monte Carlo uncertainty only; not field-to-field uncertainty')


def summarize(values,truth,fullbox,patch,mask,s):
    bands=Bands(48,6.766,[0,.08,.16,.32,np.inf]);power=np.array(bands.compare(truth,truth)['truth_power'])
    out=dict(draws=len(values['power']),mean_power_ratio=(values['power'].mean(0)/power).tolist(),
        median_correlation=np.median(values['correlation'],axis=0).tolist(),
        mean_abs_log_power_error=float(np.abs(np.log(values['power'].mean(0)/power)).mean()),
        onepoint_draw_mean=values['onepoint'].mean(0).tolist(),
        onepoint_draw_std=values['onepoint'].std(0,ddof=1).tolist(),
        regional=summarize_calibration(values['regional'],patch['regional'],levels=s['coverage_levels']),tidal={})
    for boundary in s['boundaries']:
        out['tidal'][boundary]={}
        for reference,target in [('matched_patch',patch[boundary]),('fullbox_reference',fullbox)]:
            out['tidal'][boundary][reference]={}
            for component,name in enumerate(NAMES):
                out['tidal'][boundary][reference][name]={}
                for label,select in [('all',np.ones(len(mask),bool)),('observed',mask),('unobserved',~mask)]:
                    out['tidal'][boundary][reference][name][label]=summarize_calibration(
                        values[boundary][:,:,component],target[:,component],select,s['coverage_levels'])
    return out


def report(root):
    p.require_compute();m=verify(root);s=m['spec'];prep=json.loads((root/'PREPARED.json').read_text())
    if p.sha256(root/prep['file'])!=prep['sha256']:raise ValueError('prepared drift')
    with np.load(root/prep['file']) as f:truths={k:f[k] for k in f.files if not k.endswith('__condition')}
    records=[];refinement=[];inputs={};trends=[]
    for branch in s['branches']:
        for anchor in s['anchors']:
            prefix=anchor+'__';truth=truths[prefix+'truth'];patch={b:truths[prefix+b] for b in s['boundaries']+['regional']}
            summaries={};last=None
            for update in s['checkpoints']:
                values,hashes=ensemble(root,m,branch,update,anchor,250,128,include_fields=update==5120);inputs.update(hashes)
                summary=summarize(values,truth,truths[prefix+'fullbox'],patch,truths[prefix+'support'],s)
                records.append(dict(branch=branch,anchor=anchor,update=update,steps=250,**summary));summaries[update]=summary
                if update==5120:last=values
            e0,e1=(summaries[k]['mean_abs_log_power_error'] for k in [2048,5120])
            improvement=(e0-e1)/max(e0,1e-12)
            trends.append(dict(branch=branch,anchor=anchor,relative_power_error_improvement=improvement,
                strong_power_progress=improvement>=s['checkpoint_strong_improvement_fraction'],
                power_plateau_candidate=abs(improvement)<=s['checkpoint_plateau_fraction'],
                caveat='screen only; CRPS, calibration, solver and seed consistency required before training decision'))
            previous={k:v[:s['refinement_draws']] for k,v in last.items()}
            previous_steps=250
            for steps in s['refinement_steps']:
                current,hashes=ensemble(root,m,branch,5120,anchor,steps,s['refinement_draws'],True);inputs.update(hashes)
                power=paired_change(previous['power'],current['power'])
                width=lambda x:np.diff(np.quantile(x['periodic'],[.05,.95],axis=0),axis=0)[0].mean(0)
                wa,wb=width(previous),width(current)
                spread=np.sqrt(np.mean(np.var(current['delta'],axis=0,ddof=1)))
                refinement.append(dict(branch=branch,anchor=anchor,from_steps=previous_steps,to_steps=steps,
                    power=power,power_point_screen=bool(np.max(np.abs(power['relative_mean_change']))<=s['sampler_power_relative_tolerance']),
                    patch90_width_relative_change=((wb-wa)/np.maximum(wa,1e-30)).tolist(),
                    coupled_rms_over_posterior_rms_spread=float(np.sqrt(np.mean((current['delta']-previous['delta'])**2))/max(spread,1e-30))))
                previous=current;previous_steps=steps
            print('ASSESSED',branch,anchor,flush=True)
    folder=root/'analysis';folder.mkdir(exist_ok=True)
    durable.publish_json(folder/'RESULTS.json',dict(manifest_sha256=p.sha256(root/'MANIFEST.json'),inputs=inputs,
        records=records,checkpoint_trends=trends,sampler_refinement=refinement,oracle_boundary=prep['references'],
        sample_total=3584,independent_heldout_phases=1,heldout_patches=2,full_field_sbc=False,
        no_automatic_training=True,production_ready=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);report(parser.parse_args().root)
