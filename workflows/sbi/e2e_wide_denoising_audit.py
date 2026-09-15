"""Training-panel localization of denoising errors; no fitting or output correction."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

import h5py
import numpy as np
import torch

from workflows.sbi import e2e_wide_pipeline as p
from workflows.sbi.e2e_wide_continue import PARENT, checked_binding
from workflows.sbi.e2e_wide_research_canary import preflight, NORMALIZATION

ROOT=p.SCRATCH_ROOT/'wide_pipeline_v1'
TRAIN384=ROOT/'continue_20260914_58309454'
EVALS={192:ROOT/'eval_20260914_58305867',384:ROOT/'eval384_20260914_58309454'}
RATIOS=(.05,.2,1.,5.,20.)


def bridge(method, ratio):
    """Same sigma/alpha, not falsely identical time conventions across objectives."""
    if ratio<=0:
        raise ValueError('positive noise/signal ratio required')
    if method=='cfm':
        t=1/(1+ratio)
        return t,t,1-t
    if method=='diffusion':
        t=2*math.atan(ratio)/math.pi
        return t,math.cos(t*math.pi/2),math.sin(t*math.pi/2)
    raise ValueError('unknown method')


def clean_estimate(method,state,velocity,t):
    if method=='cfm':
        return state+(1-t)*velocity
    if method=='diffusion':
        return math.cos(t*math.pi/2)*state-math.sin(t*math.pi/2)*velocity
    raise ValueError('unknown method')


class Bands:
    def __init__(self,n,cell,edges):
        w=np.hanning(n)
        self.window=w[:,None,None]*w[None,:,None]*w[None,None,:]
        self.sum_window=self.window.sum()
        self.norm=float((self.window**2).sum()*n**3)
        k=np.fft.fftfreq(n,d=cell)*2*np.pi
        kz=np.fft.rfftfreq(n,d=cell)*2*np.pi
        radius=np.sqrt(k[:,None,None]**2+k[None,:,None]**2+kz[None,None,:]**2)
        self.masks=[(radius>lo)&(radius<=hi) for lo,hi in zip(edges[:-1],edges[1:])]
        self.weights=np.full(len(kz),2.);self.weights[0]=1
        if n%2==0:
            self.weights[-1]=1

    def fft(self,x):
        x=np.asarray(x,dtype=np.float64)
        return np.fft.rfftn((x-np.sum(x*self.window)/self.sum_window)*self.window)

    def cross(self,a,b):
        v=np.real(a*np.conj(b))*self.weights/self.norm
        return np.array([v[m].sum() for m in self.masks])

    def compare(self,pred,truth,noise=None,velocity_error=None):
        fp,ft=self.fft(pred),self.fft(truth)
        pp,pt=self.cross(fp,fp),self.cross(ft,ft)
        cross=self.cross(fp,ft);err=self.cross(fp-ft,fp-ft)
        out={'truth_power':pt.tolist(),'prediction_power':pp.tolist(),'error_power':err.tolist(),
             'power_ratio':(pp/np.maximum(pt,1e-30)).tolist(),
             'gain':(cross/np.maximum(pt,1e-30)).tolist(),
             'correlation':(cross/np.sqrt(np.maximum(pp*pt,1e-60))).tolist(),
             'error_over_truth_power':(err/np.maximum(pt,1e-30)).tolist(),
             'mean_error':float(np.mean(pred)-np.mean(truth))}
        if noise is not None:
            fn=self.fft(noise);pn=self.cross(fn,fn)
            leakage=self.cross(fp-ft,fn)
            out['noise_leakage_coefficient']=(leakage/np.maximum(pn,1e-30)).tolist()
            out['noise_correlated_error_power']=(leakage**2/np.maximum(pn,1e-30)).tolist()
        if velocity_error is not None:
            fv=self.fft(velocity_error);v=self.cross(fv,fv)
            out['velocity_error_power']=v.tolist()
            out['velocity_error_band_fraction']=(v/np.maximum(v.sum(),1e-30)).tolist()
        return out

    def decomposition(self,coarse,fine):
        fc,ff=self.fft(coarse),self.fft(fine)
        pc,pf,cross=self.cross(fc,fc),self.cross(ff,ff),2*self.cross(fc,ff)
        total=self.cross(fc+ff,fc+ff)
        if not np.allclose(total,pc+pf+cross,rtol=1e-10,atol=1e-10):
            raise ValueError('component power does not close')
        return {k:v.tolist() for k,v in {'coarse':pc,'fine':pf,'cross':cross,'total':total}.items()}


def medians(rows,keys):
    return {k:np.median([r[k] for r in rows],axis=0).tolist() for k in keys}


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    device=p.runtime();c,_,_,_=preflight();ds=p.dataset_for(c,NORMALIZATION)
    base=p.provenance(c,ds);extended=checked_binding(TRAIN384,base)
    previous=json.loads((EVALS[384]/'EVALUATION_COMPLETE.json').read_text())
    anchors=previous['registration']['refinement_anchors']
    selected=[r for r in ds.rows if r['anchor_id'] in anchors]
    if len(anchors)!=24 or len(set(anchors))!=24 or len(selected)!=24:
        raise ValueError('expected unchanged 24-anchor panel')
    out=p.output_path(c,args.output);out.mkdir(parents=True,exist_ok=False)
    bands={'coarse':Bands(96,13.532,[0,.04,.08,.16,np.inf]),
           'fine':Bands(96,3.383,[0,.08,.16,.32,np.inf])}
    models={};checkpoints={}
    for step in (192,384):
        for method in ('cfm','diffusion'):
            for stage in ('coarse','fine'):
                path=(PARENT if step==192 else TRAIN384)/f'{method}_{stage}'/f'step_{step:06d}.pt'
                state=p.load_checkpoint(path,base if step==192 else extended,stage,method)
                model=p.build_model(c,stage,device).eval();model.load_state_dict(state['model'])
                models[step,method,stage]=model;checkpoints[f'{step}/{method}/{stage}']=p.sha256(path)
    registration={'job_id':os.environ['SLURM_JOB_ID'],'node':socket.gethostname(),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.REPO,text=True).strip(),
        'source_sha256':p.sha256(__file__),'base_binding_sha256':p.digest(base),'continuation_binding_sha256':p.digest(extended),
        'checkpoint_sha256':checkpoints,'anchors':anchors,'noise_signal_ratios':list(RATIOS),'noise_replicates':2,
        'evaluation_receipt_sha256':{str(step):p.sha256(root/'EVALUATION_COMPLETE.json') for step,root in EVALS.items()},
        'coarse_band_edges':[0,.04,.08,.16,'inf'],'fine_band_edges':[0,.08,.16,.32,'inf'],
        'claim':'training-only denoising localization; true-coarse generation is an oracle diagnostic, not inference',
        'heldout_payloads_read':False,'training_performed':False}
    p.write_json(out/'AUDIT_STARTED.json',registration);start=time.monotonic()
    records=[];components=[];oracle_rows=[];trajectories=[]
    completed=0
    for index,row in enumerate(ds.rows):
        if row['anchor_id'] not in anchors:
            continue
        item=ds[index]
        true_coarse=ds.inverse_target(item['coarse_target'][0],'coarse')
        true_fine=ds.inverse_target(item['fine_target'][0],'fine')
        up_true=p.coarse_to_fine(true_coarse,fine_side=96,factor=4)
        with h5py.File(row['shard'],'r') as f:
            truth=f[row['group']]['delta_r7_gaussian'][:]
        if not np.allclose(up_true+true_fine,truth,atol=2e-5,rtol=2e-5):
            raise ValueError('physical truth component sum mismatch')
        true_parts=bands['fine'].decomposition(up_true,true_fine)
        for stage in ('coarse','fine'):
            target=p.tensor(item[stage+'_target'],device)
            condition=p.tensor(item[stage+'_condition'],device)
            wide=p.tensor(item['coarse_condition'],device) if stage=='fine' else None
            target_np=target[0,0].cpu().numpy()
            scale=ds.normalization['targets'][stage]['std']
            for rep in range(2):
                seed=p.seed_for(915,row['anchor_id'],rep,stage)
                noise=torch.randn(target.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed))
                noise_np=noise[0,0].cpu().numpy()
                for method in ('cfm','diffusion'):
                    for ratio in RATIOS:
                        t,alpha,sigma=bridge(method,ratio)
                        noisy=alpha*target+sigma*noise
                        velocity=target-noise if method=='cfm' else alpha*noise-sigma*target
                        for step in (192,384):
                            pred=models[step,method,stage](noisy,target.new_tensor([t]),condition,wide_condition=wide)
                            clean=clean_estimate(method,noisy,pred,t)
                            error=clean-target
                            expected=(sigma if method=='cfm' else -sigma)*(pred-velocity)
                            if not torch.allclose(error,expected,atol=3e-5,rtol=3e-5):
                                raise ValueError('velocity-to-clean error identity failed')
                            metric=bands[stage].compare(clean[0,0].cpu().numpy()*scale,target_np*scale,
                                noise_np*scale,(pred-velocity)[0,0].cpu().numpy()*scale)
                            metric['noise_leakage_over_input_sigma']=(np.asarray(metric['noise_leakage_coefficient'])/sigma).tolist()
                            record={'anchor_id':row['anchor_id'],'phase':row['phase'],'method':method,'stage':stage,
                                'step':step,'ratio':ratio,'time':t,'alpha':alpha,'sigma':sigma,'replicate':rep,'seed':seed,
                                'velocity_mse':float(torch.mean((pred-velocity)**2)),
                                'clean_mse':float(torch.mean(error**2)),'metrics':metric}
                            records.append(record)
        # Decompose the SAME saved ancestral draws, including signed cross-power.
        for step in (192,384):
            for method in ('cfm','diffusion'):
                path=EVALS[step]/f'{row["anchor_id"]}_{method}.h5'
                receipt=json.loads(path.with_suffix('.json').read_text())
                if p.sha256(path)!=receipt['sample_sha256']:
                    raise ValueError('saved draw checksum changed')
                with h5py.File(path,'r') as f:
                    for draw in range(4):
                        co=f[f'{draw}/coarse_delta'][:];fi=f[f'{draw}/fine_residual'][:]
                        parts=bands['fine'].decomposition(p.coarse_to_fine(co,fine_side=96,factor=4),fi)
                        components.append({'anchor_id':row['anchor_id'],'phase':row['phase'],'method':method,
                            'step':step,'draw':draw,'truth':true_parts,'prediction':parts})
        # Fine-only generation with TRUE coarse, explicitly not a deployable draw.
        for method in ('cfm','diffusion'):
            seed=p.seed_for(c['sampling']['seed'],row['anchor_id'],'eval-0','fine')
            fine_z=p.sample_field(models[384,method,'fine'],p.tensor(item['fine_condition'],device),
                method=method,steps=c['sampling']['cfm_steps' if method=='cfm' else 'diffusion_steps'],
                generator=torch.Generator(device=device).manual_seed(seed),solver='heun',
                wide_condition=p.tensor(item['coarse_condition'],device))
            fine=ds.inverse_target(fine_z[0,0].cpu().numpy(),'fine');delta=up_true+fine
            path=out/f'{row["anchor_id"]}_{method}_true_coarse_diagnostic.h5'
            with h5py.File(path,'x') as f:
                f.attrs['oracle_diagnostic_only']=True;f.attrs['fine_seed']=seed
                f.create_dataset('fine_residual',data=fine);f.create_dataset('delta_local96',data=delta)
            oracle_rows.append({'anchor_id':row['anchor_id'],'phase':row['phase'],'method':method,
                'sample_sha256':p.sha256(path),'metrics':bands['fine'].compare(delta,truth),
                'components':bands['fine'].decomposition(up_true,fine),
                'density_below_minus_one_fraction':float(np.mean(delta < -1))})
        # Six preselected extreme-shell anchors: inspect on-sampler clean predictions.
        if row['support_stratum']=='boundary' and int(str(row['shell']).lstrip('s')) in (0,3):
            obs=ds.inference_conditions(index)
            for method in ('cfm','diffusion'):
                handles=[]
                for stage in ('coarse','fine'):
                    counter=[0]
                    def hook(model,inputs,pred,stage=stage,counter=counter):
                        call=counter[0];counter[0]+=1
                        if call%4!=0 and call!=31:
                            return
                        state,t=inputs[0],float(inputs[1][0])
                        clean=clean_estimate(method,state,pred,t)[0,0].cpu().numpy()
                        physical=ds.inverse_target(clean,stage)
                        f=bands[stage].fft(physical);power=bands[stage].cross(f,f)
                        trajectories.append({'anchor_id':row['anchor_id'],'method':method,'stage':stage,
                            'call':call,'time':t,'predicted_clean_power':power.tolist(),
                            'predicted_clean_mean':float(physical.mean()),
                            'note':'CFM even calls are accepted-state starts; odd last call is Heun proposal'})
                    handles.append(models[384,method,stage].register_forward_hook(hook))
                try:
                    co,fi,_=p.generate_pair(c,ds,obs,models[384,method,'coarse'],models[384,method,'fine'],method,'eval-0',device)
                    with h5py.File(EVALS[384]/f'{row["anchor_id"]}_{method}.h5','r') as f:
                        if not np.array_equal(co,f['0/coarse_delta'][:]) or not np.array_equal(fi,f['0/fine_residual'][:]):
                            raise ValueError('instrumented sampler changed saved realization')
                finally:
                    for handle in handles:
                        handle.remove()
        completed+=1
        p.write_json(out/f'anchor_{completed:02d}.json',{'anchor_id':row['anchor_id'],'done':True})
        print(f'DENOISING AUDIT {completed}/24 {row["anchor_id"]}',flush=True)
    summaries={}
    keys=('truth_power','prediction_power','error_power','power_ratio','gain','correlation',
          'error_over_truth_power','noise_leakage_over_input_sigma','velocity_error_band_fraction')
    for step in (192,384):
        for method in ('cfm','diffusion'):
            for stage in ('coarse','fine'):
                for ratio in RATIOS:
                    chosen=[r for r in records if (r['step'],r['method'],r['stage'],r['ratio'])==(step,method,stage,ratio)]
                    summaries[f'{step}/{method}/{stage}/{ratio}']=medians([r['metrics'] for r in chosen],keys)
    preflight()
    if (completed,len(records),len(components),len(oracle_rows),len(trajectories))!=(24,1920,384,48,216):
        raise ValueError('diagnostic panel incomplete')
    if checked_binding(TRAIN384,p.provenance(c,ds))!=extended or p.sha256(__file__)!=registration['source_sha256']:
        raise ValueError('audit provenance changed')
    result={'registration':registration,'elapsed_seconds':time.monotonic()-start,'probes':records,
        'probe_summary':summaries,'components':components,'true_coarse_diagnostics':oracle_rows,
        'sampler_trajectories':trajectories,'complete':True,'training_ready':False,'calibration_pass':None}
    p.write_json(out/'DENOISING_AUDIT_COMPLETE.json',result)
    print(f'DENOISING AUDIT COMPLETE: {len(records)} controlled probes, {len(components)} component draws, {len(oracle_rows)} oracle controls',flush=True)


if __name__=='__main__':
    main()
