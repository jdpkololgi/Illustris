"""Fixed spatial gallery and fail-closed, target-free Loa field adapter."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_observations as obs
from workflows.sbi import e2e_coupled_condition_products as products
from workflows.sbi import e2e_coupled_conditions as conditions
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi import e2e_cfm_pilot_evaluate as evaluate
from workflows.abacus_tweb import p3a_build_canonical_fields as grid_ops
from workflows.abacus_tweb import p3br_build_random_response as response_ops
from workflows.sbi.e2e_coupled_benchmark_models import FieldCondition, sample
from workflows.sbi.e2e_vdm_context_models import replicate

LOA = Path('/pscratch/sd/d/dkololgi/graphweb_desi/outputs/p12a_loa_canary_20260925_v1')
SAVED = c.ROOT.parent/'cfm_additional_20260926_v1'


def patch_counts(xyz, grid, start, shape):
    """Include the half-cell CIC halo; loss at the crop is expected, audited."""
    origin = np.asarray(grid['origin_mpc_h']) + start*grid['cell_mpc_h']
    cell = grid['cell_mpc_h']
    mask = np.all((xyz >= origin-.5*cell) &
                  (xyz < origin+(np.asarray(shape)+.5)*cell), axis=1)
    spec = grid_ops.GridSpec(tuple(origin), tuple(shape), cell, 0.)
    counts, audit = grid_ops.cic_deposit(xyz[mask], spec)
    total = float(counts.sum(dtype=np.float64))
    if abs(total-audit['deposited_weight']) > 2e-6*max(1., total):
        raise ValueError('cropped CIC conservation failed')
    if abs(audit['deposited_weight']+audit['lost_weight']-mask.sum()) > 1e-7*max(1,mask.sum()):
        raise ValueError('CIC input weight accounting failed')
    return counts, dict(**audit, grid_sum=total)


def raw_slabs(xyz, grid, start, shape, angular, boundary, selection):
    coord.validate_grid(grid)
    spec = grid_ops.GridSpec(tuple(grid['origin_mpc_h']),tuple(grid['shape']),
                            grid['cell_mpc_h'],grid['padding_mpc_h'])
    counts, audit = patch_counts(xyz, grid, start, shape)
    support = angular['support'].astype(bool) & (angular['domain']//2 == 1)
    for i in range(0, shape[0], 16):
        local = (slice(i,min(i+16,shape[0])),slice(0,shape[1]),slice(0,shape[2]))
        slices = tuple(slice(int(a+s.start),int(a+s.stop)) for a,s in zip(start,local))
        values, z = obs.response_chunk(spec,slices,support,angular['angular_response'],
                                      boundary,selection,'NGC')
        values['counts'] = counts[local]
        values['log_count_ratio_random'] = grid_ops.log_count_ratio(values['counts'],
            values['expected_counts_random'],values['exposure_apodized_random'],1e-3,1e-4)
        axes = [np.arange(s.start,s.stop) for s in slices]
        valid = ((axes[0][:,None,None]>=0)&(axes[0][:,None,None]<spec.shape[0]) &
                 (axes[1][None,:,None]>=0)&(axes[1][None,:,None]<spec.shape[1]) &
                 (axes[2][None,None,:]>=0)&(axes[2][None,None,:]<spec.shape[2]))
        for key in values:
            values[key] = np.where(valid, values[key], 0)
        values['observer_redshift'] = z # analytic, including outside cap bbox
        values['geometry_valid_fraction'] = valid.astype('f4')
        yield local, values, audit


def build_conditions(xyz, grid, midpoint, angular, boundary, selection):
    """Reproduce transform/pool ordering without constructing a survey-size cube."""
    midpoint=np.asarray(midpoint,dtype=int)
    if np.any(midpoint%8): raise ValueError('midpoint must align with coarse cells')
    local_start=midpoint-np.array([64,48,48])
    joint=np.empty((12,64,48,48),dtype='f4')
    for sl,values,local_audit in raw_slabs(xyz,grid,local_start,(128,96,96),angular,boundary,selection):
        out=slice(sl[0].start//2,sl[0].stop//2)
        for j,name in enumerate(products.LOCAL_CHANNELS):
            joint[j,out]=op.mean_pool(products.transformed(name,values[name]),2)
    wide_start=midpoint-192
    wide=np.empty((12,48,48,48),dtype='f4')
    for sl,values,wide_audit in raw_slabs(xyz,grid,wide_start,(384,384,384),angular,boundary,selection):
        out=slice(sl[0].start//8,sl[0].stop//8)
        pooled={name:op.mean_pool(values[name],4)*(64 if name in products.SUMMED else 1)
                for name in (*products.SUMMED,*products.AVERAGED,'geometry_valid_fraction')}
        pooled['log_count_ratio_random']=np.log((pooled['counts']+.5)/(pooled['expected_counts_random']+.5))
        extra=products.radial_coordinates(grid,(wide_start+np.array([sl[0].start,0,0]))//4,
                                          (4,96,96),stride=4)
        pooled.update(extra)
        for j,name in enumerate(products.WIDE_CHANNELS):
            wide[j,out]=op.mean_pool(products.transformed(name,pooled[name]),2)
    if not np.isfinite(joint).all() or not np.isfinite(wide).all():
        raise ValueError('nonfinite condition')
    return dict(joint=joint,wide=wide,support=joint[1].copy()),dict(local=local_audit,wide=wide_audit)


def compare_channels(rebuilt, original):
    rows=[]
    for region,names in [('joint',products.LOCAL_CHANNELS),('wide',products.WIDE_CHANNELS)]:
        for i,name in enumerate(names):
            a,b=rebuilt[region][i],original[region][i]
            ok=np.allclose(a,b,rtol=2e-5,atol=2e-5)
            rows.append(dict(region=region,channel=name,max_abs=float(np.max(abs(a-b))),passed=bool(ok)))
    return rows


def overlay(xyz,grid,midpoint):
    origin=np.asarray(grid['origin_mpc_h'])+(np.asarray(midpoint)-[64,48,48])*grid['cell_mpc_h']
    pos=xyz-origin
    # Same central two-cell slab in every panel, with no jitter/subsampling.
    keep=np.all((pos>=[0,0,23*6.766])&(pos<[64*6.766,48*6.766,25*6.766]),axis=1)
    return pos[keep,:2]


def render(path,title,draws,support,points,truth=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    project=lambda x:np.mean(x[:,:,23:25],axis=2)
    slabs=np.stack([project(d) for d in draws])
    panels=[] if truth is None else [('Mock truth',np.log10(project(truth)),False)]
    panels += [(f'Posterior draw {i+1}',np.log10(slabs[i]),False) for i in range(3)]
    panels += [(f'Mean of {len(draws)} draws',np.log10(slabs.mean(0)),False),
               ('Posterior SD of slab density',slabs.std(0,ddof=1),True)]
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    extent=[0,64*6.766,0,48*6.766];mask=project(support)
    for ax,(label,field,is_sd) in zip(axes.flat,panels):
        im=ax.imshow(field.T,origin='lower',extent=extent,cmap='magma' if is_sd else 'cividis',
                     vmin=0 if is_sd else -.5,vmax=1 if is_sd else .5,aspect='equal')
        ax.scatter(points[:,0],points[:,1],s=2,c='white',alpha=.6,linewidths=0,rasterized=True)
        xx=(np.arange(64)+.5)*6.766;yy=(np.arange(48)+.5)*6.766
        if np.min(mask)<.5<np.max(mask): ax.contour(xx,yy,mask.T,levels=[.5],colors='cyan',linewidths=.8)
        if np.any(mask<.5): ax.contourf(xx,yy,(mask<.5).T,levels=[.5,1.5],colors='none',hatches=['///'])
        for x in (16,32): ax.add_patch(Rectangle((x*6.766,16*6.766),16*6.766,16*6.766,
            fill=False,edgecolor='white',linestyle='--',linewidth=.8))
        ax.set(title=label,xlabel='x from patch edge [Mpc/h]',ylabel='y [Mpc/h]')
        fig.colorbar(im,ax=ax,shrink=.8,label='SD(ρ / mean ρ)' if is_sd else 'log₁₀(ρ / mean ρ)')
    for ax in list(axes.flat)[len(panels):]: ax.axis('off')
    fig.suptitle(title+'\nR=7 Mpc/h matter density; 13.532 Mpc/h XY slab; '+str(len(points))+' observed galaxies (white dots)',fontsize=13)
    fig.supxlabel('Hatching: <50% observed support in slab; dashed boxes: owned cores. Conditional samples; calibration not established.',fontsize=10)
    fig.savefig(path,dpi=160);plt.close(fig)
    return dict(galaxies=int(len(points)),draws=len(draws),slab_z_cells=[23,25],
                log_density_limits=[-.5,.5],sd_limits=[0,1],unsupported_slab_fraction=float(np.mean(mask<.5)))


def mock_gallery(output,selection):
    result=[];replay=None
    evaluate.EVAL_PHASES=('ph016','ph017')
    for phase in evaluate.EVAL_PHASES:
        receipt=c.verify_receipt(c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json')
        xyz,cap=obs.observed_points(receipt['outputs'][0]['path']);xyz=xyz[cap==1]
        geo=coord.verify_receipt(coord.ROOT/'geometry'/phase/'GEOMETRY_COMPLETE.json',payload=False)
        for kind in ('interior','boundary'):
            pair=f'{phase}_NGC_s0_{kind}_00'
            row=next(r for r in geo['pairs'] if r['pair_id']==pair)
            midpoint=np.asarray(row['center'])+[16,0,0]
            arr=conditions.load_pair(phase,pair)
            folder=SAVED/('additional'+phase[-2:])/ 'seed17_step13312_fine26624'/phase/pair/'nfe128'
            complete=json.loads((folder/'COMPLETE.json').read_text());draws=[]
            for name,sha in sorted(complete['chunks'].items()):
                if c.sha256(folder/name)!=sha: raise ValueError('saved draw changed')
                with np.load(folder/name) as f: draws.extend(f['rho'])
            target,target_sha=evaluate.truth(phase,pair)
            if len(draws)!=32: raise ValueError('incomplete saved posterior')
            points=overlay(xyz,row['grid'],midpoint)
            png=output/(pair+'.png')
            stats=render(png,f'Unseen-in-training {phase}: {kind} patch',draws,arr['support'],points,target['rho_joint'])
            np.savez_compressed(output/(pair+'_overlay.npz'),xy_mpc_h=points)
            result.append(dict(pair=pair,figure=str(png),target_sha256=target_sha,
                draw_receipt=c.file_record(folder/'COMPLETE.json',content_hash=True),**stats))
            print(json.dumps(dict(mock_figure=pair)),flush=True)
            if phase=='ph016' and kind=='interior':
                angular_receipt=c.verify_receipt(c.ROOT/'observations'/phase/'angular/ANGULAR_COMPLETE.json')
                with np.load(angular_receipt['outputs'][0]['path']) as f: angular={k:f[k] for k in f.files}
                support=angular['support'].astype(bool)&(angular['domain']//2==1)
                boundary=response_ops.angular_boundary_distance(support)
                rebuilt,audit=build_conditions(xyz,row['grid'],midpoint,angular,boundary,selection)
                checks=compare_channels(rebuilt,conditions.crop_context(arr,phase))
                replay=dict(checks=checks,cic=audit,passed=all(r['passed'] for r in checks),pair=pair)
                c.atomic_json(output/'MOCK_ADAPTER_CHECK.json',replay)
                print(json.dumps(dict(adapter_passed=replay['passed'],checks=checks)),flush=True)
    return result,replay


@torch.no_grad()
def real_draws(models,arrays,chart,first,nfe):
    t=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)[None].expand(2,*x.shape)
    base=FieldCondition(t(views.scale_channels(arrays['joint'],chart['joint'])),
        t(views.scale_channels(arrays['wide'],chart['wide'])),t(np.zeros(3,dtype='f4')),'wide')
    seeds=lambda stage:[int(c.digest(['loa-gallery-v1',17,i,stage])[:15],16) for i in (first,first+1)]
    z=sample(models['coarse'],base,'cfm',nfe//2,seeds('coarse'))
    crop=tuple(slice(*v) for v in op.layout()['joint_coarse_crop_in_wide'])
    fine=replace(base,region='joint',coarse_joint=replicate(z[(slice(None),slice(None),*crop)]),
                 coarse_wide=z,coarse_source='sampled')
    u=sample(models['fine'],fine,'cfm',nfe//2,seeds('fine'))
    rows=[];errors=[]
    for a,b in zip(z.cpu().numpy(),u.cpu().numpy()):
        wide=np.exp(a[0].astype(float)*chart['coarse_logrho']['std'][0]+chart['coarse_logrho']['mean'][0])
        rho=op.decode(wide[crop],b[0].astype(float)*chart['fine_residual']['std'][0])
        error=float(np.max(abs(op.mean_pool(rho)-wide[crop])/wide[crop]))
        if not np.isfinite(rho).all() or np.min(rho)<=0 or error>2e-6: raise ValueError('physical decode failed')
        rows.append(rho);errors.append(error)
    return np.asarray(rows),errors


def loa_gallery(output,selection,replay):
    if not replay['passed']: return dict(status='blocked',reason='mock adapter parity failed')
    ready=json.loads((LOA/'INPUTS_READY.json').read_text())
    for name,sha in ready['arrays_sha256'].items():
        if c.sha256(LOA/name)!=sha: raise ValueError('Loa input drift: '+name)
    with np.load(LOA/'catalogue.npz') as f:
        ra,dec,z=f['ra'],f['dec'],f['z'];cap=f['cap']
        xyz=coord.sky_mpc_h(ra[cap==1],dec[cap==1],z[cap==1])
    spec=grid_ops.grid_from_xyz(xyz,3.383,coord.config()['padding_mpc_h']);grid=coord.grid_record(spec)
    center=coord.sky_mpc_h(np.array(180.),np.array(10.),np.array(.22))
    midpoint=(np.rint((center-np.asarray(spec.origin))/3.383/8)*8).astype(int)
    with np.load(LOA/'random_angular.npz') as f:
        angular={k:f[k] for k in f.files}
    if 'angular_response' not in angular: angular['angular_response']=angular['response']
    with np.load(LOA/'boundary_angles.npz') as f: boundary=f['NGC']
    arrays,audit=build_conditions(xyz,grid,midpoint,angular,boundary,selection)
    fractions=[float(arrays['support'][tuple(slice(*v) for v in core)].mean())
               for core in op.layout()['owned_core_crops_in_joint']]
    if min(fractions)<.25: return dict(status='blocked',reason='insufficient owned-core support',fractions=fractions)
    np.savez_compressed(output/'loa_conditions.npz',**arrays)
    models,hashes,normalizer=evaluate.load_models(17,13312,26624);chart=views.load_chart(normalizer)
    rows=[];errors=[];chunks={}
    for first in range(0,16,2):
        tick=time.monotonic();draws,err=real_draws(models,arrays,chart,first,128)
        path=output/f'loa_draw{first:03d}.npz';np.savez_compressed(path,rho=draws)
        chunks[path.name]=c.sha256(path);rows.extend(draws);errors.extend(err)
        print(json.dumps(dict(loa_draws=first+2,seconds=time.monotonic()-tick)),flush=True)
    fine,err=real_draws(models,arrays,chart,0,256)
    np.savez_compressed(output/'loa_refinement.npz',rho=fine)
    refinement=float(np.sqrt(np.mean((fine-np.asarray(rows[:2]))**2)))
    points=overlay(xyz,grid,midpoint);np.savez_compressed(output/'loa_overlay.npz',xy_mpc_h=points)
    stats=render(output/'loa_exploratory.png','Real Loa: exploratory conditional field (no truth available)',
                 rows,arrays['support'],points)
    return dict(status='complete_exploratory',checkpoints=hashes,normalizer=normalizer,chunks=chunks,
        input_receipt=c.file_record(LOA/'INPUTS_READY.json',content_hash=True),grid=grid,
        midpoint=midpoint.tolist(),center_radec_z=[180,10,.22],core_support=fractions,cic=audit,
        max_mass_error=max(errors+err),nfe=128,refinement_nfe=256,refinement_draws=2,
        refinement_rmse=refinement,posterior_sd_rms=float(np.sqrt(np.mean(np.var(rows,axis=0,ddof=1)))),**stats)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    c.require_compute();torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    selection_path=Path(c.config()['observation']['fixed_selection_manifest'])
    selection=json.loads(selection_path.read_text())
    mocks,replay=mock_gallery(output,selection)
    c.atomic_json(output/'MOCK_GALLERY_COMPLETE.json',dict(figures=mocks))
    try: loa=loa_gallery(output,selection,replay)
    except Exception as e:
        c.atomic_json(output/'LOA_BLOCKED.json',dict(error=repr(e)));raise
    c.atomic_json(output/'COMPLETE.json',dict(mocks=mocks,adapter=replay,loa=loa,
        source=c.file_record(__file__,content_hash=True),selection=c.file_record(selection_path,content_hash=True)))
    print(json.dumps(dict(complete=True,loa_status=loa['status'])),flush=True)


if __name__=='__main__': main()
