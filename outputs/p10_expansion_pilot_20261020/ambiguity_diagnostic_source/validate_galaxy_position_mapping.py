#!/usr/bin/env python3
"""Independent native-object qualification of cut-sky galaxy coordinates.

Frozen geometry/mass tolerances; no fitted distances or per-phase refitting.
Cubic rows validate high-z centrals AND satellites. Native halos validate low-z
resolved centrals. Field-A particles validate low-z unresolved centrals. Unresolved
satellites, if found, are explicitly uncovered. Development phases only.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import fitsio
import numpy as np
from scipy.spatial import cKDTree
from galaxy_position_mapping import reconstruct_native, periodic_delta, field_indices, mapping_definition
from p12a_coordinate_sample_audit import mapped_member

ROOT=Path('/global/cfs/cdirs/desi/cosmosim/SecondGenMocks/AbacusSummit')
SIMROOT=Path('/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit')
ALLOWED=('ph000','ph002','ph003','ph004','ph005','ph006')
COLS=['RA','DEC','Z_COSMO','Z','RES','CEN','FILE_NUM','HALO_INDEX','BOX_INDEX','HALO_MASS']
TOL=.005 # Mpc/h, fixed before phase/particle checks; ~0.005 of a field cell

def stats(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    return dict(n=len(x),median=float(np.median(x)) if len(x) else None,
                p95=float(np.percentile(x,95)) if len(x) else None,max=float(np.max(x)) if len(x) else None)

def source(p):
    st=p.stat(); return dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns)

class Field:
    def __init__(self, phase, field_root=None):
        if field_root is not None:
            self.root=Path(field_root)
        elif phase=='ph000':
            self.root=Path('/pscratch/sd/d/dkololgi/AbacusSummit_densities/tweb_rank_outputs_fullgrid_v3/dens_AbacusSummit_base_c000_ph000_z0.200_ngrid2048_box2000_thr0p2/backend_optimized_ngrid_2048_rsmooth_7')
        else:
            self.root=Path(f'/pscratch/sd/d/dkololgi/abacus/p10_multiphase/{phase}/targets/tweb/backend_optimized_ngrid_2048_rsmooth_7')
        self.spans=[]; self.cache={}
        for p in sorted(self.root.glob('abacus_cactus_tweb_rank*.npz')):
            with np.load(p) as f:
                assert int(f['ngrid'])==2048 and float(f['boxsize'])==2000 and float(f['Rsmooth'])==7
                self.spans.append((int(f['x_start']),int(f['x_end']),p))
        self.spans.sort()
        if not self.spans or self.spans[0][0]!=0 or self.spans[-1][1]!=2048 or any(a[1]!=b[0] for a,b in zip(self.spans,self.spans[1:])):
            raise ValueError(f'Missing field coverage: {self.root}')
    def sample(self,pos):
        ix=field_indices(pos); out=np.full((len(pos),3),np.nan)
        for lo,hi,p in self.spans:
            m=(ix[:,0]>=lo)&(ix[:,0]<hi)
            if not m.any(): continue
            if p not in self.cache: self.cache[p]=mapped_member(p,'eig_vals')
            q=ix[m]; out[m]=self.cache[p][:,q[:,0]-lo,q[:,1],q[:,2]].T
        assert np.isfinite(out).all()
        return out
    def compare(self,a,b):
        x,y=self.sample(a),self.sample(b)
        same=np.all(field_indices(a)==field_indices(b),axis=1)
        return dict(rows=len(a),same_voxel=int(same.sum()),same_voxel_fraction=float(same.mean()),
                    max_abs_eigenvalue_difference=float(np.max(np.abs(x-y))),
                    rms_eigenvalue_difference=float(np.sqrt(np.mean((x-y)**2))),
                    changed_web_class=int(np.sum(np.sum(x>.2,axis=1)!=np.sum(y>.2,axis=1))))

def cubic_file_ranges(hdu):
    """Locate concatenated FILE_NUM blocks; BOX_INDEX is local to each block.

    Binary boundary searches avoid loading a 232-million-row column. Validate
    endpoints and 33 interior probes in every block; galaxy identity/position
    comparisons independently reject any wrong offset used by the audit.
    """
    count = hdu.get_nrows()
    def file_at(row):
        return int(hdu.read(rows=[row], columns=['FILE_NUM'])['FILE_NUM'][0])
    first, last = file_at(0), file_at(count-1)
    if (first, last) != (0,33):
        raise ValueError('Unexpected cubic FILE_NUM span')
    starts = [0]
    for target in range(1,34):
        lo, hi = starts[-1], count
        while lo < hi:
            mid = (lo+hi)//2
            if file_at(mid) < target: lo = mid+1
            else: hi = mid
        starts.append(lo)
    starts.append(count)
    for fn, (lo, hi) in enumerate(zip(starts,starts[1:])):
        if hi <= lo:
            raise ValueError('Empty cubic file block')
        probes = np.unique(np.linspace(lo,hi-1,33,dtype=np.int64))
        if not np.all(hdu.read(rows=probes,columns=['FILE_NUM'])['FILE_NUM']==fn):
            raise ValueError('Cubic FILE_NUM blocks are not contiguous')
    return starts


def read_cubic_correspondence(hdu, sample):
    starts = cubic_file_ranges(hdu)
    fn = sample['FILE_NUM'].astype(np.int64)
    bi = sample['BOX_INDEX'].astype(np.int64)
    if np.any((fn<0)|(fn>33)) or np.any(bi<0):
        raise ValueError('Invalid cubic local index')
    index = np.asarray(starts)[fn] + bi
    if np.any(index>=np.asarray(starts)[fn+1]):
        raise ValueError('BOX_INDEX outside its FILE_NUM cubic block')
    # fitsio preserves requested ordering, including duplicates; unique/inverse
    # also makes the guarantee explicit and avoids repeated reads.
    unique,inverse = np.unique(index,return_inverse=True)
    cols = ['x','y','z','FILE_NUM','HALO_MASS','cen']
    return hdu.read(rows=unique,columns=cols)[inverse], starts


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--registry',type=Path,help='Explicit versioned phase contract; reserved phases remain closed')
    ap.add_argument('--phases',nargs='+',default=list(ALLOWED))
    ap.add_argument('--samples',type=int,default=32000)
    ap.add_argument('--slabs',nargs='+',type=int,default=[0,33])
    args=ap.parse_args()
    if not(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and socket.gethostname().startswith('nid')):
        ap.error('Requires approved Slurm compute step')
    registry=None
    allowed=ALLOWED
    if args.registry:
        from p10_phase_assets import load_registry
        registry=load_registry(args.registry)
        allowed=tuple(p for p,c in registry['phases'].items() if c['truth_access'] in ('development','selection'))
    if any(p not in allowed for p in args.phases): ap.error('Development/selection phases only; reserved targets remain closed')
    if not 1000<=args.samples<=100000: ap.error('Sample count outside bounded budget')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(mapping=mapping_definition(),job_id=os.environ['SLURM_JOB_ID'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).with_name('galaxy_position_mapping.py').read_bytes()).hexdigest(),
        tolerance_mpc_h=TOL,mass_relative_tolerance=1e-5,
        sampling=dict(rows_per_phase=args.samples,method='equally spaced original rows',native_slabs=args.slabs),
        phases={},completed=False)
    if args.registry:
        report['phase_registry']={'path':str(args.registry.resolve()),'sha256':hashlib.sha256(args.registry.read_bytes()).hexdigest()}
    def save(): (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    from abacusnbody.data.read_abacus import read_asdf
    for phase in args.phases:
        started=time.monotonic(); print('START',phase,flush=True)
        name=f'AbacusSummit_base_c000_{phase}'
        cut=ROOT/f'CutSky/BGS/v0.1/z0.200/cutsky_BGS_z0.200_{name}.fits'
        cube=ROOT/f'CubicBox/BGS/v0.1/z0.200/{name}/BGS_box_{phase}.fits'
        with fitsio.FITS(cut) as f:
            ids=np.unique(np.linspace(0,f[1].get_nrows()-1,args.samples,dtype=np.int64))
            s=f[1].read(rows=ids,columns=COLS)
        pos=reconstruct_native(s['RA'],s['DEC'],s['Z_COSMO'])
        dcheck=np.arange(0,len(s),max(1,len(s)//256))
        direct=reconstruct_native(s['RA'][dcheck],s['DEC'][dcheck],s['Z_COSMO'][dcheck],direct=True)
        info=dict(sources=[source(cut),source(cube)],sample_rows=len(s),redshift_range=[float(s['Z_COSMO'].min()),float(s['Z_COSMO'].max())],
                  distance_spline_error_mpc_h=stats(np.linalg.norm(periodic_delta(pos[dcheck],direct),axis=1)),native_checks=[])
        report['phases'][phase]=info; save()
        field_root=None
        if registry and phase!='ph000':
            field_root=Path(registry['path_templates']['phase_output'].format(phase=phase))/'targets/tweb/backend_optimized_ngrid_2048_rsmooth_7'
        field=Field(phase,field_root)
        info['field_path']=str(field.root)
        info['field_sources']=[source(p) for _,_,p in field.spans]
        high=s['BOX_INDEX']>=0
        with fitsio.FITS(cube) as f:
            c,starts=read_cubic_correspondence(f[1],s[high])
        info['cubic_file_starts']=starts
        info['cubic_index_semantics']='FILE_NUM-local BOX_INDEX into concatenated cubic galaxy blocks'
        reference=np.column_stack([c[k] for k in ['x','y','z']])-1000.
        residual=np.linalg.norm(periodic_delta(pos[high],reference),axis=1)
        identity=(c['FILE_NUM']==s['FILE_NUM'][high]) & (c['HALO_MASS']==s['HALO_MASS'][high]) & (c['cen']==s['CEN'][high])
        info['cubic']={}
        for cen in [0,1]:
            m=s['CEN'][high]==cen
            info['cubic'][f'cen{cen}']=dict(rows=int(m.sum()),identity_matches=int(identity[m].sum()),
                position_error_mpc_h=stats(residual[m]),within_tolerance=int(np.sum(residual[m]<TOL)),
                field=field.compare(pos[high][m],reference[m]))
        refs=np.full_like(pos,np.nan); refs[high]=reference
        kind=np.full(len(s),'uncovered',dtype='U24'); kind[high]='cubic_galaxy'
        native_ids=np.full(len(s),-1,dtype=np.int64)
        low=s['BOX_INDEX']<0
        info['unresolved_satellites_in_sample']=int(np.sum(low&(s['RES']==0)&(s['CEN']==0)))
        for slab in args.slabs:
            base=SIMROOT/name/'halos/z0.200'
            central=low&(s['RES']==1)&(s['CEN']==1)&(s['FILE_NUM']==slab)
            unresolved=low&(s['RES']==0)&(s['CEN']==1)&(s['FILE_NUM']==slab)
            for population,mask in [('resolved_central',central),('unresolved_central',unresolved)]:
                print(phase,slab,population,int(mask.sum()),flush=True)
                if not mask.any():
                    info['native_checks'].append(dict(slab=slab,population=population,rows=0,pass_gate=False)); save(); continue
                if population=='resolved_central':
                    p=base/'halo_info'/f'halo_info_{slab:03d}.asdf'
                    cat=CompaSOHaloCatalog(p,fields=['x_L2com','N'],subsamples=False,cleaned=True,convert_units=True,verbose=False)
                    assert cat.header['SimName']==name
                    xyz=np.asarray(cat.halos['x_L2com']); mass=np.asarray(cat.halos['N'],dtype=float)*float(cat.header['ParticleMassHMsun'])/1e10
                    keep=np.flatnonzero((mass>=10)&(np.sum(periodic_delta(xyz,-1000.)**2,axis=1)<500.**2))
                    points=np.asarray(xyz[keep],dtype=float); selected_mass=mass[keep]
                    del cat,xyz,mass
                else:
                    p=base/'field_rv_A'/f'field_rv_A_{slab:03d}.asdf'
                    tab=read_asdf(p,load=['pos'],verbose=False)
                    xyz=np.asarray(tab['pos'])
                    keep=np.flatnonzero(np.sum(periodic_delta(xyz,-1000.)**2,axis=1)<500.**2)
                    points=np.asarray(xyz[keep],dtype=float)
                    del tab,xyz
                tree=cKDTree(points%2000.,boxsize=2000.)
                distance,nearest=tree.query(pos[mask]%2000.,k=2,workers=8)
                unique=(distance[:,0]<TOL)&(distance[:,1]>=TOL)
                mass_ok=np.ones(mask.sum(),dtype=bool)
                if population=='resolved_central': mass_ok=np.abs(selected_mass[nearest[:,0]]/s['HALO_MASS'][mask]-1)<1e-5
                accepted=unique&mass_ok
                rr=np.flatnonzero(mask); refs[rr[accepted]]=points[nearest[accepted,0]]
                native_ids[rr[accepted]]=keep[nearest[accepted,0]]; kind[rr[accepted]]=population
                entry=dict(slab=slab,population=population,rows=int(mask.sum()),source=source(p),
                    native_candidates=len(points),position_error_mpc_h=stats(distance[:,0]),
                    uniquely_matched=int(unique.sum()),mass_matches=int(mass_ok.sum()) if population=='resolved_central' else None,
                    accepted=int(accepted.sum()),pass_gate=bool(accepted.all()))
                if accepted.any(): entry['field']=field.compare(pos[rr[accepted]],refs[rr[accepted]])
                info['native_checks'].append(entry);save()
                del tree,points,keep,distance,nearest;gc.collect()
        groups=list(info['cubic'].values())+info['native_checks']
        info['field_gate']=all(
            v.get('field',{}).get('same_voxel_fraction',0)>=.995
            and v.get('field',{}).get('rms_eigenvalue_difference',float('inf'))<.01
            for v in groups)
        info['pass_gate']=bool(all(v['identity_matches']==v['rows'] and v['within_tolerance']==v['rows'] for v in info['cubic'].values()) and all(v['pass_gate'] for v in info['native_checks']) and info['unresolved_satellites_in_sample']==0 and info['field_gate'])
        np.savez(args.output/f'{phase}_sample.npz',original_row=ids,**{k:s[k] for k in COLS},
                 reconstructed_native=pos,independent_native=refs,reference_kind=kind,native_source_row=native_ids)
        info['elapsed_seconds']=time.monotonic()-started;save()
        print('DONE',phase,'pass',info['pass_gate'],flush=True)
        del field;gc.collect()
    report['completed']=True;report['pass_gate']=all(v['pass_gate'] for v in report['phases'].values());save()
    print('COMPLETE',args.output,report['pass_gate'],flush=True)

if __name__=='__main__': main()
