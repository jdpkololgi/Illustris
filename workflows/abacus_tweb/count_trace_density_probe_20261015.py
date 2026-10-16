"""Bounded count/true-trace diagnostic on exposed historical ph000 wedges.

No neural inference; no access to reserved phases. Keep query rows and native
R7 truth fixed while randomly thinning the already observed tracer population.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import fitsio
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from astropy.cosmology import Planck18

BASE=Path('/pscratch/sd/d/dkololgi/abacus/s2_shells')
OUT=Path('/global/u2/d/dkololgi/fmsu_tokyo_20261010/abacus_density_followup_20261015')
TAGS=['0p15_0p25','0p25_0p35','0p35_0p45','0p45_0p55']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[];sources=[];checks=[];baselines=[]
    spline_path=Path('/pscratch/sd/d/dkololgi/abacus/conditioning/ntilde_spline_v1_frozen.json')
    spline=json.loads(spline_path.read_text());h=float(Planck18.h)
    for tag in TAGS:
        pp=BASE/('shell_'+tag+'_final_points_xyz.npy')
        tp=BASE/('shell_'+tag+'_final_wedge_targets.fits')
        ap=Path('/pscratch/sd/d/dkololgi/abacus/aperture_features_v1')/('aperture_'+tag+'.npz')
        p=np.load(pp);d=fitsio.read(tp,columns=['TARGETID','RA','DEC','Z','LAMBDA1','LAMBDA2','LAMBDA3','BOX_INDEX'])
        x=np.load(ap)['X'];assert len(p)==len(d)==len(x)
        valid=(d['BOX_INDEX']>=0)
        trace=d['LAMBDA1']+d['LAMBDA2']+d['LAMBDA3']
        valid &= np.isfinite(trace)
        baselines.append(dict(shell=tag,rows=len(p),valid_truth_rows=int(valid.sum()),excluded_truth_rows=int((~valid).sum()),cached_N7_trace=float(spearmanr(x[valid,0],trace[valid]).statistic),cached_N7_lambda1=float(spearmanr(x[valid,0],d['LAMBDA1'][valid]).statistic),cached_contrast7_trace=float(spearmanr(x[valid,3],trace[valid]).statistic)))
        # Interior of the saved rectangular wedge/shell. This is a geometric
        # edge check, not a proof of survey angular completeness.
        radius=np.linalg.norm(p,axis=1)*h
        ra=np.deg2rad(d['RA']);dec=np.deg2rad(d['DEC'])
        edge=np.minimum.reduce([radius-radius.min(),radius.max()-radius,
            radius*np.cos(dec)*np.sin(ra-ra.min()),radius*np.cos(dec)*np.sin(ra.max()-ra),
            radius*np.sin(dec-dec.min()),radius*np.sin(dec.max()-dec)])
        eligible=np.flatnonzero((edge>14.)&valid)
        rng=np.random.default_rng(20261015)
        q=np.sort(rng.choice(eligible,min(10000,len(eligible)),replace=False))
        assert len(q)>100
        nt=np.interp(d['Z'][q],spline['grid_z'],spline['ntilde'])
        nt=np.maximum(nt,spline['ntilde_floor'])
        for seed in [20261015,20261016,20261017]:
            u=np.random.default_rng(seed).random(len(p))
            for frac in [1.,.3,.1]:
                if frac==1. and seed!=20261015:continue
                keep=u<frac;tree=cKDTree(p[keep])
                for R in [2.,7.,10.,14.]:
                    if frac!=1. and R not in [2.,7.]:continue
                    count=tree.query_ball_point(p[q],R/h,workers=1,return_length=True)-keep[q]
                    assert np.all(count>=0)
                    if seed==20261015 and frac==1.:
                        for j in [0,len(q)//2,len(q)-1]:
                            brute=int(np.sum(np.sum((p-p[q[j]])**2,axis=1)<=(R/h)**2)-1)
                            assert brute==count[j]
                        if R==7.:
                            cached=np.rint(np.expm1(x[q,0])).astype(int)
                            assert np.array_equal(count,cached)
                    expected=nt*frac*4*np.pi/3*(R/h)**3
                    contrast=np.log((count+1e-3)/(expected+1e-3))
                    # Show narrow-z robustness; pooled correlations can include
                    # changing radial selection and changing tracer bias.
                    sub=[]
                    for lo in np.arange(float(d['Z'].min()),float(d['Z'].max()),.02):
                        m=(d['Z'][q]>=lo)&(d['Z'][q]<lo+.02)
                        if m.sum()>=100 and np.std(count[m])>0:
                            sub.append(dict(zlo=float(lo),rows=int(m.sum()),rho=float(spearmanr(count[m],trace[q][m]).statistic)))
                    records.append(dict(shell=tag,seed=seed,fraction=frac,tracers=int(keep.sum()),query_rows=len(q),radius_hmpc=R,truth_radius_hmpc=7.,median_count=float(np.median(count)),zero_fraction=float(np.mean(count==0)),rho_trace=float(spearmanr(count,trace[q]).statistic),rho_lambda1=float(spearmanr(count,d['LAMBDA1'][q]).statistic),rho_selection_contrast_trace=float(spearmanr(contrast,trace[q]).statistic),narrow_z=sub))
        checks.append(dict(shell=tag,eligible_interior=len(eligible),query_rows=len(q),query_id_sha256=hashlib.sha256(d['TARGETID'][q].tobytes()).hexdigest(),cached_N7_exact=True,brute_force_checks=12,edge_buffer_hmpc=14.))
        for f in [pp,tp,ap]:sources.append(dict(path=str(f),bytes=f.stat().st_size,sha256=digest(f)))
        print(tag,baselines[-1],flush=True)
    result=dict(schema='abacus-count-trace-density-probe-v1',scope='Historical observed/harmonized ph000 S2 wedge, four valid shells. Native saved R7 eigenvalue trace, not P12 predictions and not R2 truth. Query galaxies fixed across thinning. Angular/radial rectangular-edge buffer; no certification of internal survey holes. Counts use each shell final selected population, not unharmonized buffer.',records=records,all_rows_cached_baselines=baselines,checks=checks,sources=sources,script_sha256=digest(Path(__file__)),spline_sha256=digest(spline_path))
    (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
