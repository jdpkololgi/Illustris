#!/usr/bin/env python3
"""Render saved ph002 tidal truth and actual selected mock galaxies; no field generation.

Bounded single-thread login execution explicitly authorized by the user on 2026-10-12.
Geometry is chosen before inspecting the field: first NGC, lowest-z, interior pair;
join its two adjacent cores and take saved z-cell 8. No truth/appearance selection.
"""
import argparse, datetime, hashlib, json, os, platform, socket, subprocess, sys
from pathlib import Path
import numpy as np
import fitsio, h5py
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Rectangle
from PIL import Image

REPO=Path(__file__).resolve().parents[2]
BASE=Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20260918_v1/cartesian_v2')
CAT=Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase/ph002/catalogues')
PAIR='ph002_NGC_s0_interior_00'
COLORS=['#9CF6D5','#5B8FF9','#E85D9E','#F2C14E']
COMPONENTS=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def bound(record):
    p=Path(record['path']);assert p.stat().st_size==record['bytes'],p
    assert sha(p)==record['sha256'],p
    return p

def extract(out):
    gp=BASE/'geometry/ph002/GEOMETRY_COMPLETE.json'
    rows=read(gp)['pairs']
    row=sorted([r for r in rows if r['cap']=='NGC' and r['shell']==0 and r['support_stratum']=='interior'],key=lambda r:r['pair_id'])[0]
    assert row['pair_id']==PAIR and row['phase']=='ph002'
    tp=BASE/f'targets/ph002/{PAIR}.json';t=read(tp)
    assert t['pass'] and t['binding']['geometry_receipt_sha256']==sha(gp)
    target=bound(t['outputs'][0])
    cfgp=REPO/'configs/e2e_coupled_coordinates_v2.json';cfg=read(cfgp)
    assert sha(cfgp)==t['coordinate_sha256']==row['grid']['coordinate_sha256']
    tablep=REPO/cfg['table_path'];assert sha(tablep)==cfg['table_sha256']
    dist=np.loadtxt(tablep);radius=CubicSpline(dist[:,0],dist[:,2],extrapolate=False)
    raw=row['grid']['cell_mpc_h'];cell=2*raw;start=np.array(row['center'])-16
    lower=np.array(row['grid']['origin_mpc_h'])+start*raw
    zindex=8;zlo=lower[2]+zindex*cell;zhi=zlo+cell
    # One z-cell of the saved cell-averaged 3D tensor, not a projected tensor average.
    with h5py.File(target) as f:
        cores=f['fullbox_tensor_cores'][:]
        rho=f['rho_joint'][16:48,16:32,16+zindex]
    assert cores.shape==(2,16,16,16,6)
    six=np.concatenate([cores[0,:,:,zindex],cores[1,:,:,zindex]],axis=0)
    matrix=np.zeros((32,16,3,3))
    for k,(a,b) in enumerate(COMPONENTS):matrix[...,a,b]=matrix[...,b,a]=six[...,k]
    val,vec=np.linalg.eigh(matrix)
    trace_err=float(np.max(np.abs(np.trace(matrix,axis1=-2,axis2=-1)-(rho-1))))
    residual=float(np.max(np.abs(matrix@vec-vec*val[...,None,:])))
    orth_err=float(np.max(np.abs(np.swapaxes(vec,-1,-2)@vec-np.eye(3))))
    assert trace_err<2e-6 and residual<1e-12 and orth_err<1e-12
    assert np.isfinite(val).all() and (np.diff(val,axis=-1)>=0).all()
    classes=(val>.2).sum(-1).astype('u1')
    pp=CAT/'bright_parent/ph002_bgs_bright_parent_linkage.fits.complete.json'
    op=CAT/'observed/ph002_bgs_bright_full_observed_with_tweb.fits.complete.json'
    parent=bound(read(pp)['output']); observed=bound(read(op)['output'])
    print('Verified saved tensor and both catalogue hashes.',flush=True)
    # Scan small fixed chunks, retaining only the predeclared real-space slab.
    candidates=[]; positions=[]
    with fitsio.FITS(str(parent)) as f:
        for first in range(0,f[1].get_nrows(),200000):
            a=f[1].read(rows=range(first,min(first+200000,f[1].get_nrows())),columns=['TARGETID','RA','DEC','Z_COSMO','BOX_INDEX'])
            ra=np.deg2rad(a['RA']);dec=np.deg2rad(a['DEC']);r=radius(a['Z_COSMO'])
            xyz=np.column_stack([np.cos(ra)*np.cos(dec),np.sin(ra)*np.cos(dec),np.sin(dec)])*r[:,None]
            m=(np.isfinite(xyz).all(1)&(xyz[:,0]>=lower[0])&(xyz[:,0]<lower[0]+32*cell)&
               (xyz[:,1]>=lower[1])&(xyz[:,1]<lower[1]+16*cell)&(xyz[:,2]>=zlo)&(xyz[:,2]<zhi))
            if m.any():candidates.append(a[m]);positions.append(xyz[m])
    if not candidates:raise ValueError('Predeclared slab has no parent galaxies; do not select another by appearance')
    a=np.concatenate(candidates);xyz=np.concatenate(positions);ids=a['TARGETID']
    sorter=np.argsort(ids);a=a[sorter];xyz=xyz[sorter];ids=ids[sorter]
    assert len(np.unique(ids))==len(ids)
    selected=[]; obsrows=[]
    with fitsio.FITS(str(observed)) as f:
        for first in range(0,f[1].get_nrows(),200000):
            b=f[1].read(rows=range(first,min(first+200000,f[1].get_nrows())),columns=['TARGETID','RA','DEC','Z','ZWARN'])
            i=np.searchsorted(ids,b['TARGETID']);valid=i<len(ids)
            valid &= ids[np.minimum(i,len(ids)-1)]==b['TARGETID']
            valid &= (b['Z']>=.15)&(b['Z']<.55)&(b['ZWARN']==0)
            if valid.any():selected.extend(i[valid]);obsrows.append(b[valid])
    if not selected:raise ValueError('No observed-success galaxies in the frozen slab')
    selected=np.array(selected);b=np.concatenate(obsrows);a=a[selected];xyz=xyz[selected]
    assert np.array_equal(a['TARGETID'],b['TARGETID'])
    assert np.max(np.abs(a['RA']-b['RA']))<1e-10 and np.max(np.abs(a['DEC']-b['DEC']))<1e-10
    assert len(np.unique(a['TARGETID']))==len(a)
    # Read only the small matching random-support hyperslab; never load the full field.
    sp=BASE/'observations/ph002/NGC_COMPLETE.json';sr=read(sp);srec=sr['outputs'][0]
    assert sr['grid']==row['grid'] and sr['pass'] and Path(srec['path']).stat().st_size==srec['bytes']
    # Large response file uses receipt size/mtime binding, not a fresh full-file hash.
    assert Path(srec['path']).stat().st_mtime_ns==srec['mtime_ns']
    sl=(slice(start[0],start[0]+64),slice(start[1],start[1]+32),slice(start[2]+2*zindex,start[2]+2*zindex+2))
    with h5py.File(srec['path']) as f: support_raw=f['support_random'][sl]
    support=support_raw.reshape(32,2,16,2,2).mean((1,3,4))
    ij=np.floor((xyz-np.array([lower[0],lower[1],zlo]))/raw).astype(int)
    supported=support_raw[ij[:,0],ij[:,1],ij[:,2]].astype(bool)
    np.savez_compressed(out/'data/tidal_overlay.npz',tensor=matrix,eigenvalues=val,eigenvectors=vec,
        classes=classes,xy=xyz[:,:2],xyz=xyz,targetid=a['TARGETID'],z_observed=b['Z'],
        z_cosmo=a['Z_COSMO'],box_index=a['BOX_INDEX'],support=support,supported=supported,
        lower=lower,cell=cell,zlo=zlo,zhi=zhi)
    meta={'phase':'ph002','simulation':'AbacusSummit_base_c000_ph002','epoch_z':.2,'pair_id':PAIR,
        'geometry_selection':'lexicographically first NGC shell-0 interior pair; both adjacent cores; z cell 8; fixed before data inspection',
        'geometry':row,'smoothing_mpc_h':7.,'cell_mpc_h':cell,'slice_thickness_mpc_h':cell,
        'slice_z_bounds_mpc_h':[zlo,zhi],'xy_bounds_mpc_h':[lower[0],lower[0]+32*cell,lower[1],lower[1]+16*cell],
        'field_statistic':'eigensystem of saved 2x2x2-averaged full-periodic-box normalized R7 tidal tensor; no local FFT, new smoothing, fit or posterior draw',
        'class_rule':'number of ordered eigenvalues strictly above 0.2: 0 void, 1 sheet, 2 filament, 3 knot',
        'coordinate_definition':'real-space observer-centred Cartesian h^-1 Mpc from RA, DEC, parent Z_COSMO and pinned DESI/c000 distance table; periodic source coordinate = (observer coordinate - 1000) mod 2000',
        'selection':'existing observed-success BGS Bright TARGETIDs, ZWARN=0 and 0.15<=observed Z<0.55; exact parent join; projected within the fixed real-space slab',
        'galaxy_count':len(a),'unsupported_count':int((~supported).sum()),'invalid_box_index_retained':int((a['BOX_INDEX']<0).sum()),
        'parent_slab_count':len(ids),'display_thinning':'none','seed':'not applicable: deterministic',
        'support_definition':'saved random-support indicator on the matching chart; hatched cells have any unsupported raw subcell; unsupported galaxies retained as crosses',
        'numerical_qa':{'trace_max_abs_error':trace_err,'eigen_equation_max_abs_error':residual,'orthogonality_max_abs_error':orth_err,'unique_targetids':True,'parent_observed_ra_dec_parity':True},
        'inputs':[{'path':str(p),'sha256':sha(p)} for p in [gp,tp,pp,op,sp,cfgp,tablep,target]],
        'large_catalogues':[read(pp)['output'],read(op)['output']],
        'response_source':{**srec,'verification':'receipt + byte size + mtime; extracted support saved/hash-bound, no full 3GB rehash'},
        'execution':{'hostname':socket.gethostname(),'slurm_action':'none','login_authorization':'explicit user instruction 2026-10-12','threads':1},
        'scientific_status':'simulation truth illustration on an exposed training phase; not posterior inference or validation',
        'allowed_claim':'the selected mock galaxies occupy the same volume as a structured underlying tidal field',
        'source_repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'renderer_path':str(Path(__file__).resolve()),'renderer_sha256':sha(__file__),
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save(out/'data/receipt.json',meta)
    print(f'Saved {len(a)} actual galaxies; {int((~supported).sum())} outside random support.',flush=True)

def render(out, only=None):
    d=np.load(out/'data/tidal_overlay.npz');meta=read(out/'data/receipt.json')
    vals=d['eigenvalues'];vec=d['eigenvectors'];xy=d['xy'];cell=float(d['cell'])
    x0,x1,y0,y1=meta['xy_bounds_mpc_h'];extent=[x0,x1,y0,y1]
    gap=np.diff(vals,axis=-1);gapmin=[gap[...,0],np.minimum(gap[...,0],gap[...,1]),gap[...,1]]
    lim=float(np.max(np.abs(vals[...,1])));lim=max(.25,np.ceil(lim*4)/4)
    cmap=plt.get_cmap('RdBu_r');norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#F5F5F5','axes.labelcolor':'#F5F5F5',
       'xtick.color':'#C8CDD4','ytick.color':'#C8CDD4','axes.edgecolor':'#C8CDD4','font.size':15,
       'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'black'})
    specs=[('T01_classes','Tidal-web environments beneath the mock galaxies',None),
           ('T02_lambda2','The tidal field beneath the mock galaxies',None),
           ('T03_e1','Tidal directions: the smallest-eigenvalue axis',0),
           ('T04_e2','Tidal directions: the intermediate axis',1),
           ('T05_e3','Tidal directions: the largest-eigenvalue axis',2)]
    if only is not None: specs=[v for v in specs if v[0]==only]
    previews=[];glyph_meta={}
    for stem,title,e in specs:
      for review in [False,True]:
        fig=plt.figure(figsize=(1920/144,1080/144),dpi=144,facecolor='black')
        def text(x,y,s,size=15,**kw):return fig.text(x,y,s,fontsize=size,**kw)
        if review:text(.045,.918,title,28,weight='bold')
        text(.045,.838,f'Abacus ph002 · simulation truth at z = 0.2 · {len(xy):,} selected mock galaxies',18)
        ax=fig.add_axes([.085,.27,.70,.52],facecolor='black')
        if e is None and stem=='T01_classes':
            im=ax.imshow(d['classes'].T,origin='lower',extent=extent,interpolation='nearest',cmap=ListedColormap(COLORS),vmin=-.5,vmax=3.5,alpha=.78)
            for j,(name,color) in enumerate(zip(['Void','Sheet','Filament','Knot'],COLORS)):
                fig.patches.append(Rectangle((.82,.70-j*.065),.017,.027,transform=fig.transFigure,facecolor=color,alpha=.78))
                text(.849,.701-j*.065,name,16)
            text(.815,.37,'λ threshold = 0.2',14)
        elif e is None:
            im=ax.imshow(vals[...,1].T,origin='lower',extent=extent,interpolation='nearest',cmap=cmap,norm=norm)
            cax=fig.add_axes([.83,.33,.015,.37]);cb=fig.colorbar(im,cax=cax)
            cb.ax.tick_params(labelsize=14,colors='#C8CDD4');cb.outline.set_edgecolor('#C8CDD4')
            text(.812,.739,'λ₂',19);text(.81,.275,'Dimensionless',13)
            text(.81,.235,'Normalized',13);text(.81,.20,'tidal tensor',13)
        else:
            # The 3D eigenvector sign is arbitrary: symmetric segments, no arrows.
            ax.set_facecolor('#111820')
            xx,yy=np.meshgrid(x0+(np.arange(32)+.5)*cell,y0+(np.arange(16)+.5)*cell,indexing='ij')
            centers=np.stack([xx,yy],-1);v=vec[...,:2,e]
            visible=(gapmin[e]>=.02)&(np.linalg.norm(v,axis=-1)>=.15)
            delta=cell*.43*v
            seg=np.stack([centers-delta,centers+delta],-2)[visible]
            ax.add_collection(LineCollection(seg,colors='#65D6E8',linewidths=1.6,zorder=2))
            bad=centers[~visible];ax.scatter(bad[:,0],bad[:,1],s=12,marker='s',facecolors='none',edgecolors='#7D8795',linewidths=.6)
            glyph_meta[f'e{e+1}']={'shown':int(visible.sum()),'omitted':int((~visible).sum())}
            text(.815,.69,f'e{e+1} direction',16,color='#65D6E8')
            text(.815,.635,'Projected in x–y',13)
            text(.815,.575,'Length shows',13);text(.815,.54,'in-plane fraction',13)
            text(.815,.44,'□  Direction',13);text(.839,.405,'suppressed*',13)
            text(.815,.30,'Hatch / ×:',13);text(.815,.265,'limited support',13)
        # Hatch actual unsupported subcell regions without treating them as zero tidal field.
        for i,j in np.argwhere(d['support']<1):
            ax.add_patch(Rectangle((x0+i*cell,y0+j*cell),cell,cell,facecolor='#20252B',alpha=.5,edgecolor='#C8CDD4',hatch='///',linewidth=0,zorder=3))
        ok=d['supported'];ax.scatter(xy[ok,0],xy[ok,1],s=19,c='#FFFFFF',edgecolors='#000000',linewidths=.55,zorder=5)
        ax.scatter(xy[~ok,0],xy[~ok,1],s=29,c='#C8CDD4',marker='x',linewidths=1.2,zorder=6)
        ax.set(xlim=(x0,x1),ylim=(y0,y1),xlabel='x [h⁻¹ Mpc]',ylabel='y [h⁻¹ Mpc]',aspect='equal')
        ax.tick_params(labelsize=15);ax.xaxis.label.set_size(18);ax.yaxis.label.set_size(18)
        text(.045,.17,'White dots: observed-success BGS mocks, shown at their real-space positions',15)
        text(.045,.122,f'R = 7 h⁻¹ Mpc · cell / slab = {cell:.3f} h⁻¹ Mpc · real-space z = {float(d["zlo"]):.2f} to {float(d["zhi"]):.2f} h⁻¹ Mpc',14)
        footer='Hatching / crosses: incomplete survey support · fixed geometry · no display thinning'
        if e is not None:footer='*Suppressed: eigenvalue gap < 0.02 or projected norm < 0.15 · axes, not flow arrows'
        text(.045,.069,footer,13,color='#C8CDD4')
        fig.canvas.draw();ren=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height()
        for t in fig.texts:
            b=t.get_window_extent(ren)
            assert b.x0>=0 and b.y0>=0 and b.x1<=w and b.y1<=h,(t.get_text(),b.bounds)
        name=stem+('_review' if review else '_plot_only')
        for ext in ['png','pdf']:fig.savefig(out/'static'/f'{name}.{ext}',dpi=144)
        plt.close(fig)
        im=Image.open(out/'static'/f'{name}.png').convert('RGB');assert im.size==(1920,1080)
        if review:
            im=im.resize((1280,720),Image.Resampling.LANCZOS);im.save(out/'qa'/f'{stem}_720p.png');previews.append(im)
    sheet=Image.new('RGB',(1280,720*len(previews)))
    for i,im in enumerate(previews):sheet.paste(im,(0,720*i))
    sheet.save(out/'qa/contact_sheet.png')
    meta.update(eigenvector_display={'projection':'3D eigendirections projected into x-y; segment length proportional to projected norm; sign-free',
       'gap_minimum':.02,'projected_norm_minimum':.15,'counts':glyph_meta},lambda2_color_limits=[-lim,lim],
       environment={'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
       output_hashes={str(p.relative_to(out)):sha(p) for folder in ['static','data','qa'] for p in sorted((out/folder).glob('*')) if p.is_file()})
    for p in (out/'static').glob('*.pdf'):assert b'/FontFile2' in p.read_bytes(),p
    meta['render_source_sha256']=sha(__file__)
    save(out/f'{only}_manifest.json',meta)
    print(f'{only}: review and plot-only PNG/PDF rendered.',flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--render-only',action='store_true')
    ap.add_argument('--single-view');args=ap.parse_args()
    for sub in ['data','static','qa']: (args.output/sub).mkdir(parents=True,exist_ok=True)
    if not args.render_only:extract(args.output)
    if args.single_view:
        render(args.output,args.single_view)
    else:
        # Isolate export backends for independent per-view PNG/PDF rendering.
        names=['T01_classes','T02_lambda2','T03_e1','T04_e2','T05_e3']
        for name in names:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(args.output),
                '--render-only','--single-view',name],check=True)
        meta=read(args.output/'T01_classes_manifest.json')
        meta['eigenvector_display']['counts']={}
        for name in names[2:]:
            meta['eigenvector_display']['counts'].update(read(args.output/f'{name}_manifest.json')['eigenvector_display']['counts'])
        sheet=Image.new('RGB',(1280,720*len(names)))
        for i,name in enumerate(names):
            im=Image.open(args.output/'qa'/f'{name}_720p.png').convert('RGB');sheet.paste(im,(0,720*i))
        sheet.save(args.output/'qa/contact_sheet.png')
        meta['output_hashes']={str(p.relative_to(args.output)):sha(p) for folder in ['static','data','qa']
            for p in sorted((args.output/folder).glob('*')) if p.is_file()}
        save(args.output/'manifest.json',meta)
if __name__=='__main__':main()
