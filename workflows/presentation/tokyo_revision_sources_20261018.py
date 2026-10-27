#!/usr/bin/env python3
"""Verify small frozen sources for Tokyo A--E, without inference or catalogue scans."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,shutil
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
import fitsio
from astropy.cosmology import Planck18
sys.path.insert(0,'/global/homes/d/dkololgi/.cache/tokyo_figure_tools')
import pymupdf
ROOT=Path('/global/homes/d/dkololgi/fmsu_tokyo_20261010');REPO=Path('/global/homes/d/dkololgi/TNG/Illustris');OUT=ROOT/'revisions_20261018';DATA=OUT/'data';DATA.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
bindings=[]
def bind(p,expected=None,name=None):
 p=Path(p);s=sha(p)
 if expected is not None:assert s==expected,(p,s,expected)
 bindings.append({'path':str(p),'sha256':s})
 if name:shutil.copy2(p,DATA/name)
 return p
# Original ph002 slice and coordinates, including both redshift columns.
m=read(ROOT/'tidal_ph002/manifest.json');p=bind(ROOT/'tidal_ph002/data/tidal_overlay.npz',m['output_hashes']['data/tidal_overlay.npz'],'tidal_overlay.npz');d=dict(np.load(p))
bind(ROOT/'tidal_ph002/manifest.json',name='tidal_manifest.json')
cfg=read(REPO/'configs/e2e_coupled_coordinates_v2.json');table=bind(REPO/cfg['table_path'],cfg['table_sha256'],'desi_distance_701f498.dat');bind(REPO/'configs/e2e_coupled_coordinates_v2.json',name='coordinates_config.json')
t=np.loadtxt(table);chi=CubicSpline(t[:,0],t[:,2],extrapolate=False)
r=np.linalg.norm(d['xyz'],axis=1);real_error=float(np.max(abs(r-chi(d['z_cosmo']))));obs_error=float(np.max(abs(r-chi(d['z_observed']))));assert real_error<1e-9 and obs_error>1
np.testing.assert_array_equal(d['xy'],d['xyz'][:,:2]);np.testing.assert_allclose(np.linalg.eigvalsh(d['tensor']),d['eigenvalues'],atol=1e-12);np.testing.assert_array_equal((d['eigenvalues']>.2).sum(-1),d['classes'])
ma=read(ROOT/'a01_ph002/manifest.json');a=dict(np.load(bind(ROOT/'a01_ph002/data/A01_density_tracers.npz',ma['output_hashes']['data/A01_density_tracers.npz'],'density_tracers.npz')))
np.testing.assert_array_equal(a['xyz'],d['xyz']);np.testing.assert_array_equal(a['targetid'],d['targetid']);np.testing.assert_allclose(a['rho_over_mean']-1,np.trace(d['tensor'],axis1=-2,axis2=-1),atol=1e-12)
# Same ph006 object and all saved joint draws, original class-conditional representatives.
mp=read(ROOT/'local_environments_selection_20261017/MANIFEST.json');p=bind(ROOT/'local_environments_selection_20261017/data/local_posterior.npz',mp['files']['data/local_posterior.npz'],'local_posterior.npz');c=dict(np.load(p));prov=read(bind(ROOT/'local_environments_selection_20261017/data/local_provenance.json',name='local_provenance.json'))
assert c['draws'].shape==(512,3);assert np.all(np.diff(c['draws'],axis=1)>=0)
counts=np.bincount((c['draws']>.2).sum(1),minlength=4);np.testing.assert_array_equal(counts/512,c['class_probabilities']);assert counts.tolist()==[209,303,0,0];assert c['chosen_draw_indices'].tolist()==[362,291]
np.testing.assert_array_equal(c['draws'][c['chosen_draw_indices']],prov['triplets']);assert int(c['target_parent_row'])==2717758 and prov['targetid']==6765665
# Bounded exact-row read: independently reconstruct all242 displayed positions from observed Z.
cat=prov['parent_catalogue_binding'];assert Path(cat['path']).stat().st_size==cat['bytes']
with fitsio.FITS(cat['path']) as f:rows=f[1].read(rows=c['parent_rows'],columns=['TARGETID','RA','DEC','Z'])
np.testing.assert_array_equal(rows['TARGETID'],c['targetid']);ra=np.deg2rad(rows['RA']);dec=np.deg2rad(rows['DEC']);rr=Planck18.comoving_distance(rows['Z']).value
xyz=rr[:,None]*np.c_[np.cos(ra)*np.cos(dec),np.sin(ra)*np.cos(dec),np.sin(dec)];delta=xyz-c['center_mpc'];local_error=float(np.max(abs(delta-c['observed_xyz_relative_mpc'])));assert local_error<1e-6
np.savez_compressed(DATA/'local_coordinate_check.npz',TARGETID=rows['TARGETID'],RA=rows['RA'],DEC=rows['DEC'],Z_observed=rows['Z'],reconstructed_xyz_mpc=xyz)
# TARP: copy original curve/envelope, not recomputed or complemented.
mt=read(ROOT/'a03_tarp/provenance_qa.json');p=bind(ROOT/'a03_tarp/plotted_arrays.npz',mt['output_hashes']['plotted_arrays.npz'],'tarp_arrays.npz');t=dict(np.load(p));saved=mt['saved_TARP'];np.testing.assert_array_equal(t['alpha'],saved['alpha']);np.testing.assert_array_equal(t['expected_coverage_probability'],saved['expected_coverage_probability']);w=mt['overlay']['half_width'];np.testing.assert_allclose(t['envelope_lower'],np.maximum(0,t['alpha']-w),atol=0,rtol=0);np.testing.assert_allclose(t['envelope_upper'],np.minimum(1,t['alpha']+w),atol=0,rtol=0)
meta={k:v for k,v in mt.items() if k not in ('saved_TARP','output_hashes')};meta['saved_curve_points']=len(t['alpha']);meta['axis_convention']='CDF of fraction of posterior draws nearer random reference than truth; x=assigned region probability, y=fraction of truths inside. No complements or data transform.';write(DATA/'tarp_provenance.json',meta)
bind(ROOT/'a03_tarp/provenance_qa.json')
# Reuse saved PDF vector contours; no fitted UMAP or synthetic point reconstruction.
pdf=bind(REPO/'workflows/gcn_paper/umap_gat_embeddings_alpha_test_predictions.pdf');doc=pymupdf.open(pdf);page=doc[0]
paths=[]
for g in page.get_drawings():
 r=g['rect']
 if g['type']=='f' and g['fill'] is not None and r.x0>45 and r.x1<365 and r.y0>60 and r.y1<400 and len(g['items'])>100:
  assert all(v[0]=='l' for v in g['items'])
  # Preserve every segment, including disconnected paths if present.
  segments=np.array([[[v[1].x,v[1].y],[v[2].x,v[2].y]] for v in g['items']])
  paths.append({'original_rgb':list(g['fill']),'segments_pdf':segments.tolist(),'original_alpha':g['fill_opacity']})
assert len(paths)==4
# The saved legend order gives void,wall,filament,cluster. Major ticks fix affine axes.
x0=61.981910705566406;xs=(143.2891082763672-x0)/2;y2=383.6811828613281;ys=(383.6811828613281-340.46795654296875)/2
for i,p in enumerate(paths):
 q=np.array(p['segments_pdf']);q[:,:,0]=(q[:,:,0]-x0)/xs;q[:,:,1]=2+(y2-q[:,:,1])/ys;p['segments_data']=q.tolist();p['class_index']=i
 # Round-trip endpoints demonstrate no movement apart from recorded affine conversion.
 z=q.copy();z[:,:,0]=z[:,:,0]*xs+x0;z[:,:,1]=y2-(z[:,:,1]-2)*ys;np.testing.assert_allclose(z,p['segments_pdf'],atol=1e-12)
paper={'source_pdf':str(pdf),'sha256':sha(pdf),'source_metadata':doc.metadata,'projection':'First column, UMAP1 vs UMAP2; saved test-set GAT+ predicted-class density contours, NOT true-class recolouring.','paths':paths,'axis_transform':{'x_zero_pdf':x0,'x_pdf_per_unit':xs,'y_two_pdf':y2,'y_pdf_per_unit':ys},'limitations':'Saved pointwise UMAP coordinates not found in repository or known legacy cache. Hidden-dimensional node embeddings and probabilities are available but are not saved projected coordinates. No UMAP refit, true-class recolouring or entropy alternative. Numerical marks reuse authored PDF paths; no per-point identities can be recovered from contours.','paper':'Kololgi et al.2026 RASTI doi:10.1093/rasti/rzag025','input_definition':'Published Paper I: galaxy-position Delaunay graph and ten graph metrics; stellar mass defines sample, not an added inference input. Simulation tidal labels provide supervision.','checkpoint':'Historical2025-11-25 model associated by the saved postprocessing script, not independently rerun or requalified. PDF is the authoritative rendered source.','published_source':'https://academic.oup.com/rasti/article/doi/10.1093/rasti/rzag025/8554208'}
write(DATA/'paperI_contours.json',paper)
for name in ['gcn_pipeline_postprocess.py']:bind(REPO/'workflows/gcn_paper'/name,name='paperI_'+name)
# Documentary source hashes, small compared with the full catalogues.
for p in [REPO/'workflows/presentation/tokyo_tidal_overlay.py',REPO/'workflows/presentation/tokyo_local_environments_20261017.py',Path('/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/lib/python3.11/site-packages/tarp/drp.py')]:bind(p)
audit={'ph002':{'displayed_count':len(d['xyz']),'radial_Z_COSMO_max_abs_error_mpc_h':real_error,'radial_Z_observed_max_abs_error_mpc_h':obs_error,'coordinate_columns':'RA,DEC,Z_COSMO from parent linkage; observed Z only for inherited selection','distance':'Pinned cosmoprimo DESI / AbacusSummitBase c000 table, h^-1 Mpc','velocities':'No displacement added or removed in plotting; Z_COSMO selected directly. Z contains radial RSD but is not used for these displayed positions.','field_chart':'Unshifted saved R7 tensor/density on observer-centred real-space chart; native source=(observer-1000) mod2000 h^-1 Mpc.','projection':'x/y, fixed z interval','xy_bounds_mpc_h':m['xy_bounds_mpc_h'],'z_bounds_mpc_h':m['slice_z_bounds_mpc_h'],'smoothing_mpc_h':7,'field_galaxy_pairing':'Same real-space chart and slab; dots selected by observational success and exact TARGETID. NOT an observed-redshift-space overlay or a field reconstructed at observed coordinates.','target_linkage':'P12-A per-galaxy tidal targets are host linked; continuous fields in these figures are saved field-grid truth, not per-dot host labels.'},'ph006':{'TARGETID':6765665,'parent_node_id':2717758,'phase':'ph006','observed_z':prov['observed_redshift'],'coordinate_columns':'RA,DEC,observed Z in canonical observed catalogue','distance':'Astropy Planck18 comoving Mpc, not DESI/c000 table and not h^-1 Mpc','max_xyz_reconstruction_error_mpc':local_error,'window':'Same original relative x/y ±40Mpc, z ±7.5Mpc; 242 plotted rows including central target','velocities':'Input observed Z retains radial RSD; no de-redshifting in plotting.','target_linkage':'Native host-linked target; displayed target location is observed redshift space, not a claim of a tidal field evaluated there.','draws':512,'counts':counts.tolist(),'draw_indices':c['chosen_draw_indices'].tolist(),'model':prov['model'],'checkpoint_sha256':prov['checkpoint']},'tarp':{'curve_points':len(t['alpha']),'unchanged':True,'envelope_half_width':w,'axis_convention':meta['axis_convention']},'paperI':{k:v for k,v in paper.items() if k!='paths'},'source_bindings':bindings,'no_new_inference':True,'no_slurm_actions':True}
write(OUT/'SOURCE_AUDIT.json',audit)
print(json.dumps({k:audit[k] for k in ['ph002','ph006','tarp']},indent=2))
