"""Scientific invariants for galaxy-position annotation and periodic sampling."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import fitsio
import numpy as np

WORKFLOW=Path(__file__).resolve().parents[1]/'workflows/abacus_tweb'
if str(WORKFLOW) not in sys.path: sys.path.insert(0,str(WORKFLOW))
from galaxy_position_mapping import reconstruct_native, periodic_delta, field_indices, comoving_distance
from tweb_grid_sampling import TWebGrid, LABEL_COLUMNS
from galaxy_position_validation_status import LABEL_POLICY_VERSION, independent_status, load_independent_index
from annotate_cutsky_with_tweb_eigs import annotate_chunk, verify_source_rows, load_qualification, sha256_file, MAPPING_VERSION


def write_grid(root,spans):
    n=8
    x,y,z=np.indices((n,n,n))
    wave=.09*np.sin(2*np.pi*x/n)+.07*np.cos(2*np.pi*y/n)+.05*np.sin(2*np.pi*z/n)
    eig=np.stack([wave-1,wave,wave+1]).astype('f4')
    cweb=np.sum(eig>.2,axis=0).astype('u1')
    for rank,(lo,hi) in enumerate(spans):
        np.savez(root/f'abacus_cactus_tweb_rank{rank:04d}.npz',eig_vals=eig[:,lo:hi],cweb=cweb[lo:hi],
                 x_start=lo,x_end=hi,ngrid=n,boxsize=2000.,threshold=.2,Rsmooth=7.)
    return eig,cweb

class CoordinateTests(unittest.TestCase):
    def test_independent_spherical_axes_and_periodic_origin(self):
        z=np.repeat(.2,4); r=comoving_distance(z,direct=True)
        got=reconstruct_native([0,90,180,0],[0,0,0,90],z)
        expected=np.full((4,3),-1000.)
        expected[0,0]+=r[0];expected[1,1]+=r[1];expected[2,0]-=r[2];expected[3,2]+=r[3]
        np.testing.assert_allclose(periodic_delta(got,expected),0,atol=1e-9)
    def test_distance_interpolation_against_direct_integration(self):
        z=np.r_[0,1,np.random.default_rng(19).uniform(0,1,100)]
        np.testing.assert_allclose(comoving_distance(z),comoving_distance(z,direct=True),rtol=0,atol=1e-8)
    def test_wrapping_matches_native_grid_not_half_box_shift(self):
        pos=np.array([[-1000,0,999.999],[1000,-2000,-.1],[3000,2000,1999.9]])
        ix=field_indices(pos)
        np.testing.assert_array_equal(ix[1],ix[2])
        self.assertEqual(ix[0,0],1024);self.assertEqual(ix[0,1],0)
    def test_invalid_inputs_do_not_fall_back_to_observed_redshift(self):
        for z in [np.nan,-.01,1.01]:
            with self.assertRaises(ValueError): reconstruct_native([0],[0],[z])
        with self.assertRaises(ValueError): reconstruct_native([0],[91],[.1])

class SamplingTests(unittest.TestCase):
    def test_full_grid_stencils_across_slab_and_periodic_boundaries(self):
        for spans in [[(0,8)],[(0,1),(1,4),(4,8)]]:
            with self.subTest(spans=spans),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);eig,cls=write_grid(root,spans);grid=TWebGrid(root)
                q=np.array([[0,0,0],[7,7,7],[1,4,6],[3,7,0],[4,2,7]])
                out=grid.sample_indices(q);idx=tuple(q.T)
                for k in range(3):
                    np.testing.assert_array_equal(out[f'LAMBDA{k+1}'],eig[k][idx])
                    lap=np.zeros_like(eig[k])
                    for axis,letter in enumerate('XYZ'):
                        plus=np.roll(eig[k],-1,axis=axis);minus=np.roll(eig[k],1,axis=axis)
                        grad=(plus-minus)*(7/(2*250))
                        np.testing.assert_allclose(out[f'DLAM{k+1}_D{letter}'],grad[idx],rtol=1e-6,atol=1e-8)
                        lap+=plus+minus-2*eig[k]
                    np.testing.assert_allclose(out[f'LAP_LAM{k+1}'],lap[idx]*49/250**2,rtol=1e-6,atol=1e-8)
                np.testing.assert_array_equal(out['CWEB'],cls[idx])
    def test_missing_slab_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_grid(root,[(0,3),(4,8)])
            with self.assertRaises(ValueError):TWebGrid(root)
    def test_satellites_with_shared_host_have_individual_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_grid(root,[(0,4),(4,8)]);grid=TWebGrid(root)
            data=np.zeros(2,dtype=[('RA','f8'),('DEC','f8'),('Z_COSMO','f4'),('Z','f4'),
               ('RES','i1'),('CEN','i1'),('HALO_MASS','f4'),('HALO_INDEX','i4')])
            data['RA']=[0,90];data['Z_COSMO']=.2;data['Z']=.8
            data['RES']=1;data['CEN']=[1,0];data['HALO_INDEX']=42;data['HALO_MASS']=100
            out=annotate_chunk(data,np.array([100,101]),grid)
            self.assertFalse(np.array_equal(out['LAMBDA1'][:1],out['LAMBDA1'][1:]))
            np.testing.assert_array_equal(out['CEN'],[1,0]);np.testing.assert_array_equal(out['SOURCE_ROW'],[100,101])
            data['Z']=.9
            changed=annotate_chunk(data,np.array([100,101]),grid)
            np.testing.assert_array_equal(out['LAMBDA1'],changed['LAMBDA1'])
            data['RES'][1]=0
            labelled=annotate_chunk(data,[100,101],grid)
            self.assertEqual(labelled['TWEB_STATUS'][1],1)
            self.assertEqual(labelled['HAS_LABEL'][1],1)
            self.assertEqual(labelled['INDEP_POS'][1],0)
            self.assertTrue(np.isfinite(labelled['LAMBDA1'][1]))
            np.testing.assert_array_equal(labelled['LAMBDA1'],changed['LAMBDA1'])
            self.assertEqual(len(labelled),2)
            checked=annotate_chunk(data,[100,101],grid,(np.array([100]),np.array([1],dtype='u1')))
            np.testing.assert_array_equal(checked['INDEP_POS'],[1,0])
            np.testing.assert_array_equal(checked['HAS_LABEL'],[1,1])

    def test_independent_status_is_exact_row_evidence_not_population_inference(self):
        index=(np.array([10,30,50]),np.array([1,2,3],dtype='u1'))
        np.testing.assert_array_equal(independent_status([50,9,30,50,51],index),[3,0,2,3,0])
        np.testing.assert_array_equal(independent_status([1,2],None),[0,0])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'evidence.npz'
            np.savez(p,source_row=index[0],independent_position_code=index[1])
            record=dict(path=str(p),sha256=sha256_file(p))
            got=load_independent_index(record)
            np.testing.assert_array_equal(got[0],index[0])
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):load_independent_index(record)

class IdentityTests(unittest.TestCase):
    def test_source_rows_preserve_order_duplicates_and_reject_wrong_metadata(self):
        data=np.zeros(3,dtype=[('RA','f8'),('DEC','f8'),('Z_COSMO','f4'),('RES','i1'),('CEN','i1'),('HALO_MASS','f4')])
        data['RA']=[10,20,30];data['Z_COSMO']=.1;data['RES']=1;data['CEN']=1
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'raw.fits';fitsio.write(p,data)
            with fitsio.FITS(p) as f:
                rows=np.array([2,0,2]);chunk=data[rows].copy();verify_source_rows(chunk,f[1],rows)
                chunk['RA'][0]=20
                with self.assertRaises(ValueError):verify_source_rows(chunk,f[1],rows)
    def test_qualification_rejects_unvalidated_phase_changed_code_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw.fits';raw.write_bytes(b'fixture');st=raw.stat()
            report=dict(completed=True,pass_gate=False,pass_supported_population_gate=True,mapping={'version':MAPPING_VERSION},
              helper_sha256=sha256_file(WORKFLOW/'galaxy_position_mapping.py'),label_policy_version=LABEL_POLICY_VERSION,allow_all_coordinate_populations=True,
              phases={'ph002':dict(pass_gate=False,pass_supported_population_gate=True,field_path=str(root),sources=[dict(path=str(raw),bytes=st.st_size,mtime_ns=st.st_mtime_ns)],field_sources=[dict(path=str(raw),bytes=st.st_size,mtime_ns=st.st_mtime_ns)])})
            p=root/'report.json';p.write_text(json.dumps(report));load_qualification(p,'ph002',root)
            report['allow_all_coordinate_populations']=False;p.write_text(json.dumps(report))
            with self.assertRaises(ValueError):load_qualification(p,'ph002',root)
            report['allow_all_coordinate_populations']=True;p.write_text(json.dumps(report))
            with self.assertRaises(ValueError):load_qualification(p,'ph001',root)
            report['helper_sha256']='changed';p.write_text(json.dumps(report))
            with self.assertRaises(ValueError):load_qualification(p,'ph002',root)
            report['helper_sha256']=sha256_file(WORKFLOW/'galaxy_position_mapping.py');p.write_text(json.dumps(report));raw.write_bytes(b'changed fixture')
            with self.assertRaises(ValueError):load_qualification(p,'ph002',root)

class PipelineMetadataTests(unittest.TestCase):
    def test_parent_identity_is_original_row_not_selected_row(self):
        from p10_build_bright_parent import with_source_rows, compact_parent_block
        raw=np.zeros(4,dtype=[('RA','f8'),('DEC','f8'),('Z','f4'),('Z_COSMO','f4'),
             ('R_MAG_APP','f4'),('FILE_NUM','i4'),('HALO_INDEX','i4'),('BOX_INDEX','i4'),
             ('RES','i1'),('CEN','i1'),('HALO_MASS','f4')])
        raw['RES']=1;raw['CEN']=[1,0,1,0]
        selected=with_source_rows(raw,200)[[1,3]]
        result=compact_parent_block(selected,np.array([1,2]),galaxy_metadata=True)
        np.testing.assert_array_equal(result['SOURCE_ROW'],[201,203])
        np.testing.assert_array_equal(result['CEN'],[0,0])
        np.testing.assert_array_equal(result['TARGETID'],[1,2])
    def test_observed_join_preserves_population_and_source_identity(self):
        from p10_build_observed_truth import output_dtype, join_successful_chunk, LSS_COLUMNS
        parent=np.zeros(2,dtype=output_dtype(True))
        parent['TARGETID']=[1,2];parent['SOURCE_ROW']=[101,900];parent['CEN']=[1,0]
        parent['RES']=1;parent['TWEB_STATUS']=1;parent['Z_COSMO']=[.19,.21]
        parent['LAMBDA1']=-.3;parent['LAMBDA2']=0;parent['LAMBDA3']=.1
        dtype=[(name,'f8') for name in LSS_COLUMNS]
        lss=np.zeros(2,dtype=dtype);lss['TARGETID']=[2,1];lss['Z_not4clus']=.2
        result,_=join_successful_chunk(lss,parent)
        np.testing.assert_array_equal(result['SOURCE_ROW'],[900,101])
        np.testing.assert_array_equal(result['CEN'],[0,1])
        np.testing.assert_array_equal(result['Z_COSMO'],parent['Z_COSMO'][[1,0]])
        parent['RES'][1]=0;parent['INDEP_POS']=[2,0]
        labelled,_=join_successful_chunk(lss,parent)
        self.assertEqual(len(labelled),2)
        np.testing.assert_array_equal(labelled['HAS_LABEL'],[1,1])
        np.testing.assert_array_equal(labelled['INDEP_POS'],[0,2])
        np.testing.assert_array_equal(labelled['SOURCE_ROW'],[900,101])
        parent['INDEP_POS'][1]=99
        with self.assertRaises(RuntimeError):join_successful_chunk(lss,parent)

    def test_cubic_index_is_local_to_file_block(self):
        from validate_galaxy_position_mapping import read_cubic_correspondence
        data=np.zeros(102,dtype=[('FILE_NUM','i4'),('HALO_MASS','f4'),('cen','i1'),('x','f4'),('y','f4'),('z','f4')])
        data['FILE_NUM']=np.repeat(np.arange(34),3);data['x']=np.arange(102)
        sample={'FILE_NUM':np.array([0,33,1,33]),'BOX_INDEX':np.array([2,0,1,0])}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cube.fits';fitsio.write(p,data)
            with fitsio.FITS(p) as f:result,starts=read_cubic_correspondence(f[1],sample)
        np.testing.assert_array_equal(result['x'],[2,99,4,99])
        self.assertEqual(starts,list(range(0,103,3)))

if __name__=='__main__':unittest.main()
