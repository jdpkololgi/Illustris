import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import healpy as hp
from scipy.ndimage import gaussian_filter
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_observations as obs
from workflows.sbi import e2e_coupled_contract as c


class CoupledResponseTests(unittest.TestCase):
    def test_response_corrected_geometry_volume_and_los(self):
        spec=obs.grid_ops.GridSpec((350.,100.,50.),(16,16,16),3.383,27.064)
        slices=(slice(2,7),slice(3,9),slice(1,8))
        n=hp.nside2npix(256)
        support=np.ones(n,dtype=bool)
        response=np.linspace(.8,1.2,n,dtype=np.float32)
        boundary=np.full(n,.1,dtype=np.float32)
        selection=json.loads(Path(c.config()['observation']['fixed_selection_manifest']).read_text())
        values,z=obs.response_chunk(spec,slices,support,response,boundary,selection,'NGC')
        gx,gy,gz=obs.grid_ops.coordinate_block(spec,slices,halo=5)
        radius=np.sqrt(gx*gx+gy*gy+gz*gz)
        redshift=coord.redshift(radius)
        shape=radius.shape
        pix=hp.vec2pix(256,np.broadcast_to(gx,shape)/radius,np.broadcast_to(gy,shape)/radius,np.broadcast_to(gz,shape)/radius)
        radial=(redshift>=.1)&(redshift<.6)&~((redshift>=.585)&(redshift<.595))
        apod=gaussian_filter(radial.astype(np.float32),sigma=1.2,mode='constant',cval=0,truncate=4)
        trim=(slice(5,10),slice(5,11),slice(5,12))
        np.testing.assert_array_equal(values['support_random'],radial[trim])
        np.testing.assert_array_equal(values['exposure_apodized_random'],apod[trim])
        np.testing.assert_array_equal(z,redshift[trim])
        curve=selection['rotations']['0']['caps']['NGC']
        nbar=(np.interp(z,curve['grid_z'],curve['ntilde'])*coord.selection_volume_jacobian(z,selection)).astype('f4')
        expected=(nbar.astype('f8')*3.383**3*response[pix[trim]].astype('f8')*radial[trim]*apod[trim].astype('f8')).astype('f4')
        np.testing.assert_allclose(values['expected_counts_random'],expected,rtol=1e-7)
        np.testing.assert_allclose(sum(values['los_'+a]**2 for a in 'xyz'),1.,atol=2e-7)

    def test_interruption_before_random_receipt_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source.fits';source.touch()
            directory=root/'angular';directory.mkdir()
            def add(counts, before):
                counts[:,2]=5
                return {'rows':20}
            original=c.atomic_json
            def stop(path,value,**kwargs):
                if path.name=='random_00.json':
                    raise RuntimeError('injected crash after NPZ before receipt')
                return original(path,value,**kwargs)
            fake_map=lambda counts: dict(counts=counts,metadata={'synthetic':True})
            with patch.object(c,'paths',return_value={'randoms':[source]}), patch.object(obs.response_ops,'add_random_file',side_effect=add), patch.object(obs.response_ops,'normalized_map',side_effect=fake_map):
                with patch.object(c,'atomic_json',side_effect=stop):
                    with self.assertRaisesRegex(RuntimeError,'injected crash'):
                        obs.angular_maps('ph007',directory)
                self.assertFalse((directory/'random_00.json').exists())
                result=obs.angular_maps('ph007',directory)
                self.assertTrue(result['pass'])
                self.assertEqual(len(list(directory.glob('random_00_generation_*.npz'))),2)
                with np.load(result['outputs'][0]['path']) as f:
                    self.assertEqual(int(f['counts'].sum()),20)

    def test_interruption_before_hdf5_receipt_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            spec=obs.grid_ops.GridSpec((350.,100.,50.),(4,4,4),3.383,27.064)
            ones=np.ones((4,4,4),dtype='f4')
            z=np.broadcast_to(np.array([.2,.3,.4,.5])[:,None,None],ones.shape)
            def chunk(*args):
                return {k:ones.copy() for k in ('support_random','angular_response',
                        'exposure_apodized_random','expected_counts_random','ntilde_h3_mpc3',
                        'distance_to_support_boundary','los_x','los_y','los_z')},z
            angular=dict(support=np.ones(4,dtype=bool),domain=np.full(4,2),angular_response=np.ones(4))
            xyz=np.zeros((64,3))
            original=c.atomic_json
            def stop(path,value,**kwargs):
                if path.name=='NGC_COMPLETE.json':
                    raise RuntimeError('injected crash after HDF5 before receipt')
                return original(path,value,**kwargs)
            with patch.object(obs.grid_ops,'grid_from_xyz',return_value=spec), \
                 patch.object(obs.grid_ops,'cic_deposit',return_value=(ones,{'lost_weight':0})), \
                 patch.object(obs.response_ops,'angular_boundary_distance',return_value=np.zeros(4)), \
                 patch.object(obs,'response_chunk',side_effect=chunk):
                with patch.object(c,'atomic_json',side_effect=stop):
                    with self.assertRaisesRegex(RuntimeError,'injected crash'):
                        obs.build_cap('ph007','NGC',xyz,angular,{}, {},directory)
                self.assertFalse((directory/'NGC_COMPLETE.json').exists())
                result=obs.build_cap('ph007','NGC',xyz,angular,{}, {},directory)
                self.assertTrue(result['pass'])
                self.assertEqual(len(list(directory.glob('NGC_response_generation_*.h5'))),2)
                self.assertEqual(result['coordinate_sha256'],c.sha256(coord.CONFIG))


if __name__=='__main__':
    unittest.main()
