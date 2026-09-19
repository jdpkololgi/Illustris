import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from workflows.sbi import e2e_coupled_condition_products as products
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_conditions as reader
from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec


class CoupledConditionTests(unittest.TestCase):
    def test_padding_is_rectangular_not_periodic(self):
        x=np.arange(60).reshape(3,4,5)
        actual=products.padded_extract(x,[-1,2,3],[3,3,4])
        self.assertEqual(actual.shape,(3,3,4))
        np.testing.assert_array_equal(actual[1:,:2,:2],x[:2,2:,3:])
        self.assertEqual(float(actual[0].sum()),0.)
        self.assertEqual(float(actual[:,:,2:].sum()),0.)
        self.assertEqual(float(products.padded_extract(x,[8,0,0],[3,4,5]).sum()),0.)

    def test_transforms_preserve_physical_number_density_units(self):
        np.testing.assert_allclose(products.transformed('counts',np.array([0,3])),[0,np.log(4)])
        h=c.config()['target']['coordinate_h']
        np.testing.assert_allclose(products.transformed('ntilde_h3_mpc3',np.array([0.])),
                                   [np.log(1e-10/h**3)],rtol=1e-7)
        with self.assertRaises(ValueError):
            products.transformed('counts',np.array([-1]))
        with self.assertRaises(ValueError):
            products.transformed('support_random',np.array([np.nan]))

    def test_factor_four_conservation_and_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            src,dst=Path(temp)/'source.h5',Path(temp)/'coarse.h5'
            shape=(19,9,7)
            grid=coord.grid_record(GridSpec((0,0,0),shape,3.383,27.064))
            with h5py.File(src,'x') as f:
                f.attrs.update(coordinate_sha256=c.sha256(coord.CONFIG),distance_unit='Mpc/h')
                for name in (*products.SUMMED,*products.AVERAGED):
                    f[name]=np.ones(shape,dtype='f4')
            qa=products.coarsen_response(src,dst,grid)
            with h5py.File(dst) as f:
                self.assertEqual(f['counts'].shape,(5,3,2))
                for a,b in qa.values(): self.assertEqual(a,b)
                self.assertEqual(float(f['counts'][:].sum()),np.prod(shape))
                self.assertEqual(float(f['geometry_valid_fraction'][-1,-1,-1]),9/64)
                np.testing.assert_array_equal(f['log_count_ratio_random'][:],0)
                self.assertFalse(f.attrs['contains_matter'])

    def test_coordinate_centers_match_block_means(self):
        grid=coord.grid_record(GridSpec((100,200,300),(128,128,128),3.383,27.064))
        raw=products.radial_coordinates(grid,[8,12,16],[8,8,8])
        wide=products.radial_coordinates(grid,[2,3,4],[2,2,2],stride=4)
        # Cartesian block means, not an incorrect average of radial distances.
        for axis in ('los_x','los_y','los_z'):
            xyz=raw[axis]*raw['observer_radius_mpc_h']
            np.testing.assert_allclose(op.mean_pool(xyz,4),
                wide[axis]*wide['observer_radius_mpc_h'],atol=1e-12)

    def test_full_shape_extract_is_targetless_and_matches_two_parents(self):
        cfg=op.layout(); shape=(144,112,112)
        grid=coord.grid_record(GridSpec((100,200,300),shape,3.383,27.064))
        x=np.broadcast_to(np.arange(shape[0],dtype='f4')[:,None,None],shape)
        raw={name:np.zeros(shape,dtype='f4') for name in products.LOCAL_CHANNELS[:-1]}
        raw['counts']=x; raw['support_random']=np.ones(shape,dtype='f4')
        coarse={name:np.zeros((36,28,28),dtype='f4') for name in products.WIDE_CHANNELS[:8]}
        row=dict(center=[48,48,48],grid=grid)
        arrays=products.extract_pair(row,raw,coarse)
        self.assertEqual(arrays['joint'].shape,(12,64,48,48))
        for center,crop in zip(([48,48,48],[80,48,48]),cfg['independent_parent_crops_in_joint']):
            expected=op.mean_pool(np.log1p(products.padded_extract(x,np.array(center)-48,[96]*3)),2)
            np.testing.assert_array_equal(arrays['joint'][0][tuple(slice(*s) for s in crop)],expected)
        np.testing.assert_array_equal(arrays['support'],1.)
        self.assertNotIn('target',arrays)

    def test_targetless_reader_and_phase_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); phase='ph007'; pair='ph007_NGC_s0_interior_00'
            folder=root/'conditions'/phase; folder.mkdir(parents=True)
            arrays=dict(joint=np.zeros((12,64,48,48),dtype='f4'),
                        wide_extended=np.zeros((12,56,56,56),dtype='f4'),
                        support=np.ones((64,48,48),dtype='f4'))
            path=folder/f'{pair}_generation_test.h5'
            with h5py.File(path,'x') as f:
                f.attrs.update(schema=products.SCHEMA,phase=phase,pair_id=pair,
                    contains_matter=False,coordinate_sha256=c.sha256(coord.CONFIG),
                    distance_unit='Mpc/h',local_channels=json.dumps(products.LOCAL_CHANNELS),
                    wide_channels=json.dumps(products.WIDE_CHANNELS))
                for name,x in arrays.items(): f.create_dataset(name,data=x,compression='lzf')
            receipt=dict(**coord.provenance(),phase=phase,pair_id=pair,contains_matter=False,
                         outputs=[c.file_record(path,content_hash=True)],**{'pass':True})
            marker=folder/f'{pair}.json'; c.atomic_json(marker,receipt)
            with patch.object(coord,'ROOT',root):
                actual=reader.load_pair(phase,pair)
                self.assertEqual(set(actual),{'joint','wide_extended','support'})
                for offset in op.layout()['context_offsets_raw']:
                    crop=reader.crop_context(actual,phase,offset)
                    self.assertEqual(crop['wide'].shape,(12,48,48,48))
                with self.assertRaises(PermissionError):
                    reader.crop_context(actual,'ph014',[32,0,0])
                with self.assertRaises(PermissionError):
                    reader.load_pair('ph001','ph001_NGC_s0_interior_00')
                with self.assertRaises(ValueError):
                    reader.load_pair(phase,'../targets')
                # A dangling external link must be rejected before HDF5 tries
                # opening it (which would raise KeyError instead).
                with h5py.File(path,'a') as f:
                    del f['joint']
                    f['joint']=h5py.ExternalLink(str(root/'never_open_target.h5'),'/truth')
                receipt['outputs']=[c.file_record(path,content_hash=True)]
                c.atomic_json(marker,receipt,replace=True)
                with self.assertRaises(PermissionError): reader.load_pair(phase,pair)
                # Even a validly hashed target outside the condition directory
                # must not be followed by the inference reader.
                receipt['outputs'][0]['path']=str(root/'targets'/'ph007'/'truth.h5')
                c.atomic_json(marker,receipt,replace=True)
                with self.assertRaises(PermissionError): reader.load_pair(phase,pair)


if __name__=='__main__': unittest.main()
