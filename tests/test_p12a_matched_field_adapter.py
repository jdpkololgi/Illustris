import unittest
import json
from pathlib import Path
import tempfile
import h5py
import numpy as np
from workflows.abacus_tweb.p12a_matched_field_adapter import derive_channels,MatchedFieldPatchAdapter,SELECTION_SCHEMA


class MatchedChannelTests(unittest.TestCase):
    def test_density_changes_only_expected_and_contrast_not_counts(self):
        raw=dict(counts=np.array([3.,5.]),exposure_binary=np.array([1.,0.]),radial_bin=np.array([1.,1.]),
                 basis_0=np.array([1.,1.]),basis_1=np.array([2.,2.]),basis_2=np.array([3.,3.]))
        selection=dict(density=[[1.,2.,4.],[2.,4.,8.]],valid=[[True]*3]*2)
        first=derive_channels(raw,selection,0);second=derive_channels(raw,selection,1)
        np.testing.assert_allclose(first['expected_counts'],17.)
        np.testing.assert_allclose(second['expected_counts'],34.)
        self.assertEqual(first['log_count_ratio'][1],0.)
        self.assertAlmostEqual(float(first['log_count_ratio'][0]),np.log(3.001/17.001),places=6)
        np.testing.assert_array_equal(raw['counts'],[3.,5.])

    def test_noninteger_bin_rejected(self):
        raw=dict(radial_bin=np.array([1.2]))
        with self.assertRaises(ValueError):derive_channels(raw,dict(density=[[1.,2.]],valid=[[True,True]]),0)

    def test_real_adapter_preserves_parent_positions_and_model_channels(self):
        from workflows.abacus_tweb.p8_train_unet_patch import model_inputs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);field=root/'field.h5';shape=(6,7,8)
            with h5py.File(field,'w') as handle:
                for name,value in dict(counts=3.,exposure_binary=1.,exposure_apodized=.8,
                                       basis_0=1.,basis_1=2.,basis_2=3.,radial_bin=1.).items():
                    handle[name]=np.full(shape,value,'f4')
            cap=dict(field_path=str(field),field_sha256='test')
            (root/'adapter_manifest.json').write_text(json.dumps(dict(phase='ph001',
                channel_order=['counts','exposure_apodized','log_count_ratio'],caps=dict(SGC=cap,NGC=cap))))
            for name,value in dict(core_voxel_start=[[2,2,2]],core_voxel_stop=[[4,5,6]],core_fold=[3],
                core_cap=[0],core_active_offsets=[0,2],core_active_parent=[10,11],
                core_active_frac_index=[[2.5,3.5,4.5],[3.,4.,5.]]).items():
                np.save(root/(name+'.npy'),np.asarray(value))
            selection=dict(fit_phases=['ph000'],omitted_phase='ph001',density=[[1.,2.,4.]]*2,valid=[[True]*3]*2)
            fit=root/'selection.json'
            fit.write_text(json.dumps(dict(schema=SELECTION_SCHEMA,**{'pass':True},density_unit='Mpc^-3',
                cosmology='Planck18',fit_phases=['ph000'],omitted_phase='ph001',selection=selection,
                contrast_epsilon=.001,phase_response={'ph001':{k:dict(path=str(field),sha256='test') for k in ('SGC','NGC')}})))
            with MatchedFieldPatchAdapter(root,selection_manifest=fit) as adapter:
                patch=adapter.extract(0,1)
            np.testing.assert_array_equal(patch.authoritative_parent_id,[10,11])
            np.testing.assert_allclose(patch.authoritative_frac_index_local,[[1.5,2.5,3.5],[2.,3.,4.]])
            normalization=dict(channels={'counts':dict(policy='zscore',mean=1.,std=2.),
                'exposure_apodized':dict(policy='identity'),'log_count_ratio':dict(policy='identity')})
            inputs,points=model_inputs(patch,normalization,'cpu')
            self.assertEqual(tuple(inputs.shape),(1,3,4,5,6))
            np.testing.assert_allclose(inputs.numpy()[0,0],(np.log(4.)-1.)/2.)
            np.testing.assert_allclose(inputs.numpy()[0,1],3.001/17.001-1.,rtol=1e-6)
            np.testing.assert_allclose(inputs.numpy()[0,2],.8)
            self.assertEqual(tuple(points.shape),(1,1,1,2,3))


if __name__=='__main__':unittest.main()
