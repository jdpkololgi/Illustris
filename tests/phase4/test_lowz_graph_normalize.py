import unittest
from unittest.mock import patch
import numpy as np
from workflows.abacus_tweb import lowz_graph_normalize as mod


class TrainOnlyGraphFeaturesTest(unittest.TestCase):
    def test_same_cap_and_redshift_bin_as_selection(self):
        selection = dict(bin_edges=[0.,.1,.2],density=[[3.,5.],[7.,0.]],valid=[[1,1],[1,0]])
        vals = mod.density_log(np.array([.05,.1,.09,.2]),np.array([0,0,1,1]),selection)
        np.testing.assert_allclose(vals,np.log([3.,5.,7.,1e-12]))
        with self.assertRaises(ValueError):
            mod.density_log(np.array([-.01]),np.array([0]),selection)

    def test_canonical_columns_and_positive_transform(self):
        raw=np.full((2,7),10.)
        result=mod.scale_raw(raw,np.array([0,1]),[np.full(6,2.),np.full(6,5.)])
        np.testing.assert_allclose(result[:,mod.SI],np.array([[5.]*6,[2.]*6])+mod.EPS)
        np.testing.assert_allclose(result[:,1],10+mod.EPS)
        np.testing.assert_array_equal(raw,10.)

    def test_refuses_development_phase_before_sampling(self):
        with patch.object(mod,"roles",return_value=("ph002",*mod.DEV)), patch.object(mod,"phase_inputs") as get:
            with self.assertRaises(ValueError):
                mod.fit({},("ph002","ph006"),dict(fit_phases=["ph002","ph006"]))
            get.assert_not_called()

    def test_fit_uses_only_named_training_phases_and_matches_edge_interface(self):
        training=("ph002","ph003")
        selection=dict(bin_edges=[0.,.1,.2],density=[[3.,5.],[7.,9.]],valid=[[1,1],[1,1]],fit_phases=training)
        rng=np.random.default_rng(42)
        raw=rng.lognormal(size=(48,7))
        cap=np.repeat([0,1],24)
        pairs=np.array([[k,k+1] for k in range(47) if k!=23],np.int32)
        feat=rng.lognormal(size=(len(pairs),5)).astype("f4")
        rec=dict(source_pairs=["pairs"],source_attrs=["attrs"])
        def load(path):
            if str(path)=="pairs": return pairs
            if str(path)=="attrs": return feat
            assert str(path).endswith("core_active_parent.npy")
            return np.arange(len(raw))
        def phase(p,d):
            assert p in training
            return rec,raw,np.tile([.05,.15],24),cap,np.ones(len(cap),bool)
        with patch.object(mod,"roles",return_value=training+mod.DEV), patch.object(mod,"N",16), \
                patch.object(mod,"array",side_effect=load), patch.object(mod,"phase_inputs",side_effect=phase):
            spec=mod.fit(dict.fromkeys(training),training,selection)
        self.assertEqual(len(spec["lambdas"]),7)
        self.assertGreater(spec["log_std"],0)
        self.assertGreater(spec["edge"]["1"]["log_density_contrast_std"],0)
