import os
import unittest
from unittest.mock import patch
from workflows.sbi import lowz_unet_fit as fit


class LowzFitTests(unittest.TestCase):
    def test_actual_epoch_coverage(self):
        _,_,ids=fit.inputs()
        for epoch in (0,1,31):
            sequence=[address for i in range(epoch*480,(epoch+1)*480)
                      for address in fit.t.selected(ids,17,i)]
            self.assertCountEqual(sequence,ids)
        with self.assertRaises(ValueError):
            fit.validate_panel(ids[:-1]+ids[:1])
        with self.assertRaises(ValueError):
            fit.validate_panel(ids[:-1]+[('ph018','forbidden')])

    def test_spectral_average(self):
        import numpy as np
        import tempfile
        from pathlib import Path
        phases=['p'+str(i) for i in range(20)]
        ids=[(p,str(i)) for p in phases for i in range(48)]
        with tempfile.TemporaryDirectory() as tmp:
            for p in phases:
                for i in (0,24):
                    path=Path(tmp)/'data'/p/str(i)
                    path.mkdir(parents=True)
                    np.savez(path/'TARGETS.npz',residual=np.ones((4,4,4)))
            with patch.object(fit.t,'ROOT',Path(tmp)),patch.object(fit.t,'SHAPE',(4,4,4)),patch.object(fit.t.c,'TRAIN',phases),patch.object(fit.t,'spectral_weights',side_effect=lambda x,a:x),patch.object(fit.t.torch,'as_tensor',side_effect=lambda x,**kw:x):
                w,rows=fit.spectral_weight(ids,{'fine_residual':{'std':[1]}})
            self.assertEqual(len(set(rows)),40)
            self.assertAlmostEqual(w[0,0,0],64.)
