import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from workflows.sbi.e2e_vdm_assessment import atomic_npz,read_chunk
from workflows.sbi.e2e_vdm_assessment_report import paired_change,summarize
from workflows.sbi import e2e_wide_pipeline as p,e2e_durable as durable


class AssessmentReportTests(unittest.TestCase):
    def test_paired_change(self):
        x=np.arange(1,65,dtype=float).reshape(16,4)
        r=paired_change(x,x*1.1)
        np.testing.assert_allclose(r['relative_mean_change'],.1)
        np.testing.assert_allclose(r['mc95'],.1,atol=1e-12)
    def test_committed_chunk_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=np.arange(40).reshape(4,10);path=atomic_npz(root,'sample',dict(delta=a))
            receipt=root/'receipt.json';bind=dict(ids=[0,1,2,3])
            durable.publish_json(receipt,dict(file=path.name,sha256=p.sha256(path),binding=bind))
            read_chunk(root,receipt,bind)
            np.testing.assert_array_equal(np.load(path)['delta'],a)
            with self.assertRaises(ValueError):read_chunk(root,receipt,dict(ids=[1,2,3,4]))
            with path.open('ab') as f:f.write(b'corruption')
            with self.assertRaises(ValueError):read_chunk(root,receipt,bind)
    def test_summary_schema_and_masks(self):
        rng=np.random.default_rng(3);truth=rng.normal(size=(48,48,48));mask=np.array([True]*4+[False]*4)
        boundaries=['periodic','zero','reflect'];features=rng.normal(size=(8,6))
        patch={b:features for b in boundaries};patch['regional']=rng.normal(size=8)
        values={b:features[None]+rng.normal(size=(8,8,6)) for b in boundaries}
        values.update(regional=rng.normal(size=(8,8)),power=rng.uniform(.1,1,size=(8,4)),
                      correlation=rng.uniform(0,1,size=(8,4)),onepoint=rng.normal(size=(8,16)))
        out=summarize(values,truth,features,patch,mask,dict(boundaries=boundaries,coverage_levels=[.5,.9]))
        self.assertEqual(out['tidal']['zero']['fullbox_reference']['gap32']['observed']['count'],4)
        self.assertEqual(out['tidal']['reflect']['matched_patch']['lambda1']['unobserved']['count'],4)
        self.assertEqual(len(out['onepoint_truth']),16)
        json.dumps(out,allow_nan=False)


if __name__=='__main__':unittest.main()
