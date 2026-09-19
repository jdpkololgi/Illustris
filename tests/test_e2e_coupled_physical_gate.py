import json
import unittest
import numpy as np
from workflows.sbi import e2e_coupled_physical_gate as gate


class PhysicalGateTests(unittest.TestCase):
    def cases(self):
        rows=[]
        for phase in ('ph007','ph008'):
            for cap in ('NGC','SGC'):
                for shell in range(4):
                    for kind in ('interior','boundary'):
                        rows.append(dict(phase=phase,cap=cap,shell=shell,support_stratum=kind,
                            offset_raw=[0,0,0],operators={name:dict(improvement=[[.3,.4,.5],[.3,.4,.5]])
                                                        for name in ('independent','joint')}))
        return rows

    def test_gate_cannot_pool_away_failure_or_missing_strata(self):
        cfg=json.loads(gate.CONFIG.read_text()); cases=self.cases()
        self.assertTrue(gate.decide(cases,cfg)['pass'])
        with self.assertRaises(ValueError): gate.decide(cases[:-1],cfg)
        for row in cases:
            row['operators']['joint']['improvement'][0][2]=.1
            row['operators']['joint']['improvement'][1][2]=.1
        self.assertFalse(gate.decide(cases,cfg)['pass'])

    def test_nonprimary_augmentation_cannot_rescue_primary(self):
        cfg=json.loads(gate.CONFIG.read_text()); cases=self.cases()
        for row in cases: row['operators']['independent']['improvement']=[[.1]*3]*2
        extra=[dict(row,offset_raw=[32,0,0],operators={name:dict(improvement=[[.99]*3]*2)
               for name in ('independent','joint')}) for row in cases]
        self.assertFalse(gate.decide(cases+extra,cfg)['pass'])

    def test_ordered_eigenvalues_and_component_layout(self):
        tensor=np.array([[[1.,0.,0.,2.,0.,3.]]])
        np.testing.assert_array_equal(gate.eigenvalues(tensor),[[[1.,2.,3.]]])


if __name__=='__main__': unittest.main()
