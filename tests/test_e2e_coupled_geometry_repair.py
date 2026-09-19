import json
import unittest
import numpy as np
from workflows.sbi import e2e_coupled_geometry_repair as repair
from workflows.sbi import e2e_coupled_geometry as geometry


def row(cap,shell,x):
    return dict(phase='ph007',cap=cap,shell=shell,support_stratum='interior',
                source_midpoint_mpc_h=[x,500.,500.],
                source_owned_core_centers_mpc_h=[[x-54.128,500.,500.],[x+54.128,500.,500.]])


class GeometryRepairTests(unittest.TestCase):
    def test_original_success_is_preserved_exactly(self):
        buckets={('NGC',0,'interior'):[row('NGC',0,300.),row('NGC',0,600.)],
                 ('SGC',0,'interior'):[row('SGC',0,1000.)]}
        cfg=json.loads(repair.CONFIG.read_text())
        expected,_=repair.select_once(buckets,1,list(buckets))
        actual,attempts=repair.select(buckets,1,'ph007',cfg)
        self.assertEqual(expected,actual); self.assertEqual(len(attempts),1)

    def test_scarcity_repair_uses_same_candidates_and_rules(self):
        buckets={('NGC',0,'interior'):[row('NGC',0,300.),row('NGC',0,900.)],
                 ('SGC',0,'interior'):[row('SGC',0,300.)]}
        cfg=json.loads(repair.CONFIG.read_text())
        self.assertIsNone(repair.select_once(buckets,1,list(buckets))[0])
        rows,attempts=repair.select(buckets,1,'ph007',cfg)
        self.assertEqual(len(rows),2); self.assertEqual(attempts[-1]['mode'],'scarcity_original_order')
        self.assertTrue(geometry.admissible(rows[1]['source_midpoint_mpc_h'],
            np.asarray(rows[1]['source_owned_core_centers_mpc_h']),[rows[0]['source_midpoint_mpc_h']],
            rows[0]['source_owned_core_centers_mpc_h']))
        self.assertEqual(rows,repair.select(buckets,1,'ph007',cfg)[0])

    def test_infeasibility_never_relaxes_geometry(self):
        buckets={('NGC',0,'interior'):[row('NGC',0,300.)],
                 ('SGC',0,'interior'):[row('SGC',0,300.)]}
        rows,attempts=repair.select(buckets,1,'ph007',json.loads(repair.CONFIG.read_text()))
        self.assertIsNone(rows); self.assertEqual(len(attempts),18)
        self.assertFalse(any(v['success'] for v in attempts))


if __name__=='__main__': unittest.main()
