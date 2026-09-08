import ast
from pathlib import Path
import unittest
import numpy as np
from workflows.sbi.p12f3_d2_native_support import NativeSupport,RULE,native_membership,require_native_record


class NativeSupportTests(unittest.TestCase):
    def test_boolean_native_guard(self):
        record=dict(galaxy_frac_index_local=np.zeros((2,3)),galaxy_native_support=np.array([True,True]),galaxy_support_rule=RULE)
        require_native_record(record)
        for value in (np.array([True,False]),np.array([1,1]),np.array([True])):
            with self.assertRaises(RuntimeError):require_native_record({**record,"galaxy_native_support":value})
        with self.assertRaises(RuntimeError):require_native_record({**record,"galaxy_support_rule":"unverified"})
        with self.assertRaises(RuntimeError):require_native_record({"galaxy_frac_index_local":np.zeros((2,3))})

    def test_empty_native_membership(self):
        self.assertEqual(native_membership(np.zeros((0,3)),1,None,None,None).shape,(0,))

    def test_native_supported_nearest_voxel_unsupported(self):
        obj=object.__new__(NativeSupport)
        obj.arrays=dict(core_active_offsets=np.array([0,1]),core_active_parent=np.array([12]),
            core_active_frac_index=np.array([[.1,.2,.3]]),core_voxel_start=np.zeros((1,3)),core_cap=np.array([1]))
        obj.manifest={"caps":{"NGC":{"origin_mpc":[200,0,0],"cell_mpc":5}}}
        obj.selection={"cosmology":{"radius_grid_mpc":[0,1000],"redshift_grid":[0,1]}}
        obj.angular=np.ones(12*256**2,dtype=bool);obj.domain=np.full(len(obj.angular),2,dtype=np.int8);obj.counts={}
        mask=np.zeros((3,3,3),dtype=bool)
        record=dict(galaxy_frac_index_local=np.array([[.1,.2,.3]]),core_bounds=np.array([[0,0,0],[3,3,3]]),support=mask)
        obj.verify_record(record,0,"candidate")
        self.assertTrue(record["galaxy_native_support"].all())
        self.assertEqual(obj.counts["candidate"][0],{"rows":1,"nearest_voxel_m0":1})
        self.assertIs(record["support"],mask)
        self.assertFalse(mask.any())
        changed={**record,"galaxy_frac_index_local":np.array([[.2,.2,.3]])}
        with self.assertRaises(RuntimeError):obj.verify_record(changed,0,"different")
        with self.assertRaises(RuntimeError):obj.verify_record(record,0,"candidate")

    def test_metric_functions_unchanged(self):
        root=Path(__file__).resolve().parents[2]
        old=Path("/global/u2/d/dkololgi/TNG/Illustris_d2_467f442/workflows/sbi/p12f3_d2_evaluate.py").read_text()
        new=(root/"workflows/sbi/p12f3_d2_evaluate_native_support.py").read_text()
        left={f.name:f for f in ast.parse(old).body if isinstance(f,ast.FunctionDef)}
        right={f.name:f for f in ast.parse(new).body if isinstance(f,ast.FunctionDef)}
        self.assertEqual(set(left),set(right))
        for name in left.keys()-{"main","parse_args","derived_physics_conditionals"}:
            self.assertEqual(ast.dump(left[name]),ast.dump(right[name]),name)
        old_function=ast.get_source_segment(old,left["derived_physics_conditionals"])
        old_function=old_function.replace(
            '        nearest = np.rint(coordinates).astype(np.int64)\n        if not np.all(support[tuple(nearest.T)]):\n            raise RuntimeError("D2 derived calibration includes an M=0 galaxy")',
            '        require_native_record(record)')
        new_function=ast.get_source_segment(new,right["derived_physics_conditionals"])
        self.assertEqual(ast.dump(ast.parse(old_function)),ast.dump(ast.parse(new_function)))


if __name__=="__main__":unittest.main()
