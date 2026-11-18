import copy
import unittest
from workflows.sbi.adopt_shared_phase_base import physical_equivalence

class PhysicalReuse(unittest.TestCase):
    def test_only_policy_prose_may_differ(self):
        old=dict(target={'radius':7},geometry={'seed':41},observation={'cell':3.383,'selection_policy':'old'})
        new=copy.deepcopy(old);new['observation']['selection_policy']='new'
        physical_equivalence(old,new)
        for section,key,value in [('target','radius',4),('geometry','seed',42),('observation','cell',5)]:
            changed=copy.deepcopy(new);changed[section][key]=value
            with self.subTest(section=section),self.assertRaises(ValueError):physical_equivalence(old,changed)

if __name__=='__main__':unittest.main()
