import unittest
import numpy as np
from workflows.sbi.p12a_galaxy_identity import identity_arrays


class GalaxyIdentityTests(unittest.TestCase):
    def table(self, n):
        x=np.zeros(n,dtype=[(k,'i8') for k in ('TARGETID','SOURCE_ROW','FILE_NUM','BOX_INDEX','HALO_INDEX','RES','CEN')]+[('HALO_MASS','f8')])
        x['TARGETID']=np.arange(n); x['SOURCE_ROW']=np.arange(n); x['HALO_MASS']=100
        return x

    def test_satellites_not_duplicates(self):
        x=self.table(3); x['BOX_INDEX']=-1; x['CEN']=[1,0,0]
        a,r=identity_arrays(x,np.ones(3,bool),[[10,0,0],[11,0,0],[12,0,0]])
        self.assertTrue(a['supervised_eligible'].all())
        self.assertEqual(len(np.unique(a['branch_local_host_group'])),1)
        self.assertFalse(r['fold_policy_qualified'])

    def test_res_namespaces_separate_hosts(self):
        x=self.table(2); x['BOX_INDEX']=-1; x['RES']=[0,1]
        a,_=identity_arrays(x,[True,True],[[10,0,0],[11,0,0]])
        self.assertEqual(len(np.unique(a['branch_local_host_group'])),2)

    def test_proven_periodic_copy_only(self):
        x=self.table(2)
        a,r=identity_arrays(x,[True,True],[[-1000,0,0],[1000,0,0]])
        self.assertEqual(a['supervised_eligible'].sum(),1)
        self.assertEqual(r['verified_high_branch_copy_rows_removed'],1)
        with self.assertRaises(ValueError): identity_arrays(x,[True,True],[[10,0,0],[11,0,0]])

    def test_low_branch_geometry_must_prove_no_copy(self):
        x=self.table(2); x['BOX_INDEX']=[-1,0]
        with self.assertRaises(ValueError): identity_arrays(x,[True,True],[[600,0,0],[1500,0,0]])

    def test_duplicate_source_rows_rejected(self):
        x=self.table(2); x['SOURCE_ROW']=0
        with self.assertRaises(ValueError): identity_arrays(x,[True,True],[[10,0,0],[11,0,0]])


if __name__=='__main__': unittest.main()
