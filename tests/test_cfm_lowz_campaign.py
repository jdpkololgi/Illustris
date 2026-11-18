import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from workflows.sbi import prepare_cfm_lowz_campaign as campaign
from workflows.sbi import e2e_wholefield_pilot_data as whole


class LowzCampaignTests(unittest.TestCase):
    def test_authority_preserves_roles_and_physics(self):
        cfg=json.loads((campaign.c.REPO/'configs/e2e_coupled_data_lowz_20261029.json').read_text())
        with patch.object(campaign.c,'config',return_value=cfg),patch.object(campaign.c,'SHARED_POLICY',True):
            campaign.authority()
        wrong=copy.deepcopy(cfg);wrong['phase_roles']['ph006']='train'
        with patch.object(campaign.c,'config',return_value=wrong),patch.object(campaign.c,'SHARED_POLICY',True):
            with self.assertRaises(ValueError):campaign.authority()

    def test_wholefield_reads_corrected_expectations(self):
        shape=(8,8,8);data={key:np.ones(shape,'f4') for key in whole.products.LOCAL_CHANNELS
                              if key!='observer_redshift'}
        data['expected_counts_random'][:]=7.
        data['exposure_apodized_random'][:]=.2
        grid=dict(origin_mpc_h=[10.,20.,30.],shape=list(shape),cell_mpc_h=3.383,
                  distance_unit='Mpc/h',coordinate_sha256=campaign.c.sha256(campaign.coord.CONFIG))
        with patch.object(whole,'SHAPE',(2,2,2)):
            values=whole.build_lowz_fine_observations(grid,[4,4,4],data)
        self.assertEqual(values.shape,(12,2,2,2))
        index=whole.products.LOCAL_CHANNELS.index('expected_counts_random')
        np.testing.assert_allclose(values[index],np.log(8.),rtol=1e-6)
        np.testing.assert_allclose(values[whole.products.LOCAL_CHANNELS.index('exposure_apodized_random')],.2)

    def test_lowz_shell_and_both_caps_required(self):
        rows=[dict(cap=cap,redshift=.1,shell=0) for cap in ('NGC','SGC')]
        with patch.object(campaign.geometry,'verify_rows',return_value={'source_nonoverlap':True}):
            self.assertTrue(campaign.verify_lowz_rows(rows,'ph001')['lowz_midpoints_in_both_caps'])
            with self.assertRaises(ValueError):campaign.verify_lowz_rows(rows[:1],'ph001')
            rows[0]['shell']=1
            with self.assertRaises(ValueError):campaign.verify_lowz_rows(rows,'ph001')

    def test_panel_fails_instead_of_relaxing_nonoverlap(self):
        row=dict(source_midpoint_mpc_h=[0,0,0],source_owned_core_centers_mpc_h=[[0,0,0],[100,0,0]])
        with patch.dict(campaign.c.ROLES,{'ph006':'development'}):
            with self.assertRaises(ValueError):
                campaign.select_panel('ph006',{('NGC',0,'boundary'):[row],('SGC',0,'boundary'):[row]})


if __name__=='__main__':unittest.main()
