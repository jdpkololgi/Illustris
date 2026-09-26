import json
import tempfile
import unittest
from pathlib import Path
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi.e2e_cfm_additional_report import report

class AdditionalReportTests(unittest.TestCase):
    def test_complete_and_incomplete_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for seed in (17,29):
                for panel,phase in [('additional16','ph016'),('additional17','ph017')]:
                    worker=root/panel/f'seed{seed}_step13312_fine26624';worker.mkdir(parents=True)
                    binding=dict(step=13312,fine_step=26624,weights='ema',evaluation_phases=[phase],seed=seed)
                    (worker/'BINDING.json').write_text(json.dumps(binding));cases=[]
                    for i in range(17):
                        folder=worker/f'case{i}';folder.mkdir();cases.append(str(folder))
                        group=dict(posterior_variance=1,mean_squared_error=1,bias=0,crps=.5,coverage={'0.9':.875})
                        scores={g:group for g in ('core_mass','block_mass','fine_mass','tidal','eigengap')}
                        scores['spectra']=[dict(sample_power=1)]*5
                        value=dict(binding=c.digest(binding),phase=phase,pair_id=str(i),nfe=128 if i<16 else 256,
                            draws=32 if i<16 else 8,chunks={},scores=scores,paired_field_rms=0,paired_base8_scores=scores)
                        (folder/'COMPLETE.json').write_text(json.dumps(value))
                    (worker/'COMPLETE.json').write_text(json.dumps(dict(binding=c.digest(binding),cases=cases)))
            report(root)
            result=json.loads((root/'COMPLETE.json').read_text())
            self.assertEqual(len(result['rows']),64)
            self.assertEqual(result['summaries']['ph016']['fine_mass']['rmse_over_spread'],1)
            (worker/'COMPLETE.json').write_text(json.dumps(dict(binding=c.digest(binding),cases=cases[:-1])))
            with self.assertRaises(ValueError):report(root)
