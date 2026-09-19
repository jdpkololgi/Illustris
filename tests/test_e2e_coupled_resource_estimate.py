import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from workflows.sbi import e2e_coupled_resource_estimate as cost


class ResourceProjectionTests(unittest.TestCase):
    def test_missing_measurements_cannot_publish_placeholder_costs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cost.coord,'ROOT',Path(tmp)):
            with self.assertRaises(FileNotFoundError): cost.run([1],[128],512)
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def fixtures(self):
        # Arithmetic-only fixture. These numbers are NOT measured GPU results.
        rows = {}
        for name in cost.gpu.CASES:
            parents = 2 if name in ('D_fine', 'I_fine') else 1
            rows[name] = dict(case=name,batch_pairs=2,synthetic_only=True,scientific_fit=False,
                parent_evaluations_per_pair=parents,seconds_per_training_pair=1.,
                host_to_device={'seconds_per_pair':.1},checkpoint_pack_write_fsync_hash_seconds=[1.,2.,3.],
                peak_gpu_reserved_bytes=100,parameters=10,
                sampling=[dict(nfe=n,seconds_per_pair=1.,network_evaluations_per_pair=n*parents)
                          for n in (64,128,256)],**{'pass':True})
        io = dict(full_panel_pairs=1664,binding=dict(phases=list(cost.c.TRAIN),batch_pairs=2),
                  measurements=[dict(pairs=26,seconds_per_pair_including_validation=v) for v in (.2,.3)])
        return rows,io

    def test_full_panel_shared_coarse_and_two_parent_counts(self):
        rows,io = self.fixtures()
        result = cost.project(rows,io,[1],[128],512)
        training = result['training'][0]
        self.assertEqual(training['distinct_factor_fits'],14)
        self.assertEqual(training['updates_per_factor'],832)
        self.assertEqual(training['checkpoint_writes_per_factor'],3)
        self.assertAlmostEqual(training['component_sum_gpu_hours'],
                               (2*1664*7*1.4+2*3*7*3)/3600)
        sample = result['confirmation_sampling'][0]
        self.assertEqual(sample['rectangular_fields'],98304)
        self.assertEqual(sample['fine_latents'],147456)
        self.assertEqual(sample['distinct_coarse_draws'],73728)
        self.assertEqual(sample['network_gpu_hours_by_nfe']['128'],96*2*128*7/3600)
        self.assertFalse(result['proposal_ready'])
        self.assertFalse(result['scientific_training_authorized'])

    def test_double_draws_doubles_sampling_not_training(self):
        rows,io = self.fixtures()
        result = cost.project(rows,io,[1,2],[128,256],512)
        a,b = result['confirmation_sampling']
        for key in ('rectangular_fields','fine_latents','distinct_coarse_draws','rectangular_float32_bytes'):
            self.assertEqual(b[key],2*a[key])
        self.assertEqual(result['training'][1]['updates_per_factor'],1664)

    def test_development_panels_charge_ladder_and_fine_only_controls(self):
        rows,io = self.fixtures()
        result = cost.project(rows,io,[1],[128],512)
        dev = result['development_sampling']
        progression,ladder,controls = dev['panels']
        self.assertEqual(progression['rectangular_fields'],24576)
        self.assertEqual(ladder['rectangular_fields'],12288)
        self.assertEqual(controls['rectangular_fields'],8192)
        self.assertEqual(controls['distinct_coarse_draws'],0)
        self.assertEqual(dev['rectangular_fields'],45056)
        self.assertEqual(dev['fine_latents'],67584)
        self.assertEqual(dev['distinct_coarse_draws'],27648)
        self.assertAlmostEqual(dev['network_gpu_hours_by_selected_nfe']['128'],
                               (32*2*32*3*7+16*2*32*3*7+16*2*32*2*4)/3600)
        self.assertFalse(dev['reuse_discount_applied'])

    def test_sampler_ladder_sums_all_nfe_costs_once(self):
        rows,io = self.fixtures()
        for row in rows.values():
            for sample in row['sampling']: sample['seconds_per_pair']=sample['nfe']/64
        dev=cost.project(rows,io,[1],[128],512)['development_sampling']
        ladder=dev['panels'][1]
        for nfe in ('64','128','256'):
            self.assertAlmostEqual(ladder['network_gpu_hours_by_selected_nfe'][nfe],
                                   16*2*32*7*(1+2+4)/3600)

    def test_checkpoint_count_changes_only_progression_panel(self):
        rows,io = self.fixtures()
        a=cost.project(rows,io,[1],[128],512,1)
        b=cost.project(rows,io,[1],[128],512,3)
        self.assertEqual(a['training'],b['training'])
        self.assertEqual(a['confirmation_sampling'],b['confirmation_sampling'])
        self.assertEqual(a['development_sampling']['panels'][1:],b['development_sampling']['panels'][1:])
        self.assertEqual(3*a['development_sampling']['panels'][0]['rectangular_fields'],
                         b['development_sampling']['panels'][0]['rectangular_fields'])
        with self.assertRaises(ValueError): cost.project(rows,io,[1],[128],512,0)

    def post_fixture(self):
        # Arithmetic fixture, NOT a measured CPU cost.
        names=('decode_seconds','common_tensor_seconds','owned_eigen_seconds',
               'fft_quantile_seconds','write_fsync_hash_verify_seconds')
        score_names=('density_calibration_seconds','eigen_gap_marginal_seconds',
                     'pointwise_vector_energy_seconds','adjacent_summary_scores_seconds')
        return dict(binding={'specification':cost.postprocess.specification()},
            synthetic_only=True,scientific_payloads_read=False,posterior_performance_evaluated=False,
            peak_host_rss_bytes=1,operator_measurements=[dict(index=i,coarse_mass_relative_error=0.,
                **dict.fromkeys(names,1.)) for i in range(8)],
            score_measurements=[dict(draws=m,owned_voxels=8192,vector_components=5,
                score_values_retained=False,**dict.fromkeys(score_names,1.)) for m in (32,128,256)],
            **{'pass':True})

    def test_postprocessing_counts_ensembles_and_names_size_proxies(self):
        rows,io=self.fixtures(); projected=cost.project(rows,io,[1],[128],512)
        result=cost.postprocess_projection(self.post_fixture(),projected['development_sampling'],
                                          projected['confirmation_sampling'])
        main=result['confirmation_panels'][0]
        self.assertEqual(main['field_decode_operator_io_seconds'],98304*6)
        self.assertEqual(main['owned_and_joint_score_seconds'],768*4)
        self.assertEqual(main['wide_scalar_score_size_proxy_seconds'],576*13.5)
        self.assertEqual(main['wide_fft_quantile_full_joint_size_proxy_seconds'],73728)
        self.assertAlmostEqual(main['four_cpu_step_hours'],674400/3600)
        self.assertFalse(result['end_to_end_analysis_measured'])

    def test_incomplete_or_unscientific_postprocessing_evidence_rejected(self):
        rows,io=self.fixtures(); p=cost.project(rows,io,[1],[128],512)
        for mutate in (lambda r:r['operator_measurements'].pop(),
                       lambda r:r['score_measurements'].pop(),
                       lambda r:r.update(posterior_performance_evaluated=True)):
            r=self.post_fixture(); mutate(r)
            with self.assertRaises(ValueError):
                cost.postprocess_projection(r,p['development_sampling'],p['confirmation_sampling'])

    def test_missing_factor_or_parent_accounting_rejected(self):
        rows,io = self.fixtures()
        bad=copy.deepcopy(rows); bad.pop('IJ_coarse')
        with self.assertRaises(ValueError): cost.project(bad,io,[1],[128],512)
        bad=copy.deepcopy(rows); bad['I_fine']['sampling'][0]['network_evaluations_per_pair']=64
        with self.assertRaises(ValueError): cost.project(bad,io,[1],[128],512)

    def test_partial_loader_nonfinite_cost_and_invalid_exposure_rejected(self):
        rows,io = self.fixtures()
        for epochs,draws,cadence in (([True],[128],512),([0],[128],512),([1],[128],0),([1,1],[128],512)):
            with self.assertRaises(ValueError): cost.project(rows,io,epochs,draws,cadence)
        bad=copy.deepcopy(rows); bad['J_fine']['seconds_per_training_pair']=float('nan')
        with self.assertRaises(ValueError): cost.project(bad,io,[1],[128],512)
        io['binding']['phases']=io['binding']['phases'][:-1]
        with self.assertRaises(ValueError): cost.project(rows,io,[1],[128],512)


if __name__ == '__main__': unittest.main()
