"""Receipt-backed unit costs and finite campaign projections, never a launcher.

Reports measured components separately from unmeasured analysis/startup/output
overheads. This is an input to the human-reviewed resource proposal, not training
authority, a convergence prediction, or a claim of end-to-end benchmark timing.
"""
import argparse
import json
import math
from pathlib import Path

from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_gpu_benchmark as gpu
from workflows.sbi import e2e_coupled_loader_benchmark as loader
from workflows.sbi import e2e_coupled_postprocess_benchmark as postprocess


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('finite positive measured cost required')
    return value


def positive_integer(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError('positive integer exposure/count required')
    return value


def development_projection(rates, checkpoints):
    """Bounded proposed development panels, not authorization or selected cases.

    Charge all repeated panels in full. In particular, do not assume the final
    checkpoint draws can be reused by a future sampler/control implementation.
    Fine-only controls reuse the frozen primary panel's sampled coarse mean or
    explicitly diagnostic truth; they do not generate another coarse ensemble.
    """
    checkpoints = positive_integer(checkpoints)
    fine = ('D_fine', 'I_fine', 'J_fine', 'CFM_fine')
    panels = [
        dict(name='checkpoint_progression', pairs=32, seeds=2, draws=32,
             repeats=checkpoints, factors=list(gpu.CASES), nfe_policy='selected'),
        dict(name='sampler_ladder', pairs=16, seeds=2, draws=32, repeats=3,
             factors=list(gpu.CASES), nfe_policy='all_64_128_256'),
        dict(name='fixed_mean_and_oracle_coarse_controls', pairs=16, seeds=2,
             draws=32, repeats=2, factors=list(fine), nfe_policy='selected')]
    for panel in panels:
        count = panel['pairs'] * panel['seeds'] * panel['draws']
        repetitions = count * panel['repeats']
        panel.update(rectangular_fields=4 * repetitions, fine_latents=6 * repetitions,
            distinct_coarse_draws=(3 * repetitions if len(panel['factors']) == 7 else 0))
        rates_by_nfe = {str(nfe): sum(rates[name]['sampling_network_seconds_per_pair'][str(nfe)]
                                    for name in panel['factors']) for nfe in (64, 128, 256)}
        panel['network_gpu_hours_by_selected_nfe'] = {
            str(nfe): (count * sum(rates_by_nfe.values()) if panel['nfe_policy']=='all_64_128_256'
                       else repetitions * rates_by_nfe[str(nfe)]) / 3600
            for nfe in (64, 128, 256)}
        panel['repeated_training_shape_transfer_proxy_gpu_hours'] = repetitions * sum(
            rates[name]['pageable_transfer_seconds_per_pair'] for name in panel['factors']) / 3600
    return dict(checkpoints=checkpoints, panels=panels,
        rectangular_fields=sum(panel['rectangular_fields'] for panel in panels),
        fine_latents=sum(panel['fine_latents'] for panel in panels),
        distinct_coarse_draws=sum(panel['distinct_coarse_draws'] for panel in panels),
        network_gpu_hours_by_selected_nfe={str(nfe): sum(
            panel['network_gpu_hours_by_selected_nfe'][str(nfe)] for panel in panels)
            for nfe in (64, 128, 256)},
        repeated_training_shape_transfer_proxy_gpu_hours=sum(
            panel['repeated_training_shape_transfer_proxy_gpu_hours'] for panel in panels),
        panel_selection='Freeze16 development pairs by phase/cap/redshift/support before predictions',
        fixed_mean_source='Empirical coarse mean from the frozen primary development ensemble',
        reuse_discount_applied=False, controls_are_diagnostic=True,
        scientific_training_authorized=False)


def postprocess_projection(record,development,confirmation):
    """Measured CPU components and explicitly identified conservative proxies."""
    if (not record['pass'] or not record['synthetic_only'] or record['scientific_payloads_read']
            or record['posterior_performance_evaluated']
            or record['binding']['specification']!=postprocess.specification()):
        raise ValueError('complete bounded synthetic postprocessing measurement required')
    operators=record['operator_measurements']; scores=record['score_measurements']
    if [row['index'] for row in operators]!=list(range(8)):
        raise ValueError('all eight synthetic operator measurements required')
    if any(row['coarse_mass_relative_error']>2e-6 for row in operators):
        raise ValueError('synthetic decode did not conserve coarse mass')
    names=('decode_seconds','common_tensor_seconds','owned_eigen_seconds',
           'fft_quantile_seconds','write_fsync_hash_verify_seconds')
    rates={name:max(positive(row[name]) for row in operators) for name in names}
    scored={row['draws']:row for row in scores}
    if set(scored)!={32,128,256} or len(scores)!=3:
        raise ValueError('full32/128/256-draw scoring measurements required')
    score_names=('density_calibration_seconds','eigen_gap_marginal_seconds',
                 'pointwise_vector_energy_seconds','adjacent_summary_scores_seconds')
    for row in scores:
        if row['owned_voxels']!=8192 or row['vector_components']!=5 or row['score_values_retained']:
            raise ValueError('scoring shape or technical-purpose mismatch')
        for name in score_names: positive(row[name])
    def cost(fields,coarse,draws):
        if fields%draws or coarse%draws: raise ValueError('whole posterior ensembles required')
        row=scored[draws]
        # Charge two full rectangular decodes for every arm; this is a proxy
        # allowance for independent-parent decode/assembly, not its measured IO.
        field_seconds=fields*(sum(rates.values())+rates['decode_seconds'])
        score_seconds=fields/draws*sum(row[name] for name in score_names)
        # Wide scalar calibration is a per-voxel operation at fixed draw count.
        # Explicitly label this size projection; it was not measured on wide arrays.
        wide_score_proxy=coarse/draws*(48**3/8192)*row['density_calibration_seconds']
        wide_fft_proxy=coarse*rates['fft_quantile_seconds']
        return dict(rectangular_fields=fields,distinct_coarse_draws=coarse,draws=draws,
            field_decode_operator_io_seconds=field_seconds,
            owned_and_joint_score_seconds=score_seconds,
            wide_scalar_score_size_proxy_seconds=wide_score_proxy,
            wide_fft_quantile_full_joint_size_proxy_seconds=wide_fft_proxy,
            four_cpu_step_hours=(field_seconds+score_seconds+wide_score_proxy+wide_fft_proxy)/3600)
    dev=[dict(name=p['name'],**cost(p['rectangular_fields'],p['distinct_coarse_draws'],p['draws']))
         for p in development['panels']]
    main=[cost(p['rectangular_fields'],p['distinct_coarse_draws'],p['draws_per_confirmation_pair'])
          for p in confirmation]
    return dict(rates_max_of_eight=rates,development_panels=dev,confirmation_panels=main,
        development_four_cpu_step_hours=sum(p['four_cpu_step_hours'] for p in dev),
        peak_process_rss_bytes=record['peak_host_rss_bytes'],
        assumptions=['Maximum observed component times, not a statistical worst-case guarantee',
            'One four-logical-CPU worker group; no unmeasured many-worker speedup',
            'Exclusive CPU-node-hours equal these step-hours if only one group uses the node',
            'Two rectangular decodes per field are a conservative independent-parent/assembly proxy',
            'Wide scalar score uses explicit voxel-count scaling; wide FFT uses full joint-size proxy',
            'Per-field IO includes a wide field each time; no shared-coarse IO discount'],
        additional_headroom_required=['Final band/window reductions and population calibration aggregation',
            'Bootstrap/reference-point diagnostics and output contention','GPU-to-host copies and orchestration'],
        end_to_end_analysis_measured=False)


def project(timings, io, epochs, draws, checkpoint_every_updates, development_checkpoints=3):
    if set(timings) != set(gpu.CASES):
        raise ValueError('all seven measured factors required; shared I/J coarse appears once')
    if (io['full_panel_pairs'] != 1664 or io['binding']['phases'] != list(c.TRAIN)
            or io['binding']['batch_pairs'] != 2 or len(io['measurements']) != 2):
        raise ValueError('complete thirteen-phase batch-two loader measurement required')
    for row in io['measurements']:
        if row['pairs'] != 26:
            raise ValueError('all26 registered loader probes required in each pass')
    read_seconds = max(positive(row['seconds_per_pair_including_validation'])
                       for row in io['measurements'])
    cadence = positive_integer(checkpoint_every_updates)
    epochs = [positive_integer(value) for value in epochs]
    draws = [positive_integer(value) for value in draws]
    if not epochs or not draws or len(set(epochs)) != len(epochs) or len(set(draws)) != len(draws):
        raise ValueError('nonempty distinct exposure and draw projections required')
    rates = {}
    for name in gpu.CASES:
        row = timings[name]
        parents = 2 if name in ('D_fine', 'I_fine') else 1
        if (row['case'] != name or row['batch_pairs'] != 2 or not row['pass']
                or not row['synthetic_only'] or row['scientific_fit']
                or row['parent_evaluations_per_pair'] != parents):
            raise ValueError('measured case/parent-count/purpose mismatch')
        measured = {entry['nfe']: entry for entry in row['sampling']}
        if set(measured) != {64, 128, 256} or len(row['sampling']) != 3:
            raise ValueError('complete measured NFE panel required')
        for nfe, entry in measured.items():
            if entry['network_evaluations_per_pair'] != nfe * parents:
                raise ValueError('independent-parent sampling accounting mismatch')
        transfer = positive(row['host_to_device']['seconds_per_pair'])
        checkpoints = row['checkpoint_pack_write_fsync_hash_seconds']
        if len(checkpoints) != 3:
            raise ValueError('all three checkpoint overhead measurements required')
        rates[name] = dict(
            gpu_training_seconds_per_pair=positive(row['seconds_per_training_pair']),
            pageable_transfer_seconds_per_pair=transfer,
            loader_seconds_per_pair=read_seconds,
            serial_training_seconds_per_pair=positive(row['seconds_per_training_pair']) + transfer + read_seconds,
            checkpoint_seconds=max(positive(value) for value in checkpoints),
            sampling_network_seconds_per_pair={str(nfe): positive(entry['seconds_per_pair'])
                                              for nfe, entry in measured.items()},
            parent_evaluations_per_pair=parents,
            peak_gpu_reserved_bytes=row['peak_gpu_reserved_bytes'], parameters=row['parameters'])
    training = []
    for exposure in epochs:
        # One epoch presents each of the1,664 paired domains once to each factor.
        # Each I/D update already includes both parent forwards/backwards.
        updates = 832 * exposure
        writes = 1 + math.ceil(updates / cadence)  # initial plus periodic/final
        components = {name: dict(
            serial_training_gpu_hours=2 * 1664 * exposure * rate['serial_training_seconds_per_pair'] / 3600,
            checkpoint_gpu_hours=2 * writes * rate['checkpoint_seconds'] / 3600)
            for name, rate in rates.items()}
        training.append(dict(epochs=exposure, updates_per_factor=updates,
            pair_presentations_per_factor=1664 * exposure, checkpoint_writes_per_factor=writes,
            distinct_factor_fits=14, factors=components,
            component_sum_gpu_hours=sum(sum(row.values()) for row in components.values())))
    sampling = []
    for count in draws:
        repetitions = 96 * 2 * count
        sampling.append(dict(draws_per_confirmation_pair=count,
            rectangular_fields=repetitions * 4, fine_latents=repetitions * 6,
            distinct_coarse_draws=repetitions * 3,
            rectangular_float32_bytes=repetitions * 4 * 64 * 48 * 48 * 4,
            unique_coarse_float32_bytes=repetitions * 3 * 48**3 * 4,
            network_gpu_hours_by_nfe={str(nfe): repetitions * sum(
                rate['sampling_network_seconds_per_pair'][str(nfe)] for rate in rates.values()) / 3600
                for nfe in (64, 128, 256)},
            repeated_training_shape_transfer_proxy_gpu_hours=repetitions * sum(
                rate['pageable_transfer_seconds_per_pair'] for rate in rates.values()) / 3600))
    development = development_projection(rates, development_checkpoints)
    return dict(rates=rates, training=training, confirmation_sampling=sampling,
        development_sampling=development,
        seeds=2, train_pairs=1664, confirmation_pairs=96, scientific_training_authorized=False,
        end_to_end_campaign_measured=False, proposal_ready=False,
        assumptions=['Serial reader plus transfer plus GPU update, with no prefetch/cache speedup',
            'Reader rate uses the slower of two measured26-pair pass averages, not a worst-case bound',
            'Per-pair D/I fine timings already charge both independent parents; no extra factor of two',
            'Shared I/J coarse fit and addressed coarse draws are counted once',
            'Sampling transfer is a conservative training-shape proxy, not a measured inference IO path',
            'Timing projections do not establish training convergence or sampler accuracy'],
        still_to_budget=['Inference condition IO, field decode, saved-draw IO and scientific diagnostics',
            'Allocation/process startup, warmup, bounded restart and scheduling headroom'])


def run(epochs, draws, checkpoint_every_updates, development_checkpoints=3):
    root = coord.ROOT / 'technical_gpu' / c.digest(gpu.binding())[:16]
    gpu_path = root / 'GPU_BENCHMARK_COMPLETE.json'
    io_path = coord.ROOT / 'technical_cpu/normalized_loader/LOADER_BENCHMARK_COMPLETE.json'
    post_path = (coord.ROOT/'technical_cpu/postprocess'/c.digest(postprocess.binding())[:16]
                 /'POSTPROCESS_BENCHMARK_COMPLETE.json')
    # Do not publish placeholder or extrapolated CPU-only timings.
    if not gpu_path.exists() or not io_path.exists() or not post_path.exists():
        raise FileNotFoundError('complete actual GPU, normalized-loader and CPU postprocessing measurements required')
    complete = coord.verify_receipt(gpu_path)
    if complete['binding'] != gpu.binding() or complete['cases'] != list(gpu.CASES):
        raise ValueError('GPU source/case panel differs from reviewed implementation')
    timings = {}
    sources = [c.file_record(gpu_path, content_hash=True), c.file_record(io_path, content_hash=True)]
    for item in complete['outputs']:
        case = coord.verify_receipt(item['path'])
        if case['binding'] != gpu.binding() or case['case'] in timings:
            raise ValueError('GPU case binding or duplication')
        saved = {Path(output['path']).name: coord.verify_receipt(output['path'], payload=False)
                 for output in case['outputs']}
        if (saved['TIMINGS.json'] != case['timing'] or not saved['REPLAY.json']['exact_fresh_process_replay']
                or saved['REPLAY.json']['binding'] != gpu.binding()):
            raise ValueError('GPU timings/replay evidence mismatch')
        timings[case['case']] = case['timing']
        sources.append(c.file_record(item['path'], content_hash=True))
    io = coord.verify_receipt(io_path, payload=False)
    normalizer = coord.ROOT / 'normalization/NORMALIZATION_COMPLETE.json'
    if io['binding'] != loader.binding(normalizer) or not io['pass']:
        raise ValueError('normalized-loader source/normalization drift')
    result = project(timings, io, epochs, draws, checkpoint_every_updates, development_checkpoints)
    post = coord.verify_receipt(post_path)
    if post['binding']!=postprocess.binding(): raise ValueError('CPU postprocessing source drift')
    result['cpu_postprocessing']=postprocess_projection(post,result['development_sampling'],
                                                       result['confirmation_sampling'])
    sources.append(c.file_record(post_path,content_hash=True))
    result['still_to_budget']=['GPU-to-host copies, final band/window and population diagnostics',
        'Output contention, process startup, warmup, bounded restart and scheduling headroom']
    result.update(**coord.provenance(), estimator_sha256=c.sha256(__file__), sources=sources)
    destination = coord.ROOT / 'resource_estimates' / (c.digest(dict(
        sources=sources, epochs=epochs, draws=draws, cadence=checkpoint_every_updates,
        development_checkpoints=development_checkpoints,
        estimator=result['estimator_sha256']))[:16] + '.json')
    if destination.exists():
        old = json.loads(destination.read_text())
        if any(old[key] != result[key] for key in (
                'rates', 'training', 'confirmation_sampling', 'development_sampling', 'cpu_postprocessing','sources')):
            raise ValueError('existing cost projection changed')
    else:
        c.atomic_json(destination, result)
    return dict(path=str(destination), training=result['training'],
                confirmation_sampling=result['confirmation_sampling'],
                development_sampling=result['development_sampling'],
                cpu_postprocessing=result['cpu_postprocessing'],proposal_ready=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, nargs='+', required=True)
    parser.add_argument('--draws', type=int, nargs='+', default=[128, 256])
    parser.add_argument('--checkpoint-every-updates', type=int, required=True)
    parser.add_argument('--development-checkpoints', type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args.epochs, args.draws, args.checkpoint_every_updates,
                         args.development_checkpoints), indent=2))
