"""Measure the actual normalized training reader, without any model fitting.

The reference path includes receipt checks, payload checksums, HDF5 reads and
normalization. It deliberately does not claim a cache or asynchronous pipeline
speedup. Two passes distinguish first-observed from repeated-read throughput;
neither pass is described as an OS-cold-cache measurement.
"""
import json
from pathlib import Path
import resource
import time

import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_normalization as norm
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi import e2e_coupled_audit_worker as audits
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_conditions as conditions


def phase_cases(phase):
    # Reject non-training roles before inspecting receipts or targets.
    if phase not in c.TRAIN:
        raise PermissionError('loader throughput probes are train-only')
    if not audits.qualified(phase):
        raise FileNotFoundError('independent training phase audit required')
    source = coord.ROOT / 'conditions' / phase / 'CONDITIONS_COMPLETE.json'
    record = coord.verify_receipt(source, payload=False)
    ids = sorted(Path(item['path']).stem for item in record['pair_receipts'])
    if len(ids) != 128 or len(set(ids)) != 128:
        raise ValueError('incomplete training pair panel')
    offsets = op.layout()['context_offsets_raw']
    index = c.TRAIN.index(phase)
    return [dict(phase=phase, pair_id=ids[i], offset=offsets[(index+j) % len(offsets)])
            for j, i in enumerate((0, 64))]


def collate(pairs):
    """Measured CPU collation only; no claimed GPU transfer or prefetch overlap."""
    import torch
    if len(pairs) != 2:
        raise ValueError('registered batch is two paired domains')
    result = []
    for part in (0, 1):
        keys = set(pairs[0][part])
        if set(pairs[1][part]) != keys:
            raise ValueError('batch fields differ')
        result.append({key: torch.from_numpy(np.stack([pair[part][key] for pair in pairs]))
                       for key in sorted(keys)})
    return tuple(result)


def binding(normalizer):
    return dict(normalization_sha256=c.sha256(normalizer),
                source_hashes={Path(module.__file__).name: c.sha256(module.__file__)
                               for module in (views, conditions, norm, op)},
                runner_sha256=c.sha256(__file__), batch_pairs=2, passes=2,
                phases=list(c.TRAIN), scientific_fit=False)


def run():
    c.require_compute()
    normalizer = coord.ROOT / 'normalization' / 'NORMALIZATION_COMPLETE.json'
    if not normalizer.exists():
        raise FileNotFoundError('complete thirteen-phase normalizer required')
    # Verifies cached global statistics and their exact source bindings.
    norm.fit()
    bound = binding(normalizer)
    cases = [case for phase in c.TRAIN for case in phase_cases(phase)]
    directory = coord.ROOT / 'technical_cpu' / 'normalized_loader'
    marker = directory / 'LOADER_BENCHMARK_COMPLETE.json'
    with c.single_writer(directory):
        if marker.exists():
            previous = coord.verify_receipt(marker, payload=False)
            if previous['binding'] != bound or previous['cases'] != cases:
                raise ValueError('completed loader benchmark binding changed')
            return previous
        import torch
        torch.set_num_threads(4)
        sources = [c.file_record(normalizer, content_hash=True)]
        for phase in c.TRAIN:
            for kind, name in [('conditions', 'CONDITIONS_COMPLETE.json'),
                               ('targets', 'TARGETS_COMPLETE.json'),
                               ('product_audit', 'LATEST_AUDIT.json')]:
                sources.append(c.file_record(coord.ROOT / kind / phase / name, content_hash=True))
        results = []
        for repeat in range(2):
            loaded = []; load_times = []; collation_times = []; input_bytes = []
            start = time.monotonic()
            for case in cases:
                before = time.monotonic()
                pair = views.load_training_pair(case['phase'], case['pair_id'],
                                               bound['normalization_sha256'], case['offset'])
                load_times.append(time.monotonic() - before)
                if any(not np.isfinite(value).all() for part in pair for value in part.values()):
                    raise ValueError('nonfinite actual training-reader output')
                input_bytes.append(sum(value.nbytes for part in pair for value in part.values()))
                loaded.append(pair)
                if len(loaded) == 2:
                    before = time.monotonic()
                    batch = collate(loaded)
                    collation_times.append(time.monotonic() - before)
                    del batch
                    loaded.clear()
                del pair
            if loaded:
                raise ValueError('unpaired final loader presentation')
            elapsed = time.monotonic() - start
            results.append(dict(pass_index=repeat, pairs=len(cases), elapsed_seconds=elapsed,
                seconds_per_pair_including_validation=elapsed / len(cases),
                strict_reader_seconds=load_times, collation_seconds_per_batch=collation_times,
                input_bytes_per_pair=input_bytes))
        result = dict(**coord.provenance(), binding=bound, cases=cases, sources=sources,
            measurements=results, full_panel_pairs=sum(128 for _ in c.TRAIN),
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            numpy_version=np.__version__, torch_version=str(torch.__version__),
            os_caches_flushed=False, persistent_cache_tested=False, gpu_transfer_tested=False,
            asynchronous_prefetch_tested=False, science_scores_evaluated=False,
            scientific_training_authorized=False, full_preparation_complete=False,
            outputs=[], **{'pass': True})
        c.atomic_json(marker, result)
        return result


if __name__ == '__main__':
    result = run()
    print(json.dumps(dict(measurements=result['measurements'],
                          peak_host_rss_bytes=result['peak_host_rss_bytes'])), flush=True)
