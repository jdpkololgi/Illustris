"""Native Loa density/eigenvalue uncertainty summaries, without model changes.

This postprocessor consumes the immutable 32-draw ensembles. Its intervals use
the same finite-ensemble order statistics as the frozen evaluation, not a new
Gaussian approximation. Stored uncertainty is conditional-model uncertainty,
not a claim of calibrated DESI or simultaneous field coverage.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

ROOT = Path('/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coherent_exterior_20261002_v1')
LEVELS = (.5, .68, .9, .95)


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def marginal_summary(draws, intervals):
    draws = np.asarray(draws)
    if len(draws) != 32 or not np.isfinite(draws).all():
        raise ValueError('Exactly 32 finite draws are required')
    ordered = np.sort(draws, axis=0)
    return dict(mean=draws.mean(0, dtype='f8').astype('f4'),
                std=draws.std(0, dtype='f8', ddof=1).astype('f4'),
                lower=np.stack([ordered[lo] for lo, hi, reference in intervals]),
                upper=np.stack([ordered[hi] for lo, hi, reference in intervals]))


def classes(eigen, threshold):
    labels = np.count_nonzero(eigen > threshold, axis=-1).astype('u1')
    probability = np.stack([(labels == cls).mean(0) for cls in range(4)], axis=-1).astype('f4')
    entropy = -np.sum(probability*np.log2(np.maximum(probability, 1/32)), axis=-1)/2
    if not np.array_equal(probability.sum(-1), np.ones(probability.shape[:-1], dtype='f4')):
        raise ValueError('Class probabilities do not sum to one')
    return probability, entropy


def stage():
    source = Path(__file__).resolve(); script = source.with_name('submit_e2e_exterior_field_summary.slurm')
    identities = {path.name: sha(path) for path in (source, script)}
    digest = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    destination = ROOT/('field_summary_source_'+digest[:20]); destination.mkdir(exist_ok=False)
    for path in (source, script):
        target = destination/path.name; shutil.copyfile(path, target)
        assert sha(target) == identities[path.name] and sha(path) == identities[path.name]
        target.chmod(0o444)
    record = dict(source=str(destination), sources=identities,
                  science_freeze_sha256=sha(ROOT/'SELECTION_FREEZE.json'),
                  purpose='Derived Loa uncertainty products only; no fitting or metric changes')
    manifest = destination/'POSTPROCESSOR.json'
    with manifest.open('x') as stream:
        stream.write(json.dumps(record, indent=2)+'\n')
    manifest.chmod(0o444)
    print(json.dumps(dict(source=str(destination), source_sha256=identities[source.name])), flush=True)


def run(freeze_path, source_sha):
    if sha(__file__) != source_sha:
        raise ValueError('Postprocessor source changed')
    selected = json.loads(freeze_path.read_text()); freeze_sha = sha(freeze_path)
    sys.path.insert(0, selected['source_snapshot'])
    from workflows.sbi import e2e_coupled_contract as c
    from workflows.sbi.e2e_exterior_batch import check_freeze
    from workflows.sbi.e2e_vdm_context_metrics import central_order_interval
    c.require_compute(); check_freeze(freeze_path)
    assert selected['draws'] == 32 and selected['sealed_reserve'] == ['ph001']
    intervals = [central_order_interval(32, level) for level in LEVELS]
    outputs = []
    for cap in ('NGC', 'SGC'):
        inputs = []
        for first in range(0, 32, 4):
            directory = ROOT/'fullcap_tides'/'Loa'/cap/f'd{first:03d}'
            record = json.loads((directory/'COMPLETE.json').read_text())
            verified = json.loads((directory/'VERIFIED.json').read_text())
            binding = json.loads((directory/'BINDING.json').read_text())
            assert record['passed'] and not record['posterior_qualified']
            assert (record['phase'], record['cap'], record['first'], record['draws']) == ('Loa', cap, first, 4)
            assert record['method'] == selected['method'] and binding['freeze_sha256'] == freeze_sha
            assert record['binding'] == sha(directory/'BINDING.json')
            assert verified['passed'] and verified['complete_receipt']['sha256'] == sha(directory/'COMPLETE.json')
            assert verified['verifier_sha256'] == selected['sources']['workflows/sbi/e2e_exterior_cap_verify.py']
            path = Path(record['output']['path'])
            assert path.resolve() == (directory/'COMPLETED_FIELD.h5').resolve()
            assert sha(path) == record['output']['sha256']
            inputs.append(dict(path=str(path), sha256=record['output']['sha256'],
                               complete_sha256=sha(directory/'COMPLETE.json'), first=first))
        out = ROOT/'loa_field_summary'/cap; out.mkdir(parents=True, exist_ok=False)
        path = out/'DENSITY_TIDAL_SUMMARY.h5'
        with ExitStack() as stack:
            sources = [stack.enter_context(h5py.File(item['path'], rdcc_nbytes=128*2**20)) for item in inputs]
            shape = sources[0]['support'].shape; origin = sources[0].attrs['origin_mpc_h']
            support = sources[0]['support'][:]; coverage = sources[0]['coverage'][:]
            for f in sources:
                assert f['eigenvalues'].shape == (4, *shape, 3)
                assert np.array_equal(f.attrs['origin_mpc_h'], origin)
                assert np.array_equal(f['support'][:], support) and np.array_equal(f['coverage'][:], coverage)
                assert f.attrs['padding_cells'] == selected['fullcap_padding'] and f.attrs['cell_mpc_h'] == 6.766
            target = stack.enter_context(h5py.File(path, 'x'))
            target.attrs.update(cap=cap, phase='Loa', draws=32, cell_mpc_h=6.766, origin_mpc_h=origin,
                                nominal_levels=LEVELS, order_lower=[v[0] for v in intervals],
                                order_upper=[v[1] for v in intervals], attainable_coverage=[v[2] for v in intervals],
                                std_ddof=1, posterior_qualified=False, source_sha256=source_sha,
                                science_freeze_sha256=freeze_sha, eigenvalue_order='ascending',
                                interval_scope='pointwise model intervals, not simultaneous field bands',
                                cap_joint_law=selected['cap_joint_law'])
            target.create_dataset('support', data=support, compression='lzf')
            target.create_dataset('coverage', data=coverage, compression='lzf')
            groups = {}
            chunks = tuple(min(n, m) for n, m in zip(shape, (8, 32, 32)))
            for name, tail in (('density', ()), ('eigenvalues', (3,))):
                group = target.create_group(name); groups[name] = group
                for statistic in ('mean', 'std'):
                    group.create_dataset(statistic, shape=(*shape, *tail), dtype='f4', chunks=(*chunks, *tail), compression='lzf')
                for statistic in ('lower', 'upper'):
                    group.create_dataset(statistic, shape=(4, *shape, *tail), dtype='f4', chunks=(1, *chunks, *tail), compression='lzf')
            probability = target.create_dataset('environment_probability', shape=(2, *shape, 4), dtype='f4',
                                                 chunks=(1, *chunks, 4), compression='lzf')
            entropy = target.create_dataset('environment_entropy', shape=(2, *shape), dtype='f4',
                                             chunks=(1, *chunks), compression='lzf')
            probability.attrs['tidal_thresholds'] = [0., .2]; entropy.attrs['normalization'] = 'Shannon entropy / log(4)'
            padding = selected['fullcap_padding']; trace_maxabs = 0.
            for lo in range(0, shape[0], 16):
                hi = min(lo+16, shape[0]); sl = slice(lo, hi)
                rho = np.concatenate([f['rho_completed'][:, padding+lo:padding+hi,
                                           padding:padding+shape[1], padding:padding+shape[2]] for f in sources])
                eigen = np.concatenate([f['eigenvalues'][:, sl] for f in sources])
                assert np.all(rho > 0) and np.all(np.diff(eigen, axis=-1) >= 0)
                summaries = dict(density=marginal_summary(rho, intervals), eigenvalues=marginal_summary(eigen, intervals))
                for name, summary in summaries.items():
                    for statistic, value in summary.items():
                        if statistic in ('mean', 'std'):
                            groups[name][statistic][sl] = value
                        else:
                            groups[name][statistic][:, sl] = value
                        assert np.isfinite(value).all()
                error = np.max(abs(summaries['eigenvalues']['mean'].sum(-1)-(summaries['density']['mean']-1)))
                trace_maxabs = max(trace_maxabs, float(error))
                assert error <= 2e-6*max(1., float(rho.max()))
                for index, threshold in enumerate((0., .2)):
                    pclass, hclass = classes(eigen, threshold)
                    probability[index, sl] = pclass; entropy[index, sl] = hclass
                print(json.dumps(dict(cap=cap, summarized_rows=hi, total_rows=shape[0])), flush=True)
        record = dict(passed=True, cap=cap, draws=32, source_sha256=source_sha, science_freeze_sha256=freeze_sha,
                      inputs=inputs, output=dict(path=str(path), sha256=sha(path)), posterior_qualified=False,
                      nominal_levels=LEVELS, attainable_coverage=[v[2] for v in intervals],
                      std_ddof=1, posterior_mean_trace_maxabs=trace_maxabs,
                      intervals='Same central finite-ensemble order statistics as frozen scoring',
                      scope='Native cellwise conditional-model summaries, not DESI calibration or simultaneous bands',
                      job=os.environ['SLURM_JOB_ID'])
        with (out/'COMPLETE.json').open('x') as stream:
            stream.write(json.dumps(record, indent=2)+'\n')
        outputs.append(dict(path=str(out/'COMPLETE.json'), sha256=sha(out/'COMPLETE.json')))
    with (ROOT/'loa_field_summary'/'COMPLETE.json').open('x') as stream:
        stream.write(json.dumps(dict(passed=True, caps=outputs, source_sha256=source_sha,
                                     science_freeze_sha256=freeze_sha, posterior_qualified=False), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', action='store_true')
    p.add_argument('--freeze', type=Path); p.add_argument('--source-sha'); a = p.parse_args()
    if a.stage:
        stage()
    else:
        if a.freeze is None or a.source_sha is None:
            p.error('--freeze and --source-sha are required for processing')
        run(a.freeze, a.source_sha)
