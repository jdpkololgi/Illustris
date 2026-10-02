"""One-time development selection and immutable production source snapshot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_exterior_campaign as campaign

ARMS = ('unconditional', 'conditional', 'learned')


def comparison(folder):
    records = [json.loads((folder/phase/'SCORES.json').read_text()) for phase in ('ph012', 'ph013')]
    if any(not r['panel_complete'] or len(r['cases']) != 16 for r in records):
        raise ValueError('Incomplete development comparison')
    scores = {}
    for name in ARMS:
        by_phase_cap = {}
        for phase, record in zip(('ph012', 'ph013'), records):
            for cap in ('NGC', 'SGC'):
                rows = [r for r in record['cases'] if r['cap'] == cap]
                if len(rows) != 8:
                    raise ValueError('Incomplete phase/cap stratum')
                by_phase_cap[phase+'_'+cap] = float(np.mean([v['crps'] for r in rows for v in r['methods'][name]['tidal']]))
        scores[name] = dict(crps=float(np.mean(list(by_phase_cap.values()))), by_phase_cap=by_phase_cap)
    return scores


def freeze(out):
    c.require_compute()
    if out.exists():
        raise ValueError('Selection freeze is immutable; do not overwrite it')
    root = campaign.ROOT
    primary, refined = comparison(root/'scores'), comparison(root/'scores_pad32_exact')
    method = min(ARMS, key=lambda name: primary[name]['crps'])
    refined_best = min(ARMS, key=lambda name: refined[name]['crps'])
    if refined_best != method:
        raise ValueError('Padding follow-up changes candidate ranking; review before freezing')
    checks_path = root/'verification/numerical_v1/CHECKS.json'
    checks = json.loads(checks_path.read_text())
    if not checks['numerical_passed'] or not checks['source_extent32_passed']:
        raise ValueError('Numerical development gate failed')
    smoke_path = root/'verification/loa_cap_completion_smoke_v3/VERIFIED.json'
    smoke = json.loads(smoke_path.read_text())
    if not smoke['passed'] or smoke['saved_interior_maxabs'] != 0:
        raise ValueError('Full-cap completion readback gate failed')
    benchmark_path = root/'verification/fullcap_benchmark_NGC/BENCHMARK.json'
    benchmark = json.loads(benchmark_path.read_text())
    if not benchmark['passed'] or benchmark['posterior_output']:
        raise ValueError('Missing targetless full-cap throughput benchmark')
    test_path = root/'verification/PRE_FREEZE_TESTS_v2.json'
    tested = json.loads(test_path.read_text())
    if not tested['passed']:
        raise ValueError('Focused test gate failed')
    for name, sha in tested['sources'].items():
        if c.sha256(c.REPO/name) != sha:
            raise ValueError('Tested source changed: '+name)
    config = json.loads(campaign.CONFIG.read_text())
    if config['preserved_sealed_reserve'] != ['ph001'] or 'ph001' in config['final_phase_inventory']:
        raise PermissionError('Sealed reserve policy changed')
    prior_path = root/'prior/PRIOR.json'
    prior = json.loads(prior_path.read_text())
    if prior['training_phases'] != list(c.TRAIN):
        raise ValueError('Prior training authority changed')
    original_binding = json.loads((root/'development/ph012/BINDING.json').read_text())
    names = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard',
                                     '--', 'workflows', 'shared', 'configs'], cwd=c.REPO).decode().split('\0')
    suffixes = {'.py', '.json', '.toml', '.yaml', '.yml', '.dat', '.txt', '.csv', '.sh', '.slurm'}
    names = sorted({name for name in names if name and Path(name).suffix in suffixes and (c.REPO/name).is_file()})
    for name in names:
        if (c.REPO/name).stat().st_size > 16*2**20:
            raise ValueError('Unexpected large source file requires review: '+name)
    sources = {name: c.sha256(c.REPO/name) for name in names}
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.REPO, text=True).strip()
    snapshot = root/('source_'+c.digest(sources)[:20])
    snapshot.mkdir(parents=True, exist_ok=False)
    bytes_copied = 0
    for name, sha in sources.items():
        source, destination = c.REPO/name, snapshot/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if c.sha256(destination) != sha or c.sha256(source) != sha:
            raise ValueError('Source changed during snapshot')
        destination.chmod(0o444)
        bytes_copied += destination.stat().st_size
    import torch, scipy, h5py, numba
    environment = dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                       scipy=scipy.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                       h5py=h5py.__version__, numba=numba.__version__)
    result = dict(schema='e2e-exterior-selection-freeze-v1', created_utc=datetime.now(timezone.utc).isoformat(),
                  method=method, selection_rule='lowest equal-phase equal-cap eigenvalue fair CRPS, primary padding16',
                  primary=primary, padding32_followup=refined, final_phases=config['final_phase_inventory'],
                  development_phases=['ph012', 'ph013'], fresh_confirmation_phases=['ph018', 'ph019'],
                  sealed_reserve=['ph001'], draws=32, nfe=128, fullcap_padding=32,
                  cap_joint_law='independent capwise modular approximation',
                  opposite_cap_observations_condition_exterior=False,
                  prior=c.file_record(prior_path, content_hash=True),
                  checkpoints=original_binding['checkpoints'], normalizer=original_binding['normalizer'],
                  sources=sources, source_snapshot=str(snapshot), source_bytes=bytes_copied,
                  git_head=revision, includes_uncommitted_task_sources=True, environment=environment,
                  evidence=[c.file_record(p, content_hash=True) for p in (checks_path, smoke_path, benchmark_path, test_path)],
                  posterior_qualified=False, fresh_scores_may_not_change_model=True,
                  qualification='failed development joint tidal calibration; final screening only',
                  numerical_limitations=['prior-domain law convergence not established',
                                         'native-source resolution convergence not established'],
                  jobs_may_start_only_after_each_targetless_cache_passes=True)
    c.atomic_json(out, result)
    out.chmod(0o444)
    c.atomic_json(snapshot/'SOURCE_MANIFEST.json', dict(git_head=revision, sources=sources, bytes=bytes_copied,
                                                      freeze=c.file_record(out, content_hash=True)))
    print(json.dumps(dict(method=method, snapshot=str(snapshot), sources=len(sources), bytes=bytes_copied,
                          final_phases=result['final_phases'], posterior_qualified=False)), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); freeze(a.out)
