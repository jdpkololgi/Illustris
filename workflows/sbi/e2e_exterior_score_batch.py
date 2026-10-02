"""Frozen, bounded final scoring after all field production completes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_exterior_campaign as campaign
from workflows.sbi.e2e_exterior_batch import check_freeze


def main():
    p = argparse.ArgumentParser(); p.add_argument('--freeze', required=True, type=Path)
    a = p.parse_args(); c.require_compute(); freeze = check_freeze(a.freeze)
    root = campaign.ROOT; logs = root/'final_score_logs'; logs.mkdir(exist_ok=False)
    def score(phase):
        with (logs/(phase+'.log')).open('x', buffering=1) as log:
            subprocess.run([sys.executable, '-u', '-m', 'workflows.sbi.e2e_exterior_final_score',
                '--phase', phase, '--samples', str(root/'fullcap_tides'), '--freeze', str(a.freeze),
                '--out', str(root/'final_scores'/phase)], stdout=log, stderr=log, check=True)
        path = root/'final_scores'/phase/'SCORES.json'; record = json.loads(path.read_text())
        if not record['panel_complete'] or len(record['cases']) != 16:
            raise ValueError('Incomplete phase scoring: '+phase)
        print(json.dumps(dict(phase=phase, scored=True)), flush=True)
        return c.file_record(path, content_hash=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(score, freeze['final_phases']))
    c.atomic_json(root/'FULL_EVALUATION_COMPLETE.json', dict(passed=True, scores=records,
        freeze=c.file_record(a.freeze, content_hash=True), posterior_qualified=False,
        note='Completion is not qualification; inspect all marginal and joint metrics.'))


if __name__ == '__main__':
    main()
