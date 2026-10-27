"""Prepare all visible corrected P12-A response inputs; never open reserved phases."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
from workflows.sbi.p12a_prepare_response_inputs import prepare, visible_roles, ROOT, CONTRACT
from workflows.abacus_tweb.p10_training_contract import atomic_json, sha256
from workflows.sbi.e2e_coupled_contract import require_compute


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4,choices=range(1,5))
    args=parser.parse_args()
    require_compute()
    _,phases=visible_roles(CONTRACT)
    output=ROOT/'p12a_response_inputs_20261026_v1'
    output.mkdir(exist_ok=True)
    results={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(prepare,phase):phase for phase in phases}
        for future in as_completed(futures):
            phase=futures[future]
            result=future.result()
            marker=Path(result['manifest'])
            if not result['pass'] or sha256(marker)!=result['manifest_sha256']:
                raise RuntimeError('Invalid response completion: '+phase)
            record=json.loads(marker.read_text())
            for key,path in record['arrays'].items():
                if sha256(Path(path))!=record['array_sha256'][key]:
                    raise RuntimeError('Response payload changed: '+phase+'/'+key)
            results[phase]={'parent_response':str(marker),'parent_response_sha256':sha256(marker),
                            'rows':record['rows'],'support_rows':record['support_rows']}
            atomic_json(output/'PROGRESS.json',{'complete':sorted(results),'required':phases})
            print(json.dumps({'completed_phase':phase,'complete':len(results),'required':len(phases)}),flush=True)
    if set(results)!=set(phases):raise RuntimeError('Incomplete response inventory')
    atomic_json(output/'adapter_inventory.json',{'schema_version':'p12a-visible-parent-response-v1',
        'pass':True,'full_contract_sha256':sha256(CONTRACT/'TRAINING_LOADER_READY.json'),
        'phases':results,'reserved_opened':False,'scientific_training':False,
        'scope':'exact parent-indexed support and boundary distance for P12-A'})

if __name__=='__main__':main()
