"""Reuse label-independent P2 arrays only when corrected P1 geometry is exact."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from workflows.abacus_tweb.audit_galaxy_truth_replacement import compare_p1
from workflows.abacus_tweb.p10_phase_assets import load_registry, sha256_file
from workflows.abacus_tweb.p10_validate_phase_products import validate_p2


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True)
    ap.add_argument('--registry',type=Path)
    ap.add_argument('--source-root',type=Path)
    args=ap.parse_args()
    repo=Path(__file__).resolve().parents[2]
    registry=load_registry(args.registry or repo/'configs/p10_phase_registry_v2.json')
    shared_policy=bool(registry.get('phase_policy'))
    if shared_policy:
        from shared.phase_policy import check_preparation_phase
        check_preparation_phase(registry,args.phase)
        source_root=Path(registry['reuse_source_root'])
        if args.source_root and args.source_root.resolve()!=source_root.resolve():
            raise ValueError('Source root differs from the versioned reuse contract')
    else:
        if args.phase not in ('ph002','ph003','ph004','ph005','ph006') or args.source_root:
            raise ValueError('No legacy geometry reuse for other phases or roots')
        source_root=Path('/pscratch/sd/d/dkololgi/abacus/p10_multiphase')
    old=source_root/args.phase
    new=Path(registry['path_templates']['phase_output'].format(phase=args.phase))
    checks=compare_p1(old/'p1_canonical',new/'p1_canonical')
    required=('parent_node_id','targetid','cap','shell','context','points')
    if not shared_policy: required+=('active',)
    if not all(checks[k] for k in required):
        raise ValueError('Geometry or selection changed; build fresh P2')
    prior=json.loads((old/'p2_graph/P2_COMPLETE.json').read_text())
    graph=json.loads((old/'p2_graph/GRAPH_COMPLETE.json').read_text())
    union=json.loads((old/'p2_union/p2b_union_manifest.json').read_text())
    gnn=json.loads(Path(prior['inputs']['gnn_metadata']).read_text())
    if not prior['pass'] or not graph['pass']:raise ValueError('Original P2 incomplete')
    hashes={r['path']:r['sha256'] for r in graph['artifacts'].values()}
    hashes[gnn['outputs']['gnn_arrays_npz']]=prior['hashes']['gnn_arrays']
    for record in union['components'].values():
        for kind in ('pairs','edge_attr'):hashes[record[kind]]=record[kind+'_sha256']
    for path,expected in hashes.items():
        if sha256_file(Path(path))!=expected:raise ValueError('Original P2 payload changed: '+path)
    def rebind(value):
        if isinstance(value,str):return value.replace(str(old),str(new))
        if isinstance(value,list):return [rebind(x) for x in value]
        if isinstance(value,dict):return {k:rebind(v) for k,v in value.items()}
        return value
    for name in ('p2_graph','p2_union'):
        dest=new/name
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir()
        for source in (old/name).iterdir():
            target=dest/source.name
            if source.name=='P2_COMPLETE.json':continue
            if source.suffix=='.json':
                payload=rebind(json.loads(source.read_text()))
                if source.name=='GRAPH_COMPLETE.json':
                    payload['p1_index_sha256']=sha256_file(new/'p1_canonical/canonical_index.npz')
                    payload['p1_points_sha256']=sha256_file(new/'p1_canonical/points.npy')
                    payload['metadata_sha256']=None  # filled after all metadata is written
                target.write_text(json.dumps(payload,indent=2)+'\n')
            else:target.symlink_to(source,target_is_directory=source.is_dir())
    marker=new/'p2_graph/GRAPH_COMPLETE.json';payload=json.loads(marker.read_text())
    payload['metadata_sha256']=sha256_file(Path(payload['metadata']))
    marker.write_text(json.dumps(payload,indent=2)+'\n')
    report=validate_p2(registry,args.phase)
    (new/'P2_GEOMETRY_REUSED.json').write_text(json.dumps(dict(
        phase=args.phase,original=str(old),geometry_checks=checks,
        active_mask_change_requires_fresh_p4=not checks['active'],
        phase_policy=registry.get('phase_policy'),
        original_p2_sha256=sha256_file(old/'p2_graph/P2_COMPLETE.json'),
        corrected_p1_sha256=sha256_file(new/'p1_canonical/manifest.json'),
        payload_sha256=hashes,**{'pass':report['pass']}),indent=2)+'\n')
    print(args.phase,'P2 reused',flush=True)

if __name__=='__main__':main()
