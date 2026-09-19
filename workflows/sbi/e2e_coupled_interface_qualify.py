"""Actual normalized-payload qualification, distinct from scientific training.

Run after the global thirteen-phase normalizer and independent phase audits.
Each phase is checkpointed; the full interface receipt requires all21 phases.
Numerical held-out QA is allowed, but no model or predictive score is used.
"""
import argparse
from contextlib import contextmanager, ExitStack
import io
import json
from pathlib import Path
import time
from unittest.mock import patch

import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi import e2e_coupled_conditions as reader
from workflows.sbi import e2e_coupled_views as views
from workflows.sbi import e2e_coupled_operators as op
from workflows.sbi import e2e_coupled_normalization as norm
from workflows.sbi import e2e_coupled_product_audit as audit
from workflows.sbi import e2e_coupled_audit_worker as audit_queue


@contextmanager
def observation_io_only(phase):
    """Fail on attempted target IO, not merely on target-derived outputs.

    Single-threaded qualification guard for Python file reads and HDF5 opens.
    The existing reader separately rejects external links/virtual storage before
    resolving them. This is an executable reader test, not an OS sandbox.
    """
    c.phase_guard(phase)
    normalizer=(coord.ROOT/'normalization/NORMALIZATION_COMPLETE.json').resolve()
    allowed=((coord.ROOT/'conditions'/phase).resolve(),(c.REPO/'configs').resolve())
    def check(path,mode='r'):
        if not isinstance(path,(str,Path)) or any(flag in mode for flag in ('w','a','+','x')):
            raise PermissionError('inference qualification permits only named read paths')
        path=Path(path).resolve()
        if path!=normalizer and not any(root in path.parents for root in allowed):
            raise PermissionError('observation inference attempted non-observation IO: '+str(path))
    original_io=io.open; original_h5=h5py.File
    def guarded_open(file,mode='r',*args,**kwargs):
        check(file,mode); return original_io(file,mode,*args,**kwargs)
    def guarded_h5(name,mode='r',*args,**kwargs):
        check(name,mode); return original_h5(name,mode,*args,**kwargs)
    with ExitStack() as stack:
        stack.enter_context(patch('builtins.open',guarded_open))
        stack.enter_context(patch('io.open',guarded_open))
        stack.enter_context(patch.object(h5py,'File',guarded_h5))
        yield


def relative_error(actual,reference):
    actual=np.asarray(actual,dtype=np.float64); reference=np.asarray(reference,dtype=np.float64)
    if actual.shape!=reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('invalid round-trip shapes/values')
    if np.any(actual<=0) or np.any(reference<=0): raise ValueError('round-trip density must stay positive')
    error=float(np.max(np.abs(actual-reference)/reference))
    if error>2e-6: raise ValueError('normalized round-trip exceeds the frozen2e-6 tolerance')
    return error


def binding(normalizer_sha256):
    return dict(normalization_sha256=normalizer_sha256,
                source_hashes={Path(module.__file__).name:c.sha256(module.__file__)
                               for module in (views,reader,op,norm,audit)},
                qualifier_sha256=c.sha256(__file__))


def qualify_phase(phase,chart,normalizer_sha256):
    c.phase_guard(phase)
    if not audit_queue.qualified(phase): raise FileNotFoundError('independent phase audit is incomplete')
    root=coord.ROOT/'interface_qualification'; directory=root/phase
    audit_pointer=coord.verify_receipt(coord.ROOT/'product_audit'/phase/'LATEST_AUDIT.json',payload=False)
    paths=[Path(audit_pointer['audit']['path']),
           coord.ROOT/'conditions'/phase/'CONDITIONS_COMPLETE.json',
           coord.ROOT/'targets'/phase/'TARGETS_COMPLETE.json']
    sources=[c.file_record(p,content_hash=True) for p in paths]
    bound=binding(normalizer_sha256)
    marker=directory/'INTERFACE_PHASE_COMPLETE.json'
    with c.single_writer(directory):
        if marker.exists():
            result=coord.verify_receipt(marker,payload=False)
            if result['binding']!=bound or result['sources']!=sources:
                raise ValueError('qualified interface source drift')
            return result
        started=time.monotonic()
        condition_record=coord.verify_receipt(paths[1],payload=False)
        target_record=coord.verify_receipt(paths[2],payload=False)
        ids=sorted(Path(row['path']).stem for row in condition_record['pair_receipts'])
        expected=128 if phase in c.TRAIN else 16
        if len(ids)!=expected or len(set(ids))!=expected: raise ValueError('wrong phase pair quota')
        target_map=audit.pair_index(target_record,ids,coord.ROOT/'targets'/phase)
        offsets=op.layout()['context_offsets_raw'] if phase in c.TRAIN else [[0,0,0]]
        max_density=max_mass=max_zero=0.; cases=0
        for pair_id in ids:
            item=target_map[pair_id]
            if c.sha256(item['path'])!=item['sha256']: raise ValueError('target receipt changed')
            record=coord.verify_receipt(item['path'])
            actual=audit.read_target(c.guarded(record['outputs'][0]['path'],phase),phase,pair_id)
            rho=actual['rho_joint']; wide=actual['coarse_rho_extended']
            for offset in offsets:
                with observation_io_only(phase):
                    observed=views.load_observations(phase,pair_id,normalizer_sha256,offset)
                if set(observed)!={'joint','wide','support','joint_center_from_wide_mpc_h'}:
                    raise ValueError('unexpected inference feature/metadata')
                if not all(np.isfinite(value).all() for value in observed.values()):
                    raise ValueError('nonfinite normalized observations')
                encoded=views.target_view(rho,wide,phase,offset,chart)
                decoded=views.decode_view(encoded['coarse_logrho'],encoded['fine_residual'],phase,offset,chart)
                max_density=max(max_density,relative_error(decoded,rho))
                max_mass=max(max_mass,relative_error(op.mean_pool(decoded),op.mean_pool(rho)))
                zero=float(np.abs(op.mean_pool(encoded['fine_residual'][0])).max())
                if zero>2e-6: raise ValueError('float32 normalized residual left its zero-mean subspace')
                max_zero=max(max_zero,zero); cases+=1
                if phase in c.TRAIN and offset==[0,0,0]:
                    fit_observed,fit_target=views.load_training_pair(phase,pair_id,normalizer_sha256)
                    if (any(not np.array_equal(observed[k],fit_observed[k]) for k in observed)
                            or any(not np.array_equal(encoded[k],fit_target[k]) for k in encoded)):
                        raise ValueError('training/inference views differ for identical inputs')
            if phase not in c.TRAIN:
                # Instrument chart access to prove the role rejection precedes IO.
                with patch.object(views,'load_chart',side_effect=AssertionError('held-out fit IO')):
                    try: views.load_training_pair(phase,pair_id,normalizer_sha256)
                    except PermissionError: pass
                    else: raise ValueError('held-out phase admitted to training reader')
        result=dict(**coord.provenance(),phase=phase,role=c.ROLES[phase],binding=bound,sources=sources,
            pairs=len(ids),offset_cases=cases,max_density_relative_error=max_density,
            max_coarse_mass_relative_error=max_mass,max_standardized_block_mean=max_zero,
            observation_io_firewall_tested=True,training_reader_agreement=phase in c.TRAIN,
            heldout_fit_rejected=phase not in c.TRAIN,science_scores_evaluated=False,
            elapsed_seconds=time.monotonic()-started,outputs=[],**{'pass':True})
        c.atomic_json(marker,result)
        print(json.dumps(dict(phase=phase,interface_pairs=len(ids),offset_cases=cases,
                             max_density_relative_error=max_density)),flush=True)
        return result


def run():
    c.require_compute(); coord.require_host_checks()
    normalizer=coord.ROOT/'normalization/NORMALIZATION_COMPLETE.json'
    if not normalizer.exists(): raise FileNotFoundError('complete global train normalizer required')
    norm.fit()  # Revalidate the exact training/source bindings; no new fit here.
    normalizer_hash=c.sha256(normalizer); chart=views.load_chart(normalizer_hash)
    root=coord.ROOT/'interface_qualification'
    with c.single_writer(root):
        results=[qualify_phase(phase,chart,normalizer_hash) for phase in c.ROLES]
        phase_sources=[c.file_record(root/phase/'INTERFACE_PHASE_COMPLETE.json',content_hash=True)
                       for phase in c.ROLES]
        pairs=sum(row['pairs'] for row in results); cases=sum(row['offset_cases'] for row in results)
        if pairs!=1792 or cases!=11776: raise ValueError('global phase/augmentation panel incomplete')
        result=dict(**coord.provenance(),binding=binding(normalizer_hash),phase_roles=c.ROLES,
            sources=[c.file_record(normalizer,content_hash=True),*phase_sources],pairs=pairs,
            offset_cases=cases,scientific_training_authorized=False,full_preparation_complete=False,
            science_scores_evaluated=False,outputs=[],**{'pass':True})
        marker=root/'INTERFACE_COMPLETE.json'
        if marker.exists():
            old=coord.verify_receipt(marker,payload=False)
            if old['binding']!=result['binding'] or old['sources']!=result['sources']:
                raise ValueError('full interface qualification drift')
            return old
        c.atomic_json(marker,result); return result


def raw_reader_smoke(phase):
    """Exercise actual targetless IO now; never stand in for normalized QA."""
    c.require_compute(); c.phase_guard(phase); coord.require_host_checks()
    if not audit_queue.qualified(phase): raise FileNotFoundError('phase audit required for reader smoke')
    source=coord.ROOT/'conditions'/phase/'CONDITIONS_COMPLETE.json'
    record=coord.verify_receipt(source,payload=False)
    ids=sorted(Path(item['path']).stem for item in record['pair_receipts'])
    expected=128 if phase in c.TRAIN else 16
    if len(ids)!=expected or len(set(ids))!=expected: raise ValueError('incomplete smoke phase')
    started=time.monotonic()
    for pair_id in ids:
        with observation_io_only(phase):
            arrays=reader.load_pair(phase,pair_id)
        if set(arrays)!={'joint','wide_extended','support'}:
            raise ValueError('unexpected targetless reader outputs')
    result=dict(**coord.provenance(),phase=phase,pairs=len(ids),binding=binding(None),
        sources=[c.file_record(source,content_hash=True)],elapsed_seconds=time.monotonic()-started,
        targets_opened_inside_reader=False,normalization_tested=False,
        full_interface_qualified=False,scientific_training_authorized=False,outputs=[],**{'pass':True})
    path=coord.ROOT/'interface_qualification'/phase/f'RAW_READER_SMOKE_{time.time_ns()}.json'
    c.atomic_json(path,result)
    return dict(path=str(path),sha256=c.sha256(path),pairs=len(ids),elapsed_seconds=result['elapsed_seconds'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-reader-phase',choices=list(c.ROLES))
    args=parser.parse_args()
    result=raw_reader_smoke(args.raw_reader_phase) if args.raw_reader_phase else run()
    print(json.dumps(result),flush=True)
