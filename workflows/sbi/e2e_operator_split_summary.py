"""Summarise bounded extent and local-refinement controls without new payload IO."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import e2e_operator_split as s
import numpy as np


def width(e):
    e=np.asarray(e);return e[...,2]-e[...,0]


def main():
    result=dict(phases=[],local_refinements=[],independent_phases=2)
    for phase in ('ph012','ph013'):
        d=json.loads((s.OUT/phase/'SPLIT.json').read_text());assert d['passed']
        rows=d['cases'];assert len(rows)==16
        e32=np.array([r['pad32'] for r in rows]);e64=np.array([r['pad64'] for r in rows])
        truth=np.array([r['fullbox'] for r in rows]);sd=np.array([r['posterior_sd'] for r in rows]).reshape(16,2,3)
        delta=(e64-e32)/sd
        result['phases'].append(dict(phase=phase,extent_shift_sd_rms=float(np.sqrt(np.mean(delta**2))),
            extent_shift_sd_max=float(abs(delta).max()),fullbox_width=float(width(truth).mean()),
            pad32_width=float(width(e32).mean()),pad64_width=float(width(e64).mean()),
            pad32_deficit_fraction=float(1-width(e32).mean()/width(truth).mean()),
            pad64_deficit_fraction=float(1-width(e64).mean()/width(truth).mean())))
        for r in rows:
            if 'refinement' not in r:continue
            truth=width(r['fullbox']);base=width(r['pad32'])
            refined16=width(r['refinement']['16']['pad32_plus_local'])
            refined32=width(r['refinement']['32']['pad32_plus_local'])
            z=(np.asarray(r['refinement']['32']['pad32_plus_local'])-np.asarray(r['refinement']['16']['pad32_plus_local']))/np.asarray(r['posterior_sd']).reshape(2,3)
            result['local_refinements'].append(dict(phase=phase,pair_id=r['pair_id'],
                baseline_deficit_fraction=(1-base/truth).tolist(),
                refined_deficit_fraction=(1-refined32/truth).tolist(),
                deficit_recovered_fraction=((refined32-base)/(truth-base)).tolist(),
                collar_convergence_max_sd=float(abs(z).max()),
                fullbox_width=truth.tolist(),baseline_width=base.tolist(),refined_width=refined32.tolist()))
    result['caveat']='Refinement is local (four pairs/eight cores); extent doubling is not an infinite-domain proof. No posterior or target definition changed.'
    s.b.c.atomic_json(s.OUT/'SUMMARY.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
