"""Tiny analytic audit of the *unchanged* block-replicated tidal operator.

No new representation, grid proposal, neural fit, held-out data or gate override.
The same periodic domain at both resolutions removes exterior-field ambiguity.
"""
import json
import numpy as np

from workflows.sbi.e2e_field_build_products import tensor_from_delta
from workflows.sbi.e2e_vdm_context_physics import repeat_spatial
from workflows.sbi.e2e_vdm_context_products import mean_pool


def audit():
    n,factor,cell=16,4,6.766
    axes=np.meshgrid(*[(np.arange(n)+.5)/n for _ in range(3)],indexing='ij',sparse=True)
    records=[]
    for mode in ((0,0,0),(1,0,0),(1,1,0),(1,1,1)):
        delta=.2*np.cos(2*np.pi*sum(k*x for k,x in zip(mode,axes)))
        delta=np.broadcast_to(delta,(n,n,n)).copy()
        coarse=mean_pool(delta,factor)
        lifted=repeat_spatial(coarse,factor)
        truth=tensor_from_delta(delta,cell)
        coarse_tide=repeat_spatial(tensor_from_delta(coarse,cell*factor),factor)
        lifted_tide=tensor_from_delta(lifted,cell)
        composite=coarse_tide+tensor_from_delta(delta-lifted,cell)
        difference=composite-truth
        commutator=coarse_tide-lifted_tide
        records.append(dict(mode=mode,
            rms_tensor_error_over_reference=float(np.sqrt(np.mean(difference**2))/np.sqrt(np.mean(truth**2))),
            density_roundtrip_max_abs=float(np.max(np.abs(lifted+(delta-lifted)-delta))),
            trace_error_max_abs=float(np.max(np.abs(composite[...,[0,3,5]].sum(-1)-delta))),
            commutator_identity_max_abs=float(np.max(np.abs(difference-commutator))),
            commutator_trace_max_abs=float(np.max(np.abs(commutator[...,[0,3,5]].sum(-1))))))
    return dict(fine_grid=n,coarse_grid=n//factor,cell_mpc_h=cell,
        records=records,changes_proposed_operator=False,exterior_uncertainty=False,
        interpretation='Exact density/mass/trace identities do not imply tensor accuracy: block lifting and the tidal operator do not commute')


if __name__=='__main__':
    print(json.dumps(audit(),indent=2,allow_nan=False))
