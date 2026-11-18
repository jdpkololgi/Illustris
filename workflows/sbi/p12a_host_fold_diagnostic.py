"""Read-only five-fold feasibility check with the unchanged P4 balance gates."""
import numpy as np
from workflows.abacus_tweb import p4_build_spatial_manifest as p4


def audit(points, cap, shell, context, eligible, host_group, origins, h=.6766):
    core_size=64./h; factor=4; super_size=core_size*factor
    ids=np.flatnonzero(context); active=np.flatnonzero(eligible)
    if np.any(eligible & ~context) or np.any(host_group[active]<0):
        raise ValueError('Missing context or host group')
    indices=np.empty((len(ids),3),np.int32)
    for c in (0,1):
        select=cap[ids]==c
        indices[select]=p4.core_indices(points[ids[select],:3],origins[c],core_size)
    cores,inverse,counts=np.unique(np.column_stack((cap[ids],indices)),axis=0,
                                 return_inverse=True,return_counts=True)
    supers,core_super=np.unique(np.column_stack((cores[:,0],np.floor_divide(cores[:,1:],factor))),
                               axis=0,return_inverse=True)
    parent_super=np.full(len(points),-1,np.int32)
    parent_super[ids]=core_super[inverse]
    active_super=parent_super[active]
    super_context=np.bincount(core_super,weights=counts,minlength=len(supers))
    super_cores=np.bincount(core_super,minlength=len(supers))
    dsu=p4.DisjointSet(len(supers))
    # Unique (host,superblock) pairs avoid millions of redundant unions.
    pairs=np.unique(np.column_stack((host_group[active],active_super)),axis=0)
    same=pairs[1:,0]==pairs[:-1,0]
    for a,b in zip(pairs[:-1,1][same],pairs[1:,1][same]):dsu.union(int(a),int(b))
    _,component=np.unique(dsu.roots(),return_inverse=True)
    n=int(component.max())+1
    counts=np.zeros((n,2,4),np.int64)
    np.add.at(counts,(component[active_super],cap[active],shell[active]),1)
    component_context=np.bincount(component,weights=super_context,minlength=n)
    component_cores=np.bincount(component,weights=super_cores,minlength=n)
    results=[]
    for name,cw,kw in [('frozen_lpt_v1',.05,.05),('boundary_balance_fallback_v1',0.,.05),
                       ('whole_component_swap_fallback_v1',0.,.05)]:
        folds=p4.greedy_balanced_folds(counts,component_context,component_cores,
                                     context_weight=cw,core_weight=kw)
        if name=='whole_component_swap_fallback_v1':folds=p4.refine_folds_by_swaps(counts,folds)
        super_fold=folds[component];active_fold=super_fold[active_super]
        lower,upper=p4.same_fold_run_bounds(supers[:,1:],supers[:,0],super_fold,origins,super_size)
        xyz=np.asarray(points[active,:3],float)
        distance=np.min(np.minimum(xyz-lower[active_super],upper[active_super]-xyz),axis=1)
        if np.any(distance < -1e-5):raise ValueError('Point outside same-fold bounds')
        distance=np.maximum(distance,0).astype(np.float32)
        dimensions=np.zeros((5,2,4),np.int64)
        np.add.at(dimensions,(active_fold,cap[active],shell[active]),1)
        totals=dimensions.sum(axis=(1,2));mean=dimensions.mean(axis=0)
        deviation=float(np.max(np.abs(dimensions-mean)/np.maximum(mean,1.)))
        super_counts=np.bincount(super_fold,minlength=5)
        medians=np.array([np.median(distance[active_fold==f]) if np.any(active_fold==f) else 0 for f in range(5)])
        ratio=lambda a: float(a.max()/a.min()) if a.min()>0 else None
        ratios=[ratio(totals),ratio(super_counts),ratio(medians)]
        crossing=int(np.sum(super_fold[pairs[:-1,1][same]]!=super_fold[pairs[1:,1][same]]))
        gates=dict(five_nonempty_folds=bool(np.all(totals>0)),
            fold_active_count_max_min_below_1p05=ratios[0] is not None and ratios[0]<1.05,
            cap_shell_relative_deviation_below_10pct=deviation<.10,
            fold_occupied_superblock_ratio_below_1p25=ratios[1] is not None and ratios[1]<1.25,
            fold_distance_medians_matched_below_25pct=ratios[2] is not None and ratios[2]<1.25,
            no_conservative_host_fold_crossing=crossing==0)
        results.append(dict(name=name,active_count_ratio=ratios[0],occupied_superblock_ratio=ratios[1],
            distance_median_ratio=ratios[2],cap_shell_max_relative_deviation=deviation,
            distance_mpc_medians=medians.tolist(),fold_counts=totals.tolist(),
            gates=gates,passed=all(gates.values())))
        if all(gates.values()):break
    return dict(core_mpc_h=64.,superblock_cores_per_axis=4,context_cores=len(cores),
        occupied_superblocks=len(supers),host_linked_components=n,
        largest_component_active_rows=int(counts.sum(axis=(1,2)).max()),
        largest_component_active_fraction=float(counts.sum(axis=(1,2)).max()/len(active)),
        candidates=results,passed=any(r['passed'] for r in results),
        support_attachment_qualified=False,production_fold_manifest_written=False)
