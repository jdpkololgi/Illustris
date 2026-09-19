"""Replicate the phone's fixed-convention host audit on registered train phases."""
import argparse
import json
import numpy as np
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import e2e_coupled_coordinates as coord
from workflows.sbi.e2e_coupled_coordinate_audit import run, periodic_difference


def summary(x):
    x = np.linalg.norm(x, axis=-1)
    return dict(median=float(np.median(x)), p95=float(np.quantile(x,.95)),
                maximum=float(x.max()), rms=float(np.sqrt(np.mean(x*x))))


def verify(phase):
    c.require_compute(); c.phase_guard(phase); coord.bind()
    if phase not in coord.config()['host_check_phases']:
        raise PermissionError('only preregistered training-phase coordinate checks')
    directory = coord.ROOT/'coordinate_audit'/phase
    with c.single_writer(directory):
        marker = directory/'CORRECTED_COMPLETE.json'
        if marker.exists():
            return coord.verify_receipt(marker)
        original = c.ROOT/'coordinate_audit'/phase/'AUDIT.json'
        if not original.exists():
            run(phase)
        record = json.loads(original.read_text())
        source = record['outputs'][0]
        if record['phase'] != phase or c.sha256(source['path']) != source['sha256']:
            raise ValueError('linked host audit changed')
        with np.load(source['path'], allow_pickle=False) as f:
            z, truth = f['Z_COSMO'], f['host_x_L2com']
            sky = coord.sky_mpc_h(f['RA'], f['DEC'], z)
            old = f['sky_planck18_mpc_h']
        variants = dict(
            corrected=summary(periodic_difference(sky-1000.,truth)),
            inherited_planck18=summary(periodic_difference(old-1000.,truth)),
            wrong_extra_h=summary(periodic_difference(sky*.6766-1000.,truth)),
            wrong_missing_h=summary(periodic_difference(sky/.6766-1000.,truth)))
        # Best scalar is diagnostic only: do not adopt a learned coordinate warp.
        scale = float(np.sum(old*sky)/np.sum(old*old))
        variants['best_scalar_only'] = summary(periodic_difference(scale*old-1000.,truth))
        radial_change = np.linalg.norm(sky,axis=1)-np.linalg.norm(old,axis=1)
        gates = dict(p95_below_0p001_mpc_h=variants['corrected']['p95']<coord.config()['host_check_p95_mpc_h_max'],
                     maximum_below_0p01=variants['corrected']['maximum']<coord.config()['host_check_max_mpc_h_max'],
                     extra_and_missing_h_rejected=min(variants[k]['p95'] for k in ('wrong_extra_h','wrong_missing_h'))>1.,
                     source_training_only=c.ROLES[phase]=='train')
        result = dict(**coord.provenance(), phase=phase, samples=len(z),
                      outputs=[source], host_sources=record['host_sources'],
                      old_audit_sha256=c.sha256(original), variants_mpc_h=variants,
                      radial_change_z_correlation=float(np.corrcoef(radial_change,z)[0,1]),
                      best_scalar_diagnostic_only=scale, transformation_fitted=False,
                      gates=gates, **{'pass':all(gates.values())})
        c.atomic_json(marker,result)
        if not result['pass']:
            raise ValueError(f'coordinate gate failed for {phase}: {variants}')
        return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=coord.config()['host_check_phases'])
    args=p.parse_args()
    for phase in ([args.phase] if args.phase else coord.config()['host_check_phases']):
        print(json.dumps(verify(phase)),flush=True)
