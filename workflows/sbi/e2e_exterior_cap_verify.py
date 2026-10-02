"""Read-back checks of saved full-cap density and ordered tidal fields."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from workflows.sbi import e2e_coupled_contract as c


def verify(root):
    c.require_compute()
    marker = root/'COMPLETE.json'; record = json.loads(marker.read_text())
    binding_path = root/'BINDING.json'
    if c.sha256(binding_path) != record['binding']:
        raise ValueError('Changed completion binding')
    binding = json.loads(binding_path.read_text())
    original_receipt = binding['input_receipt']
    if c.sha256(original_receipt['path']) != original_receipt['sha256']:
        raise ValueError('Changed density source receipt')
    original = json.loads(Path(original_receipt['path']).read_text())
    for item in (record['output'], original['field']):
        if c.sha256(item['path']) != item['sha256']:
            raise ValueError('Changed saved field')
    maximum, interior_maximum, minimum, checked = 0., 0., float('inf'), 0
    with h5py.File(original['field']['path']) as src, h5py.File(record['output']['path']) as dst:
        shape = src['rho'].shape[1:]; padding = int(dst.attrs['padding_cells'])
        if (dst['rho_completed'].shape != (record['draws'], *(n+2*padding for n in shape))
                or dst['eigenvalues'].shape != (record['draws'], *shape, 3)
                or float(dst.attrs['cell_mpc_h']) != 6.766
                or not np.array_equal(dst.attrs['origin_mpc_h'], src.attrs['origin_mpc_h'])):
            raise ValueError('Saved field geometry changed')
        if not np.array_equal(src['support'][:], dst['support'][:]) or not np.array_equal(src['coverage'][:], dst['coverage'][:]):
            raise ValueError('Footprint or source coverage changed')
        for draw in range(record['draws']):
            for lo in range(0, shape[0], 8):
                hi = min(lo+8, shape[0]); original_rho = src['rho'][draw, lo:hi]
                rho = dst['rho_completed'][draw, lo+padding:hi+padding, padding:padding+shape[1], padding:padding+shape[2]]
                eigen = dst['eigenvalues'][draw, lo:hi]
                if not np.isfinite(eigen).all() or np.any(np.diff(eigen, axis=-1) < 0):
                    raise ValueError('Nonfinite or unordered saved eigenvalues')
                if not np.isfinite(rho).all() or np.any(rho <= 0):
                    raise ValueError('Invalid saved completed density')
                error = float(np.max(abs(eigen.astype('f8').sum(-1)-(rho.astype('f8')-1))))
                maximum = max(maximum, error)
                if error > 8e-7*max(1., float(rho.max())):
                    raise ValueError('Saved density and tensor trace disagree beyond float32 rounding')
                known = np.isfinite(original_rho)
                if record['method'] != 'learned':
                    known &= src['support'][lo:hi] > 0
                if record['method'] != 'zero' and known.any():
                    difference = float(np.max(abs(original_rho[known]-rho[known])))
                    interior_maximum = max(interior_maximum, difference)
                    if difference != 0:
                        raise ValueError('Saved inferred interior was modified')
                checked += int(rho.size)
            for lo in range(0, dst['rho_completed'].shape[1], 16):
                rho = dst['rho_completed'][draw, lo:lo+16]
                if not np.isfinite(rho).all() or np.any(rho <= 0):
                    raise ValueError('Invalid saved exterior density')
                minimum = min(minimum, float(rho.min()))
    result = dict(passed=True, saved_trace_maxabs=maximum, saved_interior_maxabs=interior_maximum,
                  completed_minimum_density=minimum, interior_voxels_checked=checked,
                  complete_receipt=c.file_record(marker, content_hash=True),
                  verifier_sha256=c.sha256(__file__), posterior_qualified=False,
                  cap_joint_law=record['cap_joint_law'])
    c.atomic_json(root/'VERIFIED.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    a = p.parse_args(); verify(a.root)
