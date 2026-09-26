"""Compare pinned native radii with the historical Planck18 chart; no correction."""
import hashlib
import json
from pathlib import Path
import numpy as np
from astropy.cosmology import Planck18
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[2]

def main():
    config = ROOT/'configs/e2e_coupled_coordinates_v2.json'
    cfg = json.loads(config.read_text())
    table = ROOT/cfg['table_path']
    table_sha = hashlib.sha256(table.read_bytes()).hexdigest()
    if table_sha != cfg['table_sha256']: raise ValueError('native table changed')
    data = np.loadtxt(table)
    native = CubicSpline(data[:,0],data[:,2])
    z = np.linspace(0.15,0.55,401)
    legacy_h = cfg['legacy_mpc_to_mpc_h']
    old_r = Planck18.comoving_distance(z).value*legacy_h
    old_dr = 299792.458/Planck18.H(z).value*legacy_h
    new_r = native(z)
    jacobian = (old_r/new_r)**2*old_dr/native(z,1)
    report = dict(schema='p12a-native-fiducial-distance-comparison-v1',
                  config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                  table_sha256=table_sha,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  z=z.tolist(),planck18_times_legacy_h_mpc_h=old_r.tolist(),
                  native_radius_mpc_h=new_r.tolist(),old_to_native_volume_jacobian=jacobian.tolist(),
                  max_abs_radial_difference_mpc_h=float(np.max(np.abs(old_r-new_r))),
                  jacobian_min=float(jacobian.min()),jacobian_max=float(jacobian.max()),
                  interpretation='Compares pinned E2E native chart against Planck18 times legacy h=0.6766. This is a coordinate-chart comparison, not galaxy-host offsets or a P12 correction. Preserve P12 Mpc coordinates and frozen selection together; do not transplant E2E products.',
                  ready_for_desi_canary=False)
    out=ROOT/'docs/evidence/p12/P12A_NATIVE_DISTANCE_COMPARISON_20260924.json'
    with out.open('x') as f: json.dump(report,f,indent=2);f.write('\n')
    print({k:report[k] for k in ['max_abs_radial_difference_mpc_h','jacobian_min','jacobian_max']})

if __name__ == '__main__': main()
