"""Build a pinned background-only CLASS distance; never fit catalogue indices.

Run only in the approved compute allocation. Downloads public release sources
to a new private Scratch directory, builds CLASS 2.6.3, and binds the original
generator's c000 ini bytes. The original production solver version is unknown:
this is a source-backed candidate, still requiring all-row host validation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile

from workflows.sbi.prepare_cfm_lowz_inputs import require_compute, save_json, sha256

REPO = Path(__file__).resolve().parents[2]
COMMIT = 'a40a5d3be570a6ce9d5a0698268b04864d4d3af2'
CLASS_VERSION = '2.6.3'
NBODYKIT_COMMIT = '09c5f271aab28f6c7f831ae64d7b7af827029e0a'
CLASSYLSS_TAG = '0.2.9'
CLASSYLSS_PATCHES = ('class-2.6.0-a_max.patch','class-2.6.1-pathsize.patch')
EXPECTED_INI = {
    'root': 'abacus_cosm000.', 'omega_b': '0.02237', 'omega_cdm': '0.12',
    'h': '0.6736', 'A_s': '2.083e-09', 'n_s': '0.9649', 'alpha_s': '0.0',
    'N_ur': '2.0328', 'N_ncdm': '1.0', 'omega_ncdm': '0.0006442',
    'w0_fld': '-1.0', 'wa_fld': '0.0',
}
CLONE_ADDITIONS = {'output': 'vTk dTk mPk', 'extra metric transfer functions': 'y',
                   'Omega_Lambda': '0.'}


def check_ini(text):
    values = {}
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        name, value = map(str.strip, line.split('=', 1))
        if name in values:
            raise ValueError('Duplicate CLASS parameter')
        values[name] = value
    if values != EXPECTED_INI:
        raise ValueError('Pinned generator parameters changed')
    return values


def run(destination):
    require_compute()
    destination = Path(destination).resolve()
    root = Path('/pscratch/sd/d/dkololgi/abacus').resolve()
    if not destination.is_relative_to(root):
        raise PermissionError('Use a private bounded Abacus Scratch build')
    destination.mkdir(exist_ok=False)
    archive = destination/'class-v2.6.3.tar.gz'
    ini = destination/'CLASS_c000.ini'
    nbodykit = destination/'nbodykit_cosmology.py.txt'
    urls = {
        archive: 'https://codeload.github.com/lesgourg/class_public/tar.gz/refs/tags/v2.6.3',
        ini: f'https://raw.githubusercontent.com/amjsmith/shared_code/{COMMIT}/abacus_mocks/lookup/Cosmologies/CLASS_c000.ini',
        nbodykit: f'https://raw.githubusercontent.com/bccp/nbodykit/{NBODYKIT_COMMIT}/nbodykit/cosmology/cosmology.py',
    }
    urls.update({destination/name:f'https://raw.githubusercontent.com/bccp/classylss/{CLASSYLSS_TAG}/depends/{name}'
                 for name in CLASSYLSS_PATCHES})
    for path, url in urls.items():
        subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                        '--max-time', '120', url, '--output', str(path)], check=True)
    parameters = check_ini(ini.read_text())
    # The generator calls from_file(...).clone(). nbodykit compile_args sets
    # Omega_Lambda=0 when w0_fld/wa_fld are present, making CLASS infer Omega_fld.
    # Directly evaluating the raw ini would NOT reproduce that construction.
    cloned_ini = destination/'CLASS_c000_nbodykit_clone.ini'
    cloned_ini.write_text(ini.read_text()+'\n'+''.join(f'{k} = {v}\n' for k,v in CLONE_ADDITIONS.items()))
    with tarfile.open(archive) as source:
        for item in source.getmembers():
            p = Path(item.name)
            if (p.is_absolute() or '..' in p.parts or not p.parts
                    or p.parts[0] != 'class_public-2.6.3'
                    or item.issym() or item.islnk() or item.isdev()):
                raise ValueError('Unsafe public source archive member')
        source.extractall(destination, filter='data')
    source_root = destination/'class_public-2.6.3'
    for name in CLASSYLSS_PATCHES:
        subprocess.run(['patch','--batch','--forward','-p1','-i',str(destination/name)],
                       cwd=source_root,check=True)
    helper = REPO/'workflows/sbi/p12a_class_background.c'
    before = sha256(helper)
    binary = destination/'p12a_class_background'
    with (destination/'build.log').open('x') as log:
        subprocess.run(['make', '-j4', 'libclass.a', 'OMPFLAG=', 'HYREC='],
                       cwd=source_root, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['gcc', '-O2', '-I'+str(source_root/'include'), str(helper),
                        str(source_root/'libclass.a'), '-lm', '-o', str(binary)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if sha256(helper) != before:
        raise ValueError('Distance helper changed during build')
    result = subprocess.run([str(binary), str(cloned_ini)], cwd=source_root,
                            capture_output=True, text=True, check=True)
    measured = json.loads(result.stdout)
    if (measured['redshift'] != .16 or measured['h'] != .6736
            or abs(measured['omega_ncdm']-.0006442) > 1e-12
            or measured['Omega_Lambda'] != 0. or not .6 < measured['Omega_fld'] < .8
            or abs(measured['distance_mpc_h']-measured['conformal_difference_mpc_h']) > 1e-8
            or not 460. < measured['distance_mpc_h'] < 462.):
        raise ValueError('CLASS background identity failed')
    record = dict(schema='p12a-generator-distance-v1', job=os.environ['SLURM_JOB_ID'],
        generator_commit=COMMIT, class_version=CLASS_VERSION,
        classylss_tag=CLASSYLSS_TAG,classylss_patches=list(CLASSYLSS_PATCHES),
        solver_selection='CLASS 2.6.3 with classylss 0.2.9 patches; production version not recovered',
        parameters=parameters, clone_additions=CLONE_ADDITIONS,
        nbodykit_commit=NBODYKIT_COMMIT,
        normalized_ini=dict(path=str(cloned_ini),sha256=sha256(cloned_ini)),
        measurement=measured, catalogue_data_used=False,
        fitted_radius=False, host_mapping_qualified=False,
        sources=[dict(path=str(path), url=url, sha256=sha256(path)) for path,url in urls.items()],
        helper=dict(path=str(helper),sha256=before),
        binary=dict(path=str(binary),sha256=sha256(binary)),
        preparer=dict(path=__file__,sha256=sha256(__file__)))
    save_json(destination/'GENERATOR_DISTANCE.json', record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
