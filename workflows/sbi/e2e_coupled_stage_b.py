"""Serial, bounded HPSS transfer only; payload scientific QA runs on compute.

Run in NERSC's transfer-only xfer QOS. This command cannot request allocations, train a model,
retrieve other epochs/particle products, or delete existing staging trees.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time

from workflows.sbi import e2e_coupled_contract as c

MEMBERS = ('./halos/z0.200/field_rv_B', './halos/z0.200/halo_rv_B')


def archive(phase):
    if phase not in c.config()['approval']['hpss_b_restore_phases']:
        raise PermissionError('phase not approved for B restore')
    name = f'AbacusSummit_base_c000_{phase}'
    return f'/nersc/projects/desi/cosmosim/Abacus/{name}/Abacus_{name}_halos.tar'


def parse_listing(text):
    result = {}
    for line in text.splitlines():
        match = re.match(r'^HTAR:\s+-\S+\s+\S+\s+(\d+)\s+\S+\s+\S+\s+(\S+)', line)
        if not match:
            continue
        size, name = int(match[1]), match[2]
        if not any(name.startswith(member+'/') for member in MEMBERS):
            raise ValueError('unexpected archive member')
        if '..' in Path(name).parts or Path(name).is_absolute() or name in result:
            raise ValueError('unsafe/duplicate archive member')
        result[name] = size
    expected = {f'{directory}/{kind}_rv_B_{i:03d}.asdf'
                for directory, kind in zip(MEMBERS, ('field','halo')) for i in range(34)}
    expected |= {f'{directory}/checksums.crc32' for directory in MEMBERS}
    if set(result) != expected or min(result.values(), default=0) <= 0:
        raise ValueError('archive lacks exact 68 payloads and two CRC manifests')
    return result


def inventory(phase, directory):
    path = directory/'ARCHIVE_INVENTORY.json'
    if path.exists():
        value = json.loads(path.read_text())
        if value['config_sha256'] != c.sha256(c.CONFIG) or value['archive'] != archive(phase):
            raise ValueError('archive receipt mismatch')
        return value
    process = subprocess.run(['/usr/bin/htar','-t','-f',archive(phase), *MEMBERS],
                             check=True, text=True, capture_output=True)
    listing = process.stdout+'\n'+process.stderr
    files = parse_listing(listing)
    value = dict(**c.provenance(), phase=phase, archive=archive(phase), files=files,
                 payload_bytes=sum(files.values()), inventory_only=True)
    c.atomic_json(path, value)
    return value


def staged_bytes_match(directory, record):
    return all((directory/name).is_file() and (directory/name).stat().st_size == size
               for name,size in record['files'].items())


def used_bytes(root):
    """Bounded walk of this new experiment only, counting hardlinks once."""
    seen, total = set(), 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory)/name
            if path.is_symlink():
                continue
            st = path.stat()
            key = st.st_dev, st.st_ino
            if key not in seen:
                total += st.st_size
                seen.add(key)
    return total


def require_transfer_job():
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        raise RuntimeError('bulk restore requires a scheduled xfer job')
    info = subprocess.run(['scontrol','show','job',job,'-o'], check=True,
                          text=True,capture_output=True).stdout
    if not re.search(r'\bQOS=xfer\b',info):
        raise RuntimeError('bulk restore requires transfer-only xfer QOS')


def stage(phase, inventory_only=False):
    directory = c.guarded(c.ROOT/'particle_b'/phase, phase, output=True)
    with c.single_writer(directory):
        record = inventory(phase, directory)
        if inventory_only:
            return record
        require_transfer_job()
        c.verify_receipt(c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json', payload=False)
        marker = directory/'TRANSFER_COMPLETE.json'
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous['config_sha256'] != c.sha256(c.CONFIG) or not staged_bytes_match(directory,record):
                raise ValueError('completed transfer changed')
            return previous
        if not staged_bytes_match(directory, record):
            # Extract only missing/incorrect-size members. An interrupted member
            # is overwritten only inside this phase's registered staging tree.
            missing = [name for name,size in record['files'].items()
                       if not (directory/name).is_file() or (directory/name).stat().st_size != size]
            required = sum(record['files'][name] for name in missing)
            if used_bytes(c.ROOT)+required > c.config()['approval']['scratch_bytes']:
                raise RuntimeError('approved Scratch cap would be exceeded')
            if shutil.disk_usage(directory).free < required+(32<<30):
                raise RuntimeError('insufficient filesystem headroom')
            log = directory/f'htar_attempt_{time.time_ns()}.log'
            started = time.monotonic()
            with log.open('x') as stream:
                subprocess.run(['/usr/bin/htar','-xvf',archive(phase),*missing],
                               cwd=directory, stdout=stream, stderr=subprocess.STDOUT, check=True)
            elapsed = time.monotonic()-started
        else:
            elapsed = 0.0
        if not staged_bytes_match(directory, record):
            raise ValueError('transfer did not produce all inventory-sized members')
        value = dict(**c.provenance(), phase=phase, archive=record['archive'],
                     inventory_sha256=c.sha256(directory/'ARCHIVE_INVENTORY.json'),
                     payload_bytes=record['payload_bytes'], elapsed_seconds=elapsed,
                     crc_verified=False, scientific_qa_passed=False,
                     note='Transfer completion is NOT permission to use unverified particles')
        c.atomic_json(marker,value)
        return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phases', nargs='+', required=True)
    p.add_argument('--inventory-only', action='store_true')
    p.add_argument('--wait-for-pairing-seconds', type=int, default=0)
    a = p.parse_args()
    c.bind_run()
    deadline = time.monotonic()+a.wait_for_pairing_seconds
    for phase in a.phases:
        marker = c.ROOT/'observations'/phase/'OBSERVED_COMPLETE.json'
        while not a.inventory_only and not marker.exists():
            if time.monotonic() >= deadline:
                raise SystemExit(75)
            print(json.dumps(dict(phase=phase, waiting_for=str(marker))),flush=True)
            time.sleep(60)
        print(json.dumps(stage(phase,a.inventory_only)), flush=True)


if __name__ == '__main__':
    main()
