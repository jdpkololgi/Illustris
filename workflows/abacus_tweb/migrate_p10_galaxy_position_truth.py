#!/usr/bin/env python3
"""Prepare or execute a versioned P10 galaxy-position truth rebuild.

Existing catalogues and downstream graphs are preserved. This builds metadata,
annotation, and the observed truth join only; it does not change science masks,
train models, rewrite caches, or unseal ph001. Default is a reviewable plan.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

WORKFLOW=Path(__file__).resolve().parent
if str(WORKFLOW) not in sys.path:sys.path.insert(0,str(WORKFLOW))
from annotate_cutsky_with_tweb_eigs import load_qualification, sha256_file
from p10_phase_assets import DEFAULT_REGISTRY, load_registry


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--phase',required=True);ap.add_argument('--qualification',type=Path,required=True)
 ap.add_argument('--registry',type=Path,default=DEFAULT_REGISTRY);ap.add_argument('--output-root',type=Path)
 ap.add_argument('--execute',action='store_true');args=ap.parse_args()
 reg=load_registry(args.registry)
 if args.phase not in reg['phases'] or reg['phases'][args.phase]['role']=='sealed_blind':raise ValueError('Unregistered or sealed phase')
 phase_root=Path(reg['path_templates']['phase_output'].format(phase=args.phase))
 field=phase_root/'targets/tweb/backend_optimized_ngrid_2048_rsmooth_7'
 receipt=json.loads(args.qualification.read_text())
 if args.phase=='ph000':field=Path(receipt['phases']['ph000']['field_path'])
 load_qualification(args.qualification,args.phase,field)
 root=args.output_root or phase_root/'catalogues/galaxy_realspace_v2'
 parent=root/f'{args.phase}_bright_parent_metadata_v2.fits'
 annotated=root/f'{args.phase}_bright_parent_galaxy_realspace_v2_tweb.fits'
 observed=root/f'{args.phase}_observed_galaxy_realspace_v2_tweb.fits'
 commands=[
  [sys.executable,str(WORKFLOW/'p10_build_bright_parent.py'),'--registry',str(args.registry),'--phase',args.phase,'--output',str(parent)],
  [sys.executable,str(WORKFLOW/'annotate_cutsky_with_tweb_eigs.py'),'--phase',args.phase,'--qualification',str(args.qualification),'--cutsky',str(parent),'--tweb-dir',str(field),'--output-dir',str(root),'--output-name',annotated.name],
  [sys.executable,str(WORKFLOW/'p10_build_observed_truth.py'),'--registry',str(args.registry),'--phase',args.phase,'--annotated-parent',str(annotated),'--output',str(observed)]]
 plan=dict(phase=args.phase,qualification_sha256=sha256_file(args.qualification),output_root=str(root),commands=commands,
  downstream_action='Rebuild label arrays/caches with identical galaxy ordering; separately prove geometry reuse. Existing redshift gates remain unchanged.')
 print(json.dumps(plan,indent=2),flush=True)
 if not args.execute:return
 if not(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID') and socket.gethostname().startswith('nid')):raise RuntimeError('Execute only inside approved compute step')
 for command,output,suffix in zip(commands,[parent,annotated,observed],['.complete.json','.manifest.json','.complete.json']):
  if output.exists() or Path(str(output)+suffix).exists():
   raise FileExistsError(f'Existing migration artifact; inspect its receipt before a separately reviewed resume: {output}')
 for command in commands:subprocess.run(command,check=True)
 (root/'GALAXY_TRUTH_MIGRATION_COMPLETE.json').write_text(json.dumps(plan,indent=2)+'\n')
if __name__=='__main__':main()
