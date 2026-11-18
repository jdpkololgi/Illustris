"""Restore original FFT CPU masks after the rejected placement diagnostic."""
import json
import os
from pathlib import Path
import socket
import subprocess

if os.environ.get('SLURM_JOB_ID')!='60358875':raise RuntimeError('Only the approved continuation allocation')
node=socket.gethostname().split('.')[0]
placements={
 'nid200033': [('run_shared_phase_target_queue',0,32),('run_shared_phase_audit_queue',32,8),('prepare_shared_wholefield observations',40,8),('adapters_new_60358875',48,16),('run_shared_phase_finalize p12',64,16),('p12_remaining_60358875',80,16)],
 'nid200034': [('run_shared_phase_target_queue',0,32),('run_shared_phase_audit_queue',32,8),('prepare_shared_wholefield reuse',40,8),('adapters_existing_60358875',48,16),('run_shared_phase_normalization',64,8),('reuse_shared_phase_adapters',72,8),('prepare_shared_phase_particles',80,8),('run_shared_phase_finalize wholefield',88,8)]}
# Exact process ancestry is used for shell-driven stages; no other user/job is touched.
rows=[]
for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,ppid=,args='],text=True).splitlines():
 pid,ppid,args=line.strip().split(None,2);rows.append((int(pid),int(ppid),args))
for label,start,count in placements[node]:
 matches=[]
 for pid,ppid,args in rows:
  match=label in args
  if label=='adapters_new_60358875':match='bash -c' in args and 'for phase in ph014 ph016 ph015 ph017;' in args
  if label=='adapters_existing_60358875':match=' -m workflows.abacus_tweb.p10_prepare_training_contract ' in args and '--phases ph000 ph002 ph003 ph004 ph005' in args
  if label=='p12_remaining_60358875':match='bash -c' in args and 'reuse_corrected_p2_20261021 --phase ph017' in args
  if match and pid!=os.getpid() and 'place_shared_phase_workers' not in args:
   group=Path(f'/proc/{pid}/cgroup')
   if group.exists() and '60358875' in group.read_text():matches.append(pid)
 chosen=set(matches)
 while True:
  more={pid for pid,ppid,args in rows if ppid in chosen}
  if more<=chosen:break
  chosen|=more
 if label != 'run_shared_phase_target_queue':continue
 cpus=set(range(16)) | set(range(128,144));changed=[]
 for pid in chosen:
  try:
   for tid in Path(f'/proc/{pid}/task').iterdir():os.sched_setaffinity(int(tid.name),cpus)
   actual=os.sched_getaffinity(pid)
   if actual!=cpus:raise RuntimeError('Requested affinity was not applied')
   changed.append(pid)
  except ProcessLookupError:pass
 print(json.dumps(dict(node=node,worker=label,cpus=sorted(cpus),processes=changed)),flush=True)
