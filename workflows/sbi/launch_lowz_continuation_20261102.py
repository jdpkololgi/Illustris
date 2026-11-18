"""One user-approved CPU/desi/interactive/64-CPU/Scratch window, at most four hours.

No training or failed-stage retries. salloc releases after the directly waited
srun finishes. tmux owns the launcher, not the scientific compute.
"""
from workflows.sbi.launch_lowz_continuation_20261030 import main

if __name__ == '__main__':
    raise SystemExit(main('--continuation-20261102'))
