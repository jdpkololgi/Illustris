#!/bin/bash
# Read-only login-node monitor. All computation belongs to Slurm allocation 60483945.
while squeue -h -j 60483945 -o '%i' 2>/dev/null | grep -q '^60483945$'; do
    date -u
    squeue -h -j 60483945 -o '%i %T elapsed=%M remaining=%L node=%N'
    tail -n 5 /pscratch/sd/d/dkololgi/abacus/lowz_completion_60483945_campaign.log
    tail -n 5 /pscratch/sd/d/dkololgi/abacus/p12a_responses_60483945_queue.log 2>/dev/null
    sleep 60
done
sacct -j 60483945 --format=JobID,State,ExitCode,Elapsed -n
echo 'Allocation ended. No replacement allocation or retry was submitted.'
