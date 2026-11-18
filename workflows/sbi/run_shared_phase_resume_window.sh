#!/usr/bin/env bash
# Run real work in the allocation-owning shell so it is not idle at a prompt.
# No timeout/security setting is changed. Every FFT worker remains bounded.
set -uo pipefail
: "${SLURM_JOB_ID:?Approved allocation required}"
cd /global/homes/d/dkololgi/TNG/Illustris
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#nodes[@]} == 2 ]] || exit 2
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
export GRAPHWEB_CFM_DATA_CONFIG=configs/e2e_coupled_data_20phase_20261027.json
py=/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin/python
root=/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/coupled_20phase_20261027_v1
srun --overlap -N1 -n1 -c32 -w "${nodes[0]}" "$py" -u -m workflows.sbi.run_shared_phase_target_queue --coupled-phases ph001 ph015 ph017 --wholefield-phases ph001 ph015 ph017 --seconds 4500 > "$root/target_queue_a_${SLURM_JOB_ID}.log" 2>&1 &
first=$!
srun --overlap -N1 -n1 -c32 -w "${nodes[1]}" "$py" -u -m workflows.sbi.run_shared_phase_target_queue --coupled-phases ph014 ph016 --wholefield-phases ph004 ph005 ph006 ph014 ph016 --seconds 4500 > "$root/target_queue_b_${SLURM_JOB_ID}.log" 2>&1 &
second=$!
wait "$first"; a=$?
wait "$second"; b=$?
printf 'Foreground target queues finished: %s %s\n' "$a" "$b"
[[ "$a" == 0 && "$b" == 0 ]]
