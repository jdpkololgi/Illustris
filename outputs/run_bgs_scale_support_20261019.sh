#!/bin/bash
set -euo pipefail
cd /global/u2/d/dkololgi/TNG/Illustris
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PATH="/pscratch/sd/d/dkololgi/conda/envs/cosmic_env/bin:$PATH"
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores python workflows/abacus_tweb/plot_bgs_scale_support_redshift.py --output outputs/bgs_scale_support_redshift_20261019
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores python outputs/refresh_bgs_plot_graph_20261019.py
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores graphify global add graphify-out/graph.json --as Illustris
