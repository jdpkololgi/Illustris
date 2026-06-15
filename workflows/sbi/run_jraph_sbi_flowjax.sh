#!/bin/bash
#
# Wrapper script to run jraph_sbi_flowjax.py with correct conda environment
# Usage: ./run_jraph_sbi_flowjax.sh [arguments to pass to python script]
#

# Activate conda base environment
source /global/common/software/desi/perlmutter/desiconda/20240425-2.2.0/conda/etc/profile.d/conda.sh
conda activate base

# Run the script with all passed arguments
python3 jraph_sbi_flowjax.py "$@"
