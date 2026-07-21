#!/usr/bin/env bash
# Example only — edit paths and load the DESI software stack before running.
#
# Typical order:
#   1) source /global/common/software/desi/desi_environment.sh main
#   2) python scripts/upstream_prepare_mocks_Y3_bright.py ...  # builds forFA{real}.fits
#   3) python scripts/upstream_getpotaDA2_mock.py --mock ab2ndgen --prog BRIGHT --realization 0 ...
#
# See README.md for NGC/SGC vs tiles and ph000/mock0 alignment.
