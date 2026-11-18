# P12-A low-z encoder budget proposal (not launch approval)

Use the same fixed four complete epochs and terminal checkpoint for the full fit
and all 20 omitted-phase fits. This yields about 1.32 million optimizer updates
per omitted-phase fit (1,385,456 full-fit updates), comparable to 1,344,880
updates for the historical 20-epoch omit-ph000 encoder. Matching update count is
a budget heuristic, not matching passes over each galaxy or proof of convergence.
Do not select a different epoch independently for each fold.

Four epochs at 6,927,280 cores per common epoch and 0.13179 s/core give about
1,014 A100-hours for optimizer updates. The short crossover block range implies
roughly 892–1,177 hours before validation. The full contract adds 51,956
validation cores per epoch. Initialization, validation and checkpoint costs
have not been profiled across an entire production epoch.

Recommend staging: run one full encoder first, with a 75 A100-hour ceiling
(51 h update estimate). Check measured first-epoch end-to-end time, per-shell
development diagnostics and projected total before scheduling the OOF fits.
Provisional aggregate ceiling 1,500 A100-hours including the first fit for
encoder fitting/validation only. This is a stop-and-review ceiling, not an
estimate that all hours will be spent. Summary exports and posterior training
require separate measurement. No sealed-phase read and no production launch.
