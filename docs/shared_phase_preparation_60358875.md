# Shared-phase data preparation, 2026-10-27

## Scope and resource budget

Prepare deterministic R7 data for the shared 20-training/3-development/2-final-test policy. Final-test ph018/ph019 remain sealed. No GPU, learned encoder, posterior or CFM fit is included. Two CPU nodes, desi/interactive, Scratch: allocation 60358203 ended after 1816 seconds; replacement 60358875 has an 85-minute limit, keeping the combined maximum at 6916 seconds below the approved 7200 seconds.

## Verified CFM result

All 23 permitted phases have native matter, particle verification, geometry, conditions, targets, independent audits and whole-field completion receipts. The model normalizer uses all 20 training phases with equal 0.05 weights. Real loader canaries pass across all 23 phases and cover 960 unique training domains per complete whole-field epoch. The epoch schedule was corrected for the expanded training set; focused tests cover epochs 0 and 31 and preserve historical defaults.

Four imported native-count receipts used slabwise rather than full-array float64 reduction. Exact replay and payload hashes isolated the discrepancy to bookkeeping. The new-campaign receipt sums and dependent hashes were corrected with originals retained. Arrays and physical targets were unchanged; all four passed the original audit thresholds.

## Final P12 result

All 23 corrected phase datasets, raw adapter sets and parent-response products pass. Full-20 selection, field, graph and target transforms and all full-loader gates pass. All 20 leave-one-training-phase-out contracts pass, each excluding its omitted training phase. The runtime configuration is bound to the full completion receipt in configs/p12a_20phase_inputs_bound_20261027.json; the original unbound template remains separate.

The original finalizer completed the full contract but declined cross-fit at its internal reserve threshold. A separate bounded step completed cross-fit within the same allocation. The first response prerequisite wait expired without creating response arrays; the resumed worker completed all 23 products. All logs are preserved.

Twenty cached label-free selection-statistic sets reproduce the full selection curves exactly in both caps. Cross-fit contracts reuse those sufficient statistics while excluding the omitted phase. Cache bindings include input and implementation hashes. Angular-response reuse validates the 18-component integer-map recomposition and source lineage before writing new private P12 products; real checks passed for ph000 and ph014 provenance formats.

## Evidence and limitations

See shared_phase_preparation_60358875.json for launch and continuation source bindings, shared_phase_preparation_60358875_final.json for the passing final receipt inventory, and the canonical CFM readiness/DATA_LOADER_CANARIES.json receipt. Original failure logs remain retained. The graphify refresh previously failed with filesystem locking error 524; no locking bypass was used. Completed deterministic preparation does not establish learned posterior calibration or production qualification.

## Cleanup and remaining scientific work

All compute steps ended before the owning shell exited and released allocation 60358875. The scheduler receipt records the exact runtime. The final inventory reports all_preparation_receipts_pass=true. A direct-script inventory invocation first failed to resolve the workflows package; the package entry point then completed with exit 0.

CFM and P12 encoder data are ready for fresh training. P12-A learned summaries and posterior tables depend on future encoder fits and exports, so no posterior-training-ready dataset or calibration claim is made. No learned training was launched.
