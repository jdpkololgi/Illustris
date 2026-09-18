# Final verification handoff, 2026-09-18

Latest login21 closeout: main report58524609 completed0:0 in2896seconds;
RESULTS SHA2ca834400a766de7d6c357679f823fc9ee848f12b40ce7a562949a8186c3303f.
All three figures have been visually inspected and copied unchanged to this
evidence directory. Step4 was terminated on normal allocation release before
publishing outputs. One JSON-only recovery58526345 completed0:0 in12seconds:
CASE_REPORT_AUDIT and CLOSEOUT_AUDIT pass all688cases/384preview cases/32cells.
Final accounted usage including recovery is78.0075GPUh/2.536111CPU-nodeh;
Scratch113.64992GiB;23.0253hours to resource closeout. No new scientific run.

The login21 owner has read the added P12/joint-structure/CFM requirement and the
Gemini attachment in full, and is integrating the login35 literature review.
Do not mark the goal complete until the added Gemini sanity-check review and
the final science-log/field-plan updates are integrated.

Important interpretation for the literature owner: oracle regional mass and
the oracle delta core-summary difference are deterministic by block-mass
construction (regional RMSE/spread about7e-16). Their generic central interval
coverage numbers are roundoff/tie sensitive, not a continuous-posterior
calibration test with the quoted finite-M target. The frozen raw values must
remain, but explicitly caveat this in any pooled oracle coverage table.
Use the nondegenerate tidal/gap diagnostics and proper scores for inference;
oracle scores still refer to a different conditioning problem, not deployment.

Continuation acknowledgment, 13:21 UTC: the login21 continuation owning step4
and exec session31824 has read this handoff. It will let that existing verifier
finish, use final_report_audit.py after terminal completion, and own the final
scientific report plus SCIENCE_LOG/field-plan closeout. Do not start another
auditor or concurrent final-document edit. All extra audits share the existing
allocation; their steps are not additional node-hour charges.

The original frozen report remains active as CPU job58524609, step0, on
nid004217. The separate case-report verifier is already active as step4;
its launcher is PID223905 on login21. Do not launch a duplicate or alter it.
The app reports two in-progress turns for the same goal task; this handoff
avoids overlapping closeout edits. The goal is not complete yet.

## Additional checks completed in the current continuation

- GPU58524030: COMPLETED0:0,988seconds, all four worker codes0.
  Native09_ACCOUNTING totals78.0075GPUh and1.7283333CPU-nodeh before
  the live CPU report; the completed density preview is already included.
- CHECKPOINT_INTEGRITY.json already existed and passed; it was not rerun.
  Step1 verified30actual checkpoint payloads,745source files,14data receipts.
- Independent draw_inventory.py (commit34c7b68) ran as report job step3,
  COMPLETED0:0,173seconds. Its exclusive/exact one-CPU step did not reserve
  another allocation or overlap the report's CPU cores.
- analysis/DRAW_INTEGRITY.json records all688case identities,4160fine
  chunks/33280fine draws and984coarse chunks/7872coarse draws. All5144actual
  array payload hashes match:34,634,901,688bytes. Shared-parent bindings,
  checkpoint pointers and exact chunk sets also pass. This independently
  complements DRAW_LEDGER_INTEGRITY.json and the frozen report's own hashes.
- The metadata-only final_report_audit.py (commit985b2a9) is prepared and
  syntax-checked, but **not executed**. It requires EXPERIMENT_COMPLETE.json,
  RESULTS/FIGURES and the payload audits. It checks all688case receipts,
  all384preview density/spectral outputs and independently recomputes all32
  progression cells, then writes analysis/CLOSEOUT_AUDIT.json exclusively.
  It does not duplicate field/tidal computation or open target arrays.

## Still required

Let the existing report/case verifier finish. Inspect RESULTS, the registered
diagnostic summary, all three figures, terminal scheduler/resource accounting
and storage/deadline limits. Reconcile final scores and contrast decisions,
document H1/H2, conditional coverage, tidal/eigengap/dependence controls and one
justified next step in the science log/field-model plan. Finish the full
requirement audit before marking the goal complete. Do not relaunch training,
change frozen runtime1639851 or access sealed phases.

No new allocation was requested by these additional checks. Audit runtime is
inside the existing CPU report allocation and must not be double-counted.
