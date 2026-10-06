# Can model diversity repair the CFM coverage gaps? Compatibility pilot

## Scientific objective and answer

The proposed intervention asks whether a fixed equal-weight mixture of two
independently trained CFM models can improve both regional tidal-anisotropy
uncertainty and interior galaxy-weighted class-fraction uncertainty. It could
help when component errors differ; it cannot remove a bias shared by both
components merely by adding model labels to the draws.

**This pilot does not answer that coverage question.** It establishes a bounded
prerequisite: the models can be run under matching full-footprint conditions,
with an unchanged deterministic integrator and tested physical decoder. The
scientific hypothesis remains untested, not accepted and not falsified.

## Why the pilot was shortened

The user approved one shared GPU up to one hour and a CPU allocation up to
30minutes. Historical ph012 SGC generation alone takes2997seconds for four
NFE128 draws. The initial full-length comparison could not fit that budget.
Before results were examined, the protocol was amended to use four NFE2
trajectories over the complete SGC canvas in each of ph012/ph013, for:

1. the immutable reference implementation with seed17;
2. the extended interface with unchanged seed17;
3. the same extended interface with seed29.

SGC was chosen for its smaller computational footprint, not scientific scores.
All786/797 SGC windows remain present. No truncated spatial context is silently
treated as full-cap replay. No truth fields, class fractions, property models,
training or untouched-phase payloads were used. NFE2 is an engineering probe,
not the model's scientific posterior sampler. Only state fingerprints and
receipts are retained, not misleading NFE2 physical field products.

## Results and their meaning

| Check | ph012 SGC | ph013 SGC | Meaning |
| --- | ---: | ---: | --- |
| Frozen versus extended seed17 coarse/fine arrays | Exact bytewise fingerprint agreement | Exact bytewise fingerprint agreement | Adding the model-seed interface did not change these shortened trajectories. |
| Seed29 normalizer, observation preparation, geometry and overlap weights | Match | Match | A comparison need not change these inputs alongside model weights. |
| Seed29 coarse/fine checkpoints and generated states | Both distinct from seed17 | Both distinct from seed17 | The second arm actually uses a different trained model; this is not evidence that its differences are useful. |
| Positive exact-mass decoding, first registered window | Max relative mass error4.39e-16 | Max relative mass error4.51e-16 | Tested local decoder contracts survive; this is not an all-cell full-NFE or tidal-operator audit. |
| Peak GPU memory |14.17GiB|14.20GiB| The shortened SGC test fits comfortably; no current-pilot NGC benchmark is claimed. |
| Seed17 / seed29 NFE2 integration seconds |47.55 /47.57|48.55 /48.43| Component runtimes are similar for this engineering probe. |

All shared imported repository dependencies match their frozen baseline
sources. The new interface is fail-closed: seed29 is currently permitted only
for shortened development-phase benchmarks, not production/confirmation
sampling. Fourteen focused tests cover seed/phase boundaries, existing
full-cap guards and canvas/decoder invariants. An independent receipt audit
checks124 conditions. None of these counts measures scientific calibration.

## Feasibility of the actual coverage test

Use the existing NFE128 timings, not NFE2 posterior statistics, for planning:

- Second-model32-draw generation, both caps and both development phases:
  42.25GPUh.
- Original tidal processing:2.08GPUh, plus0.47h loading/tidal setup.
- Total approximately44.80GPUh; about79.94GiB additional density/tidal output
  under the existing storage scheme.
- A four-cap full-NFE baseline replay adds5.28GPUh sampling alone.

These are estimates from recorded seed17 runs; the full-length seed29 runtime
has not been measured. Verification, corrected-operator replay, scoring, queue
delays and retries are additional. Thus this candidate is technically feasible
but is NOT the cheap saved-draw intervention initially envisaged. The compatible
seed29 bank does not already exist in the inspected current pipeline.

The next scientific decision is whether this prospective coverage test merits
that budget. Pilot success does not imply it will work. Full-panel generation
requires new authorization, a completed execution manifest, and unchanged
registered co-primary gates; no automatic expansion, reweighting or new seeds.
B3 remains deferred. Real CIGALE data are not a substitute for the existing
mock truth in this coverage experiment.

## Provenance and operational closeout

Scratch root:
`/pscratch/sd/d/dkololgi/abacus/e2e_field_v3/cfm_mixture_pilot_20261006_v1`.
`SUMMARY.json` records exact replay and truth-independent future component
assignments. `REVIEW.json` retains the124 checks, phasewise results, historical
timing receipts, source hashes and estimate arithmetic. Component selection
is Bernoulli with probability0.5 per whole phase/draw; realized32-draw counts
are17/15 and18/14, not artificially forced to16/16.

Reference sampler SHA256:
`7940a599d41a865543a53a32fadc96c0e0491b8269e986a769237a57e080803c`.
The pre-existing NFE128 baseline binds that exact source and the unchanged
integrator. The pilot validates exact replay only at NFE2, not NFE128.

Entry points: `workflows/sbi/e2e_cfm_mixture_pilot.py` and
`workflows/sbi/e2e_cfm_mixture_pilot_audit.py`; bounded model selection lives in
`e2e_cfm_mixture_contract.py`, with regression tests in
`tests/phase4/test_e2e_cfm_mixture_contract.py`.

GPU allocation59422033/nid200312 COMPLETED0:0 in10m05s; both execution steps
COMPLETED0:0. Released immediately after validation. No separate CPU node
allocated. No original posterior products overwritten. Graphify AST refresh
again failed at filesystem flock Errno524; source queries remain usable and
this infrastructure limitation does not alter the scientific interpretation.
