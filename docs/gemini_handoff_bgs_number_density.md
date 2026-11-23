Investigate the high-redshift number-density discrepancy between DESI DR2 Loa
v2.1 **full flux-limited BGS_BRIGHT** and the Abacus SecondGen Kibo-v1 altMTL
mock used by our CFM. This is not the magnitude-limited BAO sample. The CFM
team is finishing mock calibration separately. Please do not change its files,
thin Loa to match the mock, or open the sealed ph018/ph019 data.

**Measurement.** On the intersection of random-supported NSIDE256 pixels
by cap and photometric system, ph006 Loa/mock unweighted count ratios in
observed-z shells [.01,.15,.25,.35,.45,.55] are NGC
1.030/1.027/1.113/1.274/1.918 and SGC 0.962/1.031/1.086/1.277/1.964.
Both use BRIGHT, observed z and ZWARN=0; Loa further requires GALAXY and
DELTACHI2≥25, which lack direct mock counterparts. One phase and one random
per release: not an area/completeness proof. The November replay confirmed
the exact CFM input; it did not discover or explain the discrepancy.
Receipt: /pscratch/sd/d/dkololgi/abacus/e2e_field_v3/lowz_loa_input_check_61811977/INPUT_CHECK.json

**What was tested.** Quality/flux cuts, catalogue versions, DR1 versus Loa,
assignment and redshift success, edge erosion, tile coverage and historical
IN_Y5 geometry were tested. Some matter, but none explains the full deficit:
the old mock already had too few eligible high-z targets before assignment.
Canonical Abacus v1 overshoots instead. In a ph000 parent sample at .45–.55
it had 11% fewer stored M_r<−22 galaxies than the old parent, yet 74.6% versus
34.6% passed the apparent-flux cut. Fixed-parent K/evolution/colour mapping
swaps can move counts substantially; changing the evolution parameter Q alone
worsens the v1 surplus. The realised v1 bright-end luminosity function and
the generator tables were checked. Photometric mapping is a strong lead, not
a unique physical cause; a single dark-matter snapshot does not by itself
imply a wrong tracer n(z).

**Other mocks.** Public Uchuu SV3 was also deficient at high z against *SV3*,
not Loa. The tested internal Uchuu Y3 and GLAM products lacked a selected tail
at .50–.55. Holi matched broad Loa counts more closely, but its available
preparation code uses Loa n(z) and placeholder magnitudes; the exact delivered
recipe was not proved. ANY-02/BRIGHT-02 BAO selections change the tracer.
No tested alternative is validated for the full-range galaxy–matter relation.

**Your deliverable:** distinguish established findings from hypotheses and
identify one new discriminating test. Start by putting both Abacus parents
and Loa on an independently justified common luminosity, colour and K/evolution
convention; test retention at the same apparent-flux limit before repeating
the matched observation pipeline. Matching n(z) alone is not a solution for
environment inference; check selected colours, clustering and truth linkage.

**Where to start:** ~/GraphWeb_DESI/docs/plan_mock_to_loa_alignment_investigation_20260926.md
is the chronological ledger. In the same directory read
p12a_realised_lf_20260928.md, p12a_mock_choice_20260928.md,
p12a_alignment_execution_results_20260926.md and
p12a_reference_mock_review_20260926.md. They link the plots, controls and
machine-readable evidence. The current status is **cause not definitively
established; no validated replacement or correction**.
