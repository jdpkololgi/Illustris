### 2026-09-06 - [code/diagnostic] D2 export recovery submitted; conditional evaluation blocked by genuine support mismatch

User authorized bounded D2 recovery and continuation to completion. The 100-step
export job 57928647 exited 75 at its internal 6600-second budget, after 145/256
cores; this was not the four-hour Slurm limit. Evaluation job 57928651 completed
its common 256-core evaluation but failed the derived-conditional M=1 galaxy
assertion. No scientific D2 completion or ranking is claimed.

Read-only audit allocation 57985230 (nid008393, one shared-interactive GPU)
COMPLETED 0:0 and was released after 2m46s. All 50-step archive shard hashes,
canonical support masks and float32 coordinate identities agree. Exactly
736/133698 galaxies across 84 cores have M=0 under archive rounding, canonical
float64 rounding and the alternative cell rule; there are zero rounding-induced
support changes. Thus removing the assertion alone is not a numerical repair.
The original all-row galaxy sample and support mask remain unchanged. A supported
conditional subset would require an explicit amended estimand, identical handling
for all matched references, and separate accounting of unsupported rows.

The audit also verified hashes and the frozen panel prefix for all 145 saved
100-step shards. One authorized recovery job 57985431 was submitted with the
unchanged, clean 467f442c5c54864658fdfaf948335d6e11a647fe worktree and config hash
3143ce1dfdb9546d3eb40413feab91bafa11b70718a2e4bb0ecb451080793533.
It resumes the remaining 111 cores through the existing launcher; no automatic
retry was added. Job is pending Priority at this record. Downstream evaluation,
decision and dispatcher jobs remain blocked on the old failed chain; no replacement
evaluation or scientific-contract modification has been made pending direction.

Audit source: workflows/sbi/p12f3_d2_recovery_audit.py.
Audit receipt:
docs/evidence/p12/p12f3_d2_20260906/READONLY_AUDIT_57985230.json.
Original large artifacts remain under the official D2 Scratch root.

P12-B follow-up diagnosis: joint gradients are present even at update 3000, so a
disconnected encoder is not supported by the trace. Its final logged pre-clipping
gradient norm is 12.89 (clip threshold 5), with encoder max gradient 1.616.
This is not evidence of convergence or adequate representation use. One seed,
one posterior-training phase, 3000 updates, and head/encoder rates 5e-4/1e-5
cannot establish that joint FMPE is intrinsically unhelpful. Proposed next checks
are loss/clipping trajectories, relative encoder movement and feature-use
ablations on internal data, followed only if warranted by a registered
warm-start/learning-rate/budget study with phase/seed replication. No new P12-B
training or ph006-driven tuning is launched; ph001 is not accessed.
