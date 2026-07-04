#!/bin/bash
# G4-PROPER unattended wave-1 orchestrator — designed to run inside tmux on a
# login node, INDEPENDENT of any Claude Code session or SSH connection.
#
#   tmux new-session -d -s g4_chain 'bash ~/TNG/Illustris/workflows/sbi/run_g4_chain.sh'
#
# Idempotent: every pass it checks, for each wave-1 run (B, C, D, E), whether its
# RESULTS FILE exists (done -> skip). If missing, it checks whether a live run is
# producing it (newest matching log modified within STALE_MIN minutes -> leave it
# alone). Otherwise it launches the runner (blocking salloc). A fast failure
# (<120 s, e.g. QOSMaxSubmitJobPerUserLimit) is treated as "no slot yet" and
# retried next pass; a slow failure counts against MAX_FAILS for that run.
# Exits when all four results exist. Everything is logged.
set -uo pipefail

REPO=/global/u2/d/dkololgi/TNG/Illustris
RUNS=/pscratch/sd/d/dkololgi/abacus/sbi_runs
LOGS=/pscratch/sd/d/dkololgi/logs
CHAINLOG=$LOGS/g4_chain_$(date +%Y%m%d_%H%M%S).log
STALE_MIN=25          # a live run prints at least every ~8-10 min
MAX_FAILS=2
declare -A FAILS

log() { echo "[$(date '+%F %T')] $*" | tee -a "$CHAINLOG"; }

# name | results file | log glob | launch command
ITEMS=(
  "B|$RUNS/g4_p1b_segnn_union/p1b_segnn_results_path1_wedge_union_r10hmpc_gnn_arrays.txt|$LOGS/g4_p1b_segnn_union_*.log|bash $REPO/workflows/sbi/run_g4_p1b_segnn_interactive.sh union"
  "C|$RUNS/g4_p1b_segnn_radius/p1b_segnn_results_path1_wedge_radius_r10hmpc_gnn_arrays.txt|$LOGS/g4_p1b_segnn_radius_*.log|bash $REPO/workflows/sbi/run_g4_p1b_segnn_interactive.sh radius"
  "D|$RUNS/g4_p1d_pointattn_radius/p1d_pointattn_results.txt|$LOGS/g4_p1d_pointattn_*.log|bash $REPO/workflows/sbi/run_g4_p1d_pointattn_interactive.sh"
  "E|$RUNS/g4_p1e_dgcnn_attn/p1e_dgcnn_attn_results.txt|$LOGS/g4_p1e_dgcnn_attn_*.log|bash $REPO/workflows/sbi/run_g4_p1e_dgcnn_attn_interactive.sh"
)

log "g4 chain started on $(hostname); items: B C D E"
while true; do
  all_done=1
  launched_this_pass=0
  for item in "${ITEMS[@]}"; do
    IFS='|' read -r name results logglob cmd <<< "$item"
    if [ -s "$results" ]; then continue; fi
    all_done=0
    if [ "${FAILS[$name]:-0}" -ge "$MAX_FAILS" ]; then
      log "$name: skipped permanently after $MAX_FAILS slow failures"
      continue
    fi
    newest=$(ls -t $logglob 2>/dev/null | head -1 || true)
    if [ -n "$newest" ] && [ -n "$(find "$newest" -mmin -$STALE_MIN 2>/dev/null)" ]; then
      continue   # a live run is producing this item — leave it alone
    fi
    if [ "$launched_this_pass" -eq 1 ]; then continue; fi
    log "$name: results missing, no fresh log -> launching: $cmd"
    t0=$(date +%s)
    if $cmd >> "$CHAINLOG" 2>&1; then
      log "$name: runner exited cleanly"
    else
      dt=$(( $(date +%s) - t0 ))
      if [ "$dt" -le 120 ]; then
        log "$name: fast failure (${dt}s) — likely QOS slot limit; will retry"
      else
        FAILS[$name]=$(( ${FAILS[$name]:-0} + 1 ))
        log "$name: SLOW failure (${dt}s), fail count ${FAILS[$name]}/$MAX_FAILS"
      fi
    fi
    launched_this_pass=1
  done
  if [ "$all_done" -eq 1 ]; then
    log "ALL wave-1 results present — chain complete."
    break
  fi
  sleep 300
done
