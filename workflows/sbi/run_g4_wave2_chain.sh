#!/bin/bash
# G4-PROPER wave-2 orchestrator (tmux, login-node, SSH-independent). Same
# idempotent liveness logic as run_g4_chain.sh. Items: D seed 43, D seed 44
# (seed variance for the headline point-attention result), F (DGCNN + curated
# features). Runs 2 at a time under the interactive QOS cap; exits when all three
# results exist.
#   tmux new-session -d -s g4_wave2 'bash ~/TNG/Illustris/workflows/sbi/run_g4_wave2_chain.sh'
set -uo pipefail

REPO=/global/u2/d/dkololgi/TNG/Illustris
RUNS=/pscratch/sd/d/dkololgi/abacus/sbi_runs
LOGS=/pscratch/sd/d/dkololgi/logs
CHAINLOG=$LOGS/g4_wave2_chain_$(date +%Y%m%d_%H%M%S).log
STALE_MIN=25
MAX_FAILS=2
declare -A FAILS
log() { echo "[$(date '+%F %T')] $*" | tee -a "$CHAINLOG"; }

ITEMS=(
  "D43|$RUNS/g4_p1d_pointattn_radius_seed43/p1d_pointattn_results.txt|$LOGS/g4_p1d_seed43_*.log|bash $REPO/workflows/sbi/run_g4_p1d_seed_interactive.sh 43"
  "D44|$RUNS/g4_p1d_pointattn_radius_seed44/p1d_pointattn_results.txt|$LOGS/g4_p1d_seed44_*.log|bash $REPO/workflows/sbi/run_g4_p1d_seed_interactive.sh 44"
  "F|$RUNS/g4_p1f_dgcnn_curated/p1e_dgcnn_attn_results.txt|$LOGS/g4_p1f_dgcnn_curated_*.log|bash $REPO/workflows/sbi/run_g4_p1f_dgcnn_curated_interactive.sh"
)

log "g4 wave-2 chain started on $(hostname); items: D43 D44 F"
while true; do
  all_done=1; launched_this_pass=0
  for item in "${ITEMS[@]}"; do
    IFS='|' read -r name results logglob cmd <<< "$item"
    if [ -s "$results" ]; then continue; fi
    all_done=0
    if [ "${FAILS[$name]:-0}" -ge "$MAX_FAILS" ]; then continue; fi
    newest=$(ls -t $logglob 2>/dev/null | head -1 || true)
    if [ -n "$newest" ] && [ -n "$(find "$newest" -mmin -$STALE_MIN 2>/dev/null)" ] \
       && ! tail -6 "$newest" 2>/dev/null | grep -qE \
            'salloc: error|Relinquishing|Job allocation .* revoked|EXITED|Traceback|OutOfMemory|CANCELLED'; then
      continue
    fi
    if [ "$launched_this_pass" -eq 1 ]; then continue; fi
    log "$name: launching: $cmd"
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
  if [ "$all_done" -eq 1 ]; then log "ALL wave-2 results present — chain complete."; break; fi
  sleep 300
done
