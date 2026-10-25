#!/usr/bin/env bash
# Rewind autopilot runner (rules: AUTOPILOT.md).
# Starts one headless Claude Code run at a time and reads results/autopilot/status.json after each:
#   WAITING_JOB → wait until its PID exits, then start the next run
#   DONE_PHASE  → start the next run (next phase); stop after phase 8 (step 8.2 is never run)
#   NEEDS_USER, FAILED, a non-zero claude exit, a missing status.json → stop
# Usage:
#   bash scripts/autopilot.sh                     # the loop
#   bash scripts/autopilot.sh --smoke LOG PROMPT  # one run with the same flags (permission check)
# Stop gracefully: touch results/autopilot/STOP. Stop now: kill "$(cat results/autopilot/runner.pid)".
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

AP=results/autopilot
STATUS=$AP/status.json
POLL_S=${AUTOPILOT_POLL_S:-300}  # how often to check a WAITING_JOB pid (seconds)
MAX_RUNS=40  # backstop against a runaway loop

PROMPT='Read CLAUDE.md, AUTOPILOT.md and PROGRESS.md. Continue from the current phase and step. Follow the phase routine (build, Done-when check, PROMPT R self-review, PROGRESS.md, commit per step). Stop and write status.json at a long job, at a decision the rules can'"'"'t make, on failure, or at the end of the phase.'

# Flags verified against `claude --help` of Claude Code 2.1.310.
ALLOWED='Read,Edit,Write,Glob,Grep,Bash(uv *),Bash(git *),Bash(cmake *),Bash(make *),Bash(g++ *),Bash(nohup *),Bash(setsid *),Bash(pkill -F results/logs/*),Bash(ls),Bash(ls *),Bash(cat *),Bash(tail *),Bash(head *),Bash(wc *),Bash(grep *),Bash(nvidia-smi),Bash(nvidia-smi *),Bash(ps *),Bash(df *),Bash(du *),Bash(date),Bash(date *),Bash(mkdir *),Bash(echo *)'
DISALLOWED='Bash(git push*),Bash(git reset --hard*),Bash(git clean*),Bash(sudo*),Bash(rm -rf*),Bash(rm -fr*)'

# The runner may be launched from inside a Claude Code session: drop that session's markers so each run is an
# independent top-level session.
unset CLAUDECODE CLAUDE_PID CLAUDE_EFFORT CLAUDE_CODE_ENTRYPOINT CLAUDE_CODE_EXECPATH CLAUDE_CODE_SESSION_ID \
      CLAUDE_CODE_CHILD_SESSION CLAUDE_CODE_SESSION_ATTENDED CLAUDE_CODE_MESSAGING_SOCKET CLAUDE_CODE_MESSAGING_TOKEN
export REWIND_AUTOPILOT=1  # read by the local .git/hooks/pre-push guard

mkdir -p "$AP"
exec 9>"$AP/runner.lock"
flock -n 9 || { echo "another autopilot runner holds $AP/runner.lock" >&2; exit 1; }
echo $$ > "$AP/runner.pid"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }

stop() {  # stop REASON [EXIT_CODE]
  log "STOP: $1"
  printf '\n## %s — runner stopped\n- %s\n' "$(date '+%F %H:%M')" "$1" >> "$AP/log.md"
  exit "${2:-0}"
}

CHILD=
on_signal() {
  [[ -n "$CHILD" ]] && kill -TERM "$CHILD" 2>/dev/null
  stop "runner received a signal; current claude run terminated" 130
}
trap on_signal TERM INT

run_claude() {  # run_claude LOG NAME PROMPT → return code of claude
  claude -p "$3" --permission-mode acceptEdits --permission-prompts none \
    --allowedTools "$ALLOWED" --disallowedTools "$DISALLOWED" \
    --output-format stream-json --verbose --name "$2" > "$1" 2>&1 &
  CHILD=$!
  local rc=0
  wait "$CHILD" || rc=$?
  CHILD=
  return "$rc"
}

summary() {  # the final result event of a stream-json log, compacted
  jq -R -c 'fromjson? | select(.type == "result")
            | {subtype, is_error, num_turns, cost_usd: .total_cost_usd, denials: (.permission_denials | length)}' "$1" | tail -1
}

field() { jq -r --arg k "$1" '.[$k] // empty' "$STATUS"; }

if [[ "${1:-}" == --smoke ]]; then
  rc=0; run_claude "$2" autopilot-smoke "$3" || rc=$?
  log "smoke run exited rc=$rc $(summary "$2")"
  exit "$rc"
fi

n=$(find "$AP" -maxdepth 1 -name 'run_*.log' | sed -E 's/.*run_([0-9]+)\.log/\1/' | sort -n | tail -1)
n=${n:-0}
runs=0
log "runner started (pid $$), last run number $n"

while :; do
  [[ -f "$AP/STOP" ]] && stop "STOP file found"

  if [[ -f "$STATUS" ]]; then
    jq -e . "$STATUS" > /dev/null || stop "status.json is not valid JSON" 1
    status=$(field status); phase=$(field phase); step=$(field step)
    log "status: $status (phase $phase, step $step): $(field reason)"
    [[ "$step" =~ ^(8\.[2-9]|9\.|10\.) ]] && stop "reached step $step; step 8.2 onward is not run by the autopilot"
    case "$status" in
      DONE_PHASE)
        [[ "$phase" =~ ^[0-9]+$ ]] || stop "status.json phase '$phase' is not an integer" 1
        (( phase >= 8 )) && stop "phase $phase done (to Gate 8.1); step 8.2 is Ashutosh's"
        ;;
      WAITING_JOB)
        pid=$(field pid)
        [[ "$pid" =~ ^[0-9]+$ ]] || stop "WAITING_JOB without an integer pid" 1
        log "waiting for pid $pid (checking every ${POLL_S}s)"
        while kill -0 "$pid" 2>/dev/null; do
          sleep "$POLL_S" &
          wait $! || true
          [[ -f "$AP/STOP" ]] && stop "STOP file found while waiting for pid $pid"
        done
        log "pid $pid has exited"
        ;;
      NEEDS_USER|FAILED) stop "$status at step $step: $(field reason)" ;;
      *) stop "unknown status '$status'" 1 ;;
    esac
  fi

  runs=$((runs + 1))
  (( runs > MAX_RUNS )) && stop "MAX_RUNS=$MAX_RUNS reached" 1
  n=$((n + 1))
  run_log=$AP/run_$n.log
  touch "$AP/.run_start"
  log "run $n starting → $run_log"
  rc=0; run_claude "$run_log" "autopilot-run-$n" "$PROMPT" || rc=$?
  result=$(summary "$run_log")
  log "run $n exited rc=$rc $result"
  (( rc != 0 )) && stop "claude exited with code $rc in run $n (see $run_log)" 1
  [[ -n "$result" && "$(jq -r '.is_error' <<< "$result")" == true ]] && stop "run $n ended with an error result (see $run_log)" 1
  [[ "$STATUS" -nt "$AP/.run_start" ]] || stop "run $n did not write $STATUS (see $run_log)" 1
done
