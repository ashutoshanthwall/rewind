# AUTOPILOT — unattended run of Phases 4 → 8.1

Written Mon 26 Oct 2026, before any Phase 4 result exists, on Ashutosh's instruction. Every headless run started by
`scripts/autopilot.sh` reads this file together with `CLAUDE.md` and `PROGRESS.md`.

## 1. Scope and precedence

- Covers step 4.1 through Gate 8.1 (final config frozen and committed). **Step 8.2 is never run by the autopilot.**
- CLAUDE.md still applies. Where this file differs, this file wins, for autopilot runs only:
  - Rule 1: a run continues step after step until the end of the current phase.
  - "Stop at GATE x", "update config only after I agree" and the 👤 boxes in the step prompts are replaced by §3.
  - Rule 6 is extended only by §4.
- Never edit `CLAUDE.md`, `AUTOPILOT.md` or `scripts/autopilot.sh`.

## 2. What one run does

1. Read `CLAUDE.md` (Part 1 + the Part 2 sections of the current phase), this file, `PROGRESS.md`, the end of
   `results/autopilot/log.md` and `results/autopilot/status.json` (if present).
2. If the previous run ended `WAITING_JOB`: check that job first (PID gone, log ends cleanly, outputs complete).
   If it failed, find the root cause (PROMPT D). One relaunch is allowed if the cause is clear and fixed;
   otherwise end with `FAILED`.
3. Work through the remaining steps of the current phase in order. For each step:
   build → check its **Done when** → PROMPT R self-review (inline) → update `PROGRESS.md` → commit `phaseX.Y: <summary>`.
4. End the run (§7) at a long job, at a decision §3 can't make, on failure, or at the end of the phase.
   A run never starts the next phase.

## 3. Decisions (written in advance)

- **Gates 4.4, 5.1, 6.2, 8.1:** the confirmed gate rule in `PROGRESS.md` (open-issue item 8) plus the decision rules in
  CLAUDE.md 1.7. Adopt the more complex variant only if its mean is better **and** it wins on more queries than it
  loses (paired, per query); otherwise keep the simpler one. Always report wins / ties / losses and the paired
  bootstrap CI of the difference.
- **Step 4.1 dedup threshold:** the setting with the fewest kept frames whose tuning pipeline recall is ≥ 95 %.
  Write it to `config.yaml`. If no setting reaches 95 %: `NEEDS_USER`.
- **Step 4.2 pretrained tag:** the agreed tuning method in `PROGRESS.md` (open-issue item 5). Near-tie: `NEEDS_USER`.
- **Gate 5.1:** a model that does not install, or does not fit 6 GB, within its 1-hour timebox is dropped
  (documented). If no video model is left, the action branch is dropped (`action.enabled: false`).
- **Gate 6.2:** drop (`events.enabled: false`) if the paired rule says drop on dev `vehicle_event` queries, or if the
  branch is not working by **Wed 28 Oct 2026 13:00** local time (check with `date`). Keeping it also needs
  "reasonable" tuning precision/recall (CLAUDE.md 1.7), which has no agreed number: if the paired rule says keep,
  end with `NEEDS_USER` and the precision/recall table.
- **Step 7.4 fusion weights:** no tuning objective is agreed yet. After the tuning sweep, end with `NEEDS_USER` unless
  `PROGRESS.md` records an objective by then.
- **Gate 8.1:** before freezing, recompute the deciding comparisons independently from the saved per-query results.
  Then freeze and commit the config as step 8.1 says, and end with `DONE_PHASE` (phase 8, step 8.1).
- **Licenses:** read every new package's or model's license from its actual license file and record it. MIT, BSD or
  Apache-2.0: proceed. Anything else: `NEEDS_USER`.
- **Anything a rule does not clearly decide:** `NEEDS_USER`. For every decision, record the evidence, the rule applied
  and the decision in `PROGRESS.md` (rule 16).

## 4. Pre-approved (extends rule 6)

- Installing the packages needed for **LanguageBind-Video**, **X-CLIP** (HF `transformers`) and **ByteTrack** into the
  project with `uv`, or into a separate uv environment under `envs/<name>/` (gitignored) if they conflict with the
  current torch. Timebox: 1 hour per model from its first install command; record the start time in `log.md`.
- Model weight downloads up to **5 GB in total** across all runs. Keep a running total in `log.md`.
- Anything beyond that (other packages, more weights, any other download over 1 GB): `NEEDS_USER`.
- `engine/` is empty. Step 7.1 needs Ashutosh to put the existing HNSW engine there (👤). If it is still empty at
  step 7.1: `NEEDS_USER`.

## 5. Never

- `git push`, `sudo`, rewriting git history (`--amend`, rebase, `reset --hard`, force), `git clean`.
- Deleting files outside `artifacts/` and `results/`.
- Touching `data/raw/`.
- Reading the test split: no test labels, events, queries or metrics. (Indexing test footage without labels is
  part of the pipeline and allowed, as in Phase 3.)
- Running step 8.2.
- Editing `eval/queries/`, or changing `config.yaml` values outside the current step.

## 6. Long jobs

- Any job expected to run longer than ~15 minutes:
  `setsid nohup <command> > results/logs/<job>.log 2>&1 & echo $! > results/logs/<job>.pid`
- Check with `ps -p <pid>` that the PID is the job itself (not a launcher that already exited) and that the log is
  growing. Then end the run with `WAITING_JOB` and that PID.
- One GPU job at a time. Check `df -h /` before any job that writes more than 1 GB.
- To stop your own job: `pkill -F results/logs/<job>.pid`. Never kill other processes.
- Run Python only through `uv run` (`uv run python …`, `uv run pytest`). Bare `python3` is not on the allowlist,
  and a denied command can't be approved in a headless run.

## 7. Ending a run

1. Commit every finished step first.
2. Write `results/autopilot/status.json` (gitignored):
   ```json
   {"status": "DONE_PHASE", "phase": 4, "step": "4.4", "pid": null, "reason": "Phase 4 done; Gate 4.4 decided: ..."}
   ```
   - `status`: `DONE_PHASE` (phase finished) · `WAITING_JOB` (long job running; `pid` set) · `NEEDS_USER` (a decision
     or approval outside these rules) · `FAILED` (could not finish a step).
   - `phase`: integer. `step`: the last step worked on, as `"X.Y"`. `pid`: integer when `WAITING_JOB`, else `null`.
   - `reason`: one or two sentences. For `NEEDS_USER`, the exact question and the options.
3. Append a dated entry to `results/autopilot/log.md` and commit it:
   ```
   ## 2026-10-26 14:05 — WAITING_JOB — step 4.2
   - what was done, decisions taken, what the next run starts with
   - weights downloaded so far: X GB of 5 GB
   ```

## 8. For Ashutosh: watching, stopping, resuming

- Human-readable state: `results/autopilot/log.md`, `results/autopilot/status.json`, `results/autopilot/wrapper.log`.
  `run_<n>.log` is the raw stream-json transcript of run n.
- Stop after the current run: `touch results/autopilot/STOP`. Stop now: `kill "$(cat results/autopilot/runner.pid)"`
  (also ends the running claude; a long job launched with `setsid` keeps running).
- Resume after `NEEDS_USER` / `FAILED`: record the answer in `PROGRESS.md` (or this file), delete `results/autopilot/STOP`
  if present and `results/autopilot/status.json`, then relaunch:
  `nohup systemd-inhibit --what=idle:sleep:handle-lid-switch --why="Rewind autopilot" bash scripts/autopilot.sh >> results/autopilot/wrapper.log 2>&1 &`
- Known stops ahead: step 7.1 if `engine/` is still empty; Gate 6.2 if the paired rule says keep; step 7.4 (no
  agreed weight objective).
