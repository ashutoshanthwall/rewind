# Autopilot log

## 2026-10-26 01:35 — NEEDS_USER — step 4.2
- **4.1 done** (commit `d03c500`): motion-aware dedup; tuning sweep, rule "fewest kept frames at recall >= 0.95"
  -> `dedup_motion_pixel_thresh: 1.0` (62.9 % of candidates kept, pipeline recall ±2 s 0.990, strict 0.941).
  Flagged: quiet cameras pay for it (G330 recall 0.83, n = 23; IR G474 0.85, n = 33); dev is G330-heavy.
- **4.2 so far:** GPU preprocessing without centre crop (cosine vs open_clip PIL path >= 0.9993); crop selection with
  `min_crop_px` on the box's longer side and `crop_rank: moving_first` (tuning actor coverage 0.331 vs 0.169);
  embedding job, store, NumPy search core, collect (4.3) and dev ablation (4.4) code written, tested (314 pass) and
  smoke-tested end to end on 2 clips. Gate rule operational form pre-registered (`rewind/eval/gate.py`, commit
  `0eeb29a`) before any dev result. Tuning embeddings for all 3 candidate tags exist (24.2 min job).
- **Stopped:** pretrained-tag check is a near-tie (laion2b best mean AP@50 0.105 but 12 wins / 16 losses vs
  datacomp_xl). See status.json for the question and options.
- **Process deviation:** `setsid` and the `&` operator are denied in this headless session (as is `$!`), so the
  AUTOPILOT.md 6 launch line cannot run. The 24-min job ran via the harness background runner (`uv run`, no `&`;
  PID file written by the job itself via `--pid-file`) and the run waited for it instead of ending WAITING_JOB.
  To restore the documented flow, allow `setsid` (or `nohup ... &`) in the runner's permission list.
- **Next run starts with:** set `appearance.pretrained` (and `appearance.model` if openai) from the answer; launch
  `scripts.embed_appearance` for all footage (tuning clips of the chosen tag are already done and will be skipped;
  ~250 clips, est. ~1 h single model); then `scripts.collect_appearance` (4.3) and
  `scripts.appearance_search --ablation` (4.4, Gate 4.4 by the pre-registered rule).
- Weights downloaded so far: 0 GB of 5 GB (all three CLIP tags were already cached).
