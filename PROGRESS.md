# PROGRESS — Rewind

Deadline: Fri 30 Oct 2026 · Current phase: **1 — Ingest (1.1, 1.2 done; 1.3 built, 👤 5-event check pending); next: 1.4 — Throughput**

---

## Step 0.1 — Repo, config, experiment log (Sat 24 Oct)

**Done**
- Repository layout from CLAUDE.md 1.8: `rewind/` package with empty subpackages (ingest, motion, detect, track, embed, search, eval, api); `.gitkeep` in empty tracked dirs; `data/` and `artifacts/` subdirs created locally (gitignored).
- `configs/config.yaml`: section 1.9 copied verbatim. 14 keys are still `TBD` (parsed by YAML as the string `"TBD"`); later steps must refuse to run on a `TBD` value rather than default it.
- `rewind/experiment_log.py`: `log_experiment(split, metrics, notes)` appends to `results/experiments.csv` with columns `timestamp, git_hash, config_hash, split, metrics, notes`.
- `results/experiments.csv`: header only (no dummy rows in the real log).
- `tests/test_experiment_log.py`: 22 tests, all passing (run in a throwaway venv with pytest 9.1.1, Python 3.14.8). Each test uses its own temporary git repo.
- `pyproject.toml` (pytest settings only), `.gitignore`.
- Self-review (PROMPT R) found 4 bugs. A failing test was written for each first (8 failed), then the code was fixed (22 pass):
  1. Untracked new files didn't mark the hash `-dirty`.
  2. Appending to the tracked `experiments.csv` made every following row `-dirty`. Changes under `results/` are now excluded from the dirty check.
  3. `None`, string and bool metric values were logged silently (e.g. as JSON `null`).
  4. `notes=None` was written silently as an empty cell; non-string metric keys were silently converted to strings.

**Decisions**
- `git_hash` gets `-dirty` appended when the working tree differs from HEAD: modified tracked files or new untracked (non-ignored) files, ignoring `results/` (experiment outputs). A row never claims a clean commit it didn't run on.
- `config_hash` = SHA-256 of the raw bytes of `config.yaml`. This is simple and needs no dependency. A comment-only edit also changes the hash, which is acceptable for verifying the frozen config at 8.2.
- Valid splits: `tuning, dev, test, none`. `none` is for runs that use no split (throughput, scaling). This set is a module constant because it is evaluation protocol, not a tunable.
- Metrics: a flat dict of string → finite number, stored as one JSON column with sorted keys. Anything else raises. Nested breakdowns go to `results/reports/`.
- `data/selection.txt` is tracked (exception to `data/*` in `.gitignore`), since the footage selection is a reviewable decision.

**Open issues**
- Known limitation: the git and config hashes are read when the row is logged, not when the run started. If code or config is edited during a multi-hour job, the row describes the end state. Don't edit tracked files while a logged job is running.
- No conda/mamba on this machine. Step 0.2 must decide between installing Miniforge and using a plain venv.
- System Python is 3.14.8, which is very new. PyTorch/open_clip wheels may not exist for it yet (VERIFY at 0.2). A conda env pinned to 3.11/3.12 avoids this.

**Key numbers**
- 22/22 tests pass (0.52 s). 14 `TBD` keys in `config.yaml`. 0 rows in `results/experiments.csv` (header only).
- `config.yaml` was diffed against the CLAUDE.md 1.9 block and is identical, apart from 3 added header comment lines.

**Verify by hand**
- Read `rewind/experiment_log.py` (≈100 lines) and be able to explain the `-dirty` rule and why `results/` is excluded.
- `git status --short` shows nothing from `data/` or `artifacts/`.
- Re-run the tests in the real step 0.2 environment (`python -m pytest -v`). So far they have only run in a throwaway venv.

---

## Step 0.2 — Environment + GPU check (Sat 24 Oct)

**Done**
- Environment built with **uv**, not Miniforge (reasons below): `pyproject.toml` (dependencies + PyTorch cu126 index), `.python-version` (3.11), `uv.lock` (exact versions + hashes).
- `scripts/gpu_check.py`:
  - Takes the model name from `config.yaml` and the weight tag from `--pretrained` (the config value stays `TBD`).
  - Checks that the model/tag pair is compatible before loading anything.
  - Loads CLIP on the GPU in fp16 and checks that every Linear/Conv2d weight is fp16.
  - Embeds 3 solid-colour images and requires each to match its own colour prompt best.
  - Exits non-zero on any failure.
- `tests/test_gpu_check.py`: 9 tests for the model/tag compatibility guard (no GPU or download needed).
- Toolchain checked by the user: ffmpeg 8.1.3 (libx264 + h264_nvenc), cmake 4.3.0, g++ 16.2.1 C++20 compile OK, aws-cli 2.37.0, MEVA bucket lists with `--no-sign-request`.
- The step 0.1 open issues (no conda, Python 3.14) are resolved; the 0.1 tests re-run in the real environment and pass.
- Self-review (PROMPT R) fixed 4 problems:
  1. The QuickGELU guard depended on open_clip's warning text. It is now an explicit metadata check.
  2. The guard had no tests.
  3. A stale VERIFY comment on `output_dim` was left after it had been checked.
  4. The VRAM figure left out the CUDA context.

**Key numbers**
- Python 3.11.17 (uv-managed, `~/.local/share/uv/python`); torch 2.14.1+cu126; torchvision 0.29.1; open-clip-torch 3.3.0; av 18.1.0; numpy 2.4.6; opencv-python-headless 5.0.0.93. 71 packages; downloading them took 17 min.
- Tests: **31/31 pass** (22 logger + 9 guard), 2.4 s.
- GPU check (ViT-B-16-quickgelu + openai, run with a scratch config): **PASS**.
  - Embeddings: shape (3, 512), fp16; 82.9% of 149.6M params in fp16.
  - Similarity: matching-colour 0.326–0.338 vs highest non-matching 0.277.
  - VRAM: torch peak allocated 370 MiB; whole device 795 of 5857 MiB in use.
- Disk: free space on `/` went from 135 → 129 GB. du reports `.venv` 6.9 GB and `~/.cache/uv` 6.8 GB, but these likely share blocks through hardlinks, so the drop in free space is the real cost. openai ViT-B/16 weights: 599 MB in the HF cache.

**Decisions**
- **uv instead of Miniforge.**
  - ffmpeg, cmake and g++ already come from Fedora, so conda's ability to ship non-Python tools isn't needed.
  - `uv.lock` gives exact reproducibility.
  - torch comes as the official CUDA wheels.
  - The system Python is untouched (`python-preference = "only-managed"`; confirmed via `sys.base_prefix`).
  - **Deviation from CLAUDE.md 1.8:** `environment.yml` is replaced by `pyproject.toml` + `uv.lock` + `.python-version`.
- **Package choices:**
  - `pyyaml` added, to read config.
  - `opencv-python-headless` instead of `opencv-python` (no GUI needed).
  - **PyAV instead of decord** (decord's last release was 2021).
  - CUDA 12.6 wheels (driver supports 13.4).
  - No detector or tracker packages.
- **fp16 means open_clip's mixed precision** (`precision="fp16"`): Conv/Linear/attention weights in fp16; embeddings, LayerNorm and logit_scale in fp32. Checked in open_clip 3.3.0 `factory.py`.
- **Model and weight tag must match on QuickGELU.**
  - open_clip only warns on a mismatch and then runs the weights with the wrong activation. Rewind treats a mismatch as an error.
  - `ViT-B-16` works with e.g. `laion2b_s34b_b88k` and `datacomp_xl_s13b_b90k`.
  - `ViT-B-16-quickgelu` works with `openai`, `dfn2b` and `metaclip_*`.

**Open issues**
- **`appearance.pretrained` is still TBD.** Choosing a tag from the QuickGELU group also means changing `appearance.model` to `ViT-B-16-quickgelu`. Decide on evidence (e.g. at 4.4) and ask before changing config.
- **The project config (`ViT-B-16`) has no GPU-check PASS yet**; it would need a compatible tag (e.g. `laion2b_s34b_b88k`, ~600 MB download). The only PASS so far used a scratch config with the quickgelu model.
- **Scripts can't `import rewind`.** The project isn't installed as a package; tests work only via pytest's `pythonpath`. The first script that needs the package (likely 1.1) needs a build backend in `pyproject.toml`. That is a new dependency, so ask first.
- **Possible separate environment later:** LanguageBind (Gate 5.1) may need an older torch. If so, it gets its own environment then.
- **Optional cleanup, your call:** `uv cache clean` would reclaim part of `~/.cache/uv`. It's a deletion, so I haven't run it.
- VERIFY: the PyTorch conda channel stopping new builds (2024) was stated from memory and not checked. It only affects the reasoning for choosing uv.

**Verify by hand**
- Read `scripts/gpu_check.py`; be able to explain why fp16 here is mixed precision and why a QuickGELU mismatch is a silent quality bug.
- Optional: `uv run python scripts/gpu_check.py --pretrained laion2b_s34b_b88k` for a PASS on the unchanged project config.

---

## Step 0.3 — Annotations + footage selection, GATE 0.3 (Sat 24 Oct)

**Done**
- **MEVA annotations:** official Kitware repo `gitlab.kitware.com/meva/meva-data-repo`, commit `421841a`, **CC-BY-4.0**
  (read from its LICENSE). Blobless + sparse checkout in `data/annotations/meva-data-repo/` (138 MB; box files
  `geom.yml` not fetched yet). Format: KPF YAML per 5-min clip; activity timespans in frames (`tsr0`).
  Source, format, facts and the exact checkout commands: `docs/meva_notes.md`.
- `scripts/survey_meva.py`: joins the S3 listing (3,956 videos, 475.9 GB), the clip/camera table and all annotations
  into `data/annotations/clip_survey.csv`. Third-party labels are attached to IR clips only, in separate `ir_*` columns.
- `scripts/select_footage.py`: camera/date choice → `data/selection.txt` (S3 key + size, generating command in the
  header) and `results/reports/selection_0.3.md` (per camera and per camera x hour window).
- `scripts/check_contrib_quality.py`: compares third-party labels with Kitware on shared RGB clips (quality only).
- `tests/test_survey_meva.py`: 22 tests (clip names incl. midnight and zero-length, KPF parsing, IR flag,
  half-open time overlap, contrib name mapping, max-not-sum, contrib labels can never land on RGB clips).
- PROMPT R on the first proposal fixed 6 problems (IR flag per camera, duplicate handling, untested overlap logic,
  "events" overclaim → "activity records", tsr0 off-by-one flag, README wording).

**Key numbers**
- Activity names: 37 (`documents/activity-names.txt`); vehicle start/stop are exactly **`vehicle_starts`** and **`vehicle_stops`**.
- **Selection: 300 clips, 24.99 h (RGB 20.74 h + IR 4.25 h), 39.86 GB.**
  - RGB: G436 (hospital), G424, G336 (school), G340 (bus), G330 (school, indoor), annotated clips only,
    on the annotated sessions of 2018-03-07, 03-11 and 03-15. 16,527 activity records (per clip, not unique events),
    981 `vehicle_starts`, 1,000 `vehicle_stops`. RGB clips: 120 evaluation-level, 129 training-level.
  - IR: **G474** (school, thermal), clips co-timed with the RGB selection. Third-party labels on 29 clips (2.42 h):
    148 records, 12 `vehicle_starts`, 15 `vehicle_stops`. No IR labels on 03-07.
  - Co-timing: 47 of 53 five-minute RGB slots have all 5 cameras. 16 thirty-minute blocks contain footage.
- Third-party label completeness vs Kitware on shared RGB clips: CMU-v2 0.23x records (0.22x vehicle start/stop),
  UMD-IBM 0.45x (0.50x).
- 53/53 tests pass.

**Decisions (GATE 0.3, approved by Ashutosh)**
- **Sessions instead of one continuous window.** MEVA records in sessions (no camera has more than 3.65 h of video on
  any date), so the selection uses the same annotated sessions on 3 dates. Deviation from CLAUDE.md 1.5.
  39.8 GB accepted.
- **IR camera G474** instead of G479. Kitware labels no IR footage (its 9 IR "labels" are `empty_37` = nothing
  present); third-party contrib labels do. G474 verified thermal by eye (glowing tyres and people) plus the MEVA abstract.
- **Contrib labels for IR only.** RGB ground truth stays Kitware-only; contrib labels never enter RGB evaluation.
- **Selection unit:** annotated RGB clips only, so unannotated RGB footage isn't indexed as unlabelled ground truth.

**Instructions recorded for later steps (from GATE 0.3)**
- **Step 2.1 (splits):** assign evaluation-level-labelled footage to **dev and test** wherever possible;
  tuning may take training-level footage.
- **Step 2.3 (metrics) and step 8.2 (test run):** for **IR**, report only recall-type metrics: Recall@k,
  seconds-to-first-hit on labelled events, event-rule recall. **Never precision or mAP for IR**, because the
  third-party labels are known to be incomplete.
- **Step 8.3 (failure analysis):** add the failure cause **"correct but unlabelled"**.

**Open issues**
- VERIFY at 1.3: `tsr0` end frame inclusive or exclusive (`empty_37` spans `[0, 9000]`, the clip has frames 0-8999).
- VERIFY at 1.3: how CMU and UMD cut timespans (numbering, boundary-crossing activities, agreement with Kitware) and
  how to merge two sources on the same IR clip (de-duplicate by temporal overlap, never sum).
- VERIFY: whether training-level Kitware labels are as complete as evaluation-level ones (affects 129 RGB clips).
- Camera views assumed fixed across the 3 dates (indoor/outdoor flag never changes); confirm by eye. This matters
  for per-camera mean-centering.
- `drop-4-hadcv22` (another video drop) was not surveyed; Kitware/contrib annotation folders only cover `drops-123`.
- `updates-r13/` has 2 replacement-looking files, neither in the selection.
- The swap changes (contrib parsing) did not get a separate PROMPT R pass.

**Verify by hand**
- Open `results/reports/selection_0.3.md`; the selected cameras, dates and totals should match the table above.
- Look at 1 frame per selected camera on each of the 3 dates (same view?).
- Read `docs/meva_notes.md`, especially the third-party-label rules.

---

## Step 0.4 — Download (Sat 24 → Sun 25 Oct)

**Done**
- `scripts/download.py`: reads `data/selection.txt`, downloads each key with `aws s3 cp --no-sign-request` to
  `data/raw/<name>.part`, checks **size and S3 ETag**, then renames it to the final name. Re-running resumes:
  files already under their final name with the right size are skipped. A final file with the wrong size is never
  touched (the script stops). Checks that 10 GB stays free (`--min-free-gb`). Logs to the terminal and
  `data/download.log`. When everything is verified it sets files to `0444` and `data/raw/` to `0555`. It refuses
  to lock if files outside the selection are present, and refuses to download into a locked folder.
  `--verify-existing` re-checks size + ETag of the files already there and changes nothing.
- **Why the ETag check:** the MEVA objects have no stored checksum and mostly multipart ETags (`<md5>-<parts>`),
  which `aws s3 cp` cannot check (no checksum fields came back from `head-object --checksum-mode ENABLED`).
  The script recomputes the ETag locally: plain MD5 for single-part objects, MD5 of the 8 MiB part MD5s for
  multipart ones. The 8 MiB part size was checked on all 1,073 objects under `drops-123-r13/2018-03-07/`.
  A file whose part count doesn't fit 8 MiB is refused, never passed.
- `tests/test_download.py`: 16 tests. They cover selection parsing (bad lines, duplicate names, empty file),
  resume planning, refusing a wrong-size file, permissions, ETag vectors (RFC 1321 `MD5("abc")`, 3-part toy case,
  impossible part count), the `download_one` rename only after the checks, a same-size corrupt download caught
  only by the ETag and kept as `.part`, and stray files.
- End-to-end test before the full run, on 2 G474 clips in the scratchpad (not `data/raw/`): the first run
  downloaded both, the second skipped both, the permissions were set, and a missing file in the locked folder
  was refused.
- PROMPT R found 4 problems and fixed them: size-only verification (ETag added), download loop untested
  (`download_one` extracted and tested), stray files only warned about (now an error), and my report claiming
  "verifies" for a size-only check.

**Key numbers**
- Ashutosh ran the full download in their own terminal in one run, without restarts:
  2026-10-24 22:04:03 → 2026-10-25 00:15:22 (**2 h 11 min, 5.1 MB/s average**). Pre-run estimate from one timed
  182 MB clip: about 1.9 h.
- **300/300 files, 39,855,653,762 bytes**. File names and sizes are identical to `data/selection.txt`.
  Permissions: files `-r--r--r--`, folder `dr-xr-xr-x`.
- `--verify-existing`: **size + ETag match for all 300 files** (115 s, read-only). Negative control: flipping one
  byte in a scratch copy of a real 182 MB file kept the size but failed the ETag check.
- Disk: 92 GB free on `/` after the download.
- 69/69 tests pass.

**Decisions**
- `data/raw/` is flat (file names are unique; the duplicate-name check enforces it). The S3 key prefix is
  recoverable from `data/selection.txt`.
- The bucket name, the 8 MiB part size and `--min-free-gb` stay in the script, not `config.yaml`. They are dataset
  facts and a disk-safety option, not pipeline tunables.
- To add files later, the folder must be unlocked by hand (`chmod u+w data/raw`). The script never unlocks it itself.

**Open issues**
- An ETag match proves the files equal the bucket objects, not that the bucket content is authentic. Integrity from
  import onward will be backed by SHA-256 at step 1.1.
- Scripts still import each other by path, not as the `rewind` package (no build backend). Revisit at 1.1 if
  scripts need `rewind.*`; adding a build backend changes `pyproject.toml`, so ask first.
- Carried over from 0.3: `tsr0` end inclusive/exclusive; CMU/UMD timespan conventions; training-level label
  completeness; camera views fixed across dates; `drop-4-hadcv22` not surveyed.

**Verify by hand**
- `ls data/raw | wc -l` → 300; `stat -c '%A' data/raw` → `dr-xr-xr-x`; try `touch data/raw/x` → it must fail.
- Play 1 RGB and 1 IR file from `data/raw/` (e.g. with `ffplay`) to confirm they decode.
- `tail -3 data/download.log` should end with `verified size + ETag of all 300 files`.

---

## Step 1.1 — Manifest (Sun 25 Oct)

**Done**
- `rewind/ingest/manifest.py` (reused by later steps):
  - parses filenames (`date.HH-MM-SS.HH-MM-SS.site.Gnnn.r13.avi`; impossible dates/times raise; end after midnight → next day);
  - frame ↔ seconds ↔ wall-clock conversion (`Video.frame_time`, `Video.frame_at`, bounded to existing frames);
  - container probe that also reads every packet and requires exactly `n_frames` packets with decode timestamps 0..n−1;
  - SHA-256 and the SQLite schema (`videos`, STRICT, `modality` restricted to `rgb`/`ir`).
- `scripts/build_manifest.py`:
  - checks every name before hashing or writing anything: the camera must be in the MEVA clip table, each camera has one site, every RGB clip has a Kitware annotation file;
  - per file: probe, name vs container duration check, hash, insert; resumable, one commit per file;
  - `--verify` re-hashes and re-probes everything and changes nothing;
  - writes `results/reports/manifest_1.1.md`.
  - Run from the repo root as `uv run python -m scripts.build_manifest`. That makes `rewind` importable with no build backend, which closes the 0.2/0.4 open issue without a new dependency.
- `tests/test_manifest.py`: 54 tests.
  - Filename parsing and timestamp conversion, with hand-computed values: midnight, out of range, and a round-trip of all 9,000 frames.
  - Probe on generated AVIs; SHA-256 FIPS test vector.
  - `build`/`verify` end to end on toy files: resume, changed size refused, duration mismatch refused with nothing written, one flipped byte detected, missing row detected.
  - Clip-table modality, annotation index, pre-validation.
- PROMPT R found 6 problems and fixed them:
  1. Frame timestamps unusable (see Decisions).
  2. `n_frames` assumed from the header, not checked.
  3. Annotation and site checks ran after inserting rows.
  4. A clip in both Kitware sources was silently overwritten.
  5. `build`/`verify` were untested and hard-wired to repo paths.
  6. DST was not mentioned, and a stale VERIFY was left in `docs/meva_notes.md`.
- Ashutosh checked by eye: camera views are the same on 03-07, 03-11 and 03-15 for all 6 cameras (`compare_<camera>.jpg`, one frame per date side by side). G474 thermal contrast changes a lot between dates; look at this again at 4.3.

**Key numbers**
- **300 videos, 25.01 h (RGB 20.76 h, IR 4.25 h), 39.86 GB.** The selection said 24.99 h because it used filename durations.
- All files: H.264, constant 30/1 fps, one stream. RGB 1920x1080, G474 352x240. Frames per clip: 8,831–9,013.
- Per camera: G330 51 clips / 4.25 h, G336 51 / 4.25, G340 49 / 4.09, G424 50 / 4.17, G436 48 / 4.00, G474 51 / 4.25. Each camera has 9–10 sessions.
- Annotation files: RGB evaluation-level 120 and training-level 129. G474 has third-party labels on 29 clips, plus 3 Kitware `empty_37` files.
- First build: about 1 min, plus a resume after the tolerance stop. `--verify` (re-hash + re-probe of 39.9 GB): 45 s, **ALL MATCH**. Two hashes cross-checked with `sha256sum`. A corrupted hash in a scratch DB copy was reported and gave exit 1.
- Tests: **123/123 pass**.

**Decisions**
- **Frame index = count of decoded frames from 0; time = index / fps.** These files are H.264 with B-frames in AVI, which stores no presentation timestamps. ffmpeg invents packet pts in decode order (pts = dts + 1), so decoded `frame.pts` / `frame.time` are scrambled (e.g. 1, 3, 2, 6), and ffprobe shows `pts=N/A`. Never use `frame.pts`. Don't seek by timestamp without checking where you land. This applies to every step that decodes video.
- **Wall-clock of frame 0 = filename start time.** Filename times are cut to whole seconds at both ends (15 names say 299 s for 300.0–300.33 s of video), so times are accurate to about ±1 s. Duration comes from the container, never from the name.
- **New config key `ingest.max_name_duration_diff_s: 2.0`.**
  - The first value, 1.0, was mine, set from a misread printout. It stopped the build at a real +1.33 s clip.
  - The observed range over all files is −0.77 to +1.33 s. The errors this check exists for (wrong fps, truncated file) are tens of seconds.
  - The same key is the session gap tolerance in the report.
- **Modality per camera** from the clip table (camera set `IR`), because 7 selected clips (session edges) are missing from the table.
- **Not an experiment:** no row in `experiments.csv`.

**Open issues**
- **Step 1.2 must compare proxy frame counts with the originals**, not just durations within 1 s. With invented timestamps, ffmpeg could drop, duplicate or reorder frames, and playback would drift from the indexed times.
- **Steps 1.4 / 3.1 / 4.x:** decode sequentially and count frames (see Decisions). Seeking needs its own check.
- VERIFY: the sub-second part of the start time is unknown (±1 s).
- VERIFY: times are naive local times, and 2018-03-11 is the US DST switch date. No selected clip spans 02:00–03:00. Never subtract times across dates.
- VERIFY: the clip table's cross-camera offsets all equal the filename start-time differences × 30 (no extra sync information), with the opposite sign to the clip-table README. Irrelevant while we never link cameras.
- `tsr0`: strong evidence for 0-based frames with an inclusive end. No activity ends at or past `n_frames`; in 134 of 221 annotated RGB clips the last activity ends exactly on `n_frames − 1`. Confirm at 1.3, together with the CMU/UMD conventions.
- Carried over: training-level label completeness; `drop-4-hadcv22` not surveyed; `appearance.pretrained` still TBD.
- Resolved here: the `import rewind` problem; 30 fps checked per file; camera views fixed across dates.

**Verify by hand**
- Pick 2 clips, run the query below, then play the G330 file. The camera, start time and duration must match the filename, and G330 must be the gym:
  `uv run python -c "import sqlite3; c=sqlite3.connect('data/rewind.sqlite'); [print(*r, sep=' | ') for r in c.execute(\"select clip, camera_id, modality, start_time, duration_s, n_frames from videos where clip in ('2018-03-11.13-50-00.13-54-59.school.G330', '2018-03-15.15-55-00.16-00-00.school.G474')\")]"`
  `ffplay data/raw/2018-03-11.13-50-00.13-54-59.school.G330.r13.avi`
- Read `rewind/ingest/manifest.py`, especially the frame index rule and `FRAME_SNAP_DECIMALS`. Be able to explain why `frame.pts` can't be trusted here and why the snap is needed (datetime stores 1/30 s as 33,333 µs).
- Open `results/reports/manifest_1.1.md` and check the totals above.

---

## Step 1.2 — Playback proxies (Sun 25 Oct)

**Done**
- `scripts/make_proxies.py`: ffmpeg → H.264 MP4, at most 720p (`scale=-2:'min(720,ih)'`, IR 352x240 not upscaled),
  `-movflags +faststart`, no audio. Frame-exact: `setpts=N/(fps*TB)` + `-fps_mode passthrough`, so proxy frame i is
  original frame i (decoded order, see the 1.1 frame index rule) and is shown at exactly i / fps.
  Each proxy is written as `.mp4.part` and renamed only after all checks pass:
  - same frame count (header and packet count) and fps as the original; duration within `proxy.max_duration_diff_s`;
  - every frame's presentation time is exactly i / fps of the original (sorted packet pts, exact fractions);
  - content alignment: over the first `proxy.align_check_frames` frames, frame-difference images of proxy and
    original (grayscale, 320 px wide) must correlate best at shift 0 among shifts −3..+3.
  Then a row goes into a new `proxies` table (path, frames, fps, duration, size, alignment correlations, full
  settings JSON). Resumable; a finished `.mp4` without a row is refused, not adopted (its settings are unknown).
  A failing clip is logged, gets no row, the run continues and exits 1 with the list; a re-run retries only those.
  Disk/config/DB problems stop the run at once. Log: `data/proxy/make_proxies.log`.
- `tests/test_proxies.py`: 13 tests. Hand-computed two-pixel correlation case; identical/static/shifted sequences;
  `make_one` on a generated moving-square AVI (frame count, no upscaling, record + changed-size detection);
  1920x1080 → 1280x720; a proxy that lost frame 0 but kept the count is rejected; a proxy with a 1-frame
  timestamp gap is rejected; decoded proxy times are exactly i/30; wrong frame count, unrecorded leftover proxy and
  non-integer fps are refused.
- Full run by Ashutosh in their own terminal: 299/300 on the first pass, 1 failed the alignment check (below);
  retried after raising `align_check_frames`: 300/300.
- Independent check of all 300 rows after the run: frame counts and fps equal the originals 300/300; max
  |duration difference| 0.00065 s; `moov` before `mdat` (faststart) 300/300; files present with recorded size
  300/300; no `.part` left; sorted packet pts == i/fps exactly 300/300 (also re-run through the final
  `probe_proxy`); best alignment shift = 0 in 300/300.
- 👤 Ashutosh opened the G340 proxy (03-07 10:55) and the retried G474 proxy (03-11 17:20) in Firefox: both play,
  seek fine, right views.
- PROMPT R found and fixed:
  1. Nothing checked that proxy frame i is *shown* at i / fps (only counts and total duration). Added the pts check
     to `probe_proxy` against the original's fps from the manifest. A first version compared against the
     container's `average_rate`, which is computed from the duration and shifts with a gap; the new test caught it.
  2. Integer fps was a comment only; `ffmpeg_command` now refuses a non-integer rate (`fps:g` would round it).

**Key numbers**
- **300 proxies, 8.60 GB** (RGB 249 clips 8.50 GB, avg 34.1 MB per 5 min; IR 51 clips 0.10 GB, avg 2.0 MB).
  Per camera: G330 1.75, G336 2.00, G340 1.05, G424 1.31, G436 2.40, G474 0.10 GB.
- Run time: 10:19:06 → 12:05:44 = **1 h 47 min for 25.0 h of video** (~14× realtime; pre-run estimate 1.75 h).
  ~22–25 s per RGB clip, ~3.5 s per IR clip.
- Alignment margin (corr at shift 0 minus best other shift): median 0.274, min 0.015, 19 of 300 below 0.05
  (mostly G474 and quiet G340/G330 openings).
- Disk: 83 GB free on `/` after the run.
- Tests: **136/136 pass**.

**Decisions**
- **libx264 veryfast CRF 23, not h264_nvenc:** 35 MB vs 88 MB per 5-min clip at similar speed (20.6 s vs 14 s per
  test clip). **keyint 250** (x264 default): G340 35.3 MB vs 64.5 MB at keyint 60; decoding a whole 250-frame GOP at
  720p (worst-case seek) takes 0.10 s vs 0.07 s. Browser check confirmed seeking is fine.
- **Alignment check by frame-difference correlation, not per-frame error.** In near-static CCTV the per-frame
  error is dominated by compression noise (G340: mean abs diff 0.982 at shift 0 vs 0.978 at +1); motion between
  frames pins the shift (0.55 vs 0.30).
- **`proxy.align_check_frames` 300 → 900 after the run.** `2018-03-11.17-20-00 G474` opens with ~10 s of nearly
  static thermal video (mean frame change 0.04 grey levels): shift 0 vs +1 tied at 300 frames (0.1536 vs 0.1539).
  Over 900 frames shift 0 wins (0.163 vs 0.139), over the whole clip clearly (0.235 vs 0.184), and in every
  300-frame window after the first. Same frame count and pts. So: not enough evidence, not a bad proxy. 900 frames
  costs 9.2 s per RGB clip vs 3.1 s (3000 would cost 32 s). The other 299 rows keep `align_check_frames: 300` in
  their settings JSON; they all passed at 300.
- **Separate `proxies` table** (one row per video) rather than columns on `videos`: keeps the evidence manifest
  unchanged and stores how each proxy was made and checked.
- New config section `proxy:` (codec, crf, preset, keyint, max height, duration tolerance, alignment-check
  parameters). `--min-free-gb` (default 10) stays a command-line safety option: it never changes the output.
- Cleanup approved by Ashutosh before the full run: dropped a stale `proxies` table from a test run (1 row, old
  `align_mad` column) and deleted 4 test proxies + log made with different settings.
- **Not an experiment:** no row in `experiments.csv`.

**Open issues / limitations**
- The alignment check covers only each clip's first 30 s (10 s for the 299 first-pass rows) and is weak on quiet
  thermal clips (min margin 0.015). The strong guarantees for the whole clip are the frame count and exact pts;
  together they rule out dropped/duplicated frames that change the count or timing, but not a content swap in
  the middle with the count preserved (no known ffmpeg path does that with passthrough).
- Proxy playback time t ↔ original frame `floor(t × 30)` holds by construction; browsers report `currentTime` with
  their own rounding, so the UI (step 9.1) must map times with the same snap as `seconds_to_frame`.

**Verify by hand**
- Done: browser play + seek on G340 and G474 proxies.
- Optional: read `scripts/make_proxies.py` `ffmpeg_command` and be able to explain `setpts=N/(fps*TB)` and
  `-fps_mode passthrough`, and why the alignment check uses difference images.

---

## Step 1.3 — Annotated events table (Sun 25 Oct) — 👤 5-event check PENDING

**Done**
- Box files fetched: the Kitware `geom.yml` of the selected clips (298 sparse-checkout paths incl. contrib IR files;
  0.958 GB by GitLab API sizes, under the 1 GB ask-first limit; 2 min 18 s; git store +100 MB).
  `scripts/annotation_patterns.py` now generates these patterns from `data/selection.txt` (checked: selects exactly
  the 252 Kitware box files present), and the recipe in `docs/meva_notes.md` uses it.
- Investigation before coding (5 read-only investigators, each re-checked by an independent skeptic with their own
  scripts): KPF semantics, format variants, clip-boundary handling, contrib IR labels, boxes vs video by eye and by
  optical flow. Results in `docs/meva_notes.md` → new section "Annotation conventions".
- `rewind/ingest/annotations.py`: Kitware parsing (both KPF line styles; activities/types via yaml, ~1 GB of geom
  line by line with key regexes, anything unrecognised raises), contrib NIST `.json` parsing (IR only),
  `merge_contrib`, clip-boundary linking (`boundary_links`, `group_ids`), IoU helpers. Every consistency check raises:
  dense geom coverage per actor, spans inside the clip, types for every actor, no orphan tracks, no duplicate ids,
  contrib localization shape and box coverage.
- `scripts/build_events.py`: builds `clip_labels`, `events`, `event_boxes` in `data/rewind.sqlite` in one transaction
  (report generated inside it); `--rebuild` replaces them and refuses once `eval/queries/HASHES` exists (frozen
  queries reference event ids). `--spot-check` prints the 5 events to check by eye with ready ffplay commands.
  Report: `results/reports/events_1.3.md`.
- Schema (in `rewind/ingest/manifest.py`): `events` (event_id, video_id, camera_id, activity_type, start/end frame
  (inclusive), start_s/end_s (proxy time), start_time/end_time (wall-clock ISO, end exclusive, fixed width),
  label_source, sources, src_ids, src_status, actor_types, n_actors, n_actors_with_boxes, at_clip_start/end,
  group_id); `event_boxes` (event_id, actor index, frame, x1..y2); `clip_labels` (status events/empty/none).
  `Video.frame_end_time` added (one rounding, = `end_time` on the last frame).
- `tests/test_annotations.py`: 27 tests. Both KPF line styles from real lines; both empty-marker styles alone;
  every guard; contrib JSON (x,y,w,h conversion, zero-length drop, boxes outside the span refused, other-clip keys
  refused); hand-computed IoU (25/175, 90/110); merge transitivity in the order that breaks a naive union;
  boundary linking (each exclusion alone, one-to-one on both sides, boxes taken at A's last / B's first frame);
  end-to-end builds on toy trees: RGB with a linked cut pair and an empty clip (times hand-computed), IR with two
  merged sources / no labels / a Kitware marker, Kitware activities on IR refused, no linking across cameras.
  Mutation check on a scratch copy: 5 mutants that the first test version let through now each fail.
- PROMPT R as an independent review (5 lenses: time/frame arithmetic, silent failures/config, test quality with
  hand mutation testing, independent audit with its own parser, project rules; each finding re-checked by a
  skeptic). The audit re-derived **all 9,678,632 Kitware boxes, every per-clip count, all 220 links and all 51 IR
  clips: no data error**. 20 smaller findings, all fixed:
  1. ISO times without microseconds when they were 0 (918 rows): pandas could not parse the column → fixed width.
  2. `frame_end_time` rounded twice: 5,508 end times 1 µs early → one rounding; clip-end events now equal the
     video end exactly (416/416).
  3. Report written after the commit → inside the transaction.
  4. `--rebuild` could silently renumber events after Gate 2.2 → refuses when `eval/queries/HASHES` exists.
  5. Contrib boxes outside the span were trimmed silently → raise (0 cases in the data).
  6. Kitware boxes not cut to the event span (1 actor, 18 frames) → cut, same rule as contrib.
  7. Spot check: fps hardcoded as 30; `ffplay -ss` starts at the previous keyframe (up to 8 s early) with no clock
     → fps from the manifest, commands draw a clock and the boxes, instructions explain the pre-roll.
  8. Fresh clone could not fetch the box files → `scripts/annotation_patterns.py` + recipe.
  9. `docs/meva_notes.md` stale, the referenced "Annotation conventions" section missing → written.
  10. Box-timing findings recorded nowhere → VERIFY notes in code and docs, caveat in the IR spot-check entry.
  11. Config comment counts wrong (25/19 → 24/18; the IoU values are continuous around 0.3: a judgement call).
  12. IR clips without events merged three different reasons → kept 'none' (not 'empty': IR labels are
      incomplete), reasons counted in the report.
  13. Test gaps (8): see the test list above.
- 👤 Ashutosh: chose to run step 1.4 before doing the 5-event check (their call); the check is still to do.

**Key numbers**
- **16,702 events** in 300 clips: Kitware RGB **16,527** (= the 0.3 survey exactly; 7,187 evaluation-level,
  9,340 training-level), contrib IR **175**. 35 of 37 activity types occur.
- **Vehicle start/stop:** RGB 981 `vehicle_starts`, 1,000 `vehicle_stops` (G336 266/273, G340 128/127,
  G424 292/296, G436 295/304; none on the indoor G330); IR 12 / 14.
- Per camera: G330 6,080 · G336 3,034 · G340 1,009 · G424 3,100 · G436 3,304 · G474 175.
- Clips: RGB 221 with events + 28 `empty_37` (G330 14, G340 5, G424 4, G336 3, G436 2); IR 29 with events + 22
  without (18 no annotation, 3 Kitware empty_37 marker, 1 contrib file with 0 records).
- Contrib: 205 raw records → 3 zero-length dropped → 27 merged away (duplicates/overlaps) → 175 events.
- Clip boundaries: 244 back-to-back clip pairs; **220 links** → 16,482 groups (198 chains: 182 of 2 pieces, 11 of 3, 4 of 4, 1 of 5).
  Events at a clip edge: 400 start on frame 0, 416 end on the last frame.
- `event_boxes`: 9,717,873 rows. DB 249 MB. Build 50 s.
- Tests: **163/163 pass**.

**Decisions**
- **Event = one annotated record within one clip of one camera**, as in the source files. The same real activity
  can appear on several cameras (never linked: no cross-camera linking) and, when cut at a clip boundary, as 2+
  pieces sharing a `group_id` (within one camera, consecutive clips; a link between annotation pieces, not an
  identity). Evaluation (2.x) should score against groups, so a cut event is neither counted twice nor missed.
- **Deviation from CLAUDE.md 1.3:** `boxes` is a separate table `event_boxes` (9.7M rows) instead of a column, and
  `events` has extra columns (frames, proxy seconds, source, flags, group_id) that later steps need.
- **RGB = Kitware only; IR = contrib only** (Gate 0.3). Contrib read from the `.json` (complete boxes), merged with
  box IoU >= 0.3 (`annotations.contrib_merge_min_box_iou`; time-only merging joined two different vehicles).
  Contrib object types ('Person' everywhere) not used.
- **Boundary links** need same type + box IoU >= 0.5 at the join (`annotations.boundary_merge_min_box_iou`), one-to-one.
- **`src_status` kept, not filtered** (meaning undocumented). Boxes stored as annotated (not clipped to the frame).
- **IR clips without events are 'none', never 'empty'**: contrib labels are incomplete, and Kitware has no real IR labels.
- **Spot check stratified**: one vehicle start/stop and one other activity from each Kitware level (they come from
  different annotation pipelines, incl. the two line styles) + one IR event; seed 13.
- New config section `annotations:` (contrib sources, two IoU thresholds), values justified in comments.
- Not an experiment: no row in `experiments.csv`.

**Open issues**
- 👤 **5-event check not done yet** (below). If an event is not where the table says, step 1.3 is not done.
- VERIFY: meaning of `src_status` (253 records are not_good/unaudited) and of `empty_37` (checked-and-absent vs
  not annotated: matters for the 28 empty RGB clips, 14 of them on G330, when counting false positives).
- VERIFY: contrib frame base (0- vs 1-based; 1 frame); CMU-v2 box timing (seen 20-40 frames ahead on 1 track);
  Kitware boxes possibly 0-1 frame ahead (3 tracks). Event time spans are unaffected.
- Carried over: training-level label completeness (9,340 of the RGB events are training-level).
- Step 2.1 must keep a group's pieces in one split (block boundaries on clip boundaries, or assign by group).
- Step 6.1: vehicle start/stop pieces at a clip edge are truncated (start/end reflect the cut, not the motion).
- `event_id` is a running number: stable for the same annotations + config, but rebuilt ids can change.
  Frozen queries (2.2) should also store (clip, src_ids) so ground truth can be re-resolved.

**Verify by hand — 👤 the 5-event check (most important manual check of the week)**
- From the repo root, print the events and commands: `uv run python -m scripts.build_events --spot-check`
- For each event: run its `ffplay` line, wait for the yellow clock to reach the in-file start time, watch until
  the end time. Does the named activity happen there, at the red box? Note "ok" or what you saw instead.
- Event 3 is IR with a third-party label: judge the time span, not the box.
- Optional: open `results/reports/events_1.3.md`; read `docs/meva_notes.md` "Annotation conventions".
