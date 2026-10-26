# PROGRESS — Rewind

Deadline: Fri 30 Oct 2026 · Current phase: **6 done → 7 next** · Phases 0–4 done (Gate 4.4: pad/motion/raw) · Gate 5.1: LanguageBind ADOPTED (7/3/3) · 5.2 action index DONE (300 clips, 44,682 windows, verified) · Gate 6.2: event branch DROPPED (rule, part 1)

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
- 👤 **Done by Ashutosh, 25 Oct (after Phase 1): passed.** DB rows match the filenames for both clips (camera, date,
  start time, modality; G330 300.33 s vs 299 s in the name = +1.33 s, G474 300.27 s vs 300 s = +0.27 s, both within
  `ingest.max_name_duration_diff_s` 2.0). The G330 clip's Kitware files (kitware-meva-training: activities, types,
  geom) carry exactly the clip's name. The original G330 file plays and shows the expected indoor scene.

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

## Step 1.3 — Annotated events table (Sun 25 Oct) — 👤 5-event check passed

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
- 👤 5-event check: passed 5/5 (below). Event 3 (IR, UMD-IBM label) showed the stop ~1.5-2 s after the labelled
  start: one more data point for the contrib timing VERIFY item.
- VERIFY: meaning of `src_status` (253 records are not_good/unaudited) and of `empty_37` (checked-and-absent vs
  not annotated: matters for the 28 empty RGB clips, 14 of them on G330, when counting false positives).
- VERIFY: contrib frame base (0- vs 1-based; 1 frame); CMU-v2 box timing (seen 20-40 frames ahead on 1 track);
  Kitware boxes possibly 0-1 frame ahead (3 tracks). Event time spans are unaffected.
- Carried over: training-level label completeness (9,340 of the RGB events are training-level).
- Step 2.1 must keep a group's pieces in one split (block boundaries on clip boundaries, or assign by group).
- Step 6.1: vehicle start/stop pieces at a clip edge are truncated (start/end reflect the cut, not the motion).
- `event_id` is a running number: stable for the same annotations + config, but rebuilt ids can change.
  Frozen queries (2.2) should also store (clip, src_ids) so ground truth can be re-resolved.

**👤 5-event check — done by Ashutosh, 25 Oct, all 5 passed** (seed 13; ffplay with the label box drawn at the
event's middle frame, fixed in place, and a playback clock). Observations, as reported:
1. vehicle_stops, G436, kitware-eval, 2:11.9-2:13.7: "a car stopped for a few seconds where the red box was, then
   left; the red box was there even after the car left" (expected: the box is drawn once, not tracked).
2. vehicle_stops, G424, kitware-train, 4:46.8-4:47.8: passed, like 1.
3. vehicle_stops, G474 IR, UMD-IBM, 4:38.4-4:40.8: passed, "car stops 40 to 42" (≈1.5-2 s after the labelled start).
4. person_carries_heavy_object, G330, kitware-eval, 3:12.5-3:28.3: passed; "the boxes were in a fixed position and
   objects were being carried at the back too".
5. person_talks_to_person, G424, kitware-train, 4:01.8-4:49.4: passed; "they passed exactly through those 2 red boxes".
Conclusion: label frame numbers = proxy playback time (frame / 30) for both Kitware levels and the IR contrib labels.

**How the check was run (to repeat it)**
- From the repo root, print the events and commands: `uv run python -m scripts.build_events --spot-check`
- For each event: run its `ffplay` line, wait for the yellow clock to reach the in-file start time, watch until
  the end time. Does the named activity happen there, at the red box? Note "ok" or what you saw instead.
- Event 3 is IR with a third-party label: judge the time span, not the box.
- Optional: open `results/reports/events_1.3.md`; read `docs/meva_notes.md` "Annotation conventions".

---

## Step 1.4 — Throughput (Sun 25 Oct)

**Done**
- `scripts/throughput.py`: decodes 1 h of RGB (12 clips over the 5 RGB cameras, seed 14) and 1 h of IR video
  sequentially from frame 0 (every clip must give exactly its manifest frame count) in three modes: decode only;
  keep 1 fps at full resolution (appearance branch); keep 5 fps scaled down to at most 640 px wide (motion pass;
  never scaled up). Compares PyAV slice vs frame threading on one clip (same frames by hash). CLIP: `ViT-B-16`
  fp16 (open_clip mixed precision, untrained weights: same compute, tag still TBD) on 1,000 real 1 fps frames,
  preprocessing timed on the CPU, image encoder timed per batch size 1-256 with peak VRAM. Projects every stage to
  the whole selection; reads all earlier runs from `results/experiments.csv` for a run-to-run table and a
  slowest-run projection. Logs one experiments.csv row per run (split `none`). Report:
  `results/reports/throughput.md` (regenerated by each run).
- `tests/test_throughput.py`: 9 tests. Sampling step (30/1 → 30, 30/5 → 6, 30/7 refused); frame 0 and every k-th
  kept, under both threading modes, and the kept array is that frame (gray level = 12 × index); never scales up;
  wrong frame count refused; seeded round-robin clip choice; minute rounding; run history incl. old metric names.
- 4 runs (machine otherwise idle; load 0.24-0.28 at start of runs 3/4):
  1. git c388f15: **IR 5 fps mode invalid**: IR frames were scaled up from 352 to 640 px (found reading the
     numbers; fixed in 3f50859). Its other numbers are valid.
  2. git 3f50859: RGB 5 fps 20× realtime, IR modes ~40 % slower than all other runs; RGB decode-only unchanged.
  3. git 3f50859: matches run 1. 4. git d3cbc82 (final script, report from this run): matches runs 1 and 3.
- Diagnosing run 2 (cause **unverified**): one clip, 5 fps mode: 8.8 s unpinned (twice), 7.0 s pinned to the
  performance cores (CPU 0-11), 13.9 s pinned to the efficiency cores (12-15). Runs 3 and 4 logged the CPU package
  at up to 90 °C / 88 °C. Run 2 started minutes after run 1 had ended; no temperature log for it. Likely core
  placement and/or thermal state of the laptop.
- Bug found while preparing the run: the first version held 1,000 full 1080p frames in RAM (6.2 GB of ~9 GB free)
  before preprocessing; stopped before it got there, now preprocessed as decoded (~0.6 GB).
- PROMPT R (2 reviewers + 2 skeptics). Confirmed and fixed: the report contained fixed text the code never checked
  ("one run each", "load average checked", "2-3 GB per hour", "decoding uses all cores" even if slice won, CPU
  model missing); run-to-run spread and the invalid run-1 value not shown; old and new IR 5 fps under one metric
  name; crop preprocessing (possibly the largest cost) missing from the projections; "parallelises over workers"
  unmeasured; VRAM conclusion not stated; page-cache caveat probably wrong; times floored instead of rounded;
  decode test checked indices but not frame content. Now every claim in the report is computed by the script.

**Key numbers** (run 4; runs 1 and 3 agree within a few %)
- Decoding, × realtime: RGB decode only **56×**, keep 1 fps **51×**, keep 5 fps at 640 px **34×** (run 2: 20×);
  IR 1,121× / 745× / 267×. Frame threading 5.1 s vs slice 11.5 s per 5-min 1080p clip, identical frames.
- CLIP ViT-B/16 fp16: **~300 img/s, flat from batch 16 to 256** (best: 16 in all 4 runs, 298-303 img/s, but the
  plateau differences are within run-to-run noise). Batch 256 uses 2,325 of 5,857 MiB: **VRAM is not the limit**.
  Batch 1: 153-212 img/s.
- CPU preprocessing (open_clip default transform, 1 process): **47-48 img/s** for 1080p frames: the slowest stage.
- Projections for the whole selection (25.01 h), stages timed separately:
  decode at 1 fps 25 min (slowest run 27); decode at 5 fps **38 min (slowest run 1 h 03)**; CLIP 90,161 full
  frames 5 min; with 4 crops each (upper bound, 450,805 images) 25 min; preprocessing of full frames 32 min in one
  process; crop preprocessing not measured. RGB video is 1.9 GB per hour.
- Tests: **172/172 pass**.

**Decisions**
- Decode with PyAV frame threading (`thread_type = "AUTO"`): 2.2× faster, identical frames.
- CLIP batch size: any value 16-256 gives the same speed; choose by memory headroom at step 4.2 (e.g. 64). Not
  written to config (step 4.2's decision).
- Measured both 1 h RGB and 1 h IR (CLAUDE.md asks for "one hour"): they differ ~20× in decode cost, so the
  projection needs both.
- Untrained CLIP weights for timing (`appearance.pretrained` is still TBD; compute is identical).

**Open issues**
- **Step 4.2: the default CLIP preprocessing centre-crops 16:9 frames to a square, cutting ~22 % on each side.**
  For CCTV that can drop the event; decide pad vs tile vs crop at 4.2.
- Step 4.2: preprocessing (~47 img/s per process) will dominate unless spread over workers or done more cheaply
  (e.g. scaling during decode). Not measured.
- Step 3.1: the 5 fps scaling step is sensitive to core type (7.0 vs 13.9 s per clip); plan on the slowest-run
  figure (~1 h) for the decode part of the motion pass.
- `results/experiments.csv` row 1 (c388f15): its IR 5 fps value is invalid. Ashutosh left the call to me; its notes
  now end with "INVALID: decode_ir_keep_5_fps_x_realtime and proj_decode_keep_5_fps_h (...)". Only that notes field
  was edited (numbers untouched, checked by diff); the report's history still finds all 4 runs.
- Not measured: detector (Gate 3.3), MOG2, crop extraction/preprocessing, action branch, an end-to-end run.

**Verify by hand**
- Read `results/reports/throughput.md` (2 min): the run-to-run table and the projection table.

---

## Step 2.1 — Splits (Sun 25 Oct)

**Done**
- `rewind/eval/splits.py`: sessions → ~30-min blocks → constrained random assignment → boundary-event flags; guarded
  accessors `split_video_ids` / `split_event_ids` (test refused unless `allow_test=True`, step 8.2 only).
- `scripts/make_splits.py`: writes DB tables `blocks`, `video_blocks`, `event_splits`, the committed split
  `eval/splits/blocks.csv` + `clips.csv`, and `results/reports/splits_2.1.md`. Refuses to rerun without `--rebuild`,
  and refuses `--rebuild` once any `eval/queries/worksheet_*.csv` or `HASHES` exists.
- Config section `splits:` (seed 21, block target 30 min, 15/15/70 ± 5 %, minimum counts, boundary pad 2 s).
- `tests/test_splits.py`: 37 tests (28 functions, some parametrised): hand-computed toy cases, an end-to-end run of the
  script on a toy DB (boundary flags, rebuild guards, determinism), and checks on the committed split: no block in two
  splits, no wall-clock overlap between blocks of different sessions and splits, every clip in one block, DB = CSV,
  stored boundary flags = recomputed. Suite: 209 pass.

**Method**
- 9 sessions of continuous *selected* footage (20-50 min) → 10 blocks (only the 50-min 03-11 morning session is cut,
  at 11:40; both halves ended up in test). Blocks of different splits are ≥ 299.8 s apart (03-07 tuning ends
  17:15:00.23, test starts 17:20:00; next 593 s on 03-15) → 0 boundary-flagged events with the 2 s pad.
- All 55,980 assignments (3 splits non-empty) are checked against the requirements; seed 21 picks one uniformly.

**PROMPT R (review)** — 4 review lenses + a skeptic per finding (mutation testing, independent recompute from the DB,
timestamps/test-split access, silent failures/config). No finding changed the split; all were LOW. Fixed:
- a crash in the report when a split has no activities (found by the new end-to-end test);
- 16 surviving mutations → new tests: midpoint rule, each minimum on the test split, inclusive edges, running
  session end, the script end to end; `target_fraction` now must name all 3 splits and sum to 1;
- the refusal guard now ignores empty tables left by a failed run; `block_id` uniqueness asserted;
- wrong claims: "sessions are recording breaks" (docstring), "≥ 5 min apart" (here), "3^12 is seconds" (config).
- After the fixes the split was rebuilt: `blocks.csv` / `clips.csv` byte-identical to draw 2.

**Draw history (honest record)**
- **Draw 1** (requirements without the camera rule; 38 valid): tuning = 03-11 13:50 block; dev = 03-07 16:50 +
  03-15 14:50; test = the other 7. Independent review found **dev had 23 G330 (indoor gym) activities vs 4,669 in
  test**: G330 is busy only on 03-11 (930-1,475 activities per block) and almost idle on other dates (2-19).
  The "≥ 25 activity types" check never bound (every block has 27-32 types), so it gave no protection.
- **Ashutosh chose** (25 Oct) to add `min_activities_per_rgb_camera: 20` for dev and test (every RGB camera's scene
  in dev and test), after seeing draw 1 but **before any query, worksheet or search existed**, and without seeing
  what the new rule would draw.
- **Draw 2 = the split** (23 valid, same seed 21): tuning = 03-07 16:50 + 03-15 14:50 (4.83 h, 19.3 %);
  dev = 03-11 13:50 (3.50 h, 14.0 %); test = the other 7 blocks (16.67 h, 66.7 %).

**Decisions**
- **Blocks follow sessions** (deviation from "split the window into 30-min blocks"): the selection is 9 separate
  sessions on 3 dates (Gate 0.3), not one window. Each session is cut into max(1, round(L / 30 min)) blocks on clip
  boundaries; each clip goes to the block holding its midpoint, so all cameras of a 5-min slot stay together.
- **Constrained random assignment:** every assignment is checked against fixed requirements (15/15 ± 5 % of
  camera-hours for tuning/dev; per split ≥ 25 types, ≥ 20 RGB vehicle starts and stops, ≥ 10 labelled IR activities;
  dev and test ≥ 20 activities on every RGB camera; Gate 0.3: tuning's evaluation-level share ≤ overall). Seed 21
  picks uniformly among the valid ones. Only label metadata is used, never search results. Binding in practice:
  the fraction band, the IR minimum, the Gate 0.3 rule and the camera rule.
- **Boundary events** (may be visible in another split: span ± 2 s overlaps another split's block, or a cut
  activity's pieces in two splits) are flagged in `event_splits.boundary`, not deleted. Currently 0.
- **Test-split guard:** `split_video_ids` / `split_event_ids` raise `PermissionError` for test unless
  `allow_test=True` (step 8.2 only). Later steps must read splits through them, not raw SQL.
- **The committed split is `eval/splits/*.csv`** (data/ is not in git); a test checks DB = CSV.
- Not an experiment: no row in `experiments.csv`.
- Reviews from now on (Ashutosh, 25 Oct): inline for routine steps; the multi-agent review only at Gate 2.2, 8.1 and
  8.2 (the 2.1 review took ~42 min).

**Key numbers (draw 2)** — activities (cut activities counted once): tuning 3,550 / dev 1,811 / test 11,121;
RGB vehicle_starts/stops 350/351 · 40/42 · 590/607; IR activities 33 · 44 · 98 (IR vehicle start/stop 8 · 8 · 10);
G330 activities 23 · 1,296 · 4,669; types 32 · 31 · 35.

**Open issues (draw 2)**
- **Dev is the light-traffic, gym-heavy regime**: G330 72 % of dev activities; G340 only 21 activities; 82 RGB
  vehicle start/stop events (test 1,197, mostly dense 03-07/03-15 traffic). Gate 6.2 dev precision will not see
  traffic queues; report 6.1/8.2 event-rule results per camera.
- **Dev labels are mostly training-level** (157 evaluation-level vs 1,610 training-level activities). The Gate 0.3
  rule was implemented only as "tuning not more evaluation-level than overall" (tuning 36 % ≤ 48 %).
- Tuning has dense traffic (good for 6.1 thresholds) but almost no indoor activity (G330 23): the 4.1 dedup sweep
  and 7.4 fusion weights are tuned on outdoor scenes only.
- Query counts per split will be small (~15/15/70 of ~60): set per-split quotas by qtype/IR at 2.2 (critique).
- `min_activity_types` never binds; kept as a check only.
- **Session gaps are not recording breaks.** On 03-07 MEVA has (unannotated, unselected) clips for 17:10-17:20, so the
  16:50 (tuning) and 17:20 (test) sessions are one continuous recording ~5 min apart; a scripted scenario could carry
  across. Separation there is only the time gap.
- **Step 2.3 / 3.2:** the witness ±10 min window (600 s) is longer than the smallest gap between splits (299.8 s):
  clip every time window to the query split's own footage (`split_video_ids`), or it reaches into test footage.

**Rules for step 2.2**
- Sample worksheet events through `split_event_ids` (boundary-flagged events left out); test-split worksheets are
  the one place before 8.2 that may pass `allow_test=True`, to write queries, never to search.
- Make sure dev gets enough `action` and `vehicle_event` queries for Gates 5.1 / 6.2 (dev has only 82 RGB vehicle
  start/stop events, all light traffic) and test enough IR queries for the RGB vs IR breakdown.
- Store `(clip, src_ids)` with each query, not just `event_id` (see 1.3).

**Verify by hand**
- Open `results/reports/splits_2.1.md`: the per-split table should match the key numbers above; the per-block table
  shows which blocks are in dev (03-11 13:50) and tuning (03-07 16:50, 03-15 14:50).
- Read `rewind/eval/splits.py` module docstring and `meets_requirements`: be able to explain why blocks follow
  sessions, why the assignment is random among valid ones, and why draw 1 was replaced (camera rule).
- `uv run pytest -q tests/test_splits.py` → 37 passed.

---

## Step 2.2 — Queries, GATE 2.2 (Sun 25 Oct)

**Queries drafted by Claude Code from labels and keyframes in witness style, reviewed by the user, frozen before any
search code existed.** (Deviation from CLAUDE.md 1.10 / step 2.2, which have the user write them; Ashutosh's decision,
25 Oct.)

**Done**
- `rewind/eval/queries.py` + `scripts/make_worksheets.py`: seeded plan (config `queries:`, seed 22) fixing, before
  any keyframe was looked at, each row's event, kind, qtype, specificity and modality; worksheets
  `eval/queries/worksheet_<split>.csv` and one keyframe sheet per row (middle frame with label boxes; first frame,
  last frame, zoom), decoded sequentially from the proxies. Sheets are gitignored (regenerable, 22 MB).
- Drafting (multi-agent): 4 writers (labels + sheets only, fixed kind/qtype/specificity, witness rules: everyday
  words, no label/technical/caption wording, only details visible in the sheet, no colours on IR), then 1 critic who
  revised 17 of 60 (6 category queries with one keyframe's setting, 4 wrong specificity, 5 details not visible —
  phone, typing, age, direction, "coffee table" — 2 repeated openings). Then 1 count fix (tes-s11 "a few" → "two")
  and casing/full stop made uniform (no wording change).
- **1 replacement:** tes-s20 (IR vehicle_stops, event 9538): label box on a parked car, no vehicle stops in the span
  (writer and critic independently). New event drawn by a seeded rule (same type and modality, unused):
  9532 (same clip, 230.3-234.2 s). Logged in `eval/queries/replacements.csv`; query written from its sheet.
- `scripts/review_queries.py`: local HTML with 15 seeded-random queries next to their sheets → **approved by
  Ashutosh** (25 Oct).
- `scripts/freeze_queries.py`: validates every row (fields, wording lint, specific events exist / in split / not
  boundary-flagged / match the DB; category types have ≥ 10 eligible RGB activities), writes
  `eval/queries/queries_<split>.csv` and `eval/queries/HASHES` (sha256sum format: 3 query files, 3 worksheets,
  replacements.csv). Refuses to run twice; `make_splits --rebuild` and `make_worksheets` refuse too.
- `load_queries(queries_dir, split, allow_test=False)` checks all hashes on every load and refuses test (8.2 only).
- Tests: `tests/test_queries.py` (plan, 5) — caught a bug: category types could qualify by counting IR events
  while examples are RGB-only; fixed, the real plan was unchanged (re-derived, identical).
  `tests/test_frozen_queries.py` (7): hashes match, no unhashed CSV, an edit is detected, frozen shape, test guard,
  and a scan of `rewind/` + `scripts/` that fails if any file outside an allow-list (the 2.1/2.2 tools) passes
  `allow_test`, opens query/worksheet files directly, or queries `event_splits` / `video_blocks` / `blocks`
  (step 8.2 must add its script to that list on purpose). Suite: 228 pass.

**PROMPT R** (after the freeze, before commit; inline code review + 1 independent agent re-checking the frozen
ground truth without our code). Frozen content: **no problem found** — all 7 hashes; all 42 specific events
(exist, right split, not boundary-flagged, single piece, ≥ 1 s, fields = DB, qtype ↔ type, IR = G474 contrib);
all 18 category types (≥ 38 eligible RGB activities); all 60 sheets (top panel = frame (start+end)//2 by image
matching, every label box at the right place); worksheets = query files; the replacement reason checked by eye.
Code fixes (no frozen file touched):
- Keyframe sheets were gitignored as "regenerable" but `make_worksheets` refuses after the freeze →
  `--sheets-only` redraws them from the frozen worksheets; all 60 verified **byte-identical** to the reviewed ones.
- Undocumented: actor boxes are drawn at the nearest annotated frame — 10 boxes in 7 rows came from 7-591 frames
  away (secondary actors or first/last panel; worst tes-s09's dropped-off person, 203 frames off on the middle
  panel). The 7 queries were re-read: each describes the right subject. Now documented; drawing unchanged.
- Wording lint banned "box" (a witness may see a box) → allowed; lint tests added. Category rows now also checked
  for qtype ↔ type at freezing. Zoom fallback labelled. Test-split access lint (above).
- Note: `replacements.csv` reason says "proxy frames 238-256 s": it means proxy *times* 238-256 s (frames
  ~7140-7680); the labelled span was 249.7-252.8 s. Frozen, so left as is.

**Key numbers** — 60 queries: 18 category + 42 specific; tuning 9 / dev 17 / test 34.

| split | category (vehicle_event / action) | specific vehicle_event / action / appearance | of which IR |
|---|---|---|---|
| tuning | 1 / 2 | 2 / 2 / 2 | 1 |
| dev | 2 / 3 | 4 / 4 / 4 | 2 |
| test | 2 / 8 | 6 / 9 / 9 | 4 |

Specificity in thirds per split (tuning 3/3/3, dev 6/6/5, test 12/11/11 vague/moderate/detailed).
HASHES: queries_tuning 6c70f93c…, queries_dev caea1717…, queries_test b033a97f….

**Decisions**
- Category ground truth = every activity of the type **in the query's split** (event columns empty in the frozen
  file); specific ground truth = one activity (event_id, plus clip + src_ids to re-resolve if ids are rebuilt).
- Specificity is part of each query (`specificity` column): breakdown by vague / moderate / detailed is possible.
- Small/distant subjects were not replaced: only a missing/wrong label justifies a replacement.

**Open issues (for 2.3 and later)**
- Risk of bias: the queries come from the same assistant that will build the search (mitigated: seeded plan, label +
  keyframe only, critic, user review, frozen before any search code). State this in the README / report.
- dev-s05 (IR, turns left) and dev-s09 (IR, starts) describe the same moment; tes-s05 and tes-s13 are two people
  entering dark cars at the same entrance seconds apart. Scored as written.
- Vague specific queries match many events but have one correct answer; seconds-watched will punish them (intended).
- Same activity on overlapping cameras: still to decide at 2.3 whether a same-type event on another camera at the same
  time counts as a hit for a specific query (MEVA has no cross-camera links).
- IR (G474) ground truth is third-party and incomplete: recall-type metrics only (Gate 0.3). Event 9538 (IR
  vehicle_stops) looks mislabelled; it stays in the events table.

**Verify by hand**
- `cd eval/queries && sha256sum -c HASHES` → 7 × OK.
- Optional: open `eval/queries/keyframes/tes-s09.jpg` — the cyan box (dropped-off person) is from ~7 s earlier.
- Skim `eval/queries/queries_dev.csv` (17 rows): these drive Gates 4.4, 5.1, 6.2.

---

## Step 2.3 — Metrics + simple baselines (Sun 25 Oct)

**Done**
- `rewind/eval/metrics.py` (pure functions): Recall@k, seconds watched with the miss rule (specific); mAP@k and
  instance Recall@k with don't-care spans (category); one-to-one temporal-overlap precision/recall for detected vs
  annotated events (used from 6.1); seeded percentile bootstrap CI.
- `rewind/eval/baselines.py`: 60 s segments (each clip cut into round(d / 60) equal pieces), sequential, random
  (seeded), witness ±10 min.
- `scripts/run_baselines.py`: tuning + dev only, through the guarded accessors (test refused); 6 rows in
  `results/experiments.csv` (3 baselines × 2 splits); `results/reports/baselines_2.3.md` (summary with CIs,
  breakdowns by qtype / modality / specificity, per-query tables). `--no-log` re-renders the report only.
- `tests/test_metrics.py`: 17 tests, every expected value hand-computed in a comment (incl. category ground truth on
  an in-memory DB). Suite: 245 pass.

**Key numbers** (median seconds watched before the event starts, specific queries; Recall@10; category mAP@10)
| split | sequential | random | witness (perfect) |
|---|---|---|---|
| tuning (6 specific, 3 category) | 8,091 s (135 min), R@10 0.056, mAP@10 0.35 | 9,063 s, R@10 0.027, mAP@10 0.29 | 3,126 s (52 min), R@10 0.056 |
| dev (12 specific, 5 category) | 5,932 s [5,476–6,561] (99 min), R@10 0.083, mAP@10 0.10 | 6,413 s, R@10 0.050, mAP@10 0.09 | 3,237 s [2,325–3,754] (54 min), R@10 0.083 |

**Decisions (definitions, in the metrics docstring)**
- **Hit = overlap on the same camera** (no cross-camera credit: MEVA has no cross-camera links). Settles the 2.2 open
  question in the strict direction.
- **Seconds watched** = full clips before the first hit + time into the hit clip until the event starts. **Miss rule**
  read literally: no hit in the list → the list's duration + a full review of all footage in the split. Baselines rank
  all footage, so they never miss; the rule bites only on truncated lists (Rewind).
- **Camera order**: sequential and witness are averaged exactly over all 720 camera orders (no arbitrary order).
  Random: mean of 100 seeded shuffles per query.
- **Witness time = the event's true middle** (a perfect witness: the baseline's best case); specific queries only.
- **Category**: ground truth = Kitware RGB activities of the type in the split (cut pieces joined); boundary-flagged =
  don't care; IR segments left out of category ranking (Gate 0.3). AP@k: a clip is relevant only if it finds a not
  yet found activity; denominator min(k, #activities).
- **Event matching**: the annotated span is widened by `metrics.event_tolerance_s` (2 s), once.
- Config sections `metrics:` and `baselines:` added.

**PROMPT R** (inline). Found and fixed:
- **Bug (found by a sanity check before logging):** G330 clips are 300.33 s, so fixed 60 s cutting left a 0.33 s
  sliver per clip, each costing one rank (dev-s04 ranked 10 instead of 9). Now equal pieces; test added. The logged
  rows were computed after the fix.
- **Bug:** `match_events` padded both spans in the overlap test (a 2 s tolerance allowed a 4 s gap) and measured the
  overlap with one padded span. Now only the label is widened, once; regression test. Not used by any logged result.
- Untested runner logic (`category_truth`, averaging over rankings) → toy tests.
- Report cosmetics (`nan`, `12.0`) fixed via `--no-log` re-render; `experiments.csv` byte-unchanged and the report
  byte-identical after the review fixes (logged numbers still valid).

**Open issues**
- Sequential beats random on R@k only because 5 of 12 dev events fall in the first 5 min of the dev block (scripted
  footage), not because sequential review is better.
- Baseline category mAP is far from 0 (busy roads: many 60 s segments contain a vehicle event); compare Rewind with
  these baselines, not with 0. Tuning mAP (0.35) is high because of dense traffic (351 vehicle_stops activities).
- Small samples: 6 tuning and 12 dev specific queries → wide CIs; breakdown cells have n = 2-5.
- Motion-ranked baseline: step 3.2.

**Health check after Phase 2** (25 Oct, read-only): 245 tests pass; 300 originals read-only, 3 random re-hashed OK;
300 proxies with equal frame counts; every event has a split, every video a block; 7/7 query hashes OK; baseline
report reproduces byte-identically; test split refused by both loaders; 83 GB free. Found and fixed (commit
"phase2.3: fix"):
- `run_baselines`, `build_events`, `review_queries` crashed at their final print when given an output path outside
  the repo (`Path.relative_to`), after outputs/log rows were written → `rewind/paths.display_path`; test added.
  `run_baselines` now writes its report before logging.
- The 6 step-2.3 rows carry git `979c37e-dirty` (logged before the commit; metrics.py changed afterwards, the
  re-run is byte-identical, so they reproduce from `9579b17`). `run_baselines` now refuses to log with uncommitted
  code (`--allow-dirty` to override): future rows name the exact commit. Suite: 246 pass.

**Verify by hand**
- Hand check of one toy case (witness ordering, expected 190 s): **done by Claude Code at Ashutosh's request**, not by
  Ashutosh — an independent hand check is still open if wanted:
  `sed -n '/def test_witness_window_first_then_the_rest/,/^def /p' tests/test_metrics.py`
- `less results/reports/baselines_2.3.md` (dev summary as above).

---

## Open-issue pass after Phase 2 (Sun 25 Oct) — 8 items

Not a plan step: Ashutosh asked to work through the open issues/VERIFY items one by one before Phase 3. No pipeline
code, config or data changed. The checks first ran as scratch scripts; they were then moved into `scripts/` (second
commit) and re-run from there, reproducing the numbers below (only an uncited G436 median moved 4.6 → 4.5 px by
rounding). Run from the repo root:
- item 2: `uv run python -m scripts.check_camera_views --out <folder>` (CSV, contact sheet, one image per flagged pair)
- item 4: `uv run python -m scripts.check_centre_crop --pretrained laion2b_s34b_b88k datacomp_xl_s13b_b90k`
  (and `--model ViT-B-16-quickgelu --pretrained openai`); verifies each tag's transform, then the tuning numbers
- item 6: `uv run python -m scripts.check_contrib_timing`
Items 3 and 5 were one-off inline queries / `curl` lookups (not scripted); their sources are named in each item.

**1. Independent hand check of the headline metric — done by Ashutosh.** New toy case (2 cameras × 300 s, 60 s
segments, GT on B 130-140 s): sequential 430 s, witness (t = 100 s, ±60 s) 310 s, miss case 690 s, short list 35 s.
All four hand answers equal `seconds_watched` output. Closes the 2.3 "verify by hand" item.

**2. Camera views across the 3 dates — VERIFY closed. Decision (Ashutosh): accept and document.**
- Method: proxy frame at 60 s of every clip (300); SIFT + RANSAC homography vs the previous clip of the same camera
  (and vs the camera's first clip); flagged pairs checked by eye.
- G330, G424, G436: fixed (median 0.1-0.2 px at 640 wide between consecutive clips; max 6.5 px).
- G474 (IR): fixed by eye (poles/path in place); SIFT finds too few matches across dates (thermal contrast changes).
- **G340: one view change on 03-07 between 11:10 and 16:50** (slight tilt, ~26 px at 640 wide ≈ 4 %); stable from
  16:50 through 03-11 and 03-15. The 4 morning clips are all in **test**.
- **G336: small temporary change on 03-07 17:00-17:10** (2 clips, 22-48 px corner displacement, looks like a
  slight zoom; back by 17:20; filenames switch from :06 to :00 there). Both clips are **tuning**.
- Consequence for 4.3: one mean per camera; 6/300 clips get slightly weaker background removal. Not per-view means.
- Disclosure: this check decoded one frame from every clip, including test clips. No labels, nothing tuned.

**3. `empty_37` and `src_status` — no official definition exists** (searched: repo READMEs, Kitware `status.csv`
(activity counts only) and `list-of-annotated-meva-clips.txt` fetched via `git show`, the 3 PDFs in `documents/`,
web search). Evidence-based reading:
- `empty_37` = "clip annotated, none of the 37 activities present": all 9 evaluation-level empty clips of ours are in
  Kitware's annotated-clip list (which counts `empty_37` as a row); the marker never co-occurs with a real activity in
  any of 1,146 files; never-annotated clips have no file at all; the eval README says all instances are annotated.
- Training-level (19 of our 28 empty RGB clips) has no official statement. 👤 Ashutosh watched the busiest one
  (dev, `2018-03-11.14-05-02.14-10-01.school.G424`): **nothing there**. Keep counting hits on empty clips as wrong
  (current metric behaviour); no change.
- `src_status`: meaning unknown. Evaluation-level events only have good/janitor/gladiator2; `not_good` and
  `unaudited` occur only in training-level labels (fits the "reduced audit" wording; inference). Tuning+dev: 122 of
  5,358 Kitware events (2.3 %). 2 of 12 dev specific queries target `unaudited` events (written from visible
  keyframes, so usable). **Decision (Ashutosh): option A — keep all events as ground truth**, meaning recorded as
  unknown.

**4. CLIP centre-crop — measured on tuning. Decision (Ashutosh): 2 overlapping square tiles as main, pad-to-square as
the ablation (at 4.4).**
- open_clip ViT-B-16 preprocessing verified: `Resize(224)` (short side) + `CenterCrop(224)`. Keeps the central 56 %
  of the width on RGB (1920×1080), 68 % on IR (352×240).
- Tuning, 3,586 events / 1.53 M actor box-frames, event "lost" = >90 % of box area outside the square:
  **28 % lost overall**, 13 % partly cut. G424 42 %, G436 40 %, G330 52 % (n = 23), G340 15 %, G336 9 %, G474 0 %.
  Vehicle start/stop 134/709 lost; door/structure activities ~50 % lost.
- Plan for 4.2: tiles x 0-1080 and 840-1920 (full res, 6 vectors/frame instead of 5); object crops padded to a
  square, never centre-cropped. New config key (e.g. `appearance.frame_mode: tiles2 | pad`) added at 4.2, with ask.
  If 4.4 shows no difference, take pad (simpler).

**5. `appearance.pretrained` candidates — three tags, decision at the start of 4.2 on tuning.**
| tag | model name | weights source (open_clip `hf_hub`) | revision | license (verified) |
|---|---|---|---|---|
| `laion2b_s34b_b88k` | `ViT-B-16` | `laion/CLIP-ViT-B-16-laion2B-s34B-b88K` (safetensors) | `7288da5a` | **MIT** (model card front matter) |
| `datacomp_xl_s13b_b90k` | `ViT-B-16` | `laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K` (.bin only; open_clip loads with `weights_only=True`) | `d110532e` | **MIT** (model card front matter) |
| `openai` | `ViT-B-16-quickgelu` | `timm/vit_base_patch16_clip_224.openai` (safetensors) | `977e3dd0` | **Apache-2.0** on the timm copy (card front matter); original release `openai/CLIP` is **MIT** (its LICENSE file) |
- Licenses were checked before downloading. Ruled out on license: `dfn2b` (apple-amlr; terms not read), `metaclip_*`
  (CC-BY-NC-4.0).
- Published benchmarks (open_clip `openclip_retrieval_results.csv` / `openclip_results.csv`), text→image retrieval
  COCO R@1 / Flickr R@1: laion2b 0.423 / 0.698, datacomp_xl 0.402 / 0.676, openai 0.331 / 0.622. ObjectNet:
  datacomp_xl 0.635, laion2b 0.560, openai 0.553. Used only to choose candidates; web benchmarks don't decide.
- Downloads approved by Ashutosh (~1.2 GB); disk 83 → 82 GB free. **GPU check PASS, exit 0, all three**, re-run with
  `HF_HUB_OFFLINE=1` (all cached). laion2b and datacomp_xl ran on the unchanged project config (`ViT-B-16`) — the
  project config's first PASS. openai ran with a scratch config set to `ViT-B-16-quickgelu`. All: (3, 512) fp16,
  366 MiB peak allocated. Colour test, matching vs best non-matching: laion2b 0.291-0.325 vs ≤0.235; datacomp_xl
  0.306-0.317 vs ≤0.211.
- Decision method (agreed): at the start of 4.2, on **tuning**, templated prompts from activity names (e.g. "a person
  opens a vehicle door") → mAP over tuning events per tag; the 9 tuning queries as a secondary check. Keeps dev free
  for the 4.4 ablation. `config.yaml` unchanged until then (`pretrained: TBD`).
- **⚠ All three model cards (identical text) state:** "**Any** deployed use case of the model - whether commercial or
  not - is currently out of scope", image search in a constrained environment is "not recommended unless there is
  thorough in-domain testing", and "use cases which would fall under the domain of surveillance and facial
  recognition are always out-of-scope regardless of performance". This is an intended-use statement, not a license
  term, and it applies to every CLIP variant in the plan. Rewind is a non-deployed research prototype on scripted
  footage, with no face recognition and a held-out in-domain evaluation — but it is CCTV search. Must be stated
  plainly in the README limitations and the final report (10.1), and is worth having an answer for in front of a
  judge.

**6. Contrib (IR) label frame base and timing — VERIFY closed. Decision (Ashutosh): no code change; document.**
- **Convention: 0-based frames, inclusive end** (same as Kitware KPF). NIST's JSON spec (p9) says frames begin at 1
  and a track ends with an *empty* box at the first absent frame; neither source follows it: in all 848 local .json
  records (CMU-v2 510, UMD-IBM 338) the "off" frame carries a real box and there are 0 empty terminators.
  **Proven 0-based for CMU-v2** (14 records start on frame 0). **UMD-IBM** never uses frame 0, so its base is
  unprovable from data: **≤ 1 frame (33 ms) either way**; exact start offsets vs Kitware show no spike at 0 or +1
  (0: 64, +1: 58, −1: 41) — label noise ≫ 1 frame.
- **Timing vs Kitware**, measured on shared RGB clips **outside the selection** (no split touched; CMU 838 clips,
  UMD 471), contrib − Kitware, one-to-one match by type + temporal IoU > 0:

| | start median | start p10..p90 | end median | end p10..p90 | within 2 s (start / end) |
|---|---|---|---|---|---|
| UMD-IBM, all (n = 3,516) | +0.63 s | −0.4..+1.9 s | −0.40 s | −1.5..+0.6 s | 88 % / 91 % |
| CMU-v2, all (n = 5,442) | −0.13 s | −3.3..+1.2 s | +0.03 s | −1.3..+2.8 s | 78 % / 81 % |
| UMD-IBM, vehicle start/stop (n = 137) | +0.73 s | −0.5..+2.3 s | −0.27 s | −1.9..+1.2 s | 87 % / 85 % |
| CMU-v2, vehicle start/stop (n = 278) | −1.07 s | −5.7..+1.0 s | +0.68 s | −1.2..+5.1 s | 59 % / 68 % |

  UMD spans are slightly tighter (start later, end earlier; fits the 1.3 eye check); CMU vehicle spans are wider and
  noisier (start ~1 s early, tail to −5.7 s). With tIoU ≥ 0.3 the picture is the same, a bit narrower. Contrib labels
  match only 18 % (CMU) / 28 % (UMD) of Kitware activities (incompleteness, as at 0.3).
- **The 2 s label widening is a config value:** `metrics.event_tolerance_s: 2.0` (`configs/config.yaml`).
  `match_events` / `precision_recall` take `tolerance_s` as a required argument (no default); the only literal 2.0s
  are toy inputs in tests. No caller exists yet — **step 6.1/8.2 must pass the config value.**
- **IR caveat for 8.2 (pre-committed):** IR ground truth = third-party labels; incomplete (recall-type metrics only)
  and offset from Kitware conventions by ~0.5-1 s (CMU vehicle events up to ~6 s). Hits use clip overlap with ±5 s
  context, so specific-query hits are rarely affected; seconds-watched shifts by ~1 s. State this next to every IR
  number. **IR event-rule recall in 8.2 is reported per label source (CMU-v2 vs UMD-IBM)**, plus merged.

**7. Dev is the light-traffic regime — cannot be fixed; reporting rules pre-committed.**
- Facts (2.1): dev = 03-11 13:50 block; G330 (indoor gym) 72 % of dev activities; G340 21 activities; RGB vehicle
  start/stop 40/42 in dev vs 590/607 in test (dense 03-07/03-15 traffic); dev labels mostly training-level. Tuning has
  dense traffic but almost no indoor activity (G330 23).
- Re-splitting is not allowed: queries are frozen (Gate 2.2) and `make_splits --rebuild` refuses once HASHES exist.
- Pre-committed:
  - 6.1 (tuning) and 8.2 (test): event-rule precision/recall **per camera**, plus overall; IR recall per label source.
  - Gate 6.2: the dev decision states its n (6 vehicle_event queries: 2 category + 4 specific) and that dev has
    82 RGB vehicle start/stop events and little queueing; the precision/recall half of the gate is on **tuning**
    (CLAUDE.md 1.7), where queues exist.
  - 4.1 dedup and 7.4 fusion weights are tuned on outdoor-heavy tuning: the 4.4/8.1 dev reports show G330 (indoor)
    separately, to show whether tuning transfers.
  - 8.2: per-camera breakdown of the headline metric as well (descriptive, small n).

**8. Small samples — cannot add queries after freezing; reporting rules pre-committed.**
- Query counts: tuning 9 (3 category, 6 specific), dev 17 (5, 12), test 34 (10, 24). Dev by qtype: action 7
  (3 category + 4 specific; Gate 5.1), vehicle_event 6 (Gate 6.2), appearance 4. IR specific: dev 2, test 4.
- Pre-committed:
  - Every headline and gate number with a bootstrap 95 % CI (`metrics.bootstrap_n`, `ci_level`), per-query table,
    median and worst case (as CLAUDE.md 1.10).
  - Breakdown cells with **n < 5 queries**: per-query values and counts only, labelled "descriptive, n = k"; no CI,
    no claims (e.g. RGB vs IR on test: 4 IR specific queries).
  - Gate comparisons (4.4, 5.1, 6.2, 8.1) are **paired per query**: report wins / ties / losses and a paired
    bootstrap CI of the difference. Needs a small addition to `rewind/eval/metrics.py` (paired bootstrap) at 4.4.
  - **CONFIRMED (Ashutosh, 25 Oct) — gate decision rule:** adopt the more complex variant only if its mean is better
    **and** it wins on more queries than it loses; otherwise keep the simpler one (CLAUDE.md 8.1: "if two are close,
    pick the simpler one"). The paired bootstrap CI of the difference is always reported but is **not** required
    to exclude 0 (with n = 6-7 at Gates 5.1/6.2 it almost never would).

**Process notes**
- Items 3 and 4 read split labels with raw SQL joins on `blocks.split` in scratch scripts, not through the guarded
  accessors in `rewind/eval/splits.py`. Event-level analyses were filtered to tuning/dev. Exception: item 3 counted
  clip-level label status over all clips incl. test (28 empty RGB clips: 9 evaluation-level, all in Kitware's
  annotated list; 19 training-level) — the same counts the 1.3 report already holds. No test events, boxes or
  queries were read; nothing was tuned.
- Item 3 read `queries_tuning.csv` / `queries_dev.csv` with pandas directly, bypassing `load_queries`' hash check
  (read-only; hashes unchanged — see below).
- Item 6 used only clips outside the selection; items 7-8 used numbers already recorded in 2.1/2.2.

**Verify by hand**
- Contact sheets: re-create with `scripts.check_camera_views --out <folder>` (`camera_views_sheet.jpg`; G340 and
  G336 changes are among the `flagged_*.jpg` before/after pairs).
- Read one model card's "Out-of-Scope Use" section yourself, e.g.
  https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K

---

## Standing policy: minimise manual work (Sun 25 Oct) — CLAUDE.md rule 16

Not a plan step. Ashutosh's instruction, added to CLAUDE.md 1.1 as **rule 16**. Rules 3 and 12 carry a note that rule 16
amends them. No pipeline code, config or data changed.

**Policy**
- 👤 eyeball checks are replaced by measurements wherever annotations allow. Claude may view images and plots itself,
  and reports say so ("viewed by Claude, n = …"). Ashutosh is asked only when no measurement is possible.
- Claude decides gates using the pre-agreed rules and records the evidence, the rule applied and the decision here.
  Ashutosh is asked only if the rule doesn't clearly decide, or the decision involves licenses, deletions or
  `data/raw/`.
- Long jobs: Claude launches them with `nohup`, logging to `results/logs/<job>.log` with a PID file
  `results/logs/<job>.pid`, and gives the log path. No polling loops; Claude checks the log when Ashutosh next
  writes or asks.
- Unchanged (rule 6): ask first before any download over 1 GB, new dependency, deletion, `data/raw/` access or
  out-of-step config change.

**What this means for the remaining gates**
- **Gate 3.3 (detector):** has a license side (Ultralytics AGPL-3.0 vs YOLOX Apache-2.0), so **Ashutosh decides**.
  Claude provides measured detection recall vs MEVA annotated actor boxes instead of the 50-frame eyeball check.
  VERIFY: the boxes (`geom.yml`) aren't fetched yet; check the download size first (ask if over 1 GB).
- **Gates 4.4, 5.1, 6.2, 8.1:** Claude decides using the confirmed paired rule (open-issue item 8). Adopt the more
  complex variant only if its mean is better **and** it wins on more queries than it loses; otherwise keep the
  simpler one. Gate 6.2 also has the Wed 28 Oct 13:00 hard stop and the precision/recall check on tuning.
- **Step 4.2 pretrained tag:** decided by the agreed tuning method (open-issue item 5). Claude decides and asks only
  on a near-tie or a license question. All three candidates' licenses are already verified as permissive.
- Step 8.3 failure causes: Claude assigns them where measurement or viewing supports it, and says which were viewed.
  Gate 8.2 test results stay final whatever they are.

**Housekeeping**
- `results/logs/` added to `.gitignore`: logs and PID files are run-time state and aren't committed. Key numbers
  from them go into this file and `results/reports/`.

---

## Phase 3 order (Sun 25 Oct) — reordered by Ashutosh

To decode the footage only once: **3.1 as a 10-minute test → 3.3 detector smoke test + Gate 3.3 → one combined
full pass (MOG2 + chosen detector on the same 5 fps frames = outputs of 3.1 and 3.4) launched in the background →
3.2 motion baseline after the run.** Runs under rule 16 (measured checks, Claude decides rule-clear gates).

---

## Step 3.1 — MOG2 motion pass, 10-minute test (Sun 25 Oct)

**Done**
- `rewind/video.py`: `sampled_frames` (sequential PyAV decode, AUTO threading, every k-th frame from 0 as BGR,
  scaled down to a max width, frame count checked against the manifest at the end), `keep_step`, `scaled_size`.
  `scripts/throughput.py` now imports `keep_step` from here (was a duplicate).
- `rewind/motion/mog2.py`: per sample: grayscale MOG2 mask (shadows = not foreground) → opening → activity =
  foreground fraction; motion boxes = connected components of the dilated mask with area ≥ `min_box_area_px`,
  scaled back to original pixels. Per-second activity = mean of the second's samples. Warm-up: the first `warmup_s`
  of samples train a fresh model per clip, then the clip runs from frame 0 (no first-frame burst). `MotionConfig`
  refuses missing/TBD keys.
- `configs/config.yaml` `motion`: OpenCV defaults (history 500 samples = 100 s, varThreshold 16, shadows detected),
  warm-up 5 s, open 3 px, merge-dilate 9 px, **`min_box_area_px: 100`** (was TBD; a smoke value at 640 px wide, not
  tuned). New section `motion_check` (seed 31, 20 random windows per event).
- `scripts/check_motion.py`: the measured replacement for the eyeball check (rule 16), tuning only. Writes
  `results/reports/motion_3.1_<camera>.md`, `results/figures/motion_3.1_<camera>.png`, one experiments.csv row.
- `tests/test_motion.py`: 14 tests (static scene → no activity/boxes incl. no warm-up burst; moving square found
  and boxed; scale-back to 1920x1080 hand-computed; small-blob filter; per-second mean incl. short last second;
  TBD refusal; sampling grid + frame-count check; AUC, box coverage, event seconds hand-computed).

**Key numbers** (10 min each, tuning, 2018-03-07 16:50, Kitware events; experiments.csv rows at git `d394f25`)

| camera | events | seconds inside ≥1 event | whole-frame activity AUC, event vs random window | local box-coverage AUC (all / vehicle / other) | median box coverage event vs random |
|---|---|---|---|---|---|
| G436 | 274 | 99 % | 0.552 | **0.869** / 0.932 / 0.823 | 0.60 vs 0.00 |
| G340 | 104 | 70 % | 0.553 (per second, covered vs not: 0.597) | **0.809** / 0.848 / 0.759 | 0.36 vs 0.00 |

- Local measure: annotated actor boxes at the 5 fps samples, fraction covered by the best motion box; null = the
  same boxes moved to 20 random non-overlapping times in the same clip ("same place, other time").
- Decode + MOG2: 30 s per 10 min of 1080p video (**20× realtime**, one process).
- Plots viewed by Claude (n = 2): activity peaks line up with vehicle passes (G340: bus at 0:45, 5:20, 6:10);
  person-only events late in the G340 window give little whole-frame activity.

**Interpretation / decisions**
- MOG2 finds motion where labelled actors are (local AUC 0.81-0.87). Numbers look right; no stop.
- The **whole-frame activity score barely separates events from random windows** (AUC 0.55) on these cameras,
  because labelled events cover 70-99 % of the time and vehicles dominate whole-frame motion. Consequence: the
  motion-ranked baseline (3.2) will be weak on busy scenes — that is the realistic picture of motion search, not a
  bug. For 4.1 dedup, use motion *inside regions* (as CLAUDE.md already specifies), not whole-frame activity.
- One MOG2 model per clip (no state across clips): resumable per clip; costs a 5 s warm-up per 5-min clip.

**Open issues**
- `min_box_area_px` not tuned; IR (352 px wide, not downscaled) uses the same pixel threshold.
- The warm-up trains on the first 5 s including any moving object; objects in the first seconds may be partly
  absorbed (slight under-detection there).

---

## Step 3.3 — Detector smoke test, GATE 3.3 (Sun 25 Oct)

**Done**
- **Install (both within the timebox).**
  - Both detectors went into throwaway venvs in the session scratchpad. They reuse the main `.venv`'s torch through
    a `.pth` path, because the uv cache had been cleaned and a fresh torch would have been a >1 GB download.
  - Ultralytics 8.4.188 installed with `--no-deps` plus its small non-torch deps. Its telemetry was switched off:
    `sync: false` in `~/.config/Ultralytics/settings.json`.
  - YOLOX 0.3.0 (PyPI sdist) had to be built against torch with `--no-build-isolation`. First attempt failed: the venv
    was on system Python without `Python.h`; it worked once recreated on uv's managed 3.11.17.
  - Main `pyproject.toml`/`uv.lock` unchanged.
- **Weights** in `artifacts/weights/` (gitignored), sha256:
  - `yolov8n.pt`: `f59b3d83…`
  - `yolov8s.pt`: `1f47a78b…`
  - `yolox_s.pth`: `f55ded71…` (in config). From the upstream release `0.1.1rc0`; GitHub was serving it at ~15 KB/s,
    so the rest came via 8 parallel HTTP ranges from the same URL.
  - Integrity of `yolox_s.pth`: size = server content-length (72,089,125 B); the 8.9 MB that overlaps with the
    independent sequential download is byte-identical; loads with 462 tensors. VERIFY: no official checksum is
    published.
- **Licenses verified from the files.** Ultralytics: `LICENSE` in the wheel = **AGPL-3.0**. YOLOX: upstream
  `LICENSE` at commit `6ddff48` = **Apache-2.0** (the PyPI sdist ships no LICENSE file).
- `rewind/detect/detectors.py`: one interface for both backends. `class_ids` takes COCO names.
  - YOLOX preprocessing is as upstream `ValTransform(legacy=False)`, **verified** on real frames: legacy input gives
    recall 0, non-legacy 0.30/0.76.
  - Fixed: YOLOX postprocess edits a tensor in place, so it must run inside `inference_mode`.
- `scripts/detector_smoke.py` (rule 16 measurement, tuning only):
  - All detector settings run on one shared decode; the MOG2 motion boxes are scored as the no-detector baseline.
  - Writes JSON per setting, `results/reports/detector_3.3.md`, 50 drawn frames per setting under
    `results/figures/detector_smoke/` (gitignored), and experiments rows.
  - Method: recall of annotated actor boxes on the 5 fps grid, greedy one-to-one at IoU ≥ 0.5, 2,680 frames
    (≤ 60 per tuning clip).
  - Pre-registered before any run: overall = mean of the cells RGB person / RGB vehicle / IR.
- `tests/test_detector_smoke.py`: 8 tests (IoU, greedy and centre matching, dedup, size buckets, spec parsing).

**Key numbers** (`results/reports/detector_3.3.md`)

| setting | RGB person | RGB vehicle | IR | overall (pre-reg.) | pooled | detector fps RGB | projected full pass |
|---|---|---|---|---|---|---|---|
| YOLOv8n 640 | 0.139 | 0.517 | 0.008 | 0.221 | 0.206 | 322 | 1.9 h |
| YOLOv8s 640 | 0.212 | 0.601 | 0.008 | 0.274 | 0.279 | 229 | 2.0 h |
| YOLOX-s 640 | 0.171 | 0.644 | 0.004 | 0.273 | 0.255 | 113 | 2.4 h |
| YOLOv8n 1280 | 0.303 | 0.691 | 0.000 | 0.331 | 0.368 | 109 | 2.7 h |
| YOLOv8s 1280 | 0.397 | 0.789 | 0.004 | 0.397 | 0.460 | 68 | 3.4 h |
| **YOLOX-s 1280** | **0.407** | **0.830** | 0.004 | **0.414** | **0.476** | 25 | 6.1 h |
| motion boxes (no detector) | 0.101 | 0.419 | 0.083 | 0.201 | 0.160 | – | – |

- **By object size:** 62 % of RGB person boxes are < 5 % of the frame height; YOLOX-s 1280 finds 0.24 of those. Objects
  ≥ 15 % of the frame height are found 83-97 % by every detector.
- **IR ground truth unusable at IoU 0.5:** 231 of 242 IR boxes are CMU-v2 (loose, out of time). Viewed by Claude
  (n = 3 IR frames): the car is detected, but inside a GT box ~4x larger.
- **Viewed by Claude** (n = 4 RGB frames): parked cars and the annotated bus are boxed (motion crops miss parked
  cars); the misses are tiny, distant people.

**GATE 3.3 — decision: YOLOX-s, RGB 1280 px / IR 640 px (Ashutosh chose the recommended option, 25 Oct)**
- **YOLOX vs YOLOv8: the rule decides.** YOLOv8 is never ≥ 5 points higher at the same input size:
  - 640: +0.1 pt overall, +2.4 pt pooled;
  - 1280: −1.7 pt overall, −1.6 pt pooled;
  - the same holds RGB-only.
- **The "no detector if both < 50 %" clause did not decide clearly, so Ashutosh was asked.**
  - Below 50 %: the pre-registered mean of cells (0.414) and pooled (0.476).
  - Above 50 %: RGB-only mean (0.618).
  - The IR cell is ~0 for every detector because of the ground truth, not the detector.
  - The alternative (motion crops) is worse on every RGB cell.
  - Ashutosh kept the detector. **This is a documented deviation from the literal rule.**
- **IR at 640 px:** upscaling 352x240 thermal frames to 1280 broke detection. Example: the same frame gave a car
  at 0.72 at 640 and "clock 0.49" at 1280. IR IoU-0.3 recall is 0.103 at 640 vs 0.037 at 1280.
- **No license problem:** Apache-2.0. Ultralytics (AGPL) is not a project dependency; `UltralyticsDetector` stays as an
  optional, lazily imported backend used only for this comparison.
- **YOLOX vendored**, not added as a package:
  - Why: the sdist compiles a C++ op against torch at build time and pins onnx 1.8.1 (no Python 3.11 wheel).
  - What: `rewind/detect/yolox_vendor/` holds the 8 inference modules copied verbatim. Only the `yolo_head.py`
    imports were changed (loguru → logging; relative imports). Includes `LICENSE` and `NOTICE.md`.
  - Check: outputs are **bit-identical** to the pip package (max abs diff 0.0 on a random batch).
  - **No new pip dependency.**
- **Config:**
  - `detection` filled: enabled, backend yolox, model yolox-s, weights path/url/sha256, license, imgsz
    `{rgb: 1280, ir: 640}`, `nms_iou: 0.65`, `batch_size: 16`.
  - New section `detection_check`.
  - `appearance.crop_source` stays `detection_then_motion`.

**Self-review (PROMPT R) — problems found and fixed**
1. `sampled_frames` crashed while building its own error message when given a `str` path → `Path(path)`.
2. YOLOX postprocess ran outside `inference_mode` → moved inside.
3. Scratch venvs were first created on system Python (different base than `.venv`) → recreated on managed 3.11.17.

**Known limitations of the 3.3 evidence (recorded, not fixed)**
- The 4 Ultralytics experiments rows carry git `e3ddc97-dirty`: the process loaded code from `79aee4c` at start;
  later uncommitted edits (motion baseline) did not affect the running process.
- The smoke test's projections use CPU decode time at the shared 1280 decode width, so the 640 rows overstate
  CPU time.

**Open issues**
- Precision cannot be measured (MEVA only boxes actors of annotated activities). IR detection quality is
  unmeasured beyond eyeballing.
- Reproducing the YOLOv8 rows needs Ultralytics installed by hand (scratch venv recipe above). The YOLOX rows
  reproduce in the main env.

---

## Phase 3 full pass — launched (Sun 25 Oct, 18:49)

- `scripts/index_pass.py` covers steps 3.1 and 3.4 in one decode.
  - Per clip: 5 fps samples decoded once at max(motion width, detection input size); MOG2 at 640 px; YOLOX-s at the
    modality's input size; a decode thread overlaps CPU decoding with GPU detection.
  - Outputs: `artifacts/motion/<camera>/<clip>.{seconds,boxes}.parquet` and
    `artifacts/detections/<camera>/<clip>.parquet` (original-pixel boxes, `t_s = frame / fps`), plus a
    `.done.json` marker written last.
  - Resumable. It refuses to mix settings (settings hash of config `motion` + `detection`) and stops on any error.
- `tests/test_index_pass.py`: 4 tests:
  - prefetch order and error propagation;
  - settings hash;
  - detector config refusals (TBD, missing imgsz, wrong weights hash);
  - one clip end to end with a fake detector (box scale-back, timestamps, markers).
- Short test (scratch output):
  - 1 RGB clip in 69.7 s (63,499 detections), 1 IR clip in 16.0 s.
  - **Projection 5.1 h** for 300 clips (< 8 h, so launched without asking, per Ashutosh's instruction).
  - The IR clip's MOG2 output is identical to the earlier 3.3 test (6,196 boxes).
- **Running:** `nohup uv run python -m scripts.index_pass`, PID 2559056 (python; `uv run` parent 2559052). Log `results/logs/index_pass.log`, PID file
  `results/logs/index_pass.pid`. First ETA 5.46 h (started 18:49, expected done ~00:15). It appends one experiments row at the end. Re-run
  the same command to resume after an interruption.
- Note: for RGB, MOG2 now gets frames decoded at 1280 and area-resized to 640 (3.1 decoded directly at 640). This is
  a small pixel-level difference; the 3.1 check numbers are not re-measured on it.

---

## Full pass — mid-run output check (Sun 25 Oct, ~19:12)

- `scripts/check_index_pass.py` (read-only, reusable for the final check), with a test in `tests/test_index_pass.py`.
  For every finished clip it checks: files and marker present, settings hash, sample and per-second counts, the
  activity range, frames on the 5 fps grid, `t_s = frame / fps`, boxes inside the frame, detection classes and
  scores against config.
- At 21/300 clips (all G330, the indoor gym): **0 problems**, no leftover `.tmp`. One detection box overhanging the
  bottom edge by 16.7 px (a person cut by the frame) tripped my first, absolute 16 px tolerance → tolerance is now
  3 % of the frame size.
- Viewed by Claude (n = 4 frames):
  - **Two empty clips are correct.** G330 03-07 16:55 and 17:25 show an empty gym: activity 0, 0 detections.
  - **Static false positive.** G330 on 03-11: two small dark objects lying on the bleachers are detected as `car`
    (score 0.30-0.57) on nearly every sample, 5,970 detections over 8 clips (2.8 % of G330 detections).
- **Open issue for 4.2 / 6.1:**
  - 4.2 crop choice should rank by size × score; these boxes are 39x22 px at low score, so they rarely reach the top
    4 in a crowded gym frame.
  - A UI must not present the crop class as fact.
  - The 6.1 stop/start rules are unaffected: they need a moving track first.
  - Also measure static repeated boxes per camera after the pass.
- IR (G474, processed last) not checked yet; covered by the final check after the pass.

---

## Step 3.2 — Motion-ranked baseline: PREP DONE, final run waits for the full pass (Sun 25 Oct, 19:25)

**Done (code + tests, committed)**
- `rewind/eval/baselines.py`:
  - `segment_activity`: mean per-second MOG2 activity over a 60 s segment; each clip second is weighted by its
    overlap with the segment, and a short last second counts only for the part that exists.
  - `motion_ranked`: segments by descending activity in one ranking over all cameras (raw activity, no per-camera
    normalisation); ties broken by camera, then time.
  - Fixed before any result was seen: score = mean (not max); raw activity.
- `scripts/run_baselines.py`:
  - adds `motion` as a 4th system: specific queries rank all segments, category queries the RGB segments only (as
    the other baselines);
  - `--log-systems` so 3.2 logs only the motion rows (the other three were logged at 2.3);
  - `--art`; refuses to run if any tuning/dev clip lacks its index_pass done marker.
- `rewind/motion/store.py`: one place for the index_pass output paths and `load_activity`. `index_pass.py` and
  `check_index_pass.py` now use it (no change in behaviour; tests pass).
- Tests: 2 new hand-computed cases in `tests/test_metrics.py` (overlap weighting incl. a partial last second and
  refusals; ranking and ties, plus seconds-watched on the ranking). Suite: 275 pass.
- Plumbing trial on the finished G330 clips only (not a result, not logged): segments and scores line up; the full
  script correctly refuses while clips are missing.

**To do after the pass (exact commands)**
1. `uv run python -m scripts.check_index_pass --report results/reports/index_pass_check.md` → expect 0 problems.
2. `uv run python -m scripts.run_baselines --log-systems motion --report results/reports/baselines_3.2.md`
   → check that sequential / random / witness are identical to `baselines_2.3.md` (deterministic), then write
   the short comparison table and the "Done when" for 3.2.
3. PROMPT R on 3.2, PROGRESS entry, commit `phase3.2: ...`.

**Expectation (from 3.1, stated before the run):** whole-frame activity barely separates events from random
windows (AUC 0.55), so the motion baseline is likely not much better than sequential on busy cameras.

---

## Phase 3 full pass — finished (Sun 25 Oct, ~23:17) — step 3.4 "Done when" met

- Finished in **4.47 h** for 300 clips (25.0 h of video, 5.6x realtime); smoke-test projection was 5.1 h.
  The experiments row was logged by the script (split `none`, settings `afe8a9f91a22f367`).
  - 450,206 samples, 1,430,491 motion boxes, 5,511,258 detections.
  - Disk: `artifacts/motion` 18 MB, `artifacts/detections` 89 MB.
- Detector recorded in config: YOLOX-s, Apache-2.0, weights sha256 in `detection.weights_sha256`, RGB 1280 / IR 640.
- **Final output check** (`results/reports/index_pass_check.md`): 300/300 clips, no leftover `.tmp`.
  - **1 flag, reviewed, not a pipeline error.** One G340 clip has 50 `car` boxes running up to 479 px below the
    frame. Viewed by Claude (n = 2): a car parked right in front of the camera is cut by the frame edge, and YOLOX
    predicts its full extent past the edge. The box around the visible car fits, so the scale-back is correct.
    Every other clip overhangs ≤ 28 px. The tolerance is kept, so a real scaling bug would still trip it.
  - **Detection counts per camera / class** (the 3.4 "Done when"):

    | camera | mod | person | car | truck | bus | motorcycle | bicycle | total |
    |---|---|---|---|---|---|---|---|---|
    | G330 | rgb | 724,253 | 7,059 | 0 | 0 | 2 | 83 | 731,397 |
    | G336 | rgb | 102,847 | 854,343 | 79,491 | 961 | 128 | 2,554 | 1,040,324 |
    | G340 | rgb | 114,188 | 797,860 | 106,094 | 1,651 | 43 | 831 | 1,020,667 |
    | G424 | rgb | 137,315 | 1,626,385 | 238,828 | 9,762 | 317 | 978 | 2,013,585 |
    | G436 | rgb | 289,084 | 329,720 | 34,259 | 3,399 | 2,445 | 4,289 | 663,196 |
    | G474 | ir | 14,855 | 23,076 | 3,498 | 50 | 98 | 512 | 42,089 |

    Most `car` detections on the outdoor cameras are parked cars, re-detected on every sample. IR has ~25x fewer
    detections per clip than RGB, which matches the 3.3 finding that IR objects are small at 640.
- **Second static false positive.** G330 03-15 15:55 has 0 motion but a `person` on every sample. Viewed by Claude
  (n = 1): a bag at the bottom edge of the gym floor, scores 0.30-0.53.

---

## Step 3.2 — Motion-ranked baseline (Sun 25 Oct, 23:45)

- Ran `scripts.run_baselines --log-systems motion` on tuning and dev → `results/reports/baselines_3.2.md`; 2 rows
  logged (motion only). **Sequential / random / witness reproduce `baselines_2.3.md` exactly** (134 of 134
  table rows identical), so adding motion changed nothing else.
- Comparison (specific queries: median seconds watched before the first hit and Recall@10; category: mAP@10):

  | split | metric | sequential | random | witness ±10 min | **motion** |
  |---|---|---|---|---|---|
  | tuning (6 spec / 3 cat) | median s watched | 8,091 | 9,063 | 3,126 | **9,734** |
  | | R@10 | 0.056 | 0.027 | 0.056 | **0** |
  | | mAP@10 | 0.350 | 0.286 | — | **0.487** |
  | dev (12 spec / 5 cat) | median s watched | 5,932 | 6,413 | 3,237 | **1,513** (CI 179–2,985) |
  | | R@10 | 0.083 | 0.050 | 0.083 | **0.333** |
  | | mAP@10 | 0.103 | 0.088 | — | **0.178** |

- **Reading (diagnosis on tuning and dev, not tuning anything):**
  - The dev win is partly a **camera effect**, not motion picking out events. The ranking is global on raw activity.
    In dev, the indoor gym G330 is the busiest camera: 16 of the top 20 segments are G330 (the other 4 are G474 IR),
    and 6 of the 12 dev specific queries are G330 events. Within G330, those events rank 1, 4, 13, 17, 25 and 33 of
    35 segments, i.e. near random.
  - The two dev IR queries (dev-s05, dev-s09) both sit in the same G474 segment at global rank 2. That is what
    gives the 62 s results.
  - In tuning, G340 holds 10 of the top 20 segments, but no tuning query is on G340, so motion is the worst system
    on time watched.
  - Category: motion helps mainly for one activity, dev-c03 (231 instances, AP@10 0.89). For the other four dev
    category queries it puts 0 instances in its top 10.
  - My expectation stated before the run ("likely not much better than sequential") was **wrong for dev and right
    for tuning**. The baseline is high-variance and depends on which cameras the queries fall on.
- **Implication:** motion is the strongest baseline on dev, so Rewind must beat a median of ~1,500 s there, not
  ~3,200 s (witness). The 1.10 rules are unchanged; no baseline is re-defined after seeing results. Step 8.2
  reports all four baselines on test as planned.
- **Self-review (PROMPT R, inline):**
  - No silent failures: missing markers and segment mismatches raise.
  - No new tunables: segment length comes from `baselines.segment_s`.
  - Tests use hand-computed answers.
  - Only tuning and dev are read: the split loop is hard-coded to (tuning, dev); `index_pass` reads footage of
    every split but no labels.
  - The overlap weighting for partial seconds is tested.
  - Nothing found to fix. Suite: **275 pass**.
- **Open issues carried forward:**
  - (a) **4.2 must clip detection boxes to the frame** before cropping.
  - (b) **Static false positives** (bleacher `car`s, a bag as `person`) and many parked-car re-detections. 4.1
    dedup ("detections unchanged") and 4.2 crop ranking should handle these; measuring repeated static boxes per
    camera is still to do (cheap, at 4.1).
  - (c) `motion.min_box_area_px` is still the untuned smoke value.
  - (d) Detector precision is not measured; IR quality has only been looked at by eye.

---

## Phase 3 summary (Sun 25 Oct)

- **Order (Ashutosh):** 3.1 test → 3.3 gate → one combined full pass for 3.1 + 3.4 → 3.2.
- **3.1 MOG2:** whole-frame activity barely separates labelled events from random windows (AUC 0.55, because events
  cover 70–99 % of the time), but box-local activity does (AUC 0.81–0.87). 20x realtime alone.
- **Gate 3.3:** YOLOX-s (Apache-2.0), RGB 1280 / IR 640, chosen by Ashutosh after the literal "<50 %" clause was
  ambiguous (RGB recall: person 0.41, vehicle 0.83; IR recall unmeasurable against the loose CMU boxes). YOLOv8 was
  never ≥ 5 points better. The YOLOX code is vendored (no new pip dependency).
- **Full pass:** 300 clips, 25 h of video, in 4.47 h. Output check: 0 pipeline errors, 1 reviewed flag.
- **3.2:** motion baseline beats the other baselines on dev (median 1,513 s) and loses on tuning (9,734 s). The dev
  win comes largely from the busiest camera holding half the dev queries.
- **Process lessons:**
  - The full pass ran single-core-bound at ~55 % GPU because the pipeline wasn't profiled before launch.
    Ashutosh chose to leave it.
  - A PID-file mistake (it recorded the launcher, not the job) and three wrong clock times in PROGRESS were fixed.
- **Next: Phase 4** (`/clear` + PROMPT S recommended first). 4.1 dedup sweep on tuning, then 4.2 embeddings, with
  open issues (a)–(c) above.

---

## Verification pass over Phases 0–3 (Sun 25 Oct, 23:55, asked by Ashutosh: "verify everything till now quickly and fix")

Inline checks, all run now:

| check | result |
|---|---|
| Test suite | 276 pass (275 + 1 new) |
| Frozen queries: `eval/queries/HASHES` (7 files) | all OK; no commit touches `eval/queries/` after Gate 2.2 (979c37e) |
| Raw evidence: SHA-256 of all 300 originals vs manifest | **300/300 match**; 0 writable files in `data/raw/` |
| Proxies | 300 files present; frame count and duration (≤ 1 s) match the originals for all 300; frame alignment best at offset 0 for all 300 |
| Splits | 58 / 42 / 200 clips (tuning / dev / test); 0 clips in two splits |
| Test-split access | only via `check_split_access` (refuses unless `allow_test`); only `make_worksheets` (step 2.2, permitted) passes it |
| Error handling | 3 `except` clauses outside vendored code: all re-raise or list the failure and exit 1 |
| Secrets / large files | none tracked; `data/` and `artifacts/` untracked except `data/selection.txt` |
| `uv.lock` | consistent with `pyproject.toml` |
| PROGRESS numbers vs `experiments.csv` | 3.1 AUCs, 3.3 recalls/projections, 2.3 and 3.2 baseline medians all match |
| 3.1 claim "decoding RGB at 1280 then resizing to 640 is a small difference" (was unmeasured) | **measured:** per-second activity, full pass vs direct-640 decode, on both 3.1 clips: Pearson r 0.9997 / 0.9990, rank agreement 0.998 / 1.000 |

**Found and fixed: git hash read at the end of long jobs**
- `log_experiment` read the git and config hashes when the row was written. A commit made while a long job ran was
  therefore logged as that job's code.
- This happened twice:
  - the 3.3 Ultralytics rows (`e3ddc97-dirty`, already documented);
  - the full-pass row (row 19, `3b9dcc0`), whose process loaded code from **`ad3f975`** at 18:49. The diff
    `ad3f975..3b9dcc0` is only the `rewind/motion/store.py` path refactor (same paths, same outputs).
- The row is left as written (the log is append-only); this note is its correction.
- Fix: `run_stamp()` in `rewind/experiment_log.py` takes both hashes at the start. `index_pass`, `detector_smoke` and
  `throughput` take the stamp at start and pass it to `log_experiment` (`stamp=`). The stamp also hashes the
  `--config` file actually used; before, the logger always hashed the default config. New test: a commit and a
  config edit during a run leave the stamped row unchanged.

Not re-checked here (done at their steps, unchanged since): annotation parsing vs NIST spec, the 5-event manual
check (1.3). The `VERIFY` markers in `annotations.py`, `manifest.py` and `build_events.py` remain open by design
(undocumented dataset fields, ±1 s start times).

---

## Autopilot set up (Mon 26 Oct, 00:30) — Phases 4 → 8.1 run unattended

- Ashutosh's instruction. Rules in `AUTOPILOT.md` (they win over CLAUDE.md for autopilot runs); runner
  `scripts/autopilot.sh` starts one headless `claude -p` run per phase/long job and stops at `NEEDS_USER`, `FAILED`,
  a claude error, or after Gate 8.1. Step 8.2 is never run by the autopilot.
- Checked before launch: flags against `claude --help` (2.1.310); one real headless run with the runner's flags
  (allowed: ls, nvidia-smi, date, git status, uv, pkill -F, Write; denied: git push, sudo, rm -rf, git reset --hard,
  bare python3); the loop on a fake `claude` (WAITING_JOB → wait → DONE_PHASE → NEEDS_USER; stop after phase 8;
  stop on missing status.json; stop on non-zero exit; kill of the runner ends the child).
- Local `.git/hooks/pre-push` (not committed) refuses pushes while `REWIND_AUTOPILOT` is set.
- Known stops ahead: 7.1 (`engine/` empty, 👤), Gate 6.2 if the paired rule says keep (no agreed precision/recall
  bar), 7.4 (no agreed weight objective).

---

## Step 4.1 — Sampling + motion-aware dedup (Mon 26 Oct, autopilot run 1)

**Done**
- `rewind/embed/dedup.py`: 1 fps candidates (on the 5 fps motion/detection grid); each is compared with the **last
  kept** frame and dropped only if (1) the share of analysis-frame pixels (gray, 640 px wide) changed by >
  `dedup_pixel_delta` inside the union of both frames' MOG2 boxes is below `dedup_motion_pixel_thresh`, (2) the
  detections are essentially unchanged and (3) it is < `max_gap_s` after the last kept frame. `Deduper` (online,
  used by the 4.2 embedding job) and `keep_flags` (list) share one rule.
- "Essentially unchanged" = every detection with score >= `dedup_det_min_score` (0.5) in either frame has a
  same-class box with IoU >= `dedup_det_iou` (0.5) in the other. Why a score floor (measured on tuning before the
  sweep, consecutive 1 fps frames): exact detection sets are identical on 0-3 % of outdoor seconds (14-33 boxes per
  frame, low-score boxes flicker), so a literal "set unchanged" would keep ~98 % of outdoor frames whatever the pixel
  threshold. With the floor: outdoor 1-36 %, G330 90 %, G474 81 % unchanged. Floor, IoU and pixel delta (25 grey
  levels) were fixed before the sweep and are not tuned.
- `scripts/dedup_sweep.py`: decodes every tuning clip once at full resolution (the 4.2 path), applies the 8 swept
  thresholds to the same candidates; `results/reports/dedup_4.1.md`; 8 experiments rows (git `bcecd4f`).
- **Pipeline recall, fixed before the run:** share of tuning events (3,586: Kitware RGB + contrib IR, per piece) with
  >= 1 kept frame in the event's frame span widened by `metrics.event_tolerance_s` (2 s; a hit clip carries 5 s of
  context). Strict (no widening) reported next to it.
- `tests/test_dedup.py`: 10 tests (mask scaling and clipping, changed share incl. the delta boundary and the region
  union, detection rule incl. flicker and class change, max gap, comparison with the last kept frame not the
  previous one (slow drift), threshold 0 keeps all, recall window edges).

**Key numbers** (tuning, 17,436 candidates)

| threshold | kept | recall ±2 s | strict | G330 kept / recall (n=23) | G474 kept / recall (n=33) |
|---|---|---|---|---|---|
| 0 (keep all) | 100 % | 1.000 | 0.977 | 100 % / 1.000 | 100 % / 1.000 |
| 0.0003 | 80.9 % | 0.997 | 0.964 | 21.7 % / 0.957 | 97.4 % / 1.000 |
| 0.003 | 72.4 % | 0.991 | 0.948 | 19.9 % / 0.826 | 71.1 % / 1.000 |
| **1.0 (chosen)** | **62.9 %** | **0.990** | 0.941 | 19.3 % / 0.826 | 25.7 % / 0.848 |

Outdoor cameras keep 67-99 % of frames at every threshold (something changes almost every second in traffic).

**Decision (rule, AUTOPILOT.md 3: fewest kept frames with tuning pipeline recall >= 0.95):
`dedup_motion_pixel_thresh: 1.0`** — written to config. Every swept value reached >= 0.99. At 1.0 the pixel test
never keeps a frame: detection changes + the 10 s max gap decide alone (a grid value fixed before the run as the
"detections + max gap only" setting).

**Open issues (flagged, not acted on: the rule decides)**
- **The quiet cameras pay for it:** G330 recall 0.83 (n = 23) and IR G474 0.85 (n = 33) at 1.0 vs 0.96 / 1.00 at
  0.0003. The overall number is dominated by the busy outdoor cameras (3,530 of 3,586 events). Dev is G330-heavy
  (72 % of dev activities), so this is the tuning → dev transfer risk pre-committed in open-issue item 7: the 4.4
  and 8.1 reports show G330 separately. If 4.4 shows G330 dev queries missing because their frames were dropped,
  that is a finding to report, not a reason to re-tune on dev.
- The ±2 s recall is lenient by design (a kept frame within 2 s leads to the event through the clip's context);
  strict recall drops 0.977 → 0.941.

---

## Step 4.2 — Embeddings: frames + crops (Mon 26 Oct, autopilot run 1)

**Done (code)**
- `rewind/embed/clip_model.py`: open_clip fp16 with **preprocessing on the GPU, never a centre crop** (item 4):
  frame views `tile0` (x 0..H), `tile1` (x W-H..W), `pad` (letterboxed W x W, black bars); object crops = box widened
  by `crop_pad_frac` per side, grown to a square around its centre (context, not bars; black outside the frame);
  bicubic + antialias resize, the model's own mean/std. `check_pretrained_compatible` moved here from
  `scripts/gpu_check.py` (re-exported there).
  **Measured vs open_clip's PIL path** (32 real tiles, G424): embedding cosine min 0.9993 / mean 0.9998 (laion2b and
  openai); input mean |diff| 0.0014 normalised units.
- `rewind/embed/crops.py`: detection crops (clipped, `min_crop_px` on the **longer** side, ranked by `crop_rank`,
  <= 4) and motion crops (by area, <= 4, `dup` when chosen detections cover >= `crop_dup_cover`), stored separately;
  `variant()` / `rewind/embed/store.variant_mask` build none / motion / detection_then_motion from one run.
- `scripts/embed_appearance.py`: per clip, 1 fps decode at full resolution -> `Deduper` (4.1) -> kept frames ->
  3 frame views + crops -> CLIP -> `<clip>.npy` (fp16, L2-normalised) + `<clip>.parquet` (video, camera, modality,
  frame, t_s, type, view, crop_source, crop_rank, dup, class, score, box) + done marker; resumable, settings hash
  (model+tag, sampling/dedup, crop settings, views, preprocessing version, index_pass settings). Several models in one
  decode (tag check). `--pid-file`: the job writes its own PID.
- `scripts/check_crop_ranking.py` (tuning, boxes only): annotated actors covered by the chosen crops.
- Tests: `test_crops.py` (5), `test_clip_views.py` (5), `test_embed_store.py` (4); suite 314 pass.

**Crop decisions (tuning, `results/reports/crops_4.2.md`, rows in experiments.csv)**
- **`min_crop_px` is tested on a box's longer side** (the embedded square is built from it). The first check used
  the shorter side: even all detections then covered only 8 % of annotated people (standing people are ~2.5x taller
  than wide); on the longer side all detections cover 0.398 (Gate 3.3 recall 0.407: consistent). The first row
  (git `bcecd4f`) used the shorter side and is superseded by the row at `7761279`.
- **`crop_rank: moving_first`** (rule fixed in the script before the run: replaces score_area only if better on the
  mean of RGB person / vehicle coverage by the detection_then_motion crops): **0.331 vs 0.169** (person 0.148 vs
  0.047, vehicle 0.514 vs 0.291). On outdoor cameras score x area fills every slot with parked cars.
- Viewed by Claude (n = 1 frame, 11 images, G424): tiles, padded frame and crops are the right pixels; detection
  crops = moving car, van, walking person, truck; one motion crop = a car door opening that no detection covers.
  Detector class vs CLIP text ("a photo of a car / person") on that clip's detection crops: car 611/612, person
  296/312.

**Open issues**
- The 4-crop cap limits person coverage to 0.148 (all detections: 0.398). `max_crops_per_frame` stays 4 (CLAUDE.md
  value); a candidate for a later ablation, not changed here.
- A motion box around a walking person is larger than the detection (9 px dilation at 640 px), so it is often not
  marked `dup` (coverage < 0.5): some people are embedded twice in detection_then_motion. Not tuned.
- **Tag-check embeddings (tuning, 3 tags in one decode):** 58 clips, 17,436 candidates -> **10,974 kept** (exactly
  the 4.1 sweep's count at 1.0: the online `Deduper` matches the sweep), 102,112 vectors per tag (9.3 per kept
  frame), 24.2 min, peak VRAM 2,863 MiB with 3 models loaded. Experiments row at `9fb1566`.

**Pretrained tag — NEAR-TIE, NEEDS_USER** (`results/reports/pretrained_4.2.md`, 6 rows at `7d34b10`; agreed method
item 5, rule fixed in `--tag-check` before the run). 29 Kitware types with >= 10 tuning activities, templated
prompts ("a person opens vehicle door"), main variant (tiles2, detection_then_motion, centered on tuning vectors):

| tag | type mAP@50 centered (decides) | mAP@10 centered | mAP@50 / @10 raw | 6 specific: R@10, median s | 3 category mAP@10 |
|---|---|---|---|---|---|
| laion2b_s34b_b88k | **0.105** | 0.161 | 0.105 / 0.155 | 0.167, 9,628 | 0.173 |
| datacomp_xl_s13b_b90k | 0.094 | 0.141 | 0.111 / 0.189 | 0.167, 9,391 | 0.224 |
| openai (quickgelu) | 0.080 | 0.130 | 0.078 / 0.122 | 0.000, 567 | 0.122 |

- laion2b has the best mean but vs datacomp_xl wins 12 types, ties 1, **loses 16** (mean diff +0.012, 95 % CI
  [−0.009, +0.035]) -> the rule's "wins on more types than it loses" fails -> near-tie. vs openai: 17/0/12, +0.025
  [+0.008, +0.046]. laion2b is much better on vehicle_starts/stops (0.35/0.41 vs 0.19/0.20) and
  person_opens_vehicle_door; datacomp_xl on door/phone/pick-up types. openai is behind on both counts.
- Retrieval is weak on tuning for every tag (type mAP@50 ~0.1; 2-3 of the 6 specific tuning queries are missed by
  every tag within the top 200 vectors). Centering helps the specific queries (median 9.6k vs 18.3k s, laion2b) but
  not the category means. Noted for 4.4; nothing tuned on it.

**Open issues**
- **Long-job launch deviation (AUTOPILOT.md 6):** in this headless run `setsid` and `&` both need approval and are
  denied, so `setsid nohup ... &` cannot be used. Jobs run through the harness background runner instead
  (`uv run ...` with run_in_background, max 2 h), PID via `--pid-file`; the run stays alive and continues after
  the job instead of ending with WAITING_JOB. Resumable per clip if the session dies.

---

## Gate rule — operational form, pre-registered (Mon 26 Oct, commit `0eeb29a`, before any dev search result)

The confirmed rule (item 8) says "mean better AND more wins than losses", but dev mixes specific and category
queries. Operational form, in `rewind/eval/gate.py` with 6 hand-worked tests, used at Gates 4.4 / 5.1 / 6.2 / 8.1:
- each query by its own metric: specific = seconds watched (headline), category = AP@50; a specific query missed by
  both variants is a tie (their miss costs differ only by list length);
- "mean better" per family; with both families present the complex variant must be strictly better in one and not
  worse in the other; adopt = mean better AND wins > losses over all queries.
- Gate 4.4 order (`scripts/appearance_search.py --ablation` docstring): crop_source none -> motion ->
  detection_then_motion (progressive), then mean_center on vs off, then frame_mode tiles2 vs pad; each at the
  choices made before it and the config's other values.
- Step 4.2 tag rule (`--tag-check` docstring): highest mean AP@50 over templated activity prompts on tuning, and it
  must win more types than it loses against each other tag; otherwise near-tie -> NEEDS_USER.

---

## Autopilot retired (Mon 26 Oct, ~09:50) — back to interactive sessions

- **Ashutosh's instruction:** the autopilot is retired; work continues interactively. `AUTOPILOT.md`,
  `scripts/autopilot.sh` and `results/autopilot/` stay in the repo for reference only. CLAUDE.md (incl. rule 16) is
  the only rule set again.
- It made **one** run (run 1, 00:30 → 01:22, 190 turns, ~$11.1, 10 permission denials) covering step 4.1 and step 4.2
  up to the tag check, then stopped at `NEEDS_USER` (pretrained tag near-tie).
- **Check of every long job it started** (transcript `results/autopilot/run_1.log` searched for every background
  launch; logs, PID files, outputs and experiments rows checked):

  | job | log | outputs | status |
  |---|---|---|---|
  | `scripts.dedup_sweep` (4.1, launched twice; the 2nd overwrote the 1st) | `dedup_sweep.log` ends with `choice: … 1.0` | `dedup_4.1.md`, 8 rows at `bcecd4f` | complete |
  | `scripts.check_crop_ranking` (4.2, 2 logged runs + 1 `--no-log` render) | both logs end with the decision line | `crops_4.2.md`, rows at `bcecd4f` (superseded) and `7761279` | complete |
  | `scripts.embed_appearance --tuning-only`, 3 tags | `embed_tagcheck.log`: 58/58, `done: … 102112 vectors`; PID dead | per tag: 58/58 `.npy` + `.parquet` + `.done.json`, rows = vectors, fp16, finite, unit norm (max dev ≤ 2e-3), settings hash = current config's (laion2b `db5765f2a902ab59`); 102,112 vectors, 10,974 kept, 17,436 candidates per tag; no stray/partial files | complete |
  | `scripts.appearance_search --tag-check` | `tag_check.log` ends `DECISION: NEAR_TIE` | `pretrained_4.2.md`, 6 rows at `7d34b10` | complete |

  The three `setsid nohup … &` launch attempts were denied (never executed); the 4th (harness background runner)
  is the one that ran. No orphan processes (all PID files point to dead processes). **Nothing needs re-running.**
  (The Phase 3 `index_pass` predates the autopilot and was checked on 25 Oct.)

## Step 4.2 — pretrained tag DECIDED: `laion2b_s34b_b88k` (Ashutosh, Mon 26 Oct)

- Evidence (`results/reports/pretrained_4.2.md`, 6 experiments rows at `7d34b10`; tuning, 29 templated Kitware types,
  tiles2 + detection_then_motion, mean-centered): type mAP@50 **laion2b 0.105**, datacomp_xl 0.094, openai 0.080.
  laion2b vs datacomp_xl: 12 wins / 1 tie / 16 losses, mean diff +0.012, 95 % CI [−0.009, +0.035] → the agreed rule
  did not decide (near-tie). vs openai: 17/0/12, +0.025 [+0.008, +0.046]. laion2b is clearly better on
  vehicle_starts / vehicle_stops (AP@50 0.35 / 0.41 vs 0.19 / 0.20).
- **Decision: laion2b_s34b_b88k** (best mean on the gate's primary metric = the tie-break rule below). Honest caveat:
  the CI against datacomp_xl includes zero; this is a judgement call, not a measured win.
- `configs/config.yaml`: `appearance.pretrained: laion2b_s34b_b88k`. The 58 tuning clips already embedded with this
  tag (same settings hash) are reused by the full run.

## Step 4.2 — full appearance embedding run: DONE (Mon 26 Oct, 09:51 → 10:36)

- `scripts.embed_appearance` (laion2b_s34b_b88k, settings `db5765f2a902ab59`), launched with `nohup … &` (log
  `results/logs/embed_appearance.log`, PID file); the 58 tuning clips from the tag check were reused (same hash).
  242 clips / 20.2 h in **44.6 min (27× real time)**, peak VRAM 2,097 MiB, no errors; experiments row at `4194389`.
- **Verified (all 300 clips):** `.npy` + `.parquet` + `.done.json` each; rows = vectors; fp16, finite, unit norm;
  one settings hash; no partial files.
- **Totals:** 90,161 candidates → **49,689 kept frames (55.1 %)** → **456,293 vectors** (149,067 frame views,
  181,876 detection crops, 125,350 motion crops); **459 MB** on disk.

  | camera | modality | clips | vectors |
  |---|---|---|---|
  | G330 | rgb | 51 | 83,201 |
  | G336 | rgb | 51 | 94,568 |
  | G340 | rgb | 49 | 62,423 |
  | G424 | rgb | 50 | 96,247 |
  | G436 | rgb | 48 | 104,797 |
  | G474 | ir | 51 | 15,057 |

  Detection crops by class: car 93,991, person 74,650, truck 11,136, bus 1,180, bicycle 701, motorcycle 218.
- Review (PROMPT R, inline): no defects found in `embed_appearance.py`, `clip_model.py`, `crops.py`, `dedup.py`,
  `store.py`; no label or test-split access. Note: IR G474 has few crops (15k vectors, 9.7k of them frame views):
  detector recall on IR is low (Gate 3.3), so IR queries rely mostly on whole-frame views.

## Step 4.3 — Mean-centering: DONE (Mon 26 Oct, 10:38)

- `scripts.collect_appearance` (9 s, 3.1 GB RAM): `…/index/vectors.npy` (raw, 456,293 × 512 fp16, vector_id = row),
  `meta.parquet`, `camera_means.parquet` (36 rows = 2 frame modes × 3 crop sources × 6 cameras, all footage, no
  labels), and the centered main variant `centered_tiles2_detection_then_motion.npy` (284,780 rows) + ids. **0.72 GiB.**
  Both variants (raw, centered) exist for the 4.4 ablation; other variants are centered at load time from the means.
- **Checked independently** (scratch script): ids sequential; G330 / G474 means recomputed from the raw vectors match
  exactly (max diff 0.0); centered norms 0.9999–1.0001; main variant = tile0/tile1 + ≤ 4 crops per frame.
- **What centering does** (3,000 random same-camera frame-view pairs per camera): raw cosine mean 0.85–0.92, sd
  0.04–0.08 (background dominates) → centered mean 0.28–0.76, sd 0.12–0.28. Whether this helps retrieval is decided
  at Gate 4.4, not here.

## Step 4.4 — dev ablation + GATE 4.4: DECIDED `pad/motion/raw` (Mon 26 Oct)

- `scripts.appearance_search --ablation` (15 s): 12 variants × 17 dev queries (12 specific, 5 category), 12
  experiments rows at `282334b`; report `results/reports/ablation_4.4.md`.
- **The pre-registered rule, applied literally, picks `pad/none/raw`, the worst variant on the headline metric**
  (median 13,515 s watched, 4/12 hits; worse than all dev baselines: sequential 5,932, random 6,413, witness 3,237,
  motion 1,513). Cause: at the first step, motion crops are much better on specific queries (mean +5,407 s, CI
  [+2,095, +8,760], 7/4/6) but slightly worse on the 5 category queries (AP@50 −0.028, CI [−0.062, +0.004]); "not
  worse in either family" vetoes on that point estimate, and the greedy order then never revisits crops.
- Gate code checked: it does exactly what was pre-registered (no bug).
- **Tie-break reading** (computed on dev with the same results, scratch script): treat the family conflict as "rule
  does not decide" and apply the agreed tie-break (better mean on the primary metric = seconds watched, CLAUDE.md
  1.10), wins > losses still required → **`pad/motion/raw`**: median 394 s, 9/12 hits, R@10 0.333, category mAP@50
  0.110 (best category value of all 12). Detection crops not adopted over motion (6/4/7, −17 s); centering not
  adopted (3/2/12 with motion crops, category −0.050 CI excludes 0); tiles2 not adopted over pad.
- Observation (not acted on): centering hurts once crops are in; the camera mean is taken over frames and crops
  together. A per-camera-per-type mean is a candidate fix, testable on tuning.
- **GATE 4.4 decision (Ashutosh, after seeing the dev results): `pad/motion/raw`** — the tie-break reading above.
  Honest record: (1) this reading was chosen AFTER the dev numbers were known; (2) the agreed tie-break rule was
  written for variants of equal complexity, and crops vs no crops is not that, so this is Ashutosh's explicit
  gate decision, not an automatic rule application. The literal rule's outcome (`pad/none/raw`) stays in
  `results/reports/ablation_4.4.md` as the evidence. Test split untouched; 8.1 re-runs the dev ablations.
  Reproducible: `uv run python -m scripts.gate44_tiebreak` (same numbers; log kept locally, logs are gitignored);
  `tiebreak_decision` has a hand-worked test in `tests/test_gate.py` (suite 315 pass).
- **Config:** `appearance.frame_mode: pad`, `crop_source: motion`, `mean_center: false`. 4.3 index re-collected for
  the new main variant (175,039 rows; `centered_pad_motion.npy` kept for ablation). **Detection crops are not used
  for retrieval** (they lost to motion crops, 6/4/7); detections still drive dedup (4.1) and the event branch (Phase 6).
- **Look at the results instead of the 👤 watch-5 check (viewed by Claude, n = 12 dev specific queries, 24 frames:
  ground-truth mid frame vs the top-1 clip's best hit with its box):** boxes land on the objects they claim (time,
  proxy seek and box scaling consistent). **No top-1 clip is the correct event**; first correct ranks 2, 2, 2, 9, 14,
  25, 27, 63, 81 and 3 misses. Failure patterns: crowded gym (G330) — a query about one person in a crowd matches
  any single walking person; distant cars give tiny crops (G424/G436 far road); IR queries (s05, s09 on G474) never
  rank an IR clip first — top-1 is an RGB car (s09: a red SUV at the G336 roundabout); vehicle_event R@10 is 0.
  So the gain over baselines comes from ranking, not top-1 precision: median 394 s watched vs motion baseline 1,513.

**Open issues after Phase 4**
- Dev is small (12 specific + 5 category queries): every gate decision here is noisy; CIs are wide.
- Centering hurts once crops are in (one camera mean over frames and crops together). Per-camera-per-type means are
  a candidate fix — test on **tuning** only if time allows; not scheduled.
- vehicle_event queries: R@10 0 on dev — the event branch (Phase 6) is where this can improve.
- IR retrieval is weak (few IR crops: detector recall on IR is low, Gate 3.3).

## Pre-agreed decision rules (Ashutosh, Mon 26 Oct) — apply without asking (CLAUDE.md rule 16)

- **Tie-break:** when two variants have equal complexity and the gate rule does not decide, choose the better mean on
  the gate's primary metric, and record in PROGRESS.md that the tie-break was used.
- **Gate 6.2 (event branch):** keep the event branch only if, on **tuning** with **Kitware RGB** labels, event-rule
  **precision ≥ 0.5 and recall ≥ 0.5**, **AND** the confirmed gate rule (`rewind/eval/gate.py`) passes on **dev
  `vehicle_event`** queries (event branch on vs off). Otherwise drop it. The Wed 28 Oct 13:00 hard stop still applies.
- **Step 7.4 (fusion weights):** choose the weights that **maximise mean Recall@10 on tuning specific-event
  queries**; ties broken by **lower median seconds-watched**; grid **{0, 0.25, 0.5, 1, 2}** per non-appearance branch
  (appearance fixed at 1.0).
- **Rule changes are pre-registered (Ashutosh, Mon 26 Oct, after Gate 4.4), for every remaining gate (5.1, 6.2, 7.4,
  8.1):** any change to a decision rule must be written in PROGRESS.md (and committed) **before** that gate's results
  are computed. **No further post-hoc reinterpretations.** If a rule turns out not to decide, or decides something
  that looks wrong, report it as it stands and let Ashutosh decide; the record must show which rule was
  pre-registered and that the decision departed from it.
  - Consequence: the Gate 4.4 tie-break reading (families conflict → headline metric decides) was a one-off post-hoc
    decision. It is **not** carried forward. Gates 5.1 / 6.2 / 8.1 use the confirmed paired rule
    (`rewind/eval/gate.py`) as written, plus the tie-break above for equal-complexity variants only, unless a change
    is recorded here before the gate runs.

## Step 8.1 ablation list (dev; CLAUDE.md step 8.1 + additions)

From CLAUDE.md 8.1: appearance-only; + action (if enabled); + events (if enabled); full fusion; crop_source none /
motion / detection_then_motion; HNSW vs brute force (metrics + latency); all broken down by qtype. CLAUDE.md's
"mean-centering on / off" is replaced by the centering comparison below.
Additions:
- **Per-camera-per-type mean-centering (Ashutosh, Mon 26 Oct; pre-registered before any 8.1 result):** each vector
  centered by the mean of its own camera AND type (frame views and crops get separate means), then re-normalised; no
  labels. Why: at Gate 4.4 centering lost once crops were in (3/2/12 with motion crops), plausibly because one camera
  mean over frames and crops together suits neither.
  - **Exactly one comparison:** per-camera-per-type centering (complex) **vs the current config, no centering**
    (simple), at the config's other values, under the confirmed paired rule (`rewind/eval/gate.py`) as written.
    Adopt per-type centering only if that rule adopts it.
  - **Single per-camera centering is not re-tested** (rejected at Gate 4.4).
  - Needs code before 8.1: means per (camera, type) in `scripts/collect_appearance.py` and the centering mode in
    `scripts/appearance_search.py`, with a hand-computed test.

## Engine copied into `engine/` (Mon 26 Oct, before step 7.1)

- Source only, unmodified, from `semantic_image_retrieval_using_hnsw_indexing` at upstream commit `6f84dcb`; what was
  and was not copied is in `engine/UPSTREAM.md` (no Flickr data, results, Python pipeline, web page or build output).
- Checked: files byte-identical to upstream; out-of-tree Release build (scratchpad) builds all 6 targets with no
  warnings; `ctest` 1/1 passed (`sir_tests`; the two checks that need Flickr data / its venv skip themselves).
- **For step 7.1:** its tests use a small home-made harness (`tests/tests.h`), not GoogleTest as CLAUDE.md 7.1 says;
  the engine has a Python text-encoder subprocess (`src/text_encoder.cpp`) that Rewind replaces with pybind11.

## Step 5.1 — PRE-REGISTRATION (Mon 26 Oct, written and committed before any 5.1 result)

Scope trimmed by Ashutosh (Mon 26 Oct). Everything below is fixed **before** any window vector, query score or
metric of step 5.1 is computed. No post-hoc reinterpretation (rule above).

**Candidates (two, not three)**
- **(a) Mean-pooled CLIP, the simple variant.** Built from the existing step 4.2 appearance vectors
  (ViT-B-16 / `laion2b_s34b_b88k`, frame view `pad` = letterboxed whole frame, raw, no centering: the Gate 4.4
  config). No new embedding run. Window vector = mean of the 1 fps vectors at the sample times inside the window
  (4 per 4 s window), L2-normalised again. A sample time whose frame dedup dropped (step 4.1) takes the vector of
  the last kept frame at or before it: dedup judged that frame essentially unchanged, and the max gap of 10 s
  guarantees one exists. **Recorded approximation:** 4 frames at 1 fps (some carried forward), not the 8 frames
  every 0.5 s that (b) sees. So (a) sees less motion detail than a true 8-frame mean-pooled CLIP would. This makes
  the bar slightly easier for (b) to clear, not harder.
- **(b) LanguageBind-Video, the complex variant.** Checkpoint **`LanguageBind/LanguageBind_Video_V1.5_FT`**
  (ViT-L/14, 8 frames, temporal attention per layer, then frame averaging; **projection dim 768**; MIT code and
  weights, training data VIDAL-10M CC-BY-NC-4.0 to be cited; zero-shot MSR-VTT R@1 42.8 per the paper). Chosen
  over `LanguageBind_Video` (LoRA: needs peft), `_FT` (older) and `Huge_V1.5_FT` (H/14, 1024-d, 4.8 GB fp32: tight
  on 6 GB for +2 R@1 on web video). Sources: HF config.json and the repo (research notes from this session).
  VERIFY on load: projection dim and num_frames from the downloaded config.
  - Input: 8 frames per window at s, s + 0.5, ..., s + 3.5 s (decoded frame round(t * fps)). Each frame becomes
    the **same `pad` view as (a)**: whole frame letterboxed into a square (no centre crop: it lost 28 % of tuning
    events, open-issue item 4), resized to 224 (bicubic, antialias) and normalised with the OpenAI CLIP mean/std
    (LanguageBind's own constants). LanguageBind's own transform is **not** used: it centre-crops and applies a
    random horizontal flip at inference (`processing_video.py`, decord branch). Honest caveat: the model was
    trained on centre crops, and letterboxed frames are a domain shift for it. Both candidates get the same view,
    so the comparison is between the models, not the crops.
  - fp16 inference. Embeddings L2-normalised by us (`get_image_features` / `get_text_features` are not normalised).
  - Text: the query text verbatim, CLIP tokenizer, max 77 tokens. No prompt templates for either candidate.
- **X-CLIP dropped (Ashutosh).** Reason: it is Kinetics-400 trained (a narrow action vocabulary, human-centred
  clips) and must be cut down to plain text embeddings to stay a dual encoder (its full forward conditions the text
  on the video). That removes the part of the model that made it strong, so it is unlikely to beat (a), and the
  1-hour setup is better spent on one serious candidate.

**Windows (identical for both)**: `rewind/embed/windows.py`. Per clip, [s, s + 4) for s = 0, 2, 4, ... while
s + 4 ≤ duration (action.window_s 4, stride_s 2). A window's search time is its centre s + 2. All 42 dev clips
(3.50 h, 6 cameras incl. IR G474). Deviation from CLAUDE.md "~2 h of dev footage": dev ground truth is split-wide,
so a subset would score correct answers outside it as misses.

**Search**: the top `search.k_per_index` (200) windows by cosine → clips with `rewind/search/core.py` unchanged
(merge 5 s, cap 60 s, mean top-3, ±5 s context). Specific queries search every camera; category queries search RGB
windows only (as at 4.4). NumPy brute force.

**Queries**: frozen dev queries with qtype `action` or `vehicle_event`: 5 category (c01–c05) + 8 specific
(s05–s12) = **13**. Appearance-qtype queries are not used.

**Reported**: per candidate, median and mean seconds watched, R@1/5/10 (specific), mAP@10/50 (category); per-query
table with the better candidate; wins / ties / losses; the paired bootstrap 95 % CI of the mean difference per
family (specific seconds, category AP@50). Cost: (b) peak VRAM (`torch.cuda.max_memory_allocated`) and windows/s
(end to end incl. decoding, and model only). (a) has no new GPU run; its cost is the step 4.2 run (shared with the
appearance branch) plus CPU pooling.

**Decision rule (GATE 5.1)**: `rewind/eval/gate.py` `paired_decision(simple=(a), complex=(b))` **as written**
(specific queries by seconds watched with a miss in both = tie; category by AP@50; adopt (b) only if the mean is
better (specific better and category not worse, or the reverse) AND wins > losses). The rule always returns
adopt / not adopt, so it decides by construction; there is no near-tie clause.
- adopt → `action.enabled: true`, `action.model: LanguageBind_Video_V1.5_FT`, then step 5.2 (full indexing).
- not adopt → **drop the action branch** (`action.enabled: false`). Mean-pooled CLIP is the bar, not a branch
  (CLAUDE.md 1.7). Phase 5 ends.

**Stop conditions (no gate result computed)**
- Setup timebox: **1 hour**, from install approval to the first correct 8-frame embedding. Install or load failure,
  or no fit in 6 GB VRAM (out of memory at batch size 1 window), means LanguageBind is dropped, so the action
  branch is dropped. Record it and stop the phase.
- **Implementation sanity check, before any dev query is scored** (GitHub issue #51 reports that FT checkpoints
  gave bad results for one user; UNVERIFIED): (1) the weights load with `strict=True` (no missing or unexpected
  keys); (2) fp16 vs fp32 window embeddings agree (cosine ≥ 0.99 on 16 tuning windows); (3) on 2 tuning clips with
  vehicle activity, the AUC of "a vehicle starts moving" / "a vehicle stops" prompts at telling windows that overlap
  an annotated `vehicle_starts` / `vehicle_stops` from those that don't, shown next to (a)'s AUC. Tuning only, for
  information. If (1) or (2) fails, or (3) is ≤ 0.55 for (b), treat it as an implementation bug: debug it within
  the timebox, never swap models silently. If it is not fixed, stop and report to Ashutosh.

**If adopted (step 5.2)**: `scripts/embed_action.py` resumable over all clips (dev clips reused, same settings
hash), test on 10 min, then a background run with log and PID under `results/logs/`. **Note for step 7.1:**
LanguageBind vectors are **768-d** (appearance is 512-d), so the engine must take the vector dimension per named
index from the data or the config, not as a constant.

## Step 5.1 — setup, sanity check: (3) FAILED → stopped for Ashutosh (Mon 26 Oct, 11:22 → 11:45)

**Built** (no dev result computed yet):
- `rewind/embed/windows.py` (windows, 8-frame indices, mean-pooled CLIP windows; hand-computed tests in
  `tests/test_windows.py`), `rewind/eval/metrics.auc` (+ hand-computed test).
- LanguageBind: model code vendored in `rewind/embed/languagebind_vendor/` (MIT, upstream commit `7070c53`) with the
  transformers **v4.30.2** CLIP attention/MLP/text-embedding code (Apache-2.0), because newer transformers removed
  `_expand_mask` and would silently drop the text tower's causal mask. Changes listed in its `NOTICE.md`. Wrapper
  `rewind/embed/languagebind.py`. `tests/test_languagebind.py` (tiny random model, CPU): text tower is causal
  (**mutation-checked:** swapping in transformers 5.x `CLIPAttention` makes it fail), and windows in a batch are
  embedded independently.
- `scripts/embed_action.py` (resumable per clip, `--sanity`, `--encode-queries`), `scripts/action_bakeoff.py`
  (Gate 5.1 scoring), `scripts/check_languagebind.py` (debugging), `scripts/fetch_languagebind.sh` (resumable
  curl + sha256). `scripts/appearance_search.query_rows` got an optional qtype filter.
- **Dependencies (approved by Ashutosh):** `transformers` 5.20.2 and `einops` 0.8.2 in the main env (no torch change).
  Weights `LanguageBind_Video_V1.5_FT` at HF commit `5d53aab`, 2.11 GB, sha256 verified (`artifacts/weights/`). The
  first download (huggingface_hub) stalled and was stopped; its partial file was in
  `artifacts/weights/LanguageBind_Video_V1.5_FT/.cache/`. (Correction, 26 Oct: it was 805 MB, not ~260 MB as first
  written; deleted with Ashutosh's approval, see "Third-party code and licenses".)
- Config: `action.languagebind` (weights dir, revision, sha256, license), `action.batch_windows: 4`,
  `action.sanity` (the pre-registered thresholds).

**Measured:** config verified: projection dim **768**, 8 frames, temporal attention, `lora_r 0`. Load 8 s; VRAM 1.0 GB
loaded, **peak 1.45 GB** at 4 windows/batch (fits 6 GB easily). Model only **~4.2 windows/s** (1, 4 and 8 windows per
batch are the same speed); end to end ~3.3 windows/s (149 windows / 45 s per 5-min clip incl. decoding). Projected:
dev 3.5 h ≈ 32 min, all 25.0 h ≈ 3.8 h (corrected: the footage is 25.0 h over 300 clips; 20.2 h was the appearance run without the 58 reused tuning clips). Measured later: 4.8 windows/s end to end on dev.

**Pre-registered sanity check** (`uv run python -m scripts.embed_action --sanity`, log
`results/logs/action_sanity.log`; tuning clips G436 hospital 14:55 (39 vehicle start/stop events) and G424 school
14:50 (36)):
1. Weights: **ok, with one deviation from the literal wording "strict=True".** Strict loading failed on exactly one
   key, `text_model.embeddings.position_ids`: the checkpoint doesn't store this constant 0..76 index buffer. Every
   learned tensor matched and there were no unexpected keys. Fix: load non-strict and accept only that missing key,
   after checking that the buffer equals 0..76. Any other mismatch is still an error.
2. fp16 vs fp32 over 16 windows: cosine min **0.9999** → ok.
3. AUC of prompt vs annotated windows (298 windows; (a) = mean-pooled CLIP on the same windows):

   | prompt | positive windows | AUC (b) LanguageBind | AUC (a) CLIP |
   |---|---|---|---|
   | "a vehicle starts moving" | 112 | 0.575 | 0.585 |
   | "a vehicle stops" | 96 | **0.481** | 0.559 |

   **(b) ≤ 0.55 on "a vehicle stops" → check (3) FAILED.** Per the pre-registration this is treated as a possible
   implementation bug, so I debugged it:
   - **Known-answer test** (`scripts.check_languagebind --known-answer`, log `results/logs/check_languagebind.log`):
     LanguageBind's own sample videos and captions from its `inference.py`. Each video matches its own caption with
     both preprocessings: cosine 0.28 / 0.25 on the right caption vs 0.00 / 0.02 on the wrong one (pad); softmax
     1.0 / 0.0. The implementation reproduces the model's intended behaviour.
   - **Centre crop instead of pad** (diagnostic, tuning only, `--centre-crop-auc`): AUC 0.600 (starts) / 0.512
     (stops). It still fails, so the letterbox is not the cause.
   - **Conclusion: no implementation bug found.** The weak AUC looks like the model's real behaviour on these CCTV
     scenes and generic prompts. Note that CLIP is also weak here (0.56–0.59): in these busy scenes, 32–38 % of
     windows contain a labelled start/stop somewhere in the frame.
- The pre-registration says: "If it is not fixed, stop and report to Ashutosh." **Stopped here; no dev window or
  query has been embedded or scored.** Setup timebox: 11:22 → 11:45 used (23 of 60 min).

**Decision (Ashutosh, Mon 26 Oct, ~11:50, before any dev result): run the Gate 5.1 dev bake-off.** Check (3) is
counted as answered: it was meant to catch implementation bugs, and debugging found none. This is Ashutosh's explicit
departure from the pre-registered "stop and report" clause, recorded here before the dev run. Nothing else changes:
same candidates, windows, queries, metrics and paired rule as pre-registered. Caveat for the record: check (3) only
probed vehicle prompts; the action-qtype queries were not covered by it.

## Step 5.1 — GATE 5.1: ADOPT LanguageBind (pre-registered rule, Mon 26 Oct, ~12:20)

- Dev embedding: `scripts.embed_action --split dev`, 42 clips / 3.50 h → **6,256 windows in 21.6 min** (4.8 windows/s
  end to end, 5.9 model only, peak VRAM 1,449 MiB); outputs checked: fp16, 768-d, finite, unit norm, one settings
  hash `c755729736ca4da5`, windows = `window_starts()`. Query texts: `--encode-queries dev`.
- Bake-off: `uv run python -m scripts.action_bakeoff --model LanguageBind_Video_V1.5_FT`, report
  `results/reports/action_5.1.md`, 2 experiments rows. **Honest note:** the rows carry git `08bee44-dirty`. The
  uncommitted diff was only a new check in `action_bakeoff.TextLookup` (text vectors must have the window vectors'
  settings hash; it passed), committed right after. The run is deterministic.

  | 13 dev queries (5 category, 8 specific) | (a) mean-pooled CLIP | (b) LanguageBind |
  |---|---|---|
  | specific median s watched | 6,902 | 6,948 |
  | specific mean s | 6,905 | 6,831 |
  | specific R@10 | 0.000 | 0.375 |
  | category mAP@10 / mAP@50 | 0.018 / 0.034 | 0.279 / 0.102 |
  | by qtype, specific R@10 (n = 4 each): action / vehicle_event | 0 / 0 | 0.50 / 0.25 |

- **Paired rule (`gate.py`, as pre-registered): wins / ties / losses 7 / 3 / 3**. Specific mean diff +74 s, CI
  [−4,750, +4,890] (n = 8). Category AP@50 mean diff +0.068, CI [−0.038, +0.177] (n = 5). Mean better in both
  families → **adopt = True. The rule decides by construction, so no stop was needed.**
- **Decision: action branch kept with `LanguageBind_Video_V1.5_FT`** (`action.enabled: true`). Honest reading:
  - Both CIs include 0. On specific queries the two are tied in substance: 3 of 8 are misses for both, and
    medians are 6,902 vs 6,948 s.
  - The gain is mainly at the top of the list (R@10 0 → 0.375; category mAP@10 0.018 → 0.279) and on the two
    vehicle category queries (AP@50 0.009 / 0.001 → 0.188 / 0.231).
  - It lost the person_puts_down_object category query (0.127 → 0.000) and the IR action query.
  - Dev is small (13 queries), so this is a noisy decision, recorded as such.
- Sanity check (3) had failed on tuning ("a vehicle stops" AUC 0.48). On dev the vehicle-stop category query went the
  other way (AP@50 0.001 → 0.231). The two tests measure different things (window-level AUC on 2 busy clips vs.
  ranked clips across the split), so neither result overrides the other.
- **Note for step 7.1: LanguageBind vectors are 768-d** (appearance 512-d). The engine must take the dimension per
  named index from the data or config, not as a constant. Step 7.4 fusion weight for `action` is still TBD (tuning).

## Step 5.2 — full action indexing: LAUNCHED (Mon 26 Oct, 12:11)

- `nohup uv run python -m scripts.embed_action --pid-file results/logs/embed_action.pid > results/logs/embed_action.log 2>&1 &`
  at commit `8e37f28` (clean). Same settings hash `c755729736ca4da5`: the 42 dev clips are reused; **258 clips /
  21.5 h to do, first ETA 134 min (≈ 14:25)**. Peak VRAM 1.45 GB. Output ≈ 37.5k windows × 768 × fp16 ≈ 58 MB.
- The script test the step asks for ("test on 10 minutes") is covered by the dev run: the same script on 42 clips /
  3.5 h, with every output checked (see GATE 5.1).
- Weights license re-checked in the downloaded model card: `license: mit`.
- **To do when it finishes:** check every clip has `.npy` + `.parquet` + `.done.json` with one settings hash, rows =
  windows, fp16 / 768-d / finite / unit norm, and no `.tmp` files; check the experiments row; record totals here.

## Step 5.2 — full action indexing: DONE and verified (Mon 26 Oct, closed 14:30)

- **Interrupted once by a power cut.** Run 1 (12:11 → 13:49) did 178 of its 258 clips. On battery it could not have
  finished: the battery holds about 40 Wh, the machine drew about 49 W, and the GPU was capped at about 30 W. So
  Ashutosh stopped it (`kill`, 13:49) right after clip 178 had been written: 0 `.tmp` files, 220 markers. Run 2
  (resume on mains, 13:51 → 14:29) skipped the 220 done clips and did the last 80. Same settings hash
  `c755729736ca4da5` in both runs (the resume refuses other settings).
- **Verification:** `uv run python -m scripts.check_action_pass` → **PASS: every check**. It checks:
  - all 300 clips have `.npy` + `.parquet` + `.done.json`, with no `.tmp` files left;
  - one settings hash;
  - rows = marker windows = `window_starts()`, and `t_s` = window centre;
  - video / camera / modality match the manifest;
  - vectors are fp16, 768-d, finite, with |norm − 1| <= 2e-3.

  | part | clips | windows |
  |---|---|---|
  | step 5.1 dev run (reused) | 42 | 6,256 |
  | 5.2 run 1 (stopped at the power cut) | 178 | 26,519 |
  | 5.2 run 2 (resume) | 80 | 11,907 |
  | **total** | **300** | **44,682** |

  25.01 h of video; 66.9 MiB on disk; peak VRAM 1,449 MiB; sum of per-clip wall time 158.3 min. Per camera: G330 51,
  G336 51, G474 51, G424 50, G340 49, G436 48 clips.
- **Speed:** 32 s per clip at first. It rose to 45–47 s while the track pass ran, and again in the last ~20 min of
  run 1, with the GPU at a power cap. Run 2 averaged 28.8 s per clip (5.17 windows/s, 10.4x realtime).
- **experiments.csv: honest note.** Run 1 was stopped before its end, so it logged **no row**. Run 2's row covers
  only its 80 clips / 11,907 windows. Its hashes are git `e93bb08-dirty` (only the then-uncommitted
  `scripts/check_action_pass.py` was untracked) and config `bbd7237…`: the config had changed since `8e37f28` in the
  `events` section only, and the action settings hash is unchanged. The full-footage totals are the table above.
- **Count reconciliation:** my mid-run "124 of 258" was wrong. The checker had counted 124 of **300** clips, which
  includes the 42 dev clips, i.e. 82 of 258 in run 1's own numbering (its log showed ~[79–82/258] then). "95 of 258"
  was run 1's own counter. The marker times confirm the split 42 / 178 / 80.
- New: `scripts/check_action_pass.py` (read-only output checker; reusable after any re-run).

## Phase 6 — PRE-REGISTRATION (Mon 26 Oct, ~12:45; written and committed before any track, rule output or metric)

Decided by Ashutosh in planning (Mon 26 Oct): **tracker** = reference ByteTrack vendored (MIT), **scipy** the only new
dependency (approved); **Gate 6.2 P/R applied per type**; **dev ON/OFF = event clips first**. Everything below is
fixed now; no post-hoc reinterpretation (rule above). Tuning is free, but the selection rule is fixed here anyway.

**Tracker (step 6.1; reference defaults, not tuned)**
- Code: `FoundationVision/ByteTrack` (formerly `ifzhang/ByteTrack`) at commit `d1bf019`, `yolox/tracker/`
  (`byte_tracker.py`, `basetrack.py`, `kalman_filter.py`, `matching.py`), LICENSE MIT (c) 2021 Yifu Zhang. Vendored in
  `rewind/track/bytetrack_vendor/` with changes listed in its NOTICE.md (numpy 2: `np.float` -> `float`; `lap.lapjv`
  -> `scipy.optimize.linear_sum_assignment` on the same extended cost matrix `lap` builds for `cost_limit`;
  `cython_bbox` -> NumPy IoU with the same +1 pixel convention; no torch; boxes in original pixels (no rescale);
  the global id counter -> one counter per tracker).
- Settings = the reference demo defaults: `track_thresh 0.5` (new tracks need score > 0.6), `match_thresh 0.8`,
  `track_buffer 30` at a nominal 30 fps, i.e. lost tracks kept **1 s** (`frame_rate` = `detection.fps` = 5 -> 5
  frames), `mot20 false`. Second association uses detections with 0.1 < score <= 0.5 (ours start at 0.3).
- Input: detections of `events.vehicle_classes` (car, truck, bus, motorcycle) pooled **class-agnostically** (YOLOX
  flips car/truck on one vehicle; a per-class tracker would cut the track there). Track class = majority class.
- **One tracker per clip, created fresh** (id counter at 0). No state crosses clips or cameras; no appearance
  features; no re-identification. The stored `track` column is a within-clip index (0..n-1, renumbered by first
  appearance), documented as never an identity. **No track index reaches SQLite, reports or the UI.**
- Output rows: frames where the track was matched to a detection, with that **detection's** box and score (not the
  Kalman box). Run on all 300 clips (indexing, like detection; no labels read).

- **Added before any result (Mon 26 Oct, ~13:05, while writing `track_clip`):** pooling the vehicle classes needs a
  class-agnostic merge first. YOLOX's per-class NMS can box one vehicle as both car and truck, and two near-identical
  boxes would start two tracks (two "events" for one vehicle). Per frame, vehicle boxes with IoU >= `detection.nms_iou`
  (0.65) to a higher-score vehicle box are dropped. Not tuned. Only output seen at this point: one test clip's
  track count (G436 10:55, 78 tracks), no speed, rule or metric.

**Speed and rules (step 6.1)**
- Per track sample i at time t_i: the track's samples in [t_i - w/2, t_i + w/2] (w = `smooth_s`); speed = distance
  between the box centres of the first and last of them / their time difference. Undefined (NaN) if that time
  difference is < w/2. Unit `px_per_s` (original pixels) or `box_h_per_s` (divided by the median box height of those
  samples, so a distant car is not "slow" just because it is small).
- Label per sample: moving if speed >= `stop_speed`, stopped if < `stop_speed`, none if undefined. Runs = maximal
  consecutive samples with one label; a none sample ends a run. Run duration = last time - first time.
- `vehicle_stops` at the boundary between a moving run of duration >= `min_moving_s` and the stopped run right
  after it, of duration >= `min_stopped_s`. `vehicle_starts`: the reverse. Detected span = [last sample of the first
  run, first sample of the second run) (one sample interval); event time = first sample of the second run.
- Best box of a detected event: the track's highest-score box in [event time - min_moving_s, event time +
  min_stopped_s] (ties: larger area). Crop = that box padded like appearance crops (`appearance.crop_pad_frac`), saved
  as a JPEG under `artifacts/event_crops/` (it is also the result thumbnail; CLAUDE.md 1.4 allows thumbnails).

**Matching and evaluation (step 6.1)**
- `rewind.eval.metrics.match_events` (same camera and type, annotated span widened by `metrics.event_tolerance_s` =
  2.0 s, one-to-one, greedy by overlap).
- Ground truth: **Kitware** `vehicle_starts` / `vehicle_stops` (`label_source` kitware-eval or kitware-train) in
  **tuning** clips that carry Kitware labels (all RGB tuning clips, `empty_37` clips included: their detections
  count as false positives). Detected events are scored only in those clips. G330 (gym) is included (it should have
  none). G474 (IR): recall vs contrib labels reported for information only (contrib labels are incomplete).
- Reported per camera and per type (P, R, F1, counts), pooled per type, plus two **informational** checks that are
  not part of any rule: (i) a **time-shift control**: every detected span moved +30 s, circular within its clip, to
  show what dense ~3 s labels give by chance; (ii) **dev held-out** P/R per camera at the frozen setting (tuning P/R
  is measured on the data it was tuned on, so it is optimistic).

**Tuning grid (tuning split only; 144 settings)**
- `speed_unit` in {px_per_s, box_h_per_s}; `stop_speed` in {5, 10, 20, 40} px/s or {0.05, 0.1, 0.2, 0.4} box-h/s;
  `min_stopped_s` in {0.6, 1, 2}; `min_moving_s` in {0.6, 1, 2}; `smooth_s` in {0.6, 1.0}.
- **Selection:** maximise the mean over the two types of pooled F1 (tuning RGB cameras with Kitware labels); ties ->
  first in grid order (unit, speed, min_stopped, min_moving, smooth, each ascending as listed). One setting shared by
  both types. If the chosen value of a numeric parameter is at a grid edge, the grid is extended one step beyond
  that edge (x2 or /2 for speeds; +1 s or -0.2 s for durations; smooth 0.4 / 1.6) once, and the selection rerun.
- Config: `events.stop_speed_px_per_s` is replaced by `events.stop_speed` + `events.speed_unit` (+ `smooth_s`,
  tracker settings, `types`).

**GATE 6.2 — operational form (extends the pre-agreed rule above)**
- **Part 1 (tuning, per type):** a type passes if its pooled tuning P >= 0.5 **and** R >= 0.5 at the selected
  setting. `events.types` = the passing types. If none passes, the event branch is dropped (no dev run needed).
- **Part 2 (dev):** OFF = appearance-only search at the Gate 4.4 config (`scripts.appearance_search.Index`, pad /
  motion / raw). ON = for a query whose text contains an intent keyword (`events.intent_keywords`, **unchanged from
  CLAUDE.md 1.9**) of an enabled type: the detected events of the matching enabled type(s) in the split (RGB only for
  category queries, like appearance), each a hit at its event time scored by cosine(CLIP crop vector, CLIP query
  vector), top `search.k_per_index`, grouped by `rewind/search/core.group_hits` (same SearchConfig) -> these clips
  first, then the OFF list. A query without intent gets the OFF list (a tie).
- Intent matching: lowercase; a keyword matches as a whole word or whole phrase (`\b...\b`); no stemming.
- Queries: the 6 dev `vehicle_event` queries (dev-c01, dev-c02, dev-s09, dev-s10, dev-s11, dev-s12), scored as in
  Gates 4.4 / 5.1; decision by `rewind/eval/gate.py` `paired_decision` as written (OFF simple, ON complex). **Keep the
  event branch iff Part 1 leaves >= 1 type AND Part 2 adopts ON.** Otherwise drop (`events.enabled: false`).
- **Recorded before any result:** the dev query texts have been read (by Claude, while planning). With the
  unchanged keyword list, dev-s11 ("A car pulling away on the far road.") matches no keyword ("pulls away" is listed,
  "pulling away" is not), so it is a tie by construction; the list is **not** edited after reading dev queries. A
  query whose only intent type is disabled by Part 1 is also a tie.
- Reported outside the rule: intent matches over all 17 dev queries (false intents on non-vehicle queries).
- Hard stop Wed 28 Oct 13:00 (CLAUDE.md).

## Step 6.1 — tracking + stop/start rules: DONE (Mon 26 Oct, 12:45 → 13:00)

**Built**
- `rewind/track/bytetrack_vendor/`: reference ByteTrack tracker (FoundationVision/ByteTrack `d1bf019`, MIT), changes
  in its NOTICE.md. **Checked against the real code:** lap 0.4.0's `_lapjv.pyx` pads to 2N x 2N with `cost_limit`
  (not what I first assumed); the scipy replacement uses a smaller matrix with the same objective (each match changes
  the total by cost − thresh). `tests/test_bytetrack.py` checks it against brute force on 300 random matrices.
- `rewind/track/tracks.py` (tracker wrapper, cross-class merge, speed, rules, best box), `rewind/track/store.py`,
  `scripts/track_pass.py`, `scripts/event_sweep.py`; `rewind/eval/metrics.match_count_us` (integer-microsecond
  `match_events`, tested equal to it on 300 random cases incl. ties).
- Tests: `test_bytetrack.py` (10), `test_track_rules.py` (13), `test_event_sweep.py` (5), 2 new in
  `test_metrics.py`. **Mutation-checked:** 6 mutations of `tracks.py` (threshold `>=` → `>`, run adjacency, NaN rule,
  min duration, box matching, index order) each make a test fail (the index-order one only after the test was
  strengthened). Suite: **357 pass**.
- Dependency: `scipy` 1.17.1 (approved in planning).

**Track pass** (`uv run python -m scripts.track_pass`, background, CPU): 300 clips in **15.2 min**; 4,119,469 vehicle
detections → **8,083 tracks**, 3,263,689 track rows; `artifacts/tracks/` 47 MB; settings `4d324505831e47ad`. G330
(gym) has 0 vehicle detections. Honest note: the experiments rows of the track pass, the sweep and the dev check
carry `d999bb0-dirty`: the 6.1 code was not yet committed; it is committed unchanged in `phase6.1` (later edits
touched only `clip_events`' output columns and `embed_events`, not tracking or the rules).
Side effect seen: while tracking ran, step 5.2's GPU job slowed from 32 to 45 s per clip (GPU at a software power
cap; CPU load was only ~3.5/16). It went back to 32 s when tracking ended (laptop power sharing, likely).

**Sweep (tuning, pre-registered; `results/reports/events_6.1.md`, full grid `events_6.1_grid.csv`)**
- 47 Kitware tuning clips (3.9 h incl. G330), labels: 351 stops, 350 starts. 144 settings, then the one-time edge
  extension (min_stopped_s 0.4, min_moving_s 3.0, smooth_s 1.6 added) → 384 settings; 76 s.
- **Selected:** `box_h_per_s`, stop_speed 0.2 box-h/s, min_stopped_s 1.0, min_moving_s 2.0, smooth_s **1.6** (the
  extended edge: per the pre-registration, extended once only). Mean F1 0.487.

  | pooled (tuning) | detected | labelled | matched | P | R | time-shift control P / R |
  |---|---|---|---|---|---|---|
  | vehicle_stops | 272 | 351 | 156 | **0.574** | **0.444** | 0.188 / 0.145 |
  | vehicle_starts | 296 | 350 | 153 | **0.517** | **0.437** | 0.179 / 0.151 |

  Per camera (stops / starts, P–R): G336 0.49–0.43 / 0.43–0.44; G340 0.58–0.42 / 0.48–0.39; G424 0.55–0.46 /
  0.52–0.43; G436 0.74–0.46 / 0.69–0.47; G330 0 detections, 0 labels. IR G474 vs contrib (info): stops 1/5, starts 1/3.
- **No setting of all 384 reaches P >= 0.5 and R >= 0.5 for either type** (best min(P, R): stops 0.480, starts 0.466).
- **Dev held-out check (information only, `events_6.1_dev.md`):** stops P 0.583 / R 0.500 (21 of 42), starts P 0.500
  / R 0.475 (19 of 40); shift control 0.111 / 0.095 and 0.026 / 0.025. Small n (95 % interval on 21/42 ≈ 0.35–0.65).
- **Where recall is lost (scratch diagnostic, tuning, not part of any rule):** 87 % of the 701 labelled events have a
  track covering the labelled vehicle (IoU >= 0.3 on >= half its 5 fps label frames); the rule fires near (same type,
  ±2 s) only 54 % of those. So the loss is mostly the speed rule, not detection/tracking. Coverage is lowest for small
  vehicles (73 % for label boxes <= 34 px tall; G336 76–78 %).
- **👤 check replaced (rule 16; viewed by Claude, n = 10: 5 random detected stops + 5 starts on tuning, one frame each
  at the best box, labels overlaid):** 4 coincide with a label (SUV by the cones G424, car on the road G436, driveway
  start G436, a parking-row start G336). 4 are parked cars in crowded rows with no label (G340 x2, G424, G336), which
  looks like box-centre jitter when a passing vehicle occludes them; the G336 one is the same car and place as the
  labelled start 100 s later. 2 are small distant cars (G424, one by a stop sign) with no label: possibly real
  stop-and-go that is not labelled. A single frame can't show motion, so this is an impression, not a measurement.

## Step 6.2 — GATE 6.2: DROP the event branch (pre-registered rule, part 1; Mon 26 Oct, ~13:00)

- **Rule applied as written:** part 1 needs, per type, pooled tuning P >= 0.5 AND R >= 0.5 at the selected setting.
  vehicle_stops 0.574 / **0.444**, vehicle_starts 0.517 / **0.437** → **both fail** → `events.types: []` →
  "If none passes, the event branch is dropped (no dev run needed)". **The rule clearly decides; part 2 (dev queries)
  was not run.** The result does not depend on the selection objective: no grid setting passes either type.
- **Confirmed by Ashutosh (26 Oct): no override** — the branch stays dropped per the pre-registered rule; the dev held-out
  numbers (stops 0.58 / 0.50) do not change it.
- **Config:** `events.enabled: false`, `events.types: []`, the selected rule values kept for the record,
  `search.weights.events: 0` (step 7.4 tunes `action` only).
- Not run because the branch is dropped (written and unit-tested in this step; kept in the repo for the record):
  `scripts/build_vehicle_events.py` (SQLite `detected_events`), `scripts/embed_events.py` (crop + CLIP),
  `rewind/search/events.py` (intent + event clips), `scripts/event_gate.py` (part 2), `rewind/video.frames_at`.
  `query_rows` now passes the query row to `ranked(..., query=)` (the existing rankers ignore it; tests unchanged).
- What this means for the product: Rewind has **no evidence for "stopping" or "starting"** beyond what appearance and
  the action branch (LanguageBind) give. `vehicle_event` queries go through appearance + action only. On dev, the
  action branch's vehicle_event R@10 was 0.25 (Gate 5.1).

## Phase 6 summary (Mon 26 Oct)

- Within-camera ByteTrack (vendored, MIT, one tracker per clip, no identities, no cross-clip/camera links): 8,083
  tracks over 25 h in 15 min on CPU. Stop/start rules tuned on 701 Kitware tuning labels: best pooled P 0.57 / 0.52 at
  R 0.44 / 0.44, well above the time-shift control (~0.15–0.19) but below the pre-agreed 0.5 / 0.5 bar for both types,
  with no setting passing → **Gate 6.2: event branch dropped** (decided by the pre-registered rule, a day before the
  Wed 13:00 hard stop). Dev held-out (information): stops 0.58 / 0.50, starts 0.50 / 0.48 on ~40 labels each.
- Failure causes seen: the rule misses about half of the tracked labelled events; parked cars jitter under
  occlusion (false stops/starts); small distant vehicles are tracked less (73 %).
- Open issues / ideas (not scheduled): hysteresis (two thresholds) or a minimum displacement instead of a single
  speed threshold; bottom-centre instead of box centre (occlusion from the top/side); tracking across clip boundaries
  (now one tracker per 5-min clip).
- **Verify by hand (Ashutosh):** read `rewind/track/tracks.py` (~190 lines) and the NOTICE of the vendored tracker;
  if you disagree with dropping on the tuning numbers while dev stops reached 0.58 / 0.50, that would be a departure
  from the pre-registered rule and must be recorded as yours.

## Third-party code and licenses (for README credits) — housekeeping (Mon 26 Oct, ~12:30)

- **Partial download deleted** (Ashutosh approved): `artifacts/weights/LanguageBind_Video_V1.5_FT/.cache/` held an
  **805 MB** `.incomplete` file from the stalled huggingface_hub download plus lock/metadata files (PROGRESS had said
  ~260 MB). The complete weights are untouched: `pytorch_model.bin` sha256 re-checked = config `679550de…29d`.
- **Vendored code checked against upstream** (raw GitHub files fetched to the scratchpad, 26 Oct):
  - `rewind/embed/languagebind_vendor/LICENSE` byte-identical to LanguageBind `LICENSE` at `7070c53`;
    `LICENSE-transformers` byte-identical to transformers `LICENSE` at `v4.30.2`.
  - LanguageBind's own `.py` files have **no license header upstream**, so none was lost. `configuration_video.py`
    is byte-identical. `modeling_video.py` differs only by the two import changes in NOTICE.md; it now also starts
    with a 2-line provenance comment (source, commit, MIT, copyright, "modified, see NOTICE.md").
  - `clip_v4_30_2.py` keeps the original Apache-2.0 header. **Fixed:** its comment and NOTICE.md said "copied
    unchanged", but the import block was rewritten (absolute imports, config classes from `configuration_video`).
    Apache-2.0 §4(b) requires modified files to say so; both now do. Function and class bodies are unchanged (every
    line outside the import block and the provenance comment occurs in upstream `modeling_clip.py`).
  - `tests/test_languagebind.py`: 2 pass after the comment changes.
- **Credits list for the README (step 10.1):**

  | Component | Where | License | Source / version |
  |---|---|---|---|
  | LanguageBind video model code | `rewind/embed/languagebind_vendor/` (`LICENSE`, `NOTICE.md`) | MIT, (c) 2023 PKU-YUAN's Group | github.com/PKU-YuanGroup/LanguageBind @ `7070c53`; Zhu et al., ICLR 2024 |
  | transformers CLIP pieces | `rewind/embed/languagebind_vendor/clip_v4_30_2.py` (`LICENSE-transformers`) | Apache-2.0, (c) 2021 The OpenAI Team Authors and The HuggingFace Team | transformers `v4.30.2`, modified imports |
  | LanguageBind_Video_V1.5_FT weights | `artifacts/weights/` (not in git) | MIT (model card); training data VIDAL-10M is CC-BY-NC-4.0 | HF `LanguageBind/LanguageBind_Video_V1.5_FT` @ `5d53aab` |
  | YOLOX model code | `rewind/detect/yolox_vendor/` (`LICENSE`, `NOTICE.md`) | Apache-2.0, (c) Megvii Inc. | PyPI sdist yolox 0.3.0; Ge et al. 2021 |
  | YOLOX-s weights | `artifacts/weights/yolox_s.pth` | Apache-2.0 | YOLOX release `0.1.1rc0` |
  | CLIP ViT-B-16 weights `laion2b_s34b_b88k` | HF cache (not in git) | MIT (model card, verified at step 4.2 prep) | HF `laion/CLIP-ViT-B-16-laion2B-s34B-b88K` @ `7288da5a`; Radford et al. 2021 |
  | open_clip library | pip dependency | VERIFY at 10.1 (not vendored) | `open-clip-torch` 3.3.0 |
  | MEVA annotations | `data/annotations/meva-data-repo/` | CC-BY-4.0 (its LICENSE, step 0.3) | Kitware `meva-data-repo` @ `421841a`; Corona et al., WACV 2021 |
  | MEVA video | `data/raw/` | VERIFY at 10.1 (mevadata.org terms) | `s3://mevadata-public-01/` |
  | ByteTrack tracker | `rewind/track/bytetrack_vendor/` (`LICENSE`, `NOTICE.md`) | MIT, (c) 2021 Yifu Zhang | github.com/FoundationVision/ByteTrack @ `d1bf019`; Zhang et al., ECCV 2022. Used at step 6.1; the event branch was dropped at Gate 6.2 |
  | scipy | pip dependency (step 6.1) | BSD-3-Clause (VERIFY at 10.1) | 1.17.1 |

## Step 7.1 — engine inspection (read-only; Mon 26 Oct ~13:20, while 5.2 runs; nothing compiled or changed)

Source: `engine/` (upstream `6f84dcb`, unmodified; build verified earlier: 6 targets, no warnings, `ctest` 1/1).

**Build.** CMake >= 3.20, C++20, single-config Release by default with `-O3 -march=native -DNDEBUG` (forced into the
cache; benchmarks rely on it). `sir_core` static library = every `src/*.cpp` + `dsa/*.cpp` (GLOB); 4 apps
(`sir_cli`, `sir_benchmark`, `sir_build_index`, `sir_gate`); one test program `sir_tests` registered with ctest.
Warnings `-Wall -Wextra`. No pybind11, no install target, no third-party libraries (everything incl. JSON, SHA-256,
MT19937-64, arrays, heaps is hand-written). `-march=native` means a binary is not portable to another CPU (fine:
built on the machine that runs it; README must say "build locally").

**Real API (what Rewind will use).**
- `EmbeddingStore` (`data.h`): owns one row-major **float32** block `[rows x dim]`; **row i = id i**. Two ways in:
  `load(binPath, manifestPath, expectedDim, expectedModel, normTolerance)` (raw little-endian float32 file + JSON
  manifest: model, dim, rows, `l2_normalized`, dtype `"float32"` only, sha256 — all checked, then every row) or
  `setData(dim, FloatArray, normTolerance, sha256 = "")` (copies element by element from a `FloatArray`).
  `validate` rejects NaN/Inf/zero norm and **|norm − 1| > normTolerance (1e-5 in config)**.
- `RetrievalIndex` interface: `insert(id, embedding)`, `search(query, queryDim, k, CandidateArray& out)` → results
  best first as `{int id, float score}`, score = dot (`1 − dist`), ties by ascending id. `BruteForceIndex` (exact,
  `insertAll()`, O(N log k) with a TopK heap) and `HNSWIndex` (`setup(store, HnswParams)`, `insert`, `search`,
  `setEfSearch`; ef = max(efSearch, k); `IndexManager::build` inserts rows 0..N−1 in order).
- `IndexManager::save / load / validate / rebuild`: binary format v2 = header (magic, version, model, dim, store rows,
  nodes, M/efC/efS, seed, flags, **SHA-256 of the embedding file**, entry/maxLevel, RNG state) + per-node layers +
  SHA-256 checksum. Load rejects corruption and a stale header (model, dim, rows, embedding hash). Only the graph is
  saved; vectors live in the separate `.bin`. `save` refuses a store without a source SHA-256 (so `setData` callers
  must pass one). Nodes may be absent (level −1), so a partial graph is representable.
- `SearchEngine` + `QueryEncoder` / `TextEncoderProcess`: Flickr-specific text → caption-metadata path through a
  Python subprocess. **Not needed:** Rewind embeds text in Python and passes vectors (pybind11, step 7.3).
- `HnswParams` defaults: M 16, efConstruction 200, efSearch 80, seed 42, heuristic selection, keep pruned, protect
  last in-edge (config `engine/config/config.json`, Flickr paths and `ViT-B-32/openai`, `embedding_dim 512`).

**Data layout / hardcoded assumptions (what 7.1 must change or work around).**
- **Vector size is NOT a compile-time constant**: dim comes from the store (`store.dim()`), the HNSW reads
  `firstRow + id * dims`, the index header stores dim. The only 512 is `config.json`'s `embedding_dim` (the apps'
  expected dim). So 512-d appearance and 768-d action indexes are possible as separate stores/indexes.
- **float32 only** (`dtype` must be `"float32"`); our vectors are fp16 `.npy` → convert to float32 and
  **re-normalise in float32** before handing them over (fp16 rounding gives |norm − 1| ~1e-3, which the 1e-5 check
  rejects; `check_action_pass` allows 2e-3).
- **ids = row indices 0..N−1, int** — no external ids and no metadata: our int64 vector_id ↔ (video, t_s, type, box)
  mapping must be kept by the caller (Python metadata parquet) or added (7.1: "int64 ids mapped to metadata").
- **No filtering** (camera / time range): to be added in 7.1 (options: over-fetch + post-filter, per-camera indexes,
  or a filter predicate inside `searchLayer`; decide in 7.1 and test against filtered brute force).
- **No named indexes**: one store + one index per process/app; 7.1 adds "multiple named indexes".
- **32-bit int sizes — overflow risk for 7.2:** `EmbeddingStore::load` allocates `new float[m.rows * m.dim]` with an
  **int** multiply (overflows past 2^31 floats ≈ 4.19 M rows at 512-d, 2.8 M at 768-d); `FloatArray`/`IntArray`
  sizes and `setData`'s loop index are `int`. Real sizes are far below (appearance 175,039 x 512 = 89.6 M floats;
  action ~44.7 k x 768), but the **synthetic 5 M benchmark (step 7.2) would overflow** → fix to `long long` first
  (and 5 M x 512 float32 = 10.2 GB, close to the 16 GB RAM limit anyway).
- **Not thread-safe:** `HNSWIndex::search` reuses a mutable visited-epoch array. FastAPI must serialise calls to
  one index (a lock), or keep one index per worker.
- `dot()` is a plain scalar loop in one `.cpp` so every distance is bit-identical (strict FP: no reassociation, so
  `-O3` does not vectorise the sum). Fine at our sizes; a speed lever only if 7.2 needs it.
- Memory: the store holds float32 vectors (appearance ≈ 358 MB, action ≈ 137 MB); graph per node `IntArray` per layer.

**Test harness.** Home-made (`tests/tests.h`: `check(cond, msg)`; `test_main.cpp` runs 16 named test functions, catches
exceptions per test, prints failures, exits 1 on any failure). Tests use `randomUnitVectors(n, d, seed)` and
`setData`; HNSW tests cover graph invariants, exactness at efSearch = N, recall growth over an ef sweep on 5,000
random vectors (>= 0.99 at ef 120), determinism, level distribution, bad-argument rejection; persistence tests cover
save/load round trip and corruption. Two tests skip themselves without Flickr data / the Python venv. **CLAUDE.md 7.1
asks for GoogleTest:** recommendation — keep the existing harness (no new dependency, already wired into ctest, same
style for new tests) unless Ashutosh prefers GoogleTest (a new dependency → ask, rule 6).

**Not used by Rewind:** `dsa/member1_inverted_index.cpp` (caption hash table / inverted index), `sir_cli`,
`sir_gate`, `sir_benchmark` (Flickr protocol; its timing helpers `timeQueries`, `recallAtK`, `collectSystemInfo` are
reusable for 7.2), `src/text_encoder.cpp`.

**Open questions for 7.1 (decide there):** filter strategy; whether ids stay row indices with a Python-side map
(simplest) or become int64 inside the engine; GoogleTest vs existing harness.

## Findings for the report (collected for results/SUMMARY.md and the failure analysis, step 10.2)

- **F1 — Event branch: tracking is not the bottleneck; the stop/start rule is (Phase 6, tuning, 26 Oct).** Of 701
  labelled Kitware vehicle start/stop events on tuning, **87 %** have a ByteTrack track covering the labelled vehicle
  (IoU >= 0.3 on >= half its 5 fps label frames), but the selected rule fires near (same type, ±2 s) only **54 %** of
  those tracked events. Coverage drops for small vehicles (73 % for label boxes <= 34 px tall; G336 76–78 %).
  Of 10 detected events viewed by Claude, 4 matched a label, **4 were parked cars in crowded rows whose box centre
  jitters when a passing vehicle occludes them** (false stops/starts), and 2 were small distant cars (possibly real,
  unlabelled stop-and-go). Result: best tuning P/R 0.574/0.444 (stops) and 0.517/0.437 (starts), chance level
  (+30 s shift) ~0.15–0.19; no grid setting reaches 0.5/0.5 → branch dropped at Gate 6.2. Evidence:
  `results/reports/events_6.1.md`, `events_6.1_grid.csv`, `events_6.1_dev.md`; coverage numbers:
  **`scripts/event_coverage.py`** → `results/reports/events_6.1_coverage.md` (tuning only; thresholds in
  `events.coverage_check`; toy tests in `tests/test_event_coverage.py`). It reproduces the first (scratch) run: 701
  labels, 87 % tracked, fires near 54 % of tracked / 13 % of untracked, same size bins. One deliberate difference:
  "fires nearby" now uses strict positive overlap like `match_events` (the scratch used inclusive ends); only G340
  stops changed (47 % → 43 %). The report's git hash is `d5e443d-dirty` because the then-uncommitted
  `scripts/check_action_pass.py` (step 5.2) was in the tree; the coverage code ran as committed in `d5e443d`.

---

## Later (deferred, not in current scope)

- **Stop/start rule robust to box jitter (from Finding F1; Ashutosh, 26 Oct).** Smooth box jitter (e.g. a median or
  Kalman-smoothed box centre, bottom-centre instead of centre, hysteresis with two speed thresholds) and ignore
  near-static tracks (e.g. a minimum net displacement over the track before any stop/start can fire), so parked cars
  occluded by passing traffic stop producing false events. Re-tune on tuning only and re-run Gate 6.2 as
  pre-registered. Not scheduled: the branch stays dropped for the 30 Oct deliverable.
- **SigLIP ViT-B/16 (`webli`) as the appearance encoder.** On open_clip's published retrieval table it beats every
  CLIP B/16 candidate on text→image retrieval (COCO R@1 0.478 vs 0.423 for the best permissive CLIP, laion2b).
  **Deferred (Ashutosh, 25 Oct): scope stays CLIP ViT-B/16** — a different model family is a scope change (CLAUDE.md
  1.7), web benchmarks don't predict CCTV performance, and there is no time in the 30 Oct schedule to re-validate.
  License, fp16 VRAM and preprocessing not checked (VERIFY if revisited).
