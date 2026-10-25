# PROGRESS — Rewind

Deadline: Fri 30 Oct 2026 · Current phase: **2 — Evaluation set: done; open-issue pass done (8 items, see end); standing policy rule 16 added · Phase 3: 3.1 test ✔ · Gate 3.3 ✔ (YOLOX-s) · full pass running (results/logs/index_pass.log) · 3.2 after the run**

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

## Later (deferred, not in current scope)

- **SigLIP ViT-B/16 (`webli`) as the appearance encoder.** On open_clip's published retrieval table it beats every
  CLIP B/16 candidate on text→image retrieval (COCO R@1 0.478 vs 0.423 for the best permissive CLIP, laion2b).
  **Deferred (Ashutosh, 25 Oct): scope stays CLIP ViT-B/16** — a different model family is a scope change (CLAUDE.md
  1.7), web benchmarks don't predict CCTV performance, and there is no time in the 30 Oct schedule to re-validate.
  License, fp16 VRAM and preprocessing not checked (VERIFY if revisited).
