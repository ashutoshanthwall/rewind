# PROGRESS — Rewind

Deadline: Fri 30 Oct 2026 · Current phase: **1 — Ingest (1.1 done); next: 1.2 — Playback proxies**

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
