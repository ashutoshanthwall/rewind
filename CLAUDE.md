# REWIND — Semantic Video Search for CCTV (24-hour footage build)

> **Who reads this file**
> - **Claude Code** reads it at the start of every session (rename it to `CLAUDE.md` in the repo root). Part 1 is your standing instructions. Part 2 is the build plan. You only act on the step the user's prompt names.
> - **Ashutosh (the user)** uses it as the playbook. Every step has a **👤 You** box (what to do yourself and when) and a **💬 Prompt** box (what to paste into Claude Code).
>
> The submission deck `Rewind_SAARTHI26_Team_Neural_Nexus.pdf` is in `docs/` for background. **Where the deck and this file disagree, this file wins** (MEVA instead of VIRAT; appearance + detection + optional action and event branches instead of CLIP-only; within-camera tracking wording).

---

# PART 1 — STANDING INSTRUCTIONS FOR CLAUDE CODE

## 1.1 Working rules

1. **Only do the step the user's prompt names.** Never start the next step on your own.
2. For every step: say what you will do → do it → check it against the step's **Done when** → report results honestly → stop.
3. **🚦 Gates:** at a gate, present the evidence and wait. The user decides. *(Amended by rule 16: decide yourself when a pre-agreed rule clearly decides.)*
4. **Never fabricate results.** Every number you report must come from code you actually ran. If something was not run or failed, say so plainly.
5. **Unverified facts:** mark with `VERIFY:` in code comments and reports. Several facts here (MEVA file naming, annotation format, activity names, model APIs, licenses) are best-effort and must be checked against real files, real library source and real license texts.
6. **Ask first** before: any download over 1 GB, any deletion, anything touching `data/raw/`, adding a new dependency, or changing `configs/config.yaml` values outside the current step.
7. **The test split is sacred** (see 1.10). Nothing reads test data except step 8.2.
8. **No silent failure.** No `try/except` that returns zeros, empty lists or defaults to hide errors. Fail loudly with a clear message.
9. **No hardcoded tunables.** Every threshold, rate, weight, class list, keyword list and model name lives in `configs/config.yaml`.
10. **Tests are real.** No tests that only assert on mocks or trivially pass. Metric and rule code must be tested on toy cases with hand-computed answers.
11. **Every experiment** appends one row to `results/experiments.csv` (timestamp, git hash, config hash, split, metrics, notes).
12. **Long jobs** (download, transcoding, detection, embedding): write a resumable script with checkpoints and a progress log, then give the user the exact command to run in their own terminal. Do not run multi-hour jobs inside the session. *(Amended by rule 16: launch them yourself in the background.)*
13. **At the end of every step:** update `PROGRESS.md` (what was done, key numbers, decisions, open issues) and commit with message `phaseX.Y: <summary>`.
14. Prefer **simple, readable, maintainable** code. This is a solo project with a fixed deadline, and the user must be able to explain every file.
15. **Never claim an event** (a vehicle stopped, a person entered) unless the observations and the configured rule support it. Similarity is not proof.
16. **Minimise the user's manual work** (standing policy, 25 Oct). Where this conflicts with rules 3 or 12 or a 👤 box in Part 2, this rule wins.
    - **Measure instead of eyeballing.** Replace 👤 eyeball checks with measurements wherever annotations allow (e.g. detection recall vs annotated boxes; activity score during labelled events vs random windows). Claude may also view images and plots itself and must say so in reports ("viewed by Claude, n = …"). Ask the user only when no measurement is possible.
    - **Gates:** decide them using the pre-agreed decision rules in CLAUDE.md / PROGRESS.md, and record evidence, rule applied and decision in PROGRESS.md. Ask the user only if the rule does not clearly decide, or the decision involves licenses, deletions, or anything in `data/raw/`.
    - **Long jobs:** launch in the background with `nohup`, log to a file under `results/logs/`, write a PID file next to it, tell the user the log path. Don't poll in a loop; check the log when the user next messages or asks.
    - Rule 6 still holds: ask first before any download over 1 GB or any new dependency (and before deletions, `data/raw/`, out-of-step config changes).

## 1.2 What we are building

**One line:** Describe what you saw, get the clip.

**Problem.** After a theft, hit-and-run or missing-person case, investigators review CCTV by hand: fast-forwarding hours of footage, guessing timestamps from witness statements, using motion search (which flags everything on a busy scene), or splitting footage among people. It is slow and tiring, and key seconds get missed. They don't need every hour, only the right few seconds.

**Solution.** Rewind indexes multi-camera footage once, then answers plain-English queries ("a white car stops near the building") with ranked clips and a per-camera timeline. It is an **offline, retrospective search tool**, not a live-monitoring system.

Retrieval branches:
- **Appearance branch (core):** CLIP embeddings of full frames + object crops (from a detector, with motion-region crops as fallback) → "what was visible".
- **Action branch (gated):** a video-text model over short sliding windows → "what happened", kept only if it beats a mean-pooled CLIP baseline on dev.
- **Event branch (gated, time-boxed):** short within-camera tracks of detected vehicles + explicit rules for **vehicle stops / vehicle starts** → events single frames cannot show.
- **Search core:** the user's own **C++ HNSW engine**, extended for this project. Results from active branches are fused, grouped into clips with context, and shown in a web UI.

**Principles (part of the product, not optional):**
- **Evidence untouched:** originals read-only and SHA-256 fingerprinted; playback uses separate proxy copies.
- **No face recognition, no person re-identification, no cross-camera linking.** If the event branch is kept: object tracks are **short-lived and within one camera only**, used solely as motion evidence; track IDs are never treated as people or identities and are never linked across cameras.
- **Humans decide:** Rewind finds candidate clips; people judge them.
- **Never claim absence:** UI always shows coverage ("searched X h across Y cameras"), never a bare "no results".
- **Honest evaluation:** frozen held-out test set, real baselines, published failure cases.

## 1.3 Scope and definition of done (deadline: Fri 30 Oct 2026)

This project is complete when **all** of these are true:

1. **~24 hours** of MEVA footage (4–6 co-timed cameras, incl. ≥1 thermal IR) is ingested, hashed, proxied, motion-scored, detected and indexed.
2. The appearance branch works with detection crops (or motion crops if the detector failed its gate — documented).
3. The action branch is working **or** dropped by Gate 5.1 (documented). The event branch is working **or** dropped by Gate 6.2 (documented).
4. Queries run through the **C++ engine** via a Python API, with camera/time filters, fusion of active branches and clip grouping.
5. The **web UI** searches, plays clips, shows the timeline and the coverage line.
6. **Test-set results** exist from exactly one run: Rewind vs 4 baselines, dev ablations, RGB vs IR and query-type breakdowns, scaling benchmark, failure analysis.
7. A **fresh clone** runs from the README.
8. A **screen recording** of the working demo exists, plus a list of demo queries that work and one documented failure.

**Out of scope:** face recognition, person re-identification, cross-camera tracking, zone/line-crossing rules (need manual geometry per camera), live/streaming ingest and alerts, model fine-tuning, VLM reranking, FAISS or other ANN libraries (unless HNSW fails a required benchmark), caption-based retrieval (BLIP + sentence embeddings), any footage beyond the selected ~24 h.

## 1.4 Constraints

| Constraint | Value |
|---|---|
| Timeline | Sat 24 Oct → Fri 30 Oct 2026 |
| People | 1 (solo) |
| GPU | RTX 3050 Laptop, **6 GB VRAM** → fp16 inference only; no training of large models |
| CPU / RAM | Intel i7-13620H, 16 GB RAM → stream video; never load whole videos into memory |
| Disk | ~**134 GB free** on `/` (btrfs) for the whole project |
| Languages | Python (pipeline, models, eval, API) + C++20 (search engine) |

**Disk budget (estimates; confirm with real sizes):**

| Item | Approx. |
|---|---|
| MEVA originals, ~24 h | ~38 GB |
| Proxies (720p H.264) | ~10–15 GB |
| Detections + tracks (Parquet) | ~1–3 GB |
| Embeddings + indexes (fp16) | ~1–3 GB |
| Model weights + environment | ~10 GB |
| Headroom | ~60 GB |

**Never extract frames to disk as image files** (except the few query keyframes and result thumbnails). Decode on the fly.

## 1.5 Dataset: MEVA

**MEVA — Multiview Extended Video with Activities** (Corona et al., WACV 2021, arXiv 2012.00914).
- Long untrimmed continuous video, scripted + spontaneous background activity, many cameras with overlapping and non-overlapping views, **RGB and thermal IR**.
- Public release: ~328 h ground-camera video; annotations for 37 activity types (temporal extents + actor/prop boxes). The activity list is expected to include vehicle events such as starts/stops/turns and person entering/exiting through structures — **VERIFY exact names in the annotations**.
- License: CC-BY-4.0 (VERIFY against the MEVA site; cite the dataset in the README).
- Video: public AWS bucket, no account needed: `aws s3 ls --no-sign-request s3://mevadata-public-01/`
- Annotations: distributed separately from video (VERIFY: find the official source via mevadata.org; confirm format).

**Selection:** ~24 h total = **4–6 cameras over the same ~4–5 h window**, prioritising (1) annotation coverage, (2) co-timed cameras, (3) ≥1 thermal IR camera, (4) at least some cameras with vehicle traffic (needed for the event branch).
**Known limitation:** activities are scripted, so events are cleaner than real incidents.

## 1.6 Architecture

```
INDEXING (offline, once)
 MEVA .avi ─► ingest: SHA-256 + manifest (SQLite), originals read-only
          │      └─► ffmpeg proxy (H.264, 720p, faststart) for browser playback
          ├─► motion pass: OpenCV MOG2 (~5 fps, downscaled)
          │      └─► per-second activity score + motion boxes  (dedup, motion baseline, fallback crops)
          ├─► detection pass: detector (~5 fps) on people + vehicle classes [Gate 3.3]
          │      └─► boxes, classes, scores per frame
          ├─► appearance branch: 1 fps sample → motion-aware dedup (max gap 10 s)
          │      → CLIP ViT-B/16 fp16 on full frame + ≤4 crops (detections first, motion boxes fallback)
          │      → per-camera mean-centering → vectors + metadata
          ├─► action branch [Gate 5.1]: windows of 8 frames / ~4 s, stride 2 s
          │      → chosen video-text encoder → vectors + metadata
          └─► event branch [Gate 6.2]: ByteTrack on vehicle detections (within camera)
                 → speed over time → rules: vehicle_stops / vehicle_starts → event records
                 → CLIP embedding of each event's best vehicle crop
                           │
           C++ HNSW engine: named indexes, metadata ids, camera/time filters, save/load

QUERY
 text ─► CLIP text encoder ──► appearance index search ──────────────┐
      ├► video-model text encoder ──► action index search (if kept) ─┤
      └► event intent (keyword list in config) ─► event records whose │
         crop embedding matches the query (if kept) ─────────────────┤
                                                                     ▼
  per-branch z-scores → weighted fusion → merge hits ≤5 s apart → cap clip ≤60 s
  → rank by mean of top-3 hits → +5 s context
  → FastAPI → web UI: result cards (box highlight + why it matched), proxy playback,
    per-camera timeline, coverage line
```

| Decision | Reason |
|---|---|
| Detection crops as primary crops | MOG2 only sees moving things; parked cars and standing people get absorbed into the background. A detector finds people/vehicles whether or not they move, and labels each crop with a class. |
| Keep MOG2 | Cheap activity score for dedup, the motion baseline (current practice), and fallback crops if detection misses something. Complementary to detection, not replaced by it. |
| Event branch = tracks + explicit rules | Single-frame embeddings can't show "stopping". A track gives speed over time, so "vehicle stops" becomes a testable rule tied to MEVA labels. Only stop/start rules: they need no per-camera geometry. |
| Event intent via keyword list | Transparent, testable, tunable on the tuning split. Rewind never infers an event the rule didn't produce. |
| Action branch gated | Web benchmarks don't predict CCTV performance; must beat mean-pooled CLIP on dev. |
| Dual-encoder video models only | Video and text must embed independently so footage is indexed once. |
| Per-camera mean-centering | Static background dominates fixed-camera embeddings and compresses scores. Uses no labels. |
| Motion-aware dedup | Global embeddings barely change when something small moves; embedding-based dedup would drop the frames that matter. |
| z-score fusion | Scores from different branches aren't comparable. |
| Clip cap 60 s, top-3 mean | Prevents chain-merged giant clips; robust to one spurious frame. |
| pybind11 + FastAPI, one process | Solo: fewer moving parts. The C++ engine stays the search core. |
| HNSW, no FAISS | Reuse the existing engine; ~hundreds of thousands of vectors makes brute force noticeably slower. Must be **measured** (step 7.2). |

## 1.7 Models and libraries

| Role | Choice | Notes |
|---|---|---|
| Appearance | CLIP **ViT-B/16** via `open_clip`, fp16, 512-d | Record the exact pretrained tag in config. |
| Detection | **Option A:** Ultralytics YOLOv8 (n or s) — **AGPL-3.0**. **Option B:** YOLOX (s or m) — **Apache-2.0**. | Decided at Gate 3.3 on detection quality, speed and license. AGPL obliges the whole project to follow AGPL terms if distributed. VERIFY current license texts. COCO classes used: person, car, truck, bus, motorcycle, bicycle (+ backpack, handbag, suitcase if useful). |
| Tracking | **ByteTrack** (within camera) | VERIFY license of the chosen implementation (reference repo vs Ultralytics built-in). Tracks are motion evidence only. |
| Motion | OpenCV **MOG2** | Activity score, dedup, motion baseline, fallback crops. |
| Action candidate (a) — baseline | **Mean-pooled CLIP B/16** over each window | The bar every video model must beat. |
| Action candidate (b) | **LanguageBind-Video** (ViT-L/14) | Dual encoder. VERIFY install, fp16, VRAM on 6 GB. |
| Action candidate (c) | **X-CLIP** `microsoft/xclip-base-patch32` (HF) | Kinetics-400 trained. Its full forward conditions text on video, which breaks indexing → **plain text embeddings only** (VERIFY in HF `XCLIPModel` source). |

**Decision rules:**
- **Gate 3.3 (detector):** adopt if, on a 10-min smoke clip, it detects most visible people/vehicles (user eyeballs ~50 frames) and full-run time projects to fit overnight. Otherwise motion crops only.
- **Gate 5.1 (action):** adopt a video model only if it beats mean-pooled CLIP on action-type **dev** queries. Otherwise drop the action branch.
- **Gate 6.2 (event):** keep the event branch only if the stop/start rules reach reasonable precision/recall against MEVA vehicle start/stop annotations on the tuning split **and** it is working by **Wed 28 Oct, 13:00**. Otherwise drop it and document.

## 1.8 Repository layout

```
rewind/
├── CLAUDE.md               # this file
├── PROGRESS.md             # running log, updated at the end of every step
├── README.md               # setup, reproduce from scratch, licenses, dataset citation
├── environment.yml
├── configs/config.yaml     # all tunables
├── docs/                   # deck PDF, DEMO.md, notes
├── data/
│   ├── raw/                # MEVA originals — READ-ONLY
│   ├── proxy/              # playback copies
│   ├── annotations/
│   ├── selection.txt
│   └── rewind.sqlite       # manifest, annotated events, splits, detected events
├── artifacts/
│   ├── motion/             # activity scores + motion boxes
│   ├── detections/         # per-frame boxes (Parquet)
│   ├── tracks/             # within-camera tracks (Parquet)
│   ├── embeddings/         # fp16 .npy + metadata .parquet per branch
│   └── indexes/
├── rewind/                 # Python package
│   ├── ingest/ motion/ detect/ track/ embed/ search/ eval/ api/
├── engine/                 # C++20 HNSW + pybind11 bindings, CMake, GoogleTest
├── ui/                     # single-page web UI
├── scripts/                # long-running job entry points
├── eval/
│   ├── queries/            # frozen query files + HASHES
│   └── splits/
├── results/
│   ├── experiments.csv
│   ├── reports/
│   └── figures/
└── tests/                  # pytest
```

**Existing engine:** `github.com/ashutoshanthwall/semantic_image_retrieval_using_hnsw_indexing` — C++ HNSW over CLIP embeddings (Flickr30k: 31,013 images; top-10 recall 0.946 vs exact; 0.54 ms/query; 13.9× faster than brute force). Prior evidence, not a guarantee for CCTV. **Inspect its real API and build before designing bindings.**

## 1.9 Config (initial; tuned only on tuning/dev splits)

```yaml
sampling:
  appearance_fps: 1
  max_gap_s: 10
  dedup_motion_pixel_thresh: TBD    # step 4.1, tuning split
motion:
  fps: 5
  downscale_width: 640
  min_box_area_px: TBD
detection:
  enabled: TBD                      # Gate 3.3
  model: TBD                        # yolov8n/s or yolox_s/m
  license: TBD                      # recorded for README
  fps: 5
  imgsz: 640
  conf_thresh: 0.3
  classes: [person, car, truck, bus, motorcycle, bicycle]
appearance:
  model: ViT-B-16
  pretrained: TBD                   # exact open_clip tag
  max_crops_per_frame: 4
  crop_source: detection_then_motion   # detection_then_motion | motion | none
  crop_pad_frac: 0.2
  min_crop_px: 32
  mean_center: true
action:
  enabled: TBD                      # Gate 5.1
  model: TBD
  frames_per_window: 8
  window_s: 4
  stride_s: 2
events:
  enabled: TBD                      # Gate 6.2
  tracker: bytetrack
  vehicle_classes: [car, truck, bus, motorcycle]
  stop_speed_px_per_s: TBD          # tuned on tuning split
  min_stopped_s: TBD
  min_moving_s: TBD
  intent_keywords:
    vehicle_stops: [stop, stops, stopped, stopping, park, parks, parked, parking, pulls over, halts]
    vehicle_starts: [start, starts, starting, drives off, pulls away, leaves, departs]
search:
  k_per_index: 200
  ef_search: 200                    # must be >= k
  fusion: zscore_weighted
  weights: {appearance: 1.0, action: TBD, events: TBD}   # tuned on tuning split
  merge_gap_s: 5
  max_clip_s: 60
  clip_rank: mean_top3
  context_s: 5
```

## 1.10 Evaluation protocol (hard rules)

**Splits — by time block, never by event.** ~30-min blocks → **tuning 15% / dev 15% / test 70%**. All cameras in a block go to the same split (MEVA cameras overlap; one event seen by two cameras must not leak across splits).

**Queries — written by the user, before the search system exists, then frozen.**
- *Category* (~15–20): from activity types, e.g. "a person opens a car trunk". Many correct answers.
- *Specific-event* (~40): one event + appearance detail from one keyframe, e.g. "a person in a dark jacket carries a box toward the building". One correct answer.
- Include **vehicle start/stop events** among both kinds (needed to evaluate the event branch honestly).
- Each query is tagged with a type: `appearance`, `action`, or `vehicle_event`.
- Frozen files + SHA-256 in `eval/queries/HASHES`. Never edited after freezing.

**Metrics.**
- Category: **mAP@k**, **Recall@k** over all labelled instances.
- Specific-event: **Recall@1/5/10** (hit = clip temporally overlaps ground truth) and the **headline metric: seconds of video watched before the first correct clip** (sum of ranked clip durations incl. context; a miss in top N costs the full sequential review).
- Per-query table, median, worst case, bootstrap 95% CI.
- Event rules (if kept): precision/recall of detected vehicle stop/start events vs annotations (temporal overlap).
- Also: pipeline recall (events with ≥1 kept frame/window/track), end-to-end query latency, indexing throughput.
- Breakdowns: RGB vs IR; appearance vs action vs vehicle_event queries.

**Baselines.** Sequential viewing · random order (seeded) · witness time ±10 min · **motion-ranked segments**.

**Rules.** Tuning split: free. Dev: gates and ablations. **Test: one run, step 8.2, frozen config.** No rewording, tweaking or swapping after seeing test results.

## 1.11 Known limitations (report honestly)
Single frames can't show actions (mitigated by action/event branches) · small objects (mitigated by crops) · detector misses in thermal IR and at distance · tracks break under occlusion and at low fps; a stop rule can fire on traffic queues · thermal IR breaks colour queries · weak on attribute binding/counting/negation · scripted dataset · CLIP's documented demographic biases (Radford et al., 2021) — never present Rewind as identifying people · automation bias (coverage line) · SHA-256 proves integrity since import, not authenticity; legal admissibility out of scope · solo query writing (mitigated by freezing first).

---

# PART 2 — BUILD PLAN WITH PROMPTS

## 2.0 How Ashutosh runs each session

**👤 Daily routine**
1. Open the terminal in the repo, start Claude Code.
2. Paste **PROMPT S (session start)**.
3. For each step: switch to **Plan Mode** (Shift+Tab twice) → paste the step's 💬 prompt → read the plan; ask about anything unclear → switch Plan Mode off → tell it to proceed.
4. After it finishes: do the **👤 You** checks for that step, then paste **PROMPT R (review)**.
5. Run `git diff HEAD~1` yourself and scan for the red flags in 2.12.
6. Paste **PROMPT E (end of step)**.
7. **At the end of each phase:** type `/clear`, then paste **PROMPT S** to start the next phase with fresh context.

**Schedule at a glance**

| Day | Phases |
|---|---|
| Sat 24 | 0 Setup |
| Sun 25 | 1 Ingest · 2 Evaluation set |
| Mon 26 | 3 Motion + detection · 4 Appearance index |
| Tue 27 | 5 Action bake-off (AM) · 6 Tracking + event rules (PM) · action indexing overnight |
| Wed 28 | 6 finishes **by 13:00 or is dropped** · 7 Engine (PM) |
| Thu 29 | 7 API + fusion (AM) · 8 Evaluation (PM) |
| Fri 30 | 9 UI (AM) · 10 Packaging (PM) |

**Reusable prompts**

**PROMPT S — session start**
```
Read CLAUDE.md (Part 1 fully, and only the Part 2 section for the step I name next) and PROGRESS.md. Summarise in 5 lines: where the project stands, the last decision made, and any open issues. Do not change anything yet.
```

**PROMPT R — self-review after a step**
```
Review everything you changed in this step as a strict reviewer. Check: silent failures, hardcoded values that belong in config.yaml, tests that pass trivially, any access to the test split, off-by-one errors in timestamps/frames, track IDs treated as identities or linked across cameras, and anything marked VERIFY that you assumed instead of checked. List problems found, then fix them. Re-run the tests and report results.
```

**PROMPT E — end of step**
```
Update PROGRESS.md for this step: what was done, key numbers, decisions, open issues, and anything I must verify by hand. Commit with message "phaseX.Y: <summary>". Then stop.
```

**PROMPT X — explain it to me**
```
Explain <file or function> to me as if I must defend it to a technical judge: what it does, why it's designed this way, what alternatives exist, and how it could fail. Then ask me 3 questions to check I understood.
```

**PROMPT D — debugging**
```
<paste the error / describe the wrong behaviour>. Find the root cause before changing code. Show me your hypothesis and the evidence for it, then propose the smallest fix. Don't add try/except to hide it.
```

---

## PHASE 0 — Setup · Sat 24 Oct

### Step 0.1 — Repo, config, experiment log
- **Do:** layout from 1.8, `configs/config.yaml` (1.9), `results/experiments.csv` + logging helper, `PROGRESS.md`, `.gitignore` (exclude `data/`, `artifacts/`, model weights).
- **Why:** every number must trace back to code and settings.
- **Done when:** a dummy experiment writes a correct row; first commit exists.

**👤 You:** create `rewind/`, `git init`, put this file in as `CLAUDE.md` and the deck PDF in `docs/`. Start Claude Code there.
**💬 Prompt:**
```
Do Phase 0, step 0.1 from CLAUDE.md. Create the repository layout, config.yaml with the initial values from section 1.9, experiments.csv with a logging helper (timestamp, git hash, config hash, split, metrics, notes), PROGRESS.md and .gitignore. Write a pytest test proving the logger writes a correct row.
```

### Step 0.2 — Environment + GPU check
- **Do:** `environment.yml` (PyTorch CUDA build, open_clip, opencv-python, PyAV or decord, numpy, pandas, pyarrow, fastapi, uvicorn, pybind11, pytest); check ffmpeg, CMake, C++20 compiler, AWS CLI; GPU smoke test (CLIP B/16 fp16 embeds one image). Detector/tracker packages are added later at step 3.3, not now.
- **Why:** CUDA/driver problems are the biggest time sink; find them on day one.
- **Done when:** script prints embedding shape and GPU name.

**👤 You:** run the install commands yourself. If anything fails, use PROMPT D.
**💬 Prompt:**
```
Do step 0.2. Write environment.yml and scripts/gpu_check.py that loads open_clip ViT-B-16 on the GPU in fp16 and embeds one test image. Give me the exact commands to create the environment and check ffmpeg, cmake, a C++20 compiler and the AWS CLI. Don't run installs yourself. Don't add detector or tracker packages yet.
```

### Step 0.3 — Annotations first, then footage selection 🚦
- **Do:** get MEVA annotations; table of camera × time window × annotated events × modality × vehicle-event count; propose ~24 h (4–6 co-timed cameras, ≥1 IR, some vehicle traffic); write `data/selection.txt`.
- **Why:** unlabelled video can't be evaluated; co-timed cameras behave like one site; vehicle events are needed for the event branch.
- **Done when:** 🚦 **GATE 0.3** — you approve.

**👤 You:** check: ≈24 h, fits disk, IR included, several hundred annotated events, and a healthy number of vehicle start/stop events.
**💬 Prompt:**
```
Do step 0.3. First find and download the official MEVA annotations (tell me the source and confirm the format; mark anything uncertain VERIFY). List the exact activity type names, especially vehicle start/stop ones. Build a table of cameras x time windows with annotated event counts, vehicle start/stop counts, modality (RGB/IR) and estimated size. Propose about 24 hours: 4-6 cameras over the same 4-5 hour window, at least one thermal IR camera, some cameras with vehicle traffic, maximising annotation coverage. Write data/selection.txt. Stop at GATE 0.3.
```

### Step 0.4 — Download
- **Do:** resumable download from `data/selection.txt` into `data/raw/`, size checks, then read-only permissions.
- **Done when:** all files present and verified.

**👤 You:** run it yourself in a separate terminal (tmux if possible); continue with Phase 1 prep meanwhile.
**💬 Prompt:**
```
Do step 0.4. Write scripts/download.py: reads data/selection.txt, downloads each file with aws s3 cp --no-sign-request into data/raw/, skips completed files, logs progress, verifies sizes, then sets data/raw/ read-only. Give me the exact command; don't run it in this session.
```

**👤 End of Phase 0:** PROMPT E → `/clear`.

---

## PHASE 1 — Ingest · Sun 25 Oct (morning)

### Step 1.1 — Manifest
- **Do:** SQLite `videos`: path, sha256, camera_id, start_time, duration_s, fps, width, height, modality. Parse names (VERIFY scheme), cross-check with annotations and container metadata.
- **Why:** every stage maps (file, frame) → (camera, wall-clock time); hashes back the evidence claim.
- **Done when:** all files listed; re-hash matches; pytest covers filename parsing.

**👤 You:** open 2 files and confirm DB camera/time match filename and annotations.
**💬 Prompt:**
```
PROMPT S first. Do step 1.1. Build the ingest manifest in data/rewind.sqlite. Verify the filename scheme against real files and annotations, read fps/resolution/duration from the container, compute SHA-256. Add pytest tests for filename parsing and timestamp conversion. Report a summary per camera.
```

### Step 1.2 — Playback proxies
- **Do:** ffmpeg → H.264 MP4, 720p, `-movflags +faststart`; resumable.
- **Done when:** proxy plays and seeks in a browser; durations within 1 s.

**👤 You:** run it yourself; open one proxy in the browser and seek.
**💬 Prompt:**
```
Do step 1.2. Write scripts/make_proxies.py (ffmpeg, H.264, 720p, -movflags +faststart), resumable, recording proxy paths in the manifest and checking durations within 1 second. Test on one file and show me; give me the full-run command.
```

### Step 1.3 — Annotated events table
- **Do:** parse annotations → `events` (event_id, activity_type, camera_id, start_time, end_time, boxes).
- **Done when:** counts per type printed; 5 events spot-checked.

**👤 You:** play the proxy at 5 events' times (include 1–2 vehicle stop/start events) and confirm. **Most important manual check of the week.**
**💬 Prompt:**
```
Do step 1.3. Parse MEVA annotations into an events table restricted to the selected footage. Print counts per activity type and camera. Give me 5 random events (at least 2 vehicle start/stop) with camera, proxy file and in-file timestamps so I can check by eye.
```

### Step 1.4 — Throughput
- **Do:** time decoding 1 h (keep 1 fps and 5 fps), CLIP B/16 fp16 on 1,000 frames.
- **Done when:** `results/reports/throughput.md` with projected times.

**💬 Prompt:**
```
Do step 1.4. Measure decode throughput for one hour of video (sequential decode, keeping 1 fps and 5 fps) and CLIP ViT-B/16 fp16 throughput on 1000 frames at the best batch size for 6 GB VRAM. Write results/reports/throughput.md with projected total times for the full ~24 h.
```

**👤 End of Phase 1:** PROMPT E (no `/clear`; Phase 2 continues).

---

## PHASE 2 — Evaluation set · Sun 25 Oct (afternoon)

### Step 2.1 — Splits
- **Do:** ~30-min blocks → tuning/dev/test 15/15/70, all cameras of a block together.
- **Done when:** test asserts no block spans splits; per-split counts (incl. IR and vehicle events) printed.

**💬 Prompt:**
```
Do step 2.1. Split the window into ~30-minute blocks assigned to tuning/dev/test at roughly 15/15/70 with a fixed seed, all cameras of a block in the same split. Check each split has a mix of activity types, some IR footage and some vehicle start/stop events; report counts. Add a test that no block appears in two splits.
```

### Step 2.2 — Queries 🚦
- **Do:** Claude Code builds worksheets (label, camera, time, keyframe). **You** write the queries. Then freeze + hash.
- **Done when:** 🚦 **GATE 2.2** — you declare them final.

**👤 You:** your job, not Claude Code's. Look at each keyframe and write a plain witness-style query. ~15–20 category + ~40 specific-event across splits, including vehicle start/stop events, each tagged `appearance` / `action` / `vehicle_event`. Don't run any search before freezing. Budget 2–3 hours.
**💬 Prompt (before you write):**
```
Do step 2.2, part 1. For each split, sample events across activity types (include vehicle start/stop events) and produce eval/queries/worksheet_<split>.csv with event_id, activity_type, camera, times and a keyframe saved under eval/queries/keyframes/. Leave blank columns 'query', 'kind' (category/specific) and 'qtype' (appearance/action/vehicle_event). Do not write any query text yourself.
```
**💬 Prompt (after you write):**
```
Do step 2.2, part 2. Validate my worksheets (no blanks, valid ids, kind and qtype set), convert them into frozen query files per split, compute SHA-256 hashes into eval/queries/HASHES, and add a check that fails if any query file changes. GATE 2.2: confirm and stop.
```

### Step 2.3 — Metrics + simple baselines
- **Do:** metrics from 1.10 (incl. event precision/recall); baselines sequential, random, witness ±10 min on tuning + dev.
- **Done when:** toy-case tests pass; baseline rows logged.

**👤 You:** recompute one headline-metric toy case by hand.
**💬 Prompt:**
```
Do step 2.3. Implement the metrics in section 1.10 (Recall@k, mAP@k, seconds-watched with the miss rule, bootstrap CI, and temporal-overlap precision/recall for detected events) with pytest toy cases whose answers are hand-computed in comments. Implement sequential, random (seeded) and witness ±10 min baselines; run on tuning and dev only; log and summarise.
```

**👤 End of Phase 2:** PROMPT E → `/clear`.

---

## PHASE 3 — Motion + detection · Mon 26 Oct (morning)

### Step 3.1 — MOG2 pass
- **Do:** per camera, downscaled, ~5 fps; per-second activity score + motion boxes.
- **Done when:** activity plot for one camera peaks at annotated events.

**👤 You:** run the full job yourself; look at the plot.
**💬 Prompt:**
```
PROMPT S first. Do step 3.1. Write scripts/motion_pass.py (OpenCV MOG2, downscaled, ~5 fps, resumable) storing per-second activity scores and merged motion boxes per camera. Test on 10 minutes of one camera, plot the activity score with annotated events overlaid, then give me the full-run command.
```

### Step 3.2 — Motion baseline
**💬 Prompt:**
```
Do step 3.2. Implement the motion-ranked baseline (rank segments by activity score, same grouping and metrics), evaluate on tuning and dev, log, and compare with the other baselines in a short table.
```

### Step 3.3 — Detector smoke test 🚦
- **Do:** set up both detector options (or whichever installs within ~45 min each) on a 10-min tuning clip incl. some IR: detect configured classes at ~5 fps; save annotated sample frames; measure fps on GPU; project full-run time; report license of each.
- **Why:** detection is a new dependency; prove it helps and fits before building on it.
- **Done when:** 🚦 **GATE 3.3** — you choose Option A, Option B, or no detector.

**👤 You:** look at ~50 annotated frames (RGB and IR). Are most visible people and vehicles boxed? Any glaring misses? Consider the license (AGPL vs Apache) for your repo. Decide.
**💬 Prompt:**
```
Do step 3.3. Ask me before adding packages. Set up a detector smoke test for Option A (Ultralytics YOLOv8n/s) and Option B (YOLOX-s), timeboxing each install to ~45 minutes. On a 10-minute tuning clip that includes some IR footage, run detection at ~5 fps on the classes in config, save ~50 annotated sample frames to results/figures/detector_smoke/, measure GPU fps, project full ~24 h runtime, and report each option's license (verify from the actual license files). Stop at GATE 3.3.
```

### Step 3.4 — Full detection pass
- **Do:** chosen detector at ~5 fps over all footage; Parquet per camera (frame time, class, box, score); resumable.
- **Done when:** detection counts per camera/class reported.

**👤 You:** run it yourself (overnight-style job; can run during Phase 4 work on already-finished cameras, or after).
**💬 Prompt:**
```
Do step 3.4 with the detector I chose. Write scripts/detect_pass.py (resumable, checkpointed, progress log) writing per-camera Parquet with timestamp, class, box, score. Test on 10 minutes, then give me the full-run command. Record the model and license in config.yaml and PROGRESS.md.
```
*(If Gate 3.3 chose "no detector", skip 3.4 and set `crop_source: motion`.)*

## PHASE 4 — Appearance index · Mon 26 Oct (afternoon)

### Step 4.1 — Sampling + motion-aware dedup
- **Do:** 1 fps; drop a frame only if pixel change in motion regions < threshold **and** its detection set is unchanged; max gap 10 s; tune on tuning split.
- **Done when:** pipeline recall reported (target ≥95% on tuning).

**💬 Prompt:**
```
PROMPT S first. Do step 4.1. Implement 1 fps sampling with dedup that drops a frame only if pixel change inside motion regions is below threshold AND its detections (if enabled) are essentially unchanged, plus the 10 s max-gap rule. Sweep the threshold on tuning and report kept frames and pipeline recall per value. Recommend a value; update config only after I agree.
```

### Step 4.2 — Embeddings: frames + crops
- **Do:** CLIP B/16 fp16; full frame + ≤4 crops (largest/most confident detections first, motion boxes to fill remaining slots, skip tiny crops); fp16 `.npy` + metadata (vector_id, camera, timestamp, type frame/crop, crop_source, class, box, split).
- **Done when:** counts + sizes reported; resumable.

**👤 You:** run the full job yourself.
**💬 Prompt:**
```
Do step 4.2. Write scripts/embed_appearance.py: for kept frames, embed the full frame plus up to 4 padded crops chosen by crop_source in config (detections first, motion boxes as fallback, skip crops under min_crop_px), with CLIP ViT-B/16 fp16, batched, resumable. Save fp16 vectors and a metadata parquet (vector_id, camera, timestamp, type, crop_source, class, box, split). Test on 10 minutes, then give me the full-run command.
```

### Step 4.3 — Mean-centering
- **Do:** per-camera mean over all that camera's indexed vectors (no labels), subtract, re-normalise; config flag.
- **Done when:** both variants saved.

**💬 Prompt:**
```
Do step 4.3. Add per-camera mean-centering: compute each camera's mean embedding over all its indexed vectors (no labels used), subtract, re-normalise. Control with appearance.mean_center and save both variants for ablation.
```

### Step 4.4 — First end-to-end search (NumPy) 🚦
- **Do:** brute-force search + grouping on dev; ablate crops none/motion/detection and centering on/off.
- **Done when:** 🚦 **GATE 4.4** — dev numbers + 5 example queries shown.

**👤 You:** watch the top clips for 5 queries; note failures and causes.
**💬 Prompt:**
```
Do step 4.4. Build NumPy brute-force search: text -> CLIP text embedding -> cosine top-k -> clip grouping per config. Run all dev queries for crop_source none / motion / detection_then_motion, each with and without mean-centering; log all. Show me 5 dev queries with top-3 clips (camera, times, proxy path, matched box and class). Stop at GATE 4.4.
```

**👤 End of Phase 4:** PROMPT E → `/clear`.

---

## PHASE 5 — Action branch · Tue 27 Oct (morning)

### Step 5.1 — Model bake-off 🚦
- **Do:** ~2 h dev video, identical windows, (a) mean-pooled CLIP, (b) LanguageBind-Video, (c) X-CLIP plain text; action-type dev queries; VRAM + speed.
- **Done when:** 🚦 **GATE 5.1** — you decide per 1.7.

**👤 You:** drop any model that won't install or fit 6 GB within ~1 hour.
**💬 Prompt:**
```
PROMPT S first. Do step 5.1. Implement windowing (8 frames, 4 s, stride 2 s) and three encoders: mean-pooled CLIP B/16, LanguageBind-Video, and X-CLIP with plain text embeddings only (verify in HF source that this avoids the video-conditioned prompt generator). Run on ~2 hours of dev footage with action-type dev queries. Report metrics, peak VRAM, windows/sec. Timebox each model's setup to ~1 hour. Stop at GATE 5.1.
```

### Step 5.2 — Full action indexing (overnight Tue → Wed)
**👤 You:** start before bed. *(Skip if Gate 5.1 dropped the branch.)*
**💬 Prompt:**
```
Do step 5.2 with the chosen model. Write scripts/embed_action.py (resumable, checkpoint every N windows, progress log, fp16 + metadata parquet). Test on 10 minutes, then give me the full-run command.
```

## PHASE 6 — Tracking + event rules · Tue 27 Oct (PM) → **hard stop Wed 28 Oct 13:00**

*(Skip this phase entirely if Gate 3.3 chose no detector.)*

### Step 6.1 — Within-camera tracking + stop/start rules
- **Do:** ByteTrack on vehicle detections per camera; per-track speed (box-centre px/s, smoothed); rules: `vehicle_stops` = moving ≥ `min_moving_s` then speed < threshold for ≥ `min_stopped_s`; `vehicle_starts` = the reverse. Store event records (camera, time, type, track box, best crop). Tune thresholds on the **tuning** split against MEVA vehicle start/stop annotations.
- **Why:** gives Rewind evidence for "stopping", which single frames can't provide, tested against real labels.
- **Done when:** precision/recall vs annotations on tuning reported; unit tests for the rules on synthetic tracks.

**👤 You:** watch 5 detected stops and 5 detected starts on the proxy. Real or not? Watch out for traffic queues and occlusion.
**💬 Prompt:**
```
PROMPT S first. Do step 6.1. Ask before adding the tracker package. Run ByteTrack on vehicle detections, within each camera only, never linking across cameras or storing identities. Compute smoothed per-track speed and implement the vehicle_stops / vehicle_starts rules from section 1.9 config. Write unit tests on synthetic tracks with hand-computed expected events. Tune thresholds on the tuning split against MEVA vehicle start/stop annotations and report precision/recall per setting. Store event records with camera, time, type, track box and best crop path in SQLite. Give me 5 detected stops and 5 starts to check by eye.
```

### Step 6.2 — Event branch in search 🚦
- **Do:** CLIP-embed each event's best vehicle crop; event intent = query contains a configured keyword; candidates = matching-type events ranked by crop–query similarity; dev results for `vehicle_event` queries with branch on/off.
- **Done when:** 🚦 **GATE 6.2** — keep or drop (rules in 1.7; hard stop Wed 13:00).

**👤 You:** if it isn't clearly helping on dev `vehicle_event` queries by Wed 13:00, **drop it**, set `events.enabled: false`, and move on. Documenting a dropped branch is a valid result.
**💬 Prompt:**
```
Do step 6.2. CLIP-embed each event record's best vehicle crop. Implement event intent detection via the keyword lists in config. For queries with intent, return events of that type ranked by crop-to-query cosine similarity, grouped into clips. Evaluate vehicle_event dev queries with the event branch on vs off (NumPy search is fine here). Report results and stop at GATE 6.2.
```

**👤 End of Phase 6:** PROMPT E → `/clear`.

---

## PHASE 7 — C++ engine + search · Wed 28 Oct (PM) → Thu 29 Oct (AM)

### Step 7.1 — Extend the HNSW engine
- **Do:** named indexes, int64 id ↔ metadata, camera/time filtering, save/load, inner product on normalised vectors; GoogleTest.
- **Done when:** tests pass; indexes load in seconds.

**👤 You:** before this step, put the existing engine into `engine/` (copy or submodule).
**💬 Prompt:**
```
PROMPT S first. Do step 7.1. Inspect the existing engine in engine/ (real API, data layout, build) and summarise before changing anything. Then extend: multiple named indexes, int64 ids mapped to metadata, camera and time-range filtering, save/load, inner-product search on normalised vectors. GoogleTest for each feature, including recall vs brute force on random data.
```

### Step 7.2 — Scaling benchmark
- **Do:** recall@200 + latency vs brute force at real sizes, then synthetic 1M (and 5M if RAM allows); `ef_search ≥ 200`.
- **Done when:** `results/reports/scaling.md` + plot.

**💬 Prompt:**
```
Do step 7.2. Benchmark the engine vs exact brute force: recall@200 and latency at our real index sizes, then synthetic 1M and 5M 512-d vectors if 16 GB RAM allows (report if not). Sweep ef_search (>= 200). Write results/reports/scaling.md and save the plot to results/figures/.
```

### Step 7.3 — pybind11 + FastAPI
- **Done when:** `/search` returns grouped clips; end-to-end latency measured.

**💬 Prompt:**
```
Do step 7.3. Add pybind11 bindings (CMake) and a FastAPI app that loads encoders, indexes and event records once at startup and exposes /search (query, cameras, time range, k) and /coverage. Measure end-to-end latency over 20 queries; report median and p95.
```

### Step 7.4 — Fusion + grouping
- **Do:** per-branch z-scores → weighted fusion of active branches (tune weights on tuning) → merge ≤5 s → cap 60 s → mean top-3 → context. Each result records which branches matched ("why it matched").
- **Done when:** edge-case tests pass; weights tuned.

**💬 Prompt:**
```
Do step 7.4. Implement per-branch z-score normalisation and weighted fusion over whichever branches are enabled, merging within merge_gap_s, the max_clip_s cap, mean-top-3 ranking and context padding. Each clip must record which branches contributed. Tests for edge cases (overlaps, video start/end, chain-merging, a branch with no hits, a disabled branch). Tune weights on the tuning split only and report the sweep.
```

**👤 End of Phase 7:** PROMPT E → `/clear`.

---

## PHASE 8 — Evaluation · Thu 29 Oct (PM)

### Step 8.1 — Dev ablations → freeze config 🚦
- **Done when:** 🚦 **GATE 8.1** — you approve the final config; committed + hashed.

**👤 You:** pick the best dev config; if two are close, pick the simpler one.
**💬 Prompt:**
```
PROMPT S first. Do step 8.1. Run dev ablations: appearance-only; + action (if enabled); + events (if enabled); full fusion; crop_source none/motion/detection; mean-centering on/off; HNSW vs brute force (metrics + latency). Write results/reports/ablations.md, one row per variant, broken down by qtype. Recommend a final config. Stop at GATE 8.1.
```

### Step 8.2 — The single test run
**👤 You:** only after the config is committed. Whatever the numbers are, they are the numbers.
**💬 Prompt:**
```
Do step 8.2. Verify query hashes and that the committed config matches the frozen config hash. Run Rewind and all four baselines on the test split exactly once, plus event-rule precision/recall on test if events are enabled. Write results/reports/test_results.md: per-query table, summary with bootstrap CIs, RGB vs IR and per-qtype breakdowns, headline metric vs each baseline. Save charts to results/figures/. Change nothing after the run.
```

### Step 8.3 — Failure analysis
**👤 You:** watch each failure and assign the cause yourself.
**💬 Prompt:**
```
Do step 8.3. Collect the ~10 worst test failures with query, qtype, ground-truth time, what Rewind returned instead, and thumbnails of both. Put them in results/reports/failures.md with an empty 'cause' column for me (object too small / IR / detector miss / track broken / action too brief / ambiguous query / dropped by dedup / other).
```

---

## PHASE 9 — UI · Fri 30 Oct (morning)

### Step 9.1 — Single-page UI
- **Do:** search box; camera + time filters; result cards (thumbnail with box highlight, class, camera, time range, score bar, "matched by: appearance / action / event"); proxy playback of the clip range; per-camera timeline with hit markers; coverage line; never a bare "no results".
- **Done when:** 3 demo queries work end-to-end in the browser.

**👤 You:** try 10 of your own queries; note anything slow or confusing.
**💬 Prompt:**
```
PROMPT S first. Do step 9.1. Build a single-page UI in ui/ served by FastAPI: search box, camera and time filters, result cards with thumbnail and highlighted box, class, camera, time range, score bar and which branches matched; clicking plays the proxy from clip start to end; a per-camera timeline with hit markers; a coverage line ('searched X h across Y cameras'). Never a bare 'no results'. Keep it simple. Give me the start command.
```

## PHASE 10 — Packaging · Fri 30 Oct (afternoon)

### Step 10.1 — README + fresh-clone test
**👤 You:** clone into a new folder and follow the README exactly.
**💬 Prompt:**
```
PROMPT S first. Do step 10.1. Write README.md: what Rewind is, setup, data download, full pipeline commands in order, running the API/UI, reproducing results, dataset citation and license, detector/tracker licenses, and known limitations (1.11). Add a smoke-test path that rebuilds a tiny index from 10 minutes of footage and runs one query.
```

### Step 10.2 — Results package
**💬 Prompt:**
```
Do step 10.2. Make sure results/figures/ has clean, labelled figures for: Rewind vs baselines (headline metric and Recall@k), ablations by qtype, scaling plot, RGB vs IR, event-rule precision/recall (if enabled), and an architecture diagram showing only the branches actually kept. Write results/SUMMARY.md: one page with key numbers, each linked to its report and experiments.csv row, plus a list of branches dropped at gates and why.
```

### Step 10.3 — Demo queries + recording
**👤 You:** choose 3 strong queries and 1 honest failure; screen-record the full UI demo yourself.
**💬 Prompt:**
```
Do step 10.3. From test results, list candidate demo queries that work well (camera/time, which branches matched, why they succeed) and 2 instructive failures. Write docs/DEMO.md with the chosen queries, expected results and a fallback plan if the live demo breaks.
```

**👤 Final check (Fri evening):** every item in 1.3 is true. Update PROGRESS.md one last time. Project complete.

---

## 2.11 If you fall behind — cut in this order
1. UI polish (keep function, drop styling)
2. Event branch (Phase 6) — already hard-stopped Wed 13:00
3. Action branch (ship appearance + detection crops + motion; document why)
4. Synthetic scaling beyond real size

**Never cut:** the frozen query set, the single test run, the baselines.

## 2.12 Red flags in `git diff` after every step
- `try/except` that swallows errors or returns empty/zero results
- numbers, class lists or keywords in code that belong in `config.yaml`
- tests that only check mocks or can't fail
- any reference to the test split outside step 8.2
- changes to `eval/queries/` after Gate 2.2
- timestamp arithmetic without a test (frame ↔ second ↔ wall-clock)
- track IDs persisted as identities, or any linking of tracks across cameras
- an event claimed in results/UI without the rule that produced it
- new dependencies you didn't approve

## 2.13 References
1. Corona et al., MEVA, WACV 2021 — arXiv 2012.00914
2. Radford et al., CLIP, ICML 2021 — arXiv 2103.00020
3. Malkov & Yashunin, HNSW, IEEE TPAMI 2020 — arXiv 1603.09320
4. Zhang et al., ByteTrack, ECCV 2022 — arXiv 2110.06864
5. Ge et al., YOLOX — arXiv 2107.08430
6. Ultralytics YOLOv8 (software, AGPL-3.0)
7. Ni et al., X-CLIP, ECCV 2022
8. Zhu et al., LanguageBind, ICLR 2024
9. Luo et al., CLIP4Clip — arXiv 2104.08860
10. Keval & Sasse, "Not the Usual Suspects", Security Journal, 2010
11. City Security Magazine — Met Police CCTV review after the 2011 London riots
