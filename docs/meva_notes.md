# MEVA: source, format and facts checked in step 0.3

Everything here was checked against real files on 2026-10-24 unless marked VERIFY.

## Sources
- **Video:** `s3://mevadata-public-01/drops-123-r13/<date>/<hour>/<clip>.r13.avi` (public, `--no-sign-request`).
  3,956 clips, 475.9 GB. `updates-r13/` holds 2 more files (not used; VERIFY whether they replace drop files).
- **Annotations:** https://gitlab.kitware.com/meva/meva-data-repo, commit `421841a` (2021-12-13).
  Cloned (blobless + sparse) into `data/annotations/meva-data-repo/`: activities/types YAML, metadata and docs only;
  `geom.yml` (boxes) not checked out yet. Fetch them for the selected clips only, at step 1.3.
- **License:** CC-BY-4.0, read from the repo's `LICENSE` file ("MEVA dataset by Kitware Inc. and IARPA").
  Cite Corona et al., WACV 2021 (arXiv 2012.00914).

## Annotation format (KPF YAML, one file set per 5-min clip)
- `annotation/DIVA-phase-2/MEVA/kitware/` = evaluation-level (769 clips). README: "all instances of the 37 ActEV
  activities should be annotated". `kitware-meva-training/` = training-level (1,443 clips), README: same process
  "with the exception of a reduced audit step". Whether training-level labels are as complete is not stated: VERIFY
  (it affects how many correct results look like false positives in the 129 training-level selected RGB clips).
  No clip appears in both.
- `<clip>.activities.yml`: records like
  `{act: {act2: {vehicle_stops: 1.0}, id2: 9, timespan: [{tsr0: [2301, 2380]}], actors: [{id1: 3, timespan: [...]}]}}`.
  `tsr0` = [start_frame, end_frame] within the clip. **Off-by-one risk, VERIFY at 1.3:** `empty_37` records span
  `[0, 9000]` but the probed 5-min RGB clip has 9,000 frames (0-8999). Either the end is exclusive or labels can
  run one frame past the video. Check against geom.yml frame numbers and the KPF spec before converting to seconds.
- Counts from these files are **records per clip**, not unique events: an activity spanning a 5-min clip boundary
  is annotated in both clips (VERIFY how it is split), and one event seen by two cameras is annotated per camera.
- `<clip>.types.yml`: actor id → object type, e.g. `{types: {cset3: {vehicle: 1.0}, id1: 1}}`.
- `<clip>.geom.yml`: per-frame actor boxes (not yet downloaded; format VERIFY at 1.3).
- **`empty_37`** is not an activity. It is a whole-clip record (`tsr0: [0, 9000]`) plus a meta line "was marked empty-37",
  meaning none of the 37 activities is present. 45 clips, **all 9 annotated IR clips are of this kind**.

## Activity names (exact, `documents/activity-names.txt`, 37)
Vehicle start/stop: **`vehicle_starts`** (2,087 instances in all annotations), **`vehicle_stops`** (2,151).
Other vehicle: `vehicle_turns_left`, `vehicle_turns_right`, `vehicle_reverses`, `vehicle_makes_u_turn`,
`vehicle_picks_up_person`, `vehicle_drops_off_person`. Full list and counts: `uv run python scripts/survey_meva.py`.

## Clip naming and time
- `date.start.end.site.camera`, e.g. `2018-03-11.16-30-08.16-35-08.hospital.G436`; times are local time
  (per annotation README). Start seconds differ per camera (G479 16:30:00 vs G436 16:30:08).
- 26 clips cross midnight (`23-55-00.00-00-00`); 1 clip is zero-length (`10-01-05.10-01-05`, 115 KB).
- Cross-camera sync: `metadata/meva-clip-camera-and-time-table.txt` gives frame offsets within a camera set.
  IR cameras are not in the camera topology (camera set `IR`, no offsets).

## Cameras and recording
- **IR cameras: G474, G475, G476, G479** (camera set `IR`), H.264 352x240, 30 fps, ~55 kbit/s, grayscale.
  **G474 checked thermal by eye** (2018-03-11 16:15 clip): tyres glow white (black in visible light), people
  uniformly bright incl. heads; the MEVA abstract (arXiv 2012.00914) says "38 RGB and thermal IR cameras".
  The other IR cameras were not checked frame by frame.
- RGB probe (G436): H.264 1920x1080, 30 fps, ~4.5 Mbit/s. A 5-min clip has 9000 frames, matching `tsr0: [0, 9000]`
  → **30 fps** (VERIFY per file from the container at 1.1).
- Cameras record in **sessions**: no camera has more than 3.65 h of video on any date. A continuous 4-5 h window
  does not exist. Each RGB camera has ~7.5-8 annotated hours spread over 7 dates.
- `metadata/meva-camera-daily-status.txt`: no camera changes indoor/outdoor across days. Fixed view across days: VERIFY by eye.
- **Kitware has no real activity labels for IR footage** (see `empty_37`). Third-party labels exist (below).

## Third-party (contrib) labels — used for IR only (Gate 0.3)
- `annotation/DIVA-phase-2/MEVA/contrib/`: CMU-v2 (1,111 clips) and UMD-IBM (637) claim "All 37 ActEV activities";
  IBM-person-vehicle (217) and IBM-person-person (90) cover subsets. All: "Scope: Training", old activity names
  (mapped with `documents/activity-name-mapping.csv`; nothing left unmapped), KPF YAML with `act2` as a `!!set`,
  files named `<clip>-activities.yml` (dash, not dot).
- They label 139 distinct IR clips; G474 (selected): 29 clips (2.42 h), 148 records (max over sources),
  12 `vehicle_starts`, 15 `vehicle_stops`. None of them labels a clip Kitware marked `empty_37`.
- **Incomplete:** on RGB clips labelled by both (`scripts/check_contrib_quality.py`), CMU-v2 has 0.23x and UMD-IBM
  0.45x as many records as Kitware (vehicle starts+stops 0.22x / 0.50x). Part may be different conventions.
- **Rules:** RGB ground truth is Kitware only; contrib labels never enter RGB evaluation. For IR, report only
  recall-type metrics (Recall@k, seconds-to-first-hit on labelled events, event-rule recall), never precision or mAP.
- **VERIFY at 1.3, how CMU and UMD clip timespans:** frame numbering (0-based? inclusive end?), whether an activity
  spanning a 5-min clip boundary is cut at the boundary or dropped, how their start/end points compare with
  Kitware's on shared RGB clips, and how to merge two sources labelling the same IR clip (counts here take the
  max per clip; events must be de-duplicated by temporal overlap, not summed).

## Reproducing the annotation checkout (40 MB + 94 MB contrib)
```bash
cd data/annotations
git clone --filter=blob:none --no-checkout --depth 1 https://gitlab.kitware.com/meva/meva-data-repo.git
cd meva-data-repo && git sparse-checkout init --no-cone
cat > .git/info/sparse-checkout <<'PATTERNS'
/README.md
/LICENSE
/metadata/*.txt
/metadata/*.md
/documents/*
/annotation/DIVA-phase-2/MEVA/README.md
/annotation/DIVA-phase-2/MEVA/kitware/**/*.activities.yml
/annotation/DIVA-phase-2/MEVA/kitware/**/*.types.yml
/annotation/DIVA-phase-2/MEVA/kitware-meva-training/**/*.activities.yml
/annotation/DIVA-phase-2/MEVA/kitware-meva-training/**/*.types.yml
/annotation/DIVA-phase-2/MEVA/contrib/README.md
/annotation/DIVA-phase-2/MEVA/contrib/*/README*
/annotation/DIVA-phase-2/MEVA/contrib/**/*G47[4-9]*
/annotation/DIVA-phase-2/MEVA/contrib/CMU-v2/**/*-activities.yml
/annotation/DIVA-phase-2/MEVA/contrib/UMD-IBM/**/*-activities.yml
PATTERNS
git checkout master   # pinned facts above refer to commit 421841a
cd ../../.. && uv run python scripts/survey_meva.py && uv run python scripts/check_contrib_quality.py
```
