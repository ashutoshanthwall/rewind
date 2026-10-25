# MEVA: source, format and facts checked in step 0.3

Everything here was checked against real files on 2026-10-24 unless marked VERIFY.

## Sources
- **Video:** `s3://mevadata-public-01/drops-123-r13/<date>/<hour>/<clip>.r13.avi` (public, `--no-sign-request`).
  3,956 clips, 475.9 GB. `updates-r13/` holds 2 more files (not used; VERIFY whether they replace drop files).
- **Annotations:** https://gitlab.kitware.com/meva/meva-data-repo, commit `421841a` (2021-12-13).
  Cloned (blobless + sparse) into `data/annotations/meva-data-repo/`: activities/types YAML, metadata and docs, and
  (since step 1.3) the Kitware box files `geom.yml` of the selected clips only (252 files, 0.96 GB;
  `scripts/annotation_patterns.py`), plus all contrib files of the IR cameras.
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
  `tsr0` = [start_frame, end_frame] within the clip, 0-based, **inclusive end** (settled at 1.3, see
  "Annotation conventions"; the `[0, 9000]` of `empty_37` is a fixed placeholder, not evidence either way).
- Counts from these files are **records per clip**, not unique events: an activity crossing a 5-min clip boundary
  is annotated in both clips, and one event seen by two cameras is annotated per camera.
- `<clip>.types.yml`: actor id → object type, e.g. `{types: {cset3: {vehicle: 1.0}, id1: 1}}`.
- `<clip>.geom.yml`: per-frame actor boxes `{geom: {id1, id0, ts0 (frame), [ts1 (s)], g0: "x1 y1 x2 y2", [keyframe]}}`.
- **`empty_37`** is not an activity: a whole-clip act record (`tsr0: [0, 9000]`, no actors, key `src`) plus a meta line
  "was marked empty-37", or (second style) only a meta line "... was marked empty_37". The README wording suggests
  "none of the 37 activities is present"; **VERIFY** whether that means checked-and-absent or not annotated.
  45 clips in the release have the first style; all 9 annotated IR clips are of this kind.

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
- RGB probe (G436): H.264 1920x1080, 30 fps, ~4.5 Mbit/s.
- **Checked on all 300 selected files at 1.1:** H.264 in AVI, constant 30/1 fps, one video stream; packet count =
  header frame count (8,831-9,013 per clip) with decode timestamps 0..n-1. Activity `tsr0` end frames never reach
  `n_frames`, and in 134 of 221 annotated RGB clips the last activity ends exactly on frame `n_frames - 1`:
  strong evidence that `tsr0` is 0-based with an inclusive end (confirmed at 1.3, see "Annotation conventions"). `[0, 9000]` appears only on
  `empty_37` markers.
- **Frame index rule:** the files have B-frames and AVI stores no presentation timestamps, so decoded
  `frame.pts` / `frame.time` are scrambled (ffprobe: `pts=N/A`). Frame index = count of decoded frames from 0;
  time = index / 30. Never use `frame.pts`; don't seek by timestamp without checking where you land.
- Filename times are whole seconds at both ends (15 selected names say 299 s for 300.0-300.33 s of video), so
  wall-clock is accurate to about +/- 1 s. The clip table's cross-camera frame offsets all equal
  (ref start - clip start) x 30 from the filename times, i.e. they carry no extra sync information, and the sign
  is the opposite of the clip-table README's equation (VERIFY which is right; we never link cameras).
  7 selected clips (session edges) are missing from the clip table.
- Cameras record in **sessions**: no camera has more than 3.65 h of video on any date. A continuous 4-5 h window
  does not exist. Each RGB camera has ~7.5-8 annotated hours spread over 7 dates.
- `metadata/meva-camera-daily-status.txt`: no camera changes indoor/outdoor across days. Fixed view across 03-07/11/15 checked by eye at 1.1 for all 6 cameras.
- **Kitware has no real activity labels for IR footage** (see `empty_37`). Third-party labels exist (below).

## Annotation conventions (checked at step 1.3, 2026-10-25, on the selected clips)
Code: `rewind/ingest/annotations.py`, `scripts/build_events.py`; report `results/reports/events_1.3.md`. Each item was
found by one investigator and re-checked by an independent skeptic with their own scripts.
- **Frames:** 0-based and equal to our decoded-frame index. On 249 RGB clips: geom frame 0 used 867 times,
  n_frames-1 903 times, never >= n_frames; 416 activities end exactly on n_frames-1, none past it, also on clips
  with 8,831 or 9,005-9,009 frames; `ts1` = `ts0`/30 on all 1,753,192 lines that have it.
- **Inclusive end:** each actor's geom covers exactly frames start..end of its `tsr0` (26,711/26,711 actors).
  An event covers [start/30, (end+1)/30) s of its clip.
- **Boxes:** `g0` = x1 y1 x2 y2 (KPF spec p6; never x2<x1; values pile up at 1919/1079). Dense: one box per frame
  of each actor span (interpolated between keyframes, VERIFY), nothing outside it. 4,409 of 9.68M boxes overshoot the
  frame by a few px (4 by >= 50 px): stored as annotated, clip before cropping. 1 actor's span runs 18 frames past
  its activity; boxes are stored for the event's frames only.
- **Two line styles** in Kitware files: python-repr `- {'geom': {...}}` and YAML flow `- { geom: {...} }` (66 selected
  clips, all evaluation-level, 64-bit hash ids; `keyframe` written True/true/False or left out). A parser that matches
  one style silently drops the other (happened twice during the investigation).
- **Clip boundaries:** MEVA-Annotation-Definitions.pdf p3: an activity running across a clip boundary is annotated in
  each clip, with unrelated ids (no KPF `tag` links). 220 such pairs are linked by `events.group_id` (same camera,
  back-to-back clips, same type, end on the last frame / start on frame 0, box IoU >= 0.5 at the join; one-to-one).
  Same type alone is not enough: about 1/3 of end-of-clip pieces have no overlapping box in the next clip.
  Pieces at a clip edge are truncated: their start/end reflect the cut, not the motion (relevant for 6.1).
- **Ids:** `id1` is unique per actor per activity (26,711 actors = 26,711 ids), not a persistent object; not stored.
- **`src_status`** (Kitware): good 13,334, janitor 2,670, not_good 157, gladiator2 154, gladiator_1 112,
  unaudited 96, refiner_1 2, refiner_agreement 2. Meaning undocumented (VERIFY). Stored, not filtered.
- **Box timing, VERIFY (few tracks):** Kitware boxes follow vehicles to within a few frames (optical flow on 3
  tracks: best shift 0 to +1 frame). CMU-v2 IR boxes were seen 20-40 frames ahead of the vehicle on 1 track; shifts
  against UMD-IBM vary in sign. Do not use contrib boxes as frame-exact.
- **Coverage of the selection:** RGB 249 clips: 221 with events, 28 empty_37 (8 act-style, 20 meta-only; G330 14).
  IR (G474) 51 clips: 29 with contrib events, 22 without (18 no annotation at all, 3 Kitware empty_37 markers,
  1 contrib .json with 0 records). Kitware RGB: 16,527 events (981 vehicle_starts, 1,000 vehicle_stops).

## Third-party (contrib) labels — used for IR only (Gate 0.3)
- `annotation/DIVA-phase-2/MEVA/contrib/`: CMU-v2 (1,111 clips) and UMD-IBM (637) claim "All 37 ActEV activities";
  IBM-person-vehicle (217) and IBM-person-person (90) cover subsets. All: "Scope: Training", old activity names
  (mapped with `documents/activity-name-mapping.csv`; nothing left unmapped), KPF YAML with `act2` as a `!!set`,
  files named `<clip>-activities.yml` (dash, not dot).
- They label 139 distinct IR clips; G474 (selected): 29 clips (2.42 h), 148 records (max over sources),
  12 `vehicle_starts`, 15 `vehicle_stops`. None of them labels a clip Kitware marked `empty_37`.
  After step 1.3's merging: 205 raw records → 3 zero-length dropped → 27 merged away → **175 events**
  (12 `vehicle_starts`, 14 `vehicle_stops`). The 0.3 numbers took the max per clip over sources, hence the difference.
- **Step 1.3 reads the per-clip NIST `.json` files, not the `-geom.yml`**: the geom files hold boxes for only 27/125
  CMU-v2 and 15/69 UMD-IBM actors, and IBM-person-vehicle reuses one id (99) across activities. The .json has a box on
  every frame start..end of every record and the same activity counts as the YAML (46/46 files). Box = (x, y, x+w, y+h);
  object types are all 'Person' (also on vehicles) and are not used. **VERIFY:** NIST's JSON spec (p9) says frames
  begin at 1 and the end key is the first absent frame; the data (start frame 0 used, boxes on the end frame, ends
  never reach n_frames, YAML tsr0 equal) fit 0-based inclusive like Kitware. Difference: 1 frame.
- **Merging (config `annotations.contrib_merge_min_box_iou` 0.3):** same type + time overlap + mean box IoU >= 0.3,
  within and across sources (UMD-IBM has 8 exact duplicates; CMU/UMD overlap on 24 pairs). Time overlap alone merged
  two different vehicles. Contrib start/end differ from Kitware's by ~1-3 s on shared RGB clips (CMU start p10..p90
  -101..+38 frames, no constant ±1 offset).
- **Incomplete:** on RGB clips labelled by both (`scripts/check_contrib_quality.py`), CMU-v2 has 0.23x and UMD-IBM
  0.45x as many records as Kitware (vehicle starts+stops 0.22x / 0.50x). Part may be different conventions.
- **Rules:** RGB ground truth is Kitware only; contrib labels never enter RGB evaluation. For IR, report only
  recall-type metrics (Recall@k, seconds-to-first-hit on labelled events, event-rule recall), never precision or mAP.
- Still open: whether contrib sources cut boundary-crossing activities like Kitware (only 2 G474 contrib events
  start on frame 0; none end on the last frame).

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
(cd ../../.. && uv run python -m scripts.annotation_patterns) >> .git/info/sparse-checkout  # boxes, selected clips (step 1.3)
git checkout master   # pinned facts above refer to commit 421841a
cd ../../.. && uv run python scripts/survey_meva.py && uv run python scripts/check_contrib_quality.py
```
