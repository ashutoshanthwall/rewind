# Rewind — semantic search over multi-camera CCTV footage (research demo)

*Describe what you saw, get the clip.* Rewind indexes ~25 h of multi-camera footage from the MEVA dataset once, then
ranks short clips against a plain-English description ("a white bus parked by the walkway"), with an optional
"around this time ± minutes" window, a per-camera timeline, playback of the matched clip and a coverage line
("searched X h across Y cameras").

> **Status:** draft README written at the pre-Phase-10 review (27 Oct 2026). Setup, the full pipeline and the
> fresh-clone test are completed in step 10.1; until then the commands below are the committed entry points, not a
> tested walkthrough.

## Research demo, not a deployed surveillance system

Rewind is a solo student research prototype, evaluated on scripted, consented footage (MEVA). It is **not** a
deployed or deployable surveillance system:

- It does no face recognition, no person re-identification and no cross-camera linking. It returns *candidate*
  clips ranked by similarity; people judge them. Similarity is not proof, and an empty or wrong result never means an
  event did not happen.
- Its appearance model's card (OpenCLIP ViT-B-16 `laion2b_s34b_b88k`,
  [huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K](https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K),
  read 27 Oct 2026) states: *"Any deployed use case of the model - whether commercial or not - is currently out of
  scope"* and *"Certain use cases which would fall under the domain of surveillance and facial recognition are always
  out-of-scope regardless of performance of the model."* Any real-world use would need a different model choice, a
  legal and ethical review, and an evaluation on real (unscripted) footage.
- CLIP-style models carry documented demographic biases (Radford et al., 2021). Rewind must never be presented as
  identifying people.

## What was built, and what the plan dropped

Planned (CLAUDE.md 1.2): appearance branch with detection crops + action branch + event branch, fused.
**Delivered and tested:** the appearance branch only — CLIP ViT-B/16 (fp16) over 1 fps frames (near-duplicates dropped,
at least one frame every 10 s) plus up to 4 **motion** crops per frame, searched with the author's C++ HNSW engine
(ef 800, 200 hits per index), grouped into ≤ 60 s clips with ±5 s context. Dropped or unused, each by a pre-registered
dev/tuning rule (PROGRESS.md): detection crops (Gate 4.4), mean-centering (4.4, 8.1), the action branch (8.1: weight 0
on an 8/1/8 tie; LanguageBind is still used to reproduce the ablations), the vehicle stop/start event branch (6.2:
rule precision/recall < 0.5). The YOLOX detector is run at indexing (it feeds frame dedup) but its crops are not
searched.

## Results (single test run, step 8.2)

`results/reports/test_results.md` — 24 specific + 10 category queries, 16.7 h of test footage, run once with a frozen
config. Quote these numbers only with their caveats:

| | Rewind | sequential | witness ±10 min | motion-ranked |
|---|---|---|---|---|
| median seconds watched before the first correct clip [95% CI] | 1,173 [376, 61,036] | 30,507 | 2,977 | 20,003 |
| specific events in Rewind's clip list | 13 / 24 | (all footage) | (all) | (all) |

- The median is fragile: it sits on the 13th hit; one fewer hit would put it near 30,800 s. **No paired-saving CI
  excludes zero** against any baseline. On the **mean**, Rewind (28,277 s) loses to the witness (2,927 s) and motion
  (20,676 s) baselines, partly because a Rewind miss is charged a full 16.7 h review while baselines never miss (the
  sequential baseline's expected cost is about half the footage).
- When found, the event's clip took 146–1,412 s of review; only 1 of 24 was in the top 10 (R@10 0.042). 6 of the 13
  hits count through neighbouring frames and the ±5 s context (no retrieved frame inside the event).
- Category mAP, instance recall and R@k are **not comparable across systems**: baselines rank 60 s segments, Rewind
  clips are ~10–12 s, and these metrics count units, not seconds.
- IR (thermal): 1 of 4 specific events found; reported with recall-type metrics only (third-party labels).
- End-to-end query latency at the frozen config: median 36 ms, p95 39 ms (20 queries, local GPU). At this index size
  (≤ 175k vectors per branch) exact NumPy search is faster than the HNSW setting used (about 19 vs 26 ms per branch at
  recall@200 0.995); a speed advantage at larger scale was **not** shown.

**How the test queries were written:** drafted by Claude Code agents from the annotation labels and keyframes in
witness style (a seeded plan fixed which events, query types and specificity), reviewed and approved by Ashutosh,
and frozen (SHA-256 in `eval/queries/HASHES`) before any search code existed. This is a deviation from the original
plan, in which the author writes them; every later decision was made on tuning/dev only.

## Known weaknesses (evidence in `results/reports/review_pre10.md`, `failures.md`)

- **Crop budget, not only detection:** retrieval searches ≤ 4 motion crops per frame, which cover ~6 % of annotated
  people and ~38 % of vehicles (tuning); the detector itself finds 0.41 of RGB people (0.24 of small ones) and ~0 in IR.
- **Scene context:** crop matches ignore the scene, so "outside a building" can still return the indoor gym (G330).
  Workaround in the UI: leave indoor cameras out ("Outdoor only").
- **Small and distant objects** are below what CLIP ViT-B/16 at 224 px resolves; **thermal IR** is weak (colour words
  cannot match; few IR vectors).
- **List depth was never tuned** (200 hits per index); on dev a deeper list (800) found 11/12 instead of 9/12, at the
  cost of more clips ahead of each hit (reviewers' exploratory scratch run, not a committed script). The UI offers "Search deeper" as a user fallback outside the tested setting.
- **Few queries per decision:** every gate used 6–12 specific queries, half of dev on the easy indoor camera; treat
  "dropped at gate X" as "not shown to help here", not "does not help".
- Coverage: footage is sampled (1 fps, near-duplicates dropped), so "searched X h" counts footage, not every second.
  The dataset is scripted, so events are cleaner than real incidents.

## Setup and pipeline (entry points; tested walkthrough in step 10.1)

Python 3.11 with `uv` (`pyproject.toml`, `uv.lock`; CUDA 12.6 PyTorch wheels), a system `ffmpeg` with the libx264
encoder (`ffmpeg -encoders | grep libx264`), CMake + a C++20 compiler, the AWS CLI (public MEVA bucket), and an NVIDIA
GPU (developed on 6 GB). Build the engine module on the machine that runs it (`-march=native`):
`cmake -S engine -B engine/build -DREWIND_PYTHON=ON -Dpybind11_DIR=$(uv run python -m pybind11 --cmakedir) -DPython_EXECUTABLE=$(uv run python -c 'import sys; print(sys.executable)') && cmake --build engine/build -j8`.

Order (all `uv run python -m scripts.<name>` from the repo root): `download` → `build_manifest` → `make_proxies` →
`build_events` → `make_splits --from-committed` (restores the frozen split) → `index_pass` (motion + detection) →
`embed_appearance` → `collect_appearance` → `build_indexes` → API/UI:
`uv run uvicorn rewind.api.app:app --port 8000` → http://localhost:8000. Optional, only to reproduce the action-branch
ablations: `bash scripts/fetch_languagebind.sh` (weights + config + tokenizer, sha256-checked) and `embed_action`.

## Credits and licenses

Rewind code and the C++ engine: MIT (`LICENSE`). Third parties keep their own terms:

| Component | Use in Rewind | License / terms |
|---|---|---|
| **MEVA dataset** — Corona et al., "MEVA: A Large-Scale Multiview, Multimodal Video Dataset for Activity Detection", WACV 2021 (arXiv 2012.00914); Kitware Inc. and IARPA | all footage and annotations | CC BY 4.0 (`data/annotations/meva-data-repo/LICENSE`; mevadata.org). Frames shown in `results/figures/` and in the UI are resized, cropped and annotated by Rewind |
| **CLIP** method — Radford et al., ICML 2021 | appearance model architecture | — |
| **OpenCLIP** (Ilharco et al., 2021, Zenodo, doi:10.5281/zenodo.5143773) and the `laion2b_s34b_b88k` ViT-B-16 weights, trained on **LAION-2B** (Schuhmann et al., "LAION-5B: An open large-scale dataset for training next generation image-text models", NeurIPS 2022 Datasets and Benchmarks) | appearance encoder and weights | weights MIT per the model card; card excludes deployed and surveillance use (above) |
| **YOLOX** — Ge et al., arXiv 2107.08430 (vendored code + YOLOX-s release weights) | detection at indexing | Apache-2.0 (`rewind/detect/yolox_vendor/LICENSE`, `NOTICE.md`) |
| **ByteTrack** — Zhang et al., ECCV 2022 (vendored) | event-branch experiments (dropped) | MIT |
| **LanguageBind** — Zhu et al., ICLR 2024 (vendored code; `LanguageBind_Video_V1.5_FT` weights) | action-branch ablations and the secondary result only; not used by the frozen config | code and weights MIT per the card ("the majority of this project"); training data VIDAL-10M is **CC BY-NC 4.0** — treat as research / non-commercial use |
| Hugging Face transformers (parts vendored with LanguageBind) | model code | Apache-2.0 |
| **FFmpeg** (system binary, libx264) and **PyAV** wheels | proxies and decoding | FFmpeg LGPL/GPL depending on the build; PyAV is BSD-3-Clause but its wheels bundle GPL x264/x265. Used as tools; review the GPL terms before redistributing an environment or container |
| OpenCV, PyTorch, NumPy, pandas, FastAPI, uvicorn, pybind11 | pipeline, API | their own permissive licenses |

Open license items are tracked in PROGRESS.md ("VERIFY register").
