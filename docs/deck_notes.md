# Deck notes — corrections for the submission deck (`docs/Rewind_SAARTHI26_Team_Neural_Nexus.pdf`)

The deck was written before the build. Where it and the measured results disagree, use these statements (pre-Phase-10
review, 27 Oct 2026; sources in brackets). The PDF itself is not edited.

## Research demo, not a deployed surveillance system

The deck names police and investigation teams as the audience. Say instead that Rewind is a research prototype
evaluated on scripted, consented MEVA footage, with no face recognition, re-identification or cross-camera linking,
and that it is not deployable as is: the model card of its appearance model (OpenCLIP ViT-B-16 `laion2b_s34b_b88k`)
states *"Any deployed use case of the model - whether commercial or not - is currently out of scope"* and *"Certain use
cases which would fall under the domain of surveillance and facial recognition are always out-of-scope regardless of
performance of the model."* (huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K, read 27 Oct 2026). Real use would
need a different model, a legal/ethical review and evaluation on real footage.

## Speed ("Fast at scale / own C++ index, built for speed")

Replace with measured numbers:
- **End-to-end query latency at the frozen config: median 36 ms, p95 39 ms** (20 queries over HTTP, one local GPU)
  [`results/reports/api_frozen.md`].
- **At our size, exact search is faster than the HNSW setting we ship:** appearance branch, 175k vectors, HNSW at
  ef 800 (recall@200 0.995 vs exact) takes about 26 ms per query vs about 19 ms for exact NumPy brute force
  [`results/reports/scaling.md`, merged appearance branch]. On dev with the split filter: 27.6 vs 12.0 ms
  [`results/reports/ablation_8.1.md`].
- **No speed advantage at larger scale was shown.** The 1M / 2M benchmarks used random vectors, on which HNSW recall
  was 0.02–0.07, so their latencies cannot be compared with exact search. **Do not quote "3 ms vs 106 ms at 1M".**
- The C++ engine is still the search core (named per-camera indexes, time filters, save/load); say so without a
  speed claim.

## Accuracy

Quote the test median **with its CI and the hit count**: 1,173 s [376, 61,036] to the first correct clip, 13 of 24
specific events found, vs 30,507 s sequential and 2,977 s for a witness-time window; no paired CI excludes zero, and
on the mean Rewind loses to the witness and motion baselines [`results/reports/test_results.md`]. Do not quote
ratios of medians. Category metrics are not comparable across systems (60 s baseline segments vs ~10–12 s clips).

## Architecture

Show only what was kept: appearance branch (CLIP B/16 frames + motion crops) → C++ HNSW engine → clips. The action
and event branches and detection crops were evaluated and not kept (see README "What was built").

## Queries

The test queries were drafted by Claude Code agents from labels and keyframes, reviewed and approved by Ashutosh,
and frozen before any search code existed.
