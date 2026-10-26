# Engine provenance

Copied (source only) from https://github.com/ashutoshanthwall/semantic_image_retrieval_using_hnsw_indexing
at commit `6f84dcb1df21f2758fc696c57ca6b4ea797561da` on 2026-10-26, unmodified.

Copied: `CMakeLists.txt`, `README.md`, `include/`, `src/`, `dsa/`, `apps/`, `tests/`, `config/config.json`
(the apps' default config path).
Not copied: `data/`, `evaluation/` (Flickr30k data and results), `scripts/` and `requirements.txt` (the
Flickr Python pipeline and text-encoder server), `web/`, build output. Tests that need those files skip themselves
(`real metadata.json loads`, `testTextEncoder`).

Changes for Rewind start at step 7.1 and are tracked in this repo's git history.

## Rewind changes (step 7.1, 26 Oct)

- `src/data.cpp`, `include/data.h`: `floatCount()` (rows x dim in 64 bits, checked) replaces the int multiply in
  `EmbeddingStore::load` (overflowed past 2^31 floats); `EmbeddingStore::setRows()` copies a caller's buffer
  (Python bindings) and records its SHA-256 like `load()`.
- New `include/collection.h`, `src/collection.cpp`: `NamedIndex` (store + HNSW + per-row times + a mutex) with
  time-range filtering (oversample, exact fallback) and `IndexCollection` (named indexes). HNSW internals unchanged.
- New `python/rewind_engine.cpp` (pybind11, built with `-DREWIND_PYTHON=ON`); `CMakeLists.txt` links Threads and
  builds the module optionally.
- New `tests/test_collection.cpp` (same `check()` harness; registered in `test_main.cpp`, `tests.h`).
