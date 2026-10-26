# Engine provenance

Copied (source only) from https://github.com/ashutoshanthwall/semantic_image_retrieval_using_hnsw_indexing
at commit `6f84dcb1df21f2758fc696c57ca6b4ea797561da` on 2026-10-26, unmodified.

Copied: `CMakeLists.txt`, `README.md`, `include/`, `src/`, `dsa/`, `apps/`, `tests/`, `config/config.json`
(the apps' default config path).
Not copied: `data/`, `evaluation/` (Flickr30k data and results), `scripts/` and `requirements.txt` (the
Flickr Python pipeline and text-encoder server), `web/`, build output. Tests that need those files skip themselves
(`real metadata.json loads`, `testTextEncoder`).

Changes for Rewind start at step 7.1 and are tracked in this repo's git history.
