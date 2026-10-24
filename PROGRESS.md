# PROGRESS — Rewind

Deadline: Fri 30 Oct 2026 · Current phase: **0 — Setup**

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
