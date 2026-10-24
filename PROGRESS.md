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
