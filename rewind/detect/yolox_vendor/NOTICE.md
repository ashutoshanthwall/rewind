# Vendored third-party code: YOLOX

- **Source:** PyPI sdist `yolox-0.3.0.tar.gz`, sha256 `1b8ee68846434408c40074f0806259fe3b9387181e562dbc435ea666533d8d08`
  (upstream https://github.com/Megvii-BaseDetection/YOLOX).
- **Copyright:** (c) Megvii Inc. and its affiliates. **License:** Apache License 2.0 (full text in `LICENSE`, taken from
  the upstream repository at commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`).
- **Files** (copied from `yolox/models/` and `yolox/utils/`): darknet.py, network_blocks.py, yolo_pafpn.py,
  yolo_head.py, yolox.py, losses.py (models); boxes.py, compat.py (utils).
- **Changes by Rewind** (Apache-2.0 §4(b)): in `yolo_head.py` only, the imports were changed: `loguru` was replaced by
  the stdlib `logging` logger, and `from yolox.utils import ...` by relative imports of `boxes` and `compat`.
  `__init__.py` and this file are new. No model code was changed.
- **Why vendored:** the PyPI package compiles a C++ evaluation op against torch at build time and pins export
  dependencies (onnx 1.8.1) that have no Python 3.11 wheels. Rewind only needs inference (step 3.3, PROGRESS.md).
- **Weights:** `yolox_s.pth` from the upstream release `0.1.1rc0`, sha256 in `configs/config.yaml`
  (`detection.weights_sha256`).
