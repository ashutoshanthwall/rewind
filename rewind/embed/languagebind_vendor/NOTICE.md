# Vendored third-party code: LanguageBind (video) + transformers v4.30.2 CLIP pieces

- **LanguageBind source:** https://github.com/PKU-YuanGroup/LanguageBind at commit
  `7070c53375661cdb235801176b564b45f96f0648`, files `languagebind/video/configuration_video.py` and
  `languagebind/video/modeling_video.py`. **Copyright** (c) 2023 PKU-YUAN's Group. **License:** MIT (full text in
  `LICENSE`). Paper: Zhu et al., *LanguageBind*, ICLR 2024. The training data (VIDAL-10M) is CC-BY-NC-4.0; Rewind only
  runs inference with the released weights (model card license: MIT).
- **Changes by Rewind** (in `modeling_video.py` only; `configuration_video.py` is byte-identical):
  1. `from peft import ...` moved inside `convert_to_lora`, after its `lora_r == 0` early return (V1.5_FT has
     `lora_r: 0`, so peft is never imported).
  2. `CLIPMLP, CLIPAttention, CLIPTextEmbeddings, _expand_mask, CLIPOutput, clip_loss` are imported from
     `clip_v4_30_2.py` instead of the installed transformers. `CLIPVisionEmbeddings` is no longer imported (the file
     defines its own class of that name, which shadowed the import).
  No model code was changed.
- **`clip_v4_30_2.py`:** copied unchanged from transformers v4.30.2 `src/transformers/models/clip/modeling_clip.py`
  (sha256 `1feb4c05e271dd181aa260eac13b10a5fcc1bd4d3cac68506e2a9bfee4024e49`, raw GitHub at tag `v4.30.2`):
  `_expand_mask`, `contrastive_loss`, `clip_loss`, `CLIPOutput`, `CLIPTextEmbeddings`, `CLIPAttention`, `CLIPMLP`,
  plus the imports they need. **License:** Apache-2.0 (full text in `LICENSE-transformers`), Copyright 2021 The
  OpenAI Team Authors and The HuggingFace Team.
- **Why vendored:** the LanguageBind repo has no installable package (`import languagebind` imports every modality and
  pulls in decord, pytorchvideo, torchaudio, peft) and pins transformers 4.30.2, which needs huggingface-hub < 1.0
  (Rewind has 2.3). Newer transformers removed `_expand_mask` and dropped the `causal_attention_mask` argument of
  `CLIPAttention.forward`: the text tower would then run without its causal mask and give wrong embeddings without an
  error. The v4.30.2 copies keep the exact attention code the model was trained with
  (`tests/test_languagebind.py` checks the causal mask and window grouping on a tiny random model).
- **Weights:** `LanguageBind/LanguageBind_Video_V1.5_FT` at Hugging Face commit and sha256 in `configs/config.yaml`
  (`action.languagebind`), fetched by `scripts/fetch_languagebind.sh`.
