#!/usr/bin/env bash
# Step 5.1: resumable download of LanguageBind_Video_V1.5_FT weights (pinned revision), checked against the
# sha256 Hugging Face reports for the LFS file. Approved by Ashutosh (26 Oct). Usage: bash scripts/fetch_languagebind.sh
set -euo pipefail
REV=5d53aab2ec8944c4babd16fb30cab845686e296f
SHA=679550de18a870da915eeb6ed32960a632d704d403815633d80b4d31ee21629d
DIR=artifacts/weights/LanguageBind_Video_V1.5_FT
mkdir -p "$DIR"
if [ ! -f "$DIR/pytorch_model.bin" ]; then
  curl -fL --retry 20 --retry-all-errors --retry-delay 5 -C - -o "$DIR/pytorch_model.bin.part" \
    "https://huggingface.co/LanguageBind/LanguageBind_Video_V1.5_FT/resolve/$REV/pytorch_model.bin"
  echo "$SHA  $DIR/pytorch_model.bin.part" | sha256sum -c -
  mv "$DIR/pytorch_model.bin.part" "$DIR/pytorch_model.bin"
fi
echo "done: $DIR/pytorch_model.bin"
