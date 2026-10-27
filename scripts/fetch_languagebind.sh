#!/usr/bin/env bash
# Step 5.1: resumable download of LanguageBind_Video_V1.5_FT (pinned revision): the weights AND the config and
# tokenizer files the loader needs (rewind/embed/languagebind.py), each checked against its sha256. Without config.json
# / the tokenizer files transformers silently falls back to defaults (pre-Phase-10 review, repro-1), so all are fetched.
# Hashes of the small files: downloaded at this revision on 27 Oct 2026 and identical to the files used for every
# reported action-branch number. Approved by Ashutosh (26 Oct; small files 27 Oct). Usage: bash scripts/fetch_languagebind.sh
set -euo pipefail
REV=5d53aab2ec8944c4babd16fb30cab845686e296f
DIR=artifacts/weights/LanguageBind_Video_V1.5_FT
URL="https://huggingface.co/LanguageBind/LanguageBind_Video_V1.5_FT/resolve/$REV"
mkdir -p "$DIR"
fetch() {   # fetch <file> <sha256>: skip if present and correct; download to .part, check, then rename
  local f=$1 sha=$2
  if [ -f "$DIR/$f" ] && echo "$sha  $DIR/$f" | sha256sum -c --status -; then echo "ok: $f"; return; fi
  curl -fL --retry 20 --retry-all-errors --retry-delay 5 -C - -o "$DIR/$f.part" "$URL/$f"
  echo "$sha  $DIR/$f.part" | sha256sum -c -
  mv "$DIR/$f.part" "$DIR/$f"
}
fetch config.json             da88f2d1b978ddf53fd4a49b7698312c0af804c5f19fe600c12fdff3e4a28016
fetch tokenizer.json          d8b124290bc4bcd18cd3f72747f525e2a1d8c266cf3089e52da38ee417564ac5
fetch vocab.json              5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363
fetch merges.txt              f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85
fetch tokenizer_config.json   c098282a97423d1c621c4e1e8f30be5d82d6071d90345d437323e85b827e32eb
fetch special_tokens_map.json c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac
fetch pytorch_model.bin       679550de18a870da915eeb6ed32960a632d704d403815633d80b4d31ee21629d
echo "done: $DIR"
