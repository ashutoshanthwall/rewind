"""Step 0.4: download the selected MEVA clips into data/raw/, verify size and content, then make data/raw/ read-only.

- Reads data/selection.txt (lines "s3_key<TAB>size_bytes"; '#' lines are comments).
- Resumable: a file whose final name exists with the expected size is skipped. Downloads go to
  "<name>.part" and are renamed only after size AND S3 ETag match, so an interrupted or corrupt download
  never sits under a final name. Re-running simply continues.
- Content check: the MEVA objects carry no stored checksum, and most have multipart ETags ("<md5>-<parts>"),
  which `aws s3 cp` cannot check. So we recompute the ETag locally (MD5 of 8 MiB part MD5s, or plain MD5
  for single-part objects) and compare it with the bucket listing.
- A final file with the WRONG size is never deleted or overwritten: the script stops and says so.
- Progress is printed and appended to a log file.
- When every file is present and verified, files become 0444 and the directory 0555 (evidence untouched).

Usage:
  uv run python scripts/download.py                     download / resume, verify, lock
  uv run python scripts/download.py --verify-existing   only re-check size + ETag of files already in --dest
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUCKET = "mevadata-public-01"  # dataset fact: public MEVA bucket (CLAUDE.md 1.5)
# How the MEVA uploader split files. Checked 2026-10-25 on the 1,073 objects under drops-123-r13/2018-03-07/:
# every multipart ETag's part count equals ceil(size / 8 MiB). A wrong guess cannot pass silently: it would
# make every ETag mismatch, and etag_of() also refuses a part count that doesn't fit this size.
PART_SIZE = 8 * 1024 * 1024


def parse_selection(path: Path) -> list[tuple[str, int]]:
    """Return [(s3_key, size_bytes)]; raise on malformed lines or duplicate file names."""
    entries = []
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 2 or not fields[1].isdigit() or int(fields[1]) <= 0:
            raise ValueError(f"{path}:{n}: expected 's3_key<TAB>positive size', got {line!r}")
        entries.append((fields[0], int(fields[1])))
    if not entries:
        raise ValueError(f"{path} lists no files")
    names = [Path(key).name for key, _ in entries]
    dups = {n for n in names if names.count(n) > 1}
    if dups:
        raise ValueError(f"Duplicate file names in {path} (data/raw/ is flat): {sorted(dups)[:5]}")
    return entries


def plan(entries: list[tuple[str, int]], dest: Path) -> list[tuple[str, int]]:
    """Return the entries still to download. Raise if a final file exists with the wrong size."""
    todo = []
    for key, size in entries:
        final = dest / Path(key).name
        if final.exists():
            actual = final.stat().st_size
            if actual != size:
                raise RuntimeError(f"{final} exists with {actual} bytes, selection says {size}. "
                                   "Not touching it: inspect and remove it by hand if it is a bad download.")
            continue
        todo.append((key, size))
    return todo


def etag_of(path: Path, s3_etag: str, part_size: int = PART_SIZE) -> str:
    """Recompute the S3 ETag of a local file in the same form as s3_etag (plain MD5, or multipart '<md5>-<n>')."""
    if "-" not in s3_etag:
        md5 = hashlib.md5()
        with path.open("rb") as f:
            while block := f.read(part_size):
                md5.update(block)
        return md5.hexdigest()
    n_parts = int(s3_etag.split("-")[1])
    expected_parts = -(-path.stat().st_size // part_size)
    if n_parts != expected_parts:
        raise ValueError(f"{path}: S3 ETag has {n_parts} parts, but {part_size}-byte parts give {expected_parts}; "
                         "the uploader's part size is unknown for this file, cannot verify")
    digests = []
    with path.open("rb") as f:
        while block := f.read(part_size):
            digests.append(hashlib.md5(block).digest())
    return f"{hashlib.md5(b''.join(digests)).hexdigest()}-{len(digests)}"


def check_etag(path: Path, s3_etag: str) -> None:
    local = etag_of(path, s3_etag)
    if local != s3_etag:
        raise RuntimeError(f"Content mismatch for {path}: local ETag {local}, S3 ETag {s3_etag}")


def list_etags(keys: list[str]) -> dict[str, str]:
    """ETag (quotes stripped) for each key, from one bucket listing per folder."""
    etags = {}
    for prefix in sorted({k.rsplit("/", 1)[0] + "/" for k in keys}):
        out = subprocess.run(["aws", "s3api", "list-objects-v2", "--no-sign-request", "--bucket", BUCKET,
                              "--prefix", prefix, "--query", "Contents[].[Key,ETag]", "--output", "json"],
                             check=True, capture_output=True, text=True).stdout
        etags.update({k: e.strip('"') for k, e in json.loads(out) or []})
    missing = [k for k in keys if k not in etags]
    if missing:
        raise RuntimeError(f"{len(missing)} selected keys not found in the bucket listing, e.g. {missing[:3]}")
    return {k: etags[k] for k in keys}


def aws_copy(key: str, out: Path) -> None:
    subprocess.run(["aws", "s3", "cp", "--no-sign-request", "--only-show-errors", f"s3://{BUCKET}/{key}", str(out)],
                   check=True)


def download_one(key: str, size: int, s3_etag: str, dest: Path, copy=aws_copy) -> Path:
    """Fetch key to <name>.part, check size and ETag, then rename to the final name. On mismatch the .part stays."""
    final = dest / Path(key).name
    part = final.with_name(final.name + ".part")
    copy(key, part)
    actual = part.stat().st_size
    if actual != size:
        raise RuntimeError(f"Size mismatch for {key}: got {actual} bytes, expected {size}. "
                           f"Left as {part}; re-run to retry.")
    check_etag(part, s3_etag)
    part.rename(final)
    return final


def stray_files(dest: Path, entries: list[tuple[str, int]]) -> list[str]:
    return sorted({f.name for f in dest.iterdir()} - {Path(k).name for k, _ in entries})


def make_read_only(dest: Path, entries: list[tuple[str, int]]) -> None:
    for key, _ in entries:
        (dest / Path(key).name).chmod(0o444)
    dest.chmod(0o555)


def log(msg: str, log_path: Path) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with log_path.open("a") as f:
        f.write(line + "\n")


def verify_existing(entries: list[tuple[str, int]], dest: Path, log_path: Path) -> None:
    """Read-only: check that every selected file is present with the right size and ETag. Changes nothing."""
    missing = plan(entries, dest)
    if missing:
        raise RuntimeError(f"{len(missing)} selected files are missing from {dest}, e.g. {missing[0][0]}")
    etags = list_etags([k for k, _ in entries])
    t0 = time.monotonic()
    for i, (key, _) in enumerate(entries, start=1):
        check_etag(dest / Path(key).name, etags[key])
        if i % 50 == 0 or i == len(entries):
            log(f"ETag check {i}/{len(entries)} ok ({time.monotonic() - t0:.0f} s)", log_path)
    log(f"verified size + ETag of all {len(entries)} files in {dest}", log_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--selection", type=Path, default=REPO_ROOT / "data" / "selection.txt")
    p.add_argument("--dest", type=Path, default=REPO_ROOT / "data" / "raw")
    p.add_argument("--log", type=Path, default=REPO_ROOT / "data" / "download.log")
    p.add_argument("--min-free-gb", type=float, default=10.0,
                   help="free space that must remain on the disk after the download (safety margin)")
    p.add_argument("--verify-existing", action="store_true",
                   help="only check size + ETag of the files already in --dest; download and chmod nothing")
    args = p.parse_args()

    entries = parse_selection(args.selection)
    if args.verify_existing:
        verify_existing(entries, args.dest, args.log)
        return

    args.dest.mkdir(parents=True, exist_ok=True)
    todo = plan(entries, args.dest)
    total_bytes = sum(s for _, s in entries)
    todo_bytes = sum(s for _, s in todo)
    log(f"selection: {len(entries)} files, {total_bytes / 1e9:.2f} GB; to download: {len(todo)} files, "
        f"{todo_bytes / 1e9:.2f} GB -> {args.dest}", args.log)

    if todo:
        if not (args.dest.stat().st_mode & 0o200):
            raise PermissionError(f"{args.dest} is read-only but {len(todo)} files are missing. "
                                  f"If that is intended, run: chmod u+w {args.dest}")
        free = shutil.disk_usage(args.dest).free
        if free - todo_bytes < args.min_free_gb * 1e9:
            raise RuntimeError(f"Not enough disk: {free / 1e9:.1f} GB free, need {todo_bytes / 1e9:.1f} GB "
                               f"+ {args.min_free_gb} GB margin")
    etags = list_etags([k for k, _ in todo]) if todo else {}

    done_bytes = total_bytes - todo_bytes
    t0 = time.monotonic()
    fetched = 0
    for i, (key, size) in enumerate(todo, start=1):
        final = download_one(key, size, etags[key], args.dest)
        done_bytes += size
        fetched += size
        rate = fetched / max(time.monotonic() - t0, 1e-9)
        eta_min = (total_bytes - done_bytes) / rate / 60
        log(f"[{i}/{len(todo)}] {final.name} {size / 1e6:.1f} MB, ETag ok | "
            f"{done_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB | {rate / 1e6:.1f} MB/s | ETA {eta_min:.0f} min", args.log)

    # Final check of the whole selection, then lock it. Stray files would stay writable inside the
    # evidence folder, so refuse to lock instead of only warning.
    if plan(entries, args.dest):
        raise RuntimeError("Files still missing after the download loop; re-run.")
    stray = stray_files(args.dest, entries)
    if stray:
        raise RuntimeError(f"{len(stray)} files in {args.dest} are not in the selection: {stray[:5]}. "
                           "Move or remove them by hand, then re-run to lock the folder.")
    make_read_only(args.dest, entries)
    log(f"verified {len(entries)} files ({total_bytes / 1e9:.2f} GB); {args.dest} is now read-only", args.log)


if __name__ == "__main__":
    main()
