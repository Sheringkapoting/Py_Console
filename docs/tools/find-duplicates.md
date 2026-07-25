# find_duplicates.py

Detects duplicate **still images** (SHA-256 exact match + perceptual-hash
near-match) and moves duplicates into `<src>/Duplicate/`. Dry-run by
default; generates a dark-theme HTML report either way.

**Install**: `requirements.txt` (Tier 1) — `pip install -r requirements.txt`

**Architecture note**: this tool has its own independent implementation of
duplicate-grouping/HTML-reporting, separate from `dup_finder_core.py`.
See `docs/architecture.md` for why it wasn't migrated onto the shared
engine when that was extracted.

## Flags

| Flag | Default | Description |
|---|---|---|
| `--src` | prompted | Source image folder — repeat for multiple (max 5). Duplicates are detected across all of them combined |
| `--threshold` | `8` | Perceptual-hash Hamming distance (0–64). Lower = stricter |
| `--recursive` | off | Include subfolders |
| `--exact-only` | off | Skip the perceptual-hash pass — byte-identical only |
| `--execute` | off | Move duplicates (default: dry run, no files moved) |
| `--report` | `<first src>/duplicate_report.html` | HTML report output path |

## Examples

```bash
python find_duplicates.py                              # interactive
python find_duplicates.py --src "D:\Photos"             # dry run
python find_duplicates.py --src "D:\Photos" --execute   # move duplicates
python find_duplicates.py --threshold 6                 # stricter matching
python find_duplicates.py --exact-only                  # skip perceptual pass

# Multiple source folders — detects duplicates across all of them combined
python find_duplicates.py --src "D:\Photos" --src "D:\Backup" --src "E:\Camera"
```

## Multiple source folders

Pass `--src` up to 5 times (or, when run interactively without `--src`,
answer "yes" to "Add another source folder?" after each entry) to scan
several folders as one combined pool — a file in folder A and a file in
folder B can be detected as duplicates of each other, not just duplicates
within their own folder. Collection and hashing/pHash computation run in
parallel across files (and across folders, when collecting). Each
duplicate is still moved into *its own* source folder's `Duplicate/` —
never a shared one — so a single-folder run behaves exactly as before,
and a multi-folder run keeps each folder's cleanup self-contained. The
summary table and HTML report both gain a per-file/per-folder breakdown
when more than one source is given; with a single `--src`, console and
report output are unchanged.

## Notes

- Selects which copy to keep by: watermark-vote (clean copy wins), then
  highest resolution, then smallest file size.
- ESC aborts cleanly mid-scan (see `docs/conventions.md`).
- `.gif`/`.webp` are in this tool's scan set too — it hashes only their
  first frame. For animation-aware duplicate detection, use
  `find_media_duplicates.py` instead.
