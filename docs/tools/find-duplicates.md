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
| `--src` | prompted | Source image folder |
| `--threshold` | `8` | Perceptual-hash Hamming distance (0–64). Lower = stricter |
| `--recursive` | off | Include subfolders |
| `--exact-only` | off | Skip the perceptual-hash pass — byte-identical only |
| `--execute` | off | Move duplicates (default: dry run, no files moved) |
| `--report` | `<src>/duplicate_report.html` | HTML report output path |

## Examples

```bash
python find_duplicates.py                              # interactive
python find_duplicates.py --src "D:\Photos"             # dry run
python find_duplicates.py --src "D:\Photos" --execute   # move duplicates
python find_duplicates.py --threshold 6                 # stricter matching
python find_duplicates.py --exact-only                  # skip perceptual pass
```

## Notes

- Selects which copy to keep by: watermark-vote (clean copy wins), then
  highest resolution, then smallest file size.
- ESC aborts cleanly mid-scan (see `docs/conventions.md`).
- `.gif`/`.webp` are in this tool's scan set too — it hashes only their
  first frame. For animation-aware duplicate detection, use
  `find_media_duplicates.py` instead.
