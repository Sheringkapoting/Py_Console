# find_media_duplicates.py

Same duplicate-detection model as `find_duplicates.py`, extended to
**animated GIF/WebP and video** via frame-sampled perceptual hashing.
Built on the shared `src/scripts/dup_finder_core.py` engine — see
`docs/architecture.md` for the `MediaHandler` pattern.

**Install**: `requirements.txt` (Tier 1) — `pip install -r requirements.txt`

## Flags

| Flag | Default | Description |
|---|---|---|
| `--src` | prompted | Source media folder — repeat for multiple (max 5). Duplicates are detected across all of them combined |
| `--media-type` | `auto` | `animated` (GIF/WebP), `video`, or `auto` (both — writes two reports) |
| `--threshold` | `8` | Mean Hamming distance across sampled frames (0–64). Lower = stricter |
| `--recursive` | off | Include subfolders |
| `--exact-only` | off | Skip near-match pass — byte-identical only |
| `--execute` | off | Move duplicates (default: dry run) |
| `--report` | `<first src>/<type>_report.html` | HTML report path (suffixed per type under `--media-type auto`) |

## Examples

```bash
python find_media_duplicates.py                                # interactive, both media types
python find_media_duplicates.py --src "D:\Media"                # dry run
python find_media_duplicates.py --src "D:\Media" --execute      # move duplicates
python find_media_duplicates.py --media-type animated           # GIF/animated WebP only
python find_media_duplicates.py --media-type video               # video only

# Multiple source folders — detects duplicates across all of them combined
python find_media_duplicates.py --src "D:\Media" --src "E:\Archive" --src "F:\Camera"
```

## Multiple source folders

Pass `--src` up to 5 times (or, interactively, answer "yes" to "Add
another source folder?" after each entry) to scan several folders as one
combined pool — a file in folder A and a file in folder B can be detected
as duplicates of each other. Collection, SHA-256 hashing, and animated/
still-image signature computation all run in parallel across files (and
across folders, when collecting); video signature extraction (via
moviepy/ffmpeg) is capped at a smaller worker count since it isn't
reliably thread-safe at full width. Each duplicate is still moved into
*its own* source folder's `Duplicate/` — never a shared one — so a
single-folder run behaves exactly as before. The summary table and HTML
report both gain a per-file/per-folder breakdown when more than one
source is given; `--media-type auto` still prompts for `--src` once, not
once per media type, regardless of how many folders are entered.

## Notes

- Animated GIF/WebP: samples up to 8 frames by relative position, phashes
  each, compares by mean Hamming distance.
- Video: duration pre-filter (skips comparison if durations differ by
  more than ~5%/2s) then the same frame-sampled phash comparison, via
  `moviepy`.
- `--media-type auto` prompts for `--src` once, not once per media type.
- ESC aborts cleanly mid-scan; when both media types are running, ESC
  skips the remaining type rather than killing the process outright.
