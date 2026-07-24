# find_media_duplicates.py

Same duplicate-detection model as `find_duplicates.py`, extended to
**animated GIF/WebP and video** via frame-sampled perceptual hashing.
Built on the shared `src/scripts/dup_finder_core.py` engine — see
`docs/architecture.md` for the `MediaHandler` pattern.

**Install**: `requirements.txt` (Tier 1) — `pip install -r requirements.txt`

## Flags

| Flag | Default | Description |
|---|---|---|
| `--src` | prompted | Source media folder |
| `--media-type` | `auto` | `animated` (GIF/WebP), `video`, or `auto` (both — writes two reports) |
| `--threshold` | `8` | Mean Hamming distance across sampled frames (0–64). Lower = stricter |
| `--recursive` | off | Include subfolders |
| `--exact-only` | off | Skip near-match pass — byte-identical only |
| `--execute` | off | Move duplicates (default: dry run) |
| `--report` | `<src>/<type>_report.html` | HTML report path (suffixed per type under `--media-type auto`) |

## Examples

```bash
python find_media_duplicates.py                                # interactive, both media types
python find_media_duplicates.py --src "D:\Media"                # dry run
python find_media_duplicates.py --src "D:\Media" --execute      # move duplicates
python find_media_duplicates.py --media-type animated           # GIF/animated WebP only
python find_media_duplicates.py --media-type video               # video only
```

## Notes

- Animated GIF/WebP: samples up to 8 frames by relative position, phashes
  each, compares by mean Hamming distance.
- Video: duration pre-filter (skips comparison if durations differ by
  more than ~5%/2s) then the same frame-sampled phash comparison, via
  `moviepy`.
- `--media-type auto` prompts for `--src` once, not once per media type.
- ESC aborts cleanly mid-scan; when both media types are running, ESC
  skips the remaining type rather than killing the process outright.
