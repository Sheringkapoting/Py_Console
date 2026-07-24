# smart_video_converter.py

Converts video to animated **GIF or WebP** (or does AI-assisted format
selection) with frame-rate/size/quality/time-range control and batch
processing. Its module docstring says "Video to GIF Converter" but it
also handles WebP output — see `docs/architecture.md`'s note on stale
docstrings not matching actual capability.

**Install**: `requirements.txt` (Tier 1) — `pip install -r requirements.txt`

## Flags

| Flag | Default | Description |
|---|---|---|
| `input` (positional) / `--input` | — | Video file or directory (directory = batch mode) |
| `-o, --output` | inferred | Output file or directory |
| `--format` | — | `gif`, `webp`, or `ai` (AI-assisted selection) |
| `--workers` | — | Max worker threads for batch mode |
| `--fps` | `10` | Frames per second |
| `--width` | — | Target width in px (max 800, aspect preserved) |
| `--start` / `--end` | `0` / full | Time range in seconds |
| `--quality` | `85` | 1–100 |
| `--no-optimize` | off | Disable GIF size optimization |
| `--progress` | `simple` | `simple`, `verbose`, or `none` |
| `--keepSourceFile` | `true` | `true`/`false` — preserve source after conversion |
| `--no-recycle-bin` | off | Permanently delete instead of recycle-bin (needs `send2trash`, optional dep) |
| `-v, --verbose` | off | Verbose logging |

`--params`/`--list-params` and `--delete-source` still parse (backward
compatibility) but are hidden from `--help` — `argparse.SUPPRESS`. Use
`--keepSourceFile false` instead of `--delete-source`.

## Examples

```bash
python smart_video_converter.py input.mp4 --format gif --fps 15
python smart_video_converter.py input.mp4 -o out.webp --format webp --quality 90
python smart_video_converter.py "D:\Videos" --format gif --workers 4   # batch
```

## Notes

- `psutil` (process priority) and `send2trash` (recycle-bin delete) are
  both optional — the script degrades gracefully without them, see
  `docs/dependencies.md`.
- `--keepSourceFile` defaults to preserving the source; only
  `--no-recycle-bin` makes a delete permanent rather than recoverable.
