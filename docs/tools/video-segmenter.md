# video_segmenter.py

Splits a video into short segments (15s by default, every 45s) via
stream copy — no re-encoding, so it's fast but cut points snap to
keyframes rather than exact times.

**Install**: `requirements.txt` (Tier 1) — ffmpeg via the bundled
`imageio-ffmpeg` binary, no system install needed (see
`docs/dependencies.md`).

## Flags

| Flag | Default | Description |
|---|---|---|
| `input_file` (positional) | required | Path to input video |
| `--segment-duration` | `15.0` | Seconds per segment |
| `--segment-gap` | `45.0` | Seconds between segment start times |
| `--output-dir` | same as input | Output directory |
| `--workers` | `4` | Parallel workers |
| `--retries` | `2` | Max retries per failed segment |

## Examples

```bash
python video_segmenter.py input.mp4                                   # defaults: 15s segments, 45s gap
python video_segmenter.py input.mp4 --segment-duration 10 --segment-gap 40
python video_segmenter.py input.mp4 --output-dir "D:\Clips"
python video_segmenter.py input.mp4 --workers 8                       # faster parallel segmentation
```

## Notes

- Non-destructive by nature — produces new segment files, doesn't modify
  or delete the source, so there's no dry-run flag to look for.
- ESC aborts cleanly mid-batch.
