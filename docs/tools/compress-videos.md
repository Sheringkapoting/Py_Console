# compress_videos.py

Batch video compression with automatic hardware-encoder detection
(NVENC/QSV/AMF/VideoToolbox) and codec selection (SVT-AV1 > H.265 >
H.264). Quality measured via VMAF, falling back to SSIM/PSNR if the
local FFmpeg build lacks `libvmaf`.

**Install**: `requirements.txt` (Tier 1) — **plus system FFmpeg/FFprobe on
PATH**, which this tool specifically requires (unlike most other video
tools in this repo — see `docs/dependencies.md`'s "ffmpeg: two different
sourcing strategies").

## Flags

| Flag | Default | Description |
|---|---|---|
| `--input` | prompted | Source video folder |
| `--codec` | auto | Codec choice (see `CODEC_PROFILES`) |
| `--crf` | codec default | Constant rate factor (quality) |
| `--preset` | codec default | Encoder speed/quality preset |
| `--hw` | auto-detect | Hardware encoder override |
| `--output` | in place | Output folder (default: replace originals) |
| `--backup` | off | Back up originals before replacing |
| `--backup-dir` | `<input>/_originals` | Custom backup location |
| `--execute` | off | Skip the dry-run prompt and compress immediately |
| `--dry-run` | off | Preview only — see Notes below, this is not simply the inverse of `--execute` |
| `--no-log` | off | Skip writing a JSON log file |

## Examples

```bash
python compress_videos.py                                    # interactive
python compress_videos.py --input "D:\Videos"                # interactive dry-run-style flow
python compress_videos.py --codec av1 --crf 32 --execute
python compress_videos.py --codec h265 --hw auto --execute
python compress_videos.py --dry-run
```

## Notes

- Unlike `find_duplicates.py`/`find_media_duplicates.py`/`face_sorter.py`,
  this tool has **both** `--execute` and `--dry-run` as explicit flags
  (neither is a silent default) — check `args._dry_set` in the source if
  you're relying on exact behavior when neither flag is passed. See
  `docs/conventions.md` for the three different safety patterns in use
  across this repo.
- ESC aborts cleanly mid-batch.
