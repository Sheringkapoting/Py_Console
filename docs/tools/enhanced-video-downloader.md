# enhanced_video_downloader.py

Extracts/downloads videos from web pages and social platforms without
login, via multi-strategy extraction (direct `<video>`/`<source>` tags,
embedded/linked media URLs, social-platform patterns, streaming
manifests) with `yt-dlp` fallback for YouTube/Facebook/Instagram/
Twitter-X/TikTok. GUI (Tkinter) and CLI in one script.

**Install**: `requirements.txt` (Tier 1) — `pip install -r requirements.txt`

## Flags

| Flag | Default | Description |
|---|---|---|
| `url` (positional) | — | Page to analyze; omit to launch the GUI |
| `-o, --output` | `downloads` | Output directory |
| `-q, --quality` | `best` | `best`, `worst`, `bestvideo`, `worstvideo` (applied when `yt-dlp` is used) |
| `-v, --verbose` | off | Verbose logging |
| `--gui` | off | Force GUI mode even with a URL argument |
| `--batch` | — | File with multiple URLs, one per line |

## Examples

```bash
python enhanced_video_downloader.py                                    # GUI mode
python enhanced_video_downloader.py https://example.com/video-page     # single URL, CLI
python enhanced_video_downloader.py URL -q bestvideo -o "D:\Downloads"
python enhanced_video_downloader.py --batch urls.txt                   # batch CLI
```

## Notes

- Streaming manifests (`.m3u8`, `.mpd`) may need additional tooling
  depending on site-specific restrictions — not guaranteed downloadable
  by this tool alone.
- ESC aborts cleanly mid-download in CLI mode.
