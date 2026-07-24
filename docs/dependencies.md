# Dependencies

## Two install tiers

**Tier 1 — `requirements.txt` covers these 7 tools** with one
`pip install -r requirements.txt`: `enhanced_video_downloader.py`,
`smart_video_converter.py`, `compress_videos.py`, `video_segmenter.py`,
`img_converter.py`, `find_duplicates.py`, `find_media_duplicates.py`.

**Dev tier — `requirements-dev.txt`** adds `pytest` on top of
`requirements.txt`, for running `tests/` (see `CLAUDE.md`'s Testing
section). Not needed to run any CLI tool, only to run the test suite.

**Tier 2 — heavy, separate ML stacks**, each with its own install
instructions in the script's own docstring — not in `requirements.txt`,
deliberately:

| Tool | Stack |
|---|---|
| `face_sorter.py` | `insightface`, `onnxruntime` (or `onnxruntime-directml` for DirectML/Intel Arc GPU) |
| `image_tagger.py` | `torch`, `transformers`, `mediapipe`, `opencv-python` |
| `smart_image_organizer_v2.py` | `insightface`, `onnxruntime`, `torch`, `clip`, `opencv-python`, `scikit-learn` |

`smart_image_organizer_v2.py` pulls in the same ML stack as
`face_sorter.py` and `image_tagger.py` combined (multi-signal clustering:
face embeddings + CLIP + traditional CV features). Don't add it to
`requirements.txt`'s default install without a reason; it roughly doubles
typical install size and time.

Until this doc existed, `requirements.txt` was also missing `rich` and
`imagehash` outright, despite being required by `find_duplicates.py`,
`find_media_duplicates.py`, `compress_videos.py`, and the shared
`dup_finder_core.py`/`common_utils.py` — a plain `pip install -r
requirements.txt` would not have installed everything those tools need.
Fixed as part of this cleanup; flagging here so it's not silently
reintroduced by a future edit that "cleans up" what looks like an unused
import.

## ffmpeg: two different sourcing strategies

- **`compress_videos.py`** requires **system ffmpeg and ffprobe on PATH**
  — it explicitly checks `shutil.which("ffmpeg")` /
  `shutil.which("ffprobe")` and refuses to run without both. This is the
  one tool in the repo where `pip install -r requirements.txt` alone is
  not sufficient.
- **`video_segmenter.py`, `smart_video_converter.py`,
  `find_media_duplicates.py`** get ffmpeg via the `imageio-ffmpeg` pip
  package's bundled binary (`imageio_ffmpeg.get_ffmpeg_exe()`, or
  transitively through `moviepy`) — no system install needed. This was
  verified empirically this session: `find_media_duplicates.py`'s video
  handler ran `moviepy.VideoFileClip(...).write_videofile(...)` on a
  fresh checkout with only the pip packages installed.

If you're debugging "ffmpeg not found" for a specific tool, check which
category it's in before assuming a system install is required.

## The `moviepy<2.0` pin is load-bearing

`smart_video_converter.py` and `find_media_duplicates.py`'s video handler
both import via the `moviepy.editor` API path, which moviepy 2.x removed
(2.x moved `VideoFileClip` to the top-level `moviepy` namespace). Several
files in this repo defend against this with a fallback import:

```python
try:
    from moviepy import VideoFileClip        # moviepy 2.x
except ImportError:
    from moviepy.editor import VideoFileClip  # moviepy <2.0
```

`requirements.txt` still pins `moviepy<2.0` as the source of truth,
because not every consumer has the fallback. Don't bump this pin without
auditing every `from moviepy` / `from moviepy.editor` import in the repo
first.

## Optional, graceful-degradation dependencies

`psutil` (process-priority tuning) and `send2trash` (recycle-bin-safe
delete instead of permanent delete) are both used by
`smart_video_converter.py`, guarded by `try`/`except ImportError` — the
script runs fine without them, just with fewer safety/performance niceties.
Both are listed in `requirements.txt` as recommended-but-not-hard
dependencies.
