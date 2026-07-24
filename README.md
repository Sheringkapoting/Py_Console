<div align="center">

<img alt="Build" src="https://img.shields.io/badge/build-passing-brightgreen" />
<img alt="Coverage" src="https://img.shields.io/badge/coverage-N/A-lightgrey" />
<img alt="Version" src="https://img.shields.io/badge/version-dev-blue" />

</div>

# Py_Console Utilities

A collection of standalone, directly-runnable Python CLI tools for media
processing: duplicate detection, format conversion, compression, face-based
sorting, and semantic tagging. Each tool is self-contained — run it with
`python <tool>.py --help`.

For architecture, coding conventions, and dependency notes, see
[`docs/`](docs/). For how Claude Code (or any agent) should work in this
repo, see [`CLAUDE.md`](CLAUDE.md).

---

## Tools

| Script | Purpose | Docs |
|---|---|---|
| `find_duplicates.py` | Detects duplicate **images** (exact SHA-256 + perceptual-hash near-match), moves duplicates to `Duplicate/`, generates an HTML report. | [tools/find-duplicates.md](docs/tools/find-duplicates.md) |
| `find_media_duplicates.py` | Same duplicate-detection model, extended to **animated GIF/WebP and video** via frame-sampled perceptual hashing. Shares its engine with `find_duplicates.py` via `src/scripts/dup_finder_core.py`. | [tools/find-media-duplicates.md](docs/tools/find-media-duplicates.md) |
| `img_converter.py` | Converts `.webp` / `.png` / `.jpeg` / `.heic` images to `.jpg` (skips animated WebP), with decompression-bomb protection. | [tools/img-converter.md](docs/tools/img-converter.md) |
| `compress_videos.py` | Batch video compression with hardware-encoder detection (NVENC/QSV/AMF/VideoToolbox), SVT-AV1/H.265/H.264, VMAF quality measurement. | [tools/compress-videos.md](docs/tools/compress-videos.md) |
| `smart_video_converter.py` | Converts video to animated **GIF or WebP** with frame-rate/size/quality/time-range control, batch processing. | [tools/smart-video-converter.md](docs/tools/smart-video-converter.md) |
| `video_segmenter.py` | Splits video into short segments at a configurable duration/gap via stream copy (no re-encode). | [tools/video-segmenter.md](docs/tools/video-segmenter.md) |
| `face_sorter.py` | Sorts images into per-person folders using **InsightFace (ArcFace/SCRFD, `buffalo_l` models)**. | [tools/face-sorter.md](docs/tools/face-sorter.md) |
| `smart_image_organizer_v2.py` | Clusters images by **combined signals** (not face-only) and renames each cluster sequentially so related images sort together. | [tools/smart-image-organizer-v2.md](docs/tools/smart-image-organizer-v2.md) |
| `image_tagger.py` | Semantic tagging via **CLIP** (scene/theme classification) + **MediaPipe Pose** (posture labels); outputs a structured `tags.json`. | [tools/image-tagger.md](docs/tools/image-tagger.md) |
| `enhanced_video_downloader.py` | Extracts/downloads videos from web pages and social platforms without login, with GUI + CLI, `yt-dlp` fallback. | [tools/enhanced-video-downloader.md](docs/tools/enhanced-video-downloader.md) |

---

## Installation

- Prerequisites: Python 3.8+ on Windows, macOS, or Linux
- Recommended: use a virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

Some tools have additional install steps (InsightFace/ArcFace for
`face_sorter.py`, CLIP/MediaPipe for `image_tagger.py`, hardware FFmpeg
builds for `compress_videos.py`). Each script documents its own
requirements at the top of the file and in `--help`; see
[`docs/dependencies.md`](docs/dependencies.md) for the shared/pinned
dependencies and why they're pinned.

---

## Usage

Every tool is self-documenting:

```bash
python find_duplicates.py --help
python face_sorter.py --help
```

Each tool also has a short doc page under [`docs/tools/`](docs/tools/)
(linked in the table above) with its flags, examples, and any
tool-specific caveats worth knowing before running it.

Most tools follow the same interaction model — see
[`docs/conventions.md`](docs/conventions.md) for the shared patterns
(dry-run by default, `--execute` to act, ESC to abort, HTML reports), and
[`docs/runbooks/`](docs/runbooks/) for step-by-step procedures (adding a
new tool, extending the duplicate finder to a new media type, recovering
from a large-file-in-git-history situation).

---

## Contribution Guidelines

- Use feature branches and clear, atomic commits.
- Follow PEP 8 for code style; keep functions cohesive and well-named.
- Include docstrings and helpful log messages where appropriate.
- For new CLI options, update the script's own help text.
- Open an issue before large refactors to discuss scope and impact.

### Code of Conduct

We aim to foster an open, respectful community:

- Be kind and constructive in discussions and reviews.
- No harassment, discrimination, or disrespectful behavior.
- Assume positive intent; focus on technical merit.
- Report issues to maintainers for prompt attention.

---

## License

No explicit license file is present. By default, all rights are reserved to the project owner. If you intend to open-source the project, please add a `LICENSE` file (e.g., MIT, Apache-2.0) and update this section accordingly.

---

## Maintainers & Contact

- Maintainers: TBD
- Contact: Please add your preferred contact (email or GitHub) here.
