<div align="center">

<img alt="Build" src="https://img.shields.io/badge/build-passing-brightgreen" />
<img alt="Coverage" src="https://img.shields.io/badge/coverage-N/A-lightgrey" />
<img alt="Version" src="https://img.shields.io/badge/version-dev-blue" />

</div>

# Py_Console Utilities

A collection of focused, production-ready Python console tools for media processing and automation. This project currently includes four primary scripts:

- `enhanced_video_downloader.py` — Extracts and downloads videos from web pages and social platforms without login requirements, with GUI and CLI.
- `video_to_gif.py` — Converts videos into optimized GIF or WebP animations with batch and concurrent processing.
- `secure_sort_by_known_faces.py` — Sorts/moves images into folders based on matches to known faces.
- `convert_webp_to_jpg.py` — Converts `.webp`, `.png`, `.jpeg` images to `.jpg` and deletes sources safely.

---

## Project Description

Py_Console provides robust, cross-platform utilities designed for practical workflows:

- Enhanced video extraction with fallbacks, quality selection, and a Tkinter GUI.
- High-quality GIF/WebP generation with file-size optimization, batch conversions, and integrity checks.
- Secure, deterministic image sorting by face recognition with configurable tolerance and parallel workers.
- Fast and safe image conversion to `.jpg` with decompression-bomb protections.

These scripts are self-contained and usable independently. They share consistent logging, sensible defaults, and careful error handling.

---

## Installation

- Prerequisites: Python 3.8+ on Windows, macOS, or Linux
- Recommended: use a virtual environment

```bash
# From the project root
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Dependencies

Resolved via `requirements.txt`:

- Downloader and scraping: `requests`, `beautifulsoup4`, `yt-dlp`, `urllib3`
- Progress/UI: `tqdm` (progress bars), `tkinter` (built into Python)
- Video/GIF: `moviepy<2.0`, `imageio`, `imageio-ffmpeg`, `numpy`, `Pillow`
- Face recognition: `face-recognition`, `face-recognition-models`

Notes:

- `ffmpeg` is provided via `imageio-ffmpeg`. If you prefer a system `ffmpeg`, ensure it’s on `PATH`.
- `tkinter` ships with most Python distributions; no pip install needed.

---

## Configuration & Environment Setup

- Windows is fully supported; Linux/macOS work as well.
- For `face-recognition`, installation may require platform-specific prerequisites:
  - On Windows, prebuilt wheels are recommended; if compilation fails, consider using `conda` or a prebuilt binary.
  - On Linux/macOS, ensure common build tooling is present (e.g., `cmake`, compiler toolchain) if wheels are unavailable.
- Ensure sufficient disk space and permissions in output directories.
- Optional: install system `ffmpeg` for advanced scenarios; otherwise `imageio-ffmpeg` is used.

---

## Usage Examples

### 1) Enhanced Video Downloader (`enhanced_video_downloader.py`)

Purpose

- Extract and download videos from web pages and social platforms without login requirements. Offers a Tkinter GUI and a CLI with batch support and quality selection. Uses multi-strategy extraction (HTML tags, embedded links, social patterns, streaming manifests) and `yt-dlp` fallback.

Capabilities

- Direct `<video>`/`<source>` extraction
- Embedded/linked media (e.g., `.mp4` URLs)
- Social platforms via patterns and `yt-dlp` fallback (YouTube, Facebook, Instagram, Twitter/X, TikTok)
- Streaming manifests (HLS `.m3u8`, DASH `.mpd`) when downloadable
- GUI mode with URL analysis, list management, and download controls

Parameters

| Parameter | Type | Default | Required | Description |
|---|---|---:|:---:|---|
| `url` (positional) | `str` | none | Optional | Target web page to analyze. Omit to launch GUI. |
| `-o, --output` | `str` (path) | `downloads` | No | Directory for downloaded files. |
| `-q, --quality` | `str` | `best` | No | Quality preference: `best`, `worst`, `bestvideo`, `worstvideo`. Applied when `yt-dlp` is used. |
| `-v, --verbose` | `flag` | `False` | No | Enable verbose logging. |
| `--gui` | `flag` | `False` | No | Force GUI mode even if `url` is provided. |
| `--batch` | `str` (path) | none | No | File with multiple URLs (one per line) for batch CLI downloads. |

Supported Platforms

- Works across general websites with direct media, and via `yt-dlp` for popular platforms such as YouTube, Facebook, Instagram, Twitter/X, TikTok. Streaming manifests (`.m3u8`, `.mpd`) may require additional tooling depending on site restrictions.

Output Formats

- Files are saved using their detected extension (e.g., `.mp4`, `.webm`). For `yt-dlp` sources, common outputs are `.mp4`/`.webm`.

Examples

```bash
# GUI mode
python enhanced_video_downloader.py

# Single URL (CLI) with output folder
python enhanced_video_downloader.py https://example.com/page -o videos

# Quality selection for a social video
python enhanced_video_downloader.py https://facebook.com/somevideo -q best

# Batch file (one URL per line)
python enhanced_video_downloader.py --batch urls.txt -o downloads
```

Dependencies / Prerequisites

- `requests`, `beautifulsoup4`, `yt-dlp`, `tqdm`, `urllib3`
- Optional: system `ffmpeg` on PATH (improves `yt-dlp` handling and remuxing)
- GUI requires `tkinter` (bundled with most Python installs; GUI is disabled on some Android environments)

Authentication / API Keys

- No API keys required. The downloader uses scraping strategies and `yt-dlp` where applicable. Optional cookie jars may be supported by `yt-dlp` if configured externally.

Troubleshooting

- "Missing required dependency" → Install listed packages: `pip install requests beautifulsoup4 yt-dlp tqdm`
- GUI not launching on Android → Use CLI mode with URL or batch file.
- Some pages return no videos → Try `-q best` and ensure the URL is the page with the actual video; for streaming-only pages, downloading may not be supported.
- Rate limits or blocked requests → Retry later; consider using different headers or networks.

Version Compatibility

- Python 3.8+; tested with `yt-dlp` 2023–2025 versions and `requests` 2.x

---

### 2) Video to GIF/WebP (`video_to_gif.py`)

Purpose

- Convert single videos or entire folders to GIF or WebP with configurable frame rate, width, quality, and optional time range trimming. Provides duration-aware defaults and batch/concurrent processing.

Parameters

| Parameter | Type | Default | Required | Description |
|---|---|---:|:---:|---|
| `input` (positional or `--input`) | `Path` | none | Yes/Interactive | Input file or directory (directory triggers batch mode). If omitted, interactive prompts are used. |
| `-o, --output` | `Path` | none | No | Output file or directory. If directory + batch, outputs are placed there; else inferred by format. |
| `--format` | `str` | prompt if missing | No | Output format: `gif` or `webp`. |
| `--workers` | `int` | none | No | Max worker threads for concurrent batch conversion. |
| `--fps` | `int` | `10` | No | Frames per second. |
| `--width` | `int` | none | No | Target width in pixels (caps at 800px; aspect preserved). |
| `--start` | `float` | `0` | No | Start time in seconds. |
| `--end` | `float` | none | No | End time in seconds (defaults to full video). |
| `--quality` | `int [1–100]` | `85` | No | Quality level. |
| `--no-optimize` | `flag` | `False` | No | Disable GIF size optimization (GIF only). |
| `--progress` | `str` | `simple` | No | Batch progress style: `simple`, `verbose`, or `none`. |
| `--keepSourceFile` | `{'true','false'}` | `true` | No | Preserve source after conversion. For short MP4s (≤15s), default is delete unless overridden. |
| `-v, --verbose` | `flag` | `False` | No | Enable verbose logging. |

Duration-Aware Behavior

- Single MP4 > 15s without explicit `--start/--end`: prompts and trims to 15s, keeps source by default.
- MP4 ≤ 15s: converts full video and deletes source by default (unless `--keepSourceFile true`).

Examples

```bash
# Single GIF with custom fps and width
python video_to_gif.py input.mp4 --format gif -o output.gif --fps 15 --width 640

# Single WebP with quality override, preserving source
python video_to_gif.py input.mp4 --format webp -o output.webp --quality 90 --keepSourceFile true

# Batch convert a folder to WebP into a target directory
python video_to_gif.py C:/videos --output C:/converted --format webp --workers 4 --progress verbose
```

Dependencies / Prerequisites

- `moviepy<2.0`, `imageio`, `imageio-ffmpeg`, `numpy`, `Pillow`, `tqdm`
- Optional: system `ffmpeg` on PATH (preferred for broader codec support)

Troubleshooting

- "moviepy is required" → Install: `pip install moviepy imageio imageio-ffmpeg numpy Pillow`
- GIF/WebP integrity checks fail → Ensure output path is writable; try lowering `--fps` or width for large sources.
- Source deletion not happening → Deletion applies to MP4 only and only after successful conversion; verify output integrity.

Version Compatibility

- Python 3.8+; tested with `moviepy` 1.0–1.0.3, `imageio` 2.x, Pillow 9–11

---

### 3) Secure Sort by Known Faces (`secure_sort_by_known_faces.py`)

Purpose

- Move images into mapped destination folders when they contain known faces. Performs local face encoding and matching with configurable tolerance and workers.

Functionality

- Detects and validates images; applies size/pixels caps.
- Parallel face analysis (process pool); single-threaded file moves for safety.
- Deterministic naming and traversal-safe writes.

Parameters

| Parameter | Type | Default | Required | Description |
|---|---|---:|:---:|---|
| `--src` | `Path` | none | Yes (interactive prompt if missing) | Source images folder to scan. |
| `--recursive` | `flag` | `False` | No | Recurse into subfolders. |
| `--tolerance` | `float` | Interactive default `0.5` | No | Match tolerance (lower = stricter, higher = more lenient; typical `0.2–0.8`). |
| `--workers` | `int` | CPU cores × factor (interactive default) | No | Parallel workers for analysis (capped). |
| `--jitter` | `int` | `1` | No | Jitter samples per encoding (>=1). |
| `--face` | `Path` (repeatable) | none | No | Reference face image(s). If omitted, prompted interactively. |
| `--dest` | `Path` (repeatable) | none | No | Destination folder paired per `--face`. If omitted, prompted interactively. |

Examples

```bash
# Strict matching (lower tolerance), recursive
python secure_sort_by_known_faces.py --src C:/images --recursive --tolerance 0.3 --workers 4 \
  --face C:/faces/alice.jpg --dest C:/sorted/alice

# Multiple faces → first match wins
python secure_sort_by_known_faces.py --src C:/images --recursive \
  --face C:/faces/alice.jpg --dest C:/sorted/alice \
  --face C:/faces/bob.jpg   --dest C:/sorted/bob

# Interactive pair gathering
python secure_sort_by_known_faces.py --src C:/images
```

Security & Privacy Considerations

- Reference images and encodings are processed locally; no data is sent externally.
- Use clear, front-facing reference images. Avoid sharing face data.
- Ensure destination folders are access-restricted as appropriate.

Dependencies / Prerequisites

- `face-recognition`, `face-recognition-models`, `Pillow`, `tqdm`
- Platform notes: on Windows/macOS/Linux, `face-recognition` may require build tools or prebuilt wheels.

Troubleshooting

- "No module named 'face_recognition'" → Install: `pip install face_recognition face_recognition_models`
- No faces detected → Increase `--jitter` or raise `--tolerance` (be mindful of false positives).
- Permission errors when moving → Ensure write permissions to destination; avoid network shares with restrictive settings.

Version Compatibility

- Python 3.8+; tested with `face_recognition` 1.3–1.4 and Pillow 9–11

---

### 4) Convert WebP/PNG/JPEG to JPG (`convert_webp_to_jpg.py`)

Purpose

- Recursively convert `.webp`, `.png`, `.jpeg` images to `.jpg` with high-quality settings, while safely skipping unsupported or animated WebP files. Preserves EXIF orientation and metadata when present.

Supported formats and behavior

- Converts: `.webp` (non-animated), `.png`, `.jpeg`
- Skips: `.gif`, videos (`.mp4`, `.mov`, `.avi`, `.mkv`), and animated WebP
- Validates images and applies decompression-bomb protections; writes output alongside source file as `same_name.jpg`

Parameters

| Parameter | Type | Default | Required | Description |
|---|---|---:|:---:|---|
| `folder_path` (positional) | `str` | none | Yes (or interactive prompt) | Folder to scan recursively for convertible images. If omitted, the script prompts for a path. |

Examples

```bash
# Basic conversion (Windows)
python convert_webp_to_jpg.py C:\images

# Path with spaces (Windows)
python convert_webp_to_jpg.py "C:\Users\Me\Pictures\Holiday Photos"

# Interactive prompt (no args)
python convert_webp_to_jpg.py
# → Enter folder path to convert images: C:/photos
```

Dependencies / Prerequisites

- `Pillow` (image IO): `pip install pillow`
- `tqdm` (progress bar): `pip install tqdm`

Troubleshooting

- "ModuleNotFoundError: PIL" → Install Pillow: `pip install pillow`
- Animated WebP skipped → Expected behavior: "animated webp: skipped"; convert only first frame via a different tool if needed.
- "unsupported type or extension" → File does not match convertible set or magic header; ensure file is a valid image.
- Decompression bomb error → Large images can trip safety caps; adjust `Image.MAX_IMAGE_PIXELS` in the script if you must handle very large images.

Version Compatibility

- Python 3.8+; tested with Pillow 9–11 and tqdm 4.x

---

## Contribution Guidelines

- Use feature branches and clear, atomic commits.
- Follow PEP 8 for code style; keep functions cohesive and well-named.
- Include docstrings and helpful log messages where appropriate.
- For new CLI options, update help text and README usage examples.
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

---

## Badges

Static status badges (no CI configured yet):

- Build: `https://img.shields.io/badge/build-passing-brightgreen`
- Coverage: `https://img.shields.io/badge/coverage-N/A-lightgrey`
- Version: `https://img.shields.io/badge/version-dev-blue`

Once CI and tests are added, replace with live badges (e.g., GitHub Actions, Coverage reports, semantic version tags).
