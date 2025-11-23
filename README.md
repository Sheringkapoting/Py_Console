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

Extract and download videos from a URL, with optional GUI and batch file:

```bash
# GUI mode
python enhanced_video_downloader.py

# Single URL (CLI)
python enhanced_video_downloader.py https://example.com/page

# Output folder and quality selection
python enhanced_video_downloader.py https://youtube.com/watch?v=xyz -o downloads -q best

# Batch file (one URL per line)
python enhanced_video_downloader.py --batch urls.txt -o downloads

# Verbose logs
python enhanced_video_downloader.py https://example.com -v
```

Command-line flags:

- `url` (positional, optional for GUI)
- `-o, --output` output directory (default: `downloads`)
- `-q, --quality` one of `best`, `worst`, `bestvideo`, `worstvideo` (default: `best`)
- `-v, --verbose` verbose logging
- `--gui` force GUI mode
- `--batch` path to a file with URLs (one per line)

Key features:

- Multi-strategy extraction: direct links, HTML5 tags, embedded iframes, social media via `yt-dlp`, streaming patterns.
- Download strategies: direct HTTP, `yt-dlp`, streaming.
- Tkinter GUI with analysis, batch list management, and threaded downloads.

---

### 2) Video to GIF/WebP (`video_to_gif.py`)

Convert a single video or a directory of videos into GIF or WebP, with quality and size controls:

```bash
# Single conversion to GIF
python video_to_gif.py input.mp4 --format gif -o output.gif --fps 15 --width 640

# Single conversion to WebP
python video_to_gif.py input.mp4 --format webp -o output.webp --fps 12

# Batch convert a folder concurrently
python video_to_gif.py --batch C:/videos --output C:/converted --format webp --workers 4

# Delete source after successful conversion
python video_to_gif.py input.mp4 --format gif --delete-source
```

Command-line flags:

- `input` (positional) file or directory
- `-o, --output` output file or directory
- `--batch` convert all videos in the input directory
- `--format {gif,webp}` select output format
- `--workers` max worker threads for batch concurrent conversion
- `--fps` frames per second (default: 10)
- `--width` target width (px), aspect preserved (max 800px)
- `--start` start time (sec, default: 0)
- `--end` end time (sec)
- `--quality [1-100]` quality (default: 85)
- `--no-optimize` disable GIF size optimization
- `--delete-source` remove source after successful conversion
- `-v, --verbose` verbose logging

Highlights:

- Intelligent FPS and palette optimization, integrity checks for GIF/WebP.
- Batch conversion with sequential or concurrent processing.
- Safe source deletion with hash-based verification.

---

### 3) Secure Sort by Known Faces (`secure_sort_by_known_faces.py`)

Move images into mapped destination folders when they contain known faces:

```bash
# Basic usage with explicit mappings
python secure_sort_by_known_faces.py \
  --src C:/images \
  --face C:/faces/alice.jpg --dest C:/sorted/alice \
  --face C:/faces/bob.jpg   --dest C:/sorted/bob \
  --tolerance 0.5 --workers 4 --recursive

# Interactive mapping if --face/--dest not provided
python secure_sort_by_known_faces.py --src C:/images --recursive
```

Command-line flags:

- `--src` source folder (required)
- `--recursive` recurse subfolders
- `--tolerance` face match tolerance (typical 0.2–0.8; default 0.5 if interactive)
- `--workers` parallel workers (capped relative to CPU cores)
- `--jitter` jitter samples for encodings (>=1)
- `--face` reference face image (repeatable)
- `--dest` destination folder for each face (repeatable)

Safeguards:

- File-size caps and decompression-bomb protections via Pillow.
- Deterministic moves with unique naming to avoid collisions.
- Interactive prompts to complete missing parameters.

---

### 4) Convert WebP/PNG/JPEG to JPG (`convert_webp_to_jpg.py`)

Recursively convert `.webp`, `.png`, `.jpeg` to `.jpg` and delete sources after successful conversion:

```bash
# Provide a folder path as argument or via prompt
python convert_webp_to_jpg.py C:/images
```

Behavior:

- Skips `.gif` and `.jpg` files.
- Validates images, converts to RGB JPEG, writes alongside source, deletes source on success.
- Progress and summary via `tqdm`.

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
