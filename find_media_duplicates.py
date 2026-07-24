#!/usr/bin/env python3
"""
find_media_duplicates.py ─ Duplicate animated WebP / GIF / video detector & mover
====================================================================================
Same detection model as find_duplicates.py, extended to media with a time
dimension:

  Pass 1 — SHA-256 hash     : byte-identical files       (exact duplicates)
  Pass 2 — Frame-sampled
           perceptual hash  : visually identical media    (re-encodes, re-saves,
                                                             transcodes, resizes)

Each animated GIF/WebP or video is reduced to a small set of evenly-sampled
frames (by relative position through the clip/animation); each frame gets a
perceptual hash (imagehash.phash). Two files are a near-duplicate when the
mean Hamming distance across their sampled frames is <= --threshold (videos
are additionally pre-filtered by duration — wildly different lengths are
never compared).

Primary selection per group:
  1. Highest resolution   (width x height)     <- best quality
  2. Most frames / longest duration            <- least likely re-encoded down
  3. Smallest file size   (tiebreaker)          <- most efficient encoding

Duplicates are moved to <src>/Duplicate/ and renamed {stem}_dup{N}{ext}, via
the same shared engine (UI, progress bars, HTML report, ESC termination,
move workflow) as find_duplicates.py — see src/scripts/dup_finder_core.py.

Requirements:
    pip install Pillow imagehash rich moviepy numpy

Usage:
    python find_media_duplicates.py                                  # interactive, both media types
    python find_media_duplicates.py --src "D:\\Media"                # direct, dry run
    python find_media_duplicates.py --src "D:\\Media" --execute      # move files
    python find_media_duplicates.py --media-type animated            # GIF/animated WebP only
    python find_media_duplicates.py --media-type video                # video only
    python find_media_duplicates.py --threshold 6                     # stricter (default 8)
    python find_media_duplicates.py --exact-only                      # skip perceptual pass
    python find_media_duplicates.py --recursive                       # include subfolders

Note: find_duplicates.py already scans .gif/.webp as static images (hashing
only their first frame). Running both tools over the same folder means a
GIF/animated WebP can be evaluated by each tool independently — that's
expected: find_duplicates.py's pass is a cheap first-frame check,
find_media_duplicates.py's is the animation-aware one.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── Shared engine (UI, progress, HTML report, ESC termination, move workflow) ─
_scripts_dir = str(Path(__file__).parent / "src" / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
import dup_finder_core as dfc
from common_utils import format_size
del _scripts_dir

console = dfc.console


# ── Dependency check ──────────────────────────────────────────────────────────

def _check_deps(need_video: bool) -> None:
    missing = []
    for pkg, mod in [("Pillow", "PIL"), ("imagehash", "imagehash"), ("rich", "rich")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if need_video:
        for pkg, mod in [("moviepy", "moviepy"), ("numpy", "numpy")]:
            try:
                __import__(mod)
            except ImportError:
                missing.append(pkg)
    if missing:
        print(f"\n  ✗  Missing packages. Run:\n\n     pip install {' '.join(missing)}\n")
        sys.exit(1)


try:
    import imagehash
    _HAS_IMAGEHASH = True
except ImportError:
    imagehash = None          # type: ignore[assignment]
    _HAS_IMAGEHASH = False

from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        VideoFileClip = None   # type: ignore[assignment]


# ── Animated GIF / WebP handler ───────────────────────────────────────────────

@dataclass
class _AnimSig:
    frame_count: int
    size: tuple[int, int]
    frame_hashes: list = field(default_factory=list)


class AnimatedMediaHandler(dfc.MediaHandler):
    """Animated GIF and animated WebP — frame-sampled perceptual hashing."""

    display_name = "Animated Media"
    item_noun    = "animated GIF/WebP files"
    report_title = "Duplicate Animated Media Report"
    report_emoji = "\U0001F300"
    report_slug  = "duplicate_animated"

    SAMPLE_FRAMES = 8

    def __init__(self) -> None:
        self._cache: dict[Path, Optional[_AnimSig]] = {}

    # -- collection --------------------------------------------------------
    def collect(self, src: Path, recursive: bool) -> list[Path]:
        pattern = "**/*" if recursive else "*"
        out: list[Path] = []
        for p in src.glob(pattern):
            if not p.is_file():
                continue
            if dfc.DUPLICATE_FOLDER in p.relative_to(src).parts:
                continue
            suf = p.suffix.lower()
            if suf == ".gif":
                out.append(p)
            elif suf == ".webp":
                try:
                    with Image.open(p) as im:
                        if getattr(im, "is_animated", False):
                            out.append(p)
                except (UnidentifiedImageError, OSError):
                    continue
        return sorted(out)

    # -- signature -----------------------------------------------------------
    def _sample_indices(self, n_frames: int) -> list[int]:
        n = max(1, n_frames)
        k = min(self.SAMPLE_FRAMES, n)
        if k <= 1:
            return [0]
        return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})

    def _load(self, path: Path) -> Optional[_AnimSig]:
        if path in self._cache:
            return self._cache[path]
        sig: Optional[_AnimSig] = None
        try:
            with Image.open(path) as im:
                n_frames = getattr(im, "n_frames", 1)
                size = im.size
                hashes = []
                for idx in self._sample_indices(n_frames):
                    try:
                        im.seek(idx)
                        hashes.append(imagehash.phash(im.convert("RGB")))
                    except Exception:
                        continue
                sig = _AnimSig(frame_count=n_frames, size=size, frame_hashes=hashes)
        except Exception:
            sig = None
        self._cache[path] = sig
        return sig

    def similarity_signature(self, path: Path) -> Any:
        return self._load(path)

    def bucket_key(self, signature: Any):
        if not signature or not signature.frame_hashes:
            return "_unknown"
        return str(signature.frame_hashes[0])[:2]

    def similarity_distance(self, sig_a: Any, sig_b: Any) -> float:
        if not sig_a or not sig_b or not sig_a.frame_hashes or not sig_b.frame_hashes:
            return float("inf")
        pairs = list(zip(sig_a.frame_hashes, sig_b.frame_hashes))
        if not pairs:
            return float("inf")
        return sum((ha - hb) for ha, hb in pairs) / len(pairs)

    # -- selection / reporting ---------------------------------------------
    def select_primary(self, group: list[Path]) -> Path:
        def key(p: Path):
            sig = self._load(p)
            pixels = sig.size[0] * sig.size[1] if sig else 0
            frames = sig.frame_count if sig else 0
            return (-pixels, -frames, self.file_size(p))
        return min(group, key=key)

    def thumbnail_b64(self, path: Path) -> str:
        try:
            with Image.open(path) as im:
                im.seek(0)
                return dfc.b64_jpeg_thumbnail(im.convert("RGB"))
        except Exception:
            return ""

    def metadata_line(self, path: Path) -> str:
        sig = self._load(path)
        size_str = format_size(self.file_size(path))
        if sig:
            return f"{sig.size[0]}\u00d7{sig.size[1]} \u00b7 {sig.frame_count} frames \u00b7 {size_str}"
        return size_str


# ── Video handler ──────────────────────────────────────────────────────────────

@dataclass
class _VideoSig:
    duration: float
    size: tuple[int, int]
    frame_hashes: list = field(default_factory=list)


class VideoMediaHandler(dfc.MediaHandler):
    """Video files — duration pre-filter + frame-sampled perceptual hashing."""

    display_name = "Video"
    item_noun    = "videos"
    report_title = "Duplicate Video Report"
    report_emoji = "\U0001F3AC"
    report_slug  = "duplicate_video"

    VIDEO_EXTS = {
        ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv",
        ".webm", ".m4v", ".3gp", ".ogv", ".ts", ".mts",
    }
    SAMPLE_FRAMES = 8
    DURATION_TOLERANCE_SEC   = 2.0
    DURATION_TOLERANCE_RATIO = 0.05

    def __init__(self) -> None:
        self._cache: dict[Path, Optional[_VideoSig]] = {}

    # -- collection --------------------------------------------------------
    def collect(self, src: Path, recursive: bool) -> list[Path]:
        pattern = "**/*" if recursive else "*"
        return sorted(
            p for p in src.glob(pattern)
            if p.is_file()
            and p.suffix.lower() in self.VIDEO_EXTS
            and dfc.DUPLICATE_FOLDER not in p.relative_to(src).parts
        )

    # -- signature -----------------------------------------------------------
    def _load(self, path: Path) -> Optional[_VideoSig]:
        if path in self._cache:
            return self._cache[path]
        sig: Optional[_VideoSig] = None
        if VideoFileClip is not None:
            try:
                with VideoFileClip(str(path)) as clip:
                    duration = float(clip.duration or 0)
                    size = (int(clip.w), int(clip.h))
                    hashes = []
                    if duration > 0:
                        for i in range(self.SAMPLE_FRAMES):
                            t = min(duration * (i + 0.5) / self.SAMPLE_FRAMES,
                                    max(duration - 0.05, 0))
                            try:
                                frame = clip.get_frame(t)
                                hashes.append(imagehash.phash(Image.fromarray(frame)))
                            except Exception:
                                continue
                    sig = _VideoSig(duration=duration, size=size, frame_hashes=hashes)
            except Exception:
                sig = None
        self._cache[path] = sig
        return sig

    def similarity_signature(self, path: Path) -> Any:
        return self._load(path)

    def bucket_key(self, signature: Any):
        if not signature or signature.duration <= 0:
            return "_unknown"
        return round(signature.duration / 3)

    def similarity_distance(self, sig_a: Any, sig_b: Any) -> float:
        if not sig_a or not sig_b or not sig_a.frame_hashes or not sig_b.frame_hashes:
            return float("inf")
        tol = max(self.DURATION_TOLERANCE_SEC,
                   self.DURATION_TOLERANCE_RATIO * max(sig_a.duration, sig_b.duration))
        if abs(sig_a.duration - sig_b.duration) > tol:
            return float("inf")
        pairs = list(zip(sig_a.frame_hashes, sig_b.frame_hashes))
        if not pairs:
            return float("inf")
        return sum((ha - hb) for ha, hb in pairs) / len(pairs)

    # -- selection / reporting ---------------------------------------------
    def select_primary(self, group: list[Path]) -> Path:
        def key(p: Path):
            sig = self._load(p)
            pixels = sig.size[0] * sig.size[1] if sig else 0
            duration = sig.duration if sig else 0
            return (-pixels, -duration, self.file_size(p))
        return min(group, key=key)

    def thumbnail_b64(self, path: Path) -> str:
        if VideoFileClip is None:
            return ""
        sig = self._load(path)
        try:
            with VideoFileClip(str(path)) as clip:
                t = min((sig.duration * 0.3) if sig and sig.duration else 0,
                        max(clip.duration - 0.05, 0))
                frame = clip.get_frame(t)
                return dfc.b64_jpeg_thumbnail(Image.fromarray(frame))
        except Exception:
            return ""

    def metadata_line(self, path: Path) -> str:
        sig = self._load(path)
        size_str = format_size(self.file_size(path))
        if sig and sig.duration:
            mm, ss = divmod(int(sig.duration), 60)
            return f"{sig.size[0]}\u00d7{sig.size[1]} \u00b7 {mm}:{ss:02d} \u00b7 {size_str}"
        return size_str


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="find_media_duplicates",
        description="Detect exact and visually-similar duplicate animated WebP/GIF/video "
                     "files and move them to Duplicate/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--src",         type=Path, default=None,
                     help="Source media folder (prompted if omitted)")
    ap.add_argument("--media-type",  choices=["auto", "animated", "video"], default="auto",
                     help="Media family to scan: animated GIF/WebP, video, or both (default: auto = both)")
    ap.add_argument("--threshold",   type=int,  default=8,
                     help="Perceptual hash Hamming distance 0-64, averaged over sampled "
                          "frames (default 8). Lower = stricter.")
    ap.add_argument("--recursive",   action="store_true",
                     help="Scan subfolders recursively")
    ap.add_argument("--exact-only",  action="store_true",
                     help="Skip near-duplicate pass — exact byte-identical only")
    ap.add_argument("--execute",     action="store_true",
                     help="Move duplicate files (default: dry run — no files moved)")
    ap.add_argument("--report",      type=Path, default=None,
                     help="HTML report output path (default: <src>/<type>_report.html; "
                          "with --media-type auto, two reports are written)")
    return ap


def _report_path_for(args, handler: dfc.MediaHandler, both: bool) -> Optional[Path]:
    if args.report is None:
        return None
    if not both:
        return args.report
    return args.report.with_name(f"{args.report.stem}_{handler.report_slug.split('_', 1)[-1]}{args.report.suffix}")


def _resolve_src(args) -> Path:
    """Prompt for / validate the source folder once, shared across handlers."""
    if args.src is None:
        while True:
            raw = dfc.Prompt.ask("  [cyan]Source folder[/cyan]").strip().strip('"').strip("'")
            p = Path(raw)
            if p.exists() and p.is_dir():
                return p
            console.print(f"    [red]✗  Not found:[/red] {p}")
    if not args.src.exists() or not args.src.is_dir():
        console.print(f"[red]  ✗  Source folder not found: {args.src}[/red]")
        sys.exit(1)
    return Path(args.src)


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    need_video = args.media_type in ("auto", "video")
    _check_deps(need_video)

    handlers: list[dfc.MediaHandler] = []
    if args.media_type in ("auto", "animated"):
        handlers.append(AnimatedMediaHandler())
    if args.media_type in ("auto", "video"):
        if VideoFileClip is None:
            console.print("[yellow]  ⚠  moviepy not available — skipping video scan.[/yellow]")
        else:
            handlers.append(VideoMediaHandler())

    if not handlers:
        console.print("[red]  ✗  No media handlers available for the requested --media-type.[/red]")
        sys.exit(1)

    # Resolve the source folder once (interactive prompt fires at most once,
    # even when both handlers run back to back under --media-type auto).
    args.src = _resolve_src(args)

    tm = dfc.make_termination_manager()
    both = len(handlers) > 1

    for handler in handlers:
        if tm.is_terminating():
            console.print("[yellow]  ⚠  ESC pressed — skipping remaining media types.[/yellow]")
            break

        run_args = argparse.Namespace(**vars(args))
        run_args.report = _report_path_for(args, handler, both)
        dfc.run_workflow(handler, run_args, tm)
        if both:
            console.print(dfc.Rule(style="dim"))


if __name__ == "__main__":
    main()
