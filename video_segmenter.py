#!/usr/bin/env python3

"""
Video Segmentation and Processing System - More Memory-Efficient

Key changes:
- Use ffprobe to get duration (no VideoFileClip).
- Use pure ffmpeg for segment -> webp, no per-frame Python processing.
- Remove unused / heavy imports (MoviePy, numpy, PIL).
- Simplify GC and error handling.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import gc
import subprocess

from tqdm import tqdm

try:
    import imageio_ffmpeg
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

# --- Conversion Config ---

try:
    from video_to_gif import ConversionConfig
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class ConversionConfig:
        default_fps: int = 10
        max_width: int = 800
        default_quality: int = 85
        webp_lossless: bool = False
        webp_method: int = 4
        webp_animation: bool = True
        supported_formats: tuple = (
            ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv",
            ".webm", ".m4v", ".3gp", ".ogv", ".ts", ".mts"
        )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("video_segmenter.log"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("VideoSegmenter")

# --- Termination Handler ---

try:
    import msvcrt
except ImportError:
    msvcrt = None


class TerminationManager:
    def __init__(self):
        self._terminate = False
        self._lock = threading.Lock()

    def request_terminate(self):
        with self._lock:
            if not self._terminate:
                self._terminate = True
                print("\n[!] Termination requested by user (ESC). Stopping safely...")
                logger.warning("Termination requested by user.")

    def is_terminating(self):
        with self._lock:
            return self._terminate


_termination = TerminationManager()


def _monitor_esc():
    """Monitor ESC key in a separate thread (Windows only)."""
    if msvcrt is None:
        return
    while not _termination.is_terminating():
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch == b"\x1b":  # ESC
                _termination.request_terminate()
                break
        time.sleep(0.1)


def start_esc_listener():
    t = threading.Thread(target=_monitor_esc, daemon=True)
    t.start()


# --- ffprobe helpers ---


def get_video_duration_ffprobe(input_path: Path) -> float:
    """Get video duration via ffprobe (seconds, float)."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffprobe_exe = ffmpeg_exe.replace("ffmpeg", "ffprobe")

    cmd = [
        ffprobe_exe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        duration_str = result.stdout.decode("utf-8", errors="ignore").strip()
        return float(duration_str)
    except Exception as e:
        logger.error(f"Error getting video duration via ffprobe: {e}")
        logger.error(f"ffprobe stderr: {result.stderr.decode('utf-8', errors='ignore') if 'result' in locals() else ''}")
        raise


# --- Main Class ---


class VideoSegmenter:
    def __init__(self, input_path: str, max_workers: int = 4):
        self.input_path = Path(input_path).absolute()
        self.output_dir = self.input_path.parent / self.input_path.stem
        self.config = ConversionConfig()
        self.max_workers = max_workers

        self.duration = 0.0
        self.segments: List[Tuple[float, float]] = []

        # Progress tracking
        self.progress_bar = None
        self.progress_lock = threading.Lock()
        self.current_progress = 0.0

    def validate_input(self) -> bool:
        if not self.input_path.exists():
            logger.error(f"Input file not found: {self.input_path}")
            return False
        if self.input_path.suffix.lower() not in self.config.supported_formats:
            logger.error(f"Unsupported video format: {self.input_path.suffix}")
            return False
        return True

    def update_progress(self, increment: float):
        with self.progress_lock:
            self.current_progress += increment
            if self.progress_bar:
                self.progress_bar.n = min(100, int(self.current_progress))
                self.progress_bar.refresh()

    def analyze_video(self) -> bool:
        """Determine duration and compute segments without loading video frames."""
        try:
            logger.info(f"Analyzing video: {self.input_path}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {self.output_dir}")

            if _termination.is_terminating():
                return False

            # Duration via ffprobe
            self.duration = get_video_duration_ffprobe(self.input_path)
            logger.info(f"Video duration: {self.duration:.2f}s")

            current_time = 0.0
            while current_time < self.duration:
                end_time = min(current_time + 15.0, self.duration)
                if end_time > current_time:
                    self.segments.append((current_time, end_time))
                current_time += 45.0

            logger.info(f"Calculated {len(self.segments)} segments")
            self.update_progress(10)
            return True
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            return False

    def build_ffmpeg_cmd(self, start: float, end: float, output_path: Path) -> list:
        """Build ffmpeg command for direct segment -> webp conversion."""
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        # Duration for segment
        duration = max(0.0, end - start)

        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss", str(start),
            "-t", str(duration),
            "-i", str(self.input_path),
            # Scale to max_width while keeping aspect ratio, ensuring even sizes.
            "-vf",
            f"scale='min({self.config.max_width},iw)':-2",
            "-r", str(self.config.default_fps),
            "-c:v", "libwebp",
            "-lossless", "1" if self.config.webp_lossless else "0",
            "-compression_level", str(self.config.webp_method),
            "-q:v", str(self.config.default_quality),
            "-loop", "0",
            "-an",
            str(output_path),
        ]
        return cmd

    def process_segment(self, segment_idx: int, start: float, end: float) -> bool:
        if _termination.is_terminating():
            return False

        output_filename = (
            f"{self.input_path.stem}_part_{segment_idx+1:03d}_"
            f"{int(start)}s_{int(end)}s.webp"
        )
        output_path = self.output_dir / output_filename
        process = None

        try:
            cmd = self.build_ffmpeg_cmd(start, end, output_path)
            logger.info(
                f"Starting ffmpeg for segment {segment_idx+1}: "
                f"{start:.2f}s -> {end:.2f}s"
            )

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Poll for termination while ffmpeg runs
            while True:
                if process.poll() is not None:
                    break
                if _termination.is_terminating():
                    logger.info(f"Segment {segment_idx+1} processing aborted by user.")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    if output_path.exists():
                        try:
                            output_path.unlink()
                        except OSError as oe:
                            logger.warning(f"Failed to remove partial output {output_path}: {oe}")
                    return False
                time.sleep(0.2)

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                logger.error(f"FFmpeg exited with error {process.returncode} for segment {segment_idx+1}")
                logger.error(f"FFmpeg stderr: {stderr.decode('utf-8', errors='ignore')}")
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except OSError as oe:
                        logger.warning(f"Failed to remove failed output {output_path}: {oe}")
                return False

            if not _termination.is_terminating():
                logger.info(f"Saved segment {segment_idx+1}: {output_path}")
                return True
            else:
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except OSError as oe:
                        logger.warning(f"Failed to remove output after termination {output_path}: {oe}")
                return False

        except Exception as e:
            logger.error(f"Error processing segment {segment_idx}: {e}")
            if process:
                try:
                    process.kill()
                except Exception:
                    pass
            if output_path.exists():
                try:
                    output_path.unlink()
                except OSError as oe:
                    logger.warning(f"Failed to remove output after exception {output_path}: {oe}")
            return False
        finally:
            # Batch GC at the end of each segment is usually enough
            gc.collect()

    def run(self):
        start_esc_listener()

        if not self.validate_input():
            return

        self.progress_bar = tqdm(total=100, desc="Overall Progress", unit="%")

        if not self.analyze_video():
            self.progress_bar.close()
            return

        if not self.segments:
            logger.warning("No segments to process.")
            self.progress_bar.close()
            return

        progress_per_segment = 90.0 / len(self.segments)
        logger.info(
            f"Starting processing of {len(self.segments)} segments "
            f"with {self.max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_segment = {}
            for i, (start, end) in enumerate(self.segments):
                if _termination.is_terminating():
                    break
                future = executor.submit(self.process_segment, i, start, end)
                future_to_segment[future] = (i, start, end)

            for future in as_completed(future_to_segment):
                if _termination.is_terminating():
                    # Fast stop: do not wait for remaining
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                try:
                    success = future.result()
                    if success:
                        self.update_progress(progress_per_segment)
                except Exception as e:
                    logger.error(f"Exception in future: {e}")

        if _termination.is_terminating():
            logger.info("Process terminated by user.")
        else:
            self.progress_bar.n = 100
            self.progress_bar.refresh()
            logger.info("Processing complete.")

        self.progress_bar.close()


def main():
    parser = argparse.ArgumentParser(description="Video Segmentation System")
    parser.add_argument("input_file", help="Path to input video file")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    args = parser.parse_args()

    segmenter = VideoSegmenter(args.input_file, max_workers=args.workers)
    segmenter.run()


if __name__ == "__main__":
    main()