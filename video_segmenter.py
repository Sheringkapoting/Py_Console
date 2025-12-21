#!/usr/bin/env python3

"""
Video Segmentation System

Creates multiple short video segments from input video:
- Each segment is 15 seconds long (configurable with --segment-duration)
- New segments start every 45 seconds (configurable with --segment-gap)
- Output segments are saved in MP4 format
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
import shutil

from tqdm import tqdm

try:
    import imageio_ffmpeg
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

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

# Supported video formats
SUPPORTED_FORMATS = (
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv",
    ".webm", ".m4v", ".3gp", ".ogv", ".ts", ".mts"
)

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
    
    # ffprobe is usually in the same directory as ffmpeg in imageio_ffmpeg wheels
    # Or we can use ffmpeg -i to parse duration if ffprobe is missing.
    # imageio_ffmpeg usually only bundles ffmpeg.
    # Let's try ffmpeg -i first as it is guaranteed to exist.
    
    cmd = [ffmpeg_exe, "-i", str(input_path)]
    
    try:
        # ffmpeg prints info to stderr
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Parse output for "Duration: HH:MM:SS.mm"
        import re
        pattern = re.compile(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}\.\d+)")
        match = pattern.search(result.stderr)
        
        if match:
            hours, minutes, seconds = map(float, match.groups())
            return hours * 3600 + minutes * 60 + seconds
            
        logger.error(f"Could not parse duration from ffmpeg output.")
        return 0.0
        
    except Exception as e:
        logger.error(f"Error getting video duration via ffmpeg: {e}")
        raise


# --- Main Class ---


class VideoSegmenter:
    def __init__(
        self,
        input_path: str,
        segment_duration: float = 15.0,
        segment_gap: float = 45.0,
        output_dir: str | None = None,
        max_workers: int = 4,
        max_retries: int = 2
    ):
        self.input_path = Path(input_path).absolute()
        self.output_dir = Path(output_dir) if output_dir else (self.input_path.parent / self.input_path.stem)
        self.segment_duration = max(0.1, float(segment_duration))
        self.segment_gap = max(self.segment_duration, float(segment_gap))
        self.max_workers = max_workers
        self.max_retries = max(0, int(max_retries))

        self.duration = 0.0
        self.segments: List[Tuple[float, float]] = []

        # Progress tracking
        self.progress_bar = None
        self.progress_lock = threading.Lock()
        self.frames_processed = 0
        self._start_time = None

    def validate_input(self) -> bool:
        if not self.input_path.exists():
            logger.error(f"Input file not found: {self.input_path}")
            return False
        if self.input_path.suffix.lower() not in SUPPORTED_FORMATS:
            logger.error(f"Unsupported video format: {self.input_path.suffix}")
            return False
        return True

    def update_progress(self, segment_idx: int):
        """Update progress bar."""
        with self.progress_lock:
            if self.progress_bar and len(self.segments) > 0:
                progress = min(100, int(((segment_idx + 1) / len(self.segments)) * 100))
                self.progress_bar.n = progress
                self.progress_bar.set_postfix_str(f"seg={segment_idx+1}/{len(self.segments)}")
                self.progress_bar.refresh()

    def analyze_video(self) -> bool:
        """Determine video duration and compute segment time ranges."""
        try:
            logger.info(f"Analyzing video: {self.input_path}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {self.output_dir}")

            if _termination.is_terminating():
                return False

            # Get duration via ffmpeg
            self.duration = get_video_duration_ffprobe(self.input_path)
            
            if self.duration <= 0:
                logger.error("Invalid video duration detected.")
                return False
                
            logger.info(f"Video duration: {self.duration:.2f}s")

            # Calculate segment time ranges
            current_time = 0.0
            while current_time < self.duration:
                end_time = min(current_time + self.segment_duration, self.duration)
                if end_time > current_time:
                    self.segments.append((current_time, end_time))
                current_time += self.segment_gap

            logger.info(f"Calculated {len(self.segments)} segments")
            logger.info(f"  - Segment duration: {self.segment_duration}s")
            logger.info(f"  - Gap between segments: {self.segment_gap}s")
            
            if self.progress_bar:
                self.progress_bar.n = 5
                self.progress_bar.refresh()
            return True
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            return False

    def segment_video(self, segment_idx: int, start: float, end: float) -> bool:
        """Create a video segment from start to end time."""
        if _termination.is_terminating():
            return False

        output_filename = f"{self.input_path.stem}_segment_{segment_idx+1:03d}.mp4"
        output_path = self.output_dir / output_filename
        process = None

        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            duration = max(0.0, end - start)
            
            # Use stream copy for fast segmentation without re-encoding
            cmd = [
                ffmpeg_exe,
                "-y",
                "-ss", str(start),
                "-t", str(duration),
                "-i", str(self.input_path),
                "-c", "copy",  # Copy streams without re-encoding
                "-an",  # No audio
                str(output_path),
            ]
            
            logger.debug(f"Segmenting {segment_idx+1}/{len(self.segments)}: {start:.1f}s-{end:.1f}s -> {output_filename}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
            )

            # Monitor ffmpeg execution
            while True:
                if process.poll() is not None:
                    break
                
                if _termination.is_terminating():
                    logger.info(f"Segment {segment_idx+1} creation aborted by user.")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    if output_path.exists():
                        try:
                            output_path.unlink()
                        except OSError:
                            pass
                    return False
                time.sleep(0.1)

            stdout, stderr = process.communicate()
            stderr_text = stderr.decode('utf-8', errors='ignore') if stderr else ""

            if process.returncode != 0:
                logger.error(f"FFmpeg exited with error {process.returncode} for segment {segment_idx+1}")
                logger.error(f"Segment details: {start:.1f}s-{end:.1f}s, Duration: {end-start:.1f}s")
                if stderr_text:
                    logger.error(f"FFmpeg stderr: {stderr_text[-500:]}")
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except OSError:
                        pass
                return False

            if not _termination.is_terminating():
                logger.info(f"Created segment {segment_idx+1}: {output_filename}")
                self.update_progress(segment_idx)
                return True
            else:
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except OSError:
                        pass
                return False

        except Exception as e:
            logger.error(f"Error creating segment {segment_idx}: {e}")
            if process:
                try:
                    process.kill()
                except Exception:
                    pass
            if output_path.exists():
                try:
                    output_path.unlink()
                except OSError:
                    pass
            return False
        finally:
            gc.collect()

    def _start_memory_monitor(self):
        """Optional memory monitoring (minimal for segmentation)."""
        pass  # Memory management not critical for stream-copy segmentation

    def run(self):
        """Main segmentation workflow."""
        start_esc_listener()
        self._start_time = time.time()

        if not self.validate_input():
            return

        self.progress_bar = tqdm(
            total=100,
            desc="Segmenting",
            unit="%",
            bar_format='{desc}: {percentage:3.0f}% |{bar}| [{postfix}]'
        )

        if not self.analyze_video():
            self.progress_bar.close()
            return

        if not self.segments:
            logger.warning("No segments to process.")
            self.progress_bar.close()
            return

        logger.info(
            f"Starting segmentation of {len(self.segments)} segments "
            f"with {self.max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for i, (start, end) in enumerate(self.segments):
                if _termination.is_terminating():
                    break
                future = executor.submit(self.segment_video, i, start, end)
                futures[future] = (i, start, end)

            for future in as_completed(futures):
                if _termination.is_terminating():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                try:
                    success = future.result()
                    if not success:
                        i, start, end = futures[future]
                        logger.warning(f"Segment {i+1} failed. Retrying (max {self.max_retries} times)...")
                        
                        retries = 0
                        while retries < self.max_retries and not _termination.is_terminating():
                            wait_time = min(2 ** retries, 10)  # Exponential backoff
                            logger.info(f"Waiting {wait_time}s before retry {retries+1}/{self.max_retries}")
                            time.sleep(wait_time)
                            
                            logger.info(f"Retrying segment {i+1}...")
                            ok = self.segment_video(i, start, end)
                            if ok:
                                logger.info(f"Segment {i+1} succeeded on retry {retries+1}")
                                break
                            retries += 1
                        
                        if retries >= self.max_retries:
                            logger.error(f"Segment {i+1} failed after {self.max_retries} retries. Skipping.")
                except Exception as e:
                    logger.error(f"Exception in segmentation: {e}")

        if _termination.is_terminating():
            logger.info("Segmentation terminated by user.")
        else:
            self.progress_bar.n = 100
            self.progress_bar.refresh()
            logger.info("Segmentation complete.")

        self.progress_bar.close()
        
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        if elapsed > 0:
            logger.info(f"Total time: {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Video Segmentation Tool - Creates short video segments from input video"
    )
    parser.add_argument(
        "input_file",
        help="Path to input video file"
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=15.0,
        help="Duration of each segment in seconds (default: 15.0)"
    )
    parser.add_argument(
        "--segment-gap",
        type=float,
        default=45.0,
        help="Time gap between segment starts in seconds (default: 45.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for segments (default: same as input file name)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Maximum retries per failed segment (default: 2)"
    )
    
    args = parser.parse_args()

    segmenter = VideoSegmenter(
        input_path=args.input_file,
        segment_duration=args.segment_duration,
        segment_gap=args.segment_gap,
        output_dir=args.output_dir,
        max_workers=args.workers,
        max_retries=args.retries,
    )
    segmenter.run()


if __name__ == "__main__":
    main()
