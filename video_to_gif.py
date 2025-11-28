#!/usr/bin/env python3
"""
Video to GIF Converter

A comprehensive tool for converting video files to animated GIFs with various
customization options including:
- Frame rate control
- Size optimization
- Quality settings
- Time range selection
- Batch processing
- Source file deletion after successful conversion

Dependencies:
- moviepy: Video processing and manipulation
- Pillow (PIL): Image processing and GIF optimization
- numpy: Numerical operations for frame processing
- tqdm: Progress bars
- pathlib: Modern path handling
"""

import os
import sys
import argparse
import hashlib
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import logging
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    import moviepy
    from moviepy.editor import VideoFileClip
    try:
        from moviepy.video.fx import resize
    except ImportError:
        # Fallback for older versions
        resize = None
except ImportError:
    print("Error: moviepy is required. Install with: pip install moviepy")
    sys.exit(1)

try:
    import imageio
except ImportError:
    print("Error: imageio is required. Install with: pip install imageio")
    sys.exit(1)

try:
    from PIL import Image, ImageSequence
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found. Progress bars will be disabled.")
    tqdm = None

# === Keyboard ESC termination support ===
class _TerminationManager:
    """Global termination manager to perform graceful shutdown on ESC.

    - Registers cleanup callbacks for resources/files
    - Provides immediate termination with exit code 0 after cleanup
    - Cross-platform keyboard listener uses pynput if available, else fallbacks
    """

    def __init__(self):
        self._cleanup = []
        self._event = threading.Event()

    def register_cleanup(self, fn):
        if callable(fn):
            self._cleanup.append(fn)

    def request_terminate(self, reason: str = "Escape key"):
        if self._event.is_set():
            return
        self._event.set()
        try:
            msg = "Termination requested via Escape. Cleaning up and exiting (code 0)."
            print(msg)
            logging.info(msg)
        except Exception:
            pass
        # Run cleanup callbacks in LIFO order
        for fn in reversed(self._cleanup):
            try:
                fn()
            except Exception as e:
                try:
                    logging.error(f"Cleanup error: {e}")
                except Exception:
                    pass
        # Immediate, graceful termination
        os._exit(0)

    def is_terminating(self) -> bool:
        return self._event.is_set()


_termination = _TerminationManager()


def _flush_and_close_logging():
    """Flush and close all logging handlers to prevent descriptor leaks."""
    try:
        logger = logging.getLogger()
        for h in logger.handlers[:]:
            try:
                h.flush()
                h.close()
            except Exception:
                pass
        logger.handlers = []
    except Exception:
        pass


def _start_keyboard_escape_listener():
    """Start a background listener to detect ESC key presses.

    Prefers pynput for true keydown events across OS. Falls back to platform
    polling on Windows (msvcrt) and POSIX (select+stdin) without capturing
    other keys. Only reacts to ESC, so other shortcuts are unaffected.
    """
    # Try pynput first for reliable keydown events
    try:
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.esc:
                    _termination.request_terminate("Escape key")
            except Exception:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()
        return
    except Exception:
        pass

    # Windows fallback using msvcrt
    if os.name == 'nt':
        try:
            import msvcrt
        except Exception:
            return

        def _win_poll():
            while not _termination.is_terminating():
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b'\x1b',):  # ESC
                            _termination.request_terminate("Escape key")
                    time.sleep(0.05)
                except Exception:
                    time.sleep(0.1)

        t = threading.Thread(target=_win_poll, daemon=True)
        t.start()
    else:
        # POSIX fallback using select on stdin
        try:
            import select
        except Exception:
            return

        def _posix_poll():
            while not _termination.is_terminating():
                try:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        ch = sys.stdin.read(1)
                        if ch == '\x1b':  # ESC
                            _termination.request_terminate("Escape key")
                except Exception:
                    time.sleep(0.1)

        t = threading.Thread(target=_posix_poll, daemon=True)
        t.start()


def _setup_escape_termination():
    """Register cleanup tasks and start the ESC listener."""
    _termination.register_cleanup(_flush_and_close_logging)
    _start_keyboard_escape_listener()


@dataclass
class ConversionConfig:
    """Configuration class for video to GIF conversion settings."""
    default_fps: int = 10
    max_width: int = 800
    default_quality: int = 85
    max_file_size_mb: int = 50
    target_size_mb: int = 12
    compression_level: int = 30
    eliminate_local_color_tables: bool = False
    delete_source: bool = True
    verify_before_delete: bool = True
    max_workers: int = 4
    # WebP defaults
    webp_lossless: bool = False
    webp_method: int = 4  # 0-6, higher = better compression, slower
    webp_animation: bool = True
    
    # Supported video formats
    supported_formats: tuple = (
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv',
        '.webm', '.m4v', '.3gp', '.ogv', '.ts', '.mts'
    )


class ConversionResult:
    """Result object for conversion operations."""
    def __init__(self, success: bool, output_path: Optional[Path] = None, 
                 error: Optional[str] = None, source_deleted: bool = False):
        self.success = success
        self.output_path = output_path
        self.error = error
        self.source_deleted = source_deleted
        self.timestamp = datetime.now()


class VideoToGifConverter:
    """
    A comprehensive video to GIF converter with optimization features and source file management.
    """
    
    def __init__(self, config: Optional[ConversionConfig] = None, logger: Optional[logging.Logger] = None):
        self.config = config or ConversionConfig()
        self.logger = logger or self._setup_logger()
        self._lock = threading.Lock()  # For thread-safe operations
        
        # Performance tracking
        self._conversion_stats = {
            'total_conversions': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'files_deleted': 0,
            'total_size_saved': 0
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('video_to_gif')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def is_supported_video(self, file_path: Path) -> bool:
        """Check if the file is a supported video format."""
        return file_path.suffix.lower() in self.config.supported_formats
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file for verification."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _verify_gif_integrity(self, gif_path: Path) -> bool:
        """Verify that the created GIF file is valid and complete."""
        try:
            with Image.open(gif_path) as img:
                # Check if it's a valid GIF
                if img.format != 'GIF':
                    return False
                
                # Try to iterate through all frames
                frame_count = 0
                for frame in ImageSequence.Iterator(img):
                    frame_count += 1
                    if frame_count > 10000:  # Reasonable limit
                        break
                
                # Check file size is reasonable
                file_size = gif_path.stat().st_size
                if file_size < 1024:  # Less than 1KB is suspicious
                    return False
                
                return frame_count > 0
        except Exception as e:
            self.logger.error(f"Error verifying GIF integrity: {e}")
            return False
    
    def _verify_webp_integrity(self, webp_path: Path) -> bool:
        try:
            with Image.open(webp_path) as img:
                if img.format != 'WEBP':
                    return False
                frame_count = getattr(img, 'n_frames', 1)
                count = 0
                for _ in ImageSequence.Iterator(img):
                    count += 1
                    if count > 10000:
                        break
                file_size = webp_path.stat().st_size
                if file_size < 1024:
                    return False
                return frame_count > 0
        except Exception as e:
            self.logger.error(f"Error verifying WebP integrity: {e}")
            return False
    
    def _safe_delete_source(self, source_path: Path, output_path: Path) -> bool:
        """Safely delete the source video after successful conversion.

        Returns True only when the file is confirmed removed. Logs explicit
        reasons on failure, including permission and filesystem errors.
        """
        # Only auto-delete MP4 sources
        if source_path.suffix.lower() != '.mp4':
            self.logger.info(f"Skipping deletion for non-MP4 source: {source_path}")
            return False
        try:
            if self.config.verify_before_delete:
                ext = output_path.suffix.lower()
                if ext == '.gif':
                    valid = self._verify_gif_integrity(output_path)
                elif ext == '.webp':
                    valid = self._verify_webp_integrity(output_path)
                else:
                    valid = output_path.exists() and output_path.stat().st_size >= 1024
                if not valid:
                    self.logger.warning(f"Output integrity check failed, keeping source file: {source_path}")
                    return False
            if not source_path.exists():
                self.logger.warning(f"Source file no longer exists: {source_path}")
                return False
            if not output_path.exists():
                self.logger.error(f"Output file does not exist, cannot delete source: {source_path}")
                return False
            out_size = output_path.stat().st_size
            if out_size < 1024:
                self.logger.error(f"Output file too small ({out_size} bytes), keeping source: {source_path}")
                return False
            original_size = source_path.stat().st_size
            try:
                source_path.unlink()
            except PermissionError as pe:
                self.logger.error(f"Permission denied deleting source {source_path}: {pe}")
                return False
            except FileNotFoundError:
                # Race condition: already removed
                pass
            except OSError as oe:
                self.logger.error(f"OS error deleting source {source_path}: {oe}")
                return False

            # Verify removal
            if source_path.exists():
                self.logger.error(f"Deletion verification failed; source still exists: {source_path}")
                return False
            with self._lock:
                self._conversion_stats['files_deleted'] += 1
                self._conversion_stats['total_size_saved'] += original_size
            self.logger.info(f"Successfully deleted source file: {source_path} (saved {original_size / (1024*1024):.1f}MB)")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting source file {source_path}: {e}")
            return False
    
    def get_video_info(self, video_path: Path) -> dict:
        """Get basic information about the video file."""
        try:
            with VideoFileClip(str(video_path)) as clip:
                return {
                    'duration': clip.duration,
                    'fps': clip.fps,
                    'size': (clip.w, clip.h),
                    'format': video_path.suffix.lower()
                }
        except Exception as e:
            self.logger.error(f"Error reading video info: {e}")
            return {}
    
    def optimize_gif_size(self, gif_path: Path, target_size_mb: float = 10.0, quality: int = 85) -> bool:
        """
        Optimize GIF file size through multiple strategies with improved memory management.
        
        Args:
            gif_path: Path to the GIF file to optimize
            target_size_mb: Target file size in MB
            quality: Starting quality level (1-100)
        
        Returns:
            True if optimization was successful, False otherwise
        """
        if not gif_path.exists():
            self.logger.error(f"GIF file not found: {gif_path}")
            return False
        
        original_size = gif_path.stat().st_size / (1024 * 1024)
        target_size_bytes = target_size_mb * 1024 * 1024
        
        if original_size <= target_size_mb:
            self.logger.info(f"GIF already within target size: {original_size:.1f}MB <= {target_size_mb}MB")
            return True
        
        self.logger.info(f"Optimizing GIF size: {original_size:.1f}MB -> {target_size_mb}MB target")
        
        # Create backup
        backup_path = gif_path.with_suffix('.gif.backup')
        try:
            import shutil
            shutil.copy2(gif_path, backup_path)
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False
        
        try:
            # Strategy 1: Quality reduction with frame optimization
            if self._optimize_by_quality(gif_path, target_size_bytes, quality):
                backup_path.unlink(missing_ok=True)
                return True
            
            # Strategy 2: Frame reduction if quality optimization failed
            if self._optimize_by_frame_reduction(gif_path, backup_path, target_size_bytes):
                backup_path.unlink(missing_ok=True)
                return True
            
            # Strategy 3: Color palette reduction
            if self._optimize_by_palette_reduction(gif_path, backup_path, target_size_bytes):
                backup_path.unlink(missing_ok=True)
                return True
            
            # If all strategies failed, restore backup
            self.logger.warning(f"Could not optimize to target size with any strategy. Restoring original.")
            import shutil
            shutil.move(backup_path, gif_path)
            return False
                
        except Exception as e:
            self.logger.error(f"Error during GIF optimization: {e}")
            # Restore backup on error
            if backup_path.exists():
                try:
                    import shutil
                    shutil.move(backup_path, gif_path)
                except Exception as restore_error:
                    self.logger.error(f"Failed to restore backup: {restore_error}")
            return False
    
    def _optimize_by_quality(self, gif_path: Path, target_size_bytes: int, start_quality: int) -> bool:
        """Optimize GIF by reducing quality iteratively."""
        try:
            with Image.open(gif_path) as img:
                # Extract frames more efficiently
                frames_data = self._extract_frames_efficiently(img)
                if not frames_data:
                    return False
                
                frames, durations = frames_data
                
                # Iterative quality reduction
                for quality in range(start_quality, 19, -10):
                    try:
                        # Use temporary file to avoid corrupting original during optimization
                        temp_path = gif_path.with_suffix('.tmp.gif')
                        
                        frames[0].save(
                            temp_path,
                            save_all=True,
                            append_images=frames[1:],
                            duration=durations,
                            loop=0,
                            optimize=True,
                            quality=quality
                        )
                        
                        if temp_path.stat().st_size <= target_size_bytes:
                            # Success - replace original
                            import shutil
                            shutil.move(temp_path, gif_path)
                            current_size_mb = gif_path.stat().st_size / (1024 * 1024)
                            self.logger.info(f"Quality optimization successful: {current_size_mb:.1f}MB (quality: {quality})")
                            return True
                        
                        # Clean up temp file
                        temp_path.unlink(missing_ok=True)
                        
                    except Exception as e:
                        self.logger.debug(f"Quality {quality} failed: {e}")
                        continue
                
                return False
                
        except Exception as e:
            self.logger.error(f"Quality optimization failed: {e}")
            return False
    
    def _optimize_by_frame_reduction(self, gif_path: Path, backup_path: Path, target_size_bytes: int) -> bool:
        """Optimize GIF by reducing frame count."""
        try:
            # Restore from backup for frame reduction attempt
            import shutil
            shutil.copy2(backup_path, gif_path)
            
            with Image.open(gif_path) as img:
                frames_data = self._extract_frames_efficiently(img)
                if not frames_data:
                    return False
                
                frames, durations = frames_data
                original_frame_count = len(frames)
                
                # Try reducing frames by 2x, 3x, 4x
                for reduction_factor in [2, 3, 4]:
                    try:
                        # Keep every nth frame
                        reduced_frames = frames[::reduction_factor]
                        reduced_durations = [d * reduction_factor for d in durations[::reduction_factor]]
                        
                        if len(reduced_frames) < 2:  # Need at least 2 frames for animation
                            continue
                        
                        temp_path = gif_path.with_suffix('.tmp.gif')
                        
                        reduced_frames[0].save(
                            temp_path,
                            save_all=True,
                            append_images=reduced_frames[1:],
                            duration=reduced_durations,
                            loop=0,
                            optimize=True,
                            quality=75
                        )
                        
                        if temp_path.stat().st_size <= target_size_bytes:
                            shutil.move(temp_path, gif_path)
                            current_size_mb = gif_path.stat().st_size / (1024 * 1024)
                            self.logger.info(f"Frame reduction successful: {current_size_mb:.1f}MB ({len(reduced_frames)}/{original_frame_count} frames)")
                            return True
                        
                        temp_path.unlink(missing_ok=True)
                        
                    except Exception as e:
                        self.logger.debug(f"Frame reduction {reduction_factor}x failed: {e}")
                        continue
                
                return False
                
        except Exception as e:
            self.logger.error(f"Frame reduction optimization failed: {e}")
            return False
    
    def _optimize_by_palette_reduction(self, gif_path: Path, backup_path: Path, target_size_bytes: int) -> bool:
        """Optimize GIF by reducing color palette."""
        try:
            # Restore from backup for palette reduction attempt
            import shutil
            shutil.copy2(backup_path, gif_path)
            
            with Image.open(gif_path) as img:
                frames_data = self._extract_frames_efficiently(img)
                if not frames_data:
                    return False
                
                frames, durations = frames_data
                
                # Try different palette sizes
                for colors in [128, 64, 32, 16]:
                    try:
                        # Convert frames to reduced palette
                        reduced_frames = []
                        for frame in frames:
                            # Convert to P mode with reduced palette
                            if frame.mode != 'P':
                                frame = frame.convert('RGB')
                            reduced_frame = frame.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
                            reduced_frames.append(reduced_frame)
                        
                        temp_path = gif_path.with_suffix('.tmp.gif')
                        
                        reduced_frames[0].save(
                            temp_path,
                            save_all=True,
                            append_images=reduced_frames[1:],
                            duration=durations,
                            loop=0,
                            optimize=True
                        )
                        
                        if temp_path.stat().st_size <= target_size_bytes:
                            shutil.move(temp_path, gif_path)
                            current_size_mb = gif_path.stat().st_size / (1024 * 1024)
                            self.logger.info(f"Palette reduction successful: {current_size_mb:.1f}MB ({colors} colors)")
                            return True
                        
                        temp_path.unlink(missing_ok=True)
                        
                    except Exception as e:
                        self.logger.debug(f"Palette reduction to {colors} colors failed: {e}")
                        continue
                
                return False
                
        except Exception as e:
            self.logger.error(f"Palette reduction optimization failed: {e}")
            return False
    
    def _extract_frames_efficiently(self, img: Image.Image) -> Optional[tuple]:
        """Extract frames and durations efficiently with memory management."""
        try:
            frames = []
            durations = []
            
            for frame_num in range(img.n_frames):
                img.seek(frame_num)
                # Create a copy to avoid issues with frame references
                frame = img.copy()
                frames.append(frame)
                
                # Get frame duration (default to 100ms if not available)
                duration = img.info.get('duration', 100)
                durations.append(max(duration, 20))  # Minimum 20ms per frame
            
            return frames, durations
            
        except Exception as e:
            self.logger.error(f"Failed to extract frames: {e}")
            return None
    
    def calculate_optimal_fps(self, video_duration: float, target_size_mb: float, width: int) -> int:
        """Calculate optimal FPS based on video characteristics and target size."""
        # Base FPS calculation considering video length and target size
        base_fps = self.config.default_fps
        
        # Adjust FPS based on video duration (longer videos need lower FPS)
        if video_duration > 30:  # Very long videos
            fps_adjustment = 0.6
        elif video_duration > 15:  # Medium videos
            fps_adjustment = 0.75
        elif video_duration > 5:   # Short videos
            fps_adjustment = 0.9
        else:  # Very short videos
            fps_adjustment = 1.0
        
        # Adjust FPS based on target size (smaller targets need lower FPS)
        if target_size_mb <= 8:
            size_adjustment = 0.7
        elif target_size_mb <= 12:
            size_adjustment = 0.85
        elif target_size_mb <= 20:
            size_adjustment = 1.0
        else:
            size_adjustment = 1.2
        
        # Adjust FPS based on width (larger widths need lower FPS for same file size)
        if width >= 800:
            width_adjustment = 0.8
        elif width >= 600:
            width_adjustment = 0.9
        else:
            width_adjustment = 1.0
        
        optimal_fps = int(base_fps * fps_adjustment * size_adjustment * width_adjustment)
        return max(5, min(optimal_fps, 15))  # Clamp between 5-15 FPS
    
    def convert_video_to_gif(self, 
                           video_path: Path,
                           output_path: Optional[Path] = None,
                           start_time: float = 0,
                           end_time: Optional[float] = None,
                           fps: int = 10,
                           width: Optional[int] = None,
                           quality: int = 85,
                           optimize_size: bool = True,
                           delete_source: Optional[bool] = None,
                           keepSourceFile: Optional[bool] = True,
                           retry_attempts: int = 3) -> ConversionResult:
        """
        Convert a video file to an animated GIF.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output GIF (optional)
            start_time: Start time in seconds
            end_time: End time in seconds (None for full video)
            fps: Frames per second for the GIF
            width: Target width in pixels (maintains aspect ratio)
            quality: Quality setting (1-100)
            optimize_size: Whether to optimize file size
            delete_source: Deprecated. Whether to delete source file after conversion (overrides config)
            keepSourceFile: New. If True, preserves source; if False, attempts deletion
        
        Returns:
            ConversionResult object with success status, output path, and deletion info
        """
        # Determine if source should be deleted (MP4-only by default)
        if keepSourceFile is not None:
            should_delete_source = (not keepSourceFile) and (video_path.suffix.lower() == '.mp4')
        elif delete_source is not None:
            # Backward compatibility if keepSourceFile not supplied
            should_delete_source = delete_source and (video_path.suffix.lower() == '.mp4')
        else:
            should_delete_source = False  # default preserve for backward compatibility
        
        # Update statistics
        with self._lock:
            self._conversion_stats['total_conversions'] += 1
        
        # Validate input (non-retryable errors)
        if not video_path.exists():
            error_msg = f"Video file not found: {video_path}"
            self.logger.error(error_msg)
            with self._lock:
                self._conversion_stats['failed_conversions'] += 1
            return ConversionResult(success=False, error=error_msg)
        
        if not self.is_supported_video(video_path):
            error_msg = f"Unsupported video format: {video_path.suffix}"
            self.logger.error(error_msg)
            with self._lock:
                self._conversion_stats['failed_conversions'] += 1
            return ConversionResult(success=False, error=error_msg)
        
        # Retry logic for conversion process
        last_error = None
        for attempt in range(retry_attempts):
            try:
                return self._perform_conversion(
                    video_path, output_path, start_time, end_time, 
                    fps, width, quality, optimize_size, should_delete_source
                )
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Conversion attempt {attempt + 1}/{retry_attempts} failed: {e}")
                
                # Wait before retry (exponential backoff)
                if attempt < retry_attempts - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s...
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # All attempts failed
        error_msg = f"All {retry_attempts} conversion attempts failed. Last error: {last_error}"
        self.logger.error(error_msg)
        with self._lock:
            self._conversion_stats['failed_conversions'] += 1
        return ConversionResult(success=False, error=error_msg)
    
    def convert_video_to_webp(self, 
                           video_path: Path,
                           output_path: Optional[Path] = None,
                           start_time: float = 0,
                           end_time: Optional[float] = None,
                           fps: int = 10,
                           width: Optional[int] = None,
                           quality: int = 85,
                           optimize_size: bool = True,
                           delete_source: Optional[bool] = None,
                           keepSourceFile: Optional[bool] = True,
                           retry_attempts: int = 3) -> ConversionResult:
        """
        Convert a video file to animated WebP.

        Args:
            delete_source: Deprecated. Whether to delete source file after conversion (overrides config)
            keepSourceFile: New. If True, preserves source; if False, attempts deletion
        """
        # Determine if source should be deleted (MP4-only by default)
        if keepSourceFile is not None:
            should_delete_source = (not keepSourceFile) and (video_path.suffix.lower() == '.mp4')
        elif delete_source is not None:
            should_delete_source = delete_source and (video_path.suffix.lower() == '.mp4')
        else:
            should_delete_source = False  # default preserve for backward compatibility
        with self._lock:
            self._conversion_stats['total_conversions'] += 1
        if not video_path.exists():
            error_msg = f"Video file not found: {video_path}"
            self.logger.error(error_msg)
            with self._lock:
                self._conversion_stats['failed_conversions'] += 1
            return ConversionResult(success=False, error=error_msg)
        if not self.is_supported_video(video_path):
            error_msg = f"Unsupported video format: {video_path.suffix}"
            self.logger.error(error_msg)
            with self._lock:
                self._conversion_stats['failed_conversions'] += 1
            return ConversionResult(success=False, error=error_msg)
        last_error = None
        for attempt in range(retry_attempts):
            try:
                # Determine output
                if output_path is None:
                    output_path = video_path.with_suffix('.webp')
                self.logger.info(f"Converting {video_path.name} to WebP...")
                with VideoFileClip(str(video_path)) as clip:
                    # Apply time range
                    if end_time is not None:
                        clip = clip.subclip(start_time, min(end_time, clip.duration))
                    elif start_time > 0:
                        clip = clip.subclip(start_time)
                    # Determine target width as in GIF conversion
                    if width is not None:
                        target_width = width
                        if target_width > self.config.max_width:
                            self.logger.info(f"Width {target_width}px exceeds maximum {self.config.max_width}px, using {self.config.max_width}px")
                            target_width = self.config.max_width
                    else:
                        original_width = clip.w
                        if original_width > self.config.max_width:
                            target_width = self.config.max_width
                            self.logger.info(f"Original width {original_width}px exceeds maximum, using {self.config.max_width}px")
                        else:
                            target_width = original_width
                    # Collect frames and save via Pillow to control WebP params
                    frames: List[Image.Image] = []
                    for frame in clip.iter_frames(fps=fps, dtype='uint8'):
                        img = Image.fromarray(frame)
                        if target_width != clip.w:
                            new_height = int(img.height * (target_width / img.width))
                            img = img.resize((target_width, new_height), Image.LANCZOS)
                        frames.append(img)

                    if not frames:
                        raise ValueError("No frames extracted from video for WebP conversion")

                    duration_ms = int(max(1, round(1000 / max(1, fps))))

                    # Log applied WebP parameters
                    self.logger.info(
                        f"Using WebP params: quality={quality}, lossless={self.config.webp_lossless}, "
                        f"method={self.config.webp_method}, animated={self.config.webp_animation}, fps={fps}"
                    )

                    if self.config.webp_animation and len(frames) > 1:
                        frames[0].save(
                            str(output_path),
                            format="WEBP",
                            save_all=True,
                            append_images=frames[1:],
                            loop=0,
                            duration=duration_ms,
                            quality=quality,
                            lossless=self.config.webp_lossless,
                            method=self.config.webp_method,
                        )
                    else:
                        frames[0].save(
                            str(output_path),
                            format="WEBP",
                            quality=quality,
                            lossless=self.config.webp_lossless,
                            method=self.config.webp_method,
                        )
                # Verify integrity
                is_valid = self._verify_webp_integrity(output_path)
                if not is_valid:
                    raise ValueError("WebP integrity check failed")
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"WebP created successfully: {output_path} ({file_size_mb:.1f}MB)")
                with self._lock:
                    self._conversion_stats['successful_conversions'] += 1
                source_deleted = False
                if should_delete_source:
                    source_deleted = self._safe_delete_source(video_path, output_path)
                return ConversionResult(success=True, output_path=output_path, source_deleted=source_deleted)
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"WebP conversion attempt {attempt + 1}/{retry_attempts} failed: {e}")
                if attempt < retry_attempts - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        error_msg = f"All {retry_attempts} WebP conversion attempts failed. Last error: {last_error}"
        self.logger.error(error_msg)
        with self._lock:
            self._conversion_stats['failed_conversions'] += 1
        return ConversionResult(success=False, error=error_msg)

    def _perform_conversion(self, video_path: Path, output_path: Optional[Path], 
                          start_time: float, end_time: Optional[float], fps: int, 
                          width: Optional[int], quality: int, optimize_size: bool, 
                          should_delete_source: bool) -> ConversionResult:
        """Perform the actual conversion with comprehensive error handling."""
        try:
            
            # Generate output path if not provided
            if output_path is None:
                output_path = video_path.with_suffix('.gif')
            
            self.logger.info(f"Converting {video_path.name} to GIF...")
            
            # Load video clip
            with VideoFileClip(str(video_path)) as clip:
                # Apply time range
                if end_time is not None:
                    clip = clip.subclip(start_time, min(end_time, clip.duration))
                elif start_time > 0:
                    clip = clip.subclip(start_time)
                
                # Enhanced: Use original video width as default, capped at max_width
                if width is not None:
                    # User specified width - apply maximum constraint if needed
                    target_width = width
                    if target_width > self.config.max_width:
                        self.logger.info(f"Width {target_width}px exceeds maximum {self.config.max_width}px, using {self.config.max_width}px")
                        target_width = self.config.max_width
                else:
                    # No width specified - use original video width, capped at max_width
                    original_width = clip.w
                    if original_width > self.config.max_width:
                        target_width = self.config.max_width
                        self.logger.info(f"Original width {original_width}px exceeds maximum, using {self.config.max_width}px")
                    else:
                        target_width = original_width
                        self.logger.info(f"Using original video width: {original_width}px")
                
                # Enhanced: Calculate optimal FPS for target size if optimization is enabled
                if optimize_size and fps == self.config.default_fps:  # Only auto-adjust if using default FPS
                    optimal_fps = self.calculate_optimal_fps(clip.duration, self.config.target_size_mb, target_width)
                    if optimal_fps != fps:
                        self.logger.info(f"Adjusting FPS for size optimization: {fps} → {optimal_fps} FPS")
                        fps = optimal_fps
                
                # Apply resizing if target width differs from original
                if target_width != clip.w:
                    if resize is not None:
                        clip = clip.resize(width=target_width)
                    else:
                        # Fallback resize method for different moviepy versions
                        clip = clip.resized(width=target_width)
                    
                    self.logger.info(f"Resizing video to {target_width}px width (maintaining aspect ratio)")
                else:
                    self.logger.info(f"Using original video dimensions: {clip.w}x{clip.h}")
                
                # Convert to GIF with simple output
                self.logger.info(f"Writing GIF with {fps} FPS...")
                clip.write_gif(
                    str(output_path),
                    fps=fps
                )
                
                # Optimize file size if requested
                if optimize_size:
                    self.optimize_gif_size(output_path, target_size_mb=self.config.target_size_mb, quality=quality)
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"GIF created successfully: {output_path} ({file_size_mb:.1f}MB)")
            
            # Update success statistics
            with self._lock:
                self._conversion_stats['successful_conversions'] += 1
            
            # Attempt to delete source file if requested
            source_deleted = False
            if should_delete_source:
                source_deleted = self._safe_delete_source(video_path, output_path)
            
            return ConversionResult(
                success=True, 
                output_path=output_path, 
                source_deleted=source_deleted
            )
            
        except Exception as e:
            error_msg = f"Error converting video to GIF: {e}"
            self.logger.error(error_msg)
            # Re-raise the exception to be caught by retry logic
            raise
    
    def batch_convert(self, 
                     input_dir: Path,
                     output_dir: Optional[Path] = None,
                     max_workers: Optional[int] = None,
                     use_concurrent: bool = True,
                     **kwargs) -> List[ConversionResult]:
        """
        Convert multiple video files to GIFs with unified progress tracking and optional concurrency.
        
        Args:
            input_dir: Directory containing video files
            output_dir: Output directory (optional)
            max_workers: Maximum number of concurrent workers (None for auto-detect)
            use_concurrent: Whether to use concurrent processing
            **kwargs: Additional arguments for convert_video_to_gif
        
        Returns:
            List of ConversionResult objects for all processed files
        """
        if not input_dir.exists() or not input_dir.is_dir():
            self.logger.error(f"Input directory not found: {input_dir}")
            return []
        
        if output_dir is None:
            output_dir = input_dir
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find MP4 video files only (per new requirement)
        video_files = []
        for file_path in input_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() == '.mp4':
                video_files.append(file_path)
        
        if not video_files:
            self.logger.warning(f"No supported video files found in {input_dir}")
            return []
        
        total_files = len(video_files)
        self.logger.info(f"Found {total_files} video files to convert")
        
        # Determine processing method
        if use_concurrent and len(video_files) > 1:
            return self._batch_convert_concurrent(video_files, output_dir, max_workers, **kwargs)
        else:
            return self._batch_convert_sequential(video_files, output_dir, **kwargs)
    
    def _batch_convert_sequential(self, video_files: List[Path], output_dir: Path, **kwargs) -> List[ConversionResult]:
        """Sequential batch conversion with unified progress tracking."""
        total_files = len(video_files)
        results = []
        processed_files = 0
        output_format = kwargs.get('output_format', 'gif').lower()
        progress_style = kwargs.get('progress', 'simple')
        # Filter kwargs passed to conversion functions
        kwargs_filtered = kwargs.copy()
        kwargs_filtered.pop('output_format', None)
        kwargs_filtered.pop('progress', None)
        
        if tqdm and progress_style != 'none':
            # Unified, simple progress bar for batch processing
            bar_desc = "Batch Converting" if progress_style == 'simple' else "Batch Converting (verbose)"
            with tqdm(total=total_files, desc=bar_desc, unit="files") as batch_pbar:
                for i, video_path in enumerate(video_files, 1):
                    # Update progress bar description with current file info
                    filename = video_path.name
                    batch_pbar.set_description(f"Converting [{i}/{total_files}]")
                    
                    ext = '.gif' if output_format == 'gif' else '.webp'
                    output_path = output_dir / f"{video_path.stem}{ext}"
                    
                    # Temporarily disable individual file progress to avoid conflicts
                    kwargs_no_progress = kwargs.copy()
                    kwargs_no_progress.pop('output_format', None)
                    kwargs_no_progress.pop('progress', None)
                    
                    try:
                        if output_format == 'gif':
                            result = self.convert_video_to_gif(
                                video_path=video_path,
                                output_path=output_path,
                                **kwargs_no_progress
                            )
                        else:
                            result = self.convert_video_to_webp(
                                video_path=video_path,
                                output_path=output_path,
                                **kwargs_no_progress
                            )
                        
                        results.append(result)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to convert {video_path.name}: {e}")
                        results.append(ConversionResult(success=False, error=str(e)))
                    
                    processed_files = i
                    batch_pbar.update(1)
                    
                    # Final status update
                    if i == total_files:
                        batch_pbar.set_description("Batch Conversion Complete")
        else:
            # Fallback without tqdm - simple counter display
            print(f"\nBatch converting {total_files} files...")
            for i, video_path in enumerate(video_files, 1):
                print(f"\n[{i}/{total_files}] Processing: {video_path.name}")
                
                ext = '.gif' if output_format == 'gif' else '.webp'
                output_path = output_dir / f"{video_path.stem}{ext}"
                
                try:
                    if output_format == 'gif':
                        result = self.convert_video_to_gif(
                            video_path=video_path,
                            output_path=output_path,
                            **kwargs_filtered
                        )
                    else:
                        result = self.convert_video_to_webp(
                            video_path=video_path,
                            output_path=output_path,
                            **kwargs_filtered
                        )
                    
                    results.append(result)
                    if result.success:
                        print(f"✓ Successfully converted: {video_path.name}")
                    else:
                        print(f"✗ Failed to convert: {video_path.name}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to convert {video_path.name}: {e}")
                    print(f"✗ Error converting {video_path.name}: {e}")
                    results.append(ConversionResult(success=False, error=str(e)))
                
                processed_files = i
            
            # Final summary (kept concise)
            successful = sum(1 for r in results if r.success)
            success_rate = (successful/total_files)*100 if total_files > 0 else 0
            print(f"\n=== Batch Conversion Complete ===")
            print(f"Total files: {total_files}")
            print(f"Successfully converted: {successful}")
            print(f"Failed: {total_files - successful}")
            print(f"Success rate: {success_rate:.1f}%")
        
        successful = sum(1 for r in results if r.success)
        self.logger.info(f"Sequential batch conversion complete: {successful}/{total_files} files converted successfully")
        return results
    
    def _batch_convert_concurrent(self, video_files: List[Path], output_dir: Path, max_workers: Optional[int], **kwargs) -> List[ConversionResult]:
        """Concurrent batch conversion with thread pool."""
        total_files = len(video_files)
        max_workers = max_workers or min(self.config.max_workers, len(video_files))
        output_format = kwargs.get('output_format', 'gif').lower()
        progress_style = kwargs.get('progress', 'simple')
        self.logger.info(f"Starting concurrent conversion with {max_workers} workers")
        results = [None] * total_files
        completed_count = 0
        # Filter kwargs passed to conversion functions
        kwargs_filtered = kwargs.copy()
        kwargs_filtered.pop('output_format', None)
        kwargs_filtered.pop('progress', None)
        
        def convert_single_file(index_and_path):
            index, video_path = index_and_path
            ext = '.gif' if output_format == 'gif' else '.webp'
            output_path = output_dir / f"{video_path.stem}{ext}"
            try:
                if output_format == 'gif':
                    result = self.convert_video_to_gif(
                        video_path=video_path,
                        output_path=output_path,
                        **kwargs_filtered
                    )
                else:
                    result = self.convert_video_to_webp(
                        video_path=video_path,
                        output_path=output_path,
                        **kwargs_filtered
                    )
                return index, result
            except Exception as e:
                self.logger.error(f"Failed to convert {video_path.name}: {e}")
                return index, ConversionResult(success=False, error=str(e))
        
        if tqdm and progress_style != 'none':
            desc = "Concurrent Converting" if progress_style == 'simple' else f"Concurrent Converting ({max_workers} workers)"
            with tqdm(total=total_files, desc=desc, unit="files") as pbar:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = { executor.submit(convert_single_file, (i, video_path)): i for i, video_path in enumerate(video_files) }
                    for future in as_completed(future_to_index):
                        try:
                            index, result = future.result()
                            results[index] = result
                            completed_count += 1
                            pbar.update(1)
                        except Exception as e:
                            self.logger.error(f"Task execution error: {e}")
                            index = future_to_index[future]
                            results[index] = ConversionResult(success=False, error=str(e))
                            completed_count += 1
                            pbar.update(1)
        else:
            print(f"\nConcurrent batch converting {total_files} files with {max_workers} workers...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = { executor.submit(convert_single_file, (i, video_path)): i for i, video_path in enumerate(video_files) }
                for future in as_completed(future_to_index):
                    try:
                        index, result = future.result()
                        results[index] = result
                        completed_count += 1
                        successful = sum(1 for r in results if r and r.success)
                        print(f"Progress: {completed_count}/{total_files} | Success: {successful} | Failed: {completed_count - successful}")
                    except Exception as e:
                        self.logger.error(f"Task execution error: {e}")
                        index = future_to_index[future]
                        results[index] = ConversionResult(success=False, error=str(e))
                        completed_count += 1
        results = [r for r in results if r is not None]
        successful = sum(1 for r in results if r.success)
        self.logger.info(f"Concurrent batch conversion complete: {successful}/{total_files} files converted successfully")
        return results


def main():
    """Main function for command-line interface."""
    # Setup global ESC termination listener for immediate graceful exit
    _setup_escape_termination()

    parser = argparse.ArgumentParser(
        description="Convert videos to GIF or WebP with duration-aware behavior and optional source preservation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples
==============
Single File (GIF):
  python video_to_gif.py input.mp4 --format gif -o output.gif --fps 15 --width 640
  python video_to_gif.py input.mp4 --format gif -o output.gif --keepSourceFile false  # delete source after success

Single File (WebP):
  python video_to_gif.py input.mp4 --format webp -o output.webp --fps 12 --quality 85
  python video_to_gif.py input.mp4 --format webp -o output.webp --keepSourceFile true   # explicitly keep source

Batch Directory:
  python video_to_gif.py C:/videos --output C:/converted --format webp --workers 4
  python video_to_gif.py C:/videos --output C:/converted --format gif --keepSourceFile false

Duration-Aware Trim (>15s MP4):
  # When a single MP4 exceeds 15 seconds and no start/end are provided, you will be prompted for format
  python video_to_gif.py long_video.mp4  # trims to 15s (source kept by default)

Parameters
==========
Input/Output:
  input (positional) or --input PATH   Input file or directory (directory triggers batch)
  -o, --output PATH                    Output file or directory
  --format {gif,webp}                  Output format (GIF or WebP)
  --workers N                          Max worker threads in batch mode
  --progress {simple,verbose,none}     Progress style for batch mode
  --keepSourceFile {true,false}        Preserve source (true) or delete (false); default true

Time & Size:
  --start SECONDS                      Start time (default: 0)
  --end SECONDS                        End time (default: full video)
  --fps N                              Frames per second (default: 10)
  --width PX                           Target width (caps at 800px by default)
  --quality [1-100]                    Quality (default: 85)
  --no-optimize                        Disable GIF size optimization (GIF only)

Notes:
  - WebP defaults: quality=85, lossless=disabled, method=4, animation=enabled.
  - Source preservation/deletion controlled via --keepSourceFile (default: true). Deletion applies to MP4 only.
  - GIF defaults: fps=10, max width capped at 800px unless overridden.

Deprecated:
  --params              Deprecated; use --help. Still accepted with a warning.
  --delete-source       Deprecated; use --keepSourceFile false.
        """
    )

    # Input arguments
    parser.add_argument(
        'input',
        nargs='?',
        help='Input video file or directory (directory triggers batch mode)'
    )

    # Optional alias for positional input to support --input usage
    parser.add_argument(
        '--input',
        dest='input',
        help='Input video file or directory (alias for positional input)'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output file or directory (extension inferred by format)'
    )

    # Note: Batch mode is automatic based on directory input

    # Format selection and workers
    parser.add_argument(
        '--format',
        choices=['gif', 'webp'],
        help='Output format: gif or webp (prompted if omitted)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        help='Max worker threads for batch concurrent conversion'
    )

    # Conversion settings
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second (default: 10)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        help='Target width in pixels (max: 800px, maintains aspect ratio)'
    )
    
    parser.add_argument(
        '--start',
        type=float,
        default=0,
        help='Start time in seconds (default: 0)'
    )
    
    parser.add_argument(
        '--end',
        type=float,
        help='End time in seconds (default: full video)'
    )
    
    parser.add_argument(
        '--quality',
        type=int,
        default=85,
        choices=range(1, 101),
        metavar='[1-100]',
        help='Quality setting 1-100 (default: 85)'
    )
    
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='Disable GIF file size optimization'
    )
    
    # --delete-source removed: deletion happens automatically on success for MP4 sources
    
    # Logging
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    # Backward compatibility: deprecated --params (hidden)
    parser.add_argument(
        '--params', '--list-params',
        dest='list_params',
        action='store_true',
        help=argparse.SUPPRESS
    )
    
    # Progress display style
    parser.add_argument(
        '--progress',
        choices=['simple', 'verbose', 'none'],
        default='simple',
        help='Progress style for batch mode: simple (default), verbose, or none'
    )

    # File preservation/deletion control
    parser.add_argument(
        '--keepSourceFile',
        dest='keepSourceFile',
        choices=['true', 'false'],
        help='Preserve source file after processing (true/false). Default: true. Overrides duration-based defaults.'
    )

    # Backward compatibility: deprecated --delete-source (hidden). Maps to keepSourceFile=false
    parser.add_argument(
        '--delete-source',
        dest='deprecated_delete_source',
        action='store_true',
        help=argparse.SUPPRESS
    )
    
    args = parser.parse_args()

    # Handle deprecated flags
    if getattr(args, 'deprecated_delete_source', False):
        print("[DEPRECATED] --delete-source is deprecated. Use --keepSourceFile false.")
        if args.keepSourceFile is None:
            args.keepSourceFile = 'false'
    if getattr(args, 'list_params', False):
        print("[DEPRECATED] --params is deprecated. Use --help for comprehensive documentation.\n")
        parser.print_help()
        return
    
    # Setup logging with timestamps
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Interactive mode if no input provided
    if not args.input:
        print("Video Converter (GIF/WebP)")
        print("=" * 50)
        
        input_path = input("Enter video file path or directory: ").strip().strip('"\'')
        if not input_path:
            print("No input provided. Exiting.")
            return
        
        args.input = input_path
        
        # Prompt for format if not specified
        if not args.format:
            choice = input("Select output format [gif/webp] (default: gif): ").strip().lower()
            args.format = 'webp' if choice == 'webp' else 'gif'
        
        # Optional output path/dir
        output_path = input("Enter output file or directory (optional): ").strip().strip('"\'')
        if output_path:
            args.output = output_path
    
    # Create converter with configuration
    config = ConversionConfig()
    converter = VideoToGifConverter(config=config)
    
    input_path = Path(args.input)

    # Helper: prompt for output format (WEBP/GIF), default WEBP
    def _prompt_format(default: str = 'webp') -> str:
        try:
            choice = input("Select output format [WEBP/GIF] (default: WEBP): ").strip()
        except Exception:
            choice = ''
        if not choice:
            return default.lower()
        choice_lower = choice.lower()
        if choice_lower in ('webp', 'gif'):
            return choice_lower
        print("Invalid selection. Defaulting to WEBP.")
        return default.lower()

    # Helper: prompt for output directory (mandatory)
    def _prompt_output_dir() -> Path:
        while True:
            try:
                out = input("Enter output directory path (mandatory): ").strip().strip('"\'')
            except Exception:
                out = ''
            if not out:
                print("Output directory is required.")
                continue
            p = Path(out)
            try:
                p.mkdir(parents=True, exist_ok=True)
                return p
            except Exception as e:
                print(f"Cannot create/use output directory: {e}")
                continue

    # Helper: prompt optional FPS (returns None if empty)
    def _prompt_fps_optional() -> Optional[int]:
        try:
            val = input("Enter FPS (optional; press Enter to use original): ").strip()
        except Exception:
            val = ''
        if not val:
            return None
        try:
            fps_i = int(val)
            if fps_i <= 0:
                print("FPS must be a positive integer. Using original.")
                return None
            return fps_i
        except Exception:
            print("Invalid FPS. Using original.")
            return None
    
    try:
        if input_path.is_dir():
            # Batch mode: process all MP4s; trim if >15s, delete source for ≤15s
            # Ensure output directory
            output_dir = Path(args.output) if args.output else _prompt_output_dir()

            mp4_files = [p for p in input_path.glob('**/*.mp4') if p.is_file()]
            if not mp4_files:
                print("No MP4 files found in directory.")
                return

            # Format selection: prompt if not provided
            selected_format = (args.format.lower() if args.format else _prompt_format())
            if selected_format not in ('webp', 'gif'):
                print("Invalid format provided. Defaulting to WEBP.")
                selected_format = 'webp'

            # Optional FPS selection: if not provided, use original per file
            fps_override = args.fps if args.fps is not None else _prompt_fps_optional()

            successful: List[ConversionResult] = []
            failed: List[ConversionResult] = []
            deleted_success: int = 0
            deleted_fail: int = 0
            deletion_fail_details: List[str] = []

            # Interpret CLI keepSourceFile override
            cli_keep: Optional[bool] = None
            if args.keepSourceFile is not None:
                cli_keep = True if str(args.keepSourceFile).lower() == 'true' else False

            print(f"Processing {len(mp4_files)} MP4 file(s) → {selected_format.upper()} | Output: {output_dir}")

            for src in mp4_files:
                try:
                    info = converter.get_video_info(src)
                    if not info or 'duration' not in info or info['duration'] is None:
                        logging.error(f"Unable to determine duration for: {src}")
                        continue
                    duration = float(info['duration'])
                    orig_fps = info.get('fps') or converter.config.default_fps
                    # Decide FPS
                    fps_used = fps_override if fps_override is not None else int(max(1, int(orig_fps)))

                    # Output path in specified output directory with new extension
                    out_path = (output_dir / src.name).with_suffix('.' + selected_format)

                    logging.info(
                        f"File: {src.name} | duration={duration:.2f}s | type={selected_format.upper()} | fps={fps_used}"
                    )

                    if duration > 15:
                        # Trim to 15s, preserve source by default
                        keep_flag = cli_keep if cli_keep is not None else True
                        if selected_format == 'webp':
                            res = converter.convert_video_to_webp(
                                video_path=src,
                                output_path=out_path,
                                start_time=0,
                                end_time=15,
                                fps=fps_used,
                                width=None,
                                quality=converter.config.default_quality,
                                optimize_size=False,
                                keepSourceFile=keep_flag
                            )
                        else:
                            res = converter.convert_video_to_gif(
                                video_path=src,
                                output_path=out_path,
                                start_time=0,
                                end_time=15,
                                fps=fps_used,
                                width=None,
                                quality=converter.config.default_quality,
                                optimize_size=False,
                                keepSourceFile=keep_flag
                            )
                    else:
                        # Process full video, delete source on success by default
                        keep_flag = cli_keep if cli_keep is not None else False
                        if selected_format == 'webp':
                            res = converter.convert_video_to_webp(
                                video_path=src,
                                output_path=out_path,
                                start_time=0,
                                end_time=None,
                                fps=fps_used,
                                width=None,
                                quality=converter.config.default_quality,
                                optimize_size=False,
                                keepSourceFile=keep_flag
                            )
                        else:
                            res = converter.convert_video_to_gif(
                                video_path=src,
                                output_path=out_path,
                                start_time=0,
                                end_time=None,
                                fps=fps_used,
                                width=None,
                                quality=converter.config.default_quality,
                                optimize_size=False,
                                keepSourceFile=keep_flag
                            )

                    if res.success:
                        successful.append(res)
                        logging.info(f"Success: {res.output_path}")
                        # Verify deletion status for short videos
                        if duration <= 15:
                            if res.source_deleted:
                                deleted_success += 1
                            else:
                                # Verify existence and record failure detail
                                if src.exists():
                                    deleted_fail += 1
                                    deletion_fail_details.append(f"Not deleted: {src}")
                    else:
                        failed.append(res)
                        logging.error(f"Failed: {src} → {selected_format.upper()} | {res.error}")
                except Exception as e:
                    logging.error(f"Error processing {src}: {e}")

            # Summary
            print("\n=== Batch Conversion Summary ===")
            print(f"Total MP4 files: {len(mp4_files)}")
            print(f"Successful: {len(successful)}")
            print(f"Failed: {len(failed)}")
            print(f"Deleted (≤15s): {deleted_success}")
            if deleted_fail > 0:
                print(f"Deletion failures (≤15s): {deleted_fail}")
                for msg in deletion_fail_details[:10]:
                    print(f"  ✗ {msg}")
            if failed:
                for r in failed:
                    print(f"  ✗ {r.error}")
            return
        
        else:
            # Single file mode
            output_path = Path(args.output) if args.output else None

            # Single file: if MP4 exceeds 15s and no explicit start/end, run trim flow
            if input_path.suffix.lower() == '.mp4' and (args.start == 0) and (args.end is None):
                info = converter.get_video_info(input_path)
                if not info or 'duration' not in info or info['duration'] is None:
                    print("Unable to determine video duration. Aborting.")
                    sys.exit(1)
                if info['duration'] > 15:
                    selected_format = (args.format.lower() if args.format else _prompt_format())
                    if selected_format not in ('webp', 'gif'):
                        print("Invalid format. Defaulting to WEBP.")
                        selected_format = 'webp'

                    orig_fps = info.get('fps') or converter.config.default_fps
                    try:
                        fps_val = int(max(1, min(int(orig_fps), 30)))
                    except Exception:
                        fps_val = converter.config.default_fps

                    # Output filename same as input, new extension, same directory
                    out_path = (input_path.parent / input_path.name).with_suffix('.' + selected_format)

                    logging.info(f"Trimming to 15s and converting to {selected_format.upper()} at {fps_val}fps")
                    # Interpret CLI override for keepSourceFile
                    cli_keep_single: Optional[bool] = None
                    if args.keepSourceFile is not None:
                        cli_keep_single = True if str(args.keepSourceFile).lower() == 'true' else False
                    keep_flag_single = cli_keep_single if cli_keep_single is not None else True
                    if selected_format == 'webp':
                        result = converter.convert_video_to_webp(
                            video_path=input_path,
                            output_path=out_path,
                            start_time=0,
                            end_time=15,
                            fps=fps_val,
                            width=None,
                            quality=converter.config.default_quality,
                            optimize_size=False,
                            keepSourceFile=keep_flag_single
                        )
                        success_label = "WEBP"
                    else:
                        result = converter.convert_video_to_gif(
                            video_path=input_path,
                            output_path=out_path,
                            start_time=0,
                            end_time=15,
                            fps=fps_val,
                            width=None,
                            quality=converter.config.default_quality,
                            optimize_size=False,
                            keepSourceFile=keep_flag_single
                        )
                        success_label = "GIF"
                else:
                    print("Video is 15 seconds or shorter. No processing needed.")
                    return
            else:
                # Fallback to existing single-file conversion behavior
                # Select conversion method based on format (default GIF)
                # Interpret CLI override for keepSourceFile (default True)
                keep_flag_sf: bool = True
                if args.keepSourceFile is not None:
                    keep_flag_sf = True if str(args.keepSourceFile).lower() == 'true' else False
                if args.format and args.format.lower() == 'webp':
                    result = converter.convert_video_to_webp(
                        video_path=input_path,
                        output_path=output_path,
                        start_time=args.start,
                        end_time=args.end,
                        fps=args.fps,
                        width=args.width,
                        quality=args.quality,
                        optimize_size=not args.no_optimize,
                        keepSourceFile=keep_flag_sf
                    )
                    success_label = "WEBP"
                else:
                    result = converter.convert_video_to_gif(
                        video_path=input_path,
                        output_path=output_path,
                        start_time=args.start,
                        end_time=args.end,
                        fps=args.fps,
                        width=args.width,
                        quality=args.quality,
                        optimize_size=not args.no_optimize,
                        keepSourceFile=keep_flag_sf
                    )
                    success_label = "GIF"
            
            if result.success:
                print(f"\n{success_label} created successfully: {result.output_path}")
                
                # Show file info
                file_size_mb = result.output_path.stat().st_size / (1024 * 1024)
                video_info = converter.get_video_info(input_path)
                
                print(f"File size: {file_size_mb:.1f}MB")
                if video_info:
                    print(f"Original video: {video_info['duration']:.1f}s, {video_info['size'][0]}x{video_info['size'][1]}")
                
                if result.source_deleted:
                    print("✓ Source video file deleted successfully")
                
                # Print statistics
                stats = converter._conversion_stats
                print(f"\n=== Conversion Statistics ===")
                print(f"Total conversions: {stats['total_conversions']}")
                print(f"Successful: {stats['successful_conversions']}")
                if stats['files_deleted'] > 0:
                    print(f"Files deleted: {stats['files_deleted']}")
                    print(f"Space saved: {stats['total_size_saved'] / (1024*1024):.1f}MB")
            else:
                fmt_label = (args.format or 'gif').upper()
                print(f"Failed to convert video to {fmt_label}: {result.error}")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nConversion cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
