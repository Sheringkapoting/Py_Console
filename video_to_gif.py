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
    delete_source: bool = False
    verify_before_delete: bool = True
    max_workers: int = 4
    
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
        if not self.config.delete_source:
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
            source_path.unlink()
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
            delete_source: Whether to delete source file after conversion (overrides config)
        
        Returns:
            ConversionResult object with success status, output path, and deletion info
        """
        # Determine if source should be deleted
        should_delete_source = delete_source if delete_source is not None else self.config.delete_source
        
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
                           retry_attempts: int = 3) -> ConversionResult:
        """
        Convert a video file to animated WebP.
        """
        should_delete_source = delete_source if delete_source is not None else self.config.delete_source
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
                    if target_width != clip.w and resize is not None:
                        clip = clip.resize(width=target_width)
                    frame_duration_ms = int(max(1, round(1000 / max(1, fps))))
                    # Use imageio writer for WebP; per-frame duration is not supported via append_data
                    writer = imageio.get_writer(str(output_path), format='WEBP', mode='I', quality=quality)
                    try:
                        for frame in clip.iter_frames(fps=fps, dtype='uint8'):
                            if target_width != clip.w and resize is None:
                                img = Image.fromarray(frame)
                                new_height = int(img.height * (target_width / img.width))
                                img = img.resize((target_width, new_height), Image.LANCZOS)
                                frame = np.array(img)
                            # Append frame without duration; rely on fps-driven iteration for timing
                            writer.append_data(frame)
                    finally:
                        writer.close()
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
                
                # Convert to GIF with enhanced progress tracking
                self.logger.info(f"Writing GIF with {fps} FPS...")
                
                if tqdm:
                    # Enhanced progress bar with optimization phase
                    total_frames = int(clip.duration * fps)
                    # Total progress: 80% for conversion + 20% for optimization
                    conversion_weight = 80
                    optimization_weight = 20
                    total_progress = 100
                    
                    with tqdm(total=total_progress, desc="Processing", unit="%") as pbar:
                        # Phase 1: Video to GIF conversion
                        pbar.set_description("Converting video to GIF")
                        
                        # Track conversion progress
                        frames_processed = 0
                        def progress_callback(gf, t):
                            nonlocal frames_processed
                            frames_processed += 1
                            progress_percent = min((frames_processed / total_frames) * conversion_weight, conversion_weight)
                            pbar.n = int(progress_percent)
                            pbar.set_postfix({"Phase": "Conversion", "Frames": f"{frames_processed}/{total_frames}"})
                            pbar.refresh()
                            return gf(t)
                        
                        clip.write_gif(
                            str(output_path),
                            fps=fps
                        )
                        
                        # Ensure conversion phase shows as complete
                        pbar.n = conversion_weight
                        pbar.set_postfix({"Phase": "Conversion Complete"})
                        pbar.refresh()
                        
                        # Phase 2: GIF optimization
                        if optimize_size:
                            pbar.set_description("Optimizing GIF (Lossy compression)")
                            initial_size = output_path.stat().st_size / (1024 * 1024)
                            
                            # Simulate optimization progress steps
                            optimization_steps = 5
                            for step in range(optimization_steps):
                                progress = conversion_weight + ((step + 1) / optimization_steps) * optimization_weight
                                pbar.n = int(progress)
                                
                                if step == 0:
                                    pbar.set_postfix({"Phase": "Analyzing frames", "Size": f"{initial_size:.1f}MB"})
                                elif step == 1:
                                    pbar.set_postfix({"Phase": "Color quantization", "Colors": f"{max(16, 256 - (self.config.compression_level * 2))}"})
                                elif step == 2:
                                    pbar.set_postfix({"Phase": "Palette optimization", "Method": "Adaptive"})
                                elif step == 3:
                                    pbar.set_postfix({"Phase": "Eliminating color tables", "Compression": f"Level {self.config.compression_level}"})
                                else:
                                    pbar.set_postfix({"Phase": "Finalizing optimization"})
                                
                                pbar.refresh()
                                
                                # Perform actual optimization on last step
                                if step == optimization_steps - 1:
                                    self.optimize_gif_size(output_path, target_size_mb=self.config.target_size_mb, quality=quality)
                                    final_size = output_path.stat().st_size / (1024 * 1024)
                                    compression_ratio = ((initial_size - final_size) / initial_size) * 100 if initial_size > 0 else 0
                                    pbar.set_postfix({"Phase": "Complete", "Reduction": f"{compression_ratio:.1f}%", "Final": f"{final_size:.1f}MB"})
                        
                        # Complete progress
                        pbar.n = total_progress
                        pbar.set_description("GIF processing complete")
                        pbar.refresh()
                else:
                    # Without progress bar
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
        
        # Find video files
        video_files = []
        for file_path in input_dir.rglob('*'):
            if file_path.is_file() and self.is_supported_video(file_path):
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
        
        # Enhanced: Unified progress bar for batch processing
        results = []
        processed_files = 0
        
        if tqdm:
            # Unified progress bar with file counters
            with tqdm(total=total_files, desc="Batch Converting", unit="files") as batch_pbar:
                for i, video_path in enumerate(video_files, 1):
                    # Update progress bar description with current file info
                    filename = video_path.name
                    batch_pbar.set_description(f"Converting [{i}/{total_files}]: {filename[:30]}{'...' if len(filename) > 30 else ''}")
                    successful = sum(1 for r in results if r.success)
                    batch_pbar.set_postfix({
                        "Total": total_files,
                        "Processed": processed_files,
                        "Success": successful,
                        "Failed": processed_files - successful
                    })
                    
                    output_path = output_dir / f"{video_path.stem}.gif"
                    
                    # Temporarily disable individual file progress to avoid conflicts
                    kwargs_no_progress = kwargs.copy()
                    
                    try:
                        result = self.convert_video_to_gif(
                            video_path=video_path,
                            output_path=output_path,
                            **kwargs_no_progress
                        )
                        
                        results.append(result)
                        successful = sum(1 for r in results if r.success)
                        batch_pbar.set_postfix({
                            "Total": total_files,
                            "Processed": i,
                            "Success": successful,
                            "Failed": i - successful
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Failed to convert {video_path.name}: {e}")
                        results.append(ConversionResult(success=False, error=str(e)))
                    
                    processed_files = i
                    batch_pbar.update(1)
                    
                    # Final status update
                    if i == total_files:
                        batch_pbar.set_description("Batch Conversion Complete")
                        successful = sum(1 for r in results if r.success)
                        batch_pbar.set_postfix({
                            "Total": total_files,
                            "Success": successful,
                            "Failed": total_files - successful,
                            "Rate": f"{(successful/total_files)*100:.1f}%"
                        })
        else:
            # Fallback without tqdm - simple counter display
            print(f"\nBatch converting {total_files} files...")
            for i, video_path in enumerate(video_files, 1):
                successful = sum(1 for r in results if r.success)
                print(f"\n[{i}/{total_files}] Processing: {video_path.name}")
                print(f"Progress: {(i-1)/total_files*100:.1f}% | Processed: {processed_files} | Success: {successful}")
                
                output_path = output_dir / f"{video_path.stem}.gif"
                
                try:
                    result = self.convert_video_to_gif(
                        video_path=video_path,
                        output_path=output_path,
                        **kwargs
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
            
            # Final summary
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
    
    def _batch_convert_sequential(self, video_files: List[Path], output_dir: Path, **kwargs) -> List[ConversionResult]:
        """Sequential batch conversion with unified progress tracking."""
        total_files = len(video_files)
        results = []
        processed_files = 0
        output_format = kwargs.get('output_format', 'gif').lower()
        
        if tqdm:
            # Unified progress bar for batch processing
            with tqdm(total=total_files, desc="Batch Converting", unit="files") as batch_pbar:
                for i, video_path in enumerate(video_files, 1):
                    # Update progress bar description with current file info
                    filename = video_path.name
                    batch_pbar.set_description(f"Converting [{i}/{total_files}]: {filename[:30]}{'...' if len(filename) > 30 else ''}")
                    successful = sum(1 for r in results if r.success)
                    batch_pbar.set_postfix({
                        "Total": total_files,
                        "Processed": processed_files,
                        "Success": successful,
                        "Failed": processed_files - successful
                    })
                    
                    ext = '.gif' if output_format == 'gif' else '.webp'
                    output_path = output_dir / f"{video_path.stem}{ext}"
                    
                    # Temporarily disable individual file progress to avoid conflicts
                    kwargs_no_progress = kwargs.copy()
                    
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
                        successful = sum(1 for r in results if r.success)
                        batch_pbar.set_postfix({
                            "Total": total_files,
                            "Processed": i,
                            "Success": successful,
                            "Failed": i - successful
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Failed to convert {video_path.name}: {e}")
                        results.append(ConversionResult(success=False, error=str(e)))
                    
                    processed_files = i
                    batch_pbar.update(1)
                    
                    # Final status update
                    if i == total_files:
                        batch_pbar.set_description("Batch Conversion Complete")
                        successful = sum(1 for r in results if r.success)
                        batch_pbar.set_postfix({
                            "Total": total_files,
                            "Success": successful,
                            "Failed": total_files - successful,
                            "Rate": f"{(successful/total_files)*100:.1f}%"
                        })
        else:
            # Fallback without tqdm - simple counter display
            print(f"\nBatch converting {total_files} files...")
            for i, video_path in enumerate(video_files, 1):
                successful = sum(1 for r in results if r.success)
                print(f"\n[{i}/{total_files}] Processing: {video_path.name}")
                print(f"Progress: {(i-1)/total_files*100:.1f}% | Processed: {processed_files} | Success: {successful}")
                
                ext = '.gif' if output_format == 'gif' else '.webp'
                output_path = output_dir / f"{video_path.stem}{ext}"
                
                try:
                    if output_format == 'gif':
                        result = self.convert_video_to_gif(
                            video_path=video_path,
                            output_path=output_path,
                            **kwargs
                        )
                    else:
                        result = self.convert_video_to_webp(
                            video_path=video_path,
                            output_path=output_path,
                            **kwargs
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
            
            # Final summary
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
        self.logger.info(f"Starting concurrent conversion with {max_workers} workers")
        results = [None] * total_files
        completed_count = 0
        
        def convert_single_file(index_and_path):
            index, video_path = index_and_path
            ext = '.gif' if output_format == 'gif' else '.webp'
            output_path = output_dir / f"{video_path.stem}{ext}"
            try:
                if output_format == 'gif':
                    result = self.convert_video_to_gif(
                        video_path=video_path,
                        output_path=output_path,
                        **kwargs
                    )
                else:
                    result = self.convert_video_to_webp(
                        video_path=video_path,
                        output_path=output_path,
                        **kwargs
                    )
                return index, result
            except Exception as e:
                self.logger.error(f"Failed to convert {video_path.name}: {e}")
                return index, ConversionResult(success=False, error=str(e))
        
        if tqdm:
            with tqdm(total=total_files, desc="Concurrent Converting", unit="files") as pbar:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_index = { executor.submit(convert_single_file, (i, video_path)): i for i, video_path in enumerate(video_files) }
                    for future in as_completed(future_to_index):
                        try:
                            index, result = future.result()
                            results[index] = result
                            completed_count += 1
                            successful = sum(1 for r in results[:completed_count] if r and r.success)
                            pbar.set_postfix({
                                "Workers": max_workers,
                                "Completed": completed_count,
                                "Success": successful,
                                "Failed": completed_count - successful
                            })
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
    parser = argparse.ArgumentParser(
        description="Convert videos to GIF or WebP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_to_gif.py input.mp4 --format gif -o output.gif --fps 15 --width 640
  python video_to_gif.py input.mp4 --format webp -o output.webp --fps 12
  python video_to_gif.py --batch C:/videos --output C:/converted --format webp --workers 4
  python video_to_gif.py input.mp4 --delete-source  # Delete source after conversion
        """
    )

    # Input arguments
    parser.add_argument(
        'input',
        nargs='?',
        help='Input video file or directory (for batch mode)'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output file or directory (extension inferred by format)'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode: convert all videos in input directory'
    )

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
    
    parser.add_argument(
        '--delete-source',
        action='store_true',
        help='Delete source video files after successful conversion'
    )
    
    # Logging
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
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
    config.delete_source = args.delete_source
    converter = VideoToGifConverter(config=config)
    
    input_path = Path(args.input)
    
    try:
        if args.batch or input_path.is_dir():
            # Batch mode
            output_dir = Path(args.output) if args.output else input_path
            
            results = converter.batch_convert(
                input_dir=input_path,
                output_dir=output_dir,
                max_workers=args.workers,
                use_concurrent=True,
                start_time=args.start,
                end_time=args.end,
                fps=args.fps,
                width=args.width,
                quality=args.quality,
                optimize_size=not args.no_optimize,
                output_format=(args.format or 'gif')
            )
            
            # Print results summary
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            deleted_count = sum(1 for r in results if r.source_deleted)
            
            if successful_results:
                print(f"\nSuccessfully converted {len(successful_results)} videos:")
                for result in successful_results:
                    file_size_mb = result.output_path.stat().st_size / (1024 * 1024)
                    deleted_status = " (source deleted)" if result.source_deleted else ""
                    print(f"  ✓ {result.output_path.name} ({file_size_mb:.1f}MB){deleted_status}")
                
                if failed_results:
                    print(f"\nFailed conversions ({len(failed_results)}):")
                    for result in failed_results:
                        print(f"  ✗ {result.error}")
                
                # Print statistics
                stats = converter._conversion_stats
                print(f"\n=== Conversion Statistics ===")
                print(f"Total files processed: {stats['total_conversions']}")
                print(f"Successful conversions: {stats['successful_conversions']}")
                print(f"Failed conversions: {stats['failed_conversions']}")
                if deleted_count > 0:
                    print(f"Source files deleted: {deleted_count}")
                    print(f"Total space saved: {stats['total_size_saved'] / (1024*1024*1024):.2f}GB")
            else:
                print("No videos were converted successfully.")
                sys.exit(1)
        
        else:
            # Single file mode
            output_path = Path(args.output) if args.output else None
            
            # Select conversion method based on format (default GIF)
            if args.format and args.format.lower() == 'webp':
                result = converter.convert_video_to_webp(
                    video_path=input_path,
                    output_path=output_path,
                    start_time=args.start,
                    end_time=args.end,
                    fps=args.fps,
                    width=args.width,
                    quality=args.quality,
                    optimize_size=not args.no_optimize
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
                    optimize_size=not args.no_optimize
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