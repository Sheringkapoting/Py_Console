#!/usr/bin/env python3
"""
Image Utilities Module

Consolidated image processing utilities including:
- Format detection and validation
- Image conversion functionality
- File operations and safety checks
- Unicode path handling
"""

import os
import struct
from typing import List, Tuple, Optional, Dict, Any, Set
from pathlib import Path
from PIL import Image, UnidentifiedImageError, ImageFile
import mimetypes
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Safety: mitigate decompression bombs
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 50_000_000  # ~50MP cap


class ImageUtils:
    """
    Consolidated image processing utilities with format detection and conversion.
    """
    
    # Supported formats
    SUPPORTED_INPUT_FORMATS = {'.webp', '.png', '.jpeg', '.jpg', '.bmp', '.tiff', '.tif'}
    CONVERTIBLE_FORMATS = {'.webp', '.png', '.jpeg', '.bmp', '.tiff', '.tif'}
    SKIP_FORMATS = {'.gif', '.jpg'}  # Already JPG or animated
    
    # Magic number signatures for format detection
    MAGIC_SIGNATURES = {
        b'\xFF\xD8\xFF': 'JPEG',
        b'\x89PNG\r\n\x1a\n': 'PNG',
        b'RIFF': 'WEBP',  # Need to check further for WEBP
        b'GIF87a': 'GIF',
        b'GIF89a': 'GIF',
        b'BM': 'BMP',
        b'II*\x00': 'TIFF',  # Little-endian TIFF
        b'MM\x00*': 'TIFF',  # Big-endian TIFF
    }
    
    # File size limits
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MIN_FILE_SIZE = 100  # 100 bytes
    
    # Default conversion settings
    DEFAULT_QUALITY = 95
    DEFAULT_MAX_WORKERS = 4
    
    def __init__(self, 
                 quality: int = DEFAULT_QUALITY,
                 max_workers: int = DEFAULT_MAX_WORKERS,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize ImageUtils.
        
        Args:
            quality: JPEG quality for conversions (1-100)
            max_workers: Maximum worker threads for batch processing
            logger: Optional logger instance
        """
        self.quality = max(1, min(100, quality))
        self.max_workers = max(1, max_workers)
        self.logger = logger or self._setup_logger()
        self.stats = self._init_stats()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _init_stats(self) -> Dict[str, Any]:
        """Initialize processing statistics."""
        return {
            'processed': 0,
            'converted': 0,
            'skipped': 0,
            'errors': 0,
            'total_size_saved': 0,
            'error_details': []
        }
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = self._init_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.stats.copy()
    
    def detect_format(self, file_path: Path) -> Optional[str]:
        """
        Detect image format using multiple methods.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Format name or None if not detected
        """
        try:
            # First try magic number detection
            format_by_magic = self._detect_by_magic(file_path)
            if format_by_magic:
                return format_by_magic
            
            # Fall back to PIL detection
            return self._detect_by_pillow(file_path)
            
        except Exception as e:
            self.logger.warning(f"Format detection failed for {file_path}: {e}")
            return None
    
    def _detect_by_magic(self, file_path: Path) -> Optional[str]:
        """Detect format using magic number signatures."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(12)  # Read first 12 bytes
                
            for signature, format_name in self.MAGIC_SIGNATURES.items():
                if header.startswith(signature):
                    # Special check for WEBP
                    if signature == b'RIFF' and b'WEBP' in header:
                        return 'WEBP'
                    elif signature == b'RIFF':
                        continue
                    return format_name
                    
        except Exception:
            pass
        return None
    
    def _detect_by_pillow(self, file_path: Path) -> Optional[str]:
        """Detect format using PIL."""
        try:
            with Image.open(file_path) as img:
                return img.format
        except Exception:
            return None
    
    def is_supported_format(self, file_path: Path) -> bool:
        """
        Check if file format is supported for processing.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if format is supported
        """
        return file_path.suffix.lower() in self.SUPPORTED_INPUT_FORMATS
    
    def should_convert(self, file_path: Path) -> bool:
        """
        Check if file should be converted.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file should be converted
        """
        suffix = file_path.suffix.lower()
        return suffix in self.CONVERTIBLE_FORMATS and suffix not in self.SKIP_FORMATS
    
    def validate_image_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Validate image file for processing.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if file exists
            if not file_path.exists():
                return False, "File does not exist"
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size < self.MIN_FILE_SIZE:
                return False, f"File too small ({file_size} bytes)"
            if file_size > self.MAX_FILE_SIZE:
                return False, f"File too large ({file_size} bytes)"
            
            # Check if it's a supported format
            if not self.is_supported_format(file_path):
                return False, f"Unsupported format: {file_path.suffix}"
            
            # Try to open with PIL
            with Image.open(file_path) as img:
                img.verify()
            
            return True, "Valid image file"
            
        except UnidentifiedImageError:
            return False, "Cannot identify image file"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def convert_single_image(self, file_path: Path, 
                           output_path: Optional[Path] = None,
                           delete_original: bool = True) -> Tuple[bool, str, Optional[Path]]:
        """
        Convert a single image to JPG format.
        
        Args:
            file_path: Input image path
            output_path: Optional output path (auto-generated if None)
            delete_original: Whether to delete the original file
            
        Returns:
            Tuple of (success, message, output_path)
        """
        try:
            # Validate input file
            is_valid, error_msg = self.validate_image_file(file_path)
            if not is_valid:
                self.stats['errors'] += 1
                return False, error_msg, None
            
            # Check if conversion is needed
            if not self.should_convert(file_path):
                self.stats['skipped'] += 1
                return True, "File skipped (already JPG or unsupported)", None
            
            # Generate output path if not provided
            if output_path is None:
                output_path = file_path.with_suffix('.jpg')
            
            # Convert image
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparency
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as JPG
                img.save(output_path, 'JPEG', quality=self.quality, optimize=True)
            
            # Calculate size savings
            original_size = file_path.stat().st_size
            new_size = output_path.stat().st_size
            size_saved = original_size - new_size
            self.stats['total_size_saved'] += size_saved
            
            # Delete original if requested
            if delete_original and output_path != file_path:
                file_path.unlink()
            
            self.stats['converted'] += 1
            self.stats['processed'] += 1
            
            return True, f"Converted successfully (saved {size_saved} bytes)", output_path
            
        except Exception as e:
            self.stats['errors'] += 1
            error_msg = f"Conversion failed: {str(e)}"
            self.stats['error_details'].append(f"{file_path}: {error_msg}")
            return False, error_msg, None
    
    def convert_images_batch(self, 
                           folder: Path, 
                           recursive: bool = True,
                           delete_originals: bool = True,
                           progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Convert multiple images in a folder.
        
        Args:
            folder: Source folder path
            recursive: Whether to process subdirectories
            delete_originals: Whether to delete original files
            progress_callback: Optional progress callback function
            
        Returns:
            Processing statistics
        """
        try:
            # Collect image files
            files = self._collect_image_files(folder, recursive)
            
            if not files:
                return {'error': 'No convertible images found'}
            
            self.logger.info(f"Found {len(files)} images to process")
            
            # Process files with progress tracking
            with tqdm(total=len(files), desc="Converting images") as pbar:
                for i, file_path in enumerate(files):
                    success, message, output_path = self.convert_single_image(
                        file_path, delete_original=delete_originals
                    )
                    
                    if progress_callback:
                        progress_callback(i + 1, len(files), str(file_path))
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Converted': self.stats['converted'],
                        'Errors': self.stats['errors']
                    })
            
            return self.get_stats()
            
        except Exception as e:
            self.logger.error(f"Batch conversion failed: {e}")
            return {'error': str(e)}
    
    def _collect_image_files(self, folder: Path, recursive: bool) -> List[Path]:
        """Collect image files from folder."""
        files = []
        
        try:
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for file_path in folder.glob(pattern):
                if file_path.is_file() and self.should_convert(file_path):
                    files.append(file_path)
                    
        except Exception as e:
            self.logger.error(f"Error collecting files: {e}")
        
        return files
    
    def get_image_info(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an image file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary with image information or None if error
        """
        try:
            with Image.open(file_path) as img:
                return {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height,
                    'file_size': file_path.stat().st_size,
                    'has_transparency': img.mode in ('RGBA', 'LA', 'P')
                }
        except Exception as e:
            self.logger.error(f"Error getting image info for {file_path}: {e}")
            return None