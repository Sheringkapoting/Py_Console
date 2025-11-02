#!/usr/bin/env python3
"""
Face Recognition Utilities Module

Consolidated face recognition functionality including:
- Face detection and encoding
- Face matching and comparison
- Image sorting by known faces
- Batch processing with progress tracking
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import logging
from dataclasses import dataclass

import face_recognition
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageFile
from tqdm import tqdm

# Pillow hardening / safety
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 50_000_000  # ~50MP cap


@dataclass
class FaceMatch:
    """Result of a face matching operation."""
    is_match: bool
    confidence: float
    distance: float
    reference_name: Optional[str] = None


class FaceRecognitionUtils:
    """
    Consolidated face recognition utilities with detection, matching, and sorting.
    """
    
    # Supported image formats
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    
    # File size limits
    MAX_BYTES = 50 * 1024 * 1024  # 50 MB per file
    
    # Default settings
    DEFAULT_TOLERANCE = 0.5
    DEFAULT_JITTER = 1
    DEFAULT_MODEL = 'hog'  # 'hog' for speed, 'cnn' for accuracy
    
    def __init__(self, 
                 tolerance: float = DEFAULT_TOLERANCE,
                 jitter: int = DEFAULT_JITTER,
                 model: str = DEFAULT_MODEL,
                 max_workers: Optional[int] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize face recognition utilities.
        
        Args:
            tolerance: Face matching tolerance (lower = stricter)
            jitter: Number of jitter samples for encoding
            model: Detection model ('hog' for speed, 'cnn' for accuracy)
            max_workers: Maximum worker processes for batch operations
            logger: Optional logger instance
        """
        self.tolerance = tolerance
        self.jitter = max(1, jitter)
        self.model = model
        self.max_workers = max_workers or min(4, os.cpu_count() or 1)
        self.logger = logger or self._setup_logger()
        
        # Reference faces storage
        self.reference_faces = {}  # {name: {'encoding': array, 'dest_dir': Path}}
        
        # Processing statistics
        self.stats = self._init_stats()
        
        # Threading control
        self._cancel_flag = threading.Event()
    
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
            'matched': 0,
            'unmatched': 0,
            'errors': 0,
            'moved_files': {},  # {dest_dir: count}
            'error_details': []
        }
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = self._init_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.stats.copy()
    
    def detect_faces(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        Detect all faces in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of face detection results with locations and encodings
        """
        try:
            # Load and validate image
            image = face_recognition.load_image_file(str(image_path))
            
            # Find face locations
            face_locations = face_recognition.face_locations(image, model=self.model)
            
            if not face_locations:
                return []
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(
                image, face_locations, num_jitters=self.jitter
            )
            
            # Combine results
            faces = []
            for i, (location, encoding) in enumerate(zip(face_locations, face_encodings)):
                top, right, bottom, left = location
                face_info = {
                    'index': i,
                    'location': location,
                    'encoding': encoding,
                    'size': {
                        'width': right - left,
                        'height': bottom - top,
                        'area': (right - left) * (bottom - top)
                    },
                    'quality': self._assess_face_quality(image, location)
                }
                faces.append(face_info)
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Face detection failed for {image_path}: {e}")
            return []
    
    def _assess_face_quality(self, image: np.ndarray, location: Tuple[int, int, int, int]) -> float:
        """Assess face quality based on size and clarity."""
        try:
            top, right, bottom, left = location
            face_width = right - left
            face_height = bottom - top
            face_area = face_width * face_height
            
            # Size score (larger faces are generally better)
            size_score = min(1.0, face_area / 10000)  # Normalize to 100x100 pixels
            
            # Aspect ratio score (closer to 1:1 is better for faces)
            aspect_ratio = face_width / max(face_height, 1)
            aspect_score = 1.0 - abs(aspect_ratio - 1.0)
            
            # Position score (faces closer to center are often better)
            img_height, img_width = image.shape[:2]
            face_center_x = (left + right) / 2
            face_center_y = (top + bottom) / 2
            img_center_x = img_width / 2
            img_center_y = img_height / 2
            
            distance_from_center = np.sqrt(
                ((face_center_x - img_center_x) / img_width) ** 2 +
                ((face_center_y - img_center_y) / img_height) ** 2
            )
            position_score = max(0.0, 1.0 - distance_from_center)
            
            # Combined quality score
            quality = (size_score * 0.5 + aspect_score * 0.3 + position_score * 0.2)
            return min(1.0, max(0.0, quality))
            
        except Exception:
            return 0.5  # Default quality score
    
    def extract_best_face_encoding(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Extract the best face encoding from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Face encoding array or None if no face found
        """
        faces = self.detect_faces(image_path)
        
        if not faces:
            return None
        
        # Return encoding of the highest quality face
        best_face = max(faces, key=lambda f: f['quality'])
        return best_face['encoding']
    
    def match_face(self, face_encoding: np.ndarray, 
                   reference_encodings: List[np.ndarray],
                   reference_names: Optional[List[str]] = None) -> FaceMatch:
        """
        Match a face encoding against reference encodings.
        
        Args:
            face_encoding: Face encoding to match
            reference_encodings: List of reference face encodings
            reference_names: Optional list of reference names
            
        Returns:
            FaceMatch result
        """
        try:
            # Calculate distances
            distances = face_recognition.face_distance(reference_encodings, face_encoding)
            
            # Find best match
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            
            # Determine if it's a match
            is_match = best_distance <= self.tolerance
            confidence = max(0.0, 1.0 - (best_distance / 1.0))  # Convert distance to confidence
            
            reference_name = None
            if reference_names and is_match:
                reference_name = reference_names[best_match_index]
            
            return FaceMatch(
                is_match=is_match,
                confidence=confidence,
                distance=best_distance,
                reference_name=reference_name
            )
            
        except Exception as e:
            self.logger.error(f"Face matching failed: {e}")
            return FaceMatch(is_match=False, confidence=0.0, distance=1.0)
    
    def add_reference_face(self, face_image_path: Path, 
                          destination_dir: Path,
                          name: Optional[str] = None) -> bool:
        """
        Add a reference face for sorting.
        
        Args:
            face_image_path: Path to the reference face image
            destination_dir: Directory where matched images will be moved
            name: Optional name for the reference face
            
        Returns:
            True if reference face was added successfully
        """
        try:
            # Extract face encoding
            encoding = self.extract_best_face_encoding(face_image_path)
            
            if encoding is None:
                self.logger.error(f"No face found in reference image: {face_image_path}")
                return False
            
            # Generate name if not provided
            if name is None:
                name = face_image_path.stem
            
            # Sanitize name
            name = self._sanitize_name(name)
            
            # Create destination directory if it doesn't exist
            destination_dir.mkdir(parents=True, exist_ok=True)
            
            # Store reference face
            self.reference_faces[name] = {
                'encoding': encoding,
                'dest_dir': destination_dir,
                'source_image': face_image_path
            }
            
            self.logger.info(f"Added reference face '{name}' -> {destination_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add reference face {face_image_path}: {e}")
            return False
    
    def clear_reference_faces(self) -> None:
        """Clear all reference faces."""
        self.reference_faces.clear()
        self.logger.info("Cleared all reference faces")
    
    def sort_images(self, source_dir: Path,
                   recursive: bool = True,
                   progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
        """
        Sort images by matching them against reference faces.
        
        Args:
            source_dir: Directory containing images to sort
            recursive: Whether to process subdirectories
            progress_callback: Optional progress callback function
            
        Returns:
            Processing statistics
        """
        try:
            if not self.reference_faces:
                return {'error': 'No reference faces added'}
            
            # Collect images to process
            images = self._collect_images(source_dir, recursive)
            
            if not images:
                return {'error': 'No images found to process'}
            
            self.logger.info(f"Found {len(images)} images to process")
            
            # Prepare reference data
            reference_encodings = [ref['encoding'] for ref in self.reference_faces.values()]
            reference_names = list(self.reference_faces.keys())
            
            # Process images with progress tracking
            with tqdm(total=len(images), desc="Sorting images") as pbar:
                for i, image_path in enumerate(images):
                    if self._cancel_flag.is_set():
                        break
                    
                    self._process_single_image(image_path, reference_encodings, reference_names)
                    
                    if progress_callback:
                        progress_callback(i + 1, len(images), str(image_path))
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Matched': self.stats['matched'],
                        'Unmatched': self.stats['unmatched'],
                        'Errors': self.stats['errors']
                    })
            
            return self.get_stats()
            
        except Exception as e:
            self.logger.error(f"Image sorting failed: {e}")
            return {'error': str(e)}
    
    def _collect_images(self, source_dir: Path, recursive: bool) -> List[Path]:
        """Collect image files from directory."""
        images = []
        
        try:
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for file_path in source_dir.glob(pattern):
                if (file_path.is_file() and 
                    file_path.suffix.lower() in self.VALID_EXTS and
                    self._validate_image_file(file_path)):
                    images.append(file_path)
                    
        except Exception as e:
            self.logger.error(f"Error collecting images: {e}")
        
        return images
    
    def _validate_image_file(self, file_path: Path) -> bool:
        """Validate image file for processing."""
        try:
            # Check file size
            if file_path.stat().st_size > self.MAX_BYTES:
                return False
            
            # Try to open with PIL
            with Image.open(file_path) as img:
                img.verify()
            
            return True
            
        except Exception:
            return False
    
    def _process_single_image(self, image_path: Path, 
                            reference_encodings: List[np.ndarray],
                            reference_names: List[str]) -> None:
        """Process a single image for face matching and sorting."""
        try:
            self.stats['processed'] += 1
            
            # Extract face encoding
            face_encoding = self.extract_best_face_encoding(image_path)
            
            if face_encoding is None:
                self.stats['unmatched'] += 1
                return
            
            # Match against reference faces
            match_result = self.match_face(face_encoding, reference_encodings, reference_names)
            
            if match_result.is_match and match_result.reference_name:
                # Move image to appropriate directory
                dest_dir = self.reference_faces[match_result.reference_name]['dest_dir']
                success = self._move_image_safely(image_path, dest_dir)
                
                if success:
                    self.stats['matched'] += 1
                    self.stats['moved_files'][str(dest_dir)] = \
                        self.stats['moved_files'].get(str(dest_dir), 0) + 1
                else:
                    self.stats['errors'] += 1
            else:
                self.stats['unmatched'] += 1
                
        except Exception as e:
            self.stats['errors'] += 1
            error_msg = f"Processing failed: {str(e)}"
            self.stats['error_details'].append(f"{image_path}: {error_msg}")
            self.logger.error(f"Error processing {image_path}: {e}")
    
    def _move_image_safely(self, src_path: Path, dest_dir: Path) -> bool:
        """Safely move image to destination directory."""
        try:
            # Ensure destination directory exists
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique destination path
            dest_path = self._generate_unique_path(dest_dir, src_path.name)
            
            # Move file
            shutil.move(str(src_path), str(dest_path))
            
            self.logger.debug(f"Moved {src_path} -> {dest_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to move {src_path} to {dest_dir}: {e}")
            return False
    
    def _generate_unique_path(self, dest_dir: Path, filename: str) -> Path:
        """Generate a unique file path in the destination directory."""
        base_path = dest_dir / filename
        
        if not base_path.exists():
            return base_path
        
        # Generate unique name with counter
        name_stem = base_path.stem
        suffix = base_path.suffix
        counter = 1
        
        while True:
            new_name = f"{name_stem}_{counter}{suffix}"
            new_path = dest_dir / new_name
            
            if not new_path.exists():
                return new_path
            
            counter += 1
            
            # Safety check to prevent infinite loop
            if counter > 10000:
                raise ValueError(f"Cannot generate unique filename for {filename}")
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for use as directory/file name."""
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        sanitized = re.sub(r'\s+', '_', sanitized)  # Replace spaces with underscores
        sanitized = sanitized.strip('._')  # Remove leading/trailing dots and underscores
        
        # Ensure name is not empty
        if not sanitized:
            sanitized = "unknown"
        
        return sanitized
    
    def cancel_processing(self) -> None:
        """Cancel ongoing processing operations."""
        self._cancel_flag.set()
        self.logger.info("Processing cancellation requested")
    
    def set_tolerance(self, tolerance: float) -> None:
        """Update face matching tolerance."""
        self.tolerance = max(0.0, min(1.0, tolerance))
        self.logger.info(f"Face matching tolerance set to {self.tolerance}")
    
    def get_reference_faces_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all reference faces."""
        info = {}
        for name, ref_data in self.reference_faces.items():
            info[name] = {
                'destination_dir': str(ref_data['dest_dir']),
                'source_image': str(ref_data['source_image']),
                'encoding_shape': ref_data['encoding'].shape
            }
        return info