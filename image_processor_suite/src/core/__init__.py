#!/usr/bin/env python3
"""
Core functionality package for Image Processor Suite

This package contains the core business logic for:
- Image conversion operations
- Face recognition and sorting
"""

# Import main classes for easy access
from .image_utils import ImageUtils
from .face_recognition_utils import FaceRecognitionUtils

__all__ = [
    "ImageUtils",
    "FaceRecognitionUtils",
]