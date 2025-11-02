#!/usr/bin/env python3
"""
Image Processor Suite

A comprehensive image processing application that combines:
- WEBP to JPG conversion
- Face recognition and secure sorting

Author: Image Processor Suite Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Image Processor Suite Team"
__email__ = "contact@imageprocessor.com"
__license__ = "MIT"
__description__ = "A comprehensive image processing application with conversion and face recognition capabilities"

# Import main classes for easy access
from ..core.image_utils import ImageUtils
from ..core.face_recognition_utils import FaceRecognitionUtils
from ..gui.main_interface import MainApplication

# Define what gets imported with "from image_processor_suite import *"
__all__ = [
    "ImageUtils",
    "FaceRecognitionUtils", 
    "MainApplication",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
]

# Package metadata
METADATA = {
    "name": "image-processor-suite",
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "license": __license__,
    "description": __description__,
    "url": "https://github.com/imageprocessor/image-processor-suite",
    "python_requires": ">=3.8",
}