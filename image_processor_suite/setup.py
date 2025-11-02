#!/usr/bin/env python3
"""
Setup script for Image Processor Suite

A comprehensive image processing application that combines:
- WEBP to JPG conversion
- Face recognition and secure sorting
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="image-processor-suite",
    version="1.0.0",
    author="Image Processor Suite Team",
    author_email="contact@imageprocessor.com",
    description="A comprehensive image processing application with conversion and face recognition capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/imageprocessor/image-processor-suite",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "Pillow>=10.0.0",
        "face-recognition>=1.3.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "image-processor=image_processor_suite.main:main",
        ],
        "gui_scripts": [
            "image-processor-gui=image_processor_suite.main:main_gui",
        ],
    },
    include_package_data=True,
    package_data={
        "image_processor_suite": ["assets/*", "config/*"],
    },
    zip_safe=False,
)