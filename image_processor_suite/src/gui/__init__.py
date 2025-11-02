#!/usr/bin/env python3
"""
GUI package for Image Processor Suite

Provides a comprehensive graphical user interface built with tkinter including:
- Main application window
- Image conversion interface
- Face recognition and sorting interface
- Progress tracking and feedback
- Configuration management
"""

from .main_interface import MainApplication

__all__ = [
    "MainApplication",
]