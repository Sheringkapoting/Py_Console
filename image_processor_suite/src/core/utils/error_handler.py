#!/usr/bin/env python3
"""
Error Handler Module

Provides enhanced error handling and user feedback for Unicode and other processing issues.
"""

import logging
import traceback
from typing import Optional, Dict, Any, Callable, Tuple
from pathlib import Path
import tkinter as tk
from tkinter import messagebox


class UnicodeErrorHandler:
    """
    Specialized error handler for Unicode-related issues in image processing.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, gui_mode: bool = True):
        """
        Initialize the Unicode error handler.
        
        Args:
            logger: Logger instance for error reporting
            gui_mode: Whether to show GUI error dialogs
        """
        self.logger = logger or logging.getLogger(__name__)
        self.gui_mode = gui_mode
        self.error_stats = {
            'unicode_errors': 0,
            'path_errors': 0,
            'encoding_errors': 0,
            'file_access_errors': 0,
            'total_errors': 0
        }
    
    def handle_unicode_path_error(self, file_path: Path, error: Exception, 
                                 operation: str = "processing") -> Tuple[bool, str]:
        """
        Handle Unicode path-related errors with detailed feedback.
        
        Args:
            file_path: The problematic file path
            error: The exception that occurred
            operation: Description of the operation being performed
            
        Returns:
            Tuple of (success, error_message)
        """
        self.error_stats['unicode_errors'] += 1
        self.error_stats['total_errors'] += 1
        
        # Analyze the error type
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Create user-friendly error message
        if "UnicodeDecodeError" in error_type or "UnicodeEncodeError" in error_type:
            self.error_stats['encoding_errors'] += 1
            user_msg = (
                f"Unicode encoding issue with file: {file_path.name}\n\n"
                f"The file path contains characters that cannot be processed properly.\n"
                f"Consider renaming the file to use only ASCII characters.\n\n"
                f"Technical details: {error_msg}"
            )
        elif "FileNotFoundError" in error_type or "PermissionError" in error_type:
            self.error_stats['file_access_errors'] += 1
            user_msg = (
                f"File access error: {file_path.name}\n\n"
                f"The file may have been moved, deleted, or access is restricted.\n"
                f"Please check file permissions and try again.\n\n"
                f"Technical details: {error_msg}"
            )
        else:
            self.error_stats['path_errors'] += 1
            user_msg = (
                f"Path processing error: {file_path.name}\n\n"
                f"There was an issue processing this file path during {operation}.\n"
                f"This may be due to special characters or path length limitations.\n\n"
                f"Technical details: {error_msg}"
            )
        
        # Log the error
        self.logger.error(
            f"Unicode path error during {operation}: {file_path} - {error_type}: {error_msg}"
        )
        
        # Show GUI dialog if in GUI mode
        if self.gui_mode:
            try:
                messagebox.showerror(
                    f"Unicode Error - {operation.title()}",
                    user_msg
                )
            except Exception:
                # Fallback if GUI is not available
                print(f"ERROR: {user_msg}")
        
        return False, user_msg
    
    def handle_face_recognition_error(self, file_path: Path, error: Exception) -> Tuple[bool, str]:
        """
        Handle face recognition specific errors.
        
        Args:
            file_path: The problematic file path
            error: The exception that occurred
            
        Returns:
            Tuple of (success, error_message)
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        if "UnicodeError" in error_type or "UnicodeDecodeError" in error_type:
            return self.handle_unicode_path_error(file_path, error, "face recognition")
        
        # Handle other face recognition errors
        user_msg = (
            f"Face recognition error: {file_path.name}\n\n"
            f"Could not process this image for face detection.\n"
            f"The image may be corrupted, in an unsupported format, or too small.\n\n"
            f"Technical details: {error_msg}"
        )
        
        self.logger.error(f"Face recognition error: {file_path} - {error_type}: {error_msg}")
        
        if self.gui_mode:
            try:
                messagebox.showwarning(
                    "Face Recognition Error",
                    user_msg
                )
            except Exception:
                print(f"WARNING: {user_msg}")
        
        return False, user_msg
    
    def handle_image_conversion_error(self, file_path: Path, error: Exception) -> Tuple[bool, str]:
        """
        Handle image conversion specific errors.
        
        Args:
            file_path: The problematic file path
            error: The exception that occurred
            
        Returns:
            Tuple of (success, error_message)
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        if "UnicodeError" in error_type or "UnicodeDecodeError" in error_type:
            return self.handle_unicode_path_error(file_path, error, "image conversion")
        
        # Handle other conversion errors
        user_msg = (
            f"Image conversion error: {file_path.name}\n\n"
            f"Could not convert this image.\n"
            f"The image may be corrupted, in an unsupported format, or protected.\n\n"
            f"Technical details: {error_msg}"
        )
        
        self.logger.error(f"Image conversion error: {file_path} - {error_type}: {error_msg}")
        
        if self.gui_mode:
            try:
                messagebox.showwarning(
                    "Image Conversion Error",
                    user_msg
                )
            except Exception:
                print(f"WARNING: {user_msg}")
        
        return False, user_msg
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all errors encountered.
        
        Returns:
            Dictionary containing error statistics and recommendations
        """
        recommendations = []
        
        if self.error_stats['unicode_errors'] > 0:
            recommendations.append(
                "Consider renaming files with special characters to use only ASCII characters."
            )
        
        if self.error_stats['encoding_errors'] > 0:
            recommendations.append(
                "Check file system encoding settings and ensure proper Unicode support."
            )
        
        if self.error_stats['file_access_errors'] > 0:
            recommendations.append(
                "Verify file permissions and ensure files haven't been moved or deleted."
            )
        
        return {
            'statistics': self.error_stats.copy(),
            'recommendations': recommendations,
            'has_unicode_issues': self.error_stats['unicode_errors'] > 0
        }
    
    def show_error_summary(self) -> None:
        """
        Display a summary of all errors encountered during processing.
        """
        summary = self.get_error_summary()
        
        if summary['statistics']['total_errors'] == 0:
            return
        
        summary_text = "Processing completed with the following issues:\n\n"
        
        # Add statistics
        stats = summary['statistics']
        summary_text += f"Total errors: {stats['total_errors']}\n"
        if stats['unicode_errors'] > 0:
            summary_text += f"Unicode-related errors: {stats['unicode_errors']}\n"
        if stats['encoding_errors'] > 0:
            summary_text += f"Encoding errors: {stats['encoding_errors']}\n"
        if stats['file_access_errors'] > 0:
            summary_text += f"File access errors: {stats['file_access_errors']}\n"
        
        # Add recommendations
        if summary['recommendations']:
            summary_text += "\nRecommendations:\n"
            for i, rec in enumerate(summary['recommendations'], 1):
                summary_text += f"{i}. {rec}\n"
        
        # Show summary
        if self.gui_mode:
            try:
                messagebox.showinfo("Processing Summary", summary_text)
            except Exception:
                print(f"SUMMARY: {summary_text}")
        else:
            print(f"SUMMARY: {summary_text}")
    
    def reset_stats(self) -> None:
        """
        Reset error statistics.
        """
        for key in self.error_stats:
            self.error_stats[key] = 0


def create_error_handler(logger: Optional[logging.Logger] = None, 
                        gui_mode: bool = True) -> UnicodeErrorHandler:
    """
    Factory function to create a Unicode error handler.
    
    Args:
        logger: Logger instance
        gui_mode: Whether to show GUI dialogs
        
    Returns:
        Configured UnicodeErrorHandler instance
    """
    return UnicodeErrorHandler(logger=logger, gui_mode=gui_mode)