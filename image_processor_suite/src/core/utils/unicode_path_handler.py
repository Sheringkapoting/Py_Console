"""Unicode path handling utilities for robust file operations.

This module provides utilities to handle Unicode characters in file paths
across different operating systems and libraries, particularly for the
face_recognition library which can have issues with non-ASCII characters.
"""

import os
import re
import sys
from pathlib import Path
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class UnicodePathHandler:
    """Handles Unicode path encoding and normalization for cross-platform compatibility."""
    
    @staticmethod
    def normalize_path(path: Union[str, Path]) -> str:
        """Normalize a path for safe use with external libraries.
        
        Args:
            path: Input path as string or Path object
            
        Returns:
            Normalized path string safe for external library use
            
        Raises:
            ValueError: If path cannot be properly encoded
        """
        try:
            # Convert to Path object for consistent handling
            path_obj = Path(path)
            
            # Resolve to absolute path to handle relative paths and symlinks
            abs_path = path_obj.resolve()
            
            # Convert to string and handle Unicode encoding
            path_str = str(abs_path)
            
            # On Windows, try to use short path names for problematic Unicode
            if sys.platform == 'win32':
                try:
                    import ctypes
                    from ctypes import wintypes
                    
                    # Get short path name if available
                    buffer_size = ctypes.windll.kernel32.GetShortPathNameW(path_str, None, 0)
                    if buffer_size > 0:
                        buffer = ctypes.create_unicode_buffer(buffer_size)
                        if ctypes.windll.kernel32.GetShortPathNameW(path_str, buffer, buffer_size):
                            short_path = buffer.value
                            # Only use short path if it's actually shorter (contains ~)
                            if '~' in short_path:
                                logger.debug(f"Using short path: {short_path} for {path_str}")
                                return short_path
                except Exception as e:
                    logger.debug(f"Could not get short path for {path_str}: {e}")
            
            # Ensure proper UTF-8 encoding with error handling
            try:
                # Test if the path can be encoded/decoded properly
                encoded = path_str.encode('utf-8', errors='strict')
                decoded = encoded.decode('utf-8')
                return decoded
            except UnicodeError:
                # Fallback: use replacement characters for problematic Unicode
                logger.warning(f"Unicode encoding issue with path: {path_str}")
                encoded = path_str.encode('utf-8', errors='replace')
                return encoded.decode('utf-8')
                
        except Exception as e:
            raise ValueError(f"Failed to normalize path '{path}': {e}")
    
    @staticmethod
    def safe_path_for_library(path: Union[str, Path], library_name: str = "face_recognition") -> str:
        """Get a safe path string for use with external libraries.
        
        Args:
            path: Input path
            library_name: Name of the library for logging purposes
            
        Returns:
            Safe path string for library use
        """
        try:
            normalized = UnicodePathHandler.normalize_path(path)
            
            # Verify the path exists
            if not Path(normalized).exists():
                # Try with original path if normalized doesn't exist
                original_str = str(path)
                if Path(original_str).exists():
                    logger.warning(f"Using original path for {library_name}: {original_str}")
                    return original_str
                else:
                    raise FileNotFoundError(f"Path does not exist: {path}")
            
            return normalized
            
        except Exception as e:
            logger.error(f"Failed to create safe path for {library_name}: {e}")
            # Last resort: return string representation of original path
            return str(path)
    
    @staticmethod
    def validate_unicode_path(path: Union[str, Path]) -> tuple[bool, Optional[str]]:
        """Validate if a path can be safely used with Unicode-sensitive operations.
        
        Args:
            path: Path to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            path_str = str(path)
            
            # Check for null bytes
            if '\x00' in path_str:
                return False, "Path contains null bytes"
            
            # Check if path can be encoded/decoded
            try:
                path_str.encode('utf-8').decode('utf-8')
            except UnicodeError as e:
                return False, f"Unicode encoding error: {e}"
            
            # Check if path exists or parent directory exists
            path_obj = Path(path)
            if not path_obj.exists() and not path_obj.parent.exists():
                return False, "Path and parent directory do not exist"
            
            # Check for extremely long paths (Windows limitation)
            if sys.platform == 'win32' and len(path_str) > 260:
                return False, "Path exceeds Windows MAX_PATH limitation"
            
            return True, None
            
        except Exception as e:
            return False, f"Path validation error: {e}"
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize a filename by removing or replacing problematic characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename safe for filesystem operations
        """
        # Characters that are problematic on various filesystems
        invalid_chars = '<>:"/\\|?*'
        
        # Replace invalid characters with underscores
        sanitized = filename
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Remove control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
        
        # Ensure filename is not empty and not just dots
        sanitized = sanitized.strip('. ')
        if not sanitized:
            sanitized = 'unnamed_file'
        
        # Limit length to reasonable size
        if len(sanitized) > 200:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:200-len(ext)] + ext
        
        return sanitized

# Convenience functions for common operations
def safe_face_recognition_path(path: Union[str, Path]) -> str:
    """Get a path safe for face_recognition library operations."""
    return UnicodePathHandler.safe_path_for_library(path, "face_recognition")

def safe_pil_path(path: Union[str, Path]) -> str:
    """Get a path safe for PIL/Pillow operations."""
    return UnicodePathHandler.safe_path_for_library(path, "PIL")

def validate_path(path: Union[str, Path]) -> tuple[bool, Optional[str]]:
    """Validate if a path is safe for Unicode operations."""
    return UnicodePathHandler.validate_unicode_path(path)

def sanitize_name(name: str, replacement: str = "_") -> str:
    """
    Sanitize a filename by replacing problematic characters.
    
    Args:
        name: Original filename
        replacement: Character to use as replacement
        
    Returns:
        Sanitized filename
    """
    # Remove or replace problematic characters
    problematic_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(problematic_chars, replacement, name)
    
    # Remove control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
    
    # Trim whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure not empty
    if not sanitized:
        sanitized = "unnamed_file"
    
    return sanitized


def sanitize_path_component(component: str, max_length: int = 255) -> str:
    """
    Sanitize a single path component (filename or directory name).
    
    Args:
        component: Path component to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized path component
    """
    # First apply basic sanitization
    sanitized = sanitize_name(component)
    
    # Handle length limitations
    if len(sanitized) > max_length:
        # Try to preserve file extension
        if '.' in sanitized:
            name, ext = sanitized.rsplit('.', 1)
            max_name_length = max_length - len(ext) - 1
            if max_name_length > 0:
                sanitized = name[:max_name_length] + '.' + ext
            else:
                sanitized = sanitized[:max_length]
        else:
            sanitized = sanitized[:max_length]
    
    # Remove reserved names on Windows
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    base_name = sanitized.split('.')[0].upper()
    if base_name in reserved_names:
        sanitized = f"file_{sanitized}"
    
    return sanitized


def create_safe_filename(original_path: Path, destination_dir: Path) -> Path:
    """
    Create a safe filename in the destination directory, handling conflicts.
    
    Args:
        original_path: Original file path
        destination_dir: Destination directory
        
    Returns:
        Safe path in destination directory
    """
    # Sanitize the filename
    safe_name = sanitize_path_component(original_path.name)
    safe_path = destination_dir / safe_name
    
    # Handle filename conflicts
    counter = 1
    while safe_path.exists():
        if '.' in safe_name:
            name, ext = safe_name.rsplit('.', 1)
            safe_path = destination_dir / f"{name}_{counter}.{ext}"
        else:
            safe_path = destination_dir / f"{safe_name}_{counter}"
        counter += 1
    
    return safe_path


def validate_and_sanitize_path(file_path: Union[str, Path]) -> Tuple[bool, str, Optional[Path]]:
    """
    Validate a path and provide a sanitized alternative if needed.
    
    Args:
        file_path: Path to validate and potentially sanitize
        
    Returns:
        Tuple of (is_valid, message, sanitized_path)
    """
    try:
        path_obj = Path(file_path)
        
        # First try basic validation
        is_valid, error_msg = validate_path(path_obj)
        
        if is_valid:
            return True, "Path is valid", path_obj
        
        # If validation failed, try to create a sanitized version
        if path_obj.parent.exists():
            sanitized_name = sanitize_path_component(path_obj.name)
            sanitized_path = path_obj.parent / sanitized_name
            
            # Check if sanitized path would be valid
            sanitized_valid, _ = validate_path(sanitized_path)
            if sanitized_valid:
                return False, f"Original path invalid, suggested: {sanitized_path}", sanitized_path
        
        return False, f"Path cannot be sanitized: {error_msg}", None
        
    except Exception as e:
        return False, f"Path validation error: {e}", None