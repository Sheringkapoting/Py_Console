#!/usr/bin/env python3
"""
Common Utilities Module
Shared utilities for video/image processing scripts to reduce code duplication
and improve maintainability.
"""

import os
import sys
import time
import threading
from typing import Optional, Callable
from pathlib import Path


def format_size(size_bytes: int) -> str:
    """Format bytes in human readable format with optimized calculation."""
    if size_bytes == 0:
        return "0B"
    
    size_names = ("B", "KB", "MB", "GB", "TB")
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f}{size_names[i]}"


class TerminationManager:
    """
    Unified termination manager for graceful shutdown across all scripts.
    Handles ESC key detection on Windows and Unix-like systems.
    """
    
    def __init__(self):
        self._terminate_flag = threading.Event()
        self._cleanup_callbacks = []
        self._lock = threading.Lock()
        self._monitor_thread = None
        
    def register_cleanup(self, callback: Callable):
        """Register a cleanup callback to be called on termination."""
        with self._lock:
            if callable(callback):
                self._cleanup_callbacks.append(callback)
    
    def request_terminate(self, reason: str = "User request"):
        """Request termination and run cleanup callbacks."""
        if self._terminate_flag.is_set():
            return
            
        self._terminate_flag.set()
        
        try:
            print(f"\n[INFO] Termination requested: {reason}")
        except Exception:
            pass
        
        # Run cleanup callbacks in reverse order
        with self._lock:
            for callback in reversed(self._cleanup_callbacks):
                try:
                    callback()
                except Exception as e:
                    try:
                        print(f"[WARN] Cleanup error: {e}")
                    except Exception:
                        pass
    
    def is_terminating(self) -> bool:
        """Check if termination has been requested."""
        return self._terminate_flag.is_set()
    
    def start_monitoring(self):
        """Start background ESC key monitoring."""
        if self._monitor_thread is not None:
            return
            
        self._monitor_thread = threading.Thread(
            target=self._monitor_escape_key,
            daemon=True
        )
        self._monitor_thread.start()
    
    def _monitor_escape_key(self):
        """Monitor for ESC key press in background thread."""
        # Try Windows first
        try:
            import msvcrt
            while not self.is_terminating():
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\x1b':  # ESC key
                        self.request_terminate("Escape key pressed")
                        return
                time.sleep(0.1)
            return
        except ImportError:
            pass
        
        # Try Unix-like systems
        try:
            import select
            import termios
            import tty
            
            old_settings = None
            try:
                old_settings = termios.tcgetattr(sys.stdin)
                tty.setraw(sys.stdin.fileno())
                
                while not self.is_terminating():
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key == '\x1b':  # ESC key
                            self.request_terminate("Escape key pressed")
                            break
            finally:
                if old_settings:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except (ImportError, OSError):
            # Platform doesn't support keyboard monitoring
            pass


class ProgressBarHelper:
    """
    Unified progress bar helper with consistent formatting across all scripts.
    """
    
    @staticmethod
    def create_bar(total: int, desc: str = "Processing", unit: str = "item", 
                   show_rate: bool = True, mininterval: float = 0.15):
        """
        Create a standardized tqdm progress bar.
        
        Args:
            total: Total number of items
            desc: Description text
            unit: Unit name (e.g., 'file', 'img', 'video')
            show_rate: Whether to show processing rate
            mininterval: Minimum update interval in seconds
        """
        try:
            from tqdm import tqdm
            
            if show_rate:
                bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            else:
                bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
            
            return tqdm(
                total=total,
                desc=desc,
                unit=unit,
                dynamic_ncols=True,
                mininterval=mininterval,
                bar_format=bar_format,
                leave=True
            )
        except ImportError:
            # Fallback if tqdm not available
            return None
    
    @staticmethod
    def format_postfix(processed: int = 0, success: int = 0, failed: int = 0, 
                      **kwargs) -> dict:
        """
        Create standardized postfix dictionary for progress bars.
        
        Args:
            processed: Number of items processed
            success: Number of successful operations
            failed: Number of failed operations
            **kwargs: Additional custom metrics
        """
        postfix = {}
        
        if processed > 0:
            postfix['✓'] = success
        if failed > 0:
            postfix['✗'] = failed
        
        # Add custom metrics
        postfix.update(kwargs)
        
        return postfix


def validate_file_path(path: Path, must_exist: bool = True, 
                      allowed_extensions: Optional[tuple] = None) -> bool:
    """
    Validate file path with common checks.
    
    Args:
        path: Path to validate
        must_exist: Whether file must exist
        allowed_extensions: Tuple of allowed extensions (e.g., ('.mp4', '.avi'))
    
    Returns:
        True if valid, False otherwise
    """
    try:
        if must_exist and not path.exists():
            return False
        
        if path.is_dir():
            return False
        
        if allowed_extensions and path.suffix.lower() not in allowed_extensions:
            return False
        
        return True
    except Exception:
        return False


def validate_directory(path: Path, create: bool = False) -> bool:
    """
    Validate directory path and optionally create it.
    
    Args:
        path: Directory path to validate
        create: Whether to create directory if it doesn't exist
    
    Returns:
        True if valid/created, False otherwise
    """
    try:
        if path.exists():
            return path.is_dir()
        
        if create:
            path.mkdir(parents=True, exist_ok=True)
            return True
        
        return False
    except Exception:
        return False


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
    
    Returns:
        Sanitized filename
    """
    import re
    
    # Remove invalid characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    safe = safe.strip('. ')
    
    # Ensure not empty
    if not safe:
        safe = 'unnamed'
    
    # Truncate if too long
    if len(safe) > max_length:
        name, ext = os.path.splitext(safe)
        max_name_len = max_length - len(ext)
        safe = name[:max_name_len] + ext
    
    return safe


class SimpleTimer:
    """Simple timer for measuring elapsed time."""
    
    def __init__(self):
        self.start_time = time.time()
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def reset(self):
        """Reset the timer."""
        self.start_time = time.time()
    
    def format_elapsed(self) -> str:
        """Get formatted elapsed time string."""
        elapsed = self.elapsed()
        
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            return f"{hours}h {minutes}m"
