#!/usr/bin/env python3
"""
Enhanced Video Downloader - No Login Required

A comprehensive Python script that extracts and downloads videos from web pages
without requiring user authentication. Replicates LJ Video Downloader functionality
while bypassing login dependencies through advanced web scraping techniques.

Author: AI Assistant
Version: 2.0
"""

import argparse
import os
import sys
import re
import json
import signal
import logging
from urllib.parse import urljoin, urlparse, parse_qs
from pathlib import Path
import time
from typing import List, Dict, Optional, Tuple
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from http.cookiejar import MozillaCookieJar, LWPCookieJar

# Mobile/Android compatibility detection
try:
    import platform
    ANDROID_MODE = 'ANDROID_ROOT' in os.environ or 'TERMUX_VERSION' in os.environ
except ImportError:
    ANDROID_MODE = False

# Cross-platform keyboard input detection
try:
    import msvcrt  # Windows
    WINDOWS_PLATFORM = True
except ImportError:
    WINDOWS_PLATFORM = False
    try:
        import termios
        import tty
        import select
    except ImportError:
        pass

try:
    import requests
    from bs4 import BeautifulSoup
    import yt_dlp
    from tqdm import tqdm
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install required packages: pip install requests beautifulsoup4 yt-dlp tqdm")
    sys.exit(1)


class EnhancedVideoDownloader:
    """Enhanced video downloader that bypasses login requirements."""
    
    def __init__(self, output_dir: str = "downloads", quality: str = "best", gui_mode: bool = False):
        """Initialize the enhanced video downloader."""
        self.output_dir = Path(output_dir)
        self.quality = quality
        self.gui_mode = gui_mode
        self.terminated = False
        self.successful_downloads = 0
        self.failed_downloads = 0
        self.progress_bar = None
        self.current_status = ""
        self.android_mode = ANDROID_MODE
        
        # Enhanced session configuration for bypassing restrictions
        self.session = requests.Session()
        self._setup_enhanced_session()
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Setup keyboard monitoring for termination (skip on Android)
        if not self.android_mode:
            self.setup_keyboard_monitor()
        
        # Create output directory
        self.create_output_directory()
        
        # GUI components
        self.root = None
        self.url_var = None
        self.progress_var = None
        self.status_var = None
        self.video_listbox = None
        self.download_button = None
        
    def _setup_enhanced_session(self):
        """Setup enhanced session with advanced bypass techniques."""
        # Mobile-friendly headers for Android compatibility
        if self.android_mode:
            user_agent = 'Mozilla/5.0 (Linux; Android 12; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36'
        else:
            user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            
        # Comprehensive headers to mimic real browser behavior
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        })
        
        # Setup retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Disable SSL verification for problematic sites (use with caution)
        self.session.verify = False
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
    def setup_keyboard_monitor(self):
        """Setup keyboard monitoring for graceful termination."""
        def keyboard_monitor():
            """Monitor keyboard input for termination signal."""
            try:
                if WINDOWS_PLATFORM:
                    # Windows implementation
                    while not self.terminated:
                        if msvcrt.kbhit():
                            key = msvcrt.getch()
                            if key == b'\x1b':  # Escape key
                                self.terminated = True
                                print("\n⏹️  Termination requested by user (Escape key pressed)")
                                break
                        time.sleep(0.1)
                else:
                    # Unix-like systems implementation
                    old_settings = None
                    try:
                        old_settings = termios.tcgetattr(sys.stdin)
                        tty.setraw(sys.stdin.fileno())
                        
                        while not self.terminated:
                            if select.select([sys.stdin], [], [], 0.1)[0]:
                                key = sys.stdin.read(1)
                                if key == '\x1b':  # Escape key
                                    self.terminated = True
                                    print("\n⏹️  Termination requested by user (Escape key pressed)")
                                    break
                    finally:
                        if old_settings:
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except Exception as e:
                self.logger.debug(f"Keyboard monitor error: {e}")
        
        # Start keyboard monitor in separate thread
        monitor_thread = threading.Thread(target=keyboard_monitor, daemon=True)
        monitor_thread.start()
        
        # Setup signal handlers as backup
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, 'terminated', True))
        signal.signal(signal.SIGTERM, lambda s, f: setattr(self, 'terminated', True))
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('enhanced_video_downloader.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        # Set verbose flag
        self.verbose = False
        
    def create_output_directory(self):
        """Create output directory if it doesn't exist."""
        try:
            # Android-specific directory handling
            if self.android_mode:
                # Use Android-friendly paths
                if not self.output_dir.is_absolute():
                    # Use internal storage or external storage if available
                    android_storage = Path('/storage/emulated/0/Download') if Path('/storage/emulated/0/Download').exists() else Path.home()
                    self.output_dir = android_storage / self.output_dir.name
                    
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory: {self.output_dir}")
            
            # Check write permissions (important on Android)
            test_file = self.output_dir / '.write_test'
            try:
                test_file.write_text('test')
                test_file.unlink()
            except Exception as e:
                self.logger.warning(f"Write permission issue: {e}")
                if self.android_mode:
                    # Fallback to home directory on Android
                    self.output_dir = Path.home() / 'downloads'
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
            raise
            
    def fetch_page_enhanced(self, url: str) -> Optional[str]:
        """Enhanced page fetching that bypasses authentication requirements."""
        try:
            self.logger.info(f"Fetching page: {url}")
            
            # Multiple bypass strategies
            strategies = [
                self._fetch_direct,
                self._fetch_with_referrer,
                self._fetch_with_mobile_agent,
                self._fetch_with_proxy_headers
            ]
            
            for strategy in strategies:
                try:
                    content = strategy(url)
                    if content and len(content) > 1000:  # Reasonable content length
                        return content
                except Exception as e:
                    self.logger.debug(f"Strategy failed: {e}")
                    continue
                    
            self.logger.warning(f"All fetch strategies failed for {url}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to fetch page {url}: {e}")
            return None
            
    def _fetch_direct(self, url: str) -> Optional[str]:
        """Direct fetch strategy using simple requests."""
        try:
            # Use simple requests.get instead of session for better compatibility
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            self.logger.debug(f"Direct fetch failed: {e}")
            return None
        
    def _fetch_with_referrer(self, url: str) -> Optional[str]:
        """Fetch with referrer header to bypass some restrictions."""
        headers = {'Referer': f"{urlparse(url).scheme}://{urlparse(url).netloc}"}
        response = self.session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        # Ensure proper encoding
        if response.encoding is None:
            response.encoding = 'utf-8'
        return response.text
        
    def _fetch_with_mobile_agent(self, url: str) -> Optional[str]:
        """Fetch with mobile user agent."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1'
        }
        response = self.session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        # Ensure proper encoding
        if response.encoding is None:
            response.encoding = 'utf-8'
        return response.text
        
    def _fetch_with_proxy_headers(self, url: str) -> Optional[str]:
        """Fetch with proxy-like headers."""
        headers = {
            'X-Forwarded-For': '8.8.8.8',
            'X-Real-IP': '8.8.8.8',
            'Via': '1.1 proxy'
        }
        response = self.session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        # Ensure proper encoding
        if response.encoding is None:
            response.encoding = 'utf-8'
        return response.text
        
    def extract_videos_enhanced(self, url: str) -> List[Dict[str, str]]:
        """Enhanced video extraction with multiple detection methods."""
        videos = []
        
        # Fetch page content
        content = self.fetch_page_enhanced(url)
        if not content:
            return videos
            
        # Multiple extraction strategies - prioritize direct methods for sites with many videos
        extraction_methods = [
            self._extract_direct_video_links,
            self._extract_embedded_videos,
            self._extract_social_media_videos,
            self._extract_streaming_videos
        ]
        
        # Try direct extraction first
        for method in extraction_methods:
            try:
                method_videos = method(content, url)
                videos.extend(method_videos)
            except Exception as e:
                self.logger.debug(f"Extraction method {method.__name__} failed: {e}")
        
        # Only use yt-dlp if no videos found with direct methods
        if not videos:
            try:
                ytdlp_videos = self._extract_with_ytdlp(content, url)
                videos.extend(ytdlp_videos)
            except Exception as e:
                self.logger.debug(f"yt-dlp extraction failed: {e}")
                
        # Remove duplicates while preserving order
        seen = set()
        unique_videos = []
        for video in videos:
            video_key = video.get('url', '')
            if video_key not in seen:
                seen.add(video_key)
                unique_videos.append(video)
                
        return unique_videos
        
    def _extract_direct_video_links(self, html: str, base_url: str) -> List[Dict[str, str]]:
        """Extract direct video links from HTML."""
        videos = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find video tags
        video_count = 0
        for video_tag in soup.find_all('video'):
            # Skip promotional/UI videos (like add_to_home.mp4)
            if video_tag.get('src') and 'add_to_home' not in video_tag.get('src', ''):
                video_count += 1
                videos.append({
                    'url': urljoin(base_url, video_tag['src']),
                    'title': f'Video {video_count}',
                    'format': 'mp4',
                    'quality': 'unknown'
                })
                
            # Check source tags within video - this is where the main content is
            for source in video_tag.find_all('source'):
                src_url = source.get('src')
                if src_url and 'add_to_home' not in src_url:
                    video_count += 1
                    # Extract title from surrounding context
                    title = f'Video {video_count}'
                    
                    # Try to find a more descriptive title from nearby text
                    parent = video_tag.parent
                    if parent:
                        # Look for text content near the video
                        text_content = parent.get_text(strip=True)
                        if text_content and len(text_content) < 100:
                            # Use first meaningful text as title
                            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
                            if lines:
                                title = lines[0][:50]  # Limit title length
                    
                    videos.append({
                        'url': urljoin(base_url, src_url),
                        'title': title,
                        'format': source.get('type', 'mp4').split('/')[-1] if source.get('type') else 'mp4',
                        'quality': source.get('data-quality', 'unknown')
                    })
                    
        return videos
        
    def _extract_embedded_videos(self, html: str, base_url: str) -> List[Dict[str, str]]:
        """Extract embedded video links."""
        videos = []
        
        # Common video URL patterns
        patterns = [
            r'https?://[^\s"\'>]+\.(?:mp4|avi|mov|wmv|flv|webm|mkv|m4v)',
            r'"(https?://[^"]+\.(?:mp4|avi|mov|wmv|flv|webm|mkv|m4v))"',
            r"'(https?://[^']+\.(?:mp4|avi|mov|wmv|flv|webm|mkv|m4v))'",
            # Specific pattern for desifakes.net encoded URLs
            r'https?://video\.desifakes\.net/vh/dl\?url=encoded\$[a-f0-9]+\.mp4',
            r'src="(https?://video\.desifakes\.net/vh/dl\?url=encoded\$[a-f0-9]+\.mp4)"',
        ]
        
        video_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            for match in matches:
                url = match if isinstance(match, str) else match[0]
                # Skip promotional videos
                if 'add_to_home' not in url:
                    video_count += 1
                    videos.append({
                        'url': urljoin(base_url, url),
                        'title': f'Embedded Video {video_count}',
                        'format': url.split('.')[-1].split('?')[0].lower(),  # Handle URLs with parameters
                        'quality': 'unknown'
                    })
                
        return videos
        
    def _extract_social_media_videos(self, html: str, base_url: str) -> List[Dict[str, str]]:
        """Extract social media video links."""
        videos = []
        
        # Social media specific patterns
        social_patterns = {
            'facebook': r'"(?:hd_src|sd_src)":\s*"([^"]+)"',
            'instagram': r'"video_url":\s*"([^"]+)"',
            'twitter': r'"video_url":\s*"([^"]+)"',
            'tiktok': r'"playAddr":\s*"([^"]+)"'
        }
        
        for platform, pattern in social_patterns.items():
            matches = re.findall(pattern, html)
            for match in matches:
                # Decode escaped characters
                url = match.replace('\\/', '/').replace('\\u0026', '&')
                videos.append({
                    'url': url,
                    'title': f'{platform.title()} Video',
                    'format': 'mp4',
                    'quality': 'unknown'
                })
                
        return videos
        
    def _extract_streaming_videos(self, html: str, base_url: str) -> List[Dict[str, str]]:
        """Extract streaming video links (HLS, DASH, etc.)."""
        videos = []
        
        # Streaming patterns
        streaming_patterns = [
            r'https?://[^\s"\'>]+\.m3u8[^\s"\'>]*',
            r'https?://[^\s"\'>]+\.mpd[^\s"\'>]*',
            r'"(https?://[^"]+\.m3u8[^"]*)"',
            r'"(https?://[^"]+\.mpd[^"]*)"'
        ]
        
        for pattern in streaming_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            for match in matches:
                url = match if isinstance(match, str) else match[0]
                format_type = 'hls' if '.m3u8' in url else 'dash'
                videos.append({
                    'url': urljoin(base_url, url),
                    'title': f'Streaming Video ({format_type.upper()})',
                    'format': format_type,
                    'quality': 'adaptive'
                })
                
        return videos
        
    def _extract_with_ytdlp(self, html: str, base_url: str) -> List[Dict[str, str]]:
        """Extract videos using yt-dlp as fallback."""
        videos = []
        
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False  # Get full info to determine format
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(base_url, download=False)
                
                if info:
                    if 'entries' in info:
                        # Playlist
                        for entry in info['entries'][:10]:  # Limit to first 10
                            if entry:
                                # Determine format from available formats or extension
                                format_ext = 'mp4'  # Default
                                if 'formats' in entry and entry['formats']:
                                    # Get the best format's extension
                                    best_format = entry['formats'][-1]
                                    format_ext = best_format.get('ext', 'mp4')
                                elif 'ext' in entry:
                                    format_ext = entry['ext']
                                
                                videos.append({
                                    'url': entry.get('webpage_url', entry.get('url', base_url)),
                                    'title': entry.get('title', 'Video'),
                                    'format': format_ext,
                                    'quality': 'best'
                                })
                    else:
                        # Single video
                        format_ext = 'mp4'  # Default
                        if 'formats' in info and info['formats']:
                            # Get the best format's extension
                            best_format = info['formats'][-1]
                            format_ext = best_format.get('ext', 'mp4')
                        elif 'ext' in info:
                            format_ext = info['ext']
                        
                        videos.append({
                            'url': info.get('webpage_url', base_url),
                            'title': info.get('title', 'Video'),
                            'format': format_ext,
                            'quality': 'best'
                        })
                        
        except Exception as e:
            self.logger.debug(f"yt-dlp extraction failed: {e}")
            
        return videos
        
    def download_video_enhanced(self, video_info: Dict[str, str], output_path: Path) -> bool:
        """Enhanced video download with multiple strategies and mobile optimization."""
        try:
            url = video_info['url']
            title = video_info.get('title', 'video')
            format_ext = video_info.get('format', 'mp4')
            
            self.logger.info(f"Downloading: {title}")
            
            # Handle 'various' format by letting yt-dlp determine the extension
            if format_ext == 'various':
                # Use original output path without extension for yt-dlp to determine
                output_path = output_path.with_suffix('')
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Android-specific optimizations
            if self.android_mode:
                # Check available storage space
                try:
                    import shutil
                    free_space = shutil.disk_usage(output_path.parent).free
                    if free_space < 100 * 1024 * 1024:  # Less than 100MB
                        self.logger.warning(f"Low storage space: {free_space / 1024 / 1024:.1f} MB")
                except Exception:
                    pass
            
            # Try different download strategies
            # For desifakes URLs, always prioritize direct download
            if 'desifakes.net' in url:
                strategies = [
                    self._download_direct,
                    self._download_streaming
                ]
            # For 'various' format or known video platforms, prioritize yt-dlp
            elif format_ext == 'various' or any(domain in url for domain in ['youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com', 'twitch.tv']):
                strategies = [
                    self._download_with_ytdlp,
                    self._download_direct,
                    self._download_streaming
                ]
            else:
                strategies = [
                    self._download_direct,
                    self._download_with_ytdlp,
                    self._download_streaming
                ]
            
            for strategy in strategies:
                try:
                    if strategy(url, output_path):
                        self.successful_downloads += 1
                        return True
                except Exception as e:
                    self.logger.debug(f"Download strategy failed: {e}")
                    continue
                    
            self.failed_downloads += 1
            return False
            
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            self.failed_downloads += 1
            return False
            
    def _download_direct(self, url: str, output_path: Path) -> bool:
        """Direct download strategy with mobile optimization."""
        # Android-specific optimizations
        timeout = 60 if self.android_mode else 30  # Longer timeout for mobile networks
        chunk_size = 4096 if self.android_mode else 8192  # Smaller chunks for mobile
        
        response = self.session.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            if total_size > 0:
                desc = output_path.name[:30] + '...' if len(output_path.name) > 30 else output_path.name
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc,
                         disable=self.android_mode and not sys.stdout.isatty()) as pbar:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if self.terminated:
                            return False
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))
                            
                            # Mobile-friendly progress updates
                            if self.android_mode and downloaded % (chunk_size * 100) == 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\rProgress: {progress:.1f}%", end='', flush=True)
            else:
                # No content-length header
                downloaded = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if self.terminated:
                        return False
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Show progress for unknown size downloads on mobile
                        if self.android_mode and downloaded % (chunk_size * 50) == 0:
                            print(f"\rDownloaded: {downloaded / 1024 / 1024:.1f} MB", end='', flush=True)
                            
        if self.android_mode:
            print()  # New line after progress
            
        return True
        
    def _download_with_ytdlp(self, url: str, output_path: Path) -> bool:
        """Download using yt-dlp."""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Remove extension from output path for yt-dlp template
            output_template = str(output_path.with_suffix(''))
            
            ydl_opts = {
                'outtmpl': f'{output_template}.%(ext)s',
                'format': self.quality if self.quality != 'best' else 'best[ext=mp4]/best',
                'quiet': not self.verbose,
                'no_warnings': not self.verbose,
                'extractaudio': False,
                'writeinfojson': False,
                'writedescription': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Check if any file was downloaded in the output directory
            output_dir = output_path.parent
            downloaded_files = list(output_dir.glob(f"{output_path.stem}.*"))
            
            if downloaded_files:
                # If a file was downloaded but with different extension, rename it
                downloaded_file = downloaded_files[0]
                if downloaded_file != output_path:
                    try:
                        downloaded_file.rename(output_path)
                    except Exception:
                        # If rename fails, just check if the downloaded file exists
                        pass
                return True
            
            return output_path.exists()
            
        except Exception as e:
            self.logger.error(f"yt-dlp download failed: {e}")
            return False
        
    def _download_streaming(self, url: str, output_path: Path) -> bool:
        """Download streaming content (HLS/DASH)."""
        if '.m3u8' in url or '.mpd' in url:
            # Use yt-dlp for streaming content
            return self._download_with_ytdlp(url, output_path)
        return False
        
    def create_gui(self):
        """Create the GUI interface."""
        if self.android_mode:
            # Check if GUI is available on Android
            try:
                self.root = tk.Tk()
                self.root.title("Enhanced Video Downloader - Mobile")
                # Smaller window for mobile screens
                self.root.geometry("400x600")
                # Make it touch-friendly
                self.root.option_add('*Font', 'TkDefaultFont 12')
            except Exception as e:
                self.logger.warning(f"GUI not available on Android: {e}")
                return False
        else:
            self.root = tk.Tk()
            self.root.title("Enhanced Video Downloader - No Login Required")
            self.root.geometry("900x700")
        
        # Variables
        self.url_var = tk.StringVar()
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.batch_mode_var = tk.BooleanVar()
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # URL input section
        url_section = ttk.LabelFrame(main_frame, text="URL Input", padding="5")
        url_section.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Single URL mode
        ttk.Radiobutton(url_section, text="Single URL", variable=self.batch_mode_var, value=False, command=self.toggle_input_mode).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(url_section, text="Batch URLs", variable=self.batch_mode_var, value=True, command=self.toggle_input_mode).grid(row=0, column=1, sticky=tk.W)
        
        # Single URL input
        self.single_url_frame = ttk.Frame(url_section)
        self.single_url_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Entry(self.single_url_frame, textvariable=self.url_var, width=70).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(self.single_url_frame, text="Analyze", command=self.analyze_url).grid(row=0, column=1, padx=(5, 0))
        
        # Batch URL input
        self.batch_url_frame = ttk.Frame(url_section)
        self.batch_url_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        
        ttk.Label(self.batch_url_frame, text="Enter URLs (one per line):").grid(row=0, column=0, sticky=tk.W)
        
        batch_text_frame = ttk.Frame(self.batch_url_frame)
        batch_text_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        
        self.batch_text = tk.Text(batch_text_frame, height=4, width=70)
        self.batch_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        batch_scrollbar = ttk.Scrollbar(batch_text_frame, orient=tk.VERTICAL, command=self.batch_text.yview)
        batch_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.batch_text.configure(yscrollcommand=batch_scrollbar.set)
        
        batch_button_frame = ttk.Frame(self.batch_url_frame)
        batch_button_frame.grid(row=2, column=0, columnspan=2, pady=(5, 0))
        
        ttk.Button(batch_button_frame, text="Load from File", command=self.load_urls_from_file).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(batch_button_frame, text="Analyze All", command=self.analyze_batch_urls).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(batch_button_frame, text="Clear", command=lambda: self.batch_text.delete(1.0, tk.END)).grid(row=0, column=2)
        
        # Initially hide batch mode
        self.batch_url_frame.grid_remove()
        
        # Video list
        ttk.Label(main_frame, text="Available Videos:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        
        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.video_listbox = tk.Listbox(list_frame, height=10)
        self.video_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.video_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.video_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Download controls
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.download_button = ttk.Button(control_frame, text="Download Selected", command=self.download_selected)
        self.download_button.grid(row=0, column=0, padx=(0, 5))
        
        ttk.Button(control_frame, text="Download All", command=self.download_all).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(control_frame, text="Choose Output Folder", command=self.choose_output_folder).grid(row=0, column=2)
        
        # Progress bar
        ttk.Label(main_frame, text="Progress:").grid(row=5, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100).grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Status
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=7, column=0, columnspan=2, sticky=tk.W)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)  # Video list row
        url_section.columnconfigure(0, weight=1)
        self.single_url_frame.columnconfigure(0, weight=1)
        batch_text_frame.columnconfigure(0, weight=1)
        batch_text_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Store video data
        self.video_data = []
        self.batch_urls = []
        
    def analyze_url(self):
        """Analyze URL and populate video list."""
        url = self.url_var.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a URL")
            return
            
        self.status_var.set("Analyzing URL...")
        self.video_listbox.delete(0, tk.END)
        self.video_data.clear()
        
        # Run analysis in separate thread
        def analyze():
            try:
                videos = self.extract_videos_enhanced(url)
                
                # Update GUI in main thread
                self.root.after(0, lambda: self._update_video_list(videos))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to analyze URL: {e}"))
                self.root.after(0, lambda: self.status_var.set("Ready"))
                
        threading.Thread(target=analyze, daemon=True).start()
        
    def toggle_input_mode(self):
        """Toggle between single URL and batch URL input modes."""
        if self.batch_mode_var.get():
            # Show batch mode
            self.single_url_frame.grid_remove()
            self.batch_url_frame.grid()
        else:
            # Show single mode
            self.batch_url_frame.grid_remove()
            self.single_url_frame.grid()
            
    def load_urls_from_file(self):
        """Load URLs from a text file."""
        file_path = filedialog.askopenfilename(
            title="Select URL file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.batch_text.delete(1.0, tk.END)
                    self.batch_text.insert(1.0, content)
                    
                self.status_var.set(f"Loaded URLs from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
                
    def analyze_batch_urls(self):
        """Analyze multiple URLs from batch input."""
        urls_text = self.batch_text.get(1.0, tk.END).strip()
        if not urls_text:
            messagebox.showerror("Error", "Please enter URLs or load from file")
            return
            
        # Parse URLs
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        if not urls:
            messagebox.showerror("Error", "No valid URLs found")
            return
            
        self.batch_urls = urls
        self.status_var.set(f"Analyzing {len(urls)} URLs...")
        self.video_listbox.delete(0, tk.END)
        self.video_data.clear()
        
        # Run batch analysis in separate thread
        def analyze_batch():
            all_videos = []
            
            for i, url in enumerate(urls):
                if self.terminated:
                    break
                    
                try:
                    self.root.after(0, lambda u=url, idx=i: self.status_var.set(f"Analyzing ({idx+1}/{len(urls)}): {u[:50]}..."))
                    
                    videos = self.extract_videos_enhanced(url)
                    
                    # Add source URL info to each video
                    for video in videos:
                        video['source_url'] = url
                        video['source_index'] = i + 1
                        
                    all_videos.extend(videos)
                    
                except Exception as e:
                    self.logger.error(f"Failed to analyze {url}: {e}")
                    
            # Update GUI in main thread
            self.root.after(0, lambda: self._update_video_list(all_videos))
            self.root.after(0, lambda: self.status_var.set(f"Found {len(all_videos)} videos from {len(urls)} URLs"))
            
        threading.Thread(target=analyze_batch, daemon=True).start()
        
    def clear_batch_urls(self):
        """Clear the batch URL text area."""
        self.batch_text.delete(1.0, tk.END)
        self.status_var.set("Batch URLs cleared")
        
    def _update_video_list(self, videos):
        """Update the video list in GUI."""
        self.video_data = videos
        
        for i, video in enumerate(videos):
            title = video.get('title', 'Unknown')
            format_info = video.get('format', 'unknown')
            quality = video.get('quality', 'unknown')
            
            # Add source URL info for batch processing
            if 'source_index' in video:
                source_info = f"[{video['source_index']}] "
                display_text = f"{source_info}{i+1}. {title} [{format_info}] ({quality})"
            else:
                display_text = f"{i+1}. {title} [{format_info}] ({quality})"
                
            self.video_listbox.insert(tk.END, display_text)
            
        self.status_var.set(f"Found {len(videos)} videos")
        
    def download_selected(self):
        """Download selected videos."""
        selection = self.video_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Please select videos to download")
            return
            
        selected_videos = [self.video_data[i] for i in selection]
        self._download_videos(selected_videos)
        
    def download_all(self):
        """Download all videos."""
        if not self.video_data:
            messagebox.showerror("Error", "No videos available for download")
            return
            
        self._download_videos(self.video_data)
        
    def _download_videos(self, videos):
        """Download specified videos."""
        def download():
            total = len(videos)
            
            for i, video in enumerate(videos):
                if self.terminated:
                    break
                    
                # Update progress
                progress = (i / total) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                
                # Generate filename
                title = video.get('title', 'video')
                format_ext = video.get('format', 'mp4')
                safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
                filename = f"{safe_title}.{format_ext}"
                output_path = self.output_dir / filename
                
                # Update status
                self.root.after(0, lambda t=title: self.status_var.set(f"Downloading: {t}"))
                
                # Download
                success = self.download_video_enhanced(video, output_path)
                
                if success:
                    self.root.after(0, lambda: self.status_var.set(f"Downloaded: {title}"))
                else:
                    self.root.after(0, lambda: self.status_var.set(f"Failed: {title}"))
                    
            # Final update
            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, lambda: self.status_var.set(f"Completed: {self.successful_downloads} successful, {self.failed_downloads} failed"))
            
        threading.Thread(target=download, daemon=True).start()
        
    def choose_output_folder(self):
        """Choose output folder."""
        folder = filedialog.askdirectory(initialdir=str(self.output_dir))
        if folder:
            self.output_dir = Path(folder)
            self.status_var.set(f"Output folder: {folder}")
            
    def run_gui(self):
        """Run the GUI application."""
        if self.android_mode and tk is None:
            print("❌ GUI mode not available on Android. Using CLI mode instead.")
            return False
            
        gui_created = self.create_gui()
        if gui_created is False:
            print("❌ Failed to create GUI. Falling back to CLI mode.")
            return False
            
        if self.root:
            self.root.mainloop()
            return True
        return False
        
    def run_cli(self, url: str):
        """Run in command-line mode."""
        print(f"🔍 Analyzing URL: {url}")
        
        videos = self.extract_videos_enhanced(url)
        
        if not videos:
            print("❌ No videos found")
            return False
            
        print(f"✅ Found {len(videos)} videos:")
        for i, video in enumerate(videos, 1):
            title = video.get('title', 'Unknown')
            format_info = video.get('format', 'unknown')
            quality = video.get('quality', 'unknown')
            print(f"  {i}. {title} [{format_info}] ({quality})")
            
        print("\n📥 Starting downloads...")
        print("Press Escape key to terminate gracefully\n")
        
        for i, video in enumerate(videos, 1):
            if self.terminated:
                break
                
            title = video.get('title', 'video')
            format_ext = video.get('format', 'mp4')
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
            filename = f"{safe_title}.{format_ext}"
            output_path = self.output_dir / filename
            
            print(f"📥 Downloading ({i}/{len(videos)}): {title}")
            
            success = self.download_video_enhanced(video, output_path)
            
            if success:
                print(f"✅ Downloaded: {filename}")
            else:
                print(f"❌ Failed: {title}")
                
        print(f"\n🎉 Download completed!")
        print(f"✅ Successful: {self.successful_downloads}")
        print(f"❌ Failed: {self.failed_downloads}")
        
        return self.successful_downloads > 0


def main():
    """Main function to handle command line arguments and run the downloader."""
    parser = argparse.ArgumentParser(
        description="Enhanced Video Downloader - Extract and download videos without login requirements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_video_downloader.py                    # GUI mode
  python enhanced_video_downloader.py https://example.com/page
  python enhanced_video_downloader.py https://youtube.com/watch?v=xyz -o videos/
  python enhanced_video_downloader.py https://facebook.com/video -q worst
  python enhanced_video_downloader.py --batch urls.txt  # Batch processing
        """
    )
    
    parser.add_argument(
        'url',
        nargs='?',
        help='URL of the web page to extract videos from (optional for GUI mode)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='downloads',
        help='Output directory for downloaded videos (default: downloads)'
    )
    
    parser.add_argument(
        '-q', '--quality',
        default='best',
        choices=['best', 'worst', 'bestvideo', 'worstvideo'],
        help='Video quality preference (default: best)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Force GUI mode even with URL argument'
    )
    
    parser.add_argument(
        '--batch',
        help='Path to file containing multiple URLs (one per line)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create downloader instance
    downloader = EnhancedVideoDownloader(
        output_dir=args.output, 
        quality=args.quality,
        gui_mode=args.gui or (not args.url and not args.batch)
    )
    
    try:
        if args.gui or (not args.url and not args.batch):
            # GUI mode
            if ANDROID_MODE:
                print("📱 Android detected - attempting GUI mode...")
            else:
                print("🚀 Starting Enhanced Video Downloader GUI...")
                
            gui_success = downloader.run_gui()
            if not gui_success and ANDROID_MODE:
                print("📱 GUI not available on Android. Please provide a URL for CLI mode.")
                print("Usage: python script.py <URL>")
                sys.exit(1)
        elif args.batch:
            # Batch CLI mode
            try:
                with open(args.batch, 'r', encoding='utf-8') as f:
                    urls = [url.strip() for url in f.readlines() if url.strip()]
                    
                if not urls:
                    print("❌ No valid URLs found in batch file")
                    sys.exit(1)
                    
                print(f"🔍 Processing {len(urls)} URLs from batch file...")
                
                all_videos = []
                for i, url in enumerate(urls, 1):
                    if downloader.terminated:
                        break
                        
                    print(f"\n📋 Analyzing URL ({i}/{len(urls)}): {url}")
                    
                    try:
                        videos = downloader.extract_videos_enhanced(url)
                        
                        if videos:
                            # Add source info
                            for video in videos:
                                video['source_url'] = url
                                video['source_index'] = i
                            all_videos.extend(videos)
                            print(f"✅ Found {len(videos)} video(s)")
                        else:
                            print("❌ No videos found")
                            
                    except Exception as e:
                        print(f"❌ Error analyzing {url}: {e}")
                        
                if not all_videos:
                    print("\n❌ No videos found from any URLs")
                    sys.exit(1)
                    
                print(f"\n📥 Starting downloads for {len(all_videos)} videos...")
                print("Press Escape key to terminate gracefully\n")
                
                for i, video in enumerate(all_videos, 1):
                    if downloader.terminated:
                        break
                        
                    title = video.get('title', 'video')
                    format_ext = video.get('format', 'mp4')
                    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
                    filename = f"{safe_title}.{format_ext}"
                    output_path = downloader.output_dir / filename
                    
                    source_info = f" [Source {video.get('source_index', '?')}]"
                    print(f"📥 Downloading ({i}/{len(all_videos)}): {title}{source_info}")
                    
                    success = downloader.download_video_enhanced(video, output_path)
                    
                    if success:
                        print(f"✅ Downloaded: {filename}")
                    else:
                        print(f"❌ Failed: {title}")
                        
                print(f"\n🎉 Batch download completed!")
                print(f"✅ Successful: {downloader.successful_downloads}")
                print(f"❌ Failed: {downloader.failed_downloads}")
                
            except FileNotFoundError:
                print(f"❌ Batch file not found: {args.batch}")
                sys.exit(1)
            except Exception as e:
                print(f"❌ Error processing batch file: {e}")
                sys.exit(1)
        else:
            # Single URL CLI mode
            success = downloader.run_cli(args.url)
            
            if downloader.terminated:
                print("\n⏹️  Video download process was terminated by user")
                sys.exit(0)
            elif not success:
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user")
        if hasattr(downloader, 'successful_downloads') and downloader.successful_downloads > 0:
            print(f"✅ {downloader.successful_downloads} video(s) were successfully downloaded before interruption")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()