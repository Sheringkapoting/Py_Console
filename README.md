# Enhanced Video Downloader - No Login Required

A powerful, cross-platform video downloader that replicates all core features of LJ Video Downloader while eliminating login dependencies. Works seamlessly on Windows, Linux, macOS, and Android devices.

## Features

- 🎥 **Multi-format Support**: Downloads MP4, WebM, AVI, MOV, FLV, MKV, and more
- 🌐 **Social Media Integration**: Works with YouTube, Facebook, Instagram, Twitter, TikTok, Vimeo, and more
- 🔍 **Smart Detection**: Finds videos in HTML5 video tags, iframes, and direct links
- 📁 **Organized Downloads**: Creates output directories and handles file naming automatically
- 🎯 **Quality Preservation**: Maintains original video quality with customizable options
- 📊 **Progress Tracking**: Real-time download progress with interactive indicators
- 🛡️ **Error Handling**: Robust error handling and retry mechanisms
- ⚡ **Graceful Termination**: Press Escape key anytime to safely terminate the download process
- 🎮 **Interactive Mode**: User-friendly prompts for URL and output directory selection

## Installation

1. **Clone or download the script files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

- `requests` - HTTP library for web scraping
- `beautifulsoup4` - HTML parsing
- `yt-dlp` - Video extraction from social media platforms
- `lxml` - Enhanced HTML parsing (optional)
- `tqdm` - Progress bars (optional)

## Usage

### Basic Usage

```bash
# Download videos from a webpage
python video_downloader.py https://example.com/page-with-videos

# Specify output directory
python video_downloader.py https://example.com/videos -o my_videos/

# Choose video quality
python video_downloader.py https://youtube.com/watch?v=xyz -q best
```

### Command Line Options

```
usage: video_downloader.py [-h] [-o OUTPUT] [-q {best,worst,bestvideo,worstvideo}] [-v] 
                          [--cookies COOKIES] [--login-url LOGIN_URL] [--login] url

Extract and download videos from web pages

positional arguments:
  url                   URL of the web page to extract videos from

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output directory for downloaded videos (default: downloads)
  -q {best,worst,bestvideo,worstvideo}, --quality {best,worst,bestvideo,worstvideo}
                        Video quality preference (default: best)
  -v, --verbose         Enable verbose logging
  --cookies COOKIES     Path to cookies file (Netscape/Mozilla format)
  --login-url LOGIN_URL
                        URL of the login page (if different from main page)
  --login               Prompt for username and password for authentication
```

### Examples

#### 1. Download from YouTube
```bash
python video_downloader.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

#### 2. Download from Facebook
```bash
python video_downloader.py "https://www.facebook.com/watch/?v=123456789" -o facebook_videos/
```

#### 3. Download from a forum or blog
```bash
python video_downloader.py "https://example-forum.com/post/123" -q best -v
```

#### 4. Download with specific quality
```bash
# Best quality (default)
python video_downloader.py https://example.com -q best

# Worst quality (smaller files)
python video_downloader.py https://example.com -q worst

# Best video only (no audio)
python video_downloader.py https://example.com -q bestvideo
```

#### 5. Authentication Examples
```bash
# Login with username/password prompt
python video_downloader.py https://members-only-site.com/videos --login

# Use saved cookies from browser
python video_downloader.py https://private-site.com/videos --cookies cookies.txt

# Specify custom login URL and authenticate
python video_downloader.py https://site.com/videos --login-url https://site.com/login --login
```

## How It Works

The script uses a multi-layered approach to find and download videos:

### 1. Social Media Detection
- Uses `yt-dlp` to handle popular platforms (YouTube, Facebook, Instagram, etc.)
- Automatically detects and extracts video URLs from social media embeds

### 2. HTML Parsing
- Scans HTML5 `<video>` tags and their `<source>` elements
- Finds video links in `<a>` tags pointing to video files
- Detects embedded videos in `<iframe>` elements

### 3. Pattern Matching
- Uses regex patterns to find video URLs in JavaScript and page source
- Supports various video file extensions and streaming formats

### 4. Download Methods
- **Direct HTTP Download**: For simple video files
- **yt-dlp Integration**: For complex streaming videos and social media
- **Progress Tracking**: Shows download progress for large files

## Supported Platforms

### Social Media
- YouTube (youtube.com, youtu.be)
- Facebook (facebook.com)
- Instagram (instagram.com)
- Twitter (twitter.com)
- TikTok (tiktok.com)
- Vimeo (vimeo.com)
- Dailymotion (dailymotion.com)
- Twitch (twitch.tv)
- Reddit (reddit.com)

### Video Formats
- MP4, AVI, MOV, WMV, FLV
- WebM, MKV, M4V, 3GP, OGV
- TS, M3U8 (streaming formats)

## Output Structure

```
downloads/
├── video_1640995200.mp4
├── facebook_video.mp4
├── youtube_video.webm
└── forum_clip.avi
```

- Videos are saved with descriptive names when possible
- Duplicate downloads are automatically skipped
- Original file extensions are preserved
- Timestamps are added for unnamed videos

## Logging

The script creates detailed logs in `video_downloader.log`:

```
2024-01-15 10:30:15 - INFO - Processing URL: https://example.com
2024-01-15 10:30:16 - INFO - Found 3 potential video(s)
2024-01-15 10:30:17 - INFO - Downloading: https://example.com/video.mp4 -> video.mp4
2024-01-15 10:30:25 - INFO - Successfully downloaded: video.mp4
```

## Termination Control

The video downloader includes robust termination handling that allows you to safely stop the download process at any time:

### How to Terminate

- **Press `Escape`** at any point during the download process
- The script will detect the interruption and gracefully terminate
- Partial downloads are automatically cleaned up
- A summary of completed downloads is displayed

### What Happens During Termination

1. **Immediate Response**: The script detects the termination signal instantly
2. **Current Download**: Any ongoing download is stopped and partial files are removed
3. **Progress Summary**: Shows how many videos were successfully downloaded before termination
4. **Clean Exit**: The process exits cleanly without leaving corrupted files

### Example Termination Output

```
^C
⏹️  Download interrupted by user
✅ 3 video(s) were successfully downloaded before interruption
```

## Authentication & Login-Protected Sites

The script supports downloading videos from login-protected websites using several authentication methods:

### 1. Interactive Login (--login)

Prompts you to enter username and password:

```bash
python video_downloader.py https://members-site.com/videos --login
```

The script will:
- Automatically detect login forms on the page
- Prompt for your credentials securely
- Handle the login process and maintain the session
- Save cookies for future use (if --cookies is specified)

### 2. Cookie-Based Authentication (--cookies)

Use cookies exported from your browser:

```bash
python video_downloader.py https://private-site.com/videos --cookies cookies.txt
```

#### Exporting Cookies from Browser

**Chrome/Edge:**
1. Install "Get cookies.txt" extension
2. Visit the website and log in
3. Click the extension icon and download cookies.txt

**Firefox:**
1. Install "cookies.txt" add-on
2. Visit the website and log in
3. Click the add-on icon and export cookies

**Manual Method (All Browsers):**
1. Open Developer Tools (F12)
2. Go to Application/Storage tab
3. Copy cookies and create a Netscape format file

### 3. Custom Login URL (--login-url)

Specify a different login page:

```bash
python video_downloader.py https://site.com/videos --login-url https://site.com/auth/login --login
```

### 4. Anti-Bot Protection

The script includes several features to bypass common anti-bot measures:

- **User Agent Rotation**: Automatically rotates between realistic browser user agents
- **Browser Headers**: Sends authentic browser headers (Accept, Language, etc.)
- **Request Timing**: Adds random delays to simulate human behavior
- **Session Management**: Maintains cookies and session state
- **Referer Handling**: Sets appropriate referer headers

### 5. CAPTCHA Detection

When CAPTCHA or bot protection is detected, the script will:
- Display a clear warning message
- Provide step-by-step solutions
- Suggest using browser cookies after manual verification

### Authentication Examples

```bash
# Forum with login requirement
python video_downloader.py "https://forum.example.com/thread/123" --login

# Social media with exported cookies
python video_downloader.py "https://private-group.com/videos" --cookies browser_cookies.txt

# Site with separate login page
python video_downloader.py "https://site.com/members/videos" --login-url "https://site.com/signin" --login

# Combine authentication with other options
python video_downloader.py "https://premium-site.com/videos" --login --cookies session.txt -q best -v
```

### Security Notes

- Credentials are never stored permanently
- Cookies are saved only if explicitly specified
- Use secure networks when entering passwords
- Respect website terms of service
- Some sites may detect and block automated access

## Troubleshooting

### Common Issues

1. **"No videos found"**
   - The page might use JavaScript to load videos dynamically
   - Try using the direct video URL if available
   - Some sites may block automated access

2. **"Download failed"**
   - Check your internet connection
   - The video might be geo-restricted
   - Try a different quality setting

3. **"Missing dependencies"**
   - Run: `pip install -r requirements.txt`
   - Ensure you have Python 3.7+ installed

4. **"HTTP Error 400/403: Access Denied"**
   - The site may require login: use `--login` or `--cookies`
   - CAPTCHA protection detected: complete verification in browser first
   - Try using cookies exported from your browser
   - Some sites block automated access entirely

5. **"Login failed" or "Authentication required"**
   - Verify your username and password are correct
   - Check if the site uses two-factor authentication
   - Try using browser cookies instead of credentials
   - Ensure the login URL is correct (use `--login-url` if needed)

6. **"CAPTCHA or bot protection detected"**
   - Open the URL in your browser and complete verification
   - Export cookies after verification and use `--cookies`
   - Try again later when protection may be less strict
   - Some sites permanently block automated access

7. **"Termination not working"**
   - Ensure the Escape key is being detected properly
   - On some systems, terminal focus may be required for key detection
   - Use the test script (`test_termination.py`) to verify termination functionality
   - Ctrl+C still works as a backup termination method

### Debug Mode

Use the `-v` flag for detailed logging:

```bash
python video_downloader.py https://example.com -v
```

## Legal Considerations

⚠️ **Important**: This tool is for educational and personal use only. Please:

- Respect copyright laws and terms of service
- Only download videos you have permission to download
- Be mindful of the content creators' rights
- Use responsibly and ethically

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the script.

## License

This project is provided as-is for educational purposes. Use at your own risk and responsibility.