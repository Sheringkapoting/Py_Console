#!/usr/bin/env python3
"""
Generate a comprehensive PDF report documenting:
- convert_webp_to_jpg.py
- enhanced_video_downloader.py
- secure_sort_by_known_faces.py
- video_to_gif.py

Output file: Py_Console_Scripts_Report.pdf

Dependencies:
  pip install reportlab
"""

import os
import sys
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional

from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
    Preformatted, PageBreak, ListFlowable, ListItem
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.lib.units import inch

ROOT = Path(r"c:\Codes\Py_Console")

FILES = {
    "convert_webp_to_jpg.py": ROOT / "convert_webp_to_jpg.py",
    "enhanced_video_downloader.py": ROOT / "enhanced_video_downloader.py",
    "secure_sort_by_known_faces.py": ROOT / "secure_sort_by_known_faces.py",
    "video_to_gif.py": ROOT / "video_to_gif.py",
}

OUTPUT_PDF = ROOT / "Py_Console_Scripts_Report.pdf"

def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

def parse_structure(code: str) -> Dict[str, Any]:
    result = {"functions": [], "classes": []}
    if not code:
        return result
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                result["functions"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                cls = {"name": node.name, "methods": []}
                for n in node.body:
                    if isinstance(n, ast.FunctionDef):
                        cls["methods"].append(n.name)
                result["classes"].append(cls)
    except Exception:
        pass
    return result

def heading(text: str, level: int) -> Paragraph:
    style_name = f"Heading{level}" if level in (1, 2, 3) else "Heading2"
    p = Paragraph(text, STYLES[style_name])
    p.outlineLevel = level - 1
    return p

def code_block(text: str) -> Preformatted:
    return Preformatted(text.strip("\n"), STYLES["Code"])

def bullet_list(items: List[str]) -> ListFlowable:
    return ListFlowable([ListItem(Paragraph(i, STYLES["BodyText"]), leftIndent=12) for i in items],
                        bulletType="bullet", start=None)

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    page_num = canvas.getPageNumber()
    canvas.drawRightString(LETTER[0] - inch * 0.5, 0.5 * inch, f"Page {page_num}")
    canvas.restoreState()

def make_toc() -> TableOfContents:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(fontName="Helvetica-Bold", fontSize=12, name='TOCHeading1', leftIndent=20, firstLineIndent=-10, spaceBefore=8, leading=14),
        ParagraphStyle(fontName="Helvetica", fontSize=10, name='TOCHeading2', leftIndent=30, firstLineIndent=-10, spaceBefore=2, leading=12),
        ParagraphStyle(fontName="Helvetica-Oblique", fontSize=9, name='TOCHeading3', leftIndent=40, firstLineIndent=-10, spaceBefore=0, leading=11),
    ]
    return toc

def add_diagram(title: str, boxes: List[str], connections: List[tuple]) -> Drawing:
    dw, dh = 500, 150
    d = Drawing(dw, dh)
    d.add(String(10, dh-20, title, fontName="Helvetica-Bold", fontSize=12))
    x0, y = 20, dh - 70
    bw, bh, gap = 110, 40, 20
    positions = []
    for i, label in enumerate(boxes):
        x = x0 + i * (bw + gap)
        positions.append((x, y))
        d.add(Rect(x, y, bw, bh, strokeColor=colors.darkblue, fillColor=colors.whitesmoke))
        d.add(String(x + 6, y + 12, label, fontName="Helvetica", fontSize=9))
    for a, b in connections:
        xa, ya = positions[a]
        xb, yb = positions[b]
        d.add(Line(xa + bw, ya + bh/2, xb, yb + bh/2, strokeColor=colors.grey))
    return d

def build_section_convert_webp_to_jpg(flow: List[Any], code: str, struct: Dict[str, Any]):
    flow.append(heading("1. WebP to JPG Converter", 1))
    flow.append(Paragraph("File: c:\\Codes\\Py_Console\\convert_webp_to_jpg.py", STYLES["BodyText"]))
    flow.append(Spacer(1, 8))

    flow.append(heading("Core Functionality", 2))
    flow.append(Paragraph("Converts `.webp`, `.png`, and `.jpeg` images to `.jpg` format. Processes files recursively in a specified folder.", STYLES["BodyText"]))

    flow.append(heading("Key Features", 2))
    flow.append(bullet_list([
        "Batch processing with a progress bar (`tqdm`).",
        "Safety guards for decompression bombs via Pillow settings.",
        "Strict validation: ensure file exists, non-zero size, basic verify() pass.",
        "Deletes the source file only after successful conversion.",
    ]))
    flow.append(Paragraph("Note: The current implementation does not support quality adjustment or output directory selection via CLI. Conversion uses Pillow defaults and saves beside the source.", STYLES["Italic"]))

    flow.append(heading("Inputs and Outputs", 2))
    flow.append(bullet_list([
        "Input: Folder path provided via CLI arg or interactive prompt.",
        "Output: JPG files saved in place; original sources deleted on success.",
        "Supported input extensions: `.webp`, `.png`, `.jpeg` (skips `.gif`, `.jpg`).",
    ]))

    flow.append(heading("Example Usage", 2))
    flow.append(code_block(r"""
# Convert all convertible images under 'D:\Pictures\WebP' (Windows)
python c:\Codes\Py_Console\convert_webp_to_jpg.py "D:\Pictures\WebP"
"""))

    flow.append(heading("Sample Runtime Output", 2))
    flow.append(code_block("""
Converting: 120img | Converted=115 | Deleted=115
[SUMMARY]
 Total files considered: 120
 Converted: 115
 Deleted originals: 115
 Failed: 5
"""))

    flow.append(heading("Code Structure Overview", 2))
    flow.append(bullet_list([
        f"Functions: {', '.join(struct['functions']) or 'None'}",
        "No classes used; single-file utility.",
    ]))

    flow.append(heading("Dependencies", 2))
    flow.append(bullet_list([
        "Pillow (`PIL`): image loading and conversion.",
        "tqdm: progress bars.",
    ]))

    flow.append(heading("Recommended Enhancements", 2))
    flow.append(bullet_list([
        "Add `--quality` CLI option to control JPEG quality (e.g., 90).",
        "Add `--output` CLI option to specify output directory.",
        "Add `--keep-source` flag to retain original files.",
    ]))

    flow.append(add_diagram(
        "Conversion Pipeline",
        ["List Files", "Filter", "Verify", "Convert", "Save JPG", "Delete Src"],
        [(0,1),(1,2),(2,3),(3,4),(4,5)]
    ))
    flow.append(PageBreak())

def build_section_enhanced_video_downloader(flow: List[Any], code: str, struct: Dict[str, Any]):
    flow.append(heading("2. Enhanced Video Downloader", 1))
    flow.append(Paragraph("File: c:\\Codes\\Py_Console\\enhanced_video_downloader.py", STYLES["BodyText"]))
    flow.append(Spacer(1, 8))

    flow.append(heading("Core Functionality", 2))
    flow.append(Paragraph("Extracts and downloads videos from web pages without login. Supports GUI and CLI modes, multiple extraction strategies, and robust download flows.", STYLES["BodyText"]))

    flow.append(heading("Key Features", 2))
    flow.append(bullet_list([
        "Multiple format support: direct file links, social media embeds, streaming (HLS/DASH).",
        "Resolution/quality selection via `-q/--quality` (forwarded to `yt-dlp`).",
        "Progress tracking with `tqdm` for direct downloads and GUI status updates.",
        "Android-aware behavior (timeouts, chunk sizes, storage checks).",
        "Robust session with retries, headers, and fetch strategies (referrer/mobile/proxy).",
        "Termination via Escape key; graceful stopping.",
    ]))

    flow.append(heading("Inputs and Outputs", 2))
    flow.append(bullet_list([
        "Input: Single URL or batch URLs (GUI/CLI).",
        "Output: Video files saved to output directory (`-o/--output`, default `downloads`).",
        "Quality: `-q best|worst|bestvideo|worstvideo` passed to `yt-dlp`.",
    ]))

    flow.append(heading("Example Usage", 2))
    flow.append(code_block(r"""
# GUI mode
python c:\Codes\Py_Console\enhanced_video_downloader.py

# CLI single URL, 1080p (best MP4 if available)
python c:\Codes\Py_Console\enhanced_video_downloader.py "https://www.youtube.com/watch?v=XXXX" -q best -o "D:\Videos"
"""))

    flow.append(heading("Code Structure Overview", 2))
    flow.append(bullet_list([
        "Class: EnhancedVideoDownloader",
        f"Methods: {', '.join(next((c['methods'] for c in struct['classes'] if c['name']=='EnhancedVideoDownloader'), []))}",
        "GUI helpers: create_gui(), analyze_url(), download_selected(), download_all(), run_gui()",
        "CLI: run_cli(), main()",
    ]))

    flow.append(heading("Dependencies", 2))
    flow.append(bullet_list([
        "requests, BeautifulSoup (bs4): page fetching and parsing.",
        "yt-dlp: resilient downloads for common platforms and playlists.",
        "tqdm: progress bars for direct downloads.",
        "tkinter: GUI (desktop environments).",
        "urllib3 Retry, HTTPAdapter: robust session retries.",
    ]))

    flow.append(heading("Use Cases", 2))
    flow.append(bullet_list([
        "Download public videos from landing pages with embedded players.",
        "Batch extraction for multiple URLs with preview list and selective download.",
        "Fallback to `yt-dlp` for complex streaming or site-specific formats.",
    ]))

    flow.append(add_diagram(
        "Download Flow",
        ["URL", "Fetch", "Extract", "Select", "Download", "Output"],
        [(0,1),(1,2),(2,3),(3,4),(4,5)]
    ))
    flow.append(PageBreak())

def build_section_secure_sort_by_known_faces(flow: List[Any], code: str, struct: Dict[str, Any]):
    flow.append(heading("3. Secure Face Recognition Sorter", 1))
    flow.append(Paragraph("File: c:\\Codes\\Py_Console\\secure_sort_by_known_faces.py", STYLES["BodyText"]))
    flow.append(Spacer(1, 8))

    flow.append(heading("Core Functionality", 2))
    flow.append(Paragraph("Analyzes images and moves them into destination folders if they contain known faces. Interactive prompts when CLI args are missing.", STYLES["BodyText"]))

    flow.append(heading("Key Features", 2))
    flow.append(bullet_list([
        "Face detection and matching using `face_recognition` encodings.",
        "Known faces database via user-provided reference images (face->dest pairs).",
        "Secure processing: size/pixel caps, safe atomic moves, traversal prevention.",
        "Parallelized face analysis with ProcessPoolExecutor; single-threaded moves.",
        "Quiet output: single progress bar with concise final summary.",
    ]))

    flow.append(heading("Inputs and Outputs", 2))
    flow.append(bullet_list([
        "Input: `--src` folder; optional `--recursive`.",
        "Parameters: `--tolerance`, `--workers`, `--jitter`; multiple `--face` and `--dest` pairs.",
        "Output: Images moved to first matched destination folder only.",
    ]))

    flow.append(heading("Example Usage", 2))
    flow.append(code_block(r"""
# Interactive mode to define pairs:
python c:\Codes\Py_Console\secure_sort_by_known_faces.py

# CLI mode with pairs:
python c:\Codes\Py_Console\secure_sort_by_known_faces.py --src "D:\Photos" --recursive \
  --tolerance 0.5 --workers 8 --jitter 2 \
  --face "D:\Refs\Mom.jpg" --dest "D:\Photos\Sorted\Mom" \
  --face "D:\Refs\Dad.jpg" --dest "D:\Photos\Sorted\Dad"
"""))

    flow.append(heading("Sample Runtime Output", 2))
    flow.append(code_block("""
[INFO] Processing 250 images from D:\\Photos
Processing: 100%|##########| 250/250 [moved=180 Mom:120 | Dad:60] 
[SUMMARY]
 D:\\Photos\\Sorted\\Mom: 120 moved
 D:\\Photos\\Sorted\\Dad: 60 moved
 Total moved: 180
 Unmatched/no-face: 70
 Errors: 0
"""))

    flow.append(heading("Code Structure Overview", 2))
    flow.append(bullet_list([
        f"Functions: {', '.join(struct['functions'])}",
        "Encodings: load_ref_encoding() builds reference encodings for known faces.",
        "Worker: _analyze_image_worker() encodes and compares faces in parallel.",
        "Filesystem safety: unique_dest(), within(), safe_move() with atomic replace.",
    ]))

    flow.append(heading("Dependencies", 2))
    flow.append(bullet_list([
        "face_recognition: face encodings and comparisons.",
        "Pillow: image validation and safety limits.",
        "tqdm: progress visualization.",
    ]))

    flow.append(heading("Use Cases", 2))
    flow.append(bullet_list([
        "Organizing photo library by detected family members.",
        "Sorting event photos into folders per attendee.",
        "Filtering out unmatched images to a separate holding folder (user-defined).",
    ]))

    flow.append(add_diagram(
        "Sorting Pipeline",
        ["Image", "Validate", "Encode", "Compare", "First Match", "Safe Move"],
        [(0,1),(1,2),(2,3),(3,4),(4,5)]
    ))
    flow.append(PageBreak())

def build_section_video_to_gif(flow: List[Any], code: str, struct: Dict[str, Any]):
    flow.append(heading("4. Video to GIF Converter", 1))
    flow.append(Paragraph("File: c:\\Codes\\Py_Console\\video_to_gif.py", STYLES["BodyText"]))
    flow.append(Spacer(1, 8))

    flow.append(heading("Core Functionality", 2))
    flow.append(Paragraph("Converts video files to animated GIF or WebP with advanced optimization, time range selection, and optional batch mode.", STYLES["BodyText"]))

    flow.append(heading("Key Features", 2))
    flow.append(bullet_list([
        "Frame rate control via `--fps` and adaptive FPS calculation.",
        "Duration selection via `--start` and `--end` (seconds).",
        "Size optimization: multi-strategy GIF optimizer (quality/frame/palette).",
        "Resolution control via `--width` (maintains aspect ratio, capped by config).",
        "Batch conversion with concurrency (`--batch`, `--workers`).",
        "Output format selection: `--format gif|webp`.",
        "Optional source deletion with integrity verification (`--delete-source`).",
    ]))

    flow.append(heading("Inputs and Outputs", 2))
    flow.append(bullet_list([
        "Input: Single video file or directory for batch mode.",
        "Output: GIF or WebP file(s) to specified output path/dir (`-o/--output`).",
        "Supported video formats: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v, .3gp, .ogv, .ts, .mts",
    ]))

    flow.append(heading("Example Usage", 2))
    flow.append(code_block(r"""
# Create a 5-second GIF at 15fps
python c:\Codes\Py_Console\video_to_gif.py "D:\Clips\sample.mp4" --format gif -o "D:\Clips\sample.gif" --fps 15 --start 0 --end 5

# Batch convert a folder to WebP with 12fps
python c:\Codes\Py_Console\video_to_gif.py "D:\Clips" --batch --format webp --fps 12 --workers 4 -o "D:\Clips\Converted"
"""))

    flow.append(heading("Sample Runtime Output", 2))
    flow.append(code_block("""
Convert: sample.mp4 -> sample.gif
[INFO] Extracting frames (fps=15, width=640)
[INFO] Optimizing GIF: quality→frames→palette
[INFO] Success: output sample.gif (11.8MB)
"""))

    flow.append(heading("Code Structure Overview", 2))
    classes = [c for c in struct["classes"] if c["name"] in ("ConversionConfig", "VideoToGifConverter")]
    converter_methods = next((c["methods"] for c in classes if c["name"] == "VideoToGifConverter"), [])
    flow.append(bullet_list([
        "Classes: ConversionConfig, VideoToGifConverter, ConversionResult.",
        f"VideoToGifConverter methods: {', '.join(converter_methods) if converter_methods else 'See source'}",
        "Batch conversion: batch_convert() with concurrent/sequential options.",
        "Optimization internals: _optimize_by_quality(), _optimize_by_frame_reduction(), _optimize_by_palette_reduction().",
    ]))

    flow.append(heading("Dependencies", 2))
    flow.append(bullet_list([
        "moviepy: Video decoding and clip handling.",
        "Pillow: GIF/WebP frame processing and saving.",
        "numpy: frame manipulation.",
        "imageio: format IO helpers.",
        "tqdm: progress (when available).",
    ]))

    flow.append(heading("Use Cases", 2))
    flow.append(bullet_list([
        "Creating short animations or meme GIFs from video clips.",
        "Generating lightweight previews (WebP) for web or mobile use.",
        "Batch processing event recordings into optimized shareable formats.",
    ]))

    flow.append(add_diagram(
        "Conversion Flow",
        ["Open Clip", "Crop Range", "Resize/FPS", "Encode", "Optimize", "Output"],
        [(0,1),(1,2),(2,3),(3,4),(4,5)]
    ))

def build_front_matter(flow: List[Any], toc: TableOfContents):
    flow.append(Paragraph("Py_Console Scripts – Technical Documentation", STYLES["Title"]))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph("Comprehensive analysis and usage guide for the primary console utilities, including functional overview, code structure, dependencies, and sample scenarios.", STYLES["BodyText"]))
    flow.append(Spacer(1, 12))
    flow.append(heading("Table of Contents", 2))
    flow.append(toc)
    flow.append(PageBreak())

def compute_structures(source_map: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    structs = {}
    for name, path in source_map.items():
        code = read_file(path)
        structs[name] = parse_structure(code)
    return structs

def register_heading_for_toc(flowables: List[Any], doc: BaseDocTemplate):
    def after(flowable):
        if hasattr(flowable, 'outlineLevel'):
            text = flowable.getPlainText()
            level = flowable.outlineLevel
            doc.notify('TOCEntry', (level, text, doc.page))
    doc.afterFlowable = after

def main():
    global STYLES
    STYLES = getSampleStyleSheet()

    # Update existing Title style instead of redefining it
    title_style = STYLES['Title']
    title_style.fontName = "Helvetica-Bold"
    title_style.fontSize = 18
    title_style.spaceAfter = 10
    title_style.alignment = 1

    # Safe add or update Italic style
    if 'Italic' in STYLES:
        STYLES['Italic'].fontName = "Helvetica-Oblique"
        STYLES['Italic'].fontSize = 9
        STYLES['Italic'].textColor = colors.grey
    else:
        STYLES.add(ParagraphStyle(
            name="Italic",
            parent=STYLES["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor=colors.grey
        ))

    # Safe add Code style (unlikely to exist by default)
    if 'Code' not in STYLES:
        STYLES.add(ParagraphStyle(
            name="Code",
            fontName="Courier",
            fontSize=9,
            leading=11,
            backColor=colors.whitesmoke,
            leftIndent=6,
            rightIndent=6,
            spaceBefore=6,
            spaceAfter=6
        ))
    codes = {name: read_file(path) for name, path in FILES.items()}
    structs = compute_structures(FILES)

    doc = BaseDocTemplate(str(OUTPUT_PDF), pagesize=LETTER, leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=0.75*inch, bottomMargin=0.75*inch)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
    doc.addPageTemplates([PageTemplate(id='page', frames=[frame], onPage=on_page)])

    toc = make_toc()

    flow: List[Any] = []
    build_front_matter(flow, toc)
    register_heading_for_toc(flow, doc)

    build_section_convert_webp_to_jpg(flow, codes["convert_webp_to_jpg.py"], structs["convert_webp_to_jpg.py"])
    build_section_enhanced_video_downloader(flow, codes["enhanced_video_downloader.py"], structs["enhanced_video_downloader.py"])
    build_section_secure_sort_by_known_faces(flow, codes["secure_sort_by_known_faces.py"], structs["secure_sort_by_known_faces.py"])
    build_section_video_to_gif(flow, codes["video_to_gif.py"], structs["video_to_gif.py"])

    doc.build(flow)
    print(f"[OK] Generated: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()