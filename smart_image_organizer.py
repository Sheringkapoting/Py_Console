#!/usr/bin/env python3
"""
secure_sort_by_known_faces.py (robust, quiet output, interactive if needed)

Key features:
- Quiet runtime: only a single tqdm progress bar and a concise final summary.
- Progress bar shows: moved={total_moved} plus per-destination counts beside bar.
- Interactive input if CLI args are missing: prompts for src, recursive, tolerance,
  workers, jitter, and an arbitrary number of (face, dest) pairs.
- Safe filesystem operations: unique naming, traversal prevention, size/pixel caps.
- Parallelized face analysis; single-threaded file moves.
- Moves to first matched destination only (no extra copies).
- Series continuation: detects existing numbered series and continues numbering.

Dependencies:
  pip install face_recognition pillow tqdm

Usage Examples:
  # Face-based sorting
  python secure_sort_by_known_faces.py --src /photos --recursive --face person1.jpg --dest /photos/person1
  
  # Series grouping with continuation (default behavior)
  python secure_sort_by_known_faces.py --src /photos --series-mode date --series-prefix vacation
  
  # Series grouping without continuation (start from 001)
  python secure_sort_by_known_faces.py --src /photos --series-mode date --no-continue-series
"""

import argparse
import os
import re
import sys
import shutil
import warnings
import threading
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

try:
    import face_recognition
except ImportError:
    face_recognition = None
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageFile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Pillow hardening / safety
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 50_000_000  # ~50MP cap

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}
MAX_BYTES = 50 * 1024 * 1024  # 50 MB per file
CPU_MAX_FACTOR = 4  # cap workers to <= CPU cores * factor

# Global termination flag
TERMINATION_FLAG = threading.Event()

# -------------------------
# Utilities
# -------------------------

def check_escape_key():
    """Check for Escape key press in a non-blocking way."""
    try:
        import msvcrt
        if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':  # Escape key
            return True
    except ImportError:
        # Non-Windows platforms - use alternative method
        try:
            import select
            import termios
            import tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    if sys.stdin.read(1) == '\x1b':
                        return True
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except (ImportError, OSError):
            pass
    return False

def monitor_escape_key():
    """Background thread to monitor for Escape key presses."""
    while not TERMINATION_FLAG.is_set():
        if check_escape_key():
            TERMINATION_FLAG.set()
            print("\n[INFO] Termination requested (Esc key pressed)")
            break
        time.sleep(0.1)  # Check every 100ms

def is_image_path(p: Path) -> bool:
    return p.suffix.lower() in VALID_EXTS

def sanitize_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]", "", name)
    name = name.lstrip(".") or "image"
    return name

def unique_dest(dest_dir: Path, base_name: str) -> Path:
    base_name = sanitize_filename(base_name)
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    candidate = dest_dir / (stem + suffix)
    i = 1
    while candidate.exists():
        candidate = dest_dir / f"{stem}({i}){suffix}"
        i += 1
    return candidate

def within(parent: Path, child: Path) -> bool:
    try:
        parent_resolved = parent.resolve(strict=True)
    except FileNotFoundError:
        parent_resolved = parent.resolve(strict=False)
    try:
        child_resolved = child.resolve(strict=True)
    except FileNotFoundError:
        child_resolved = child.parent.resolve(strict=False)
    return parent_resolved == child_resolved or parent_resolved in child_resolved.parents

def safe_move(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = unique_dest(dest_dir, src.name)
    if not within(dest_dir, dest):
        raise RuntimeError("Refusing to write outside destination directory.")
    try:
        os.replace(src, dest)  # atomic if same FS
        return dest
    except Exception:
        with src.open("rb") as rf, open(dest, "xb") as wf:
            shutil.copyfileobj(rf, wf, length=1024 * 1024)
        src.unlink(missing_ok=False)
        return dest

def validate_image(path: Path) -> None:
    """Validate image file with optimized checks."""
    if not path.exists() or path.is_dir():
        raise ValueError("Not a file path")
    if not is_image_path(path):
        raise ValueError("Unsupported extension")
    
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_BYTES:
            raise ValueError(f"Unexpected file size: {size} bytes")
        
        # Quick validation without full loading
        with Image.open(path) as im:
            im.verify()
    except (UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise ValueError(f"Invalid image content: {e}")

def load_ref_encoding(image_path: Path, jitter: int) -> Optional[list]:
    """
    Load reference face encoding with enhanced face detection and quality checks.
    """
    try:
        if face_recognition is None:
            raise RuntimeError("face_recognition dependency is not available")
        validate_image(image_path)
        img = face_recognition.load_image_file(str(image_path))
        
        # Detect face locations first
        face_locations = face_recognition.face_locations(img, model="hog")
        if not face_locations:
            return None
        
        # Filter faces by size (avoid tiny faces)
        valid_faces = []
        img_height, img_width = img.shape[:2]
        min_face_size = min(img_height, img_width) * 0.1  # 10% of image dimension
        
        for i, location in enumerate(face_locations):
            top, right, bottom, left = location
            face_width = right - left
            face_height = bottom - top
            
            # Check if face is large enough
            if face_width >= min_face_size and face_height >= min_face_size:
                valid_faces.append((i, location))
        
        if not valid_faces:
            return None
        
        # Use the largest valid face
        largest_face = max(valid_faces, key=lambda x: (x[1][1] - x[1][3]) * (x[1][2] - x[1][0]))
        face_idx = largest_face[0]
        
        # Get encodings with higher jitter for better accuracy
        encs = face_recognition.face_encodings(img, num_jitters=max(1, int(jitter)), known_face_locations=[face_locations[face_idx]])
        if not encs:
            return None
            
        return encs[0]
    except Exception:
        return None

def analyze_faces_enhanced(img_path: Path, tolerance: float, known_encs: List, jitter: int, 
                          min_face_size: float = 0.08, max_face_size: float = 0.9, 
                          face_model: str = "hog", strict_matching: bool = False) -> tuple:
    """
    Enhanced face analysis with quality checks and precise matching.
    Checks for termination flag to allow graceful shutdown.
    """
    # Check for termination before processing
    if TERMINATION_FLAG.is_set():
        return (str(img_path), False, [False] * len(known_encs), "Terminated by user")
        
    try:
        if face_recognition is None:
            raise RuntimeError("face_recognition dependency is not available")
        
        # Validate image once at the beginning
        validate_image(img_path)
        img = face_recognition.load_image_file(str(img_path))
        
        # Detect face locations with fallback
        try:
            face_locations = face_recognition.face_locations(img, model=face_model)
        except:
            face_locations = face_recognition.face_locations(img, model="hog")
        
        if not face_locations:
            return (str(img_path), False, [False] * len(known_encs), None)
        
        # Filter faces by quality with optimized calculations
        img_height, img_width = img.shape[:2]
        min_face_pixels = min(img_height, img_width) * min_face_size
        max_face_pixels = min(img_height, img_width) * max_face_size
        
        valid_faces = []
        for i, location in enumerate(face_locations):
            top, right, bottom, left = location
            face_width = right - left
            face_height = bottom - top
            aspect_ratio = face_width / face_height if face_height > 0 else 0
            
            # Optimized size and aspect ratio checks
            if (min_face_pixels <= face_width <= max_face_pixels and 
                min_face_pixels <= face_height <= max_face_pixels and
                0.7 <= aspect_ratio <= 1.5):
                valid_faces.append((i, location))
        
        if not valid_faces:
            return (str(img_path), False, [False] * len(known_encs), None)
        
        # Check for termination after face detection
        if TERMINATION_FLAG.is_set():
            return (str(img_path), False, [False] * len(known_encs), "Terminated by user")
            
        # Get encodings for valid faces only
        valid_locations = [loc for idx, loc in valid_faces]
        encodings = face_recognition.face_encodings(img, num_jitters=max(1, int(jitter)), known_face_locations=valid_locations)
        
        if not encodings:
            return (str(img_path), False, [False] * len(known_encs), None)
        
        # Optimized matching with pre-calculated tolerance levels
        match_results = []
        for known_enc in known_encs:
            best_match = False
            
            tolerance_levels = (
                [tolerance * 0.5, tolerance * 0.7, tolerance * 0.9] 
                if strict_matching else 
                [tolerance * 0.6, tolerance * 0.8, tolerance]
            )
            
            for enc in encodings:
                for tol_level in tolerance_levels:
                    if face_recognition.compare_faces([known_enc], enc, tolerance=tol_level)[0]:
                        best_match = True
                        break
                if best_match:
                    break
            
            match_results.append(best_match)
        
        return (str(img_path), True, match_results, None)
        
    except Exception as e:
        return (str(img_path), False, [False] * len(known_encs), str(e))

def collect_images(src: Path, recursive: bool) -> List[Path]:
    paths: List[Path] = []
    if recursive:
        for root, _, files in os.walk(src, topdown=True):
            root_path = Path(root)
            for fname in files:
                p = root_path / fname
                try:
                    if p.exists() and not p.is_dir() and is_image_path(p):
                        paths.append(p)
                except Exception:
                    continue
    else:
        try:
            for p in src.iterdir():
                try:
                    if p.exists() and not p.is_dir() and is_image_path(p):
                        paths.append(p)
                except Exception:
                    continue
        except PermissionError:
            return []
    return paths

def _series_group_by_date(images: List[Path], threshold_minutes: int) -> List[List[Path]]:
    if not images:
        return []
    threshold_seconds = max(1, threshold_minutes) * 60
    images_sorted = sorted(images, key=lambda p: p.stat().st_mtime)
    groups: List[List[Path]] = []
    current_group: List[Path] = [images_sorted[0]]
    last_time = images_sorted[0].stat().st_mtime
    for img in images_sorted[1:]:
        t = img.stat().st_mtime
        if t - last_time <= threshold_seconds:
            current_group.append(img)
        else:
            groups.append(current_group)
            current_group = [img]
        last_time = t
    if current_group:
        groups.append(current_group)
    return groups

def _series_group_by_name(images: List[Path]) -> List[List[Path]]:
    groups_map: Dict[str, List[Path]] = {}
    for img in images:
        stem = img.stem.lower()
        m = re.match(r"([a-zA-Z]+)", stem)
        key = m.group(1) if m else stem
        groups_map.setdefault(key, []).append(img)
    groups: List[List[Path]] = []
    for key, vals in groups_map.items():
        vals_sorted = sorted(vals, key=lambda p: p.name.lower())
        groups.append(vals_sorted)
    groups.sort(key=lambda g: g[0].stat().st_mtime if g else 0.0)
    return groups

def _series_group_by_visual(images: List[Path], threshold: float) -> List[List[Path]]:
    if not images:
        return []
    vectors: Dict[Path, np.ndarray] = {}
    valid_images: List[Path] = []
    for img in images:
        try:
            validate_image(img)
            with Image.open(img) as im:
                im = im.convert("RGB")
                im = im.resize((16, 16))
                arr = np.asarray(im, dtype=np.float32) / 255.0
                vec = arr.reshape(-1)
                vectors[img] = vec
                valid_images.append(img)
        except Exception:
            continue
    groups: List[List[Path]] = []
    centers: List[np.ndarray] = []
    for img in sorted(valid_images, key=lambda p: p.stat().st_mtime):
        v = vectors[img]
        if not centers:
            centers.append(v)
            groups.append([img])
            continue
        dists = [np.linalg.norm(v - c) for c in centers]
        best_idx = int(np.argmin(dists))
        if dists[best_idx] <= max(1e-6, threshold):
            groups[best_idx].append(img)
            n = len(groups[best_idx])
            centers[best_idx] = (centers[best_idx] * (n - 1) + v) / float(n)
        else:
            centers.append(v)
            groups.append([img])
    return groups

def validate_series_prefix(prefix: str) -> str:
    """
    Validate and sanitize series prefix.
    """
    if not prefix or not prefix.strip():
        return "series"
    return sanitize_filename(prefix.strip())

def detect_existing_series(images: List[Path], prefix: str) -> Dict[str, int]:
    """
    Detect existing numbered series in images and return the highest series number for each prefix.
    """
    series_pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_(\d+)")
    existing_series: Dict[str, int] = {}
    
    for img in images:
        match = series_pattern.match(img.stem)
        if match:
            series_num = int(match.group(1))
            existing_series[prefix] = max(existing_series.get(prefix, 0), series_num)
    
    return existing_series

def filter_already_renamed_images(images: List[Path], prefix: str) -> Tuple[List[Path], int]:
    """
    Filter out images that are already part of a series and return only new images.
    Returns (new_images, count_of_filtered_images)
    """
    series_pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_(\d+)")
    new_images = []
    filtered_count = 0
    
    for img in images:
        if series_pattern.match(img.stem):
            filtered_count += 1
        else:
            new_images.append(img)
    
    return new_images, filtered_count

def get_next_series_number(existing_series: Dict[str, int], prefix: str) -> int:
    """
    Get the next series number based on existing series.
    """
    return existing_series.get(prefix, 0) + 1

def _rename_series_groups(groups: List[List[Path]], prefix: str, continue_series: bool = True) -> None:
    """
    Rename series groups with optional continuation of existing numbering.
    Includes Esc key termination support.
    """
    total_images = sum(len(g) for g in groups)
    if total_images == 0:
        return
    
    # Detect existing series if continuation is enabled
    all_images = [img for group in groups for img in group]
    existing_series = detect_existing_series(all_images, prefix) if continue_series else {}
    series_index = get_next_series_number(existing_series, prefix)
    
    use_tqdm = "tqdm" in globals() and tqdm is not None
    
    # Start escape key monitoring thread
    monitor_thread = threading.Thread(target=monitor_escape_key, daemon=True)
    monitor_thread.start()
    
    try:
        with tqdm(total=total_images, desc="Renaming series", unit="img", dynamic_ncols=True, mininterval=0.1,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as bar:
            for group_idx, group in enumerate(groups, 1):
                # Check for termination before processing each group
                if TERMINATION_FLAG.is_set():
                    print("\n[INFO] Terminating series renaming...")
                    break
                    
                group_sorted = sorted(group, key=lambda p: p.stat().st_mtime)
                series_label = sanitize_filename(f"{prefix}_{series_index:03d}")
                count = 1
                
                # Update progress bar description with current series info
                bar.set_description(f"Series {series_index:03d} ({len(group)} imgs)")
                
                for img in group_sorted:
                    # Check for termination before processing each image
                    if TERMINATION_FLAG.is_set():
                        print("\n[INFO] Terminating series renaming...")
                        break
                        
                    ext = img.suffix.lower()
                    new_base = f"{series_label}_{count:03d}{ext}"
                    dest_dir = img.parent
                    dest_path = unique_dest(dest_dir, new_base)
                    
                    try:
                        os.replace(img, dest_path)
                    except Exception:
                        with img.open("rb") as rf, open(dest_path, "xb") as wf:
                            shutil.copyfileobj(rf, wf, length=1024 * 1024)
                        img.unlink(missing_ok=False)
                    
                    count += 1
                    bar.update(1)
                    
                    # Update postfix with current progress
                    bar.set_postfix({
                        'series': f"{series_index}/{len(groups)}",
                        'in_series': f"{count-1}/{len(group)}",
                        'renamed': bar.n
                    })
                
                # Check for termination after each group
                if TERMINATION_FLAG.is_set():
                    break
                    
                series_index += 1
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        # Ensure termination flag is set for cleanup
        TERMINATION_FLAG.set()
        
    # Exit cleanly if terminated early
    if TERMINATION_FLAG.is_set() and bar.n < total_images:
        print(f" Status: Terminated early (renamed {bar.n} of {total_images} images)")
        sys.exit(0)

# -------------------------
# Worker (parallel)
# -------------------------

def _analyze_image_worker(img_path_str: str, tolerance: float, known_encs: List, jitter: int,
                          min_face_size: float, max_face_size: float, face_model: str, strict_matching: bool) -> tuple:
    """
    Worker function for parallel face analysis using enhanced detection.
    Checks for termination flag to allow graceful shutdown.
    """
    # Check for termination at start of worker
    if TERMINATION_FLAG.is_set():
        img_path = Path(img_path_str)
        return (str(img_path), False, [False] * len(known_encs), "Terminated by user")
        
    img_path = Path(img_path_str)
    return analyze_faces_enhanced(img_path, tolerance, known_encs, jitter, 
                                 min_face_size, max_face_size, face_model, strict_matching)

# -------------------------
# CLI + Interactive prompts
# -------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Move images into mapped folders when they contain known faces."
    )
    ap.add_argument("--src", type=Path, help="Source images folder")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--tolerance", type=float, help="Face match tolerance (0.2-0.8 typical)")
    ap.add_argument("--min-face-size", type=float, default=0.08, help="Minimum face size as fraction of image (0.05-0.2)")
    ap.add_argument("--max-face-size", type=float, default=0.9, help="Maximum face size as fraction of image (0.5-1.0)")
    ap.add_argument("--face-model", choices=["hog", "cnn"], default="hog", help="Face detection model (hog=fast, cnn=accurate)")
    ap.add_argument("--strict-matching", action="store_true", help="Use stricter face matching for better precision")
    ap.add_argument("--workers", type=int, help="Parallel workers")
    ap.add_argument("--jitter", type=int, help="Jitter samples for encodings (>=1)")
    ap.add_argument("--face", action="append", type=Path, help="Reference face image (repeatable)")
    ap.add_argument("--dest", action="append", type=Path, help="Dest folder for each face (repeatable)")
    ap.add_argument("--series-mode", choices=["date", "name", "visual"], help="Group and rename images into series")
    ap.add_argument("--series-threshold-minutes", type=int, default=30, help="Max gap (minutes) within a date-based series")
    ap.add_argument("--series-visual-threshold", type=float, default=0.25, help="Max distance for visual series grouping (0-1)")
    ap.add_argument("--series-prefix", type=str, default="series", help="Prefix for renamed series files")
    ap.add_argument("--no-continue-series", action="store_true", help="Do not continue existing series numbering (start from 001)")
    return ap.parse_args()

def prompt_bool(prompt: str, default: bool) -> bool:
    yn = "Y/n" if default else "y/N"
    while True:
        ans = input(f"{prompt} [{yn}]: ").strip().lower()
        if ans == "" and default is not None:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False

def prompt_float(prompt: str, default: float, min_v: float, max_v: float) -> float:
    while True:
        ans = input(f"{prompt} [{default}]: ").strip()
        if ans == "":
            return default
        try:
            v = float(ans)
            if v < min_v or v > max_v:
                continue
            return v
        except ValueError:
            continue

def prompt_int(prompt: str, default: int, min_v: int, max_v: int) -> int:
    while True:
        ans = input(f"{prompt} [{default}]: ").strip()
        if ans == "":
            return default
        try:
            v = int(ans)
            if min_v <= v <= max_v:
                return v
        except ValueError:
            continue

def gather_pairs_interactively() -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    print("\nAdd (face image -> destination folder) pairs.")
    while True:
        face_str = input("Reference face image path (Enter to finish): ").strip()
        if face_str == "":
            break
        face = Path(face_str)
        if not face.exists():
            print("File not found.")
            continue
        dest_str = input("Destination folder path: ").strip()
        if not dest_str:
            print("Destination required.")
            continue
        dest = Path(dest_str)
        pairs.append((face, dest))
    return pairs

def interactive_fill(args: argparse.Namespace) -> argparse.Namespace:
    while args.src is None:
        src_str = input("Source folder: ").strip()
        if src_str:
            p = Path(src_str)
            if p.exists() and p.is_dir():
                args.src = p
            else:
                print("Invalid folder.")

    if not args.recursive:
        args.recursive = prompt_bool("Recurse subfolders?", False)
    if args.tolerance is None:
        args.tolerance = prompt_float("Tolerance (0.2-0.8 typical)", 0.5, 0.05, 1.0)
    if args.min_face_size is None:
        args.min_face_size = prompt_float("Minimum face size (0.05-0.2)", 0.08, 0.05, 0.2)
    if args.max_face_size is None:
        args.max_face_size = prompt_float("Maximum face size (0.5-1.0)", 0.9, 0.5, 1.0)
    if args.face_model is None:
        model_choice = input("Face detection model [hog/cnn] (default: hog): ").strip().lower()
        args.face_model = model_choice if model_choice in ["cnn", "hog"] else "hog"
    if not args.strict_matching:
        args.strict_matching = prompt_bool("Use strict face matching?", False)
    
    cpu = os.cpu_count() or 1
    max_w = cpu * CPU_MAX_FACTOR
    if args.workers is None:
        args.workers = prompt_int("Workers", cpu, 1, max_w)
    if args.jitter is None or args.jitter < 1:
        args.jitter = prompt_int("Jitter (>=1)", 1, 1, 20)

    pairs: List[Tuple[Path, Path]] = []
    if args.face and args.dest:
        if len(args.face) == len(args.dest):
            pairs.extend(zip(args.face, args.dest))
    if not pairs:
        pairs = gather_pairs_interactively()

    valid_pairs: List[Tuple[Path, Path, list]] = []
    for (face, dest) in pairs:
        enc = load_ref_encoding(face, args.jitter)
        if enc is None:
            print(f"[WARN] Skipping: no face in {face}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        valid_pairs.append((face, dest, enc))

    if not valid_pairs:
        print("No valid face/dest pairs.")
        sys.exit(2)
    args._mapping = valid_pairs
    return args

def run_series_mode(args: argparse.Namespace) -> None:
    """
    Run series grouping and renaming mode.
    """
    if args.src is None:
        while args.src is None:
            src_str = input("Source folder: ").strip()
            if not src_str:
                continue
            p = Path(src_str)
            if p.exists() and p.is_dir():
                args.src = p
            else:
                print("Invalid folder.")
    if not args.recursive:
        use_rec = input("Recurse subfolders? [y/N]: ").strip().lower()
        if use_rec in ("y", "yes"):
            args.recursive = True
    
    # Validate and sanitize series prefix
    args.series_prefix = validate_series_prefix(args.series_prefix)
    
    images = collect_images(args.src, args.recursive)
    total = len(images)
    if total == 0:
        print("No images found.")
        return
    
    # Check for existing series before processing (use all images for detection)
    existing_series = detect_existing_series(images, args.series_prefix)
    continue_series = not args.no_continue_series
    
    # Filter out already renamed images for processing
    images, filtered_count = filter_already_renamed_images(images, args.series_prefix)
    new_total = len(images)
    
    if filtered_count > 0:
        print(f"[INFO] Skipped {filtered_count} already renamed images")
    
    if new_total == 0:
        print("No new images to process (all are already in series).")
        return
    
    if existing_series and continue_series:
        max_series = max(existing_series.values())
        next_series = get_next_series_number(existing_series, args.series_prefix)
        print(f"[INFO] Detected existing series '{args.series_prefix}' up to #{max_series:03d}")
        print(f"[INFO] Will continue numbering from #{next_series:03d}")
    elif args.no_continue_series:
        print("[INFO] Starting fresh series numbering from 001")
    else:
        print("[INFO] No existing series detected, starting from 001")
    
    # Group images
    try:
        if args.series_mode == "date":
            if args.series_threshold_minutes <= 0:
                print("Invalid series-threshold-minutes; must be > 0.")
                sys.exit(2)
            groups = _series_group_by_date(images, args.series_threshold_minutes)
        elif args.series_mode == "name":
            groups = _series_group_by_name(images)
        else:  # visual
            if args.series_visual_threshold <= 0:
                print("Invalid series-visual-threshold; must be > 0.")
                sys.exit(2)
            groups = _series_group_by_visual(images, args.series_visual_threshold)
    except Exception as e:
        print(f"[ERROR] Failed to group images: {e}")
        sys.exit(2)
    
    if not groups:
        print("No groups formed.")
        return
    
    print(f"[INFO] Grouped {new_total} new images into {len(groups)} series")
    for idx, g in enumerate(groups, 1):
        print(f" Series {idx:03d}: {len(g)} images")
    
    # Confirm before renaming if there are existing series
    if existing_series and continue_series:
        confirm = input(f"\nContinue renaming {new_total} new images? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Operation cancelled.")
            return
    
    # Rename with series continuation
    print("[INFO] Press ESC key to terminate renaming gracefully")
    try:
        _rename_series_groups(groups, args.series_prefix, continue_series)
        print("Series grouping and renaming completed.")
    except Exception as e:
        print(f"[ERROR] Failed during renaming: {e}")
        sys.exit(2)

# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()
    if args.series_mode and (args.face or args.dest):
        print("Cannot use --face/--dest together with --series-mode.")
        sys.exit(2)
    if args.series_mode:
        print(f"[MODE] Series mode: {args.series_mode}")
        run_series_mode(args)
        return
    if face_recognition is None:
        print("Face-based mode requires 'face_recognition'. Install it or use --series-mode.")
        sys.exit(2)
    print("[MODE] Face-based authentication mode")
    args = interactive_fill(args)
    images = collect_images(args.src, args.recursive)
    total = len(images)
    if total == 0:
        print("No images found.")
        return

    known_encs = [enc for _, _, enc in args._mapping]
    dest_dirs = [dest for _, dest, _ in args._mapping]
    moved_to: Dict[str, int] = {str(d): 0 for d in dest_dirs}
    unmatched = 0
    errors = 0
    total_moved = 0

    print(f"[INFO] Processing {total} images from {args.src}")
    print("[INFO] Press ESC key to terminate processing gracefully\n")
    
    # Start escape key monitoring thread
    monitor_thread = threading.Thread(target=monitor_escape_key, daemon=True)
    monitor_thread.start()

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_analyze_image_worker, str(p), args.tolerance, known_encs, args.jitter,
                               args.min_face_size, args.max_face_size, args.face_model, args.strict_matching) 
                      for p in images]
            with tqdm(total=total, desc="Processing images", unit="img", dynamic_ncols=True, mininterval=0.1,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
                pbar.set_postfix({'moved': 0, 'unmatched': 0, 'errors': 0}, refresh=False)
                
                for fut in as_completed(futures):
                    # Check for termination before processing result
                    if TERMINATION_FLAG.is_set():
                        print("\n[INFO] Terminating processing...")
                        # Cancel remaining futures
                        for remaining_fut in futures:
                            if not remaining_fut.done():
                                remaining_fut.cancel()
                        break
                        
                    img_path_str, has_faces, match_bools, err = fut.result()
                    img_path = Path(img_path_str)
                    processed = pbar.n  # Get current processed count

                    if err:
                        errors += 1
                        if "Terminated by user" in str(err):
                            break
                        pbar.set_postfix({'moved': total_moved, 'unmatched': unmatched, 'errors': errors}, refresh=False)
                        pbar.update(1)
                        continue
                        
                    if not has_faces:
                        unmatched += 1
                        pbar.set_postfix({'moved': total_moved, 'unmatched': unmatched, 'errors': errors}, refresh=False)
                        pbar.update(1)
                        continue
                        
                    matches = [i for i, ok in enumerate(match_bools) if ok]
                    if not matches:
                        unmatched += 1
                        pbar.set_postfix({'moved': total_moved, 'unmatched': unmatched, 'errors': errors}, refresh=False)
                        pbar.update(1)
                        continue
                        
                    first_dest = dest_dirs[matches[0]]
                    try:
                        _ = safe_move(img_path, first_dest)
                        moved_to[str(first_dest)] += 1
                        total_moved += 1
                    except Exception:
                        errors += 1
                    
                    # Update progress bar with optimized postfix
                    pbar.set_postfix({'moved': total_moved, 'unmatched': unmatched, 'errors': errors}, refresh=False)
                    pbar.update(1)
                    
                    # Check for termination after each update
                    if TERMINATION_FLAG.is_set():
                        print("\n[INFO] Terminating processing...")
                        # Cancel remaining futures
                        for remaining_fut in futures:
                            if not remaining_fut.done():
                                remaining_fut.cancel()
                        break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        
    # Ensure termination flag is set for cleanup
    TERMINATION_FLAG.set()
    
    # Calculate final processed count
    processed_count = total_moved + unmatched + errors

    print("\n[SUMMARY]")
    print(f" Processed: {processed_count}/{total} images")
    for d in dest_dirs:
        print(f" {d}: {moved_to[str(d)]} moved")
    print(f" Total moved: {total_moved}")
    print(f" Unmatched/no-face: {unmatched}")
    if errors:
        print(f" Errors: {errors}")
    if TERMINATION_FLAG.is_set() and processed_count < total:
        print(f" Status: Terminated early (processed {processed_count} of {total} images)")
        
    # Exit cleanly if terminated early
    if TERMINATION_FLAG.is_set() and processed_count < total:
        sys.exit(0)

if __name__ == "__main__":
    main()
