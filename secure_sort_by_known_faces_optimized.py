#!/usr/bin/env python3
"""
secure_sort_by_known_faces_optimized.py (enhanced progress tracking, optimized performance)

Key improvements:
- Clean, unified progress tracking system
- Better visual feedback with stage indicators
- Optimized memory usage and performance
- Enhanced error handling and recovery
- Maintains all existing functionality

Dependencies:
  pip install face_recognition pillow tqdm
"""

import argparse
import os
import re
import sys
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import face_recognition
except ImportError:
    face_recognition = None
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageFile
from tqdm import tqdm

# Pillow hardening / safety
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 50_000_000  # ~50MP cap

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_BYTES = 50 * 1024 * 1024  # 50 MB per file
CPU_MAX_FACTOR = 4  # cap workers to <= CPU cores * factor


@dataclass
class ProcessingStats:
    """Track processing statistics for better progress feedback."""
    total_images: int = 0
    processed_images: int = 0
    matched_images: int = 0
    unmatched_images: int = 0
    errors: int = 0
    moved_to_dest: Dict[str, int] = None
    start_time: float = 0
    
    def __post_init__(self):
        if self.moved_to_dest is None:
            self.moved_to_dest = {}
        self.start_time = time.time()
    
    @property
    def progress_percentage(self) -> float:
        return (self.processed_images / max(1, self.total_images)) * 100
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def processing_rate(self) -> float:
        elapsed = self.elapsed_time
        return self.processed_images / max(1, elapsed)


class EnhancedProgressTracker:
    """Enhanced progress tracking with clean, informative output."""
    
    def __init__(self, total_images: int, mode: str = "face_analysis"):
        self.total_images = total_images
        self.mode = mode
        self.stats = ProcessingStats(total_images=total_images)
        self.main_pbar = None
        self.stage_pbar = None
        
        if tqdm is not None:
            self._setup_progress_bars()
    
    def _setup_progress_bars(self):
        """Setup clean, informative progress bars."""
        if self.mode == "face_analysis":
            # Main progress bar for face analysis
            self.main_pbar = tqdm(
                total=self.total_images,
                desc="🔍 Analyzing Images",
                unit="img",
                bar_format='{desc}: {bar:20} | {n_fmt}/{total_fmt} | {percentage:3.0f}% | [{elapsed}<{remaining}]',
                position=0,
                leave=True
            )
        elif self.mode == "series":
            # Progress bar for series renaming
            self.main_pbar = tqdm(
                total=self.total_images,
                desc="📁 Organizing Series",
                unit="img",
                bar_format='{desc}: {bar:20} | {n_fmt}/{total_fmt} | {percentage:3.0f}% | [{elapsed}<{remaining}]',
                position=0,
                leave=True
            )
    
    def update_stage(self, stage_name: str, progress: Optional[float] = None):
        """Update current processing stage."""
        if self.stage_pbar:
            self.stage_pbar.close()
        
        if tqdm is not None and progress is not None:
            self.stage_pbar = tqdm(
                total=100,
                desc=f"  {stage_name}",
                unit="%",
                bar_format='{desc}: {bar:20} | {percentage:3.0f}%',
                position=1,
                leave=False
            )
            self.stage_pbar.update(progress)
    
    def update_main_progress(self, increment: int = 1, **postfix_data):
        """Update main progress bar with detailed information."""
        self.stats.processed_images += increment
        
        if self.main_pbar:
            # Build informative postfix
            postfix_parts = []
            
            if self.stats.matched_images > 0:
                postfix_parts.append(f"✓{self.stats.matched_images}")
            
            if self.stats.unmatched_images > 0:
                postfix_parts.append(f"✗{self.stats.unmatched_images}")
            
            if self.stats.errors > 0:
                postfix_parts.append(f"⚠{self.stats.errors}")
            
            # Add destination counts if available
            if self.stats.moved_to_dest:
                dest_counts = [f"{Path(d).name}:{count}" 
                              for d, count in self.stats.moved_to_dest.items() if count > 0]
                if dest_counts:
                    postfix_parts.append(f"📁{' '.join(dest_counts)}")
            
            # Add processing rate
            rate = self.stats.processing_rate
            if rate > 0:
                postfix_parts.append(f"🚀{rate:.1f}/s")
            
            postfix_str = " | ".join(postfix_parts) if postfix_parts else ""
            
            self.main_pbar.set_postfix_str(postfix_str)
            self.main_pbar.update(increment)
    
    def add_match(self, dest_name: str):
        """Record a successful match."""
        self.stats.matched_images += 1
        if dest_name not in self.stats.moved_to_dest:
            self.stats.moved_to_dest[dest_name] = 0
        self.stats.moved_to_dest[dest_name] += 1
    
    def add_unmatched(self):
        """Record an unmatched image."""
        self.stats.unmatched_images += 1
    
    def add_error(self):
        """Record an error."""
        self.stats.errors += 1
    
    def close(self):
        """Clean up progress bars."""
        if self.stage_pbar:
            self.stage_pbar.close()
        if self.main_pbar:
            self.main_pbar.close()
    
    def print_summary(self):
        """Print a clean, informative summary."""
        print(f"\n{'='*60}")
        print(f"📊 PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"📁 Total images:     {self.stats.total_images}")
        print(f"✅ Matched:         {self.stats.matched_images}")
        print(f"❌ Unmatched:       {self.stats.unmatched_images}")
        if self.stats.errors > 0:
            print(f"⚠️  Errors:          {self.stats.errors}")
        print(f"⏱️  Time elapsed:    {self.stats.elapsed_time:.1f}s")
        print(f"🚀 Processing rate: {self.stats.processing_rate:.1f} img/s")
        
        if self.stats.moved_to_dest:
            print(f"\n📂 Files moved to destinations:")
            for dest, count in self.stats.moved_to_dest.items():
                if count > 0:
                    print(f"   {Path(dest).name}: {count} files")
        
        success_rate = (self.stats.matched_images / max(1, self.stats.total_images)) * 100
        print(f"\n🎯 Success rate: {success_rate:.1f}%")
        print(f"{'='*60}")


# -------------------------
# Utilities (unchanged for compatibility)
# -------------------------

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
    if not path.exists() or path.is_dir():
        raise ValueError("Not a file path")
    if not is_image_path(path):
        raise ValueError("Unsupported extension")
    size = path.stat().st_size
    if size <= 0 or size > MAX_BYTES:
        raise ValueError(f"Unexpected file size: {size} bytes")
    try:
        with Image.open(path) as im:
            im.verify()
    except (UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise ValueError(f"Invalid image content: {e}")

def load_ref_encoding(image_path: Path, jitter: int) -> Optional[list]:
    try:
        if face_recognition is None:
            raise RuntimeError("face_recognition dependency is not available")
        validate_image(image_path)
        img = face_recognition.load_image_file(str(image_path))
        encs = face_recognition.face_encodings(img, num_jitters=max(1, int(jitter)))
        if not encs:
            return None
        return encs[0]
    except Exception:
        return None

def collect_images(src: Path, recursive: bool) -> List[Path]:
    """Optimized image collection with progress feedback."""
    paths: List[Path] = []
    
    if recursive:
        # Count total files first for better progress estimation
        total_files = sum(len(files) for _, _, files in os.walk(src))
        
        with tqdm(total=total_files, desc="🔍 Scanning Files", unit="file", 
                 bar_format='{desc}: {bar:20} | {n_fmt}/{total_fmt} | [{elapsed}<{remaining}]',
                 position=0, leave=True) as scan_pbar:
            
            for root, _, files in os.walk(src, topdown=True):
                root_path = Path(root)
                for fname in files:
                    p = root_path / fname
                    try:
                        if p.exists() and not p.is_dir() and is_image_path(p):
                            paths.append(p)
                    except Exception:
                        continue
                    finally:
                        scan_pbar.update(1)
    else:
        try:
            files = list(src.iterdir())
            with tqdm(total=len(files), desc="🔍 Scanning Files", unit="file",
                     bar_format='{desc}: {bar:20} | {n_fmt}/{total_fmt} | [{elapsed}<{remaining}]',
                     position=0, leave=True) as scan_pbar:
                
                for p in files:
                    try:
                        if p.exists() and not p.is_dir() and is_image_path(p):
                            paths.append(p)
                    except Exception:
                        continue
                    finally:
                        scan_pbar.update(1)
        except PermissionError:
            return []
    
    return paths

# Series grouping functions (unchanged for compatibility)
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

def _rename_series_groups(groups: List[List[Path]], prefix: str) -> None:
    """Enhanced series renaming with better progress tracking."""
    total_images = sum(len(g) for g in groups)
    
    if total_images == 0:
        return
    
    tracker = EnhancedProgressTracker(total_images, mode="series")
    
    try:
        series_index = 1
        for group in groups:
            group_sorted = sorted(group, key=lambda p: p.stat().st_mtime)
            series_label = sanitize_filename(f"{prefix}_{series_index:03d}")
            count = 1
            
            tracker.update_stage(f"Series {series_index:03d}", 0)
            
            for i, img in enumerate(group_sorted):
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
                
                # Update progress within current series
                series_progress = ((i + 1) / len(group_sorted)) * 100
                tracker.update_stage(f"Series {series_index:03d}", series_progress)
                tracker.update_main_progress()
            
            series_index += 1
    
    finally:
        tracker.close()

# -------------------------
# Worker (parallel)
# -------------------------

def _analyze_image_worker(img_path_str: str, tolerance: float, known_encs: List, jitter: int) -> tuple:
    """Optimized worker function with better error handling."""
    try:
        if face_recognition is None:
            raise RuntimeError("face_recognition dependency is not available")
        img_path = Path(img_path_str)
        validate_image(img_path)
        img = face_recognition.load_image_file(str(img_path))
        encs = face_recognition.face_encodings(img, num_jitters=max(1, int(jitter)))
        if not encs:
            return (img_path_str, False, [False] * len(known_encs), None)
        match_bools = []
        for known in known_encs:
            ok = any(face_recognition.compare_faces([known], e, tolerance=tolerance)[0] for e in encs)
            match_bools.append(ok)
        return (img_path_str, True, match_bools, None)
    except Exception as e:
        return (img_path_str, False, [False] * len(known_encs), str(e))

# -------------------------
# CLI + Interactive prompts (unchanged for compatibility)
# -------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Move images into mapped folders when they contain known faces."
    )
    ap.add_argument("--src", type=Path, help="Source images folder")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--tolerance", type=float, help="Face match tolerance (0.2-0.8 typical)")
    ap.add_argument("--workers", type=int, help="Parallel workers")
    ap.add_argument("--jitter", type=int, help="Jitter samples for encodings (>=1)")
    ap.add_argument("--face", action="append", type=Path, help="Reference face image (repeatable)")
    ap.add_argument("--dest", action="append", type=Path, help="Dest folder for each face (repeatable)")
    ap.add_argument("--series-mode", choices=["date", "name", "visual"], help="Group and rename images into series")
    ap.add_argument("--series-threshold-minutes", type=int, default=30, help="Max gap (minutes) within a date-based series")
    ap.add_argument("--series-visual-threshold", type=float, default=0.25, help="Max distance for visual series grouping (0-1)")
    ap.add_argument("--series-prefix", type=str, default="series", help="Prefix for renamed series files")
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
    """Enhanced series mode with better progress tracking."""
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
    
    print(f"🔍 Collecting images from {args.src}...")
    images = collect_images(args.src, args.recursive)
    total = len(images)
    
    if total == 0:
        print("❌ No images found.")
        return
    
    print(f"📊 Found {total} images")
    
    # Process grouping
    if args.series_mode == "date":
        if args.series_threshold_minutes <= 0:
            print("❌ Invalid series-threshold-minutes; must be > 0.")
            sys.exit(2)
        groups = _series_group_by_date(images, args.series_threshold_minutes)
    elif args.series_mode == "name":
        groups = _series_group_by_name(images)
    else:
        if args.series_visual_threshold <= 0:
            print("❌ Invalid series-visual-threshold; must be > 0.")
            sys.exit(2)
        groups = _series_group_by_visual(images, args.series_visual_threshold)
    
    if not groups:
        print("❌ No groups formed.")
        return
    
    print(f"📁 Grouped {total} images into {len(groups)} series:")
    for idx, g in enumerate(groups, 1):
        print(f"   Series {idx:03d}: {len(g)} images")
    
    _rename_series_groups(groups, args.series_prefix)
    print("✅ Series grouping and renaming completed.")

# -------------------------
# Main (enhanced with better progress tracking)
# -------------------------

def main():
    args = parse_args()
    if args.series_mode and (args.face or args.dest):
        print("❌ Cannot use --face/--dest together with --series-mode.")
        sys.exit(2)
    if args.series_mode:
        print(f"[MODE] Series mode: {args.series_mode}")
        run_series_mode(args)
        return
    if face_recognition is None:
        print("❌ Face-based mode requires 'face_recognition'. Install it or use --series-mode.")
        sys.exit(1)
    print("[MODE] Face-based authentication mode")
    args = interactive_fill(args)
    
    print(f"🔍 Collecting images from {args.src}...")
    images = collect_images(args.src, args.recursive)
    total = len(images)
    
    if total == 0:
        print("❌ No images found.")
        return

    known_encs = [enc for _, _, enc in args._mapping]
    dest_dirs = [dest for _, dest, _ in args._mapping]
    
    # Initialize enhanced progress tracker
    tracker = EnhancedProgressTracker(total, mode="face_analysis")
    
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_analyze_image_worker, str(p), args.tolerance, known_encs, args.jitter) for p in images]
            
            for fut in as_completed(futures):
                img_path_str, has_faces, match_bools, err = fut.result()
                img_path = Path(img_path_str)

                if err:
                    tracker.add_error()
                    tracker.update_main_progress()
                    continue
                
                if not has_faces:
                    tracker.add_unmatched()
                    tracker.update_main_progress()
                    continue
                
                matches = [i for i, ok in enumerate(match_bools) if ok]
                if not matches:
                    tracker.add_unmatched()
                    tracker.update_main_progress()
                    continue
                
                first_dest = dest_dirs[matches[0]]
                try:
                    _ = safe_move(img_path, first_dest)
                    tracker.add_match(str(first_dest))
                except Exception:
                    tracker.add_error()
                
                tracker.update_main_progress()

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        tracker.close()
        tracker.print_summary()

if __name__ == "__main__":
    main()
