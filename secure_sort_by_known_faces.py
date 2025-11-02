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

Dependencies:
  pip install face_recognition pillow tqdm
"""

import argparse
import os
import re
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import face_recognition
from PIL import Image, UnidentifiedImageError, ImageFile
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Pillow hardening / safety
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 50_000_000  # ~50MP cap

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_BYTES = 50 * 1024 * 1024  # 50 MB per file
CPU_MAX_FACTOR = 4  # cap workers to <= CPU cores * factor

# -------------------------
# Utilities
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
        validate_image(image_path)
        img = face_recognition.load_image_file(str(image_path))
        encs = face_recognition.face_encodings(img, num_jitters=max(1, int(jitter)))
        if not encs:
            return None
        return encs[0]
    except Exception:
        return None

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

# -------------------------
# Worker (parallel)
# -------------------------

def _analyze_image_worker(img_path_str: str, tolerance: float, known_encs: List, jitter: int) -> tuple:
    try:
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
# CLI + Interactive prompts
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

# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()
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

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_analyze_image_worker, str(p), args.tolerance, known_encs, args.jitter) for p in images]
            with tqdm(total=total, desc="Processing", unit="img") as pbar:
                for fut in as_completed(futures):
                    img_path_str, has_faces, match_bools, err = fut.result()
                    img_path = Path(img_path_str)

                    if err:
                        errors += 1
                        pbar.update(1)
                        continue
                    if not has_faces:
                        unmatched += 1
                        pbar.update(1)
                        continue
                    matches = [i for i, ok in enumerate(match_bools) if ok]
                    if not matches:
                        unmatched += 1
                        pbar.update(1)
                        continue
                    first_dest = dest_dirs[matches[0]]
                    try:
                        _ = safe_move(img_path, first_dest)
                        moved_to[str(first_dest)] += 1
                        total_moved += 1
                    except Exception:
                        errors += 1
                    # Update progress bar postfix
                    per_dest = " | ".join(f"{Path(d).name}:{moved_to[str(d)]}" for d in dest_dirs)
                    pbar.set_postfix_str(f"moved={total_moved} {per_dest}")
                    pbar.update(1)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    print("\n[SUMMARY]")
    for d in dest_dirs:
        print(f" {d}: {moved_to[str(d)]} moved")
    print(f" Total moved: {total_moved}")
    print(f" Unmatched/no-face: {unmatched}")
    if errors:
        print(f" Errors: {errors}")

if __name__ == "__main__":
    main()