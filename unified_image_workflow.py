#!/usr/bin/env python3
"""
unified_image_workflow.py  —  5-Stage Image Management Pipeline
===============================================================
Stage 1 · Convert   webp / png / jpeg / heic  →  jpg
Stage 2 · Compress  Optimize every .jpg (quality=85, progressive)
Stage 3 · Dedup     SHA-256 exact  +  perceptual-hash near-match
                    +  face-embedding burst-dedup (cosine ≥ 0.93)
Stage 4 · FaceSort  ArcFace recognition against reference Faces/ folder
                    → matched persons   →  <src>/<PersonName>/
                    → group photos (2+ faces) →  <src>/Group/
                    → unmatched         →  remain in <src>/ for Stage 5
Stage 5 · Organize  CLIP + ArcFace + background colour + date-modified
                    → agglomerative clustering + sequential rename
                    {prefix}_{series:03d}_{seq:03d}{ext}

Performance notes
─────────────────
• InsightFace buffalo_l loaded ONCE and reused across Stages 3, 4, 5.
• Single unified embedding cache (_workflow_cache.json) with schema:
      { abs_path: { mtime, face, clip, face_count } }
• Process priority lowered; ONNX/OMP thread count capped.
• ESC key aborts gracefully and saves cache.

Requirements
────────────
  pip install Pillow imagehash insightface onnxruntime-directml \\
              opencv-python numpy scikit-learn rich tqdm psutil pillow-heif
  # CLIP (optional, improves Stage 5):
  pip install clip-anytorch torch-directml torchvision

Usage
─────
  python unified_image_workflow.py                        # fully interactive
  python unified_image_workflow.py --src "D:\\Photos" --faces "D:\\Refs"
  python unified_image_workflow.py --execute --recursive
"""

from __future__ import annotations

# ── Thread / priority caps — BEFORE any heavy import ─────────────────────────
import os
_OMP_THREADS = 4
os.environ.setdefault("OMP_NUM_THREADS",           str(_OMP_THREADS))
os.environ.setdefault("MKL_NUM_THREADS",           str(_OMP_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS",      str(_OMP_THREADS))
os.environ.setdefault("ONNXRUNTIME_THREAD_COUNT",  str(_OMP_THREADS))

try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HAS_HEIC = True
except ImportError:
    _HAS_HEIC = False

try:
    import imagehash as _imagehash_mod
    _HAS_IMAGEHASH = True
except ImportError:
    _HAS_IMAGEHASH = False

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _cv2 = None
    _HAS_CV2 = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    from insightface.app import FaceAnalysis as _FaceAnalysis
    _HAS_INSIGHTFACE = True
except ImportError:
    _FaceAnalysis = None
    _HAS_INSIGHTFACE = False

try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _torch = None
    _HAS_TORCH = False

try:
    import clip as _clip_module
    _HAS_CLIP = True
except ImportError:
    _clip_module = None
    _HAS_CLIP = False

try:
    from sklearn.cluster import AgglomerativeClustering as _AgglomerativeClustering
    _HAS_SKLEARN = True
except ImportError:
    _AgglomerativeClustering = None
    _HAS_SKLEARN = False

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 80_000_000

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()

# ── Constants ─────────────────────────────────────────────────────────────────
_IMAGE_EXTS       = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}
_CONVERT_EXTS     = {".webp", ".png", ".jpeg"}
if _HAS_HEIC:
    _CONVERT_EXTS.add(".heic")
_DEFAULT_MODELS   = r"C:\Users\Sunil\Downloads\Quick Share"
_MAX_SIDE         = 1280          # cap long edge before inference
_FACE_THRESHOLD   = 0.40          # ArcFace cosine sim for face sort
_FACE_DUP_THRESH  = 0.93          # burst-dedup threshold (Stage 3)
_MIN_DET_SCORE    = 0.65          # minimum face detection confidence
_MIN_FACE_RATIO   = 0.005         # face bbox ≥ 0.5% of image area
_MIN_SEC_RATIO    = 0.18          # secondary face ≥ 18% of primary
_PHASH_THRESHOLD  = 8             # Hamming distance for perceptual dup
_CACHE_FILE       = "_workflow_cache.json"
_DUP_FOLDER       = "Duplicate"

# ── Shared utilities: TerminationManager (ESC) + dedup primitives ─────────────
import sys as _sys_tmp
_scripts_dir = str(Path(__file__).parent.parent / "src" / "scripts")
_new_dir     = str(Path(__file__).parent)
for _d in (_scripts_dir, _new_dir):
    if _d not in _sys_tmp.path:
        _sys_tmp.path.insert(0, _d)

from common_utils import TerminationManager as _TerminationManager
_tm = _TerminationManager()
_tm.start_monitoring()

# Import pure dedup primitives from find_duplicates.py (same directory) to
# avoid ~100 lines of duplicated union-find + hashing code.
try:
    from find_duplicates import _UF, _sha256, _dup_dest, _uf_to_groups
except Exception:
    class _UF:                                                 # type: ignore[no-redef]
        def __init__(self, n):
            self.parent = list(range(n)); self.rank = [0] * n
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
            return x
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry: return
            if self.rank[rx] < self.rank[ry]: rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]: self.rank[rx] += 1
    def _sha256(p):                                            # type: ignore[no-redef]
        try:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
            return h.hexdigest()
        except OSError: return None
    def _dup_dest(dup_dir, p):                                 # type: ignore[no-redef]
        stem, suf = p.stem, p.suffix
        c = dup_dir / f"{stem}_dup{suf}"; i = 1
        while c.exists(): c = dup_dir / f"{stem}_dup{i}{suf}"; i += 1
        return c
    def _uf_to_groups(uf, n, images):                         # type: ignore[no-redef]
        clusters: Dict[int, List[Path]] = defaultdict(list)
        for i in range(n): clusters[uf.find(i)].append(images[i])
        return [v for v in clusters.values() if len(v) > 1]

del _sys_tmp, _scripts_dir, _new_dir, _d


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def sanitize_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]", "", name)
    return name.lstrip(".") or "image"


def unique_dest(dest_dir: Path, base_name: str) -> Path:
    base_name = sanitize_filename(base_name)
    stem, suffix = Path(base_name).stem, Path(base_name).suffix
    candidate = dest_dir / (stem + suffix)
    i = 1
    while candidate.exists():
        candidate = dest_dir / f"{stem}({i}){suffix}"
        i += 1
    return candidate


def safe_rename(src: Path, dest: Path) -> None:
    if src.resolve() == dest.resolve():
        return
    try:
        os.replace(src, dest)
    except Exception:
        with src.open("rb") as rf, open(dest, "xb") as wf:
            shutil.copyfileobj(rf, wf, length=1 << 20)
        src.unlink()


def resize_for_inference(img_bgr, max_side: int = _MAX_SIDE):
    """Cap longer edge at max_side using INTER_AREA (no accuracy loss for ArcFace)."""
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img_bgr
    scale = max_side / longest
    return _cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                       interpolation=_cv2.INTER_AREA)


def collect_images(
    src: Path,
    recursive: bool,
    exclude_dirs: Optional[set] = None,
) -> List[Path]:
    """
    Collect image files from src.
    exclude_dirs: set of folder *names* (not full paths) to skip entirely.
    """
    if exclude_dirs is None:
        exclude_dirs = set()
    results: List[Path] = []
    if recursive:
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for fname in files:
                p = Path(root) / fname
                if p.suffix.lower() in _IMAGE_EXTS:
                    results.append(p)
    else:
        for p in src.iterdir():
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                results.append(p)
    return sorted(results, key=lambda p: (p.stat().st_mtime, p.name.lower()))


def l2_normalize(vec) -> Optional[object]:
    if vec is None or not _HAS_NUMPY:
        return None
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-12 else None


def cosine_sim(a, b) -> float:
    if not _HAS_NUMPY:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED EMBEDDING CACHE
#  Schema: { abs_path_str: { mtime, face, clip, face_count } }
# ══════════════════════════════════════════════════════════════════════════════

def cache_load(src: Path) -> dict:
    path = src / _CACHE_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def cache_save(cache: dict, src: Path) -> None:
    try:
        (src / _CACHE_FILE).write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass


def cache_get(cache: dict, p: Path) -> Tuple[Optional[object], Optional[object], int]:
    """Returns (face_emb, clip_emb, face_count). All None/0 on miss."""
    key   = str(p.resolve())
    entry = cache.get(key)
    if not entry or abs(entry.get("mtime", 0) - p.stat().st_mtime) >= 1.0:
        return None, None, 0
    face = np.array(entry["face"], dtype=np.float32) if entry.get("face") else None
    clip = np.array(entry["clip"], dtype=np.float32) if entry.get("clip") else None
    return face, clip, int(entry.get("face_count", 0))


def cache_set(
    cache: dict,
    p: Path,
    face_emb: Optional[object],
    clip_emb: Optional[object],
    face_count: int,
) -> None:
    cache[str(p.resolve())] = {
        "mtime":      p.stat().st_mtime,
        "face":       face_emb.tolist() if face_emb is not None else None,
        "clip":       clip_emb.tolist() if clip_emb is not None else None,
        "face_count": face_count,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ONNX / InsightFace loader
# ══════════════════════════════════════════════════════════════════════════════

def load_face_app(models_root: str, allowed_modules=None):
    """Load InsightFace buffalo_l once; reused by Stages 3, 4, 5."""
    if not _HAS_INSIGHTFACE:
        console.print("[yellow]⚠  insightface not installed — face features disabled[/yellow]")
        return None

    import onnxruntime as ort
    ort.set_default_logger_severity(3)

    avail = ort.get_available_providers()
    providers = [ep for ep in ("DmlExecutionProvider", "CPUExecutionProvider")
                 if ep in avail] or ["CPUExecutionProvider"]
    ep_label = providers[0]

    if ep_label == "CPUExecutionProvider":
        console.print(
            f"  [yellow]⚠  DirectML unavailable — CPU mode "
            f"(threads capped={_OMP_THREADS})[/yellow]\n"
            f"  [dim]  Fix: pip uninstall onnxruntime -y  &&  "
            f"pip install onnxruntime-directml[/dim]"
        )
    else:
        console.print(f"  [green]✓[/green] [dim]ONNX: {ep_label}  "
                      f"(threads={_OMP_THREADS})[/dim]")

    if allowed_modules is None:
        allowed_modules = ["detection", "recognition"]

    app = _FaceAnalysis(
        name="buffalo_l",
        root=models_root,
        providers=providers,
        allowed_modules=allowed_modules,
    )
    app.prepare(
        ctx_id=0 if ep_label != "CPUExecutionProvider" else -1,
        det_size=(640, 640),
    )
    return app


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 · CONVERT  (webp / png / jpeg / heic  →  jpg)
# ══════════════════════════════════════════════════════════════════════════════

def _is_webp_animated(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            h = f.read(12)
            if len(h) < 12 or h[:4] != b"RIFF" or h[8:12] != b"WEBP":
                return False
            while True:
                ch = f.read(8)
                if len(ch) < 8:
                    break
                fourcc, size = ch[:4], int.from_bytes(ch[4:], "little")
                if fourcc == b"VP8X":
                    d = f.read(size)
                    if len(d) >= 1 and d[0] & 0x02:
                        return True
                elif fourcc == b"ANIM":
                    return True
                else:
                    f.seek(size + (size % 2), os.SEEK_CUR)
        return False
    except Exception:
        return False


def _is_heic_animated(path: str) -> bool:
    if not _HAS_HEIC:
        return False
    try:
        with Image.open(path) as im:
            return getattr(im, "is_animated", False) or getattr(im, "n_frames", 1) > 1
    except Exception:
        return False


def _should_convert(path: str) -> bool:
    lo = path.lower()
    if lo.endswith((".jpg", ".gif")):
        return False
    if lo.endswith(".webp") and _is_webp_animated(path):
        return False
    if lo.endswith(".heic") and _is_heic_animated(path):
        return False
    return any(lo.endswith(ext) for ext in _CONVERT_EXTS)


def _convert_one(file_path: str) -> Tuple[bool, str]:
    """Convert a single image to JPG. Returns (ok, err_msg)."""
    try:
        if not os.path.isfile(file_path) or os.path.getsize(file_path) <= 0:
            return False, "empty or missing"
        # Verify integrity first
        with Image.open(file_path) as im:
            try:
                im.verify()
            except Exception as e:
                return False, f"corrupt: {e}"
        # Re-open after verify (Pillow requirement)
        with Image.open(file_path) as im:
            exif = im.info.get("exif")
            icc  = im.info.get("icc_profile")
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                bg   = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[3])
                rgb = bg
            else:
                rgb = im.convert("RGB")
            out = Path(file_path).with_suffix(".jpg")
            if out.resolve() == Path(file_path).resolve():
                return False, "same path"
            kw: dict = {"quality": 85, "optimize": True, "progressive": True, "subsampling": 2}
            if exif:
                kw["exif"] = exif
            if icc:
                kw["icc_profile"] = icc
            rgb.save(str(out), "JPEG", **kw)
        os.remove(file_path)
        return True, ""
    except Exception as e:
        return False, str(e)


def run_conversion(src: Path) -> None:
    console.print()
    console.rule("[bold cyan]Stage 1 / Convert[/bold cyan]")

    candidates = [
        f for f in src.rglob("*")
        if f.is_file() and _should_convert(str(f))
    ]
    if not candidates:
        console.print("  [dim]No convertible images found.[/dim]")
        return

    converted = failed = 0
    orig_total = sum(f.stat().st_size for f in candidates if f.exists())
    conv_total = 0
    t0 = time.time()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(bar_width=36), MofNCompleteColumn(),
                  TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("[cyan]Converting…[/cyan]", total=len(candidates))
        for f in candidates:
            ok, _ = _convert_one(str(f))
            if ok:
                converted += 1
                jpg = f.with_suffix(".jpg")
                if jpg.exists():
                    conv_total += jpg.stat().st_size
            else:
                failed += 1
                conv_total += f.stat().st_size if f.exists() else 0
            prog.advance(task)

    saved = orig_total - conv_total
    console.print(
        f"  [green]✓[/green]  {converted} converted  ·  {failed} failed  ·  "
        f"saved {format_size(saved)} ({saved/max(orig_total,1)*100:.1f}%)  "
        f"({time.time()-t0:.1f}s)"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 · COMPRESS  (optimize existing JPGs in-place)
# ══════════════════════════════════════════════════════════════════════════════

def _compress_one(path: Path) -> Tuple[bool, bool]:
    """Returns (ok, reduced). ok=False on error; reduced=True if file shrank."""
    tmp = Path(str(path) + ".tmp")
    try:
        orig = path.stat().st_size
        with Image.open(path) as im:
            exif = im.info.get("exif")
            icc  = im.info.get("icc_profile")
            rgb  = im.convert("RGB")
            kw: dict = {"quality": 85, "optimize": True, "progressive": True, "subsampling": 2}
            if exif:
                kw["exif"] = exif
            if icc:
                kw["icc_profile"] = icc
            rgb.save(str(tmp), "JPEG", **kw)
        if tmp.stat().st_size < orig:
            os.replace(str(tmp), str(path))
            return True, True
        tmp.unlink(missing_ok=True)
        return True, False
    except Exception:
        tmp.unlink(missing_ok=True)
        return False, False


def run_compression(src: Path) -> None:
    console.print()
    console.rule("[bold cyan]Stage 2 / Compress[/bold cyan]")

    jpgs = sorted(src.rglob("*.jpg"))
    if not jpgs:
        console.print("  [dim]No .jpg files found.[/dim]")
        return

    compressed = skipped = failed = 0
    saved_bytes = 0
    orig_total  = sum(f.stat().st_size for f in jpgs if f.exists())
    t0 = time.time()

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(bar_width=36), MofNCompleteColumn(),
                  TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("[cyan]Compressing…[/cyan]", total=len(jpgs))
        for f in jpgs:
            orig_sz = f.stat().st_size if f.exists() else 0
            ok, reduced = _compress_one(f)
            if not ok:
                failed += 1
            elif reduced:
                compressed += 1
                saved_bytes += orig_sz - (f.stat().st_size if f.exists() else 0)
            else:
                skipped += 1
            prog.advance(task)

    console.print(
        f"  [green]✓[/green]  {compressed} compressed  ·  {skipped} already optimal  ·  "
        f"{failed} failed  ·  saved {format_size(saved_bytes)} "
        f"({saved_bytes/max(orig_total,1)*100:.1f}%)  ({time.time()-t0:.1f}s)"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 · DEDUP  (SHA-256 + perceptual hash + face-embedding burst dedup)
# ══════════════════════════════════════════════════════════════════════════════
# _UF, _sha256, _dup_dest, _uf_to_groups imported from find_duplicates above.

def _phash(p: Path) -> Optional[object]:
    if not _HAS_IMAGEHASH:
        return None
    try:
        with Image.open(p) as im:
            return _imagehash_mod.phash(im.convert("RGB"))
    except Exception:
        return None


def _image_quality(p: Path) -> Tuple[int, int]:
    """Returns (pixels, filesize) for primary selection (max pixels, min size)."""
    try:
        with Image.open(p) as im:
            w, h = im.size
        return w * h, p.stat().st_size
    except Exception:
        return 0, p.stat().st_size


def _select_primary(group: List[Path]) -> Path:
    return min(group, key=lambda p: (-_image_quality(p)[0], _image_quality(p)[1]))


def run_dedup(
    src: Path,
    images: List[Path],
    face_app,
    cache: dict,
    execute: bool,
    phash_threshold: int = _PHASH_THRESHOLD,
    face_dup_threshold: float = _FACE_DUP_THRESH,
) -> List[Path]:
    """
    Find and move duplicate images.  Returns list of paths that remain (non-duplicates).
    """
    console.print()
    console.rule("[bold cyan]Stage 3 / Duplicate Detection[/bold cyan]")

    n = len(images)
    if n == 0:
        console.print("  [dim]No images to check.[/dim]")
        return images

    uf = _UF(n)

    # ── Pass 1: SHA-256 exact ────────────────────────────────────────────────
    console.print(Rule("Pass 1 · SHA-256 exact", style="dim"))
    sha_map: Dict[str, List[int]] = {}
    with Progress(SpinnerColumn(), TextColumn("[cyan]Hashing…[/cyan]"),
                  BarColumn(bar_width=36), MofNCompleteColumn(),
                  TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("", total=n)
        for i, p in enumerate(images):
            h = _sha256(p)
            if h:
                sha_map.setdefault(h, []).append(i)
            prog.advance(task)
    exact_groups = sum(1 for v in sha_map.values() if len(v) > 1)
    for idxs in sha_map.values():
        if len(idxs) > 1:
            for j in idxs[1:]:
                uf.union(idxs[0], j)
    console.print(f"  [green]✓[/green]  {exact_groups} exact-duplicate group(s)\n")

    # ── Pass 2: perceptual hash ──────────────────────────────────────────────
    if _HAS_IMAGEHASH:
        console.print(Rule("Pass 2 · Perceptual hash", style="dim"))
        phashes = [None] * n
        with Progress(SpinnerColumn(), TextColumn("[cyan]pHash…[/cyan]"),
                      BarColumn(bar_width=36), MofNCompleteColumn(),
                      TimeElapsedColumn(), console=console) as prog:
            task = prog.add_task("", total=n)
            for i, p in enumerate(images):
                phashes[i] = _phash(p)
                prog.advance(task)

        bucket: Dict[str, List[int]] = {}
        for i, h in enumerate(phashes):
            if h is not None:
                bucket.setdefault(str(h)[:2], []).append(i)

        near = 0
        with Progress(SpinnerColumn(), TextColumn("[cyan]Comparing…[/cyan]"),
                      BarColumn(bar_width=36), TaskProgressColumn(),
                      console=console) as prog:
            task = prog.add_task("", total=len(bucket))
            for idxs in bucket.values():
                for a in range(len(idxs)):
                    for b in range(a + 1, len(idxs)):
                        ia, ib = idxs[a], idxs[b]
                        ha, hb = phashes[ia], phashes[ib]
                        if ha is None or hb is None:
                            continue
                        if uf.find(ia) == uf.find(ib):
                            continue
                        if (ha - hb) <= phash_threshold:
                            uf.union(ia, ib)
                            near += 1
                prog.advance(task)
        console.print(f"  [green]✓[/green]  {near} near-duplicate pair(s) merged\n")

    # ── Pass 3: face-embedding burst dedup ──────────────────────────────────
    if face_app is not None and _HAS_NUMPY and _HAS_CV2:
        console.print(Rule("Pass 3 · Face-embedding burst dedup", style="dim"))
        console.print(f"  [dim]threshold: cosine ≥ {face_dup_threshold}[/dim]\n")

        face_embs: List[Optional[object]] = [None] * n
        with Progress(SpinnerColumn(), TextColumn("[cyan]Face embeddings…[/cyan]"),
                      BarColumn(bar_width=36), MofNCompleteColumn(),
                      TimeElapsedColumn(), console=console) as prog:
            task = prog.add_task("", total=n)
            for i, p in enumerate(images):
                # Check cache first
                fe, _, _ = cache_get(cache, p)
                if fe is None:
                    try:
                        img = _cv2.imread(str(p))
                        if img is not None:
                            img = resize_for_inference(img)
                            faces = face_app.get(img)
                            if faces:
                                f = max(faces, key=lambda f:
                                        (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                                fe = l2_normalize(np.asarray(f.embedding, np.float32))
                    except Exception:
                        pass
                face_embs[i] = fe
                prog.advance(task)

        # Find near-identical face pairs (very strict threshold)
        face_near = 0
        emb_indices = [i for i in range(n) if face_embs[i] is not None]
        emb_arr = np.stack([face_embs[i] for i in emb_indices]) if emb_indices else None

        if emb_arr is not None and len(emb_indices) > 1:
            sim_matrix = emb_arr @ emb_arr.T
            for row in range(len(emb_indices)):
                for col in range(row + 1, len(emb_indices)):
                    ia, ib = emb_indices[row], emb_indices[col]
                    if uf.find(ia) == uf.find(ib):
                        continue
                    if float(sim_matrix[row, col]) >= face_dup_threshold:
                        uf.union(ia, ib)
                        face_near += 1
        console.print(f"  [green]✓[/green]  {face_near} face-duplicate pair(s) merged\n")

    # ── Build groups ─────────────────────────────────────────────────────────
    groups = _uf_to_groups(uf, n, images)
    if not groups:
        console.print(Panel.fit(
            Text("No duplicates found.", style="bold green"),
            border_style="green",
        ))
        return images

    total_dups  = sum(len(g) - 1 for g in groups)
    total_bytes = sum(
        p.stat().st_size
        for g in groups
        for p in g
        if p != _select_primary(g) and p.exists()
    )

    # Summary table
    tbl = Table(show_lines=False, header_style="bold cyan", min_width=72)
    tbl.add_column("Primary (keep)",   style="green",   max_width=40)
    tbl.add_column("Copies",           justify="right", style="bold white")
    tbl.add_column("Size (primary)",   justify="right", style="dim")
    for g in sorted(groups, key=lambda g: -len(g)):
        primary = _select_primary(g)
        sz = primary.stat().st_size if primary.exists() else 0
        tbl.add_row(primary.name, str(len(g) - 1), f"{sz/1024:.1f} KB")
    console.print(tbl)
    console.print(
        f"\n  [bold cyan]{len(groups)}[/bold cyan] group(s)  ·  "
        f"[bold]{total_dups}[/bold] duplicate(s)  ·  "
        f"[dim]{total_bytes/1024/1024:.1f} MB recoverable[/dim]\n"
    )

    if not execute:
        if not Confirm.ask("  Move duplicates now?", default=False):
            return images

    # ── Move ─────────────────────────────────────────────────────────────────
    dup_dir = src / _DUP_FOLDER
    dup_dir.mkdir(exist_ok=True)
    moved = errors = 0
    dup_set: set = set()

    with Progress(SpinnerColumn(), TextColumn("[cyan]Moving duplicates…[/cyan]"),
                  BarColumn(bar_width=36), TaskProgressColumn(),
                  console=console) as prog:
        task = prog.add_task("", total=total_dups)
        for g in groups:
            primary = _select_primary(g)
            for dup in g:
                if dup == primary:
                    continue
                dest = _dup_dest(dup_dir, dup)
                try:
                    shutil.move(str(dup), str(dest))
                    dup_set.add(str(dup.resolve()))
                    moved += 1
                except Exception as exc:
                    console.print(f"  [red]ERR[/red]  {dup.name}: {exc}")
                    errors += 1
                prog.advance(task)

    console.print(
        f"\n  [green]✓[/green]  {moved} duplicates moved to "
        f"[cyan]{_DUP_FOLDER}/[/cyan]  ·  {errors} errors\n"
    )
    return [p for p in images if str(p.resolve()) not in dup_set]


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 · FACE SORT  (ArcFace match against reference Faces/ folder)
# ══════════════════════════════════════════════════════════════════════════════

def _load_references(face_app, faces_folder: Path, cache: dict) -> Dict[str, List]:
    """Load reference embeddings per person from the Faces/ folder."""
    refs: Dict[str, List] = defaultdict(list)
    ref_images = sorted(
        p for p in faces_folder.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    if not ref_images:
        console.print(f"[red]  ✗  No reference images in: {faces_folder}[/red]")
        return {}

    tbl = Table(title="Reference faces", show_lines=False, header_style="bold cyan")
    tbl.add_column("File",   style="white")
    tbl.add_column("Person", style="green")
    tbl.add_column("Status", justify="center")

    for img_path in ref_images:
        person = re.sub(r"_\d+$", "", img_path.stem)
        fe, _, _ = cache_get(cache, img_path)
        if fe is None:
            try:
                img = _cv2.imread(str(img_path))
                if img is None:
                    tbl.add_row(img_path.name, person, "[red]can't read[/red]")
                    continue
                faces = face_app.get(img)
                if not faces:
                    tbl.add_row(img_path.name, person, "[red]no face[/red]")
                    continue
                f = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                fe = l2_normalize(np.asarray(f.embedding, np.float32))
                cache_set(cache, img_path, fe, None, 1)
            except Exception as e:
                tbl.add_row(img_path.name, person, f"[red]{e}[/red]")
                continue
        refs[person].append(fe)
        tbl.add_row(img_path.name, person, "[green]✓[/green]")

    console.print(tbl)
    return dict(refs)


def _match_face(
    img_path: Path,
    face_app,
    refs: Dict[str, List],
    threshold: float,
    cache: dict,
) -> Tuple[List[dict], int]:
    """
    Returns (confirmed: list[dict], face_count: int).

    confirmed = every person matched above threshold, each entry:
        { person, score, is_female, bbox }
    face_count = number of significant female faces detected.

    Group rule (female-priority):
        len([c for c in confirmed if c["is_female"]]) >= 2  →  Group
        len([...]) == 1  →  that female's folder
        0 females but ≥1 male confirmed  →  best male's folder
        0 confirmed  →  stays in src (Stage 5)

    Cache note: only the primary (female-priority) face embedding is stored.
    On cache-hit, at most ONE match is reconstructable; full multi-person
    detection only occurs on the first inference pass.
    """
    def _is_female(f) -> bool:
        return getattr(f, "sex", None) == "F"

    # ── Cache hit: single embedding — limited to one match ───────────────
    fe, _, fc = cache_get(cache, img_path)
    if fe is not None:
        best_person, best_score = None, -1.0
        for person, embs in refs.items():
            s = max(cosine_sim(fe, e) for e in embs)
            if s > best_score:
                best_score, best_person = s, person
        if best_person and best_score >= threshold:
            # Cache stores female-priority embedding; fc ≥ 1 means a female
            # face was the primary at inference time.
            is_female_cached = fc >= 1
            return [{"person": best_person, "score": round(best_score, 4),
                     "is_female": is_female_cached, "bbox": [0, 0, 0, 0]}], fc
        return [], fc

    # ── Full inference ────────────────────────────────────────────────────
    try:
        img = _cv2.imread(str(img_path))
        if img is None:
            return [], 0
        img_small = resize_for_inference(img)
        h, w = img_small.shape[:2]
        img_area = h * w
        faces = face_app.get(img_small)
        if not faces:
            return [], 0

        def _area(f) -> float:
            return max(0.0, float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))

        valid = [f for f in faces
                 if f.det_score >= _MIN_DET_SCORE
                 and _area(f) / img_area >= _MIN_FACE_RATIO]
        if not valid:
            return [], 0

        primary_area = max(_area(f) for f in valid)
        significant  = [f for f in valid
                        if _area(f) >= primary_area * _MIN_SEC_RATIO]

        # face_count = significant female faces (informational; drives cache tag)
        face_count = sum(1 for f in significant if _is_female(f))

        # ── Per-face recognition with seen_persons guard ─────────────────
        # Process females first so the female-priority embedding gets cached.
        confirmed: List[dict] = []
        seen_persons: set     = set()
        sorted_faces = sorted(significant,
                              key=lambda f: (not _is_female(f), -_area(f)))

        for face in sorted_faces:
            best_person, best_score = None, -1.0
            for person, embs in refs.items():
                if person in seen_persons:
                    continue                          # one match per person
                s = max(cosine_sim(face.embedding, e) for e in embs)
                if s > best_score:
                    best_score, best_person = s, person
            if best_person and best_score >= threshold:
                confirmed.append({
                    "person":    best_person,
                    "score":     round(best_score, 4),
                    "is_female": _is_female(face),
                    "bbox":      [int(x) for x in face.bbox],
                })
                seen_persons.add(best_person)

        # ── Cache: store female-priority face embedding ───────────────────
        female_sig = [f for f in significant if _is_female(f)]
        cache_face = (max(female_sig, key=_area) if female_sig
                      else max(significant, key=_area))
        fe_store = l2_normalize(np.asarray(cache_face.embedding, np.float32))
        cache_set(cache, img_path, fe_store, None, face_count)

        return confirmed, face_count

    except Exception:
        return [], 0


def run_face_sort(
    src: Path,
    images: List[Path],
    face_app,
    refs: Dict[str, List],
    cache: dict,
    threshold: float,
    execute: bool,
) -> List[Path]:
    """
    Move face-matched images to per-person subfolders and Group/.
    Returns list of remaining images (unmatched → stay in src for Stage 5).
    """
    console.print()
    console.rule("[bold cyan]Stage 4 / Face Sort[/bold cyan]")

    if face_app is None or not refs:
        console.print("  [yellow]⚠  Face app or references unavailable — skipping Stage 4.[/yellow]")
        return images

    matches: List[dict] = []
    t0 = time.time()

    console.print("  [dim](ESC to abort)[/dim]\n")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(bar_width=36), TaskProgressColumn(),
                  TimeElapsedColumn(), TimeRemainingColumn(),
                  console=console, refresh_per_second=6) as prog:
        task = prog.add_task("[cyan]Matching faces…[/cyan]", total=len(images))
        for img_path in images:
            if _tm.is_terminating():
                console.print("\n  [yellow]⚠  Aborted.[/yellow]")
                break
            confirmed, fc = _match_face(img_path, face_app, refs, threshold, cache)

            # ── Female-priority grouping (mirrors face_sorter.py) ────────
            # Group ONLY when 2+ recognized female faces present.
            #   • 1 female + any males (known/unknown)  → her folder
            #   • 1 female + blurred/unrecognized       → her folder
            #   • 2+ recognized females                 → Group
            #   • 0 females, ≥1 recognized male         → best male's folder
            #   • 0 recognized                          → stays in src (Stage 5)
            female_hits   = [c for c in confirmed if c["is_female"]]
            other_hits    = [c for c in confirmed if not c["is_female"]]
            n_rec_females = len(female_hits)

            if n_rec_females >= 2:
                dest_folder = "Group"
                best_score  = max(c["score"] for c in female_hits)
            elif n_rec_females == 1:
                dest_folder = female_hits[0]["person"]
                best_score  = female_hits[0]["score"]
            elif other_hits:
                best_male   = max(other_hits, key=lambda c: c["score"])
                dest_folder = best_male["person"]
                best_score  = best_male["score"]
            else:
                dest_folder = None
                best_score  = 0.0

            matches.append({"path": img_path, "folder": dest_folder,
                            "score": best_score, "face_count": fc})

            n_matched = sum(1 for m in matches if m["folder"] and m["folder"] != "Group")
            n_group   = sum(1 for m in matches if m["folder"] == "Group")
            n_other   = sum(1 for m in matches if m["folder"] is None)
            prog.update(task, description=(
                f"[cyan]Matching faces[/cyan]  "
                f"[green]{n_matched} matched[/green]  "
                f"[magenta]{n_group} group[/magenta]  "
                f"[dim]{n_other} others[/dim]"
            ))
            prog.advance(task)

    elapsed = time.time() - t0
    n_matched = sum(1 for m in matches if m["folder"] and m["folder"] != "Group")
    n_group   = sum(1 for m in matches if m["folder"] == "Group")
    n_other   = sum(1 for m in matches if m["folder"] is None)

    console.print(
        f"\n  [green]✓[/green]  {len(images)} images in {elapsed:.1f}s  ·  "
        f"[green]{n_matched}[/green] matched  "
        f"[magenta]{n_group}[/magenta] group  "
        f"[dim]{n_other}[/dim] unmatched\n"
    )

    # Summary table
    by_folder: Dict[str, List] = defaultdict(list)
    for m in matches:
        by_folder[m["folder"] or "Others (remain)"].append(m)

    tbl = Table(title="Sort preview", show_lines=False, header_style="bold cyan")
    tbl.add_column("Destination",  style="green")
    tbl.add_column("Count",        justify="right", style="bold white")
    tbl.add_column("Avg score",    justify="right", style="yellow")
    for folder, ms in sorted(by_folder.items()):
        scores = [m["score"] for m in ms]
        avg    = sum(scores) / len(scores) if scores else 0.0
        tbl.add_row(folder, str(len(ms)), f"{avg:.3f}")
    console.print(tbl)

    to_move = [m for m in matches if m["folder"] is not None]
    if not to_move:
        console.print("\n  [yellow]No images to move.[/yellow]")
        return images

    if not execute:
        if not Confirm.ask("\n  Move matched files now?", default=False):
            return images

    moved = errors = 0
    moved_paths: set = set()

    with Progress(SpinnerColumn(), TextColumn("[cyan]Moving…[/cyan]"),
                  BarColumn(bar_width=36), TaskProgressColumn(),
                  console=console) as prog:
        task = prog.add_task("", total=len(to_move))
        for m in to_move:
            dest_dir = src / m["folder"]
            dest_dir.mkdir(exist_ok=True)
            dest = dest_dir / m["path"].name
            if dest.exists():
                dest = dest_dir / (m["path"].stem + "_dup" + m["path"].suffix)
            try:
                shutil.move(str(m["path"]), str(dest))
                moved_paths.add(str(m["path"].resolve()))
                moved += 1
            except Exception as exc:
                console.print(f"  [red]ERR[/red]  {m['path'].name}: {exc}")
                errors += 1
            prog.advance(task)

    console.print(
        f"\n  [green]✓[/green]  {moved} moved  ·  {errors} errors\n"
    )
    # Return only images that were NOT moved (stay for Stage 5)
    return [p for p in images if str(p.resolve()) not in moved_paths]


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 · ORGANIZER  (CLIP + ArcFace + bg colour + date → cluster + rename)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _ImgRec:
    path:       Path
    mtime:      float
    face_emb:   Optional[object]
    clip_emb:   Optional[object]
    bg_hist:    Optional[object]
    face_count: int
    width:      int
    height:     int


def _bg_mask(img_bgr) -> object:
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:int(.08*h), :]  = 255
    mask[int(.92*h):, :]  = 255
    mask[:, :int(.08*w)]  = 255
    mask[:, int(.92*w):]  = 255
    return mask


def _bg_hist(img_bgr) -> object:
    hsv  = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2HSV)
    mask = _bg_mask(img_bgr)
    h    = _cv2.calcHist([hsv], [0, 1, 2], mask, [16, 8, 8],
                         [0, 180, 0, 256, 0, 256])
    h    = _cv2.normalize(h, h).flatten().astype(np.float32)
    s    = h.sum()
    if s > 0:
        h /= s
    return h


def _extract_one(
    p: Path,
    face_app,
    clip_model,
    clip_preprocess,
    clip_device,
    cache: dict,
) -> Optional[_ImgRec]:
    try:
        if not p.exists() or p.stat().st_size <= 0:
            return None
        pil     = Image.open(p).convert("RGB")
        w, h_px = pil.size
        mtime   = p.stat().st_mtime

        img_bgr = _cv2.cvtColor(np.array(pil), _cv2.COLOR_RGB2BGR)
        img_bgr = resize_for_inference(img_bgr)

        fe, ce, fc = cache_get(cache, p)
        cache_hit  = fe is not None or ce is not None

        if not cache_hit:
            # Face embedding
            if face_app is not None:
                try:
                    faces = face_app.get(img_bgr)
                    fc    = len(faces)
                    if faces:
                        lf = max(faces, key=lambda f:
                                 (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                        fe = l2_normalize(np.asarray(lf.embedding, np.float32))
                except Exception:
                    pass

            # CLIP embedding
            if clip_model is not None and _HAS_TORCH:
                try:
                    with _torch.no_grad():
                        inp = clip_preprocess(pil).unsqueeze(0).to(clip_device)
                        raw = clip_model.encode_image(inp)[0].detach().cpu().numpy().astype(np.float32)
                        ce  = l2_normalize(raw)
                except Exception:
                    pass

            cache_set(cache, p, fe, ce, fc)

        bg = _bg_hist(img_bgr) if _HAS_CV2 and _HAS_NUMPY else None

        return _ImgRec(
            path=p, mtime=mtime, face_emb=fe, clip_emb=ce,
            bg_hist=bg, face_count=fc, width=w, height=h_px,
        )
    except Exception:
        return None


def _build_distance_matrix(
    records: List[_ImgRec],
    w_face=0.40, w_clip=0.28, w_bg=0.17, w_time=0.15,
    time_half_life_hours=72.0,
    face_bonus=0.08,
) -> object:
    n = len(records)
    MISS_FACE, MISS_CLIP, MISS_BG = 0.68, 0.60, 0.70

    # Face distance
    face_has = np.array([r.face_emb is not None for r in records])
    if face_has.any():
        F = np.zeros((n, 512), dtype=np.float32)
        for i, r in enumerate(records):
            if r.face_emb is not None:
                F[i] = r.face_emb
        sim_f = np.clip(F @ F.T, -1.0, 1.0)
        D_f   = (1.0 - sim_f) / 2.0
        no_f  = ~face_has
        D_f[no_f, :] = MISS_FACE
        D_f[:, no_f] = MISS_FACE
        np.fill_diagonal(D_f, 0.0)
    else:
        D_f = np.full((n, n), MISS_FACE, np.float32)
        np.fill_diagonal(D_f, 0.0)

    # CLIP distance
    clip_has = np.array([r.clip_emb is not None for r in records])
    if clip_has.any():
        cdim = next(r.clip_emb.shape[0] for r in records if r.clip_emb is not None)
        C    = np.zeros((n, cdim), dtype=np.float32)
        for i, r in enumerate(records):
            if r.clip_emb is not None:
                C[i] = r.clip_emb
        sim_c = np.clip(C @ C.T, -1.0, 1.0)
        D_c   = (1.0 - sim_c) / 2.0
        no_c  = ~clip_has
        D_c[no_c, :] = MISS_CLIP
        D_c[:, no_c] = MISS_CLIP
        np.fill_diagonal(D_c, 0.0)
    else:
        D_c = np.full((n, n), MISS_CLIP, np.float32)
        np.fill_diagonal(D_c, 0.0)

    # Background chi-square (chunked to stay within RAM)
    _CHUNK = 128
    bg_has = np.array([r.bg_hist is not None for r in records])
    if bg_has.any():
        bdim = next(r.bg_hist.shape[0] for r in records if r.bg_hist is not None)
        BG   = np.zeros((n, bdim), dtype=np.float32)
        for i, r in enumerate(records):
            if r.bg_hist is not None:
                BG[i] = r.bg_hist
        D_b = np.zeros((n, n), dtype=np.float32)
        for rs in range(0, n, _CHUNK):
            re_  = min(rs + _CHUNK, n)
            chk  = BG[rs:re_, np.newaxis, :]
            full = BG[np.newaxis, :, :]
            denom = chk + full + 1e-10
            D_b[rs:re_, :] = np.clip(0.5 * np.sum((chk - full)**2 / denom, axis=2), 0.0, 1.0)
        no_b = ~bg_has
        D_b[no_b, :] = MISS_BG
        D_b[:, no_b] = MISS_BG
        np.fill_diagonal(D_b, 0.0)
    else:
        D_b = np.full((n, n), MISS_BG, np.float32)
        np.fill_diagonal(D_b, 0.0)

    # Time distance
    mt   = np.array([r.mtime for r in records], dtype=np.float64)
    dt   = np.abs(mt[:, None] - mt[None, :]) / 3600.0
    D_t  = np.clip(1.0 - np.exp(-dt / max(1e-6, time_half_life_hours)), 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(D_t, 0.0)

    D = (w_face * D_f + w_clip * D_c + w_bg * D_b + w_time * D_t).astype(np.float32)

    # Face presence bonus/penalty
    both_f = face_has[:, None] & face_has[None, :]
    one_f  = face_has[:, None] ^ face_has[None, :]
    D = np.where(both_f & (D_f < 0.22), np.clip(D - face_bonus, 0.0, 1.0), D)
    D = np.where(one_f, np.clip(D + 0.04, 0.0, 1.0), D)

    # Face-count penalty (solo vs group)
    fc  = np.array([r.face_count for r in records], dtype=np.float32)
    fcd = np.abs(fc[:, None] - fc[None, :])
    D_fc = np.where(fcd == 0, 0.0, np.where(fcd == 1, 0.20, 0.35)).astype(np.float32)
    np.fill_diagonal(D_fc, 0.0)
    D = np.clip(D + D_fc, 0.0, 1.0)

    np.fill_diagonal(D, 0.0)
    return D.astype(np.float32)


def _order_cluster(idxs: List[int], records: List[_ImgRec], D) -> List[int]:
    if len(idxs) <= 2:
        return sorted(idxs, key=lambda i: records[i].mtime)
    ordered = sorted(idxs, key=lambda i: records[i].mtime)
    used, seq = {ordered[0]}, [ordered[0]]
    while len(seq) < len(idxs):
        last = seq[-1]
        rem  = [i for i in idxs if i not in used]
        def score(i):
            tg = abs(records[i].mtime - records[last].mtime)
            return D[last, i] + min(tg / (3600.0 * 24.0 * 30.0), 1.0) * 0.05
        nxt = min(rem, key=score)
        seq.append(nxt)
        used.add(nxt)
    return seq


def _detect_series(images: List[Path], prefix: str) -> int:
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)_(\d+)$")
    return max((int(m.group(1)) for p in images for m in [pat.match(p.stem)] if m), default=0)


def _filter_renamed(images: List[Path], prefix: str) -> Tuple[List[Path], int]:
    pat     = re.compile(rf"^{re.escape(prefix)}_(\d+)_(\d+)$")
    fresh   = [p for p in images if not pat.match(p.stem)]
    skipped = len(images) - len(fresh)
    return fresh, skipped


def _rename_clusters(clusters: List[List[Path]], prefix: str,
                     continue_series: bool, all_images: List[Path]) -> None:
    prefix    = sanitize_filename(prefix)
    start_idx = (_detect_series(all_images, prefix) + 1) if continue_series else 1

    # Two-phase rename: tmp → final (avoids name collisions in large clusters)
    tmp_moves: List[Tuple[Path, Path]] = []
    for cluster in clusters:
        for p in cluster:
            tmp = unique_dest(p.parent, f"__tmp__{p.stem}{p.suffix}")
            safe_rename(p, tmp)
            tmp_moves.append((p, tmp))
    tmp_map = {orig: tmp for orig, tmp in tmp_moves}

    si = start_idx
    for cluster in clusters:
        for ii, orig in enumerate(cluster, 1):
            tmp   = tmp_map[orig]
            ext   = orig.suffix.lower()
            dest  = unique_dest(orig.parent, f"{prefix}_{si:03d}_{ii:03d}{ext}")
            safe_rename(tmp, dest)
        si += 1


def run_organizer(
    src: Path,
    images: List[Path],
    face_app,
    cache: dict,
    models_root: str,
    prefix: str,
    cluster_threshold: float,
    time_half_life_hours: float,
    w_face: float, w_clip: float, w_bg: float, w_time: float,
    execute: bool,
    force: bool = False,
) -> None:
    console.print()
    console.rule("[bold cyan]Stage 5 / Organize & Rename[/bold cyan]")

    if not images:
        console.print("  [dim]No images remaining to organize.[/dim]")
        return
    if not _HAS_NUMPY or not _HAS_CV2:
        console.print("  [yellow]⚠  numpy / opencv not available — skipping Stage 5.[/yellow]")
        return
    if _AgglomerativeClustering is None:
        console.print("  [yellow]⚠  scikit-learn not installed — skipping Stage 5.[/yellow]")
        return

    prefix = sanitize_filename(prefix)
    all_existing = list(images)

    if not force:
        images, skipped = _filter_renamed(images, prefix)
        if skipped:
            console.print(f"  [dim]Skipped {skipped} already-renamed files.[/dim]")
    if not images:
        console.print("  [yellow]All images already renamed. Use --force to re-cluster.[/yellow]")
        return

    # Load CLIP once
    clip_model = clip_preprocess = None
    clip_device = "cpu"
    if _HAS_CLIP and _HAS_TORCH:
        try:
            # Try DirectML for Intel Arc, fallback to CUDA then CPU
            try:
                import torch_directml
                clip_device = torch_directml.device()
            except ImportError:
                if _torch.cuda.is_available():
                    clip_device = "cuda"
            clip_model, clip_preprocess = _clip_module.load("ViT-B/32", device=clip_device)
            console.print(f"  [green]✓[/green] CLIP loaded (device={clip_device})")
        except Exception as e:
            console.print(f"  [yellow]⚠  CLIP unavailable: {e}[/yellow]")

    # ── Extract features ─────────────────────────────────────────────────────
    console.print(f"\n  Extracting features from {len(images)} images…")
    console.print("  [dim](ESC to abort)[/dim]")

    records: List[_ImgRec] = []
    failed: List[str] = []

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
                  console=console) as prog:
        task = prog.add_task("Extracting…", total=len(images))
        for p in images:
            if _tm.is_terminating():
                console.print("\n  [yellow]⚠  Aborted.[/yellow]")
                break
            rec = _extract_one(p, face_app, clip_model, clip_preprocess, clip_device, cache)
            if rec:
                records.append(rec)
            else:
                failed.append(p.name)
            prog.advance(task)

    cache_save(cache, src)   # persist updated embeddings

    console.print(
        f"\n  Extracted [bold]{len(records)}[/bold] images"
        + (f"  ({len(failed)} failed)" if failed else "")
    )
    if not records:
        console.print("[yellow]  No valid images after feature extraction.[/yellow]")
        return

    # ── Cluster ──────────────────────────────────────────────────────────────
    console.print()
    console.rule("[dim]Clustering[/dim]")

    if len(records) == 1:
        clustered_paths = [[records[0].path]]
    else:
        total_w = w_face + w_clip + w_bg + w_time
        wf, wc, wb, wt = w_face/total_w, w_clip/total_w, w_bg/total_w, w_time/total_w
        console.print(
            f"  Weights — face:{wf:.2f}  clip:{wc:.2f}  bg:{wb:.2f}  "
            f"time:{wt:.2f}  threshold:{cluster_threshold}"
        )
        console.print(f"  [dim]Building {len(records)}×{len(records)} distance matrix…[/dim]")
        D = _build_distance_matrix(records, wf, wc, wb, wt, time_half_life_hours)

        labels = _AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=cluster_threshold,
        ).fit_predict(D)

        clusters: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(labels):
            clusters.setdefault(int(lbl), []).append(idx)

        clustered_paths = []
        for _, idxs in sorted(clusters.items(),
                               key=lambda kv: min(records[i].mtime for i in kv[1])):
            ordered = _order_cluster(idxs, records, D)
            clustered_paths.append([records[i].path for i in ordered])

    console.print(f"  Formed [bold]{len(clustered_paths)}[/bold] clusters")

    # Cluster preview
    start = (_detect_series(all_existing, prefix) + 1)
    tbl = Table(title="Cluster preview (top 25)", show_lines=False)
    tbl.add_column("Series", style="cyan", justify="right")
    tbl.add_column("Images", justify="right")
    tbl.add_column("Sample",  style="dim")
    for ci, cluster in enumerate(clustered_paths[:25]):
        tbl.add_row(f"{prefix}_{start+ci:03d}", str(len(cluster)), cluster[0].name)
    if len(clustered_paths) > 25:
        tbl.add_row("…", f"+{len(clustered_paths)-25}", "")
    console.print(tbl)

    # ── Rename ────────────────────────────────────────────────────────────────
    console.print()
    console.rule("[dim]Renaming[/dim]")

    if not execute:
        console.print("[bold yellow]DRY RUN[/bold yellow] — showing first 30 renames:\n")
        shown = 0
        for ci, cluster in enumerate(clustered_paths):
            for j, p in enumerate(cluster, 1):
                if shown >= 30:
                    break
                new_name = f"{prefix}_{start+ci:03d}_{j:03d}{p.suffix.lower()}"
                console.print(f"  {p.name}  →  [bold]{new_name}[/bold]")
                shown += 1
            if shown >= 30:
                break
        console.print()
        if not Confirm.ask("Apply renames now?", default=False):
            console.print("[dim]No files changed.[/dim]")
            return

    _rename_clusters(clustered_paths, prefix, continue_series=True, all_images=all_existing)
    console.print(
        f"\n  [green]✓[/green]  Renamed "
        f"[bold]{sum(len(c) for c in clustered_paths)}[/bold] files  "
        f"into [bold]{len(clustered_paths)}[/bold] series\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING & INTERACTIVE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="unified_image_workflow",
        description="5-stage image pipeline: convert → compress → dedup → face-sort → organize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--src",           type=Path,  default=None,
                    help="Source image folder (prompted if omitted)")
    ap.add_argument("--faces",         type=Path,  default=None,
                    help="Reference faces folder (default: <src>/Faces)")
    ap.add_argument("--models",        type=str,   default=_DEFAULT_MODELS,
                    help="Root folder containing models/buffalo_l/")
    ap.add_argument("--recursive",     action="store_true",
                    help="Scan subfolders recursively")
    ap.add_argument("--execute",       action="store_true",
                    help="Actually move/rename files (default: dry-run with prompts)")
    ap.add_argument("--prefix",        type=str,   default="IMG",
                    help="Rename prefix for Stage 5 (default: IMG)")
    ap.add_argument("--face-threshold",  type=float, default=_FACE_THRESHOLD,
                    help=f"Face sort similarity threshold (default {_FACE_THRESHOLD})")
    ap.add_argument("--phash-threshold", type=int,   default=_PHASH_THRESHOLD,
                    help=f"pHash Hamming threshold for dedup (default {_PHASH_THRESHOLD})")
    ap.add_argument("--cluster-threshold", type=float, default=0.48,
                    help="Organizer clustering distance threshold (default 0.48)")
    ap.add_argument("--time-half-life",    type=float, default=72.0,
                    help="Time-signal half-life in hours (default 72)")
    ap.add_argument("--weight-face", type=float, default=0.40)
    ap.add_argument("--weight-clip", type=float, default=0.28)
    ap.add_argument("--weight-bg",   type=float, default=0.17)
    ap.add_argument("--weight-time", type=float, default=0.15)
    ap.add_argument("--skip-convert",  action="store_true", help="Skip Stage 1")
    ap.add_argument("--skip-compress", action="store_true", help="Skip Stage 2")
    ap.add_argument("--skip-dedup",    action="store_true", help="Skip Stage 3")
    ap.add_argument("--skip-facesort", action="store_true", help="Skip Stage 4")
    ap.add_argument("--skip-organize", action="store_true", help="Skip Stage 5")
    ap.add_argument("--force",         action="store_true",
                    help="Re-cluster already-renamed files in Stage 5")
    return ap


def _interactive_setup(args: argparse.Namespace) -> argparse.Namespace:
    console.print()
    console.print(Panel.fit(
        Text.assemble(
            ("Unified Image Workflow", "bold cyan"),
            ("  ·  ", "dim"),
            ("Convert → Compress → Dedup → FaceSort → Organize", "dim"),
        ),
        border_style="cyan",
    ))
    console.print()

    if args.src is None:
        while True:
            raw = Prompt.ask("  [cyan]Source folder[/cyan]").strip().strip('"').strip("'")
            p   = Path(raw)
            if p.exists() and p.is_dir():
                args.src = p
                break
            console.print(f"    [red]✗  Not found:[/red] {p}")

    if args.faces is None:
        default_faces = args.src / "Faces"
        if default_faces.exists():
            args.faces = default_faces
            console.print(f"  [dim]Faces folder auto-detected: {args.faces}[/dim]")
        else:
            while True:
                raw = Prompt.ask(
                    "  [cyan]Reference faces folder[/cyan]  "
                    f"[dim](or press Enter to skip face sort)[/dim]",
                    default=""
                ).strip().strip('"').strip("'")
                if not raw:
                    args.faces = None
                    args.skip_facesort = True
                    break
                p = Path(raw)
                if p.exists() and p.is_dir():
                    args.faces = p
                    break
                console.print(f"    [red]✗  Not found:[/red] {p}")

    if not args.recursive:
        args.recursive = Confirm.ask(
            "\n  Scan [bold]subfolders[/bold] recursively?", default=False
        )

    console.print()
    console.print(Rule("Ready", style="cyan"))
    console.print(f"  Source    : {args.src}")
    console.print(f"  Faces     : {args.faces or '— (face sort skipped)'}")
    console.print(f"  Models    : {args.models}")
    console.print(f"  Recursive : {'yes' if args.recursive else 'no'}")
    console.print(f"  Mode      : [bold {'red' if args.execute else 'yellow'}]"
                  f"{'EXECUTE — files will be moved' if args.execute else 'DRY RUN — prompts before each move'}[/bold "
                  f"{'red' if args.execute else 'yellow'}]")
    console.print()

    if not Confirm.ask("  Proceed?", default=True):
        console.print("[yellow]  Aborted.[/yellow]")
        sys.exit(0)

    return args


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.src is None:
        args = _interactive_setup(args)
    else:
        if not args.src.exists() or not args.src.is_dir():
            console.print(f"[red]Source folder not found: {args.src}[/red]")
            sys.exit(1)

    src = Path(args.src)

    # ── Stage 1: Convert ────────────────────────────────────────────────────
    if not args.skip_convert:
        run_conversion(src)

    # ── Stage 2: Compress ───────────────────────────────────────────────────
    if not args.skip_compress:
        run_compression(src)

    # ── Load InsightFace ONCE — used by Stages 3, 4, 5 ─────────────────────
    face_app = None
    need_face = (
        not args.skip_dedup
        or (not args.skip_facesort and args.faces)
        or not args.skip_organize
    )
    if need_face:
        console.print()
        console.rule("[bold cyan]Loading InsightFace[/bold cyan]")
        # Stage 4 (face sort) needs genderage; Stage 3/5 only need det+rec.
        # Load with genderage so it's available if face sort runs.
        modules = ["detection", "recognition"]
        if not args.skip_facesort and args.faces:
            modules.append("genderage")
        with console.status("[cyan]Loading buffalo_l…[/cyan]"):
            t0       = time.time()
            face_app = load_face_app(args.models, allowed_modules=modules)
        if face_app:
            console.print(f"  [green]✓[/green]  Models ready  ({time.time()-t0:.1f}s)\n")

    # ── Load unified embedding cache ─────────────────────────────────────────
    cache = cache_load(src)

    # Exclude system folders from image collection
    exclude = {_DUP_FOLDER, "_workflow_cache.json"}
    if args.faces:
        exclude.add(Path(args.faces).name)

    images = collect_images(src, args.recursive, exclude_dirs=exclude)
    console.print(f"\n  [cyan]{len(images)}[/cyan] images found in {src}\n")

    if not images:
        console.print("[yellow]  No images found. Exiting.[/yellow]")
        sys.exit(0)

    # ── Stage 3: Dedup ──────────────────────────────────────────────────────
    remaining = images
    if not args.skip_dedup:
        remaining = run_dedup(
            src=src,
            images=images,
            face_app=face_app,
            cache=cache,
            execute=args.execute,
            phash_threshold=args.phash_threshold,
        )
        cache_save(cache, src)

    # ── Stage 4: Face Sort ──────────────────────────────────────────────────
    if not args.skip_facesort and args.faces and face_app is not None:
        console.print()
        console.rule("[bold cyan]Loading reference faces[/bold cyan]")
        refs = _load_references(face_app, Path(args.faces), cache)
        if refs:
            remaining = run_face_sort(
                src=src,
                images=remaining,
                face_app=face_app,
                refs=refs,
                cache=cache,
                threshold=args.face_threshold,
                execute=args.execute,
            )
            cache_save(cache, src)
        else:
            console.print("  [yellow]⚠  No valid references loaded — skipping face sort.[/yellow]")
    elif not args.skip_facesort:
        console.print("\n  [dim]Stage 4 skipped (no Faces folder specified).[/dim]")

    # ── Stage 5: Organize ───────────────────────────────────────────────────
    if not args.skip_organize:
        run_organizer(
            src=src,
            images=remaining,
            face_app=face_app,
            cache=cache,
            models_root=args.models,
            prefix=args.prefix,
            cluster_threshold=args.cluster_threshold,
            time_half_life_hours=args.time_half_life,
            w_face=args.weight_face,
            w_clip=args.weight_clip,
            w_bg=args.weight_bg,
            w_time=args.weight_time,
            execute=args.execute,
            force=args.force,
        )
        cache_save(cache, src)

    console.print()
    console.print(Panel.fit(
        Text.assemble(("Pipeline complete.", "bold green")),
        border_style="green",
    ))
    console.print()


if __name__ == "__main__":
    main()
