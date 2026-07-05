#!/usr/bin/env python3
"""
smart_image_organizer_v2.py  —  Multi-Signal Image Clustering & Sequential Renaming
====================================================================================
Clusters images using COMBINED signals, then renames each cluster sequentially so
related images sit together when sorted by name.

  Signals used
  ────────────
  • ArcFace face identity  (InsightFace buffalo_l, 512-dim cosine)
  • CLIP ViT-B/32          (scene / theme / attire similarity)
  • Background HSV hist    (outer-border colour, ignores face centre)
  • Date-modified          (exponential decay over configurable half-life)

  Output naming
  ─────────────
  {prefix}_{series:03d}_{seq:03d}{ext}   e.g.  IMG_001_001.jpg

  Quick start (interactive)
  ─────────────────────────
  python smart_image_organizer_v2.py

  Command-line
  ────────────
  python smart_image_organizer_v2.py \\
    --src "C:/Photos" --recursive \\
    --models-root "C:/Users/Sunil/Downloads/Quick Share" \\
    --prefix album --cluster-threshold 0.48 --dry-run

  Install
  ───────
  # Standard (CPU):
  pip install insightface onnxruntime clip-anytorch torch torchvision \\
              opencv-python Pillow numpy scikit-learn rich psutil
  # Recommended for Samsung Galaxy Book 5 / Intel Arc (DirectML):
  pip uninstall onnxruntime -y
  pip install insightface onnxruntime-directml clip-anytorch torch-directml \\
              torchvision opencv-python Pillow numpy scikit-learn rich psutil
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── ESC-key abort (via shared TerminationManager) ─────────────────────────────
import sys as _sys_tmp
_scripts_dir = str(Path(__file__).parent / "src" / "scripts")
if _scripts_dir not in _sys_tmp.path:
    _sys_tmp.path.insert(0, _scripts_dir)
from common_utils import TerminationManager as _TerminationManager
del _sys_tmp, _scripts_dir
_tm = _TerminationManager()
_tm.start_monitoring()

# ── Cap ONNX/OpenMP threads BEFORE onnxruntime / torch are imported ──────────
_OMP_THREADS = 4
os.environ.setdefault("OMP_NUM_THREADS",          str(_OMP_THREADS))
os.environ.setdefault("MKL_NUM_THREADS",          str(_OMP_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS",     str(_OMP_THREADS))
os.environ.setdefault("ONNXRUNTIME_THREAD_COUNT", str(_OMP_THREADS))

# ── Lower process priority so PC stays responsive ─────────────────────────────
try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 50_000_000

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}
MAX_BYTES   = 50 * 1024 * 1024
_DEFAULT_MODELS_ROOT = r"C:\Users\Sunil\Downloads\Quick Share"
_MAX_SIDE   = 1280   # cap longer edge before inference

# ── Optional heavy dependencies ───────────────────────────────────────────
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None

try:
    import clip as _clip_module
except ImportError:
    _clip_module = None

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

try:
    from sklearn.cluster import AgglomerativeClustering
except ImportError:
    AgglomerativeClustering = None

# ── Rich console (optional but highly recommended) ────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn,
        MofNCompleteColumn, TimeRemainingColumn, TextColumn,
    )
    from rich.table import Table
    from rich.prompt import Prompt, Confirm, FloatPrompt, IntPrompt
    _console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class _Console:          # bare fallback
        def print(self, *a, **kw):
            print(re.sub(r"\[/?[^\]]*\]", "", " ".join(str(x) for x in a)))
        def rule(self, title="", **kw):
            print(f"\n{'─'*60}  {title}")
    _console = _Console()


def _ask(prompt: str, default: str = "") -> str:
    if HAS_RICH:
        return Prompt.ask(prompt, default=default)
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def _confirm(prompt: str, default: bool = False) -> bool:
    if HAS_RICH:
        return Confirm.ask(prompt, default=default)
    suffix = "[Y/n]" if default else "[y/N]"
    val = input(f"{prompt} {suffix}: ").strip().lower()
    if val in ("y", "yes"):
        return True
    if val in ("n", "no"):
        return False
    return default


def _ask_float(prompt: str, default: float) -> float:
    if HAS_RICH:
        return FloatPrompt.ask(prompt, default=default)
    try:
        return float(input(f"{prompt} [{default}]: ").strip() or default)
    except ValueError:
        return default


@dataclass
class ImageRecord:
    path: Path
    mtime: float
    face_emb: Optional[np.ndarray]
    clip_emb: Optional[np.ndarray]
    bg_hist: Optional[np.ndarray]
    face_count: int
    width: int
    height: int


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


def collect_images(src: Path, recursive: bool) -> List[Path]:
    paths: List[Path] = []
    if recursive:
        for root, _, files in os.walk(src):
            for fname in files:
                p = Path(root) / fname
                if p.is_file() and is_image_path(p):
                    paths.append(p)
    else:
        for p in src.iterdir():
            if p.is_file() and is_image_path(p):
                paths.append(p)
    return sorted(paths, key=lambda p: (p.stat().st_mtime, p.name.lower()))


def validate_image(path: Path) -> None:
    if not path.exists() or path.is_dir():
        raise ValueError("Not a file")
    if not is_image_path(path):
        raise ValueError("Unsupported extension")
    size = path.stat().st_size
    if size <= 0 or size > MAX_BYTES:
        raise ValueError(f"Invalid file size: {size}")
    with Image.open(path) as im:
        im.verify()


def l2_normalize(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vec is None:
        return None
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(vec)
    if n <= 1e-12:
        return None
    return vec / n


def cosine_distance(a: Optional[np.ndarray], b: Optional[np.ndarray], missing_penalty: float = 0.65) -> float:
    if a is None and b is None:
        return 0.5
    if a is None or b is None:
        return missing_penalty
    sim = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return (1.0 - sim) / 2.0


def chi_square_distance(a: Optional[np.ndarray], b: Optional[np.ndarray], missing_penalty: float = 0.7) -> float:
    if a is None and b is None:
        return 0.5
    if a is None or b is None:
        return missing_penalty
    denom = a + b + 1e-10
    d = 0.5 * np.sum(((a - b) ** 2) / denom)
    return float(np.clip(d, 0.0, 1.0))


def compute_time_distance(t1: float, t2: float, half_life_hours: float = 24.0 * 3.0) -> float:
    dt_hours = abs(t1 - t2) / 3600.0
    score = 1.0 - math.exp(-dt_hours / max(1e-6, half_life_hours))
    return float(np.clip(score, 0.0, 1.0))


def detect_background_mask(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mx1, mx2 = int(0.08 * w), int(0.92 * w)
    my1, my2 = int(0.08 * h), int(0.92 * h)
    mask[:my1, :] = 255
    mask[my2:, :] = 255
    mask[:, :mx1] = 255
    mask[:, mx2:] = 255
    return mask


def compute_background_hsv_hist(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = detect_background_mask(img_bgr)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


class FeatureExtractor:
    """
    Extracts face, CLIP, and background features per image.
    Face and CLIP are optional — if their deps are missing the signals are
    simply omitted and the distance matrix re-weights accordingly.
    """

    def __init__(
        self,
        models_root: str = _DEFAULT_MODELS_ROOT,
        providers: Optional[List[str]] = None,
        det_size: Tuple[int, int] = (640, 640),
    ):
        if cv2 is None:
            raise RuntimeError("opencv-python is required. Run: pip install opencv-python")

        # ── Resolve best ONNX execution provider (DirectML > CPU) ───────
        # CUDAExecutionProvider intentionally excluded: Galaxy Book 5 has Intel
        # Arc, no NVIDIA GPU. onnxruntime-directml lists CUDAExecutionProvider
        # anyway → "cublasLt64_12.dll missing" spam → CPU fallback. Skip it.
        import onnxruntime as ort
        ort.set_default_logger_severity(3)   # suppress provider-load warnings
        _avail = ort.get_available_providers()
        if providers is None:
            providers = [ep for ep in
                         ("DmlExecutionProvider", "CPUExecutionProvider")   # CUDA removed
                         if ep in _avail] or ["CPUExecutionProvider"]
        _gpu_ep = providers[0] != "CPUExecutionProvider"
        if _gpu_ep:
            _console.print(f"  [green]✓[/green] [dim]ONNX provider: {providers[0]}  (threads capped={_OMP_THREADS})[/dim]")
        else:
            _console.print(
                f"  [yellow]⚠ DirectML unavailable — CPU mode (threads capped={_OMP_THREADS})[/yellow]\n"
                f"  [dim]  Fix: pip uninstall onnxruntime -y  &&  "
                f"pip install onnxruntime-directml[/dim]"
            )

        # ── Resolve best PyTorch device for CLIP ──────────────────────────
        # Priority: torch-directml (Intel Arc) > CUDA > CPU
        self.device = "cpu"
        if torch is not None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                try:
                    import torch_directml          # pip install torch-directml
                    self.device = torch_directml.device()
                    _console.print(f"  [dim]CLIP device: torch-directml (Intel Arc)[/dim]")
                except ImportError:
                    pass                           # stay on CPU

        # ── ArcFace / InsightFace ─────────────────────────────────────────
        # allowed_modules skips 3 unused models (2d106det, 1k3d68, genderage)
        # → ~40-50% less inference work per image
        self.face_app = None
        if FaceAnalysis is not None:
            try:
                self.face_app = FaceAnalysis(
                    name="buffalo_l",
                    root=models_root,
                    providers=providers,
                    allowed_modules=["detection", "recognition"],
                )
                self.face_app.prepare(ctx_id=0 if _gpu_ep else -1, det_size=det_size)
                _console.print("[green]✓[/green] ArcFace loaded (detection + recognition only)")
            except Exception as e:
                _console.print(f"[yellow]⚠ ArcFace unavailable: {e}[/yellow]")
                self.face_app = None
        else:
            _console.print("[yellow]⚠ insightface not installed — face signal disabled[/yellow]")

        # ── CLIP ──────────────────────────────────────────────────────────
        self.clip_model = None
        self.clip_preprocess = None
        if _clip_module is not None and torch is not None:
            try:
                self.clip_model, self.clip_preprocess = _clip_module.load(
                    "ViT-B/32", device=self.device
                )
                _console.print(f"[green]✓[/green] CLIP loaded (ViT-B/32, device={self.device})")
            except Exception as e:
                _console.print(f"[yellow]⚠ CLIP unavailable: {e}[/yellow]")
        else:
            _console.print("[yellow]⚠ clip-anytorch / torch not installed — CLIP signal disabled[/yellow]")

    @staticmethod
    def _resize_for_inference(img_bgr: np.ndarray, max_side: int = _MAX_SIDE) -> np.ndarray:
        """Cap longer edge at max_side (INTER_AREA). Reduces cost for 4K+ phone photos.
        ArcFace crops faces to 112×112 regardless of input size → no accuracy loss."""
        h, w = img_bgr.shape[:2]
        longest = max(h, w)
        if longest <= max_side:
            return img_bgr
        scale = max_side / longest
        return cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    def extract(self, path: Path, cache: Optional[dict] = None) -> ImageRecord:
        validate_image(path)

        pil = Image.open(path).convert("RGB")
        width, height = pil.size
        mtime = path.stat().st_mtime

        img_rgb = np.array(pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr = self._resize_for_inference(img_bgr)

        # ── Cache lookup — skip inference if embeddings already known ─────
        face_emb: Optional[np.ndarray] = None
        clip_emb: Optional[np.ndarray] = None
        cache_hit = False
        if cache is not None:
            face_emb, clip_emb = _cache_get(cache, path)
            cache_hit = (face_emb is not None or clip_emb is not None
                         or str(path.resolve()) in cache)

        # ── Face embedding (skip if cached) ───────────────────────────────
        face_count = 0
        if not cache_hit and self.face_app is not None:
            try:
                faces = self.face_app.get(img_bgr)
                face_count = len(faces)
                if faces:
                    largest = max(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    )
                    face_emb = l2_normalize(np.asarray(largest.embedding, dtype=np.float32))
            except Exception:
                pass

        # ── CLIP embedding (skip if cached) ───────────────────────────────
        if not cache_hit and self.clip_model is not None and self.clip_preprocess is not None:
            try:
                with torch.no_grad():
                    inp = self.clip_preprocess(pil).unsqueeze(0).to(
                        self.device if not hasattr(self.device, "type") else self.device
                    )
                    raw = self.clip_model.encode_image(inp)[0].detach().cpu().numpy().astype(np.float32)
                    clip_emb = l2_normalize(raw)
            except Exception:
                pass

        # ── Write to cache if we just computed fresh embeddings ───────────
        if not cache_hit and cache is not None:
            _cache_set(cache, path, face_emb, clip_emb)

        # ── Background histogram — fast cv2, no need to cache ─────────────
        bg_hist = compute_background_hsv_hist(img_bgr)

        return ImageRecord(
            path=path,
            mtime=mtime,
            face_emb=face_emb,
            clip_emb=clip_emb,
            bg_hist=bg_hist,
            face_count=face_count,
            width=width,
            height=height,
        )


# ── Embedding cache ───────────────────────────────────────────────────────────
# JSON sidecar: {abs_path: {"mtime": float, "face": [...] | null, "clip": [...] | null}}
# Keyed by abs path + mtime. Skips heavy inference on repeat runs.
# Both ArcFace (~400ms) and CLIP (~200ms) cached → repeat runs near-instant.

def _cache_load(src_folder: Path) -> dict:
    cache_file = src_folder / "_organizer_cache.json"
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _cache_save(cache: dict, src_folder: Path) -> None:
    cache_file = src_folder / "_organizer_cache.json"
    try:
        cache_file.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass


def _cache_get(cache: dict, path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (face_emb, clip_emb) from cache, or (None, None) on miss/stale."""
    key = str(path.resolve())
    entry = cache.get(key)
    if not entry or abs(entry.get("mtime", 0) - path.stat().st_mtime) >= 1.0:
        return None, None
    face = np.array(entry["face"], dtype=np.float32) if entry.get("face") else None
    clip = np.array(entry["clip"], dtype=np.float32) if entry.get("clip") else None
    return face, clip


def _cache_set(cache: dict, path: Path,
               face_emb: Optional[np.ndarray],
               clip_emb: Optional[np.ndarray]) -> None:
    cache[str(path.resolve())] = {
        "mtime": path.stat().st_mtime,
        "face":  face_emb.tolist() if face_emb is not None else None,
        "clip":  clip_emb.tolist() if clip_emb is not None else None,
    }


def build_distance_matrix(
    records: List[ImageRecord],
    w_face: float = 0.40,
    w_clip: float = 0.28,
    w_bg: float = 0.17,
    w_time: float = 0.15,
    time_half_life_hours: float = 72.0,
    face_presence_bonus: float = 0.08,
) -> np.ndarray:
    """
    Vectorized N×N distance matrix using numpy broadcasting.
    Replaces O(N²) pure-Python loop — ~50-100× faster for N > 200.
    """
    n = len(records)
    MISSING_FACE = 0.68
    MISSING_CLIP = 0.60
    MISSING_BG   = 0.70

    # ── Face cosine distance matrix ───────────────────────────────────────
    face_has = np.array([r.face_emb is not None for r in records])  # (N,) bool
    if face_has.any():
        F = np.zeros((n, 512), dtype=np.float32)
        for i, r in enumerate(records):
            if r.face_emb is not None:
                F[i] = r.face_emb
        # cosine similarity via dot product (embeddings already L2-normalised)
        sim_face = np.clip(F @ F.T, -1.0, 1.0)
        D_face = (1.0 - sim_face) / 2.0                              # [0, 1]
        # fill rows/cols where face missing with penalty
        no_face = ~face_has
        D_face[no_face, :] = MISSING_FACE
        D_face[:, no_face] = MISSING_FACE
        np.fill_diagonal(D_face, 0.0)
    else:
        D_face = np.full((n, n), MISSING_FACE, dtype=np.float32)
        np.fill_diagonal(D_face, 0.0)

    # ── CLIP cosine distance matrix ───────────────────────────────────────
    clip_has = np.array([r.clip_emb is not None for r in records])
    if clip_has.any():
        clip_dim = next(r.clip_emb.shape[0] for r in records if r.clip_emb is not None)
        C = np.zeros((n, clip_dim), dtype=np.float32)
        for i, r in enumerate(records):
            if r.clip_emb is not None:
                C[i] = r.clip_emb
        sim_clip = np.clip(C @ C.T, -1.0, 1.0)
        D_clip = (1.0 - sim_clip) / 2.0
        no_clip = ~clip_has
        D_clip[no_clip, :] = MISSING_CLIP
        D_clip[:, no_clip] = MISSING_CLIP
        np.fill_diagonal(D_clip, 0.0)
    else:
        D_clip = np.full((n, n), MISSING_CLIP, dtype=np.float32)
        np.fill_diagonal(D_clip, 0.0)

    # ── Background chi-square distance matrix (chunked to save RAM) ─────────
    # Full (N,N,D) broadcast needs ~N²×D×4 bytes — 21 GiB for N=2374, D=1024.
    # Instead compute in row-chunks of _BG_CHUNK rows at a time:
    #   peak RAM = _BG_CHUNK × N × D × 4 bytes  ≈ 128×2374×1024×4 ≈ 1.2 GiB  ✓
    _BG_CHUNK = 128
    bg_has = np.array([r.bg_hist is not None for r in records])
    if bg_has.any():
        bg_dim = next(r.bg_hist.shape[0] for r in records if r.bg_hist is not None)
        BG = np.zeros((n, bg_dim), dtype=np.float32)
        for i, r in enumerate(records):
            if r.bg_hist is not None:
                BG[i] = r.bg_hist
        D_bg = np.zeros((n, n), dtype=np.float32)
        for row_start in range(0, n, _BG_CHUNK):
            row_end  = min(row_start + _BG_CHUNK, n)
            chunk    = BG[row_start:row_end, np.newaxis, :]   # (chunk, 1, D)
            full     = BG[np.newaxis, :, :]                   # (1, N, D)
            denom    = chunk + full + 1e-10
            D_bg[row_start:row_end, :] = np.clip(
                0.5 * np.sum((chunk - full) ** 2 / denom, axis=2), 0.0, 1.0
            )
        no_bg = ~bg_has
        D_bg[no_bg, :] = MISSING_BG
        D_bg[:, no_bg] = MISSING_BG
        np.fill_diagonal(D_bg, 0.0)
    else:
        D_bg = np.full((n, n), MISSING_BG, dtype=np.float32)
        np.fill_diagonal(D_bg, 0.0)

    # ── Time distance matrix (vectorized) ────────────────────────────────
    mtimes = np.array([r.mtime for r in records], dtype=np.float64)
    dt_hours = np.abs(mtimes[:, np.newaxis] - mtimes[np.newaxis, :]) / 3600.0
    D_time = np.clip(1.0 - np.exp(-dt_hours / max(1e-6, time_half_life_hours)), 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(D_time, 0.0)

    # ── Combine ───────────────────────────────────────────────────────────
    D = (w_face * D_face + w_clip * D_clip + w_bg * D_bg + w_time * D_time).astype(np.float32)

    # Face presence bonus/penalty (vectorized)
    both_face = face_has[:, np.newaxis] & face_has[np.newaxis, :]  # (N,N) bool
    one_face  = face_has[:, np.newaxis] ^ face_has[np.newaxis, :]
    D = np.where(both_face & (D_face < 0.22), np.clip(D - face_presence_bonus, 0.0, 1.0), D)
    D = np.where(one_face, np.clip(D + 0.04, 0.0, 1.0), D)

    # ── Solo vs group penalty ─────────────────────────────────────────────
    # Images with different face counts (solo=1 vs couple/group=2+) get pushed
    # apart so they don't merge into the same cluster.
    # Same count  → penalty 0.0
    # Differ by 1 → penalty 0.20  (solo vs couple)
    # Differ by 2+→ penalty 0.35  (solo vs group)
    fc = np.array([r.face_count for r in records], dtype=np.float32)
    fc_diff = np.abs(fc[:, np.newaxis] - fc[np.newaxis, :])
    D_fc = np.where(fc_diff == 0, 0.0,
           np.where(fc_diff == 1, 0.20, 0.35)).astype(np.float32)
    np.fill_diagonal(D_fc, 0.0)
    D = np.clip(D + D_fc, 0.0, 1.0)

    np.fill_diagonal(D, 0.0)
    return D.astype(np.float32)


def cluster_records(distance_matrix: np.ndarray, distance_threshold: float) -> np.ndarray:
    if AgglomerativeClustering is None:
        raise RuntimeError("scikit-learn is required for clustering")

    kwargs = dict(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )

    try:
        model = AgglomerativeClustering(**kwargs)
    except TypeError:
        kwargs["affinity"] = "precomputed"
        kwargs.pop("metric", None)
        model = AgglomerativeClustering(**kwargs)

    labels = model.fit_predict(distance_matrix)
    return labels


def order_cluster_indices(indices: List[int], records: List[ImageRecord], D: np.ndarray) -> List[int]:
    if len(indices) <= 2:
        return sorted(indices, key=lambda i: records[i].mtime)

    ordered_by_time = sorted(indices, key=lambda i: records[i].mtime)
    used = {ordered_by_time[0]}
    sequence = [ordered_by_time[0]]

    while len(sequence) < len(indices):
        last = sequence[-1]
        remaining = [i for i in indices if i not in used]

        def score(i: int) -> Tuple[float, float]:
            time_gap = abs(records[i].mtime - records[last].mtime)
            return (D[last, i] + min(time_gap / (3600.0 * 24.0 * 30.0), 1.0) * 0.05, time_gap)

        nxt = min(remaining, key=score)
        sequence.append(nxt)
        used.add(nxt)

    return sequence


def detect_existing_series(images: List[Path], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_(\d+)$")
    highest = 0
    for img in images:
        m = pattern.match(img.stem)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest


def filter_already_renamed_images(images: List[Path], prefix: str) -> Tuple[List[Path], int]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_(\d+)$")
    fresh = []
    skipped = 0
    for img in images:
        if pattern.match(img.stem):
            skipped += 1
        else:
            fresh.append(img)
    return fresh, skipped


def safe_rename(src: Path, dest: Path) -> None:
    if src.resolve() == dest.resolve():
        return
    try:
        os.replace(src, dest)
    except Exception:
        with src.open("rb") as rf, open(dest, "xb") as wf:
            shutil.copyfileobj(rf, wf, length=1024 * 1024)
        src.unlink()


def rename_clusters(
    clustered_paths: List[List[Path]],
    prefix: str,
    continue_series: bool,
    all_existing_images: List[Path],
) -> None:
    prefix = sanitize_filename(prefix)
    start_idx = detect_existing_series(all_existing_images, prefix) + 1 if continue_series else 1

    temp_moves: List[Tuple[Path, Path]] = []
    for cluster in clustered_paths:
        for p in cluster:
            tmp = unique_dest(p.parent, f"__tmp__{p.stem}{p.suffix}")
            safe_rename(p, tmp)
            temp_moves.append((p, tmp))

    tmp_map = {orig: tmp for orig, tmp in temp_moves}

    series_idx = start_idx
    for cluster in clustered_paths:
        item_idx = 1
        for original_path in cluster:
            tmp_path = tmp_map[original_path]
            ext = original_path.suffix.lower()
            new_name = f"{prefix}_{series_idx:03d}_{item_idx:03d}{ext}"
            dest = unique_dest(original_path.parent, new_name)
            safe_rename(tmp_path, dest)
            item_idx += 1
        series_idx += 1


def _interactive_setup(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in any missing CLI arguments interactively."""
    _console.print(Panel(
        "[bold]Smart Image Organizer v2[/bold]\n"
        "Multi-signal clustering: face · CLIP · background · date\n"
        "Sequential renaming: [cyan]{prefix}_{series:03d}_{seq:03d}{ext}[/cyan]",
        title="[bold cyan]Setup[/bold cyan]",
        border_style="cyan",
    ) if HAS_RICH else "=== Smart Image Organizer v2 ===")

    # Source folder
    while args.src is None:
        raw = _ask("[bold]Source folder[/bold]")
        p = Path(raw)
        if p.exists() and p.is_dir():
            args.src = p
        else:
            _console.print("[red]Folder not found. Try again.[/red]")

    if not args.recursive:
        args.recursive = _confirm("Recurse into subfolders?", default=False)

    if args.models_root == _DEFAULT_MODELS_ROOT:
        ans = _ask(
            "Models root (parent of models/buffalo_l/)",
            default=_DEFAULT_MODELS_ROOT,
        )
        args.models_root = ans

    if args.prefix == "series":
        args.prefix = _ask("Rename prefix", default="IMG")

    if not args.dry_run:
        args.dry_run = _confirm("Dry run first (preview renames)?", default=True)

    return args


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-signal image clustering + sequential renaming.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--src", type=Path, default=None, help="Source image folder (prompted if omitted)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--models-root", type=str, default=_DEFAULT_MODELS_ROOT,
                    help="Parent of models/buffalo_l/ (default: %(default)s)")
    ap.add_argument("--prefix", type=str, default="series", help="Rename prefix (default: series)")
    ap.add_argument("--cluster-threshold", type=float, default=0.48,
                    help="Agglomerative distance threshold 0–1 (default: 0.48)")
    ap.add_argument("--time-half-life-hours", type=float, default=72.0,
                    help="Time proximity half-life in hours (default: 72)")
    ap.add_argument("--weight-face", type=float, default=0.40)
    ap.add_argument("--weight-clip", type=float, default=0.28)
    ap.add_argument("--weight-bg",   type=float, default=0.17)
    ap.add_argument("--weight-time", type=float, default=0.15)
    ap.add_argument("--no-continue-series", action="store_true",
                    help="Restart series numbering from 001 instead of continuing")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview renames without touching files")
    ap.add_argument("--force", action="store_true",
                    help="Re-cluster files that already match the prefix pattern (bypass skip)")
    args = ap.parse_args()

    # Fill missing args interactively when run without --src
    if args.src is None:
        args = _interactive_setup(args)
    else:
        if not args.src.exists() or not args.src.is_dir():
            raise SystemExit(f"Source folder not found: {args.src}")

    # ── Discover ──────────────────────────────────────────────────────────
    _console.rule("[bold cyan]1 / 5  Discovering images[/bold cyan]")
    images = collect_images(args.src, args.recursive)
    if not images:
        raise SystemExit("No images found.")
    _console.print(f"Found [bold]{len(images)}[/bold] images in {args.src}")

    prefix = sanitize_filename(args.prefix)
    existing_all = list(images)
    if getattr(args, "force", False):
        skipped = 0
        _console.print("[dim]--force: skipping already-renamed filter, re-clustering all files[/dim]")
    else:
        images, skipped = filter_already_renamed_images(images, prefix)
        if skipped:
            _console.print(f"[dim]Skipped {skipped} already-renamed files (use --force to re-cluster)[/dim]")
    if not images:
        _console.print("[yellow]All images are already renamed. Run with --force to re-cluster.[/yellow]")
        return

    # ── Load models ───────────────────────────────────────────────────────
    _console.rule("[bold cyan]2 / 5  Loading models[/bold cyan]")
    try:
        extractor = FeatureExtractor(models_root=args.models_root)
    except RuntimeError as e:
        raise SystemExit(str(e))

    # ── Extract features ──────────────────────────────────────────────────
    _console.rule("[bold cyan]3 / 5  Extracting features[/bold cyan]")

    # ESC-key abort listener is already running via _tm.start_monitoring() at import time
    _console.print("  [dim](Press [bold]ESC[/bold] at any time to abort and save progress)[/dim]")

    # Load embedding cache — skips ArcFace+CLIP inference for unchanged files
    emb_cache = _cache_load(args.src)
    cache_hits_before = sum(
        1 for p in images
        if str(p.resolve()) in emb_cache
        and abs(emb_cache[str(p.resolve())].get("mtime", 0) - p.stat().st_mtime) < 1.0
    )
    if cache_hits_before:
        _console.print(f"[dim]Cache: {cache_hits_before}/{len(images)} embeddings already stored "
                       f"(~{cache_hits_before * 600 // 1000}s saved)[/dim]")

    records: List[ImageRecord] = []
    failed_names: List[str] = []

    if HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=_console,
        ) as prog:
            task = prog.add_task("Extracting features", total=len(images))
            for p in images:
                if _tm.is_terminating():
                    _console.print("\n  [yellow]⚠  ESC pressed — saving cache and stopping.[/yellow]")
                    break
                try:
                    records.append(extractor.extract(p, cache=emb_cache))
                except (ValueError, UnidentifiedImageError, RuntimeError, Exception) as e:
                    failed_names.append(f"{p.name}: {e}")
                prog.advance(task)
    else:
        for i, p in enumerate(images, 1):
            if _tm.is_terminating():
                print("\n  ESC pressed — saving cache and stopping.")
                break
            try:
                records.append(extractor.extract(p, cache=emb_cache))
            except Exception as e:
                failed_names.append(f"{p.name}: {e}")
            if i % 25 == 0 or i == len(images):
                print(f"  {i}/{len(images)} processed")

    # Persist updated cache
    _cache_save(emb_cache, args.src)

    _console.print(
        f"Extracted [bold]{len(records)}[/bold] images"
        + (f"  ({len(failed_names)} failed)" if failed_names else "")
    )
    if failed_names:
        for msg in failed_names[:10]:
            _console.print(f"  [dim red]SKIP {msg}[/dim red]")
        if len(failed_names) > 10:
            _console.print(f"  [dim]... and {len(failed_names)-10} more[/dim]")

    if len(records) == 0:
        raise SystemExit("No valid images after feature extraction.")

    # ── Cluster ───────────────────────────────────────────────────────────
    _console.rule("[bold cyan]4 / 5  Clustering[/bold cyan]")

    if len(records) == 1:
        clustered_paths = [[records[0].path]]
    else:
        total_w = args.weight_face + args.weight_clip + args.weight_bg + args.weight_time
        wf = args.weight_face / total_w
        wc = args.weight_clip / total_w
        wb = args.weight_bg  / total_w
        wt = args.weight_time / total_w

        _console.print(
            f"Weights — face:[cyan]{wf:.2f}[/cyan]  "
            f"clip:[cyan]{wc:.2f}[/cyan]  "
            f"bg:[cyan]{wb:.2f}[/cyan]  "
            f"time:[cyan]{wt:.2f}[/cyan]  "
            f"threshold:[cyan]{args.cluster_threshold}[/cyan]"
        )

        _console.print(f"[dim]Building {len(records)}×{len(records)} distance matrix…[/dim]")
        D = build_distance_matrix(
            records,
            w_face=wf, w_clip=wc, w_bg=wb, w_time=wt,
            time_half_life_hours=args.time_half_life_hours,
        )

        if AgglomerativeClustering is None:
            raise SystemExit("scikit-learn is required. Run: pip install scikit-learn")
        labels = cluster_records(D, args.cluster_threshold)

        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(idx)

        clustered_paths = []
        for _, idxs in sorted(
            clusters.items(),
            key=lambda kv: min(records[i].mtime for i in kv[1]),
        ):
            ordered = order_cluster_indices(idxs, records, D)
            clustered_paths.append([records[i].path for i in ordered])

    _console.print(f"Formed [bold]{len(clustered_paths)}[/bold] clusters")

    # Show cluster table
    if HAS_RICH:
        tbl = Table(title="Cluster preview (top 25)", show_lines=False)
        tbl.add_column("Series", style="cyan", justify="right")
        tbl.add_column("Images", justify="right")
        tbl.add_column("Sample", style="dim")
        start = (detect_existing_series(existing_all, prefix) + 1
                 if not args.no_continue_series else 1)
        for ci, cluster in enumerate(clustered_paths[:25]):
            tbl.add_row(f"{prefix}_{start+ci:03d}", str(len(cluster)), cluster[0].name)
        if len(clustered_paths) > 25:
            tbl.add_row("…", f"+{len(clustered_paths)-25}", "")
        _console.print(tbl)
    else:
        start = (detect_existing_series(existing_all, prefix) + 1
                 if not args.no_continue_series else 1)
        for ci, cluster in enumerate(clustered_paths[:25]):
            print(f"  {prefix}_{start+ci:03d}: {len(cluster)} images  e.g. {cluster[0].name}")
        if len(clustered_paths) > 25:
            print(f"  … and {len(clustered_paths)-25} more")

    # ── Rename ────────────────────────────────────────────────────────────
    _console.rule("[bold cyan]5 / 5  Renaming[/bold cyan]")

    if args.dry_run:
        _console.print("[bold yellow]DRY RUN[/bold yellow] — showing first 30 renames:\n")
        shown = 0
        start = (detect_existing_series(existing_all, prefix) + 1
                 if not args.no_continue_series else 1)
        for ci, cluster in enumerate(clustered_paths):
            for j, p in enumerate(cluster, 1):
                if shown >= 30:
                    _console.print(
                        f"[dim]  … and {sum(len(c) for c in clustered_paths)-30} more[/dim]"
                    )
                    break
                new_name = f"{prefix}_{start+ci:03d}_{j:03d}{p.suffix.lower()}"
                _console.print(f"  {p.name}  →  [bold]{new_name}[/bold]")
                shown += 1
            if shown >= 30:
                break

        _console.print()
        if _confirm("Apply renames now?", default=False):
            args.dry_run = False
        else:
            _console.print("[dim]No files changed.[/dim]")
            return

    if not args.dry_run:
        rename_clusters(
            clustered_paths=clustered_paths,
            prefix=prefix,
            continue_series=not args.no_continue_series,
            all_existing_images=existing_all,
        )
        _console.print(
            f"[green]✓[/green] Renamed [bold]{sum(len(c) for c in clustered_paths)}[/bold] files"
        )

    # ── Summary ───────────────────────────────────────────────────────────
    _console.print()
    if HAS_RICH:
        summary = Table(title="Summary", show_header=False, border_style="green")
        summary.add_column("", style="bold")
        summary.add_column("")
        summary.add_row("Source", str(args.src))
        summary.add_row("Images found", str(len(existing_all)))
        summary.add_row("Processed", str(len(records)))
        summary.add_row("Clusters", str(len(clustered_paths)))
        summary.add_row(
            "Signals active",
            "  ".join(filter(None, [
                "Face" if extractor.face_app else None,
                "CLIP" if extractor.clip_model else None,
                "Background", "Date",
            ])),
        )
        _console.print(summary)
    else:
        print(f"\nDone — {len(records)} images → {len(clustered_paths)} clusters")

    _console.print("\n[bold green]All done![/bold green]")


if __name__ == "__main__":
    main()