#!/usr/bin/env python3
"""
face_sorter.py  ─  ArcFace-powered face recognition image sorter
=================================================================
Detects faces in a folder of images and sorts them into per-person
subfolders using InsightFace's buffalo_l models (SCRFD + ArcFace).

All folder paths are asked interactively at runtime — no code edits needed.

Requirements (install once):
    # Standard (CPU only):
    pip install insightface onnxruntime opencv-python pillow tqdm rich psutil
    # Recommended for Samsung Galaxy Book 5 / Intel Arc GPU (DirectML):
    pip uninstall onnxruntime -y
    pip install insightface onnxruntime-directml opencv-python pillow tqdm rich psutil

Usage:
    python face_sorter.py                          # fully interactive
    python face_sorter.py --execute                # skip dry-run prompt
    python face_sorter.py --src "D:\\Photos" --faces "D:\\Refs"
    python face_sorter.py --threshold 0.45 --recursive
    python face_sorter.py --help
"""

from __future__ import annotations

# ── DEFAULTS  (overridden by CLI args or interactive prompts) ─────────────────
_DEFAULT_MODELS_ROOT = r"C:\Users\Sunil\Downloads\Quick Share"
_DEFAULT_THRESHOLD   = 0.40
_DEFAULT_MIN_DET     = 0.65   # raised from 0.50 — rejects blurry/distant/background faces
_IMAGE_EXTS          = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_MAX_SIDE            = 1280   # pre-resize: cap longer edge before inference
_OMP_THREADS         = 4      # ONNX CPU thread cap (change to os.cpu_count()//2 if desired)

# ── Group-detection face quality gates ───────────────────────────────────────
# Only "significant" secondary faces trigger Group.  Filters out background
# crowds, distant faces, bokeh blur, and edge-of-frame partial faces.
_MIN_FACE_IMG_RATIO   = 0.005  # face bbox area must be ≥ 0.5% of image area
_MIN_SECONDARY_RATIO  = 0.18   # secondary face area must be ≥ 18% of largest face area
# ─────────────────────────────────────────────────────────────────────────────

import os
# ── Cap ONNX/OpenMP threads BEFORE onnxruntime is imported ───────────────────
os.environ.setdefault("OMP_NUM_THREADS",         str(_OMP_THREADS))
os.environ.setdefault("MKL_NUM_THREADS",         str(_OMP_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS",    str(_OMP_THREADS))
os.environ.setdefault("ONNXRUNTIME_THREAD_COUNT", str(_OMP_THREADS))

# ── Lower process priority so PC stays responsive ────────────────────────────
try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)   # Windows
except Exception:
    pass   # psutil absent or non-Windows: silently skip

import argparse
import base64
import io
import json
import re
import shutil
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ── ESC-key abort (via shared TerminationManager) ─────────────────────────────
import sys as _sys_tmp
_scripts_dir = str(Path(__file__).parent / "src" / "scripts")
if _scripts_dir not in _sys_tmp.path:
    _sys_tmp.path.insert(0, _scripts_dir)
from common_utils import TerminationManager as _TerminationManager
del _sys_tmp, _scripts_dir
_tm = _TerminationManager()
_tm.start_monitoring()


# ── Dependency check ──────────────────────────────────────────────────────────

def _check_deps() -> None:
    missing = []
    for pkg, import_name in [
        ("insightface",   "insightface"),
        ("onnxruntime",   "onnxruntime"),
        ("opencv-python", "cv2"),
        ("pillow",        "PIL"),
        ("tqdm",          "tqdm"),
        ("rich",          "rich"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("\n  ✗  Missing packages. Install them with:\n")
        print(f"     pip install {' '.join(missing)}\n")
        sys.exit(1)

_check_deps()

# ── Imports (safe after dep-check) ───────────────────────────────────────────

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.rule import Rule
from rich.text import Text
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn,
)

console = Console()


# ── Interactive setup ─────────────────────────────────────────────────────────

def _prompt_folder(label: str, default: Optional[Path] = None) -> Path:
    """
    Ask the user to type a folder path.  Keeps prompting until the path exists.
    Pressing Enter accepts the default (if one is provided).
    """
    while True:
        hint = f" [dim](default: {default})[/dim]" if default else ""
        raw  = Prompt.ask(f"  [cyan]{label}[/cyan]{hint}").strip().strip('"').strip("'")

        if raw == "" and default:
            path = default
        else:
            path = Path(raw)

        if path.exists():
            return path

        console.print(f"    [red]✗  Path not found:[/red] {path}")


def _prompt_models_root(default: Path) -> Path:
    """Ask for models root and verify buffalo_l lives under it."""
    while True:
        path = _prompt_folder("Models root folder  (contains models\\buffalo_l\\)", default)
        buffalo = path / "models" / "buffalo_l"
        if buffalo.exists():
            return path
        console.print(
            f"    [red]✗  buffalo_l not found under:[/red] {buffalo}\n"
            f"       Expected structure: <root>\\models\\buffalo_l\\det_10g.onnx …"
        )


def interactive_setup(args: argparse.Namespace) -> argparse.Namespace:
    """
    Fill any missing args interactively, then confirm with the user before
    proceeding.  Already-supplied CLI args are respected as-is.
    """
    console.print()
    console.print(Panel.fit(
        Text.assemble(
            ("ArcFace Face Sorter", "bold cyan"),
            ("  ·  ", "dim"),
            ("InsightFace buffalo_l + ONNX Runtime", "dim"),
        ),
        border_style="cyan",
    ))
    console.print()
    console.print("  Answer the prompts below — press [bold]Enter[/bold] to accept defaults.\n")

    # ── Source folder ────────────────────────────────────────────────────────
    if args.src is None:
        args.src = _prompt_folder("Source folder  (images to sort)")
    else:
        console.print(f"  [cyan]Source folder[/cyan]  : {args.src}")

    # ── Reference faces folder ───────────────────────────────────────────────
    if args.faces is None:
        default_faces = args.src / "Faces"
        args.faces = _prompt_folder(
            "Faces folder   (reference photos, one per person)",
            default_faces if default_faces.exists() else None,
        )
    else:
        console.print(f"  [cyan]Faces folder[/cyan]   : {args.faces}")

    # ── Models root ──────────────────────────────────────────────────────────
    if args.models is None:
        args.models = str(_prompt_models_root(Path(_DEFAULT_MODELS_ROOT)))
    else:
        console.print(f"  [cyan]Models root[/cyan]    : {args.models}")

    # ── Recursive scan ───────────────────────────────────────────────────────
    if not args.recursive_set:          # only ask if not passed as CLI flag
        args.recursive = Confirm.ask(
            "\n  Scan [bold]subfolders[/bold] of source recursively?",
            default=False,
        )

    # ── Threshold ────────────────────────────────────────────────────────────
    console.print(
        f"\n  [dim]Similarity threshold: {args.threshold}  "
        "(0–1; raise to be stricter, e.g. 0.45)[/dim]"
    )

    # ── Report path default ──────────────────────────────────────────────────
    if args.report is None:
        args.report = args.src / "face_sort_report.html"

    # ── Confirm ──────────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("Ready to run", style="cyan"))
    console.print(f"  Source      : [white]{args.src}[/white]")
    console.print(f"  Faces       : [white]{args.faces}[/white]")
    console.print(f"  Models      : [white]{args.models}[/white]")
    console.print(f"  Recursive   : [white]{'yes' if args.recursive else 'no'}[/white]")
    console.print(f"  Threshold   : [white]{args.threshold}[/white]")
    console.print(f"  Mode        : [bold {'yellow' if not args.execute else 'red'}]"
                  f"{'DRY RUN (preview only)' if not args.execute else 'EXECUTE — files will be moved'}[/bold "
                  f"{'yellow' if not args.execute else 'red'}]")
    console.print(f"  Report      : [white]{args.report}[/white]")
    console.print()

    if not Confirm.ask("  Proceed?", default=True):
        console.print("[yellow]  Aborted.[/yellow]")
        sys.exit(0)

    return args


# ── InsightFace loader ────────────────────────────────────────────────────────

def load_face_app(models_root: str):
    from insightface.app import FaceAnalysis
    import onnxruntime as ort

    # Suppress noisy ONNX provider-load warnings (missing CUDA DLLs etc.)
    ort.set_default_logger_severity(3)   # 0=Verbose … 3=Error … 4=Fatal

    # Galaxy Book 5 has Intel Arc, no NVIDIA — skip CUDAExecutionProvider entirely.
    # CUDAExecutionProvider listed by onnxruntime-directml even without NVIDIA GPU,
    # causing "cublasLt64_12.dll missing" spam before falling back to CPU anyway.
    available = ort.get_available_providers()
    providers: list[str] = []
    for ep in ("DmlExecutionProvider", "CPUExecutionProvider"):   # CUDA removed
        if ep in available:
            providers.append(ep)
    if not providers:
        providers = ["CPUExecutionProvider"]

    ep_label = providers[0]
    if ep_label == "CPUExecutionProvider":
        console.print(
            f"  [yellow]⚠ DirectML unavailable — running on CPU "
            f"(threads capped={_OMP_THREADS})[/yellow]\n"
            f"  [dim]  Fix: pip uninstall onnxruntime -y  &&  "
            f"pip install onnxruntime-directml[/dim]"
        )
    else:
        console.print(
            f"  [green]✓[/green] [dim]ONNX provider: {ep_label}  "
            f"(threads capped={_OMP_THREADS})[/dim]"
        )

    # genderage added so we can count female faces for Group detection.
    # 2d106det.onnx and 1k3d68.onnx (landmarks) still skipped — not needed.
    app = FaceAnalysis(
        name            = "buffalo_l",
        root            = models_root,
        providers       = providers,
        allowed_modules = ["detection", "recognition", "genderage"],
    )
    app.prepare(ctx_id=0 if providers[0] != "CPUExecutionProvider" else -1,
                det_size=(640, 640))
    return app


# ── Cosine similarity ─────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── Reference loading ─────────────────────────────────────────────────────────

def load_references(app, faces_folder: Path) -> dict[str, list[np.ndarray]]:
    """
    Detect the largest face in every reference image and build:
        { person_name: [embedding, …] }

    Multiple files for the same person are merged automatically:
        Kajal.jpg + Kajal_01.jpg  →  'Kajal' gets two embeddings.
    """
    refs: dict[str, list[np.ndarray]] = defaultdict(list)
    images = sorted(
        p for p in faces_folder.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )

    if not images:
        console.print(f"[red]  ✗  No reference images in: {faces_folder}[/red]")
        sys.exit(1)

    table = Table(
        title="Reference faces",
        show_lines=False,
        header_style="bold cyan",
        min_width=62,
    )
    table.add_column("File",   style="white")
    table.add_column("Person", style="green")
    table.add_column("Det ✓",  style="yellow", justify="right")
    table.add_column("Status", justify="center")

    for img_path in images:
        person = re.sub(r"_\d+$", "", img_path.stem)   # Kajal_01 → Kajal
        img    = cv2.imread(str(img_path))
        if img is None:
            table.add_row(img_path.name, person, "—", "[red]can't read[/red]")
            continue

        faces = app.get(img)
        if not faces:
            table.add_row(img_path.name, person, "—", "[red]no face[/red]")
            continue

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        refs[person].append(face.embedding)
        table.add_row(img_path.name, person, f"{face.det_score:.2f}", "[green]✓[/green]")

    console.print(table)
    console.print(
        f"  [cyan]Loaded[/cyan] [bold]{len(refs)}[/bold] persons: "
        + ", ".join(sorted(refs))
    )
    return dict(refs)


# ── Per-image matcher ─────────────────────────────────────────────────────────

def _resize_for_inference(img: np.ndarray, max_side: int = _MAX_SIDE) -> np.ndarray:
    """Cap longer edge at max_side using INTER_AREA.  Reduces decode+inference cost
    for high-res phone photos (4K+) without affecting ArcFace accuracy — the
    recognition model crops faces to 112×112 regardless of input size."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def match_image(
    img_path: Path,
    app,
    refs: dict[str, list[np.ndarray]],
    threshold: float,
    min_det: float = _DEFAULT_MIN_DET,
    cache: dict | None = None,
) -> tuple[list[dict], int]:
    """Detect faces in one image.  Returns (matches, face_count).
    face_count = number of valid faces detected (above min_det score).
    If *cache* is provided, cached embeddings are reused and new ones stored."""
    img = cv2.imread(str(img_path))
    if img is None:
        return [], 0

    # ── Check embedding cache before running inference ───────────────────
    cached_emb, cached_face_count = (
        _get_cached_embedding(cache, img_path) if cache is not None else (None, 0)
    )
    if cached_emb is not None:
        best_person: Optional[str] = None
        best_score = -1.0
        for person, embeddings in refs.items():
            score = max(cosine_sim(cached_emb, e) for e in embeddings)
            if score > best_score:
                best_score, best_person = score, person
        if best_person and best_score >= threshold:
            return [{
                "person":     best_person,
                "score":      round(best_score, 4),
                "bbox":       [0, 0, 0, 0],
                "img_path":   img_path,
                "face_count": cached_face_count,
            }], cached_face_count
        return [], cached_face_count

    # ── Full inference path ──────────────────────────────────────────────
    img = _resize_for_inference(img)
    h, w = img.shape[:2]
    img_area = h * w

    faces = app.get(img)
    if not faces:
        return [], 0

    def _bbox_area(f) -> float:
        return max(0.0, float((f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])))

    # ── Gate 1: det_score ≥ min_det (rejects blurry / low-confidence) ───
    valid_faces = [f for f in faces if f.det_score >= min_det]

    # ── Gate 2: absolute size — face bbox ≥ 0.5% of image area ─────────
    # Eliminates tiny crowd/background faces regardless of det_score.
    valid_faces = [f for f in valid_faces
                   if _bbox_area(f) / img_area >= _MIN_FACE_IMG_RATIO]

    if not valid_faces:
        return [], 0

    # ── Primary face = largest valid face ───────────────────────────────
    primary_area = max(_bbox_area(f) for f in valid_faces)

    # ── Gate 3: relative size — secondary faces ≥ 18% of primary area ──
    # Eliminates background figures and distant faces when a clear
    # primary subject is present.
    significant_faces = [f for f in valid_faces
                         if _bbox_area(f) >= primary_area * _MIN_SECONDARY_RATIO]

    # ── Gender filter — only CONFIRMED female faces count toward Group ──
    # Unknown sex (attr absent) is NOT defaulted to female — avoids
    # counting distant/blurry faces whose gender model is uncertain.
    def _is_confirmed_female(f) -> bool:
        return getattr(f, "sex", None) == "F"

    face_count = sum(1 for f in significant_faces if _is_confirmed_female(f))

    confirmed: list[dict]  = []
    seen_persons: set[str] = set()

    # Run recognition only on significant faces (skip tiny background ones)
    for face in significant_faces:
        best_person = None
        best_score  = -1.0

        for person, embeddings in refs.items():
            if person in seen_persons:
                continue
            score = max(cosine_sim(face.embedding, e) for e in embeddings)
            if score > best_score:
                best_score, best_person = score, person

        if best_person and best_score >= threshold:
            confirmed.append({
                "person":     best_person,
                "score":      round(best_score, 4),
                "bbox":       face.bbox.astype(int).tolist(),
                "img_path":   img_path,
                "face_count": face_count,
                "is_female":  _is_confirmed_female(face),   # gender of THIS detected face
            })
            seen_persons.add(best_person)

    # ── Cache: store largest CONFIRMED-FEMALE face embedding ────────────
    female_sig = [f for f in significant_faces if _is_confirmed_female(f)]
    cache_face = (
        max(female_sig,    key=_bbox_area) if female_sig else
        max(significant_faces, key=_bbox_area)
    )
    if cache is not None:
        _set_cached_embedding(cache, img_path, cache_face.embedding, face_count)

    return confirmed, face_count


# ── Embedding cache ───────────────────────────────────────────────────────────
# Stores {abs_path: {"mtime": float, "embedding": [float, ...]}}
# Key = absolute path string.  Invalidated when file mtime changes.
# Saves ~400-600 ms per cached image on repeat runs.

def _load_cache(cache_path: Path) -> dict:
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict, cache_path: Path) -> None:
    try:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass   # cache write failure is non-fatal


def _get_cached_embedding(cache: dict, img_path: Path) -> tuple[Optional[np.ndarray], int]:
    """Returns (embedding, face_count).  face_count=1 for old cache entries (backward compat)."""
    key   = str(img_path.resolve())
    entry = cache.get(key)
    if entry and abs(entry["mtime"] - img_path.stat().st_mtime) < 1.0:
        return np.array(entry["embedding"], dtype=np.float32), int(entry.get("face_count", 1))
    return None, 0


def _set_cached_embedding(cache: dict, img_path: Path, emb: np.ndarray, face_count: int = 1) -> None:
    cache[str(img_path.resolve())] = {
        "mtime":      img_path.stat().st_mtime,
        "embedding":  emb.tolist(),
        "face_count": face_count,
    }


# ── Image collection ──────────────────────────────────────────────────────────

_SKIP_FOLDERS = {"Others", "Group"}   # auto-created destination folders

def collect_images(src: Path, recursive: bool, faces: Path) -> list[Path]:
    """
    Gather all image files from src.
    - recursive=False : main folder only
    - recursive=True  : all subfolders, but always skips the Faces folder
                        and the auto-created Others / Group destination folders.
    """
    glob_pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in src.glob(glob_pattern)
        if p.is_file()
        and p.suffix.lower() in _IMAGE_EXTS
        and faces not in p.parents                       # skip Faces folder
        and p.parent != faces
        and not any(part in _SKIP_FOLDERS                # skip Others / Group
                    for part in p.relative_to(src).parts)
    )


# ── HTML report ───────────────────────────────────────────────────────────────

_THUMB = 150


def _b64_thumb(path: Path, bbox: list[int] | None) -> str:
    try:
        with Image.open(path) as im:
            im  = ImageOps.exif_transpose(im).convert("RGB")
            iw, ih = im.size
            im2 = im.copy()
            im2.thumbnail((_THUMB, _THUMB), Image.LANCZOS)
            if bbox:
                sx, sy = im2.width / iw, im2.height / ih
                x1, y1, x2, y2 = bbox
                ImageDraw.Draw(im2).rectangle(
                    [int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)],
                    outline="#00FF66", width=2,
                )
            buf = io.BytesIO()
            im2.save(buf, format="JPEG", quality=75)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def write_html_report(
    matches: list[dict],
    out_path: Path,
    threshold: float,
    dry_run: bool,
    recursive: bool,
) -> None:
    by_person: dict[str, list[dict]] = defaultdict(list)
    for m in matches:
        by_person[m["person"]].append(m)

    mode_label = "DRY RUN — preview only, no files moved" if dry_run else "⚠ EXECUTED — files have been moved"
    mode_color = "#f0c040" if dry_run else "#ff6060"

    # Sort order: named persons → Group → Others
    special = {"Group", "Others"}
    sorted_persons = (
        sorted(p for p in by_person if p not in special)
        + [p for p in ("Group", "Others") if p in by_person]
    )

    cards_html = []
    for person in sorted_persons:
        group = sorted(by_person[person], key=lambda m: -m["score"])
        folder_color = (
            "#d88ffc" if person == "Group" else
            "#888888" if person == "Others" else
            "#6bf"
        )
        imgs  = "".join(
            f"""<div class="card">
              <img src="data:image/jpeg;base64,{_b64_thumb(m['img_path'], m['bbox'])}"
                   title="{m['img_path'].name}">
              <div class="fname">{m['img_path'].name}</div>
              <div class="score">★ {m['score']:.3f}</div>
            </div>"""
            for m in group
        )
        cards_html.append(f"""
        <section>
          <h2 style="color:{folder_color}">{person}
            <span class="count">{len(group)} image{"s" if len(group)!=1 else ""}</span>
          </h2>
          <div class="grid">{imgs}</div>
        </section>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Face Sort Report</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0 }}
    body  {{ background:#0f0f17; color:#e0e0e0; font-family:system-ui,sans-serif; padding:24px }}
    h1    {{ color:#6bf; font-size:1.6rem; margin-bottom:6px }}
    .meta {{ color:#888; font-size:.9rem; margin-bottom:28px }}
    .mode {{ color:{mode_color}; font-weight:bold }}
    section {{ background:#16161f; border:1px solid #2a2a3f; border-radius:10px;
               padding:16px; margin-bottom:20px }}
    h2    {{ color:#cde; font-size:1.1rem; margin-bottom:14px }}
    .count{{ font-weight:normal; color:#888; font-size:.85rem; margin-left:8px }}
    .grid {{ display:flex; flex-wrap:wrap; gap:10px }}
    .card {{ text-align:center }}
    .card img {{ width:{_THUMB}px; height:{_THUMB}px; object-fit:cover; border-radius:6px }}
    .fname{{ font-size:9px; color:#aaa; max-width:{_THUMB}px; word-break:break-all; margin-top:4px }}
    .score{{ font-size:11px; color:#4fc; margin-top:2px }}
  </style>
</head>
<body>
  <h1>Face Sort Report</h1>
  <p class="meta">
    Mode: <span class="mode">{mode_label}</span> &nbsp;|&nbsp;
    Threshold: {threshold} &nbsp;|&nbsp;
    Recursive: {'yes' if recursive else 'no'} &nbsp;|&nbsp;
    Matched: {sum(1 for m in matches if m['person'] not in ('Others','Group'))} &nbsp;|&nbsp;
    Group: {sum(1 for m in matches if m['person']=='Group')} &nbsp;|&nbsp;
    Others: {sum(1 for m in matches if m['person']=='Others')} &nbsp;|&nbsp;
    Persons: {len([p for p in by_person if p not in ('Others','Group')])}
  </p>
  {"".join(cards_html)}
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(matches: list[dict]) -> None:
    by_person: dict[str, list[dict]] = defaultdict(list)
    for m in matches:
        by_person[m["person"]].append(m)

    table = Table(
        title="Sort summary",
        show_lines=False,
        header_style="bold cyan",
        min_width=62,
    )
    table.add_column("Folder",   style="green")
    table.add_column("Images",   justify="right", style="bold white")
    table.add_column("Avg sim",  justify="right", style="yellow")
    table.add_column("Min",      justify="right", style="dim")
    table.add_column("Max",      justify="right", style="dim")

    # Named persons first (sorted), then Group, then Others
    special = {"Group", "Others"}
    for person in sorted(p for p in by_person if p not in special):
        scores = [m["score"] for m in by_person[person]]
        table.add_row(
            person,
            str(len(scores)),
            f"{sum(scores)/len(scores):.3f}",
            f"{min(scores):.3f}",
            f"{max(scores):.3f}",
        )
    for label, color in (("Group", "magenta"), ("Others", "dim")):
        if label in by_person:
            scores = [m["score"] for m in by_person[label]]
            avg_s  = sum(scores) / len(scores)
            table.add_row(
                f"[{color}]{label}[/{color}]",
                f"[{color}]{len(scores)}[/{color}]",
                f"[{color}]{avg_s:.3f}[/{color}]",
                f"[{color}]{min(scores):.3f}[/{color}]",
                f"[{color}]{max(scores):.3f}[/{color}]",
            )

    console.print(table)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog        = "face_sorter",
        description = "Sort images into per-person folders using ArcFace recognition.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Examples:
  python face_sorter.py                           # fully interactive
  python face_sorter.py --execute                 # interactive, then move files
  python face_sorter.py --src "D:\\Photos" --faces "D:\\Refs" --execute
  python face_sorter.py --recursive               # include subfolders
  python face_sorter.py --threshold 0.45          # stricter matching
""",
    )
    ap.add_argument(
        "--src",       type=Path,  default=None,
        help="Folder of images to sort (prompted if omitted)",
    )
    ap.add_argument(
        "--faces",     type=Path,  default=None,
        help="Folder of reference photos — one per person (prompted if omitted)",
    )
    ap.add_argument(
        "--models",    type=str,   default=None,
        help=f"Root folder containing models\\buffalo_l\\ (default: {_DEFAULT_MODELS_ROOT})",
    )
    ap.add_argument(
        "--report",    type=Path,  default=None,
        help="Path for the HTML report (default: <src>\\face_sort_report.html)",
    )
    ap.add_argument(
        "--threshold", type=float, default=_DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold 0–1 (default {_DEFAULT_THRESHOLD})",
    )
    ap.add_argument(
        "--recursive", action="store_true", default=False,
        help="Scan subfolders of source recursively (default: main folder only)",
    )
    ap.add_argument(
        "--execute",   action="store_true",
        help="Move matched files into person subfolders (default: dry run)",
    )
    return ap


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Track whether --recursive was explicitly passed so interactive_setup
    # knows not to ask again.
    args.recursive_set = "--recursive" in sys.argv

    # Fill any missing paths / confirm settings interactively.
    args = interactive_setup(args)

    dry_run = not args.execute
    src     = Path(args.src)
    faces   = Path(args.faces)
    report  = Path(args.report)

    # ── Load models ──────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("Loading models", style="cyan"))
    with console.status("[cyan]Loading SCRFD detector + ArcFace recogniser …[/cyan]"):
        t0  = time.time()
        app = load_face_app(args.models)
    console.print(f"  [green]✓[/green]  Models ready  ({time.time()-t0:.1f}s)\n")

    # ── Load reference faces ─────────────────────────────────────────────────
    console.print(Rule("Reference faces", style="cyan"))
    refs = load_references(app, faces)
    if not refs:
        console.print("[red]  ✗  No reference faces loaded.[/red]")
        sys.exit(1)
    console.print()

    # ── Collect source images ────────────────────────────────────────────────
    all_images = collect_images(src, args.recursive, faces)
    if not all_images:
        console.print(f"[yellow]  ⚠  No images found in: {src}[/yellow]")
        sys.exit(0)

    console.print(Rule("Processing images", style="cyan"))
    scan_scope = "recursively" if args.recursive else "main folder only"
    console.print(f"  [cyan]{len(all_images)}[/cyan] images found  ({scan_scope})\n")

    # ── Load embedding cache ────────────────────────────────────────────────
    cache_path = src / "face_sort_cache.json"
    emb_cache  = _load_cache(cache_path)
    cache_size_before = len(emb_cache)
    cache_hits  = 0

    console.print("  [dim](Press [bold]ESC[/bold] at any time to abort and save progress)[/dim]\n")

    # ── Match loop ───────────────────────────────────────────────────────────
    # all_matches holds every image result — person match, Group, or Others.
    # "person" field doubles as destination subfolder name.
    all_matches: list[dict] = []
    t0 = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=36),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=6,
    ) as progress:
        task = progress.add_task("[cyan]Matching faces[/cyan]", total=len(all_images))
        for img_path in all_images:
            if _tm.is_terminating():
                console.print("\n  [yellow]⚠  ESC pressed — saving cache and stopping.[/yellow]")
                break
            had_cached = _get_cached_embedding(emb_cache, img_path)[0] is not None
            hits, face_count = match_image(img_path, app, refs, args.threshold, cache=emb_cache)
            if had_cached:
                cache_hits += 1

            # ── Categorise  (female-priority grouping) ──────────────────
            # Rule: Group ONLY when 2+ RECOGNIZED female faces present.
            #   • 1 recognized female + any males (known/unknown) → her folder.
            #   • 1 recognized female + blurred/unrecognized       → her folder.
            #   • 2+ recognized females                            → Group.
            #   • 0 recognized females, 1+ recognized males        → best male's folder.
            #   • 0 recognized at all                              → Others.
            female_hits   = [h for h in hits if h.get("is_female", False)]
            other_hits    = [h for h in hits if not h.get("is_female", False)]
            n_rec_females = len(female_hits)

            if n_rec_females >= 2:
                # 2+ recognized females → Group
                all_matches.append({
                    "person":     "Group",
                    "score":      round(max(h["score"] for h in female_hits), 4),
                    "bbox":       female_hits[0]["bbox"],
                    "img_path":   img_path,
                    "face_count": face_count,
                })
            elif n_rec_females == 1:
                # Exactly 1 recognized female → her folder (female takes priority over any males)
                all_matches.append(female_hits[0])
            elif other_hits:
                # No recognized females; pick highest-scoring recognized male
                best_male = max(other_hits, key=lambda h: h["score"])
                all_matches.append(best_male)
            else:
                # No recognized faces at all → Others
                all_matches.append({
                    "person":     "Others",
                    "score":      0.0,
                    "bbox":       [0, 0, 0, 0],
                    "img_path":   img_path,
                    "face_count": face_count,
                })

            n_matched = sum(1 for m in all_matches if m["person"] not in ("Others", "Group"))
            n_group   = sum(1 for m in all_matches if m["person"] == "Group")
            n_others  = sum(1 for m in all_matches if m["person"] == "Others")
            progress.advance(task)
            progress.update(
                task,
                description=(
                    f"[cyan]Matching faces[/cyan]  "
                    f"[green]{n_matched} matched[/green]  "
                    f"[magenta]{n_group} group[/magenta]  "
                    f"[dim]{n_others} others[/dim]"
                ),
            )

    elapsed = time.time() - t0
    rate    = len(all_images) / elapsed if elapsed > 0 else 0
    n_matched = sum(1 for m in all_matches if m["person"] not in ("Others", "Group"))
    n_group   = sum(1 for m in all_matches if m["person"] == "Group")
    n_others  = sum(1 for m in all_matches if m["person"] == "Others")
    console.print(
        f"\n  [green]✓[/green]  Done — "
        f"[bold]{len(all_images)}[/bold] images in [bold]{elapsed:.1f}s[/bold] "
        f"([dim]{rate:.1f} img/s[/dim])  ·  "
        f"[bold green]{n_matched}[/bold green] matched  "
        f"[bold magenta]{n_group}[/bold magenta] group  "
        f"[dim]{n_others}[/dim] others\n"
    )

    # ── Save embedding cache & print stats ───────────────────────────────────
    _save_cache(emb_cache, cache_path)
    new_entries = len(emb_cache) - cache_size_before
    console.print(
        f"  [dim]Cache: {cache_hits}/{len(all_images)} hits, "
        f"{new_entries} new entries saved → {cache_path.name}[/dim]\n"
    )

    # ── Summary table ────────────────────────────────────────────────────────
    console.print(Rule("Results", style="cyan"))
    if all_matches:
        print_summary(all_matches)
    else:
        console.print("  [yellow]No matches found. Try lowering --threshold.[/yellow]")
    console.print()

    # ── Dry run: offer to continue ────────────────────────────────────────────
    if dry_run and all_matches:
        persons_only = [m for m in all_matches if m["person"] not in ("Others", "Group")]
        n_persons    = len(set(m["person"] for m in persons_only))
        console.print(Panel.fit(
            Text.assemble(
                ("Dry run complete.\n", "bold yellow"),
                (f"  {len(persons_only)} matched  ·  "
                 f"{n_group} → Group  ·  {n_others} → Others  ·  "
                 f"{n_persons} person folder(s)\n\n", "white"),
                ("Open the HTML report to review, then run again with  ", "dim"),
                ("--execute", "bold cyan"),
                ("  to move the files.", "dim"),
            ),
            border_style="yellow",
        ))

        console.print()
        if Confirm.ask("  Move files now?", default=False):
            args.execute = True
            dry_run      = False
        else:
            # Generate report from original paths (files not moved yet)
            _gen_report(all_matches, report, args.threshold, dry_run, args.recursive)
            return

    # ── Move files ────────────────────────────────────────────────────────────
    if not dry_run:
        console.print(Rule("Moving files", style="cyan"))
        moved = errors = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=36),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Moving …[/cyan]", total=len(all_matches))
            for m in all_matches:
                dest_dir = src / m["person"]   # "Kajal", "Group", or "Others"
                dest_dir.mkdir(exist_ok=True)
                dest = dest_dir / m["img_path"].name
                if dest.exists():
                    dest = dest_dir / (m["img_path"].stem + "_dup" + m["img_path"].suffix)
                try:
                    shutil.move(str(m["img_path"]), str(dest))
                    m["img_path"] = dest   # ← update to new location for report
                    moved += 1
                except Exception as exc:
                    console.print(f"  [red]ERR[/red]  {m['img_path'].name}: {exc}")
                    errors += 1
                progress.advance(task)

        dest_folders = set(m["person"] for m in all_matches)
        console.print(
            f"\n  [green]✓[/green]  {moved} files moved  ·  {errors} errors\n"
        )
        console.print(Panel.fit(
            Text.assemble(
                ("Done! ", "bold green"),
                (f"{moved} images sorted into {len(dest_folders)} folders: ", "white"),
                (", ".join(sorted(dest_folders)), "cyan"),
                (".", "white"),
            ),
            border_style="green",
        ))

    # ── Save JSON log ─────────────────────────────────────────────────────────
    # (after move so img_path reflects final location)
    log_path = report.with_suffix(".json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            [{**m, "img_path": str(m["img_path"])} for m in all_matches],
            f, indent=2,
        )
    console.print(f"  [green]✓[/green]  Log     →  {log_path}\n")

    # ── HTML report — generated AFTER move so thumbnails read from new paths ──
    _gen_report(all_matches, report, args.threshold, dry_run, args.recursive)

    console.print()


def _gen_report(
    all_matches: list[dict],
    report: Path,
    threshold: float,
    dry_run: bool,
    recursive: bool,
) -> None:
    """Generate JSON log + HTML report then print the link. Runs synchronously
    so thumbnails are always read AFTER files have been moved to their final
    destinations (fixes the blank-thumbnail race condition)."""
    console.print(f"  [dim]Generating HTML report …[/dim]")
    write_html_report(all_matches, report, threshold, dry_run, recursive)
    console.print(f"  [green]✓[/green]  Report  →  [link={report.as_uri()}]{report}[/link]\n")


if __name__ == "__main__":
    main()
