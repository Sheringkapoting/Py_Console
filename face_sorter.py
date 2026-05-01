#!/usr/bin/env python3
"""
face_sorter.py  ─  ArcFace-powered face recognition image sorter
=================================================================
Detects faces in a folder of images and sorts them into per-person
subfolders using InsightFace's buffalo_l models (SCRFD + ArcFace).

All folder paths are asked interactively at runtime — no code edits needed.

Requirements (install once):
    pip install insightface onnxruntime opencv-python pillow tqdm rich

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
_DEFAULT_MIN_DET     = 0.50
_IMAGE_EXTS          = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import base64
import io
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ── ESC key termination support ────────────────────────────────────────────────
try:
    import msvcrt
    _HAS_MSVC = True
except ImportError:
    _HAS_MSVC = False

class TerminationManager:
    """Manages graceful termination on ESC key press."""
    def __init__(self):
        self._should_terminate = False
    
    def check_esc(self) -> bool:
        """Check if ESC key was pressed (Windows only)."""
        if not _HAS_MSVC:
            return False
        try:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\x1b':  # ESC key
                    self._should_terminate = True
                    return True
        except Exception:
            pass
        return False
    
    @property
    def should_terminate(self) -> bool:
        return self._should_terminate

# Global termination manager
termination_manager = TerminationManager()


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
    app = FaceAnalysis(
        name      = "buffalo_l",
        root      = models_root,
        providers = ["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
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

def match_image(
    img_path: Path,
    app,
    refs: dict[str, list[np.ndarray]],
    threshold: float,
    min_det: float = _DEFAULT_MIN_DET,
) -> list[dict]:
    """Detect faces in one image and return all matches above threshold."""
    img = cv2.imread(str(img_path))
    if img is None:
        return []

    faces = app.get(img)
    if not faces:
        return []

    confirmed: list[dict]   = []
    seen_persons: set[str]  = set()

    for face in faces:
        if face.det_score < min_det:
            continue

        best_person: Optional[str] = None
        best_score = -1.0

        for person, embeddings in refs.items():
            if person in seen_persons:
                continue
            score = max(cosine_sim(face.embedding, e) for e in embeddings)
            if score > best_score:
                best_score, best_person = score, person

        if best_person and best_score >= threshold:
            confirmed.append({
                "person":   best_person,
                "score":    round(best_score, 4),
                "bbox":     face.bbox.astype(int).tolist(),
                "img_path": img_path,
            })
            seen_persons.add(best_person)

    return confirmed


# ── Image collection ──────────────────────────────────────────────────────────

def collect_images(src: Path, recursive: bool, faces: Path) -> list[Path]:
    """
    Gather all image files from src.
    - recursive=False : main folder only
    - recursive=True  : all subfolders, but always skips the Faces folder
                        and any destination subfolders that look like person names.
    """
    glob_pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in src.glob(glob_pattern)
        if p.is_file()
        and p.suffix.lower() in _IMAGE_EXTS
        and faces not in p.parents        # never scan the Faces folder
        and p.parent != faces
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
    unmatched_images: list[Path] | None = None,
) -> None:
    by_person: dict[str, list[dict]] = defaultdict(list)
    for m in matches:
        by_person[m["person"]].append(m)

    mode_label = "DRY RUN — preview only, no files moved" if dry_run else "⚠ EXECUTED — files have been moved"
    mode_color = "#f0c040" if dry_run else "#ff6060"
    
    # Count total persons including Others if there are unmatched images
    total_persons = len(by_person)
    if unmatched_images:
        total_persons += 1  # Others folder

    cards_html = []
    # Add matched persons sections
    for person in sorted(by_person):
        group = sorted(by_person[person], key=lambda m: -m["score"])
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
          <h2>{person}
            <span class="count">{len(group)} image{"s" if len(group)!=1 else ""}</span>
          </h2>
          <div class="grid">{imgs}</div>
        </section>""")
    
    # Add Others section for unmatched images
    if unmatched_images:
        unmatched_imgs = "".join(
            f"""<div class="card">
              <img src="data:image/jpeg;base64,{_b64_thumb(img_path, None)}"
                   title="{img_path.name}">
              <div class="fname">{img_path.name}</div>
              <div class="score">—</div>
            </div>"""
            for img_path in sorted(unmatched_images)
        )
        cards_html.append(f"""
        <section>
          <h2>Others
            <span class="count">{len(unmatched_images)} image{"s" if len(unmatched_images)!=1 else ""}</span>
          </h2>
          <div class="grid">{unmatched_imgs}</div>
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
    Total matches: {len(matches)} &nbsp;|&nbsp;
    Unmatched: {len(unmatched_images) if unmatched_images else 0} &nbsp;|&nbsp;
    Persons: {total_persons}
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
        title="Match summary",
        show_lines=False,
        header_style="bold cyan",
        min_width=58,
    )
    table.add_column("Person",  style="green")
    table.add_column("Images",  justify="right",  style="bold white")
    table.add_column("Avg sim", justify="right",  style="yellow")
    table.add_column("Min",     justify="right",  style="dim")
    table.add_column("Max",     justify="right",  style="dim")

    for person in sorted(by_person):
        scores = [m["score"] for m in by_person[person]]
        table.add_row(
            person,
            str(len(scores)),
            f"{sum(scores)/len(scores):.3f}",
            f"{min(scores):.3f}",
            f"{max(scores):.3f}",
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

    # ── Match loop ───────────────────────────────────────────────────────────
    all_matches: list[dict] = []
    matched_images: set[Path] = set()  # Track images that had matches
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
            # Check for ESC key termination
            if termination_manager.check_esc():
                console.print("\n[yellow]⚠ Face matching cancelled by user (ESC key).[/yellow]")
                console.print(f"[dim]Processed {len(all_matches)} matches before cancellation.[/dim]")
                break
                
            hits = match_image(img_path, app, refs, args.threshold)
            if hits:
                matched_images.add(img_path)
            all_matches.extend(hits)
            progress.advance(task)
            progress.update(
                task,
                description=(
                    f"[cyan]Matching faces[/cyan]  "
                    f"[green]{len(all_matches)} match{'es' if len(all_matches)!=1 else ''}[/green]"
                ),
            )

    # ── Identify unmatched images ───────────────────────────────────────────────
    unmatched_images = [img for img in all_images if img not in matched_images]
    
    elapsed = time.time() - t0
    rate    = len(all_images) / elapsed if elapsed > 0 else 0
    console.print(
        f"\n  [green]✓[/green]  Done — "
        f"[bold]{len(all_images)}[/bold] images in [bold]{elapsed:.1f}s[/bold] "
        f"([dim]{rate:.1f} img/s[/dim])  ·  "
        f"[bold cyan]{len(all_matches)}[/bold cyan] matches  ·  "
        f"[bold yellow]{len(unmatched_images)}[/bold yellow] unmatched\n"
    )

    # ── Summary table ────────────────────────────────────────────────────────
    console.print(Rule("Results", style="cyan"))
    if all_matches:
        print_summary(all_matches)
    else:
        console.print("  [yellow]No matches found. Try lowering --threshold.[/yellow]")
    
    if unmatched_images:
        console.print(f"  [yellow]Unmatched images[/yellow]: {len(unmatched_images)}")
    console.print()

    # ── Save JSON log ────────────────────────────────────────────────────────
    log_path = report.with_suffix(".json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            [{**m, "img_path": str(m["img_path"])} for m in all_matches],
            f, indent=2,
        )

    # ── HTML report ──────────────────────────────────────────────────────────
    with console.status("[cyan]Generating HTML report …[/cyan]"):
        write_html_report(all_matches, report, args.threshold, dry_run, args.recursive, unmatched_images)
    console.print(f"  [green]✓[/green]  Report  →  [link={report.as_uri()}]{report}[/link]")
    console.print(f"  [green]✓[/green]  Log     →  {log_path}\n")

    # ── Dry run: offer to continue ────────────────────────────────────────────
    total_files_to_move = len(all_matches) + len(unmatched_images)
    if dry_run and total_files_to_move > 0:
        matched_persons = len(set(m['person'] for m in all_matches)) if all_matches else 0
        console.print(Panel.fit(
            Text.assemble(
                ("Dry run complete.\n", "bold yellow"),
                (f"  {len(all_matches)} matches across "
                 f"{matched_persons} persons.\n", "white"),
                (f"  {len(unmatched_images)} unmatched images will move to Others folder.\n\n", "yellow"),
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
            console.print()
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
            # Move matched images first
            task = progress.add_task("[cyan]Moving matched files …[/cyan]", total=len(all_matches))
            for m in all_matches:
                # Check for ESC key termination
                if termination_manager.check_esc():
                    console.print("\n[yellow]⚠ File moving cancelled by user (ESC key).[/yellow]")
                    console.print(f"[dim]{moved} files moved before cancellation.[/dim]")
                    break
                    
                dest_dir = src / m["person"]
                dest_dir.mkdir(exist_ok=True)
                dest = dest_dir / m["img_path"].name
                if dest.exists():
                    dest = dest_dir / (m["img_path"].stem + "_dup" + m["img_path"].suffix)
                try:
                    shutil.move(str(m["img_path"]), str(dest))
                    moved += 1
                except Exception as exc:
                    console.print(f"  [red]ERR[/red]  {m['img_path'].name}: {exc}")
                    errors += 1
                progress.advance(task)
            
            # Move unmatched images to Others folder
            if unmatched_images:
                task_others = progress.add_task("[cyan]Moving unmatched files to Others …[/cyan]", total=len(unmatched_images))
                others_dir = src / "Others"
                others_dir.mkdir(exist_ok=True)
                
                for img_path in unmatched_images:
                    # Check for ESC key termination
                    if termination_manager.check_esc():
                        console.print("\n[yellow]⚠ File moving cancelled by user (ESC key).[/yellow]")
                        console.print(f"[dim]{moved} files moved before cancellation.[/dim]")
                        break
                        
                    dest = others_dir / img_path.name
                    if dest.exists():
                        dest = others_dir / (img_path.stem + "_dup" + img_path.suffix)
                    try:
                        shutil.move(str(img_path), str(dest))
                        moved += 1
                    except Exception as exc:
                        console.print(f"  [red]ERR[/red]  {img_path.name}: {exc}")
                        errors += 1
                    progress.advance(task_others)

        console.print(
            f"\n  [green]✓[/green]  {moved} files moved  ·  {errors} errors\n"
        )
        
        matched_persons = len(set(m['person'] for m in all_matches)) if all_matches else 0
        summary_parts = [f"{moved} images sorted"]
        if matched_persons > 0:
            summary_parts.append(f"{matched_persons} person folders")
        if len(unmatched_images) > 0:
            summary_parts.append(f"{len(unmatched_images)} to Others folder")
        
        console.print(Panel.fit(
            Text.assemble(
                ("Done! ", "bold green"),
                (" ".join(summary_parts) + ".", "white"),
            ),
            border_style="green",
        ))

    console.print()


if __name__ == "__main__":
    main()
