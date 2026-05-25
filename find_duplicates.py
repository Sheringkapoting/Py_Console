#!/usr/bin/env python3
"""
find_duplicates.py  ─  Duplicate image detector & mover
========================================================
Two-pass detection:
  Pass 1 — SHA-256 hash    : byte-identical files  (exact duplicates)
  Pass 2 — Perceptual hash : visually identical files (re-saves, slight crops,
                              watermark variants, compression artefacts)

Primary image selection per group:
  1. Highest pixel count  (width × height)   ← best quality
  2. Smallest file size   (tiebreaker)        ← most efficient encoding

Duplicates are moved to  <src>/Duplicate/  and renamed  {stem}_dup{N}{ext}.

Requirements:
    pip install Pillow imagehash rich tqdm

Usage:
    python find_duplicates.py                              # interactive
    python find_duplicates.py --src "D:\\Photos"           # direct, dry run
    python find_duplicates.py --src "D:\\Photos" --execute # move files
    python find_duplicates.py --threshold 6                # stricter (default 8)
    python find_duplicates.py --exact-only                 # skip perceptual pass
    python find_duplicates.py --recursive                  # include subfolders
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

# ── Dependency check ──────────────────────────────────────────────────────────

def _check_deps() -> None:
    missing = []
    for pkg, mod in [("Pillow", "PIL"), ("imagehash", "imagehash"),
                     ("rich", "rich"), ("tqdm", "tqdm")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n  ✗  Missing packages. Run:\n\n     pip install {' '.join(missing)}\n")
        sys.exit(1)

# ── Imports (optional deps guarded so this module is safely importable) ──────

try:
    import imagehash
    _HAS_IMAGEHASH = True
except ImportError:
    imagehash = None          # type: ignore[assignment]
    _HAS_IMAGEHASH = False

from PIL import Image, ImageFile, UnidentifiedImageError
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
from tqdm import tqdm

console = Console()

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}
_DUPLICATE_FOLDER = "Duplicate"
_THUMB = 160          # thumbnail px for HTML report

# Watermark detection tunables (cross-image diff approach)
_WM_DIFF_THRESH = 20    # min per-pixel diff to count as "changed" region
_WM_MIN_PIXELS  = 200   # min changed pixels needed to attempt detection
_WM_EDGE_RATIO  = 1.3   # edge-density ratio to call one copy watermarked

# ── Union-Find for grouping ───────────────────────────────────────────────────

class _UF:
    """Path-compressed union-find (disjoint set)."""
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ── Image metadata ────────────────────────────────────────────────────────────

def _image_meta(path: Path) -> tuple[int, int, int]:
    """Returns (pixels, filesize, error).  pixels=0 on failure."""
    try:
        with Image.open(path) as im:
            w, h = im.size
        return w * h, path.stat().st_size, 0
    except Exception:
        return 0, path.stat().st_size, 1


# ── Hashing ───────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _phash(path: Path):
    if not _HAS_IMAGEHASH:
        return None
    try:
        with Image.open(path) as im:
            return imagehash.phash(im.convert("RGB"))
    except (UnidentifiedImageError, Exception):
        return None


# ── Image collection ──────────────────────────────────────────────────────────

def collect_images(src: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in src.glob(pattern)
        if p.is_file()
        and p.suffix.lower() in _IMAGE_EXTS
        and _DUPLICATE_FOLDER not in p.relative_to(src).parts
    )


# ── Watermark detection ───────────────────────────────────────────────────────

def _watermark_scores(group: list[Path]) -> dict[Path, int]:
    """
    Cross-compare every pair in the group via pixel diff + edge density.

    For each pair (A, B):
      1. Resize both to same size, convert to grayscale.
      2. Compute per-pixel diff; isolate "changed" region (diff > threshold).
      3. Compare edge density of A vs B inside that region.
         Higher edge density = extra structure = likely watermark on that copy.
      4. Vote for the copy with higher edge density as "watermarked".

    Returns {path: vote_count}.  0 votes = clean (or indeterminate).
    Works for watermarks placed anywhere — centre, corner, diagonal.
    """
    import numpy as np
    from PIL import ImageFilter, ImageOps

    votes: dict[Path, int] = {p: 0 for p in group}

    # ── load & cache (resize to smallest common size for fair comparison) ──
    try:
        dims = []
        for p in group:
            try:
                with Image.open(p) as im:
                    dims.append(im.size)
            except Exception:
                pass
        if not dims:
            return votes
        min_w = min(d[0] for d in dims)
        min_h = min(d[1] for d in dims)

        cache: dict[Path, tuple] = {}
        for p in group:
            try:
                with Image.open(p) as im:
                    im = (ImageOps.exif_transpose(im)
                          .convert("L")
                          .resize((min_w, min_h), Image.LANCZOS))
                    arr   = np.array(im, dtype=np.float32)
                    edges = np.array(im.filter(ImageFilter.FIND_EDGES),
                                     dtype=np.float32)
                    cache[p] = (arr, edges)
            except Exception:
                pass

        paths = [p for p in group if p in cache]
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                pa, pb = paths[i], paths[j]
                arr_a, edges_a = cache[pa]
                arr_b, edges_b = cache[pb]

                diff      = np.abs(arr_a - arr_b)
                mask      = diff > _WM_DIFF_THRESH
                n_changed = int(mask.sum())
                if n_changed < _WM_MIN_PIXELS:
                    continue   # images too similar — no watermark signal

                ea = float(edges_a[mask].mean())
                eb = float(edges_b[mask].mean())

                if ea > eb * _WM_EDGE_RATIO:
                    votes[pa] += 1   # A has extra structure → A watermarked
                elif eb > ea * _WM_EDGE_RATIO:
                    votes[pb] += 1   # B watermarked
    except Exception:
        pass

    return votes


# ── Primary selector ──────────────────────────────────────────────────────────

def _select_primary(group: list[Path]) -> Path:
    """
    Select best image to keep.  Priority (ascending sort key = preferred):
      1. Watermark votes  (0 = clean/indeterminate → wins; higher = watermarked)
      2. Highest pixel count  (negated — larger pixels = smaller key)
      3. Smallest file size   (tiebreaker for equal resolution)
    """
    try:
        wm_votes = _watermark_scores(group)
    except Exception:
        wm_votes = {p: 0 for p in group}

    def key(p: Path) -> tuple[int, int, int]:
        pixels, size, _ = _image_meta(p)
        return (wm_votes.get(p, 0), -pixels, size)

    return min(group, key=key)


# ── Dedup logic ───────────────────────────────────────────────────────────────

def find_duplicate_groups(
    images: list[Path],
    threshold: int,
    exact_only: bool,
) -> list[dict[str, Any]]:
    """
    Returns list of group dicts, each with:
      - "paths":      list[Path]   (2+ files that are duplicates)
      - "match_type": "exact" | "near"
    Groups with only one image (unique) are excluded.
    """
    n = len(images)
    uf = _UF(n)
    exact_indices: set[int] = set()   # indices involved in byte-identical groups

    # ── Pass 1: SHA-256 exact match ──────────────────────────────────────
    console.print(Rule("Pass 1 / SHA-256 exact matching", style="cyan"))
    sha_map: dict[str, list[int]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("[cyan]Hashing files[/cyan]", total=n)
        for i, p in enumerate(images):
            h = _sha256(p)
            if h:
                sha_map.setdefault(h, []).append(i)
            prog.advance(task)

    n_exact_groups = 0
    for idxs in sha_map.values():
        if len(idxs) > 1:
            n_exact_groups += 1
            exact_indices.update(idxs)
            for j in idxs[1:]:
                uf.union(idxs[0], j)

    console.print(f"  [green]✓[/green]  {n_exact_groups} exact-duplicate group(s) found\n")

    if exact_only:
        raw = _uf_to_groups(uf, n, images)
        return [{"paths": g, "match_type": "exact"} for g in raw]

    # ── Pass 2: perceptual hash near-match ───────────────────────────────
    console.print(Rule("Pass 2 / Perceptual hash near-match", style="cyan"))
    console.print(f"  [dim]Hamming threshold: {threshold} / 64 bits[/dim]\n")

    phashes: list[Optional[imagehash.ImageHash]] = [None] * n

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("[cyan]Computing pHash[/cyan]", total=n)
        for i, p in enumerate(images):
            phashes[i] = _phash(p)
            prog.advance(task)

    # Compare all pairs — O(N²) but each op is a fast 64-bit XOR popcount.
    # Bucket by first 2 hex chars of hash to reduce comparisons by ~256×.
    near_pairs = 0
    bucket: dict[str, list[int]] = {}
    for i, h in enumerate(phashes):
        if h is None:
            continue
        key = str(h)[:2]
        bucket.setdefault(key, []).append(i)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=36),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("[cyan]Comparing hashes[/cyan]", total=len(bucket))
        for idxs in bucket.values():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    ia, ib = idxs[a], idxs[b]
                    ha, hb = phashes[ia], phashes[ib]
                    if ha is None or hb is None:
                        continue
                    if uf.find(ia) == uf.find(ib):
                        continue   # already same group
                    if (ha - hb) <= threshold:
                        uf.union(ia, ib)
                        near_pairs += 1
            prog.advance(task)

    console.print(f"\n  [green]✓[/green]  {near_pairs} additional near-duplicate pair(s) merged\n")

    # Tag each final group as exact or near
    path_to_idx = {p: i for i, p in enumerate(images)}
    raw = _uf_to_groups(uf, n, images)
    result: list[dict[str, Any]] = []
    for group in raw:
        idxs = {path_to_idx[p] for p in group}
        match_type = "exact" if idxs <= exact_indices else "near"
        result.append({"paths": group, "match_type": match_type})
    return result


def _uf_to_groups(uf: _UF, n: int, images: list[Path]) -> list[list[Path]]:
    from collections import defaultdict
    clusters: dict[int, list[Path]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(images[i])
    return [v for v in clusters.values() if len(v) > 1]


# ── Destination naming ────────────────────────────────────────────────────────

def _dup_dest(dup_dir: Path, path: Path) -> Path:
    """Return a unique path inside dup_dir with _dup[N] suffix."""
    stem, suffix = path.stem, path.suffix
    candidate = dup_dir / f"{stem}_dup{suffix}"
    n = 1
    while candidate.exists():
        candidate = dup_dir / f"{stem}_dup{n}{suffix}"
        n += 1
    return candidate


# ── Thumbnail helper ─────────────────────────────────────────────────────────

def _b64_thumb(path: Path) -> str:
    """Return base64 JPEG thumbnail string, or '' on failure."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((_THUMB, _THUMB), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=75)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


# ── HTML report ───────────────────────────────────────────────────────────────

def write_html_report(
    groups: list[dict[str, Any]],
    out_path: Path,
    threshold: int,
    dry_run: bool,
    recursive: bool,
    src: Path,
) -> None:
    """Generate dark-theme HTML report with Exact and Near-match sections."""

    exact_groups = [g for g in groups if g["match_type"] == "exact"]
    near_groups  = [g for g in groups if g["match_type"] == "near"]

    total_exact = sum(len(g["paths"]) - 1 for g in exact_groups)
    total_near  = sum(len(g["paths"]) - 1 for g in near_groups)
    total_dups  = total_exact + total_near

    mode_label = "DRY RUN — preview only, no files moved" if dry_run else "⚠ EXECUTED — files have been moved"
    mode_color = "#f0c040" if dry_run else "#ff6060"

    def _card(p: Path, is_primary: bool) -> str:
        thumb = _b64_thumb(p)
        img_src = f"data:image/jpeg;base64,{thumb}" if thumb else ""
        w = h = 0
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            pass
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        label       = "PRIMARY" if is_primary else "DUP"
        badge_bg    = "#00c853" if is_primary else "#d32f2f"
        border_col  = "#00e676" if is_primary else "#ff5252"
        return (
            f'<div class="card">'
            f'<span class="badge" style="background:{badge_bg}">{label}</span>'
            f'<img src="{img_src}" title="{p.name}" '
            f'     style="border:2px solid {border_col}">'
            f'<div class="fname">{p.name}</div>'
            f'<div class="fmeta">{w}×{h} · {size_kb:.1f} KB</div>'
            f'</div>'
        )

    def _render_group(g_info: dict) -> str:
        paths   = g_info["paths"]
        primary = _select_primary(paths)
        cards   = "".join(_card(p, p == primary) for p in paths)
        return f'<div class="group">{cards}</div>'

    def _render_section(title: str, grps: list[dict], color: str) -> str:
        if not grps:
            return ""
        n_dups  = sum(len(g["paths"]) - 1 for g in grps)
        content = "".join(_render_group(g) for g in sorted(grps, key=lambda g: -len(g["paths"])))
        return (
            f'<section>'
            f'<h2 style="color:{color}">{title}'
            f'  <span class="count">{len(grps)} group(s) · {n_dups} duplicate(s)</span>'
            f'</h2>'
            f'{content}'
            f'</section>'
        )

    exact_sec = _render_section(
        "🔒 Exact Duplicates (byte-identical)", exact_groups, "#4fc3f7"
    )
    near_sec  = _render_section(
        f"🔍 Near Duplicates (Hamming ≤ {threshold})", near_groups, "#ce93d8"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Duplicate Finder Report</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0 }}
    body    {{ background:#0f0f17; color:#e0e0e0;
               font-family:system-ui,sans-serif; padding:24px }}
    h1      {{ color:#6bf; font-size:1.6rem; margin-bottom:6px }}
    .meta   {{ color:#888; font-size:.9rem; margin-bottom:28px; line-height:2 }}
    .mode   {{ color:{mode_color}; font-weight:bold }}
    section {{ background:#16161f; border:1px solid #2a2a3f; border-radius:10px;
               padding:16px 20px; margin-bottom:20px }}
    h2      {{ font-size:1.1rem; margin-bottom:14px }}
    .count  {{ font-weight:normal; color:#888; font-size:.85rem; margin-left:8px }}
    .group  {{ display:flex; flex-wrap:wrap; gap:14px; margin-bottom:14px;
               padding:10px; background:#1e1e2a; border-radius:8px }}
    .card   {{ text-align:center }}
    .badge  {{ font-size:9px; font-weight:bold; color:#fff; padding:2px 6px;
               border-radius:3px; margin-bottom:4px; display:inline-block }}
    .card img {{ width:{_THUMB}px; height:{_THUMB}px; object-fit:cover;
                 border-radius:6px; display:block }}
    .fname  {{ font-size:9px; color:#aaa; max-width:{_THUMB}px;
               word-break:break-all; margin-top:4px }}
    .fmeta  {{ font-size:9px; color:#4fc; margin-top:2px }}
  </style>
</head>
<body>
  <h1>🗂 Duplicate Image Finder Report</h1>
  <p class="meta">
    Source: {src}<br>
    Mode: <span class="mode">{mode_label}</span><br>
    Threshold: {threshold} / 64 bits &nbsp;|&nbsp;
    Recursive: {'yes' if recursive else 'no'} &nbsp;|&nbsp;
    Groups: {len(groups)} &nbsp;|&nbsp;
    Total duplicates: {total_dups} &nbsp;|&nbsp;
    Exact: {total_exact} &nbsp;|&nbsp;
    Near: {total_near}
  </p>
  {exact_sec}
  {near_sec}
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="find_duplicates",
        description="Detect exact and visually identical duplicate images and move them to Duplicate/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--src",        type=Path,  default=None,
                    help="Source image folder (prompted if omitted)")
    ap.add_argument("--threshold",  type=int,   default=8,
                    help="Perceptual hash Hamming distance 0–64 (default 8). Lower = stricter.")
    ap.add_argument("--recursive",  action="store_true",
                    help="Scan subfolders recursively")
    ap.add_argument("--exact-only", action="store_true",
                    help="Skip perceptual pass — exact byte-identical only")
    ap.add_argument("--execute",    action="store_true",
                    help="Move duplicate files (default: dry run — no files moved)")
    ap.add_argument("--report",     type=Path,  default=None,
                    help="HTML report output path (default: <src>/duplicate_report.html)")
    return ap


def main() -> None:
    _check_deps()
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    parser = build_parser()
    args   = parser.parse_args()

    console.print()
    console.print(Panel.fit(
        Text.assemble(
            ("Duplicate Image Finder", "bold cyan"),
            ("  ·  ", "dim"),
            ("SHA-256 exact  +  perceptual hash near-match", "dim"),
        ),
        border_style="cyan",
    ))
    console.print()

    # ── Source folder ────────────────────────────────────────────────────
    if args.src is None:
        while True:
            raw = Prompt.ask("  [cyan]Source folder[/cyan]").strip().strip('"').strip("'")
            p   = Path(raw)
            if p.exists() and p.is_dir():
                args.src = p
                break
            console.print(f"    [red]✗  Not found:[/red] {p}")
    else:
        if not args.src.exists() or not args.src.is_dir():
            console.print(f"[red]  ✗  Source folder not found: {args.src}[/red]")
            sys.exit(1)

    src     = Path(args.src)
    dry_run = not args.execute
    report  = args.report if args.report else src / "duplicate_report.html"

    console.print(f"  [cyan]Source     [/cyan] : {src}")
    console.print(f"  [cyan]Mode       [/cyan] : [bold {'yellow' if dry_run else 'red'}]"
                  f"{'DRY RUN — no files moved' if dry_run else 'EXECUTE — files will be moved'}[/bold "
                  f"{'yellow' if dry_run else 'red'}]")
    console.print(f"  [cyan]Threshold  [/cyan] : {args.threshold} / 64 bits")
    console.print(f"  [cyan]Recursive  [/cyan] : {'yes' if args.recursive else 'no'}")
    console.print(f"  [cyan]Exact only [/cyan] : {'yes' if args.exact_only else 'no'}")
    console.print(f"  [cyan]Report     [/cyan] : {report}")
    console.print()

    # ── Collect ──────────────────────────────────────────────────────────
    console.print(Rule("Collecting images", style="cyan"))
    images = collect_images(src, args.recursive)
    if not images:
        console.print(f"[yellow]  ⚠  No images found in: {src}[/yellow]")
        sys.exit(0)
    console.print(f"  [cyan]{len(images)}[/cyan] images found\n")

    # ── Detect ───────────────────────────────────────────────────────────
    groups = find_duplicate_groups(images, args.threshold, args.exact_only)

    if not groups:
        console.print(Panel.fit(
            Text.assemble(("No duplicates found.", "bold green")),
            border_style="green",
        ))
        return

    # ── Summary table ────────────────────────────────────────────────────
    total_dups  = sum(len(g["paths"]) - 1 for g in groups)
    total_bytes = 0

    console.print(Rule("Duplicate groups", style="cyan"))
    table = Table(
        show_lines=False,
        header_style="bold cyan",
        min_width=78,
    )
    table.add_column("Primary (keep)",  style="green",   max_width=36)
    table.add_column("Type",            style="cyan",    justify="center")
    table.add_column("Copies",          justify="right", style="bold white")
    table.add_column("Resolution",      justify="right", style="yellow")
    table.add_column("Size (primary)",  justify="right", style="dim")

    for g_info in sorted(groups, key=lambda g: -len(g["paths"])):
        group   = g_info["paths"]
        mtype   = g_info["match_type"]
        primary = _select_primary(group)
        pixels, size, _ = _image_meta(primary)
        dups    = [p for p in group if p != primary]
        total_bytes += sum(p.stat().st_size for p in dups)
        w = h = 0
        try:
            with Image.open(primary) as im:
                w, h = im.size
        except Exception:
            pass
        type_label = "[cyan]exact[/cyan]" if mtype == "exact" else "[magenta]near[/magenta]"
        table.add_row(
            primary.name,
            type_label,
            str(len(dups)),
            f"{w}×{h}" if w else "—",
            f"{size/1024:.1f} KB",
        )

    console.print(table)
    console.print(
        f"\n  [bold cyan]{len(groups)}[/bold cyan] group(s)  ·  "
        f"[bold]{total_dups}[/bold] duplicate file(s)  ·  "
        f"[dim]{total_bytes/1024/1024:.1f} MB recoverable[/dim]\n"
    )

    # ── Dry run offer ────────────────────────────────────────────────────
    if dry_run:
        # Generate report from current (pre-move) paths
        console.print(f"  [dim]Generating HTML report …[/dim]")
        write_html_report(groups, report, args.threshold, True, args.recursive, src)
        console.print(f"  [green]✓[/green]  Report → [link={report.as_uri()}]{report}[/link]\n")

        console.print(Panel.fit(
            Text.assemble(
                ("Dry run complete.\n", "bold yellow"),
                (f"  {total_dups} duplicates would be moved to {_DUPLICATE_FOLDER}/.\n\n", "white"),
                ("Run again with  ", "dim"),
                ("--execute", "bold cyan"),
                ("  to move files.", "dim"),
            ),
            border_style="yellow",
        ))
        console.print()
        if not Confirm.ask("  Move duplicates now?", default=False):
            return
        dry_run = False

    # ── Move duplicates ──────────────────────────────────────────────────
    console.print(Rule("Moving duplicates", style="cyan"))
    dup_dir = src / _DUPLICATE_FOLDER
    dup_dir.mkdir(exist_ok=True)

    moved = errors = 0
    dest_map: dict[Path, Path] = {}   # original_path → moved_path (for report)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=36),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("[cyan]Moving …[/cyan]", total=total_dups)
        for g_info in groups:
            group   = g_info["paths"]
            primary = _select_primary(group)
            for dup in group:
                if dup == primary:
                    continue
                dest = _dup_dest(dup_dir, dup)
                try:
                    shutil.move(str(dup), str(dest))
                    dest_map[dup] = dest
                    moved += 1
                except Exception as exc:
                    console.print(f"  [red]ERR[/red]  {dup.name}: {exc}")
                    errors += 1
                prog.advance(task)

    # Update paths in groups to reflect new locations (for thumbnail reading)
    for g_info in groups:
        g_info["paths"] = [dest_map.get(p, p) for p in g_info["paths"]]

    console.print(
        f"\n  [green]✓[/green]  {moved} duplicates moved to [cyan]{dup_dir}[/cyan]  ·  {errors} errors\n"
    )

    # ── HTML report ──────────────────────────────────────────────────────
    console.print(f"  [dim]Generating HTML report …[/dim]")
    write_html_report(groups, report, args.threshold, False, args.recursive, src)
    console.print(f"  [green]✓[/green]  Report → [link={report.as_uri()}]{report}[/link]\n")

    console.print(Panel.fit(
        Text.assemble(
            ("Done! ", "bold green"),
            (f"{moved} duplicate files moved. ", "white"),
            (f"Primaries kept in original location.", "dim"),
        ),
        border_style="green",
    ))
    console.print()


if __name__ == "__main__":
    main()
