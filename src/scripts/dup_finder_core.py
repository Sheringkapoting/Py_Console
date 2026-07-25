#!/usr/bin/env python3
"""
dup_finder_core.py ─ shared engine for the duplicate-finder family
====================================================================
Extracted from find_duplicates.py so every duplicate-finder tool (still
images, animated GIF/WebP, video, ...) shares one Union-Find, SHA-256
exact-match pass, near-match pass, Rich UI/progress, HTML report shell,
ESC-termination wiring, and dry-run/execute move workflow.

This module knows nothing about any specific media format. Format-specific
behavior — how to hash a frame, how to pick which copy to keep, how to
render a thumbnail — is supplied by a MediaHandler implementation passed
into run_workflow().

Not meant to be run directly; import it from a per-media-type script
(e.g. find_duplicates.py, find_media_duplicates.py).
"""

from __future__ import annotations

import base64
import concurrent.futures as cf
import hashlib
import io
import os
import shutil
import sys
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Hashable, Optional, Union

_scripts_dir = str(Path(__file__).parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from common_utils import TerminationManager
del _scripts_dir

from PIL import Image
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

DUPLICATE_FOLDER = "Duplicate"
THUMB = 160          # thumbnail px for HTML report
MAX_SOURCES = 5       # max source folders accepted per run


# ── ESC-key termination ───────────────────────────────────────────────────────

def make_termination_manager() -> TerminationManager:
    """Standard ESC-abort setup, shared across all duplicate-finder tools."""
    tm = TerminationManager()
    tm.start_monitoring()
    return tm


# ── Union-Find for grouping ───────────────────────────────────────────────────

class UnionFind:
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


def _uf_to_groups(uf: UnionFind, n: int, items: list[Path]) -> list[list[Path]]:
    clusters: dict[int, list[Path]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(items[i])
    return [v for v in clusters.values() if len(v) > 1]


# ── Exact hashing ─────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# ── Destination naming ────────────────────────────────────────────────────────

def unique_dest_path(dup_dir: Path, path: Path) -> Path:
    """Return a unique path inside dup_dir with a _dup[N] suffix."""
    stem, suffix = path.stem, path.suffix
    candidate = dup_dir / f"{stem}_dup{suffix}"
    n = 1
    while candidate.exists():
        candidate = dup_dir / f"{stem}_dup{n}{suffix}"
        n += 1
    return candidate


# ── Thumbnail helper ─────────────────────────────────────────────────────────

def b64_jpeg_thumbnail(im: Image.Image, size: int = THUMB, quality: int = 75) -> str:
    """Encode an already-loaded PIL frame as a base64 JPEG thumbnail."""
    try:
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


# ── MediaHandler strategy interface ───────────────────────────────────────────

class MediaHandler(ABC):
    """
    Format-specific strategy plugged into the shared dedup workflow.

    Subclasses implement how to collect candidate files, compute a
    similarity signature, measure distance between two signatures, pick
    the best copy to keep, and render report thumbnails/metadata. The
    exact-duplicate pass (SHA-256) and the whole Rich UI / HTML report /
    move workflow are handled generically by this module.
    """

    display_name: str = "Media"     # e.g. "Image", "Animated Media", "Video"
    item_noun: str = "files"        # e.g. "images", "animated GIF/WebP files"
    report_title: str = "Duplicate Finder Report"
    report_emoji: str = "\U0001F5C2"
    report_slug: str = "duplicate"  # default report filename stem

    # Cap on concurrent workers for similarity_signature() specifically.
    # None = use the shared default pool size. Override downward for
    # handlers whose signature step isn't safely fan-out-able at full
    # width (e.g. video decoding via subprocess-backed libraries).
    signature_workers: Optional[int] = None

    @abstractmethod
    def collect(self, src: Path, recursive: bool) -> list[Path]:
        """Return sorted candidate files under src (excluding Duplicate/)."""

    def exact_hash(self, path: Path) -> Optional[str]:
        return sha256_file(path)

    @abstractmethod
    def similarity_signature(self, path: Path) -> Any:
        """Return a signature object, or None if it can't be computed."""

    def bucket_key(self, signature: Any) -> Hashable:
        """Coarse bucket to cut down pairwise comparisons. Default: one bucket."""
        return "_"

    @abstractmethod
    def similarity_distance(self, sig_a: Any, sig_b: Any) -> float:
        """Lower = more similar. Return float('inf') for "definitely not a match"."""

    @abstractmethod
    def select_primary(self, group: list[Path]) -> Path:
        """Choose which file in a duplicate group to keep."""

    @abstractmethod
    def thumbnail_b64(self, path: Path) -> str:
        """Base64 JPEG thumbnail for the HTML report, or '' on failure."""

    @abstractmethod
    def metadata_line(self, path: Path) -> str:
        """Short human string for the HTML card, e.g. '1920x1080 . 4.2 MB'."""

    def file_size(self, path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0


# ── Progress bar factory ──────────────────────────────────────────────────────

def _progress_bar(show_remaining: bool = True) -> Progress:
    cols: list = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=36),
    ]
    if show_remaining:
        cols += [MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn()]
    else:
        cols += [TaskProgressColumn(), TimeElapsedColumn()]
    return Progress(*cols, console=console)


# ── Parallel execution helper ─────────────────────────────────────────────────

def _default_workers() -> int:
    return min(32, (os.cpu_count() or 4) + 4)


def parallel_compute(
    items: list,
    fn,
    tm: TerminationManager,
    prog: Progress,
    task,
    max_workers: Optional[int] = None,
) -> list:
    """
    Run fn(item) for every item on a thread pool, advancing prog/task once
    per completion, and return results in the same order as `items`.

    A thread pool (not a process pool) is used deliberately: handlers cache
    unpicklable objects (PIL images, moviepy clips) and per-file work here
    is dominated by file I/O and native (GIL-releasing) image/hash code, so
    threads already parallelize the actual bottleneck.

    On ESC (tm.is_terminating()), stops waiting on stragglers and cancels
    whatever hasn't started yet — the same "stop, don't crash" behavior as
    the serial loops elsewhere in this module. Exceptions raised by `fn`
    propagate out of this call, matching a plain serial loop calling `fn`
    directly.
    """
    results: list = [None] * len(items)
    if not items:
        return results
    with cf.ThreadPoolExecutor(max_workers=max_workers or _default_workers()) as ex:
        futures = {ex.submit(fn, item): i for i, item in enumerate(items)}
        try:
            for fut in cf.as_completed(futures):
                if tm.is_terminating():
                    break
                results[futures[fut]] = fut.result()
                prog.advance(task)
        finally:
            for fut in futures:
                fut.cancel()
    return results


# ── Multi-source collection ───────────────────────────────────────────────────

def collect_from_sources(
    handler: MediaHandler,
    srcs: list[Path],
    recursive: bool,
) -> tuple[list[Path], dict[Path, Path]]:
    """
    Collect candidate files from every source folder combined. Each
    folder's handler.collect() call is independent I/O, so with more than
    one source they run concurrently. Returns (items, item_src) where
    item_src maps each file back to the source-folder root it came from
    (used later to route moved duplicates back to *their own* folder's
    Duplicate/, and to label folders in the report/summary).

    Files that resolve to the same on-disk path from more than one source
    (overlapping or nested folders) are only counted once, keeping the
    exact-hash pass from seeing a file "duplicate itself".
    """
    if len(srcs) == 1:
        per_src = {srcs[0]: handler.collect(srcs[0], recursive)}
    else:
        per_src = {}
        with cf.ThreadPoolExecutor(max_workers=min(len(srcs), _default_workers())) as ex:
            futures = {ex.submit(handler.collect, s, recursive): s for s in srcs}
            for fut, s in futures.items():
                per_src[s] = fut.result()

    items: list[Path] = []
    item_src: dict[Path, Path] = {}
    seen: set[Path] = set()
    for s in srcs:
        for p in per_src.get(s, []):
            resolved = p.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            items.append(p)
            item_src[p] = s
    return items, item_src


# ── Dedup logic ───────────────────────────────────────────────────────────────

def find_duplicate_groups(
    items: list[Path],
    handler: MediaHandler,
    threshold: float,
    exact_only: bool,
    tm: TerminationManager,
) -> list[dict[str, Any]]:
    """
    Returns list of group dicts, each with:
      - "paths":      list[Path]   (2+ files that are duplicates)
      - "match_type": "exact" | "near"
    Groups with only one file (unique) are excluded.
    """
    n = len(items)
    uf = UnionFind(n)
    exact_indices: set[int] = set()

    # ── Pass 1: SHA-256 exact match ──────────────────────────────────────
    console.print(Rule("Pass 1 / SHA-256 exact matching", style="cyan"))
    sha_map: dict[str, list[int]] = {}

    with _progress_bar() as prog:
        task = prog.add_task(f"[cyan]Hashing {handler.item_noun}[/cyan]", total=n)
        hashes = parallel_compute(items, handler.exact_hash, tm, prog, task)
    if tm.is_terminating():
        console.print("\n  [yellow]⚠  ESC pressed — stopping.[/yellow]")

    for i, h in enumerate(hashes):
        if h:
            sha_map.setdefault(h, []).append(i)

    n_exact_groups = 0
    for idxs in sha_map.values():
        if len(idxs) > 1:
            n_exact_groups += 1
            exact_indices.update(idxs)
            for j in idxs[1:]:
                uf.union(idxs[0], j)

    console.print(f"  [green]✓[/green]  {n_exact_groups} exact-duplicate group(s) found\n")

    if exact_only:
        raw = _uf_to_groups(uf, n, items)
        return [{"paths": g, "match_type": "exact"} for g in raw]

    # ── Pass 2: similarity near-match ────────────────────────────────────
    console.print(Rule("Pass 2 / Similarity near-match", style="cyan"))
    console.print(f"  [dim]Distance threshold: {threshold}[/dim]\n")

    def _safe_signature(p: Path) -> Any:
        try:
            return handler.similarity_signature(p)
        except Exception:
            return None

    with _progress_bar() as prog:
        task = prog.add_task("[cyan]Computing signatures[/cyan]", total=n)
        signatures: list[Any] = parallel_compute(
            items, _safe_signature, tm, prog, task,
            max_workers=handler.signature_workers,
        )
    if tm.is_terminating():
        console.print("\n  [yellow]⚠  ESC pressed — stopping.[/yellow]")

    near_pairs = 0
    bucket: dict[Hashable, list[int]] = defaultdict(list)
    for i, sig in enumerate(signatures):
        if sig is None:
            continue
        bucket[handler.bucket_key(sig)].append(i)

    with _progress_bar(show_remaining=False) as prog:
        task = prog.add_task("[cyan]Comparing signatures[/cyan]", total=len(bucket))
        for idxs in bucket.values():
            if tm.is_terminating():
                prog.advance(task)
                continue
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    ia, ib = idxs[a], idxs[b]
                    if uf.find(ia) == uf.find(ib):
                        continue
                    try:
                        dist = handler.similarity_distance(signatures[ia], signatures[ib])
                    except Exception:
                        continue
                    if dist <= threshold:
                        uf.union(ia, ib)
                        near_pairs += 1
            prog.advance(task)

    console.print(f"\n  [green]✓[/green]  {near_pairs} additional near-duplicate pair(s) merged\n")

    path_to_idx = {p: i for i, p in enumerate(items)}
    raw = _uf_to_groups(uf, n, items)
    result: list[dict[str, Any]] = []
    for group in raw:
        idxs = {path_to_idx[p] for p in group}
        match_type = "exact" if idxs <= exact_indices else "near"
        result.append({"paths": group, "match_type": match_type})
    return result


# ── HTML report ───────────────────────────────────────────────────────────────

def write_html_report(
    groups: list[dict[str, Any]],
    out_path: Path,
    threshold: float,
    dry_run: bool,
    recursive: bool,
    src: Union[Path, list[Path]],
    handler: MediaHandler,
    item_src: Optional[dict[Path, Path]] = None,
) -> None:
    """
    Generate dark-theme HTML report with Exact and Near-match sections.

    `src` accepts either a single Path (original, single-folder behavior)
    or a list of Paths (combined multi-folder run). When multiple sources
    are given, pass `item_src` (path -> owning source root, as returned by
    collect_from_sources()) to label each card with its originating folder
    and add a per-folder breakdown section.
    """
    srcs = list(src) if isinstance(src, (list, tuple)) else [src]

    exact_groups = [g for g in groups if g["match_type"] == "exact"]
    near_groups  = [g for g in groups if g["match_type"] == "near"]

    total_exact = sum(len(g["paths"]) - 1 for g in exact_groups)
    total_near  = sum(len(g["paths"]) - 1 for g in near_groups)
    total_dups  = total_exact + total_near

    mode_label = "DRY RUN — preview only, no files moved" if dry_run else "⚠ EXECUTED — files have been moved"
    mode_color = "#f0c040" if dry_run else "#ff6060"

    def _card(p: Path, is_primary: bool) -> str:
        img_src = ""
        try:
            thumb = handler.thumbnail_b64(p)
            if thumb:
                img_src = f"data:image/jpeg;base64,{thumb}"
        except Exception:
            pass
        try:
            meta = handler.metadata_line(p)
        except Exception:
            meta = ""
        label      = "PRIMARY" if is_primary else "DUP"
        badge_bg   = "#00c853" if is_primary else "#d32f2f"
        border_col = "#00e676" if is_primary else "#ff5252"
        folder_tag = ""
        if item_src and len(srcs) > 1:
            root = item_src.get(p)
            if root is not None:
                folder_tag = f'<div class="ffolder">{root.name or str(root)}</div>'
        return (
            f'<div class="card">'
            f'<span class="badge" style="background:{badge_bg}">{label}</span>'
            f'<img src="{img_src}" title="{p.name}" '
            f'     style="border:2px solid {border_col}">'
            f'<div class="fname">{p.name}</div>'
            f'{folder_tag}'
            f'<div class="fmeta">{meta}</div>'
            f'</div>'
        )

    def _render_group(g_info: dict) -> str:
        paths   = g_info["paths"]
        primary = handler.select_primary(paths)
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
        f"🔍 Near Duplicates (distance ≤ {threshold})", near_groups, "#ce93d8"
    )

    multi_source = len(srcs) > 1
    if multi_source:
        source_heading = f"Sources ({len(srcs)})"
        source_lines   = "<br>".join(f"&nbsp;&nbsp;{s}" for s in srcs)
        source_value   = f"<br>{source_lines}"
    else:
        source_heading = "Source"
        source_value   = str(srcs[0])

    folder_sec = ""
    if multi_source and item_src:
        counts: dict[Path, int] = {s: 0 for s in srcs}
        for g in groups:
            for p in g["paths"]:
                root = item_src.get(p)
                if root in counts:
                    counts[root] += 1
        rows = "".join(
            f'<tr><td>{s}</td><td>{counts[s]}</td></tr>' for s in srcs
        )
        folder_sec = (
            '<section>'
            '<h2 style="color:#8ab4f8">📁 Per-folder breakdown'
            '  <span class="count">files involved in duplicate groups, by source</span>'
            '</h2>'
            f'<table class="folders">{rows}</table>'
            '</section>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{handler.report_title}</title>
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
    .card img {{ width:{THUMB}px; height:{THUMB}px; object-fit:cover;
                 border-radius:6px; display:block; background:#000 }}
    .fname  {{ font-size:9px; color:#aaa; max-width:{THUMB}px;
               word-break:break-all; margin-top:4px }}
    .fmeta  {{ font-size:9px; color:#4fc; margin-top:2px }}
    .ffolder {{ font-size:9px; color:#8ab4f8; margin-top:2px }}
    table.folders {{ width:100%; border-collapse:collapse; font-size:.85rem }}
    table.folders td {{ padding:4px 8px; border-bottom:1px solid #2a2a3f }}
    table.folders td:last-child {{ text-align:right; color:#4fc }}
  </style>
</head>
<body>
  <h1>{handler.report_emoji} {handler.report_title}</h1>
  <p class="meta">
    {source_heading}: {source_value}<br>
    Mode: <span class="mode">{mode_label}</span><br>
    Threshold: {threshold} &nbsp;|&nbsp;
    Recursive: {'yes' if recursive else 'no'} &nbsp;|&nbsp;
    Groups: {len(groups)} &nbsp;|&nbsp;
    Total duplicates: {total_dups} &nbsp;|&nbsp;
    Exact: {total_exact} &nbsp;|&nbsp;
    Near: {total_near}
  </p>
  {folder_sec}
  {exact_sec}
  {near_sec}
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


# ── Source-folder resolution (prompt / validate, 1–MAX_SOURCES folders) ──────

def resolve_source_folders(src_arg: Union[None, Path, list]) -> list[Path]:
    """
    Normalize args.src into a validated list of 1..MAX_SOURCES existing
    directories.

    - None            -> interactively prompt (loop, capped at MAX_SOURCES).
    - a single Path    -> validated and wrapped in a 1-element list
                          (unchanged single-folder CLI behavior).
    - a list of Paths  -> validated, deduped by resolved path, and capped.
    """
    if src_arg is None:
        srcs: list[Path] = []
        while True:
            label = f"Source folder {len(srcs) + 1}/{MAX_SOURCES}" if srcs else "Source folder"
            raw = Prompt.ask(f"  [cyan]{label}[/cyan]").strip().strip('"').strip("'")
            p = Path(raw)
            if not (p.exists() and p.is_dir()):
                console.print(f"    [red]✗  Not found:[/red] {p}")
                continue
            if any(p.resolve() == s.resolve() for s in srcs):
                console.print(f"    [yellow]⚠  Already added:[/yellow] {p}")
                continue
            srcs.append(p)
            if len(srcs) >= MAX_SOURCES:
                console.print(f"  [dim]Maximum of {MAX_SOURCES} source folders reached.[/dim]")
                break
            if not Confirm.ask("  Add another source folder?", default=False):
                break
        return srcs

    raw_list = list(src_arg) if isinstance(src_arg, (list, tuple)) else [src_arg]
    if not raw_list:
        console.print("[red]  ✗  No source folder given.[/red]")
        sys.exit(1)
    if len(raw_list) > MAX_SOURCES:
        console.print(f"[red]  ✗  Too many source folders ({len(raw_list)}); max {MAX_SOURCES}.[/red]")
        sys.exit(1)

    srcs = []
    seen: set[Path] = set()
    for p in raw_list:
        p = Path(p)
        if not p.exists() or not p.is_dir():
            console.print(f"[red]  ✗  Source folder not found: {p}[/red]")
            sys.exit(1)
        resolved = p.resolve()
        if resolved in seen:
            continue   # same folder passed twice — dedupe silently
        seen.add(resolved)
        srcs.append(p)
    return srcs


# ── Shared workflow ────────────────────────────────────────────────────────────

def run_workflow(handler: MediaHandler, args, tm: TerminationManager) -> None:
    """
    Generic collect -> detect -> summarize -> dry-run -> confirm -> move ->
    report pipeline, shared by every duplicate-finder tool. Behavior and
    console/report copy mirror find_duplicates.py's original workflow.
    """
    console.print()
    console.print(Panel.fit(
        Text.assemble(
            (f"Duplicate {handler.display_name} Finder", "bold cyan"),
            ("  ·  ", "dim"),
            ("SHA-256 exact  +  perceptual near-match", "dim"),
        ),
        border_style="cyan",
    ))
    console.print()

    # ── Source folder(s) ─────────────────────────────────────────────────
    srcs = resolve_source_folders(args.src)

    dry_run = not args.execute
    report  = args.report if args.report else srcs[0] / f"{handler.report_slug}_report.html"
    args.src = srcs   # keep args.src consistent as a list for downstream/tests

    if len(srcs) == 1:
        console.print(f"  [cyan]Source     [/cyan] : {srcs[0]}")
    else:
        console.print(f"  [cyan]Sources ({len(srcs)})[/cyan] :")
        for i, s in enumerate(srcs, 1):
            console.print(f"                   {i}. {s}")
    console.print(f"  [cyan]Mode       [/cyan] : [bold {'yellow' if dry_run else 'red'}]"
                  f"{'DRY RUN — no files moved' if dry_run else 'EXECUTE — files will be moved'}[/bold "
                  f"{'yellow' if dry_run else 'red'}]")
    console.print(f"  [cyan]Threshold  [/cyan] : {args.threshold}")
    console.print(f"  [cyan]Recursive  [/cyan] : {'yes' if args.recursive else 'no'}")
    console.print(f"  [cyan]Exact only [/cyan] : {'yes' if args.exact_only else 'no'}")
    console.print(f"  [cyan]Report     [/cyan] : {report}")
    console.print()

    # ── Collect (combined across all sources) ────────────────────────────
    console.print(Rule(f"Collecting {handler.item_noun}", style="cyan"))
    items, item_src = collect_from_sources(handler, srcs, args.recursive)
    if not items:
        where = str(srcs[0]) if len(srcs) == 1 else f"any of the {len(srcs)} source folders"
        console.print(f"[yellow]  ⚠  No {handler.item_noun} found in: {where}[/yellow]")
        return
    if len(srcs) == 1:
        console.print(f"  [cyan]{len(items)}[/cyan] {handler.item_noun} found\n")
    else:
        per_folder = Counter(item_src[p] for p in items)
        console.print(f"  [cyan]{len(items)}[/cyan] {handler.item_noun} found across {len(srcs)} folders (combined):")
        for s in srcs:
            console.print(f"    [dim]{per_folder.get(s, 0):>6}[/dim]  {s}")
        console.print()

    # ── Detect ───────────────────────────────────────────────────────────
    groups = find_duplicate_groups(items, handler, args.threshold, args.exact_only, tm)

    if not groups:
        console.print(Panel.fit(
            Text.assemble(("No duplicates found.", "bold green")),
            border_style="green",
        ))
        return

    # ── Summary table ────────────────────────────────────────────────────
    total_dups  = sum(len(g["paths"]) - 1 for g in groups)
    total_bytes = 0

    multi_source = len(srcs) > 1

    console.print(Rule("Duplicate groups", style="cyan"))
    table = Table(
        show_lines=False,
        header_style="bold cyan",
        min_width=78,
    )
    table.add_column("Primary (keep)",      style="green",   max_width=36)
    if multi_source:
        table.add_column("Folder",          style="blue",    max_width=20)
    table.add_column("Type",                style="cyan",    justify="center")
    table.add_column("Copies",              justify="right", style="bold white")
    table.add_column("Details (primary)",   justify="right", style="yellow")

    for g_info in sorted(groups, key=lambda g: -len(g["paths"])):
        group   = g_info["paths"]
        mtype   = g_info["match_type"]
        primary = handler.select_primary(group)
        dups    = [p for p in group if p != primary]
        total_bytes += sum(handler.file_size(p) for p in dups)
        try:
            details = handler.metadata_line(primary)
        except Exception:
            details = "—"
        type_label = "[cyan]exact[/cyan]" if mtype == "exact" else "[magenta]near[/magenta]"
        row = [primary.name]
        if multi_source:
            root = item_src.get(primary)
            row.append((root.name or str(root)) if root else "—")
        row += [type_label, str(len(dups)), details]
        table.add_row(*row)

    console.print(table)
    console.print(
        f"\n  [bold cyan]{len(groups)}[/bold cyan] group(s)  ·  "
        f"[bold]{total_dups}[/bold] duplicate file(s)  ·  "
        f"[dim]{total_bytes/1024/1024:.1f} MB recoverable[/dim]\n"
    )

    # ── Dry run offer ────────────────────────────────────────────────────
    if dry_run:
        console.print(f"  [dim]Generating HTML report …[/dim]")
        write_html_report(groups, report, args.threshold, True, args.recursive,
                           srcs if multi_source else srcs[0], handler, item_src)
        console.print(f"  [green]✓[/green]  Report → [link={report.as_uri()}]{report}[/link]\n")

        console.print(Panel.fit(
            Text.assemble(
                ("Dry run complete.\n", "bold yellow"),
                (f"  {total_dups} duplicates would be moved to {DUPLICATE_FOLDER}/.\n\n", "white"),
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
    # Each duplicate moves into *its own* source folder's Duplicate/ — not a
    # single shared one — so single-folder behavior is unchanged and a
    # multi-folder run still leaves each folder's cleanup self-contained.
    console.print(Rule("Moving duplicates", style="cyan"))
    dup_dirs: dict[Path, Path] = {}
    for s in srcs:
        d = s / DUPLICATE_FOLDER
        d.mkdir(exist_ok=True)
        dup_dirs[s] = d

    moved = errors = 0
    dest_map: dict[Path, Path] = {}

    with _progress_bar(show_remaining=False) as prog:
        task = prog.add_task("[cyan]Moving …[/cyan]", total=total_dups)
        for g_info in groups:
            group   = g_info["paths"]
            primary = handler.select_primary(group)
            for dup in group:
                if dup == primary:
                    continue
                dup_root = item_src.get(dup, srcs[0])
                dest = unique_dest_path(dup_dirs[dup_root], dup)
                try:
                    shutil.move(str(dup), str(dest))
                    dest_map[dup] = dest
                    item_src[dest] = dup_root
                    moved += 1
                except Exception as exc:
                    console.print(f"  [red]ERR[/red]  {dup.name}: {exc}")
                    errors += 1
                prog.advance(task)

    for g_info in groups:
        g_info["paths"] = [dest_map.get(p, p) for p in g_info["paths"]]

    if len(srcs) == 1:
        console.print(
            f"\n  [green]✓[/green]  {moved} duplicates moved to [cyan]{dup_dirs[srcs[0]]}[/cyan]  ·  {errors} errors\n"
        )
    else:
        console.print(
            f"\n  [green]✓[/green]  {moved} duplicates moved to each source's "
            f"[cyan]{DUPLICATE_FOLDER}/[/cyan] folder  ·  {errors} errors\n"
        )

    # ── HTML report ──────────────────────────────────────────────────────
    console.print(f"  [dim]Generating HTML report …[/dim]")
    write_html_report(groups, report, args.threshold, False, args.recursive,
                       srcs if multi_source else srcs[0], handler, item_src)
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
