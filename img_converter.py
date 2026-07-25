#!/usr/bin/env python3
import os
import sys
import time
from typing import List, Tuple
from pathlib import Path
from PIL import Image, UnidentifiedImageError, ImageFile

# ── ESC-key abort (via shared TerminationManager) ─────────────────────────────
_scripts_dir = str(Path(__file__).parent / "src" / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from common_utils import format_size, TerminationManager as _TerminationManager
del _scripts_dir
_tm = _TerminationManager()
_tm.start_monitoring()

# ── Dependency check ──────────────────────────────────────────────────────────

def _importable(name: str) -> bool:
    try: __import__(name); return True
    except ImportError: return False

def _check_deps() -> None:
    missing = [pkg for pkg, mod in [("rich", "rich")] if not _importable(mod)]
    if missing:
        print(f"\n  ✗  Run:  pip install {' '.join(missing)}\n")
        sys.exit(1)

_check_deps()

# ── Imports (after dep-check) ─────────────────────────────────────────────────

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.rule import Rule
from rich.table import Table
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

console = Console()

# HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HAS_HEIC = True
except ImportError:
    _HAS_HEIC = False
    console.print("[yellow]  ⚠  pillow-heif not found. .heic support disabled.[/yellow]")

# Safety: mitigate decompression bombs
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 80_000_000  # ~50MP cap; adjust if needed

CONVERTIBLE_EXTS = ['.webp', '.png', '.jpeg']
if _HAS_HEIC:
    CONVERTIBLE_EXTS.append('.heic')
CONVERTIBLE_EXTS = tuple(CONVERTIBLE_EXTS)
SKIP_EXTS = ('.gif', '.jpg')  # .jpg is already target; .gif we skip per original code

def is_heic_animated(path: str) -> Tuple[bool, str]:
    """Check if HEIC file is animated or has multiple frames."""
    if not _HAS_HEIC:
        return (False, "heic support not available")
    
    try:
        with Image.open(path) as im:
            # Check if image is animated or has multiple frames
            is_animated = getattr(im, 'is_animated', False)
            n_frames = getattr(im, 'n_frames', 1)
            
            if is_animated or n_frames > 1:
                return (True, f"animated heic ({n_frames} frames)")
            return (False, "")
    except Exception as e:
        return (False, str(e))

def is_webp_animated(path: str) -> Tuple[bool, str]:
    try:
        with open(path, 'rb') as f:
            header = f.read(12)
            if len(header) < 12:
                return (False, "truncated header")
            if header[0:4] != b'RIFF' or header[8:12] != b'WEBP':
                return (False, "not webp riff")
            while True:
                chunk_hdr = f.read(8)
                if len(chunk_hdr) < 8:
                    break
                fourcc = chunk_hdr[0:4]
                size = int.from_bytes(chunk_hdr[4:8], 'little')
                if fourcc == b'VP8X':
                    data = f.read(size)
                    if len(data) < size:
                        return (False, "truncated vp8x")
                    flags = data[0]
                    if flags & 0x02:
                        return (True, "vp8x animated")
                elif fourcc == b'ANIM':
                    f.seek(size, os.SEEK_CUR)
                    return (True, "anim chunk")
                else:
                    f.seek(size, os.SEEK_CUR)
                if size % 2 == 1:
                    f.seek(1, os.SEEK_CUR)
        return (False, "")
    except Exception as e:
        return (False, str(e))



def list_files_recursive(folder: str) -> List[str]:
    files: List[str] = []
    for root, _, fnames in os.walk(folder, topdown=True):
        for fname in fnames:
            files.append(os.path.join(root, fname))
    return files

def should_convert(path: str) -> bool:
    lower = path.lower()
    if lower.endswith(SKIP_EXTS):
        return False
    if lower.endswith('.webp'):
        animated, _ = is_webp_animated(path)
        if animated:
            return False
    if lower.endswith('.heic'):
        animated, _ = is_heic_animated(path)
        if animated:
            return False
    return lower.endswith(CONVERTIBLE_EXTS)

def target_jpg_path(src_path: str) -> str:
    root, _ = os.path.splitext(src_path)
    return root + ".jpg"


def convert_one(file_path: str) -> Tuple[bool, str]:
    """
    Returns (ok, err_msg). On success, ok=True and err_msg="".
    On failure, ok=False and err_msg contains brief text.
    """
    try:
        # Validate quickly before loading fully
        if not os.path.isfile(file_path):
            return (False, "not a file")
        size = os.path.getsize(file_path)
        if size <= 0:
            return (False, "zero size")
        if file_path.lower().endswith('.webp'):
            animated, _ = is_webp_animated(file_path)
            if animated:
                return (False, "animated webp")
        if file_path.lower().endswith('.heic'):
            animated, _ = is_heic_animated(file_path)
            if animated:
                return (False, "animated heic")

        # Convert to RGB JPG
        with Image.open(file_path) as im:
            # Check for HEIC animation/multi-frame
            if _HAS_HEIC and file_path.lower().endswith('.heic'):
                if getattr(im, 'is_animated', False) or getattr(im, 'n_frames', 1) > 1:
                    return (False, "animated/multi-frame heic")

            # Verify content
            try:
                im.verify()
            except (UnidentifiedImageError, Image.DecompressionBombError) as e:
                return (False, f"invalid image: {e}")
        # Re-open to actually convert after verify() (Pillow requirement)
        with Image.open(file_path) as im2:
            exif = im2.info.get("exif")
            icc = im2.info.get("icc_profile")

            if im2.mode in ("RGBA", "LA") or (im2.mode == "P" and "transparency" in im2.info):
                rgba = im2.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[3])
                rgb = bg
            else:
                rgb = im2.convert("RGB")
            out_path = target_jpg_path(file_path)
            # Ensure we do not overwrite an existing different file unintentionally
            if os.path.abspath(out_path) == os.path.abspath(file_path):
                # Shouldn’t happen because we skip .jpg inputs
                return (False, "same path as source")

            save_kwargs = {
                "quality": 85,
                "optimize": True,
                "progressive": True,
                "subsampling": 2,
            }
            if exif:
                save_kwargs["exif"] = exif
            if icc:
                save_kwargs["icc_profile"] = icc

            rgb.save(out_path, "JPEG", **save_kwargs)
        return (True, "")
    except Exception as e:
        return (False, str(e))

def delete_source(file_path: str) -> Tuple[bool, str]:
    try:
        os.remove(file_path)
        return (True, "")
    except Exception as e:
        return (False, str(e))

def convert_images(folder: str) -> None:
    if not os.path.isdir(folder):
        console.print(f"[red]  ✗  The path '{folder}' is not a valid directory.[/red]")
        return

    all_files = list_files_recursive(folder)
    total_candidates = [f for f in all_files if should_convert(f)]
    total = len(total_candidates)

    if total == 0:
        exts = ".webp, .png, .jpeg" + (", .heic" if _HAS_HEIC else "")
        console.print(f"[yellow]  ⚠  No convertible images found ({exts}).[/yellow]")
        return

    converted = 0
    deleted = 0
    failed = 0
    total_original_size = 0
    total_converted_size = 0
    start_time = time.time()

    console.print()
    console.print(Rule("Converting", style="cyan"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=6,
    ) as progress:
        task = progress.add_task("[cyan]Converting[/cyan]", total=total)

        for file_path in total_candidates:
            if _tm.is_terminating():
                console.print("\n[red]  Cancelled (Esc pressed).[/red]")
                break

            fname = os.path.basename(file_path)
            progress.update(task, description=f"[white]{fname[:40]}[/white]")

            original_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            total_original_size += original_size

            ok, err = convert_one(file_path)

            if ok:
                converted_size = os.path.getsize(target_jpg_path(file_path)) if os.path.exists(target_jpg_path(file_path)) else 0
                total_converted_size += converted_size

                # Only delete original if conversion succeeded
                del_ok, _ = delete_source(file_path)
                converted += 1
                if del_ok:
                    deleted += 1

                ratio = (1 - converted_size / original_size) * 100 if original_size > 0 else 0
                console.print(
                    f"  [green]✓[/green]  {fname[:45]}  "
                    f"{format_size(original_size)} → {format_size(converted_size)}  "
                    f"[green]−{ratio:.1f}%[/green]"
                )
            else:
                failed += 1
                console.print(f"  [red]✗[/red]  {fname[:45]}  {err}")

            progress.advance(task)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_time = time.time() - start_time
    console.print()
    console.print(Rule("Conversion Summary", style="cyan"))

    table = Table(show_lines=False, header_style="bold cyan", min_width=50)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold white", justify="right")

    table.add_row("Total files processed", str(total))
    table.add_row("Successfully converted", str(converted))
    table.add_row("Original files deleted", str(deleted))
    table.add_row("Failed conversions", str(failed))
    table.add_row("Total processing time", f"{elapsed_time:.1f}s")

    if total_original_size > 0:
        table.add_row("Original total size", format_size(total_original_size))
        table.add_row("Converted total size", format_size(total_converted_size))
        space_saved = total_original_size - total_converted_size
        table.add_row(
            "Space saved",
            f"{format_size(space_saved)} ({(1 - total_converted_size/total_original_size)*100:.1f}%)",
        )
        if converted > 0:
            avg_rate = converted / elapsed_time if elapsed_time > 0 else 0
            table.add_row("Average processing rate", f"{avg_rate:.1f} files/sec")

    console.print(table)

def _jpg_paths(folder: str) -> List[str]:
    return [f for f in list_files_recursive(folder) if f.lower().endswith('.jpg')]

def compress_one_jpg(file_path: str) -> Tuple[bool, str, bool]:
    try:
        if not os.path.isfile(file_path):
            return (False, "not a file", False)
        orig_size = os.path.getsize(file_path)
        if orig_size <= 0:
            return (False, "zero size", False)
        tmp_path = file_path + ".tmp"
        with Image.open(file_path) as im:
            exif = im.info.get("exif")
            icc = im.info.get("icc_profile")
            rgb = im.convert("RGB")

            save_kwargs = {
                "quality": 85,
                "optimize": True,
                "progressive": True,
                "subsampling": 2,
            }
            if exif:
                save_kwargs["exif"] = exif
            if icc:
                save_kwargs["icc_profile"] = icc

            rgb.save(tmp_path, "JPEG", **save_kwargs)
        new_size = os.path.getsize(tmp_path)
        if new_size < orig_size:
            os.replace(tmp_path, file_path)
            return (True, "", True)
        else:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return (True, "no reduction", False)
    except KeyboardInterrupt:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return (False, str(e), False)

def compress_jpgs(folder: str) -> None:
    jpgs = _jpg_paths(folder)
    total = len(jpgs)
    if total == 0:
        console.print("[yellow]  ⚠  No .jpg files found to compress.[/yellow]")
        return

    compressed = 0
    deleted = 0
    skipped = 0
    failed = 0
    aborted = False

    total_original_size = sum(os.path.getsize(f) for f in jpgs if os.path.exists(f))
    total_compressed_size = 0
    total_saved = 0
    start_time = time.time()

    console.print()
    console.print(Rule("Compressing", style="cyan"))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=28),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=6,
        ) as progress:
            task = progress.add_task("[cyan]Compressing[/cyan]", total=total)

            for file_path in jpgs:
                if _tm.is_terminating():
                    aborted = True
                    break

                fname = os.path.basename(file_path)
                progress.update(task, description=f"[white]{fname[:40]}[/white]")

                original_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                ok, err, replaced = compress_one_jpg(file_path)

                if ok:
                    if replaced:
                        compressed += 1
                        deleted += 1
                        new_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                        saved = original_size - new_size
                        total_saved += saved
                        total_compressed_size += new_size
                        ratio = (saved / original_size) * 100 if original_size > 0 else 0
                        console.print(
                            f"  [green]🗜[/green]  {fname[:45]}  "
                            f"{format_size(original_size)} → {format_size(new_size)}  "
                            f"[green]−{ratio:.1f}%[/green]"
                        )
                    else:
                        skipped += 1
                        total_compressed_size += original_size
                        console.print(f"  [yellow]⏭[/yellow]  {fname[:45]}  no reduction")
                else:
                    failed += 1
                    total_compressed_size += original_size
                    console.print(f"  [red]✗[/red]  {fname[:45]}  {err}")

                progress.advance(task)
    except KeyboardInterrupt:
        aborted = True

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_time = time.time() - start_time
    if aborted:
        console.print("\n[red]  Cancelled (Esc pressed).[/red]")

    console.print()
    console.print(Rule("Compression Summary", style="cyan"))

    table = Table(show_lines=False, header_style="bold cyan", min_width=50)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold white", justify="right")

    table.add_row("Total .jpg files processed", str(total))
    table.add_row("Successfully compressed", str(compressed))
    table.add_row("Skipped (no reduction)", str(skipped))
    table.add_row("Failed compressions", str(failed))
    table.add_row("Total processing time", f"{elapsed_time:.1f}s")

    if total_original_size > 0:
        table.add_row("Original total size", format_size(total_original_size))
        table.add_row("Final total size", format_size(total_compressed_size))
        table.add_row(
            "Total space saved",
            f"{format_size(total_saved)} ({(total_saved/total_original_size)*100:.1f}%)",
        )
        if compressed > 0:
            avg_rate = compressed / elapsed_time if elapsed_time > 0 else 0
            table.add_row("Average compression rate", f"{avg_rate:.1f} files/sec")

    console.print(table)

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]Image Converter[/bold cyan]\n"
        "[dim]Converts .webp / .png / .jpeg" + (" / .heic" if _HAS_HEIC else "") + " to .jpg, deletes originals on success[/dim]",
        border_style="cyan",
    ))

    # Accept folder path from command line or prompt
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = console.input("[cyan]Enter folder path to convert images:[/cyan] ").strip()
    convert_images(folder_path)
    try:
        run_compress = Confirm.ask("\nWould you like to run the compression process?", default=False)
    except KeyboardInterrupt:
        console.print("\n[yellow]  ⚠  Compression process prompt cancelled by user.[/yellow]")
        sys.exit(0)
    if run_compress:
        compress_jpgs(folder_path)
    else:
        console.print("[dim]Compression skipped.[/dim]")

