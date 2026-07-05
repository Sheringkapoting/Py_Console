#!/usr/bin/env python3
"""
compress_videos.py  ─  Batch video compression with hardware acceleration
==========================================================================
Compresses video files using FFmpeg with automatic hardware-encoder
detection (NVENC / QSV / AMF / VideoToolbox) and the latest codec choices
(SVT-AV1 > H.265 > H.264).  Quality is measured with VMAF (Netflix's
perceptual metric, industry standard since ~2020) with automatic fallback
to SSIM/PSNR when libvmaf is not compiled into the local FFmpeg build.

Codec quality notes
───────────────────
  SVT-AV1  — best compression (~50 % smaller than H.264 at equal quality);
              CRF 30–38 typical; presets 4–8 balance speed vs quality
  H.265    — ~40 % better than H.264; CRF 24–28 typical
  H.264    — widest compatibility; CRF 18–24 typical

Requirements:
    pip install rich tqdm

    FFmpeg ≥ 6.0 with libsvtav1 is recommended.
    Download from https://ffmpeg.org/download.html
    (BtbN Windows builds include SVT-AV1, NVENC, QSV, VMAF by default.)

Usage:
    python compress_videos.py                         # interactive
    python compress_videos.py --input "D:\\Videos"    # direct
    python compress_videos.py --codec av1 --crf 32 --execute
    python compress_videos.py --codec h265 --hw auto --execute
    python compress_videos.py --dry-run               # preview, no changes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── ESC-key abort (via shared TerminationManager) ─────────────────────────────
_scripts_dir = str(Path(__file__).parent / "src" / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from common_utils import TerminationManager as _TerminationManager
del _scripts_dir
_tm = _TerminationManager()
_tm.start_monitoring()

# ── Dependency check ──────────────────────────────────────────────────────────

def _check_deps() -> None:
    missing = [pkg for pkg, mod in [("rich", "rich"), ("tqdm", "tqdm")]
               if not _importable(mod)]
    if missing:
        print(f"\n  ✗  Run:  pip install {' '.join(missing)}\n")
        sys.exit(1)

def _importable(name: str) -> bool:
    try: __import__(name); return True
    except ImportError: return False

_check_deps()

# ── Imports (after dep-check) ─────────────────────────────────────────────────

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

console = Console()

# ── Supported formats ─────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".flv",
              ".webm", ".ts", ".mts", ".m2ts"}

# ── Codec profiles  ───────────────────────────────────────────────────────────
# Each entry: (software_encoder, hw_nvenc, hw_qsv, hw_amf, hw_videotoolbox,
#              default_crf, crf_range_hint, preset_hint)

CODEC_PROFILES = {
    "av1": dict(
        soft     = "libsvtav1",
        nvenc    = "av1_nvenc",
        qsv      = "av1_qsv",
        amf      = "av1_amf",
        vtb      = None,                 # VideoToolbox AV1 encode not yet stable
        default_crf = 32,
        crf_hint    = "28–38  (lower = better quality)",
        preset_hint = "5–8  (lower = slower, higher quality)",
        extra_soft  = ["-svtav1-params", "tune=0"],   # tune=0 = visual quality
    ),
    "h265": dict(
        soft     = "libx265",
        nvenc    = "hevc_nvenc",
        qsv      = "hevc_qsv",
        amf      = "hevc_amf",
        vtb      = "hevc_videotoolbox",
        default_crf = 26,
        crf_hint    = "22–30  (lower = better quality)",
        preset_hint = "ultrafast / fast / medium / slow / veryslow",
        extra_soft  = ["-tag:v", "hvc1"],  # Apple/browser compatibility
    ),
    "h264": dict(
        soft     = "libx264",
        nvenc    = "h264_nvenc",
        qsv      = "h264_qsv",
        amf      = "h264_amf",
        vtb      = "h264_videotoolbox",
        default_crf = 23,
        crf_hint    = "18–28  (lower = better quality)",
        preset_hint = "ultrafast / fast / medium / slow / veryslow",
        extra_soft  = [],
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# FFmpeg / hardware capability detection
# ══════════════════════════════════════════════════════════════════════════════

def _run_quiet(*cmd: str) -> str:
    """Run a command and return stdout+stderr as a string (no exceptions)."""
    try:
        r = subprocess.run(list(cmd), capture_output=True, text=True, timeout=10)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def ffmpeg_ok() -> bool:
    return bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))


def detect_encoders() -> dict[str, bool]:
    """
    Ask FFmpeg which encoders are compiled in.
    Returns a dict  encoder_name → available.
    """
    out  = _run_quiet("ffmpeg", "-encoders")
    all_encoders: set[str] = set()
    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and len(parts[0]) == 6:   # e.g. "V..... libx264"
            all_encoders.add(parts[1].lower())

    needed = set()
    for p in CODEC_PROFILES.values():
        for key in ("soft", "nvenc", "qsv", "amf", "vtb"):
            if p[key]:
                needed.add(p[key])

    return {enc: enc.lower() in all_encoders for enc in needed}


def detect_vmaf() -> bool:
    """Return True if the libvmaf filter is available in this FFmpeg build."""
    return "libvmaf" in _run_quiet("ffmpeg", "-filters")


def choose_encoder(codec: str, hw_pref: str, avail: dict[str, bool]) -> tuple[str, bool]:
    """
    Return (encoder_name, is_hardware) for the given codec and hw preference.

    hw_pref: "auto" | "nvenc" | "qsv" | "amf" | "vtb" | "cpu"
    Falls back to software if the requested hardware encoder is unavailable.
    """
    p = CODEC_PROFILES[codec]
    hw_map = {"nvenc": p["nvenc"], "qsv": p["qsv"], "amf": p["amf"], "vtb": p["vtb"]}

    if hw_pref == "cpu":
        return p["soft"], False

    if hw_pref == "auto":
        # Try each hardware option in preference order
        for key in ("nvenc", "qsv", "amf", "vtb"):
            enc = hw_map[key]
            if enc and avail.get(enc):
                return enc, True
        return p["soft"], False

    # Explicit hardware request
    enc = hw_map.get(hw_pref)
    if enc and avail.get(enc):
        return enc, True
    console.print(f"  [yellow]⚠  {hw_pref} not available for {codec}, falling back to CPU.[/yellow]")
    return p["soft"], False


# ══════════════════════════════════════════════════════════════════════════════
# FFprobe  — video metadata
# ══════════════════════════════════════════════════════════════════════════════

def probe_duration_ms(path: str) -> Optional[int]:
    """Return video duration in milliseconds, or None on failure."""
    out = _run_quiet(
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    )
    try:
        return int(float(out.strip()) * 1000)
    except (ValueError, TypeError):
        return None


def probe_video_info(path: str) -> dict:
    """Return codec, width, height, bitrate of the first video stream."""
    out = _run_quiet(
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,bit_rate",
        "-of", "json",
        path,
    )
    try:
        data    = json.loads(out)
        streams = data.get("streams", [])
        if streams:
            s = streams[0]
            return {
                "codec":   s.get("codec_name", "unknown"),
                "width":   s.get("width", 0),
                "height":  s.get("height", 0),
                "bitrate": int(s.get("bit_rate") or 0),
            }
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# FFmpeg command builders
# ══════════════════════════════════════════════════════════════════════════════

def build_encode_cmd(
    in_path: str,
    out_path: str,
    codec: str,
    encoder: str,
    is_hw: bool,
    crf: int,
    preset: str,
) -> list[str]:
    """
    Build the FFmpeg encode command.

    Hardware encoders (NVENC/QSV/AMF) use -cq instead of -crf because
    the CRF parameter is a software-encoder concept.
    """
    p = CODEC_PROFILES[codec]

    base = ["ffmpeg", "-y", "-i", in_path]

    if is_hw:
        # Hardware path: quality via -cq (NVENC/QSV/AMF equivalent of CRF)
        base += [
            "-c:v", encoder,
            "-cq",  str(crf),
            "-preset", preset,
        ]
    else:
        # Software path
        base += [
            "-c:v", encoder,
            "-crf",    str(crf),
            "-preset", preset,
        ]
        base += p.get("extra_soft", [])

    base += [
        "-map",      "0",      # include all streams (audio, subtitles, …)
        "-c:a",      "copy",   # pass audio through unchanged
        "-progress", "pipe:1",
        "-nostats",
        "-loglevel", "error",
        out_path,
    ]
    return base


def build_vmaf_cmd(orig: str, compressed: str) -> list[str]:
    """
    VMAF quality assessment command (FFmpeg libvmaf filter).
    VMAF score range: 0–100; 93+ is perceptually transparent.
    """
    return [
        "ffmpeg",
        "-i", orig,
        "-i", compressed,
        "-filter_complex",
        "[0:v][1:v]libvmaf=log_fmt=json:log_path=-:n_threads=4",
        "-f", "null", "-",
    ]


def build_ssim_psnr_cmd(orig: str, compressed: str) -> list[str]:
    """Fallback quality metric when VMAF is not available."""
    return [
        "ffmpeg",
        "-i", orig,
        "-i", compressed,
        "-filter_complex", "[0:v][1:v]ssim;[0:v][1:v]psnr",
        "-f", "null", "-",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Quality metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_quality(orig: str, compressed: str, use_vmaf: bool) -> dict[str, float]:
    """
    Compute VMAF (primary) or SSIM + PSNR (fallback).

    VMAF reference:
        93–100 : perceptually transparent / reference quality
        80–93  : high quality, negligible artifacts
        60–80  : acceptable for streaming
        < 60   : visible degradation
    """
    if use_vmaf:
        cmd = build_vmaf_cmd(orig, compressed)
        out = _run_quiet(*cmd)
        # Parse VMAF JSON log written to stderr/stdout
        try:
            # FFmpeg writes the VMAF JSON to the log_path=- (stdout equivalent)
            for line in out.splitlines():
                line = line.strip()
                if '"VMAF score"' in line or '"vmaf"' in line.lower():
                    # Try to extract the mean VMAF score from JSON output
                    import re
                    m = re.search(r'"mean"\s*:\s*([\d.]+)', out)
                    if m:
                        return {"vmaf": round(float(m.group(1)), 3)}
                    m = re.search(r'"VMAF score"\s*:\s*([\d.]+)', out)
                    if m:
                        return {"vmaf": round(float(m.group(1)), 3)}
            # Try full JSON parse
            start = out.find('{"version"')
            if start == -1:
                start = out.find('{"VMAF')
            if start != -1:
                data = json.loads(out[start:out.rfind('}') + 1])
                score = (
                    data.get("pooled_metrics", {}).get("vmaf", {}).get("mean")
                    or data.get("VMAF score")
                )
                if score is not None:
                    return {"vmaf": round(float(score), 3)}
        except Exception:
            pass

    # SSIM / PSNR fallback
    cmd = build_ssim_psnr_cmd(orig, compressed)
    out = _run_quiet(*cmd)
    metrics: dict[str, float] = {}
    for line in out.splitlines():
        if "SSIM" in line and "All:" in line:
            try:
                metrics["ssim"] = float(line.split("All:")[-1].split()[0])
            except Exception:
                pass
        if "PSNR" in line and "average:" in line:
            try:
                for part in line.split():
                    if part.startswith("average:"):
                        metrics["psnr"] = float(part.split(":")[1])
                        break
            except Exception:
                pass
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Progress-aware FFmpeg runner
# ══════════════════════════════════════════════════════════════════════════════

def run_ffmpeg(
    cmd: list[str],
    duration_ms: Optional[int],
    rich_progress: Progress,
    task_id,
) -> tuple[int, float]:
    """
    Stream FFmpeg's  -progress pipe:1  output to update a Rich progress task.
    Returns (return_code, elapsed_seconds).
    """
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True,
    )

    def _kill() -> None:
        try:
            proc.stdin.write("q\n"); proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.terminate()

    try:
        total_set = duration_ms and duration_ms > 0
        if total_set:
            rich_progress.update(task_id, total=duration_ms, completed=0)

        while True:
            if _tm.is_terminating():
                _kill()
                break

            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue

            line = line.strip()
            if line.startswith("out_time_ms=") and total_set:
                try:
                    ms = int(line.split("=", 1)[1])
                    rich_progress.update(task_id, completed=min(ms, duration_ms))
                except ValueError:
                    pass
            elif line == "progress=end":
                if total_set:
                    rich_progress.update(task_id, completed=duration_ms)
                break

    finally:
        # Drain pipes to prevent zombie processes
        try: proc.stdout.read()
        except Exception: pass
        try: proc.stderr.read()
        except Exception: pass

    rc = proc.returncode
    return (rc if rc is not None else 1), time.time() - t0


# ══════════════════════════════════════════════════════════════════════════════
# Single-file compression
# ══════════════════════════════════════════════════════════════════════════════

def format_size(n: int) -> str:
    if n <= 0:
        return "0 B"
    units, f = ["B", "KB", "MB", "GB", "TB"], float(n)
    i = 0
    while f >= 1024 and i < len(units) - 1:
        f /= 1024; i += 1
    return f"{f:.2f} {units[i]}"


def compress_one(
    in_path: Path,
    codec: str,
    encoder: str,
    is_hw: bool,
    crf: int,
    preset: str,
    out_folder: Optional[Path],
    backup_folder: Optional[Path],
    use_vmaf: bool,
    rich_progress: Progress,
    file_task_id,
) -> dict:
    """
    Compress a single video file.  Returns a result dict for the JSON log.
    """
    result: dict = {
        "file":      str(in_path),
        "status":    "error",
        "error":     None,
        "orig_size": 0,
        "new_size":  0,
        "savings_pct": 0.0,
        "elapsed_s": 0.0,
        "quality":   {},
    }

    if not in_path.is_file():
        result["error"] = "not a file"; return result

    orig_size = in_path.stat().st_size
    result["orig_size"] = orig_size

    # Determine output path
    if out_folder:
        out_folder.mkdir(parents=True, exist_ok=True)
        tmp = out_folder / (in_path.stem + "_compressed.mp4")
        final_dest = out_folder / in_path.name
    else:
        tmp = in_path.with_suffix(".tmp_compress.mp4")
        final_dest = in_path            # will replace original

    cmd          = build_encode_cmd(str(in_path), str(tmp), codec, encoder, is_hw, crf, preset)
    duration_ms  = probe_duration_ms(str(in_path))

    rc, elapsed  = run_ffmpeg(cmd, duration_ms, rich_progress, file_task_id)
    result["elapsed_s"] = round(elapsed, 2)

    if rc != 0 or not tmp.exists():
        try: tmp.unlink(missing_ok=True)
        except Exception: pass
        result["error"] = "ffmpeg non-zero exit" if rc != 0 else "output not created"
        return result

    new_size = tmp.stat().st_size
    result["new_size"] = new_size

    if new_size >= orig_size:
        try: tmp.unlink()
        except Exception: pass
        result["status"] = "skipped"
        result["error"]  = "compressed ≥ original (kept original)"
        return result

    # Compute quality before replacing
    result["quality"] = compute_quality(str(in_path), str(tmp), use_vmaf)

    # Optional backup of original
    if backup_folder:
        backup_folder.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(in_path), str(backup_folder / in_path.name))
        except Exception as e:
            result["error"] = f"backup failed: {e}"; return result

    # Move compressed file into place
    try:
        shutil.move(str(tmp), str(final_dest))
    except Exception as e:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass
        result["error"] = f"move failed: {e}"; return result

    savings = (1 - new_size / orig_size) * 100
    result.update(status="ok", savings_pct=round(savings, 1))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Batch runner
# ══════════════════════════════════════════════════════════════════════════════

def run_batch(
    paths: list[Path],
    codec: str,
    encoder: str,
    is_hw: bool,
    crf: int,
    preset: str,
    out_folder: Optional[Path],
    backup_folder: Optional[Path],
    use_vmaf: bool,
    dry_run: bool,
    log_path: Optional[Path],
) -> None:

    if not paths:
        console.print("[yellow]  ⚠  No videos to process.[/yellow]")
        return

    total   = len(paths)
    stats   = defaultdict(int)
    results = []

    console.print()
    console.print(Rule("Compressing", style="cyan"))

    enc_label = f"[{'green' if is_hw else 'yellow'}]{encoder}[/]"
    mode_label = "[bold yellow]DRY RUN[/bold yellow]" if dry_run else f"{enc_label} · CRF {crf} · {preset}"

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

        batch_task = progress.add_task(
            f"[cyan]Batch[/cyan]  {mode_label}",
            total=total,
        )
        file_task = progress.add_task("[dim]Waiting …[/dim]", total=None)

        for path in paths:
            if _tm.is_terminating():
                console.print("\n[red]  Cancelled (Esc pressed).[/red]")
                break

            fname = path.name
            progress.update(file_task, description=f"[white]{fname[:50]}[/white]", total=None, completed=0)

            if dry_run:
                info = probe_video_info(str(path))
                size = path.stat().st_size if path.exists() else 0
                console.print(
                    f"  [dim]DRY[/dim]  {fname}  "
                    f"{format_size(size)}  "
                    f"{info.get('codec','?')} "
                    f"{info.get('width','?')}×{info.get('height','?')}"
                )
                stats["dry"] += 1
                progress.advance(batch_task)
                continue

            res = compress_one(
                path, codec, encoder, is_hw, crf, preset,
                out_folder, backup_folder, use_vmaf,
                progress, file_task,
            )
            results.append(res)

            # Build one-line result string
            if res["status"] == "ok":
                stats["ok"] += 1
                q = res["quality"]
                q_str = ""
                if "vmaf" in q:
                    q_str = f"  VMAF {q['vmaf']:.1f}"
                elif "ssim" in q:
                    q_str = f"  SSIM {q['ssim']:.4f}"
                console.print(
                    f"  [green]✓[/green]  {fname[:45]}  "
                    f"{format_size(res['orig_size'])} → {format_size(res['new_size'])}  "
                    f"[green]−{res['savings_pct']:.1f}%[/green]  "
                    f"{res['elapsed_s']:.1f}s{q_str}"
                )
            elif res["status"] == "skipped":
                stats["skipped"] += 1
                console.print(f"  [yellow]─[/yellow]  {fname[:45]}  {res['error']}")
            else:
                stats["failed"] += 1
                console.print(f"  [red]✗[/red]  {fname[:45]}  {res['error']}")

            progress.advance(batch_task)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("Summary", style="cyan"))

    table = Table(show_lines=False, header_style="bold cyan", min_width=50)
    table.add_column("Metric",  style="dim")
    table.add_column("Value",   style="bold white", justify="right")

    table.add_row("Total considered",   str(total))
    if dry_run:
        table.add_row("Dry-run scanned", str(stats["dry"]))
    else:
        ok_results = [r for r in results if r["status"] == "ok"]
        saved_bytes = sum(r["orig_size"] - r["new_size"] for r in ok_results)
        table.add_row("Compressed",      str(stats["ok"]))
        table.add_row("Skipped (no gain)", str(stats["skipped"]))
        table.add_row("Failed",          str(stats["failed"]))
        table.add_row("Total saved",     format_size(saved_bytes))
        if ok_results:
            avg_saving = sum(r["savings_pct"] for r in ok_results) / len(ok_results)
            table.add_row("Avg size reduction", f"{avg_saving:.1f}%")
            # Quality summary
            vmaf_scores = [r["quality"]["vmaf"] for r in ok_results if "vmaf" in r["quality"]]
            if vmaf_scores:
                table.add_row("Avg VMAF",    f"{sum(vmaf_scores)/len(vmaf_scores):.1f} / 100")
            ssim_scores = [r["quality"]["ssim"] for r in ok_results if "ssim" in r["quality"]]
            if ssim_scores and not vmaf_scores:
                table.add_row("Avg SSIM",    f"{sum(ssim_scores)/len(ssim_scores):.4f}")

    console.print(table)

    # ── JSON log ──────────────────────────────────────────────────────────────
    if log_path and results:
        payload = {
            "run_at":   datetime.now().isoformat(timespec="seconds"),
            "codec":    codec,
            "encoder":  encoder,
            "hardware": is_hw,
            "crf":      crf,
            "preset":   preset,
            "dry_run":  dry_run,
            "results":  results,
        }
        try:
            log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            console.print(f"\n  [green]✓[/green]  Log  →  {log_path}")
        except Exception as e:
            console.print(f"\n  [yellow]⚠  Could not write log: {e}[/yellow]")


# ══════════════════════════════════════════════════════════════════════════════
# Interactive setup
# ══════════════════════════════════════════════════════════════════════════════

def interactive_setup(args: argparse.Namespace, cap: dict) -> argparse.Namespace:
    """Fill any unset args interactively and confirm before processing."""

    console.print()
    console.print(Panel.fit(
        Text.assemble(
            ("Video Compressor", "bold cyan"),
            ("  ·  ", "dim"),
            ("FFmpeg  NVENC / QSV / AMF / SVT-AV1", "dim"),
        ),
        border_style="cyan",
    ))

    # Capability banner
    hw_list = [k for k in ("nvenc", "qsv", "amf", "vtb") if cap.get(k)]
    soft_list = [c for c in CODEC_PROFILES if cap.get(f"soft_{c}")]
    console.print(
        f"  Hardware encoders : [green]{', '.join(hw_list) if hw_list else 'none detected'}[/green]"
    )
    console.print(
        f"  Software codecs   : [cyan]{', '.join(soft_list)}[/cyan]"
    )
    console.print(
        f"  Quality metric    : [cyan]{'VMAF' if cap['vmaf'] else 'SSIM / PSNR (VMAF not in this FFmpeg build)'}[/cyan]\n"
    )

    # Input path
    if args.input is None:
        raw = Prompt.ask("  [cyan]Input folder or video file[/cyan]").strip().strip('"\'')
        args.input = raw

    # Codec
    if not getattr(args, "_codec_set", False):
        default_codec = "av1" if cap.get("soft_av1") else "h265"
        args.codec = Prompt.ask(
            "  [cyan]Codec[/cyan]",
            choices=list(CODEC_PROFILES.keys()),
            default=default_codec,
        )

    profile = CODEC_PROFILES[args.codec]

    # CRF
    if not getattr(args, "_crf_set", False):
        args.crf = int(Prompt.ask(
            f"  [cyan]CRF[/cyan] [dim]({profile['crf_hint']})[/dim]",
            default=str(profile["default_crf"]),
        ))

    # Preset
    if not getattr(args, "_preset_set", False):
        default_preset = "6" if args.codec == "av1" else "medium"
        args.preset = Prompt.ask(
            f"  [cyan]Preset[/cyan] [dim]({profile['preset_hint']})[/dim]",
            default=default_preset,
        )

    # Hardware
    if not getattr(args, "_hw_set", False):
        hw_choices = ["auto", "cpu"] + [k for k in ("nvenc", "qsv", "amf", "vtb") if cap.get(k)]
        args.hw = Prompt.ask(
            "  [cyan]Hardware acceleration[/cyan]",
            choices=hw_choices,
            default="auto",
        )

    # Output folder (optional)
    if not getattr(args, "_out_set", False):
        raw = Prompt.ask(
            "  [cyan]Output folder[/cyan] [dim](Enter = replace originals in place)[/dim]",
            default="",
        ).strip().strip('"\'')
        args.output = raw or None

    # Backup
    if not getattr(args, "_backup_set", False) and not args.output:
        args.backup = Confirm.ask(
            "  Keep backups of originals before replacing?",
            default=True,
        )

    if not getattr(args, "_dry_set", False):
        args.dry_run = Confirm.ask("  Dry run first (no files changed)?", default=True)

    # Confirmation panel
    console.print()
    console.print(Rule("Settings", style="cyan"))
    console.print(f"  Input      : [white]{args.input}[/white]")
    console.print(f"  Codec      : [white]{args.codec}[/white]   CRF {args.crf}   Preset {args.preset}")
    console.print(f"  Hardware   : [white]{args.hw}[/white]")
    console.print(f"  Output     : [white]{args.output or 'replace originals'}[/white]")
    console.print(f"  Backup     : [white]{getattr(args,'backup',False)}[/white]")
    console.print(f"  Mode       : [bold {'yellow' if args.dry_run else 'red'}]"
                  f"{'DRY RUN (no changes)' if args.dry_run else 'EXECUTE — files will be modified'}[/bold "
                  f"{'yellow' if args.dry_run else 'red'}]")
    console.print()

    if not Confirm.ask("  Proceed?", default=True):
        console.print("[yellow]  Aborted.[/yellow]")
        sys.exit(0)

    return args


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog            = "compress_videos",
        description     = "Batch video compression with hardware acceleration.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Codecs (best → most compatible):
  av1    SVT-AV1: ~50% better than H.264, CRF 28–38
  h265   libx265: ~40% better than H.264, CRF 22–30
  h264   libx264: widest device compatibility,  CRF 18–28

Hardware options:
  auto   (default) try nvenc → qsv → amf → vtb → cpu
  nvenc  NVIDIA GPU   (H.264 / H.265 / AV1 on RTX 40xx)
  qsv    Intel GPU    (Quick Sync)
  amf    AMD GPU      (AMF/VCE)
  vtb    Apple Mac    (VideoToolbox)
  cpu    force software encoding

Examples:
  python compress_videos.py --input "D:\\Videos" --codec av1 --execute
  python compress_videos.py --input video.mp4 --hw nvenc --crf 28
  python compress_videos.py --dry-run
""",
    )
    ap.add_argument("--input",   type=str, default=None)
    ap.add_argument("--codec",   choices=list(CODEC_PROFILES.keys()), default=None)
    ap.add_argument("--crf",     type=int, default=None)
    ap.add_argument("--preset",  type=str, default=None)
    ap.add_argument("--hw",      type=str, default=None,
                    choices=["auto","cpu","nvenc","qsv","amf","vtb"])
    ap.add_argument("--output",  type=str, default=None,
                    help="Output folder (default: replace originals in place)")
    ap.add_argument("--backup",  action="store_true", default=False,
                    help="Back up originals before replacing")
    ap.add_argument("--backup-dir", type=str, default=None, dest="backup_dir",
                    help="Custom backup folder (default: <input>/_originals)")
    ap.add_argument("--execute", action="store_true", default=False,
                    help="Skip dry-run prompt and compress immediately")
    ap.add_argument("--dry-run", action="store_true", default=False, dest="dry_run")
    ap.add_argument("--no-log",  action="store_true", default=False, dest="no_log",
                    help="Do not write a JSON log file")
    return ap


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not ffmpeg_ok():
        console.print("[red]  ✗  ffmpeg / ffprobe not found on PATH.[/red]")
        console.print("     Download from: https://ffmpeg.org/download.html")
        sys.exit(1)

    # ── Detect capabilities once ─────────────────────────────────────────────
    with console.status("[cyan]Detecting FFmpeg capabilities …[/cyan]"):
        avail  = detect_encoders()
        has_vmaf = detect_vmaf()

    # Summarise hw availability per type
    cap: dict = {"vmaf": has_vmaf}
    for codec, p in CODEC_PROFILES.items():
        cap[f"soft_{codec}"] = avail.get(p["soft"], False)
    for hw in ("nvenc", "qsv", "amf", "vtb"):
        cap[hw] = any(avail.get(p[hw], False) for p in CODEC_PROFILES.values() if p[hw])

    # ── Parse CLI ────────────────────────────────────────────────────────────
    parser = build_parser()
    args   = parser.parse_args()
    args._codec_set  = "--codec"   in sys.argv
    args._crf_set    = "--crf"     in sys.argv
    args._preset_set = "--preset"  in sys.argv
    args._hw_set     = "--hw"      in sys.argv
    args._out_set    = "--output"  in sys.argv
    args._backup_set = "--backup"  in sys.argv or "--backup-dir" in sys.argv
    args._dry_set    = "--dry-run" in sys.argv or "--execute" in sys.argv

    if args.execute:
        args.dry_run = False

    args = interactive_setup(args, cap)

    # ── Resolve encoder ──────────────────────────────────────────────────────
    hw_pref    = args.hw or "auto"
    encoder, is_hw = choose_encoder(args.codec, hw_pref, avail)

    # ── Collect videos ───────────────────────────────────────────────────────
    in_path = Path(args.input)
    if not in_path.exists():
        console.print(f"[red]  ✗  Not found: {in_path}[/red]")
        sys.exit(1)

    if in_path.is_dir():
        videos = sorted(
            p for p in in_path.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )
    elif in_path.suffix.lower() in VIDEO_EXTS:
        videos = [in_path]
    else:
        console.print(f"[red]  ✗  Unsupported file type: {in_path.suffix}[/red]")
        sys.exit(1)

    if not videos:
        console.print("[yellow]  ⚠  No supported video files found.[/yellow]")
        sys.exit(0)

    console.print(f"\n  [cyan]{len(videos)}[/cyan] video(s) found\n")

    out_folder    = Path(args.output) if args.output else None
    backup_folder = None
    if getattr(args, "backup", False) and not out_folder:
        backup_folder = (
            Path(args.backup_dir) if getattr(args, "backup_dir", None)
            else in_path / "_originals"
        )

    log_path = None
    if not args.no_log:
        base = out_folder or (in_path if in_path.is_dir() else in_path.parent)
        log_path = base / f"compress_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    run_batch(
        paths         = videos,
        codec         = args.codec,
        encoder       = encoder,
        is_hw         = is_hw,
        crf           = args.crf,
        preset        = args.preset,
        out_folder    = out_folder,
        backup_folder = backup_folder,
        use_vmaf      = has_vmaf,
        dry_run       = args.dry_run,
        log_path      = log_path,
    )


if __name__ == "__main__":
    main()
