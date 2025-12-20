#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import time
import threading
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
try:
    import msvcrt
    _HAS_MSVC = True
except Exception:
    _HAS_MSVC = False
try:
    import select, termios, tty
    _HAS_POSIX_TTY = True
except Exception:
    _HAS_POSIX_TTY = False
try:
    import msvcrt
    _HAS_MSVC = True
except Exception:
    _HAS_MSVC = False
try:
    import select, termios, tty
    _HAS_POSIX_TTY = True
except Exception:
    _HAS_POSIX_TTY = False


SUPPORTED_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def format_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    i = 0
    while f >= 1024 and i < len(units) - 1:
        f /= 1024
        i += 1
    return f"{f:.2f} {units[i]}"


def is_video_file(path: str) -> bool:
    lower = path.lower()
    return any(lower.endswith(ext) for ext in SUPPORTED_VIDEO_EXTS)


def list_videos(folder: str) -> List[str]:
    files: List[str] = []
    for root, _, fnames in os.walk(folder, topdown=True):
        for fname in fnames:
            p = os.path.join(root, fname)
            if is_video_file(p):
                files.append(p)
    return files


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class CancelMonitor:
    """Monitors for Esc key and exposes a cancellation event.

    Windows: uses msvcrt.kbhit/getch to detect Esc (\x1b).
    POSIX: enables raw mode on stdin and uses select to detect Esc.
    """

    def __init__(self) -> None:
        self.event = threading.Event()
        self._t: Optional[threading.Thread] = None
        self._restore: Optional[Tuple[int, bytes]] = None

    def _run_windows(self) -> None:
        while not self.event.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b"\x1b",):
                    self.event.set()
                    break
            time.sleep(0.05)

    def _run_posix(self) -> None:
        try:
            if not sys.stdin.isatty():
                return
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            self._restore = (fd, termios.tcgetattr(fd))
            tty.setcbreak(fd)
            while not self.event.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r:
                    ch = sys.stdin.read(1)
                    if ch == "\x1b":
                        self.event.set()
                        break
        finally:
            try:
                if self._restore:
                    termios.tcsetattr(self._restore[0], termios.TCSADRAIN, self._restore[1])
            except Exception:
                pass

    def start(self) -> None:
        if self._t and self._t.is_alive():
            return
        target = self._run_windows if _HAS_MSVC else self._run_posix
        self._t = threading.Thread(target=target, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self.event.set()
        if self._t and self._t.is_alive():
            try:
                self._t.join(timeout=0.5)
            except Exception:
                pass


def ffprobe_duration_ms(path: str) -> Optional[int]:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        s = out.decode("utf-8", errors="ignore").strip()
        if not s:
            return None
        dur = float(s)
        return int(dur * 1000)
    except Exception:
        return None


def build_ffmpeg_cmd(in_path: str, out_path: str, codec: str, crf: int, preset: str) -> List[str]:
    if codec == "h264":
        return [
            "ffmpeg",
            "-y",
            "-i",
            in_path,
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-map",
            "0",
            "-c:a",
            "copy",
            "-progress",
            "pipe:1",
            "-nostats",
            "-loglevel",
            "error",
            out_path,
        ]
    else:
        return [
            "ffmpeg",
            "-y",
            "-i",
            in_path,
            "-c:v",
            "libx265",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-map",
            "0",
            "-c:a",
            "copy",
            "-progress",
            "pipe:1",
            "-nostats",
            "-loglevel",
            "error",
            out_path,
        ]


def compute_metrics(orig_path: str, new_path: str) -> Dict[str, float]:
    try:
        cmd = [
            "ffmpeg",
            "-i",
            orig_path,
            "-i",
            new_path,
            "-filter_complex",
            "[0:v][1:v]ssim;[0:v][1:v]psnr",
            "-f",
            "null",
            "-",
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        text = (p.stdout or "") + (p.stderr or "")
        metrics: Dict[str, float] = {}
        for line in text.splitlines():
            if "SSIM" in line and "All:" in line:
                try:
                    tail = line.split("All:")[-1]
                    val = float(tail.split()[0])
                    metrics["ssim"] = val
                except Exception:
                    pass
            if "average:" in line and "PSNR" in line:
                try:
                    for part in line.split():
                        if part.startswith("average:"):
                            val = float(part.split(":")[1])
                            metrics["psnr"] = val
                            break
                except Exception:
                    pass
        return metrics
    except Exception:
        return {}


def _run_ffmpeg_with_progress(cmd: List[str], duration_ms: Optional[int], cancel_event: Optional[threading.Event] = None) -> Tuple[int, float]:
    start_t = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True)
    last_ms = 0
    try:
        if duration_ms and duration_ms > 0:
            with tqdm(total=duration_ms, desc="File", unit="ms", dynamic_ncols=True, mininterval=0.2, file=sys.stdout, leave=False) as fbar:
                while True:
                    if cancel_event and cancel_event.is_set():
                        try:
                            if p.stdin:
                                p.stdin.write("q\n")
                                p.stdin.flush()
                        except Exception:
                            pass
                        try:
                            p.wait(timeout=3)
                        except Exception:
                            try:
                                p.terminate()
                            except Exception:
                                pass
                        break
                    line = p.stdout.readline()
                    if not line:
                        if p.poll() is not None:
                            break
                        continue
                    line = line.strip()
                    if line.startswith("out_time_ms="):
                        try:
                            val = int(line.split("=", 1)[1])
                            last_ms = val
                            fbar.n = min(val, duration_ms)
                            fbar.refresh()
                        except Exception:
                            pass
                    elif line.startswith("progress=") and line.endswith("end"):
                        break
        else:
            while True:
                if cancel_event and cancel_event.is_set():
                    try:
                        if p.stdin:
                            p.stdin.write("q\n")
                            p.stdin.flush()
                    except Exception:
                        pass
                    try:
                        p.wait(timeout=3)
                    except Exception:
                        try:
                            p.terminate()
                        except Exception:
                            pass
                    break
                if p.poll() is not None:
                    break
                time.sleep(0.1)
    finally:
        # Drain remaining output to avoid zombies
        try:
            if p.stdout:
                p.stdout.read()
        except Exception:
            pass
        try:
            if p.stderr:
                p.stderr.read()
        except Exception:
            pass
    elapsed = time.time() - start_t
    rc = p.returncode
    return rc if rc is not None else 1, elapsed


def compress_one(in_path: str, codec: str, crf: int, preset: str, cancel_event: Optional[threading.Event] = None) -> Tuple[bool, str, Dict[str, float], int, int, float]:
    try:
        if not os.path.isfile(in_path):
            return False, "not a file", {}, 0, 0, 0.0
        if not is_video_file(in_path):
            return False, "unsupported type", {}, 0, 0, 0.0
        orig_size = os.path.getsize(in_path)
        tmp_out = in_path + ".tmp.mp4"
        cmd = build_ffmpeg_cmd(in_path, tmp_out, codec, crf, preset)
        duration_ms = ffprobe_duration_ms(in_path)
        rc, elapsed = _run_ffmpeg_with_progress(cmd, duration_ms, cancel_event)
        if rc != 0 or not os.path.exists(tmp_out):
            return False, "ffmpeg error", {}, orig_size, 0, elapsed
        new_size = os.path.getsize(tmp_out)
        if new_size < orig_size:
            metrics = compute_metrics(in_path, tmp_out)
            os.replace(tmp_out, in_path)
            return True, "", metrics, orig_size, new_size, elapsed
        else:
            try:
                os.remove(tmp_out)
            except Exception:
                pass
            return True, "no reduction", {}, orig_size, new_size, elapsed
    except Exception as e:
        try:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass
        return False, str(e), {}, 0, 0, 0.0


def run_batch(paths: List[str], codec: str, crf: int, preset: str, dry_run: bool) -> None:
    total = len(paths)
    if total == 0:
        print("[INFO] No videos to process.")
        return
    compressed = 0
    replaced = 0
    skipped = 0
    failed = 0
    monitor = CancelMonitor()
    monitor.start()
    aborted = False
    with tqdm(total=total, desc="Compressing", unit="vid", dynamic_ncols=True, mininterval=0.3, file=sys.stdout) as pbar:
        for p in paths:
            if monitor.event.is_set():
                aborted = True
                break
            if dry_run:
                size = os.path.getsize(p) if os.path.exists(p) else 0
                pbar.set_postfix_str(
                    f"DryRun size={format_size(size)} | codec={codec} | crf={crf} | preset={preset}"
                )
                pbar.update(1)
                continue
            ok, err, metrics, orig_size, new_size, secs = compress_one(p, codec, crf, preset, monitor.event)
            if ok:
                compressed += 1
                if new_size < orig_size:
                    replaced += 1
                else:
                    skipped += 1
            else:
                if err == "ffmpeg error" and monitor.event.is_set():
                    aborted = True
                    break
                failed += 1
            ratio = (new_size / orig_size) if orig_size else 0
            ssim = metrics.get("ssim")
            psnr = metrics.get("psnr")
            details = (
                f"{os.path.basename(p)} | {format_size(orig_size)} → {format_size(new_size)} "
                f"| ratio={ratio:.2f} | {secs:.2f}s"
            )
            if ssim is not None:
                details += f" | SSIM={ssim:.4f}"
            if psnr is not None:
                details += f" | PSNR={psnr:.2f}"
            pbar.set_postfix_str(
                f"Compressed={compressed} | Replaced={replaced} | Skipped={skipped} | Failed={failed}"
            )
            pbar.write(details)
            pbar.update(1)
    monitor.stop()
    if aborted:
        print("\n[INFO] Compression cancelled by user (Esc).")
        print("\n[SUMMARY]")
        print(f" Total videos considered: {total}")
        print(f" Compressed: {compressed}")
        print(f" Replaced originals: {replaced}")
        print(f" Skipped: {skipped}")
        print(f" Failed: {failed}")
        sys.exit(130)
    print("\n[SUMMARY]")
    print(f" Total videos considered: {total}")
    print(f" Compressed: {compressed}")
    print(f" Replaced originals: {replaced}")
    print(f" Skipped: {skipped}")
    print(f" Failed: {failed}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch video compression using FFmpeg")
    parser.add_argument("--input", type=str, help="Input folder or single video file path")
    parser.add_argument("--codec", type=str, choices=["h264", "h265"], default="h265")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", type=str, default="medium")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    if not ffmpeg_available():
        print("[ERROR] ffmpeg/ffprobe not found on PATH.")
        sys.exit(1)
    args = parse_args()
    in_path = args.input
    if not in_path:
        in_path = input("Enter input folder or video file path: ").strip()
    if not os.path.exists(in_path):
        print(f"[ERROR] The path '{in_path}' does not exist.")
        sys.exit(1)
    if os.path.isdir(in_path):
        vids = list_videos(in_path)
    else:
        if is_video_file(in_path):
            vids = [in_path]
        else:
            print("[ERROR] Unsupported file type.")
            sys.exit(1)
    if len(vids) == 0:
        print("[INFO] No supported videos found.")
        sys.exit(0)
    run_batch(vids, args.codec, args.crf, args.preset, args.dry_run)


if __name__ == "__main__":
    main()
class CancelMonitor:
    """Monitors for Esc key and exposes a cancellation event.

    Windows: uses msvcrt.kbhit/getch to detect Esc (\x1b).
    POSIX: enables raw mode on stdin and uses select to detect Esc.
    """

    def __init__(self) -> None:
        self.event = threading.Event()
        self._t: Optional[threading.Thread] = None
        self._restore: Optional[Tuple[int, bytes]] = None

    def _run_windows(self) -> None:
        while not self.event.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b"\x1b",):
                    self.event.set()
                    break
            time.sleep(0.05)

    def _run_posix(self) -> None:
        try:
            if not sys.stdin.isatty():
                return
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            self._restore = (fd, termios.tcgetattr(fd))
            tty.setcbreak(fd)
            while not self.event.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r:
                    ch = sys.stdin.read(1)
                    if ch == "\x1b":
                        self.event.set()
                        break
        finally:
            try:
                if self._restore:
                    termios.tcsetattr(self._restore[0], termios.TCSADRAIN, self._restore[1])
            except Exception:
                pass

    def start(self) -> None:
        if self._t and self._t.is_alive():
            return
        self._t = threading.Thread(target=self._run_windows if _HAS_MSVC else self._run_posix, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self.event.set()
        if self._t and self._t.is_alive():
            try:
                self._t.join(timeout=0.5)
            except Exception:
                pass
